#!/usr/bin/env python3
"""
main_robot.py  —  感知 → 决策 → 电机 整合主循环
=================================================
用法:
    python3 main_robot.py                    # 全功能
    python3 main_robot.py --dry-run          # 不驱动电机
    python3 main_robot.py --no-vlm           # 关 VLM
    python3 main_robot.py --max-speed 40     # 限速
    python3 main_robot.py --scenario3 --target-room 1208 --no-vlm --max-speed 35
"""

import argparse
import os
import signal
import socket
import sys
import threading
import time
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PKG_DIR)
sys.path.insert(0, os.path.join(_PKG_DIR, "perception"))

from task6_arbitrator import Task6Arbitrator
from scenario2_controller import Scenario2Controller
from global_ocr import GlobalOcrReader

CAMERA_ID      = 0
FRAME_WIDTH    = 640
FRAME_HEIGHT   = 480
PREVIEW_PORT   = 8091
JPEG_QUALITY   = 70
YOLO_PRIORITY  = 10
OCR_PRIORITY   = 0
GEOM_CLASS_FILTER = {0,24,28,39,56}
SORT_MAX_AGE   = 5
SORT_MIN_HITS  = 3
SORT_IOU       = 0.3
YOLO_SCORE_THRES = 0.4
YOLO_NMS_THRES   = 0.45


class GeometryStream:
    def __init__(self):
        from bpu_perception_stream import BpuYoloDetector, DEFAULT_MODEL
        from sort_tracker import Sort
        from task3_core import Task3Controller, Target

        self._yolo = BpuYoloDetector(
            model_path=DEFAULT_MODEL,
            score_thres=YOLO_SCORE_THRES,
            nms_thres=YOLO_NMS_THRES,
        )
        self._yolo.set_scheduling_params(priority=YOLO_PRIORITY, bpu_cores=[0])
        self._tracker = Sort(max_age=SORT_MAX_AGE,
                             min_hits=SORT_MIN_HITS,
                             iou_threshold=SORT_IOU)
        self._task3 = Task3Controller()
        self._Target = Target
        self._class_filter = set(GEOM_CLASS_FILTER)

    def step(self, frame, frame_id=0, front_distance=99.9):
        dets, cls = self._yolo.detect(frame)
        if len(dets) > 0:
            mask = np.array([int(c) in self._class_filter for c in cls], dtype=bool)
            dets = dets[mask]
        tracks = self._tracker.update(dets) if len(dets) > 0 else self._tracker.update()

        targets = []
        preview = []
        for trk in tracks:
            x1, y1, x2, y2, tid = trk[0], trk[1], trk[2], trk[3], trk[4]
            w, h = x2 - x1, y2 - y1
            targets.append(self._Target(
                track_id=int(tid),
                bbox_x=float(x1), bbox_y=float(y1),
                bbox_w=float(w),  bbox_h=float(h),
                confidence=1.0,
            ))
            preview.append({
                "id": int(tid), "cx": float((x1+x2)/2), "cy": float((y1+y2)/2),
                "w": float(w), "h": float(h), "conf": 1.0,
            })
        task3_cmd = self._task3.process(targets, frame_id=frame_id,
                                         front_distance=front_distance)
        return preview, task3_cmd


from ultrasonic import Ultrasonic


class FrameBuffer:
    def __init__(self):
        self._lock = threading.Lock()
        self._jpeg = None
        self._id = 0
    def update(self, jpeg_bytes, frame_id):
        with self._lock:
            self._jpeg = jpeg_bytes
            self._id = frame_id
    def get(self):
        with self._lock:
            return self._jpeg, self._id

FRAME_BUFFER = FrameBuffer()


class StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args): pass
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b'<html><body style="margin:0;background:#111">'
                b'<img src="/stream" style="width:100%;max-width:640px">'
                b'</body></html>')
            return
        if self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            last_id = -1
            try:
                while True:
                    jpeg, fid = FRAME_BUFFER.get()
                    if jpeg is None or fid == last_id:
                        time.sleep(0.03)
                        continue
                    last_id = fid
                    self.wfile.write(b"--frame\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpeg)))
                    self.end_headers()
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                return
            return
        self.send_response(404)
        self.end_headers()


def start_http_server(port):
    server = ThreadingHTTPServer(("0.0.0.0", port), StreamHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "<your-ip>"

def open_camera(camera_id, width, height):
    cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

def draw_overlay(frame, dets, ocr_results, result, fps, sem, motor_info=""):
    out = frame.copy()
    for d in dets:
        try:
            x1 = int(d["cx"]-d["w"]/2); y1 = int(d["cy"]-d["h"]/2)
            x2 = int(d["cx"]+d["w"]/2); y2 = int(d["cy"]+d["h"]/2)
            cv2.rectangle(out, (x1,y1), (x2,y2), (0,230,0), 2)
            cv2.putText(out, f"#{d['id']} h={int(d['h'])}", (x1, max(12,y1-4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,230,0), 1)
        except Exception: continue
    for r in ocr_results:
        try:
            pts = np.array(r["box"], dtype=np.int32).reshape(-1,1,2)
            cv2.polylines(out, [pts], True, (0,180,255), 2)
        except Exception: continue
    cv2.rectangle(out, (0,0), (FRAME_WIDTH,60), (25,25,25), -1)
    src = getattr(result, "source", "?")
    src = getattr(src, "value", src)
    reason = getattr(result, "reason", "")
    cv2.putText(out, f"fps {fps:4.1f} | src {src}", (6,16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
    cv2.putText(out, f"sem p={sem.get('person_present')} c={sem.get('crowded')}",
                (6,34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,220,255), 1)
    cv2.putText(out, motor_info[:60], (6,54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100,255,100), 1)
    return out


def main():
    ap = argparse.ArgumentParser(description="感知+电机 整合主循环")
    ap.add_argument("--camera", type=int, default=CAMERA_ID)
    ap.add_argument("--width", type=int, default=FRAME_WIDTH)
    ap.add_argument("--height", type=int, default=FRAME_HEIGHT)
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument("--ocr-every", type=int, default=7)
    ap.add_argument("--scenario2", action="store_true", help="启用场景2状态机")
    ap.add_argument("--scenario3", action="store_true",
                    help="启用场景3：OCR 门牌识别 + 偏移转向 + 超声波停车")
    ap.add_argument("--target-room", type=str, default="1207", help="★ 目标房间号")
    ap.add_argument("--approach-speed", type=int, default=15,
                    help="OCR 锁定门牌后的慢速 duty，建议 15")
    ap.add_argument("--approach-ocr-every", type=int, default=5,
                    help="门牌追踪激活后的 OCR 间隔帧数，建议比普通 ocr-every 小")
    ap.add_argument("--x-stable-px", type=int, default=40,
                    help="OCR x 稳定阈值：近期样本都在中位数 ±该像素内")
    ap.add_argument("--stable-samples", type=int, default=3,
                    help="至少几个 OCR 样本稳定后才允许转向")
    ap.add_argument("--lost-fast-frames", type=int, default=60,
                    help="慢速后连续多少控制帧没看到 target-room，就恢复 GO_SPEED")
    ap.add_argument("--center-deadband-px", type=int, default=45,
                    help="目标中心离画面中心小于该像素，就认为已对准")
    ap.add_argument("--door-stop-dist", type=float, default=0.40,
                    help="门牌追踪时超声波停车距离，单位 m")

    # Scenario3 tuning. These only affect --scenario3.
    # 你现场主要可以调这些；也可以直接改 scenario3_controller.py 顶部常量。
    ap.add_argument("--s3-search-dir", choices=("left", "right"), default="left",
                    help="场景3没看到目标门牌时，原地扫描方向")
    ap.add_argument("--s3-search-speed", type=int, default=12,
                    help="场景3搜索门牌时的原地转向 duty")
    ap.add_argument("--s3-align-speed", type=int, default=12,
                    help="场景3看到目标后，按 OCR 偏移转向的 duty")
    ap.add_argument("--s3-approach-speed", type=int, default=12,
                    help="场景3对准后慢速靠近 duty")
    ap.add_argument("--s3-center-deadband-px", type=int, default=45,
                    help="场景3目标 x 离画面中心小于该像素，就认为对准")
    ap.add_argument("--s3-x-stable-px", type=int, default=35,
                    help="场景3 OCR x 稳定阈值")
    ap.add_argument("--s3-stable-samples", type=int, default=3,
                    help="场景3需要几个 OCR x 样本稳定后再转向")
    ap.add_argument("--s3-lost-hold-frames", type=int, default=45,
                    help="场景3 OCR 暂时丢失后，保持上一判断多少帧")
    ap.add_argument("--s3-door-stop-dist", type=float, default=0.35,
                    help="场景3超声波停车距离，单位 m")
    ap.add_argument("--s3-max-active-sec", type=float, default=0.0,
                    help="场景3锁定目标后的最长运行秒数；0=不启用")

    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--port", type=int, default=PREVIEW_PORT)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-speed", type=int, default=80)
    args = ap.parse_args()

    if args.scenario2 and args.scenario3:
        ap.error("--scenario2 和 --scenario3 不能同时启用；一次只跑一个场景")

    # Scenario3 必须用 OCR。用户如果手滑 --ocr-every 0，这里自动兜底。
    if args.scenario3 and args.ocr_every <= 0:
        print("[warn] --scenario3 需要 OCR；已自动把 --ocr-every 改为 5", file=sys.stderr)
        args.ocr_every = 5

    MAX_PWM = args.max_speed

    print("[init] 几何流...", file=sys.stderr)
    geom = GeometryStream()

    ocr = None
    if args.ocr_every > 0:
        print("[init] OCR...", file=sys.stderr)
        ocr = GlobalOcrReader(priority=OCR_PRIORITY)

    print(f"[init] Task6 (VLM={'off' if args.no_vlm else 'on'})...", file=sys.stderr)
    arbitrator = Task6Arbitrator(enable_vlm=not args.no_vlm)

    ultrasonic = Ultrasonic()   # pin 用 ultrasonic.py 里的默认值
    ultrasonic.start()
    scenario2 = Scenario2Controller(
        target_room=args.target_room,
        frame_width=args.width,
        enable_door_nav=True,
        approach_speed=args.approach_speed,
        door_turn_speed=args.approach_speed,
        center_deadband_px=args.center_deadband_px,
        x_stable_px=args.x_stable_px,
        stable_samples=args.stable_samples,
        lost_fast_frames=args.lost_fast_frames,
        door_stop_dist=args.door_stop_dist,
    ) if args.scenario2 else None

    scenario3 = None
    if args.scenario3:
        from scenario3_controller import Scenario3Controller
        from task3_core import RobotState

        s3_search_state = (
            RobotState.TURN_RIGHT if args.s3_search_dir == "right"
            else RobotState.TURN_LEFT
        )
        scenario3 = Scenario3Controller(
            target_room=args.target_room,
            frame_width=args.width,
            search_turn_speed=args.s3_search_speed,
            search_turn_state=s3_search_state,
            align_turn_speed=args.s3_align_speed,
            approach_speed=args.s3_approach_speed,
            center_deadband_px=args.s3_center_deadband_px,
            x_stable_px=args.s3_x_stable_px,
            stable_samples=args.s3_stable_samples,
            lost_hold_frames=args.s3_lost_hold_frames,
            door_stop_dist=args.s3_door_stop_dist,
            max_active_sec=args.s3_max_active_sec,
        )
        print(
            f"[init] Scenario3 target={args.target_room} "
            f"search={args.s3_search_dir}@{args.s3_search_speed} "
            f"align={args.s3_align_speed} approach={args.s3_approach_speed} "
            f"stop={args.s3_door_stop_dist:.2f}m",
            file=sys.stderr,
        )

    mc = None
    if not args.dry_run:
        from motor_controller import MotorController
        print("[init] 电机控制器...", file=sys.stderr)
        mc = MotorController(auto_start=True)
    else:
        print("[init] DRY-RUN 模式", file=sys.stderr)

    http_server = None
    if not args.no_stream:
        try:
            http_server = start_http_server(args.port)
            print(f"[http] 预览: http://{get_local_ip()}:{args.port}", file=sys.stderr)
        except OSError as e:
            print(f"[http] 预览失败: {e}", file=sys.stderr)

    cap = open_camera(args.camera, args.width, args.height)
    stop_flag = {"v": False}
    signal.signal(signal.SIGINT, lambda s, f: stop_flag.update(v=True))

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    n = 0
    cur_fps = 0.0
    t_fps, n_fps = time.time(), 0
    last_log = time.time()
    ocr_results = []
    last_sem = {"person_present": None, "crowded": None, "age": -1.0}
    motor_info = ""
    last_ocr_run_frame = -10**9

    print("[run] 主循环启动", file=sys.stderr)
    try:
        while not stop_flag["v"]:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            t0 = time.time()
            front_distance = ultrasonic.distance
            dets, task3_cmd = geom.step(frame, frame_id=n,
                                         front_distance=front_distance)
            yolo_ms = (time.time() - t0) * 1000

            arbitrator.update_frame(frame)

            # OCR is expensive, so normally run it every args.ocr_every frames.
            # Scenario2 door-nav / Scenario3 need fresher OCR for door-number tracking.
            ocr_updated = False
            ocr_period = args.ocr_every
            if scenario2 is not None and getattr(scenario2, "door_nav_active", False):
                ocr_period = max(1, args.approach_ocr_every)
            if scenario3 is not None:
                # 场景3全程靠 OCR 找两个纸板里的目标门牌，所以保持较高 OCR 频率。
                ocr_period = max(1, min(args.ocr_every, args.approach_ocr_every))
            if ocr is not None and ocr_period > 0 and (n - last_ocr_run_frame) >= ocr_period:
                t_ocr = time.time()
                ocr_results = ocr.read(frame)
                last_ocr_run_frame = n
                ocr_updated = True
                ocr_ms = (time.time() - t_ocr) * 1000
            else:
                ocr_ms = 0.0

            if scenario3 is not None:
                final_cmd = scenario3.update(
                    ocr_results=(ocr_results if ocr_updated else None),
                    front_distance=front_distance,
                    frame_id=n,
                )
                result = SimpleNamespace(
                    source="scenario3",
                    reason=final_cmd.reason,
                    command=final_cmd,
                )
            elif scenario2 is not None:
                final_cmd = scenario2.update(
                    front_distance,
                    ocr_results=(ocr_results if ocr_updated else None),
                    frame_id=n,
                )
                result = SimpleNamespace(
                    source="scenario2",
                    reason=final_cmd.reason,
                    command=final_cmd,
                )
            else:
                nav2_cmd = None
                result = arbitrator.arbitrate(task3_cmd, nav2_cmd)
                final_cmd = getattr(result, "command", None)

            if mc is not None and final_cmd is not None:
                from task3_core import RobotState
                state = final_cmd.state
                ls = abs(final_cmd.left_speed)
                rs = abs(final_cmd.right_speed)
                if ls <= 1.0: ls *= MAX_PWM
                else: ls = min(ls, MAX_PWM)
                if rs <= 1.0: rs *= MAX_PWM
                else: rs = min(rs, MAX_PWM)

                if state in (RobotState.STOP, RobotState.IDLE):
                    actual_l, actual_r = 0, 0
                    mc.stop()
                elif state == RobotState.TURN_LEFT:
                    actual_l, actual_r = -ls, rs
                    mc.set_motors(actual_l, actual_r)
                elif state == RobotState.TURN_RIGHT:
                    actual_l, actual_r = ls, -rs
                    mc.set_motors(actual_l, actual_r)
                else:
                    actual_l, actual_r = ls, rs
                    mc.set_motors(actual_l, actual_r)
                motor_info = f"L={actual_l:+.0f} R={actual_r:+.0f} {state.value}"
            elif final_cmd is not None:
                motor_info = f"[DRY] L={final_cmd.left_speed:.2f} R={final_cmd.right_speed:.2f} {final_cmd.state.value}"

            sem_getter = getattr(arbitrator, "get_semantic_state", None)
            if callable(sem_getter):
                try:
                    s = sem_getter()
                    if s is not None: last_sem = s
                except Exception: pass

            if not args.no_stream:
                vis = draw_overlay(frame, dets, ocr_results, result,
                                    cur_fps, last_sem, motor_info)
                ok, jpeg = cv2.imencode(".jpg", vis, encode_params)
                if ok:
                    FRAME_BUFFER.update(jpeg.tobytes(), n)

            n += 1
            now = time.time()
            if now - t_fps >= 0.5:
                cur_fps = (n - n_fps) / (now - t_fps)
                t_fps, n_fps = now, n
            if now - last_log >= 1.0:
                print(
                    f"[stats] f={n} fps={cur_fps:.1f} yolo={yolo_ms:.0f}ms "
                    f"{motor_info} | reason={getattr(final_cmd, 'reason', '')}",
                    file=sys.stderr
                )
                last_log = now

    finally:
        stop_flag["v"] = True
        if mc is not None:
            mc.stop()
            time.sleep(0.1)
            mc.cleanup()
        cap.release()
        ultrasonic.stop()
        try: arbitrator.stop()
        except Exception: pass
        if http_server: http_server.shutdown()
        print("[run] 已退出", file=sys.stderr)


if __name__ == "__main__":
    main()
