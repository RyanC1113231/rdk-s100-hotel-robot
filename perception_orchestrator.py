"""
perception_orchestrator.py
==========================
RDK S100 单进程编排器：一台摄像头喂三条消费流，统一出决策。

这是把你已有的几块拼到一起跑的"总进程"。核心解决的问题：
你原来 combined_perception.py / global_ocr3.py / vlm_semantic.py 各自
cv2.VideoCapture 开一次摄像头，三个进程抢同一个 /dev/video0 开不起来。
这里改成 **一个进程独占摄像头**，每帧分发给三个消费者：

    相机采集 (本主循环, 唯一拥有者)
        ├── 几何流  YOLO+SORT  每帧同步取   → Task3 状态  → 高频反应式
        ├── 语义流  VLM        update_frame 推最新帧 → 后台线程跑(~5-8s/次)
        └── OCR     PaddleOCR  触发式(默认每N帧) → 读门牌
        ↓
    Task6 仲裁器  →  最终 MotorCommand  →  (ROS 接入后 publish; 现在打日志)

三条流互不阻塞：
  - YOLO 每帧同步 (它快, BPU)
  - VLM 在 Task6Arbitrator 内部的后台线程, 只取你 update_frame() 推进去的
    最新帧, 跳过中间积压 —— 慢消费者不拖快消费者
  - OCR 触发式, 不每帧跑 (每帧全图 OCR 会拖死 fps)

────────────────────────────────────────────────────────────
全部模块已接好 ✅ —— YOLO(BpuYoloDetector)+SORT、Task3、VLM、OCR、仲裁 都锁定了。
现在直接能跑 (相机/线程/VLM/OCR/仲裁/退出/预览 全写全)。

只剩两个"以后接"的占位 (现在用安全默认值, 不影响先跑起来, 搜 ★ 可定位)：
   ① 超声波 front_distance: 现在固定 99.9(前方无障碍)。接 HC-SR04 或 ROS
      距离 topic 后, 换主循环里那一行即可。
   ② Nav2 指令 / 指令输出: 现在 nav2_cmd=None、result.command 只打日志。
      接 ROS 后: nav2_cmd 换成订阅值, result.command publish 成 /cmd_vel。
────────────────────────────────────────────────────────────

⚠️ 跑之前先改一行：vlm_semantic.py 里 THREADS = 8 → 改成 4
   板子 6 核 (nproc=6, ARM 无超线程)。VLM 每次推理那几秒会抢满线程, 设 4 给
   YOLO+OCR+主循环留 2 核, 否则那几秒安全关键的 YOLO 路径会被饿到掉帧。
   先按 4 跑, 用 htop 看 VLM 爆发时 YOLO fps 掉多少, 再微调。

用法：
    python3 perception_orchestrator.py                 # 全开 + 浏览器预览
    python3 perception_orchestrator.py --no-vlm        # 只测 YOLO+OCR
    python3 perception_orchestrator.py --ocr-every 0   # 关 OCR (省算力)
    python3 perception_orchestrator.py --no-stream     # 不开浏览器预览
    浏览器: http://<tailscale-ip>:8091

依赖 (都是你已有的模块, 同目录 import)：
    task6_arbitrator.Task6Arbitrator      (内部持有并启动 VLMSemanticStream)
    global_ocr3.GlobalOcrReader
    task3_core.Task3Controller / Target
    bpu_perception_stream3.BpuYoloDetector / DEFAULT_MODEL   (按你实际文件名)
    sort_tracker.Sort
"""

import argparse
import os
import signal
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

# ── 路径设置 ──────────────────────────────────────────────────
# 本文件放在扁平目录 robot_tasks/ (与 task3_core.py / task6_arbitrator.py 同级)。
# 扁平目录会作为脚本目录自动进 sys.path; 这里再显式把 perception/ 子目录加进来,
# 这样 bpu_perception_stream3 / global_ocr3 / sort_tracker / vlm_semantic 都能 import。
# (不依赖 task6_arbitrator 内部那个 path 插入的副作用, 自己保证能跑。)
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))   # robot_tasks/
sys.path.insert(0, _PKG_DIR)
sys.path.insert(0, os.path.join(_PKG_DIR, "perception"))

# ── 你已有的模块 (已知接口, 直接 import) ──────────────────────
from task6_arbitrator import Task6Arbitrator        # 扁平目录; 内部启动 VLM 后台线程
from global_ocr3 import GlobalOcrReader             # perception/; read(frame)->[{"box","text","center"}]
# task3_core / bpu_perception_stream3 / sort_tracker 在 GeometryStream 里 import


# ============================================================
# 配置
# ============================================================

CAMERA_ID      = 0
FRAME_WIDTH    = 640
FRAME_HEIGHT   = 480
PREVIEW_PORT   = 8091          # 避开 8080/8081/8090
JPEG_QUALITY   = 70

# BPU 优先级 (数值含义以你 demo 为准; YOLO 必须高于 OCR, 它是安全关键)
YOLO_PRIORITY  = 10
OCR_PRIORITY   = 0


# ============================================================
# 几何流适配器 (YOLO BPU 检测 + SORT 跟踪 + Task3) —— 已全部锁定 ✅
# ============================================================
#
# 设计说明: bpu_perception_stream3.py 里的 PerceptionPipeline 自带 CameraReader
# (后台线程独占摄像头), 不能直接用——会和本编排器抢 /dev/video。所以这里只复用
# 它的两个无状态零件: BpuYoloDetector (检测) + Sort (跟踪), 喂本编排器的帧。
# 等价于把 PerceptionPipeline.step() 里 "detect → 类别过滤 → SORT" 那段搬出来,
# 去掉它自己的相机读取。

# 类别过滤: COCO class 0 = person。Task3 是避人逻辑, 默认只跟人。
# 要把行李车等也纳入避让, 往这个集合加对应 COCO 类号即可。
GEOM_CLASS_FILTER = {0}

# SORT 参数 (与 bpu_perception_stream3.py 默认一致)
SORT_MAX_AGE   = 5
SORT_MIN_HITS  = 3
SORT_IOU       = 0.3

# YOLO 置信度 / NMS
YOLO_SCORE_THRES = 0.4
YOLO_NMS_THRES   = 0.45


class GeometryStream:
    """
    几何流: 一帧进 → (预览检测列表, Task3 的 MotorCommand) 出。

    step(frame, frame_id, front_distance) 返回:
        dets:       预览用检测列表 [{"id","cx","cy","w","h","conf"}] (中心格式, 画框用)
        task3_cmd:  Task3Controller.process() 的 MotorCommand
                    (.state=RobotState / .left_speed / .right_speed / .reason),
                    直接喂给 Task6Arbitrator.arbitrate()
    """

    def __init__(self):
        # 模块名按你 S100 上实际文件名来 (你传的是 bpu_perception_stream3.py)。
        # 若文件叫 bpu_perception_stream.py, 把下面 import 的模块名改掉即可。
        from bpu_perception_stream3 import BpuYoloDetector, DEFAULT_MODEL
        from sort_tracker import Sort
        from task3_core import Task3Controller, Target

        # --- YOLO BPU 检测器 (最高优先级, 安全关键) ---
        self._yolo = BpuYoloDetector(
            model_path=DEFAULT_MODEL,
            score_thres=YOLO_SCORE_THRES,
            nms_thres=YOLO_NMS_THRES,
        )
        self._yolo.set_scheduling_params(priority=YOLO_PRIORITY, bpu_cores=[0])

        # --- SORT 跟踪器 (检测本身不带 track_id, ID 在这里产生) ---
        self._tracker = Sort(max_age=SORT_MAX_AGE,
                             min_hits=SORT_MIN_HITS,
                             iou_threshold=SORT_IOU)

        # --- Task3 控制器 ---
        self._task3 = Task3Controller()
        self._Target = Target
        self._class_filter = set(GEOM_CLASS_FILTER)

    def step(self, frame, frame_id=0, front_distance=99.9):
        # ① 检测: detect() 返回 (dets, cls)
        #    dets: (N,5) = [x1, y1, x2, y2, score]   (xyxy 像素坐标)
        #    cls:  (N,)  = COCO 类号
        dets, cls = self._yolo.detect(frame)

        # ② 类别过滤 (只留 person)
        if len(dets) > 0:
            mask = np.array([int(c) in self._class_filter for c in cls], dtype=bool)
            dets = dets[mask]

        # ③ SORT 跟踪 → tracks: (M,5) = [x1, y1, x2, y2, track_id]
        tracks = self._tracker.update(dets) if len(dets) > 0 else self._tracker.update()

        # ④ tracks → Task3 的 Target 对象。
        #    Task3 的 Target 用 *左上角* 坐标 (lateral = bbox_x + bbox_w/2 算中心),
        #    而 tracks 给的就是 xyxy, 所以 bbox_x=x1, bbox_y=y1, 不用换算。
        #    confidence: Task3 的 process() 不用它, 填 1.0 即可。
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
                "id": int(tid),
                "cx": float((x1 + x2) / 2.0), "cy": float((y1 + y2) / 2.0),
                "w": float(w), "h": float(h), "conf": 1.0,
            })

        # ⑤ Task3 决策 (吃 Target 列表 + 超声波前向距离, <0.5m 内部急停兜底)
        task3_cmd = self._task3.process(
            targets, frame_id=frame_id, front_distance=front_distance
        )
        return preview, task3_cmd


# ============================================================
# 相机
# ============================================================

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
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)     # 丢积压帧, 永远拿最新
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[cam] Opened camera {camera_id}: {w}x{h}", file=sys.stderr)
    return cap


# ============================================================
# MJPEG 浏览器预览 (合并视图: YOLO 框 + OCR 框 + 决策 HUD)
# ============================================================

class FrameBuffer:
    def __init__(self):
        self._lock = threading.Lock()
        self._jpeg = None
        self._fid = -1

    def update(self, jpeg, fid):
        with self._lock:
            self._jpeg, self._fid = jpeg, fid

    def get(self):
        with self._lock:
            return self._jpeg, self._fid


FRAME_BUFFER = FrameBuffer()

HTML_PAGE = """<!doctype html><html><head><title>Orchestrator</title>
<style>body{background:#1a1a1a;color:#ddd;font-family:sans-serif;margin:0;
padding:16px;text-align:center}h2{margin:6px}img{max-width:100%;border:2px
solid #444;border-radius:8px}.i{color:#888;font-size:13px;margin-top:6px}</style>
</head><body><h2>S100 Orchestrator · YOLO + VLM + OCR</h2>
<img src="/stream.mjpg"/><div class="i">geometric (every frame) · semantic (bg) · ocr (gated)</div>
</body></html>"""


class StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
            return
        if self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            last = -1
            try:
                while True:
                    jpeg, fid = FRAME_BUFFER.get()
                    if jpeg is None or fid == last:
                        time.sleep(0.02)
                        continue
                    last = fid
                    self.wfile.write(b"--FRAME\r\n")
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


def draw_overlay(frame, dets, ocr_results, result, fps, sem):
    """合并视图叠加。所有绘制都容错, 任何字段缺失都不崩。"""
    out = frame.copy()

    # YOLO 框 (绿)
    for d in dets:
        try:
            x1 = int(d["cx"] - d["w"] / 2); y1 = int(d["cy"] - d["h"] / 2)
            x2 = int(d["cx"] + d["w"] / 2); y2 = int(d["cy"] + d["h"] / 2)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 230, 0), 2)
            cv2.putText(out, f"#{d['id']}", (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 0), 1)
        except Exception:
            continue

    # OCR 框 (橙)
    for r in ocr_results:
        try:
            pts = np.array(r["box"], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], True, (0, 180, 255), 2)
            lbl = (r.get("text", "") or "?").encode("ascii", "replace").decode()
            cv2.putText(out, lbl[:16], (r["box"][0][0], max(12, r["box"][0][1] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1)
        except Exception:
            continue

    # 决策 HUD (顶栏)
    cv2.rectangle(out, (0, 0), (FRAME_WIDTH, 46), (25, 25, 25), -1)
    src = getattr(result, "source", "?")
    src = getattr(src, "value", src)
    reason = getattr(result, "reason", "")
    supp = getattr(result, "nav2_suppressed", False)
    line1 = f"fps {fps:4.1f} | src {src} | nav2_supp {supp}"
    cv2.putText(out, line1, (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    line2 = (f"sem person={sem.get('person_present')} crowd={sem.get('crowded')} "
             f"age={sem.get('age', -1):.0f}s | {reason[:34]}")
    cv2.putText(out, line2, (6, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 255), 1)
    return out


# ============================================================
# 主编排
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="S100 单进程编排器")
    ap.add_argument("--camera", type=int, default=CAMERA_ID)
    ap.add_argument("--width", type=int, default=FRAME_WIDTH)
    ap.add_argument("--height", type=int, default=FRAME_HEIGHT)
    ap.add_argument("--no-vlm", action="store_true", help="关语义流(只 YOLO+OCR)")
    ap.add_argument("--ocr-every", type=int, default=15,
                    help="每 N 帧跑一次 OCR; 0=关。每帧 OCR 会拖死 fps")
    ap.add_argument("--no-stream", action="store_true", help="不开浏览器预览")
    ap.add_argument("--port", type=int, default=PREVIEW_PORT)
    args = ap.parse_args()

    print("[init] 几何流 (YOLO + Task3) ...", file=sys.stderr)
    geom = GeometryStream()
    
    from ultrasonic import Ultrasonic
    us = Ultrasonic(trig_pin=11, echo_pin=13)
    us.start()

    ocr = None
    if args.ocr_every > 0:
        print("[init] OCR ...", file=sys.stderr)
        ocr = GlobalOcrReader(priority=OCR_PRIORITY)

    print(f"[init] Task6 仲裁器 (VLM={'off' if args.no_vlm else 'on'}) ...",
          file=sys.stderr)
    # Task6Arbitrator 内部创建并 start() VLMSemanticStream 的后台线程
    arbitrator = Task6Arbitrator(enable_vlm=not args.no_vlm)

    http_server = None
    if not args.no_stream:
        try:
            http_server = start_http_server(args.port)
            print(f"[http] 浏览器打开: http://{get_local_ip()}:{args.port}",
                  file=sys.stderr)
            print(f"[http] (Tailscale 用 100.x 那个 IP)", file=sys.stderr)
        except OSError as e:
            print(f"[http] 预览启动失败: {e}", file=sys.stderr)

    cap = open_camera(args.camera, args.width, args.height)

    stop = {"v": False}
    signal.signal(signal.SIGINT, lambda s, f: stop.update(v=True))

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    n = 0
    cur_fps = 0.0
    t_fps, n_fps = time.time(), 0
    last_log = time.time()
    ocr_results = []
    last_sem = {"person_present": None, "crowded": None, "age": -1.0}

    print("[run] 主循环启动。Ctrl-C 退出。", file=sys.stderr)
    try:
        while not stop["v"]:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            t0 = time.time()

            # ── 1. 几何流 (每帧同步) ──────────────────────────────
            # ★ 超声波: 接上 HC-SR04 读数(米)。现在没接, 用 99.9 = "前方无障碍",
            #   Task3 的超声波急停分支就不会误触发。ROS 接入后换成订阅到的距离。
            front_distance = us.distance
            dets, task3_cmd = geom.step(frame, frame_id=n,
                                        front_distance=front_distance)
            yolo_ms = (time.time() - t0) * 1000

            # ── 2. 语义流 (推最新帧, VLM 在后台自己跑) ─────────────
            arbitrator.update_frame(frame)

            # ── 3. OCR (触发式: 每 ocr_every 帧。真实部署里换成
            #         "靠近门口才读门牌"的高层触发) ──────────────────
            if ocr is not None and n % args.ocr_every == 0:
                t_ocr = time.time()
                ocr_results = ocr.read(frame)
                ocr_ms = (time.time() - t_ocr) * 1000
            else:
                ocr_ms = 0.0

            # ── 4. 仲裁 (几何 + 语义 + Nav2) ─────────────────────
            # ★ ROS 接入后: nav2_cmd 换成订阅到的 Nav2 指令。现在没接, 传 None,
            #   仲裁器走 "Task3 安全状态 / VLM 拥挤" 分支即可。
            #   传进去的是 Task3 的 MotorCommand (仲裁器读它的 .state 判断是否压制
            #   Nav2)。若你 arbitrate() 期望的是 RobotState, 改成 task3_cmd.state。
            nav2_cmd = None
            result = arbitrator.arbitrate(task3_cmd, nav2_cmd)

            # ── 5. 输出 ──────────────────────────────────────────
            # ★ ROS 接入后: 这里把 result.command publish 成 /cmd_vel 或你的
            #   ControlCommand topic。现在没接, 打日志验证逻辑。
            final_cmd = getattr(result, "command", None)

            # ── 6. 取语义状态供 HUD ──────────────────────────────
            # get_semantic_state() 在 VLM 关闭 或 语义过期(>MAX_SEMANTIC_AGE)时
            # 返回 None。只在拿到 dict 时更新, 否则保持上次/初始占位, 避免 .get() 崩。
            sem_getter = getattr(arbitrator, "get_semantic_state", None)
            if callable(sem_getter):
                try:
                    s = sem_getter()
                    if s is not None:
                        last_sem = s
                except Exception:
                    pass

            # ── 7. 预览 ──────────────────────────────────────────
            if not args.no_stream:
                vis = draw_overlay(frame, dets, ocr_results, result, cur_fps, last_sem)
                ok, jpeg = cv2.imencode(".jpg", vis, encode_params)
                if ok:
                    FRAME_BUFFER.update(jpeg.tobytes(), n)

            # ── 8. 统计 ──────────────────────────────────────────
            n += 1
            now = time.time()
            if now - t_fps >= 0.5:
                cur_fps = (n - n_fps) / (now - t_fps)
                t_fps, n_fps = now, n
            if now - last_log >= 1.0:
                texts = [r.get("text", "") for r in ocr_results]
                print(f"[stats] f={n} fps={cur_fps:.1f} yolo={yolo_ms:.0f}ms "
                      f"ocr={ocr_ms:.0f}ms dets={len(dets)} "
                      f"src={getattr(getattr(result,'source',''),'value',result.source if hasattr(result,'source') else '?')} "
                      f"sem={last_sem.get('person_present')}/{last_sem.get('crowded')} "
                      f"cmd={final_cmd} texts={texts}", file=sys.stderr)
                last_log = now

    finally:
        stop["v"] = True
        cap.release()
        try:
            arbitrator.stop()        # 关 VLM 后台线程
        except Exception:
            pass
        if http_server:
            http_server.shutdown()
        print("[run] 已退出, 资源已释放。", file=sys.stderr)


if __name__ == "__main__":
    main()
