"""
bpu_perception_stream.py
------------------------
bpu_perception.py 的扩展版本: 增加浏览器 MJPEG 预览。

S100 SSH 场景下,用浏览器实时看带 bbox 的画面,比 X11 forwarding 快得多。

用法:
    # 启动 (会同时输出 JSONL 和开 HTTP 服务)
    python3 bpu_perception_stream.py --output /tmp/visual.jsonl --port 8080

    # Mac 浏览器打开:
    #   http://<s100_ip>:8080
    #   例如: http://100.110.96.7:8080

    # 只看视频不录 JSONL: 加 --no-output
    python3 bpu_perception_stream.py --port 8080 --no-output

JSONL schema 与 bpu_perception.py 一致 (generator schema)。
HTTP 服务为只读 MJPEG stream + 简单 HTML 页面,无任何写操作,只在本地网络可用。
"""

import argparse
import json
import os
import signal
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


# ============================================================
# 路径配置 (与 bpu_perception.py 保持一致)
# ============================================================

UTILS_PATH = "/app/pydev_demo"
DEFAULT_MODEL = "/opt/hobot/model/s100/basic/yolo11m_detect_nashe_640x640_nv12.hbm"

if UTILS_PATH not in sys.path:
    sys.path.insert(0, UTILS_PATH)

import hbm_runtime
from utils import preprocess_utils as preprocess
from utils import postprocess_utils as postprocess

from sort_tracker import Sort


DEFAULT_CLASSES = [0,24,28,39,56,73]


# ============================================================
# BPU 检测器 (与 bpu_perception.py 完全一致,直接复制)
# ============================================================

class BpuYoloDetector:
    def __init__(self, model_path: str, score_thres: float = 0.25,
                 nms_thres: float = 0.45, classes_num: int = 80):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"BPU model not found: {model_path}")

        print(f"[bpu] Loading model: {model_path}", file=sys.stderr)
        self.model = hbm_runtime.HB_HBMRuntime(model_path)

        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]
        self.output_quants = self.model.output_quants[self.model_name]

        self.input_H = self.input_shapes[self.input_names[0]][1]
        self.input_W = self.input_shapes[self.input_names[0]][2]

        self.score_thres = score_thres
        self.nms_thres = nms_thres
        self.classes_num = classes_num
        self.conf_thres_raw = -np.log(1.0 / score_thres - 1.0)
        self.resize_type = 1
        self.reg = 16
        self.strides = [8, 16, 32]
        self.anchor_sizes = [80, 40, 20]
        self.weights_static = np.arange(
            self.reg, dtype=np.float32)[np.newaxis, np.newaxis, :]

        print(f"[bpu] Model loaded. Input: {self.input_W}x{self.input_H}, "
              f"outputs: {len(self.output_names)} tensors", file=sys.stderr)

    def set_scheduling_params(self, priority=None, bpu_cores=None):
        kwargs = {}
        if priority is not None:
            kwargs["priority"] = {self.model_name: priority}
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {self.model_name: bpu_cores}
        if kwargs:
            self.model.set_scheduling_params(**kwargs)

    def pre_process(self, img):
        resize_img = preprocess.resized_image(
            img, self.input_W, self.input_H, self.resize_type)
        y, uv = preprocess.bgr_to_nv12_planes(resize_img)
        return {self.model_name: {
            self.input_names[0]: y,
            self.input_names[1]: uv,
        }}

    def forward(self, input_tensor):
        return self.model.run(input_tensor)[self.model_name]

    def post_process(self, outputs, img_w, img_h):
        all_bboxes, all_scores, all_ids = [], [], []
        fp32_outputs = postprocess.dequantize_outputs(outputs, self.output_quants)

        for i, (stride, anchor_size) in enumerate(
                zip(self.strides, self.anchor_sizes)):
            cls_key = self.output_names[2 * i]
            box_key = self.output_names[2 * i + 1]
            scores, ids, valid_idx = postprocess.filter_classification(
                fp32_outputs[cls_key], self.conf_thres_raw)
            bboxes = postprocess.decode_boxes(
                fp32_outputs[box_key], valid_idx,
                anchor_size, stride, self.weights_static)
            all_bboxes.append(bboxes)
            all_scores.append(scores)
            all_ids.append(ids)

        if not all_bboxes or sum(len(b) for b in all_bboxes) == 0:
            return (np.empty((0, 4), dtype=np.float32),
                    np.empty((0,), dtype=np.int64),
                    np.empty((0,), dtype=np.float32))

        bboxes = np.concatenate(all_bboxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        ids = np.concatenate(all_ids, axis=0)

        keep = postprocess.NMS(bboxes, scores, ids, self.nms_thres)
        if len(keep) == 0:
            return (np.empty((0, 4), dtype=np.float32),
                    np.empty((0,), dtype=np.int64),
                    np.empty((0,), dtype=np.float32))

        xyxy = postprocess.scale_coords_back(
            bboxes[keep], img_w, img_h,
            self.input_W, self.input_H, self.resize_type)
        return xyxy, ids[keep], scores[keep]

    def detect(self, img_bgr):
        h, w = img_bgr.shape[:2]
        inp = self.pre_process(img_bgr)
        outs = self.forward(inp)
        xyxy, cls, score = self.post_process(outs, w, h)
        if len(xyxy) == 0:
            return np.empty((0, 5), dtype=np.float32), np.empty((0,), dtype=np.int64)
        dets = np.concatenate([xyxy, score[:, None]], axis=1).astype(np.float32)
        return dets, cls.astype(np.int64)


# ============================================================
# MJPEG 推流服务器 (用 Python 标准库,无外部依赖)
# ============================================================

class FrameBuffer:
    """线程安全的最新帧缓存. 主线程写, HTTP 线程读."""
    def __init__(self):
        self._lock = threading.Lock()
        self._jpeg_bytes = None
        self._frame_id = -1

    def update(self, jpeg_bytes, frame_id):
        with self._lock:
            self._jpeg_bytes = jpeg_bytes
            self._frame_id = frame_id

    def get(self):
        with self._lock:
            return self._jpeg_bytes, self._frame_id


# 全局帧缓存 (HTTP handler 和主线程共享)
FRAME_BUFFER = FrameBuffer()


HTML_PAGE = """<!doctype html>
<html><head><title>BPU Perception Stream</title>
<style>
  body { background:#222; color:#ddd; font-family:sans-serif;
         margin:0; padding:20px; text-align:center; }
  h2 { margin: 8px 0; }
  img { max-width:100%; border:2px solid #444; border-radius:8px; }
  .info { color:#999; font-size:14px; margin-top:8px; }
</style></head>
<body>
  <h2>BPU Perception Stream</h2>
  <img src="/stream.mjpg" alt="live stream"/>
  <div class="info">RDK S100 · YOLO + SORT · refresh page if stalled</div>
</body></html>
"""


class StreamHandler(BaseHTTPRequestHandler):
    """处理 / (HTML) 和 /stream.mjpg (MJPEG)."""

    def log_message(self, format, *args):
        # 静默 HTTP 访问日志,避免干扰主进程 stderr
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
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            last_id = -1
            try:
                while True:
                    jpeg, fid = FRAME_BUFFER.get()
                    if jpeg is None:
                        time.sleep(0.02)
                        continue
                    # 只在帧更新时推送,避免重复发同一帧
                    if fid == last_id:
                        time.sleep(0.01)
                        continue
                    last_id = fid
                    self.wfile.write(b"--FRAME\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpeg)))
                    self.end_headers()
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                # 浏览器关闭连接,正常退出
                return
            return

        self.send_response(404)
        self.end_headers()


def start_http_server(port: int):
    """在后台线程启动 HTTP 服务."""
    server = ThreadingHTTPServer(("0.0.0.0", port), StreamHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def get_local_ip():
    """猜测本机 IP,给用户看连接 URL."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "<your-s100-ip>"


# ============================================================
# 异步摄像头读取
# ============================================================

class CameraReader:
    """
    后台线程持续 cap.read(),只保留最新一帧,旧帧丢弃。
    主线程通过 read_latest() 拿帧,永不阻塞。

    这样摄像头读取(I/O) 和 BPU 推理(计算) 真正并行,
    消除"主线程串行等待 cap.read()"导致的 fps 上限。
    同时丢弃积压帧,保证拿到的永远是最新画面,降低控制延迟。
    """

    def __init__(self, camera_id: int, width: int, height: int):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap = None
        self._latest_frame = None
        self._latest_seq = 0           # 自增序号,主线程用来判断是否拿到新帧
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._captured_count = 0       # 摄像头实际抓到的帧数(用于诊断真实采集速率)
        self._t_start = 0.0

    def open(self):
        # 强制 V4L2 后端
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_id}")

        # 顺序: FOURCC -> 宽高 -> FPS
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        fcc_s = "".join([chr((fcc >> (8 * i)) & 0xFF) for i in range(4)])
        self.width, self.height = w, h
        print(f"[cam] Opened camera {self.camera_id}: {w}x{h} fourcc={fcc_s}",
              file=sys.stderr)

        # 启动后台读帧线程
        self._t_start = time.time()
        self._thread = threading.Thread(
            target=self._loop, name="camera-reader", daemon=True)
        self._thread.start()

    def _loop(self):
        """后台线程: 不停 read,把最新帧塞到 _latest_frame,旧帧丢弃."""
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.005)
                continue
            with self._lock:
                self._latest_frame = frame
                self._latest_seq += 1
                self._captured_count += 1

    def read_latest(self, last_seq: int = -1, timeout: float = 1.0
                    ) -> Tuple[Optional[np.ndarray], int]:
        """
        拿最新一帧。
        last_seq: 调用方上次拿到的序号,只在有新帧时才返回(避免重复处理同一帧)。
        返回 (frame, seq). 超时返回 (None, last_seq)。
        """
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._lock:
                if self._latest_seq > last_seq and self._latest_frame is not None:
                    return self._latest_frame.copy(), self._latest_seq
            time.sleep(0.002)
        return None, last_seq

    def capture_fps(self) -> float:
        """后台线程的实测采集 fps(摄像头吐帧速率,不是主线程消费速率)."""
        elapsed = time.time() - self._t_start
        return self._captured_count / elapsed if elapsed > 0 else 0.0

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()


# ============================================================
# 主管线 (与 bpu_perception.py 相同,加上帧推送到 FRAME_BUFFER)
# ============================================================

class PerceptionPipeline:
    def __init__(self, camera_id=0, model_path=DEFAULT_MODEL,
                 classes=None, score_thres=0.4, nms_thres=0.45,
                 sort_max_age=5, sort_min_hits=3, sort_iou=0.3,
                 image_size=(640, 480), priority=0, bpu_cores=(0,),
                 jpeg_quality=70):
        self.camera_id = camera_id
        self.classes_filter = set(classes) if classes else set(DEFAULT_CLASSES)
        self.image_size = image_size
        self.jpeg_quality = jpeg_quality

        self.detector = BpuYoloDetector(
            model_path=model_path,
            score_thres=score_thres,
            nms_thres=nms_thres,
        )
        self.detector.set_scheduling_params(
            priority=priority, bpu_cores=list(bpu_cores))

        self.tracker = Sort(max_age=sort_max_age,
                            min_hits=sort_min_hits,
                            iou_threshold=sort_iou)

        self.camera = None
        self.last_seq = -1
        self.frame_id = 0
        self.t0 = None

    def open_camera(self):
        self.camera = CameraReader(self.camera_id,
                                   self.image_size[0],
                                   self.image_size[1])
        self.camera.open()
        self.image_size = (self.camera.width, self.camera.height)

    def close(self):
        if self.camera is not None:
            self.camera.close()

    def _filter_by_class(self, dets, cls):
        if len(dets) == 0:
            return dets, cls
        mask = np.array([int(c) in self.classes_filter for c in cls], dtype=bool)
        return dets[mask], cls[mask]

    def _build_record(self, tracks, dets):
        targets = []
        for trk in tracks:
            x1, y1, x2, y2, tid = trk
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1
            conf = 1.0
            if len(dets) > 0:
                best_iou = 0.0
                for det in dets:
                    iou = self._iou(trk[:4], det[:4])
                    if iou > best_iou:
                        best_iou = iou
                        conf = float(det[4])
            targets.append({
                "target_id": int(tid),
                "pixel_x": float(cx),
                "pixel_y": float(cy),
                "bbox_width": float(w),
                "bbox_height": float(h),
                "confidence": float(conf),
            })
        if self.t0 is None:
            self.t0 = time.time()
        return {
            "frame_id": self.frame_id,
            "timestamp": round(time.time() - self.t0, 4),
            "targets": targets,
        }

    def _draw_overlay(self, frame, record, fps):
        """在 frame 上画 bbox + 文字,返回处理后的 BGR 图."""
        out = frame.copy()
        for tgt in record["targets"]:
            cx, cy = tgt["pixel_x"], tgt["pixel_y"]
            w, h = tgt["bbox_width"], tgt["bbox_height"]
            x1 = int(cx - w / 2); y1 = int(cy - h / 2)
            x2 = int(cx + w / 2); y2 = int(cy + h / 2)
            label = f"#{tgt['target_id']} {tgt['confidence']:.2f}"
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out, (x1, max(0, y1 - th - 8)),
                          (x1 + tw + 6, y1), (0, 255, 0), -1)
            cv2.putText(out, label, (x1 + 3, max(th + 2, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        # 左上角 HUD
        hud = f"frame {record['frame_id']} | fps {fps:.1f} | targets {len(record['targets'])}"
        cv2.rectangle(out, (0, 0), (340, 26), (0, 0, 0), -1)
        cv2.putText(out, hud, (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        return out

    def step(self, current_fps=0.0):
        """
        处理一帧。
        返回 (record, frame_with_overlay, dropped_frames).
          - record: 同 generator schema, 可直接写 JSONL
          - dropped_frames: 摄像头从上次到这次抓了多少帧, 减去本帧, 即丢弃的旧帧数.
                            正常应接近 0, 大于 0 说明主线程消费跟不上.
        """
        frame, seq = self.camera.read_latest(last_seq=self.last_seq, timeout=1.0)
        if frame is None:
            return None, None, 0
        dropped = seq - self.last_seq - 1 if self.last_seq >= 0 else 0
        self.last_seq = seq

        dets, cls = self.detector.detect(frame)
        dets, cls = self._filter_by_class(dets, cls)
        tracks = self.tracker.update(dets) if len(dets) > 0 else self.tracker.update()
        record = self._build_record(tracks, dets)
        overlay = self._draw_overlay(frame, record, current_fps)
        self.frame_id += 1
        return record, overlay, dropped

    @staticmethod
    def _iou(b1, b2):
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        w = max(0.0, x2 - x1); h = max(0.0, y2 - y1)
        inter = w * h
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter / (a1 + a2 - inter + 1e-9)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="RDK S100 BPU + SORT + MJPEG browser preview")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--classes", type=int, nargs="+", default=None)
    parser.add_argument("--score-thres", type=float, default=0.4)
    parser.add_argument("--nms-thres", type=float, default=0.45)
    parser.add_argument("--output", type=str, default=None,
                        help="JSONL output path. Default: /tmp/stream.jsonl")
    parser.add_argument("--no-output", action="store_true",
                        help="Don't write JSONL, only stream video")
    parser.add_argument("--port", type=int, default=8080,
                        help="HTTP server port for browser preview")
    parser.add_argument("--no-stream", action="store_true",
                        help="Disable HTTP server (behaves like bpu_perception.py)")
    parser.add_argument("--jpeg-quality", type=int, default=70,
                        help="MJPEG quality 1-100 (lower = less bandwidth)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--sort-max-age", type=int, default=5)
    parser.add_argument("--sort-min-hits", type=int, default=3)
    parser.add_argument("--sort-iou", type=float, default=0.3)
    parser.add_argument("--priority", type=int, default=0)
    parser.add_argument("--bpu-cores", type=int, nargs="+", default=[0])
    args = parser.parse_args()

    # 启动 HTTP 服务
    http_server = None
    if not args.no_stream:
        try:
            http_server = start_http_server(args.port)
            ip = get_local_ip()
            print(f"[http] Stream server started.", file=sys.stderr)
            print(f"[http] Open in browser:  http://{ip}:{args.port}",
                  file=sys.stderr)
        except OSError as e:
            print(f"[http] Failed to start server on port {args.port}: {e}",
                  file=sys.stderr)
            print(f"[http] Continuing without browser preview.", file=sys.stderr)

    # JSONL 输出
    output_file = None
    if not args.no_output:
        path = args.output or "/tmp/stream.jsonl"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        output_file = open(path, "w", buffering=1)
        print(f"[main] Writing JSONL to: {path}", file=sys.stderr)

    pipeline = PerceptionPipeline(
        camera_id=args.camera,
        model_path=args.model_path,
        classes=args.classes,
        score_thres=args.score_thres,
        nms_thres=args.nms_thres,
        sort_max_age=args.sort_max_age,
        sort_min_hits=args.sort_min_hits,
        sort_iou=args.sort_iou,
        image_size=(args.width, args.height),
        priority=args.priority,
        bpu_cores=args.bpu_cores,
        jpeg_quality=args.jpeg_quality,
    )

    stop_flag = {"stop": False}

    def sigint_handler(sig, frame):
        stop_flag["stop"] = True
        print("\n[main] SIGINT received, stopping...", file=sys.stderr)

    signal.signal(signal.SIGINT, sigint_handler)

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]

    try:
        pipeline.open_camera()
        t_start = time.time()
        last_log = t_start
        last_fps_calc = t_start
        frames_at_last_calc = 0
        frames_logged = 0
        current_fps = 0.0

        while not stop_flag["stop"]:
            record, overlay, dropped = pipeline.step(current_fps=current_fps)
            if record is None:
                time.sleep(0.05)
                continue

            # JSONL
            if output_file is not None:
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            frames_logged += 1

            # 推到 HTTP buffer
            if not args.no_stream and overlay is not None:
                ok, jpeg = cv2.imencode(".jpg", overlay, encode_params)
                if ok:
                    FRAME_BUFFER.update(jpeg.tobytes(), record["frame_id"])

            now = time.time()

            # 更新 current_fps (用于 HUD,每 0.5 秒重算一次)
            if now - last_fps_calc >= 0.5:
                current_fps = (frames_logged - frames_at_last_calc) / (now - last_fps_calc)
                last_fps_calc = now
                frames_at_last_calc = frames_logged

            # stderr 心跳: 区分"消费 fps"(主线程处理速率) 和 "采集 fps"(摄像头吐帧速率)
            if now - last_log >= 1.0:
                consume_fps = frames_logged / (now - t_start)
                capture_fps = pipeline.camera.capture_fps()
                n_targets = len(record["targets"])
                print(f"[stats] frame={record['frame_id']} "
                      f"consume_fps={consume_fps:.1f} capture_fps={capture_fps:.1f} "
                      f"targets={n_targets} dropped={dropped}", file=sys.stderr)
                last_log = now

            if args.max_frames > 0 and frames_logged >= args.max_frames:
                print(f"[main] Reached max_frames={args.max_frames}, stopping.",
                      file=sys.stderr)
                break

    finally:
        pipeline.close()
        if output_file is not None:
            output_file.close()
        if http_server is not None:
            http_server.shutdown()
        print("[main] Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
