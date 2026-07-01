"""
combined_perception.py
----------------------
单进程同时跑 YOLO + OCR + SORT,共享一个摄像头。

为什么需要这个:
  Linux 下 /dev/video0 是独占设备,不能两个进程同时打开.
  所以要在一个进程里读一次摄像头,把帧分发给 YOLO 和 OCR 两个模型.

复用关系 (不重复代码):
  - bpu_perception_stream.py 里的 BpuYoloDetector, CameraReader, Sort
  - global_ocr.py 里的 GlobalOcrReader

每帧流程:
  CameraReader → 一帧 BGR
      ├→ YOLO 检测 → SORT 跟踪 → 绿框
      └→ OCR (det+rec)        → 橙框 + 文字

输出:
  - YOLO JSONL: 跟 generator schema 一致 (task3 直接消费)
  - OCR  JSONL: 单独 schema (text + box),不污染 task3
  - 浏览器: 一个页面 (端口默认 8090),YOLO 绿框和 OCR 橙框画在同一画面

用法:
    python3 combined_perception.py
    # 浏览器: http://100.110.96.7:8090

    # 自定义输出:
    python3 combined_perception.py \
        --yolo-out /tmp/yolo.jsonl --ocr-out /tmp/ocr.jsonl --port 8090
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

import cv2
import numpy as np


# ============================================================
# 路径 (与两个源文件保持一致)
# ============================================================

UTILS_PATH = "/app/pydev_demo"
if UTILS_PATH not in sys.path:
    sys.path.insert(0, UTILS_PATH)

# 把 perception 目录加到 sys.path,以便 import bpu_perception_stream / global_ocr
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# 复用现有类
from bpu_perception_stream import (
    BpuYoloDetector, CameraReader, DEFAULT_MODEL as DEFAULT_YOLO_MODEL,
)
from sort_tracker import Sort
from global_ocr import (
    GlobalOcrReader,
    DEFAULT_DET_MODEL as DEFAULT_OCR_DET_MODEL,
    DEFAULT_REC_MODEL as DEFAULT_OCR_REC_MODEL,
    DEFAULT_LABEL as DEFAULT_OCR_LABEL,
)
from vlm_semantic import VLMSemanticStream


DEFAULT_YOLO_CLASSES = [0]  # person


# ============================================================
# MJPEG 推流 (跟单模块版本同款,但只有这一个,所以用 module-level buffer)
# ============================================================

class FrameBuffer:
    def __init__(self):
        self._lock = threading.Lock()
        self._jpeg = None
        self._fid = -1

    def update(self, jpeg, fid):
        with self._lock:
            self._jpeg = jpeg
            self._fid = fid

    def get(self):
        with self._lock:
            return self._jpeg, self._fid


FRAME_BUFFER = FrameBuffer()


HTML_PAGE = """<!doctype html><html><head><title>Combined Perception</title>
<style>body{background:#222;color:#ddd;font-family:sans-serif;margin:0;padding:20px;
text-align:center}h2{margin:8px 0}img{max-width:100%;border:2px solid #444;border-radius:8px}
.info{color:#999;font-size:14px;margin-top:8px}
.legend{margin-top:10px;font-size:13px}
.swatch{display:inline-block;width:14px;height:14px;vertical-align:middle;margin:0 4px;border-radius:3px}
.yolo{background:#00ff00}.ocr{background:#ffa500}</style></head>
<body><h2>YOLO + OCR Combined</h2><img src="/stream.mjpg"/>
<div class="legend"><span class="swatch yolo"></span>YOLO(person)
<span class="swatch ocr"></span>OCR(text)</div>
<div class="info">RDK S100 · single camera, two models · UTF-8 text in console</div>
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


# ============================================================
# 主循环逻辑 (跟两个源文件一致的字段处理,合并到一处)
# ============================================================

def filter_yolo_by_class(dets, cls, classes_filter):
    if len(dets) == 0:
        return dets, cls
    mask = np.array([int(c) in classes_filter for c in cls], dtype=bool)
    return dets[mask], cls[mask]


def iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    w = max(0.0, x2 - x1); h = max(0.0, y2 - y1)
    inter = w * h
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-9)


def build_yolo_record(frame_id, t0, tracks, dets):
    """跟 generator schema 一致的 YOLO 输出."""
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
                v = iou(trk[:4], det[:4])
                if v > best_iou:
                    best_iou = v
                    conf = float(det[4])
        targets.append({
            "target_id": int(tid),
            "pixel_x": float(cx),
            "pixel_y": float(cy),
            "bbox_width": float(w),
            "bbox_height": float(h),
            "confidence": float(conf),
        })
    return {
        "frame_id": frame_id,
        "timestamp": round(time.time() - t0, 4),
        "targets": targets,
    }


def build_ocr_record(frame_id, t0, ocr_results):
    """OCR 独立 schema (跟 task3 schema 分开,不污染)."""
    texts = []
    for r in ocr_results:
        texts.append({
            "text": r["text"],
            "box": r["box"],         # 四点框
            "center": r["center"],
        })
    return {
        "frame_id": frame_id,
        "timestamp": round(time.time() - t0, 4),
        "texts": texts,
    }


def draw_combined_overlay(frame, yolo_record, ocr_results, fps,
                          vlm_state=None):
    """在同一画面上画 YOLO 绿框 + OCR 橙框 + VLM 语义 HUD."""
    out = frame.copy()

    # YOLO: 绿色
    for tgt in yolo_record["targets"]:
        cx, cy = tgt["pixel_x"], tgt["pixel_y"]
        w, h = tgt["bbox_width"], tgt["bbox_height"]
        x1 = int(cx - w / 2); y1 = int(cy - h / 2)
        x2 = int(cx + w / 2); y2 = int(cy + h / 2)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"#{tgt['target_id']} {tgt['confidence']:.2f}"
        cv2.putText(out, label, (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

    # OCR: 橙色
    for r in ocr_results:
        pts = np.array(r["box"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], isClosed=True,
                      color=(0, 165, 255), thickness=2)  # BGR=橙
        label = r["text"][:20] if r["text"] else ""
        try:
            ascii_label = label.encode("ascii", "replace").decode("ascii")
        except Exception:
            ascii_label = "?"
        if ascii_label:
            cv2.putText(out, ascii_label,
                        (r["box"][0][0], max(15, r["box"][0][1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

    # HUD 第一行: fps / YOLO / OCR
    n_y = len(yolo_record["targets"])
    n_o = len(ocr_results)
    hud = f"fps {fps:.1f} | YOLO {n_y} | OCR {n_o}"

    # HUD 第二行: VLM 语义状态
    if vlm_state is not None and vlm_state["age"] < 60:
        p = "Y" if vlm_state["person_present"] else "N"
        c = "Y" if vlm_state["crowded"] else "N"
        o = "Y" if vlm_state["obstacle_present"] else "N"
        vlm_hud = f"VLM: person={p} crowded={c} obstacle={o} | age={vlm_state['age']:.1f}s"
        hud_h = 48  # 两行高度
    else:
        vlm_hud = None
        hud_h = 26  # 单行

    cv2.rectangle(out, (0, 0), (520, hud_h), (0, 0, 0), -1)
    cv2.putText(out, hud, (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    if vlm_hud:
        cv2.putText(out, vlm_hud, (6, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1)
    return out


# ============================================================
# main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Combined YOLO+OCR perception (single camera)")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--port", type=int, default=8090,
                        help="HTTP port (default 8090 to avoid 8080/8081)")
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--jpeg-quality", type=int, default=70)
    parser.add_argument("--max-frames", type=int, default=0)

    # YOLO
    parser.add_argument("--yolo-model", default=DEFAULT_YOLO_MODEL)
    parser.add_argument("--yolo-classes", type=int, nargs="+", default=None)
    parser.add_argument("--yolo-score-thres", type=float, default=0.4)
    parser.add_argument("--yolo-nms-thres", type=float, default=0.45)
    parser.add_argument("--sort-max-age", type=int, default=5)
    parser.add_argument("--sort-min-hits", type=int, default=3)
    parser.add_argument("--sort-iou", type=float, default=0.3)
    parser.add_argument("--yolo-out", type=str, default="/tmp/yolo.jsonl",
                        help="YOLO JSONL output (generator schema). Use 'none' to disable.")

    # OCR
    parser.add_argument("--ocr-det-model", default=DEFAULT_OCR_DET_MODEL)
    parser.add_argument("--ocr-rec-model", default=DEFAULT_OCR_REC_MODEL)
    parser.add_argument("--ocr-label", default=DEFAULT_OCR_LABEL)
    parser.add_argument("--ocr-threshold", type=float, default=0.5)
    parser.add_argument("--ocr-ratio-prime", type=float, default=2.7)
    parser.add_argument("--ocr-out", type=str, default="/tmp/ocr.jsonl",
                        help="OCR JSONL output. Use 'none' to disable.")

    # BPU 调度
    parser.add_argument("--priority", type=int, default=0)
    parser.add_argument("--bpu-cores", type=int, nargs="+", default=[0])

    # VLM
    parser.add_argument("--no-vlm", action="store_true",
                        help="禁用 VLM 语义流（节省资源）")

    args = parser.parse_args()

    yolo_classes = set(args.yolo_classes) if args.yolo_classes else set(DEFAULT_YOLO_CLASSES)

    # ---- 初始化 ----
    print("[init] Loading YOLO detector...", file=sys.stderr)
    yolo = BpuYoloDetector(
        model_path=args.yolo_model,
        score_thres=args.yolo_score_thres,
        nms_thres=args.yolo_nms_thres,
    )
    yolo.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)

    tracker = Sort(max_age=args.sort_max_age,
                   min_hits=args.sort_min_hits,
                   iou_threshold=args.sort_iou)

    print("[init] Loading OCR reader (det+rec)...", file=sys.stderr)
    ocr = GlobalOcrReader(
        det_model_path=args.ocr_det_model,
        rec_model_path=args.ocr_rec_model,
        label_file=args.ocr_label,
        ratio_prime=args.ocr_ratio_prime,
        threshold=args.ocr_threshold,
        priority=args.priority,
        bpu_cores=args.bpu_cores,
    )

    # HTTP
    http_server = None

    # VLM 语义流
    vlm_stream = None
    if not args.no_vlm:
        print("[init] Starting VLM semantic stream...", file=sys.stderr)
        vlm_stream = VLMSemanticStream()
        vlm_stream.start()

    if not args.no_stream:
        try:
            http_server = start_http_server(args.port)
            ip = get_local_ip()
            print(f"[http] Stream server started.", file=sys.stderr)
            print(f"[http] Open in browser: http://{ip}:{args.port}", file=sys.stderr)
            print(f"[http] (via Tailscale use your 100.x.x.x IP)", file=sys.stderr)
        except OSError as e:
            print(f"[http] Failed to start server: {e}", file=sys.stderr)

    # JSONL 文件
    yolo_fp = None
    if args.yolo_out and args.yolo_out.lower() != "none":
        os.makedirs(os.path.dirname(args.yolo_out) or ".", exist_ok=True)
        yolo_fp = open(args.yolo_out, "w", buffering=1)
        print(f"[main] YOLO JSONL → {args.yolo_out}", file=sys.stderr)

    ocr_fp = None
    if args.ocr_out and args.ocr_out.lower() != "none":
        os.makedirs(os.path.dirname(args.ocr_out) or ".", exist_ok=True)
        ocr_fp = open(args.ocr_out, "w", buffering=1)
        print(f"[main] OCR JSONL → {args.ocr_out}", file=sys.stderr)

    # 摄像头 (异步读帧)
    camera = CameraReader(args.camera, args.width, args.height)
    camera.open()

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]

    stop = {"v": False}
    def sigint_handler(sig, frame):
        stop["v"] = True
        print("\n[main] SIGINT received, stopping...", file=sys.stderr)
    signal.signal(signal.SIGINT, sigint_handler)

    t0 = time.time()
    last_seq = -1
    frame_id = 0
    last_log = t0
    cur_fps = 0.0
    last_fps_t = t0
    last_fps_n = 0

    try:
        while not stop["v"]:
            frame, seq = camera.read_latest(last_seq=last_seq, timeout=1.0)
            if frame is None:
                time.sleep(0.05)
                continue
            dropped = seq - last_seq - 1 if last_seq >= 0 else 0
            last_seq = seq

            # --- YOLO + SORT ---
            t_yolo = time.time()
            dets, cls = yolo.detect(frame)
            dets, cls = filter_yolo_by_class(dets, cls, yolo_classes)
            if len(dets) > 0:
                tracks = tracker.update(dets)
            else:
                tracks = tracker.update()
            yolo_record = build_yolo_record(frame_id, t0, tracks, dets)
            yolo_ms = (time.time() - t_yolo) * 1000

            # --- OCR ---
            t_ocr = time.time()
            ocr_results = ocr.read(frame)
            ocr_record = build_ocr_record(frame_id, t0, ocr_results)
            ocr_ms = (time.time() - t_ocr) * 1000

            # --- 输出 JSONL ---
            if yolo_fp is not None:
                yolo_fp.write(json.dumps(yolo_record, ensure_ascii=False) + "\n")
            if ocr_fp is not None:
                ocr_fp.write(json.dumps(ocr_record, ensure_ascii=False) + "\n")

            # --- 浏览器预览 ---
            if vlm_stream is not None:
                vlm_stream.update_frame(frame)
            vlm_state = vlm_stream.get_semantic_state() if vlm_stream else None

            if not args.no_stream:
                overlay = draw_combined_overlay(frame, yolo_record, ocr_results,
                                               cur_fps, vlm_state)
                ok, jpeg = cv2.imencode(".jpg", overlay, encode_params)
                if ok:
                    FRAME_BUFFER.update(jpeg.tobytes(), frame_id)

            frame_id += 1

            now = time.time()
            if now - last_fps_t >= 0.5:
                cur_fps = (frame_id - last_fps_n) / (now - last_fps_t)
                last_fps_t = now
                last_fps_n = frame_id
            if now - last_log >= 1.0:
                texts = [r["text"] for r in ocr_results]
                cap_fps = camera.capture_fps()
                vlm_info = ""
                if vlm_state and vlm_state["age"] < 60:
                    vlm_info = (f" | VLM: p={vlm_state['person_present']}"
                                f" c={vlm_state['crowded']}"
                                f" o={vlm_state['obstacle_present']}"
                                f" age={vlm_state['age']:.1f}s")
                print(f"[stats] frame={frame_id} fps={cur_fps:.1f} "
                      f"capture={cap_fps:.1f} dropped={dropped} | "
                      f"yolo={yolo_ms:.0f}ms ({len(yolo_record['targets'])}) "
                      f"ocr={ocr_ms:.0f}ms ({len(texts)}) "
                      f"texts={texts}{vlm_info}", file=sys.stderr)
                last_log = now
            if args.max_frames > 0 and frame_id >= args.max_frames:
                print(f"[main] Reached max_frames={args.max_frames}, stopping.",
                      file=sys.stderr)
                break
    finally:
        camera.close()
        if vlm_stream:
            vlm_stream.stop()
        if yolo_fp:
            yolo_fp.close()
        if ocr_fp:
            ocr_fp.close()
        if http_server:
            http_server.shutdown()
        print("[main] Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
