"""
global_ocr.py
-------------
RDK S100 PaddleOCR (det + rec) 全局文字检测,带浏览器 MJPEG 预览。

基于官方 demo: /app/pydev_demo/08_OCR_sample/01_paddleOCR/paddle_ocr.py
改造为: 摄像头持续读 → 每帧全图 OCR → 浏览器实时看识别结果。

两段流水线:
  1. PaddleOCR_Det (BPU, 640x640 NV12): 找文字区域,输出四点框 + 裁剪图
  2. PaddleOCR_Rec (BPU, 48x320 RGB):  逐个裁剪图认字,CTC 解码成字符串

形式:
  - GlobalOcrReader 类: 可被 stream/其他代码调用
  - main(): 独立跑摄像头 + 浏览器预览

用法:
    # 独立跑,浏览器看
    python3 global_ocr.py
    # 浏览器打开 http://<tailscale-ip>:8081

    # 跑测试图 (不开摄像头,验证流水线)
    python3 global_ocr.py --test-img /app/res/assets/gt_2322.jpg --save out.jpg

    # 当库用:
    from global_ocr import GlobalOcrReader
    reader = GlobalOcrReader()
    results = reader.read(frame_bgr)   # -> [(box4pt, text, ...), ...]

注意:
  - OCR det+rec 后处理较重 (透视矫正 + 逐框 rec + CTC 解码),
    每帧全图 OCR 会比 YOLO 慢很多. 这是"全局每帧"模式,
    实测后再决定要不要改触发式.
  - 端口默认 8081, 避免和 bpu_perception_stream.py 的 8080 冲突,
    两个可以同时跑,各看各的.

依赖:
  - hbm_runtime, pyclipper, PIL, opencv, numpy, scipy
  - demo 的 utils (preprocess/postprocess/common/draw)
  - demo 的 postprocess.rec_postprocess.CTCLabelDecode
"""

import argparse
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ============================================================
# 路径配置 (确认后按需修改这两行)
# ============================================================

# demo 根目录 (utils 在这里)
UTILS_PATH = "/app/pydev_demo"
# OCR demo 目录 (CTCLabelDecode 的 postprocess 包在这里)
OCR_DEMO_PATH = "/app/pydev_demo/08_OCR_sample/01_paddleOCR"

DEFAULT_DET_MODEL = "/opt/hobot/model/s100/basic/cn_PP-OCRv3_det_infer-deploy_640x640_nv12.hbm"
DEFAULT_REC_MODEL = "/opt/hobot/model/s100/basic/cn_PP-OCRv3_rec_infer-deploy_48x320_rgb.hbm"
DEFAULT_LABEL = "/app/res/labels/ppocr_keys_v1.txt"

for p in (UTILS_PATH, OCR_DEMO_PATH):
    if p not in sys.path:
        sys.path.insert(0, p)

import hbm_runtime
import pyclipper
import utils.preprocess_utils as pre_utils
import utils.postprocess_utils as post_utils
import utils.common_utils as common
import utils.draw_utils as draw

# 注意: 不 import demo 的 CTCLabelDecode,因为它所在的文件死 import 了 paddle.
# 我们这里内置一个等价的纯 numpy 实现 (跟 demo 行为一致).


# ============================================================
# 内置 CTC Decoder (PaddleOCR 风格,无 paddle 依赖)
# ============================================================

class CTCLabelDecode:
   

    def __init__(self, character_dict_path=None, use_space_char=False):
        # 字符表第一位是 'blank' (CTC 占位符),会被忽略
        self.character_str = ['blank', '<']
        if character_dict_path is not None:
            with open(character_dict_path, 'rb') as fin:
                for line in fin.readlines():
                    self.character_str.append(
                        line.decode('utf-8').strip('\n').strip('\r\n'))
        if use_space_char:
            self.character_str.append(' ')
        self.character = self.character_str
        self.ignored_tokens = [0]  # 0 = blank

    def __call__(self, preds, labels=None):
        if isinstance(preds, list):
            preds = np.array(preds)
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        return self._decode(preds_idx, preds_prob, is_remove_duplicate=True)

    def _decode(self, text_index, text_prob, is_remove_duplicate=True):
        result_list = []
        for b in range(len(text_index)):
            selection = np.ones(len(text_index[b]), dtype=bool)
            if is_remove_duplicate:
                # CTC 去重: 连续相同的索引只保留一次
                selection[1:] = text_index[b][1:] != text_index[b][:-1]
            char_list = []
            # 跟 demo 一致: 忽略最后 2 个 token
            for i, idx in enumerate(text_index[b][:-2]):
                if idx in self.ignored_tokens or not selection[i]:
                    continue
                if 0 <= idx < len(self.character):
                    char_list.append(self.character[idx])
            text = ''.join(char_list)
            mean_prob = float(np.mean(text_prob[b][:-2])) if len(text_prob[b]) > 2 else 0.0
            result_list.append((text, mean_prob))
        return result_list


# ============================================================
# det 的轮廓膨胀 (从 demo 复制)
# ============================================================

def dilate_contours(contours, ratio_prime: float):
 
    dilated_polys = []
    for poly in contours:
        poly = poly[:, 0, :]
        arc_length = cv2.arcLength(poly, True)
        if arc_length == 0:
            continue
        D_prime = (cv2.contourArea(poly) * ratio_prime / arc_length)
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)

        # pco.Execute 返回 list[list[(x,y)]] (可能 0 个/1 个/多个多边形)
        executed = pco.Execute(D_prime)

        # 只接受恰好 1 个多边形 (跟 demo 原意一致),其他情况跳过
        if not executed or len(executed) != 1:
            continue

        # demo 原版形状: np.array(pco.Execute(D_prime)) 对单多边形是 (1, N, 2)
        # 我们保持一致 (下游 cv2.contourArea / minAreaRect 期望这个形状)
        try:
            dilated_poly = np.array([executed[0]], dtype=np.int_)
        except (ValueError, TypeError):
            continue
        if dilated_poly.size == 0 or dilated_poly.ndim != 3:
            continue

        dilated_polys.append(dilated_poly)
    return dilated_polys


# ============================================================
# 检测模型 (从 demo 的 PaddleOCR_Det 改造)
# ============================================================

class PaddleOcrDet:
    def __init__(self, model_path: str, ratio_prime: float = 2.7,
                 threshold: float = 0.5, min_box_area: float = 100):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"OCR det model not found: {model_path}")
        print(f"[ocr] Loading det model: {model_path}", file=sys.stderr)
        self.model = hbm_runtime.HB_HBMRuntime(model_path)
        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]
        self.output_quants = self.model.output_quants[self.model_name]
        self.input_H = self.input_shapes[self.input_names[0]][1]
        self.input_W = self.input_shapes[self.input_names[0]][2]
        self.ratio_prime = ratio_prime
        self.threshold = threshold
        self.min_box_area = min_box_area

    def set_scheduling_params(self, priority=None, bpu_cores=None):
        kwargs = {}
        if priority is not None:
            kwargs["priority"] = {self.model_name: priority}
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {self.model_name: bpu_cores}
        if kwargs:
            self.model.set_scheduling_params(**kwargs)

    def pre_process(self, img):
        # 注意: resize_type=0 (直接拉伸, 非letterbox), INTER_AREA
        resize_img = pre_utils.resized_image(
            img, self.input_W, self.input_H, 0, cv2.INTER_AREA)
        y, uv = pre_utils.bgr_to_nv12_planes(resize_img)
        return {self.model_name: {
            self.input_names[0]: y,
            self.input_names[1]: uv,
        }}

    def forward(self, input_tensor):
        return self.model.run(input_tensor)[self.model_name]

    def post_process(self, outputs, img, img_w, img_h):
        """返回 (boxes_list, cropped_images)."""
        preds = outputs[self.output_names[0]]
        preds = np.where(preds > self.threshold, 255, 0).astype(np.uint8).squeeze()
        preds = cv2.resize(preds, (img_w, img_h))

        contours, _ = cv2.findContours(
            preds, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dilated_polys = dilate_contours(contours, self.ratio_prime)
        boxes_list = post_utils.get_bounding_boxes(dilated_polys, self.min_box_area)

        cropped_images = []
        for box in boxes_list:
            cropped = post_utils.crop_and_rotate_image(img, box)
            cropped_images.append(cropped)
        return boxes_list, cropped_images

    def detect(self, img):
        h, w = img.shape[:2]
        inp = self.pre_process(img)
        outs = self.forward(inp)
        return self.post_process(outs, img, w, h)


# ============================================================
# 识别模型 (从 demo 的 PaddleOCR_Rec 改造)
# ============================================================

class PaddleOcrRec:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"OCR rec model not found: {model_path}")
        print(f"[ocr] Loading rec model: {model_path}", file=sys.stderr)
        self.model = hbm_runtime.HB_HBMRuntime(model_path)
        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]
        self.output_quants = self.model.output_quants[self.model_name]
        # rec 输入是 NCHW: shape[2]=H, shape[3]=W
        self.input_H = self.input_shapes[self.input_names[0]][2]
        self.input_W = self.input_shapes[self.input_names[0]][3]

    def set_scheduling_params(self, priority=None, bpu_cores=None):
        kwargs = {}
        if priority is not None:
            kwargs["priority"] = {self.model_name: priority}
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {self.model_name: bpu_cores}
        if kwargs:
            self.model.set_scheduling_params(**kwargs)

    def pre_process(self, img):
        # resize 到 48x320, 归一化, BGR->RGB, NHWC->NCHW
        image_resized = cv2.resize(img, dsize=(self.input_W, self.input_H))
        image_resized = (image_resized / 255.0).astype(np.float32)
        input_image = image_resized[:, :, [2, 1, 0]]  # BGR->RGB
        input_image = input_image[None].transpose(0, 3, 1, 2)  # NHWC->NCHW
        return input_image

    def forward(self, input_tensor):
        return self.model.run(input_tensor)[self.model_name]

    def post_process(self, outputs, postprocess_op):
        preds = outputs[self.output_names[0]]
        sim_pred = postprocess_op(preds)[0][0]
        return sim_pred

    def recognize(self, cropped_img, postprocess_op):
        if cropped_img is None or cropped_img.size == 0:
            return ""
        if cropped_img.shape[0] < 2 or cropped_img.shape[1] < 2:
            return ""
        inp = self.pre_process(cropped_img)
        outs = self.forward(inp)
        return self.post_process(outs, postprocess_op)


# ============================================================
# 全局 OCR 读取器 (det + rec 组装)
# ============================================================

class GlobalOcrReader:
    """
    全图 OCR. 调 read(frame) 返回所有识别到的文字。

    返回: List[dict], 每个:
      {
        "box": [[x,y],[x,y],[x,y],[x,y]],  # 四点框 (int)
        "text": str,                        # 识别的字符串
        "center": [cx, cy],                 # 框中心
      }
    """

    def __init__(self,
                 det_model_path: str = DEFAULT_DET_MODEL,
                 rec_model_path: str = DEFAULT_REC_MODEL,
                 label_file: str = DEFAULT_LABEL,
                 ratio_prime: float = 2.7,
                 threshold: float = 0.5,
                 min_box_area: float = 100,
                 priority: int = 0,
                 bpu_cores=(0,),
                 # 输出过滤参数 (在 rec 之后,只影响 read() 的返回)
                 filter_empty: bool = True,    # 过滤掉 rec 解码出空串的框
                 min_box_w: int = 20,          # 过滤掉宽 < 这个值的框 (噪声)
                 min_box_h: int = 10,          # 过滤掉高 < 这个值的框 (噪声)
                 suppress_demo_prints: bool = True):  # 屏蔽 demo 的 "width: H height: W" 噪声
        self.det = PaddleOcrDet(det_model_path, ratio_prime, threshold, min_box_area)
        self.rec = PaddleOcrRec(rec_model_path)
        self.det.set_scheduling_params(priority=priority, bpu_cores=list(bpu_cores))
        self.rec.set_scheduling_params(priority=priority, bpu_cores=list(bpu_cores))
        self.ctc = CTCLabelDecode(label_file)
        self.filter_empty = filter_empty
        self.min_box_w = min_box_w
        self.min_box_h = min_box_h
        # 屏蔽 utils.postprocess_utils.crop_and_rotate_image 里的 print 噪声
        if suppress_demo_prints:
            self._monkey_patch_silence_demo_prints()
        print(f"[ocr] GlobalOcrReader ready.", file=sys.stderr)

    @staticmethod
    def _monkey_patch_silence_demo_prints():
        """
        demo 的 utils.postprocess_utils.crop_and_rotate_image 在每个裁剪框
        都 print("width:", w, "height:", h),会刷屏. 这里把它替换成静默版本.
        """
        try:
            import utils.postprocess_utils as pu

            orig = pu.crop_and_rotate_image

            def silent_crop(img, box):
                # 抑制 print 的最简单方式: 临时重定向 stdout
                import io
                import contextlib
                with contextlib.redirect_stdout(io.StringIO()):
                    return orig(img, box)

            pu.crop_and_rotate_image = silent_crop
        except Exception as e:
            print(f"[ocr] (warn) failed to silence demo prints: {e}",
                  file=sys.stderr)

    def read(self, frame_bgr) -> List[dict]:
        boxes_list, cropped_images = self.det.detect(frame_bgr)
        results = []
        for box, crop in zip(boxes_list, cropped_images):
            # 提前过滤小框,不浪费 rec 时间
            box_int = np.array(box).reshape(-1, 2).astype(int)
            w = box_int[:, 0].max() - box_int[:, 0].min()
            h = box_int[:, 1].max() - box_int[:, 1].min()
            if w < self.min_box_w or h < self.min_box_h:
                continue

            text = self.rec.recognize(crop, self.ctc)
            # 过滤空文本
            if self.filter_empty and (text is None or text.strip() == ""):
                continue

            cx = int(box_int[:, 0].mean())
            cy = int(box_int[:, 1].mean())
            results.append({
                "box": box_int.tolist(),
                "text": text,
                "center": [cx, cy],
            })
        return results

    @staticmethod
    def draw_overlay(frame_bgr, results, fps: float = 0.0):
        """把 OCR 结果画在画面上 (框 + 文字). 返回 BGR 图。

        注意: cv2.putText 不支持中文,会显示成 '?'. 这里只画框 + ASCII 文字,
        中文/完整文字仍在结构化结果里. 调试看框位置足够;
        要看中文渲染用 PIL (draw_text),但每帧 PIL 太慢,预览用 cv2 即可。
        """
        out = frame_bgr.copy()
        for r in results:
            pts = np.array(r["box"], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], isClosed=True, color=(0, 200, 255), thickness=2)
            # 文字标签 (ASCII 能显示,中文显示为?,但框位置准确)
            label = r["text"][:20] if r["text"] else "?"
            x, y = r["center"]
            try:
                ascii_label = label.encode("ascii", "replace").decode("ascii")
            except Exception:
                ascii_label = "?"
            cv2.putText(out, ascii_label, (r["box"][0][0], max(15, r["box"][0][1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        hud = f"OCR | fps {fps:.1f} | texts {len(results)}"
        cv2.rectangle(out, (0, 0), (300, 26), (0, 0, 0), -1)
        cv2.putText(out, hud, (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        return out


# ============================================================
# MJPEG 浏览器预览 (与 stream 版同款)
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

HTML_PAGE = """<!doctype html><html><head><title>OCR Stream</title>
<style>body{background:#222;color:#ddd;font-family:sans-serif;margin:0;padding:20px;
text-align:center}h2{margin:8px 0}img{max-width:100%;border:2px solid #444;border-radius:8px}
.info{color:#999;font-size:14px;margin-top:8px}</style></head>
<body><h2>PaddleOCR Stream</h2><img src="/stream.mjpg"/>
<div class="info">RDK S100 · det + rec · text in console is full UTF-8</div>
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
# 摄像头 (与 stream 版同款, 强制 V4L2 + MJPG)
# ============================================================

def open_camera(camera_id, width, height):
    cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[cam] Opened camera {camera_id}: {w}x{h}", file=sys.stderr)
    return cap


# ============================================================
# main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="RDK S100 PaddleOCR global text detection + browser preview")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--det-model-path", type=str, default=DEFAULT_DET_MODEL)
    parser.add_argument("--rec-model-path", type=str, default=DEFAULT_REC_MODEL)
    parser.add_argument("--label-file", type=str, default=DEFAULT_LABEL)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--ratio-prime", type=float, default=2.7)
    parser.add_argument("--port", type=int, default=8081,
                        help="HTTP port (default 8081, avoid clash with 8080)")
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--jpeg-quality", type=int, default=70)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--priority", type=int, default=0)
    parser.add_argument("--bpu-cores", type=int, nargs="+", default=[0])
    parser.add_argument("--test-img", type=str, default=None,
                        help="Run OCR on a single image instead of camera (validation)")
    parser.add_argument("--save", type=str, default=None,
                        help="With --test-img: save annotated result here")
    args = parser.parse_args()

    reader = GlobalOcrReader(
        det_model_path=args.det_model_path,
        rec_model_path=args.rec_model_path,
        label_file=args.label_file,
        ratio_prime=args.ratio_prime,
        threshold=args.threshold,
        priority=args.priority,
        bpu_cores=args.bpu_cores,
    )

    # --- 单图模式 (验证流水线,不开摄像头) ---
    if args.test_img:
        print(f"[main] Single-image mode: {args.test_img}", file=sys.stderr)
        img = common.load_image(args.test_img)
        t0 = time.time()
        results = reader.read(img)
        dt = time.time() - t0
        print(f"[main] OCR took {dt*1000:.0f}ms, found {len(results)} texts:")
        for r in results:
            print(f"  text='{r['text']}'  center={r['center']}")
        if args.save:
            overlay = reader.draw_overlay(img, results, fps=1.0/dt if dt else 0)
            cv2.imwrite(args.save, overlay)
            print(f"[main] Saved annotated image to: {args.save}")
        return

    # --- 摄像头 + 浏览器模式 ---
    http_server = None
    if not args.no_stream:
        try:
            http_server = start_http_server(args.port)
            ip = get_local_ip()
            print(f"[http] Stream server started.", file=sys.stderr)
            print(f"[http] Open in browser:  http://{ip}:{args.port}", file=sys.stderr)
            print(f"[http] (via Tailscale use your 100.x.x.x IP)", file=sys.stderr)
        except OSError as e:
            print(f"[http] Failed to start server: {e}", file=sys.stderr)

    cap = open_camera(args.camera, args.width, args.height)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]

    import signal
    stop = {"v": False}
    signal.signal(signal.SIGINT, lambda s, f: stop.update(v=True))

    t_start = time.time()
    last_log = t_start
    n = 0
    cur_fps = 0.0
    last_fps_t = t_start
    last_fps_n = 0

    try:
        while not stop["v"]:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            t0 = time.time()
            results = reader.read(frame)
            ocr_ms = (time.time() - t0) * 1000

            if not args.no_stream:
                overlay = reader.draw_overlay(frame, results, fps=cur_fps)
                ok, jpeg = cv2.imencode(".jpg", overlay, encode_params)
                if ok:
                    FRAME_BUFFER.update(jpeg.tobytes(), n)

            n += 1
            now = time.time()
            if now - last_fps_t >= 0.5:
                cur_fps = (n - last_fps_n) / (now - last_fps_t)
                last_fps_t = now
                last_fps_n = n
            if now - last_log >= 1.0:
                texts = [r["text"] for r in results]
                print(f"[stats] frame={n} fps={cur_fps:.1f} "
                      f"ocr={ocr_ms:.0f}ms texts={texts}", file=sys.stderr)
                last_log = now
            if args.max_frames and n >= args.max_frames:
                break
    finally:
        cap.release()
        if http_server:
            http_server.shutdown()
        print("[main] Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
