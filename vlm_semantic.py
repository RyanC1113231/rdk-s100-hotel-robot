"""
vlm_semantic.py
---------------
VLM semantic stream wrapper for Task6 arbitrator.

Architecture:
  - Background thread: fetch latest frame -> run llama-smolvlm-bpu-cli -> parse output -> update semantic state
  - Main thread calls get_semantic_state() non-blocking
  - Each inference restarts process (~3s load), state updates ~every 8s

Usage:
    from vlm_semantic import VLMSemanticStream

    stream = VLMSemanticStream()
    stream.start()

    # Call every frame in main loop
    stream.update_frame(frame_bgr)

    # Task6 arbitrator reads state
    state = stream.get_semantic_state()
    # state = {
    #     "person_present": True/False,
    #     "crowded": True/False,
    #     "obstacle_present": True/False,
    #     "description": "raw description text",
    #     "timestamp": float,
    #     "age": float,   # seconds since last update
    # }

    stream.stop()

Path config:
    Modify constants below for actual deployment paths.
"""

import subprocess
import threading
import time
import os
import cv2
import numpy as np
import tempfile
import logging

logger = logging.getLogger(__name__)

# ============================================================
# Path config (modify for deployment)
# ============================================================

LLAMA_BIN = os.path.expanduser(
    "~/ros2_ws/llama.cpp_vlm_bpu/llama.cpp/build/bin/llama-smolvlm-bpu-cli"
)
GGUF_MODEL = os.path.expanduser(
    "~/ros2_ws/models/smolvlm/SmolVLM2-256M-Video-Instruct-Q8_0.gguf"
)
HBM_MODEL = os.path.expanduser(
    "~/ros2_ws/models/smolvlm/SigLip_int16_SmolVLM2_256M_Instruct_S100.hbm"
)

# Inference params
THREADS = 8
TEMPERATURE = 0.5
PROMPT = "Describe the scene."

# Semantic parsing keywords
PERSON_KEYWORDS = [
    "person", "people", "man", "woman", "child", "crowd",
    "human", "standing", "walking", "sitting"
]
CROWDED_KEYWORDS = [
    "crowd", "many people", "group of people", "crowded",
    "numerous", "packed", "filled with people"
]
EMPTY_KEYWORDS = [
    "empty", "no one", "nobody", "no people", "vacant",
    "unoccupied", "deserted"
]
# Obstacle keywords based on real VLM output
OBSTACLE_KEYWORDS = [
    "box", "boxes", "cardboard",           # cardboard box (frequent VLM output)
    "bottle", "waterbottle",                # bottle (frequent VLM output)
    "suitcase", "luggage", "bag",           # luggage
    "cart", "trolley",                      # cart
    "chair", "table", "furniture", "shelf", # furniture
    "cone", "barrier",                      # road barrier
    "bucket", "bin", "trash",               # trash bin
    "object on the floor", "placed on the floor",  # VLM????
]


# ============================================================
# Semantic parsing
# ============================================================

def parse_description(text: str) -> dict:
    """
    Extract structured semantics from description text.
    Returns person_present, crowded, obstacle_present fields.
    """
    text_lower = text.lower()

    person_present = any(kw in text_lower for kw in PERSON_KEYWORDS)
    crowded = any(kw in text_lower for kw in CROWDED_KEYWORDS)
    obstacle_present = any(kw in text_lower for kw in OBSTACLE_KEYWORDS)

    # If explicitly empty, override person detection
    if any(kw in text_lower for kw in EMPTY_KEYWORDS):
        person_present = False
        crowded = False

    return {
        "person_present": person_present,
        "crowded": crowded,
        "obstacle_present": obstacle_present,
    }


# ============================================================
# Single inference (stateless, for standalone testing)
# ============================================================

def run_vlm_inference(image_path: str, timeout: int = 30) -> str:
    """
    Run one VLM inference on given image path. Returns description text.
    Returns empty string on failure.
    """
    cmd = [
        LLAMA_BIN,
        "-m", GGUF_MODEL,
        "--mmproj", HBM_MODEL,
        "--image", image_path,
        "-p", PROMPT,
        "--temp", str(TEMPERATURE),
        "--threads", str(THREADS),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        # Extract generated text after "=== Generating response ===" marker
        output = result.stdout
        marker = "=== Generating response ==="
        if marker in output:
            text = output.split(marker)[-1].strip()
            # Remove trailing performance stats lines
            lines = [l for l in text.splitlines() if l.strip() and not l.startswith("llama_perf")]
            return " ".join(lines).strip()
        return output.strip()
    except subprocess.TimeoutExpired:
        logger.warning("[vlm] inference timeout")
        return ""
    except Exception as e:
        logger.error(f"[vlm] inference error: {e}")
        return ""


# ============================================================
# Frame buffer (thread-safe)
# ============================================================

class _FrameBuffer:
    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._updated = False

    def put(self, frame_bgr):
        with self._lock:
            self._frame = frame_bgr.copy()
            self._updated = True

    def get_if_updated(self):
        """Return (frame, is_new). is_new=False if no update since last call."""
        with self._lock:
            if self._frame is None:
                return None, False
            frame = self._frame.copy()
            is_new = self._updated
            self._updated = False
            return frame, is_new


# ============================================================
# VLM semantic stream main class
# ============================================================

class VLMSemanticStream:
    """
    Background thread running VLM inference for Task6 semantic state.

    Thread-safe: get_semantic_state() can be called from any thread.
    """

    def __init__(self):
        self._frame_buf = _FrameBuffer()
        self._lock = threading.Lock()
        self._state = {
            "person_present": False,
            "crowded": False,
            "obstacle_present": False,
            "description": "",
            "timestamp": 0.0,
            "age": float("inf"),
        }
        self._running = False
        self._thread = None
        self._tmp_dir = tempfile.mkdtemp(prefix="vlm_frames_")

    # ----------------------------------------------------------
    # Public interface
    # ----------------------------------------------------------

    def start(self):
        """Start background inference thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("[vlm] VLMSemanticStream started")

    def stop(self):
        """Stop background thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("[vlm] VLMSemanticStream stopped")

    def update_frame(self, frame_bgr: np.ndarray):
        """
        Update frame for inference (called externally each frame).
        Non-blocking, returns immediately.
        """
        self._frame_buf.put(frame_bgr)

    def get_semantic_state(self) -> dict:
        """
        Return latest semantic state (non-blocking).
        age field = seconds since last successful inference.
        """
        with self._lock:
            state = self._state.copy()
        state["age"] = time.time() - state["timestamp"] if state["timestamp"] > 0 else float("inf")
        return state

    # ----------------------------------------------------------
    # Background inference loop
    # ----------------------------------------------------------

    def _loop(self):
        tmp_path = os.path.join(self._tmp_dir, "current_frame.jpg")
        while self._running:
            frame, _ = self._frame_buf.get_if_updated()

            if frame is None:
                time.sleep(0.1)
                continue

            # Save frame to temp file
            ok = cv2.imwrite(tmp_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                logger.warning("[vlm] failed to write temp frame")
                continue

            # Run inference
            t0 = time.time()
            description = run_vlm_inference(tmp_path)
            elapsed = time.time() - t0
            logger.info(f"[vlm] inference done in {elapsed:.1f}s | '{description}'")

            if not description:
                continue

            # Parse semantics
            parsed = parse_description(description)

            # Update state
            with self._lock:
                self._state = {
                    "person_present": parsed["person_present"],
                    "crowded": parsed["crowded"],
                    "obstacle_present": parsed["obstacle_present"],
                    "description": description,
                    "timestamp": time.time(),
                    "age": 0.0,
                }

    # ----------------------------------------------------------
    # Destructor
    # ----------------------------------------------------------

    def __del__(self):
        self.stop()


# ============================================================
# Standalone test entry
# ============================================================

if __name__ == "__main__":
    import sys
    import argparse

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="VLM semantic stream test")
    parser.add_argument("--image", type=str, default=None,
                        help="Single image path (no camera)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera ID (default 0)")
    parser.add_argument("--duration", type=int, default=30,
                        help="Camera mode duration in seconds (default 30)")
    args = parser.parse_args()

    if args.image:
        # Single image test mode
        print(f"[test] Running on image: {args.image}")
        desc = run_vlm_inference(args.image)
        print(f"[test] Description: {desc}")
        parsed = parse_description(desc)
        print(f"[test] person_present:   {parsed['person_present']}")
        print(f"[test] crowded:          {parsed['crowded']}")
        print(f"[test] obstacle_present: {parsed['obstacle_present']}")

    else:
        # Camera stream mode
        stream = VLMSemanticStream()
        stream.start()

        cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)

        print(f"[test] Running for {args.duration}s, reading camera {args.camera}...")
        t_end = time.time() + args.duration
        while time.time() < t_end:
            ret, frame = cap.read()
            if ret:
                stream.update_frame(frame)
            state = stream.get_semantic_state()
            print(f"[state] person={state['person_present']} "
                  f"crowded={state['crowded']} "
                  f"obstacle={state['obstacle_present']} "
                  f"age={state['age']:.1f}s | {state['description'][:60]}")
            time.sleep(1.0)

        cap.release()
        stream.stop()
        print("[test] Done.")
