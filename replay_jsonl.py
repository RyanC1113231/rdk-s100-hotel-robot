"""
replay_jsonl.py
---------------
回放脚本: 读取 JSONL 流式文件 或 generator 的整包 JSON,
逐帧产出 frame dict,可按原始 timestamp 节奏播放 (模拟实时输入)。

支持两种输入:
  1. JSONL (每行一帧, usb_perception.py 的输出)
  2. JSON  (generator 的 {description, version, frames: [...]})
     自动识别。

用法:
    # 按真实时间节奏回放 (timestamp 间隔决定速度)
    python replay_jsonl.py --input stream.jsonl

    # 加速 5 倍 / 减速到 0.5 倍
    python replay_jsonl.py --input stream.jsonl --speed 5.0
    python replay_jsonl.py --input stream.jsonl --speed 0.5

    # 不按时间节奏,以最快速度走完
    python replay_jsonl.py --input stream.jsonl --no-realtime

    # 限定起止帧
    python replay_jsonl.py --input stream.jsonl --start 100 --end 200

    # 输出到 stdout 给下游消费 (默认行为)
    python replay_jsonl.py --input stream.jsonl | python task3_core.py --stdin

    # 也可以当库用:
    from replay_jsonl import iter_frames
    for frame in iter_frames("stream.jsonl"):
        controller.step(frame)
"""

import argparse
import json
import sys
import time
from pathlib import Path


def _detect_format(path):
    """返回 'jsonl' 或 'json' (generator整包)."""
    with open(path, "r", encoding="utf-8") as f:
        # 跳过空白行
        first = ""
        for line in f:
            s = line.strip()
            if s:
                first = s
                break
    if not first:
        raise ValueError(f"{path} is empty")
    # JSONL 每行是单帧 dict (含 frame_id 或 targets),且通常较短
    # 整包 JSON 一般以 { 开始,后面跟 description/version/frames
    if first.startswith("{") and ("frames" in first or path.lower().endswith(".json")):
        # 可能是整包 (但有可能JSONL第一行就是单帧),用更严格判断:
        # 尝试整文件 json.load 一次
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict) and "frames" in obj:
                return "json"
        except json.JSONDecodeError:
            pass
    return "jsonl"


def iter_frames(path):
    """
    通用迭代器: 不管输入是 JSONL 还是整包 JSON,都按帧 dict yield。
    每个 dict 至少含: frame_id, timestamp, targets
    """
    fmt = _detect_format(path)
    if fmt == "json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        for frame in obj.get("frames", []):
            yield frame
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                yield json.loads(s)


def main():
    parser = argparse.ArgumentParser(
        description="Replay JSONL/JSON scenario files frame by frame")
    parser.add_argument("--input", required=True, help="Input .jsonl or .json file")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (default 1.0 = realtime)")
    parser.add_argument("--no-realtime", action="store_true",
                        help="Ignore timestamps, output as fast as possible")
    parser.add_argument("--start", type=int, default=0,
                        help="Start at frame index (inclusive)")
    parser.add_argument("--end", type=int, default=-1,
                        help="End at frame index (inclusive). -1 = all")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL path. Default: stdout")
    parser.add_argument("--verbose", action="store_true",
                        help="Log progress to stderr")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"[error] File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        out_fp = open(args.output, "w", buffering=1)
    else:
        out_fp = sys.stdout

    fmt = _detect_format(args.input)
    print(f"[init] Input format: {fmt}", file=sys.stderr)

    t_wall_start = time.time()
    t_data_start = None
    n_emitted = 0

    try:
        for i, frame in enumerate(iter_frames(args.input)):
            if i < args.start:
                continue
            if args.end >= 0 and i > args.end:
                break

            # 时间节奏控制
            if not args.no_realtime:
                ts = frame.get("timestamp", None)
                if ts is not None:
                    if t_data_start is None:
                        t_data_start = ts
                    target_wall = t_wall_start + (ts - t_data_start) / args.speed
                    sleep_for = target_wall - time.time()
                    if sleep_for > 0:
                        time.sleep(sleep_for)

            out_fp.write(json.dumps(frame, ensure_ascii=False) + "\n")
            n_emitted += 1

            if args.verbose and n_emitted % 30 == 0:
                print(f"[replay] emitted {n_emitted} frames "
                      f"(latest frame_id={frame.get('frame_id')})", file=sys.stderr)

    except KeyboardInterrupt:
        print("\n[replay] Interrupted by user.", file=sys.stderr)
    finally:
        if args.output:
            out_fp.close()
        elapsed = time.time() - t_wall_start
        print(f"[replay] Done. Emitted {n_emitted} frames in {elapsed:.2f}s.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
