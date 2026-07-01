#!/usr/bin/env python3
"""
voice_launcher.py  --  voice-controlled robot launcher for S100
===============================================================
Listens to Yahboom AI Voice Module (CI1302) via UART.
Launches/stops main_robot.py (full perception + avoidance + motor).

Voice commands:
  "ni hao xiao ya"    -> wake module
  "xiao che qian jin"  (0x04,0x70,0x71) -> start main_robot.py
  "xiao che ting zhi"  (0x01) -> stop  main_robot.py
  "ting che"           (0x02) -> stop
  "xiao che xiu mian"  (0x03) -> stop

Usage:
  python3 voice_launcher.py
  python3 voice_launcher.py --speed 40
"""

import serial
import subprocess
import time
import signal
import sys
import os
import argparse

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MAIN_ROBOT = "/root/ros2_ws/src/robot_tasks/robot_tasks/main_robot.py"
MAIN_CWD   = "/root/ros2_ws/src/robot_tasks/robot_tasks"
PYTHON     = "/usr/bin/python3"

# ---------------------------------------------------------------------------
# UART Protocol
# ---------------------------------------------------------------------------
FRAME_HEAD = bytes([0xAA, 0x55])
FRAME_TAIL = 0xFB
FRAME_LEN  = 5
TTS_INIT_DONE = 0x67


class FrameParser:
    def __init__(self):
        self._buf = bytearray()

    def feed(self, data: bytes):
        self._buf.extend(data)
        while len(self._buf) >= FRAME_LEN:
            idx = self._buf.find(FRAME_HEAD)
            if idx < 0:
                self._buf = self._buf[-1:]
                return
            if idx > 0:
                self._buf = self._buf[idx:]
            if len(self._buf) < FRAME_LEN:
                return
            if self._buf[4] == FRAME_TAIL:
                func_type = self._buf[2]
                cmd_id    = self._buf[3]
                self._buf = self._buf[FRAME_LEN:]
                yield (func_type, cmd_id)
            else:
                self._buf = self._buf[2:]


def tts_speak(ser, content_id):
    ser.write(bytes([0xAA, 0x55, 0xFF, content_id, 0xFB]))
    time.sleep(0.005)
    ser.flushInput()


# ---------------------------------------------------------------------------
# Process manager
# ---------------------------------------------------------------------------
class RobotProcess:
    def __init__(self):
        self._proc = None

    @property
    def running(self):
        return self._proc is not None and self._proc.poll() is None

    def start(self, extra_args=None):
        if self.running:
            print("[Voice] main_robot already running (PID %d)" % self._proc.pid)
            return False
        cmd = [PYTHON, MAIN_ROBOT]
        if extra_args:
            cmd.extend(extra_args)
        print("[Voice] >>> Starting: %s" % " ".join(cmd))
        self._proc = subprocess.Popen(cmd, cwd=MAIN_CWD)
        print("[Voice] main_robot started, PID = %d" % self._proc.pid)
        return True

    def stop(self):
        if not self.running:
            print("[Voice] main_robot not running")
            return False
        pid = self._proc.pid
        print("[Voice] >>> Stopping main_robot (PID %d)..." % pid)
        self._proc.terminate()
        try:
            self._proc.wait(timeout=5)
            print("[Voice] Stopped gracefully")
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
            print("[Voice] Force killed")
        self._proc = None
        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--speed", type=int, default=50,
                    help="Max motor speed passed to main_robot.py")
    ap.add_argument("--no-vlm", action="store_true",
                    help="Pass --no-vlm to main_robot.py")
    args = ap.parse_args()

    # Build extra args for main_robot.py
    robot_args = ["--max-speed", str(args.speed)]
    if args.no_vlm:
        robot_args.append("--no-vlm")

    # Open serial
    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.1)
    except serial.SerialException as e:
        print("[Voice] Cannot open %s: %s" % (args.port, e))
        sys.exit(1)
    print("[Voice] UART ready: %s @ %d" % (args.port, args.baud))

    robot = RobotProcess()
    fp = FrameParser()

    # TTS announce
    time.sleep(0.3)
    tts_speak(ser, TTS_INIT_DONE)
    print("[Voice] Ready. Say wake word then command.")
    print()

    def shutdown(sig=None, frame=None):
        print("\n[Voice] Shutting down...")
        robot.stop()
        ser.close()
        sys.exit(0)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        while True:
            data = ser.read(ser.in_waiting or 1)
            if not data:
                continue

            for func_type, cmd_id in fp.feed(data):
                if func_type != 0x00:
                    if func_type == 0x03:
                        print("[Voice] Module woken up")
                    continue

                if cmd_id == (0x04,0x70,0x71):      # "xiao che qian jin"
                    print("[Voice] >>> START ROBOT")
                    robot.start(robot_args)

                elif cmd_id in (0x01, 0x02):  # "ting zhi" / "ting che"
                    print("[Voice] >>> STOP ROBOT")
                    robot.stop()

                elif cmd_id == 0x03:    # "xiu mian"
                    print("[Voice] >>> SLEEP -> stop")
                    robot.stop()

                else:
                    print("[Voice] cmd_id=0x%02X (no handler)" % cmd_id)

    except Exception as e:
        print("[Voice] Error: %s" % e)
    finally:
        robot.stop()
        ser.close()


if __name__ == "__main__":
    main()
