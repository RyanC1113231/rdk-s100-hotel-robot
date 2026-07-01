#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scenario3_controller.py  --  Scenario 3 target door-board selection controller

Scenario 3:
  Robot drives forward.
  Two paper door-number boards may appear ahead, roughly at 11 o'clock and 1 o'clock.
  OCR may see both, e.g. 1207 and 1208.
  The controller selects target_room, decides whether it is on the left/right/center,
  then performs a fixed turn toward that target.
  After the fixed turn, it drives forward and stops by ultrasonic distance.

Expected OCR result format from GlobalOcrReader.read(frame):
  [
    {"box": [[x,y], [x,y], [x,y], [x,y]], "text": "1207", "center": [cx, cy]},
    ...
  ]

Main entry:
  s3 = Scenario3Controller(target_room="1207")
  cmd = s3.update(ocr_results=ocr_results, front_distance=front_distance, frame_id=n)

Returns:
  task3_core.MotorCommand
"""

from __future__ import annotations

import re
import time
from typing import Optional, Sequence

from task3_core import RobotState, MotorCommand


# ============================================================
# SCENARIO3_TUNE_HERE
# 你之后主要只改这一块
# ============================================================

DEFAULT_TARGET_ROOM = "1207"
FRAME_WIDTH = 640

# 1) 没看到目标门牌前：直走，不再原地扫描
CRUISE_SPEED = 18

# 2) 看到目标门牌后：先停一下，避免 OCR 刚识别到就冲过去
LOCK_STOP_SEC = 0.20

# 3) 目标门牌在左/右时：执行固定时间转向
#    你说“转的角度我会决定”，现场就调下面两个参数：
TARGET_TURN_SPEED = 12       # 转向速度 duty，建议 8~18
TARGET_TURN_SEC = 0.70       # 转向持续时间，角度主要靠这个调，建议 0.3~1.5

# 4) 如果目标 x 离中心小于这个范围，就认为不用转，直接靠近
CENTER_DEADBAND_PX = 60

# 5) 转完之后：直走靠近目标纸板
APPROACH_SPEED = 14

# 6) 最后用超声波停车
DOOR_STOP_DIST = 0.35        # m，建议 0.30~0.45

# 7) OCR 容错设置
#    看到目标一次就锁定，不需要持续识别；防止转向/靠近时 OCR 丢失又乱变。
REQUIRE_TARGET_HITS = 1      # 要求识别到目标几次才锁定，建议 1 或 2

# 8) 安全兜底：看到目标并开始动作后，最多跑多久。
#    0 表示关闭这个超时停车。
MAX_ACTIVE_SEC = 0.0


class S3:
    CRUISE = "cruise"          # 直走，等待 OCR 看到目标
    LOCK_STOP = "lock_stop"    # 看到目标后短暂停一下
    TURN_TO_TARGET = "turn_to_target"
    APPROACH = "approach"
    DONE = "done"


class Scenario3Controller:
    """
    Scenario 3 behavior:

      CRUISE:
        drive forward; OCR watches both boards.
      LOCK_STOP:
        target_room acquired; stop briefly.
      TURN_TO_TARGET:
        fixed-time turn left/right based on target x.
      APPROACH:
        drive straight toward the selected target board.
      DONE:
        stop.

    It does NOT scan in place before seeing the target.
    It does NOT continuously chase OCR x while approaching.
    """

    def __init__(
        self,
        target_room: str = DEFAULT_TARGET_ROOM,
        frame_width: int = FRAME_WIDTH,

        # Keep these names compatible with previous main_robot.py args.
        search_turn_speed: float = CRUISE_SPEED,     # repurposed as cruise speed
        search_turn_state: RobotState = RobotState.TURN_LEFT,  # unused, kept compatible
        align_turn_speed: float = TARGET_TURN_SPEED,
        approach_speed: float = APPROACH_SPEED,
        center_deadband_px: int = CENTER_DEADBAND_PX,
        x_stable_px: int = 35,                       # unused, kept compatible
        stable_samples: int = 3,                     # unused, kept compatible
        lost_hold_frames: int = 45,                  # unused, kept compatible
        door_stop_dist: float = DOOR_STOP_DIST,
        max_active_sec: float = MAX_ACTIVE_SEC,

        # New direct tuning args. main_robot may not pass these yet, so defaults work.
        target_turn_sec: float = TARGET_TURN_SEC,
        lock_stop_sec: float = LOCK_STOP_SEC,
        require_target_hits: int = REQUIRE_TARGET_HITS,
    ):
        self.target_room = str(target_room)
        self.frame_width = int(frame_width)
        self.frame_center_x = self.frame_width / 2.0

        self.cruise_speed = float(search_turn_speed)
        self.align_turn_speed = float(align_turn_speed)
        self.approach_speed = float(approach_speed)
        self.center_deadband_px = int(center_deadband_px)
        self.door_stop_dist = float(door_stop_dist)

        self.target_turn_sec = float(target_turn_sec)
        self.lock_stop_sec = float(lock_stop_sec)
        self.require_target_hits = max(1, int(require_target_hits))
        self.max_active_sec = float(max_active_sec)

        self.state = S3.CRUISE
        self._t_enter = time.time()
        self._active_since: Optional[float] = None

        self._target_hits = 0
        self._target_x: Optional[float] = None
        self._target_raw_text: str = ""
        self._target_side = "center"   # "left" / "right" / "center"

        self._last_seen_texts = []

    @property
    def done(self) -> bool:
        return self.state == S3.DONE

    def reset(self):
        self.state = S3.CRUISE
        self._t_enter = time.time()
        self._active_since = None
        self._target_hits = 0
        self._target_x = None
        self._target_raw_text = ""
        self._target_side = "center"
        self._last_seen_texts = []

    def _enter(self, state: str):
        self.state = state
        self._t_enter = time.time()

    @staticmethod
    def _digits(text: str) -> str:
        """
        OCR 容错:
          I/l/|/! -> 1
          O/o/Q/D -> 0
          S/s -> 5
          B -> 8
        """
        s = str(text or "")
        trans = str.maketrans({
            "N": "2",
            "o": "0", "O": "0",
            "B": "8",
        })
        s = s.translate(trans)
        return "".join(re.findall(r"\d", s))

    @staticmethod
    def _box_area(r: dict) -> float:
        try:
            pts = r.get("box", [])
            xs = [float(p[0]) for p in pts]
            ys = [float(p[1]) for p in pts]
            return max(0.0, max(xs) - min(xs)) * max(0.0, max(ys) - min(ys))
        except Exception:
            return 0.0

    def _find_target(self, ocr_results: Optional[Sequence[dict]]) -> Optional[dict]:
        if not ocr_results:
            self._last_seen_texts = []
            return None

        candidates = []
        seen = []

        for r in ocr_results:
            if not isinstance(r, dict):
                continue

            raw = str(r.get("text", "") or "")
            digits = self._digits(raw)
            if raw or digits:
                seen.append(f"{raw}->{digits}")

            center = r.get("center")
            if not (isinstance(center, (list, tuple)) and len(center) >= 2):
                continue

            # Match exact target or containing target. Example: "Room1207" -> "1207".
            if digits == self.target_room or self.target_room in digits:
                candidates.append(r)

        self._last_seen_texts = seen

        if not candidates:
            return None

        # If multiple boxes match target, choose the largest.
        return max(candidates, key=self._box_area)

    def _update_target_from_ocr(self, ocr_results: Optional[Sequence[dict]]) -> bool:
        target = self._find_target(ocr_results)
        if target is None:
            return False

        try:
            x = float(target["center"][0])
        except Exception:
            return False

        self._target_hits += 1
        self._target_x = x
        self._target_raw_text = str(target.get("text", "") or "")

        err = x - self.frame_center_x
        if err < -self.center_deadband_px:
            self._target_side = "left"
        elif err > self.center_deadband_px:
            self._target_side = "right"
        else:
            self._target_side = "center"

        return self._target_hits >= self.require_target_hits

    def _turn_command(self, d: float) -> MotorCommand:
        if self._target_side == "left":
            return MotorCommand(
                self.align_turn_speed,
                self.align_turn_speed,
                RobotState.TURN_LEFT,
                f"S3 target={self.target_room} on LEFT x={self._target_x:.0f}; "
                f"fixed turn {self.target_turn_sec:.2f}s d={d:.2f}",
            )
        if self._target_side == "right":
            return MotorCommand(
                self.align_turn_speed,
                self.align_turn_speed,
                RobotState.TURN_RIGHT,
                f"S3 target={self.target_room} on RIGHT x={self._target_x:.0f}; "
                f"fixed turn {self.target_turn_sec:.2f}s d={d:.2f}",
            )

        return MotorCommand(
            self.approach_speed,
            self.approach_speed,
            RobotState.FORWARD,
            f"S3 target={self.target_room} centered x={self._target_x:.0f}; no turn d={d:.2f}",
        )

    def update(
        self,
        ocr_results: Optional[Sequence[dict]],
        front_distance: float,
        frame_id: int = 0,
    ) -> MotorCommand:
        d = float(front_distance)
        now = time.time()
        el = now - self._t_enter

        if self.state == S3.DONE:
            return MotorCommand(0, 0, RobotState.STOP, "S3 DONE")

        # If already active / acquired target, ultrasonic is allowed to stop.
        # Before target is acquired, do NOT stop just because ultrasonic is close;
        # scenario3 wants target-room-triggered navigation.
        if self.state in (S3.LOCK_STOP, S3.TURN_TO_TARGET, S3.APPROACH):
            if d <= self.door_stop_dist:
                self._enter(S3.DONE)
                return MotorCommand(
                    0, 0, RobotState.STOP,
                    f"S3 ultrasonic stop after target={self.target_room} d={d:.2f}",
                )

        if self.max_active_sec > 0 and self._active_since is not None:
            if now - self._active_since >= self.max_active_sec:
                self._enter(S3.DONE)
                return MotorCommand(
                    0, 0, RobotState.STOP,
                    f"S3 timeout stop target={self.target_room} d={d:.2f}",
                )

        # ---------------- CRUISE ----------------
        if self.state == S3.CRUISE:
            acquired = self._update_target_from_ocr(ocr_results)
            if acquired:
                self._active_since = now
                self._enter(S3.LOCK_STOP)
                return MotorCommand(
                    0, 0, RobotState.STOP,
                    f"S3 acquired target={self.target_room} text='{self._target_raw_text}' "
                    f"x={self._target_x:.0f} side={self._target_side} seen={self._last_seen_texts}",
                )

            return MotorCommand(
                self.cruise_speed,
                self.cruise_speed,
                RobotState.FORWARD,
                f"S3 cruise searching target={self.target_room} seen={self._last_seen_texts}",
            )

        # ---------------- LOCK_STOP ----------------
        if self.state == S3.LOCK_STOP:
            if el < self.lock_stop_sec:
                return MotorCommand(
                    0, 0, RobotState.STOP,
                    f"S3 lock stop target={self.target_room} side={self._target_side} t={el:.2f}",
                )

            # If target is centered, skip fixed turn.
            if self._target_side == "center":
                self._enter(S3.APPROACH)
                return MotorCommand(
                    self.approach_speed,
                    self.approach_speed,
                    RobotState.FORWARD,
                    f"S3 target centered -> approach d={d:.2f}",
                )

            self._enter(S3.TURN_TO_TARGET)
            return self._turn_command(d)

        # ---------------- TURN_TO_TARGET ----------------
        if self.state == S3.TURN_TO_TARGET:
            if el < self.target_turn_sec:
                return self._turn_command(d)

            self._enter(S3.APPROACH)
            return MotorCommand(
                self.approach_speed,
                self.approach_speed,
                RobotState.FORWARD,
                f"S3 fixed turn done side={self._target_side} -> approach d={d:.2f}",
            )

        # ---------------- APPROACH ----------------
        if self.state == S3.APPROACH:
            return MotorCommand(
                self.approach_speed,
                self.approach_speed,
                RobotState.FORWARD,
                f"S3 approach target={self.target_room} side={self._target_side} d={d:.2f}",
            )

        return MotorCommand(0, 0, RobotState.STOP, f"S3 unknown state={self.state}")
