#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scenario2_controller.py  --  ultrasonic + OCR door-number navigation FSM.

Flow:
  CRUISE      drive straight until front ultrasonic enters danger zone
  SCAN_LEFT   rotate left in place, watch front ultrasonic
              front open (sustained) -> TRAVERSE ; ~90deg no opening -> TURN_RIGHT
  TURN_RIGHT  rotate right until front open, or fixed-time if BLIND_RIGHT=True
  TRAVERSE    drive into the opening while watching OCR door numbers
              once target room is seen -> slow down -> wait for x to stabilize
              -> turn toward target -> creep forward -> ultrasonic stop
  DONE        stop

Expected OCR result format from GlobalOcrReader.read(frame):
  [
    {"box": [[x,y], [x,y], [x,y], [x,y]], "text": "1207", "center": [cx, cy]},
    ...
  ]

This file embeds the previous DoorOcrNavigator directly into Scenario2Controller,
so main_robot.py does not need to import door_ocr_navigator.py anymore.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import re
import time
from typing import Optional, Sequence

from task3_core import RobotState, MotorCommand

# ==================== scenario2 tunables ====================
CRUISE_SPEED = 30        # forward duty while cruising
SCAN_SPEED   = 20        # rotate duty while scanning (slow = cleaner sweep)
TURN_SPEED   = 30        # rotate duty for right turn
GO_SPEED     = 28        # forward duty through opening before OCR locks target

DANGER_DIST = 0.45       # m. front closer than this in CRUISE -> start scanning
OPEN_DIST   = 0.50       # m. front farther than this counts as "open" (99.9 also counts)
OPEN_CONFIRM_SEC = 1.20 # front must stay open this long before committing

SCAN_LEFT_MAX_SEC = 3.0  # time to rotate ~90deg left at SCAN_SPEED -- calibrate
TURN_RIGHT_MAX_SEC = 4.0 # fallback cap for right turn (~180-200deg) -- calibrate

BLIND_RIGHT = False
TURN_RIGHT_BLIND_SEC = 3.0

# ==================== OCR door-nav tunables ====================
DEFAULT_TARGET_ROOM = "1207"
FRAME_WIDTH = 640
APPROACH_SPEED = 15          # slow speed after seeing target room
DOOR_TURN_SPEED = 15         # in-place turn speed while centering target door
CENTER_DEADBAND_PX = 45      # |target_x - frame_center| <= this -> go forward
X_STABLE_PX = 40             # stability: recent x values within median ± this px
STABLE_SAMPLES = 3           # how many target OCR samples are required before turning
LOST_FAST_FRAMES = 60        # after target is acquired, OCR lost this long -> creep slowly (no release)
DOOR_STOP_DIST = 0.30        # m. ultrasonic stop distance once target room is acquired


class S2:
    CRUISE = "cruise"
    SCAN_LEFT = "scan_left"
    TURN_RIGHT = "turn_right"
    TRAVERSE = "traverse"
    DONE = "done"


@dataclass
class DoorNavStatus:
    active: bool
    done: bool
    command: Optional[MotorCommand]
    reason: str
    target_x: Optional[float] = None
    stable: bool = False
    lost_frames: int = 0


class DoorOcrNavigator:
    """Small OCR stabilizer/controller embedded for Scenario2Controller."""

    def __init__(
        self,
        target_room: str = DEFAULT_TARGET_ROOM,
        frame_width: int = FRAME_WIDTH,
        approach_speed: float = APPROACH_SPEED,
        turn_speed: float = DOOR_TURN_SPEED,
        center_deadband_px: int = CENTER_DEADBAND_PX,
        x_stable_px: int = X_STABLE_PX,
        stable_samples: int = STABLE_SAMPLES,
        lost_fast_frames: int = LOST_FAST_FRAMES,
        stop_distance_m: float = DOOR_STOP_DIST,
    ):
        self.target_room = str(target_room)
        self.frame_width = int(frame_width)
        self.frame_center_x = self.frame_width / 2.0
        self.approach_speed = float(approach_speed)
        self.turn_speed = float(turn_speed)
        self.center_deadband_px = int(center_deadband_px)
        self.x_stable_px = int(x_stable_px)
        self.stable_samples = max(1, int(stable_samples))
        self.lost_fast_frames = int(lost_fast_frames)
        self.stop_distance_m = float(stop_distance_m)

        self._active = False
        self._done = False
        self._last_seen_frame: Optional[int] = None
        self._x_hist = deque(maxlen=self.stable_samples)
        self._last_target = None

    @property
    def active(self) -> bool:
        return self._active and not self._done

    @property
    def done(self) -> bool:
        return self._done

    def reset(self):
        self._active = False
        self._done = False
        self._last_seen_frame = None
        self._x_hist.clear()
        self._last_target = None

    @staticmethod
    def _digits(text: str) -> str:
        return "".join(re.findall(r"\d", str(text or "")))

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
            return None

        candidates = []
        for r in ocr_results:
            if not isinstance(r, dict):
                continue
            digits = self._digits(r.get("text", ""))
            # Accept exact "1207" or strings containing it, e.g. "Room 1207".
            if digits == self.target_room or self.target_room in digits:
                center = r.get("center")
                if isinstance(center, (list, tuple)) and len(center) >= 2:
                    candidates.append(r)

        if not candidates:
            return None

        # If already tracking one, prefer the candidate closest to previous x.
        # Otherwise prefer largest OCR box; it is usually the closer/real door plate.
        if self._last_target is not None:
            last_x = float(self._last_target["center"][0])
            return min(candidates, key=lambda r: abs(float(r["center"][0]) - last_x))
        return max(candidates, key=self._box_area)

    def _is_stable(self) -> bool:
        if len(self._x_hist) < self.stable_samples:
            return False
        xs = sorted(float(x) for x in self._x_hist)
        med = xs[len(xs) // 2]
        return max(abs(x - med) for x in xs) <= self.x_stable_px

    def update(
        self,
        ocr_results: Optional[Sequence[dict]],
        frame_id: int,
        front_distance: float,
    ) -> DoorNavStatus:
        """
        Call every control frame while Scenario2 is in TRAVERSE.

        Simplified behavior for demo stability:
          - Before target room is seen: inactive, Scenario2 keeps normal traverse.
          - Once target room is seen once: door_nav owns the robot.
          - It does NOT turn left/right based on OCR x offset.
          - It only creeps forward slowly and stops by ultrasonic distance.
          - If OCR later loses the target, it still keeps creeping until ultrasonic stops.
        """
        if self._done:
            cmd = MotorCommand(0, 0, RobotState.STOP, "door_nav done")
            return DoorNavStatus(True, True, cmd, cmd.reason)

        # 1) Try to acquire / refresh the target room from fresh OCR results.
        target = self._find_target(ocr_results)
        if target is not None:
            self._active = True
            self._last_seen_frame = int(frame_id)
            self._last_target = target
            self._x_hist.append(float(target["center"][0]))

        # 2) Target has never been seen: do not take control yet.
        if not self._active:
            return DoorNavStatus(False, False, None, "door_nav inactive")

        # 3) Once the target has been seen, ultrasonic becomes the final stop condition.
        #    This remains true even if OCR loses the number later.
        lost_frames = 0
        if self._last_seen_frame is not None:
            lost_frames = int(frame_id) - int(self._last_seen_frame)

        if front_distance <= self.stop_distance_m:
            self._done = True
            cmd = MotorCommand(
                0, 0, RobotState.STOP,
                f"door_nav ultrasonic stop d={front_distance:.2f}m target={self.target_room}",
            )
            return DoorNavStatus(
                True, True, cmd, cmd.reason,
                target_x=(self._x_hist[-1] if self._x_hist else None),
                stable=True,
                lost_frames=lost_frames,
            )

        # 4) No OCR-offset turning anymore. After target is acquired, just creep forward.
        #    This avoids random left/right turns caused by unstable OCR box centers.
        cmd = MotorCommand(
            self.approach_speed, self.approach_speed, RobotState.FORWARD,
            f"door_nav acquired {self.target_room}; creep straight until ultrasonic stop "
            f"d={front_distance:.2f} lost={lost_frames}",
        )
        return DoorNavStatus(
            True, False, cmd, cmd.reason,
            target_x=(self._x_hist[-1] if self._x_hist else None),
            stable=True,
            lost_frames=lost_frames,
        )


class Scenario2Controller:
    def __init__(
        self,
        target_room: str = DEFAULT_TARGET_ROOM,
        frame_width: int = FRAME_WIDTH,
        enable_door_nav: bool = True,
        approach_speed: float = APPROACH_SPEED,
        door_turn_speed: float = DOOR_TURN_SPEED,
        center_deadband_px: int = CENTER_DEADBAND_PX,
        x_stable_px: int = X_STABLE_PX,
        stable_samples: int = STABLE_SAMPLES,
        lost_fast_frames: int = LOST_FAST_FRAMES,
        door_stop_dist: float = DOOR_STOP_DIST,
    ):
        self.state = S2.CRUISE
        self._t_enter = time.time()
        self._open_since = None
        self._door_nav = DoorOcrNavigator(
            target_room=target_room,
            frame_width=frame_width,
            approach_speed=approach_speed,
            turn_speed=door_turn_speed,
            center_deadband_px=center_deadband_px,
            x_stable_px=x_stable_px,
            stable_samples=stable_samples,
            lost_fast_frames=lost_fast_frames,
            stop_distance_m=door_stop_dist,
        ) if enable_door_nav else None

    @property
    def done(self):
        return self.state == S2.DONE

    @property
    def door_nav_active(self) -> bool:
        return bool(self._door_nav is not None and self._door_nav.active)

    def reset(self):
        self.state = S2.CRUISE
        self._t_enter = time.time()
        self._open_since = None
        if self._door_nav is not None:
            self._door_nav.reset()

    def _enter(self, s):
        self.state = s
        self._t_enter = time.time()
        self._open_since = None

    def _confirm_open(self, d, now):
        """True once front has been >= OPEN_DIST continuously for OPEN_CONFIRM_SEC."""
        if d >= OPEN_DIST:                    # 99.9 (no echo / >4m) counts as open
            if self._open_since is None:
                self._open_since = now
            return (now - self._open_since) >= OPEN_CONFIRM_SEC
        self._open_since = None
        return False

    def update(
        self,
        front_distance,
        destination_reached: bool = False,
        ocr_results: Optional[Sequence[dict]] = None,
        frame_id: int = 0,
    ):
        """
        front_distance: meters (99.9 = no echo / out of range).
        destination_reached: legacy bool; only used when door_nav is disabled.
        ocr_results: GlobalOcrReader results; pass fresh results on OCR frames.
        frame_id: current camera/control frame id.
        Returns a MotorCommand for this frame.
        """
        now = time.time()
        d = front_distance
        el = now - self._t_enter

        if self.state == S2.CRUISE:
            if d < DANGER_DIST:
                self._enter(S2.SCAN_LEFT)
                return MotorCommand(0, 0, RobotState.STOP,
                                    f"S2 danger d={d:.2f} -> scan left")
            return MotorCommand(CRUISE_SPEED, CRUISE_SPEED, RobotState.FORWARD,
                                f"S2 cruise d={d:.2f}")

        if self.state == S2.SCAN_LEFT:
            if self._confirm_open(d, now):
                self._enter(S2.TRAVERSE)
                return MotorCommand(GO_SPEED, GO_SPEED, RobotState.FORWARD,
                                    f"S2 left opening d={d:.2f} -> go")
            if el > SCAN_LEFT_MAX_SEC:
                self._enter(S2.TURN_RIGHT)
                return MotorCommand(TURN_SPEED, TURN_SPEED, RobotState.TURN_RIGHT,
                                    "S2 left blocked -> turn right")
            return MotorCommand(SCAN_SPEED, SCAN_SPEED, RobotState.TURN_LEFT,
                                f"S2 scan-left d={d:.2f} t={el:.1f}")

        if self.state == S2.TURN_RIGHT:
            if BLIND_RIGHT:
                if el > TURN_RIGHT_BLIND_SEC:
                    self._enter(S2.TRAVERSE)
                    return MotorCommand(GO_SPEED, GO_SPEED, RobotState.FORWARD,
                                        "S2 blind-180 done -> go")
                return MotorCommand(TURN_SPEED, TURN_SPEED, RobotState.TURN_RIGHT,
                                    f"S2 blind-right t={el:.1f}")
            # sensor-confirmed right turn
            if self._confirm_open(d, now):
                self._enter(S2.TRAVERSE)
                return MotorCommand(GO_SPEED, GO_SPEED, RobotState.FORWARD,
                                    f"S2 right opening d={d:.2f} -> go")
            if el > TURN_RIGHT_MAX_SEC:
                self._enter(S2.TRAVERSE)
                return MotorCommand(GO_SPEED, GO_SPEED, RobotState.FORWARD,
                                    "S2 right timeout -> go (check setup)")
            return MotorCommand(TURN_SPEED, TURN_SPEED, RobotState.TURN_RIGHT,
                                f"S2 scan-right d={d:.2f} t={el:.1f}")

        if self.state == S2.TRAVERSE:
            # New behavior: OCR door-nav owns the robot after target_room is acquired.
            # It slows down, waits for x stability, turns toward the target, and uses
            # ultrasonic distance for the final stop.
            if self._door_nav is not None:
                door_status = self._door_nav.update(
                    ocr_results=ocr_results,
                    frame_id=frame_id,
                    front_distance=d,
                )
                if door_status.done:
                    self._enter(S2.DONE)
                    return door_status.command
                if door_status.active and door_status.command is not None:
                    return door_status.command

            # Legacy fallback, useful if enable_door_nav=False.
            if destination_reached:
                self._enter(S2.DONE)
                return MotorCommand(0, 0, RobotState.STOP,
                                    "S2 destination reached -> DONE")
            return MotorCommand(GO_SPEED, GO_SPEED, RobotState.FORWARD,
                                f"S2 traverse d={d:.2f}")

        # DONE
        return MotorCommand(0, 0, RobotState.STOP, "S2 DONE")
