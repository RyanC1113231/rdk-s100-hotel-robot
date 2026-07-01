"""
Task 3 Core Controller (v4 - S-curve evade)

Decision chain:
  YOLO bbox + ultrasonic front_distance -> threat -> FSM -> MotorCommand

v4 changes:
  - DANGER no longer STOP; triggers EVADE (arc turn while moving forward)
  - Only EMERGENCY (ultrasonic < threshold) hard-stops
  - Added EVADE state with direction lock + post-evade STRAIGHTEN counter-steer
  - WARNING zone uses arc turns (differential forward), not in-place spins
  - S-curve: EVADE arc -> STRAIGHTEN counter-arc -> FORWARD
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

# ==================== bbox thresholds (pixels) ====================
DANGER_BBOX_ENTER = 400       # bbox_h > 400 enter danger -> EVADE
DANGER_BBOX_EXIT  = 350       # bbox_h < 350 exit danger
WARNING_BBOX_ENTER = 280      # bbox_h > 280 enter warning -> arc adjust
WARNING_BBOX_EXIT  = 230      # bbox_h < 230 exit warning
EMERGENCY_BBOX = 470          # bbox_h > 470 emergency stop

# ---- ultrasonic emergency ----
FRONT_DISTANCE_EMERGENCY = 0.5   # < 0.5m unconditional stop

# ---- general parameters ----
LATERAL_THRESHOLD = 0.15
MAX_SPEED = 1.0
SLOW_SPEED = 0.5
IMAGE_WIDTH = 640
IMAGE_CENTER_X = IMAGE_WIDTH / 2

# ---- Track ID jump tolerance ----
ID_MATCH_BBOX_H_THRESHOLD = 80
ID_MATCH_LATERAL_THRESHOLD = 0.3

# ---- Persistence buffer ----
PERSIST_HOLD_FRAMES = 5
PERSIST_RAMP_FRAMES = 3

# ---- bbox_h_rate ----
APPROACH_RATE_THRESHOLD = 15
RETREAT_RATE_THRESHOLD = -10

# ---- EVADE parameters ----
EVADE_OUTER_SPEED = 0.7       # outer wheel speed during evade arc
EVADE_INNER_SPEED = 0.1       # inner wheel speed (slow, keeps moving fwd)
EVADE_MIN_FRAMES = 5          # minimum evade frames before allowing exit
STRAIGHTEN_SPEED = 0.6        # speed during counter-steer phase
STRAIGHTEN_RATIO = 1.6        # outer/inner ratio during straighten (gentler than evade)
STRAIGHTEN_FRAMES = 8         # how long to counter-steer after evade

# ---- WARNING arc parameters ----
WARNING_ARC_RATIO = 1.5       # outer/inner wheel ratio for warning arc turns


class ThreatLevel(Enum):
    SAFE = "safe"
    WARNING = "warning"
    DANGER = "danger"

class RobotState(Enum):
    FORWARD = "forward"
    SLOW = "slow"
    TURN_LEFT = "turn_left"       # in-place spin (stage manager only)
    TURN_RIGHT = "turn_right"     # in-place spin (stage manager only)
    EVADE = "evade"               # arc turn away from obstacle (keeps moving)
    STRAIGHTEN = "straighten"     # counter-steer after evade to resume heading
    STOP = "stop"
    IDLE = "idle"
    BACKWARD = "backward"
    IN_PLACE_TURN = "in_place_turn"

class EvadeDir(Enum):
    NONE = "none"
    LEFT = "left"       # evade to the left (obstacle is on right)
    RIGHT = "right"     # evade to the right (obstacle is on left/center)

@dataclass
class Target:
    track_id: int
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    confidence: float

    @property
    def lateral(self) -> float:
        """bbox center normalized to [-1, +1], negative=left, positive=right"""
        center_x = self.bbox_x + self.bbox_w / 2
        return (center_x - IMAGE_CENTER_X) / IMAGE_CENTER_X

    @property
    def is_emergency(self) -> bool:
        return self.bbox_h > EMERGENCY_BBOX

@dataclass
class MotorCommand:
    left_speed: float = 0.0
    right_speed: float = 0.0
    state: RobotState = RobotState.IDLE
    reason: str = ""


class Task3Controller:
    def __init__(self):
        self.reset()

    def reset(self):
        self._prev_targets: list[Target] = []
        self._prev_state = RobotState.FORWARD
        self._state_frames = 0
        self._threat_level = ThreatLevel.SAFE
        self._history: list[MotorCommand] = []
        self._last_known_closest_id = None
        self._last_known_bbox_h = None
        self._last_known_lateral = None
        # ---- persistence buffer ----
        self._disappeared_frames = 0
        self._last_valid_cmd: Optional[MotorCommand] = None
        # ---- bbox_h_rate ----
        self._prev_bbox_h: Optional[float] = None
        self._bbox_h_rate: float = 0.0
        # ---- evade state ----
        self._evade_dir = EvadeDir.NONE
        self._evade_frames = 0
        self._straighten_frames = 0

    def process(self, targets: list[Target], frame_id: int = 0,
                front_distance: float = 99.9) -> MotorCommand:
        """Main entry point."""
        self._prev_targets = targets

        # ---- ultrasonic emergency (highest priority) ----
        if front_distance < FRONT_DISTANCE_EMERGENCY:
            cmd = MotorCommand(0.0, 0.0, RobotState.STOP,
                               f"US emergency! front={front_distance:.2f}m")
            self._abort_evade()
            self._commit_state(cmd)
            self._last_valid_cmd = MotorCommand(0.0, 0.0, RobotState.STOP, cmd.reason)
            self._push_history(cmd)
            return cmd

        # ---- find closest target (largest bbox_h) ----
        closest = None
        if targets:
            closest = max(targets, key=lambda t: t.bbox_h)

            # Track ID jump tolerance
            if (self._last_known_closest_id is not None
                    and closest.track_id != self._last_known_closest_id
                    and self._last_known_bbox_h is not None
                    and self._last_known_lateral is not None):
                bbox_h_drift = abs(closest.bbox_h - self._last_known_bbox_h)
                lat_drift = abs(closest.lateral - self._last_known_lateral)
                if (bbox_h_drift < ID_MATCH_BBOX_H_THRESHOLD
                        and lat_drift < ID_MATCH_LATERAL_THRESHOLD):
                    closest.track_id = self._last_known_closest_id

        # ---- update target memory & bbox_h_rate ----
        if closest:
            self._disappeared_frames = 0
            self._last_known_closest_id = closest.track_id
            self._last_known_lateral = closest.lateral

            if self._prev_bbox_h is not None:
                self._bbox_h_rate = closest.bbox_h - self._prev_bbox_h
            else:
                self._bbox_h_rate = 0.0
            self._prev_bbox_h = closest.bbox_h
            self._last_known_bbox_h = closest.bbox_h
        else:
            self._disappeared_frames += 1
            self._bbox_h_rate = 0.0

        # ---- threat level update (bbox_h + hysteresis) ----
        if closest is not None:
            h = closest.bbox_h
            if self._threat_level == ThreatLevel.DANGER:
                if h < DANGER_BBOX_EXIT:
                    self._threat_level = ThreatLevel.WARNING
            elif self._threat_level == ThreatLevel.WARNING:
                if h > DANGER_BBOX_ENTER:
                    self._threat_level = ThreatLevel.DANGER
                elif h < WARNING_BBOX_EXIT:
                    self._threat_level = ThreatLevel.SAFE
            else:  # SAFE
                if h > DANGER_BBOX_ENTER:
                    self._threat_level = ThreatLevel.DANGER
                elif h > WARNING_BBOX_ENTER:
                    self._threat_level = ThreatLevel.WARNING

        # ---- decision ----
        cmd = self._decide(closest)
        self._push_history(cmd)
        return cmd

    def _decide(self, closest: Optional[Target]) -> MotorCommand:
        # ==== Phase check: are we in STRAIGHTEN (post-evade counter-steer)? ====
        if self._straighten_frames > 0:
            cmd = self._do_straighten(closest)
            self._commit_state(cmd)
            return cmd

        # ==== target disappeared -> persistence buffer ====
        if closest is None:
            # If we were evading and target disappeared, finish with straighten
            if self._evade_dir != EvadeDir.NONE:
                self._straighten_frames = STRAIGHTEN_FRAMES
                self._evade_frames = 0
                cmd = self._do_straighten(closest)
                self._commit_state(cmd)
                return cmd
            cmd = self._handle_disappeared()
            self._commit_state(cmd)
            return cmd

        h = closest.bbox_h
        lat = closest.lateral
        rate = self._bbox_h_rate

        # ==== EMERGENCY: bbox too large -> hard stop ====
        if h > EMERGENCY_BBOX:
            self._abort_evade()
            cmd = MotorCommand(0.0, 0.0, RobotState.STOP,
                               f"EMERGENCY bbox_h={h:.0f}px")
            self._last_valid_cmd = cmd
            self._commit_state(cmd)
            return cmd

        # ==== DANGER: active EVADE (arc turn away, keep moving) ====
        if self._threat_level == ThreatLevel.DANGER:
            cmd = self._do_evade(closest)
            self._last_valid_cmd = MotorCommand(
                cmd.left_speed, cmd.right_speed, cmd.state, cmd.reason)
            self._commit_state(cmd)
            return cmd

        # ==== If we just exited DANGER (evade_dir still set), start straighten ====
        if self._evade_dir != EvadeDir.NONE:
            self._straighten_frames = STRAIGHTEN_FRAMES
            self._evade_frames = 0
            cmd = self._do_straighten(closest)
            self._commit_state(cmd)
            return cmd

        # ==== WARNING: slow + arc adjust ====
        if self._threat_level == ThreatLevel.WARNING:
            speed = SLOW_SPEED
            if rate > APPROACH_RATE_THRESHOLD:
                speed = SLOW_SPEED * 0.5

            if abs(lat) > LATERAL_THRESHOLD:
                if lat < 0:
                    # obstacle left -> arc right (left fast, right slow)
                    inner = speed
                    outer = speed * WARNING_ARC_RATIO
                    cmd = MotorCommand(outer, inner, RobotState.SLOW,
                                       f"WARNING arc-R lat={lat:.2f} h={h:.0f} rate={rate:+.1f}")
                else:
                    # obstacle right -> arc left (right fast, left slow)
                    inner = speed
                    outer = speed * WARNING_ARC_RATIO
                    cmd = MotorCommand(inner, outer, RobotState.SLOW,
                                       f"WARNING arc-L lat={lat:.2f} h={h:.0f} rate={rate:+.1f}")
            else:
                # obstacle centered in WARNING -> slow and prepare to evade
                # slightly favor one side to avoid head-on stall
                cmd = MotorCommand(speed, speed / 1.15, RobotState.SLOW,
                                   f"WARNING center h={h:.0f} rate={rate:+.1f} (drift-R)")

            self._last_valid_cmd = MotorCommand(
                cmd.left_speed, cmd.right_speed, cmd.state, cmd.reason)
            self._commit_state(cmd)
            return cmd

        # ==== SAFE: full speed + micro-adjust ====
        if abs(lat) > LATERAL_THRESHOLD:
            if lat < 0:
                cmd = MotorCommand(MAX_SPEED , MAX_SPEED / 1.15, RobotState.FORWARD,
                                   f"micro-R lat={lat:.2f} h={h:.0f}")
            else:
                cmd = MotorCommand(MAX_SPEED / 1.15, MAX_SPEED, RobotState.FORWARD,
                                   f"micro-L lat={lat:.2f} h={h:.0f}")
        else:
            cmd = MotorCommand(MAX_SPEED, MAX_SPEED, RobotState.FORWARD,
                               f"SAFE h={h:.0f} rate={rate:+.1f}")

        self._last_valid_cmd = MotorCommand(
            cmd.left_speed, cmd.right_speed, cmd.state, cmd.reason)
        self._commit_state(cmd)
        return cmd

    # ----------------------------------------------------------------
    #  EVADE: arc turn away from obstacle while maintaining forward motion
    # ----------------------------------------------------------------
    def _do_evade(self, closest: Target) -> MotorCommand:
        lat = closest.lateral
        h = closest.bbox_h
        rate = self._bbox_h_rate

        # Lock evade direction on first frame
        if self._evade_dir == EvadeDir.NONE:
            # Choose: turn away from obstacle's lateral position
            # If obstacle is center or left of center -> evade right
            # If obstacle is right of center -> evade left
            if lat <= 0.1:
                self._evade_dir = EvadeDir.RIGHT
            else:
                self._evade_dir = EvadeDir.LEFT
            self._evade_frames = 0

        self._evade_frames += 1

        # Apply arc: outer wheel fast, inner wheel slow, both forward
        if self._evade_dir == EvadeDir.RIGHT:
            # Turn right: left wheel fast (outer), right wheel slow (inner)
            cmd = MotorCommand(
                EVADE_OUTER_SPEED, EVADE_INNER_SPEED, RobotState.EVADE,
                f"EVADE-R f{self._evade_frames} lat={lat:.2f} h={h:.0f} rate={rate:+.1f}")
        else:
            # Turn left: right wheel fast (outer), left wheel slow (inner)
            cmd = MotorCommand(
                EVADE_INNER_SPEED, EVADE_OUTER_SPEED, RobotState.EVADE,
                f"EVADE-L f{self._evade_frames} lat={lat:.2f} h={h:.0f} rate={rate:+.1f}")

        return cmd

    # ----------------------------------------------------------------
    #  STRAIGHTEN: counter-steer after evade to resume original heading
    # ----------------------------------------------------------------
    def _do_straighten(self, closest: Optional[Target]) -> MotorCommand:
        self._straighten_frames -= 1
        remaining = self._straighten_frames

        # Gentle counter-steer: opposite direction of evade
        if self._evade_dir == EvadeDir.RIGHT:
            # We evaded right, now steer left to straighten
            inner = STRAIGHTEN_SPEED
            outer = STRAIGHTEN_SPEED * STRAIGHTEN_RATIO
            cmd = MotorCommand(
                inner, outer, RobotState.STRAIGHTEN,
                f"STRAIGHTEN-L rem={remaining}")
        else:
            # We evaded left, now steer right to straighten
            inner = STRAIGHTEN_SPEED
            outer = STRAIGHTEN_SPEED * STRAIGHTEN_RATIO
            cmd = MotorCommand(
                outer, inner, RobotState.STRAIGHTEN,
                f"STRAIGHTEN-R rem={remaining}")

        # Check if new DANGER appeared during straighten -> re-evade
        if closest is not None and self._threat_level == ThreatLevel.DANGER:
            self._straighten_frames = 0
            self._evade_dir = EvadeDir.NONE  # re-evaluate direction
            return self._do_evade(closest)

        # Straighten complete -> reset evade state
        if remaining <= 0:
            self._abort_evade()

        self._last_valid_cmd = MotorCommand(
            cmd.left_speed, cmd.right_speed, cmd.state, cmd.reason)
        return cmd

    def _abort_evade(self):
        """Reset all evade state."""
        self._evade_dir = EvadeDir.NONE
        self._evade_frames = 0
        self._straighten_frames = 0

    # ----------------------------------------------------------------
    #  Persistence buffer (unchanged)
    # ----------------------------------------------------------------
    def _handle_disappeared(self) -> MotorCommand:
        total_persist = PERSIST_HOLD_FRAMES + PERSIST_RAMP_FRAMES

        if self._last_valid_cmd is None:
            self._clear_target_memory()
            return MotorCommand(MAX_SPEED, MAX_SPEED, RobotState.FORWARD,
                                "no target, forward")

        # Phase 1: hold last command
        if self._disappeared_frames <= PERSIST_HOLD_FRAMES:
            held = self._last_valid_cmd
            return MotorCommand(
                held.left_speed, held.right_speed, held.state,
                f"[hold {self._disappeared_frames}/{PERSIST_HOLD_FRAMES}] {held.reason}")

        # Phase 2: ramp back to forward
        if self._disappeared_frames <= total_persist:
            ramp_step = self._disappeared_frames - PERSIST_HOLD_FRAMES
            t = ramp_step / PERSIST_RAMP_FRAMES
            held = self._last_valid_cmd
            left = held.left_speed + (MAX_SPEED - held.left_speed) * t
            right = held.right_speed + (MAX_SPEED - held.right_speed) * t
            return MotorCommand(
                round(left, 3), round(right, 3), RobotState.FORWARD,
                f"[ramp {ramp_step}/{PERSIST_RAMP_FRAMES}]")

        # Phase 3: fully resumed
        self._clear_target_memory()
        return MotorCommand(MAX_SPEED, MAX_SPEED, RobotState.FORWARD,
                            "no target, forward")

    def _clear_target_memory(self):
        self._last_known_closest_id = None
        self._last_known_bbox_h = None
        self._last_known_lateral = None
        self._last_valid_cmd = None
        self._prev_bbox_h = None
        self._bbox_h_rate = 0.0
        self._threat_level = ThreatLevel.SAFE
        self._abort_evade()

    def _commit_state(self, cmd: MotorCommand):
        if cmd.state != self._prev_state:
            self._state_frames = 0
        else:
            self._state_frames += 1
        self._prev_state = cmd.state

    def _push_history(self, cmd: MotorCommand):
        self._history.append(cmd)
        if len(self._history) > 50:
            self._history.pop(0)

    def get_history(self, n: int = 10) -> list[MotorCommand]:
        return self._history[-n:] if self._history else []
