#!/usr/bin/env python3
"""
motor_controller.py  --  S100 + MDD10A dual motor controller
============================================================
Pin layout (BOARD numbering, 3.3V I2C expander):
  Left  PWM = 37    DIR = 29
  Right PWM = 31    DIR = 36
  GND       = 39

DIR polarity: LOW = forward, HIGH = backward
"""

import Hobot.GPIO as GPIO
import threading
import time

LEFT_PWM_PIN  = 37
LEFT_DIR_PIN  = 29
RIGHT_PWM_PIN = 31
RIGHT_DIR_PIN = 36

PWM_FREQ = 100
PWM_PERIOD = 1.0 / PWM_FREQ


class MotorController:
    def __init__(self, auto_start=True):
        GPIO.setmode(GPIO.BOARD)
        for p in [LEFT_PWM_PIN, LEFT_DIR_PIN, RIGHT_PWM_PIN, RIGHT_DIR_PIN]:
            GPIO.setup(p, GPIO.OUT, initial=GPIO.LOW)

        self._lock = threading.Lock()
        self._left_duty = 0
        self._right_duty = 0
        self._left_fwd = True
        self._right_fwd = True
        self._running = False
        self._thread = None

        if auto_start:
            self.start()

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._pwm_loop, daemon=True,
                                         name="motor-pwm")
        self._thread.start()
        print("[motor] PWM thread started")

    def cleanup(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
        for p in [LEFT_PWM_PIN, RIGHT_PWM_PIN]:
            try:
                GPIO.output(p, GPIO.LOW)
            except Exception:
                pass
        GPIO.cleanup()
        print("[motor] cleaned up")

    def set_motors(self, left: float, right: float):
        """left/right: -100~+100, positive=forward negative=backward"""
        with self._lock:
            self._left_fwd = (left >= 0)
            self._right_fwd = (right >= 0)
            self._left_duty = int(min(abs(left), 100))
            self._right_duty = int(min(abs(right), 100))

    def forward(self, speed=50):
        self.set_motors(speed, speed)

    def backward(self, speed=50):
        self.set_motors(-speed, -speed)

    def turn_left(self, speed=50):
        self.set_motors(-speed, speed)

    def turn_right(self, speed=50):
        self.set_motors(speed, -speed)

    def stop(self):
        self.set_motors(0, 0)

    def apply_command(self, cmd):
        """Map MotorCommand to motor output.

        State mapping:
          STOP / IDLE          -> (0, 0)
          TURN_LEFT            -> in-place spin: (-ls, +rs)
          TURN_RIGHT           -> in-place spin: (+ls, -rs)
          IN_PLACE_TURN        -> in-place spin: use speed signs directly
          BACKWARD             -> (-ls, -rs)
          EVADE / STRAIGHTEN / -> differential forward arc: (+ls, +rs)
            SLOW / FORWARD       (speeds already encode the turn ratio)
        """
        from task3_core import RobotState

        state = cmd.state
        ls = abs(cmd.left_speed)
        rs = abs(cmd.right_speed)

        # Normalize 0-1 range to 0-100 duty cycle
        if ls <= 1.0:
            ls *= 100
        if rs <= 1.0:
            rs *= 100

        if state in (RobotState.STOP, RobotState.IDLE):
            self.set_motors(0, 0)

        elif state == RobotState.TURN_LEFT:
            # In-place spin: left backward, right forward
            self.set_motors(-ls, rs)

        elif state == RobotState.TURN_RIGHT:
            # In-place spin: left forward, right backward
            self.set_motors(ls, -rs)

        elif state == RobotState.IN_PLACE_TURN:
            # Stage manager controls signs via left_speed/right_speed directly
            l_raw = cmd.left_speed
            r_raw = cmd.right_speed
            if abs(l_raw) <= 1.0:
                l_raw *= 100
            if abs(r_raw) <= 1.0:
                r_raw *= 100
            self.set_motors(l_raw, r_raw)

        elif state == RobotState.BACKWARD:
            self.set_motors(-ls, -rs)

        else:
            # FORWARD, SLOW, EVADE, STRAIGHTEN:
            # Both wheels forward, differential speed creates arc turn
            self.set_motors(ls, rs)

    def _pwm_loop(self):
        while self._running:
            with self._lock:
                ld = self._left_duty
                rd = self._right_duty
                lf = self._left_fwd
                rf = self._right_fwd

            # DIR: LOW=forward, HIGH=backward
            GPIO.output(LEFT_DIR_PIN, GPIO.LOW if lf else GPIO.HIGH)
            GPIO.output(RIGHT_DIR_PIN, GPIO.LOW if rf else GPIO.HIGH)

            if ld == 0 and rd == 0:
                GPIO.output(LEFT_PWM_PIN, GPIO.LOW)
                GPIO.output(RIGHT_PWM_PIN, GPIO.LOW)
                time.sleep(PWM_PERIOD)
                continue

            on_l = PWM_PERIOD * ld / 100.0
            on_r = PWM_PERIOD * rd / 100.0

            if ld > 0:
                GPIO.output(LEFT_PWM_PIN, GPIO.HIGH)
            if rd > 0:
                GPIO.output(RIGHT_PWM_PIN, GPIO.HIGH)

            if on_l <= on_r:
                time.sleep(on_l)
                if ld > 0:
                    GPIO.output(LEFT_PWM_PIN, GPIO.LOW)
                time.sleep(on_r - on_l)
                if rd > 0:
                    GPIO.output(RIGHT_PWM_PIN, GPIO.LOW)
                remain = PWM_PERIOD - on_r
            else:
                time.sleep(on_r)
                if rd > 0:
                    GPIO.output(RIGHT_PWM_PIN, GPIO.LOW)
                time.sleep(on_l - on_r)
                if ld > 0:
                    GPIO.output(LEFT_PWM_PIN, GPIO.LOW)
                remain = PWM_PERIOD - on_l

            if remain > 0:
                time.sleep(remain)


if __name__ == "__main__":
    mc = MotorController()
    try:
        print("forward 50%, 3s")
        mc.forward(50)
        time.sleep(3)

        print("backward 50%, 3s")
        mc.backward(50)
        time.sleep(3)

        print("turn_left 40%, 2s")
        mc.turn_left(40)
        time.sleep(2)

        print("turn_right 40%, 2s")
        mc.turn_right(40)
        time.sleep(2)

        mc.stop()
        time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        mc.cleanup()
        print("test done")
