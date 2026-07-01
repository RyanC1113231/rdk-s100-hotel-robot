#!/usr/bin/env python3
"""
ultrasonic.py — HC-SR04 超声波测距模块 (RDK S100)

后台线程持续测距，主循环随时读 .distance 拿最新值。

硬件接线:
  VCC  → 5V
  GND  → GND
  TRIG → I2C expander GPIO pin (3.3V, ≥2V 满足 HC-SR04 要求)
  ECHO → ★ SoC 原生 GPIO pin (脉冲计时需要快速响应，expander 太慢)
         ★ HC-SR04 echo 输出 5V，SoC 原生引脚仅耐 1.8V
         ★ 必须加分压电阻: 例如 1kΩ + 1.8kΩ 分压, 5V → ~1.67V

用法:
  from ultrasonic import Ultrasonic
  us = Ultrasonic(trig_pin=★, echo_pin=★)
  us.start()
  dist = us.distance          # 米, 超时/异常返回 NO_OBSTACLE
  us.stop()

在 perception_orchestrator3.py 中接入:
  第 412 行 front_distance = 99.9
  改为     front_distance = us.distance
"""

import time
import threading
import sys

# ===================== ★ 引脚配置 (BOARD 编号) ★ =====================
# 根据实际接线修改。参考 40-pin header 定义:
#   expander pins (3.3V, 慢): 31, 36, 37, 38 ... (31/36/37/38 已被电机占用)
#   SoC 原生 pins (1.8V, 快): 32, 33 ...
TRIG_PIN = 11   # ★ 改成实际 TRIG 引脚 (expander pin, 3.3V)
ECHO_PIN = 13   # ★ 改成实际 ECHO 引脚 (SoC 原生 pin, 加分压!)

# ===================== 参数 =====================
NO_OBSTACLE = 99.9            # 无障碍/超时时返回的距离 (米)
MAX_RANGE_M = 4.0             # HC-SR04 最大量程
TIMEOUT_S = MAX_RANGE_M * 2 / 340.0 + 0.01   # echo 超时 (~0.035s)
MEASURE_INTERVAL = 0.06       # 测量间隔 (秒), HC-SR04 建议 ≥60ms
MEDIAN_WINDOW = 3             # 中值滤波窗口

# ===================== GPIO 抽象层 =====================
_gpio_mod = None

def _init_gpio():
    """尝试加载 Hobot.GPIO, 失败则回退 sysfs"""
    global _gpio_mod
    if _gpio_mod is not None:
        return

    try:
        import Hobot.GPIO as GPIO   # noqa: N811
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)
        _gpio_mod = GPIO
        print("[ultrasonic] GPIO backend: Hobot.GPIO", file=sys.stderr)
    except ImportError:
        _gpio_mod = _SysfsGPIO()
        print("[ultrasonic] GPIO backend: sysfs fallback", file=sys.stderr)


class _SysfsGPIO:
    """最小 sysfs GPIO 封装, 仅在 Hobot.GPIO 不可用时使用"""
    BOARD = "BOARD"
    OUT = "out"
    IN = "in"
    HIGH = 1
    LOW = 0

    # ★ BOARD 编号 → Linux GPIO 编号的映射表
    # 需要根据 S100 实际 pinout 补全
    _BOARD_TO_LINUX = {
        # board_pin: linux_gpio_number
        # 示例 (需要你 `cat /sys/kernel/debug/gpio` 确认):
        # 32: 443, 33: 444, ...
    }

    def __init__(self):
        self._exported = set()

    def setmode(self, mode):
        pass

    def setwarnings(self, v):
        pass

    def setup(self, pin, direction):
        gpio_num = self._to_linux(pin)
        if gpio_num not in self._exported:
            self._write("/sys/class/gpio/export", str(gpio_num))
            time.sleep(0.05)
            self._exported.add(gpio_num)
        self._write(f"/sys/class/gpio/gpio{gpio_num}/direction", direction)

    def output(self, pin, value):
        gpio_num = self._to_linux(pin)
        self._write(f"/sys/class/gpio/gpio{gpio_num}/value", str(value))

    def input(self, pin):
        gpio_num = self._to_linux(pin)
        return int(self._read(f"/sys/class/gpio/gpio{gpio_num}/value"))

    def cleanup(self, pin=None):
        pins = [pin] if pin else list(self._exported)
        for p in pins:
            try:
                gpio_num = self._to_linux(p) if p in (self._BOARD_TO_LINUX or {}) else p
                self._write("/sys/class/gpio/unexport", str(gpio_num))
            except Exception:
                pass

    def _to_linux(self, board_pin):
        if board_pin in self._BOARD_TO_LINUX:
            return self._BOARD_TO_LINUX[board_pin]
        raise ValueError(f"BOARD pin {board_pin} 未在 _BOARD_TO_LINUX 映射表中, "
                         f"请运行 cat /sys/kernel/debug/gpio 查找对应编号")

    @staticmethod
    def _write(path, val):
        with open(path, "w") as f:
            f.write(val)

    @staticmethod
    def _read(path):
        with open(path, "r") as f:
            return f.read().strip()


# ===================== 超声波传感器 =====================

class Ultrasonic:
    def __init__(self, trig_pin=TRIG_PIN, echo_pin=ECHO_PIN):
        if trig_pin == 0 or echo_pin == 0:
            raise ValueError("★ 请设置 TRIG_PIN 和 ECHO_PIN! 参见文件顶部注释。")
        self._trig = trig_pin
        self._echo = echo_pin
        self._distance = NO_OBSTACLE
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._buffer = []   # 中值滤波缓冲

    @property
    def distance(self) -> float:
        """最新测距结果 (米)。线程安全。"""
        with self._lock:
            return self._distance

    def start(self):
        """初始化 GPIO 并启动后台测距线程。"""
        _init_gpio()
        gpio = _gpio_mod
        gpio.setup(self._trig, gpio.OUT)
        gpio.setup(self._echo, gpio.IN)
        gpio.output(self._trig, gpio.LOW)
        time.sleep(0.1)   # 让传感器稳定

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="ultrasonic")
        self._thread.start()
        print(f"[ultrasonic] 启动 trig={self._trig} echo={self._echo}", file=sys.stderr)

    def stop(self):
        """停止后台线程并清理 GPIO。"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        try:
            _gpio_mod.cleanup(self._trig)
            _gpio_mod.cleanup(self._echo)
        except Exception:
            pass
        print("[ultrasonic] 已停止", file=sys.stderr)

    # ── 内部 ──

    def _measure_once(self) -> float:
        """单次测距, 返回距离 (米)。失败返回 NO_OBSTACLE。"""
        gpio = _gpio_mod

        # 发送 10μs 触发脉冲
        gpio.output(self._trig, gpio.HIGH)
        time.sleep(0.00001)          # 10μs
        gpio.output(self._trig, gpio.LOW)

        # 等待 echo 上升沿
        t_start = time.time()
        deadline = t_start + TIMEOUT_S
        while gpio.input(self._echo) == gpio.LOW:
            if time.time() > deadline:
                return NO_OBSTACLE

        pulse_start = time.time()

        # 等待 echo 下降沿
        while gpio.input(self._echo) == gpio.HIGH:
            if time.time() > deadline:
                return NO_OBSTACLE

        pulse_end = time.time()

        # 距离 = 声速 × 时间 / 2
        dist = (pulse_end - pulse_start) * 340.0 / 2.0

        if dist > MAX_RANGE_M:
            return NO_OBSTACLE

        return round(dist, 3)

    def _loop(self):
        """后台持续测距 + 中值滤波。"""
        while self._running:
            d = self._measure_once()
            self._buffer.append(d)
            if len(self._buffer) > MEDIAN_WINDOW:
                self._buffer.pop(0)

            # 中值滤波
            sorted_buf = sorted(self._buffer)
            median = sorted_buf[len(sorted_buf) // 2]

            with self._lock:
                self._distance = median

            time.sleep(MEASURE_INTERVAL)


# ===================== 独立测试 =====================
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--trig", type=int, default=TRIG_PIN, help="TRIG pin (BOARD)")
    p.add_argument("--echo", type=int, default=ECHO_PIN, help="ECHO pin (BOARD)")
    args = p.parse_args()

    us = Ultrasonic(trig_pin=args.trig, echo_pin=args.echo)
    us.start()
    try:
        while True:
            d = us.distance
            bar = "#" * min(int(d * 10), 50) if d < NO_OBSTACLE else "---"
            print(f"\r距离: {d:6.3f}m  [{bar:<50s}]", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()
    finally:
        us.stop()
