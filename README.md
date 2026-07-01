# RDK S100 Hotel Corridor Robot

A vision-based hotel corridor robot prototype built on the D-Robotics RDK S100 platform. The project integrates real-time object detection, multi-object tracking, OCR door-number recognition, ultrasonic emergency stopping, optional VLM semantic understanding, motor control, MJPEG preview, and voice-triggered startup.

This repository is intended as a competition/demo codebase and a reference implementation for small indoor service-robot experiments.

## Features

- **Real-time geometric perception** using YOLO on the RDK S100 BPU.
- **Multi-object tracking** using a lightweight SORT tracker.
- **Reactive obstacle avoidance** based on bounding-box size, target position, and ultrasonic distance.
- **OCR door-number recognition** using PaddleOCR models on the RDK S100.
- **Scenario 2 navigation demo**: when the front path is blocked, the robot scans for an open path with the ultrasonic sensor, then moves forward and uses OCR to approach a target room number.
- **Scenario 3 target selection demo**: when two door-number boards appear, the robot selects the target room number and turns toward it.
- **Ultrasonic emergency stop** using HC-SR04 distance readings.
- **MDD10A dual motor driver control** through RDK S100 GPIO.
- **Browser MJPEG preview** for remote debugging over SSH/LAN.
- **Voice command launcher** using a Yahboom CI1302 voice module over UART.

## Hardware

Main hardware used in this project:

- D-Robotics RDK S100
- USB camera
- HC-SR04 ultrasonic sensor
- Cytron MDD10A dual motor driver
- Two DC geared motors
- External motor battery/power supply
- Yahboom CI1302 AI voice module, optional

## Important Wiring Notes

### Motor driver

Default motor pins are defined in `motor_controller.py`:

| Signal | BOARD pin |
| --- | --- |
| Left PWM | 37 |
| Left DIR | 29 |
| Right PWM | 31 |
| Right DIR | 36 |
| GND | 39 |

DIR polarity:

- `LOW` = forward
- `HIGH` = backward

### Ultrasonic sensor

Default pins are defined in `ultrasonic.py`:

| Signal | Default BOARD pin | Notes |
| --- | --- | --- |
| TRIG | 11 | 3.3 V GPIO output |
| ECHO | 13 | Fast SoC GPIO input, voltage divider required |

The HC-SR04 ECHO pin outputs 5 V. The RDK S100 GPIO input is not 5 V tolerant, so a resistor voltage divider is required before connecting ECHO to the board.

## Repository Structure

Recommended structure:

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── main_robot.py
├── motor_controller.py
├── ultrasonic.py
├── task3_core.py
├── scenario2_controller.py
├── scenario3_controller.py
├── global_ocr.py
├── bpu_perception_stream.py
├── combined_perception.py
├── sort_tracker.py
├── vlm_semantic.py
├── voice_launcher.py
└── docs/
```

## Quick Start

Run the main robot loop without driving the motors:

```bash
python3 main_robot.py --dry-run --no-vlm
```

Run the main robot loop with motors enabled and speed limited:

```bash
python3 main_robot.py --no-vlm --max-speed 35
```

Run Scenario 2 with a target room number:

```bash
python3 main_robot.py --scenario2 --target-room 1207 --no-vlm --max-speed 35
```

Run without the browser preview:

```bash
python3 main_robot.py --scenario2 --target-room 1207 --no-vlm --max-speed 35 --no-stream
```

Start the voice launcher:

```bash
python3 voice_launcher.py --target-room 1207 --speed 35 --ocr-every 8
```

Run the voice launcher in normal Task3 mode instead of Scenario 2:

```bash
python3 voice_launcher.py --task3 --speed 35
```

## Browser Preview

When MJPEG preview is enabled, the program prints a URL similar to:

```text
http://<s100-ip>:8091
```

Open this URL in a browser on the same network to view the live camera stream with overlays.

## Models

Model files are **not included** in this repository.

Expected model paths are defined in the source files, for example:

- YOLO HBM model path in `bpu_perception_stream.py`
- PaddleOCR det/rec HBM model paths in `global_ocr.py`
- SmolVLM GGUF/HBM model paths in `vlm_semantic.py`

Please download or prepare the required models separately and update the paths in the corresponding files if needed.

## Dependencies

Install Python dependencies:

```bash
pip3 install -r requirements.txt
```

Some dependencies are provided by the RDK S100 system image or D-Robotics SDK and cannot be installed from PyPI, including:

- `Hobot.GPIO`
- `hbm_runtime`
- `/app/pydev_demo` utility modules

## Safety Notes

- Always test with the wheels lifted off the ground first.
- Use `--dry-run` before enabling motors.
- Keep `--max-speed` low during early testing.
- Verify motor direction before running autonomous modes.
- Use a voltage divider on the HC-SR04 ECHO pin.
- Do not power motors directly from the RDK S100.

## Project Status

This is a working prototype for a hotel corridor robot demo. It focuses on practical integration of perception, decision-making, and motor execution on RDK S100 hardware.

Current limitations:

- The navigation logic is demo-oriented rather than a full SLAM/Nav2 stack.
- VLM inference is slower than YOLO/OCR and should be used carefully on limited compute.
- Door-number OCR depends on lighting, camera angle, and printed text quality.
- Motion timing and turning angles may need hardware-specific calibration.

## License

This project is released under the MIT License. See `LICENSE` for details.
