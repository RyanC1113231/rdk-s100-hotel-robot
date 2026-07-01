# RDK S100 酒店走廊机器人

这是一个基于地瓜机器人 RDK S100 平台开发的视觉感知酒店走廊机器人原型。项目集成了实时目标检测、多目标跟踪、OCR 门牌识别、超声波急停、可选 VLM 语义理解、电机控制、MJPEG 浏览器预览以及语音触发启动等功能。

本仓库主要作为竞赛 / Demo 代码库，也可作为小型室内服务机器人实验的参考实现。

## 功能特性

- **实时几何感知**：使用 RDK S100 BPU 运行 YOLO 目标检测。
- **多目标跟踪**：使用轻量级 SORT 跟踪器进行目标 ID 维护。
- **反应式避障**：根据目标框大小、目标位置以及超声波距离进行避障决策。
- **OCR 门牌识别**：基于 RDK S100 上的 PaddleOCR 模型识别门牌号。
- **场景 2 导航 Demo**：当前方路径被阻挡时，机器人利用超声波扫描开放通路，随后前进，并通过 OCR 识别目标房间号进行靠近。
- **场景 3 目标选择 Demo**：当画面中出现两个门牌纸板时，机器人识别目标房间号，并朝目标方向转向。
- **超声波急停**：使用 HC-SR04 测距结果作为近距离安全兜底。
- **MDD10A 双电机驱动控制**：通过 RDK S100 GPIO 控制双电机驱动板。
- **浏览器 MJPEG 实时预览**：支持在 SSH / 局域网环境下远程查看带有检测框和状态信息的画面。
- **语音命令启动器**：通过 Yahboom CI1302 语音模块的 UART 指令启动或停止主程序。

## 硬件组成

本项目主要使用以下硬件：

- 地瓜机器人 RDK S100
- USB 摄像头
- HC-SR04 超声波传感器
- Cytron MDD10A 双路电机驱动板
- 两个直流减速电机
- 外部电机电池 / 电源
- Yahboom CI1302 AI 语音模块，可选

## 重要接线说明

### 电机驱动

默认电机引脚在 `motor_controller.py` 中定义：

| 信号 | BOARD 引脚 |
| --- | --- |
| 左电机 PWM | 37 |
| 左电机 DIR | 29 |
| 右电机 PWM | 31 |
| 右电机 DIR | 36 |
| GND | 39 |

DIR 方向极性：

- `LOW` = 前进
- `HIGH` = 后退

### 超声波传感器

默认引脚在 `ultrasonic.py` 中定义：

| 信号 | 默认 BOARD 引脚 | 说明 |
| --- | --- | --- |
| TRIG | 11 | 3.3 V GPIO 输出 |
| ECHO | 13 | 快速 SoC GPIO 输入，必须使用分压电路 |

HC-SR04 的 ECHO 引脚输出为 5 V。RDK S100 的 GPIO 输入不能直接承受 5 V，因此在连接 ECHO 到开发板前，必须使用电阻分压电路降压。

## 仓库结构

推荐结构如下：

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
