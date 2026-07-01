"""
Task 6 多任务冲突仲裁器 (v2)

v2 新增: 接入 VLM 语义流（双流融合）
  - 高频几何流: Task 3 (YOLO+SORT, 15-30fps) → 反应式避障
  - 低频语义流: VLM (每~5秒) → 场景级别判断

仲裁优先级（从高到低）:
  1. Task 3 安全状态 (STOP/SLOW/TURN) → 无条件覆盖，安全最优先
  2. VLM 语义：场景拥挤 或 检测到障碍物 → 降速（即使 Task 3 说 FORWARD）
  3. Nav2 导航指令 → 正常执行
  4. 兜底前进

语义流使用原则:
  - 语义定方向（判断场景类型/风险倾向）
  - 几何定时机（瞬时距离/速度判断交给 Task 3）
  - 语义状态超过 MAX_SEMANTIC_AGE 秒未更新则忽略（数据过期）
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'perception'))
from vlm_semantic import VLMSemanticStream

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np
import logging

from task3_core import MotorCommand, RobotState


logger = logging.getLogger(__name__)

# 语义状态超过此秒数未更新则忽略
MAX_SEMANTIC_AGE = 30.0

# ==================== 数据结构 ====================

class CommandSource(Enum):
    TASK3      = "task3"      # 避障模块
    NAV2       = "nav2"       # 导航模块
    ARBITRATOR = "task6"      # 仲裁器自身（兜底用）

@dataclass
class ArbitrationResult:
    """仲裁输出"""
    command:        MotorCommand    # 最终电机指令
    source:         CommandSource   # 这条指令来自谁
    nav2_suppressed: bool           # Nav2 是否被压制
    reason:         str             # 仲裁原因
    semantic_active: bool           # 本次决策是否受语义流影响

# Task 3 中需要压制 Nav2 的状态
_SAFETY_STATES = {
    RobotState.STOP,
    RobotState.SLOW,
    RobotState.TURN_LEFT,
    RobotState.TURN_RIGHT,
}

# ==================== 仲裁器 ====================

class Task6Arbitrator:
    def __init__(self, enable_vlm: bool = True):
        """
        enable_vlm: 是否启用 VLM 语义流。
                    False 时退化为 v1 行为，方便调试对比。
        """
        self._enable_vlm = enable_vlm
        self._vlm: Optional[VLMSemanticStream] = None
        if enable_vlm:
            self._vlm = VLMSemanticStream()
            self._vlm.start()
            logger.info("[task6] VLM semantic stream enabled")
        self.reset()

    def reset(self):
        self._history: list[ArbitrationResult] = []
        self._suppress_count = 0    # 连续压制 Nav2 的帧数

    # ----------------------------------------------------------
    # 外部接口：喂摄像头帧给 VLM
    # ----------------------------------------------------------

    def update_frame(self, frame_bgr: np.ndarray):
        """
        每帧调用，把摄像头帧传给 VLM 语义流。
        非阻塞，直接返回。
        """
        if self._vlm is not None:
            self._vlm.update_frame(frame_bgr)

    # ----------------------------------------------------------
    # 核心仲裁
    # ----------------------------------------------------------

    def arbitrate(self, task3_cmd: MotorCommand,
                  nav2_cmd: Optional[MotorCommand] = None) -> ArbitrationResult:
        """
        每帧调用一次。

        task3_cmd: Task 3 控制器的输出（必须有）
        nav2_cmd:  Nav2 导航的期望指令（可以为 None）
        """

        # 读取语义状态（非阻塞）
        semantic = self._get_semantic()
        semantic_valid = semantic is not None
        crowded      = semantic["crowded"]      if semantic_valid else False
        person_present = semantic["person_present"] if semantic_valid else False
        obstacle_present = semantic["obstacle_present"] if semantic_valid else False
        semantic_active = False

        # ---- 情况 1: Task 3 有避障动作 → 安全优先，无条件覆盖 ----
        if task3_cmd.state in _SAFETY_STATES:
            self._suppress_count += 1
            result = ArbitrationResult(
                command=task3_cmd,
                source=CommandSource.TASK3,
                nav2_suppressed=nav2_cmd is not None,
                reason=f"安全优先: {task3_cmd.reason}",
                semantic_active=False,
            )
            self._record(result)
            return result

        # ---- 情况 2: Task 3 输出 FORWARD（几何安全）----
        self._suppress_count = 0

        # 2a: 语义流有效 且 (场景拥挤 或 有障碍物) → 降速，压制 Nav2
        if semantic_valid and (crowded or obstacle_present):
            semantic_active = True
            cur_speed = (task3_cmd.left_speed + task3_cmd.right_speed) / 2.0
            slow_speed = min(cur_speed * 0.4, 0.15)
            if crowded:
                reason_text = "语义: 场景拥挤，降速通行"
            else:
                reason_text = "语义: 检测到障碍物，降速通行"
            slow_cmd = MotorCommand(
                left_speed=slow_speed,
                right_speed=slow_speed,
                state=RobotState.SLOW,
                reason=reason_text,
            )
            result = ArbitrationResult(
                command=slow_cmd,
                source=CommandSource.ARBITRATOR,
                nav2_suppressed=nav2_cmd is not None,
                reason=f"语义降速: crowded={crowded}, obstacle={obstacle_present}, person={person_present}",
                semantic_active=True,
            )
            self._record(result)
            logger.debug(f"[task6] semantic override: crowded={crowded} obstacle={obstacle_present} → slow")
            return result

        # 2b: Nav2 有导航指令 → 执行 Nav2
        if nav2_cmd is not None:
            result = ArbitrationResult(
                command=nav2_cmd,
                source=CommandSource.NAV2,
                nav2_suppressed=False,
                reason=f"安全，执行导航: {nav2_cmd.reason}",
                semantic_active=semantic_active,
            )
            self._record(result)
            return result

        # 2c: 兜底 → 直接前进
        result = ArbitrationResult(
            command=task3_cmd,
            source=CommandSource.TASK3,
            nav2_suppressed=False,
            reason="安全，无导航任务，正常前进",
            semantic_active=semantic_active,
        )
        self._record(result)
        return result

    # ----------------------------------------------------------
    # 内部工具
    # ----------------------------------------------------------

    def _get_semantic(self) -> Optional[dict]:
        """
        获取语义状态。超过 MAX_SEMANTIC_AGE 则返回 None（视为过期）。
        VLM 未启用时也返回 None。
        """
        if self._vlm is None:
            return None
        state = self._vlm.get_semantic_state()
        if state["age"] > MAX_SEMANTIC_AGE:
            return None
        return state

    def _record(self, result: ArbitrationResult):
        self._history.append(result)
        if len(self._history) > 50:
            self._history.pop(0)

    # ----------------------------------------------------------
    # 公开查询
    # ----------------------------------------------------------

    def get_history(self, n: int = 10) -> list[ArbitrationResult]:
        return self._history[-n:] if self._history else []

    @property
    def suppress_count(self) -> int:
        """当前连续压制 Nav2 的帧数"""
        return self._suppress_count

    def get_semantic_state(self) -> Optional[dict]:
        """暴露语义状态，供外部调试/日志使用"""
        return self._get_semantic()

    # ----------------------------------------------------------
    # 析构
    # ----------------------------------------------------------

    def __del__(self):
        if self._vlm is not None:
            self._vlm.stop()
