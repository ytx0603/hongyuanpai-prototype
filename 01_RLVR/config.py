"""RLVR 训练配置"""
from dataclasses import dataclass, field
from typing import List

@dataclass
class RLVRConfig:
    # 奖励函数权重
    weight_structure: float = 0.35    # 结构合理性权重
    weight_material: float = 0.25      # 材料兼容性权重
    weight_history: float = 0.40       # 历史逻辑符合度权重

    # RL 训练
    hidden_dim: int = 256
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 3e-4
    gamma: float = 0.99               # 折扣因子
    clip_epsilon: float = 0.2         # GRPO clip 阈值
    entropy_coef: float = 0.01        # 熵正则系数

    # Self-play
    self_play_rounds: int = 5         # 每轮对弈次数
    replay_buffer_size: int = 1000

    # SFT
    sft_epochs: int = 10
    sft_lr: float = 1e-4

    # 数据
    num_samples: int = 500            # 模拟样本数
    state_dim: int = 128              # 状态空间维度
    action_dim: int = 64              # 动作空间维度（修复参数）

    # 路径
    model_save_path: str = "./checkpoints"
    log_path: str = "./logs"
