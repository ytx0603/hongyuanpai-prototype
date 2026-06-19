"""3D-CNN 监测配置"""
from dataclasses import dataclass

@dataclass
class CNN3DConfig:
    # 点云处理
    voxel_size: float = 0.01          # 体素大小（米）
    grid_dim: int = 64                # 体素网格维度 (64x64x64)
    num_points_per_sample: int = 4096  # 每样本点数

    # 3D-CNN 架构
    input_channels: int = 1
    conv1_channels: int = 32
    conv2_channels: int = 64
    conv3_channels: int = 128
    fc_dim: int = 256

    # 变化分类
    num_change_types: int = 5         # 裂缝/剥落/风化/生物病害/无变化

    # 趋势预测
    trend_hidden: int = 128
    trend_seq_len: int = 6            # 6期时序数据

    # 训练
    batch_size: int = 16
    learning_rate: float = 1e-3
    num_epochs: int = 50

    # 路径
    checkpoint_dir: str = "./checkpoints"
