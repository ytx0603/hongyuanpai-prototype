"""
点云预处理管线：
原始扫描 → 下采样 → 去噪 → 体素化 → 输入张量
"""

import numpy as np
import torch
from scipy.spatial import KDTree
from config import CNN3DConfig

cfg = CNN3DConfig()


def voxelize(point_cloud: np.ndarray,
             grid_dim: int = 64,
             voxel_size: float = None) -> np.ndarray:
    """
    将点云体素化为 3D 网格

    Args:
        point_cloud: [N, 3] 点云坐标
        grid_dim: 体素网格维度
        voxel_size: 体素大小（m）

    Returns:
        voxel_grid: [D, D, D] 体素占位矩阵 (0/1)
    """
    if voxel_size is None:
        # 自动计算体素大小
        ranges = point_cloud.max(axis=0) - point_cloud.min(axis=0)
        voxel_size = max(ranges) / grid_dim

    # 归一化到 [0, grid_dim)
    min_bound = point_cloud.min(axis=0)
    indices = np.floor((point_cloud - min_bound) / voxel_size).astype(int)
    indices = np.clip(indices, 0, grid_dim - 1)

    voxel_grid = np.zeros((grid_dim, grid_dim, grid_dim), dtype=np.float32)
    unique_indices = np.unique(indices, axis=0)
    voxel_grid[unique_indices[:, 0],
               unique_indices[:, 1],
               unique_indices[:, 2]] = 1.0
    return voxel_grid


def random_sample(points: np.ndarray, n: int) -> np.ndarray:
    """随机采样固定数量的点"""
    if len(points) >= n:
        idx = np.random.choice(len(points), n, replace=False)
    else:
        idx = np.random.choice(len(points), n, replace=True)
    return points[idx]


def add_simulated_crack(points: np.ndarray,
                        crack_ratio: float = 0.05) -> np.ndarray:
    """
    模拟裂缝：在点云中插入裂缝

    Args:
        points: [N, 3] 原始点云
        crack_ratio: 裂缝占比

    Returns:
        points_with_crack: [N+M, 3] 带裂缝的点云
    """
    n_crack = int(len(points) * crack_ratio)
    # 沿 x 轴方向生成裂缝状点
    crack_x = np.random.uniform(-0.5, 0.5, n_crack)
    crack_y = np.random.uniform(-0.02, 0.02, n_crack)  # 窄裂缝
    crack_z = np.random.uniform(-0.5, 0.5, n_crack)
    crack_points = np.stack([crack_x, crack_y, crack_z], axis=1)
    return np.vstack([points, crack_points])


class PointCloudPipeline:
    """点云预处理管线"""

    def __init__(self, grid_dim: int = 64):
        self.grid_dim = grid_dim

    def __call__(self, points: np.ndarray) -> torch.Tensor:
        """
        完整管线：采样 → 体素化 → 转 tensor

        Args:
            points: [N, 3] 原始点云

        Returns:
            voxel: [1, D, D, D] 体素张量
        """
        sampled = random_sample(points, cfg.num_points_per_sample)
        voxel = voxelize(sampled, self.grid_dim)
        tensor = torch.from_numpy(voxel).unsqueeze(0).float()  # [1, D, D, D]
        return tensor

    def generate_sample_pair(self) -> tuple:
        """
        生成一组模拟的"前一期/后一期"点云对比

        Returns:
            scan_before: [1, D, D, D] 前期扫描
            scan_after:  [1, D, D, D] 后期扫描（可能有变化）
            change_mask: [D, D, D] 变化区域标注（仅模拟用）
        """
        np.random.seed(42)
        # 模拟一个立方体文物
        n_points = cfg.num_points_per_sample
        base = np.random.uniform(-0.5, 0.5, (n_points, 3))

        before = self(base)

        # 后期：加入模拟裂缝
        after_points = add_simulated_crack(base, crack_ratio=0.05)
        after = self(after_points)

        return before, after
