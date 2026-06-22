# -*- coding: utf-8 -*-
# 点云预处理 + 变化模拟
# 把 3D 扫描的点云变成体素网格，喂给 CNN
# 顺便模拟了几种常见的文物损毁类型，用来生成训练数据

import numpy as np
import torch
from config import CNN3DConfig

cfg = CNN3DConfig()


def voxelize(pts, grid_dim=64, vsize=None):
    # pts: [N, 3] → grid: [D, D, D] 0/1 占位
    if vsize is None:
        span = pts.max(axis=0) - pts.min(axis=0)
        vsize = max(span) / grid_dim

    lo = pts.min(axis=0)
    idx = np.floor((pts - lo) / vsize).astype(int)
    idx = np.clip(idx, 0, grid_dim - 1)

    grid = np.zeros((grid_dim, grid_dim, grid_dim), dtype=np.float32)
    uq = np.unique(idx, axis=0)
    grid[uq[:, 0], uq[:, 1], uq[:, 2]] = 1.0
    return grid


def random_sample(pts, n):
    # 固定点数下采样（多了无放回，少了有放回）
    if len(pts) >= n:
        return pts[np.random.choice(len(pts), n, replace=False)]
    return pts[np.random.choice(len(pts), n, replace=True)]


# ===== 物理变化模拟函数 =====
# 每个函数模拟一种文物损毁类型，产生肉眼可见的物理差异

def add_simulated_crack(pts, ratio=0.05):
    # 裂缝：沿 x 轴插入窄带状点，模拟结构开裂
    n = int(len(pts) * ratio)
    cx = np.random.uniform(-0.5, 0.5, n)
    cy = np.random.uniform(-0.02, 0.02, n)  # y 极窄 → 线状
    cz = np.random.uniform(-0.5, 0.5, n)
    return np.vstack([pts, np.column_stack([cx, cy, cz])])


def simulate_spalling(pts, ratio=0.08):
    # 剥落：去掉表面最外侧的一层点 → 模拟材料脱落
    n_rm = int(len(pts) * ratio)
    th = pts[:, 1].max() - 0.05
    surface = np.where(pts[:, 1] > th)[0]
    if len(surface) > n_rm:
        rm = np.random.choice(surface, n_rm, replace=False)
    else:
        rm = surface
    return np.delete(pts, rm, axis=0)


def simulate_weathering(pts, noise=0.03):
    # 风化：全局加随机噪声，表面比内部影响大 → 模拟侵蚀
    n = np.random.randn(*pts.shape) * noise
    dist = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts + n * (1.0 + dist * 2.0)


def simulate_biodeterioration(pts, ratio=0.06):
    # 生物病害：在表面随机位置生成团块 → 模拟霉菌/苔藓
    n = int(len(pts) * ratio)
    nc = np.random.randint(1, 4)
    blobs = []
    for _ in range(nc):
        ctr = np.random.uniform(-0.3, 0.3, 3)
        blobs.append(np.random.randn(n // nc, 3) * 0.06 + ctr)
    return np.vstack([pts] + blobs)


def apply_change(pts, change_type):
    # 根据类型标签施加物理变化
    # 0=无 1=裂缝 2=剥落 3=风化 4=生物病害
    if change_type == 0:
        return pts.copy()
    elif change_type == 1:
        return add_simulated_crack(pts)
    elif change_type == 2:
        return simulate_spalling(pts)
    elif change_type == 3:
        return simulate_weathering(pts)
    elif change_type == 4:
        return simulate_biodeterioration(pts)
    return pts.copy()


class PointCloudPipeline:
    # 点云 → 体素的完整管线
    def __init__(self, gdim=64):
        self.gdim = gdim

    def __call__(self, pts):
        # 下采样 → 体素化 → tensor
        sp = random_sample(pts, cfg.num_points_per_sample)
        vx = voxelize(sp, self.gdim)
        return torch.from_numpy(vx).unsqueeze(0).float()

    def gen_pair(self, change_type=0):
        # 生成一对 before/after 扫描，after 施加了给定的物理变化
        n = cfg.num_points_per_sample
        base = np.random.uniform(-0.5, 0.5, (n, 3))
        before = self(base)
        after = self(apply_change(base, change_type))
        return before, after

    # 旧接口兼容（别的文件可能会调）
    generate_sample_pair = gen_pair
