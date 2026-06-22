# -*- coding: utf-8 -*-
# 3DCNN 变化检测：输入两期扫描的体素，输出变化类型 + 概率图
# 五分类：无变化 / 裂缝 / 剥落 / 风化 / 生物病害
#
# 数据集的问题：之前 label 和物理变化是独立随机的（各生成各的），
# 导致 label=裂缝 但 after 数据里没裂缝 → 模型学了个寂寞
# 后来改成 apply_change(change_type) 先施加物理变化再贴标签（见 pointcloud_pipeline.py）

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from config import CNN3DConfig
from model import CNN3D
from pointcloud_pipeline import PointCloudPipeline

cfg = CNN3DConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 分类标签，后面展示用
CHANGE_NAMES = ["无变化", "裂缝", "剥落", "风化", "生物病害"]


class ChangeDetector(nn.Module):
    # 双时相变化检测器
    # 思路：两个时间点的扫描过同一个 encoder → 差分 → 融合 → 分类+分割
    def __init__(self):
        super().__init__()
        self.backbone = CNN3D()

        # 融合层：把 before 特征和差分拼起来（conv3_channels*2 = 256）
        self.fusion = nn.Sequential(
            nn.Conv3d(cfg.conv3_channels * 2, cfg.conv1_channels, 1),
            nn.BatchNorm3d(cfg.conv1_channels),
            nn.ReLU(),
        )
        self.seg_head = nn.Conv3d(cfg.conv1_channels, 1, 1)  # 变化区域 heatmap
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
            nn.Linear(cfg.conv1_channels, cfg.fc_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(cfg.fc_dim, cfg.num_change_types),
        )

    def forward(self, x_before, x_after):
        # 双期特征提取
        f1, _ = self.backbone.encoder(x_before)
        f2, _ = self.backbone.encoder(x_after)
        diff = torch.abs(f1 - f2)
        fused = self.fusion(torch.cat([f1, diff], dim=1))

        change_map = torch.sigmoid(self.seg_head(fused))    # [B, 1, D, D, D]
        change_logits = self.cls_head(fused)                # [B, 5]
        return change_logits, change_map


def train_step(model, opt, xb, xa, y):
    model.train()
    logits, _ = model(xb, xa)
    loss = F.cross_entropy(logits, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    loss_sum = 0.0
    for xb, xa, y in loader:
        xb, xa, y = xb.to(device), xa.to(device), y.to(device)
        logits, _ = model(xb, xa)
        loss_sum += F.cross_entropy(logits, y).item()
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    n = max(len(loader), 1)
    return loss_sum / n, correct / max(total, 1)


def make_dataset(n=200):
    # 生成模拟数据集
    # 每种变化类型对应一种物理模拟，label 和数据是对齐的
    pipe = PointCloudPipeline()
    xs, ys = [], []
    for _ in range(n):
        ct = np.random.randint(0, cfg.num_change_types)
        bf, af = pipe.generate_sample_pair(change_type=ct)
        xs.append((bf, af))
        ys.append(ct)
    return xs, torch.tensor(ys)


def demo():
    print("="*50)
    print("3DCNN change detection demo")
    print(f"device: {device}")
    print("="*50)

    model = ChangeDetector().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # 造数据
    data, labels = make_dataset(200)
    n_tr = 160
    tr_x, te_x = data[:n_tr], data[n_tr:]
    tr_y, te_y = labels[:n_tr], labels[n_tr:]

    # 训练循环
    for ep in range(cfg.num_epochs):
        ep_loss = 0.0
        for i in range(0, len(tr_x), cfg.batch_size):
            batch = tr_x[i:i+cfg.batch_size]
            by = tr_y[i:i+cfg.batch_size].to(device)
            xb = torch.stack([b[0] for b in batch]).to(device)
            xa = torch.stack([b[1] for b in batch]).to(device)
            ep_loss += train_step(model, opt, xb, xa, by)

        if (ep + 1) % 10 == 0:
            # 拼测试集
            txb = torch.stack([te_x[i][0] for i in range(len(te_x))])
            txa = torch.stack([te_x[i][1] for i in range(len(te_x))])
            loader = [(txb[i:i+cfg.batch_size].to(device),
                      txa[i:i+cfg.batch_size].to(device),
                      te_y[i:i+cfg.batch_size].to(device))
                     for i in range(0, len(te_x), cfg.batch_size)]
            tl, ta = evaluate(model, loader)
            print(f"  ep {ep+1}/{cfg.num_epochs}  "
                  f"tr_loss={ep_loss:.2f}  te_loss={tl:.4f}  te_acc={ta:.1%}")

    Path(cfg.checkpoint_dir).mkdir(exist_ok=True)
    save_path = f"{cfg.checkpoint_dir}/change_detector.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n[done] model -> {save_path}")


if __name__ == "__main__":
    demo()
