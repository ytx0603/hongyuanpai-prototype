"""
趋势预测模块：
基于多期监测数据，预测文物劣化的发展趋势。
例如："以当前速度，这个裂缝再过6个月会扩展到什么程度"
"""

import torch
import torch.nn as nn
import numpy as np
from config import CNN3DConfig
from model import CNN3D, TrendPredictionHead

cfg = CNN3DConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrendPredictor(nn.Module):
    """
    劣化趋势预测器

    输入多期三维扫描的特征序列，
    输出未来劣化程度的定量预测。
    """

    def __init__(self):
        super().__init__()
        self.backbone = CNN3D()
        self.trend_head = TrendPredictionHead()

    def forward(self, scan_sequence):
        """
        Args:
            scan_sequence: [B, T, 1, D, D, D] T 期扫描数据

        Returns:
            trend_score: [B, 1] 预测的劣化程度 (0~1)
        """
        B, T = scan_sequence.shape[:2]
        # 提取每期特征
        features = []
        for t in range(T):
            x = scan_sequence[:, t]  # [B, 1, D, D, D]
            feat, _ = self.backbone.encoder(x)
            feat = feat.mean(dim=[2, 3, 4])  # [B, C] 全局平均池化
            features.append(feat)

        seq_feat = torch.stack(features, dim=1)  # [B, T, C]
        trend = self.trend_head(seq_feat)        # [B, 1]
        return torch.sigmoid(trend)


def generate_synthetic_sequence(num_samples: int = 32,
                                 seq_len: int = 6) -> torch.Tensor:
    """
    生成模拟的时序扫描数据

    模拟劣化逐渐加重的过程。
    """
    B, T, D = num_samples, seq_len, cfg.grid_dim
    sequence = []
    for t in range(T):
        # 每期增加一点噪声模拟劣化
        noise = torch.randn(B, 1, D, D, D) * 0.02 * (t + 1)
        base = torch.rand(B, 1, D, D, D) * 0.3
        scan = torch.clamp(base + noise, 0, 1)
        sequence.append(scan)
    return torch.stack(sequence, dim=1)


def demo():
    """趋势预测 demo"""
    print("="*50)
    print("劣化趋势预测 Demo")
    print(f"设备: {device}")
    print("="*50)

    model = TrendPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 生成模拟数据
    print("\n[数据] 生成模拟时序扫描...")
    seq = generate_synthetic_sequence(64, cfg.trend_seq_len).to(device)
    # 模拟目标：后期劣化程度
    targets = torch.rand(64, 1).to(device) * 0.5 + 0.3

    # 训练
    print("[训练]...")
    n_train = 48
    train_seq, train_tgt = seq[:n_train], targets[:n_train]
    test_seq, test_tgt = seq[n_train:], targets[n_train:]

    for epoch in range(30):
        model.train()
        pred = model(train_seq)
        loss = nn.MSELoss()(pred, train_tgt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_pred = model(test_seq)
                test_loss = nn.MSELoss()(test_pred, test_tgt)
            print(f"  Epoch {epoch+1}/30  train_loss: {loss.item():.4f}  "
                  f"test_loss: {test_loss.item():.4f}")

    print(f"\n[完成] 趋势预测模型训练完成")


if __name__ == "__main__":
    demo()
