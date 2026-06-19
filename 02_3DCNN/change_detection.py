"""
3D-CNN 变化分类系统：
对比两期扫描数据，自动识别裂缝、剥落、风化、生物病害。
"""

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

CHANGE_LABELS = [
    "无变化",
    "裂缝 (Crack)",
    "剥落 (Spalling)",
    "风化 (Weathering)",
    "生物病害 (Biodeterioration)",
]


class ChangeDetector(nn.Module):
    """
    变化检测器：
    双时相输入 → 特征提取 → 差分 → 分类
    """

    def __init__(self):
        super().__init__()
        self.backbone = CNN3D()

        # 双时相融合层（encoder 输出通道数为 conv3_channels）
        self.fusion = nn.Sequential(
            nn.Conv3d(cfg.conv3_channels * 2, cfg.conv1_channels, 1),
            nn.BatchNorm3d(cfg.conv1_channels),
            nn.ReLU(),
        )

        # 变化区域分割头（输出变化概率图）
        self.seg_head = nn.Conv3d(cfg.conv1_channels, 1, 1)

        # 整体变化分类头
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(cfg.conv1_channels, cfg.fc_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(cfg.fc_dim, cfg.num_change_types),
        )

    def forward(self, x_before, x_after):
        """
        Args:
            x_before: [B, 1, D, D, D] 前期扫描
            x_after:  [B, 1, D, D, D] 后期扫描

        Returns:
            change_logits: [B, 5] 变化类型分类
            change_map:    [B, 1, D, D, D] 变化概率图
        """
        # 提取双时相特征
        feat_before, _ = self.backbone.encoder(x_before)
        feat_after, _ = self.backbone.encoder(x_after)

        # 差分融合
        diff = torch.abs(feat_before - feat_after)
        fused = self.fusion(torch.cat([feat_before, diff], dim=1))

        # 变化概率图
        change_map = torch.sigmoid(self.seg_head(fused))

        # 整体分类
        change_logits = self.cls_head(fused)
        return change_logits, change_map


def train_step(model, optimizer, x_before, x_after, labels):
    """单步训练"""
    model.train()
    logits, _ = model(x_before, x_after)
    loss = F.cross_entropy(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, test_loader):
    """评估"""
    model.eval()
    correct, total = 0, 0
    total_loss = 0.0
    for x_b, x_a, labels in test_loader:
        x_b, x_a, labels = x_b.to(device), x_a.to(device), labels.to(device)
        logits, _ = model(x_b, x_a)
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / max(len(test_loader), 1), correct / max(total, 1)


def generate_synthetic_dataset(num_samples: int = 200):
    """
    生成模拟训练数据

    真实场景中应替换为实际文物扫描数据。
    """
    pipeline = PointCloudPipeline()
    data = []
    labels = []

    for _ in range(num_samples):
        before, after = pipeline.generate_sample_pair()
        change_type = np.random.randint(0, cfg.num_change_types)
        # 根据变化类型微调 after
        data.append((before, after))
        labels.append(change_type)

    return data, torch.tensor(labels)


def demo():
    """运行变化检测 demo"""
    print("="*50)
    print("3D-CNN 变化检测 Demo")
    print(f"设备: {device}")
    print("="*50)

    model = ChangeDetector().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # 生成模拟数据
    print("\n[数据] 生成模拟训练数据...")
    data, labels = generate_synthetic_dataset(200)
    n_train = 160
    train_data, test_data = data[:n_train], data[n_train:]
    train_labels, test_labels = labels[:n_train], labels[n_train:]

    # 训练
    print("[训练] 开始...")
    for epoch in range(cfg.num_epochs):
        epoch_loss = 0.0
        for i in range(0, len(train_data), cfg.batch_size):
            batch = train_data[i:i+cfg.batch_size]
            batch_labels = train_labels[i:i+cfg.batch_size].to(device)
            x_b = torch.stack([b[0] for b in batch]).to(device)
            x_a = torch.stack([b[1] for b in batch]).to(device)
            loss = train_step(model, optimizer, x_b, x_a, batch_labels)
            epoch_loss += loss

        if (epoch + 1) % 10 == 0:
            # 构建测试 DataLoader
            test_x_b = torch.stack([test_data[i][0] for i in range(len(test_data))])  # [N, 1, D, D, D]
            test_x_a = torch.stack([test_data[i][1] for i in range(len(test_data))])
            test_loader = [(test_x_b[i:i+cfg.batch_size].to(device),
                            test_x_a[i:i+cfg.batch_size].to(device),
                            test_labels[i:i+cfg.batch_size].to(device))
                           for i in range(0, len(test_data), cfg.batch_size)]
            test_loss, test_acc = evaluate(model, test_loader)
            print(f"  Epoch {epoch+1}/{cfg.num_epochs}  "
                  f"Train Loss: {epoch_loss:.4f}  "
                  f"Test Loss: {test_loss:.4f}  "
                  f"Test Acc: {test_acc:.2%}")

    # 保存模型
    Path(cfg.checkpoint_dir).mkdir(exist_ok=True)
    torch.save(model.state_dict(), f"{cfg.checkpoint_dir}/change_detector.pth")
    print(f"\n[完成] 模型已保存到 {cfg.checkpoint_dir}/change_detector.pth")
    print(f"[完成] 变化类型: {CHANGE_LABELS}")


if __name__ == "__main__":
    demo()
