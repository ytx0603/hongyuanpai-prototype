"""
3D-CNN 模型架构：
带有跳连接（skip connections）的轻量化 3D 卷积网络。
从论文六（Frontiers 2026）的遥感架构迁移而来。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CNN3DConfig

cfg = CNN3DConfig()


class Conv3DBlock(nn.Module):
    """3D 卷积块：Conv3D + BN + ReLU + Dropout"""

    def __init__(self, in_ch, out_ch, kernel_size=3, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm3d(out_ch)
        self.dropout = nn.Dropout3d(dropout)

    def forward(self, x):
        return self.dropout(F.relu(self.bn(self.conv(x))))


class SkipBlock3D(nn.Module):
    """带跳连接的 3D 残差块"""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = Conv3DBlock(channels, channels)
        self.conv2 = Conv3DBlock(channels, channels)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class Encoder3D(nn.Module):
    """3D-CNN 编码器：下采样路径"""

    def __init__(self, in_ch=1):
        super().__init__()
        self.enc1 = Conv3DBlock(in_ch, cfg.conv1_channels)   # 1→32
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = Conv3DBlock(cfg.conv1_channels, cfg.conv2_channels)  # 32→64
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = Conv3DBlock(cfg.conv2_channels, cfg.conv3_channels)  # 64→128
        self.pool3 = nn.MaxPool3d(2)

    def forward(self, x):
        x1 = self.enc1(x)           # [B, 32, 64, 64, 64]
        x = self.pool1(x1)          # [B, 32, 32, 32, 32]
        x2 = self.enc2(x)           # [B, 64, 32, 32, 32]
        x = self.pool2(x2)          # [B, 64, 16, 16, 16]
        x3 = self.enc3(x)           # [B, 128, 16, 16, 16]
        x = self.pool3(x3)          # [B, 128, 8, 8, 8]
        return x, (x1, x2, x3)


class Decoder3D(nn.Module):
    """3D-CNN 解码器：上采样路径（带跳连接）"""

    def __init__(self):
        super().__init__()
        self.up3 = nn.ConvTranspose3d(cfg.conv3_channels, cfg.conv2_channels, 2, 2)
        self.dec3 = Conv3DBlock(cfg.conv2_channels * 2, cfg.conv2_channels)
        self.up2 = nn.ConvTranspose3d(cfg.conv2_channels, cfg.conv1_channels, 2, 2)
        self.dec2 = Conv3DBlock(cfg.conv1_channels * 2, cfg.conv1_channels)
        self.up1 = nn.ConvTranspose3d(cfg.conv1_channels, cfg.conv1_channels, 2, 2)
        self.dec1 = Conv3DBlock(cfg.conv1_channels + 1, cfg.conv1_channels)

    def forward(self, x, skip_connections):
        x1, x2, x3 = skip_connections

        x = self.up3(x)                           # [B, 64, 16, 16, 16]
        x = torch.cat([x, x3], dim=1)             # skip connection
        x = self.dec3(x)

        x = self.up2(x)                           # [B, 32, 32, 32, 32]
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)

        x = self.up1(x)                           # [B, 32, 64, 64, 64]
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)                          # [B, 32, 64, 64, 64]
        return x


class ChangeDetectionHead(nn.Module):
    """变化分类头"""

    def __init__(self):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(cfg.conv1_channels, cfg.fc_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(cfg.fc_dim, cfg.num_change_types),
        )

    def forward(self, x):
        x = self.global_pool(x).flatten(1)  # [B, 32]
        return self.fc(x)                    # [B, 5]


class TrendPredictionHead(nn.Module):
    """趋势预测头：基于多期特征预测劣化趋势"""

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=cfg.conv1_channels,
            hidden_size=cfg.trend_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )
        self.fc = nn.Sequential(
            nn.Linear(cfg.trend_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # 预测劣化程度
        )

    def forward(self, seq_features):
        """
        Args:
            seq_features: [B, T, C] 多期特征序列
        Returns:
            trend: [B, 1] 预测的劣化趋势值
        """
        lstm_out, _ = self.lstm(seq_features)
        last = lstm_out[:, -1, :]  # 取最后时间步
        return self.fc(last)


class CNN3D(nn.Module):
    """
    完整 3D-CNN 模型：
    编码器 → 解码器 → 变化分类头 + 趋势预测头
    """

    def __init__(self):
        super().__init__()
        self.encoder = Encoder3D(cfg.input_channels)
        self.decoder = Decoder3D()
        self.change_head = ChangeDetectionHead()
        self.trend_head = TrendPredictionHead()

    def forward(self, x, return_all=False):
        """
        Args:
            x: [B, 1, D, D, D] 体素输入
            return_all: 是否返回所有输出

        Returns:
            change_logits: [B, 5] 变化分类 logits
            trend_pred: [B, 1] 趋势预测（可选）
        """
        encoded, skips = self.encoder(x)
        decoded = self.decoder(encoded, skips)
        change_logits = self.change_head(decoded)

        if return_all:
            return change_logits, decoded
        return change_logits
