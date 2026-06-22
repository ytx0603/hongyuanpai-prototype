# -*- coding: utf-8 -*-
# 3D CNN — U-Net with skip connections
# 基于 Frontiers 2026 的遥感架构，迁移到文物体素数据

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CNN3DConfig

cfg = CNN3DConfig()


class Conv3DBlock(nn.Module):
    # Conv3d → BN → ReLU → Dropout3d
    def __init__(self, in_ch, out_ch, k=3, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, k, padding=k//2)
        self.bn = nn.BatchNorm3d(out_ch)
        self.drop = nn.Dropout3d(dropout)
    def forward(self, x):
        return self.drop(F.relu(self.bn(self.conv(x))))


class SkipBlock3D(nn.Module):
    # 残差块：x + conv(conv(x))
    def __init__(self, ch):
        super().__init__()
        self.c1 = Conv3DBlock(ch, ch)
        self.c2 = Conv3DBlock(ch, ch)
    def forward(self, x):
        return x + self.c2(self.c1(x))


class Encoder3D(nn.Module):
    # 下采样：1→32→64→128 通道，空间逐层减半
    def __init__(self, in_ch=1):
        super().__init__()
        self.enc1 = Conv3DBlock(in_ch, cfg.conv1_channels)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = Conv3DBlock(cfg.conv1_channels, cfg.conv2_channels)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = Conv3DBlock(cfg.conv2_channels, cfg.conv3_channels)
        self.pool3 = nn.MaxPool3d(2)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        return self.pool3(x3), (x1, x2, x3)


class Decoder3D(nn.Module):
    # 上采样 + skip connection 拼接
    def __init__(self):
        super().__init__()
        self.up3 = nn.ConvTranspose3d(cfg.conv3_channels, cfg.conv2_channels, 2, 2)
        self.dec3 = Conv3DBlock(cfg.conv2_channels * 2, cfg.conv2_channels)
        self.up2 = nn.ConvTranspose3d(cfg.conv2_channels, cfg.conv1_channels, 2, 2)
        self.dec2 = Conv3DBlock(cfg.conv1_channels * 2, cfg.conv1_channels)
        self.up1 = nn.ConvTranspose3d(cfg.conv1_channels, cfg.conv1_channels, 2, 2)
        self.dec1 = Conv3DBlock(cfg.conv1_channels + 1, cfg.conv1_channels)  # +1 是把原始输入也拼上

    def forward(self, x, skips):
        x1, x2, x3 = skips
        x = self.dec3(torch.cat([self.up3(x), x3], dim=1))
        x = self.dec2(torch.cat([self.up2(x), x2], dim=1))
        x = self.dec1(torch.cat([self.up1(x), x1], dim=1))
        return x


class ChangeDetectionHead(nn.Module):
    # 全局池化 → FC → 5 分类
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(cfg.conv1_channels, cfg.fc_dim),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(cfg.fc_dim, cfg.num_change_types),
        )
    def forward(self, x):
        return self.fc(self.pool(x).flatten(1))


class TrendPredictionHead(nn.Module):
    # LSTM over 多期特征 → 预测劣化值
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(cfg.conv1_channels, cfg.trend_hidden, 2,
                           batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(nn.Linear(cfg.trend_hidden, 64), nn.ReLU(),
                               nn.Linear(64, 1))
    def forward(self, seq):
        # seq: [B, T, C]
        out, _ = self.lstm(seq)
        return self.fc(out[:, -1, :])


class CNN3D(nn.Module):
    # 完整模型：encoder + decoder + change head
    # TrendPredictionHead 独立使用，不挂在这里
    def __init__(self):
        super().__init__()
        self.encoder = Encoder3D(cfg.input_channels)
        self.decoder = Decoder3D()
        self.change_head = ChangeDetectionHead()

    def forward(self, x, return_feat=False):
        # x: [B, 1, D, D, D] → logits: [B, 5]
        enc, skips = self.encoder(x)
        dec = self.decoder(enc, skips)
        logits = self.change_head(dec)
        return (logits, dec) if return_feat else logits
