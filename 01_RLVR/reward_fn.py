"""
三维度奖励函数：
1. 结构合理性 — 曲率连续性 + 法向量一致性
2. 材料兼容性 — 物理化学参数匹配度
3. 历史逻辑符合度 — 语义相似度 vs 知识图谱
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import RLVRConfig

cfg = RLVRConfig()


class StructureReward(nn.Module):
    """结构合理性评分：评估修复区域的几何连续性"""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.curvature_net = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
        self.normal_net = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, repair_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            repair_features: [B, D] 修复区域的特征表示
        Returns:
            score: [B, 1] 0~1 结构合理性评分
        """
        curvature_score = self.curvature_net(repair_features)
        normal_score = self.normal_net(repair_features)
        # 加权融合：曲率 0.6 + 法向量 0.4
        score = 0.6 * curvature_score + 0.4 * normal_score
        return score


class MaterialReward(nn.Module):
    """材料兼容性评分：评估修复材料的物理化学匹配度"""

    def __init__(self, material_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(material_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
        )
        self.scorer = nn.Linear(32, 1)

    def forward(self,
                original_material: torch.Tensor,
                repair_material: torch.Tensor) -> torch.Tensor:
        """
        Args:
            original_material: [B, D] 原始材料属性
            repair_material: [B, D]   修复材料属性
        Returns:
            score: [B, 1] 0~1 材料兼容性评分
        """
        orig_feat = self.encoder(original_material)
        repair_feat = self.encoder(repair_material)
        # 余弦相似度 + 可学习的残差
        cos_sim = F.cosine_similarity(orig_feat, repair_feat, dim=1, eps=1e-8).unsqueeze(1)
        residual = self.scorer(torch.abs(orig_feat - repair_feat))
        score = torch.sigmoid(cos_sim + residual)
        return score


class HistoryReward(nn.Module):
    """历史逻辑符合度评分：评估修复方案是否符合历史记载"""

    def __init__(self, kg_emb_dim: int = 128):
        super().__init__()
        self.semantic_encoder = nn.Sequential(
            nn.Linear(kg_emb_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.temporal_encoder = nn.Sequential(
            nn.Linear(kg_emb_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.combine = nn.Sequential(
            nn.Linear(192, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self,
                repair_embed: torch.Tensor,
                kg_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            repair_embed: [B, D] 修复方案的语义编码
            kg_embed:     [B, D] 知识图谱中对应时期/文物的标准编码
        Returns:
            score: [B, 1] 0~1 历史逻辑符合度评分
        """
        sem = self.semantic_encoder(repair_embed)
        temporal = self.temporal_encoder(kg_embed)
        combined = torch.cat([sem, temporal], dim=1)
        score = self.combine(combined)
        return score


class CompositeReward(nn.Module):
    """综合奖励：三个维度的加权求和"""

    def __init__(self):
        super().__init__()
        self.structure_rwd = StructureReward()
        self.material_rwd = MaterialReward()
        self.history_rwd = HistoryReward()

    def forward(self,
                repair_feat: torch.Tensor,
                orig_material: torch.Tensor,
                repair_material: torch.Tensor,
                repair_embed: torch.Tensor,
                kg_embed: torch.Tensor) -> dict:
        s = self.structure_rwd(repair_feat)
        m = self.material_rwd(orig_material, repair_material)
        h = self.history_rwd(repair_embed, kg_embed)

        total = (cfg.weight_structure * s +
                 cfg.weight_material * m +
                 cfg.weight_history * h)
        return {
            "total": total,           # [B, 1] 综合评分
            "structure": s,           # [B, 1]
            "material": m,            # [B, 1]
            "history": h,             # [B, 1]
        }
