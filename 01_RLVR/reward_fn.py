# -*- coding: utf-8 -*-
# 三维度奖励函数：结构合理性 / 材料兼容性 / 历史符合度
# v2: CompositeReward 可传权重参数，不依赖全局 cfg

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import RLVRConfig


class StructureReward(nn.Module):
    # 结构评分：曲率连续性 + 法向量一致性
    def __init__(self, dim=64):
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

    def forward(self, feat):
        # feat: [B, D] → score: [B, 1]
        c = self.curvature_net(feat)
        n = self.normal_net(feat)
        return 0.6 * c + 0.4 * n  # 曲率 6 法向 4


class MaterialReward(nn.Module):
    # 材料评分：原始材料 vs 修复材料的匹配度
    def __init__(self, material_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(material_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
        )
        self.scorer = nn.Linear(32, 1)

    def forward(self, orig, repair):
        # orig/repair: [B, D] → score: [B, 1]
        of = self.encoder(orig)
        rf = self.encoder(repair)
        cos = F.cosine_similarity(of, rf, dim=1, eps=1e-8).unsqueeze(1)
        res = self.scorer(torch.abs(of - rf))
        return torch.sigmoid(cos + res)


class HistoryReward(nn.Module):
    # 历史评分：修复方案是否符合 KG 记载的历史背景
    def __init__(self, kg_emb_dim=128):
        super().__init__()
        self.sem_enc = nn.Sequential(
            nn.Linear(kg_emb_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.tmp_enc = nn.Sequential(
            nn.Linear(kg_emb_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.out = nn.Sequential(
            nn.Linear(192, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, repair_emb, kg_emb):
        # repair_emb/kg_emb: [B, D] → score: [B, 1]
        sem = self.sem_enc(repair_emb)
        tmp = self.tmp_enc(kg_emb)
        return self.out(torch.cat([sem, tmp], dim=1))


class CompositeReward(nn.Module):
    # 三个维度的加权和
    def __init__(self, w_struct=None, w_mat=None, w_hist=None):
        super().__init__()
        self.structure_rwd = StructureReward()
        self.material_rwd = MaterialReward()
        self.history_rwd = HistoryReward()
        d = RLVRConfig()
        self.ws = w_struct or d.weight_structure
        self.wm = w_mat or d.weight_material
        self.wh = w_hist or d.weight_history

    def forward(self, repair_feat, orig_material, repair_material, repair_embed, kg_embed):
        s = self.structure_rwd(repair_feat)
        m = self.material_rwd(orig_material, repair_material)
        h = self.history_rwd(repair_embed, kg_embed)
        total = self.ws * s + self.wm * m + self.wh * h
        return {"total": total, "structure": s, "material": m, "history": h}
