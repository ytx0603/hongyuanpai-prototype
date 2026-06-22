# -*- coding: utf-8 -*-
# SFT: 在专家标注上做监督学习，让模型先学会"专家怎么修"
# 然后 RLVR 的 self-play 再在这个基础上自己探索
# 真数据场景下这批标注来自文保专家的实际修复方案记录

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from config import RLVRConfig

cfg = RLVRConfig()


class SFTFineTuner:
    # SFT: 拿 expert demo 做 behavior cloning
    def __init__(self, policy_net):
        self.policy = policy_net
        self.opt = torch.optim.Adam(policy_net.parameters(), lr=cfg.sft_lr)
        self.loss_fn = nn.MSELoss()

    def prepare_data(self, states, expert_actions, batch_size=16):
        ds = TensorDataset(states, expert_actions)
        return DataLoader(ds, batch_size=batch_size, shuffle=True)

    def train_epoch(self, loader):
        total, n = 0.0, 0
        dev = next(self.policy.parameters()).device
        for s, a in loader:
            s, a = s.to(dev), a.to(dev)
            pred = self.policy(s)
            loss = self.loss_fn(pred, a)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            total += loss.item()
            n += 1
        return total / max(n, 1)

    def generate_synthetic_expert_data(self, n=100, sd=128, ad=64):
        # 模拟专家行为：状态→动作有一定规律 + 噪声
        torch.manual_seed(42)
        s = torch.randn(n, sd)
        a = torch.tanh(s[:, :ad] * 0.8 + torch.randn(n, ad) * 0.1)
        return s, a

    def run_sft(self, n_epochs=None):
        n_epochs = n_epochs or cfg.sft_epochs
        print(f"[SFT] generating {cfg.num_samples} synthetic expert samples...")
        s, a = self.generate_synthetic_expert_data(cfg.num_samples, cfg.state_dim, cfg.action_dim)
        loader = self.prepare_data(s, a)
        losses = []
        for ep in range(n_epochs):
            l = self.train_epoch(loader)
            losses.append(l)
            if (ep + 1) % 5 == 0:
                print(f"[SFT] ep {ep+1}/{n_epochs}  loss={l:.6f}")
        print(f"[SFT] done. final loss={losses[-1]:.6f}")
        return losses
