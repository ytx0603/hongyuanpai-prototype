"""
SFT (Supervised Fine-Tuning) 微调：
用专家标注的高质量修复方案对模型做监督学习，
提升采纳率（从 68% → 85%+）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from config import RLVRConfig

cfg = RLVRConfig()


class SFTFineTuner:
    """
    SFT 微调器

    在少量专家标注数据上做监督学习，
    让模型学会"专家会怎么做"，
    再结合 RLVR 的"自己能探索"能力。
    """

    def __init__(self, policy_net: nn.Module):
        self.policy = policy_net
        self.optimizer = torch.optim.Adam(
            policy_net.parameters(), lr=cfg.sft_lr
        )
        self.criterion = nn.MSELoss()

    def prepare_data(self,
                     states: torch.Tensor,
                     expert_actions: torch.Tensor,
                     batch_size: int = 16) -> DataLoader:
        """准备专家标注数据集的 DataLoader"""
        dataset = TensorDataset(states, expert_actions)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train_epoch(self, dataloader: DataLoader) -> float:
        """训练一个 epoch，返回平均 loss"""
        total_loss = 0.0
        n_batches = 0
        device = next(self.policy.parameters()).device

        for states, expert_actions in dataloader:
            states = states.to(device)
            expert_actions = expert_actions.to(device)
            predicted = self.policy(states)
            loss = self.criterion(predicted, expert_actions)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def generate_synthetic_expert_data(self,
                                        num_samples: int = 100,
                                        state_dim: int = 128,
                                        action_dim: int = 64) -> tuple:
        """
        生成模拟的专家标注数据（用于 demo）

        真实场景中，这部分数据来自文保专家的实际修复方案。
        """
        torch.manual_seed(42)
        states = torch.randn(num_samples, state_dim)
        # 模拟专家行为：有特定的分布偏好
        expert_actions = torch.tanh(
            states[:, :action_dim] * 0.8 +
            torch.randn(num_samples, action_dim) * 0.1
        )
        return states, expert_actions

    def run_sft(self, num_epochs: int = None) -> list:
        """完整运行 SFT 训练流程"""
        num_epochs = num_epochs or cfg.sft_epochs
        print(f"[SFT] 生成 {cfg.num_samples} 条模拟专家数据...")
        states, expert_actions = self.generate_synthetic_expert_data(
            num_samples=cfg.num_samples,
            state_dim=cfg.state_dim,
            action_dim=cfg.action_dim,
        )

        dataloader = self.prepare_data(states, expert_actions)
        losses = []

        for epoch in range(num_epochs):
            loss = self.train_epoch(dataloader)
            losses.append(loss)
            if (epoch + 1) % 5 == 0:
                print(f"[SFT] Epoch {epoch+1}/{num_epochs}  Loss: {loss:.6f}")

        print(f"[SFT] 完成！最终 Loss: {losses[-1]:.6f}")
        return losses
