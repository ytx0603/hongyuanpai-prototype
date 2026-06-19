"""
GRPO (Group Relative Policy Optimization)：
通过组内相对比较来优化策略，
不依赖绝对评分，只依赖"这个方案比其他方案好多少"。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import RLVRConfig

cfg = RLVRConfig()


class GRPOOptimizer:
    """
    GRPO 策略优化器

    核心思想：
    1. 对同一文物残片，Actor 生成 K 个修复方案
    2. 用 Reward 函数对 K 个方案评分
    3. 用组内均值做 baseline，计算 advantage
    4. 用 clipped surrogate objective 更新策略
    """

    def __init__(self, policy_net: nn.Module, lr: float = None):
        self.policy = policy_net
        self.optimizer = torch.optim.Adam(
            policy_net.parameters(),
            lr=lr or cfg.learning_rate
        )

    def compute_advantage(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        计算组内 advantage

        Args:
            rewards: [B, K] 每组的 K 个评分

        Returns:
            advantages: [B, K] 归一化后的优势值
        """
        # 组内均值作为 baseline
        baseline = rewards.mean(dim=1, keepdim=True)  # [B, 1]
        std = rewards.std(dim=1, keepdim=True) + 1e-8
        advantages = (rewards - baseline) / std
        return advantages

    def update(self,
               states: torch.Tensor,
               actions: torch.Tensor,
               old_log_probs: torch.Tensor,
               rewards: torch.Tensor) -> dict:
        """
        一次 GRPO 更新

        Args:
            states: [B, D] 状态
            actions: [B, K, D] 每组 K 个动作
            old_log_probs: [B, K] 旧策略的 log prob
            rewards: [B, K] reward 评分

        Returns:
            loss: 策略损失
            approx_kl: 近似 KL 散度
        """
        B, K, D = actions.shape
        state_dim = states.shape[-1]

        # 计算 advantage
        advantages = self.compute_advantage(rewards)  # [B, K]

        # 当前策略的 log prob（简化版：用 -MSE 近似）
        flat_states = states.unsqueeze(1).expand(-1, K, -1)  # [B, K, state_dim]
        flat_actions = actions.reshape(-1, D)  # [B*K, action_dim]
        flat_states_2d = flat_states.reshape(-1, state_dim)  # [B*K, state_dim]

        new_actions = self.policy(flat_states_2d)
        # 用负 MSE 作为 log prob 近似
        new_log_probs = -F.mse_loss(new_actions, flat_actions, reduction='none').mean(dim=1)
        new_log_probs = new_log_probs.reshape(B, K)

        # 重要性采样比率
        ratio = torch.exp(new_log_probs - old_log_probs)

        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 熵正则
        entropy = -torch.mean(new_log_probs)

        # 总损失
        total_loss = policy_loss - cfg.entropy_coef * entropy

        # 更新
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        # 计算近似 KL
        approx_kl = ((ratio - 1) - (ratio.log())).mean().item()

        return {
            "policy_loss": policy_loss.item(),
            "entropy": entropy.item(),
            "approx_kl": approx_kl,
            "total_loss": total_loss.item(),
        }
