# GRPO 优化器
# 参考 https://arxiv.org/abs/2506.14245 (Wen et al.)
# 核心：组内相对比较，不依赖绝对 reward 的 scale
#
# 之前一版用 -MSE 算 log_prob，ratio 完全没区分度 → 已改

import torch
from config import RLVRConfig

cfg = RLVRConfig()


class GRPOOptimizer:
    # 对同一批 state，生成 K 个 candidate action
    # reward 打分后在组内做 baseline，算 advantage
    # 用 PPO 的 clipped surrogate 更新

    def __init__(self, policy_net, lr=None):
        self.policy = policy_net
        self.optimizer = torch.optim.Adam(
            policy_net.parameters(),
            lr=lr if lr is not None else cfg.learning_rate
        )

    def compute_advantage(self, rewards):
        # Group Relative: 组内归一化 → 看每个方案比组均值好多少
        # rewards: [B, K]
        baseline = rewards.mean(dim=1, keepdim=True)
        std = rewards.std(dim=1, keepdim=True) + 1e-8  # 防 0
        return (rewards - baseline) / std

    def update(self, states, actions, old_log_probs, rewards):
        # states: [B, state_dim]
        # actions: [B, K, action_dim] — K 个候选方案
        # old_log_probs: [B, K] — 采样时记录的 log p
        # rewards: [B, K]
        B, K, D = actions.shape
        S = states.shape[-1]

        advantages = self.compute_advantage(rewards)  # [B, K]

        # flatten 到 [B*K, ...] 好批量算
        s_flat = states.unsqueeze(1).expand(-1, K, -1).reshape(-1, S)
        a_flat = actions.reshape(-1, D)

        # 新策略下重新评估这批 action 的 log prob
        new_lp = self.policy.evaluate_log_prob(s_flat, a_flat).reshape(B, K)

        ratio = torch.exp(new_lp - old_log_probs)  # 重要：这里终于不是≈1了

        # clipped surrogate (PPO style)
        clip_val = cfg.clip_epsilon
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_val, 1 + clip_val) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 加一点熵，防止策略过早 collapse
        ent = -new_lp.mean()
        loss = policy_loss - cfg.entropy_coef * ent

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        # 近似 KL: E[ratio - 1 - log(ratio)]，当 ratio≈1 时接近 0
        approx_kl = (ratio - 1 - ratio.log()).mean().item()

        return {
            "policy_loss": policy_loss.item(),
            "entropy": ent.item(),
            "approx_kl": approx_kl,
            "total_loss": loss.item(),
        }
