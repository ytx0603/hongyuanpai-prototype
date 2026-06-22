# -*- coding: utf-8 -*-
# Self-play 模块：Actor 自己跟自己下棋，在数字孪生上反复尝试修复
#
# v1 的问题：Actor 是 deterministic 的，GRPO 的 log_prob 用 -MSE 凑数
#           结果 importance ratio ≈ 1，梯度等于白算
# v2 改成了 Gaussian stochastic policy，sample() 返回 (action, log_prob)
#           然后 evaluate_log_prob() 在参数更新后重算，ratio 就有区分度了
#
# 踩坑记录：log_prob 在 sample() 和 evaluate_log_prob() 里必须作用在
#           同一个 action 值上（见 2026-06-22 的 debug），不然 ratio=exp(A-B)
#           其中 A≠B，random noise 会吃掉 KL 信号

import torch
import torch.nn as nn
import numpy as np
from collections import deque
from config import RLVRConfig

cfg = RLVRConfig()


class SelfPlayBuffer:
    # 经验回放，队列满了自动踢最旧的
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # 不够就重复采样凑数
        n = len(self.buffer)
        idx = np.random.choice(n, batch_size, replace=batch_size > n)
        batch = [self.buffer[i] for i in idx]
        return (
            torch.stack([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]),
            torch.tensor([b[2] for b in batch], dtype=torch.float32),
            torch.stack([b[3] for b in batch]),
            torch.tensor([b[4] for b in batch], dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


class SelfPlayActor(nn.Module):
    # 高斯策略 Actor
    # 输出 mean + learnable log_std，sample 用 rsample 保证可微
    def __init__(self, state_dim=128, action_dim=64):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.mean_head = nn.Linear(256, action_dim)
        # log_std 是一个可学参数，初始值 exp(-1) ≈ 0.37
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.0))

    def _get_distribution(self, state):
        # 计算当前策略分布的 mu 和 sigma
        shared = self.shared(state)
        mean = torch.tanh(self.mean_head(shared))
        std = torch.exp(self.log_std.clamp(-20, 2))
        return mean, std

    def forward(self, state):
        # deterministic forward: 只输出均值，给 SFT 和 eval 用
        mean, _ = self._get_distribution(state)
        return mean

    def sample(self, state):
        # 随机采样一个动作 + 它的 log 概率
        # 用 rsample (reparam trick) 这样梯度能穿过采样步骤
        mean, std = self._get_distribution(state)
        dist = torch.distributions.Normal(mean, std)
        raw = dist.rsample()
        act = torch.clamp(raw, -1.0, 1.0)
        # 注意：log_prob 算在 clamped action 上（不是 raw）
        # 不然跟 evaluate_log_prob 接不上（那是根据传进来的 action 算的）
        lp = dist.log_prob(act).sum(dim=-1)
        return act, lp

    def evaluate_log_prob(self, state, action):
        # GRPO update 时用：拿当前参数重新评估旧动作的概率
        mean, std = self._get_distribution(state)
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(action).sum(dim=-1)


class SelfPlayLoop:
    # self-play 的主循环：采样 → 打分 → 存 buffer
    def __init__(self, actor, buffer):
        self.actor = actor
        self.buffer = buffer
        self.optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.learning_rate)
        # 把 action 投影回 state 空间，用来算 next_state（简单线性映射）
        self.action_projector = nn.Linear(actor.action_dim, actor.state_dim)

    def simulate_repair(self, state, add_noise=True):
        # 训练时 add_noise=True → 随机策略采样
        # eval 时 add_noise=False → 确定性输出
        if add_noise and self.training:
            return self.actor.sample(state)
        return self.actor(state), None

    def run_round(self, states, reward_fn):
        # 一轮 self-play: 采样动作 → reward 打分 → 写入 buffer
        actions, _ = self.simulate_repair(states, add_noise=True)

        device = states.device
        rewards = reward_fn(actions)

        # action_projector 可能在不同设备上（第一次用的时候挪一下）
        proj_dev = next(self.action_projector.parameters()).device
        if proj_dev != device:
            self.action_projector = self.action_projector.to(device)

        action_proj = self.action_projector(actions)
        next_states = states + 0.05 * action_proj  # 小步模拟修复效果
        for i in range(len(states)):
            done_flag = (rewards["total"][i] > 0.85).float()
            self.buffer.push(
                states[i].detach(), actions[i].detach(),
                rewards["total"][i].item(),
                next_states[i].detach(), done_flag.item()
            )

        return {
            "avg_reward": rewards["total"].mean().item(),
            "avg_structure": rewards["structure"].mean().item(),
            "avg_material": rewards["material"].mean().item(),
            "avg_history": rewards["history"].mean().item(),
            "buffer_size": len(self.buffer),
        }

    @property
    def training(self):
        return self.actor.training
