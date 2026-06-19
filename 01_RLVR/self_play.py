"""
自我对弈 (Self-Play) 循环：
AI 在数字孪生体上反复模拟修复，
每次推理结果作为下一次的输入，
在探索中自主进化。
"""

import torch
import torch.nn as nn
import numpy as np
from collections import deque
from config import RLVRConfig

cfg = RLVRConfig()


class SelfPlayBuffer:
    """自我对弈经验回放缓冲区"""

    def __init__(self, max_size: int = 1000):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states = torch.stack([b[0] for b in batch])
        actions = torch.stack([b[1] for b in batch])
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        next_states = torch.stack([b[3] for b in batch])
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class SelfPlayActor(nn.Module):
    """自我对弈中的 Actor 网络"""

    def __init__(self, state_dim: int = 128, action_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """输出修复动作（修复参数）"""
        return torch.tanh(self.net(state))  # [-1, 1]


class SelfPlayLoop:
    """
    自我对弈主循环

    工作流程：
    1. 当前 Actor 对一批文物残片生成修复方案
    2. 用 Reward 函数评分
    3. 高分的方案存入"成功经验"缓冲区
    4. 低分的方案作为"失败经验"存入
    5. 用缓冲区数据更新 Actor
    6. 重复 1-5 直到收敛
    """

    def __init__(self, actor: SelfPlayActor, buffer: SelfPlayBuffer):
        self.actor = actor
        self.buffer = buffer
        self.optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.learning_rate)
        # 动作→状态投影层
        self.action_projector = nn.Linear(actor.action_dim, actor.state_dim)

    def simulate_repair(self, state: torch.Tensor, add_noise: bool = True) -> torch.Tensor:
        """模拟一次修复，返回修复动作"""
        action = self.actor(state)
        if add_noise and self.training:
            # 探索噪声
            noise = torch.randn_like(action) * 0.1
            action = action + noise
        return action

    def run_round(self, states: torch.Tensor, reward_fn) -> dict:
        """
        运行一轮自我对弈

        Args:
            states: [B, D] 文物残片状态
            reward_fn: 奖励函数

        Returns:
            stats: 训练统计
        """
        actions = self.simulate_repair(states, add_noise=True)
        rewards = reward_fn(actions)

        # 存入缓冲区
        device = states.device
        self.action_projector = self.action_projector.to(device)
        action_proj = self.action_projector(actions)  # [B, state_dim]
        next_states = states + 0.05 * action_proj  # 模拟修复后状态
        for i in range(len(states)):
            done = (rewards["total"][i] > 0.85).float()
            self.buffer.push(
                states[i].detach(),
                actions[i].detach(),
                rewards["total"][i].item(),
                next_states[i].detach(),
                done.item()
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
