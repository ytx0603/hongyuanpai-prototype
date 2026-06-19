"""
用户画像推断模块：
通过行为轨迹、驻留时间、提问方式等无感信号推断用户类型，
实现"千人千面"的叙事内容适配。

参考论文一（IUI'25）的用户感知方法。
"""

import numpy as np
import torch
import torch.nn as nn
from enum import Enum
from config import NarrativeConfig

cfg = NarrativeConfig()


class UserType(Enum):
    GENERAL = "general"      # 普通游客
    DEEP = "deep"            # 深度爱好者
    CHILD = "child"          # 儿童
    SCHOLAR = "scholar"      # 学者/研究者


class BehaviorSignal:
    """用户行为信号"""

    def __init__(self,
                 dwell_time: float = 0,       # 驻留时间（秒）
                 move_speed: float = 0,       # 移动速度
                 question_depth: float = 0,   # 提问深度 (0~1)
                 interact_count: int = 0,     # 互动次数
                 revisit: bool = False):      # 是否重复观看
        self.dwell_time = dwell_time
        self.move_speed = move_speed
        self.question_depth = question_depth
        self.interact_count = interact_count
        self.revisit = revisit


class UserProfileEncoder(nn.Module):
    """
    用户画像编码器：

    输入行为序列 → LSTM → 用户类型分类
    完全参考论文一的"无感语境感知"方法。
    """

    def __init__(self):
        super().__init__()
        self.input_dim = 5  # dwell, speed, depth, count, revisit
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=cfg.profile_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )
        self.classifier = nn.Sequential(
            nn.Linear(cfg.profile_dim, 32),
            nn.ReLU(),
            nn.Linear(32, len(UserType)),
        )

    def forward(self, behavior_seq):
        """
        Args:
            behavior_seq: [B, T, 5] 行为序列
        Returns:
            logits: [B, 4] 用户类型 logits
        """
        lstm_out, _ = self.lstm(behavior_seq)
        last = lstm_out[:, -1, :]  # 取最后时间步
        return self.classifier(last)

    def predict_user_type(self, signals: list) -> UserType:
        """
        根据行为信号列表推断用户类型

        Args:
            signals: List[BehaviorSignal]

        Returns:
            user_type: UserType
        """
        if not signals:
            return UserType.GENERAL

        # 提取特征
        avg_dwell = np.mean([s.dwell_time for s in signals])
        avg_speed = np.mean([s.move_speed for s in signals])
        avg_depth = np.mean([s.question_depth for s in signals])
        total_interact = sum(s.interact_count for s in signals)
        any_revisit = any(s.revisit for s in signals)

        # 基于规则的快速判断（无需模型亦可工作）
        if avg_depth > 0.7:
            return UserType.SCHOLAR
        if avg_dwell > 30 and total_interact > 5:
            return UserType.DEEP
        if avg_speed < 0.3 and total_interact < 3:
            return UserType.CHILD
        return UserType.GENERAL

    def get_narrative_style(self, user_type: UserType) -> str:
        """根据用户类型返回叙事风格"""
        styles = {
            UserType.CHILD: "通俗故事型：语言生动、有冒险情节、简短段落",
            UserType.GENERAL: "平衡型：有故事有信息，图文并茂，5-8句",
            UserType.DEEP: "详细型：包含历史背景、工艺细节、相关文献",
            UserType.SCHOLAR: "学术型：包含史料出处、考古证据、争议讨论、参考文献",
        }
        return styles.get(user_type, "平衡型")


def demo():
    """用户画像 demo"""
    print("="*50)
    print("用户画像推断 Demo")
    print("="*50)

    encoder = UserProfileEncoder()

    # 模拟不同用户的行为
    test_cases = [
        ("学者", [BehaviorSignal(dwell_time=45, move_speed=0.1, question_depth=0.85, interact_count=8)]),
        ("儿童", [BehaviorSignal(dwell_time=5, move_speed=0.5, question_depth=0.1, interact_count=1)]),
        ("深度爱好者", [BehaviorSignal(dwell_time=60, move_speed=0.05, question_depth=0.6, interact_count=10, revisit=True)]),
        ("普通游客", [BehaviorSignal(dwell_time=8, move_speed=0.4, question_depth=0.2, interact_count=2)]),
    ]

    for label, signals in test_cases:
        user_type = encoder.predict_user_type(signals)
        style = encoder.get_narrative_style(user_type)
        print(f"\n  [{label}]")
        print(f"    推断类型: {user_type.value}")
        print(f"    叙事风格: {style}")


if __name__ == "__main__":
    demo()
