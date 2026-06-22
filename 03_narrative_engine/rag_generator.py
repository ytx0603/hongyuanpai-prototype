"""
RAG 叙事内容生成器：

基于知识图谱检索 + 用户画像适配，
生成"千人千面"的文物叙事内容。

三层补救机制：
1. 内容渐进式递进
2. 主动提问入口
3. 反事实纠正
"""

import numpy as np
from typing import List, Optional
from config import NarrativeConfig
from kg_builder import KnowledgeGraph
from user_profiling import UserProfileEncoder, UserType, BehaviorSignal

cfg = NarrativeConfig()


class RAGGenerator:
    """
    检索增强生成器

    在实际部署中，这里会对接 LLM API。
    Demo 版本使用模板化生成来演示完整流程。
    """

    def __init__(self, kg: KnowledgeGraph, profiler: UserProfileEncoder):
        self.kg = kg
        self.profiler = profiler
        self._content_templates = {
            UserType.CHILD: {
                "文物": "小朋友，这个{name}已经{age}岁啦！它见过好多好多故事呢…",
                "人物": "{name}是一个勇敢的人，他做了一件了不起的事情…",
                "事件": "{year}年，发生了一件大事…",
            },
            UserType.GENERAL: {
                "文物": "这件{name}出土于{location}，距今约{age}年。它见证了…",
                "人物": "{name}（{years}）是这段历史中的重要人物…",
                "事件": "{year}年的{name}，改变了…",
            },
            UserType.DEEP: {
                "文物": "{name}，{material}材质，{age}年历史。从其工艺特征来看…",
                "人物": "{name}（{years}），关于他还有一段不为人知的细节…",
                "事件": "{name}的背景、经过和影响，需要从三个层面来理解…",
            },
            UserType.SCHOLAR: {
                "文物": "{name}（ID: {id}）的断代依据主要有以下几方面：1)…",
                "人物": "关于{name}的生平，现有史料记载存在以下争议…",
                "事件": "学界对{name}的历史评价主要有三种观点…",
            },
        }

    def retrieve(self, query: str, top_k: int = 3) -> list:
        """
        从知识图谱检索相关内容

        基于关键词匹配（实际部署应使用 embedding 语义检索）。
        """
        results = []
        query_lower = query.lower()
        for eid, entity in self.kg.entities.items():
            score = 0
            if query_lower in entity.name.lower():
                score += 1.0
            if entity.attributes:
                for k, v in entity.attributes.items():
                    if query_lower in str(v).lower():
                        score += 0.5
            if score > 0:
                results.append((entity, score))

        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def generate(self,
                 entity_id: str,
                 user_type: UserType = UserType.GENERAL) -> str:
        """
        为指定实体生成适配的叙事内容

        Args:
            entity_id: 知识图谱中的实体 ID
            user_type: 用户类型

        Returns:
            narrative: 叙事文本
        """
        entity = self.kg.entities.get(entity_id)
        if not entity:
            return "未找到相关信息。"

        template_map = self._content_templates.get(user_type, self._content_templates[UserType.GENERAL])
        template = template_map.get(entity.type, "这是{name}。")

        # 填充模板（v2: 使用安全的 .get 回退，避免 KeyError）
        params = {"name": entity.name, "id": entity.id}
        if entity.attributes:
            params.update(entity.attributes)

        # 使用 defaultdict 风格的回退确保所有模板占位符都有值
        try:
            content = template.format(**{k: params.get(k, f"[{k}暂无记录]") for k in params})
            # 二次尝试：捕获模板中需要的但 params 不存在的 key
            content = template.format(**params)
        except KeyError as e:
            missed_key = str(e).strip("'")
            params[missed_key] = f"[{missed_key}暂无记录]"
            content = template.format(**params)
        except Exception:
            content = f"{entity.name} — 一件值得我们了解的文物。"

        # 添加深度入口
        if user_type in [UserType.GENERAL, UserType.CHILD]:
            content += "\n\n【想了解更多？】可以问我具体的问题，比如'这件文物是怎么发现的？'"

        return content

    def layered_generate(self,
                         entity_id: str,
                         user_type: UserType,
                         depth_level: int = 1) -> str:
        """
        分层内容生成（三层补救机制的第一层）

        depth_level:
            1 = 入口层（简短钩子）
            2 = 故事层（生动叙述）
            3 = 深度层（完整信息）
        """
        if depth_level == 1:
            # 钩子层
            return f"「{self.kg.entities.get(entity_id, '').name if self.kg.entities.get(entity_id) else '...'}」想知道它的故事吗？"
        elif depth_level == 2:
            return self.generate(entity_id, user_type)
        else:
            # 深度层：附加相关关系
            base = self.generate(entity_id, user_type)
            rels = self.kg.query_relations(entity_id)
            if rels:
                related = []
                for r in rels[:3]:
                    obj = self.kg.entities.get(r.object_id)
                    if obj:
                        related.append(f"• 关联{obj.type}：{obj.name}")
                base += "\n\n**相关信息**\n" + "\n".join(related)
            return base


def demo():
    """叙事引擎 demo"""
    print("="*50)
    print("多模态叙事引擎 Demo")
    print("="*50)

    # 初始化
    kg = KnowledgeGraph()
    kg.build_sample_graph()
    profiler = UserProfileEncoder()
    generator = RAGGenerator(kg, profiler)

    # 测试不同用户类型的内容适配
    test_entities = ["rel001", "rel002"]
    test_users = [
        ("一位带着孩子的家长", UserType.CHILD),
        ("普通参观者", UserType.GENERAL),
        ("历史爱好者", UserType.DEEP),
        ("访问学者", UserType.SCHOLAR),
    ]

    for eid in test_entities:
        entity = kg.entities.get(eid)
        print(f"\n--- {entity.name} ---")
        for label, utype in test_users:
            content = generator.generate(eid, utype)
            print(f"\n  [{label}]")
            for line in content.split('\n'):
                print(f"    {line}")

    # 分层内容展示
    print(f"\n--- 分层内容示例 ---")
    for level in [1, 2, 3]:
        content = generator.layered_generate("rel001", UserType.GENERAL, level)
        print(f"\n  [深度 Level {level}] {content[:80]}...")


if __name__ == "__main__":
    demo()
