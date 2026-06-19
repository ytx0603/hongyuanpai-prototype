"""
一键溯源模块：
每条叙事内容附带"信息来源"标签，
用户点击即可查看知识图谱中的原始出处。
确保"好看"不"戏说"。
"""

from typing import List, Optional
from dataclasses import dataclass
from config import NarrativeConfig
from kg_builder import KnowledgeGraph, KGEntity, KGRelation

cfg = NarrativeConfig()


@dataclass
class SourceTrace:
    """溯源信息"""
    entity_name: str
    entity_type: str
    source_description: str
    related_entities: List[str]
    confidence: float


class TraceabilityEngine:
    """
    溯源引擎

    为每一段叙事内容生成可追溯的"来源链"。
    """

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    def trace(self, entity_id: str) -> SourceTrace:
        """
        为指定实体生成溯源信息

        Args:
            entity_id: 实体 ID

        Returns:
            SourceTrace: 溯源信息
        """
        entity = self.kg.entities.get(entity_id)
        if not entity:
            return SourceTrace("未知", "未知", "未找到该实体的来源信息", [], 0.0)

        # 关联实体
        rels = self.kg.query_relations(entity_id)
        related = []
        for r in rels:
            obj = self.kg.entities.get(r.object_id)
            if obj:
                related.append(f"{r.predicate} → {obj.name}")

        # 来源描述
        if entity.attributes:
            attr_parts = [f"{k}: {v}" for k, v in entity.attributes.items()]
            source = f"依据{'、'.join(attr_parts)}记载"
        else:
            source = "知识图谱收录"

        return SourceTrace(
            entity_name=entity.name,
            entity_type=entity.type,
            source_description=source,
            related_entities=related,
            confidence=0.95,
        )

    def trace_text(self, text: str) -> List[SourceTrace]:
        """
        分析文本中提到的所有实体并溯源

        Args:
            text: 叙事文本

        Returns:
            List[SourceTrace]: 所有提及实体的溯源信息
        """
        traces = []
        for eid, entity in self.kg.entities.items():
            if entity.name in text:
                trace = self.trace(eid)
                traces.append(trace)
        return traces

    def format_trace_report(self, trace: SourceTrace) -> str:
        """格式化溯源报告"""
        lines = [
            f"📖 信息来源",
            f"━━━━━━━━━━━━━━━━",
            f"实体: {trace.entity_name}",
            f"类型: {trace.entity_type}",
            f"来源: {trace.source_description}",
            f"置信度: {trace.confidence:.0%}",
        ]
        if trace.related_entities:
            lines.append(f"关联信息:")
            for r in trace.related_entities:
                lines.append(f"  • {r}")
        return "\n".join(lines)


def demo():
    """溯源 demo"""
    print("="*50)
    print("一键溯源 Demo")
    print("="*50)

    kg = KnowledgeGraph()
    kg.build_sample_graph()
    engine = TraceabilityEngine(kg)

    test_entities = ["rel001", "rel003", "person002"]

    for eid in test_entities:
        entity = kg.entities.get(eid)
        if entity:
            print(f"\n  「{entity.name}」")
            trace = engine.trace(eid)
            report = engine.format_trace_report(trace)
            for line in report.split('\n'):
                print(f"    {line}")


if __name__ == "__main__":
    demo()
