"""
知识图谱构建模块：
将文物数据、历史事件、人物关系构建为可查询的知识图谱，
作为叙事引擎的"硬约束"——AI 不能超越图谱中记录的事实。
"""

import json
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from config import NarrativeConfig

cfg = NarrativeConfig()


@dataclass
class KGEntity:
    """知识图谱实体"""
    id: str
    name: str
    type: str  # 文物/人物/事件/地点/时期
    attributes: Dict = None

@dataclass
class KGRelation:
    """知识图谱关系"""
    subject_id: str
    predicate: str
    object_id: str
    confidence: float = 1.0


class KnowledgeGraph:
    """
    轻量级知识图谱

    支持实体-关系-实体的三元组存储和查询。
    在实际部署中可对接 Neo4j 或 ArangoDB。
    """

    def __init__(self):
        self.entities: Dict[str, KGEntity] = {}
        self.relations: List[KGRelation] = []
        # 索引
        self._type_index: Dict[str, List[str]] = {}
        self._subject_index: Dict[str, List[KGRelation]] = {}

    def add_entity(self, entity: KGEntity):
        self.entities[entity.id] = entity
        if entity.type not in self._type_index:
            self._type_index[entity.type] = []
        self._type_index[entity.type].append(entity.id)

    def add_relation(self, relation: KGRelation):
        self.relations.append(relation)
        if relation.subject_id not in self._subject_index:
            self._subject_index[relation.subject_id] = []
        self._subject_index[relation.subject_id].append(relation)

    def query_by_type(self, type_name: str) -> List[KGEntity]:
        """按类型查询实体"""
        ids = self._type_index.get(type_name, [])
        return [self.entities[eid] for eid in ids if eid in self.entities]

    def query_relations(self, subject_id: str) -> List[KGRelation]:
        """查询某实体的所有关系"""
        return self._subject_index.get(subject_id, [])

    def get_entity_embedding(self, entity_id: str) -> np.ndarray:
        """
        获取实体的 Embedding（用于奖励函数）

        模拟实现：用实体 ID 的 hash 生成固定维度的向量。
        实际部署中应使用 Sentence-BERT 或其他编码器。
        """
        np.random.seed(hash(entity_id) % (2**31))
        return np.random.randn(cfg.kg_embed_dim).astype(np.float32)

    def build_sample_graph(self):
        """构建示例知识图谱"""
        # 文物
        self.add_entity(KGEntity("rel001", "宁都烈士家书", "文物",
                                 {"年代": "1930年代", "材质": "纸本", "地点": "宁都"}))
        self.add_entity(KGEntity("rel002", "红军长征用过的水壶", "文物",
                                 {"年代": "1934", "材质": "金属", "地点": "瑞金"}))
        self.add_entity(KGEntity("rel003", "二苏大会议桌", "文物",
                                 {"年代": "1933", "材质": "木质", "地点": "瑞金"}))

        # 人物
        self.add_entity(KGEntity("person001", "宁都烈士（无名）", "人物",
                                 {"生卒": "1912-1931", "身份": "红军战士"}))
        self.add_entity(KGEntity("person002", "毛泽东", "人物",
                                 {"生卒": "1893-1976", "身份": "革命家"}))

        # 事件
        self.add_entity(KGEntity("event001", "宁都起义", "事件",
                                 {"时间": "1931", "地点": "宁都"}))
        self.add_entity(KGEntity("event002", "第二次全国苏维埃代表大会", "事件",
                                 {"时间": "1934", "地点": "瑞金"}))

        # 关系
        self.add_relation(KGRelation("rel001", "关联人物", "person001"))
        self.add_relation(KGRelation("rel001", "关联事件", "event001"))
        self.add_relation(KGRelation("rel003", "关联事件", "event002"))
        self.add_relation(KGRelation("event002", "参与者", "person002"))

    def to_dict(self) -> dict:
        return {
            "entities": {k: asdict(v) for k, v in self.entities.items()},
            "relations": [asdict(r) for r in self.relations],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeGraph":
        kg = cls()
        for eid, e_data in data["entities"].items():
            kg.add_entity(KGEntity(**e_data))
        for r_data in data["relations"]:
            kg.add_relation(KGRelation(**r_data))
        return kg


def demo():
    """知识图谱 demo"""
    kg = KnowledgeGraph()
    kg.build_sample_graph()

    print("="*50)
    print("知识图谱 Demo")
    print("="*50)

    for etype in ["文物", "人物", "事件"]:
        entities = kg.query_by_type(etype)
        print(f"\n[{etype}] ({len(entities)} 个)")
        for e in entities:
            rels = kg.query_relations(e.id)
            rel_str = "; ".join([f"{r.predicate} → {r.object_id}" for r in rels[:3]])
            print(f"  {e.name} ({e.id})  {rel_str}")

    print(f"\n图谱总计: {len(kg.entities)} 实体, {len(kg.relations)} 关系")


if __name__ == "__main__":
    demo()
