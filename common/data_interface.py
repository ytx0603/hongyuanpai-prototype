# -*- coding: utf-8 -*-
# 统一数据接口 — 所有模块都从这拿数据，不直接碰文件
#
# SyntheticDataProvider: 造数据用（开发/测试），基于物理规则模拟
# RealDataProvider: 接博物馆真数据（需要授权和数据路径）
# 切换只要改 get_data_provider() 的参数，代码不用动

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
from enum import Enum


class ChangeType(Enum):
    NO_CHANGE = 0
    CRACK = 1
    SPALLING = 2
    WEATHERING = 3
    BIODETERIORATION = 4


@dataclass
class ArtifactScan:
    artifact_id: str
    artifact_name: str
    point_cloud: "np.ndarray"
    scan_date: str
    resolution: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class ScanPair:
    artifact_id: str
    before: "torch.Tensor"
    after: "torch.Tensor"
    label: ChangeType
    scan_interval_days: int


@dataclass
class ExpertAnnotation:
    artifact_id: str
    state: "torch.Tensor"
    expert_action: "torch.Tensor"
    annotator: str
    annotation_date: str
    confidence: float = 1.0


@dataclass
class KnowledgeGraphRecord:
    entity_id: str
    entity_name: str
    entity_type: str
    attributes: Dict = field(default_factory=dict)
    relations: List = field(default_factory=list)


class DataProvider(ABC):
    # 抽象接口 — 所有模块通过这个拿数据
    # 切换真/假数据源只要换掉 get_data_provider 的参数
    @abstractmethod
    def get_artifact_scan(self, artifact_id): ...
    @abstractmethod
    def get_scan_pair(self, artifact_id): ...
    @abstractmethod
    def get_expert_annotations(self, num_samples=100): ...
    @abstractmethod
    def get_knowledge_graph_records(self): ...
    @property
    @abstractmethod
    def provider_name(self): ...
    @property
    @abstractmethod
    def is_synthetic(self): ...

# ============================================================
# SyntheticDataProvider — 基于物理规则的模拟数据
# ============================================================

class SyntheticDataProvider(DataProvider):
    def __init__(self, grid_dim=64, state_dim=128, action_dim=64, seed=42):
        self.grid_dim = grid_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.rng = np.random.RandomState(seed)
        self._registry = self._build_registry()

    def _build_registry(self):
        return {
            "JGS-001": {"name": "宁都烈士家书", "period": "1930", "material": "纸本"},
            "JGS-002": {"name": "红军水壶", "period": "1934", "material": "金属"},
            "JGS-003": {"name": "二苏大会议桌", "period": "1933", "material": "木质"},
            "JGS-004": {"name": "红军军号", "period": "1928", "material": "金属"},
            "JGS-005": {"name": "斗争时期标语", "period": "1927-1937", "material": "纸本"},
        }

    @property
    def provider_name(self):
        return "SyntheticDataProvider"
    @property
    def is_synthetic(self):
        return True

    def get_artifact_scan(self, artifact_id):
        info = self._registry.get(artifact_id)
        if info is None:
            raise KeyError(f"{artifact_id} not found. Available: {list(self._registry.keys())}")
        scale = 0.3 if info["material"] == "纸本" else 0.5
        cloud = self.rng.uniform(-scale, scale, (4096, 3))
        return ArtifactScan(artifact_id=artifact_id, artifact_name=info["name"],
                            point_cloud=cloud, scan_date="2024-06-15", resolution=0.1,
                            metadata={"年代": info["period"], "材质": info["material"]})

    def get_scan_pair(self, artifact_id, change_type=None):
        if change_type is None:
            change_type = ChangeType(self.rng.randint(0, 5))
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "02_3DCNN"))
        from pointcloud_pipeline import PointCloudPipeline
        pipeline = PointCloudPipeline(self.grid_dim)
        before, after = pipeline.generate_sample_pair(change_type=change_type.value)
        return ScanPair(artifact_id=artifact_id, before=before, after=after,
                        label=change_type, scan_interval_days=365)

    def get_expert_annotations(self, num_samples=100):
        torch.manual_seed(42)
        anns = []
        ids = list(self._registry.keys())
        for i in range(num_samples):
            s = torch.randn(self.state_dim)
            a = torch.tanh(torch.randn(self.action_dim) * 0.5 + s[:self.action_dim] * 0.8)
            anns.append(ExpertAnnotation(artifact_id=ids[i % len(ids)], state=s,
                                         expert_action=a, annotator="SYNTH",
                                         annotation_date="2024-07-01", confidence=0.85))
        return anns

    def get_knowledge_graph_records(self):
        return [
            KnowledgeGraphRecord("rel001", "宁都烈士家书", "文物", {"年代": "1930", "材质": "纸本", "地点": "宁都"}, [("rel001", "关联人物", "person001")]),
            KnowledgeGraphRecord("rel002", "红军水壶", "文物", {"年代": "1934", "材质": "金属", "地点": "瑞金"}, []),
            KnowledgeGraphRecord("rel003", "二苏大会议桌", "文物", {"年代": "1933", "材质": "木质", "地点": "瑞金"}, [("rel003", "关联事件", "event002")]),
            KnowledgeGraphRecord("person001", "宁都烈士", "人物", {"生卒": "1912-1931", "身份": "红军战士"}, []),
            KnowledgeGraphRecord("event001", "宁都起义", "事件", {"时间": "1931", "地点": "宁都"}, []),
        ]


def get_data_provider(provider_type="synthetic", **kwargs):
    if provider_type == "synthetic":
        return SyntheticDataProvider(**kwargs)
    elif provider_type == "real":
        raise NotImplementedError(
            "真实数据源需要博物馆数据授权。详见 DATA_POLICY.md")
    else:
        raise ValueError(f"Unknown provider: {provider_type}")