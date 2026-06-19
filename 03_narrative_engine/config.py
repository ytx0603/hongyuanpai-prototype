"""叙事引擎配置"""
from dataclasses import dataclass

@dataclass
class NarrativeConfig:
    # 知识图谱
    kg_embed_dim: int = 128
    num_entities: int = 1000   # 模拟用
    num_relations: int = 50    # 模拟用

    # 用户画像
    profile_dim: int = 64
    behavior_seq_len: int = 10  # 行为序列长度

    # RAG
    rag_top_k: int = 5
    chunk_size: int = 256
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # 校验器
    validator_threshold: float = 0.85

    # 路径
    kg_path: str = "./kg_data"
