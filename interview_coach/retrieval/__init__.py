"""retrieval 子包:语义资料库(第 7 课 RAG)。

检索器三件套(共享同一 search 接口,可无缝对比):
    LexicalKB   词典基线:字面命中(先落地,面试官立刻可查)
    VectorKB    向量检索:embedding + 余弦(embedder 可插拔,见 embeddings.py)
    HybridSearch 混合 + RRF 重排:词典与向量各取所长

配套:
    build_default_kb / build_vector_kb  从一场面试的"弹药"建库
    evaluate_retrievers / recall@k      跑对比实验量化差距
"""
from .embeddings import Embedder, OfflineLexicalEmbedder, SemanticEmbedder, get_embedder
from .hybrid import HybridSearch
from .kb import KBHit, LexicalKB, build_default_kb
from .metric import QueryCase, RetrieverReport, evaluate_retrievers, print_retriever_report, recall_at_k
from .vector import VectorKB, build_vector_kb

__all__ = [
    "KBHit", "LexicalKB", "build_default_kb", "build_vector_kb", "VectorKB",
    "HybridSearch", "Embedder", "OfflineLexicalEmbedder", "SemanticEmbedder", "get_embedder",
    "QueryCase", "RetrieverReport", "evaluate_retrievers", "print_retriever_report", "recall_at_k",
]
