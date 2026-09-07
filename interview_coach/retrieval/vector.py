"""向量库(第 7 课):把文档嵌入成稠密向量,用余弦相似度检索。

和词典基线 LexicalKB 的区别(面试要讲得清):
- LexicalKB 靠"词是否字面出现"打分 → 换说法(同义词/近义表达)就漏;
- VectorKB 靠"向量方向接近"打分 → 理论上能抓语义相近(需要真语义 embedding);
- 当前向量是"词典哈希向量"(见 embeddings.py),仍是字面维度——只有当配了语义
  embedding API,或换成本地语义模型后,这里才体现语义优势。诚实标注,别吹。

接口与 LexicalKB 完全一致:
    kb.search(query, k) -> list[KBHit]
所以上层(工具/评测)能在两个实现间无缝切换、做对比实验。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .embeddings import Embedder, OfflineLexicalEmbedder, get_embedder
from .kb import KBHit


def _cosine(a: list[float], b: list[float]) -> float:
    """两向量余弦相似度(都已 L2 归一化时,就是点积)。"""
    s = 0.0
    for x, y in zip(a, b):
        s += x * y
    return s


@dataclass
class VectorKB:
    """稠密向量检索。支持追加文档;向量在 add 时就算好,检索只做点积排序。"""
    embedder: Embedder = field(default_factory=get_embedder)
    # docs 结构:(source, text, vector)
    _docs: list[tuple[str, str, list[float]]] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"vector({self.embedder.name})"

    def __len__(self) -> int:
        return len(self._docs)

    def add(self, source: str, text: str) -> None:
        if not text.strip():
            return
        vec = self.embedder.embed([text])[0]
        self._docs.append((source, text, vec))

    def search(self, query: str, k: int = 3) -> list[KBHit]:
        if not query.strip() or not self._docs:
            return []
        qv = self.embedder.embed([query])[0]
        scored = [
            (_cosine(qv, vec), i)
            for i, (_, _, vec) in enumerate(self._docs)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        out: list[KBHit] = []
        for s, i in scored[:k]:
            if s <= 0:
                break
            src, text, _ = self._docs[i]
            out.append(KBHit(text=text, score=round(float(s), 4), source=src))
        return out


def build_vector_kb(state, embedder: Embedder | None = None) -> VectorKB:
    """和 build_default_kb 同一份弹药,但建成向量库(对比实验的"同题不同检索器")。"""
    kb = VectorKB(embedder=embedder or get_embedder())
    plan = state.plan
    for d in plan.dimensions:
        kb.add(f"维度:{d.name}", f"{d.name}——{d.why}")
    for i, rp in enumerate(plan.risk_points):
        kb.add(f"高危点r{i}:{rp.title}",
               f"{rp.title}:{rp.detail}{(' 原文:「' + rp.evidence + '」') if rp.evidence else ''}")
    for line in (state.jd or "").splitlines():
        line = line.strip()
        if len(line) >= 6:
            kb.add("JD", line)
    if state.resume:
        for line in state.resume.splitlines():
            line = line.strip()
            if len(line) >= 6:
                kb.add("简历", line)
    return kb
