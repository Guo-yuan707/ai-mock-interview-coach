"""混合检索 + 重排(第 7 课):词典(精确)与向量(近义)各取所长。

为什么混合(讲得出 why):
- 词典检索抓"字面命中",向量检索抓"语义相近",两者盲区互补:
  纯词典漏同义改写;纯向量在术语精确匹配上可能输给词典;
- 常见做法 RRF(Reciprocal Rank Fusion):把两路结果的排名倒数相加,
  不依赖各自分数尺度就能合并 —— 简单、鲁棒、不挑 retriever。

rerank 在这里 = RRF 融合重排(把原始 top-k 用融合分重排)。更重(如模型 rerank)
留给将来;这一版是可离线、可复现的确定性重排,够当对比实验。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .kb import KBHit


def _rrf_score(rank: int, k: int = 60) -> float:
    """RRF:排第 rank 的文档拿到 1/(k+rank) 分。排越前越高,尺度稳定。"""
    return 1.0 / (k + rank)


@dataclass
class HybridSearch:
    """把若干路检索器(list of .search(query,k))的结果按 RRF 融合成一个结果。"""
    retrievers: list[Any] = field(default_factory=list)
    k: int = 60                       # RRF 常数(论文常见 60)
    top_n: int = 5                    # 融合后取前几条

    def search(self, query: str, k: int | None = None) -> list[KBHit]:
        """query → 每路各检索一批 → RRF 合并 → 重排返回 top。"""
        if not self.retrievers:
            return []
        # 以"文本内容"为锚做融合(同一段原文在两路都命中 → 分叠加)
        acc: dict[str, KBHit] = {}           # text -> 命中对象
        fused: dict[str, float] = {}         # text -> RRF 总分
        for retriever in self.retrievers:
            for rank, hit in enumerate(retriever.search(query, k=k or 8), start=1):
                key = (hit.source, hit.text)
                if key not in acc:
                    acc[key] = hit
                    fused[key] = 0.0
                fused[key] += _rrf_score(rank)
        ordered = sorted(acc.items(), key=lambda kv: fused[kv[0]], reverse=True)
        return [hit for _, hit in ordered[:self.top_n]]
