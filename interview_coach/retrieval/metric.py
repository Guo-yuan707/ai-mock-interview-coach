"""检索评测指标(第 7/8 课):召回率 recall@k,用来量化"哪套检索器更强"。

为什么量化(讲得出 why):
- 不能靠感觉说"向量更好",要有数字:同一批 golden 查询,
  各自算 recall@k(前 k 条命中率),一对比实验,谁强谁弱一目了然;
- 这套指标和"面试 Eval"(第 8 课)是同一种思想:先定 golden 标准,再量化差距。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def recall_at_k(retrieved_sources: list[str], golden_sources: set[str], k: int) -> float:
    """前 k 条里,命中了多少 golden 文档 / golden 总数。

    参数:
        retrieved_sources: 检索器返回结果按序的 source 标签列表(前 k 条)
        golden_sources:    这道查询"应该命中"的 source 集合
    """
    if not golden_sources:
        return 0.0
    hit = sum(1 for s in retrieved_sources[:k] if s in golden_sources)
    return round(hit / len(golden_sources), 3)


@dataclass
class QueryCase:
    """一条 golden 查询:问题 + 应当命中的 source 标签(用于算 recall)。"""
    query: str
    golden: list[str]


@dataclass
class RetrieverReport:
    name: str
    recall: list[float]        # 每条查询的 recall@k
    avg_recall: float
    hits: int

    def line(self, k: int) -> str:
        return (f"{self.name:<22} recall@{k}={self.avg_recall:.3f} "
                f"(每问:{['%.2f' % r for r in self.recall]}) 命中{self.hits}")


def evaluate_retrievers(cases: list[QueryCase], retrievers: dict[str, Any], k: int = 3) -> list[RetrieverReport]:
    """跑对比实验:每套检索器对所有 golden 查询算 recall@k,返回报表。

    用法示例(interactive/CLI):
        evaluate_retrievers(cases, {"词典": LexicalKB, "向量": VectorKB, "混合": HybridSearch}, k=3)
    """
    reports: list[RetrieverReport] = []
    for name, retriever in retrievers.items():
        recalls: list[float] = []
        hit = 0
        for case in cases:
            result = retriever.search(case.query, k=k)
            sources = [r.source for r in result]
            r = recall_at_k(sources, set(case.golden), k)
            recalls.append(r)
            hit += 1 if r > 0 else 0
        reports.append(RetrieverReport(
            name=name,
            recall=recalls,
            avg_recall=round(sum(recalls) / len(recalls), 3) if recalls else 0.0,
            hits=hit,
        ))
    reports.sort(key=lambda x: -x.avg_recall)
    return reports


def print_retriever_report(reports: list[RetrieverReport], k: int = 3) -> str:
    """把对比报表排成几行文字(终端演示 / 文档截图用)。"""
    lines = [f"检索对比实验(recall@{k},越高越好):", "─" * 60]
    for rep in reports:
        lines.append("  " + rep.line(k))
    return "\n".join(lines)
