"""词典基线知识库(第 7 课的"对照组"):纯关键词,不调模型、不装重库。

为什么先做这个"笨版本"(讲得出 why):
- 生产级 RAG 要做"对比实验":向量检索到底比纯关键词强多少,得有个基线才好量化;
- 它零依赖、跑得快、可单测 —— 面试官 function calling 里的 retrieve_knowledge
  现在就有真实的、可查证的东西可查,而不是等 embedding 装好才能演示;
- 第 7 课在它旁边加 VectorKB(embedding+向量),两者共享同一个 search 接口,
  评测脚本把两套结果的 recall@k 摆在一起,谁强谁弱一眼见。

接口约定(其它模块只认这个):
    kb.search(query, k=3) -> list[KBHit(text, score, source)]
有了这个约定,engine / tools / 评测 都不关心背后是词典还是向量。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

# 中文段落按相邻两字(二元组)切词:能抓住"向量库""检索流程"这类组合,
# 又不会把整句当一个词;英文/数字按连续词切。这是最朴素的"中文分词降级方案"。
_CJK_RUN = re.compile(r"[一-鿿]+")
_WORD = re.compile(r"[a-zA-Z0-9]+")


@dataclass
class KBHit:
    text: str
    score: float
    source: str = ""


def _tokenize(text: str) -> list[str]:
    """把文本切成小词条:中文出二元组+单字兜底,西文出小写单词。"""
    tokens: list[str] = []
    text = (text or "").lower()
    for run in _CJK_RUN.findall(text):
        n = len(run)
        if n >= 2:
            tokens.extend(run[i:i + 2] for i in range(n - 1))
        tokens.append(run)          # 整段中文也当一条(长短语整段命中得分更高)
    tokens.extend(_WORD.findall(text))
    return tokens


def _tf(text: str) -> dict[str, int]:
    """词频向量:词 → 出现次数。"""
    out: dict[str, int] = {}
    for t in _tokenize(text):
        out[t] = out.get(t, 0) + 1
    return out


class LexicalKB:
    """词典检索:打分 = 查询词与文档词的重叠加权(词频 × 逆文档频率)。"""

    def __init__(self, docs: list[tuple[str, str]] | None = None) -> None:
        """docs: [(source, text), ...] 每一条是一个可检索的片段。"""
        self._docs: list[tuple[str, str, dict[str, int]]] = []
        self._doc_tokens: list[dict[str, int]] = []
        self._df: dict[str, int] = {}
        for source, text in (docs or []):
            if not text.strip():
                continue
            tf = _tf(text)
            self._docs.append((source, text, tf))
            self._doc_tokens.append(tf)
            for term in tf:
                self._df[term] = self._df.get(term, 0) + 1
        self._total = len(self._docs)

    def __len__(self) -> int:
        return self._total

    def add(self, source: str, text: str) -> None:
        self._docs.append((source, text, _tf(text)))
        self._doc_tokens.append(self._docs[-1][2])
        for term in self._doc_tokens[-1]:
            self._df[term] = self._df.get(term, 0) + 1
        self._total += 1

    def search(self, query: str, k: int = 3) -> list[KBHit]:
        """返回最相关的 k 条。查询词越稀有、重叠越多,分越高(idf 加权)。"""
        if not query.strip() or not self._total:
            return []
        q = _tf(query)
        n = self._total
        # 逆文档频率:只在很少文档里出现的词更有区分度
        idf = {term: math.log((n + 1) / (self._df.get(term, 0) + 1)) + 1.0 for term in q}
        qnorm = sum(q.values()) or 1.0
        scored: list[tuple[float, int]] = []
        for i, doc_tf in enumerate(self._doc_tokens):
            s = 0.0
            for term, cq in q.items():
                cd = doc_tf.get(term, 0)
                if cd:
                    s += (cq / qnorm) * math.sqrt(cd) * idf[term]
            if s > 0:
                scored.append((s, i))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            KBHit(text=self._docs[i][1], score=round(s, 4), source=self._docs[i][0])
            for s, i in scored[:k]
        ]


def build_default_kb(state) -> LexicalKB:
    """把一场面试的"弹药"建成可检索的知识库:
    评分维度、高危追问点、JD 原文、简历原文(有就给)分条入库,
    每一条都带 source 标签,方便面试官引用原文、也方便日后回查。
    """
    kb = LexicalKB()
    plan = state.plan
    for d in plan.dimensions:
        kb.add(f"维度:{d.name}", f"{d.name}——{d.why}")
    for i, rp in enumerate(plan.risk_points):
        kb.add(f"高危点r{i}:{rp.title}", f"{rp.title}:{rp.detail}{(' 原文:「' + rp.evidence + '」') if rp.evidence else ''}")
    # JD 按行切块入库,简历同理(保留出处好查证)
    for line in (state.jd or "").splitlines():
        line = line.strip()
        if len(line) >= 6:
            kb.add("JD", line)
    if state.resume:
        for line in (state.resume).splitlines():
            line = line.strip()
            if len(line) >= 6:
                kb.add("简历", line)
    return kb
