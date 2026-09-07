"""RAG v2 测试:词典/向量/混合三路检索 + recall@k 指标 + embedder 确定性(不碰网络)。"""

from interview_coach.retrieval import (
    HybridSearch,
    LexicalKB,
    OfflineLexicalEmbedder,
    QueryCase,
    VectorKB,
    evaluate_retrievers,
    print_retriever_report,
    recall_at_k,
)

DOCS = [
    ("jd1", "岗位要求熟悉向量库,能用 embedding 做语义检索。"),
    ("jd2", "岗位要求会 Python 与 pytest,做单元测试。"),
    ("res1", "候选人用过 RAG,调过向量库接口。"),
    ("res2", "候选人只写过几个 Python 小脚本。"),
]


def test_lexical_kb_top_hit():
    kb = LexicalKB(DOCS)
    hits = kb.search("向量库 embedding", k=1)
    assert hits and hits[0].source == "jd1"


def test_vector_kb_deterministic_and_top_hit():
    emb = OfflineLexicalEmbedder()
    a = emb.embed(["向量库 embedding 语义检索"])
    b = emb.embed(["向量库 embedding 语义检索"])
    assert a == b                      # 同文本 → 同向量(md5 稳定,可复现)
    kb = VectorKB(embedder=emb)
    for src, text in DOCS:
        kb.add(src, text)
    hits = kb.search("向量库 embedding", k=1)
    assert hits and hits[0].source == "jd1"


def test_hybrid_fuses_and_reranks():
    lex = LexicalKB(DOCS)
    vec = VectorKB(embedder=OfflineLexicalEmbedder())
    for src, text in DOCS:
        vec.add(src, text)
    hybrid = HybridSearch(retrievers=[lex, vec], top_n=3)
    hits = hybrid.search("单元测试", k=5)
    assert hits[0].source == "jd2"     # RRF 融合后,该命中的仍在最前


def test_recall_at_k_math():
    assert recall_at_k(["a", "b", "c"], {"a", "x"}, k=3) == 0.5
    assert recall_at_k(["a", "b"], {"c"}, k=2) == 0.0


def test_evaluate_retrievers_report():
    cases = [
        QueryCase("向量库怎么用", golden=["jd1"]),
        QueryCase("python 测试", golden=["jd2", "res2"]),
    ]
    lex = LexicalKB(DOCS)
    vec = VectorKB(embedder=OfflineLexicalEmbedder())
    for src, text in DOCS:
        vec.add(src, text)
    reports = evaluate_retrievers(cases, {"词典": lex, "向量": vec}, k=3)
    assert len(reports) == 2
    assert all(0.0 <= r.avg_recall <= 1.0 for r in reports)
    text = print_retriever_report(reports)
    assert "recall@" in text            # 报表能打出来给人看
