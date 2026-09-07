"""一键评测(第 8 课):跑一组指标,给系统"体检"。

用法:
    .venv/Scripts/python -m interview_coach.eval.run             # 离线(不用 key,秒出)
    .venv/Scripts/python -m interview_coach.eval.run --live      # 用真模型跑 AI 出题 + 真评分

三组体检:
    A 检索 recall@k   —— 词典 / 向量 / 混合三套对比(离线即可,零依赖)
    B 出题覆盖        —— 计划有没有抓住 JD 的高危主题(离线=确定性基线;--live=AI 出题)
    C 评分严格度      —— 强回答 vs 弱回答,评卷人分不分得开(离线=脚本示例;--live=真打)

设计取舍:默认离线能跑,是为了"无 key 也可演示/CI";--live 才烧 token,且每项都真。
"""
from __future__ import annotations

import argparse
import json

from .. import planner as planner_mod
from ..eval.golden import (EXPECTED_THEMES, GOLDEN_JD, GOLDEN_RESUME,
                           STRONG_ANSWER, WEAK_ANSWER)
from ..eval.metrics import plan_risks_text, report_card, strictness_gap, theme_coverage
from ..fake import FakeClient
from ..plan import InterviewPlan, RiskPoint, RubricDimension
from ..retrieval import (HybridSearch, LexicalKB, OfflineLexicalEmbedder, VectorKB,
                         evaluate_retrievers, print_retriever_report)


# ---------------- A. 检索对比(纯离线) ----------------

def run_retrieval_compare() -> str:
    """把 golden JD/简历按行切进三套检索器,自动出 golden 查询,比 recall@k。"""
    docs: list[tuple[str, str]] = []
    for i, line in enumerate(GOLDEN_JD.splitlines()):
        line = line.strip()
        if len(line) >= 4:
            docs.append((f"jd{i}", line))
    for i, line in enumerate(GOLDEN_RESUME.splitlines()):
        line = line.strip()
        if len(line) >= 4:
            docs.append((f"res{i}", line))

    lex = LexicalKB(docs)
    vec = VectorKB(embedder=OfflineLexicalEmbedder())
    for src, text in docs:
        vec.add(src, text)
    hybrid = HybridSearch(retrievers=[lex, vec], top_n=3)

    # 自动 golden:每个 JD 主题词 → 应命中包含它的那个 jd 行
    queries = []
    for theme in ["向量库", "多轮对话", "function calling"]:
        gold = [src for src, text in docs if theme.lower() in text.lower()]
        if gold:
            queries.append({"query": f"岗位要求{theme} 怎么做", "golden": gold})
    from ..retrieval.metric import QueryCase
    cases = [QueryCase(q["query"], q["golden"]) for q in queries]
    k = 2
    reports = evaluate_retrievers(cases, {"词典Lexical": lex, "向量Vector": vec, "混合Hybrid": hybrid}, k=k)
    return print_retriever_report(reports, k=k)


# ---------------- B. 出题覆盖 ----------------

def _offline_fake_ai_plan() -> FakeClient:
    """离线"AI 出题"替身:返回一份覆盖 golden 主题的计划 JSON(测覆盖逻辑本身)。"""
    plan_payload = {
        "target_hint": "偏 AI 应用",
        "dimensions": [
            {"name": "岗位理解与匹配", "why": "通用", "source_keywords": []},
            {"name": "技术深度(RAG)", "why": "JD 点名", "source_keywords": ["rag", "向量库"]},
            {"name": "表达与逻辑", "why": "通用", "source_keywords": []},
        ],
        "risk_points": [
            {"kind": "missing_jd_keyword", "title": "简历没写 RAG", "detail": "会问对 RAG 的理解与 embedding", "evidence": "RAG"},
            {"kind": "missing_jd_keyword", "title": "缺向量库", "detail": "会不会用向量库", "evidence": "向量库"},
            {"kind": "jd_probe", "title": "多轮对话上下文管理", "detail": "追问上下文方案", "evidence": "多轮对话"},
            {"kind": "unevidenced_claim", "title": "这句没证据", "detail": "提升多少,给证据", "evidence": "提升了效率"},
            {"kind": "jd_probe", "title": "pytest 测试功底", "detail": "问测试设计", "evidence": "pytest"},
            {"kind": "resume_probe", "title": "STAR 展开", "detail": "一手细节", "evidence": ""},
        ],
    }
    return FakeClient(responder=lambda kind, model, messages, jm: json.dumps(plan_payload, ensure_ascii=False))


def run_planner_coverage(mode: str = "offline", llm=None) -> str:
    """出题覆盖:确定性基线(永远测)+ --live 时另测 AI 版。"""
    if mode == "live":
        # llm 给进来(BYOK)就用它;不给则走默认(读 .env / 全局单例,CLI 行为不变)
        plan, used_mode = planner_mod.build_plan_auto(GOLDEN_JD, GOLDEN_RESUME,
                                                      use_ai=True, llm=llm)
        cov = theme_coverage(plan_risks_text(plan), EXPECTED_THEMES)
        return (f"[AI 出题] 主题覆盖 {cov['rate']:.0%} "
                f"命中:{'、'.join(cov['hit']) or '无'} 漏:{'、'.join(cov['miss']) or '无'}")
    # 离线:确定性基线(AI 版用假计划也跑一遍覆盖,验证"换 AI 不出圈")
    plan = planner_mod.build_plan(GOLDEN_JD, GOLDEN_RESUME)
    cov = theme_coverage(plan_risks_text(plan), EXPECTED_THEMES)
    fake = _offline_fake_ai_plan()
    ai_plan = planner_mod.build_plan_ai(GOLDEN_JD, GOLDEN_RESUME, llm=fake)
    cov_ai = theme_coverage(plan_risks_text(ai_plan), EXPECTED_THEMES)
    return (f"确定性基线覆盖 {cov['rate']:.0%}(命中{'、'.join(cov['hit']) or '无'});"
            f"假AI计划覆盖 {cov_ai['rate']:.0%}")


# ---------------- C. 评分严格度 ----------------

_MINIPLAN = InterviewPlan(
    dimensions=[RubricDimension("岗位理解与匹配", "是否读懂 JD"),
                RubricDimension("技术深度(RAG)", "JD 点明 RAG"),
                RubricDimension("表达与逻辑", "结构清晰")],
    risk_points=[RiskPoint("jd_probe", "讲讲 RAG", "细节")],
)
QUESTION = "请具体讲讲你做 RAG 项目的做法和量化结果。"


def _offline_score_client() -> FakeClient:
    """离线评分替身:回答里带量化证据(如 83%)→ 高分;纯夸大口吻 → 低分。"""
    def responder(kind, model, messages, jm):
        if kind != "score":
            return ""
        last = messages[-1].get("content", "")
        high = "83%" in last or "40%" in last
        payload = {
            "overall": 8 if high else 3,
            "dimensions": [
                {"dimension": "岗位理解与匹配", "score": 5 if high else 2, "comment": "x", "gap": ""},
                {"dimension": "技术深度(RAG)", "score": 5 if high else 1, "comment": "x",
                 "gap": "" if high else "没证据,会被追问穿"},
                {"dimension": "表达与逻辑", "score": 4 if high else 2, "comment": "x", "gap": ""},
            ],
            "strengths": [], "weaknesses": [], "verdict": "x",
        }
        return json.dumps(payload, ensure_ascii=False)
    return FakeClient(responder=responder)


def run_scoring(mode: str = "offline", llm=None) -> str:
    from ..evaluator import score_answer
    # 离线一律用脚本替身;live 时用传入的 client(BYOK),没传则走默认(读 .env / 全局)
    if mode != "live":
        llm = _offline_score_client()
    strong = score_answer(_MINIPLAN, QUESTION, STRONG_ANSWER, llm=llm)
    weak = score_answer(_MINIPLAN, QUESTION, WEAK_ANSWER, llm=llm)
    g = strictness_gap(strong.overall, weak.overall)
    return f"强回答 {strong.overall}/10 vs 弱回答 {weak.overall}/10 → gap {g['gap']} → {g['verdict']}"


# ---------------- 主入口 ----------------

def run_all(mode: str = "offline", llm=None) -> str:
    sec_a = run_retrieval_compare()
    sec_b = run_planner_coverage(mode, llm=llm)
    sec_c = run_scoring(mode, llm=llm)
    card = report_card([("检索对比", sec_a.splitlines()[-1] if sec_a else "无"),
                        ("出题覆盖", sec_b),
                        ("评分严格度", sec_c)])
    body = "\n\n".join([sec_a, f"B. 出题覆盖 → {sec_b}", f"C. 评分严格度 → {sec_c}", card])
    head = f"# AI 模拟面试官 · Eval 一键评测({mode})\n\nA. 检索对比(recall@k):\n"
    return head + body


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="interview_coach 一键评测(默认离线,免 key)")
    p.add_argument("--live", action="store_true", help="用真模型跑 AI 出题 + 真实评分(烧 token)")
    p.add_argument("--out", default=None, help="把报告写到文件(可选)")
    args = p.parse_args(argv)
    mode = "live" if args.live else "offline"
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    text = run_all(mode)
    print(text)
    if args.out:
        from pathlib import Path
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"\n报告已存:{args.out}")


if __name__ == "__main__":
    main()
