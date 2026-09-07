"""Eval 套件测试:纯指标 + 离线一键评测三组都能跑通(不碰网络)。"""

from interview_coach.eval.metrics import plan_risks_text, report_card, strictness_gap, theme_coverage
from interview_coach.eval.run import run_all, run_retrieval_compare
from interview_coach.plan import InterviewPlan, RiskPoint


def _plan():
    return InterviewPlan(risk_points=[
        RiskPoint("jd_probe", "讲讲 RAG 与 embedding", "检索细节"),
        RiskPoint("missing_jd_keyword", "简历没提多轮对话", "上下文管理"),
    ])


def test_theme_coverage_math():
    text = plan_risks_text(_plan())
    cov = theme_coverage(text, ["rag", "多轮对话", "python", "证据"])
    assert cov["rate"] == 0.5            # rag/多轮对话 命中,python/证据 未命中
    assert set(cov["hit"]) == {"rag", "多轮对话"}
    assert "python" in cov["miss"]


def test_strictness_gap_verdicts():
    assert strictness_gap(8, 3)["verdict"] == "严格(分得开)"
    assert strictness_gap(6, 5)["verdict"] == "偏松(区分度不足)"
    assert strictness_gap(2, 7)["verdict"] == "异常(强弱倒挂)"


def test_report_card_prints():
    text = report_card([("检索", "recall=0.8"), ("出题", "覆盖 90%")])
    assert "检索" in text and "recall=0.8" in text


def test_retrieval_compare_offline_runs():
    """离线检索对比能跑,且报表带 recall 字样(不需要任何 API)。"""
    text = run_retrieval_compare()
    assert "recall@" in text


def test_eval_run_all_offline_runs():
    """一键评测(离线)能完整跑完,不抛错、有成绩单。"""
    text = run_all("offline")
    assert "Eval 一键评测" in text
    assert "评分严格度" in text
