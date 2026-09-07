"""评卷人测试:结构化打分 + 维度名防幻觉清洗(不碰网络)。"""

import json

from interview_coach.evaluator import ScoreDim, TurnScore, score_answer
from interview_coach.fake import FakeClient
from interview_coach.plan import InterviewPlan, RiskPoint, RubricDimension


def _plan():
    return InterviewPlan(
        dimensions=[
            RubricDimension("岗位理解与匹配", "是否读懂 JD"),
            RubricDimension("技术深度(RAG)", "JD 点明 RAG"),
            RubricDimension("表达与逻辑", "结构清晰"),
        ],
        risk_points=[RiskPoint("jd_probe", "讲讲 RAG", "细节")],
    )


def _fake_returns(plan):
    """评卷 JSON:两个真实维度 + 一个"计划里没有"的维度(故意测清洗)。"""
    payload = {
        "overall": 4,
        "dimensions": [
            {"dimension": "技术深度(RAG)", "score": 2, "comment": "只讲了概念", "gap": "没给落地细节,会被追问穿"},
            {"dimension": "表达与逻辑", "score": 3, "comment": "结构一般", "gap": ""},
            {"dimension": "颜值气质", "score": 5, "comment": "模型乱加的维度"},   # ← 该被丢掉
        ],
        "strengths": ["提到了 RAG 定义"],
        "weaknesses": ["没有数字和一手细节"],
        "verdict": "偏弱,下题追落地细节",
    }
    return FakeClient(responder=lambda kind, model, messages, jm: json.dumps(payload, ensure_ascii=False))


def test_score_answer_returns_typed_turn_score():
    plan = _plan()
    fake = _fake_returns(plan)
    ts = score_answer(plan, "讲讲你的 RAG 项目?", "我做过 RAG,用了向量库。", llm=fake)
    assert isinstance(ts, TurnScore)
    assert 1 <= ts.overall <= 10
    assert fake.records[0].kind == "score"


def test_hallucinated_dimension_is_dropped():
    """模型多报计划外的维度名 → 被清洗掉,不让它给计划偷偷加戏。"""
    plan = _plan()
    ts = score_answer(plan, "q", "a", llm=_fake_returns(plan))
    names = [d.dimension for d in ts.dimensions]
    assert "颜值气质" not in names
    assert "技术深度(RAG)" in names
    assert ts.weak_dimensions == ["技术深度(RAG)"]  # 2分 < 3 → 进薄弱清单


def test_avg_and_weak():
    ts = TurnScore(overall=5, dimensions=[
        ScoreDim("岗位理解", 5),
        ScoreDim("表达与逻辑", 1),
    ])
    assert ts.avg == 3.0
    assert ts.weak_dimensions == ["表达与逻辑"]
