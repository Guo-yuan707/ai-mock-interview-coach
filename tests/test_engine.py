"""会话引擎集成测试:假模型驱动"问→答→评分→收尾"整条链路(不碰网络)。"""

import json

from interview_coach.engine import InterviewSession, SessionConfig
from interview_coach.fake import FakeClient
from interview_coach.llm import LMCall
from interview_coach.plan import InterviewPlan, RiskPoint, RubricDimension

PLAN = InterviewPlan(
    dimensions=[
        RubricDimension("岗位理解与匹配", "是否读懂 JD"),
        RubricDimension("技术深度(RAG)", "JD 点明 RAG"),
        RubricDimension("表达与逻辑", "结构清晰"),
    ],
    risk_points=[
        RiskPoint("jd_probe", "讲讲 RAG 检索", "会追问检索流程与失败场景", "RAG"),
        RiskPoint("unevidenced_claim", "这句没证据", "提升多少", "提升团队效率"),
        RiskPoint("resume_probe", "STAR 展开", "一手细节", ""),
        RiskPoint("missing_jd_keyword", "简历未提向量库", "了解吗", "向量库"),
        RiskPoint("jd_probe", "讲讲多轮对话", "上下文管理", "多轮对话"),
    ],
)

SCORE_JSON = {
    "overall": 6,
    "dimensions": [
        {"dimension": "岗位理解与匹配", "score": 3, "comment": "一般", "gap": ""},
        {"dimension": "技术深度(RAG)", "score": 2, "comment": "只讲概念", "gap": "没给一手细节,会被追问穿"},
        {"dimension": "表达与逻辑", "score": 4, "comment": "清楚", "gap": ""},
    ],
    "strengths": ["有结构"],
    "weaknesses": ["没数字"],
    "verdict": "偏弱",
}


def _make_fake(script: list) -> tuple[FakeClient, dict]:
    """script: 决策轮依次返回的内容(list of dict 或 LMCall),供 responder 取用。"""
    state = {"i": 0}

    def responder(kind, model, messages, jm):
        if kind == "decision":
            idx = state["i"]
            state["i"] += 1
            item = script[idx] if idx < len(script) else {"intent": "finish", "note": "兜底结束"}
            if isinstance(item, LMCall):
                return item
            return json.dumps(item, ensure_ascii=False)
        if kind == "question":
            return "你简历里写做过 RAG——具体讲讲检索流程,以及你踩过什么坑?"
        if kind == "score":
            return json.dumps(SCORE_JSON, ensure_ascii=False)
        return ""

    return FakeClient(responder=responder), state


def test_full_session_one_round_then_finish():
    """一问→一答→一评分→面试官主动收尾,状态全对。"""
    fake, _ = _make_fake([
        {"intent": "ask", "risk_id": "r0", "follow_up": "追检索细节"},   # 第一轮问 r0
        {"intent": "finish", "note": "高危覆盖已充分"},                   # 第二轮收尾
    ])
    sess = InterviewSession(jd="要做 RAG 问答", resume="熟练 Python", plan=PLAN,
                            cfg=SessionConfig(model="deepseek-v4-flash"), llm=fake)

    q = sess.ask_next()
    assert q is not None and q.question.startswith("你简历里写做过 RAG")
    assert q.risk_id == "r0"
    assert not sess.state.finished

    score = sess.submit("我用 chromadb 做检索,先召回再精排,遇到过长文档会切块。")
    assert score is not None
    assert score.overall == 6
    assert sess.state.turns and len(sess.state.turns) == 1
    assert sess.state.coverage()["asked"] == 1

    # 第二轮:面试官选择收尾 → ask_next 返回 None,状态 finished
    nxt = sess.ask_next()
    assert nxt is None
    assert sess.state.finished
    summary = sess.summarize()
    assert summary["turns"] == 1
    assert summary["average_overall"] == 6.0
    assert "技术深度(RAG)" in summary["weak_dimensions"]


def test_interviewer_retrieves_knowledge_then_asks():
    """决策第一步模型调 retrieve_knowledge 工具 → 引擎执行并把原文存进 last_grounding。"""
    tool_call = LMCall(finish_reason="tool_calls", tool_calls=[
        {"id": "call_1", "name": "retrieve_knowledge", "arguments": '{"query": "RAG"}'},
    ])
    fake, _ = _make_fake([tool_call, {"intent": "ask", "risk_id": "r0"}])
    sess = InterviewSession(jd="负责基于大模型做 RAG 问答", resume="熟练 Python",
                            plan=PLAN, llm=fake)
    q = sess.ask_next()
    assert q is not None
    assert "retrieve_knowledge" in q.tool_calls        # 记录:面试官真调了工具
    assert sess.state.last_grounding.strip()           # 检索到的原文被留下来了


def test_max_turns_stops_interview():
    """达到轮次上限 → 引擎自动收尾,不无限问下去(成本红线)。"""
    fake, _ = _make_fake([{"intent": "ask", "risk_id": "r0"}] * 2)
    sess = InterviewSession(jd="x", resume="y", plan=PLAN,
                            cfg=SessionConfig(model="m", max_turns=2), llm=fake)
    assert sess.ask_next() is not None
    sess.submit("答1")
    assert sess.ask_next() is not None
    sess.submit("答2")
    assert sess.ask_next() is None                 # 第 3 次问 → 已达上限,自动结束
    assert sess.state.finished
    assert "上限" in sess.state.finish_reason


def test_end_early_by_caller():
    fake, _ = _make_fake([{"intent": "ask", "risk_id": "r0"}])
    sess = InterviewSession(jd="x", resume="y", plan=PLAN, llm=fake)
    sess.ask_next()
    sess.end_early("候选人主动结束")
    assert sess.state.finished
    assert sess.ask_next() is None
