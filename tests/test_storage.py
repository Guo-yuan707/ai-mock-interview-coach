"""存储与报告测试:SQLite 落库/回放/导出(用临时库,不碰真实数据库)。"""

import json

from interview_coach import report
from interview_coach.engine import InterviewSession
from interview_coach.fake import FakeClient
from interview_coach.plan import InterviewPlan, RiskPoint, RubricDimension, plan_from_dict, plan_to_dict
from interview_coach.storage import InterviewStore

PLAN = InterviewPlan(
    dimensions=[RubricDimension("岗位理解与匹配", "是否读懂 JD"),
                RubricDimension("技术深度(RAG)", "JD 点明 RAG")],
    risk_points=[RiskPoint("jd_probe", "讲讲 RAG", "检索细节", "RAG")],
)
SCORE = {
    "overall": 7,
    "dimensions": [{"dimension": "岗位理解与匹配", "score": 4, "comment": "还行", "gap": ""},
                   {"dimension": "技术深度(RAG)", "score": 2, "comment": "缺一手细节", "gap": "没证据"}],
    "strengths": ["清晰"], "weaknesses": ["没细节"], "verdict": "中上偏弱",
}


def _finished_session(tmp_path):
    """造一场跑完的假面试(1 轮问答 + 收尾)。"""
    def responder(kind, model, messages, jm):
        if kind == "decision":
            return json.dumps({"intent": "ask", "risk_id": "r0"}, ensure_ascii=False)
        if kind == "question":
            return "讲讲你的 RAG 一手细节?"
        if kind == "score":
            return json.dumps(SCORE, ensure_ascii=False)
        return ""
    fake = FakeClient(responder=responder)
    sess = InterviewSession(jd="要会 RAG", resume="会一点", plan=PLAN, llm=fake)
    sess.ask_next()
    sess.submit("我做过 RAG,但主要是调用现成 API。")
    sess.end_early("测试结束")
    return sess


def test_plan_json_roundtrip():
    d = plan_to_dict(PLAN)
    again = plan_from_dict(d)
    assert [x.name for x in again.dimensions] == ["岗位理解与匹配", "技术深度(RAG)"]
    assert again.risk_points[0].title == "讲讲 RAG"


def test_store_save_and_replay(tmp_path):
    sess = _finished_session(tmp_path)
    summary = sess.summarize()
    store = InterviewStore(tmp_path / "t.db")

    sid = store.save_run(sess.state, summary)
    assert sid == sess.state.session_id

    listing = store.list_sessions()
    assert listing and listing[0]["id"] == sid
    assert listing[0]["n_turns"] == 1

    loaded = store.load_session(sid)
    assert loaded is not None
    assert loaded["plan"].risk_points[0].title == "讲讲 RAG"
    assert loaded["turns"][0]["answer"].startswith("我做过 RAG")
    assert loaded["turns"][0]["score"]["overall"] == 7
    store.close()


def test_report_export(tmp_path):
    sess = _finished_session(tmp_path)
    summary = sess.summarize()
    out = report.export_report(sess.state, summary, output_dir=tmp_path / "out")
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "结案报告" in text
    assert "讲讲你的 RAG 一手细节?" in text     # 逐轮回放在报告里
    assert "最该补的短板" in text
