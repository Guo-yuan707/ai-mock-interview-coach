"""CLI/入口层测试:子命令接线 + 离线整场面试能跑通(不碰真 API)。"""

from interview_coach import planner, report
from interview_coach.cli import SUBCOMMANDS, build_parser, main
from interview_coach.engine import InterviewSession
from interview_coach.eval.golden import GOLDEN_JD, GOLDEN_RESUME
from interview_coach.fake import make_offline_interview_client
from interview_coach.storage import InterviewStore
from interview_coach.config import PROJECT_ROOT


def test_parser_exposes_all_subcommands():
    p = build_parser()
    assert SUBCOMMANDS == {"plan", "interview", "history", "replay", "eval"}
    # 每个子命令都挂了一个可执行的 func(replay 需要 session_id 位置参数)
    for name in SUBCOMMANDS:
        argv = [name] + (["dummy"] if name == "replay" else [])
        ns = p.parse_args(argv)
        assert getattr(ns, "func", None) is not None, name


def test_bare_main_defaults_to_plan(capsys):
    """裸跑 main.py(无参数)→ plan 预览:确定性出题、免 key、有高危点。"""
    code = main([])
    out = capsys.readouterr().out
    assert code == 0
    assert "评分维度表" in out
    assert "高危追问点" in out
    assert "deterministic" in out          # 默认是确定性版,没有偷偷花 key


def test_main_plan_with_example_jd_arg(capsys):
    """main.py <示例JD> → 仍当 plan 预览(第 1 课习惯保留)。"""
    code = main([str(PROJECT_ROOT / "examples" / "sample-jd.txt")])
    assert code == 0
    assert "高危追问点" in capsys.readouterr().out


def test_offline_interview_end_to_end(tmp_path):
    """离线(脚本替身)能完整演一场:问→答→评分→收尾→落库→导出报告。"""
    plan = planner.build_plan(GOLDEN_JD, GOLDEN_RESUME)
    llm = make_offline_interview_client(plan)
    sess = InterviewSession(jd=GOLDEN_JD, resume=GOLDEN_RESUME,
                            plan=plan, llm=llm)

    while (turn := sess.ask_next()) is not None:
        score = sess.submit(f"第 {turn.seq} 轮的回答:我用了具体做法并给了量化结果。")
        assert score is not None and score.overall == 6   # 离线固定分
    summary = sess.summarize()
    assert sess.state.finished
    assert summary["turns"] >= 1
    assert summary["average_overall"] == 6.0
    assert summary["coverage"]["asked"] >= 1

    # 落库再读回,轮次与均分一致(第 9 课持久化闭环)
    db = tmp_path / "t.db"
    store = InterviewStore(db)
    try:
        sid = store.save_run(sess.state, summary)
        loaded = store.load_session(sid)
    finally:
        store.close()
    assert len(loaded["turns"]) == summary["turns"]
    assert loaded["turns"][0]["question"]
    assert loaded["turns"][0]["score"]["overall"] == 6

    # 报告能导出成 .md(不覆盖、可读)
    out = report.export_report(sess.state, summary, tmp_path / "reports")
    text = out.read_text(encoding="utf-8")
    assert "AI 模拟面试结案报告" in text
    assert "高危覆盖" in text
