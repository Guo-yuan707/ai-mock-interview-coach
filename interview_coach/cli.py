"""命令行入口(第 1 课骨架 → 第 9 课全流程):plan / interview / history / replay / eval。

设计原则:所有"业务逻辑"都在 engine / planner / report / storage 里,
本文件只做三件事 —— 接参数、把人话翻译成对核心的调用、把结果打给人看。
所以网页版(Streamlit)复用的是同一套核心,不是重写一遍。

子命令:
    plan        打印面试计划预览(第 1 课验收面板,默认确定性出题、免 key、可复现)
    interview   跑一场完整模拟面试(在线=真 AI;--offline=脚本替身,无 key 也能演示)
    history     列出历史面试(SQLite)
    replay      回放某一场的完整问答与评分
    eval        一键评测(默认离线,免 key)

兼容旧习惯:裸跑 `python main.py` / `python main.py 某JD` → 当 plan 预览。
"""
from __future__ import annotations

import argparse
import sys

from . import config, planner, report, storage
from .engine import InterviewSession, SessionConfig
from .fake import make_offline_interview_client
from .llm import LLMError, reset_usage
from .loader import read_text
from .plan import InterviewPlan

SUBCOMMANDS = {"plan", "interview", "history", "replay", "eval"}
SEP = "─" * 62


def _fix_console() -> None:
    """Windows 终端默认 GBK,强制 UTF-8 避免中文乱码。

    用 hasattr 兜底:在 pytest 里 stdout 是捕获对象,没有 reconfigure → 跳过即可。
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                pass


# =====================================================================
# plan:打印面试计划预览(第 1 课验收面板)
# =====================================================================

def _plan_preview(plan: InterviewPlan, mode: str) -> None:
    print(SEP)
    print(f"🎯 岗位方向提示:{plan.target_hint or '(暂无法判断)'}")
    print(f"📋 评分维度表({plan.n_dimensions} 维,评卷人按这些打分) —— 出题模式:{mode}")
    for i, d in enumerate(plan.dimensions, 1):
        print(f"  {i}. {d.name}\n     缘起:{d.why}")
    print(SEP)
    print(f"⚠️  高危追问点({plan.n_risks} 条,面试官最可能在这追穿):")
    for i, rp in enumerate(plan.risk_points, 1):
        print(f"  {i}. [{rp.kind}] {rp.title}")
        print(f"     {rp.detail}")
        if rp.evidence:
            print(f"     依据原文:「{rp.evidence}」")
    print(SEP)


def _default_paths(args) -> tuple[str, str | None]:
    """决定默认 JD / 简历路径:examples 里的合成样例,不碰真实个人资料(红线)。"""
    jd_path = args.jd or str(config.PROJECT_ROOT / "examples" / "sample-jd.txt")
    resume_path = None
    if not args.no_resume:
        resume_path = args.resume or str(config.PROJECT_ROOT / "examples" / "sample-resume.txt")
    return jd_path, resume_path


def cmd_plan(args) -> int:
    jd_path, resume_path = _default_paths(args)
    print(f"读取 JD  :{jd_path}")
    print(f"读取简历 :{resume_path or '(未提供,纯 JD 出题)'}\n")
    # plan 预览默认确定性版(快、免 key、稳定);要 AI 出题才 --ai
    use_ai = bool(args.ai)
    plan, mode = planner.build_plan_auto(jd_path, resume_path, use_ai=use_ai)
    _plan_preview(plan, mode)
    return 0


# =====================================================================
# interview:跑一场完整模拟面试(交互)
# =====================================================================

def cmd_interview(args) -> int:
    jd_path, resume_path = _default_paths(args)
    print(f"读取 JD  :{jd_path}")
    print(f"读取简历 :{resume_path or '(未提供,纯 JD 出题)'}")

    if args.offline:
        print("⚠️  offline 演示模式:面试官/评卷人由脚本替身扮演,结果非真实。")
        plan, plan_mode = planner.build_plan_auto(jd_path, resume_path, use_ai=False)
        llm = make_offline_interview_client(plan)
    else:
        # 在线:计划自动(有 key 就 AI 出题,失败/没 key 自动降级确定性),问答走真模型
        plan, plan_mode = planner.build_plan_auto(jd_path, resume_path, use_ai=None)
        llm = None
        print(f"✅ 在线模式:模型 {args.model} · 计划来源 {plan_mode}")

    cfg = SessionConfig(model=args.model, max_turns=args.max_turns)
    reset_usage()                       # 只把这之后(面试进行中)的调用记进本场账单
    sess = InterviewSession(jd=read_text(jd_path),
                            resume=read_text(resume_path) if resume_path else None,
                            plan=plan, cfg=cfg, llm=llm)
    print("(输入 /end 结束;其他内容即你的回答)\n")
    _plan_preview(plan, plan_mode)

    def stream(q: str) -> None:
        print(q, end="", flush=True)   # 打字机效果:边收边打,不换行

    print("\n" + SEP)
    try:
        while (turn := sess.ask_next(on_delta=stream)) is not None:
            print()   # 问题结束,换行
            if turn.tool_calls:
                from .tools import format_tool_result_for_show
                mark = format_tool_result_for_show(turn.tool_calls, sess.state)
                if mark:
                    print(f"  ↳ {mark}")
            print(f"\n[第 {turn.seq} 轮 你答] ", end="")
            raw = _read_answer()
            if raw is None:
                print("(输入中断,结束面试)")
                sess.end_early("候选人中断输入")
                break
            if raw.strip().lower() in ("/end", "/quit", "/exit"):
                print("(你主动结束面试)")
                sess.end_early("候选人主动结束")
                break
            score = sess.submit(raw)
            if score is not None:
                _show_score(score)
            print(SEP)
            if sess.state.finished:
                break
    except LLMError as e:
        print(f"\n❌ 调用大模型失败:{e}")
        print("小提示:没配 API key 时,可用 `--offline` 演示全流程(不出网、不花钱)。")
        return 2

    _show_summary_and_save(sess, args)
    return 0


def _read_answer() -> str | None:
    """读一行回答;中断(EOF/Ctrl-C)返回 None。"""
    try:
        return input()
    except (EOFError, KeyboardInterrupt):
        return None


def _show_score(score) -> None:
    dim = " · ".join(f"{d.dimension}:{d.score}/5" for d in score.dimensions)
    print(f"📊 本题评分:{score.overall}/10 ({dim})")
    if any(d.gap for d in score.dimensions):
        print("   ⚠️ 评卷人指出会被追问穿:" + " | ".join(d.gap for d in score.dimensions if d.gap))
    if score.verdict:
        print(f"   💬 {score.verdict}")


def _show_summary_and_save(sess: InterviewSession, args) -> None:
    summary = sess.summarize()
    if args.export:
        path = report.export_report(sess.state, summary)
        print(f"\n📄 报告已导出:{path}")
    if args.save:
        store = storage.InterviewStore()
        try:
            sid = store.save_run(sess.state, summary)
        finally:
            store.close()
        print(f"💾 已存入历史库(session id:{sid}),可用 `python main.py replay {sid}` 回放")
    print(SEP)
    print("🏁 面试结束 | " + summary["finish_reason"])
    print(f"轮次 {summary['turns']} · 场均 {summary['average_overall']}/10 · "
          f"高危覆盖 {summary['coverage']['asked']}/{summary['coverage']['total']}")
    if summary.get("dimension_avg"):
        print("维度均分:" + " · ".join(f"{k} {v}/5" for k, v in summary["dimension_avg"].items()))


# =====================================================================
# history / replay:历史与会话回放(第 9 课持久化)
# =====================================================================

def cmd_history(_args) -> int:
    store = storage.InterviewStore()
    try:
        rows = store.list_sessions()
    finally:
        store.close()
    if not rows:
        print("暂无历史记录。跑一场面试并加 --save 后会出现在这里。")
        return 0
    print(f"{'ID':<14}{'时间':<18}{'轮次':<5}{'均分':<6}{'模式':<8}结束原因")
    for r in rows:
        avg = f"{r['avg']:.1f}" if r["avg"] is not None else "-"
        why = (r["finish_reason"] or "")[:22]
        print(f"{r['id']:<14}{storage.format_timestamp(r['created_at']):<18}"
              f"{r['n_turns']:<5}{avg:<6}{r['planner_mode']:<8}{why}")
    return 0


def cmd_replay(args) -> int:
    store = storage.InterviewStore()
    try:
        sess = store.load_session(args.session_id)
    finally:
        store.close()
    if sess is None:
        print(f"找不到会话 {args.session_id},先用 `history` 看有哪些。")
        return 1
    meta = sess["meta"]
    print(SEP)
    print(f"会话 {meta['id']} · 计划:{meta.get('planner_mode', '-')} · "
          f"模型:{meta.get('model') or '-'} · 结束:{meta.get('finish_reason') or '-'}")
    print(f"评分维度:共 {len(sess['plan'].dimensions)} 维;高危点:{len(sess['plan'].risk_points)} 条\n")
    for t in sess["turns"]:
        print(f"\n▶ 第 {t['seq']} 轮 · 高危点 {t['risk_id'] or '自由追问'}")
        print(f"  面试官问:{t['question']}")
        print(f"  你答:{t['answer']}")
        sc = t.get("score") or {}
        if sc.get("overall") is not None:
            print(f"  评分:{sc['overall']}/10")
            for d in sc.get("dimensions", []):
                gap = f" ⚠️{d['gap']}" if d.get("gap") else ""
                print(f"    - {d['dimension']}:{d['score']}/5 {d.get('comment', '')}{gap}")
    return 0


# =====================================================================
# eval:一键评测(第 8 课体检)
# =====================================================================

def _cmd_eval(args) -> int:
    from .eval.run import main as eval_main
    eval_main(["--live"] if args.live else [])
    return 0


# =====================================================================
# 顶层 parser
# =====================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="interview-coach", description="AI 模拟面试官 · 命令行")
    sub = p.add_subparsers(dest="cmd")

    p_plan = sub.add_parser("plan", help="打印面试计划(第 1 课面板)")
    p_plan.add_argument("jd", nargs="?", default=None, help="JD 文本/文件(缺省用示例)")
    p_plan.add_argument("--resume", default=None)
    p_plan.add_argument("--no-resume", action="store_true")
    p_plan.add_argument("--ai", action="store_true", help="用 AI 出题(默认确定性版,免 key)")
    p_plan.set_defaults(func=cmd_plan)

    p_iv = sub.add_parser("interview", help="跑一场完整模拟面试")
    p_iv.add_argument("jd", nargs="?", default=None)
    p_iv.add_argument("--resume", default=None)
    p_iv.add_argument("--no-resume", action="store_true")
    p_iv.add_argument("--ai", action="store_true", help="强制 AI 出题(默认自动:有 key 就 AI)")
    p_iv.add_argument("--offline", action="store_true", help="无 key 演示:脚本替身跑全流程")
    p_iv.add_argument("--model", default=config.DEFAULT_MODEL, choices=config.MODELS)
    p_iv.add_argument("--max-turns", type=int, default=config.MAX_INTERVIEW_TURNS)
    p_iv.add_argument("--save", action="store_true", help="结束落 SQLite 历史库")
    p_iv.add_argument("--export", action="store_true", help="结束导出 .md 报告到 output/")
    p_iv.set_defaults(func=cmd_interview)

    p_h = sub.add_parser("history", help="列出历史面试")
    p_h.set_defaults(func=cmd_history)

    p_r = sub.add_parser("replay", help="回放某一场的完整问答与评分")
    p_r.add_argument("session_id")
    p_r.set_defaults(func=cmd_replay)

    p_e = sub.add_parser("eval", help="一键评测(默认离线/免 key)")
    p_e.add_argument("--live", action="store_true", help="用真模型跑(烧 token)")
    p_e.set_defaults(func=_cmd_eval)
    return p


def main(argv: list[str] | None = None) -> int:
    _fix_console()
    raw = list(argv) if argv is not None else sys.argv[1:]
    # 兼容第 1 课习惯:不给子命令 / 直接给 JD → 默认当 plan 预览(确定性、免 key)
    if not raw or raw[0] not in SUBCOMMANDS:
        raw = ["plan"] + raw
    parser = build_parser()
    args = parser.parse_args(raw)
    return args.func(args)
