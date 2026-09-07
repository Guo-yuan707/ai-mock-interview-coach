"""网页入口(第 2 课起强调的"网页为主"):Streamlit 起一个面试间。

复用的是与 CLI 完全相同的核心(engine / planner / evaluator / storage / report),
这里只做界面编排 —— 双入口意味着核心逻辑只有一份,不会两处越写越偏。

运行:
    .venv/Scripts/python -m streamlit run app.py
然后浏览器打开 http://localhost:8501

页面:
    ① 模拟面试   在线(真 AI+真评分)/ 离线(免 key 脚本演示)
    ② 历史回放   从 SQLite 挑一场,整场问答逐字回放 + 导出报告
    ③ 一键评测   离线跑(免 key);也可用访客自己的 key 跑真模型

在线模式 = BYOK:每个访客在左侧填自己的 DeepSeek key,只存本会话内存(st.session_state)、
不写盘/不入库/不进 git;服务端不持任何 key。Streamlit 单进程多访客,
必须给每人 new 一个 LLMClient 显式传下去(见 _load_user_client),绝不写 llm 模块的全局单例。

红线提醒(顶部常驻):真实简历/JD 只是临时粘贴进会话,不会写入任何文件/代码库;
示例一律用 examples/ 里的合成样例。
"""
from __future__ import annotations

import streamlit as st

from interview_coach import config, planner, report, storage
from interview_coach.engine import InterviewSession, SessionConfig
from interview_coach.eval.run import run_all
from interview_coach.fake import make_offline_interview_client
from interview_coach.llm import LLMClient, LLMError, reset_usage
from interview_coach.loader import read_text

st.set_page_config(page_title="AI 模拟面试官", page_icon="🎙️", layout="wide")

SAMPLE_JD = str(config.PROJECT_ROOT / "examples" / "sample-jd.txt")
SAMPLE_RESUME = str(config.PROJECT_ROOT / "examples" / "sample-resume.txt")


# =====================================================================
# 小工具
# =====================================================================

def _reset(key: str) -> None:
    st.session_state.pop(key, None)


# =====================================================================
# BYOK(自带 key):每个访客在侧边栏填自己的 DeepSeek key
# =====================================================================

def _has_user_key() -> bool:
    """当前访客是否已在侧边栏填了 key(只存会话内存,不落盘/不进库)。"""
    return bool((st.session_state.get("user_api_key") or "").strip())


def _load_user_client() -> LLMClient | None:
    """用访客自己填的 key 构造一个独立真客户端;没填返回 None。

    关键:绝不写 llm 模块的全局单例 —— Streamlit 单进程多访客,
    若沿用全局 get_client(),B 访客会用到 A 访客的 key(串号/烧别人的钱)。
    这里每人 new 一个 LLMClient,并从入口一路显式传下去(plan→session→评分),
    公共部署下也天然隔离。
    """
    if not _has_user_key():
        return None
    return LLMClient(api_key=st.session_state.user_api_key.strip())


def _score_markdown(sc) -> str:
    """把一道题的 TurnScore 压成一段评卷人可读的 markdown。"""
    lines = [f"**本题 {sc.overall}/10**"]
    for d in sc.dimensions:
        flag = f"　⚠️{d.gap}" if d.gap else ""
        lines.append(f"- {d.dimension}:{d.score}/5　{d.comment}{flag}")
    if sc.strengths:
        lines.append("👍 " + "；".join(sc.strengths))
    if sc.weaknesses:
        lines.append("👎 " + "；".join(sc.weaknesses))
    if sc.verdict:
        lines.append(f"_评语:{sc.verdict}_")
    return "\n\n".join(lines)


def _render_chat(S: dict) -> None:
    """把会话记录重放成聊天样式(每次 rerun 全量画一遍)。"""
    for e in S["messages"]:
        if e["role"] == "user":
            with st.chat_message("user"):
                st.markdown(e["text"])
        elif e.get("score"):
            with st.chat_message("assistant"):
                st.markdown("🧑‍⚖️ **评卷人**")
                st.markdown(e["text"])
        else:
            with st.chat_message("assistant"):
                st.markdown(e["text"])


def _show_plan(S: dict) -> None:
    """开场的计划小抄:评分维度 + 高危点(全部 grounded 在原文,红线)。"""
    plan = S["plan"]
    with st.expander(f"📋 本场面试计划({plan.n_dimensions} 维评分 · {plan.n_risks} 个高危点)",
                     expanded=True):
        st.caption(f"计划模式:{S['planner_mode']} · 模型:{S['model']}")
        dims = "　".join(f"**{d.name}**" for d in plan.dimensions)
        st.markdown("**评分维度(评卷人只按这些打分):**\n\n" + dims)
        for i, rp in enumerate(plan.risk_points, 1):
            ev = f"　`依据:{rp.evidence}`" if rp.evidence else ""
            st.markdown(f"{i}. [{rp.kind}] **{rp.title}**{ev}\n　{rp.detail}")


# =====================================================================
# 页面 ① :模拟面试
# =====================================================================

def _run_interview_page() -> None:
    st.header("🎙️ 模拟面试间")
    st.caption("面试官按计划多轮追问 → 评卷人逐题结构化打分 → 结束出结案报告。"
               "红线:问题与评分只依据 简历/JD 原文或高危点清单,绝不编你的经历。")
    st.session_state.setdefault("mic", {"started": False})

    if not st.session_state.mic["started"]:
        _render_start()
        return

    S = st.session_state.mic
    sess = S["session"]

    col_m, col_b, col_c = st.columns([1, 1, 4])
    with col_m:
        st.metric("已进行轮次", len(sess.state.turns))
    cov = sess.state.coverage()
    with col_b:
        st.metric("高危覆盖", f"{cov['asked']}/{cov['total']}")
    with col_c:
        if st.button("🔚 结束本场(直接总结)", key="mic_stop"):
            sess.end_early("用户主动结束")
            S["need_question"] = False
            S["awaiting"] = False
            S["finished"] = True

    # ---- 主状态机:来一答 → 接住打分 → 立刻出下一问;或开场先出第一问 ----
    awaiting = S.get("awaiting", False)
    finished = S.get("finished", False)
    ans = st.chat_input("输入你的回答,回车发送(仅文字)", key="mic_box",
                        disabled=bool(finished))
    if awaiting and ans:
        sc = sess.submit(ans)
        S["messages"].append({"role": "user", "text": ans})
        if sc is not None:
            S["messages"].append({"role": "assistant", "text": _score_markdown(sc), "score": True})
        S["awaiting"] = False
        S["need_question"] = True

    if S.get("need_question") and not S.get("finished"):
        S["need_question"] = False
        ph = st.empty()
        try:
            turn = sess.ask_next(on_delta=lambda c: ph.markdown("✍️ " + c))
        except LLMError as e:
            S["awaiting"] = False
            S["finished"] = True
            S["error"] = str(e)
            turn = None
        if turn is None:
            S["awaiting"] = False
            S["finished"] = True
        else:
            ph.empty()
            note = ""
            if turn.tool_calls:
                from interview_coach.tools import format_tool_result_for_show
                note = format_tool_result_for_show(turn.tool_calls, sess.state)
                note = ("\n\n_📎 " + note + "_") if note else ""
            rid = turn.risk_id or "自由追问"
            S["messages"].append({
                "role": "assistant", "text": f"**面试官 · 第 {turn.seq} 轮·高危点 {rid}**\n\n{turn.question}{note}",
            })
            S["awaiting"] = True
        # 结束时立刻算好总结,好让下方面板有数据
        if S.get("finished"):
            S["summary"] = sess.summarize()

    _render_chat(S)

    if S.get("finished"):
        _render_finish(S)


def _render_start() -> None:
    S = st.session_state.mic
    st.markdown("### 开始一场模拟面试")
    mode = st.radio("运行模式", ["🧪 离线演示(免 key,脚本替身跑全流程)",
                                 "🌐 在线(真 AI 追问 + 真评分,需你在左侧填自己的 key)"])
    offline = mode.startswith("🧪")
    model = st.selectbox("模型(在线模式才生效)", config.MODELS,
                         index=config.MODELS.index(config.DEFAULT_MODEL))
    max_turns = st.slider("一场最多轮数", 4, 20, config.MAX_INTERVIEW_TURNS)

    use_examples = st.checkbox("用内置合成样例(推荐:不碰真实资料)", value=True)
    if use_examples:
        jd = read_text(SAMPLE_JD)
        resume = read_text(SAMPLE_RESUME)
    else:
        jd = st.text_area("岗位 JD(可贴全文,或粘贴文件路径)", height=180)
        resume = st.text_area("候选人简历(可选;纯 JD 也开得了场)", height=180)

    st.caption("真实简历/JD 只会停留在本次会话内存里:不落盘、不进 git、不用于其它场次。")
    if not (jd or resume):
        st.warning("请至少给一份 JD。")
        return
    if st.button("🎬 开始面试", type="primary"):
        if offline:
            reset_usage()
            with st.spinner("生成面试计划(确定性版,免 key)…"):
                plan, plan_mode = planner.build_plan_auto(jd, resume, use_ai=False)
                llm = make_offline_interview_client(plan)
        else:
            llm = _load_user_client()
            if llm is None:
                st.error("🌐 在线模式需要你自己的 DeepSeek API Key。\n"
                         "请在左侧边栏「🔑 在线模式:你的 API Key」填入"
                         "(消耗你自己的额度);或改用上面的 🧪 离线演示(免 key)。")
                return
            with st.spinner("生成面试计划(AI 出题;失败自动降级确定性)…"):
                plan, plan_mode = planner.build_plan_auto(jd, resume, use_ai=True, llm=llm)
            reset_usage()   # 计划生成结束后再归零,账单只记"面试进行中"
        cfg = SessionConfig(model=model, max_turns=max_turns)
        sess = InterviewSession(jd=jd, resume=resume or None,
                                plan=plan, cfg=cfg, llm=llm)
        S.update(started=True, session=sess, plan=plan, planner_mode=plan_mode,
                 model=model, messages=[], awaiting=False, need_question=True,
                 finished=False, summary=None, offline=offline)
        if offline:
            st.toast("离线演示:面试官/评卷人是脚本替身,内容非真实,看的是流程。")
        _show_plan(S)


def _render_finish(S: dict) -> None:
    summary = S.get("summary") or S["session"].summarize()
    if S.get("error"):
        st.error("❌ 调用大模型失败:" + S["error"] + "\n\n小提示:检查左侧填的 key 是否正确/是否有余额;"
                 "只想看流程可切回 🧪 离线演示。")
    st.divider()
    st.subheader("🏁 面试结束")
    st.info(summary["finish_reason"])
    c1, c2, c3 = st.columns(3)
    c1.metric("场均分", f"{summary['average_overall']}/10")
    c2.metric("轮次", summary["turns"])
    c3.metric("高危覆盖", f"{summary['coverage']['asked']}/{summary['coverage']['total']}")
    if summary.get("dimension_avg"):
        st.markdown("**各维度均分(满分 5):**")
        for name, avg in sorted(summary["dimension_avg"].items(), key=lambda x: -x[1]):
            st.markdown(f"- {name}:{avg}")
    if summary.get("weak_dimensions"):
        st.warning("评卷人标出的薄弱维度(该补):" + "、".join(summary["weak_dimensions"]))

    txt = report.build_report_text(S["session"].state, summary)
    st.download_button("⬇️ 下载结案报告(.md)", txt,
                       file_name=f"interview-report-{S['session'].state.session_id}.md", type="primary")
    if config.DEPLOYED_PUBLIC:
        st.caption("🔒 公网多访客模式已关闭『保存到历史库』:你的内容只留在本次会话,"
                   "不会写进共享服务器库。")
    elif st.button("💾 保存到历史库", key="mic_save"):
        store = storage.InterviewStore()
        try:
            sid = store.save_run(S["session"].state, summary)
        finally:
            store.close()
        st.success(f"已保存(session id:{sid}),可在「历史回放」页查看。")
    if st.button("🔄 另开一场"):
        _reset("mic")
        st.rerun()


# =====================================================================
# 页面 ② :历史回放
# =====================================================================

def _run_history_page() -> None:
    st.header("📚 历史回放")
    store = storage.InterviewStore()
    try:
        rows = store.list_sessions()
        if not rows:
            st.info("还没有保存过的面试。去「模拟面试」跑一场,结束时点『保存到历史库』。")
            return
        by_id = {r["id"]: r for r in rows}
        label = lambda r: (f"{r['id']} · {storage.format_timestamp(r['created_at'])} · "
                           f"{r['n_turns']} 轮 · 均分 {('%.1f' % r['avg']) if r['avg'] is not None else '-'} · "
                           f"{r['planner_mode']}")
        chosen = st.selectbox("选择一场", list(by_id), format_func=lambda i: label(by_id[i]))
        sess = store.load_session(chosen)
    finally:
        store.close()

    meta, plan, turns = sess["meta"], sess["plan"], sess["turns"]
    st.caption(f"计划:{meta.get('planner_mode', '-')} · 模型:{meta.get('model') or '-'} · "
               f"结束:{meta.get('finish_reason') or '-'}")
    st.markdown(f"**评分维度({len(plan.dimensions)}):** "
                + "　".join(f"`{d.name}`" for d in plan.dimensions))

    for t in turns:
        with st.chat_message("assistant"):
            st.markdown(f"**面试官 · 第 {t['seq']} 轮·高危点 {t['risk_id'] or '自由追问'}**\n\n{t['question']}")
        with st.chat_message("user"):
            st.markdown(t["answer"] or "_（未作答）_")
        sc = t.get("score") or {}
        if sc.get("overall") is not None:
            lines = [f"**本题 {sc['overall']}/10**"]
            for d in sc.get("dimensions", []):
                flag = f"　⚠️{d['gap']}" if d.get("gap") else ""
                lines.append(f"- {d['dimension']}:{d['score']}/5　{d.get('comment', '')}{flag}")
            with st.chat_message("assistant"):
                st.caption("\n".join(lines))

    if st.button("🗑️ 删除这场", key="hist_del"):
        store = storage.InterviewStore()
        try:
            store.delete_session(chosen)
        finally:
            store.close()
        st.success("已删除。")


# =====================================================================
# 页面 ③ :一键评测
# =====================================================================

def _run_eval_page() -> None:
    st.header("🧪 一键评测")
    st.caption("A 检索 recall@k · B 出题覆盖 · C 评分严格度。默认离线免 key;"
               "点下面的『真模型评测』才会烧 token。")
    if st.button("▶️ 跑离线评测(免 key)", type="primary"):
        with st.spinner("离线评测中…"):
            st.code(run_all("offline"), language="markdown")
    live_ready = _has_user_key()
    if st.button("🔥 跑真模型评测(用你的 key)", disabled=not live_ready):
        llm = _load_user_client()
        with st.spinner("真模型评测中(烧你 key 的额度),耐心等…"):
            st.code(run_all("live", llm=llm), language="markdown")
    if not live_ready:
        st.caption("真模型评测要真调 DeepSeek(烧 token):填了左侧「🔑 在线模式」的 key 才能点,"
                   "用你自己的额度。")


# =====================================================================
# 壳
# =====================================================================

def _render_api_settings() -> None:
    """侧边栏常驻:访客填自己的 DeepSeek key(BYOK)。

    只存在 session_state(浏览器会话内存),不写任何文件 / SQLite / git;
    服务端不留。key 仅在你本会话直连 DeepSeek 用,额度自己承担。
    """
    with st.sidebar.expander("🔑 在线模式:你的 API Key", expanded=False):
        st.text_input("DeepSeek API Key(在线问答/真评分用)",
                      type="password", key="user_api_key",
                      placeholder="sk-…",
                      help="填你自己的 key,只在本会话内用,不会写盘或入库。")
        if _has_user_key():
            st.success("已填 ✓(本次会话在线模式用它,烧你的额度)")
        else:
            st.caption("不填也能玩 🧪 离线演示;想用真 AI 请去"
                       "platform.deepseek.com 创建 key 后填这里。")


def main() -> None:
    st.sidebar.title("🎙️ AI 模拟面试官")
    st.sidebar.caption("JD 定义评分维度 → 多轮追问 → 结构化打分 → 结案报告")
    if config.DEPLOYED_PUBLIC:
        pages = ["① 模拟面试", "③ 一键评测"]   # 公网多访客:关共享历史库,防互看/互删
        note = "🔒 公网多访客:已隐藏共享历史库 · 真实资料不入库"
    else:
        pages = ["① 模拟面试", "② 历史回放", "③ 一键评测"]
        note = "红线:不编你的经历 · 测试不碰真 API · 真实资料不入库"
    page = st.sidebar.radio("页面", pages)
    _render_api_settings()
    st.sidebar.divider()
    st.sidebar.caption(note)
    if page.startswith("①"):
        _run_interview_page()
    elif page.startswith("②"):
        _run_history_page()
    else:
        _run_eval_page()


if __name__ == "__main__":
    main()
