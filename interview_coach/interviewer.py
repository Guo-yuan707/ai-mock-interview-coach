"""面试官 Agent(第 5/6 课):模型决定"下一步",不是 if-else 轮询。

两个动作,各司其职:
1. agent_decide:带工具的函数调用循环。
   模型看完现场(高危点清单+进度+最近问答),自己决定:
   调 retrieve_knowledge 查原文?调 check_progress 看进度?还是直接给"决策 JSON"
   (追哪条高危点 / 怎么追 / 或收尾)。—— 这是第 5 课 Function Calling 闭环 +
   第 6 课 Agent 决策循环(模型主导)。
2. phrase_question:流式把决策"说"成一个自然的问题(打字机效果 + TTFT)。
   —— 第 3 课流式输出。设计上"先决定、再措辞"分开:
   决策要稳(JSON、低温),措辞要自然(流式),互不拖累。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import config
from .llm import LMCall, get_client
from .plan import RiskPoint
from .schemas import DecisionSchema
from .state import SessionState, format_recent_history, format_risk_roster
from .structured import extract_json
from . import tools as T


@dataclass
class InterviewerTurn:
    """一次 agent_decide 的结论:要么提问(ask),要么收尾(finish)。"""
    kind: str                                  # "ask" | "finish"
    risk_id: str = ""                          # ask:要攻的高危点;finish:空
    follow_up: str = ""                        # 追问角度,给措辞当指示
    note: str = ""                             # 内部提示(不外露给用户)
    tool_calls_used: list[str] = field(default_factory=list)  # 这步真调了哪些工具
    reason: str = ""                           # finish 时的收尾理由


def _maybe_llm(llm):
    return llm if llm is not None else get_client()


# ---------------- 决策(带工具的函数调用循环) ----------------

def _build_decide_messages(state: SessionState) -> list[dict]:
    """把"现场"拼给面试官:高危点覆盖情况 + 最近几轮 + 上一答评分。"""
    last = state.last_turn()
    last_line = ""
    if last is not None and last.score is not None:
        sc = last.score
        last_line = (
            f"上一答(针对 {last.risk_id}「{last.question[:30]}…」)评分 {sc.overall}/10,"
            f"维度均值 {sc.avg},弱项:{'、'.join(sc.weak_dimensions) or '无'}。评语:{sc.verdict}"
        )
    return [
        {"role": "system", "content": (
            "你是模拟面试的 AI 面试官,正在按一份面试计划对候选人多轮追问、挖深度。"
            "规则:① 所有问题 grounded 在 简历/JD 原文或高危点清单,不许编候选人的事实;"
            "② 追问要有压迫感——专挑没给证据、没数字、没一手细节的说法;"
            "③ 你带了工具:信息不够想引原文就调 retrieve_knowledge;想确认进度就调 "
            "check_progress;觉得追问已充分就调 finish_interview。"
            "流程:先想这一步要不要查;要查就连调工具(可多次);"
            "信息够了【只输出一个决策 JSON,不要输出任何别的话】,格式:"
            '{"intent":"ask 或 finish","risk_id":"r加序号,ask 必填、finish 可空",'
            '"follow_up":"追问角度","note":"给措辞的内部提示","retrieve_query":""}'
        )},
        {"role": "user", "content": (
            "【本轮现场】\n"
            f"高危点清单(带是否已问):\n{format_risk_roster(state)}\n\n"
            f"最近对话摘要:\n{format_recent_history(state)}\n\n"
            f"{last_line}\n\n"
            "现在决定这一步怎么走。要么调工具先补信息,要么直接输出决策 JSON。"
        )},
    ]


def _parse_decision(text: str) -> DecisionSchema | None:
    """把模型回的 JSON 决策文字解析成强类型;坏了返回 None(上层有兜底)。"""
    try:
        return DecisionSchema.model_validate(extract_json(text))
    except Exception:
        return None


def agent_decide(llm, model: str, state: SessionState, *, kb=None,
                 tools_specs: list[dict] | None = None,
                 max_rounds: int = 4) -> InterviewerTurn:
    """让面试官走一步:可调工具(闭环)→ 最后给决策。返回 InterviewerTurn。"""
    client = _maybe_llm(llm)
    specs = tools_specs if tools_specs is not None else T.build_tool_specs()
    messages = _build_decide_messages(state)
    ctx = T.ToolContext(state=state, kb=kb)
    used: list[str] = []

    for _ in range(max_rounds):
        call = client.chat_with_tools(model=model, messages=messages, tools=specs,
                                      max_tokens=config.MAX_TOKENS_DECISION, kind="decision")
        # 情况一:模型要调工具
        if call.tool_calls:
            # 先把它这轮的 assistant 消息整体记回对话(含 reasoning_content——思考模式下
            # DeepSeek 要求原样带回,漏了会 400),再逐条喂工具结果,形成闭环。
            if call.assistant_message:
                messages.append(call.assistant_message)
            else:
                # Fake 等测试替身没有 assistant_message → 自己拼一份(多个并行调用合成一条)
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": tc["id"], "type": "function",
                         "function": {"name": tc["name"], "arguments": tc.get("arguments", "")}}
                        for tc in call.tool_calls
                    ],
                })
            for tc in call.tool_calls:
                name = tc["name"]
                args = T.parse_arguments(tc.get("arguments", ""))
                res = T.run_tool(name, args, ctx)
                used.append(name)
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": res.output})
                if name == T.TOOL_RETRIEVE and res.ok:
                    state.last_grounding = res.output      # 措辞阶段可引用刚查到的原文
                if res.is_finish:
                    return InterviewerTurn(kind="finish", reason=state.finish_reason,
                                           tool_calls_used=used)
            continue  # 执行完工具,让模型带着结果再走一轮

        # 情况二:模型直接给文字 → 解析成决策
        decision = _parse_decision(call.text)
        if decision is not None:
            if decision.intent == "finish":
                return InterviewerTurn(kind="finish", reason=decision.note or decision.follow_up,
                                       tool_calls_used=used)
            if decision.intent == "retrieve":   # 兜底:JSON 说要查但没走工具 → 代查一次再续
                res = T.run_tool(T.TOOL_RETRIEVE, {"query": decision.retrieve_query or "整体情况"}, ctx)
                used.append(T.TOOL_RETRIEVE)
                state.last_grounding = res.output
                messages.append({"role": "user",
                                 "content": f"(自动补充)知识库检索结果:\n{res.output}\n请继续给出决策 JSON。"})
                continue
            # intent == ask
            return InterviewerTurn(kind="ask", risk_id=decision.risk_id,
                                   follow_up=decision.follow_up, note=decision.note,
                                   tool_calls_used=used)
        # 解析失败:不强撑,交给引擎的确定性兜底(engine 会给个安全的下一个问题)
        return InterviewerTurn(kind="ask", risk_id="", follow_up="", note="", tool_calls_used=used)

    # 工具轮次用尽仍没给出决策 → 同上,兜底
    return InterviewerTurn(kind="ask", risk_id="", tool_calls_used=used)


# ---------------- 措辞(流式把问题说出口) ----------------

def _build_phrase_messages(state: SessionState, risk: RiskPoint | None,
                           follow_up: str) -> list[dict]:
    """把"要攻哪条 + 怎么攻"变成一次流式提问的 prompt。"""
    risk_block = ""
    if risk is not None:
        risk_block = (
            f"本条高危点:\n标题:{risk.title}\n"
            f"风险:{risk.detail}\n"
            f"{('原文依据:「' + risk.evidence + '」') if risk.evidence else '(无原文,基于 JD/简历要点询问)'}"
        )
    grounding = f"\n可引用的检索原文:\n{state.last_grounding}" if state.last_grounding.strip() else ""
    return [
        {"role": "system", "content": (
            "你是模拟面试的 AI 面试官。现在把内部指示落实成【一个】真正会问出口的问题。"
            "要求:① 只问一个问题,自然、口语、有追问感,别铺垫长篇;"
            "② 不要自问自答、不要给出答案暗示、不要点破这是考题;"
            "③ 如果给了可引用原文,可以顺着原文问,但别整段照抄;"
            "④ 别重复最近已经问过的问题。"
        )},
        {"role": "user", "content": (
            f"{risk_block}\n"
            f"{('追问角度:' + follow_up) if follow_up else ''}\n"
            f"{grounding}\n\n"
            "最近对话(避免重复):\n" + format_recent_history(state) +
            "\n\n现在,请把上面的指示说成你实际会问的那一个问题。"
        )},
    ]


def phrase_question(llm, model: str, state: SessionState, risk: RiskPoint | None,
                    follow_up: str = "", *, on_delta=None) -> tuple[str, float | None]:
    """流式生成问题文字,边吐边经 on_delta 回调(打字机效果)。返回 (问题全文, TTFT毫秒)。"""
    client = _maybe_llm(llm)
    call = client.stream_chat(model=model, messages=_build_phrase_messages(state, risk, follow_up),
                              max_tokens=config.MAX_TOKENS_QUESTION,
                              on_delta=on_delta, kind="question")
    return call.text, call.ttft_ms
