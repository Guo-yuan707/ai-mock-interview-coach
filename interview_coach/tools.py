"""工具箱(第 5 课 Function Calling):面试官能主动调的工具集。

分工:
    build_tool_specs()   → 把"有哪些工具、参数长什么样"声明成模型能懂的 JSON(函数调用协议)
    run_tool(...)        → 工具真正干活的地方(检索知识库 / 查追问进度 / 结束面试)

为什么走"模型自己决定调不调"而不是写死 if-else(讲得出 why):
- 面试官是 Agent:信息不够时它应该"去查一下再问",而不是硬着头皮编;
- Function Calling 闭环 = 模型发出 tool_calls → 程序执行 → 结果以 tool 角色喂回 →
  模型带着结果继续思考 → 直到它给出最终追问/结束。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .state import SessionState

# 三个工具的"函数名",模型靠这个名字来点名
TOOL_RETRIEVE = "retrieve_knowledge"
TOOL_PROGRESS = "check_progress"
TOOL_FINISH = "finish_interview"


def build_tool_specs() -> list[dict]:
    """声明工具清单(OpenAI 函数调用协议),交给模型。

    模型看到这张表就知道:我可以查知识库 / 查进度 / 结束,各有啥参数。
    """
    return [
        {
            "type": "function",
            "function": {
                "name": TOOL_RETRIEVE,
                "description": "从本次面试的知识库(JD/评分维度/简历要点/已检索资料)中检索一段最相关的原文,"
                               "用来把你的下一个问题问得更扎实、不靠记忆编。当你觉得需要引原文时调用。",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "要查什么,例如 'RAG 检索流程'"}},
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": TOOL_PROGRESS,
                "description": "查看这场面试进行到哪:高危点覆盖几条、最近答得如何。用于决定要不要收尾或换方向。",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": TOOL_FINISH,
                "description": "结束这场面试。当你判断高危点已覆盖充分、或候选人表现已可下结论时调用,并给出收尾小结。",
                "parameters": {
                    "type": "object",
                    "properties": {"summary": {"type": "string", "description": "为什么现在结束(几句话)"}},
                    "required": ["summary"],
                },
            },
        },
    ]


@dataclass
class ToolContext:
    """工具执行时能碰到的现场:会话状态 + 可选的语义知识库。

    kb 只需有 search(query, k)->list[KBHit];没挂就返回"知识库未启用"。
    这样 retrieval 是否已实现都不影响工具本身,测试也好注入假 KB。
    """
    state: SessionState
    kb: Any | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """一次工具执行的结果。"""
    name: str
    output: str
    is_finish: bool = False            # 是否触发了"结束面试"
    ok: bool = True


def run_tool(name: str, arguments: dict | None, ctx: ToolContext) -> ToolResult:
    """工具总入口:按名字分发到真正实现。arguments 是模型给的参数 dict。"""
    arguments = arguments or {}
    if name == TOOL_RETRIEVE:
        return _retrieve(arguments.get("query", ""), ctx)
    if name == TOOL_PROGRESS:
        return _progress(ctx)
    if name == TOOL_FINISH:
        return _finish(arguments.get("summary", ""), ctx)
    return ToolResult(name=name, output=f"未知工具:{name}", ok=False)


def _retrieve(query: str, ctx: ToolContext) -> ToolResult:
    query = (query or "").strip()
    if not query:
        return ToolResult(name=TOOL_RETRIEVE, output="检索词为空。")
    if ctx.kb is None:
        return ToolResult(name=TOOL_RETRIEVE, output="本场未挂载知识库,无法检索。请基于你已知的信息继续。")
    hits = ctx.kb.search(query, k=3)
    if not hits:
        return ToolResult(name=TOOL_RETRIEVE, output=f"知识库没找到与『{query}』相关的内容。")
    lines = [f"命中 {len(hits)} 条(可引用原文):"]
    for h in hits:
        src = f"[{h.source}]" if getattr(h, "source", "") else ""
        lines.append(f"- {src} {getattr(h, 'text', '')[:200]}")
    return ToolResult(name=TOOL_RETRIEVE, output="\n".join(lines))


def _progress(ctx: ToolContext) -> ToolResult:
    cov = ctx.state.coverage()
    last = ctx.state.last_turn()
    lines = [
        f"高危覆盖:{cov['asked']}/{cov['total']}({cov['rate']:.0%})",
        f"已进行轮次:{len(ctx.state.turns)}",
    ]
    if last is not None and last.score is not None:
        lines.append(f"上一答评分:{last.score.overall}/10")
    remaining = ctx.state.remaining_risks()
    lines.append("仍未充分追问的高危点:" + ("、".join(r.title[:20] for r in remaining[:5]) or "无,可考虑收尾"))
    return ToolResult(name=TOOL_PROGRESS, output="\n".join(lines))


def _finish(summary: str, ctx: ToolContext) -> ToolResult:
    ctx.state.finished = True
    ctx.state.finish_reason = summary or "面试官认为追问已充分,主动结束"
    return ToolResult(name=TOOL_FINISH, output=f"面试结束。收尾小结:{ctx.state.finish_reason}", is_finish=True)


def parse_arguments(raw: str) -> dict:
    """把模型给的 tool arguments(字符串)解析成 dict;坏了就返回空 dict,别让一次坏参数带崩全场。"""
    try:
        obj = json.loads(raw or "{}")
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


def format_tool_result_for_show(tool_calls_used: list[str], state: SessionState) -> str:
    """给 UI 一行"面试官刚才查了知识库"之类的痕迹(可视化,可查证)。"""
    if not tool_calls_used:
        return ""
    parts = []
    if any(t.endswith("retrieve_knowledge") for t in tool_calls_used):
        parts.append("🔍 面试官主动检索了知识库再提问")
    if any(t.endswith("check_progress") for t in tool_calls_used):
        parts.append("📊 面试官查了追问进度")
    return " | ".join(parts) if parts else ""
