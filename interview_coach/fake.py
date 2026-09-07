"""假 LLM 客户端(测试 + 离线演示双用)。

为什么需要它(讲得出 why):
- 红线:测试绝不碰真实 API、不泄露真实数据 → 纯函数测试都喂"脚本化回复";
- 卖点:没有 API key 也能跑通全流程演示(网页/CLI 传 --offline),
  流式效果、评分、报告照常出来,只是内容是脚本写的、不是真 AI 想的;
- 接口与真客户端(LLMClient)同形 → 上层代码零改动切换,体现"面向接口编程"。

用法(测试):
    fake = FakeClient(responder=lambda kind, model, messages, json_mode: '{"ok": 1}')
    plan = planner.build_plan_ai(..., llm=fake)   # 断言 fake.records 里发生了什么
演示(网页/CLI 的 --offline):
    fake = make_offline_interview_client(plan)   # 没 key 也能演完一整场
"""
from __future__ import annotations

import json

import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .llm import LMCall, Usage


@dataclass
class FakeRecord:
    """记下一次"假调用"的入参,测试用来断言(比如:'decision 调用里是否带了工具')。"""
    kind: str
    model: str
    messages: list[dict]
    json_mode: bool = False
    stream: bool = False
    tools: list[dict] | None = None


# responder 返回 str → 直接当模型回复;返回 LMCall → 原样用(可控制 tool_calls / finish)
Responder = Callable[[str, str, list[dict], bool], "str | LMCall"]


def default_responder(kind: str, model: str, messages: list[dict], json_mode: bool) -> str:
    """兜底回复:尽量从最后一条 user 消息里截一段当回复,别让调用方拿到空串。"""
    last = messages[-1].get("content", "") if messages else ""
    if json_mode:
        return '{"note": "fake"}'
    snippet = (last or "（无输入）")[:30]
    return f"[fake {kind}] 收到:…{snippet}"


class FakeClient:
    """与 LLMClient 同形。stream_chat 模拟"一次吐全"来保持简单(仍有 TTFT)。"""

    def __init__(self, responder: Optional[Responder] = None,
                 model: str = "deepseek-v4-flash") -> None:
        self.responder: Responder = responder or default_responder
        self.model = model
        self.records: list[FakeRecord] = []

    def _resolve(self, kind: str, model: str, messages: list[dict],
                 json_mode: bool, stream: bool, tools: list[dict] | None) -> LMCall:
        self.records.append(FakeRecord(kind=kind, model=model, messages=messages,
                                       json_mode=json_mode, stream=stream, tools=tools))
        out = self.responder(kind, model, messages, json_mode)
        if isinstance(out, LMCall):
            return out
        return LMCall(text=out, model=model, usage=Usage(completion_tokens=len(out)))

    def chat(self, model: str, messages: list[dict], *, temperature: float = 0.2,
             max_tokens: int | None = None, json_mode: bool = False,
             kind: str = "chat") -> LMCall:
        return self._resolve(kind, model, messages, json_mode, stream=False, tools=None)

    def stream_chat(self, model: str, messages: list[dict], *, temperature: float = 0.7,
                    max_tokens: int | None = 500,
                    on_delta: Callable[[str], None] | None = None,
                    kind: str = "question") -> LMCall:
        call = self._resolve(kind, model, messages, json_mode=False,
                             stream=True, tools=None)
        call.ttft_ms = 1.0  # 假环境:象征性给个首字延迟
        if on_delta and call.text:
            on_delta(call.text)   # 假流式:一次吐全,但回调机制照走
        return call

    def chat_with_tools(self, model: str, messages: list[dict], tools: list[dict], *,
                        temperature: float = 0.2,
                        max_tokens: int | None = 700,
                        kind: str = "tools") -> LMCall:
        return self._resolve(kind, model, messages, json_mode=False,
                             stream=False, tools=tools)


# ---------------- 离线"整场面试"替身(CLI --offline 与网页演示共用) ----------------

def make_offline_interview_client(plan) -> FakeClient:
    """离线模式替身:按计划顺序出题 + 固定给分,没 key 也能完整演一场面试。

    它演示的是"流程"(问→答→结构化评分→覆盖→收尾),不是"内容":
        decision → 依次 ask r0..rN(高危点按序逐个问)最后 finish;
        question → 把该风险点的 title/detail 问成一句;
        score    → 按计划维度给固定分(维度名取真实的,评卷清洗才认)。
    上层 UI 会在界面上标注 offline,避免把脚本内容误当成真 AI 的判断。
    """
    counter = {"n": 0}
    dims = [d.name for d in plan.dimensions]
    risks = list(plan.risk_points)

    def responder(kind, model, messages, jm):
        if kind == "decision":
            i = counter["n"]
            counter["n"] += 1
            if i < len(risks):
                return json.dumps({"intent": "ask", "risk_id": f"r{i}", "follow_up": "请展开细节"},
                                  ensure_ascii=False)
            return json.dumps({"intent": "finish", "note": "高危点已按计划覆盖"}, ensure_ascii=False)
        if kind == "question":
            i = counter["n"] - 1
            rp = risks[i] if 0 <= i < len(risks) else None
            if rp is not None:
                return f"我们谈谈「{rp.title}」:{rp.detail}。请用具体例子讲。"
            return "还有想补充展示的技术点吗?"
        if kind == "score":
            payload = {
                "overall": 6,
                "dimensions": [{"dimension": d, "score": 3, "comment": "(offline 演示分)",
                                "gap": "演示环境,非真实评分"} for d in dims],
                "strengths": [], "weaknesses": [],
                "verdict": "(offline) 请用在线模式获得真实评分",
            }
            return json.dumps(payload, ensure_ascii=False)
        return ""

    return FakeClient(responder=responder)
