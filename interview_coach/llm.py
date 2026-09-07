"""LLM 统一出口(第 3/9 课):全项目唯一允许"真调大模型"的地方。

为什么单独一个模块(讲得出 why):
- 面试官/评卷人/出题器只关心"给我一段文本 / 流式给我一段文本 / 帮我调个工具",
  不关心 DeepSeek、OpenAI、还是假模型 → 接口统一,上层全部可测;
- 流式输出、TTFT 首字延迟、重试退避、用量/成本统计,这些"横切关注点"
  只写一处,任何角色自动获得,不用每个 agent 自己抄一遍。

核心 API:
    get_client()                       → 默认真客户端(读 .env,没 key 报友好错)
    LLMClient.chat(...)                → 一次性文本(json_mode 可要 JSON)
    LLMClient.stream_chat(...)         → 流式,边收边通过 on_delta 回调吐字,回传 TTFT
    LLMClient.chat_with_tools(...)     → 让模型决定"要不要调工具"(Function Calling)

测试约定(红线):pytest 一律用 FakeClient / 纯函数,绝不碰真实网络;
本文件的真网络路径只在小步冒烟脚本里手工跑。
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

from . import config

# 模块加载时读一次 .env(显式路径,避免 load_dotenv() 在某些环境找不到调用栈)
load_dotenv(config.PROJECT_ROOT / ".env")


class LLMError(RuntimeError):
    """LLM 调用失败的统一异常:上层 catch 它给用户友好提示。"""


# ---------------- 数据结构 ----------------

@dataclass
class Usage:
    """一次调用的 token 用量(响应里带;带不动就 0,显示时兜底)。"""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class LMCall:
    """一次 LLM 调用的完整结果,是 chat / stream / tools 共同的返回形状。"""
    text: str = ""                        # 模型回的文字
    finish_reason: str = "stop"           # stop=正常 / length=被截断 / tool_calls=想调工具
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    ttft_ms: float | None = None          # 首字延迟(流式才有意义;一次性调用为 None)
    latency_ms: float | None = None       # 本次总耗时
    tool_calls: list[dict[str, Any]] | None = None  # 原始 tool_calls(有则模型在请求调工具)
    assistant_message: dict[str, Any] | None = None  # 完整 assistant 消息(含 reasoning_content,
    # 若模型是"思考模式"就必须原样带回去继续对话,否则 DeepSeek 报 400) —— 见 _robust

    @property
    def truncated(self) -> bool:
        """finish_reason == length → 输出被 max_tokens 截断,上层要提示。"""
        return self.finish_reason == "length"


# ---------------- 用量 / 成本账本 ----------------

@dataclass
class CallRecord:
    """账本里的一行:一次调用花了多少 token / 毫秒(第 9 课可观测)。"""
    kind: str                       # plan / decision / question / score / eval...
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float | None = None
    ttft_ms: float | None = None
    cost_yuan: float | None = None  # 单价未配置时为 None(= 只数 token 不算钱)


# 进程内账本(简单起见用列表;第 9 课会把整场会话连同账本一起落 SQLite)
_USAGE_LOG: list[CallRecord] = []


def reset_usage() -> None:
    _USAGE_LOG.clear()


def record_usage(rec: CallRecord) -> None:
    _USAGE_LOG.append(rec)


def usage_snapshot() -> list[CallRecord]:
    """返回账本副本(深拷贝的意义不大,都是不可变字段;直接给列表拷贝)。"""
    return list(_USAGE_LOG)


def cost_yuan(model: str, usage: Usage) -> float | None:
    """按 .env 可选单价(元/百万 token)估算金额;没配单价就返回 None(不编价)。"""
    price_in = _env_float("DEEPSEEK_PRICE_IN_PER_M")
    price_out = _env_float("DEEPSEEK_PRICE_OUT_PER_M")
    if price_in is None or price_out is None:
        return None
    return (usage.prompt_tokens * price_in + usage.completion_tokens * price_out) / 1_000_000


def _env_float(key: str) -> float | None:
    raw = os.getenv(key, "").strip()
    if not raw or not raw.replace(".", "").isdigit():
        return None
    return float(raw)


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数(用于上下文预算裁剪,不用精确)。

    中文大约 1 字≈1 token,英文约 4 字符≈1 token,混合文本取折中权重。
    只用来"判断要不要裁剪",宁可估大一点,别把预算估小导致真超限。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    other = len(text) - cjk
    return int(cjk * 1.0 + other / 3.5) + 1


# ---------------- 客户端 ----------------

def _load_key() -> tuple[str, str]:
    """从 .env 读 API key 和 base_url,没配就报友好错误(不裸奔报错)。"""
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
    if not api_key or api_key == "你的key填在这里":
        raise LLMError(
            "❌ 未设置 DEEPSEEK_API_KEY。\n"
            "请把项目根目录的 .env.example 复制成 .env,填入你的 key:\n"
            "  去 https://platform.deepseek.com/api_keys 创建。"
        )
    return api_key, base_url


class LLMClient:
    """真客户端:包一层 OpenAI SDK,指向 DeepSeek。

    成员函数都接受 model / temperature / max_tokens,
    让上层把"决策要稳、追问要自然"这类差异当成参数传进来,
    而不是每个角色各写一套调用。
    """

    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 timeout_s: float = config.LLM_TIMEOUT_S,
                 max_retries: int | None = None) -> None:
        if api_key:
            # 调用方已显式给了 key(BYOK:公网访客自填)。此时服务端可能根本没有 .env
            # (设计如此),绝不能再走 _load_key() —— 它会因"没配默认 key"抛 LLMError。
            # base_url 从环境读,缺省用官方默认即可。
            base_url = (base_url or os.getenv("DEEPSEEK_BASE_URL")
                        or "https://api.deepseek.com").strip()
        else:
            # 没给 key → 才从 .env 读默认(没配就抛友好错)。
            env_key, env_base = _load_key()
            api_key, base_url = env_key, (base_url or env_base)
        # 每类调用可以单独决定"重试几次":如 AI 出题给足超时但不重试(见 build_plan_auto)
        self._max_retries = config.LLM_MAX_RETRIES if max_retries is None else max_retries
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)

    # ---- 一次性(非流式)----
    def chat(self, model: str, messages: list[dict], *,
             temperature: float = config.TEMPERATURE_DECISION,
             max_tokens: int | None = None,
             json_mode: bool = False,
             kind: str = "chat") -> LMCall:
        """非流式文本。json_mode=True 时要求模型输出 JSON(DeepSeek 需 prompt 含 'json')。"""
        kwargs: dict[str, Any] = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return self._robust(lambda: self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        ), model=model, kind=kind, stream=False)

    # ---- 流式 ----
    def stream_chat(self, model: str, messages: list[dict], *,
                    temperature: float = config.TEMPERATURE_GENERATE,
                    max_tokens: int | None = config.MAX_TOKENS_QUESTION,
                    on_delta: Callable[[str], None] | None = None,
                    kind: str = "question") -> LMCall:
        """流式:逐块通过 on_delta 吐字(打字机效果),同时测 TTFT(首字延迟)。

        一次性非流式拿不到 TTFT;流式从"发请求"到"收到第一块有效文字"的时间差,
        就是用户感知的"首字延迟"——第 3 课流式输出的核心指标。
        """
        started = time.perf_counter()
        first_char_at: float | None = None
        parts: list[str] = []
        usage = Usage()
        finish = "stop"
        prompt_text = "\n".join(m.get("content", "") or "" for m in messages)

        def _gen() -> None:
            nonlocal first_char_at, finish
            stream = self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                piece = delta.content if delta and delta.content else ""
                if piece and first_char_at is None:
                    first_char_at = time.perf_counter()  # 第一块有效文字到达
                if chunk.choices[0].finish_reason:
                    finish = chunk.choices[0].finish_reason
                if delta and getattr(delta, "tool_calls", None):
                    pass  # 流式追问一般不调工具,真遇到由调用方另行处理
                if piece:
                    parts.append(piece)
                    if on_delta:
                        on_delta(piece)
                if chunk.usage:
                    usage.prompt_tokens = chunk.usage.prompt_tokens or 0
                    usage.completion_tokens = chunk.usage.completion_tokens or 0

        self._robust(_gen, model=model, kind=kind, stream=True)

        # 流式响应多数不回传精确 usage → 拿不到就按文本量估算(仅用于预算/展示,标注"约")
        if usage.total == 0:
            usage.prompt_tokens = estimate_tokens(prompt_text)
            usage.completion_tokens = estimate_tokens("".join(parts))

        total_ms = (time.perf_counter() - started) * 1000
        ttft_ms = (first_char_at - started) * 1000 if first_char_at else None
        record_usage(CallRecord(kind=kind, model=model,
                                prompt_tokens=usage.prompt_tokens,
                                completion_tokens=usage.completion_tokens,
                                latency_ms=total_ms, ttft_ms=ttft_ms,
                                cost_yuan=cost_yuan(model, usage)))
        return LMCall(text="".join(parts), finish_reason=finish, usage=usage,
                      model=model, ttft_ms=ttft_ms, latency_ms=total_ms)

    # ---- Function Calling(第 5 课)----
    def chat_with_tools(self, model: str, messages: list[dict], tools: list[dict], *,
                        temperature: float = config.TEMPERATURE_DECISION,
                        max_tokens: int | None = config.MAX_TOKENS_DECISION,
                        kind: str = "tools") -> LMCall:
        """把工具清单交给模型,让模型自己决定:回文字 或 请求调工具。

        返回 LMCall.tool_calls 非空 = 模型在请求调用函数,由调用方(engine)
        执行并把结果以 role=tool 消息喂回去,形成"工具调用闭环"。
        """
        kwargs: dict[str, Any] = {}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return self._robust(lambda: self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        ), model=model, kind=kind, stream=False)

    # ---- 内部:重试 + 记账 ----
    def _robust(self, fn, *, model: str, kind: str, stream: bool) -> LMCall:
        """真正发请求的地方:失败重试(退避)+ 成功后记账。

        为什么手动包这层而不是每个函数 try?
        - 重试/超时/记账是"所有调用都要的",抽出来一处写;
        - 临时故障(网络抖动 / 限流 429)重试一次常能自己好;真错误就抛 LLMError 给上层。
        """
        last_err: Exception | None = None
        for attempt in range(self._max_retries + 1):
            started = time.perf_counter()
            try:
                if stream:
                    fn()                      # 流式内部已把文字喂给 on_delta
                else:
                    resp = fn()
                break
            except Exception as e:            # 重试窗口窄,先捕获所有异常统一处理
                last_err = e
                if attempt < self._max_retries:
                    time.sleep(0.8 * (attempt + 1))   # 退避:1st失败后等 0.8s
                    continue
                raise LLMError(f"调用大模型失败(已重试 {self._max_retries} 次):{e}") from e
        else:
            # 理论走不到(for...else 语义保险),但类型检查需要
            raise LLMError(f"调用大模型失败:{last_err}")

        if stream:
            return LMCall()   # 流式分支:真实结果由 stream_chat 收尾组装,这里只是兜底

        # 非流式:解析标准响应结构
        message = resp.choices[0].message
        text = message.content or ""
        finish = resp.choices[0].finish_reason or "stop"
        u = getattr(resp, "usage", None)
        usage = Usage(
            prompt_tokens=(u.prompt_tokens if u else 0) or 0,
            completion_tokens=(u.completion_tokens if u else 0) or 0,
        )
        tool_calls = None
        if getattr(message, "tool_calls", None):
            tool_calls = [
                {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                for tc in message.tool_calls
            ]
            finish = "tool_calls"

        # 把这一轮的完整 assistant 消息留住(OpenAI 协议原样):思考模式下的 reasoning_content
        # 在"继续同一段对话"时**必须原样带回去**(DeepSeek 硬性要求,漏了报 400);
        # 上层(agent_decide / parse_with_retry)要把这段消息重新喂回去时用它,而不是自己重拼。
        assistant_message: dict[str, Any] = {"role": "assistant", "content": text}
        rc = getattr(message, "reasoning_content", None)
        if rc:
            assistant_message["reasoning_content"] = rc
        if getattr(message, "tool_calls", None):
            assistant_message["content"] = None
            assistant_message["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in message.tool_calls
            ]

        total_ms = (time.perf_counter() - started) * 1000
        rec = CallRecord(kind=kind, model=model,
                         prompt_tokens=usage.prompt_tokens,
                         completion_tokens=usage.completion_tokens,
                         latency_ms=total_ms,
                         cost_yuan=cost_yuan(model, usage))
        record_usage(rec)
        return LMCall(text=text, finish_reason=finish, usage=usage, model=model,
                      latency_ms=total_ms, tool_calls=tool_calls,
                      assistant_message=assistant_message)


# 默认真客户端(惰性单例:第一次要用了才建,没 key 也在此时才报错)
_client: LLMClient | None = None


def get_client(timeout_s: float | None = None,
               max_retries: int | None = None) -> LLMClient:
    """返回真客户端。测试请自行构造 FakeClient 传进上层,别调用这个。

    默认返回共享单例(90s 超时、2 次重试)。某些调用想单独调策略时,
    可传 timeout_s / max_retries → 新建一个专用客户端(建客户端不联网,便宜)。
    如 AI 出题:180s 超时 + 不重试,避免"健康但慢的长生成"被误杀、也避免故障时干等太久。
    """
    global _client
    if timeout_s is None and max_retries is None:
        if _client is None:
            _client = LLMClient()
        return _client
    return LLMClient(timeout_s=timeout_s or config.LLM_TIMEOUT_S,
                     max_retries=max_retries)
