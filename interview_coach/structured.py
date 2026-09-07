"""结构化输出:让模型"按 schema 说话",解析失败自动带反馈重试(第 4 课)。

为什么需要这层(讲得出 why):
- 大模型默认回自由文本,但"出题 / 打分 / 决策"都必须拿到**字段齐全、类型正确**的对象,
  否则下游一接 `obj.dimensions[0].score` 就炸;
- 模型经常不老实:套 ```json 代码块、前后夹解释文字、漏字段、给超范围分数……
  所以解析要**容错**,校验失败要把"错在哪"反馈回去让它改(而不是静默吞掉)。

对外只暴露一个函数:
    parse_with_retry(llm, ..., schema) -> pydantic 对象
 失败重试 N 次仍不行 → 抛 LLMError(上层给用户友好提示,绝不产出残缺结构)。
"""
from __future__ import annotations

import json
import re
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

from . import config
from .llm import LLMError

# 泛型:调用方传"哪一种 pydantic 模型",我们就返回"哪一种的实例"
T = TypeVar("T", bound=BaseModel)

# 找文本里第一个完整的 {...} 或 [...] 段(容忍模型在 JSON 前后夹带解释/代码块)
_JSON_BLOCK = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)


def extract_json(raw: str) -> dict | list:
    """把模型返回的文字里"像 JSON 的部分"抠出来并解析。

    容错顺序:① 整段直接 loads;② 正则抠出第一个 {...} / [...] 再 loads;
    ③ 都失败 → 抛带原文的错(方便上层拿去反馈重试)。
    """
    raw = (raw or "").strip()
    # 去掉可能包裹的 markdown 代码块围栏 ```json ... ```
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()

    for candidate in (raw, *(m.group(1) for m in _JSON_BLOCK.finditer(raw))):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, (dict, list)):
                return parsed
        except json.JSONDecodeError:
            continue
    raise ValueError(f"从模型回复中解析不出 JSON。原文:\n{raw[:800]}")


def parse_with_retry(llm, model: str, messages: list[dict], schema: Type[T], *,
                     kind: str = "structured",
                     temperature: float = config.TEMPERATURE_DECISION,
                     max_tokens: int | None = None,
                     max_attempts: int = 3,
                     json_mode: bool = True) -> T:
    """核心入口:调模型 → 解析成 schema → 失败则把报错反馈回去让它改。

    流程:
        1. 把 [系统要求 + 上一步原始回复 + 报错原因] 追加进 messages,再让模型重来;
        2. 校验用 pydantic:字段缺失 / 类型错 / 越界(如分数>5)都会触发重试;
        3. max_attempts 次仍不过 → 抛 LLMError(不把坏结构放行给下游)。

    参数:
        llm:    LLMClient 或 FakeClient(测试注入假客户端,不碰真网络)
        model:  用的模型(决策/结构化建议 flash,便宜够稳)
        messages: [{"role": "system"/"user", ...}] 的 prompt(内容需含 'json' 字样,
                 DeepSeek 的 json 模式要求在 prompt 里出现 json)
        schema: 期望的 pydantic 模型,如 ScoreSchema
        kind:   记账用的调用类型标签(plan / score / decision ...)
        json_mode: 是否走 response_format=json_object(默认开,更稳)
    """
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        call = llm.chat(model=model, messages=messages, temperature=temperature,
                        max_tokens=max_tokens, json_mode=json_mode, kind=kind)
        raw = call.text
        if not raw.strip():
            # 空回复多为"思考模式先把 output token 吃光、正文没来得及吐"(DeepSeek 通病)。
            # 提示它别输出思考过程再试一次;几次都不行才当终局错误。
            if attempt < max_attempts:
                messages = messages + [{"role": "user", "content": (
                    "你上一次只输出了思考、没有给出正文。请直接输出符合要求的 JSON,"
                    "不要把推理过程写到正文里。"
                )}]
                continue
            raise LLMError(f"{kind}:模型连续 {max_attempts} 次返回空回复,无法结构化解析。")

        # 1) 先抠 JSON;2) 再用 pydantic 强校验
        try:
            obj = extract_json(raw)
            return schema.model_validate(obj)
        except (ValueError, ValidationError) as e:
            if attempt >= max_attempts:
                raise LLMError(
                    f"{kind}:模型连续 {max_attempts} 次没输出合格 JSON(最后报错:"
                    f"{_short(e)})\n最后一次原文:\n{raw[:500]}"
                ) from e
            # 把"哪里不合规"翻译成模型能懂的话,追加回去让它自查重写。
            # assistant 一条优先用完整原消息(含 reasoning_content,思考模式下必须原样带回)。
            prev_assistant = call.assistant_message or {"role": "assistant", "content": raw}
            messages = messages + [
                prev_assistant,
                {"role": "user", "content": (
                    f"你上一次的输出不合规,原因:{_short(e)}\n"
                    f"请只输出一个符合要求的 JSON,不要任何解释。"
                )},
            ]
    raise LLMError(f"{kind}:结构化解析重试后仍未成功。")  # 逻辑上到不了,类型安全用


def _short(e: BaseException) -> str:
    """把 pydantic ValidationError 压成一行人话(只说第一条错,别刷屏)。"""
    text = str(e)
    if isinstance(e, ValidationError) and e.errors():
        first = e.errors()[0]
        loc = ".".join(str(x) for x in first.get("loc", []))
        return f"字段 {loc}:{first.get('msg', text)}"
    return text[:200]
