"""结构化输出测试:JSON 容错抠取 + 校验失败自动带反馈重试(不碰网络)。"""

from pydantic import BaseModel, Field

from interview_coach.fake import FakeClient
from interview_coach.llm import LLMError
from interview_coach.structured import extract_json, parse_with_retry


class Point(BaseModel):
    x: int
    y: int


def test_extract_json_from_plain():
    assert extract_json('{"x": 1, "y": 2}') == {"x": 1, "y": 2}


def test_extract_json_inside_code_fence_and_extra_text():
    """模型常见的坏习惯:套 ```json 围栏 + 前后夹解释 → 都要能抠出来。"""
    raw = '好的,结果如下:```json\n{"x": 3, "y": 4}\n``` 请查收。'
    assert extract_json(raw) == {"x": 3, "y": 4}


def test_extract_json_raises_on_garbage():
    import pytest
    with pytest.raises(ValueError):
        extract_json("我什么都没输出")


def test_parse_with_retry_feedback_loop():
    """第一次给坏 JSON,第二次给对的 → 断言:中间多了一次带反馈的 user 消息。"""
    calls: dict[int, str] = {}

    def responder(kind, model, messages, json_mode):
        # 记录"有没有出现过反馈消息"(第 2 次起 user 内容含'不合规')
        feedback_seen = any("不合规" in m.get("content", "") for m in messages)
        calls[len(calls)] = feedback_seen
        if len(calls) == 1:
            return '{"x": "不是数字"}'        # 第一次:x 该是 int,故意给错类型
        return '{"x": 10, "y": 20}'           # 第二次:合格
    fake = FakeClient(responder=responder)
    p = parse_with_retry(fake, "m", [{"role": "user", "content": "给个 json"}],
                         schema=Point, kind="test")
    assert (p.x, p.y) == (10, 20)
    assert list(calls.values()) == [False, True]   # 第二次确实带上了反馈


def test_parse_with_retry_gives_up_with_friendly_error():
    """连续给坏数据,最终抛 LLMError 而不是返回残缺对象。"""
    import pytest
    fake = FakeClient(responder=lambda *a: '{"x": 1}')   # 永远缺 y
    with pytest.raises(LLMError):
        parse_with_retry(fake, "m", [{"role": "user", "content": "给个 json"}],
                         schema=Point, kind="test", max_attempts=2)


def test_parse_with_retry_recovers_from_empty_reply():
    """第一次返回空串(思考吃光 token 的典型症状)→ 不应立刻判死,应提示后重试。"""
    calls = {"n": 0}

    def responder(kind, model, messages, json_mode):
        calls["n"] += 1
        if calls["n"] == 1:
            return ""                       # 空回复
        return '{"x": 7, "y": 8}'
    fake = FakeClient(responder=responder)
    p = parse_with_retry(fake, "m", [{"role": "user", "content": "给个 json"}],
                         schema=Point, kind="test")
    assert (p.x, p.y) == (7, 8)
    assert calls["n"] == 2                  # 确确实重重试了一次


def test_parse_with_retry_empty_reply_eventually_raises():
    """连续空回复超过上限 → 仍抛 LLMError(不无限重试、不返回坏对象)。"""
    import pytest
    fake = FakeClient(responder=lambda *a: "")
    with pytest.raises(LLMError):
        parse_with_retry(fake, "m", [{"role": "user", "content": "给个 json"}],
                         schema=Point, kind="test", max_attempts=2)
