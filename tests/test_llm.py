"""LLM 统一出口测试:只测纯函数 + FakeClient 契约,绝不碰真实网络(红线)。"""

from interview_coach.fake import FakeClient
from interview_coach.llm import LMCall, Usage, cost_yuan, estimate_tokens, reset_usage


def test_estimate_tokens_is_positive_and_scales():
    """估算 token 数:空串=0,文本越长越多(中文按约 1 字 1 token 粗算)。"""
    assert estimate_tokens("") == 0
    zh = estimate_tokens("这是一个用于测试的中文句子,大概二十个字左右的样子吧。")
    short = estimate_tokens("hi")
    long_en = estimate_tokens("a" * 400)
    assert zh > 0
    assert long_en > estimate_tokens("hello")  # 更长 → 估算更大
    assert estimate_tokens("hello") > 0


def test_cost_yuan_none_when_prices_unset(monkeypatch):
    """没配单价(默认 .env 没有)→ 返回 None,绝不用编的价格算钱。"""
    monkeypatch.delenv("DEEPSEEK_PRICE_IN_PER_M", raising=False)
    monkeypatch.delenv("DEEPSEEK_PRICE_OUT_PER_M", raising=False)
    assert cost_yuan("deepseek-v4-flash", Usage(prompt_tokens=1000, completion_tokens=500)) is None


def test_fake_chat_records_and_returns_text():
    """FakeClient.chat 记录入参并原样返回脚本回复(不碰网络)。"""
    fake = FakeClient(responder=lambda kind, model, messages, jm: "好的,我会从 STAR 追问。")
    call = fake.chat("m", [{"role": "user", "content": "下一题?"}], json_mode=False, kind="decision")
    assert call.text.startswith("好的")
    assert fake.records[0].kind == "decision"
    assert fake.records[0].json_mode is False


def test_fake_stream_pushes_deltas_via_callback_and_has_ttft():
    """假流式:on_delta 收到整段文字,返回里有象征性 TTFT(真实流式才有这个指标)。"""
    got: list[str] = []
    fake = FakeClient(responder=lambda *a: "流式输出的这段文字")
    call = fake.stream_chat("m", [{"role": "user", "content": "问"}], on_delta=got.append, kind="question")
    assert "".join(got) == "流式输出的这段文字"
    assert call.ttft_ms is not None
    assert call.text == "流式输出的这段文字"


def test_fake_can_return_scripted_tool_calls():
    """需要断言'模型请求调工具'时,responder 可直接返回构造好的 LMCall。"""
    tool_call = LMCall(
        text="", finish_reason="tool_calls",
        tool_calls=[{"id": "t1", "name": "retrieve_knowledge", "arguments": '{"query": "RAG"}'}],
    )
    fake = FakeClient(responder=lambda *a: tool_call)
    call = fake.chat_with_tools("m", [], tools=[{}], kind="decision")
    assert call.tool_calls and call.tool_calls[0]["name"] == "retrieve_knowledge"


def test_usage_snapshot_isolated_between_fake_and_real():
    """Fake 不写全局账本(真账本只在真客户端里记,避免测试污染统计)。"""
    reset_usage()
    FakeClient().chat("m", [{"role": "user", "content": "x"}], kind="chat")
    from interview_coach.llm import usage_snapshot
    assert usage_snapshot() == []  # Fake 不记账;真 LLMClient 才记
