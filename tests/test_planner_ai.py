"""AI 版出题测试:用脚本化的 PlanSchema JSON 喂假模型,验证映射正确(不碰网络)。"""

import json
import time

from interview_coach import config
from interview_coach.fake import FakeClient
from interview_coach.plan import InterviewPlan
from interview_coach.planner import build_plan_ai, build_plan_auto

JD = "AI 应用研发,负责大模型 RAG 问答与多轮对话;要 Python/pytest 功底,熟悉向量库。"
RESUME = "熟练 Python,负责过一个小工具,提升团队效率。"

PLAN_JSON = {
    "target_hint": "偏 AI Agent / 大模型应用方向",
    "dimensions": [
        {"name": "岗位理解与匹配", "why": "是否读懂 JD", "source_keywords": []},
        {"name": "表达与逻辑", "why": "结构清晰", "source_keywords": []},
        {"name": "技术深度(RAG)", "why": "JD 点明 RAG", "source_keywords": ["rag", "向量库"]},
    ],
    "risk_points": [
        {"kind": "missing_jd_keyword", "title": "简历未体现 RAG", "detail": "会问对 RAG 的理解", "evidence": "熟悉向量库"},
        {"kind": "unevidenced_claim", "title": "这句没证据", "detail": "提升多少", "evidence": "提升团队效率"},
        {"kind": "jd_probe", "title": "讲讲多轮对话", "detail": "上下文管理", "evidence": "多轮对话"},
        {"kind": "resume_probe", "title": "STAR 展开", "detail": "一手细节", "evidence": ""},
        {"kind": "ai_probe", "title": "评测方法", "detail": "怎么量化", "evidence": ""},
    ],
}


def test_build_plan_ai_maps_to_interview_plan_shape():
    """AI 输出 → 仍是 InterviewPlan(和确定性版同形状,上层无感知)。"""
    fake = FakeClient(responder=lambda kind, model, messages, jm: json.dumps(PLAN_JSON, ensure_ascii=False))
    plan = build_plan_ai(JD, RESUME, llm=fake)
    assert plan.n_dimensions >= 2
    assert plan.n_risks >= 5
    assert plan.target_hint
    # 映射正确:维度和风险点都被搬进了 dataclass
    names = [d.name for d in plan.dimensions]
    assert "技术深度(RAG)" in names
    kinds = {r.kind for r in plan.risk_points}
    assert "unevidenced_claim" in kinds
    # 记录里应该标注为 plan 类型的调用
    assert fake.records[0].kind == "plan"
    assert fake.records[0].json_mode is True


def test_build_plan_ai_uses_pro_model_by_default():
    fake = FakeClient(responder=lambda kind, model, messages, jm: json.dumps(PLAN_JSON, ensure_ascii=False))
    build_plan_ai(JD, RESUME, llm=fake)
    assert fake.records[0].model == "deepseek-v4-pro"


def test_build_plan_auto_fast_ai_wins(monkeypatch):
    """墙钟内能出 AI 计划 → mode='ai'(快路径,纯本地)。"""
    fake = FakeClient(responder=lambda kind, model, messages, jm: json.dumps(PLAN_JSON, ensure_ascii=False))
    monkeypatch.setattr(config, "PLAN_WALL_TIMEOUT_S", 10)
    plan, mode = build_plan_auto(JD, RESUME, use_ai=True, llm=fake)
    assert mode == "ai"
    assert plan.n_risks >= 5


def test_build_plan_auto_degrades_on_wallclock_timeout(monkeypatch):
    """AI 出题超过硬墙钟 → 不无限等、马上降级确定性版(慢模型 = 后台慢吐,测墙钟拦截)。"""
    def slow(kind, model, messages, jm):
        time.sleep(0.4)                # 假装后端很慢(慢滴流)
        return json.dumps(PLAN_JSON, ensure_ascii=False)
    fake = FakeClient(responder=slow)
    monkeypatch.setattr(config, "PLAN_WALL_TIMEOUT_S", 0.05)   # 墙钟压到 50ms → 必超时
    t0 = time.time()
    plan, mode = build_plan_auto(JD, RESUME, use_ai=True, llm=fake)
    assert mode == "deterministic"     # 及时降级,而不是陪着后台线程一起等
    assert isinstance(plan, InterviewPlan)
    assert time.time() - t0 < 1.0      # 没有真等 0.4s×N 次网络重试
