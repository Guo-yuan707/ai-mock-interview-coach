"""结构化 schema(第 4 课):给"出题 / 追问决策 / 逐题评分"定死 JSON 形状。

为什么用 pydantic 而不用手写 dict 校验(讲得出 why):
- 手写校验 = 一堆 if obj.get("x") ...,散在各处还容易漏;
- pydantic 声明式:字段类型 + 范围(如分数 1~5)+ 必填,一次写完,解析失败还自带
  清晰的报错信息,配合 structured.parse_with_retry 就能"让模型自查重来"。

命名约定:带 Schema 后缀的 = 给大模型当"输出合同"的 pydantic 模型;
它们和 plan.py 里给内部用的 dataclass 分开:Schema 偏"AI 要填的表单",
dataclass 偏"模块间传的成品",两者之间由 adapter(如 planner)做映射。
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# 所有 schema 都容忍模型多塞未知字段(不因多一个键就整份重试),
# 但"必填/范围/类型"仍强校验——那是我们真正在乎的。
_EXTRA = {"extra": "ignore"}


# ---------------- AI 出题(第 6 课,替换确定性 v0) ----------------

class _DimSchema(BaseModel):
    """一条评分维度:评卷人按它打分;name 必须 = 输出计划里列过的名字。"""
    model_config = ConfigDict(**_EXTRA)
    name: str = Field(description="维度名,如 技术深度(RAG)")
    why: str = Field(description="为什么列这一维(源自 JD 的哪些信号)")
    source_keywords: list[str] = Field(default_factory=list)


class _RiskSchema(BaseModel):
    """一个高危追问点。kind 必须是受控枚举,便于程序分流处理。"""
    model_config = ConfigDict(**_EXTRA)
    kind: Literal["missing_jd_keyword", "unevidenced_claim", "jd_probe", "resume_probe", "ai_probe"]
    title: str = Field(description="一句话标题,如 简历未体现 JD 高频点:RAG")
    detail: str = Field(description="会怎么追问、风险在哪(具体、可执行)")
    evidence: str = Field(default="", description="依据的原文片段;宁可空,不许编")


class PlanSchema(BaseModel):
    """AI 版面试计划的输出合同:和确定性版的 InterviewPlan 同构。"""
    model_config = ConfigDict(**_EXTRA)
    target_hint: str = Field(default="", description="从 JD 嗅出的岗位方向(仅提示)")
    dimensions: list[_DimSchema] = Field(min_length=2, description="至少 2 维:通用维度 + JD 技术维度")
    risk_points: list[_RiskSchema] = Field(min_length=5, description="至少 5 个高危追问点")


# ---------------- 追问决策(第 6 课,模型决定下一招) ----------------

class DecisionSchema(BaseModel):
    """面试官"这一步往哪走"的决策 JSON。

    不是 if-else 写死的轮询,而是把现场(已问过哪些高危点、回答强弱)
    摊给模型,让它自己选:追同一条 / 换下一条 / 先查资料 / 收尾。
    """
    model_config = ConfigDict(**_EXTRA)
    intent: Literal["ask", "retrieve", "finish"] = "ask"
    risk_id: str = Field(default="", description="要追问的高危点 id(来自 InterviewPlan 的序号);finish 时可为空")
    retrieve_query: str = Field(default="", description="intent=retrieve 时的检索词(要去知识库查什么)")
    follow_up: str = Field(default="", description="追问角度:如 追证据 / 追细节 / 换一层问,给措辞当指引")
    note: str = Field(default="", description="给'措辞生成'的内部提示(不外露给用户)")

    @field_validator("risk_id")
    @classmethod
    def risk_id_can_be_empty_only_when_finish(cls, v: str, info) -> str:
        # 保留宽松:risk_id 空时,由 engine 用"下一个未覆盖高危点"兜底,不在这里卡死重试
        return v


# ---------------- 逐题评分(第 6 课,评卷人) ----------------

class _DimScore(BaseModel):
    """某一评分维度上的得分 + 依据。"""
    model_config = ConfigDict(**_EXTRA)
    dimension: str = Field(description="维度名,必须来自面试计划的维度表")
    score: int = Field(description="1~5 分")
    comment: str = Field(description="这维给几分的人话理由(短)")
    gap: str = Field(default="", description="扣分点:这句没证据 / 会穿帮之处;没问题就空")

    @field_validator("score")
    @classmethod
    def score_in_1_5(cls, v: int) -> int:
        if not (1 <= v <= 5):
            raise ValueError(f"分数必须在 1~5 之间,收到 {v}")
        return v


class ScoreSchema(BaseModel):
    """评卷人对"某一问的回答"的结构化评分。"""
    model_config = ConfigDict(**_EXTRA)
    overall: int = Field(description="本题整体 1~10 分")
    dimensions: list[_DimScore] = Field(min_length=1, description="逐维度打分,维度名须与计划一致")
    strengths: list[str] = Field(default_factory=list, description="答得好的点(具体,有依据)")
    weaknesses: list[str] = Field(default_factory=list, description="薄弱/会被追问穿的点(具体,不许泛泛)")
    verdict: str = Field(description="一句话结论:这回答稳不稳、下题该往哪补")

    @field_validator("overall")
    @classmethod
    def overall_in_1_10(cls, v: int) -> int:
        if not (1 <= v <= 10):
            raise ValueError(f"整体分必须在 1~10 之间,收到 {v}")
        return v
