"""面试计划的数据结构(第 1 课):先定"纸的形状",内容后面再填。

为什么先定义数据结构(讲得出 why):
- 它像一份**合同**:planner(出题)、interviewer(追问)、evaluator(打分)都围绕同一套形状转,
  谁都不需要知道别人内部怎么实现的;
- 第 6 课把"确定性出题"换成"AI 出题"时,只换**填内容的人**,形状不变 → 其它模块零改动。
  这是工程里的老话:**先约定接口,再各写实现**。
"""
from dataclasses import dataclass, field


@dataclass
class RubricDimension:
    """评分表里的一行:评卷人(evaluator)就按这些维度给回答打分。"""
    name: str                       # 维度名,如 "技术深度(RAG)"
    why: str                        # 为什么列这一维(来自 JD 的哪些信号)
    source_keywords: list[str] = field(default_factory=list)   # 触发它的 JD 关键词(可查证)


@dataclass
class RiskPoint:
    """一个"高危追问点":面试官最可能在这里把你的回答追问穿。"""
    kind: str                       # missing_jd_keyword / unevidenced_claim / jd_probe ...
    title: str                      # 一句话标题,如 "简历未体现 JD 高频点:RAG"
    detail: str                     # 具体会怎么追问、风险在哪
    evidence: str = ""              # 依据的原文片段(红线:所有判断可回原文查证)


@dataclass
class InterviewPlan:
    """一份完整的面试计划:评分维度表 + 高危追问清单。"""
    target_hint: str = ""                       # 从 JD 嗅出的岗位方向(仅提示,不作事实)
    jd_categories: dict[str, list[str]] = field(default_factory=dict)  # JD 关键词按类别分组
    dimensions: list[RubricDimension] = field(default_factory=list)
    risk_points: list[RiskPoint] = field(default_factory=list)

    @property
    def n_dimensions(self) -> int:
        return len(self.dimensions)

    @property
    def n_risks(self) -> int:
        return len(self.risk_points)


# ---------------- 序列化(落库 / 回放 / 导出报告都要重建计划) ----------------

def plan_to_dict(plan: InterviewPlan) -> dict:
    """把计划摊成普通 dict(json 可存)。"""
    return {
        "target_hint": plan.target_hint,
        "jd_categories": plan.jd_categories,
        "dimensions": [{"name": d.name, "why": d.why,
                        "source_keywords": list(d.source_keywords)} for d in plan.dimensions],
        "risk_points": [{"kind": r.kind, "title": r.title, "detail": r.detail,
                         "evidence": r.evidence} for r in plan.risk_points],
    }


def plan_from_dict(data: dict) -> InterviewPlan:
    """从 plan_to_dict 的结果重建计划(存过的旧面试能原样回放)。"""
    data = data or {}
    return InterviewPlan(
        target_hint=data.get("target_hint", ""),
        jd_categories=data.get("jd_categories", {}),
        dimensions=[RubricDimension(name=d.get("name", ""), why=d.get("why", ""),
                                    source_keywords=d.get("source_keywords", []))
                    for d in data.get("dimensions", [])],
        risk_points=[RiskPoint(kind=r.get("kind", ""), title=r.get("title", ""),
                               detail=r.get("detail", ""), evidence=r.get("evidence", ""))
                     for r in data.get("risk_points", [])],
    )
