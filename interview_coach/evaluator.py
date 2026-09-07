"""评卷人(第 6 课 Agent B):每答完一问,按计划的维度表结构化打分。

与面试官的分工:
    面试官 = 出题、追问的人(产"下一题");
    评卷人 = 记分员(产"这题你答得怎样"),按 JD 维度打分,并明确指出
    "这句没证据,会被追问穿"——绝不当老好人(PROJECT §八 红线)。

对外只一个函数:
    score_answer(plan, question, answer, ...) -> TurnScore
它和"谁问的问题"无关,只依赖:计划(维度表)+ 这一问 + 这一答。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import config
from .llm import get_client
from .plan import InterviewPlan
from .schemas import ScoreSchema
from .structured import parse_with_retry


# ---------------- 数据形状 ----------------

@dataclass
class ScoreDim:
    """某一维度上的得分(评卷人打的分,附带人话依据)。"""
    dimension: str
    score: int                  # 1~5
    comment: str = ""
    gap: str = ""               # 扣分点:这句没证据 / 会穿帮处(可空)


@dataclass
class TurnScore:
    """"这一问"的完整评分结果。"""
    overall: int                      # 1~10
    dimensions: list[ScoreDim] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    verdict: str = ""
    model: str = ""

    @property
    def avg(self) -> float:
        """维度平均(1~5)。没有维度时返回 0,由上层兜底。"""
        return round(sum(d.score for d in self.dimensions) / len(self.dimensions), 2) if self.dimensions else 0.0

    @property
    def weak_dimensions(self) -> list[str]:
        """低于 3 分的维度名(报告里汇总成"薄弱清单"用)。"""
        return [d.dimension for d in self.dimensions if d.score < 3]


# ---------------- 提示词 ----------------

def _dimensions_for_prompt(plan: InterviewPlan) -> str:
    """把计划维度表列给评卷人看,要求它只在这几维里打分(不许发明新维度)。"""
    lines = [f"- {d.name}:{d.why}" for d in plan.dimensions]
    return "\n".join(lines)


def _build_score_messages(plan: InterviewPlan, question: str, answer: str) -> list[dict]:
    return [
        {"role": "system", "content": (
            "你是严格的 AI 面试评卷人。你按给定的评分维度给候选人的这一问打分。"
            "红线:① 只许在给定维度里打分,不许发明新维度;② 打分必须有依据——"
            "候选人的原话里如果只有'我很强/我负责了'这类没有数字、没有做法、没有一手细节"
            "的说法,必须判低分并在 gap 里写明'这句没证据,会被追问穿';"
            "③ 你不安慰人,该低就低,评语具体、可复核。"
        )},
        {"role": "user", "content": f"""
请给下面这位候选人的回答打分。输出一个 JSON 对象(json)。

【评分维度(只许用这些名字)】
{_dimensions_for_prompt(plan)}

【面试官问的问题】
{question}

【候选人的回答】
{answer}

【输出格式(json 对象)】
{{
  "overall": 1到10的整数,
  "dimensions": [
    {{"dimension": "必须是上面维度名之一", "score": 1到5, "comment": "短评", "gap": "扣分点/会穿帮处(没问题就空串)"}}
  ],
  "strengths": ["有依据的优点"],
  "weaknesses": ["有依据的弱点,别泛泛"],
  "verdict": "一句话结论:这回答稳不稳、下题该往哪补"
}}

规则:每一条给定维度都要评到;给分要吝啬,拿不准就低一分。
"""},
    ]


def _normalize(plan: InterviewPlan, schema: ScoreSchema) -> TurnScore:
    """把 pydantic 结果收成内部 TurnScore,并做一层"防幻觉"清洗:
    模型若多报了计划里没有的维度名 → 丢掉(不能让它偷偷加戏)。
    """
    known = {d.name for d in plan.dimensions}
    dims = [
        ScoreDim(dimension=d.dimension, score=d.score, comment=d.comment, gap=d.gap)
        for d in schema.dimensions
        if d.dimension in known          # 只认计划里的维度
    ]
    return TurnScore(
        overall=schema.overall,
        dimensions=dims,
        strengths=list(schema.strengths),
        weaknesses=list(schema.weaknesses),
        verdict=schema.verdict,
    )


# ---------------- 主入口 ----------------

def score_answer(plan: InterviewPlan, question: str, answer: str, *,
                 llm=None, model: str | None = None) -> TurnScore:
    """评卷:对'这一问题+这一答'给一份结构化分。llm 传 FakeClient 则可离线测试。"""
    client = llm if llm is not None else get_client()
    model = model or config.DEFAULT_MODEL   # 评分逐题高频 → flash 就够(便宜快)
    schema = parse_with_retry(
        client, model, _build_score_messages(plan, question, answer),
        schema=ScoreSchema, kind="score", max_tokens=config.MAX_TOKENS_SCORE,
    )
    ts = _normalize(plan, schema)
    ts.model = model
    return ts
