"""会话状态(第 2 课):一场面试"现在进行到哪"的唯一真相。

为什么单独一个 state(讲得出 why):
- 多轮对话最容易出的事 = 状态散落各处,一轮没存、下轮就失忆;
- 这里集中存:背景(JD/简历/计划,几乎不变的"长记忆")+
  进行中(问过哪些题、每轮问答与评分、谁高危点已覆盖);
- engine / interviewer / evaluator / tools 都只跟这一个对象打交道,
  谁都不自己私藏一份"我以为问过什么"。

状态分层(对应第 2 课 Context Engineering 的"长短记忆"思想):
    长记忆  = jd / resume / plan(几乎不动,回回都要带,但只读不裁)
    短记忆  = turns(最近几轮,满了就把更早的折叠成摘要,防上下文超预算)
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from .plan import InterviewPlan, RiskPoint
from . import config


def risk_id(i: int) -> str:
    """风险点编号:第 i 个 → 'r{i}'。计划本身不带 id,用下标当稳定编号。"""
    return f"r{i}"


@dataclass
class Turn:
    """一问一答一评分,组成一轮。"""
    seq: int
    risk_id: str                     # 这一问瞄准哪个高危点(可能为 '' 表示自由问)
    question: str
    answer: str = ""                 # 还没答时为空(先提问、后补答,两拍)
    score: object | None = None      # evaluator.TurnScore(延迟 import,避免环)
    ttft_ms: float | None = None     # 这题的流式首字延迟(第 3 课指标)
    tool_calls: list[str] = field(default_factory=list)  # 问之前面试官调过哪些工具

    @property
    def answered(self) -> bool:
        return bool(self.answer.strip())


@dataclass
class SessionState:
    """一场模拟面试的全部现场。"""
    jd: str = ""
    resume: str = ""
    plan: InterviewPlan = field(default_factory=InterviewPlan)
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)
    turns: list[Turn] = field(default_factory=list)
    ask_counts: dict[str, int] = field(default_factory=dict)  # risk_id → 问过几次
    finished: bool = False
    finish_reason: str = ""
    model: str = ""                  # 实际用的会话模型
    planner_mode: str = "deterministic"  # plan 是 ai 还是 deterministic 出的
    last_grounding: str = ""         # 面试官最近一次检索到的原文(措辞阶段可引用,可查证)

    # ---------- 高危点定位 ----------
    def risk(self, rid: str) -> RiskPoint | None:
        if not rid:
            return None
        try:
            i = int(rid[1:]) if rid.startswith("r") else int(rid)
            if 0 <= i < len(self.plan.risk_points):
                return self.plan.risk_points[i]
        except ValueError:
            return None
        return None

    def remaining_risks(self) -> list[RiskPoint]:
        """还没被追问过(或追问不足)的高危点,面试官从这里挑下一个。"""
        return [rp for i, rp in enumerate(self.plan.risk_points)
                if self.ask_counts.get(risk_id(i), 0) < config.MAX_FOLLOWUPS_PER_RISK]

    def coverage(self) -> dict:
        """高危覆盖统计:问过几条 / 共几条,占百分之几(报告 + 进度工具用)。"""
        total = len(self.plan.risk_points) or 1
        asked = sum(1 for i in range(len(self.plan.risk_points))
                    if self.ask_counts.get(risk_id(i), 0) > 0)
        return {"asked": asked, "total": len(self.plan.risk_points), "rate": round(asked / total, 3)}

    # ---------- 短记忆 ----------
    def recent_turns(self, n: int | None = None) -> list[Turn]:
        n = n or config.SHORT_MEMORY_TURNS
        return self.turns[-n:]

    def last_turn(self) -> Turn | None:
        return self.turns[-1] if self.turns else None

    def register_ask(self, rid: str) -> None:
        self.ask_counts[rid] = self.ask_counts.get(rid, 0) + 1

    def next_seq(self) -> int:
        return len(self.turns) + 1


def format_recent_history(state: SessionState, max_tokens: int | None = None) -> str:
    """把最近几轮问答+评分压成给模型的紧凑摘要。

    max_tokens 若给定,则从新到旧裁剪,直到不超预算(第 2 课上下文预算)。
    只裁"短记忆",长记忆(JD/计划)由调用方决定带不带。
    """
    from .llm import estimate_tokens
    from .evaluator import TurnScore

    limit = max_tokens or config.CONTEXT_MAX_TOKENS // 3
    blocks: list[str] = []
    used = 0
    for t in reversed(state.turns):
        lines = [f"面试官问:{t.question}"]
        if t.answer:
            lines.append(f"候选答:{t.answer}")
        if isinstance(t.score, TurnScore):
            lines.append(f"评卷:{t.score.overall}/10 维度均值{t.score.avg} 弱项:{'、'.join(t.score.weak_dimensions) or '无'}")
            if t.score.verdict:
                lines.append(f"评语:{t.score.verdict}")
        block = "\n".join(lines)
        cost = estimate_tokens(block) + 20
        if used + cost > limit and blocks:
            break                       # 预算不够塞更早的轮次了
        blocks.insert(0, block)         # 逆序收集再正序拼接
        used += cost
    return "\n\n".join(blocks) if blocks else "（还没有问答发生）"


def format_risk_roster(state: SessionState) -> str:
    """把高危点清单连同"问过没问过"整理成面试官一眼能扫的名单。"""
    lines = []
    for i, rp in enumerate(state.plan.risk_points):
        rid = risk_id(i)
        asked = state.ask_counts.get(rid, 0)
        mark = "已问" if asked else "未问"
        flag = " 🔁已追" + str(asked) + "次" if asked > 1 else ""
        lines.append(f"[{rid}|{mark}{flag}] {rp.title} —— 依据:{rp.detail[:40]}")
    return "\n".join(lines)
