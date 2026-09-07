"""会话引擎(第 2 课):把"面试官 ↔ 候选人 ↔ 评卷人"装进一台状态机。

它不关心问题怎么措辞、分怎么打——那是 interviewer / evaluator 的事;
它只做编排:
    ask_next()   → 让面试官走一步(可能调工具)→ 得到一个问题(流式吐给你)
    submit()     → 收到候选人的回答 → 交给评卷人打分 → 存进会话状态
这样 UI(CLI / 网页)只需循环调用这两个方法,核心逻辑一份,两头共用(双入口)。

上下文管理在这里落地:
    长短记忆分层 + 预算(发给模型的上下文由 state 里的裁剪函数控制)。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import config, evaluator, interviewer
from . import planner as planner_mod
from .llm import LLMClient, get_client, usage_snapshot
from .plan import InterviewPlan, RiskPoint
from .retrieval import LexicalKB, build_default_kb
from .state import SessionState, Turn, risk_id


@dataclass
class SessionConfig:
    """一场面试的参数(集中、可改):默认读 config.py,网页/CLI 可覆盖。"""
    model: str = config.DEFAULT_MODEL
    max_turns: int = config.MAX_INTERVIEW_TURNS


class InterviewSession:
    """一场模拟面试的编排器。用法:
        sess = InterviewSession(jd=..., resume=...)
        while (q := sess.ask_next()):
            show(q.question)
            score = sess.submit(input())
        report = sess.summarize()
    """

    def __init__(self, jd: str, resume: str | None = None,
                 plan: InterviewPlan | None = None,
                 planner_mode: str = "deterministic",
                 cfg: SessionConfig | None = None,
                 llm: LLMClient | None = None,
                 kb: LexicalKB | None = None) -> None:
        self.plan = plan
        if self.plan is None:
            # 没给计划就现场出:有真模型且能连就用 AI,否则确定性版(永不阻塞)
            self.plan, planner_mode = planner_mod.build_plan_auto(jd, resume)
        self.state = SessionState(jd=jd or "", resume=resume or "", plan=self.plan,
                                  model=(cfg.model if cfg else config.DEFAULT_MODEL),
                                  planner_mode=planner_mode)
        self.cfg = cfg or SessionConfig()
        self.llm = llm if llm is not None else get_client()
        self.kb = kb if kb is not None else build_default_kb(self.state)
        self._pending: Turn | None = None
        # 账本起始游标:只把"这场面试期间"产生的调用记到本场头上
        # (同进程开多场(网页)时,避免把上一场的账单也算进来)
        self._usage_start = len(usage_snapshot())

    # ---------- 编排主循环的两个动作 ----------

    def ask_next(self, on_delta=None) -> Turn | None:
        """让面试官走一步。返回待答的 Turn(需要你自己把 on_delta 收到的问题给用户);
        若面试结束/达上限 → 返回 None(调用方应去 summarize)。"""
        if self.state.finished:
            return None
        if len(self.state.turns) >= self.cfg.max_turns:
            self._finish("已达到本轮最大轮数上限,面试结束")
            return None

        decision = interviewer.agent_decide(self.llm, self.cfg.model, self.state, kb=self.kb)
        if decision.kind == "finish":
            self._finish(decision.reason or "面试官认为追问已充分")
            return None

        rid = self._pick_risk(decision.risk_id)
        risk = self.state.risk(rid)
        if rid:
            self.state.register_ask(rid)
        q_text, ttft = interviewer.phrase_question(
            self.llm, self.cfg.model, self.state, risk,
            follow_up=decision.follow_up, on_delta=on_delta)
        if not q_text.strip():        # 措辞兜底:空话就退化为从计划里挖一句
            q_text = self._fallback_question(risk, decision.follow_up)
        turn = Turn(seq=self.state.next_seq(), risk_id=rid, question=q_text.strip(),
                    ttft_ms=ttft, tool_calls=list(decision.tool_calls_used))
        self._pending = turn
        return turn

    def submit(self, answer: str) -> evaluator.TurnScore | None:
        """提交候选人回答 → 评卷 → 落状态。返回本题评分(会话结束/无待答题时为 None)。"""
        turn = self._pending
        if turn is None:
            return None
        self._pending = None
        turn.answer = (answer or "").strip()
        score = evaluator.score_answer(self.state.plan, turn.question, turn.answer,
                                       llm=self.llm, model=self.cfg.model)
        turn.score = score
        self.state.turns.append(turn)
        return score

    # ---------- 结束 / 汇总 ----------

    def end_early(self, reason: str = "候选人/调用方主动结束") -> None:
        """外部(用户/网页按钮)主动结束面试。"""
        self._finish(reason)

    def _finish(self, reason: str) -> None:
        self.state.finished = True
        self.state.finish_reason = reason

    def summarize(self) -> dict:
        """结案汇总(第 9 课报告的数据基础):维度均分、高危覆盖、弱项清单、账单。"""
        turns = [t for t in self.state.turns if t.score is not None]
        scores = [t.score for t in turns]
        dim_tot: dict[str, list[int]] = {}
        for sc in scores:
            for d in sc.dimensions:
                dim_tot.setdefault(d.dimension, []).append(d.score)
        dims = {name: round(sum(v) / len(v), 2) for name, v in dim_tot.items()}
        weak = sorted({d.dimension for sc in scores for d in sc.dimensions if d.score < 3})
        record = usage_snapshot()[self._usage_start:]   # 只算这一场的调用
        usage = {
            "calls": len(record),
            "prompt_tokens": sum(r.prompt_tokens for r in record),
            "completion_tokens": sum(r.completion_tokens for r in record),
            "total_ms": sum(r.latency_ms for r in record if r.latency_ms),
            "costs": [r.cost_yuan for r in record if r.cost_yuan is not None],
        }
        avg = (sum(t.score.overall for t in turns) / len(turns)) if turns else 0.0
        return {
            "finished": self.state.finished,
            "finish_reason": self.state.finish_reason,
            "turns": len(self.state.turns),
            "average_overall": round(avg, 2),
            "dimension_avg": dims,
            "weak_dimensions": weak,
            "coverage": self.state.coverage(),
            "usage": usage,
            "usage_records": record,          # 原样给 storage 落库用
            "model": self.cfg.model,
            "planner_mode": self.state.planner_mode,
        }

    # ---------- 内部小工具 ----------

    def _pick_risk(self, decision_rid: str) -> str:
        """决定这一问攻哪个高危点:模型指定且没问爆 → 用它的;否则按覆盖度兜底。"""
        if decision_rid and self.state.risk(decision_rid) is not None:
            asked = self.state.ask_counts.get(decision_rid, 0)
            if asked < config.MAX_FOLLOWUPS_PER_RISK:
                return decision_rid
        remaining = self.state.remaining_risks()
        if remaining:
            # 第一个未问满的高危点 = 下一目标
            for i, rp in enumerate(self.state.plan.risk_points):
                if rp in remaining:
                    return risk_id(i)
        # 都问遍了 → 轮一圈问最少的那条,做整合性收尾
        if self.state.plan.risk_points:
            least = min(range(len(self.state.plan.risk_points)),
                        key=lambda i: self.state.ask_counts.get(risk_id(i), 0))
            return risk_id(least)
        return ""

    def _fallback_question(self, risk: RiskPoint | None, follow_up: str) -> str:
        """措辞真的空时,退化成一道还能问的问题(绝不让用户等一个空问题)。"""
        if risk is not None:
            return f"关于「{risk.title}」,{risk.detail}"
        return "结合你刚才的回答,能再展开讲讲你最有把握的一个技术点吗?"
