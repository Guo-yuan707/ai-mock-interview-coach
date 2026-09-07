"""SQLite 存储(第 9 课):整场会话、每轮问答与评分、每笔 LLM 账单都落库,可回放。

为什么做持久化(讲得出 why):
- 前面所有轮次都在内存里,一关程序就没了 —— 面试复盘、报告导出无从谈起;
- 落库后能:看历史清单 → 点开某一场完整回放(问过什么、答过什么、每维几分)→ 重新导出报告;
- SQLite 是单文件数据库,零安装、随项目走,够这个量级用(也呼应 JD 常写的 SQLite)。

表设计三张:
    sessions   一场面试的主档(JD/简历/计划/模式/结束原因)
    turns      每一轮:问题、回答、该轮评分
    usage      每笔大模型调用的账单(tokens/毫秒/成本)
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from . import config
from .plan import plan_from_dict, plan_to_dict


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False)


class InterviewStore:
    """对 SQLite 的一层薄封装:建库建表 + 存/查/回放。"""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path else config.DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    # ---------- 建表 ----------
    def _create_tables(self) -> None:
        cur = self._conn
        cur.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            created_at REAL,
            jd TEXT, resume TEXT,
            model TEXT, planner_mode TEXT,
            finished INTEGER, finish_reason TEXT,
            plan_json TEXT
        );
        CREATE TABLE IF NOT EXISTS turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            seq INTEGER, risk_id TEXT,
            question TEXT, answer TEXT,
            ttft_ms REAL, tool_calls TEXT,
            overall REAL, avg REAL, score_json TEXT
        );
        CREATE TABLE IF NOT EXISTS usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            kind TEXT, model TEXT,
            prompt_tokens INTEGER, completion_tokens INTEGER,
            latency_ms REAL, ttft_ms REAL, cost_yuan REAL
        );
        CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
        """)
        self._conn.commit()

    # ---------- 存 ----------
    def save_run(self, state, summary: dict) -> str:
        """整场落库(主档 + 每轮 + 账单)。返回 session_id。"""
        sid = state.session_id
        self._conn.execute(
            "INSERT OR REPLACE INTO sessions "
            "(id, created_at, jd, resume, model, planner_mode, finished, finish_reason, plan_json) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (sid, state.created_at, state.jd, state.resume, summary.get("model", ""),
             state.planner_mode, 1 if state.finished else 0, state.finish_reason,
             _json(plan_to_dict(state.plan))),
        )
        for t in state.turns:
            sc = t.score
            self._conn.execute(
                "INSERT INTO turns (session_id, seq, risk_id, question, answer, ttft_ms, "
                "tool_calls, overall, avg, score_json) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (sid, t.seq, t.risk_id, t.question, t.answer, t.ttft_ms,
                 _json(t.tool_calls),
                 (sc.overall if sc else None),
                 (sc.avg if sc else None),
                 (_json(self._score_to_dict(sc)) if sc else None)),
            )
        for rec in summary.get("usage_records", []):   # 由调用方从账本快照切出来塞进来
            self._conn.execute(
                "INSERT INTO usage (session_id, kind, model, prompt_tokens, completion_tokens, "
                "latency_ms, ttft_ms, cost_yuan) VALUES (?,?,?,?,?,?,?,?)",
                (sid, rec.kind, rec.model, rec.prompt_tokens, rec.completion_tokens,
                 rec.latency_ms, rec.ttft_ms, rec.cost_yuan),
            )
        self._conn.commit()
        return sid

    @staticmethod
    def _score_to_dict(sc) -> dict:
        """把 TurnScore 摊成 json(维度/优缺点/评语都留全,回放才讲得清)。"""
        if sc is None:
            return {}
        return {
            "overall": sc.overall,
            "dimensions": [{"dimension": d.dimension, "score": d.score,
                            "comment": d.comment, "gap": d.gap} for d in sc.dimensions],
            "strengths": list(sc.strengths),
            "weaknesses": list(sc.weaknesses),
            "verdict": sc.verdict,
        }

    # ---------- 查 / 回放 ----------
    def list_sessions(self) -> list[dict[str, Any]]:
        """历史清单(不带大字段,列表页用):每场一行概要。"""
        rows = self._conn.execute(
            "SELECT s.id, s.created_at, s.model, s.planner_mode, s.finished, s.finish_reason, "
            " (SELECT COUNT(*) FROM turns t WHERE t.session_id = s.id) AS n_turns, "
            " (SELECT AVG(overall) FROM turns t WHERE t.session_id = s.id AND overall IS NOT NULL) AS avg "
            "FROM sessions s ORDER BY s.created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def load_session(self, session_id: str) -> dict[str, Any] | None:
        """读一整场(重建计划 + 逐轮问答评分),网页"历史回放"页用。"""
        s = self._conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if s is None:
            return None
        turns = self._conn.execute(
            "SELECT * FROM turns WHERE session_id=? ORDER BY seq", (session_id,)).fetchall()
        return {
            "meta": dict(s),
            "plan": plan_from_dict(json.loads(s["plan_json"] or "{}")),
            "turns": [self._turn_dict(t) for t in turns],
            "usage": [dict(u) for u in self._conn.execute(
                "SELECT * FROM usage WHERE session_id=? ORDER BY id", (session_id,)).fetchall()],
        }

    @staticmethod
    def _turn_dict(t) -> dict:
        d = dict(t)
        d["tool_calls"] = json.loads(d.get("tool_calls") or "[]")
        d["score"] = json.loads(d.get("score_json") or "{}")
        return d

    def delete_session(self, session_id: str) -> None:
        self._conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        self._conn.execute("DELETE FROM turns WHERE session_id=?", (session_id,))
        self._conn.execute("DELETE FROM usage WHERE session_id=?", (session_id,))
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def format_timestamp(epoch: float) -> str:
    """把时间戳变成可读的中文时间(历史列表用)。"""
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(epoch))
