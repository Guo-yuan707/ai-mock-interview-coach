"""评测指标(纯函数,可单测,不碰 API)。

每个指标都回答一个能讲给面试官的问题:
    theme_coverage      → "AI 出题有没有覆盖 JD 的高危主题?"
    strictness_gap      → "评卷人分得开强弱答案吗?(会不会当老好人)"
    report_card         → 把几项合成一张成绩单
"""
from __future__ import annotations

from typing import Any


def theme_coverage(plan_text: str, themes: list[str]) -> dict:
    """高危主题覆盖率:themes 里有多少条出现在计划的风险点文本里(子串命中)。

    参数:
        plan_text: 把计划所有风险点的 title+detail 拼成的一段文本(小写化再比)
        themes:    该 JD 应当覆盖的主题词(如 ["rag", "向量库"])
    返回:
        {"hit": [...], "miss": [...], "rate": 0~1}
    """
    hay = (plan_text or "").lower()
    hit = [t for t in themes if t.lower() in hay]
    miss = [t for t in themes if t.lower() not in hay]
    rate = round(len(hit) / len(themes), 3) if themes else 0.0
    return {"hit": hit, "miss": miss, "rate": rate}


def plan_risks_text(plan) -> str:
    """把计划的风险点拼成一段可做覆盖统计的文本。"""
    parts = []
    for rp in plan.risk_points:
        parts.append(f"{rp.title} {rp.detail}")
    return "\n".join(parts)


def strictness_gap(strong_overall: float, weak_overall: float) -> dict:
    """评分严格度:强回答的均分应显著高于弱回答。

    gap = strong - weak。gap>=2 视为"分得开"(不吃老好人);0~2 偏松;
    <0 异常(评卷人可能把强弱搞反)。
    """
    gap = round(float(strong_overall) - float(weak_overall), 2)
    verdict = "严格(分得开)" if gap >= 2 else ("偏松(区分度不足)" if gap >= 0 else "异常(强弱倒挂)")
    return {"gap": gap, "verdict": verdict}


def report_card(rows: list[tuple[str, str]]) -> str:
    """把 (项目, 一行结论) 排成一张成绩单文本。"""
    lines = ["评测成绩单", "─" * 46]
    for name, line in rows:
        lines.append(f"  ▸ {name}:{line}")
    return "\n".join(lines)
