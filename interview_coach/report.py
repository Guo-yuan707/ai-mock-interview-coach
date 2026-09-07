"""结案报告(第 9 课导出):一场面试结束,产出"哪被追问穿、下一步往哪补"。

报告回答 PROJECT 定位句的三个问题:
    ① 整体面得怎么样        → 均分 / 各维度均分
    ② 高危覆盖到没          → 问了几条 / 全条里覆盖多少
    ③ 接下来该补哪          → 薄弱维度清单 + 逐轮"gap/评语"可回查

全部数据来自 state(逐轮) + summary(聚合),只做展示,不掺新逻辑。
"""
from __future__ import annotations

import time
from pathlib import Path

from . import config
from .state import SessionState, format_recent_history  # noqa: F401 (未来扩展复用)


def _bars(value: float, width: int = 20) -> str:
    """把 0~10 的分数画成可视横条(终端/网页都好读)。"""
    filled = max(0, min(width, int(round(value / 10 * width))))
    return "█" * filled + "░" * (width - filled)


def build_report_text(state: SessionState, summary: dict) -> str:
    """把一场面试整理成可读报告(markdown,可直接存文件)。"""
    cov = summary.get("coverage", state.coverage())
    usage = summary.get("usage", {})
    dims = summary.get("dimension_avg", {})
    weak = summary.get("weak_dimensions", [])
    costs = usage.get("costs") or []

    lines: list[str] = []
    a = lines.append
    a(f"# AI 模拟面试结案报告")
    a(f"\n> 面试 ID:`{state.session_id}` · 计划模式:{summary.get('planner_mode','-')} · 模型:{summary.get('model','-')}")
    a(f"> 结束原因:{state.finish_reason or '-'}")
    a("")

    a("## 一、整体结论")
    a(f"- 场均分:**{summary.get('average_overall', 0)} / 10** · 共 {summary.get('turns', 0)} 轮")
    if dims:
        a("\n**各维度均分(满分 5):**")
        for name, avg in sorted(dims.items(), key=lambda x: -x[1]):
            score10 = avg / 5 * 10
            a(f"  - {name}:{avg} {_bars(score10)}")
    a(f"\n**高危覆盖:** {cov.get('asked', 0)} / {cov.get('total', 0)} 条 已追问({cov.get('rate', 0):.0%})")

    a("\n## 二、最该补的短板(评卷人标出 <3 分的维度)")
    if weak:
        for w in weak:
            a(f"- ❌ {w}")
    else:
        a("- 没有出现明显薄弱维度(或样本不足)")

    a("\n## 三、逐轮回放(问→答→分)")
    if not state.turns:
        a("- (本场没有产生完整问答)")
    for i, t in enumerate(state.turns, 1):
        a(f"\n### 第 {i} 轮 · 高危点 `{t.risk_id or '自由追问'}`")
        if t.tool_calls:
            a(f"> 📎 面试官先调用了工具:{'、'.join(t.tool_calls)}")
        a(f"**问:** {t.question}")
        a(f"\n**答:** {(t.answer or '—')[:600]}")
        sc = t.score
        if sc is not None:
            dim_line = " · ".join(f"{d.dimension} {d.score}/5" for d in sc.dimensions)
            a(f"\n**评分:{sc.overall}/10**({dim_line})")
            if any(d.gap for d in sc.dimensions):
                gaps = [d.gap for d in sc.dimensions if d.gap]
                a(f"- ⚠️ 会被追问穿:{' | '.join(gaps)}")
            if sc.verdict:
                a(f"- 评语:{sc.verdict}")
        a("")

    a("## 四、账单(第 9 课可观测)")
    total_ms = usage.get("total_ms", 0) or 0
    a(f"- 调用 {usage.get('calls', 0)} 次 · 输入 {usage.get('prompt_tokens', 0)} tok · "
      f"输出 {usage.get('completion_tokens', 0)} tok · 总耗时 {total_ms / 1000:.1f}s")
    if costs:
        a(f"- 估算成本:¥{sum(costs):.4f}(单价来自 .env,未配单价则不计)")
    else:
        a("- 成本:未配置单价(见 .env 的 DEEPSEEK_PRICE_IN/OUT_PER_M),仅统计 token 数")

    a(f"\n---\n*报告生成于 {time.strftime('%Y-%m-%d %H:%M')},由 interview_coach 自动导出*")
    return "\n".join(lines)


def export_report(state: SessionState, summary: dict,
                  output_dir: str | Path | None = None) -> Path:
    """把报告写成 .md 文件。默认存 output/ 下,文件名带场次 id,不覆盖历史。"""
    out_dir = Path(output_dir) if output_dir else config.REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"interview-report-{state.session_id}.md"
    path.write_text(build_report_text(state, summary), encoding="utf-8")
    return path
