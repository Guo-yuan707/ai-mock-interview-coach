"""第 1 课测试:确定性出题器 planner 的纯函数测试(不碰真实 API / 真实资料)。"""

from interview_coach import planner

# 合成 JD:故意点名几类能力,方便断言
JD_TEXT = """
AI 应用研发工程师
负责基于大模型(LLM/Agent)的 RAG 问答与多轮对话开发;
要求 Python 功底与 pytest;熟悉向量库(chromadb)与 embedding;
加分:SQLite、Streamlit、Eval 评测。
"""

# 合成简历:技术面较窄,故意漏掉 JD 的 RAG/向量库,并埋了一句没数字的夸大句
RESUME_TEXT = """
熟练 Python,熟悉 pytest;
负责过一个小工具的开发,提升团队效率;
这个项目让我精通了大模型 API 调用的所有细节。
"""


def test_analyze_jd_returns_detected_categories():
    """JD 里提到的类别要被识别出来,没提的(如软素质)不应硬凑。"""
    cats = planner.analyze_jd(JD_TEXT)
    assert "大模型应用" in cats
    assert "RAG 与知识库" in cats
    assert "语言与工程" in cats
    assert "软素质" not in cats          # 不能空口出题


def test_build_plan_with_resume_gives_5_plus_risks():
    """验收:JD+简历 → ≥5 个高危追问点,评分维度非空,岗位方向有提示。"""
    plan = planner.build_plan(JD_TEXT, RESUME_TEXT)
    assert plan.n_risks >= 5, f"高危点应≥5,实际 {plan.n_risks}"
    assert plan.n_dimensions >= 2
    assert plan.target_hint  # 有方向提示


def test_missing_jd_keyword_detected():
    """JD 高频要求、简历没写的关键词,要被抓成 missing_jd_keyword。"""
    plan = planner.build_plan(JD_TEXT, RESUME_TEXT)
    kinds = [r.kind for r in plan.risk_points]
    assert "missing_jd_keyword" in kinds
    titles = [r.title for r in plan.risk_points]
    assert any("rag" in t.lower() or "向量" in t for t in titles), "简历没写 RAG/向量库应被抓出来"


def test_unevidenced_claim_flagged():
    """没数字支撑的夸大句,要被抓成 unevidenced_claim(会被追问穿)。"""
    plan = planner.build_plan(JD_TEXT, RESUME_TEXT)
    bad = [r for r in plan.risk_points if r.kind == "unevidenced_claim"]
    assert bad, "应至少抓到一句'无证据'的夸大句"
    assert any("精通" in r.evidence for r in bad), "简历里'精通大模型 API…'这句应被当无证据句抓出"


def test_no_resume_uses_jd_probe():
    """没有简历时,用 JD 高频点出探针题,且仍然 ≥5。"""
    plan = planner.build_plan(JD_TEXT, None)
    assert plan.n_risks >= 5
    assert all(r.kind == "jd_probe" for r in plan.risk_points)


def test_plan_is_deterministic():
    """同一份输入 → 同一份计划(确定性,可回归)。"""
    a = planner.build_plan(JD_TEXT, RESUME_TEXT)
    b = planner.build_plan(JD_TEXT, RESUME_TEXT)
    assert [r.title for r in a.risk_points] == [r.title for r in b.risk_points]
    assert [d.name for d in a.dimensions] == [d.name for d in b.dimensions]
