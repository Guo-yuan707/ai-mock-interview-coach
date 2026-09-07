"""确定性版出题器 v0(第 1 课):JD(+简历) → 评分维度表 + 高危追问清单。

为什么第 1 课先做"确定性版"(讲得出 why):
- 不调大模型 → 不要 API key、跑得快、结果稳定、每一条都有原文依据可查证(红线);
- 方便写单元测试:同一份 JD 永远出同一份计划;
- 第 6 课会换"AI 出题":只替换本模块内部,对外返回的 InterviewPlan 形状不变。

诚实声明:关键词命中是"笨办法",可能误伤/漏判;
真正按语义重要性出题在第 6 课交给 AI 后,本文件仍留作"确定性基线"对比。
"""
import re
import threading

from . import config
from .loader import read_text
from .llm import get_client
from .plan import InterviewPlan, RiskPoint, RubricDimension
from .schemas import PlanSchema
from .structured import parse_with_retry

# 能力类别词典:类别名 → 该类的关键词(中英都行,命中按"子串包含")
# 设计:每个类别 = 一个潜在的"评分维度";JD 提到哪几类,评分表就含哪几类。
LEXICON: dict[str, list[str]] = {
    "语言与工程": ["python", "pytest", "sqlite", "fastapi", "类型注解", "重构", "可测试"],
    "大模型应用": ["llm", "大模型", "gpt", "claude", "prompt", "提示词", "token", "流式",
                  "function calling", "工具调用", "多模态"],
    "RAG 与知识库": ["rag", "向量", "embedding", "向量库", "chromadb", "faiss", "检索",
                    "rerank", "重排", "混合检索", "知识库"],
    "Agent 与编排": ["agent", "智能体", "langgraph", "多智能体", "多轮对话", "上下文管理",
                    "工具链", "规划", "反思"],
    "系统与工程化": ["streamlit", "部署", "docker", "api", "异步", "高可用", "成本", "评测",
                    "eval", "可观测", "trace"],
    "软素质": ["沟通", "自驱", "责任心", "团队", "owner", "抗压"],
}

# 每类命中后,转成面试评分维度时用的模板话术
_CATEGORY_TO_DIM_WHY = {
    "语言与工程": "JD 要求语言与工程功底(出现了 {kws}),面试官会从代码/测试/设计上追深度。",
    "大模型应用": "JD 围绕大模型应用({kws}),会追问你对提示词/调用/边界是否真懂。",
    "RAG 与知识库": "JD 点明检索/向量相关能力({kws}),会深挖你 RAG 的落地细节与缺陷。",
    "Agent 与编排": "JD 涉及 Agent/多轮对话({kws}),会追问你的方案为什么这么设计。",
    "系统与工程化": "JD 提到工程化/上线({kws}),会关心部署、稳定、成本意识。",
    "软素质": "JD 强调软素质({kws}),会通过行为问题(如 STAR)考察。",
}

# 常见"夸大词/无证据"信号(出现在简历句子里、却无数字支撑时 = 高危)
_CLAIM_WORDS = ["精通", "熟练掌握", "熟练", "熟悉", "负责", "主导", "独立开发",
                "核心成员", "大幅", "显著", "提升", "优化", "保证", "确保"]

# 判断一行里有没有"数字证据"(百分号/具体量级都算)
_HAS_NUMBER = re.compile(r"[0-9]|%")


def analyze_jd(jd_text: str) -> dict[str, list[str]]:
    """把 JD 文本按能力类别归类出命中的关键词(空类=JD 没提,不出题)。"""
    hay = jd_text.lower()
    matched: dict[str, list[str]] = {}
    for category, keywords in LEXICON.items():
        hits = [kw for kw in keywords if kw.lower() in hay]
        if hits:
            matched[category] = hits
    return matched


def build_plan(jd_text: str, resume_text: str | None = None) -> InterviewPlan:
    """主入口:JD(+可选简历) → 一份面试计划(维度表 + ≥5 个风险点)。"""
    jd = read_text(jd_text)
    resume = read_text(resume_text) if resume_text else ""
    categories = analyze_jd(jd)

    target_hint = _guess_target(categories)
    dimensions = _build_dimensions(categories)
    risk_points = _build_risks(categories, resume)

    return InterviewPlan(
        target_hint=target_hint,
        jd_categories=categories,
        dimensions=dimensions,
        risk_points=_pad_to_at_least(risk_points, 5, categories),
    )


# ---------- 内部小工具 ----------

def _build_dimensions(categories: dict[str, list[str]]) -> list[RubricDimension]:
    """评分维度 = 通用维度(永远在)+ JD 命中的技术类别维度。"""
    dims = [
        RubricDimension("岗位理解与匹配", "是否真读懂 JD 在招什么人、要什么能力", []),
        RubricDimension("表达与逻辑", "回答是否结构清晰、先结论后展开", []),
    ]
    for category, keywords in categories.items():
        dims.append(RubricDimension(
            name=f"技术深度({category})",
            why=_CATEGORY_TO_DIM_WHY.get(category, "JD 提到该能力类别。").format(kws="、".join(keywords)),
            source_keywords=list(keywords),
        ))
    return dims


def _build_risks(categories: dict[str, list[str]], resume: str) -> list[RiskPoint]:
    """根据有没有简历,分别找风险点。"""
    if not resume:
        return _probe_from_jd(categories)

    risks: list[RiskPoint] = []
    risks += _missing_jd_keywords(categories, resume)      # 简历没写的 JD 要求
    risks += _unevidenced_claims(resume)                   # 没数字的夸大句
    risks.append(RiskPoint(
        kind="resume_probe",
        title="每个项目都会被要求按 STAR 展开",
        detail="挑简历里最满的一句,先问'结果可量化吗',再问一手细节(你具体干了哪步、别人干了哪步)。",
        evidence="",
    ))
    return risks


def _probe_from_jd(categories: dict[str, list[str]]) -> list[RiskPoint]:
    """没有简历 → 把 JD 高频点当考题目标,出'请你讲 X'式追问。"""
    risks = []
    for category, keywords in categories.items():
        for kw in keywords[:1]:          # 每类先取第一个关键词当探针
            risks.append(RiskPoint(
                kind="jd_probe",
                title=f"请你讲讲「{kw}」",
                detail=f"JD 反复要求 {category} 里的 {kw}。先问概念与适用场景,再问'你实际用过吗、遇到什么问题'。",
                evidence=kw,
            ))
    return risks


def _missing_jd_keywords(categories: dict[str, list[str]], resume: str) -> list[RiskPoint]:
    """简历对照 JD:JD 高频要求、简历却完全没提的关键词 = 最可能的追问洞。"""
    hay = resume.lower()
    risks = []
    for category, keywords in categories.items():
        missing = [kw for kw in keywords if kw.lower() not in hay]
        for kw in missing[:2]:           # 每类最多挑 2 个,别刷屏
            risks.append(RiskPoint(
                kind="missing_jd_keyword",
                title=f"简历未体现 JD 高频点:{kw}",
                detail=f"JD 在「{category}」要求 {kw},简历通篇没出现 → 面试官大概率问:'这块你了解/用过吗?'",
                evidence=kw,
            ))
    return risks


def _unevidenced_claims(resume: str) -> list[RiskPoint]:
    """扫简历句子:有夸大词、却没有数字支撑的句子 = '会被追问穿'的高危句。"""
    risks = []
    for line in resume.splitlines():
        line = line.strip()
        if len(line) < 6:                 # 太短的(标题/空行)不判
            continue
        if any(w in line for w in _CLAIM_WORDS) and not _HAS_NUMBER.search(line):
            risks.append(RiskPoint(
                kind="unevidenced_claim",
                title=f"这句没有证据,会被追问穿:「{line[:28]}…」" if len(line) > 28 else f"这句没有证据,会被追问穿:「{line}」",
                detail="只写了能力/功劳,没给数字或做法 → 会被追:'具体提升多少?怎么做到的?一手细节在哪?'",
                evidence=line,
            ))
    return risks[:4]   # v0 先只挑前 4 句,别淹没输出;第 6 课 AI 版会按严重度排序


def _pad_to_at_least(risks: list[RiskPoint], minimum: int,
                     categories: dict[str, list[str]]) -> list[RiskPoint]:
    """凑数到 ≥minimum 条:还不够就用 JD 高频点补探针题(别让验收差一条)。"""
    if len(risks) >= minimum:
        return risks
    seen = {r.title for r in risks}
    for category, keywords in categories.items():
        for kw in keywords:
            title = f"深挖 JD 高频点:{kw}"
            if title in seen:
                continue
            seen.add(title)
            risks.append(RiskPoint(
                kind="jd_probe",
                title=title,
                detail=f"JD 反复出现 {category} 的关键词 {kw},准备好'是什么/怎么做/踩过什么坑'三层答法。",
                evidence=kw,
            ))
            if len(risks) >= minimum:
                return risks
    return risks


def _guess_target(categories: dict[str, list[str]]) -> str:
    """从命中的类别粗猜岗位方向(仅提示,不作事实,红线)。"""
    if "Agent 与编排" in categories or "大模型应用" in categories:
        return "偏 AI Agent / 大模型应用方向(依据:JD 命中 Agent 与大模型类别)"
    if categories:
        return "偏工程开发方向(依据:JD 命中的类别为 " + "、".join(list(categories)[:3]) + ")"
    return ""


# =====================================================================
# AI 版出题(第 6 课):读同一份 JD/简历,让大模型按语义出计划。
# 交接点:对外仍返回 InterviewPlan —— 和确定性版同形状,
# 所以调用方(面试引擎/网页)根本不用知道现在是哪种出题器。
# 确定性版留在上面当"基线 / 兜底 / 离线演示",两者可对比。
# =====================================================================

def _maybe_llm(llm, *, timeout_s: float | None = None,
               max_retries: int | None = None):
    """测试传 FakeClient;没传就取真客户端。
    AI 出题(plan)给独立超时/重试策略:pro 长生成健康时也要 1~2 分钟 → 180s 超时;
    且不重试 → 后端真故障时 ~3 分钟内失败,由 build_plan_auto 降级,别让用户干等。
    """
    return llm if llm is not None else get_client(timeout_s=timeout_s,
                                                  max_retries=max_retries)


def _build_ai_plan_messages(jd: str, resume: str) -> list[dict]:
    """拼 AI 出题的 prompt。必须出现 'json' 字样(DeepSeek json 模式要求)。"""
    resume_section = resume if resume.strip() else "（未提供简历——那就出 JD 专业知识题,不挖简历）"
    return [
        {"role": "system", "content": (
            "你是资深 AI 面试官教练,负责根据岗位 JD(必填)和候选人简历(可选),"
            "产出一份可执行的模拟面试计划。你严格按 JSON 输出,不夹带任何解释。"
            "红线:所有判断必须 grounded 在原文;拿不准的 evidence 写空串,绝不编造原文。"
        )},
        {"role": "user", "content": f"""
请为下面的岗位设计一份模拟面试计划,输出一个 JSON 对象(不要 markdown 代码块)。

【岗位 JD】
{jd}

【候选人简历】
{resume_section}

【输出格式要求(json 对象,字段如下)】
{{
  "target_hint": "一句话方向提示,依据 JD 的关键词,仅提示不作事实",
  "dimensions": [
    {{"name": "维度名", "why": "为什么列这一维(源自 JD 哪些信号)", "source_keywords": ["触发它的JD关键词"]}}
  ],
  "risk_points": [
    {{"kind": "missing_jd_keyword|unevidenced_claim|jd_probe|resume_probe", "title": "一句话标题", "detail": "会怎么追问、风险在哪", "evidence": "依据原文片段(没有就空串)"}}
  ]
}}

【硬性要求】
1. dimensions 至少 2 条:请始终包含『岗位理解与匹配』『表达与逻辑』两条通用维度,
   其余维度从 JD 实际提到的技术能力里提炼(如 技术深度(RAG)),JD 没提的类别不要硬加;
2. risk_points 至少 5 条,按命中率从高到低;
3. 有简历时:重点抓 JD 高频要求但简历没写的(missing_jd_keyword)、
   以及简历里『有夸大词却没数字证据』的句子(unevidenced_claim);
4. 没简历时:全部出 JD 专业知识探针题(jd_probe),如『请讲讲 RAG 的检索流程与失败场景』;
5. evidence 字段必须逐字引用原文(可截断),找不到原文证据就写空串;
6. detail 要具体到"面试官现场会怎么追问",不是空话套话。
"""},
    ]


def build_plan_ai(jd_text: str, resume_text: str | None = None, *,
                  llm=None, model: str | None = None) -> InterviewPlan:
    """AI 版主入口:JD(+简历)→ 用大模型出面试计划(仍是 InterviewPlan 形状)。

    参数约定和 build_plan 一致:jd_text / resume_text 既可以是文件路径,
    也可以是直接粘贴的文本(loader.read_text 会自动判断)。
    """
    jd = read_text(jd_text)
    resume = read_text(resume_text) if resume_text else ""
    client = _maybe_llm(llm, timeout_s=config.PLAN_TIMEOUT_S,
                        max_retries=config.PLAN_MAX_RETRIES)
    model = model or config.PRO_MODEL  # 出整份计划重质量 → 用 pro

    schema = parse_with_retry(
        client, model, _build_ai_plan_messages(jd, resume),
        schema=PlanSchema, kind="plan", max_tokens=config.MAX_TOKENS_PLAN,
    )

    dims = [
        RubricDimension(name=d.name, why=d.why, source_keywords=list(d.source_keywords))
        for d in schema.dimensions
    ]
    risks = [
        RiskPoint(kind=r.kind, title=r.title, detail=r.detail, evidence=r.evidence)
        for r in schema.risk_points
    ]
    return InterviewPlan(target_hint=schema.target_hint,
                         jd_categories={},  # AI 版不维护关键词分类,留空(形状兼容)
                         dimensions=dims, risk_points=risks)


def _ai_plan_with_wallclock(jd_text: str, resume_text: str | None, *, llm=None,
                            timeout_s: float) -> InterviewPlan | None:
    """在 daemon 线程里跑 AI 出题,超过 timeout_s(硬墙钟)就放弃。

    为什么不能只靠 SDK 超时:DeepSeek 客户端超时是"空闲超时"——后端若缓慢但持续地吐字,
    永远等不到超时触发。这里用 daemon 线程 + join(timeout) 保证**铁定**在 timeout_s 内返回:
    返回 None = 超时或出错(后台线程自行消亡,不阻塞解释器退出),调用方据此降级。
    """
    box: dict = {}

    def run() -> None:
        try:
            box["plan"] = build_plan_ai(jd_text, resume_text, llm=llm)
        except Exception as e:            # AI 出题失败不阻塞面试,交给调用方降级
            box["err"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    if t.is_alive():
        return None                       # 超时:放弃这次 AI 出题(线程后台自灭)
    if "err" in box:
        return None
    return box.get("plan")


def build_plan_auto(jd_text: str, resume_text: str | None = None, *,
                    use_ai: bool | None = None, llm=None) -> tuple[InterviewPlan, str]:
    """供入口层(CLI/网页)用的总开关:
    - use_ai=True 且真模型可用 → AI 出题(带硬墙钟,超时自动放弃);
    - 失败/超时或 use_ai=False → 确定性版兜底(离线照样能跑)。
    返回 (plan, mode说明),mode 取值 'ai' 或 'deterministic',方便展示/记录。
    """
    if use_ai is None:
        # 自动模式:默认想用 AI,但只有 key 真可用才上(试一下,不行就降级)
        use_ai = _try_client_available()
    if use_ai:
        plan = _ai_plan_with_wallclock(jd_text, resume_text, llm=llm,
                                       timeout_s=config.PLAN_WALL_TIMEOUT_S)
        if plan is not None:
            return plan, "ai"
    return build_plan(jd_text, resume_text), "deterministic"


def _try_client_available() -> bool:
    """探测真客户端是否可用(有 key 且连通)。失败不抛,返回 False。"""
    try:
        get_client()
        return True
    except Exception:
        return False
