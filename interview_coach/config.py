"""集中配置:所有"会变的环境 / 路径 / 模型参数"都收敛在这一处,不散落代码里。

好处(讲得出 why):
- 想改一个地方全局生效,不用满代码找;
- 将来要支持"从命令行 / 网页界面覆盖配置"时,只需改这一处读参数的逻辑。
"""
import os
from pathlib import Path

# 项目根目录 = 本文件所在目录的上一层
# config.py 在 interview_coach/ 里 → parent 是 interview_coach → parent.parent 是项目根
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 数据目录:放语料、SQLite 数据库、评测缓存(先占位,用到哪补哪)
DATA_DIR = PROJECT_ROOT / "data"

# 默认数据库文件(SQLite 落库 + 历史回放 + 报告)
DB_PATH = DATA_DIR / "interview.db"

# 报告导出目录(第 5 课 report / CLI 导出用)
REPORT_DIR = PROJECT_ROOT / "output"

# ================= 公网多访客判定 =================
# 共享 SQLite(历史库)只在"单机/单用户"语义下安全;公网部署时任何访客都能看/删别人
# 的库 → 这类功能默认关。判定:本地开发一般有 .env(读 key);公网仓库不带 .env。
# 想手动覆盖:环境变量 APP_PUBLIC=1 / 0。
def _is_public_deploy() -> bool:
    flag = os.getenv("APP_PUBLIC", "").strip().lower()
    if flag:
        return flag in {"1", "true", "yes"}
    return not (PROJECT_ROOT / ".env").exists()


DEPLOYED_PUBLIC = _is_public_deploy()

# ================= LLM 模型配置 =================
# 双模型策略:决策/评分这类要稳的用 flash 也行;但长文/重质量用 pro。
# 全部走 DeepSeek 的 OpenAI 兼容协议,模型名见 .env / 官方控制台。
DEFAULT_MODEL = "deepseek-v4-flash"   # 快/便宜:追问生成、进度判断
PRO_MODEL = "deepseek-v4-pro"         # 稳/贵:面试计划、逐题评分
MODELS = [DEFAULT_MODEL, PRO_MODEL]   # 网页下拉框用同一份清单

# 温度(0=死板确定,1=发散):
TEMPERATURE_DECISION = 0.2   # 结构化"下一步决策/打分"要稳定可复现 → 低温
TEMPERATURE_GENERATE = 0.7   # 生成追问话术要自然像真人 → 中温

# 各调用类型输出上限(防截断,也防刷 token):
MAX_TOKENS_DECISION = 900    # "下一步" JSON:够短
MAX_TOKENS_QUESTION = 500    # 单条追问文本
MAX_TOKENS_SCORE = 2500      # 逐题评分 JSON(多维度评语较长);思考模式会先吐 reasoning 占 token,留足余量
MAX_TOKENS_PLAN = 4096       # AI 版面试计划 JSON(维度+风险点+原文依据较长,thinking 也占额)

# ================= 上下文/会话预算(第 2 课) =================
# 每次真正发给模型的上下文,压缩在 CONTEXT_MAX_TOKENS 内(超了就裁剪短记忆);
# 长记忆(JD/简历/计划)几乎不动,优先裁剪的永远是"最近几轮对话"。
CONTEXT_MAX_TOKENS = 9000
SHORT_MEMORY_TURNS = 6        # 短记忆默认保留最近 N 轮(一问一答一评分)
MAX_INTERVIEW_TURNS = 12      # 一场最多 N 轮,防无限聊(也是成本红线)
MAX_FOLLOWUPS_PER_RISK = 3    # 同一个高危点最多追 N 次,别死磕一条

# ================= LLM 客户端 =================
LLM_TIMEOUT_S = 90            # 单次请求超时(秒)
LLM_MAX_RETRIES = 2           # 请求失败重试次数(临时故障用)

# AI 出题(plan)单独的超时策略:它用 pro 且输出长(4096+thinking),健康时也常要 1~2 分钟,
# 90s 默认超时会把"正在好好生成"的请求误杀 → 给足时间;同时只试一次不重试。
PLAN_TIMEOUT_S = 180
PLAN_MAX_RETRIES = 0
# 硬墙钟上限:SDK 超时是"空闲超时",后端若"慢滴流"吐字不会触发 → 外面再套一道墙钟,
# 到点就放弃本次 AI 出题、降级确定性版(用户在自动模式下永不无限干等)。
PLAN_WALL_TIMEOUT_S = 240
