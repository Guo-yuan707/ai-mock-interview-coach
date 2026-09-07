# 🎙️ AI 模拟面试官 (Mock Interview Coach)

一份由 **JD 驱动** 的 AI 模拟面试系统:贴一份岗位 JD,系统先自动生成面试计划(评分维度 + 高危点),再由「面试官」多轮追问、逐题交给「评卷人」按结构化维度打分,结束输出结案报告。全程带**红线**:问题与评分只依据 JD/简历原文或高危点清单,**绝不编造候选人的经历**。

> 技术要点:Python · Streamlit(网页双入口)+ CLI · 通过 OpenAI 兼容协议接入 **DeepSeek**(BYOK) · SQLite 历史落库 · 确定性版「免 key」也能跑全流程。

---

## ✨ 功能

| 页面 | 做什么 |
|---|---|
| **① 模拟面试** | 真 AI 多轮追问 + 逐题结构化评分;或「离线演示」免 key 走脚本替身看全流程 |
| **② 历史回放** | (本地/单机)把一场面试从 SQLite 逐字回放、导出/删除 |
| **③ 一键评测** | 系统体检:检索 recall@k、出题主题覆盖、评分严格度(默认离线免 key) |

双入口(网页 + CLI)复用同一份核心 `interview_coach/`,逻辑只有一份,不会两处越写越偏。

## 🔐 BYOK & 隐私红线

- **在线模式 = 自带 key(BYOK)**:每位用户在网页左侧填**自己的 DeepSeek API Key**,key 只存在该浏览器会话内存中——不落盘、不入库、服务端不留。多访客各用各的 client(见 `app.py` 的 `_load_user_client()`),额度各自承担。
- 真实简历 / JD 只临时粘贴进会话,不写入任何文件或代码库;公开部署默认隐藏「共享历史库」,防止访客互看互删(见 `interview_coach/config.py` 的 `DEPLOYED_PUBLIC`)。
- 项目自带 `.env.example` 模板,真实密钥只放本地 `.env`(已被 `.gitignore` 拦截)。

## 🚀 本地运行

```bash
pip install -r requirements.txt
# 1) 复制 .env.example 为 .env,填入你自己的 DeepSeek key
cp .env.example .env

# 网页版(推荐)
streamlit run app.py
# 浏览器打开 http://localhost:8501

# 或命令行
python main.py plan 某JD.txt --ai        # AI 生成面试计划
python main.py interview --offline       # 免 key 也能跑整场演示
python main.py history / replay <id>     # 回看历史
```

**没 key 也能玩**:`🧪 离线演示` 用脚本替身跑完「计划→追问→评分→报告」全流程,看的是编排逻辑本身。

## 🧪 测试

```bash
pytest -q          # 50 个测试,全部离线、不碰真 API
python -m interview_coach.eval.run           # 一键评测(离线,免 key)
python -m interview_coach.eval.run --live    # 用真模型跑(烧你自己的 token)
```

## ☁️ 部署(Streamlit Community Cloud)

免费档即可。要点:仓库保持 public → 平台自动发现根目录 `app.py` → 在 App settings 把 Python 版本调到平台可用版本(本项目本地验证于 3.12/3.14)→ **不需要配置任何服务端 Secret**(在线模式让用户自带 key)。

## 📁 结构

```
app.py                      # 网页入口(Streamlit 界面编排)
interview_coach/            # 与 CLI 共享的核心
  ├── planner.py            #   面试计划(确定性 / AI 两版)
  ├── engine.py             #   面试会话编排
  ├── interviewer.py        #   面试官:多轮追问(可调工具)
  ├── evaluator.py          #   评卷人:结构化打分
  ├── retrieval/            #   词典 / 向量 / 混合检索
  ├── storage.py            #   SQLite 落库 + 历史回放
  ├── report.py             #   结案报告导出
  └── llm.py                #   DeepSeek(OpenAI 兼容)客户端,默认读 .env
tests/                      # 离线测试
examples/                   # 合成样例(JD / 简历)
```
