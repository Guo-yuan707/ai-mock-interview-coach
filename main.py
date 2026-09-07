"""命令行入口(薄壳):真正的实现都在 interview_coach/cli.py。

运行方式(在项目根目录):
    .venv/Scripts/python main.py                  # 面试计划预览(第 1 课面板,确定性、免 key)
    .venv/Scripts/python main.py plan 某JD.txt [--resume 简历.txt] [--ai]
    .venv/Scripts/python main.py interview [--offline] [--ai] [--model ...] [--save] [--export]
    .venv/Scripts/python main.py history          # 列出历史面试(SQLite)
    .venv/Scripts/python main.py replay <session_id>
    .venv/Scripts/python main.py eval [--live]

快捷示范(没 API key 也能跑通全流程):
    .venv/Scripts/python main.py interview --offline --save
"""
import sys

from interview_coach.cli import main as cli_main


def main() -> int:
    return cli_main()


if __name__ == "__main__":
    sys.exit(main())
