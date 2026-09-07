"""输入层(第 1 课):把 JD / 简历从"路径 或 粘贴文本"读成干净的字符串。

为什么单独一个模块:
- 上层(planner / interviewer)只想要"文本",不在乎它是从文件来还是粘贴来 → 接口统一,好测;
- 中文读取最容易在编码上翻车(Windows 默认 GBK),读入统一用 utf-8,一处把关全局安全。
"""
from pathlib import Path


def read_text(source: str) -> str:
    """入口函数:给路径就读文件;给文本就原样返回(去除首尾空白)。

    用"这串字像不像一个存在的文件路径"来判断来源,
    所以粘贴进来的普通文本永远不会被误当成文件名。
    """
    if _is_existing_file(source):
        return Path(source).read_text(encoding="utf-8")
    return source.strip()


def _is_existing_file(source: str) -> bool:
    """判断 source 是不是磁盘上真实存在的文件路径。

    - 太短(如 <1 字符)不可能是文件,省一次磁盘查询;
    - 含换行的多行粘贴内容不可能是个文件名。
    """
    if not source or len(source) < 1 or "\n" in source:
        return False
    return Path(source).exists()
