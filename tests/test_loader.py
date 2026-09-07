"""第 1 课测试:输入层 loader 的纯函数测试。"""

from interview_coach import loader


def test_pasted_text_returns_as_is():
    """粘贴进来的普通文本,不能被误当成文件路径。"""
    text = "这是一段粘贴的 JD 内容,含多行\n第二行。"
    assert loader.read_text(text) == text


def test_single_line_text_returns_as_is():
    """单行、非文件路径的文本也要原样返回。"""
    text = "AI 应用研发(2026 届)"
    assert loader.read_text(text) == text


def test_existing_file_is_read_as_utf8():
    """给真实存在的文件路径 → 读出 utf-8 内容(去掉首尾空白)。"""
    jd_path = "examples/sample-jd.txt"
    content = loader.read_text(jd_path)
    assert "AI 应用研发工程师" in content  # 确认是 utf-8 读出来的中文
