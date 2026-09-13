"""把 Streamlit Community Cloud 免费档的休眠实例叫醒,让作品集链接点开就是热的。

为什么不用 curl / UptimeRobot / cron-job.org:
    休眠时打这个链接会收到 200 OK,但返回的只是一层约 4KB 的静态 HTML 外壳,
    Python 进程根本没起来 —— 探测显示"正常",应用却在睡。真正唤醒需要
    执行 JavaScript 并建立 WebSocket(/_stcore/stream)。所以这里用无头浏览器
    "真的打开一次页面"。

判据:
    页面上出现 [data-testid="stAppViewContainer"] 才算真醒了 —— 这个节点由
    Streamlit 的 Python 进程渲染出来,拿到它才说明应用已经在跑,而不是只有外壳。

用法(本地也能跑,先装 playwright 及其 chromium):
    APP_URL=https://xxx.streamlit.app python wake_streamlit.py
"""
from __future__ import annotations

import os
import re
import sys
import time

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

# 休眠页上的唤醒按钮文案(Streamlit 改版可能改字,所以匹配宽松些)
WAKE_BUTTON = re.compile(r"get this app back up", re.IGNORECASE)

# Python 进程真起来了的证据:这个节点由 Streamlit 渲染,静态外壳里没有
APP_READY = '[data-testid="stAppViewContainer"]'

OPEN_TIMEOUT_MS = 60_000     # 打开页面本身
WAKE_TIMEOUT_MS = 180_000    # 冷启动唤醒:平台说"几秒到几十秒",给足 3 分钟
KEEP_ALIVE_S = 20            # 渲染出来后保持会话,确保这次访问被计为"活跃"


def _wake_if_sleeping(page) -> bool:
    """看到休眠页就点『get this app back up』。返回是否点了。"""
    btn = page.get_by_role("button", name=WAKE_BUTTON)
    try:
        btn.first.wait_for(state="visible", timeout=8_000)
    except PlaywrightTimeoutError:
        print("· 没看到休眠页(应用是醒着的),直接等首屏。")
        return False
    print("· 检测到休眠页,点『get this app back up』唤醒…")
    btn.first.click()
    return True


def main() -> int:
    url = (os.environ.get("APP_URL") or "").strip()
    if not url:
        print("❌ 没给 APP_URL 环境变量,不知道该叫醒谁。", file=sys.stderr)
        return 2

    t0 = time.monotonic()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        try:
            print(f"→ 打开 {url}")
            page.goto(url, wait_until="domcontentloaded", timeout=OPEN_TIMEOUT_MS)

            _wake_if_sleeping(page)

            print("→ 等首次渲染(冷启动可能要几十秒)…")
            page.wait_for_selector(APP_READY, state="attached", timeout=WAKE_TIMEOUT_MS)
            print(f"✅ 首屏渲染完成,耗时 {time.monotonic() - t0:.1f}s")

            print(f"→ 保持会话 {KEEP_ALIVE_S}s(确保这次访问被计为活跃)…")
            page.wait_for_timeout(KEEP_ALIVE_S * 1000)
        except PlaywrightTimeoutError as e:
            print(f"❌ 等待超时,应用没能渲染出来:{e}", file=sys.stderr)
            return 1
        except Exception as e:  # 网络/浏览器等其它异常,原样暴露便于排查
            print(f"❌ 出错:{type(e).__name__}: {e}", file=sys.stderr)
            return 1
        finally:
            browser.close()

    print(f"✅ 保活完成,总耗时 {time.monotonic() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
