"""Bounded Scrapling rendering for sparse script-driven HTML pages."""
import os
from urllib.parse import urlsplit

from retrieval import UnsupportedContentTypeError, parse_page


class BrowserRenderError(RuntimeError):
    """Browser startup or navigation failed; the static page remains usable."""


def render_page(url, static_length):
    if urlsplit(url).scheme not in ("http", "https"):
        raise ValueError("Only HTTP and HTTPS URLs are supported.")
    try:
        from scrapling.fetchers import StealthyFetcher
        from patchright.sync_api import TimeoutError as BrowserTimeout
    except ImportError as exc:
        raise BrowserRenderError(
            "Install requirements-browser.txt and run scrapling install."
        ) from exc

    def wait_for_content(page):
        try:
            page.wait_for_function(
                "minimum => document.body && document.body.innerText.length > minimum",
                arg=static_length + 10, timeout=5000,
            )
        except BrowserTimeout:
            pass

    # Some hosted containers cannot start Chromium through its zygote process.
    launch_options = {}
    if os.environ.get("MATCH4_BROWSER_NO_ZYGOTE") == "1":
        launch_options["extra_flags"] = ["--no-zygote"]

    try:
        response = StealthyFetcher.fetch(
            url, timeout=12000, page_action=wait_for_content, **launch_options,
        )
    except Exception as exc:
        raise BrowserRenderError(str(exc)) from exc
    if not 200 <= response.status < 300:
        raise BrowserRenderError(f"Browser navigation returned HTTP {response.status}.")
    content_type = next(
        (value for key, value in response.headers.items()
         if key.lower() == "content-type"), ""
    ).lower()
    if "text/html" not in content_type:
        raise UnsupportedContentTypeError(f"Expected HTML, received: {content_type}")
    html = response.body.decode(response.encoding or "utf-8", errors="replace")
    return parse_page(html, str(response.url))
