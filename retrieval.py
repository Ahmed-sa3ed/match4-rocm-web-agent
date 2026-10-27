from urllib.parse import urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup

from forms import parse_search_forms
from budget import remaining
from tables import format_simple_tables


MAX_HTML_BYTES = 2 * 1024 * 1024


class UnsupportedContentTypeError(ValueError):
    """A downloaded link points to content this agent cannot read."""


def fetch_page(url: str) -> dict:
    """Fetch HTML or PDF with Scrapling and return an evidence page."""
    if urlsplit(url).scheme not in {"http", "https"}:
        raise ValueError("Only HTTP and HTTPS URLs are supported.")

    response = fetch_scrapling_response(url)

    content_type = next(
        (value for key, value in response.headers.items()
         if key.lower() == "content-type"), ""
    ).lower()
    is_pdf = "application/pdf" in content_type or response.body.startswith(b"%PDF-")
    if not is_pdf and len(response.body) > MAX_HTML_BYTES:
        raise UnsupportedContentTypeError(f"HTML response exceeds {MAX_HTML_BYTES} bytes")
    if is_pdf:
        from retrieval_pdf import parse_pdf
        return parse_pdf(response.body, str(response.url))
    if "text/html" not in content_type:
        raise UnsupportedContentTypeError(f"Expected HTML or PDF, received: {content_type}")

    return parse_page(response.body.decode(response.encoding or "utf-8", errors="replace"),
                      str(response.url))


def fetch_scrapling_response(url):
    """Use Scrapling HTTP with the agent's existing retry error contract."""
    from scrapling.fetchers import Fetcher

    request = httpx.Request("GET", url)
    try:
        response = Fetcher.get(url, follow_redirects="safe", timeout=max(0.1, remaining(10)),
                               retries=1, stealthy_headers=False,
                               impersonate=None, verify=True)
    except Exception as exc:
        raise httpx.RequestError(str(exc), request=request) from exc
    if not 200 <= response.status < 300:
        raise httpx.HTTPStatusError(
            f"HTTP {response.status} for {url}", request=request,
            response=httpx.Response(response.status, request=request),
        )
    return response


def parse_page(html: str, url: str) -> dict:
    """Parse static or rendered HTML into the evidence page contract."""
    forms = parse_search_forms(html, url)

    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    has_scripts = bool(soup.find("script"))

    for element in soup.select("script, style, template"):
        element.decompose()

    links = []
    seen = set()
    content_anchors = soup.select("main a[href], article a[href]")
    for anchor in content_anchors + soup.select("a[href]"):
        target = urljoin(url, anchor["href"])
        if urlsplit(target).scheme not in {"http", "https"}:
            continue
        if target in seen:
            continue
        seen.add(target)
        links.append({
            "text": anchor.get_text(" ", strip=True),
            "url": target,
        })

    table_rows = format_simple_tables(soup)
    text = soup.get_text("\n", strip=True)

    promoted = []
    size = 0
    for row in table_rows:
        if size + len(row) + 1 > 4000:
            break
        promoted.append(row)
        size += len(row) + 1
    if promoted:
        text = "\n".join(promoted) + "\n\n" + text
    return {
        "url": url,
        "title": title,
        "text": text,
        "text_truncated": False,
        "links": links,
        "links_truncated": False,
        "forms": forms,
        "render_hint": has_scripts and len(text) < 600,
    }
