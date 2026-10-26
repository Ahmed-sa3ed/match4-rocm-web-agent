"""Scrapling HTTP retrieval compatible with the existing agent page contract.

This adapter uses Scrapling's Response selectors for page text and links.
Browser rendering is handled by the separate Scrapling browser adapter.
"""
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup

from forms import parse_search_forms
from tables import format_simple_tables
from retrieval import UnsupportedContentTypeError, fetch_scrapling_response


def fetch_page(url: str) -> dict:
    if urlsplit(url).scheme not in {"http", "https"}:
        raise ValueError("Only HTTP and HTTPS URLs are supported.")
    response = fetch_scrapling_response(url)
    content_type = next(
        (value for key, value in response.headers.items()
         if key.lower() == "content-type"), ""
    ).lower()
    if "application/pdf" in content_type or response.body.startswith(b"%PDF-"):
        from retrieval_pdf import parse_pdf
        return parse_pdf(response.body, str(response.url))
    if "text/html" not in content_type:
        raise UnsupportedContentTypeError(f"Expected HTML or PDF, received: {content_type}")

    final_url = str(response.url)
    html = response.body.decode(response.encoding or "utf-8", errors="replace")
    forms = parse_search_forms(html, final_url)
    has_scripts = bool(BeautifulSoup(html, "html.parser").find("script"))

    # Scrapling parses the title, links and visible text. Keep the established
    # table formatter, so the two versions differ in retrieval and DOM parsing
    # while sharing the same evidence and answer logic.
    title = str(response.css("title::text").get() or "").strip()
    links, seen = [], set()
    content_anchors = response.css("main a[href], article a[href]")
    for anchor in list(content_anchors) + list(response.css("a[href]")):
        target = urljoin(final_url, str(anchor.attrib.get("href", "")))
        if urlsplit(target).scheme not in {"http", "https"} or target in seen:
            continue
        seen.add(target)
        links.append({
            "text": str(anchor.get_all_text(separator=" ", strip=True)),
            "url": target,
        })

    soup = BeautifulSoup(html, "html.parser")
    for element in soup.select("script, style, template"):
        element.decompose()
    table_rows = format_simple_tables(soup)
    text = str(response.get_all_text(
        separator="\n", strip=True,
        ignore_tags=("script", "style", "template"),
    ))
    promoted, size = [], 0
    for row in table_rows:
        if size + len(row) + 1 > 4000:
            break
        promoted.append(row)
        size += len(row) + 1
    if promoted:
        text = "\n".join(promoted) + "\n\n" + text
    return {
        "url": final_url,
        "title": title,
        "text": text,
        "text_truncated": False,
        "links": links,
        "links_truncated": False,
        "forms": forms,
        "render_hint": has_scripts and len(text) < 600,
    }
