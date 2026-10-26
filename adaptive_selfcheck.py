"""Offline regression checks for adaptive browser retrieval."""
import io
import json
from contextlib import redirect_stdout
from unittest.mock import patch
from types import SimpleNamespace, ModuleType
import sys

import httpx

from agent import run_agent
from retrieval import parse_page
from retrieval_browser import BrowserRenderError
from run_agent_suite import run_case


class Model:
    def generate(self, messages):
        request = json.loads(messages[-1]["content"])
        if "request_to_analyze" in request:
            return json.dumps({"required_fields": ["price"]})
        if "page" in request:
            text = request["page"]["text"]
            return json.dumps({"price": {"value": "12 EUR",
                "quote": "Adult admission: 12 EUR.", "confidence": 0.99}}) \
                if "Adult admission: 12 EUR." in text else "{}"
        if "target_field" in request:
            pages = json.loads(messages[1]["content"])["pages"]
            index = next((i for i, page in enumerate(pages)
                          if "Adult admission: 12 EUR." in page["text"]), None)
            if index is None:
                return json.dumps({"page_index": None, "quote": "", "answer": ""})
            return json.dumps({"page_index": index, "quote": "Adult admission: 12 EUR.",
                               "answer": "12 EUR"})
        return '{"link_index": 0}' if request.get("links") else '{"link_index": null}'


url = "https://example.org/shell"
shell_html = "<html><title>Visit</title><main>Loading</main><script src='/app.js'></script></html>"
static = parse_page(shell_html, url)
rendered = parse_page("<html><main>Adult admission: 12 EUR.</main></html>", url)
assert static["render_hint"] and not rendered["render_hint"]
assert not parse_page("<html><p>Adult admission: 12 EUR.</p><script></script></html>",
                      url)["text_truncated"]

render_calls = []
with redirect_stdout(io.StringIO()) as output:
    answer = run_agent("What is adult admission?", url, Model(), None,
        fetch_page_fn=lambda target: static,
        render_page_fn=lambda target, length: render_calls.append((target, length)) or rendered,
        max_pages=1)
assert render_calls == [(url, len(static["text"]))]
assert answer["answer"] == "12 EUR"
assert answer["citations"][0]["source_url"] == url
assert answer["citations"][0]["quote"] == "Adult admission: 12 EUR."
assert "Browser fallback used" in output.getvalue()
print("PASS: sparse script page renders before one-page budget ends and grounds citation")

with redirect_stdout(io.StringIO()):
    answer = run_agent("What is adult admission?", url, Model(), None,
        fetch_page_fn=lambda target: {**rendered, "render_hint": True},
        render_page_fn=lambda *args: (_ for _ in ()).throw(AssertionError("unneeded")))
assert answer["answer"] == "12 EUR"
print("PASS: grounded static answer does not launch browser")

next_url = "https://example.org/fees"
linked = {**static, "links": [{"url": next_url, "text": "Fees"}]}
def fetch(target):
    return linked if target == url else {**rendered, "url": next_url}
with redirect_stdout(io.StringIO()) as output:
    answer = run_agent("What is adult admission?", url, Model(), None,
        fetch_page_fn=fetch,
        render_page_fn=lambda *args: (_ for _ in ()).throw(BrowserRenderError("offline")))
assert answer["answer"] == "12 EUR"
assert answer["citations"][0]["source_url"] == next_url
assert "Browser fallback failed" in output.getvalue()
print("PASS: browser failure preserves static links and verified answer")

with redirect_stdout(io.StringIO()) as output:
    answer = run_agent("What is adult admission?", url, Model(), None,
        fetch_page_fn=lambda target: static,
        render_page_fn=lambda *args: {**rendered, "url": "https://other.org/fees"})
assert answer["status"] == "abstained" and not answer.get("citations")
assert "off-origin redirect" in output.getvalue()
print("PASS: off-origin browser result cannot become evidence")

request = httpx.Request("GET", url)
response = httpx.Response(200, request=request, text=shell_html,
                          headers={"content-type": "text/html"})
case = {"id": "adaptive", "group": "real", "url": url,
        "question": "What is adult admission?",
        "expected": {"status": "answered", "answers": ["12 EUR"]}}
with patch("retrieval.fetch_scrapling_response", return_value=SimpleNamespace(
        status=200, headers={"content-type": "text/html"}, encoding="utf-8",
        url=url, body=response.content)), \
     patch("retrieval_browser.render_page", return_value=rendered):
    row = run_case({**case, "requires_browser": True}, "original", Model(), None, 1)
assert row["passed"] and row["browser_attempts"] == 1 and row["browser_exercised"]
assert row["actual"]["citations"][0]["quote"] == "Adult admission: 12 EUR."
with patch("retrieval.fetch_scrapling_response", return_value=SimpleNamespace(
        status=200, headers={"content-type": "text/html"}, encoding="utf-8",
        url=url, body=response.content)):
    baseline = run_case(case, "original", Model(), None, 1, static_only=True)
assert not baseline["passed"] and baseline["browser_attempts"] == 0
print("PASS: suite reports browser attempts and static-only baseline")

with patch("run_agent_suite.fetcher_for", return_value=fetch), \
     patch("retrieval_browser.render_page", side_effect=BrowserRenderError("offline")):
    failed_render = run_case({**case, "requires_browser": True}, "original",
                             Model(), None, 1, max_pages=2)
assert failed_render["actual"]["answer"] == "12 EUR"
assert failed_render["browser_attempts"] == 1
assert not failed_render["browser_exercised"] and not failed_render["passed"]
print("PASS: a static fallback answer cannot pass a failed browser case")


# The renderer reads the final browser DOM and URL, then reuses HTML parsing.
from retrieval_browser import render_page

class BrowserPage:
    def wait_for_function(self, expression, **kwargs):
        assert kwargs["arg"] == len(static["text"]) + 10

def fake_browser_fetch(target, **kwargs):
    assert target == url and kwargs["timeout"] == 12000
    kwargs["page_action"](BrowserPage())
    return SimpleNamespace(
        url="https://example.org/current", status=200,
        headers={"content-type": "text/html"}, encoding="utf-8",
        body=b"<html><main>Adult admission: 12 EUR.</main></html>",
    )

fetchers = ModuleType("scrapling.fetchers")
fetchers.StealthyFetcher = SimpleNamespace(fetch=fake_browser_fetch)
scrapling = ModuleType("scrapling")
scrapling.__path__ = []
patchright = ModuleType("patchright")
patchright.__path__ = []
sync_api = ModuleType("patchright.sync_api")
sync_api.TimeoutError = TimeoutError
with patch.dict(sys.modules, {"scrapling": scrapling, "scrapling.fetchers": fetchers,
                              "patchright": patchright, "patchright.sync_api": sync_api}):
    browser_page = render_page(url, len(static["text"]))
assert browser_page["url"] == "https://example.org/current"
assert "Adult admission: 12 EUR." in browser_page["text"]
print("PASS: browser adapter parses final DOM and redirected URL")


class TwoFieldModel:
    def generate(self, messages):
        request = json.loads(messages[-1]["content"])
        if "request_to_analyze" in request:
            return json.dumps({"required_fields": ["director", "price"]})
        if "page" in request:
            text = request["page"]["text"]
            facts = {}
            for field, value, quote in (
                ("director", "Maya Chen", "Director: Maya Chen."),
                ("price", "12 EUR", "Adult admission: 12 EUR."),
            ):
                if quote in text:
                    facts[field] = {"value": value, "quote": quote, "confidence": 0.99}
            return json.dumps(facts)
        if "target_field" in request:
            field = request["target_field"]
            value, quote = (("Maya Chen", "Director: Maya Chen.")
                            if field == "director" else
                            ("12 EUR", "Adult admission: 12 EUR."))
            return json.dumps({"page_index": 0, "answer": value, "quote": quote})
        return '{"link_index": null}'

static_detail = parse_page(
    '<main>Director: Maya Chen. <a href="/contact">Contact</a></main><script></script>', url)
render_detail = parse_page('<main>Adult admission: 12 EUR.</main>', url)
with redirect_stdout(io.StringIO()):
    combined = run_agent("Who is director and what is admission?", url,
        TwoFieldModel(), None, fetch_page_fn=lambda target: static_detail,
        render_page_fn=lambda *args: render_detail, max_pages=1)
assert combined["answer"] == "Maya Chen | 12 EUR"
assert [citation["quote"] for citation in combined["citations"]] == [
    "Director: Maya Chen.", "Adult admission: 12 EUR.",
]
print("PASS: browser content retains static evidence for other requested fields")
