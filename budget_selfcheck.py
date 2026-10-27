"""Check time and response-size boundaries without network access."""
from time import monotonic
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import sys

import budget
import httpx
import retrieval
from retrieval import UnsupportedContentTypeError

url = "https://example.org/"
response = SimpleNamespace(status=200, url=url, encoding="utf-8",
                           headers={"content-type": "text/html"},
                           body=b"<p>fine</p>")
fetchers = ModuleType("scrapling.fetchers")
seen = []
def get(target, **options):
    seen.append(options["timeout"])
    return response
fetchers.Fetcher = type("Fetcher", (), {"get": staticmethod(get)})
module = ModuleType("scrapling")
module.__path__ = []
with patch.dict(sys.modules, {"scrapling": module, "scrapling.fetchers": fetchers}):
    budget.set_deadline(monotonic() + 0.5)
    assert retrieval.fetch_page(url)["text"] == "fine"
    assert 0.1 <= seen[-1] <= 0.5
    budget.set_deadline(monotonic() - 1)
    try:
        retrieval.fetch_page(url)
        raise AssertionError("Expired fetch was accepted")
    except httpx.RequestError as exc:
        assert "deadline" in str(exc)
    assert len(seen) == 1
budget.set_deadline(None)
with patch.object(retrieval, "fetch_scrapling_response", return_value=SimpleNamespace(
        **{**response.__dict__, "body": b"x" * (retrieval.MAX_HTML_BYTES + 1)})):
    try:
        retrieval.fetch_page(url)
        raise AssertionError("Oversized HTML accepted")
    except UnsupportedContentTypeError as exc:
        assert "HTML response exceeds" in str(exc)
print("PASS: Scrapling fetch honors remaining time and rejects oversized HTML")
