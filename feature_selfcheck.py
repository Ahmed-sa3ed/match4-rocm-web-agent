"""Offline regression checks for retry, complete pages, chunks and evidence logs."""
import io
import json
from contextlib import redirect_stdout
from unittest.mock import patch
import sys

import httpx

import agent
import retrieval
from retrieval import UnsupportedContentTypeError


class Model:
    def __init__(self, invalid_final=False):
        self.invalid_final = invalid_final
        self.read_chunks = 0

    def generate(self, messages):
        request = json.loads(messages[-1]["content"])
        if "request_to_analyze" in request:
            return json.dumps({"required_fields": ["adult_admission"]})
        if "proposed_answer" in request:
            return json.dumps({"supported": False, "quote": "",
                               "reason": "The proposed passage does not establish this answer."})
        if "page" in request:
            self.read_chunks += 1
            if "Adult admission: 12 EUR." in request["page"]["text"]:
                return json.dumps({"adult_admission": {
                    "value": "12 EUR", "quote": "Adult admission: 12 EUR.",
                    "confidence": 1.0, "reason": "The price appears here.",
                }})
            return "{}"
        if "target_field" in request:
            original = json.loads(messages[1]["content"])
            return json.dumps({
                "page_index": next((page["page_index"] for page in original["pages"]
                                    if "Adult admission: 12 EUR." in page["text"]), 0),
                "quote": "Adult admission: ... 12 EUR." if self.invalid_final else
                         "Adult admission: 12 EUR.",
                "answer": "12 EUR",
                "reason": "The price matches the question.",
            })
        return '{"link_index": null}'


def page(url, text, links=None):
    return {"url": url, "title": "Tickets", "text": text,
            "links": links or [], "forms": []}


url = "https://example.org/start"
request = httpx.Request("GET", url)
count = [0]


def flaky(target):
    count[0] += 1
    if count[0] == 1:
        raise httpx.HTTPStatusError("HTTP 503", request=request,
                                    response=httpx.Response(503, request=request))
    return page(target, "Adult admission: 12 EUR.")

with patch.object(agent, "sleep", lambda _: None):
    assert agent.fetch_with_retry(url, flaky)["url"] == url
assert count[0] == 2

count[0] = 0

def missing(target):
    count[0] += 1
    raise httpx.HTTPStatusError("HTTP 404", request=request,
                                response=httpx.Response(404, request=request))

try:
    agent.fetch_with_retry(url, missing)
    raise AssertionError("HTTP 404 should fail")
except httpx.HTTPStatusError:
    pass
assert count[0] == 1

html = "<html><body>" + "a" * 13000 + "Adult admission: 12 EUR." + "".join(
    f'<a href="/page-{n}">Page {n}</a>' for n in range(45)) + "</body></html>"
response = httpx.Response(200, request=request, text=html,
                          headers={"content-type": "text/html"})
from types import SimpleNamespace
scrapling_response = SimpleNamespace(
    status=200, headers={"content-type": "text/html"}, encoding="utf-8",
    url=url, body=html.encode("utf-8"),
)
with patch.object(retrieval, "fetch_scrapling_response", return_value=scrapling_response):
    full = retrieval.fetch_page(url)
assert len(full["text"]) > 12000 and "12 EUR" in full["text"]
assert len(full["links"]) == 45
print("PASS: original retriever retains long page text and all 45 links")

# The actual HTTP boundary is Scrapling, including status and retry metadata.
from types import ModuleType
fake_fetchers = ModuleType("scrapling.fetchers")
seen_http = []
def fake_get(target, **options):
    seen_http.append((target, options))
    return scrapling_response
fake_fetchers.Fetcher = type("Fetcher", (), {"get": staticmethod(fake_get)})
fake_scrapling = ModuleType("scrapling")
fake_scrapling.__path__ = []
with patch.dict(sys.modules, {"scrapling": fake_scrapling,
                              "scrapling.fetchers": fake_fetchers}):
    actual = retrieval.fetch_page(url)
assert actual["url"] == url and seen_http[0][0] == url
assert seen_http[0][1]["retries"] == 1
assert seen_http[0][1]["follow_redirects"] == "safe"
print("PASS: website HTTP uses Scrapling Fetcher with bounded retry and safe redirects")


with patch.object(agent, "choose_link", side_effect=[None, 0]) as chooser:
    chosen = agent.choose_next_url("admission", ["adult_admission"],
        full, {full["url"]: full}, {full["url"]}, set(), url,
        Model(), None)
assert chosen == "https://example.org/page-40"
assert len(chooser.call_args_list[0].args[0]["links"]) == 40
assert len(chooser.call_args_list[1].args[0]["links"]) == 5
print("PASS: navigation inspects second batch after the first 40 links")

model = Model()
long_text = ("Museum history.\n" * 850) + "Adult admission: 12 EUR."
with redirect_stdout(io.StringIO()) as output:
    result = agent.run_agent("What is adult admission?", url, model, None,
                             fetch_page_fn=lambda target: page(target, long_text))
assert result["status"] == "answered" and result["answer"] == "12 EUR"
assert model.read_chunks > 1, model.read_chunks
assert "chunks:" in output.getvalue()
print("PASS: long page inspects later text chunks and answers")

image = "https://example.org/image.png"
answer = "https://example.org/fees"
links = [{"url": image, "text": "Picture"}, {"url": answer, "text": "Fees"}]

def recover(target):
    if target == url:
        return page(url, "Tickets and images", links)
    if target == image:
        raise UnsupportedContentTypeError("Expected HTML, received: image/png")
    return page(answer, "Adult admission: 12 EUR.")

with patch.object(agent, "choose_link", return_value=0):
    with redirect_stdout(io.StringIO()) as output:
        result = agent.run_agent("What is adult admission?", url,
                                 Model(), None, fetch_page_fn=recover)
assert result["status"] == "answered" and result["answer"] == "12 EUR", result
assert "image/png" in output.getvalue()
print("PASS: unsupported image is logged and another link provides the answer")

call_count = [0]
def unavailable(target):
    call_count[0] += 1
    raise httpx.RequestError("connection reset", request=request)

with patch.object(agent, "sleep", lambda _: None):
    with redirect_stdout(io.StringIO()) as output:
        result = agent.run_agent("What is adult admission?", url,
                                 Model(), None, fetch_page_fn=unavailable)
assert call_count[0] == 2
assert result["status"] == "abstained" and "connection reset" in result["failure_reason"]
assert result["fetch_failures"][0]["url"] == url
print("PASS: exhausted network retries return URL and failure reason")

with redirect_stdout(io.StringIO()) as output:
    result = agent.run_agent("What is adult admission?", url,
        Model(invalid_final=True), None,
        fetch_page_fn=lambda target: page(target, "Adult admission: 12 EUR."))
assert result["status"] == "abstained" and result["answer"] == ""
assert not result.get("citations")
assert result["selection_failures"][0]["grounded_candidate_count"] == 1
assert "semantic review found insufficient support" in result["failure_reason"]
trace = output.getvalue()
assert "confidence: 1.00" in trace and "grounded: True" in trace
assert "quote is absent from the chosen page" in trace
assert "Final selection incomplete" in trace
print("PASS: high-confidence candidate abstention records the model reply and rejection")

# Test the Scrapling adapter without requiring Scrapling in this Python environment.
import importlib
from bs4 import BeautifulSoup

class Selection(list):
    def get(self):
        return self[0] if self else None

class Node:
    def __init__(self, tag):
        self.tag = tag
        self.attrib = tag.attrs

    def get_all_text(self, separator=" ", strip=True):
        return self.tag.get_text(separator, strip=strip)

class FakeScraplingResponse:
    status = 200
    headers = {"content-type": "text/html"}
    encoding = "utf-8"
    url = "https://example.org/start"
    body = html.encode("utf-8")

    def css(self, selector):
        soup = BeautifulSoup(self.body, "html.parser")
        if selector == "title::text":
            return Selection([soup.title.get_text()] if soup.title else [])
        return Selection([Node(node) for node in soup.select(selector)])

    def get_all_text(self, separator="\n", strip=True, ignore_tags=()):
        soup = BeautifulSoup(self.body, "html.parser")
        for node in soup.select(", ".join(ignore_tags)):
            node.decompose()
        return soup.get_text(separator, strip=strip)

scraper = importlib.import_module("retrieval_scrapling")
with patch.object(scraper, "fetch_scrapling_response", return_value=FakeScraplingResponse()):
    fetched = scraper.fetch_page(url)
    assert len(fetched["text"]) > 12000 and "12 EUR" in fetched["text"]
    assert len(fetched["links"]) == 45
print("PASS: Scrapling adapter retains long page text and all 45 links")

# Exact source matching stays mandatory; only semantic answer wording may vary.
from evidence import select_facts
from validation import validate_facts

class SemanticModel:
    def __init__(self, answer, source_quote, supported=True, review_quote=None):
        self.answer = answer
        self.source_quote = source_quote
        self.supported = supported
        self.review_quote = review_quote if review_quote is not None else source_quote
        self.calls = 0

    def generate(self, messages):
        request = json.loads(messages[-1]["content"])
        self.calls += 1
        if "proposed_answer" in request:
            return json.dumps({"supported": self.supported,
                               "quote": self.review_quote if self.supported else "",
                               "reason": "The page directly supports this conclusion."
                                         if self.supported else
                                         "The page does not support the proposed conclusion."})
        return json.dumps({"page_index": 0, "quote": "Paraphrased quote.",
                           "answer": self.answer,
                           "reason": "Testing a distinct expression of the answer."})

source_quote = "They are not available for registration or transfer."
semantic_page = {"https://example.org/policy": {
    "url": "https://example.org/policy", "title": "Policy",
    "text": "Example domains. " + source_quote,
}}
semantic_request = {
    "question": "Is this example domain available for registration or transfer?",
    "required_fields": ["domain_registration_or_transfer_eligibility"],
    "pages": semantic_page, "candidate_reviews": [],
}
with redirect_stdout(io.StringIO()):
    model = SemanticModel("No", source_quote)
    facts = select_facts(semantic_request, model, None)
assert model.calls == 3 and facts["domain_registration_or_transfer_eligibility"]["value"] == "No"
assert facts["domain_registration_or_transfer_eligibility"]["quote"] == source_quote
assert validate_facts(semantic_request["required_fields"], facts, semantic_page)
assert not validate_facts(semantic_request["required_fields"], facts, semantic_page,
                          allow_semantic=True)
print("PASS: semantic paraphrase accepted with an actual source passage")

with redirect_stdout(io.StringIO()):
    model = SemanticModel("Yes", source_quote, supported=False)
    facts = select_facts(semantic_request, model, None)
assert facts is None and model.calls == 3
print("PASS: semantic reviewer can reject an unsupported answer")

with redirect_stdout(io.StringIO()):
    model = SemanticModel("No", source_quote, review_quote="Different policy text")
    facts = select_facts(semantic_request, model, None)
assert facts is None and model.calls == 3
print("PASS: model approval cannot invent a supporting page passage")

other_page = {"https://example.org/ticket": {
    "url": "https://example.org/ticket", "title": "Ticket rules",
    "text": "Customers may cancel within 24 hours without a fee.",
}}
other_request = {"question": "Is cancellation free within 24 hours?",
                 "required_fields": ["free_cancellation_within_24_hours"],
                 "pages": other_page, "candidate_reviews": []}
with redirect_stdout(io.StringIO()):
    facts = select_facts(other_request, SemanticModel("Yes",
        "Customers may cancel within 24 hours without a fee."), None)
assert facts["free_cancellation_within_24_hours"]["value"] == "Yes"
print("PASS: semantic fallback is not tied to one website or subject")

class IntegratedSemanticModel:
    def generate(self, messages):
        data = json.loads(messages[-1]["content"])
        if "request_to_analyze" in data:
            return json.dumps({"required_fields": ["domain_availability"]})
        if "proposed_answer" in data:
            return json.dumps({"supported": True, "quote": source_quote,
                               "reason": "The domain is included in the unavailable group."})
        if "page" in data:
            return json.dumps({"domain_availability": {
                "value": "No", "quote": source_quote,
                "confidence": 0.9, "reason": "The source denies availability.",
            }})
        return json.dumps({"page_index": 0, "quote": source_quote,
                           "answer": "No", "reason": "Availability is denied."})

with redirect_stdout(io.StringIO()) as output:
    single = agent.run_agent(semantic_request["question"],
        "https://example.org/policy", IntegratedSemanticModel(), None,
        fetch_page_fn=lambda target: page(target, semantic_page[target]["text"]))
assert single["status"] == "answered" and single["answer"] == "No", single
assert single["citations"] == [{
    "field": "domain_availability", "value": "No",
    "source_url": "https://example.org/policy", "quote": source_quote,
    "support_mode": "semantic",
}], single
assert "Semantic review domain_availability" in output.getvalue()
print("PASS: complete agent returns one semantically reviewed availability answer")


# High confidence may start the final review early, but cannot skip that review.
class EarlyReviewModel:
    def __init__(self, *, old_score=0.99):
        self.old_score = old_score

    def generate(self, messages):
        request = json.loads(messages[-1]["content"])
        if "request_to_analyze" in request:
            return json.dumps({"required_fields": ["adult_admission"]})
        if "page" in request:
            text = request["page"]["text"]
            if "In 2024" in text:
                return json.dumps({"adult_admission": {
                    "value": "8 EUR", "quote": "In 2024, adult admission was 8 EUR.",
                    "confidence": self.old_score, "reason": "An archived price.",
                }})
            if "Current prices" in text:
                return json.dumps({"adult_admission": {
                    "value": "12 EUR", "quote": "Current prices: adult admission is 12 EUR.",
                    "confidence": 0.99, "reason": "This is the current price.",
                }})
            return "{}"
        if "candidates" in request and "target_field" in request:
            index = next(i for i, item in enumerate(request["candidates"])
                         if item["value"] == "12 EUR")
            return json.dumps({"candidate_index": index,
                               "reason": "The source calls this the current price; the other is dated 2024."})
        if "target_field" in request:
            pages = json.loads(messages[1]["content"])["pages"]
            for current in pages:
                if "Current prices" in current["text"]:
                    return json.dumps({
                        "page_index": current["page_index"],
                        "quote": "Current prices: adult admission is 12 EUR.",
                        "answer": "12 EUR", "reason": "The page identifies a current price.",
                    })
            return json.dumps({
                "page_index": None, "quote": "", "answer": "",
                "reason": "Only a historical price is available.",
            })
        return '{"link_index": 0}'


archive_url = "https://example.org/archive"
current_url = "https://example.org/current"
archive = page(archive_url, "In 2024, adult admission was 8 EUR.",
               [{"url": current_url, "text": "Current prices"}])
current = page(current_url, "Current prices: adult admission is 12 EUR.")
fetched = []

def historical_fetch(target):
    fetched.append(target)
    return archive if target == archive_url else current

with redirect_stdout(io.StringIO()) as output:
    result = agent.run_agent("What is current adult admission?",
                             archive_url, EarlyReviewModel(), None,
                             fetch_page_fn=historical_fetch)
assert result["status"] == "answered" and result["answer"] == "12 EUR", result
assert fetched == [archive_url, current_url], fetched
assert result["conflicts"][0]["status"] == "resolved"
print("PASS: high-scoring historical price is rejected and navigation continues")

fetched.clear()
with redirect_stdout(io.StringIO()) as output:
    result = agent.run_agent("What is current adult admission?",
                             current_url, EarlyReviewModel(), None,
                             fetch_page_fn=lambda target: fetched.append(target) or current)
assert result["status"] == "answered" and result["answer"] == "12 EUR"
assert fetched == [current_url]
assert result["citations"] == [{
    "field": "adult_admission", "value": "12 EUR", "source_url": current_url,
    "quote": "Current prices: adult admission is 12 EUR.", "support_mode": "exact",
}]
assert "Early review approved | stopping after 1 page(s)" in output.getvalue()
print("PASS: verified high-confidence answer stops after one page")

fetched.clear()
with redirect_stdout(io.StringIO()) as output:
    result = agent.run_agent("What is current adult admission?",
                             archive_url, EarlyReviewModel(old_score=0.949), None,
                             max_pages=2, fetch_page_fn=historical_fetch)
assert result["status"] == "answered" and result["answer"] == "12 EUR"
assert fetched == [archive_url, current_url]
assert "Early review |" not in output.getvalue()
print("PASS: a score below 0.95 does not trigger the early review")

fetched.clear()
with redirect_stdout(io.StringIO()) as output:
    result = agent.run_agent("What is current adult admission?",
                             archive_url, EarlyReviewModel(), None,
                             max_pages=1, fetch_page_fn=historical_fetch)
assert result["status"] == "abstained" and fetched == [archive_url]
assert not result.get("citations")
print("PASS: max_pages limits successful page fetches")


# Distinct fields must keep their own final sources, including a redirected URL.
class MultiSourceModel:
    def generate(self, messages):
        request = json.loads(messages[-1]["content"])
        if "request_to_analyze" in request:
            return json.dumps({"required_fields": ["director", "admission"]})
        if "page" in request:
            return "{}"
        if "target_field" in request:
            field = request["target_field"]
            index, value, quote = (
                (0, "Maya Chen", "Director: Maya Chen.") if field == "director"
                else (1, "14 EUR", "Adult admission: 14 EUR.")
            )
            return json.dumps({"page_index": index, "answer": value, "quote": quote})
        return '{"link_index": 0}'


staff_url = "https://example.org/team"
fees_url = "https://example.org/fees"
redirected_url = "https://example.org/current-fees"
source_pages = {
    staff_url: page(staff_url, "Director: Maya Chen.",
                    [{"url": fees_url, "text": "Admission"}]),
    fees_url: page(redirected_url, "Adult admission: 14 EUR."),
}
with redirect_stdout(io.StringIO()):
    result = agent.run_agent("Who is director and what is admission?", staff_url,
                             MultiSourceModel(), None, max_pages=2,
                             fetch_page_fn=source_pages.__getitem__)
assert result == {
    "status": "answered", "answer": "Maya Chen | 14 EUR",
    "citations": [
        {"field": "director", "value": "Maya Chen", "source_url": staff_url,
         "quote": "Director: Maya Chen.", "support_mode": "exact"},
        {"field": "admission", "value": "14 EUR", "source_url": redirected_url,
         "quote": "Adult admission: 14 EUR.", "support_mode": "exact"},
    ],
}, result
print("PASS: ordered citations retain each field's verified passage and final source URL")


# Exercise CLI and report output with only model and HTTP boundaries replaced.
import local_agent
from run_agent_suite import run_case

response = httpx.Response(200, request=httpx.Request("GET", url),
                          text="<html><body>Adult admission: 12 EUR.</body></html>",
                          headers={"content-type": "text/html"})
with patch.object(retrieval, "fetch_scrapling_response", return_value=SimpleNamespace(
        status=200, headers={"content-type": "text/html"}, encoding="utf-8",
        url=url, body=response.content)), \
     patch.object(local_agent, "available_models", return_value=[{"slug": "offline"}]), \
     patch.object(local_agent, "LocalModel", return_value=Model()), \
     patch.object(sys, "argv", ["local_agent.py", "ask", url,
                                "What is adult admission?", "--model", "offline"]), \
     redirect_stdout(io.StringIO()) as output:
    local_agent.main()
rendered = output.getvalue()
assert "Answer: 12 EUR" in rendered and "[1] adult_admission: 12 EUR" in rendered
assert f"Source: {url}" in rendered and "Passage: Adult admission: 12 EUR." in rendered
assert "Support: exact" in rendered
case = {"id": "offline_citation", "group": "real", "url": url,
        "question": "What is adult admission?",
        "expected": {"status": "answered", "answers": ["12 EUR"]}}
with patch.object(retrieval, "fetch_scrapling_response", return_value=SimpleNamespace(
        status=200, headers={"content-type": "text/html"}, encoding="utf-8",
        url=url, body=response.content)):
    row = run_case(case, "original", Model(), None, 1)
assert row["passed"], row
assert row["actual"]["citations"][0]["quote"] == "Adult admission: 12 EUR."
assert json.loads(json.dumps(row))["actual"]["citations"][0]["source_url"] == url
print("PASS: CLI displays citations and suite reports preserve them without changing answer scoring")
