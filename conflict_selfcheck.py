"""Offline checks for conflicting values across fetched pages."""
import io
import json
from contextlib import redirect_stdout

import agent


OLD = "https://example.org/archive"
NEW = "https://example.org/current"
OLD_QUOTE = "In 2024, adult admission was 8 EUR."
NEW_QUOTE = "Current adult admission is 12 EUR."


def page(url, text, links=()):
    return {"url": url, "title": "Tickets", "text": text,
            "links": list(links), "forms": []}


class Model:
    def __init__(self, conflict_choice=1, final_choice=1):
        self.conflict_choice = conflict_choice
        self.final_choice = final_choice
        self.conflict_calls = 0

    def generate(self, messages):
        request = json.loads(messages[-1]["content"])
        if "request_to_analyze" in request:
            return json.dumps({"required_fields": ["adult_admission"]})
        if "page" in request:
            text = request["page"]["text"]
            if OLD_QUOTE in text:
                return json.dumps({"adult_admission": {
                    "value": "8 EUR", "quote": OLD_QUOTE,
                    "confidence": .99, "reason": "Archive price"}})
            if NEW_QUOTE in text:
                return json.dumps({"adult_admission": {
                    "value": "12 EUR", "quote": NEW_QUOTE,
                    "confidence": .99, "reason": "Current price"}})
            return "{}"
        if "candidates" in request and "target_field" in request:
            self.conflict_calls += 1
            return json.dumps({
                "candidate_index": self.conflict_choice,
                "reason": ("The current page identifies the current price; "
                           "the archive dates the old price to 2024."
                           if self.conflict_choice is not None else
                           "Neither passage establishes which price applies."),
            })
        if "target_field" in request:
            index = self.final_choice
            quote, value = ((OLD_QUOTE, "8 EUR") if index == 0 else
                            (NEW_QUOTE, "12 EUR"))
            return json.dumps({"page_index": index, "quote": quote,
                               "answer": value, "reason": "The page states this price."})
        return '{"link_index": 0}'


def run(model, new_text=NEW_QUOTE):
    pages = {
        OLD: page(OLD, OLD_QUOTE, [{"url": NEW, "text": "Current tickets"}]),
        NEW: page(NEW, new_text),
    }
    fetched = []
    def fetch(url):
        fetched.append(url)
        return pages[url]
    with redirect_stdout(io.StringIO()):
        result = agent.run_agent("What is current adult admission?", OLD,
                                 model, None, max_pages=2, fetch_page_fn=fetch)
    return result, fetched


result, fetched = run(Model())
assert fetched == [OLD, NEW], fetched
assert result["status"] == "answered" and result["answer"] == "12 EUR", result
assert result["citations"][0]["source_url"] == NEW
assert result["conflicts"][0]["status"] == "resolved"
assert [c["value"] for c in result["conflicts"][0]["candidates"]] == ["8 EUR", "12 EUR"]
print("PASS: first-page confidence cannot bypass a linked conflicting value")

result, fetched = run(Model(conflict_choice=None))
assert result["status"] == "abstained" and "Conflicting grounded values" in result["failure_reason"]
assert result["conflicts"][0]["status"] == "unresolved" and not result.get("citations")
print("PASS: ambiguous grounded claims lead to an explained abstention")

result, _ = run(Model(conflict_choice=42))
assert result["status"] == "abstained" and "invalid candidate" in result["failure_reason"]
print("PASS: invalid adjudication cannot select a fabricated source")

result, _ = run(Model(final_choice=0))
assert result["status"] == "abstained" and result["conflicts"][0]["status"] == "unresolved"
assert "disagreed" in result["failure_reason"]
print("PASS: final answer cannot contradict the conflict decision")

model = Model()
result, _ = run(model, new_text="No price listed here.")
assert model.conflict_calls == 0 and result["status"] == "abstained"
print("PASS: no conflict review runs without two grounded values")
