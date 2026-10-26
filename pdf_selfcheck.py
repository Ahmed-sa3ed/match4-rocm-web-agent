"""Offline PDF retrieval, navigation, page citation, and failure checks."""
import io
import json
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

import agent
import retrieval
import retrieval_scrapling
from retrieval import UnsupportedContentTypeError

START = "https://example.org/reports"
PDF = "https://example.org/files/fees?download=1"
QUOTE = "Adult admission is 12 EUR."


def make_pdf(*lines):
    writer = PdfWriter()
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    font_ref = writer._add_object(font)
    for line in lines:
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})
        })
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 72 720 Td ({line}) Tj ET".encode())
        page[NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


pdf_bytes = make_pdf("Old tickets are unavailable.", QUOTE)


def response(url):
    if url == START:
        return SimpleNamespace(url=url, headers={"Content-Type": "text/html"},
                               body=(f'<html><title>Reports</title><a href="{PDF}">'
                                     'Admission report</a></html>').encode(),
                               encoding="utf-8")
    return SimpleNamespace(url=PDF, headers={"Content-Type": "application/pdf"},
                           body=pdf_bytes, encoding="utf-8")


class Model:
    def generate(self, messages):
        request = json.loads(messages[-1]["content"])
        if "request_to_analyze" in request:
            return json.dumps({"required_fields": ["adult_admission"]})
        if "page" in request:
            if QUOTE in request["page"]["text"]:
                return json.dumps({"adult_admission": {
                    "value": "12 EUR", "quote": QUOTE,
                    "confidence": .98, "reason": "The report lists the fee."}})
            return "{}"
        if "target_field" in request:
            return json.dumps({"page_index": 1 if len(request["pages"]) > 1 else 0,
                               "quote": QUOTE, "answer": "12 EUR",
                               "reason": "The PDF page states the admission."})
        return '{"link_index": 0}'


with patch.object(retrieval, "fetch_scrapling_response", side_effect=response):
    pdf = retrieval.fetch_page(PDF)
    assert pdf["url"] == PDF and pdf["pdf_pages"][1]["number"] == 2
    assert QUOTE in pdf["text"] and pdf["render_hint"] is False
    with redirect_stdout(io.StringIO()):
        result = agent.run_agent("What is adult admission?", START,
                                 Model(), None, fetch_page_fn=retrieval.fetch_page,
                                 max_pages=2)
assert result["status"] == "answered" and result["answer"] == "12 EUR", result
assert result["citations"][0]["source_url"] == PDF
assert result["citations"][0]["page_number"] == 2
assert result["citations"][0]["quote"] == QUOTE
print("PASS: Scrapling follows an HTML link to a PDF and cites the extracted page")

with patch.object(retrieval, "fetch_scrapling_response", return_value=response(PDF)), \
     redirect_stdout(io.StringIO()):
    direct = agent.run_agent("What is adult admission?", PDF,
                             Model(), None, fetch_page_fn=retrieval.fetch_page,
                             max_pages=1)
assert direct["status"] == "answered" and direct["citations"][0]["page_number"] == 2
print("PASS: direct PDF start URLs use the same evidence and citation path")

with patch.object(retrieval_scrapling, "fetch_scrapling_response", return_value=response(PDF)):
    parsed = retrieval_scrapling.fetch_page(PDF)
assert parsed["text"] == pdf["text"] and parsed["pdf_pages"] == pdf["pdf_pages"]
print("PASS: both suite retrieval backends share PDF parsing")

with patch.object(retrieval, "fetch_scrapling_response", return_value=SimpleNamespace(
        url=PDF, headers={"content-type": "application/octet-stream"},
        body=pdf_bytes, encoding="utf-8")):
    assert retrieval.fetch_page(PDF)["pdf_pages"][1]["number"] == 2
print("PASS: a PDF signature works without a .pdf URL or PDF content type")

for body, expected in [(b"invalid", "signature"),
                       (make_pdf(), "no extractable text"),
                       (b"%PDF-" + b"a" * (8 * 1024 * 1024), "8 MiB")]:
    with patch.object(retrieval, "fetch_scrapling_response", return_value=SimpleNamespace(
            url=PDF, headers={"content-type": "application/pdf"},
            body=body, encoding="utf-8")):
        try:
            retrieval.fetch_page(PDF)
            raise AssertionError("Unreadable PDF was accepted")
        except UnsupportedContentTypeError as exc:
            assert expected in str(exc), str(exc)
print("PASS: malformed, image-only, and oversized PDFs fail with a useful reason")

# PDF line wrapping can make a fluent model quotation differ from the bytes
# extracted from the page. Retry once, then still require exact source text.
from retrieval_pdf import parse_pdf
EXACT = "Rosetta is scheduled to conduct anoth- er flyby of Lutetia on 10 July 2010."
FLUENT = "Rosetta is scheduled to conduct another flyby of Lutetia on 10 July 2010."
wrapped_pdf = parse_pdf(make_pdf(EXACT), PDF)

class QuoteRepairModel:
    def __init__(self, repair=True):
        self.repair = repair
        self.retries = 0
        self.final_context = ""

    def generate(self, messages):
        request = json.loads(messages[-1]["content"])
        if "request_to_analyze" in request:
            return json.dumps({"required_fields": ["flyby_date"]})
        if "page" in request:
            if "10 July 2010" not in request["page"]["text"]:
                return "{}"
            if "validation_feedback" in request:
                self.retries += 1
            quote = EXACT if self.repair and "validation_feedback" in request else FLUENT
            return json.dumps({"flyby_date": {"value": "10 July 2010",
                                               "quote": quote, "confidence": .99}})
        if "target_field" in request:
            self.final_context = request["pages"][0]["text"]
            return json.dumps({"page_index": 0, "answer": "10 July 2010",
                               "quote": EXACT, "reason": "The passage supplies the date."})
        return '{"link_index": null}'

repair_model = QuoteRepairModel()
with redirect_stdout(io.StringIO()):
    repaired = agent.run_agent("What date was the Lutetia flyby scheduled?", PDF,
                               repair_model, None, max_pages=1,
                               fetch_page_fn=lambda _: wrapped_pdf)
assert repair_model.retries == 1 and repaired["status"] == "answered"
assert repaired["citations"][0]["page_number"] == 1
assert repaired["citations"][0]["quote"] == EXACT
print("PASS: PDF quote repair copies exact extracted text and retains page citation")

long_text = "Irrelevant background. " * 300 + "\n[PDF page 2]\n" + EXACT
long_pdf = {**wrapped_pdf, "text": long_text,
            "pdf_pages": [{"number": 2, "text": EXACT}]}
fallback_model = QuoteRepairModel(repair=False)
with redirect_stdout(io.StringIO()):
    fallback = agent.run_agent("What date was the Lutetia flyby scheduled?", PDF,
                               fallback_model, None, max_pages=1,
                               fetch_page_fn=lambda _: long_pdf)
assert fallback_model.retries == 1 and EXACT in fallback_model.final_context
assert fallback["status"] == "answered" and fallback["citations"][0]["page_number"] == 2
print("PASS: final review sees relevant long-PDF text after an invalid candidate quote")
