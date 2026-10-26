import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import parse_qs, urlsplit

from agent import run_agent


PROJECT = Path(__file__).resolve().parent


class SearchTestHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        target = urlsplit(self.path)
        if target.path != "/search":
            return super().do_GET()

        self.server.search_requests.append(self.path)
        fields = parse_qs(target.query)
        query = fields.get("q", [""])[0]

        if (
            fields.get("section") == ["museum"]
            and "maps" in query.lower()
        ):
            message = "Historic maps are in Gallery 2."
        else:
            message = "No results found."

        body = (
            f"<html><title>Search results</title>"
            f"<p>{message}</p></html>"
        ).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_search_checks(model, tokenizer):
    fixture = PROJECT / "test_site/search-form.html"
    if not fixture.exists():
        raise FileNotFoundError(f"Missing test page: {fixture}")

    handler = partial(SearchTestHandler, directory=str(fixture.parent))

    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        server.search_requests = []
        Thread(target=server.serve_forever, daemon=True).start()

        try:
            start_url = (
                f"http://127.0.0.1:{server.server_port}"
                "/search-form.html"
            )

            found = run_agent(
                "In which gallery are Cedar Museum's historic maps displayed?",
                start_url,
                model,
                tokenizer,
            )
            found_requests = len(server.search_requests)

            missing = run_agent(
                "Which gallery displays Cedar Museum's dinosaur fossils?",
                start_url,
                model,
                tokenizer,
            )
            missing_requests = (
                len(server.search_requests) - found_requests
            )
        finally:
            server.shutdown()

    reports = [
        {
            "case": "Search finds answer",
            "passed": (
                found_requests > 0
                and found["status"] == "answered"
                and "Gallery 2" in found["answer"]
            ),
            "actual": found,
            "search_requests": found_requests,
        },
        {
            "case": "Search finds nothing",
            "passed": (
                missing_requests > 0
                and missing == {"status": "abstained", "answer": ""}
            ),
            "actual": missing,
            "search_requests": missing_requests,
        },
    ]

    report_path = PROJECT / "search-check-results.json"
    report_path.write_text(
        json.dumps(reports, indent=2),
        encoding="utf-8",
    )

    for report in reports:
        outcome = "PASS" if report["passed"] else "FAIL"
        print(f"{outcome} | {report['case']}")

    assert all(report["passed"] for report in reports), report_path
    return reports
