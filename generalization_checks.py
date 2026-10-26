"""Small holdout scenarios for the notebook's actual model."""

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from urllib.parse import urljoin


PAGES = {
    "aquarium/tickets.html": """
        <title>Harbor Aquarium ticket prices</title>
        <h1>Current adult admission</h1><p>Adult ticket: 23 GBP.</p>
        <a href='winter.html'>Winter 2025 ticket archive</a>
    """,
    "aquarium/winter.html": """
        <title>Harbor Aquarium winter 2025 archive</title>
        <h1>Winter 2025 tickets</h1>
        <p>In winter 2025, adult admission was 18 GBP.</p>
        <a href='tickets.html'>Current tickets</a>
    """,
    "observatory/team.html": """
        <title>Northbridge Observatory team</title>
        <h1>Current staff</h1><p>Director: Mateo Reed.</p>
        <a href='team-2021.html'>Staff history</a>
    """,
    "observatory/team-2021.html": """
        <title>Northbridge Observatory 2021 staff record</title>
        <h1>Staff in 2021</h1><p>In 2021, the director was Elena Cho.</p>
        <a href='team.html'>Current staff</a>
    """,
    "ferry/timetable-archive.html": """
        <title>Cedar Bay Ferry timetable archive</title>
        <h1>Timetable for 2023</h1><p>Last departure: 19:30.</p>
        <a href='timetable.html'>Current timetable</a>
    """,
    "ferry/timetable.html": """
        <title>Cedar Bay Ferry current timetable</title>
        <h1>Current timetable</h1><p>Last departure: 21:15.</p>
        <a href='timetable-archive.html'>Archive</a>
    """,
    "observatory/events.html": """
        <title>Northbridge Observatory events</title>
        <h1>Public events</h1><p>Public tours occur on Fridays.</p>
    """,
}


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


def run_generalization_checks(model, tokenizer):
    from agent import run_agent

    cases = [
        ("past price, current page first", "aquarium/tickets.html",
         "How much was adult admission at Harbor Aquarium in winter 2025?",
         {"status": "answered", "answer": "18 GBP"}),
        ("current price, archive first", "aquarium/winter.html",
         "What is adult admission at Harbor Aquarium now?",
         {"status": "answered", "answer": "23 GBP"}),
        ("past person, current page first", "observatory/team.html",
         "Who directed Northbridge Observatory in 2021?",
         {"status": "answered", "answer": "Elena Cho"}),
        ("current time, archive first", "ferry/timetable-archive.html",
         "What is the last Cedar Bay Ferry departure on the current timetable?",
         {"status": "answered", "answer": "21:15"}),
        ("no answer available", "observatory/events.html",
         "What is the adult ticket price at Northbridge Observatory?",
         {"status": "abstained", "answer": ""}),
    ]
    with TemporaryDirectory(prefix="match4-holdout-") as directory:
        root = Path(directory)
        for filename, html in PAGES.items():
            path = root / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(html, encoding="utf-8")

        handler = partial(QuietHandler, directory=str(root))
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{server.server_port}/"
            reports = []
            for name, path, question, expected in cases:
                actual = run_agent(
                    question, urljoin(base_url, path), model, tokenizer
                )
                reports.append({"case": name, "expected": expected,
                                "actual": actual, "passed": actual == expected})
                print("PASS" if actual == expected else "FAIL", "|", name,
                      "|", actual)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
    return reports
