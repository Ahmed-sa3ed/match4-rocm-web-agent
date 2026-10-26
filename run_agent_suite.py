"""Run repeatable local and live website checks for the Match 4 agent."""
import argparse
import io
import json
import re
import statistics
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, redirect_stdout
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from time import perf_counter
from urllib.parse import urljoin, urlsplit

from agent import run_agent
from decision import available_models
from decision import VLLMModel
from text_match import match_span

PROJECT = Path(__file__).resolve().parent
BACKENDS = ("original", "scrapling")
_case_output = ContextVar("case_output", default=None)


class RoutedStdout:
    """Send worker prints to their own trace while preserving console output."""

    def __init__(self, original):
        self.original = original

    def write(self, value):
        target = _case_output.get() or self.original
        return target.write(value)

    def flush(self):
        target = _case_output.get() or self.original
        return target.flush()

    def __getattr__(self, name):
        return getattr(self.original, name)


@contextmanager
def capture_case_output(buffer):
    if isinstance(sys.stdout, RoutedStdout):
        token = _case_output.set(buffer)
        try:
            yield
        finally:
            _case_output.reset(token)
    else:
        with redirect_stdout(buffer):
            yield


@contextmanager
def route_worker_output():
    original = sys.stdout
    sys.stdout = RoutedStdout(original)
    try:
        yield
    finally:
        sys.stdout = original



def normalized(value):
    collapsed = " ".join(unicodedata.normalize("NFKC", value).casefold().split())
    return re.sub(r"(?<=[$â‚¬Â£])\s+(?=\d)", "", collapsed).rstrip(".")


def matches_expected(expected, result):
    if not isinstance(result, dict) or result.get("status") != expected["status"]:
        return False
    answer = result.get("answer")
    if not isinstance(answer, str):
        return False
    if expected["status"] == "abstained":
        return answer == ""
    return normalized(answer) in {normalized(item) for item in expected["answers"]}


def read_suite(path):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported agent_suite.json schema_version.")
    if not isinstance(data.get("model"), str) or not data["model"].strip():
        raise ValueError("Set 'model' to a slug from local_agent.py models.")
    max_pages = data.get("max_pages", 4)
    if type(max_pages) is not int or not 1 <= max_pages <= 20:
        raise ValueError("max_pages must be an integer from 1 to 20.")
    backends = data.get("backends")
    if not isinstance(backends, list) or not backends or any(b not in BACKENDS for b in backends):
        raise ValueError("backends must be a nonempty list of original/scrapling.")
    if len(backends) != len(set(backends)):
        raise ValueError("Duplicate backends in configuration.")
    if type(data.get("repeats")) is not int or not 1 <= data["repeats"] <= 5:
        raise ValueError("repeats must be an integer from 1 to 5.")
    if data.get("answer_comparison") != "casefold_whitespace_terminal_period":
        raise ValueError("Only casefold_whitespace_terminal_period comparison is supported.")
    report_dir = data.get("report_directory")
    if not isinstance(report_dir, str) or not report_dir.strip():
        raise ValueError("Set report_directory to a relative folder name.")
    report_path = (PROJECT / report_dir).resolve()
    if not report_path.is_relative_to(PROJECT) or report_path == PROJECT:
        raise ValueError("Report directory must remain inside the project.")
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("The suite needs at least one case.")
    ids = set()
    fixture_root = (PROJECT / "test_site").resolve()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str) or not case["id"]:
            raise ValueError("Every case needs a nonempty id.")
        if case["id"] in ids:
            raise ValueError(f"Duplicate case id: {case['id']}")
        ids.add(case["id"])
        if case.get("group") not in ("local", "real"):
            raise ValueError(f"{case['id']}: group must be local or real.")
        if not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ValueError(f"{case['id']}: missing question.")
        if case["group"] == "local":
            name = case.get("fixture_path")
            if not isinstance(name, str) or not name or "url" in case:
                raise ValueError(f"{case['id']}: supply fixture_path only.")
            fixture = (fixture_root / name).resolve()
            if not fixture.is_relative_to(fixture_root) or not fixture.is_file():
                raise ValueError(f"{case['id']}: fixture file missing or outside test_site.")
        else:
            url = case.get("url")
            if (not isinstance(url, str) or "fixture_path" in case
                    or urlsplit(url).scheme not in ("http", "https")
                    or not urlsplit(url).netloc):
                raise ValueError(f"{case['id']}: supply an HTTP(S) url only.")
        for feature in ("requires_browser", "requires_pdf", "requires_conflict"):
            if type(case.get(feature, False)) is not bool:
                raise ValueError(f"{case['id']}: {feature} must be a boolean.")
        if case.get("feature_focus") not in (None, "pdf", "conflict"):
            raise ValueError(f"{case['id']}: feature_focus must be pdf or conflict.")
        expected = case.get("expected")
        if not isinstance(expected, dict) or expected.get("status") not in ("answered", "abstained"):
            raise ValueError(f"{case['id']}: expected.status is invalid.")
        answers = expected.get("answers")
        if not isinstance(answers, list) or any(not isinstance(x, str) or not x for x in answers):
            raise ValueError(f"{case['id']}: expected.answers must be a list of answers.")
        if expected["status"] == "answered" and not answers:
            raise ValueError(f"{case['id']}: answered needs accepted answers.")
        if expected["status"] == "abstained" and answers:
            raise ValueError(f"{case['id']}: abstained requires an empty answers list.")
    return data, report_path


def feature_evidence(result):
    """A correct answer alone does not show that a feature was exercised."""
    pdf = any(type(item.get("page_number")) is int and item["page_number"] > 0
              for item in result.get("citations", []))
    needed_status = "resolved" if result.get("status") == "answered" else "unresolved"
    conflict = any(
        item.get("status") == needed_status
        and len({str(candidate.get("value", "")).casefold()
                 for candidate in item.get("candidates", [])}) > 1
        for item in result.get("conflicts", [])
    )
    return pdf, conflict


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


@contextmanager
def local_fixture_server(enabled):
    if not enabled:
        yield None
        return
    fixture_dir = PROJECT / "test_site"
    handler = partial(QuietHandler, directory=str(fixture_dir))
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/"
        finally:
            server.shutdown()
            worker.join(timeout=5)


def fetcher_for(backend):
    if backend == "original":
        from retrieval import fetch_page
    elif backend == "scrapling":
        from retrieval_scrapling import fetch_page
    else:
        raise ValueError(f"Unknown backend: {backend}")
    return fetch_page


def run_case(case, backend, model, local_base, repeat, *,
             max_pages=4, static_only=False):
    url = (urljoin(local_base, case["fixture_path"])
           if case["group"] == "local" else case["url"])
    fetch = fetcher_for(backend)
    metrics = {"fetch_attempts": 0, "pages": 0, "browser_attempts": 0,
               "browser_seconds": 0.0, "retrieval_seconds": 0.0}
    static_texts = {}
    rendered_texts = {}

    def timed_fetch(target):
        metrics["fetch_attempts"] += 1
        started = perf_counter()
        try:
            page = fetch(target)
            metrics["pages"] += 1
            static_texts[page["url"]] = page["text"]
            return page
        finally:
            metrics["retrieval_seconds"] += perf_counter() - started

    def timed_render(target, static_length):
        from retrieval_browser import render_page
        metrics["browser_attempts"] += 1
        started = perf_counter()
        try:
            page = render_page(target, static_length)
            rendered_texts[page["url"]] = page["text"]
            return page
        finally:
            elapsed = perf_counter() - started
            metrics["browser_seconds"] += elapsed
            metrics["retrieval_seconds"] += elapsed

    decisions = []

    class ObservedModel:
        def generate(self, messages):
            request = json.loads(messages[-1]["content"])
            reply = model.generate(messages)
            if "target_field" in request:
                decisions.append({
                    "field": request["target_field"],
                    "candidate_reviews": request.get("candidate_reviews", []),
                    "model_reply": reply[:2000],
                })
            return reply

    buffer = io.StringIO()
    started = perf_counter()
    try:
        with capture_case_output(buffer):
            result = run_agent(case["question"], url, ObservedModel(), None,
                               fetch_page_fn=timed_fetch,
                               render_page_fn=None if static_only else timed_render,
                               max_pages=max_pages)
        error = None
    except Exception as exc:
        result = {"status": "error", "answer": ""}
        error = f"{type(exc).__name__}: {exc}"
    elapsed = perf_counter() - started
    pdf_exercised, conflict_exercised = feature_evidence(result)
    browser_exercised = any(
        source == url and source in rendered_texts
        and match_span(rendered_texts[source], citation["quote"]) is not None
        and match_span(static_texts.get(source, ""), citation["quote"]) is None
        for citation in result.get("citations", [])
        for source in (citation["source_url"],)
    )
    return {
        "id": case["id"], "group": case["group"], "repeat": repeat,
        "backend": backend, "start_url": url, "question": case["question"],
        "expected": case["expected"], "actual": result,
        "requires_browser": case.get("requires_browser", False),
        "requires_pdf": case.get("requires_pdf", False),
        "requires_conflict": case.get("requires_conflict", False),
        "pdf_exercised": pdf_exercised,
        "conflict_exercised": conflict_exercised,
        "browser_exercised": browser_exercised,
        "passed": (error is None and matches_expected(case["expected"], result)
                   and (not case.get("requires_browser") or browser_exercised)
                   and (not case.get("requires_pdf") or pdf_exercised)
                   and (not case.get("requires_conflict") or conflict_exercised)),
        "total_seconds": round(elapsed, 3),
        "retrieval_seconds": round(metrics["retrieval_seconds"], 3),
        "other_seconds": round(max(0.0, elapsed - metrics["retrieval_seconds"]), 3),
        "fetch_attempts": metrics["fetch_attempts"], "pages_fetched": metrics["pages"],
        "browser_attempts": metrics["browser_attempts"],
        "browser_seconds": round(metrics["browser_seconds"], 3),
        "error": error, "decisions": decisions, "trace": buffer.getvalue(),
    }


def summarize(rows, backends):
    by_backend = {}
    for backend in backends:
        sample = [row for row in rows if row["backend"] == backend]
        good = [row for row in sample if row["passed"]]
        by_backend[backend] = {
            "passed": len(good), "attempted": len(sample),
            "median_seconds_all": round(statistics.median(
                row["total_seconds"] for row in sample), 3),
            "median_seconds_correct": (
                round(statistics.median(row["total_seconds"] for row in good), 3)
                if good else None),
            "total_seconds": round(sum(row["total_seconds"] for row in sample), 3),
        }
    wins = {backend: 0 for backend in backends}
    if len(backends) == 2:
        pairs = {}
        for row in rows:
            pairs.setdefault((row["id"], row["repeat"]), {})[row["backend"]] = row
        for pair in pairs.values():
            if (len(pair) == 2 and all(row["passed"] for row in pair.values())):
                a, b = backends
                if pair[a]["total_seconds"] != pair[b]["total_seconds"]:
                    winner = min(backends, key=lambda name: pair[name]["total_seconds"])
                    wins[winner] += 1
    return {"backends": by_backend, "paired_speed_wins_when_both_correct": wins}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "agent_suite.json")
    parser.add_argument("--only", choices=("all", "local", "real"), default="all")
    parser.add_argument("--case", action="append", help="Run one id; repeat for multiple cases.")
    parser.add_argument("--feature", action="append", choices=("pdf", "conflict"),
                        help="Run cases exercising this feature; repeat to combine them.")
    parser.add_argument("--model", help="Override model from JSON.")
    parser.add_argument("--backends", nargs="+", choices=BACKENDS,
                        help="Override backends from JSON.")
    parser.add_argument("--repeats", type=int, help="Override repeats from JSON.")
    parser.add_argument("--max-pages", type=int,
                        help="Maximum successfully fetched pages per case (1-20).")
    parser.add_argument("--workers", type=int, default=1,
                        help="Run up to this many independent cases concurrently (1-4; default 1).")
    parser.add_argument("--static-only", action="store_true",
                        help="Disable browser rendering for baseline comparison.")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without network or model calls.")
    args = parser.parse_args(argv)
    try:
        config, report_dir = read_suite(args.config.resolve())
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    model_slug = args.model or config["model"]
    backends = args.backends or config["backends"]
    repeats = args.repeats if args.repeats is not None else config["repeats"]
    max_pages = args.max_pages if args.max_pages is not None else config.get("max_pages", 4)
    if not 1 <= max_pages <= 20:
        parser.error("--max-pages must be from 1 to 20.")
    if len(backends) != len(set(backends)) or not 1 <= repeats <= 5:
        parser.error("Use unique backends and 1â€“5 repeats.")
    if not 1 <= args.workers <= 4:
        parser.error("--workers must be between 1 and 4.")
    cases = [case for case in config["cases"]
             if args.only == "all" or case["group"] == args.only]
    if args.feature:
        cases = [case for case in cases
                 if any(case.get("feature_focus") == feature or
                        case.get(f"requires_{feature}") for feature in args.feature)]
    if args.case:
        unknown = set(args.case) - {case["id"] for case in config["cases"]}
        if unknown:
            parser.error("Unknown case id(s): " + ", ".join(sorted(unknown)))
        cases = [case for case in cases if case["id"] in args.case]
    if not cases:
        parser.error("No cases selected.")
    print(f"Model: {model_slug} | backends: {backends} | repeats: {repeats} | max pages: {max_pages}")
    print("Cases:", ", ".join(case["id"] for case in cases))
    if args.dry_run:
        print("Dry run: no requests sent.")
        return 0

    try:
        slugs = {entry["slug"] for entry in available_models()}
    except Exception as exc:
        parser.error(f"Could not check available models: {type(exc).__name__}: {exc}")
    if model_slug not in slugs:
        parser.error("Configured model unavailable; run local_agent.py models.")

    model = VLLMModel(model_slug)
    rows = []
    started_suite = perf_counter()
    with local_fixture_server(any(case["group"] == "local" for case in cases)) as base:
        jobs = []
        for repeat in range(1, repeats + 1):
            for i, case in enumerate(cases):
                order = (backends if (i + repeat) % 2 else list(reversed(backends)))
                for backend in order:
                    jobs.append((case, backend, repeat))

        if args.workers == 1:
            for case, backend, repeat in jobs:
                print(f"Running {case['id']} | {backend} | {repeat}/{repeats}", flush=True)
                row = run_case(case, backend, model, base, repeat,
                               max_pages=max_pages, static_only=args.static_only)
                rows.append(row)
                label = "PASS" if row["passed"] else "FAIL"
                print(f"  {label} | {row['total_seconds']}s | "
                      f"answer: {row['actual'].get('answer', '')!r}"
                      + (f" | {row['error']}" if row["error"] else ""), flush=True)
        else:
            with route_worker_output(), ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {}
                for index, (case, backend, repeat) in enumerate(jobs):
                    future = pool.submit(run_case, case, backend, model, base, repeat,
                                         max_pages=max_pages, static_only=args.static_only)
                    futures[future] = index
                completed = {}
                for future in as_completed(futures):
                    index = futures[future]
                    row = future.result()
                    completed[index] = row
                    label = "PASS" if row["passed"] else "FAIL"
                    print(f"  {label} | {row['id']} | {row['backend']} | "
                          f"{row['total_seconds']}s | answer: {row['actual'].get('answer', '')!r}"
                          + (f" | {row['error']}" if row["error"] else ""), flush=True)
                rows = [completed[index] for index in range(len(jobs))]

    wall_seconds = round(perf_counter() - started_suite, 3)
    summary = summarize(rows, backends)
    summary["suite_wall_seconds"] = wall_seconds
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_file": str(args.config.resolve()), "model": model_slug,
        "backends": backends, "repeats": repeats, "workers": args.workers,
        "max_pages": max_pages, "static_only": args.static_only,
        "answer_comparison": config["answer_comparison"],
        "summary": summary, "runs": rows,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archived = report_dir / f"agent-suite-{stamp}.json"
    latest = report_dir / "latest.json"
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    archived.write_text(serialized, encoding="utf-8")
    latest.write_text(serialized, encoding="utf-8")
    print("Summary:", json.dumps(summary, indent=2))
    print("Saved:", archived, "and", latest)
    return 0 if all(row["passed"] for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
