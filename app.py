"""File-output interface; Match 4's official harness contract is unconfirmed."""
import argparse
from contextlib import redirect_stdout
import json
import multiprocessing
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from urllib.parse import urlsplit

from agent import run_agent
from budget import set_deadline
from decision import LocalModel, DEFAULT_MODEL
from retrieval_browser import render_page

PROJECT = Path(__file__).resolve().parent
EMPTY = {"status": "abstained", "answer": "", "citations": []}
ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,100}")


def atomic_json(target, result):
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + target.name, dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(result, output, ensure_ascii=False, indent=2)
            output.write("\n")
        os.chmod(temporary, 0o644)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def recover_target(argv):
    """Keep a safe output identity even if argument parsing fails."""
    values = {}
    for index, token in enumerate(argv):
        flag, equal, value = token.partition("=")
        if flag in ("--query-id", "--output-dir"):
            if not equal:
                value = argv[index + 1] if index + 1 < len(argv) else ""
            if value and not value.startswith("-"):
                values[flag] = value
    query_id = values.get("--query-id", "query")
    if not ID_PATTERN.fullmatch(query_id):
        query_id = "query"
    folder = Path(values.get("--output-dir", PROJECT / "output"))
    return folder / (query_id + "_output.json")


def parser_for_query():
    parser = argparse.ArgumentParser(description="Retrieve a grounded answer from a website")
    parser.add_argument("--url", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--query-id", default="query")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "output")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-pages", type=int, default=4)
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--deadline-seconds", type=float, default=28.0,
                        help="Whole-query wall time; leave time for output recovery")
    return parser


def query_worker(args, deadline, sender):
    """Isolate model, browser and network stalls from the output supervisor."""
    try:
        set_deadline(deadline)
        with redirect_stdout(sys.stderr):
            candidate = run_agent(args.query, args.url, LocalModel(args.model), None,
                                  max_pages=args.max_pages,
                                  render_page_fn=None if args.static_only else render_page)
        if (not isinstance(candidate, dict) or candidate.get("status") not in ("answered", "abstained")
                or not isinstance(candidate.get("answer"), str)
                or not isinstance(candidate.get("citations", []), list)
                or (candidate["status"] == "answered" and not candidate["answer"])):
            raise ValueError("Agent returned an invalid result")
        sender.send(candidate)
    except Exception as exc:
        print(f"Agent failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        sender.close()


def main():
    argv = sys.argv[1:]
    parser = parser_for_query()
    if "--help" in argv or "-h" in argv:
        parser.parse_args(argv)
        return
    target = recover_target(argv)
    atomic_json(target, EMPTY)
    try:
        args = parser.parse_args(argv)
        if not ID_PATTERN.fullmatch(args.query_id):
            raise ValueError("query-id must contain 1-100 letters, digits, underscores or hyphens")
        if not 1 <= args.max_pages <= 20:
            raise ValueError("max-pages must be between 1 and 20")
        if urlsplit(args.url).scheme not in ("http", "https") or not urlsplit(args.url).hostname:
            raise ValueError("url must be an HTTP(S) URL")
        if not math.isfinite(args.deadline_seconds) or args.deadline_seconds < 1:
            raise ValueError("deadline-seconds must be a finite number of at least 1")
        started = time.monotonic()
        context = multiprocessing.get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        worker = context.Process(target=query_worker, args=(args, started + args.deadline_seconds - 0.5, sender))
        worker.start()
        sender.close()
        cutoff = started + args.deadline_seconds - 0.5
        candidate = None
        if receiver.poll(max(0, cutoff - time.monotonic())):
            try:
                candidate = receiver.recv()
            except EOFError:
                pass
        worker.join(max(0, cutoff - time.monotonic()))
        if worker.is_alive():
            worker.terminate()
            worker.join(0.2)
            if worker.is_alive():
                worker.kill()
                worker.join(0.2)
            print("Agent exceeded query deadline; preserving last valid output", file=sys.stderr)
        elif worker.exitcode:
            print(f"Agent worker exited with code {worker.exitcode}", file=sys.stderr)
        elif candidate is not None:
            atomic_json(target, candidate)
        receiver.close()
    except (Exception, SystemExit) as exc:
        print(f"Agent failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    try:
        result = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        result = EMPTY
        atomic_json(target, result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
