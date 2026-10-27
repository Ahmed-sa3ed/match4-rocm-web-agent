"""File-output interface; Match 4's official harness contract is unconfirmed."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit
from agent import run_agent
from decision import LocalModel, DEFAULT_MODEL
from retrieval_browser import render_page

def main():
    parser = argparse.ArgumentParser(description="Retrieve a grounded answer from a website")
    parser.add_argument("--url", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--query-id", default="query")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-pages", type=int, default=4)
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", args.query_id):
        parser.error("query-id must contain 1-100 letters, digits, underscores or hyphens")
    if not 1 <= args.max_pages <= 20:
        parser.error("max-pages must be between 1 and 20")
    if urlsplit(args.url).scheme not in ("http", "https") or not urlsplit(args.url).hostname:
        parser.error("url must be an HTTP(S) URL")
    with redirect_stdout(sys.stderr):
        result = run_agent(args.query, args.url, LocalModel(args.model), None,
                           max_pages=args.max_pages,
                           render_page_fn=None if args.static_only else render_page)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / (args.query_id + "_output.json")
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    main()
