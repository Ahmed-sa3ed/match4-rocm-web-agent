"""Keep agent progress separate from the command's final result."""
from contextlib import contextmanager, redirect_stdout
import os
import sys


@contextmanager
def agent_output(audit=False):
    if audit:
        previous = os.environ.get("MATCH4_MODEL_METRICS")
        os.environ["MATCH4_MODEL_METRICS"] = "1"
        try:
            with redirect_stdout(sys.stderr):
                yield
        finally:
            if previous is None:
                os.environ.pop("MATCH4_MODEL_METRICS", None)
            else:
                os.environ["MATCH4_MODEL_METRICS"] = previous
    else:
        with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink):
            yield
