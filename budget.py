"""Per-query monotonic time budget for Scrapling fetches."""
from contextvars import ContextVar
from time import monotonic

_end = ContextVar("query_end", default=None)


def set_deadline(end):
    return _end.set(end)


def remaining(default):
    end = _end.get()
    if end is None:
        return default
    left = end - monotonic()
    if left <= 0:
        raise TimeoutError("Query deadline expired")
    return min(default, left)
