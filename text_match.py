"""Match source text despite line breaks and common typography."""
import re
import unicodedata

TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
SWAP = str.maketrans({
    "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"',
    "\u2010": "-", "\u2011": "-",
    "\u2013": "-", "\u2014": "-"
})


def tokens(text):
    for match in TOKEN.finditer(text):
        part = match.group()
        if all(unicodedata.category(c) == "Cf" for c in part):
            continue
        yield (
            unicodedata.normalize("NFKC", part).translate(SWAP).casefold(),
            match.start(),
            match.end(),
        )


def match_span(source, proposed):
    """Return source span; keep every word, number, and punctuation token."""
    if not isinstance(source, str) or not isinstance(proposed, str):
        return None
    wanted = [token for token, _, _ in tokens(proposed)]
    if not wanted:
        return None
    actual = list(tokens(source))
    for start in range(len(actual) - len(wanted) + 1):
        if all(
            actual[start + j][0] == token
            for j, token in enumerate(wanted)
        ):
            return source[
                actual[start][1]:
                actual[start + len(wanted) - 1][2]
            ]
    return None
