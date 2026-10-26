"""Split page text for bounded model requests without losing source text."""


def text_chunks(text, size=5000, overlap=250):
    if size <= overlap or overlap < 0:
        raise ValueError("Chunk size must be greater than overlap.")
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = max(text.rfind("\n", start + size // 2, end),
                           text.rfind(" ", start + size // 2, end))
            if boundary > start:
                end = boundary + 1
        yield text[start:end]
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
