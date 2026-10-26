"""Turn bounded, text-based PDF responses into the agent's page contract."""
from io import BytesIO
from pathlib import PurePosixPath
from urllib.parse import unquote, urlsplit

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from retrieval import UnsupportedContentTypeError
from text_match import match_span

MAX_PDF_BYTES = 8 * 1024 * 1024
MAX_PDF_PAGES = 40
MAX_PDF_TEXT = 200_000
MAX_PAGE_STREAM_BYTES = 2 * 1024 * 1024


def parse_pdf(body, url):
    """Extract selectable text; scanned or unreadable PDFs remain unsupported."""
    if not body.startswith(b"%PDF-"):
        raise UnsupportedContentTypeError("PDF response has no PDF signature.")
    if len(body) > MAX_PDF_BYTES:
        raise UnsupportedContentTypeError("PDF exceeds the 8 MiB download limit.")
    try:
        reader = PdfReader(BytesIO(body), strict=False)
        if reader.is_encrypted:
            raise UnsupportedContentTypeError("Password-protected PDF cannot be read.")
        total = len(reader.pages)
        texts = []
        length = 0
        for number, page in enumerate(reader.pages[:MAX_PDF_PAGES], 1):
            contents = page.get_contents()
            if contents is not None and len(contents.get_data()) > MAX_PAGE_STREAM_BYTES:
                raise UnsupportedContentTypeError(
                    f"PDF page {number} exceeds the 2 MiB content-stream limit.")
            extracted = (page.extract_text() or "").strip()
            if not extracted:
                continue
            available = MAX_PDF_TEXT - length
            if available <= 0:
                break
            extracted = extracted[:available]
            texts.append({"number": number, "text": extracted})
            length += len(extracted)
            if length >= MAX_PDF_TEXT:
                break
    except (PdfReadError, OSError, ValueError, KeyError, TypeError) as exc:
        raise UnsupportedContentTypeError(f"Could not read PDF: {exc}") from exc
    if not texts:
        raise UnsupportedContentTypeError(
            "PDF has no extractable text (scanned images require OCR).")
    name = unquote(PurePosixPath(urlsplit(url).path).name) or "PDF document"
    return {
        "url": url, "title": name,
        "text": "\n\n".join(f"[PDF page {part['number']}]\n{part['text']}"
                            for part in texts),
        "text_truncated": total > MAX_PDF_PAGES or length >= MAX_PDF_TEXT,
        "links": [], "links_truncated": False, "forms": [],
        "render_hint": False, "pdf_pages": texts,
    }


def pdf_page_number(page, quote):
    """Find the actual PDF page containing a final verified quotation."""
    for part in page.get("pdf_pages", []):
        if match_span(part["text"], quote) is not None:
            return part["number"]
    return None
