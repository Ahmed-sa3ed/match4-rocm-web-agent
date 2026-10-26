from urllib.parse import urlencode, urljoin, urlsplit

from bs4 import BeautifulSoup


def parse_search_forms(html, page_url):
    """Find simple GET forms with one text/search input."""
    soup = BeautifulSoup(html, "html.parser")
    forms = []

    for element in soup.find_all("form"):
        method = element.get("method", "get").strip().lower()
        if method != "get":
            continue

        # These controls are outside our first version.
        if element.find(["select", "textarea"]):
            continue

        query_fields = []
        hidden_fields = []
        supported = True

        for control in element.find_all("input"):
            name = control.get("name")
            if not name or control.has_attr("disabled"):
                continue

            kind = control.get("type", "text").lower()

            if kind in {"text", "search"}:
                query_fields.append(name)
            elif kind == "hidden":
                hidden_fields.append(
                    (name, control.get("value", ""))
                )
            elif kind not in {"submit", "button", "reset"}:
                supported = False

        if not supported or len(query_fields) != 1:
            continue

        action_url = urljoin(
            page_url,
            element.get("action") or page_url,
        )
        if urlsplit(action_url).scheme not in {"http", "https"}:
            continue

        forms.append({
            "action_url": action_url,
            "query_field": query_fields[0],
            "hidden_fields": hidden_fields,
        })

    return forms


def build_search_url(form, query):
    """Build a GET URL from a parsed form and search text."""
    parameters = [
        *form["hidden_fields"],
        (form["query_field"], query),
    ]
    parts = urlsplit(form["action_url"])

    return parts._replace(
        query=urlencode(parameters),
        fragment="",
    ).geturl()
