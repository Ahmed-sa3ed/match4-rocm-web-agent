import json
import math
import re
from time import sleep
from collections import Counter
from urllib.parse import urldefrag, urlsplit

import httpx

from evidence import candidate_fields, select_facts
from forms import build_search_url
from navigation import choose_link, choose_route
from page_chunks import text_chunks
from planning import plan_task
from retrieval import UnsupportedContentTypeError, fetch_page
from retrieval_browser import BrowserRenderError, render_page
from search_navigation import choose_search


def same_origin(url, start_url):
    target = urlsplit(url)
    origin = urlsplit(start_url)
    return target.scheme == origin.scheme and target.netloc == origin.netloc


def canonical_url(url):
    """A #fragment is a location within a page, not another page."""
    return urldefrag(url).url


def rank_links(links, question, missing_fields):
    """Show distinctive question matches first; keep every other link."""
    def words(value):
        return set(re.findall(r"[a-z0-9]+", value.lower()))

    query = words(question + " " + " ".join(missing_fields))
    labels = [words(link.get("text", "")) for link in links]
    frequency = Counter(word for label in labels for word in label)
    count = len(links)

    def score(index):
        return sum(
            math.log((count + 1) / (frequency[word] + 1)) + 1
            for word in query & labels[index]
            if len(word) > 2
        )

    order = sorted(range(count), key=lambda index: -score(index))
    return [links[index] for index in order]


def unvisited_links(pages, visited, start_url):
    discovered = {}
    for page in pages.values():
        for link in page["links"]:
            target = canonical_url(link["url"])
            if same_origin(target, start_url) and target not in visited:
                discovered.setdefault(target, {**link, "url": target})
    return list(discovered.values())


def choose_next_url(question, missing_fields, page, pages, visited,
                    used_forms, start_url, model, tokenizer):
    links = rank_links(unvisited_links(pages, visited, start_url),
                       question, missing_fields)
    forms = [
        form for form in page["forms"]
        if same_origin(form["action_url"], start_url)
        and json.dumps(form, sort_keys=True) not in used_forms
    ][:5]
    offset = 0
    # Show every discovered link eventually, in bounded batches for the model.
    while offset < len(links) or forms:
        batch = links[offset:offset + 40]
        if batch and forms:
            choice = choose_route({
                "question": question, "missing_fields": missing_fields,
                "links": batch, "forms": forms,
                "page_text": page["text"][:1000],
            }, model, tokenizer)
            if choice is None:
                offset += len(batch)
                continue
            if choice["action"] == "link":
                return batch[choice["index"]]["url"]
            form_index, query = choice["index"], choice["query"]
        elif batch:
            index = choose_link({
                "question": question, "missing_fields": missing_fields,
                "links": batch,
            }, model, tokenizer)
            if index is not None:
                return batch[index]["url"]
            offset += len(batch)
            continue
        else:
            choice = choose_search({
                "question": question, "missing_fields": missing_fields,
                "page_text": page["text"][:1000], "forms": forms,
            }, model, tokenizer)
            if choice is None:
                break
            form_index, query = choice["form_index"], choice["query"]

        form = forms.pop(form_index)
        used_forms.add(json.dumps(form, sort_keys=True))
        target = canonical_url(build_search_url(form, query))
        if same_origin(target, start_url) and target not in visited:
            return target
    return None


def fetch_with_retry(url, fetch_page_fn, max_attempts=2):
    """Retry temporary HTTP/network failures; leave other responses alone."""
    for attempt in range(1, max_attempts + 1):
        try:
            return fetch_page_fn(url)
        except httpx.HTTPError as exc:
            status = (exc.response.status_code
                      if isinstance(exc, httpx.HTTPStatusError) else None)
            temporary = (isinstance(exc, httpx.RequestError) or
                         status in (408, 429) or
                         status is not None and 500 <= status <= 599)
            reason = f"HTTP {status}" if status is not None else f"{type(exc).__name__}: {exc}"
            print(f"Fetch attempt {attempt}/{max_attempts} | {url} | {reason}")
            if attempt == max_attempts or not temporary:
                raise
            sleep(0.25 * attempt)


def answered_result(required_fields, selected, conflicts=None):
    """Expose only final verified evidence, in the requested answer order."""
    result = {
        "status": "answered",
        "answer": " | ".join(selected[field]["value"] for field in required_fields),
        "citations": [
            {"field": field, "value": selected[field]["value"],
             "source_url": selected[field]["source_url"],
             "quote": selected[field]["quote"],
             "support_mode": selected[field].get("support_mode", "exact"),
             **({"page_number": selected[field]["page_number"]}
                if "page_number" in selected[field] else {})}
            for field in required_fields
        ],
    }
    if conflicts:
        result["conflicts"] = conflicts
    return result


def inspect_page(question, required_fields, page, model, tokenizer,
                 candidate_reviews, found_fields, page_number):
    chunks = list(text_chunks(page["text"]))
    print(f"Page {page_number} | {page['url']} | "
          f"characters: {len(page['text'])} | chunks: {len(chunks)} | "
          f"links: {len(page['links'])}")
    page_found = set()
    for number, chunk in enumerate(chunks, 1):
        missing = [field for field in required_fields if field not in page_found]
        if not missing:
            break
        chunk_page = {**page, "text": chunk}
        request = {"question": question, "required_fields": missing,
                   "page": {"url": page["url"], "text": chunk}}
        before = len(page_found)
        page_found.update(candidate_fields(
            request, chunk_page, model, tokenizer, reviews=candidate_reviews))
        found_fields.update(page_found)
        print(f"Chunk {number}/{len(chunks)} | {page['url']} | "
              f"new grounded fields: {len(page_found) - before} | "
              f"still missing: {[field for field in required_fields if field not in found_fields]}")
    return page_found


def run_agent(question, start_url, model, tokenizer, *,
              fetch_page_fn=None, render_page_fn=render_page,
              max_pages=4, early_answer_confidence=0.95):
    """Browse up to max_pages and review strong complete evidence early."""
    if type(max_pages) is not int or not 1 <= max_pages <= 20:
        raise ValueError("max_pages must be an integer between 1 and 20.")
    if (type(early_answer_confidence) not in (int, float)
            or not math.isfinite(early_answer_confidence)
            or not 0 <= early_answer_confidence <= 1):
        raise ValueError("early_answer_confidence must be between 0 and 1.")
    fetch_page_fn = fetch_page_fn or fetch_page
    required_fields = plan_task(question, model, tokenizer)
    print("Required fields:", required_fields)

    found_fields = set()
    candidate_reviews = []
    pages = {}
    visited = set()
    used_forms = set()
    fetch_failures = []
    current_url = canonical_url(start_url)
    last_page = None
    successful_pages = 0
    navigation_fields = required_fields

    # A failed URL may need a replacement; retries do not consume the page budget.
    for attempt in range(2 * max_pages):
        if successful_pages >= max_pages:
            break
        current_url = canonical_url(current_url)
        visited.add(current_url)
        try:
            page = fetch_with_retry(current_url, fetch_page_fn)
        except (httpx.HTTPError, UnsupportedContentTypeError) as exc:
            reason = f"{type(exc).__name__}: {exc}"
            fetch_failures.append({"url": current_url, "reason": reason})
            print(f"Fetch failed | {current_url} | {reason}")
            if last_page is None:
                break
            current_url = choose_next_url(
                question, navigation_fields, last_page, pages, visited,
                used_forms, start_url, model, tokenizer,
            )
            if current_url is None:
                break
            continue

        page["url"] = canonical_url(page["url"])
        if not same_origin(page["url"], start_url):
            reason = f"Redirected outside the starting origin to {page['url']}"
            fetch_failures.append({"url": current_url, "reason": reason})
            print(f"Fetch failed | {current_url} | {reason}")
            if last_page is None:
                break
            current_url = choose_next_url(
                question, navigation_fields, last_page, pages, visited,
                used_forms, start_url, model, tokenizer,
            )
            if current_url is None:
                break
            continue

        visited.add(page["url"])
        pages[page["url"]] = page
        last_page = page
        successful_pages += 1

        page_found = inspect_page(
            question, required_fields, page, model, tokenizer,
            candidate_reviews, found_fields, successful_pages,
        )
        if (render_page_fn is not None and page.get("render_hint")
                and any(field not in found_fields for field in required_fields)):
            print(f"Browser fallback | sparse script page | {page['url']}")
            try:
                rendered = render_page_fn(page["url"], len(page["text"]))
            except (BrowserRenderError, UnsupportedContentTypeError) as exc:
                print(f"Browser fallback failed | {page['url']} | {exc}")
            else:
                if not same_origin(rendered["url"], start_url):
                    print(f"Browser fallback rejected off-origin redirect | {rendered['url']}")
                elif (rendered["text"] == page["text"]
                      and rendered["links"] == page["links"]
                      and rendered["forms"] == page["forms"]):
                    print(f"Browser fallback found no new content | {page['url']}")
                else:
                    old_url = page["url"]
                    if page["text"] and page["text"] not in rendered["text"]:
                        rendered["text"] = page["text"] + "\n\n" + rendered["text"]
                    known_links = {link["url"] for link in rendered["links"]}
                    rendered["links"].extend(
                        link for link in page["links"]
                        if link["url"] not in known_links
                    )
                    known_forms = {json.dumps(form, sort_keys=True)
                                   for form in rendered["forms"]}
                    rendered["forms"].extend(
                        form for form in page["forms"]
                        if json.dumps(form, sort_keys=True) not in known_forms
                    )
                    page = rendered
                    page["url"] = canonical_url(page["url"])
                    pages.pop(old_url)
                    pages[page["url"]] = page
                    visited.add(page["url"])
                    last_page = page
                    print(f"Browser fallback used | {page['url']} | "
                          f"characters: {len(page['text'])}")
                    page_found.update(inspect_page(
                        question, required_fields, page, model, tokenizer,
                        candidate_reviews, found_fields, successful_pages,
                    ))

        missing_fields = [field for field in required_fields if field not in found_fields]
        navigation_fields = missing_fields or required_fields
        print(f"Step {successful_pages} | {page['url']} | "
              f"candidate fields: {sorted(found_fields)} | missing: {missing_fields}")

        if successful_pages >= max_pages:
            break

        # A confidence score prompts review; it never approves an answer itself.
        strong_fields = {
            review["field"] for review in candidate_reviews
            if review.get("grounded") and review.get("model_estimate_supplied")
            and review["confidence"] >= early_answer_confidence
        }
        if (all(field in strong_fields for field in required_fields)
                and (successful_pages >= max_pages
                     or not unvisited_links(pages, visited, start_url))):
            print(f"Early review | all {len(required_fields)} fields have grounded "
                  f"candidates at or above {early_answer_confidence:.2f}")
            early_request = {
                "question": question, "required_fields": required_fields,
                "pages": pages, "candidate_reviews": candidate_reviews,
                "diagnostics": [],
            }
            early_selected = select_facts(early_request, model, tokenizer)
            if early_selected is not None:
                print("Early review approved | stopping after", successful_pages,
                      "page(s)")
                return answered_result(required_fields, early_selected,
                                       early_request.get("conflicts"))
            print("Early review did not verify every field | continuing navigation")

        current_url = choose_next_url(
            question, navigation_fields, page, pages, visited,
            used_forms, start_url, model, tokenizer,
        )
        if current_url is None:
            break

    if not pages:
        reason = ("No supported page could be fetched. " + "; ".join(
            f"{item['url']} ({item['reason']})" for item in fetch_failures))
        print("Abstaining:", reason)
        return {"status": "abstained", "answer": "",
                "failure_reason": reason, "fetch_failures": fetch_failures}

    diagnostics = []
    final_request = {
        "question": question, "required_fields": required_fields,
        "pages": pages, "candidate_reviews": candidate_reviews,
        "diagnostics": diagnostics,
    }
    selected = select_facts(final_request, model, tokenizer)
    if selected is None:
        result = {"status": "abstained", "answer": ""}
        if final_request.get("conflicts"):
            result["conflicts"] = final_request["conflicts"]
        rejected_grounded = [issue for issue in diagnostics
                             if issue["grounded_candidate_count"]]
        if rejected_grounded:
            result["selection_failures"] = diagnostics
            result["failure_reason"] = "Final review rejected grounded evidence: " + "; ".join(
                f"{issue['field']}: {issue['reason']}" for issue in rejected_grounded)
        if fetch_failures:
            prefix = result.get("failure_reason", "Final review did not verify every field.")
            result["failure_reason"] = prefix + " Fetch failures encountered: " + "; ".join(
                f"{item['url']} ({item['reason']})" for item in fetch_failures)
            result["fetch_failures"] = fetch_failures
        else:
            print("Abstaining: final evidence review did not verify every field. "
                  "See the field-level validation logs above.")
        return result
    return answered_result(required_fields, selected,
                           final_request.get("conflicts"))
