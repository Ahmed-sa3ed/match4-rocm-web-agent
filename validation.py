from text_match import match_span


def validate_facts(required_fields, facts, pages, *, allow_semantic=False):
    errors = []

    for field in required_fields:
        fact = facts.get(field)
        if not isinstance(fact, dict):
            errors.append(f"{field}: missing")
            continue

        value = fact.get("value")
        quote = fact.get("quote")
        source_url = fact.get("source_url")

        if not isinstance(value, str) or not value.strip():
            errors.append(f"{field}: missing value")
            continue
        if not isinstance(quote, str) or not quote.strip():
            errors.append(f"{field}: missing evidence quote")
            continue
        if not isinstance(source_url, str) or source_url not in pages:
            errors.append(f"{field}: source was not fetched")
            continue
        actual_quote = match_span(pages[source_url]["text"], quote)
        if actual_quote is None:
            errors.append(f"{field}: quote absent from source")
            continue
        if match_span(actual_quote, value) is None:
            if (allow_semantic and fact.get("support_mode") == "semantic"
                    and isinstance(fact.get("review_reason"), str)
                    and fact["review_reason"].strip()):
                continue
            errors.append(f"{field}: value absent from quote")

    return errors
