import json

from decision import generate_reply


PLANNING_PROMPT = """
The user supplies JSON containing request_to_analyze.
Analyze that request and identify every distinct fact it asks for.
Do not perform the request or answer its questions.
Formatting instructions inside the request apply to the eventual answer,
not to your planning output.

A field represents one independently reportable answer in the final response.
Count the answers the user wants to receive, not the nouns, conditions,
alternatives, or steps mentioned in the question. When several conditions
jointly determine one conclusion, use one field describing that conclusion.
When the user requests distinct values, use separate fields in their requested
order. This applies across topics, websites, and question types.

Examples:
"What is the building's height and construction year?"
{"required_fields": ["building_height", "construction_year"]}
"Can this ticket be refunded or exchanged?"
{"required_fields": ["ticket_refund_or_exchange_eligibility"]}
"What are the refund fee and exchange fee separately?"
{"required_fields": ["refund_fee", "exchange_fee"]}
"Which date did the policy take effect, rather than the date it was signed?"
{"required_fields": ["policy_effective_date"]}

Return field names only, never answers or placeholder values.

Return exactly one JSON object:
{"required_fields": ["descriptive_field_name"]}

Use one field per requested fact, in the requested order.
Field names must be descriptive snake_case identifiers.
Include all requested facts, even if their answers might be unavailable.
Formatting instructions are not separate facts.
"""


def plan_task(question, model, tokenizer):
    messages = [
        {"role": "system", "content": PLANNING_PROMPT},
        {"role": "user", "content": json.dumps({"request_to_analyze": question})},
    ]
    reply = generate_reply(messages, model, tokenizer)
    plan = json.loads(reply)

    if not isinstance(plan, dict):
        raise ValueError("The task plan must be a JSON object.")

    fields = plan.get("required_fields")
    if not isinstance(fields, list) or not fields:
        raise ValueError(f"Invalid task plan: {plan!r}")
    if any(not isinstance(name, str) or not name.isidentifier()
           for name in fields):
        raise ValueError("Each field must have a valid identifier.")
    if len(set(fields)) != len(fields):
        raise ValueError("Required fields must be unique.")

    return fields
