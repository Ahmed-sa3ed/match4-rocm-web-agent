from unittest.mock import patch
from contextlib import redirect_stdout
from io import StringIO
import os
import httpx
from decision import LocalModel
real_client = httpx.Client

def test_response(choice, valid):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"choices": [choice]}))
    real_client = httpx.Client
    with patch('decision.httpx.Client', side_effect=lambda **kw: real_client(transport=transport, **kw)):
        try:
            result = LocalModel().generate([{"role": "user", "content": "test"}])
            assert valid and result == '{"answer":"12 EUR"}'
        except ValueError:
            assert not valid

test_response({"finish_reason":"stop", "message":{"content":'{"answer":"12 EUR"}'}}, True)
test_response({"finish_reason":"length", "message":{"content":'{"answer":'}}, False)
test_response({"finish_reason":"stop", "message":{"content":None}}, False)
print('PASS: local inference accepts complete output and rejects truncated or empty output')

# A complete JSON object can still reach the cap without an EOS token.
transport = httpx.MockTransport(lambda request: httpx.Response(200, json={
    "choices": [{"finish_reason": "length", "message": {"content": '{"answer":"12 EUR"}'}}],
    "usage": {"completion_tokens": 128},
}))
recorded = StringIO()
with patch('decision.httpx.Client', side_effect=lambda **kw: real_client(transport=transport, **kw)), \
     patch.dict(os.environ, {"MATCH4_MODEL_METRICS": "1"}), redirect_stdout(recorded):
    try:
        LocalModel().generate([{"role": "user", "content": "test"}])
    except ValueError:
        pass
    else:
        raise AssertionError("A length finish must be rejected")
assert "output_tokens: 128" in recorded.getvalue()
assert "finish: length | json_object: True" in recorded.getvalue()
print("PASS: metrics distinguish length finish from incomplete JSON")
