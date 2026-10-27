from unittest.mock import patch
import httpx
from decision import LocalModel

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
