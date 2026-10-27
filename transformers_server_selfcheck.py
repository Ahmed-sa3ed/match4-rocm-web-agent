"""Check the optional Transformers server wire contract without model downloads."""
import os
from http.server import ThreadingHTTPServer
from threading import Thread
from unittest.mock import patch

import httpx

from decision import DEFAULT_MODEL, VLLMModel, available_models
from serve_transformers import handler_for


class Backend:
    def complete(self, messages, max_tokens):
        assert messages == [{"role": "user", "content": "hello"}]
        assert max_tokens == 2048
        return '{"answer":"12 EUR"}', "stop"


server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(Backend()))
thread = Thread(target=server.serve_forever, daemon=True)
thread.start()
try:
    url = f"http://127.0.0.1:{server.server_port}"
    with patch.dict(os.environ, {"VLLM_BASE_URL": url + "/v1"}):
        assert available_models()[0]["slug"] == DEFAULT_MODEL
        answer = VLLMModel().generate([{"role": "user", "content": "hello"}])
        assert answer == '{"answer":"12 EUR"}'
        with httpx.Client(trust_env=False) as client:
            assert client.get(url + "/health").status_code == 200
            assert client.post(url + "/v1/chat/completions", json={
                "model": "unknown", "messages": [], "stream": False
            }).status_code == 400
    print("PASS: local Transformers HTTP API works with the agent client")
finally:
    server.shutdown()
    server.server_close()
    thread.join()
