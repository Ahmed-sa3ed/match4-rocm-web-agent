"""Check the optional Transformers server wire contract without model downloads."""
import os
from http.server import ThreadingHTTPServer
from threading import Thread
from unittest.mock import patch

import httpx

from decision import DEFAULT_MODEL, LocalModel, available_models
from serve import handler_for


class Backend:
    def complete(self, messages, max_tokens):
        assert messages == [{"role": "user", "content": "hello"}]
        assert max_tokens == 128
        return '{"answer":"12 EUR"}', "stop"


server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(Backend()))
thread = Thread(target=server.serve_forever, daemon=True)
thread.start()
try:
    url = f"http://127.0.0.1:{server.server_port}"
    with patch.dict(os.environ, {"MODEL_BASE_URL": url + "/v1"}):
        assert available_models()[0]["slug"] == DEFAULT_MODEL
        answer = LocalModel().generate([{"role": "user", "content": "hello"}])
        assert answer == '{"answer":"12 EUR"}'
        with httpx.Client(trust_env=False) as client:
            health = client.get(url + "/health").json()
            assert health["backend"] == "match4-checkpoint5-transformers"
            assert health["model"] == DEFAULT_MODEL
            assert client.post(url + "/v1/chat/completions", json={
                "model": "unknown", "messages": [], "stream": False
            }).status_code == 400
            try:
                LocalModel("unknown").generate([{"role": "user", "content": "hello"}])
            except RuntimeError as exc:
                assert "Unsupported model or streaming request" in str(exc)
            else:
                raise AssertionError("Expected server error to reach the client")
    print("PASS: local Transformers HTTP API works with the agent client")
finally:
    server.shutdown()
    server.server_close()
    thread.join()


# Verify the checkpoint generation settings without loading model weights.
from contextlib import nullcontext
from threading import Lock
from types import SimpleNamespace
from serve import ModelBackend

class Token(int):
    def item(self):
        return int(self)

class Tokens(list):
    @property
    def shape(self):
        return (1, len(self))

class Inputs(dict):
    def to(self, device):
        assert device == "cuda"
        return self

class Tokenizer:
    eos_token_id = 2
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        return "formatted chat"
    def __call__(self, prompt, **kwargs):
        assert prompt == "formatted chat" and kwargs == {"return_tensors": "pt"}
        return Inputs(input_ids=Tokens([Token(1)]))
    def decode(self, output, **kwargs):
        assert list(output) == [3, 2] and kwargs == {"skip_special_tokens": True}
        return '{"answer":"12 EUR"}'

class Generator:
    device = "cuda"
    def generate(self, **kwargs):
        assert kwargs["max_new_tokens"] == 128
        assert kwargs["do_sample"] is False
        assert kwargs["pad_token_id"] == 2
        return [Tokens([Token(1), Token(3), Token(2)])]

backend = ModelBackend.__new__(ModelBackend)
backend.torch = SimpleNamespace(inference_mode=nullcontext)
backend.tokenizer = Tokenizer()
backend.model = Generator()
backend.lock = Lock()
assert backend.complete([{"role": "user", "content": "hello"}], 128) == (
    '{"answer":"12 EUR"}', "stop"
)
print("PASS: checkpoint chat template and deterministic generation settings")
