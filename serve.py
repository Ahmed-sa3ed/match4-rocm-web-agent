"""Resident local Qwen model from checkpoint 5, served on loopback."""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from decision import DEFAULT_MODEL
from pathlib import Path


class ModelBackend:
    def __init__(self, model_id=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if torch.__version__ != "2.13.0+rocm10.0.0" or not torch.cuda.is_available():
            raise RuntimeError("Checkpoint requires PyTorch 2.13.0+rocm10.0.0 and an AMD GPU.")
        model_id = model_id or os.getenv("MODEL_ID", DEFAULT_MODEL)
        self.torch = torch
        cache_dir = Path(os.getenv("HF_HOME", Path(__file__).parent / "hf-cache"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        options = {"cache_dir": str(cache_dir), "trust_remote_code": False}
        revision = os.getenv("MODEL_REVISION")
        if revision:
            options["revision"] = revision
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **options)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, attn_implementation="sdpa",
            **options,
        ).to("cuda").eval()
        self.lock = threading.Lock()
        print(f"Transformers model ready on {torch.cuda.get_device_name(0)}", flush=True)

    def complete(self, messages, max_tokens):
        if not isinstance(messages, list) or not messages or not all(
            isinstance(item, dict) and item.get("role") in ("system", "user", "assistant")
            and isinstance(item.get("content"), str) for item in messages
        ):
            raise ValueError("messages must contain system/user/assistant text turns")
        if type(max_tokens) is not int or not 1 <= max_tokens <= 512:
            raise ValueError("max_tokens must be an integer between 1 and 512")
        with self.lock, self.torch.inference_mode():
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            if inputs["input_ids"].shape[-1] + max_tokens > 32768:
                raise ValueError("Request exceeds the 32768 token context budget")
            output = self.model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )[0][inputs["input_ids"].shape[-1]:]
            eos = self.tokenizer.eos_token_id
            stopped = bool(len(output) and eos is not None and output[-1].item() == eos)
            content = self.tokenizer.decode(output, skip_special_tokens=True).strip()
            finish = "stop" if stopped or len(output) < max_tokens else "length"
            return content, finish, inputs["input_ids"].shape[-1], len(output)


def handler_for(backend):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, value):
            body = json.dumps(value).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self.respond(200, {"status": "ok", "backend": "match4-checkpoint5-transformers",
                                   "model": DEFAULT_MODEL, "pid": os.getpid()})
            elif self.path == "/v1/models":
                self.respond(200, {"data": [{"id": DEFAULT_MODEL, "object": "model"}]})
            else:
                self.respond(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.respond(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 2_000_000:
                    raise ValueError("Invalid request size")
                request = json.loads(self.rfile.read(length))
                if request.get("model") != DEFAULT_MODEL or request.get("stream") is True:
                    raise ValueError("Unsupported model or streaming request")
                content, finish, prompt_tokens, completion_tokens = backend.complete(
                    request.get("messages"), request.get("max_tokens", 128)
                )
                self.respond(200, {"choices": [{"finish_reason": finish,
                             "message": {"role": "assistant", "content": content}}],
                             "usage": {"prompt_tokens": prompt_tokens,
                                       "completion_tokens": completion_tokens}})
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self.respond(400, {"error": str(exc)})
            except Exception as exc:
                print(f"Model generation failed: {exc}", flush=True)
                self.respond(500, {"error": "Model generation failed"})

        def log_message(self, *_):
            pass

    return Handler


def main():
    backend = ModelBackend()
    server = ThreadingHTTPServer(("127.0.0.1", 8000), handler_for(backend))
    print("Local Qwen server is ready on http://127.0.0.1:8000", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
