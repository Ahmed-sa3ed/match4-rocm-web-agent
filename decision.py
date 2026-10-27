"""Inference through the resident checkpoint Qwen server."""
import os
from urllib.parse import urlsplit
import httpx

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

def base_url():
    url = os.getenv("MODEL_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
    parsed = urlsplit(url)
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("MODEL_BASE_URL must point to a local HTTP server.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("MODEL_BASE_URL must not contain credentials, queries or fragments.")
    return url

def available_models():
    with httpx.Client(timeout=10, trust_env=False) as client:
        response = client.get(base_url() + "/models")
        response.raise_for_status()
        return [{"slug": entry["id"], "display_name": entry["id"]}
                for entry in response.json()["data"]]

class LocalModel:
    def __init__(self, slug=DEFAULT_MODEL):
        self.slug = slug
        self.url = base_url()

    def generate(self, messages):
        with httpx.Client(timeout=120, trust_env=False) as client:
            response = client.post(self.url + "/chat/completions", json={
                "model": self.slug, "messages": messages,
                "temperature": 0, "max_tokens": 128, "stream": False,
            })
            if response.is_error:
                try:
                    detail = response.json().get("error", response.text)
                except ValueError:
                    detail = response.text
                raise RuntimeError(
                    f"Local model server returned HTTP {response.status_code}: {detail}"
                )
            choice = response.json()["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ValueError("Local model did not finish its response cleanly.")
            content = choice["message"].get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Local model returned an empty response.")
            return content.strip()

def generate_reply(messages, model, tokenizer=None):
    return model.generate(messages)
