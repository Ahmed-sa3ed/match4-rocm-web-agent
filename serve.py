"""Keep vLLM resident for repeated CLI invocations; no public port is exposed."""
import os
import signal
import subprocess
import sys
import time
import httpx
from decision import DEFAULT_MODEL

def main():
    import torch
    if not torch.version.hip or not torch.cuda.is_available():
        raise SystemExit("An AMD GPU with ROCm PyTorch is required.")
    print(f"ROCm PyTorch {torch.__version__}; GPU: {torch.cuda.get_device_name(0)}", flush=True)
    model = os.getenv("MODEL_ID", DEFAULT_MODEL)
    command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
               "--model", model, "--served-model-name", DEFAULT_MODEL,
               "--host", "127.0.0.1", "--port", "8000", "--dtype", "bfloat16",
               "--max-model-len", "32768", "--max-num-seqs", "2",
               "--gpu-memory-utilization", "0.70", "--enforce-eager"]
    child = subprocess.Popen(command)
    def stop(*_):
        if child.poll() is None: child.terminate()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        deadline = time.monotonic() + 570
        with httpx.Client(timeout=2, trust_env=False) as client:
            while time.monotonic() < deadline:
                if child.poll() is not None:
                    raise RuntimeError(f"vLLM exited during startup: {child.returncode}")
                try:
                    response = client.get("http://127.0.0.1:8000/health")
                    if response.status_code == 200: break
                except httpx.HTTPError:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError("vLLM was not ready within the startup budget.")
        print("Local vLLM is ready. Run app.py or local_agent.py in another terminal.", flush=True)
        return child.wait()
    finally:
        stop()
        try: child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()

if __name__ == "__main__":
    raise SystemExit(main())
