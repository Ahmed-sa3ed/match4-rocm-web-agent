# Match 4 Grounded Web Retrieval on AMD ROCm

A web retrieval agent that navigates links and GET forms, renders JavaScript with Scrapling, reads tables and PDFs, resolves conflicting evidence, and answers with verified citations. The latest agent workflow uses the **local Qwen model from checkpoint 5** through a resident Transformers server. No ChatGPT login or hosted LLM API is needed.

The model is `Qwen/Qwen3-4B-Instruct-2507` with Transformers 5.20.2, BF16 weights, SDPA attention, deterministic generation and a 128-token output cap. These match the checkpoint's model configuration. The HTTP server keeps the model loaded between CLI queries; the agent's planning, extraction and review prompts are from the newer project.

> **Validation:** Offline application and local HTTP contract checks pass. A user-run AMD notebook check launched Chromium through Scrapling with `MATCH4_BROWSER_NO_ZYGOTE=1`. Full Qwen accuracy, image build, VRAM usage and the final Match 4 CLI contract still require live validation. The available competition briefs cover Matches 1–3.

## Features

- Navigate on the starting website, follow relevant links and submit discovered GET search forms.
- Extract structured text from HTML and tables; render sparse JavaScript pages with Scrapling Chromium.
- Read selectable-text PDFs and cite their page numbers.
- Validate source passages, review meaning, handle competing claims and abstain when evidence is insufficient.
- Run 33 local and public-site evaluation cases with traces and reports.

## AMD notebook: from clone to full test

Copy each cell into Jupyter separately. `!` runs a shell command and `%cd` changes the working directory. Use `/persistent` on `jupyter-hack-*` or `/workspace` on `rgapi-hackathon-*` sessions. For an existing checkout, skip the clone and `%cd` to its folder.

```python
%cd /persistent
!git clone https://github.com/Ahmed-sa3ed/match4-rocm-web-agent.git
%cd /persistent/match4-rocm-web-agent
```

Preserve the notebook's ROCm PyTorch build when installing dependencies:

```python
!python3 -c 'import torch; print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0)); assert torch.cuda.is_available()'
!python3 -c 'import torch; print("torch==" + torch.__version__)' > constraints-torch.txt
!python3 -m pip install -c constraints-torch.txt -r requirements.txt
```

Check application logic without loading the model:

```python
!python3 backend_selfcheck.py
!python3 transformers_server_selfcheck.py
!python3 app_selfcheck.py
!python3 feature_selfcheck.py
!python3 adaptive_selfcheck.py
!python3 conflict_selfcheck.py
!python3 pdf_selfcheck.py
!python3 run_agent_suite.py --dry-run
```

The tested notebook required a Chromium process setting. Set this before running browser or agent commands:

```python
import os
os.environ["MATCH4_BROWSER_NO_ZYGOTE"] = "1"
```

```python
!python3 browser_livecheck.py
```

If Chromium is missing, `!scrapling install` normally downloads browsers and Ubuntu libraries. The tested notebook could not reach Ubuntu APT mirrors, so a missing system library may require the platform's network support. This real browser check exercises a local JavaScript page and does not require the LLM.

Start Qwen once in the background. The first run downloads model weights to persistent storage. Use your actual project path if different:

```python
import os
import sys
import subprocess
from pathlib import Path

project = Path.cwd()
server_env = os.environ.copy()
server_env["HF_HOME"] = str(project / "hf-cache")
server_log = open(project / "model-server.log", "w")
server = subprocess.Popen(
    [sys.executable, "serve.py"], cwd=project,
    env=server_env, stdout=server_log, stderr=subprocess.STDOUT,
)
print("Model server PID:", server.pid)
```

Check readiness in a new cell; repeat while weights load:

```python
import urllib.request
print("Server exit code:", server.poll())
try:
    import json
    health = json.load(urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=3))
    print("Health:", health)
    assert health["backend"] == "match4-checkpoint5-transformers"
    assert health["pid"] == server.pid, "Port 8000 belongs to an older server process"
except Exception as error:
    print("Not ready:", error)
print((project / "model-server.log").read_text()[-4000:])
```

When health identifies `match4-checkpoint5-transformers` and its PID matches the new server, test the model, then local and public cases. If the PID differs, stop the older process and start `serve.py` again. If a request fails, the client now shows the server error detail; do not run the full suite until the smoke case succeeds:

```python
!python3 local_agent.py models
!python3 local_agent.py smoke --model Qwen/Qwen3-4B-Instruct-2507
!python3 run_agent_suite.py --only local --workers 1
!python3 run_agent_suite.py --workers 1
```

To measure the 128-token cap before changing it, set this in a Python cell before running selected cases. Each model call adds its stage, output token count, finish reason, JSON-object parse status and elapsed time to that case's report trace; no prompt or model response text is added by this measurement. Restart the model server after pulling the updated `serve.py` so it returns token usage.

```python
import os
os.environ["MATCH4_MODEL_METRICS"] = "1"
!python3 run_agent_suite.py --case w3c_invention --case archives_declaration_dates --case pdf_nist_si_mole --case conflict_w3c_wcag21_date --workers 1
```

Reports are written to `test_reports/`. A completed suite can contain failures; inspect its per-case traces. Public sites can change or be blocked by notebook networking. Stop the server when finished using `server.terminate()` in a Python cell.

## Ask a question

After the model server is healthy:

```python
!python3 app.py --url https://www.iana.org/help/example-domains --query "Can example.com be registered?" --query-id example-domains
!python3 local_agent.py ask https://www.iana.org/help/example-domains "Can example.com be registered?" --model Qwen/Qwen3-4B-Instruct-2507
```

The first command writes `output/example-domains_output.json`. The second prints the answer and source passages. Answers contain `status`, `answer`, and `citations` with field, value, URL, quote and support mode. PDFs add a page number. Unsupported answers abstain.

## Container packaging

The Dockerfile starts from the mandated `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0` image and keeps Qwen resident through `serve.py`. Build on an AMD machine with Docker and network access to Python, model and browser dependencies:

```bash
docker build -t match4-agent:local .
docker run -d --name match4-agent --device /dev/kfd --device /dev/dri --group-add video --ipc=host -v match4-models:/models match4-agent:local
docker logs -f match4-agent
```

After readiness, run a query inside the container:

```bash
docker exec match4-agent python3 /app/app.py --url https://www.iana.org/help/example-domains --query "Can example.com be registered?" --query-id example-domains
docker cp match4-agent:/app/output/example-domains_output.json .
```

The image has not been built on the competition platform. The exact Match 4 input/output contract is still unknown; the current `app.py` interface is provisional. The shared briefs require a public container image reference for submission, beyond this source repository. See [competition readiness](docs/COMPETITION.md).

## Scope

A starting URL is required. Navigation stays on its origin. Browser rendering is bounded to sparse script pages; PDF text must be selectable (no OCR). Page limits do not impose a complete query deadline. The 128-token model response limit follows the checkpoint but may truncate longer JSON decisions from the expanded agent; the client rejects incomplete responses. Full tests will quantify this risk before submission.

Credentials, downloaded weights, local environments and generated reports are excluded from Git. Dependencies and models retain their upstream licenses.
