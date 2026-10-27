# Notebook model test with Transformers

vLLM's current ROCm requirements cannot resolve on the tested Python 3.14 environment: `amd-quark==0.12.post1` excludes Python 3.14, and the package index has no `mooncake-transfer-engine-rocm` wheel. For notebook evaluation, `serve_transformers.py` runs the same Qwen model behind the local HTTP API expected by the agent. This does not validate vLLM or the final Docker image.

In the existing clean checkout, install the version used by the original checkpoint and check the API without model weights:

```python
%cd /persistent/match4-clean-20261027-151415
!python3 -m pip install -c constraints-torch.txt transformers==5.20.2
!python3 transformers_server_selfcheck.py
```

Start the model in a Python cell. The first run downloads weights into persistent storage:

```python
import os
import sys
import subprocess
from pathlib import Path

project = Path("/persistent/match4-clean-20261027-151415")
server_env = os.environ.copy()
server_env["HF_HOME"] = str(project / "hf-cache")
server_log = open(project / "transformers-server.log", "w")
server = subprocess.Popen(
    [sys.executable, "serve_transformers.py"], cwd=project,
    env=server_env, stdout=server_log, stderr=subprocess.STDOUT,
)
print("Model server PID:", server.pid)
```

Check readiness in another cell. The log contains startup errors:

```python
import urllib.request
print("Server exit code:", server.poll())
try:
    print("Health:", urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=3).status)
except Exception as error:
    print("Not ready:", error)
print((project / "transformers-server.log").read_text()[-4000:])
```

When health returns `200`, run the agent in order:

```python
!python3 local_agent.py models
!python3 local_agent.py smoke --model Qwen/Qwen3-4B-Instruct-2507
!python3 run_agent_suite.py --only local --workers 1
!python3 run_agent_suite.py --workers 1
```

The server serializes GPU generations to limit memory peaks. The agent client allows 60 seconds per model call, so slow generations may time out. Public cases require outbound access; a completed suite may contain failed cases. Inspect `test_reports/` for traces. To stop the server, run `server.terminate()` in a Python cell.
