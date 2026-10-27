"""Exercise CLI output recovery with a genuinely blocked model worker."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from time import monotonic, sleep


class StalledModel(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        sleep(4)
        try:
            self.send_response(500)
            self.end_headers()
        except OSError:
            pass

    def log_message(self, *args):
        pass


server = ThreadingHTTPServer(("127.0.0.1", 0), StalledModel)
thread = Thread(target=server.serve_forever, daemon=True)
thread.start()
try:
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
        env = {**os.environ, "MODEL_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1"}
        args = [sys.executable, str(Path(__file__).with_name("app.py")),
                "--url", "https://example.org/", "--query", "Admission?",
                "--query-id", "stall", "--output-dir", folder,
                "--deadline-seconds", "1.5"]
        started = monotonic()
        run = subprocess.run(args, env=env, capture_output=True, text=True, timeout=12)
        assert run.returncode == 0, run.stderr
        assert monotonic() - started < 8
        result = json.loads(run.stdout)
        assert result == {"status": "abstained", "answer": "", "citations": []}
        assert json.loads((Path(folder)/"stall_output.json").read_text()) == result
        assert "deadline" in run.stderr, run.stderr

        args[args.index("stall")] = "interrupted"
        args[-1] = "10"
        worker = subprocess.Popen(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        target = Path(folder)/"interrupted_output.json"
        limit = monotonic() + 8
        while not target.exists() and monotonic() < limit:
            sleep(0.02)
        assert json.loads(target.read_text()) == result
        worker.kill()
        worker.wait(timeout=5)
        assert json.loads(target.read_text()) == result

        malformed = subprocess.run(args[:args.index("--url")]+["--query-id", "../escape", "--output-dir", folder],
                                   env=env, capture_output=True, text=True, timeout=3)
        assert json.loads(malformed.stdout) == result
        assert json.loads((Path(folder)/"query_output.json").read_text()) == result
finally:
    server.shutdown()
    server.server_close()
print("PASS: hung model, interrupted parent, and malformed input preserve valid output")
