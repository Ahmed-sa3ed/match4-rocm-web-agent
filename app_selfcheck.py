"""Check the file-output boundary without network or model access."""
import io
import json
import sys
import tempfile
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import patch
import app

result = {"status": "answered", "answer": "12 EUR", "citations": [
    {"field": "price", "value": "12 EUR", "source_url": "https://example.org/fees",
     "quote": "Admission is 12 EUR.", "support_mode": "exact"}]}
class ImmediatePipe:
    def __init__(self):
        self.value = None
    def send(self, value):
        self.value = value
    def close(self):
        pass
    def poll(self, timeout=None):
        return self.value is not None
    def recv(self):
        return self.value

class ImmediateProcess:
    exitcode = 0
    def __init__(self, target, args):
        self.target, self.args = target, args
    def start(self):
        self.target(*self.args)
    def join(self, timeout):
        pass
    def is_alive(self):
        return False

class ImmediateContext:
    Process = ImmediateProcess
    @staticmethod
    def Pipe(duplex=False):
        channel = ImmediatePipe()
        return channel, channel

with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
    args = ['app.py', '--url', 'https://example.org/', '--query', 'Admission?',
            '--query-id', 'demo', '--output-dir', folder]
    output = io.StringIO()
    def complete(*unused, **kwargs):
        assert json.loads((Path(folder)/'demo_output.json').read_text()) == app.EMPTY
        return result
    with patch.object(app.multiprocessing, 'get_context', return_value=ImmediateContext()), patch.object(sys, 'argv', args), patch.object(app, 'run_agent', side_effect=complete), redirect_stdout(output):
        app.main()
    assert json.loads(output.getvalue()) == result
    assert json.loads((Path(folder)/'demo_output.json').read_text()) == result
    assert not list(Path(folder).glob('.demo_output.json*'))
    with patch.object(sys, 'argv', args + ['--query-id', '../escape']), redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()) as rejected:
        app.main()
    assert json.loads(rejected.getvalue()) == app.EMPTY
    assert json.loads((Path(folder)/'query_output.json').read_text()) == app.EMPTY
    with patch.object(app.multiprocessing, 'get_context', return_value=ImmediateContext()), patch.object(sys, 'argv', args), patch.object(app, 'run_agent', side_effect=RuntimeError('model stopped')), redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()) as failed:
        app.main()
    assert json.loads(failed.getvalue()) == app.EMPTY
    assert json.loads((Path(folder)/'demo_output.json').read_text()) == app.EMPTY
print('PASS: CLI publishes an early placeholder, atomically replaces it, and recovers from errors')
