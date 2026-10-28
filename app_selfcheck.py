"""Check the file-output boundary without network or model access."""
import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import patch
import app
import local_agent

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
        print('INTERNAL TRACE')
        return result
    quiet_errors = io.StringIO()
    with patch.object(app.multiprocessing, 'get_context', return_value=ImmediateContext()), patch.object(sys, 'argv', args), patch.object(app, 'run_agent', side_effect=complete), redirect_stdout(output), redirect_stderr(quiet_errors):
        app.main()
    assert json.loads(output.getvalue()) == result
    assert 'INTERNAL TRACE' not in quiet_errors.getvalue()
    assert json.loads((Path(folder)/'demo_output.json').read_text()) == result
    assert not list(Path(folder).glob('.demo_output.json*'))
    audit_output, audit_errors = io.StringIO(), io.StringIO()
    with patch.object(app.multiprocessing, 'get_context', return_value=ImmediateContext()), patch.object(sys, 'argv', args + ['--audit']), patch.object(app, 'run_agent', side_effect=complete), patch.dict(os.environ, {}, clear=False), redirect_stdout(audit_output), redirect_stderr(audit_errors):
        app.main()
    assert json.loads(audit_output.getvalue()) == result
    assert 'INTERNAL TRACE' in audit_errors.getvalue()
    with patch.object(sys, 'argv', args + ['--query-id', '../escape']), redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()) as rejected:
        app.main()
    assert json.loads(rejected.getvalue()) == app.EMPTY
    assert json.loads((Path(folder)/'query_output.json').read_text()) == app.EMPTY
    with patch.object(app.multiprocessing, 'get_context', return_value=ImmediateContext()), patch.object(sys, 'argv', args), patch.object(app, 'run_agent', side_effect=RuntimeError('model stopped')), redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()) as failed:
        app.main()
    assert json.loads(failed.getvalue()) == app.EMPTY
    assert json.loads((Path(folder)/'demo_output.json').read_text()) == app.EMPTY

    ask_args = ['local_agent.py', 'ask', 'https://example.org/', 'Admission?',
                '--model', 'example-model']
    def local_complete(*unused, **kwargs):
        print('INTERNAL TRACE')
        return result
    with patch.object(sys, 'argv', ask_args), patch.object(local_agent, 'available_models', return_value=[{'slug': 'example-model'}]), patch.object(local_agent, 'LocalModel', return_value=object()), patch('agent.run_agent', side_effect=local_complete), redirect_stdout(io.StringIO()) as answer, redirect_stderr(io.StringIO()) as errors:
        local_agent.main()
    assert answer.getvalue().strip() == '12 EUR'
    assert errors.getvalue() == ''
    with patch.object(sys, 'argv', ask_args + ['--audit']), patch.object(local_agent, 'available_models', return_value=[{'slug': 'example-model'}]), patch.object(local_agent, 'LocalModel', return_value=object()), patch('agent.run_agent', side_effect=local_complete), redirect_stdout(io.StringIO()) as answer, redirect_stderr(io.StringIO()) as errors:
        local_agent.main()
    assert answer.getvalue().strip() == '12 EUR'
    assert 'INTERNAL TRACE' in errors.getvalue() and 'Source:' in errors.getvalue()
print('PASS: CLI publishes an early placeholder, atomically replaces it, and recovers from errors')
