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
with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
    args = ['app.py', '--url', 'https://example.org/', '--query', 'Admission?',
            '--query-id', 'demo', '--output-dir', folder]
    output = io.StringIO()
    with patch.object(sys, 'argv', args), patch.object(app, 'run_agent', return_value=result), redirect_stdout(output):
        app.main()
    assert json.loads(output.getvalue()) == result
    assert json.loads((Path(folder)/'demo_output.json').read_text()) == result
    assert not list(Path(folder).glob('*.tmp'))
    with patch.object(sys, 'argv', args + ['--query-id', '../escape']), redirect_stderr(io.StringIO()):
        try:
            app.main()
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError('Unsafe output filename accepted')
print('PASS: CLI writes complete JSON and prevents query-ID path traversal')
