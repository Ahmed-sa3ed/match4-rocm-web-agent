"""Exercise Scrapling's real Chromium renderer on a local JavaScript page."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from retrieval import parse_page
from retrieval_browser import render_page

HTML = b'''<!doctype html><html><body><p>Loading...</p><script>
setTimeout(() => { document.body.innerHTML = "<p>JavaScript rendering works: 4827</p>"; }, 300);
</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(HTML)))
        self.end_headers()
        self.wfile.write(HTML)

    def log_message(self, *_):
        pass


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/"
        static = parse_page(HTML.decode(), url)
        assert "4827" not in static["text"]
        rendered = render_page(url, len(static["text"]))
        assert "JavaScript rendering works: 4827" in rendered["text"]
        print("PASS: Scrapling launched Chromium and executed JavaScript")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


if __name__ == "__main__":
    main()
