#!/usr/bin/env python3
"""
serve.py — dev HTTP server for the RL Showcase dashboard.

Serves index.html + logs/ from the project root so the frontend
can fetch logs/manifest.json and logs/<agent>/replays.json directly.

Usage:
    python3 serve.py          # http://localhost:8080
    python3 serve.py 9000     # custom port
"""
import http.server, socketserver, os, sys, webbrowser

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
os.chdir(os.path.dirname(os.path.abspath(__file__)))


class _Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        # only log non-asset requests
        if not any(args[0].endswith(ext) for ext in (".png", ".ico", ".css")):
            super().log_message(fmt, *args)


with socketserver.TCPServer(("", PORT), _Handler) as httpd:
    url = f"http://localhost:{PORT}"
    print(f"\n  RL Showcase  →  {url}")
    print(f"  Press Ctrl+C to stop\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    httpd.serve_forever()
