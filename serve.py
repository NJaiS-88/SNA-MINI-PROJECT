"""
Launch local HTTP server to view the AI Influencer Campaign interactive dashboard.
Usage: python serve.py [port]
"""

import http.server
import socketserver
import webbrowser
import os
import sys
PORT = int(os.environ.get("PORT", sys.argv[1] if len(sys.argv) > 1 else 8080))

class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        # Keep console output clean
        sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")

if __name__ == "__main__":
    web_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(web_dir)
    
    url = f"http://localhost:{PORT}/index.html"
    print("=" * 65)
    print(f"🚀 AI Influencer Campaign Dashboard is running!")
    print(f"📍 Local URL: {url}")
    print(f"💡 Press Ctrl+C to stop the server")
    print("=" * 65)
    
    try:
        webbrowser.open(url)
    except Exception:
        pass
        
    socketserver.TCPServer.allow_reuse_address = True
    try:
        with socketserver.TCPServer(("", PORT), QuietHandler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
