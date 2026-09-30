"""Preview the project on loopback only; this script never deploys or uploads."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit
import argparse
import socket
ROOT=Path(__file__).resolve().parents[1]
class Handler(SimpleHTTPRequestHandler):
    def send_head(self):
        parts=Path(unquote(urlsplit(self.path).path)).parts
        if any(p.startswith('.') or p=='__pycache__' for p in parts):
            self.send_error(404);return None
        return super().send_head()
    def list_directory(self,path):
        self.send_error(404,'Open index.html or code.html.');return None
class LocalServer(ThreadingHTTPServer):
    allow_reuse_address=False
    def server_bind(self):
        if hasattr(socket,'SO_EXCLUSIVEADDRUSE'):
            self.socket.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
        super().server_bind()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=18765);args=p.parse_args()
    server=LocalServer(('127.0.0.1',args.port),partial(Handler,directory=str(ROOT)))
    print(f'More from Less: http://127.0.0.1:{args.port}/',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
