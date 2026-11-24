"""Run the original Guard inference on one persistent request thread.

This additive CPU memory mitigation changes only HTTP request threading. Model
loading, inference arithmetic, labels, and command-line options remain in serve.py.
Use the original --device cpu and four-thread environment for the current replay.
Unlike ThreadingHTTPServer, this service cannot answer concurrent health checks
while a prediction is running. The suspected native per-request-thread memory
retention has not been established as the failure's precise allocator cause.
"""

from http.server import HTTPServer

import serve


def main():
    serve.ThreadingHTTPServer = HTTPServer
    serve.main()


if __name__ == "__main__":
    main()
