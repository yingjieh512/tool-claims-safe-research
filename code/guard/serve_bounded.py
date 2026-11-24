"""Launch original Guard inference with bounded SDPA and serial HTTP handling."""

from http.server import HTTPServer
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

from bounded_sdpa import install_bounded_sdpa
import serve


def runtime_http_server(settings):
    """Add provenance only to successful original health/predict responses."""
    class RuntimeHTTPServer(HTTPServer):
        def __init__(self, server_address, handler_class, *args, **kwargs):
            original_reply = handler_class.reply

            class RuntimeHandler(handler_class):
                def reply(self, status, value):
                    if status == 200 and self.path in ("/health", "/predict") and isinstance(value, dict):
                        value = dict(value)
                        value["guard_runtime"] = dict(settings)
                    return original_reply(self, status, value)

            super().__init__(server_address, RuntimeHandler, *args, **kwargs)

    return RuntimeHTTPServer


def runtime_settings(settings, device):
    source_root = Path(__file__).resolve().parent
    result = dict(settings)
    result.update(
        device=device, dtype="bfloat16" if str(device).startswith("cuda") else "float32",
        torch_threads=torch.get_num_threads(), all_tokens=True,
        source_sha256={name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
                       for name in ("serve.py", "jev_inference.py", "bounded_sdpa.py", "serve_bounded.py")},
    )
    return result


def main():
    runtime_parser = argparse.ArgumentParser(add_help=False)
    runtime_parser.add_argument("--device", default="cuda:0")
    runtime_args, _ = runtime_parser.parse_known_args()
    settings = runtime_settings(install_bounded_sdpa(), runtime_args.device)
    print(json.dumps({"bounded_sdpa": settings}), file=sys.stderr, flush=True)
    serve.ThreadingHTTPServer = runtime_http_server(settings)
    serve.main()


if __name__ == "__main__":
    main()
