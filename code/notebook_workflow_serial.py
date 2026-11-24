"""Run the unchanged notebook workflow with serial Guard startup and cleanup.

Only startup and owned-process cleanup are temporarily replaced. This is a
single-workflow-per-process adapter: its scoped module patches are not thread
local, so do not run the original workflow concurrently in the same process.
Serial HTTP handling alone does not prevent long-input attention memory peaks.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import threading
import time
from unittest.mock import patch
from urllib.parse import urlsplit

import notebook_workflow as original

_WORKFLOW_LOCK = threading.Lock()


def _stop_owned_tree(process):
    """Stop only a still-running process object owned by this workflow."""
    if process is None or process.poll() is not None:
        return
    if os.name == "nt" and isinstance(process.pid, int) and process.pid > 0:
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=15, check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            process.wait(timeout=10)
            return
        except (OSError, subprocess.TimeoutExpired):
            pass
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def _start_guard_serial(root, session, endpoint, python, checkpoint, base_model, callback, timeout):
    """Mirror original startup, changing only the server entry-point filename."""
    return _start_guard_entry(root, session, endpoint, python, checkpoint, base_model, callback, timeout,
                              entrypoint_name="serve_serial.py")


def _start_guard_entry(root, session, endpoint, python, checkpoint, base_model, callback, timeout,
                       *, entrypoint_name, extra_environment=None):
    """Shared additive startup implementation; the baseline workflow stays intact."""
    parts = urlsplit(endpoint)
    if parts.scheme != "http" or parts.hostname not in ("127.0.0.1", "localhost"):
        raise ValueError("Automatic Guard startup supports an HTTP loopback endpoint only")
    python, checkpoint, base_model = Path(python), Path(checkpoint), Path(base_model)
    if not python.is_file() or not checkpoint.is_dir() or not base_model.is_dir():
        raise FileNotFoundError(
            "Guard is unavailable and its local Python/checkpoint/base paths are missing. "
            "Follow code/guard/README.md, or configure existing local paths in the notebook."
        )
    entrypoint = Path(root) / "code/guard" / entrypoint_name
    if not entrypoint.is_file():
        raise FileNotFoundError("The additive Guard entry point is missing")
    environment = os.environ.copy()
    environment.pop("TYPESAFE_API_KEY", None)
    environment.pop("JEV_API_KEY", None)
    environment.update(
        PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
    )
    if extra_environment:
        environment.update(extra_environment)
    command = [
        str(python), "-B", str(entrypoint),
        "--checkpoint", str(checkpoint), "--base-model", str(base_model),
        "--device", "cpu", "--host", "127.0.0.1", "--port", str(parts.port or 80),
    ]
    log = (Path(session) / "guard_server.log").open("x", encoding="utf-8")
    process = None
    try:
        process = subprocess.Popen(
            command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        start = time.monotonic()
        last_update = 0.0
        while time.monotonic() - start < timeout:
            if process.poll() is not None:
                raise RuntimeError("Local Guard exited during startup; inspect this session's guard_server.log")
            try:
                health = original._health(endpoint)
                return process, log, health
            except (OSError, ValueError):
                pass
            elapsed = time.monotonic() - start
            if elapsed - last_update >= 10:
                original._notify(callback, "guard_starting", elapsed_seconds=round(elapsed))
                last_update = elapsed
            time.sleep(1)
        raise TimeoutError("Local Guard did not become ready within the configured startup timeout")
    except BaseException:
        _stop_owned_tree(process)
        log.close()
        raise


def run_all(*args, **kwargs):
    """Delegate experiments with scoped serial-service hooks, restored on exit.

    A ready external service is reused and is not stopped. Startup health checks
    precede replay; serial serving cannot answer concurrent health checks during
    inference. Model inputs, settings, scoring, output paths and order are unchanged.
    """
    if not _WORKFLOW_LOCK.acquire(blocking=False):
        raise RuntimeError("Only one serial notebook workflow may run in this process")
    try:
        with patch.object(original, "_start_guard", _start_guard_serial), patch.object(
            original, "_stop", _stop_owned_tree
        ):
            return original.run_all(*args, **kwargs)
    finally:
        _WORKFLOW_LOCK.release()
