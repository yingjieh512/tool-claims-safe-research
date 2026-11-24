"""Additive six-job notebook workflow; existing evidence and runners are read-only."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit, urlunsplit
from urllib.request import urlopen
import uuid

MODELS = ("guard", "jev-1.13.0", "jev-1.14.0")
EXPERIMENTS = ("full1000", "paired90")


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def _save_session(path, metadata):
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _notify(callback, event, **fields):
    if callback:
        callback({"event": event, **fields})


def _health(endpoint):
    parts = urlsplit(endpoint)
    if parts.scheme not in ("http", "https") or parts.username or parts.password:
        raise ValueError("Guard endpoint must be an HTTP(S) URL without embedded credentials")
    if parts.path.rstrip("/") != "/predict":
        raise ValueError("Use the original Guard /predict endpoint")
    url = urlunsplit((parts.scheme, parts.netloc, "/health", "", ""))
    with urlopen(url, timeout=3) as response:
        result = json.load(response)
    if result.get("status") != "ready" or result.get("model") != "jev-v3.2-2b":
        raise ValueError("Guard /health did not report the expected ready model")
    if result.get("truncation") is not False:
        raise ValueError("Guard /health must declare truncation=false")
    return result


def _stop(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def _start_guard(root, session, endpoint, python, checkpoint, base_model, callback, timeout):
    parts = urlsplit(endpoint)
    if parts.scheme != "http" or parts.hostname not in ("127.0.0.1", "localhost"):
        raise ValueError("Automatic Guard startup supports an HTTP loopback endpoint only")
    python, checkpoint, base_model = Path(python), Path(checkpoint), Path(base_model)
    if not python.is_file() or not checkpoint.is_dir() or not base_model.is_dir():
        raise FileNotFoundError(
            "Guard is unavailable and its local Python/checkpoint/base paths are missing. "
            "Follow code/guard/README.md, or configure existing local paths in the notebook."
        )
    environment = os.environ.copy()
    environment.pop("TYPESAFE_API_KEY", None)
    environment.pop("JEV_API_KEY", None)
    environment.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    command = [str(python), "-B", str(root / "code/guard/serve.py"),
               "--checkpoint", str(checkpoint), "--base-model", str(base_model),
               "--device", "cpu", "--host", "127.0.0.1", "--port", str(parts.port or 80)]
    log = (session / "guard_server.log").open("x", encoding="utf-8")
    process = None
    try:
        process = subprocess.Popen(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        start = time.monotonic()
        last_update = 0.0
        while time.monotonic() - start < timeout:
            if process.poll() is not None:
                raise RuntimeError("Local Guard exited during startup; inspect this session's guard_server.log")
            try:
                health = _health(endpoint)
                return process, log, health
            except (OSError, ValueError):
                pass
            elapsed = time.monotonic() - start
            if elapsed - last_update >= 10:
                _notify(callback, "guard_starting", elapsed_seconds=round(elapsed))
                last_update = elapsed
            time.sleep(1)
        raise TimeoutError("Local Guard did not become ready within the configured startup timeout")
    except BaseException:
        _stop(process)
        log.close()
        raise


def _run_job(root, session, experiment, model, endpoint, api_key, callback):
    output = session / experiment / model
    output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-B", "-u", str(root / "code/rerun.py"),
               "--experiment", experiment, "--model", model, "--output-dir", str(output)]
    if model == "guard":
        command += ["--endpoint", endpoint]
    environment = os.environ.copy()
    environment.update(PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    if model == "guard":
        environment.pop("TYPESAFE_API_KEY", None)
    else:
        environment["TYPESAFE_API_KEY"] = api_key
    process = None
    events = queue.Queue()
    expected = 1000 if experiment == "full1000" else 270
    started = time.monotonic()
    completed, consecutive_failures, stopped_for_errors = 0, 0, False
    with (output / "runner.log").open("x", encoding="utf-8") as log:
        try:
            process = subprocess.Popen(command, cwd=root, env=environment, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            def reader():
                try:
                    for line in process.stdout:
                        events.put(line)
                finally:
                    events.put(None)
            thread = threading.Thread(target=reader, daemon=True)
            thread.start()
            finished, last_update = False, 0.0
            while not finished:
                try:
                    line = events.get(timeout=1)
                    if line is None:
                        finished = True
                    else:
                        line = line.replace(api_key, "[REDACTED]") if api_key else line
                        log.write(line)
                        log.flush()
                        match = re.match(r"(\d+)/(\d+) id=.* error=(\S+)", line)
                        if match:
                            completed = int(match.group(1))
                            consecutive_failures = consecutive_failures + 1 if match.group(3) != "None" else 0
                            if consecutive_failures >= 5:
                                stopped_for_errors = True
                                _stop(process)
                except queue.Empty:
                    pass
                elapsed = time.monotonic() - started
                if elapsed - last_update >= 5 or finished:
                    _notify(callback, "job_progress", experiment=experiment, model=model,
                            completed=completed, expected=expected, elapsed_seconds=round(elapsed))
                    last_update = elapsed
            returncode = process.wait()
            return {"returncode": returncode, "stopped_after_repeated_errors": stopped_for_errors,
                    "elapsed_seconds": round(time.monotonic() - started, 2),
                    "output": output.relative_to(root).as_posix()}
        finally:
            _stop(process)
            if process is not None and process.stdout is not None:
                process.stdout.close()
            environment.pop("TYPESAFE_API_KEY", None)


def run_all(root, *, api_key, guard_endpoint="http://127.0.0.1:29660/predict",
            auto_start_guard=True, guard_python=None, guard_checkpoint=None, guard_base_model=None,
            guard_startup_timeout=600, callback=None):
    """Run all six jobs in a brand-new session and return its directory, including on partial failure.

    This is an explicit live operation. It never resumes or overwrites a pre-existing run.
    Only a Guard subprocess started by this function is terminated when the workflow ends.
    """
    root = Path(root).resolve()
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("A TypeSafe API key is required for the six-job live workflow")
    api_key = api_key.strip()
    for experiment in EXPERIMENTS:
        if not (root / f"inputs/{experiment}_states.jsonl").is_file():
            raise FileNotFoundError(f"Missing frozen {experiment} input")
    if not (root / "code/rerun.py").is_file():
        raise FileNotFoundError("Cannot find the existing unified runner")
    runs = root / "runs"
    if runs.is_symlink():
        raise ValueError("The runs directory must be a normal directory within the package")
    runs.mkdir(exist_ok=True)
    if runs.resolve().parent != root:
        raise ValueError("Resolved run output must remain inside the package")
    session_id = "notebook_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    session = runs / session_id
    session.mkdir(exist_ok=False)
    tasks = [{"experiment": e, "model": m, "status": "pending"} for e in EXPERIMENTS for m in MODELS]
    metadata = {"schema_version": 1, "session_id": session_id, "created_utc": _timestamp(),
                "status": "starting", "cohort": "fixed historical 90-parent selection",
                "guard_endpoint": guard_endpoint, "tasks": tasks,
                "inputs_sha256": {e: hashlib.sha256((root / f"inputs/{e}_states.jsonl").read_bytes()).hexdigest()
                                   for e in EXPERIMENTS}}
    manifest = session / "session.json"
    _save_session(manifest, metadata)
    owned_guard, guard_log = None, None
    _notify(callback, "session_started", session=session.relative_to(root).as_posix())
    try:
        try:
            metadata["guard_health"] = _health(guard_endpoint)
            metadata["guard_started_by_notebook"] = False
        except (OSError, ValueError) as exc:
            if not auto_start_guard:
                raise RuntimeError("Guard is not ready. Start it using code/guard/README.md or enable automatic startup.") from exc
            owned_guard, guard_log, health = _start_guard(
                root, session, guard_endpoint,
                guard_python or root / ".venv-guard/Scripts/python.exe",
                guard_checkpoint or root / "models/jev-guard-v3.2-2b/checkpoint",
                guard_base_model or root / "models/qwen3.5-2b", callback, guard_startup_timeout)
            metadata["guard_health"] = health
            metadata["guard_started_by_notebook"] = True
        metadata["status"] = "running"
        _save_session(manifest, metadata)
        for task in tasks:
            task.update(status="running", started_utc=_timestamp())
            _save_session(manifest, metadata)
            _notify(callback, "job_started", experiment=task["experiment"], model=task["model"])
            try:
                task.update(_run_job(root, session, task["experiment"], task["model"], guard_endpoint, api_key, callback))
                task["status"] = "finished" if task["returncode"] == 0 else "process_failed"
            except Exception as exc:
                task.update(status="process_failed", error=str(exc).replace(api_key, "[REDACTED]"))
            task["finished_utc"] = _timestamp()
            _save_session(manifest, metadata)
            _notify(callback, "job_finished", **task)
        metadata["status"] = "jobs_finished" if all(t["status"] == "finished" for t in tasks) else "finished_with_process_errors"
    except KeyboardInterrupt:
        metadata["status"] = "interrupted"
        for task in tasks:
            if task["status"] == "running":
                task["status"] = "interrupted"
    except Exception as exc:
        metadata.update(status="setup_failed", error=str(exc).replace(api_key, "[REDACTED]"))
    finally:
        _stop(owned_guard)
        if guard_log is not None:
            guard_log.close()
        metadata["finished_utc"] = _timestamp()
        _save_session(manifest, metadata)
        _notify(callback, "session_finished", session=session.relative_to(root).as_posix(), status=metadata["status"],
                error=metadata.get("error"))
    return session
