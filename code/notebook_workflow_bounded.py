"""Inactive-by-default adapter for validated bounded CPU Guard replay.

Use explicitly after model parity validation. The original workflow, frozen
inputs, scoring and six-job order are unchanged. Scoped hooks share the serial
adapter's single-workflow lock and are restored on every exit.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import notebook_workflow as original
import notebook_workflow_serial as serial


EXPECTED_RUNTIME = {
    "implementation": "query_chunked_cpu_float32_math_sdpa",
    "chunk_size": 256,
    "threshold": 4096,
    "device": "cpu",
    "dtype": "float32",
    "torch_threads": 4,
    "full_key_value_context": True,
    "all_tokens": True,
    "truncation": False,
}


def _validate_runtime(health):
    runtime = health.get("guard_runtime")
    if not isinstance(runtime, dict):
        raise RuntimeError("Guard is healthy but does not report the required bounded runtime")
    mismatches = [key for key, value in EXPECTED_RUNTIME.items() if runtime.get(key) != value]
    if mismatches:
        raise RuntimeError("Guard is healthy but its bounded runtime differs in: " + ", ".join(mismatches))
    hashes = runtime.get("source_sha256")
    names = ("serve.py", "jev_inference.py", "bounded_sdpa.py", "serve_bounded.py")
    if not isinstance(hashes, dict) or any(
        not isinstance(hashes.get(name), str) or len(hashes[name]) != 64
        or any(char not in "0123456789abcdef" for char in hashes[name]) for name in names
    ):
        raise RuntimeError("Guard is healthy but its bounded runtime source hashes are missing or invalid")


def _start_guard_bounded(root, session, endpoint, python, checkpoint, base_model, callback, timeout):
    return serial._start_guard_entry(
        root, session, endpoint, python, checkpoint, base_model, callback, timeout,
        entrypoint_name="serve_bounded.py",
        extra_environment={"JEV_SDPA_CHUNK_SIZE": "256", "JEV_SDPA_CHUNK_THRESHOLD": "4096"},
    )


def _write_runtime_record(root, session, observed):
    root, session = Path(root).resolve(), Path(session)
    metadata = json.loads((session / "session.json").read_text(encoding="utf-8"))
    health = metadata.get("guard_health", observed.get("health"))
    names = (
        "code/notebook_workflow.py", "code/notebook_workflow_serial.py", "code/notebook_workflow_bounded.py",
        "code/guard/serve.py", "code/guard/jev_inference.py", "code/guard/bounded_sdpa.py", "code/guard/serve_bounded.py",
    )
    record = {
        "schema_version": 1, "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "session_id": metadata.get("session_id", session.name), "session_status": metadata.get("status"),
        "expected_runtime": EXPECTED_RUNTIME, "actual_health": health,
        "runtime_validated": observed.get("validated", False),
        "guard_started_by_notebook": metadata.get("guard_started_by_notebook"),
        "package_source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names},
    }
    with (session / "guard_runtime.json").open("x", encoding="utf-8") as output:
        json.dump(record, output, indent=2)
        output.write("\n")


def run_all(root, **kwargs):
    """Run in a fresh session, requiring the production bounded runtime.

    A matching external service is reused. A healthy service with another runtime
    is rejected without stopping it or launching a competing process. An absent
    service follows the original automatic-start policy. Actual observed runtime
    and source hashes are saved exclusively in the new session's guard_runtime.json.
    """
    if not serial._WORKFLOW_LOCK.acquire(blocking=False):
        raise RuntimeError("Only one additive notebook workflow may run in this process")
    base_health = original._health
    observed = {}

    def health_with_runtime(endpoint):
        health = base_health(endpoint)
        observed.update(health=health, validated=False)
        _validate_runtime(health)
        observed["validated"] = True
        return health

    try:
        with patch.object(original, "_start_guard", _start_guard_bounded), patch.object(
            original, "_stop", serial._stop_owned_tree
        ), patch.object(original, "_health", health_with_runtime):
            session = original.run_all(root, **kwargs)
        _write_runtime_record(root, session, observed)
        return session
    finally:
        serial._WORKFLOW_LOCK.release()
