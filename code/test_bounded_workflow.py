"""Mock-only integration checks for bounded serving and notebook provenance."""

import argparse
from contextlib import redirect_stdout, redirect_stderr
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

import notebook_workflow as original
import notebook_workflow_bounded as bounded
import notebook_workflow_serial as serial


ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT / "code/guard"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def handler_checks():
    models = []
    class FakeClassifier:
        def __init__(self, *args):
            self.failure = None
            self.last_output = None
            models.append(self)

        def predict(self, *args):
            if self.failure:
                raise self.failure
            self.last_output = {"model": "jev-v3.2-2b", "safety": "unsafe", "prompt_tokens": 17, "truncated": False}
            return self.last_output

    with patch.dict(sys.modules, {"jev_inference": SimpleNamespace(JevClassifier=FakeClassifier)}):
        base = _load("mock_base_guard", GUARD / "serve.py")
    sys.path.insert(0, str(GUARD))
    try:
        with patch.dict(sys.modules, {"serve": base}):
            launcher = _load("mock_bounded_launcher", GUARD / "serve_bounded.py")
    finally:
        sys.path.pop(0)
    captured = {}
    def capture_server(self, address, handler, *args, **kwargs):
        captured.update(address=address, handler=handler)
    initial = dict(bounded.EXPECTED_RUNTIME)
    initial.pop("device")
    initial.pop("dtype")
    with patch.object(launcher, "install_bounded_sdpa", return_value=initial), patch.object(
        launcher.HTTPServer, "__init__", capture_server
    ), patch.object(launcher.HTTPServer, "serve_forever"), patch.dict(
        os.environ, {"JEV_API_KEY": "mock-auth-token"}
    ), patch.object(sys, "argv", ["serve_bounded.py", "--device", "cpu"]), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        launcher.main()
    handler = captured["handler"]
    baseline = handler.__bases__[0]
    model = models[0]

    def request(cls, method, path, *, authorized=True, payload=None, body=None, declared_size=None):
        raw = body if body is not None else json.dumps(payload or {}).encode()
        instance = cls.__new__(cls)
        instance.path = path
        instance.headers = {"Content-Length": str(len(raw) if declared_size is None else declared_size)}
        if authorized:
            instance.headers["Authorization"] = "Bearer mock-auth-token"
        instance.rfile, instance.wfile = io.BytesIO(raw), io.BytesIO()
        status, headers = [], {}
        instance.send_response = status.append
        instance.send_header = lambda key, value: headers.update({key: value})
        instance.end_headers = lambda: None
        getattr(instance, "do_" + method)()
        wire = instance.wfile.getvalue()
        assert int(headers["Content-Length"]) == len(wire)
        assert headers["Content-Type"] == "application/json"
        return status[0], json.loads(wire)

    cases = [
        ("GET", "/health", {}, 200),
        ("GET", "/health", {"authorized": False}, 401),
        ("GET", "/unknown", {}, 404),
        ("GET", "/predict", {}, 404),
        ("POST", "/health", {}, 404),
        ("POST", "/predict", {"authorized": False}, 401),
        ("POST", "/predict", {"payload": {"input": "mock text", "options": []}}, 200),
        ("POST", "/predict", {"declared_size": 0}, 400),
        ("POST", "/predict", {"body": b"not json"}, 400),
        ("POST", "/predict", {"payload": {"options": []}}, 400),
    ]
    checks = []
    for method, path, options, expected_status in cases:
        status, value = request(handler, method, path, **options)
        base_status, base_value = request(baseline, method, path, **options)
        assert status == base_status == expected_status
        if status == 200:
            runtime = value.pop("guard_runtime")
            assert runtime["device"] == "cpu" and runtime["dtype"] == "float32" and runtime["all_tokens"] is True
            assert set(runtime["source_sha256"]) == {"serve.py", "jev_inference.py", "bounded_sdpa.py", "serve_bounded.py"}
        else:
            assert "guard_runtime" not in value
        value.pop("latency_ms", None)
        base_value.pop("latency_ms", None)
        assert value == base_value
        checks.append(f"{method} {path} status {status}")
    assert "guard_runtime" not in model.last_output
    for failure, expected_status in ((ValueError("mock invalid input"), 400), (RuntimeError("mock inference failure"), 503)):
        model.failure = failure
        options = {"payload": {"input": "mock text", "options": []}}
        actual = request(handler, "POST", "/predict", **options)
        expected = request(baseline, "POST", "/predict", **options)
        assert actual == expected and actual[0] == expected_status and "guard_runtime" not in actual[1]
        checks.append(f"original classifier exception status {expected_status}")
    return checks


def workflow_checks():
    saved = (original._start_guard, original._stop, original._health)
    runtime = dict(bounded.EXPECTED_RUNTIME)
    runtime["source_sha256"] = {name: "0" * 64 for name in ("serve.py", "jev_inference.py", "bounded_sdpa.py", "serve_bounded.py")}
    ready = {"status": "ready", "model": "jev-v3.2-2b", "truncation": False, "guard_runtime": runtime}
    job_result = {"returncode": 0, "stopped_after_repeated_errors": False, "elapsed_seconds": 0, "output": "mock"}
    checks = []
    with tempfile.TemporaryDirectory(prefix="bounded_workflow_mock_") as directory:
        root = Path(directory).resolve()
        for name in (
            "code/notebook_workflow.py", "code/notebook_workflow_serial.py", "code/notebook_workflow_bounded.py",
            "code/guard/serve.py", "code/guard/jev_inference.py", "code/guard/bounded_sdpa.py", "code/guard/serve_bounded.py",
            "code/guard/serve_serial.py",
        ):
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
        (root / "code/rerun.py").touch()
        (root / "inputs").mkdir()
        for experiment in ("full1000", "paired90"):
            (root / f"inputs/{experiment}_states.jsonl").write_text("{}\n")
        (root / "python.exe").touch()
        (root / "checkpoint").mkdir()
        (root / "base-model").mkdir()

        with patch.object(original, "_health", return_value=ready), patch.object(original, "_run_job", return_value=job_result) as jobs, patch.object(serial, "_start_guard_entry") as start, patch.object(serial.subprocess, "run") as kill:
            session = bounded.run_all(root, api_key="mock-api-for-test")
            assert jobs.call_count == 6
            start.assert_not_called()
            kill.assert_not_called()
        record = json.loads((session / "guard_runtime.json").read_text())
        assert record["runtime_validated"] and record["actual_health"] == ready
        assert record["guard_started_by_notebook"] is False
        assert len(record["package_source_sha256"]) == 7
        assert "mock-api-for-test" not in (session / "guard_runtime.json").read_text()
        original_record = (session / "guard_runtime.json").read_bytes()
        try:
            bounded._write_runtime_record(root, session, {})
            raise AssertionError("Existing provenance was overwritten")
        except FileExistsError:
            pass
        assert (session / "guard_runtime.json").read_bytes() == original_record
        checks.append("matching_external_runtime_reused_six_jobs_and_exclusive_provenance")

        wrong = {key: value for key, value in ready.items() if key != "guard_runtime"}
        with patch.object(original, "_health", return_value=wrong), patch.object(original, "_run_job") as jobs, patch.object(serial, "_start_guard_entry") as start, patch.object(serial.subprocess, "run") as kill:
            rejected = bounded.run_all(root, api_key="mock-api-for-test")
            jobs.assert_not_called()
            start.assert_not_called()
            kill.assert_not_called()
        rejected_record = json.loads((rejected / "guard_runtime.json").read_text())
        assert rejected_record["session_status"] == "setup_failed" and not rejected_record["runtime_validated"]
        assert rejected_record["actual_health"] == wrong
        assert rejected != session
        checks.append("healthy_wrong_runtime_rejected_without_start_stop_or_jobs")

        diagnostic = dict(ready, guard_runtime=dict(runtime, threshold=0))
        try:
            bounded._validate_runtime(diagnostic)
            raise AssertionError("Diagnostic threshold accepted as production")
        except RuntimeError as error:
            assert "threshold" in str(error)
        checks.append("diagnostic_threshold_not_accepted_as_production")

        process = Mock(pid=654321)
        process.poll.return_value = None
        with patch.object(original, "_health", side_effect=[OSError("mock absent"), ready]), patch.object(original, "_run_job", return_value=job_result) as jobs, patch.object(serial.subprocess, "Popen", return_value=process) as popen, patch.object(serial.subprocess, "run") as kill, patch.dict(os.environ, {"JEV_SDPA_CHUNK_SIZE": "7", "JEV_SDPA_CHUNK_THRESHOLD": "0"}):
            fresh = bounded.run_all(root, api_key="mock-api-for-test", guard_python=root / "python.exe",
                                    guard_checkpoint=root / "checkpoint", guard_base_model=root / "base-model")
            assert jobs.call_count == 6
            call = popen.call_args
            assert call.args[0][2] == str(root / "code/guard/serve_bounded.py")
            assert call.kwargs["env"]["JEV_SDPA_CHUNK_SIZE"] == "256"
            assert call.kwargs["env"]["JEV_SDPA_CHUNK_THRESHOLD"] == "4096"
            assert call.kwargs["env"]["OMP_NUM_THREADS"] == call.kwargs["env"]["MKL_NUM_THREADS"] == "4"
            assert kill.call_args.args[0] == ["taskkill", "/PID", "654321", "/T", "/F"]
            assert call.kwargs["stdout"].closed
        fresh_record = json.loads((fresh / "guard_runtime.json").read_text())
        assert fresh_record["runtime_validated"] and fresh_record["guard_started_by_notebook"] is True
        checks.append("absent_service_starts_bounded_defaults_owned_tree_cleanup_and_provenance")

        serial_session = root / "serial-start"
        serial_session.mkdir()
        process = Mock(pid=654321)
        process.poll.return_value = None
        with patch.object(original, "_health", return_value=ready), patch.object(serial.subprocess, "Popen", return_value=process) as popen:
            _, log, _ = serial._start_guard_serial(root, serial_session, "http://127.0.0.1:29660/predict", root / "python.exe", root / "checkpoint", root / "base-model", None, 10)
            assert popen.call_args.args[0][2] == str(root / "code/guard/serve_serial.py")
            log.close()
        checks.append("shared_startup_keeps_serial_entrypoint")

    for failure in (RuntimeError("mock workflow failure"), KeyboardInterrupt()):
        with patch.object(original, "run_all", side_effect=failure):
            try:
                bounded.run_all("mock-root", api_key="mock-key")
                raise AssertionError("Expected delegated exception")
            except BaseException as error:
                assert error is failure
        assert (original._start_guard, original._stop, original._health) == saved
        assert not serial._WORKFLOW_LOCK.locked()
    checks.append("hooks_and_lock_restored_after_exception_and_interrupt")
    with patch.object(original, "run_all", side_effect=lambda *a, **kw: serial.run_all()):
        try:
            bounded.run_all("mock-root", api_key="mock-key")
            raise AssertionError("Concurrent serial workflow accepted")
        except RuntimeError as error:
            assert "Only one" in str(error)
    assert (original._start_guard, original._stop, original._health) == saved
    assert not serial._WORKFLOW_LOCK.locked()
    checks.append("shared_serial_bounded_single_workflow_lock")
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args()
    report = {"status": "passed", "handler_checks": handler_checks(), "workflow_checks": workflow_checks(),
              "real_model_loads": 0, "real_network_or_api_calls": 0, "real_process_starts_or_stops": 0}
    rendered = json.dumps(report, indent=2) + "\n"
    if arguments.report:
        with arguments.report.open("x", encoding="utf-8") as output:
            output.write(rendered)
    print(rendered)
