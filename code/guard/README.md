# Local Jev-Guard v3.2 inference

`jev_inference.py` and `serve.py` retain the upstream source bytes. The model is `hubin/jev-guard-v3.2-2b`, using `Qwen/Qwen3.5-2B`. Revision IDs, weight hashes, and the inspected local runtime are recorded in [runtime_provenance.json](runtime_provenance.json).

Reading the report or executing the offline notebook does not require these dependencies or weights. Model files are intentionally excluded from the review package.

## Windows CPU setup

From the package root, create a separate Python 3.11 environment:

```powershell
py -3.11 -m venv .venv-guard
.\.venv-guard\Scripts\python.exe -m pip install -r requirements-guard.txt
.\.venv-guard\Scripts\python.exe code\guard\fetch_models.py
```

The explicit download command writes pinned model snapshots under `models/` and checks the base weights, adapter, safety heads, and inference source hashes. The base weights are about 4.55 GB; CPU float32 loading requires additional memory. `--verify-only` checks already downloaded weights without network access.

Start the original HTTP server:

```powershell
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
.\.venv-guard\Scripts\python.exe code\guard\serve.py --checkpoint models\jev-guard-v3.2-2b\checkpoint --base-model models\qwen3.5-2b --device cpu --host 127.0.0.1 --port 29660
```

The CPU path uses float32. In a second terminal, check `http://127.0.0.1:29660/health`, then use the replay commands in [../README.md](../README.md). The runner uses `/predict` with an empty options list, `mode="single"`, and `safety_applicable=true`.

The model accepts at most 16,384 tokens; oversized inputs are rejected rather than silently shortened. Frozen paired inputs must retain their hashes. CPU inference may take substantial time; the unified runner uses a longer default timeout for Guard.

## Evidence and source scope

The archived full-test Guard predictions came from a historical hosted service. The 90-parent experiment used local CPU float32. A prior 21-short-input check found agreement on 20 of 21 labels, which does not establish full equivalence between deployments. New CPU full-test results must be recorded as a new run.

`requirements-guard.txt` records the inspected local environment, not the historical hosted server's environment. Upstream model cards, license notices, and the original `SHA256SUMS` are retained beside this document. The upstream hash list also names demo assets omitted from this minimal package. Any temporary hosted URL in those cards is historical source documentation.
