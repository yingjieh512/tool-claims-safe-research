# Code guide

All commands below run from the review package root. The offline path is the shortest way to reproduce the reported tables and figures; live replay is optional and produces new measurements.

## Implementation map

| File | Responsibility |
| --- | --- |
| `analysis.py` | Validate archived predictions; recompute confusion counts, metrics, parent-level paired differences, and bootstrap intervals |
| `prepare_inputs.py` | Rebuild all 1,000 inputs, recompute the fixed 90-parent selection, and regenerate all 270 paired inputs |
| `verify_package.py` | One-command offline verification of evidence, reconstruction, and result agreement |
| `rerun.py` | Unified full-test/paired replay for Guard and both TypeSafe versions; resume, response validation, and error accounting |
| `trajectory_audit/data.py` | Original trajectory normalization |
| `trajectory_audit/views.py` | Original label-free input serialization |
| `trajectory_audit/adversarial.py` | Original tool-result perturbations and mechanical invariant checks |
| `guard/jev_inference.py`, `guard/serve.py` | Unmodified upstream Guard inference and HTTP server |
| `guard/fetch_models.py` | Explicit, revision-pinned model download and weight hash verification |

The three core trajectory modules remain unchanged to preserve exact input construction. Some original helper functionality remains in those modules for source fidelity; only the full view and `tail_safety` family are exercised here. The experiment-specific wrappers are consolidated, with no dependency on unrelated baseline or stepwise experiments.

## Offline verification

```sh
python -B code/verify_package.py
python -B code/prepare_inputs.py
```

To write rebuilt inputs to a separate location:

```sh
python -B code/prepare_inputs.py --output-dir runs/rebuilt-inputs
```

`verify_package.py` requires the analysis dependencies. Input reconstruction and replay dry runs use only the standard library. Verification never needs a credential or running model.

## Optional model replay

First inspect a configuration without making requests:

```sh
python -B code/rerun.py --experiment full1000 --model jev-1.13.0 --dry-run
python -B code/rerun.py --experiment paired90 --model jev-1.14.0 --dry-run
python -B code/rerun.py --experiment paired90 --model guard --endpoint http://127.0.0.1:29660/predict --dry-run
```

For TypeSafe, supply your own `TYPESAFE_API_KEY` through the process environment, or use the notebook's disabled-by-default live cell and hidden credential prompt. Do not save a credential in source or notebook cells. Actual API replay may incur usage charges.

```sh
python -B code/rerun.py --experiment full1000 --model jev-1.13.0
python -B code/rerun.py --experiment full1000 --model jev-1.14.0
python -B code/rerun.py --experiment paired90 --model jev-1.13.0
python -B code/rerun.py --experiment paired90 --model jev-1.14.0
```

For Guard, start the local server using [guard/README.md](guard/README.md), then run:

```sh
python -B code/rerun.py --experiment full1000 --model guard --endpoint http://127.0.0.1:29660/predict
python -B code/rerun.py --experiment paired90 --model guard --endpoint http://127.0.0.1:29660/predict
```

A full run makes 1,000 predictions per model; paired replay makes 270. `--limit 2` selects two full-test trajectories or two complete parent triplets. `--output-dir runs/new-run` creates an independent run. Outputs must remain under `runs/`, protecting the archived `results/`.

The runner sends exact frozen prompts without further shortening, validates model identity, retains raw responses, and records errors separately from binary verdicts. It validates input/configuration identity on resume, skips previously successful predictions, and retries failed records. A lock file prevents simultaneous writers; after forced termination, confirm that the old process has stopped before addressing a stale lock.

The historical Guard hosted endpoint is not a default replay target. A new local full-test run is a different deployment measurement. API availability and outputs can change; replay does not guarantee identical scores or labels to the archive.
