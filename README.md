# When a Tool Claims Safe

Research code and evidence for **When a Tool Claims Safe: Probabilistic Decision Model As Agent Security Judge under Forged Result Attack**, by Yingjie Huang. The manuscript is included as [paper/Capstone.pdf](paper/Capstone.pdf).

The study asks whether a tool result that claims a workflow was safe can cause an agent safety auditor to miss unsafe behavior. It evaluates Jev-Guard v3.2 and TypeSafe `jev-1.13.0` / `jev-1.14.0` on 1,000 ATBench trajectories, followed by a fixed 90-parent clean/control/attack experiment.

## Start here

- **Paper measurements:** [results_overview.ipynb](results_overview.ipynb), [saved HTML report](results_overview.html), and [RESULTS.md](RESULTS.md).
- **Interactive view and optional fresh rerun:** [run_and_compare.ipynb](run_and_compare.ipynb). Default execution is offline and displays the included later rerun. Set `VIEW_RUN=None` to display the paper's historical measurements.
- **Claim-to-evidence map and paper clarifications:** [docs/evidence_map.md](docs/evidence_map.md).
- **Exploratory lexical probe:** [supplementary/lexical_probe](supplementary/lexical_probe). Its archived report and implementation are included; the original fold assignments and out-of-fold predictions were not recovered, so its paper numbers cannot be independently recomputed from preserved predictions.
- **Later completed rerun:** [executed Notebook](runs/notebook_20261116T225617Z_b4e8fb4b/completed_report/run_and_compare.executed.ipynb), [HTML](runs/notebook_20261116T225617Z_b4e8fb4b/completed_report/report.html), and [verification record](runs/notebook_20261116T225617Z_b4e8fb4b/completion_verification.json).

## Measurements reported in the paper

The full test contains 503 safe and 497 unsafe trajectories. Paired entries count unsafe parents classified as safe, with 90 parents per arm.

| Auditor | Full accuracy | Unsafe recall | Clean FN | Control FN | Attack FN | Attack minus control FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Jev-Guard v3.2 | 56.6% | 43.7% | 0/90 | 2/90 | 50/90 | 48/90 |
| TypeSafe jev-1.13.0 | 66.3% | 50.3% | 1/90 | 2/90 | 26/90 | 24/90 |
| TypeSafe jev-1.14.0 | 64.4% | 47.5% | 1/90 | 4/90 | 22/90 | 18/90 |

These measurements are preserved under `results/`. The original Guard full test used a hosted deployment; its paired experiment used local CPU float32 inference. Full-test coverage does not mean every original trajectory was unabridged: inherited preprocessing omits the middle of 52 long full-test inputs. All 90 paired parents have no such omission.

The TypeSafe implementation classifies `unsafe_probability >= 0.5` as unsafe, including ties. Guard uses its published safety-head verdict. The paper's phrase "greater than 0.5" does not describe the implemented TypeSafe tie rule; the code and archived measurements are retained unchanged.

## Later full rerun, kept separately

The November 16-17, 2026 rerun reused all frozen inputs and the original fixed cohort. It produced 3,810 valid predictions: 3,000 full-test predictions plus 810 paired predictions. These are additional measurements, not replacements for the paper's tables.

| Auditor | Full accuracy | Clean FN | Control FN | Attack FN | Net additional FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| Jev-Guard v3.2, local CPU | 57.2% | 0/90 | 2/90 | 50/90 | 48/90 |
| TypeSafe jev-1.13.0 | 66.7% | 2/90 | 4/90 | 25/90 | 21/90 |
| TypeSafe jev-1.14.0 | 64.7% | 0/90 | 4/90 | 21/90 | 17/90 |

The local Guard full test required memory recovery. The first 366 successful predictions were retained, and 634 further predictions used query-chunked CPU float32 attention with all supplied keys, values, and input tokens. No additional input truncation was introduced. Four diagnostic real-model cases matched labels and token counts; maximum absolute probability difference was approximately `1.61e-6`. This is diagnostic agreement, not a claim of bitwise equivalence for all inputs. Five earlier failed call records remain in the evidence; all affected inputs were subsequently scored successfully.

## Reproduce the analysis offline

Use Python 3.9-3.11; Python 3.11 is recommended for a new environment. From the repository root:

```sh
python -m pip install -r requirements.txt
python -B code/verify_package.py
python -m jupyterlab run_and_compare.ipynb
```

Choose **Restart Kernel and Run All Cells**. By default this reads saved predictions and refreshes all tables and six figures without any API call, model startup, download, or result overwrite. The paper-specific `results_overview.ipynb` also runs offline; it exports the historical figure files when executed.

The smaller input-reconstruction and dry-run checks require only the standard library:

```sh
python -B code/prepare_inputs.py
python -B code/rerun.py --experiment paired90 --model jev-1.14.0 --dry-run
```

## Collect a new complete run

1. Prepare the local Guard environment and pinned model weights using [code/guard/README.md](code/guard/README.md). Weights and virtual environments are not included. Downloading them is an explicit setup step.
2. Open `run_and_compare.ipynb`. Repository-relative Guard paths are the defaults; optionally set `GUARD_PYTHON`, `GUARD_CHECKPOINT`, and `GUARD_BASE_MODEL` in the environment.
3. Supply `TYPESAFE_API_KEY` through the environment or the notebook's hidden prompt. Do not put credentials in a notebook cell or tracked file.
4. Set `RUN_ALL_EXPERIMENTS=True` and `VIEW_RUN=None`, then run all cells.

The notebook runs all six jobs, creates a fresh timestamped directory under `runs/`, and then reads that directory to refresh its tables and figures. It uses [the bounded Guard workflow](code/notebook_workflow_bounded.py), which validates the serving runtime. An incompatible healthy external service is rejected; an already-running service is not stopped. The notebook stops services that it starts itself. Local CPU inference can take many hours; TypeSafe requests require account access and may incur usage charges. Model availability and outputs may change, so a new run need not reproduce archived labels exactly.

Individual replay commands and resume behavior are documented in [code/README.md](code/README.md). `--output-dir` must point under `runs/`; archive paths are protected. The fixed 90 parents are not reselected using new predictions.

## Repository layout

| Path | Purpose |
| --- | --- |
| `code/` | Input construction, prediction replay, offline metrics, and original/bounded Guard inference |
| `data/` | Original ATBench test data |
| `inputs/` | Frozen 1,000 full inputs, 270 paired inputs, cohort selection, and edit manifest |
| `results/` | Per-example predictions and summaries matching the paper |
| `figures/` | Historical plots recomputed from the paper's predictions |
| `runs/notebook_20261116T225617Z_b4e8fb4b/` | Completed later rerun and recovery audit records |
| `supplementary/lexical_probe/` | Exploratory lexical implementation and the preserved report |
| `provenance/` | Source/data hashes, original validation records, and repository copy manifest |
| `paper/` | Unmodified supplied manuscript |
| `docs/` | Evidence map and methodological clarifications |

Raw inputs and prediction archives retain their original bytes and language. Reader-facing documentation, code explanations, and figure labels are English. Run audit metadata uses portable placeholders for author-local filesystem paths. Its `record_scope` fields explain that retained historical digests and execution timestamps describe the original artifacts before this publication export. The completed report Notebook and HTML have been re-rendered offline with portable settings; numeric results and raw prediction rows are unchanged. Use the root `run_and_compare.ipynb` for new runs. The original bundle manifest and upstream `code/guard/SHA256SUMS` describe historical releases, not the current publication tree; weight hashes are unchanged. `provenance/repository_copy_manifest.json` distinguishes exact preserved files from adaptations using source and published hashes, while `provenance/research_export_validation.json` records the new offline checks.

## Scope and licensing

The paired results are conditional on 90 historically common true positives, contain only unsafe parents, and use one fixed attack template and location. Gold labels are inherited after mechanical invariant checks. These data do not establish general attack success rates, a causal account of model internals, calibrated cross-model probabilities, or results for future permission-aware defenses. See [RESULTS.md](RESULTS.md) and [the evidence map](docs/evidence_map.md).

Project-source licensing is retained in [LICENSE](LICENSE). Third-party models, data, and upstream code retain their own terms and notices; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
