# ATBench Partial-View Audit — MVP Result

Run date: 2026-09-29  
Dataset: [ATBench](https://huggingface.co/datasets/AI45Research/ATBench), 1,000 records  
Pinned data SHA-256: `80c534b5f3517c872b528a2e6b64e34495206968edd184f4aed563d39f0cca09`

## Question

Can a simple safety classifier predict ATBench's trajectory-level label without
reading the complete multi-step trajectory?

## Protocol

- Five views: full trajectory, user only, tool schema only, actions only, final answer only.
- Model: TF-IDF unigrams/bigrams plus class-balanced logistic regression.
- Evaluation: 5-fold out-of-fold predictions under both stratified random CV and
  approximate TF-IDF-cluster-held-out CV.
- Approximate groups: 50 TF-IDF/KMeans clusters derived from user requests and tool
  names without reading labels. All 50 groups contain both labels; sizes range 7–42.
- Uncertainty: 500 group-bootstrap resamples.
- Excluded from all inputs: `label`, `reason`, `risk_source`, `failure_mode`, and
  `real_world_harm`.

ATBench publishes only an official `test` split. These are custom diagnostic
cross-validation results, not leaderboard scores.

## Result

| Split | View | Macro-F1 | 95% group-bootstrap CI | AUROC | Unsafe FNR |
|---|---|---:|---:|---:|---:|
| Group held out | Full trajectory | 0.829 | [0.802, 0.850] | 0.903 | 0.179 |
| Group held out | User only | 0.681 | [0.638, 0.713] | 0.740 | 0.280 |
| Group held out | Tool schema only | 0.585 | [0.551, 0.614] | 0.627 | 0.447 |
| Group held out | Actions only | 0.658 | [0.627, 0.689] | 0.739 | 0.376 |
| **Group held out** | **Final answer only** | **0.880** | **[0.840, 0.906]** | **0.938** | **0.044** |
| Random | Full trajectory | 0.821 | [0.800, 0.839] | 0.895 | 0.189 |
| Random | Final answer only | 0.874 | [0.835, 0.901] | 0.937 | 0.050 |

The lexical classifier performs better with only the final answer than with the
complete trajectory under both split strategies. The signal persists under our
approximate clusters, but the release does not include original pair/template IDs,
so residual overlap between true scenario variants cannot be ruled out.

The paired Macro-F1 difference (`final answer only − full trajectory`) is +0.051
under the cluster-held-out split, with a conditional fixed-cluster 95% bootstrap
interval of [+0.011, +0.091]. Under the random split it is +0.053
([+0.015, +0.090]).

## Missing-answer ablation

Twenty-six trajectories have no terminal `Complete` answer, and all 26 are unsafe.
After removing all of them, preserving their original cluster assignments for the
remaining records, and rebuilding the folds, final-answer-only performance remains
high:

| Split | N | Macro-F1 | AUROC | Unsafe FNR |
|---|---:|---:|---:|---:|
| Group held out | 974 | 0.871 | 0.934 | 0.045 |
| Random | 974 | 0.871 | 0.936 | 0.051 |

After deleting these 26 records and rebuilding folds, Macro-F1 changes from 0.880 to
0.871. This does not isolate a causal marker effect, but it shows that the remaining
974 final answers still carry most of the predictive signal.

## What the lexical model uses

The strongest unsafe-associated terms include `tool`, `confirms`, `returned`,
`active`, and `tool returned`. Safe-associated terms include `security`, `reasons`,
`security reasons`, `privacy`, and `decline`. These associations suggest that
completion and refusal phrasing carries substantial label information.

This does not establish annotation leakage. The wording may legitimately summarize
whether an agent refused or completed an unsafe operation, and bag-of-words dilution
can make the much longer full trajectory harder for this particular model. The next
experiment should test a frozen compact safety judge and outcome-masked final answers.

The reported percentile intervals resample the 50 fixed inferred clusters and fixed
out-of-fold predictions. They do not include uncertainty from rerunning clustering,
fold construction, feature selection, or analysis choices. This exploratory protocol
was designed after inspecting the same 1,000 official test records; an untouched
confirmatory dataset is still needed.

## Reproduce

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/trajectory-audit download
LOKY_MAX_CPU_COUNT=4 .venv/bin/trajectory-audit run \
  --input data/raw/atbench/test.json \
  --output-dir outputs/atbench-mvp \
  --folds 5 \
  --groups 50 \
  --bootstrap-resamples 500
```
