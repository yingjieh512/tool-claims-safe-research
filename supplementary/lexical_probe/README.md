# Exploratory lexical probe

The paper reports group-held-out macro-F1 **0.829** for complete trajectories
and **0.880** for final answers alone. These are results of a separate trained
TF-IDF/logistic-regression probe, not partial-view measurements of Jev-Guard or
the TypeSafe models.

## Available historical evidence

[historical_report.md](historical_report.md) is the original English report,
dated 2026-09-29, copied without alteration. It reports a paired difference of
+0.051 with a fixed-cluster bootstrap 95% interval of [+0.011, +0.091].
The exact historical out-of-fold predictions, fold assignments, result JSON,
and execution-environment version record were **not found in the available
local archives**. Consequently, these historical numbers are documented by the
report but have not been independently recomputed for this repository.

The historical report includes commands for the original repository. Use the
commands below for this self-contained subset. No fresh lexical model training
was performed while assembling this repository.

## Protocol and code

- Dataset: the same 1,000-row ATBench release in `../../data/atbench_test.json`.
- Five views: full, user only, tool schema only, actions only, final answer only.
- Word unigram/bigram TF-IDF, at most 50,000 features, class-balanced logistic
  regression; vocabulary fitting occurs within each training fold.
- Five-fold stratified random CV and approximate group-held-out CV; seed 42.
- Fifty label-free TF-IDF/KMeans clusters from user requests and tool names.
  The cluster vectorizer uses at most 20,000 features; KMeans uses `n_init=10`.
- Five hundred bootstrap draws over the fixed inferred groups and fixed
  out-of-fold predictions, including paired final-minus-full macro-F1 differences.

Six source modules in `lexical_audit/` are exact copies of the original files.
`reporting.py` preserves the exact aggregation portion of the original module;
unused report-rendering functions are omitted. `source_manifest.json` records
source paths relative to the original repository and SHA-256 hashes. The new
`run_probe.py` is a small portable entry point calling those original algorithms.

## Validate or run a fresh replication

From the new repository root, validate the pinned input without training:

```sh
python supplementary/lexical_probe/run_probe.py
```

To perform a fresh CPU replication, use a Python 3.10+ environment:

```sh
python -m pip install -r supplementary/lexical_probe/requirements.txt
python supplementary/lexical_probe/run_probe.py --run
```

Each invocation with `--run` creates a new `runs/lexical_<timestamp>_<id>/`
directory containing configuration and dependency versions, group assignments,
fold-level predictions, and aggregate/bootstrap results. It never replaces the
historical report or archived judge predictions. Dependency constraints match
the original project; they are not asserted to recover its missing historical
runtime exactly, so new measurements may differ.

This is exploratory cross-validation on the official test records, not an
official leaderboard score. Cluster definitions are inferred proxies for task
families. The result does not establish a causal explanation for a pretrained
safety judge, annotation leakage, or the suitability of final-answer-only
auditing in deployment.
