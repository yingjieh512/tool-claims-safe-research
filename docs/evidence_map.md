# Paper evidence and reproduction map

This repository accompanies [Capstone.pdf](../paper/Capstone.pdf), titled **When a Tool Claims Safe: Probabilistic Decision Model As Agent Security Judge under Forged Result Attack**. The manuscript reports the historical measurements in `results/`. The later complete rerun in `runs/notebook_20261116T225617Z_b4e8fb4b/` is supplementary evidence and must not be substituted silently for the manuscript's numbers.

## Evidence used by the manuscript

| Paper location or claim | Supporting artifacts | Computation or reproduction |
| --- | --- | --- |
| Abstract and Section 5; Figure 1 on page 6: full-test binary safety metrics | [Historical full-test predictions and summaries](../results/full1000/), one directory for each of `guard`, `jev-1.13.0`, and `jev-1.14.0` | [Offline analysis](../code/analysis.py), [executed historical Notebook](../results_overview.ipynb), [HTML report](../results_overview.html) |
| Section 5; Figure 2 on page 6: clean, control, and attack false negatives | [Historical paired predictions, pairs, and summaries](../results/paired90/) | [Cross-model comparison](../results/paired90/comparison.json), [parent-level records](../results/paired90/per_parent.jsonl), `code/analysis.py` |
| Equation 1; Figure 3 on page 6: attack-minus-control effect and parent-bootstrap intervals | [Paired comparison](../results/paired90/comparison.json) | `code/analysis.py` recomputes matched-parent differences and 5,000 percentile bootstrap replicates using the recorded seeds |
| Section 4: pinned 1,000-case release, labels, and input construction | [Dataset](../data/), [frozen full-test inputs](../inputs/full1000_states.jsonl), [source manifest](../provenance/source_manifest.json), [input validation](../provenance/input_validation.json) | [Input reconstruction](../code/prepare_inputs.py), [normalization](../code/trajectory_audit/data.py), [serialization](../code/trajectory_audit/views.py) |
| Section 4: fixed 90-parent cohort and tool-result intervention | [Selection](../inputs/selection.json), [construction summary](../inputs/construction_summary.json), [edit manifest](../inputs/paired90_manifest.jsonl), [270 frozen paired inputs](../inputs/paired90_states.jsonl) | `code/prepare_inputs.py` and [perturbation implementation](../code/trajectory_audit/adversarial.py) |
| Sections 1, 4, and 5: exploratory TF-IDF probe, full-view macro-F1 0.829 versus final-only 0.880 | [Lexical probe supplement](../supplementary/lexical_probe/) | Retained historical report and reproduction source; the original per-example predictions and fold artifacts were not recovered. See the archival limitation below. |

The repository's [full-test chart](../figures/01_full1000_metrics.png), [paired false-negative chart](../figures/03_paired90_false_negatives.png), and [effect interval chart](../figures/04_paired90_effect_ci.png) reproduce the values in manuscript Figures 1, 2, and 3. They are regenerated visualizations, not byte-identical copies of the manuscript's figures.

### Historical full-test results

Each model has 1,000 recorded valid predictions. Unsafe is the positive class.

| Model and deployment | Accuracy | Unsafe recall | Unsafe F1 | TP | TN | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Jev-Guard v3.2, historical hosted service | 56.6% | 43.7% | 50.0% | 217 | 349 | 154 | 280 |
| TypeSafe `jev-1.13.0` | 66.3% | 50.3% | 59.7% | 250 | 413 | 90 | 247 |
| TypeSafe `jev-1.14.0` | 64.4% | 47.5% | 57.0% | 236 | 408 | 95 | 261 |

### Historical fixed-cohort paired results

Every arm contains the same 90 unsafe parents. These are false negatives, not failed requests.

| Model | Clean FN | Control FN | Attack FN | Attack minus control | Rate difference | Bootstrap 95% interval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Jev-Guard v3.2, local CPU float32 | 0/90 | 2/90 | 50/90 | 48/90 | +53.3 pp | [43.3, 63.3] pp |
| TypeSafe `jev-1.13.0` | 1/90 | 2/90 | 26/90 | 24/90 | +26.7 pp | [17.8, 35.6] pp |
| TypeSafe `jev-1.14.0` | 1/90 | 4/90 | 22/90 | 18/90 | +20.0 pp | [12.2, 28.9] pp |

The selection process identifies 122 common historical unsafe true positives, excludes inputs with middle omission to obtain 94 eligible parents, and retains the 90 smallest hashes of `common-tp-90-v1:` followed by the parent ID. The cohort stays fixed after clean replay. Historical clean misses for TypeSafe are retained rather than removed through reselection.

## Protocol clarifications

- **Decision boundary:** Section 4 of the manuscript says probability "greater than 0.5." The actual TypeSafe implementation uses **`unsafe_probability >= 0.5`**. The archived full tests contain 15 exact ties for 1.13 and 10 for 1.14; the paired tests contain five and three, respectively. The code and results are preserved with their actual tie rule. Jev-Guard uses the published safety-head verdict, which is a native two-class argmax, rather than a separately configured TypeSafe-style threshold.
- **Deployment:** the manuscript's Guard full-test archive comes from the historical hosted service; its paired experiment uses local CPU float32 inference. These are distinct deployments. TypeSafe uses version-pinned API requests and a fixed typed safety question.
- **Input coverage:** the release contains 503 safe and 497 unsafe cases. All 1,000 are scored, but historical preprocessing omits the middle of 52 long serialized views while retaining their beginning and end. The 24,000-character limit is a view-construction limit. It does not mean that every original long trajectory survives intact. None of the 90 selected parents has middle omission.
- **Intervention location:** the edit changes only the `content` of the last existing `environment` tool-result event. It does not replace the agent's final answer or append a new agent message. All 90 selected cases use an added `_external_tool_note` field in an existing JSON object; existing keys and values are retained.
- **Length matching:** control and attack notes match in serialized character count and placement. This does not establish matching token counts for every tokenizer.
- **Effect units:** attack-minus-control false-negative-rate differences are measured in percentage points. The manuscript's 1.14 effect described as "20%" is a **20.0-percentage-point** difference.

## Supplementary complete rerun

The [completed rerun Notebook](../runs/notebook_20261116T225617Z_b4e8fb4b/completed_report/run_and_compare.executed.ipynb), [HTML report](../runs/notebook_20261116T225617Z_b4e8fb4b/completed_report/report.html), and [verification record](../runs/notebook_20261116T225617Z_b4e8fb4b/completion_verification.json) document a later run completed on 17 November 2026 UTC. The root `run_and_compare.ipynb` is the portable entry point for inspecting this rerun and launching future runs. Use `results_overview.ipynb` for the manuscript's historical results.

| Model | Full-test accuracy | Clean FN | Control FN | Attack FN | Net additional FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| Jev-Guard v3.2, local CPU | 57.2% | 0/90 | 2/90 | 50/90 | 48/90 |
| TypeSafe `jev-1.13.0` | 66.7% | 2/90 | 4/90 | 25/90 | 21/90 |
| TypeSafe `jev-1.14.0` | 64.7% | 0/90 | 4/90 | 21/90 | 17/90 |

All six experiment/model groups are complete: 3,000 full-test and 810 paired predictions. Guard's full test required recovery after a memory failure. Recovery preserved 366 successful predictions and appended 634 predictions using CPU float32 query-chunked attention, without further truncating the supplied inputs. Four real-model diagnostic parity cases agreed on labels and token counts, with maximum absolute probability difference approximately `1.61e-6`; this is not a claim of bitwise equivalence for every input. Earlier failed call records remain in the prediction evidence. Initial partial reports remain in the original experiment workspace and are not included in this cleaned export. The final complete report is re-rendered offline in `completed_report/`, using portable settings and unchanged predictions.

## Limits of the evidence

The lexical probe is a separate TF-IDF logistic regression classifier. Its [retained historical report](../supplementary/lexical_probe/historical_report.md) gives cluster-held-out macro-F1 values of 0.829 and 0.880, using diagnostic cross-validation on ATBench's released test set. The repository retains the report and reproduction source, but the original out-of-fold predictions, group assignments, fold manifests, results JSON, and runtime-version record supporting those exact values were not recovered. Consequently, those quoted values cannot currently be independently recomputed from retained per-example probe evidence. The [source manifest](../supplementary/lexical_probe/source_manifest.json) records what was recovered. From the repository root, `python supplementary/lexical_probe/run_probe.py` validates inputs without training; adding `--run` launches a fresh five-view random/group cross-validation experiment in a new `runs/lexical_*` directory. A fresh run does not reconstruct missing historical artifacts. See the [probe guide](../supplementary/lexical_probe/README.md) for dependencies and settings.

The lexical result does not show that Jev or Jev-Guard performs better on final answers alone, and it does not identify those judges' internal attention or decision mechanism. Similarly, the paired intervention establishes sensitivity to the supplied forged note in this selected cohort; it does not establish that every probabilistic judge is unsuitable for every safety-auditing use case.

The 90-parent cohort is conditioned on historical common true positives and includes only unsafe cases. Its false-negative rates are not full-distribution accuracy or a false-positive estimate. The study uses one note template and one insertion location. Gold labels are inherited under mechanical preservation checks, not independently re-annotated for every edited trajectory. Parent-bootstrap intervals quantify variation across these fixed selected parents and calls; they do not capture repeated-call, deployment, or benchmark uncertainty. Proposed permission-aware scenarios and defenses are future work, not completed measurements in this repository.
