# Evidence and reproducibility records

The committee package separates preserved evidence from its presentation and execution wrappers.

| Record | Meaning |
| --- | --- |
| `source_manifest.json` | Source repository reference, per-file original locations and hashes, and the current locations of byte-preserved evidence/core modules |
| `analysis_environment.json` | Python and analysis library versions used for the validated analysis |
| `input_validation.json` | Counts and exact-match input reconstruction checks |
| `code_validation.json` | Verification of the regrouped code, six replay configurations, and response parsing |
| `package_validation.json` | Final notebook execution, evidence preservation, and package checks |
| `bundle_sha256.json` | Hashes of the delivered package's files, excluding this hash inventory itself |
| `SOURCE_LICENSE` | Original project source license |

The source commit identifies the repository baseline. The manifest also records the source worktree's modified-file state and relevant script hashes; it does not claim that the commit alone contains every experimental wrapper exactly as used. The copied data, prediction archives, selection/edit records, and trajectory core modules are identified by file hash.

The review edition changes file organization and reader-facing explanations. It does not change archived predictions, their input prompts, model labels, or cohort membership. Local Guard runtime provenance and upstream model licenses are under `code/guard/`.

`python -B code/verify_package.py` checks the immutable evidence and recomputes the results. Re-running the notebook may change its execution metadata and exported figures, so the delivery-time `bundle_sha256.json` is an inventory of the delivered snapshot, not a claim that editable presentation files will retain those hashes after further work.

No new model requests or inference are needed for any offline verification. Live replay writes a new run rather than replacing this evidence.
