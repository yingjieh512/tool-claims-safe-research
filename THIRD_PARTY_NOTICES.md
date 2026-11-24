# Third-party notices and evidence provenance

The root MIT license retains the source project's notice. It does not relicense third-party models, datasets, API services, or the manuscript.

- **ATBench:** the supplied 1,000-example test data and labels are preserved with source provenance in `provenance/source_manifest.json`. Upstream dataset terms continue to apply. No new license over the benchmark is asserted here.
- **Jev-Guard v3.2:** upstream inference source, model cards, source hashes, and release notices are retained in `code/guard/`. See `code/guard/LICENSE-NOTICE.md` and `code/guard/runtime_provenance.json`. The repository includes the study's serial and bounded-attention serving additions separately from the unchanged upstream implementation.
- **Qwen base model:** the original license is retained as `code/guard/QWEN-LICENSE`. Weights are downloaded separately using revision-pinned setup code and are not uploaded in this repository.
- **TypeSafe:** versioned API responses are experimental observations. TypeSafe's service terms and access requirements remain applicable; no API credential or hosted model weights are included.
- **Lexical probe:** source files and their inherited license/attribution are described in `supplementary/lexical_probe/`.
- **Manuscript:** `paper/Capstone.pdf` is the unmodified author-supplied paper. It is included to establish the relationship between reported claims and supporting artifacts, not as a revised paper.

The original full Guard test used a hosted deployment; the original paired test and subsequent complete Guard rerun used local CPU deployments. These deployment distinctions are part of the provenance and must be retained when reusing the results.
