"""Validate or rerun the exploratory lexical probe using the original algorithms."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_DATA_SHA256 = "80c534b5f3517c872b528a2e6b64e34495206968edd184f4aed563d39f0cca09"
VIEWS = ("full", "user_only", "tool_schema_only", "action_only", "final_answer_only")


def write_json(path, payload):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Train fresh lexical models; otherwise validate inputs only.")
    parser.add_argument("--input", type=Path, default=ROOT / "data/atbench_test.json")
    args = parser.parse_args()
    raw = args.input.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_DATA_SHA256:
        raise ValueError("The input differs from the original pinned ATBench release.")
    rows = json.loads(raw)
    if not isinstance(rows, list) or len(rows) != 1000:
        raise ValueError("Expected the original 1,000-row ATBench dataset.")
    config = {
        "status": "input_validated_no_model_training",
        "data_sha256": digest,
        "records": len(rows),
        "views": VIEWS,
        "seed": 42,
        "folds": 5,
        "groups": 50,
        "max_features": 50000,
        "bootstrap_resamples": 500,
        "historical_oof_archive_available": False,
    }
    if not args.run:
        print(json.dumps(config, indent=2))
        return 0
    if sys.version_info < (3, 10):
        raise RuntimeError("Fresh lexical training requires Python 3.10 or newer.")
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
    from lexical_audit.baseline import evaluate_tfidf_cross_validation, save_evaluation_artifacts
    from lexical_audit.data import load_records
    from lexical_audit.grouping import derive_semantic_groups
    from lexical_audit.reporting import aggregate_cross_validation
    from lexical_audit.views import build_views

    records = load_records(args.input)
    if Counter(record.label for record in records) != {0: 503, 1: 497}:
        raise ValueError("Unexpected label distribution after normalization.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / "runs" / ("lexical_" + stamp + "_" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", {
        **config,
        "status": "fresh_replication_started",
        "python": platform.python_version(),
        "dependencies": {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl")},
        "historical_environment_versions_available": False,
    })
    print("Deriving 50 label-free user/tool clusters ...", flush=True)
    groups, _, _ = derive_semantic_groups([dict(record.raw) for record in records], n_groups=50, random_state=42)
    write_json(output / "group_assignments.json", [
        {"record_index": index, "group": group} for index, group in enumerate(groups)
    ])
    rendered = [build_views(record) for record in records]
    texts = {view: [item[view] for item in rendered] for view in VIEWS}
    print("Training five-view, five-fold random and group-held-out lexical probes ...", flush=True)
    results = evaluate_tfidf_cross_validation(
        records, n_splits=5, split_strategies=("random", "group"),
        group_values=groups, seed=42, view_names=VIEWS, view_texts=texts,
        max_features=50000, bootstrap_resamples=0,
    )
    save_evaluation_artifacts(results, output)
    summary = aggregate_cross_validation(results, semantic_groups=groups, bootstrap_resamples=500, seed=42)
    summary["provenance_note"] = "Fresh replication; not the missing historical out-of-fold archive."
    write_json(output / "summary.json", summary)
    print(json.dumps({"status": "fresh_replication_complete", "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
