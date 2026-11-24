"""Verify the included evidence and recompute reported results without model calls."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]

from analysis import load_archives, compare_archived_summaries, read_jsonl
from prepare_inputs import reconstruct


def main():
    full, paired = load_archives(ROOT)
    full_states, _, selection = reconstruct(ROOT)
    if full_states != read_jsonl(ROOT / "inputs/full1000_states.jsonl"):
        raise ValueError("Rebuilt full inputs do not match the frozen archive")
    full_metrics, paired_metrics, _ = compare_archived_summaries(ROOT, full, paired)
    report = {
        "status": "passed",
        "model_calls": 0,
        "full_test_valid_predictions": len(full),
        "paired_valid_predictions": len(paired),
        "input_reconstruction": selection,
        "full_test_accuracy": full_metrics.accuracy.to_dict(),
        "paired_false_negatives": paired_metrics[["clean", "control", "attack", "net_extra"]].astype(int).to_dict(orient="index"),
        "archived_summary_and_bootstrap_agreement": True,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
