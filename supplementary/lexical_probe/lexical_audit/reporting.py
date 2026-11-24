"""Aggregate cross-validation outputs and create lightweight audit reports."""

from __future__ import annotations

from collections import defaultdict
from html import escape
import json
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import f1_score

from .baseline import bootstrap_confidence_interval, classification_metrics


def _strategy_name(split_name: str) -> str:
    return split_name.rsplit("_fold_", 1)[0]


def _paired_cluster_bootstrap_macro_f1(
    y_true: Sequence[int],
    first: Sequence[int],
    second: Sequence[int],
    groups: Sequence[str],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap a paired Macro-F1 difference using identical group draws."""

    group_indices: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        group_indices[str(group)].append(index)
    group_keys = np.asarray(list(group_indices), dtype=object)
    true_array = np.asarray(y_true)
    first_array = np.asarray(first)
    second_array = np.asarray(second)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_resamples):
        sampled = rng.choice(group_keys, size=len(group_keys), replace=True)
        indices = np.asarray(
            [index for group in sampled for index in group_indices[str(group)]],
            dtype=int,
        )
        estimates.append(
            float(
                f1_score(true_array[indices], first_array[indices], average="macro", zero_division=0)
                - f1_score(true_array[indices], second_array[indices], average="macro", zero_division=0)
            )
        )
    point = float(
        f1_score(true_array, first_array, average="macro", zero_division=0)
        - f1_score(true_array, second_array, average="macro", zero_division=0)
    )
    return {
        "metric": "macro_f1_difference",
        "estimate": point,
        "lower": float(np.quantile(estimates, 0.025)),
        "upper": float(np.quantile(estimates, 0.975)),
        "confidence": 0.95,
        "n_resamples": n_resamples,
        "sampling_units": "fixed_inferred_clusters",
        "seed": seed,
    }


def aggregate_cross_validation(
    results: Mapping[str, Any],
    *,
    semantic_groups: Sequence[str] | None = None,
    bootstrap_resamples: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    """Aggregate out-of-fold predictions once per record, strategy, and view."""

    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    fold_metrics: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for split_name, split in results.get("splits", {}).items():
        strategy = _strategy_name(str(split_name))
        for view, payload in split.get("views", {}).items():
            buckets[(strategy, str(view))].extend(payload.get("predictions", []))
            fold_metrics[(strategy, str(view))].append(payload.get("metrics", {}))

    summary: dict[str, Any] = {
        "schema_version": 1,
        "bootstrap_resamples": bootstrap_resamples,
        "strategies": {},
    }
    expected_records = int(results.get("dataset", {}).get("labelled_records", 0))
    prepared_predictions: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for (strategy, view), predictions in sorted(buckets.items()):
        predictions = sorted(predictions, key=lambda row: int(row["record_index"]))
        indices = [int(row["record_index"]) for row in predictions]
        if len(indices) != len(set(indices)):
            raise ValueError(f"duplicate out-of-fold predictions for {strategy}/{view}")
        if expected_records and len(indices) != expected_records:
            raise ValueError(
                f"incomplete out-of-fold coverage for {strategy}/{view}: "
                f"expected {expected_records}, found {len(indices)}"
            )
        prepared_predictions[(strategy, view)] = predictions
        y_true = [row["y_true"] for row in predictions]
        y_pred = [row["y_pred"] for row in predictions]
        scores = [row.get("positive_score") for row in predictions]
        score_values = None if any(value is None for value in scores) else scores
        metrics = classification_metrics(y_true, y_pred, scores=score_values)

        if semantic_groups is not None:
            groups = [semantic_groups[index] for index in indices]
        else:
            groups = [str(row.get("group", index)) for index, row in zip(indices, predictions)]

        confidence_intervals: dict[str, Any] = {}
        if bootstrap_resamples:
            for metric in ("macro_f1", "accuracy", "false_negative_rate"):
                confidence_intervals[metric] = bootstrap_confidence_interval(
                    y_true,
                    y_pred,
                    metric=metric,
                    scores=score_values,
                    groups=groups,
                    n_resamples=bootstrap_resamples,
                    seed=seed,
                )

        per_fold = fold_metrics[(strategy, view)]
        fold_summary = {}
        for metric in ("macro_f1", "accuracy", "roc_auc", "false_negative_rate"):
            values = [float(item[metric]) for item in per_fold if item.get(metric) is not None]
            if values:
                fold_summary[metric] = {
                    "mean": statistics.fmean(values),
                    "standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "folds": len(values),
                }
        summary["strategies"].setdefault(strategy, {"views": {}})["views"][view] = {
            "metrics": metrics,
            "confidence_intervals": confidence_intervals,
            "fold_summary": fold_summary,
            "records": len(predictions),
        }

    for strategy_payload in summary["strategies"].values():
        views = strategy_payload["views"]
        full_score = views.get("full", {}).get("metrics", {}).get("macro_f1")
        for view_payload in views.values():
            score = view_payload["metrics"].get("macro_f1")
            view_payload["macro_f1_retention_vs_full"] = (
                score / full_score if score is not None and full_score else None
            )
    if bootstrap_resamples:
        for strategy, strategy_payload in summary["strategies"].items():
            full_rows = prepared_predictions.get((strategy, "full"))
            final_rows = prepared_predictions.get((strategy, "final_answer_only"))
            if not full_rows or not final_rows:
                continue
            full_by_index = {int(row["record_index"]): row for row in full_rows}
            final_by_index = {int(row["record_index"]): row for row in final_rows}
            if set(full_by_index) != set(final_by_index):
                raise ValueError(f"full/final coverage differs for {strategy}")
            ordered_indices = sorted(full_by_index)
            y_true = [int(full_by_index[index]["y_true"]) for index in ordered_indices]
            if any(
                int(final_by_index[index]["y_true"]) != truth
                for index, truth in zip(ordered_indices, y_true)
            ):
                raise ValueError(f"full/final labels differ for {strategy}")
            groups = (
                [semantic_groups[index] for index in ordered_indices]
                if semantic_groups is not None
                else [str(full_by_index[index]["group"]) for index in ordered_indices]
            )
            strategy_payload["paired_differences"] = {
                "final_answer_only_minus_full": _paired_cluster_bootstrap_macro_f1(
                    y_true,
                    [int(final_by_index[index]["y_pred"]) for index in ordered_indices],
                    [int(full_by_index[index]["y_pred"]) for index in ordered_indices],
                    groups,
                    n_resamples=bootstrap_resamples,
                    seed=seed,
                )
            }
    return summary

