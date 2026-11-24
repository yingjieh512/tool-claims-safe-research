"""Low-cost TF-IDF baselines and evaluation utilities.

The main entry point, :func:`evaluate_tfidf_baseline`, evaluates every requested
trajectory view under the same random and grouped splits.  Its return value is
JSON-serializable and includes both aggregate metrics and per-record predictions.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import csv
from dataclasses import asdict, is_dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from .splits import (
    SplitResult,
    cross_validation_splits,
    record_value,
    split_records,
)


MetricCallable = Callable[[Sequence[Any], Sequence[Any]], float]


def _to_builtin(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value):
        return _to_builtin(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _ordered_labels(*arrays: Sequence[Any]) -> list[Any]:
    values: dict[tuple[str, str], Any] = {}
    for array in arrays:
        for value in array:
            values.setdefault((type(value).__name__, repr(value)), value)
    return [values[key] for key in sorted(values)]


def _positive_label(labels: Sequence[Any], requested: Any | None = None) -> Any | None:
    ordered = _ordered_labels(labels)
    if requested is not None:
        return requested
    if len(ordered) != 2:
        return None
    for preferred in (1, True, "unsafe", "harmful", "risk", "risky", "positive"):
        for label in ordered:
            if label == preferred or (
                isinstance(label, str) and label.casefold() == str(preferred).casefold()
            ):
                return label
    return ordered[-1]


def classification_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    scores: Sequence[float] | None = None,
    positive_label: Any | None = None,
) -> dict[str, Any]:
    """Compute stable binary or multiclass classification metrics."""

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0:
        raise ValueError("metrics require at least one prediction")

    labels = _ordered_labels(y_true, y_pred)
    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    balanced_accuracy = (
        float(accuracy_score(y_true, y_pred))
        if len(set(y_true)) == 1
        else float(balanced_accuracy_score(y_true, y_pred))
    )
    if len(labels) == 1:
        # sklearn warns for this valid descriptive slice because it cannot infer a
        # second class.  A 1x1 matrix is the faithful representation here; binary
        # error rates below still use the explicitly requested positive label.
        only_label = labels[0]
        matrix = [
            [
                int(
                    sum(
                        true == only_label and pred == only_label
                        for true, pred in zip(y_true, y_pred)
                    )
                )
            ]
        ]
    else:
        matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    result: dict[str, Any] = {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "labels": [_to_builtin(label) for label in labels],
        "confusion_matrix": matrix,
        "per_class": {
            str(label): {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(per_class_f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
    }

    pos_label = _positive_label(labels, positive_label)
    if pos_label is not None:
        positive = np.asarray([value == pos_label for value in y_true], dtype=bool)
        predicted_positive = np.asarray([value == pos_label for value in y_pred], dtype=bool)
        tp = int(np.sum(positive & predicted_positive))
        fn = int(np.sum(positive & ~predicted_positive))
        fp = int(np.sum(~positive & predicted_positive))
        tn = int(np.sum(~positive & ~predicted_positive))
        result.update(
            {
                "positive_label": _to_builtin(pos_label),
                "true_positive": tp,
                "false_negative": fn,
                "false_positive": fp,
                "true_negative": tn,
                "false_negative_rate": fn / (tp + fn) if tp + fn else None,
                "false_positive_rate": fp / (tn + fp) if tn + fp else None,
            }
        )
        if scores is not None:
            if len(scores) != len(y_true):
                raise ValueError("scores must have the same length as y_true")
            binary_true = positive.astype(int)
            if len(set(binary_true)) == 2:
                result["roc_auc"] = float(roc_auc_score(binary_true, scores))
                result["average_precision"] = float(
                    average_precision_score(binary_true, scores)
                )
    return _to_builtin(result)


def _metric_value(
    metric: str | MetricCallable,
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    scores: Sequence[float] | None,
    positive_label: Any | None,
) -> float:
    if callable(metric):
        return float(metric(y_true, y_pred))
    metric = metric.casefold()
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric == "macro_f1":
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    if metric == "weighted_f1":
        return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    metrics = classification_metrics(
        y_true, y_pred, scores=scores, positive_label=positive_label
    )
    if metric not in metrics or metrics[metric] is None:
        return float("nan")
    return float(metrics[metric])


def bootstrap_confidence_interval(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    metric: str | MetricCallable = "macro_f1",
    scores: Sequence[float] | None = None,
    positive_label: Any | None = None,
    groups: Sequence[Any] | None = None,
    n_resamples: int = 1_000,
    confidence: float = 0.95,
    seed: int = 42,
    stratified: bool = True,
) -> dict[str, Any]:
    """Return a deterministic percentile bootstrap confidence interval.

    Passing ``groups`` performs a cluster bootstrap, which is preferable when
    several examples share a task or template.  Otherwise, class-stratified
    item resampling is used by default.
    """

    if len(y_true) != len(y_pred) or (scores is not None and len(scores) != len(y_true)):
        raise ValueError("all prediction arrays must have the same length")
    if groups is not None and len(groups) != len(y_true):
        raise ValueError("groups must have the same length as y_true")
    if len(y_true) == 0:
        raise ValueError("bootstrap requires at least one prediction")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    true_array = np.asarray(y_true, dtype=object)
    pred_array = np.asarray(y_pred, dtype=object)
    score_array = np.asarray(scores, dtype=float) if scores is not None else None
    rng = np.random.default_rng(seed)

    if groups is not None:
        group_to_indices: dict[str, list[int]] = {}
        for index, group in enumerate(groups):
            group_to_indices.setdefault(str(group), []).append(index)
        group_keys = np.asarray(list(group_to_indices), dtype=object)
        sampling_units = "groups"
    else:
        class_to_indices: dict[tuple[str, str], list[int]] = {}
        if stratified:
            for index, label in enumerate(y_true):
                class_to_indices.setdefault((type(label).__name__, repr(label)), []).append(index)
        sampling_units = "stratified_items" if stratified else "items"

    estimates: list[float] = []
    for _ in range(n_resamples):
        if groups is not None:
            sampled_groups = rng.choice(group_keys, size=len(group_keys), replace=True)
            sample_indices = np.asarray(
                [
                    index
                    for group in sampled_groups
                    for index in group_to_indices[str(group)]
                ],
                dtype=int,
            )
        elif stratified:
            pieces = [
                rng.choice(indices, size=len(indices), replace=True)
                for indices in class_to_indices.values()
            ]
            sample_indices = np.concatenate(pieces).astype(int)
            rng.shuffle(sample_indices)
        else:
            sample_indices = rng.integers(0, len(true_array), size=len(true_array))

        sampled_scores = score_array[sample_indices] if score_array is not None else None
        estimate = _metric_value(
            metric,
            true_array[sample_indices].tolist(),
            pred_array[sample_indices].tolist(),
            scores=sampled_scores.tolist() if sampled_scores is not None else None,
            positive_label=positive_label,
        )
        if math.isfinite(estimate):
            estimates.append(estimate)

    if not estimates:
        raise ValueError("metric was undefined for every bootstrap sample")
    alpha = (1.0 - confidence) / 2.0
    point = _metric_value(
        metric,
        y_true,
        y_pred,
        scores=scores,
        positive_label=positive_label,
    )
    return {
        "metric": metric if isinstance(metric, str) else getattr(metric, "__name__", "callable"),
        "estimate": point,
        "lower": float(np.quantile(estimates, alpha)),
        "upper": float(np.quantile(estimates, 1.0 - alpha)),
        "confidence": confidence,
        "n_resamples": n_resamples,
        "n_valid_resamples": len(estimates),
        "sampling_units": sampling_units,
        "seed": seed,
    }


def _default_view_builder(record: Any, view_name: str) -> str:
    from .views import build_view

    return build_view(record, view_name)


def _record_identifier(record: Any, fallback: int) -> str:
    for field in ("record_id", "sample_id", "example_id", "id"):
        value = record_value(record, field, None)
        if value is not None and str(value).strip():
            return str(value)
    return str(fallback)


def _prepare_view_texts(
    records: Sequence[Any],
    *,
    view_names: Sequence[str] | None,
    view_texts: Mapping[str, Sequence[str]] | None,
    view_builder: Callable[[Any, str], str] | None,
) -> tuple[tuple[str, ...], dict[str, list[str]]]:
    if view_texts is not None:
        names = tuple(view_names or view_texts.keys())
        prepared: dict[str, list[str]] = {}
        for name in names:
            if name not in view_texts:
                raise KeyError(f"view_texts does not contain {name!r}")
            if len(view_texts[name]) != len(records):
                raise ValueError(f"view {name!r} has a different length than records")
            prepared[name] = [str(text) if text is not None else "" for text in view_texts[name]]
        return names, prepared

    if view_names is None:
        try:
            from .views import VIEW_NAMES

            names = tuple(VIEW_NAMES)
        except (ImportError, AttributeError):
            names = ("full",)
    else:
        names = tuple(view_names)
    builder = view_builder or _default_view_builder
    prepared = {
        name: [str(builder(record, name) or "") for record in records]
        for name in names
    }
    return names, prepared


def _fit_text_model(
    train_texts: Sequence[str],
    train_labels: Sequence[Any],
    *,
    max_features: int | None,
    ngram_range: tuple[int, int],
    class_weight: str | Mapping[Any, float] | None,
    seed: int,
) -> Any:
    clean_texts = [text if text.strip() else "[EMPTY]" for text in train_texts]
    if len(set(train_labels)) < 2:
        model = DummyClassifier(strategy="most_frequent")
        model.fit(clean_texts, train_labels)
        return model
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=ngram_range,
                    min_df=1,
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight=class_weight,
                    max_iter=1_000,
                    random_state=seed,
                    solver="liblinear",
                ),
            ),
        ]
    )
    model.fit(clean_texts, train_labels)
    return model


def evaluate_tfidf_baseline(
    records: Sequence[Any],
    *,
    view_names: Sequence[str] | None = None,
    view_texts: Mapping[str, Sequence[str]] | None = None,
    view_builder: Callable[[Any, str], str] | None = None,
    split_strategies: Sequence[str] = ("random", "group"),
    label_field: str = "label",
    group_fields: str | Sequence[str] | None = None,
    group_values: Sequence[Any] | None = None,
    precomputed_splits: Mapping[str, SplitResult] | None = None,
    test_size: float | int = 0.2,
    seed: int = 42,
    positive_label: Any | None = None,
    max_features: int | None = 50_000,
    ngram_range: tuple[int, int] = (1, 2),
    class_weight: str | Mapping[Any, float] | None = "balanced",
    bootstrap_resamples: int = 0,
) -> dict[str, Any]:
    """Evaluate all trajectory views with deterministic TF-IDF classifiers.

    Unlabelled records are skipped, while prediction rows retain their original
    input indices.  A single split is reused across views within each strategy,
    making partial-view comparisons paired and directly interpretable.
    """

    records = tuple(records)
    if not records:
        raise ValueError("records must not be empty")
    if group_values is not None and len(group_values) != len(records):
        raise ValueError("group_values must have the same length as records")
    names, all_view_texts = _prepare_view_texts(
        records,
        view_names=view_names,
        view_texts=view_texts,
        view_builder=view_builder,
    )

    original_indices: list[int] = []
    labeled_records: list[Any] = []
    labels: list[Any] = []
    for index, record in enumerate(records):
        label = record_value(record, label_field, None)
        if label is None:
            continue
        original_indices.append(index)
        labeled_records.append(record)
        labels.append(label)
    if len(labeled_records) < 2:
        raise ValueError("at least two labelled records are required")

    labeled_views = {
        name: [all_view_texts[name][index] for index in original_indices]
        for name in names
    }
    record_ids = [
        _record_identifier(record, original_index)
        for record, original_index in zip(labeled_records, original_indices)
    ]
    labeled_group_values = (
        [group_values[index] for index in original_indices]
        if group_values is not None
        else None
    )

    result: dict[str, Any] = {
        "schema_version": 1,
        "model": "tfidf_logistic_regression",
        "config": {
            "seed": seed,
            "test_size": test_size,
            "label_field": label_field,
            "group_fields": group_fields,
            "external_group_values": group_values is not None,
            "precomputed_splits": precomputed_splits is not None,
            "view_names": list(names),
            "split_strategies": list(split_strategies),
            "max_features": max_features,
            "ngram_range": list(ngram_range),
            "class_weight": class_weight,
        },
        "dataset": {
            "input_records": len(records),
            "labelled_records": len(labeled_records),
            "skipped_unlabelled": len(records) - len(labeled_records),
            "label_counts": {
                str(label): count for label, count in Counter(labels).items()
            },
        },
        "splits": {},
    }

    if precomputed_splits is not None:
        split_items = list(precomputed_splits.items())
        if not split_items:
            raise ValueError("precomputed_splits must not be empty")
    else:
        split_items = [
            (
                strategy,
                split_records(
                    labeled_records,
                    strategy=strategy,
                    test_size=test_size,
                    seed=seed,
                    label_field=label_field,
                    group_fields=group_fields,
                    group_values=labeled_group_values,
                    stratify=True,
                ),
            )
            for strategy in split_strategies
        ]

    for split_name, split in split_items:
        all_split_indices = split.train_indices + split.test_indices
        if not all_split_indices or max(all_split_indices) >= len(labeled_records):
            raise ValueError(f"precomputed split {split_name!r} has invalid record indices")
        split_payload = split.to_dict()
        split_payload["train_original_indices"] = [
            original_indices[index] for index in split.train_indices
        ]
        split_payload["test_original_indices"] = [
            original_indices[index] for index in split.test_indices
        ]
        split_payload["views"] = {}
        split_group_by_index = {
            **dict(zip(split.train_indices, split.train_groups)),
            **dict(zip(split.test_indices, split.test_groups)),
        }

        y_train = [labels[index] for index in split.train_indices]
        y_test = [labels[index] for index in split.test_indices]
        for view_name in names:
            texts = labeled_views[view_name]
            x_train = [texts[index] for index in split.train_indices]
            x_test = [texts[index] if texts[index].strip() else "[EMPTY]" for index in split.test_indices]
            model = _fit_text_model(
                x_train,
                y_train,
                max_features=max_features,
                ngram_range=ngram_range,
                class_weight=class_weight,
                seed=seed,
            )
            predictions = model.predict(x_test).tolist()
            probabilities = model.predict_proba(x_test) if hasattr(model, "predict_proba") else None
            classes = list(model.classes_)
            pos_label = _positive_label(_ordered_labels(labels), positive_label)
            positive_scores: list[float] | None = None
            if probabilities is not None and pos_label is not None and pos_label in classes:
                pos_index = classes.index(pos_label)
                positive_scores = probabilities[:, pos_index].astype(float).tolist()

            metrics = classification_metrics(
                y_test,
                predictions,
                scores=positive_scores,
                positive_label=positive_label,
            )
            prediction_rows: list[dict[str, Any]] = []
            for position, local_index in enumerate(split.test_indices):
                probability_map = (
                    {
                        str(label): float(probabilities[position, class_index])
                        for class_index, label in enumerate(classes)
                    }
                    if probabilities is not None
                    else {}
                )
                prediction_rows.append(
                    {
                        "split": split_name,
                        "view": view_name,
                        "record_index": original_indices[local_index],
                        "record_id": record_ids[local_index],
                        "group": split_group_by_index[local_index],
                        "y_true": _to_builtin(y_test[position]),
                        "y_pred": _to_builtin(predictions[position]),
                        "positive_score": (
                            positive_scores[position] if positive_scores is not None else None
                        ),
                        "confidence": (
                            float(np.max(probabilities[position]))
                            if probabilities is not None
                            else None
                        ),
                        "probabilities": probability_map,
                        "correct": bool(y_test[position] == predictions[position]),
                    }
                )

            view_payload: dict[str, Any] = {
                "metrics": metrics,
                "predictions": prediction_rows,
                "model_kind": (
                    "dummy_most_frequent"
                    if isinstance(model, DummyClassifier)
                    else "tfidf_logistic_regression"
                ),
            }
            if bootstrap_resamples:
                test_groups = [
                    split_group_by_index[index] for index in split.test_indices
                ]
                view_payload["bootstrap"] = {
                    "macro_f1": bootstrap_confidence_interval(
                        y_test,
                        predictions,
                        metric="macro_f1",
                        groups=test_groups,
                        n_resamples=bootstrap_resamples,
                        seed=seed,
                    ),
                    "accuracy": bootstrap_confidence_interval(
                        y_test,
                        predictions,
                        metric="accuracy",
                        groups=test_groups,
                        n_resamples=bootstrap_resamples,
                        seed=seed,
                    ),
                }
            split_payload["views"][view_name] = view_payload
        result["splits"][split_name] = split_payload
    return _to_builtin(result)


def evaluate_tfidf_cross_validation(
    records: Sequence[Any],
    *,
    n_splits: int = 5,
    split_strategies: Sequence[str] = ("random", "group"),
    label_field: str = "label",
    group_fields: str | Sequence[str] | None = None,
    group_values: Sequence[Any] | None = None,
    seed: int = 42,
    **evaluation_kwargs: Any,
) -> dict[str, Any]:
    """Evaluate TF-IDF views over named out-of-fold predictions.

    The returned ``splits`` mapping uses keys such as ``group_fold_0``.  This
    keeps each fold self-contained while allowing :func:`flatten_predictions`
    to concatenate all out-of-fold rows without special handling.
    """

    records = tuple(records)
    if group_values is not None and len(group_values) != len(records):
        raise ValueError("group_values must have the same length as records")
    labelled_positions = [
        index
        for index, record in enumerate(records)
        if record_value(record, label_field, None) is not None
    ]
    labelled_records = [records[index] for index in labelled_positions]
    labelled_group_values = (
        [group_values[index] for index in labelled_positions]
        if group_values is not None
        else None
    )
    named_folds: dict[str, SplitResult] = {}
    for strategy in split_strategies:
        folds = cross_validation_splits(
            labelled_records,
            strategy=strategy,
            n_splits=n_splits,
            seed=seed,
            label_field=label_field,
            group_fields=group_fields,
            group_values=labelled_group_values,
            stratify=True,
        )
        normalized = folds[0].strategy
        named_folds.update(
            {f"{normalized}_fold_{fold.fold_index}": fold for fold in folds}
        )

    results = evaluate_tfidf_baseline(
        records,
        split_strategies=split_strategies,
        label_field=label_field,
        group_fields=group_fields,
        group_values=group_values,
        precomputed_splits=named_folds,
        seed=seed,
        **evaluation_kwargs,
    )
    results["config"]["n_splits"] = n_splits
    results["config"]["cross_validation"] = True
    results["config"]["test_size"] = 1.0 / n_splits
    return results


def flatten_predictions(results: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten nested view/split predictions into CSV-friendly dictionaries."""

    rows: list[dict[str, Any]] = []
    for split_name, split_payload in results.get("splits", {}).items():
        for view_name, view_payload in split_payload.get("views", {}).items():
            for prediction in view_payload.get("predictions", []):
                row = dict(prediction)
                row.setdefault("split", split_name)
                row.setdefault("view", view_name)
                if isinstance(row.get("probabilities"), Mapping):
                    row["probabilities"] = json.dumps(
                        row["probabilities"], sort_keys=True, ensure_ascii=False
                    )
                rows.append(_to_builtin(row))
    return rows


def save_results(results: Mapping[str, Any], path: str | Path) -> Path:
    """Write the complete evaluation result to a JSON file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(_to_builtin(results), handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    return output_path


def save_predictions_csv(results: Mapping[str, Any], path: str | Path) -> Path:
    """Write all per-record predictions to one CSV file."""

    rows = flatten_predictions(results)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    default_fields = [
        "split",
        "view",
        "record_index",
        "record_id",
        "group",
        "y_true",
        "y_pred",
        "positive_score",
        "confidence",
        "probabilities",
        "correct",
    ]
    extra_fields = sorted({key for row in rows for key in row} - set(default_fields))
    fieldnames = default_fields + extra_fields
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def save_evaluation_artifacts(
    results: Mapping[str, Any], output_dir: str | Path
) -> dict[str, Path]:
    """Save ``results.json`` and ``predictions.csv`` beneath ``output_dir``."""

    directory = Path(output_dir)
    return {
        "results": save_results(results, directory / "results.json"),
        "predictions": save_predictions_csv(results, directory / "predictions.csv"),
    }


__all__ = [
    "bootstrap_confidence_interval",
    "classification_metrics",
    "evaluate_tfidf_baseline",
    "evaluate_tfidf_cross_validation",
    "flatten_predictions",
    "save_evaluation_artifacts",
    "save_predictions_csv",
    "save_results",
]
