"""Deterministic train/test splits for trajectory-audit experiments.

The grouped split deliberately keeps every inferred task/template/tool group on
only one side of the split.  The helpers accept either mapping-like records or
the ``TrajectoryRecord`` dataclass used by :mod:`trajectory_audit.data`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
import json
import math
from typing import Any

import numpy as np
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)


DEFAULT_GROUP_FIELDS: tuple[str, ...] = (
    "group_id",
    "template_id",
    "template_name",
    "template",
    "task_id",
    "task_name",
    "task",
    "scenario_id",
    "scenario_name",
    "scenario",
    "tool_id",
    "tool_name",
    "tool_names",
    "tools",
)

_CONTENT_FALLBACK_FIELDS: tuple[str, ...] = (
    "user_request",
    "user_input",
    "instruction",
    "prompt",
    "query",
    "request",
)

_ID_FALLBACK_FIELDS: tuple[str, ...] = (
    "record_id",
    "sample_id",
    "example_id",
    "id",
)

_MISSING = object()


def _record_sources(record: Any) -> list[Any]:
    """Return record containers in lookup priority order."""

    sources = [record]
    for name in ("metadata", "raw"):
        if isinstance(record, Mapping):
            nested = record.get(name)
        else:
            nested = getattr(record, name, None)
        if isinstance(nested, Mapping):
            sources.append(nested)
    return sources


def record_value(record: Any, key: str, default: Any = None) -> Any:
    """Read a field from a dict/dataclass, then from metadata and raw data."""

    for source in _record_sources(record):
        if isinstance(source, Mapping):
            if key in source:
                return source[key]
        elif hasattr(source, key):
            return getattr(source, key)
    return default


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_jsonable(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True, default=str))
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _canonical(value: Any) -> str:
    if isinstance(value, str):
        return " ".join(value.split()).casefold()
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _present(value: Any) -> bool:
    if value is _MISSING or value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (Sequence, Mapping, set, frozenset)):
        return bool(value)
    return True


def _digest(value: Any) -> str:
    return sha256(_canonical(value).encode("utf-8")).hexdigest()[:20]


def infer_group_key(
    record: Any,
    *,
    group_fields: str | Sequence[str] | None = None,
    record_index: int | None = None,
) -> str:
    """Infer a stable leakage-control group for a record.

    With explicit ``group_fields``, all available fields form a composite key.
    Otherwise the first useful task/template/scenario/tool field is used.  If
    none is present, identical user requests are grouped by a content digest;
    record identifiers and finally a digest of the record provide safe fallbacks.
    Labels are excluded from the final-record digest so they cannot define groups.
    """

    if isinstance(group_fields, str):
        requested = (group_fields,)
    elif group_fields is None:
        requested = DEFAULT_GROUP_FIELDS
    else:
        requested = tuple(group_fields)

    found: list[tuple[str, Any]] = []
    for field in requested:
        value = record_value(record, field, _MISSING)
        if _present(value):
            found.append((field, value))
            if group_fields is None:
                break
    if found:
        return "field:" + "|".join(f"{name}={_canonical(value)}" for name, value in found)

    for field in _CONTENT_FALLBACK_FIELDS:
        value = record_value(record, field, _MISSING)
        if _present(value):
            return f"content:{field}:{_digest(value)}"

    for field in _ID_FALLBACK_FIELDS:
        value = record_value(record, field, _MISSING)
        if _present(value):
            return f"id:{field}:{_canonical(value)}"

    if is_dataclass(record):
        payload = asdict(record)
    elif isinstance(record, Mapping):
        payload = dict(record)
    else:
        payload = vars(record) if hasattr(record, "__dict__") else str(record)
    if isinstance(payload, Mapping):
        ignored = {"label", "target", "y", "prediction", "score"}
        payload = {key: value for key, value in payload.items() if str(key) not in ignored}
    if _present(payload):
        return f"record:{_digest(payload)}"
    return f"index:{record_index if record_index is not None else 0}"


@dataclass(frozen=True)
class SplitResult:
    """A materialized split plus the indices and groups used to create it."""

    train: tuple[Any, ...]
    test: tuple[Any, ...]
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    train_groups: tuple[str, ...]
    test_groups: tuple[str, ...]
    strategy: str
    seed: int
    requested_test_size: float | int
    fold_index: int | None = None

    @property
    def actual_test_fraction(self) -> float:
        total = len(self.train_indices) + len(self.test_indices)
        return len(self.test_indices) / total if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "seed": self.seed,
            "requested_test_size": self.requested_test_size,
            "fold_index": self.fold_index,
            "actual_test_fraction": self.actual_test_fraction,
            "train_size": len(self.train_indices),
            "test_size": len(self.test_indices),
            "train_indices": list(self.train_indices),
            "test_indices": list(self.test_indices),
            "train_group_count": len(set(self.train_groups)),
            "test_group_count": len(set(self.test_groups)),
        }


def _target_test_count(n_records: int, test_size: float | int) -> int:
    if isinstance(test_size, bool):
        raise TypeError("test_size must be a float fraction or an integer count")
    if isinstance(test_size, int):
        target = test_size
    elif isinstance(test_size, float) and 0.0 < test_size < 1.0:
        target = math.ceil(n_records * test_size)
    else:
        raise ValueError("float test_size must be between 0 and 1")
    if target < 1 or target >= n_records:
        raise ValueError("test_size must leave at least one train and one test record")
    return target


def _can_stratify(labels: Sequence[Any], target_test: int) -> bool:
    counts = Counter(labels)
    n_classes = len(counts)
    return (
        n_classes > 1
        and min(counts.values()) >= 2
        and target_test >= n_classes
        and len(labels) - target_test >= n_classes
    )


def _group_candidate_score(
    test_indices: np.ndarray,
    *,
    labels: Sequence[Any] | None,
    target_test: int,
    n_records: int,
    stratify: bool,
) -> float:
    size_error = abs(len(test_indices) - target_test) / n_records
    if not stratify or labels is None or len(set(labels)) < 2:
        return size_error

    overall = Counter(labels)
    held_out = Counter(labels[index] for index in test_indices)
    train = overall - held_out
    distribution_error = sum(
        abs(held_out[label] / len(test_indices) - count / n_records)
        for label, count in overall.items()
    )
    missing_penalty = sum(
        1.0 for label in overall if held_out[label] == 0 or train[label] == 0
    )
    return size_error + distribution_error + missing_penalty


def split_records(
    records: Sequence[Any],
    *,
    strategy: str = "random",
    test_size: float | int = 0.2,
    seed: int = 42,
    label_field: str = "label",
    group_fields: str | Sequence[str] | None = None,
    group_values: Sequence[Any] | None = None,
    stratify: bool = True,
) -> SplitResult:
    """Split records randomly or by non-overlapping inferred groups.

    Group splitting samples several deterministic candidates and selects the one
    closest to the requested size and, when possible, the overall label mix.
    This improves small-benchmark stability without ever splitting a group.
    """

    records = tuple(records)
    if len(records) < 2:
        raise ValueError("at least two records are required")
    target_test = _target_test_count(len(records), test_size)
    indices = np.arange(len(records))
    raw_labels = [record_value(record, label_field, _MISSING) for record in records]
    have_labels = all(label is not _MISSING and label is not None for label in raw_labels)
    labels: list[Any] | None = raw_labels if have_labels else None
    if group_values is not None:
        if len(group_values) != len(records):
            raise ValueError("group_values must have the same length as records")
        groups = tuple(f"provided:{_canonical(value)}" for value in group_values)
    else:
        groups = tuple(
            infer_group_key(record, group_fields=group_fields, record_index=index)
            for index, record in enumerate(records)
        )

    strategy = strategy.casefold().replace("-", "_")
    if strategy in {"random", "stratified_random"}:
        stratify_labels = (
            labels
            if stratify and labels is not None and _can_stratify(labels, target_test)
            else None
        )
        train_idx, test_idx = train_test_split(
            indices,
            test_size=target_test,
            random_state=seed,
            shuffle=True,
            stratify=stratify_labels,
        )
        normalized_strategy = "random"
    elif strategy in {"group", "grouped", "group_holdout"}:
        unique_groups = set(groups)
        if len(unique_groups) < 2:
            raise ValueError(
                "grouped split requires at least two distinct groups; "
                "provide a more specific group field"
            )
        fraction = target_test / len(records)
        splitter = GroupShuffleSplit(
            n_splits=min(128, max(24, len(unique_groups) * 6)),
            test_size=fraction,
            random_state=seed,
        )
        candidates: list[tuple[float, tuple[int, ...], tuple[int, ...]]] = []
        for train_candidate, test_candidate in splitter.split(indices, labels, groups):
            if not len(train_candidate) or not len(test_candidate):
                continue
            score = _group_candidate_score(
                test_candidate,
                labels=labels,
                target_test=target_test,
                n_records=len(records),
                stratify=stratify,
            )
            # The index tuples provide deterministic tie-breaking.
            candidates.append(
                (
                    score,
                    tuple(sorted(int(index) for index in train_candidate)),
                    tuple(sorted(int(index) for index in test_candidate)),
                )
            )
        if not candidates:
            raise ValueError("unable to construct a non-empty grouped split")
        _, train_tuple, test_tuple = min(candidates, key=lambda item: item)
        train_idx = np.asarray(train_tuple, dtype=int)
        test_idx = np.asarray(test_tuple, dtype=int)
        normalized_strategy = "group"
    else:
        raise ValueError("strategy must be 'random' or 'group'")

    train_indices = tuple(sorted(int(index) for index in train_idx))
    test_indices = tuple(sorted(int(index) for index in test_idx))
    train_groups = tuple(groups[index] for index in train_indices)
    test_groups = tuple(groups[index] for index in test_indices)
    if normalized_strategy == "group" and set(train_groups) & set(test_groups):
        raise RuntimeError("internal error: a group appeared in both train and test")

    return SplitResult(
        train=tuple(records[index] for index in train_indices),
        test=tuple(records[index] for index in test_indices),
        train_indices=train_indices,
        test_indices=test_indices,
        train_groups=train_groups,
        test_groups=test_groups,
        strategy=normalized_strategy,
        seed=seed,
        requested_test_size=test_size,
    )


def cross_validation_splits(
    records: Sequence[Any],
    *,
    strategy: str = "group",
    n_splits: int = 5,
    seed: int = 42,
    label_field: str = "label",
    group_fields: str | Sequence[str] | None = None,
    group_values: Sequence[Any] | None = None,
    stratify: bool = True,
) -> tuple[SplitResult, ...]:
    """Build deterministic random or group-disjoint cross-validation folds.

    ``group_values`` is intended for precomputed semantic clusters.  This is
    useful when exact task/tool keys are nearly unique and therefore do not
    constitute a meaningful generalization test.
    """

    records = tuple(records)
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if len(records) < n_splits:
        raise ValueError("n_splits cannot exceed the number of records")
    labels_raw = [record_value(record, label_field, _MISSING) for record in records]
    have_labels = all(label is not _MISSING and label is not None for label in labels_raw)
    labels: list[Any] | None = labels_raw if have_labels else None
    indices = np.arange(len(records))
    if group_values is not None:
        if len(group_values) != len(records):
            raise ValueError("group_values must have the same length as records")
        groups = tuple(f"provided:{_canonical(value)}" for value in group_values)
    else:
        groups = tuple(
            infer_group_key(record, group_fields=group_fields, record_index=index)
            for index, record in enumerate(records)
        )

    normalized = strategy.casefold().replace("-", "_")
    if normalized in {"random", "stratified_random"}:
        can_stratify = (
            stratify
            and labels is not None
            and len(set(labels)) > 1
            and min(Counter(labels).values()) >= n_splits
        )
        splitter: Any
        if can_stratify:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            iterator = splitter.split(indices, labels)
        else:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            iterator = splitter.split(indices)
        normalized = "random"
    elif normalized in {"group", "grouped", "group_holdout"}:
        if len(set(groups)) < n_splits:
            raise ValueError("grouped cross-validation needs at least n_splits distinct groups")
        can_stratify = stratify and labels is not None and len(set(labels)) > 1
        if can_stratify:
            splitter = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=seed
            )
            iterator = splitter.split(indices, labels, groups)
        else:
            splitter = GroupKFold(n_splits=n_splits)
            iterator = splitter.split(indices, labels, groups)
        normalized = "group"
    else:
        raise ValueError("strategy must be 'random' or 'group'")

    folds: list[SplitResult] = []
    for fold_index, (train_idx, test_idx) in enumerate(iterator):
        train_indices = tuple(sorted(int(index) for index in train_idx))
        test_indices = tuple(sorted(int(index) for index in test_idx))
        train_groups = tuple(groups[index] for index in train_indices)
        test_groups = tuple(groups[index] for index in test_indices)
        if normalized == "group" and set(train_groups) & set(test_groups):
            raise RuntimeError("internal error: a group appeared in both train and test")
        folds.append(
            SplitResult(
                train=tuple(records[index] for index in train_indices),
                test=tuple(records[index] for index in test_indices),
                train_indices=train_indices,
                test_indices=test_indices,
                train_groups=train_groups,
                test_groups=test_groups,
                strategy=normalized,
                seed=seed,
                requested_test_size=1.0 / n_splits,
                fold_index=fold_index,
            )
        )
    return tuple(folds)


def make_split(*args: Any, **kwargs: Any) -> SplitResult:
    """Alias retained for callers that prefer an experiment-oriented name."""

    return split_records(*args, **kwargs)


__all__ = [
    "DEFAULT_GROUP_FIELDS",
    "SplitResult",
    "cross_validation_splits",
    "infer_group_key",
    "make_split",
    "record_value",
    "split_records",
]
