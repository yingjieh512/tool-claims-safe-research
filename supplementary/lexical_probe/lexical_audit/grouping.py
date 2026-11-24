"""Derive approximate task groups without using safety labels.

ATBench does not publish task-template identifiers and exact user/tool signatures are
almost all unique.  This module therefore groups records by unsupervised TF-IDF
clustering over user requests and tool names.  Labels are used only for diagnostics
after the groups have been assigned.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Any, Iterable

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+", re.IGNORECASE)
_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
_LONG_TOKEN_RE = re.compile(r"\b(?=[A-Za-z0-9_-]{8,}\b)(?=\S*[A-Za-z])(?=\S*\d)\S+\b")
_NUMBER_RE = re.compile(r"\b\d+(?:[.:/_-]\d+)*\b")


def _trace(record: dict[str, Any]) -> list[dict[str, Any]]:
    value = record.get("contents", record.get("content", []))
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def grouping_text(record: dict[str, Any]) -> str:
    """Return label-free text used to derive an approximate task group."""

    users = [
        str(message.get("content", ""))
        for message in _trace(record)
        if str(message.get("role", "")).lower() == "user"
    ]
    tools = []
    raw_tools = record.get("tool_used", [])
    if isinstance(raw_tools, list):
        tools = [
            str(tool.get("name", ""))
            for tool in raw_tools
            if isinstance(tool, dict) and tool.get("name")
        ]

    text = " USER ".join(users).lower()
    text = _URL_RE.sub(" URLTOKEN ", text)
    text = _EMAIL_RE.sub(" EMAILTOKEN ", text)
    text = _UUID_RE.sub(" IDTOKEN ", text)
    text = _LONG_TOKEN_RE.sub(" IDTOKEN ", text)
    text = _NUMBER_RE.sub(" NUMTOKEN ", text)
    text = " ".join(text.split())
    return f"{text} TOOLNAMES {' '.join(sorted(tools))}".strip()


def derive_semantic_groups(
    records: Iterable[dict[str, Any]],
    *,
    n_groups: int = 50,
    random_state: int = 42,
) -> tuple[list[str], TfidfVectorizer, KMeans]:
    """Cluster records into label-free, approximate task/tool groups.

    The fitted vectorizer and clusterer are returned so an audit can record the
    precise transformation.  This is a benchmark-analysis grouping heuristic; it
    is not a claim that the inferred groups recover ATBench's private templates.
    """

    rows = list(records)
    if len(rows) < 2:
        raise ValueError("At least two records are required to derive groups.")
    if not 2 <= n_groups <= len(rows):
        raise ValueError("n_groups must be between 2 and the number of records.")

    texts = [grouping_text(record) for record in rows]
    min_df = 2 if len(rows) >= 20 else 1
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=0.995,
        max_features=20_000,
        sublinear_tf=True,
        strip_accents="unicode",
        stop_words="english",
    )
    matrix = vectorizer.fit_transform(texts)
    if matrix.shape[1] == 0:
        raise ValueError("Grouping text produced an empty TF-IDF vocabulary.")

    model = KMeans(
        n_clusters=n_groups,
        random_state=random_state,
        n_init=10,
        max_iter=300,
    )
    assignments = model.fit_predict(matrix)
    width = max(2, len(str(n_groups - 1)))
    groups = [f"cluster_{int(value):0{width}d}" for value in assignments]
    return groups, vectorizer, model


def group_diagnostics(
    records: Iterable[dict[str, Any]],
    groups: Iterable[str],
    vectorizer: TfidfVectorizer,
    model: KMeans,
    *,
    top_terms: int = 8,
) -> list[dict[str, Any]]:
    """Describe inferred groups; labels affect this report, never assignment."""

    rows = list(records)
    group_values = list(groups)
    if len(rows) != len(group_values):
        raise ValueError("records and groups must have the same length")

    members: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(group_values):
        members[group].append(index)

    terms = vectorizer.get_feature_names_out()
    summaries = []
    for group, indices in sorted(members.items()):
        cluster_id = int(group.rsplit("_", 1)[1])
        center = model.cluster_centers_[cluster_id]
        best = center.argsort()[::-1][:top_terms]
        labels = [int(rows[index].get("label", 0)) for index in indices]
        summaries.append(
            {
                "group": group,
                "size": len(indices),
                "safe": int(Counter(labels).get(0, 0)),
                "unsafe": int(Counter(labels).get(1, 0)),
                "unsafe_rate": sum(labels) / len(labels),
                "top_terms": [str(terms[index]) for index in best if center[index] > 0],
                "sample_ids": [str(rows[index].get("id", index)) for index in indices[:5]],
            }
        )
    return summaries


def write_group_manifest(path: str, diagnostics: list[dict[str, Any]]) -> None:
    """Write human-readable group diagnostics as stable JSON."""

    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
