"""Deterministic, label-free text views of normalized agent trajectories."""

from __future__ import annotations

import json
from typing import Callable, Iterable

from .data import ToolSpec, TrajectoryEvent, TrajectoryRecord, canonical_text


VIEW_NAMES = (
    "full",
    "user_only",
    "tool_schema_only",
    "action_only",
    "final_answer_only",
)

_ALIASES = {
    "user": "user_only",
    "users": "user_only",
    "tool": "tool_schema_only",
    "tools": "tool_schema_only",
    "tool_only": "tool_schema_only",
    "tool_schema": "tool_schema_only",
    "actions": "action_only",
    "tool_calls": "action_only",
    "final": "final_answer_only",
    "answer": "final_answer_only",
    "final_answer": "final_answer_only",
}


def _header_value(value: str) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _section(title: str, items: Iterable[str]) -> str:
    materialized = [item for item in items if item]
    body = "\n\n".join(materialized) if materialized else "<NONE>"
    return f"[{title}]\n{body}"


def _render_tool(tool: ToolSpec, position: int) -> str:
    lines = [f"[TOOL {position:04d}]", f"name: {_header_value(tool.name)}"]
    lines.extend(("description:", tool.description or "<EMPTY>"))
    lines.extend(("parameters:", canonical_text(tool.parameters) or "{}"))
    return "\n".join(lines)


def _render_event(event: TrajectoryEvent) -> str:
    attributes = [
        f"kind={_header_value(event.kind)}",
        f"role={_header_value(event.role)}",
    ]
    if event.name:
        attributes.append(f"name={_header_value(event.name)}")
    if event.call_id:
        attributes.append(f"call_id={_header_value(event.call_id)}")
    lines = [f"[EVENT {event.index:04d} {' '.join(attributes)}]"]
    if event.content:
        lines.extend(("content:", event.content))
    if event.arguments is not None:
        lines.extend(("arguments:", event.arguments or "<EMPTY>"))
    return "\n".join(lines)


def _render_user(event: TrajectoryEvent, position: int) -> str:
    return f"[USER {position:04d}]\n{event.content or '<EMPTY>'}"


def _render_action(event: TrajectoryEvent, position: int) -> str:
    lines = [f"[ACTION {position:04d}]"]
    lines.append(f"name: {_header_value(event.name or 'unknown_tool')}")
    if event.call_id:
        lines.append(f"call_id: {_header_value(event.call_id)}")
    lines.extend(("arguments:", event.arguments if event.arguments is not None else "{}"))
    return "\n".join(lines)


def _full(record: TrajectoryRecord) -> str:
    return "\n\n".join(
        (
            "[VIEW full]",
            _section(
                "AVAILABLE_TOOLS",
                (_render_tool(tool, index) for index, tool in enumerate(record.tools)),
            ),
            _section("TRAJECTORY", (_render_event(event) for event in record.events)),
        )
    )


def _user_only(record: TrajectoryRecord) -> str:
    selected = [
        event
        for event in record.events
        if event.kind == "message" and event.role == "user"
    ]
    return "\n\n".join(
        (
            "[VIEW user_only]",
            _section(
                "USER_MESSAGES",
                (_render_user(event, index) for index, event in enumerate(selected)),
            ),
        )
    )


def _tool_schema_only(record: TrajectoryRecord) -> str:
    return "\n\n".join(
        (
            "[VIEW tool_schema_only]",
            _section(
                "AVAILABLE_TOOLS",
                (_render_tool(tool, index) for index, tool in enumerate(record.tools)),
            ),
        )
    )


def _action_only(record: TrajectoryRecord) -> str:
    selected = [event for event in record.events if event.kind == "tool_call"]
    return "\n\n".join(
        (
            "[VIEW action_only]",
            _section(
                "TOOL_CALLS",
                (_render_action(event, index) for index, event in enumerate(selected)),
            ),
        )
    )


def _final_answer_only(record: TrajectoryRecord) -> str:
    selected: TrajectoryEvent | None = None
    for event in reversed(record.events):
        if event.kind == "completion":
            selected = event
            break
    if selected is None:
        for event in reversed(record.events):
            if event.kind == "message" and event.role == "assistant" and event.content:
                selected = event
                break
    if selected is None:
        body = "<NONE>"
    else:
        body = selected.content or "<EMPTY>"
    return f"[VIEW final_answer_only]\n\n[FINAL_ANSWER]\n{body}"


_BUILDERS: dict[str, Callable[[TrajectoryRecord], str]] = {
    "full": _full,
    "user_only": _user_only,
    "tool_schema_only": _tool_schema_only,
    "action_only": _action_only,
    "final_answer_only": _final_answer_only,
}


def normalize_view_name(view: str) -> str:
    """Normalize a view name and reject accidental, incomparable variants."""

    normalized = view.strip().lower().replace("-", "_").replace(" ", "_")
    normalized = _ALIASES.get(normalized, normalized)
    if normalized not in _BUILDERS:
        choices = ", ".join(VIEW_NAMES)
        raise ValueError(f"unknown view {view!r}; choose one of: {choices}")
    return normalized


def build_view(record: TrajectoryRecord, view: str) -> str:
    """Render one named partial view.

    The output contains no record ID, gold label, taxonomy label, or annotation
    rationale.  This is intentional: those fields can trivially leak the target.
    """

    normalized = normalize_view_name(view)
    return _BUILDERS[normalized](record)


def render_view(record: TrajectoryRecord, view: str) -> str:
    """Alias for :func:`build_view`, convenient in plotting/model code."""

    return build_view(record, view)


def build_views(record: TrajectoryRecord) -> dict[str, str]:
    """Render all five canonical views in stable order."""

    return {view: _BUILDERS[view](record) for view in VIEW_NAMES}


__all__ = [
    "VIEW_NAMES",
    "build_view",
    "build_views",
    "normalize_view_name",
    "render_view",
]
