"""Load and normalize trajectory-safety benchmark records.

The public datasets used by this project do not share one schema.  ATBench, for
example, uses ``contents``/``tool_used`` while OpenAI-style traces usually use
``messages``/``tools``.  This module converts those shapes into a small common
representation without depending on pandas or Hugging Face ``datasets``.

The normalizer is deliberately conservative about labels: labels are stored on
the :class:`TrajectoryRecord`, but never mixed into the event text rendered for
models.  That separation prevents the audit itself from leaking the answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import io
import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence, TextIO


JSONValue = Any


class DataFormatError(ValueError):
    """Raised when an input file cannot be interpreted as trajectory records."""


@dataclass(frozen=True)
class ToolSpec:
    """A normalized tool definition, in the order exposed to the agent."""

    name: str
    description: str = ""
    parameters: JSONValue = field(default_factory=dict)
    metadata: Mapping[str, JSONValue] = field(
        default_factory=dict, repr=False, compare=False
    )


@dataclass(frozen=True)
class TrajectoryEvent:
    """One ordered event from an agent execution trace.

    ``kind`` is one of ``message``, ``reasoning``, ``tool_call``,
    ``tool_result``, ``completion``, or ``other``.  Tool arguments are
    canonicalized to text so that equal JSON objects have equal
    representations across input files.
    """

    index: int
    kind: str
    role: str = "unknown"
    content: str = ""
    name: str | None = None
    arguments: str | None = None
    call_id: str | None = None
    metadata: Mapping[str, JSONValue] = field(
        default_factory=dict, repr=False, compare=False
    )


@dataclass(frozen=True)
class TrajectoryRecord:
    """A benchmark row normalized for partial-view experiments."""

    record_id: str
    label: int | None
    events: tuple[TrajectoryEvent, ...]
    tools: tuple[ToolSpec, ...] = ()
    metadata: Mapping[str, JSONValue] = field(
        default_factory=dict, repr=False, compare=False
    )
    raw: Mapping[str, JSONValue] = field(
        default_factory=dict, repr=False, compare=False
    )

    @property
    def id(self) -> str:
        """A short alias useful in tables and downstream scripts."""

        return self.record_id


_ID_KEYS = (
    "id",
    "record_id",
    "sample_id",
    "example_id",
    "trajectory_id",
    "conv_id",
    "conversation_id",
    "task_id",
)

_TRAJECTORY_KEYS = (
    # ATBench and ATBench500 come first.
    "contents",
    "content",
    "trajectory",
    "messages",
    "conversation",
    "conversations",
    "chat",
    "history",
    "trace",
    "steps",
    "events",
    "interactions",
    "dialogue",
)

_TOOL_KEYS = (
    "tool_used",
    "tool_use",
    "tools",
    "available_tools",
    "tool_definitions",
    "tool_schemas",
    "functions",
)

_LABEL_KEYS = (
    "label",
    "safety_label",
    "safety",
    "verdict",
    "classification",
    "target",
)

_WRAPPER_KEYS = ("data", "records", "examples", "items", "samples", "rows")
_SPLIT_KEYS = ("train", "validation", "valid", "dev", "test")


def canonical_json(value: JSONValue) -> str:
    """Return stable, UTF-8-friendly JSON for arbitrary JSON-like data."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        # A custom object should not make an otherwise useful row unloadable.
        return json.dumps(str(value), ensure_ascii=False)


def canonical_text(value: JSONValue) -> str:
    """Convert a scalar or nested value to deterministic text."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if isinstance(value, (Mapping, list, tuple)):
        return canonical_json(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).strip()


def _first_present(record: Mapping[str, JSONValue], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _parse_binary(value: JSONValue) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in {
            "0",
            "safe",
            "benign",
            "false",
        }:
            return 0
        if normalized in {
            "1",
            "unsafe",
            "harmful",
            "malicious",
            "true",
        }:
            return 1
    return None


def normalize_label(record: Mapping[str, JSONValue]) -> int | None:
    """Normalize a safety label to ``0`` (safe) or ``1`` (unsafe).

    Besides direct binary labels, common boolean spellings are supported.
    ``is_safe`` and ``safe`` have inverse polarity.  A small number of datasets
    put the label in an ``annotation`` or ``labels`` object, so those containers
    are inspected one level deep as well.
    """

    def from_mapping(candidate: Mapping[str, JSONValue]) -> int | None:
        for key in ("is_unsafe", "unsafe"):
            if key in candidate:
                parsed = _parse_binary(candidate[key])
                if parsed is not None:
                    return parsed
        for key in ("is_safe", "safe"):
            if key in candidate:
                parsed = _parse_binary(candidate[key])
                if parsed is not None:
                    return 1 - parsed
        for key in _LABEL_KEYS:
            if key in candidate:
                parsed = _parse_binary(candidate[key])
                if parsed is not None:
                    return parsed
        return None

    parsed = from_mapping(record)
    if parsed is not None:
        return parsed
    for container_key in ("annotation", "annotations", "labels", "metadata"):
        nested = record.get(container_key)
        if isinstance(nested, Mapping):
            parsed = from_mapping(nested)
            if parsed is not None:
                return parsed
    return None


def _tool_spec(value: JSONValue, default_name: str | None = None) -> ToolSpec:
    if isinstance(value, str):
        parsed = _try_json(value)
        if parsed is not None and parsed != value:
            return _tool_spec(parsed, default_name=default_name)
        return ToolSpec(name=default_name or value.strip() or "unknown_tool")

    if not isinstance(value, Mapping):
        return ToolSpec(
            name=default_name or "unknown_tool", description=canonical_text(value)
        )

    outer = dict(value)
    function = outer.get("function")
    body = dict(function) if isinstance(function, Mapping) else outer
    name = _first_present(
        body, ("name", "tool_name", "function_name", "api_name", "operation_id")
    )
    if name is None:
        name = default_name or _first_present(outer, ("name", "id", "type"))
    description = _first_present(
        body, ("description", "desc", "documentation", "docstring", "usage")
    )
    parameters = _first_present(
        body, ("parameters", "input_schema", "schema", "arguments", "args")
    )
    if parameters is None:
        parameters = {}
    consumed = {
        "function",
        "name",
        "tool_name",
        "function_name",
        "api_name",
        "operation_id",
        "description",
        "desc",
        "documentation",
        "docstring",
        "usage",
        "parameters",
        "input_schema",
        "schema",
        "arguments",
        "args",
    }
    metadata = {key: item for key, item in outer.items() if key not in consumed}
    return ToolSpec(
        name=canonical_text(name) or "unknown_tool",
        description=canonical_text(description),
        parameters=parameters,
        metadata=metadata,
    )


def normalize_tools(value: JSONValue) -> tuple[ToolSpec, ...]:
    """Normalize list-, mapping-, and OpenAI-style tool definitions."""

    tools: list[ToolSpec] = []

    def visit(candidate: JSONValue, default_name: str | None = None) -> None:
        if candidate is None:
            return
        if isinstance(candidate, str):
            parsed = _try_json(candidate)
            if parsed is not None and parsed != candidate:
                visit(parsed, default_name)
            else:
                tools.append(_tool_spec(candidate, default_name))
            return
        if isinstance(candidate, Sequence) and not isinstance(
            candidate, (str, bytes, bytearray)
        ):
            for item in candidate:
                visit(item)
            return
        if isinstance(candidate, Mapping):
            definition_keys = {
                "function",
                "name",
                "tool_name",
                "function_name",
                "api_name",
                "operation_id",
                "description",
                "desc",
                "parameters",
                "input_schema",
                "schema",
            }
            if definition_keys.intersection(candidate):
                tools.append(_tool_spec(candidate, default_name))
                return
            # Some exports use {"search": {"description": ..., ...}}.
            for key, item in candidate.items():
                visit(item, default_name=str(key))
            return
        tools.append(_tool_spec(candidate, default_name))

    visit(value)
    return tuple(tools)


def _try_json(value: str) -> JSONValue | None:
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def _role(value: JSONValue, fallback: str = "unknown") -> str:
    text = canonical_text(value).lower()
    aliases = {
        "human": "user",
        "customer": "user",
        "model": "assistant",
        "ai": "assistant",
        "agent": "assistant",
        "function": "tool",
        "environment": "tool",
        "observation": "tool",
    }
    return aliases.get(text, text or fallback)


def _event_metadata(item: Mapping[str, JSONValue], consumed: set[str]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if key not in consumed}


def normalize_events(value: JSONValue) -> tuple[TrajectoryEvent, ...]:
    """Flatten nested chat/step/tool-call structures without reordering them."""

    events: list[TrajectoryEvent] = []

    def emit(
        kind: str,
        *,
        role: str = "unknown",
        content: JSONValue = "",
        name: JSONValue = None,
        arguments: JSONValue = None,
        call_id: JSONValue = None,
        metadata: Mapping[str, JSONValue] | None = None,
        allow_empty: bool = False,
    ) -> None:
        content_text = canonical_text(content)
        argument_text = None if arguments is None else canonical_text(arguments)
        name_text = canonical_text(name) or None
        call_id_text = canonical_text(call_id) or None
        if not allow_empty and not (
            content_text or argument_text or name_text or call_id_text
        ):
            return
        events.append(
            TrajectoryEvent(
                index=len(events),
                kind=kind,
                role=_role(role),
                content=content_text,
                name=name_text,
                arguments=argument_text,
                call_id=call_id_text,
                metadata=dict(metadata or {}),
            )
        )

    def tool_call(candidate: JSONValue, inherited_role: str = "assistant") -> None:
        if candidate is None:
            return
        if isinstance(candidate, str):
            parsed = _try_json(candidate)
            if parsed is not None:
                tool_call(parsed, inherited_role)
            else:
                emit("tool_call", role=inherited_role, name=candidate)
            return
        if isinstance(candidate, Sequence) and not isinstance(
            candidate, (str, bytes, bytearray)
        ):
            for call in candidate:
                tool_call(call, inherited_role)
            return
        if not isinstance(candidate, Mapping):
            emit("tool_call", role=inherited_role, arguments=candidate)
            return

        if "action" in candidate and len(candidate) == 1:
            tool_call(candidate["action"], inherited_role)
            return
        function = candidate.get("function")
        body = function if isinstance(function, Mapping) else candidate
        name = _first_present(
            body,
            ("name", "tool", "tool_name", "function_name", "api_name", "operation"),
        )
        arguments = _first_present(
            body, ("arguments", "args", "input", "parameters", "kwargs")
        )
        call_id = _first_present(
            candidate, ("id", "call_id", "tool_call_id", "tool_use_id")
        )
        consumed = {
            "function",
            "name",
            "tool",
            "tool_name",
            "function_name",
            "api_name",
            "operation",
            "arguments",
            "args",
            "input",
            "parameters",
            "kwargs",
            "id",
            "call_id",
            "tool_call_id",
            "tool_use_id",
            "type",
        }
        emit(
            "tool_call",
            role=inherited_role,
            name=name,
            arguments=arguments,
            call_id=call_id,
            metadata=_event_metadata(candidate, consumed),
        )

    def completion(candidate: JSONValue) -> bool:
        """Parse ATBench's ``Complete{...}`` terminal action.

        ``True`` means the value was a completion and must not also be emitted
        as a tool call.  Keeping completions as a separate event kind lets the
        final-answer view distinguish a real terminal response from an
        assistant preamble before a tool call.
        """

        payload: JSONValue
        if isinstance(candidate, str):
            stripped = candidate.strip()
            match = re.fullmatch(r"Complete\s*(\{.*\})", stripped, flags=re.DOTALL)
            if match is None:
                return False
            try:
                payload = json.loads(match.group(1))
            except json.JSONDecodeError:
                return False
        elif isinstance(candidate, Mapping):
            action_name = _first_present(
                candidate, ("name", "tool", "action", "operation")
            )
            if canonical_text(action_name).lower() not in {
                "complete",
                "final_answer",
            }:
                return False
            payload = _first_present(
                candidate, ("arguments", "args", "input", "parameters")
            )
            if payload is None:
                return False
            if isinstance(payload, Mapping) and not any(
                key in payload
                for key in ("response", "answer", "final_answer", "content", "message")
            ):
                return False
        else:
            return False

        if isinstance(payload, Mapping):
            response = _first_present(
                payload, ("response", "answer", "final_answer", "content", "message")
            )
            if response is None:
                response = payload
        else:
            response = payload

        # ATBench often double-encodes a quoted response inside the JSON action.
        if isinstance(response, str):
            stripped_response = response.strip()
            if stripped_response.startswith('"') and stripped_response.endswith('"'):
                try:
                    parsed_response = json.loads(stripped_response)
                except json.JSONDecodeError:
                    pass
                else:
                    if isinstance(parsed_response, str):
                        response = parsed_response
        emit("completion", role="assistant", content=response, allow_empty=True)
        return True

    def tool_result(
        candidate: JSONValue,
        *,
        fallback_name: str | None = None,
        fallback_call_id: str | None = None,
    ) -> None:
        if isinstance(candidate, Mapping):
            name = _first_present(
                candidate, ("name", "tool", "tool_name", "function_name")
            )
            call_id = _first_present(
                candidate, ("tool_call_id", "tool_use_id", "call_id", "id")
            )
            content = _first_present(
                candidate,
                ("content", "output", "result", "observation", "response", "value"),
            )
            if content is None:
                content = candidate
            consumed = {
                "name",
                "tool",
                "tool_name",
                "function_name",
                "tool_call_id",
                "tool_use_id",
                "call_id",
                "id",
                "content",
                "output",
                "result",
                "observation",
                "response",
                "value",
                "type",
            }
            emit(
                "tool_result",
                role="tool",
                content=content,
                name=name or fallback_name,
                call_id=call_id or fallback_call_id,
                metadata=_event_metadata(candidate, consumed),
                allow_empty=True,
            )
        else:
            emit(
                "tool_result",
                role="tool",
                content=candidate,
                name=fallback_name,
                call_id=fallback_call_id,
                allow_empty=True,
            )

    def content_blocks(blocks: Sequence[JSONValue], inherited_role: str) -> None:
        for block in blocks:
            if isinstance(block, str):
                emit("message", role=inherited_role, content=block, allow_empty=True)
                continue
            if not isinstance(block, Mapping):
                emit("message", role=inherited_role, content=block, allow_empty=True)
                continue
            block_type = canonical_text(block.get("type")).lower()
            if block_type in {"text", "input_text", "output_text"}:
                emit(
                    "message",
                    role=inherited_role,
                    content=_first_present(block, ("text", "content", "value")),
                    allow_empty=True,
                )
            elif block_type in {"tool_use", "tool_call", "function_call"}:
                tool_call(block, inherited_role)
            elif block_type in {
                "tool_result",
                "function_result",
                "tool_response",
            }:
                tool_result(block)
            elif block_type in {"thinking", "reasoning", "analysis"}:
                emit(
                    "reasoning",
                    role=inherited_role,
                    content=_first_present(block, ("thinking", "text", "content")),
                )
            elif "text" in block:
                emit(
                    "message",
                    role=inherited_role,
                    content=block["text"],
                    allow_empty=True,
                )
            elif "content" in block:
                visit(block["content"], inherited_role)
            else:
                emit("other", role=inherited_role, content=block)

    def message(item: Mapping[str, JSONValue], inherited_role: str) -> None:
        current_role = _role(item.get("role"), inherited_role)
        content = item.get("content")
        call_id = _first_present(item, ("tool_call_id", "tool_use_id", "call_id"))
        name = _first_present(item, ("name", "tool_name", "function_name"))

        # ATBench agent turns are ``thought`` followed by an ``action``.  The
        # action is either a JSON tool call or a ``Complete{...}`` response.
        if current_role == "assistant" and (
            "thought" in item or "action" in item
        ):
            if item.get("thought") not in (None, ""):
                emit("reasoning", role=current_role, content=item["thought"])
            if "action" in item and not completion(item["action"]):
                tool_call(item["action"], current_role)

        if current_role == "tool":
            result_value = content
            if result_value is None:
                result_value = _first_present(
                    item, ("output", "result", "observation", "response")
                )
            tool_result(
                {
                    "content": result_value,
                    "name": name,
                    "tool_call_id": call_id,
                }
            )
        elif isinstance(content, Sequence) and not isinstance(
            content, (str, bytes, bytearray)
        ):
            content_blocks(content, current_role)
        elif content is not None:
            emit("message", role=current_role, content=content, allow_empty=True)

        # OpenAI puts calls beside message content; their list order is causal.
        if "function_call" in item:
            tool_call(item["function_call"], current_role)
        if "tool_call" in item:
            tool_call(item["tool_call"], current_role)
        if "tool_calls" in item:
            tool_call(item["tool_calls"], current_role)

    def visit(candidate: JSONValue, inherited_role: str = "unknown") -> None:
        if candidate is None:
            return
        if isinstance(candidate, str):
            parsed = _try_json(candidate)
            if parsed is not None:
                visit(parsed, inherited_role)
            else:
                emit(
                    "message", role=inherited_role, content=candidate, allow_empty=True
                )
            return
        if isinstance(candidate, Sequence) and not isinstance(
            candidate, (str, bytes, bytearray)
        ):
            for item in candidate:
                visit(item, inherited_role)
            return
        if not isinstance(candidate, Mapping):
            emit("other", role=inherited_role, content=candidate)
            return

        event_type = canonical_text(candidate.get("type")).lower()
        if event_type in {"tool_use", "tool_call", "function_call"}:
            tool_call(candidate, inherited_role if inherited_role != "unknown" else "assistant")
            return
        if event_type in {"tool_result", "function_result", "tool_response"}:
            tool_result(candidate)
            return
        if event_type in {"text", "input_text", "output_text"}:
            emit(
                "message",
                role=inherited_role,
                content=_first_present(candidate, ("text", "content", "value")),
                allow_empty=True,
            )
            return
        if "role" in candidate:
            message(candidate, inherited_role)
            return

        # Session logs often wrap a JSON-encoded chat message in ``message``.
        if "message" in candidate:
            visit(candidate["message"], inherited_role)
            return

        # A bare OpenAI function-call object has no role.
        if "function" in candidate and isinstance(candidate.get("function"), Mapping):
            tool_call(candidate)
            return

        # Step-style traces use ordered fields such as thought/action/observation.
        handled = False
        last_tool_name: str | None = None
        last_call_id: str | None = None
        for key, item in candidate.items():
            normalized_key = key.lower()
            if normalized_key in {
                "events",
                "messages",
                "trajectory",
                "trace",
                "steps",
                "interactions",
                "conversation",
                "conversations",
                "dialogue",
                "contents",
                "content",
            }:
                visit(item, inherited_role)
                handled = True
            elif normalized_key in {"user", "request", "query", "instruction"}:
                emit("message", role="user", content=item)
                handled = True
            elif normalized_key in {
                "assistant",
                "agent",
                "response",
                "answer",
            }:
                emit("message", role="assistant", content=item)
                handled = True
            elif normalized_key in {"final", "final_answer"}:
                emit("completion", role="assistant", content=item)
                handled = True
            elif normalized_key in {"thought", "thinking", "reasoning", "analysis"}:
                emit("reasoning", role="assistant", content=item)
                handled = True
            elif normalized_key in {"action", "tool_call", "function_call"}:
                before = len(events)
                if not completion(item):
                    tool_call(item)
                if len(events) > before:
                    last_tool_name = events[-1].name
                    last_call_id = events[-1].call_id
                handled = True
            elif normalized_key in {
                "observation",
                "tool_result",
                "tool_response",
                "output",
            }:
                tool_result(
                    item,
                    fallback_name=last_tool_name,
                    fallback_call_id=last_call_id,
                )
                handled = True
        if not handled:
            emit("other", role=inherited_role, content=candidate)

    visit(value)
    return tuple(events)


def normalize_record(
    raw: Mapping[str, JSONValue], index: int = 0
) -> TrajectoryRecord:
    """Normalize one raw benchmark row."""

    if not isinstance(raw, Mapping):
        raise DataFormatError(
            f"record {index} must be a JSON object, got {type(raw).__name__}"
        )
    record = dict(raw)
    identifier = _first_present(record, _ID_KEYS)
    record_id = canonical_text(identifier) or f"record-{index:06d}"

    trajectory_value = _first_present(record, _TRAJECTORY_KEYS)
    if trajectory_value is None:
        raise DataFormatError(
            f"record {record_id!r} has no trajectory field; expected one of "
            + ", ".join(_TRAJECTORY_KEYS)
        )
    tools_value = _first_present(record, _TOOL_KEYS)

    structural = set(_ID_KEYS) | set(_TRAJECTORY_KEYS) | set(_TOOL_KEYS)
    structural |= set(_LABEL_KEYS) | {"safe", "unsafe", "is_safe", "is_unsafe"}
    metadata = {key: value for key, value in record.items() if key not in structural}
    return TrajectoryRecord(
        record_id=record_id,
        label=normalize_label(record),
        events=normalize_events(trajectory_value),
        tools=normalize_tools(tools_value),
        metadata=metadata,
        raw=record,
    )


def _expand_document(document: JSONValue) -> list[Mapping[str, JSONValue]]:
    if isinstance(document, Sequence) and not isinstance(
        document, (str, bytes, bytearray)
    ):
        records: list[Mapping[str, JSONValue]] = []
        for index, item in enumerate(document):
            if not isinstance(item, Mapping):
                raise DataFormatError(
                    f"record {index} must be a JSON object, got {type(item).__name__}"
                )
            records.append(item)
        return records
    if not isinstance(document, Mapping):
        raise DataFormatError(
            f"top-level JSON must be an object or array, got {type(document).__name__}"
        )

    for key in _WRAPPER_KEYS:
        value = document.get(key)
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return _expand_document(value)

    split_rows: list[Mapping[str, JSONValue]] = []
    saw_split = False
    for split in _SPLIT_KEYS:
        value = document.get(split)
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            saw_split = True
            for item in _expand_document(value):
                copied = dict(item)
                copied.setdefault("_split", split)
                split_rows.append(copied)
    if saw_split:
        return split_rows
    return [document]


def loads_records(text: str, *, format_hint: str | None = None) -> list[TrajectoryRecord]:
    """Parse records from JSON or JSON Lines text."""

    normalized_hint = (format_hint or "").lower().lstrip(".")
    if normalized_hint in {"jsonl", "ndjson"}:
        raw_records: list[Mapping[str, JSONValue]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                document = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataFormatError(
                    f"invalid JSON on line {line_number}: {exc.msg}"
                ) from exc
            raw_records.extend(_expand_document(document))
    else:
        try:
            document = json.loads(text)
        except json.JSONDecodeError as json_error:
            # Unknown extensions and in-memory buffers commonly contain JSONL.
            if normalized_hint == "json":
                raise DataFormatError(f"invalid JSON: {json_error.msg}") from json_error
            return loads_records(text, format_hint="jsonl")
        raw_records = _expand_document(document)

    return [normalize_record(record, index) for index, record in enumerate(raw_records)]


def load_records(
    source: str | Path | TextIO | Mapping[str, JSONValue] | Iterable[Mapping[str, JSONValue]],
) -> list[TrajectoryRecord]:
    """Load normalized records from a path, text stream, or Python objects.

    Paths ending in ``.jsonl`` or ``.ndjson`` are read as JSON Lines; all other
    files are first parsed as regular JSON.  For an iterable of mappings, input
    order is retained exactly.
    """

    if isinstance(source, (str, Path)):
        path = Path(source)
        text = path.read_text(encoding="utf-8-sig")
        return loads_records(text, format_hint=path.suffix)
    if isinstance(source, io.TextIOBase) or hasattr(source, "read"):
        text = source.read()
        if not isinstance(text, str):
            raise DataFormatError("input stream must return text, not bytes")
        stream_name = getattr(source, "name", "")
        suffix = Path(stream_name).suffix if stream_name else None
        return loads_records(text, format_hint=suffix)
    if isinstance(source, Mapping):
        raw_records = _expand_document(source)
    else:
        raw_records = list(source)
    return [normalize_record(record, index) for index, record in enumerate(raw_records)]


def iter_records(
    source: str | Path | TextIO | Mapping[str, JSONValue] | Iterable[Mapping[str, JSONValue]],
) -> Iterator[TrajectoryRecord]:
    """Iterate over :func:`load_records` results."""

    yield from load_records(source)


__all__ = [
    "DataFormatError",
    "ToolSpec",
    "TrajectoryEvent",
    "TrajectoryRecord",
    "canonical_json",
    "canonical_text",
    "iter_records",
    "load_records",
    "loads_records",
    "normalize_events",
    "normalize_label",
    "normalize_record",
    "normalize_tools",
]
