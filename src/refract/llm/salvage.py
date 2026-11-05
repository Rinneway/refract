"""Keep what is valid in a model answer that fails its schema.

A small model often returns one malformed item — a risk with `"impact": "critical"`, an action without a rationale —
and the whole answer fails validation, so the run ends without an insight. When the model's own repair also fails,
`salvage` keeps the rest:

- an enum value in another case, with spaces or with words around it is normalised (`"High "` → `"high"`,
  `"apply product"` → `"apply_product"`, `"medium (moderate)"` → `"medium"`);
- a single object where a list is expected becomes a one-item list, a one-item list where a value is expected
  becomes the value; a list over its maximum length keeps its first items;
- an invalid optional field is dropped, so its default applies;
- a list item that still does not validate is dropped (a risk, an action, a source id).

A required top-level field that cannot be fixed still fails the answer: code never invents content. Every change is
returned as a note for the trace.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_ENUM_ERRORS = {"enum", "literal_error"}
MAX_ROUNDS = 60


def extract_json(text: str) -> Any:
    """The JSON object in a model answer (``` fences and stray text around it removed), or None."""
    cleaned = _FENCE.sub("", (text or "").strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None


def _at(data: Any, path: tuple) -> Any:
    for p in path:
        data = data[p]
    return data


def _label(item: Any, path: tuple) -> str:
    where = ".".join(map(str, path))
    if isinstance(item, dict):
        ident = item.get("claim_id") or item.get("action_id") or item.get("name") or item.get("step_id")
        if isinstance(ident, str) and ident:
            return f"{where} ({ident})"
    return where


def _short(value: Any, limit: int = 40) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _enum_value(value: str, err: Mapping[str, Any]) -> str | None:
    """The allowed value a model meant: the same in lower case, or the only allowed value among its words."""
    norm = re.sub(r"[\s-]+", "_", value.strip().lower())
    allowed = re.findall(r"'([^']*)'", str((err.get("ctx") or {}).get("expected", "")))
    if not allowed or norm in allowed:
        return norm
    words = set(re.findall(r"[a-z_]+", norm.replace("_", " "))) | set(re.findall(r"[a-z_]+", norm))
    hits = [a for a in allowed if a in words]
    return hits[0] if len(hits) == 1 else None


def salvage(schema: type[T], data: Any) -> tuple[T | None, list[str]]:
    """Validate `data` against `schema`, fixing or dropping what fails. Returns the object (or None) and notes."""
    if not isinstance(data, dict):
        return None, []
    data = copy.deepcopy(data)
    notes: list[str] = []
    dropped_fields: dict[tuple, Any] = {}  # field → its invalid value, until we know whether it was required
    for _ in range(MAX_ROUNDS):
        try:
            return schema.model_validate(data), notes
        except ValidationError as e:
            err = e.errors()[0]
        loc = tuple(err["loc"])
        if not loc:  # a whole-object validator failed: nothing to cut
            return None, notes
        try:
            parent, key = _at(data, loc[:-1]), loc[-1]
        except (KeyError, IndexError, TypeError):
            return None, notes
        value = err.get("input")
        where = ".".join(map(str, loc))

        if err["type"] in _ENUM_ERRORS and isinstance(value, str):
            fixed = _enum_value(value, err)
            if fixed is not None and fixed != value:
                parent[key] = fixed
                notes.append(f"{where}: “{_short(value)}” read as “{fixed}”")
                continue
        limit = (err.get("ctx") or {}).get("max_length")
        if err["type"] == "too_long" and isinstance(value, list) and isinstance(limit, int):
            parent[key] = value[:limit]
            notes.append(f"{where}: {len(value)} items, kept the first {limit}")
            continue
        if err["type"] == "list_type" and value is not None and not isinstance(value, list):
            parent[key] = [value]
            notes.append(f"{where}: a single item read as a list")
            continue
        if err["type"] != "list_type" and isinstance(value, list) and len(value) == 1:
            parent[key] = value[0]
            notes.append(f"{where}: a one-item list read as its item")
            continue

        if err["type"] != "missing" and isinstance(parent, dict) and loc not in dropped_fields:
            dropped_fields[loc] = (
                value  # if the field is required, the next round drops the item that holds it
            )
            del parent[key]
            notes.append(f"{where}: invalid value “{_short(value)}” dropped")
            continue
        reason = f"{where} {_short(err['msg'], 80)}"
        if (
            err["type"] == "missing" and loc in dropped_fields
        ):  # it was required: one note for the item instead
            if notes and notes[-1].startswith(f"{where}: invalid value"):
                notes.pop()
            reason = f"{key} “{_short(dropped_fields[loc])}” is invalid"

        for j in range(len(loc) - 1, -1, -1):
            if not isinstance(loc[j], int):
                continue
            try:
                items = _at(data, loc[:j])
            except (KeyError, IndexError, TypeError):
                continue
            idx = loc[j]
            if isinstance(items, list) and isinstance(idx, int) and idx < len(items):
                notes.append(f"{_label(items[idx], loc[: j + 1])} dropped: {reason}")
                del items[idx]
                break
        else:  # a required top-level field
            if loc in dropped_fields:
                notes.append(f"{where}: “{_short(dropped_fields[loc])}” is invalid and the field is required")
            return None, notes
    return None, notes
