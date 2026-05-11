"""JSON file loading helpers with encoding fallbacks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def should_skip_json_file(path: str | Path) -> bool:
    """Return whether a path is an auxiliary file rather than a real spectrum JSON."""

    json_path = Path(path)
    return (
        json_path.name.startswith(".")
        or json_path.name.startswith("_")
        or "__MACOSX" in json_path.parts
        or any(part.startswith(".") for part in json_path.parts)
    )


def load_json_file(path: str | Path) -> dict[str, Any]:
    """Load a JSON object from disk with common text encoding fallbacks.

    Args:
        path: JSON file path.

    Returns:
        Parsed JSON object.

    Raises:
        ValueError: If the file cannot be decoded as JSON.
        TypeError: If the JSON root is not an object.
    """

    json_path = Path(path)
    payload_bytes = json_path.read_bytes()
    errors: list[str] = []
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            text = payload_bytes.decode(encoding)
            payload = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"{encoding}: {exc}")
            continue
        if not isinstance(payload, dict):
            raise TypeError(f"JSON root must be an object in {json_path}")
        return payload
    detail = "; ".join(errors)
    raise ValueError(f"Failed to decode JSON file {json_path}. Tried utf-8, utf-8-sig, gb18030, latin-1. {detail}")
