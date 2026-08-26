from __future__ import annotations

from pathlib import Path
import json

SESSION_SCHEMA = "xrd_image_toolkit_session"
SESSION_VERSION = "1.0"


def save_session(path, payload):
    record = {
        "schema": SESSION_SCHEMA,
        "schema_version": SESSION_VERSION,
        **payload,
    }
    Path(path).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return Path(path)


def load_session(path):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("schema") != SESSION_SCHEMA:
        raise ValueError("This is not an XRD Image Toolkit session file.")
    return record
