"""Deterministic serialization, hashing, and small filesystem helpers."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import json_value


def utc_now() -> str:
    """Return an RFC 3339 UTC timestamp with second precision."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Calculate the SHA-256 of a file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize *value* consistently for identity hashes."""

    return (
        json.dumps(
            json_value(value),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    """Atomically write pretty, strict JSON."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            json_value(value),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, target)


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {Path(path).name}.")
    return value


def write_csv_gzip(path: str | Path, frame: pd.DataFrame) -> None:
    """Write deterministic gzip CSV (stable header, LF, and gzip timestamp)."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline="")
    frame.to_csv(buffer, index=False, lineterminator="\n", na_rep="")
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9
        ) as zipped:
            zipped.write(buffer.getvalue().encode("utf-8"))
    os.replace(temporary, target)


def file_entry(
    path: str | Path,
    *,
    relative_to: str | Path,
    rows: int | None = None,
    columns: list[str] | None = None,
) -> dict[str, Any]:
    file_path = Path(path)
    base = Path(relative_to)
    entry: dict[str, Any] = {
        "path": file_path.relative_to(base).as_posix(),
        "sha256": sha256_file(file_path),
        "size_bytes": file_path.stat().st_size,
    }
    if rows is not None:
        entry["rows"] = int(rows)
    if columns is not None:
        entry["columns"] = list(columns)
    return entry


def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": sys.platform,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }


def source_tree_revision(repository_root: str | Path) -> dict[str, Any]:
    """Identify the runnable package source without relying on Git metadata."""

    root = Path(repository_root).resolve()
    candidates = [root / "pyproject.toml", *sorted((root / "src").rglob("*.py"))]
    files = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in candidates
        if path.is_file()
    }
    if not files:
        raise ValueError("No package source files were found for code revision hashing.")
    return {
        "kind": "source_tree_sha256",
        "value": canonical_json_hash(files),
        "file_count": len(files),
        "files": files,
        "git_commit": None,
        "note": "Content identity only; no Git commit or earlier creation date is asserted.",
    }
