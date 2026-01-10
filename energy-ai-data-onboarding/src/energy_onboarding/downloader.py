"""Pinned, streaming source download with fail-closed verification."""

from __future__ import annotations

import hashlib
import os
import ssl
import urllib.request
from pathlib import Path
from typing import Any, BinaryIO, Callable

from .io_utils import read_json, sha256_file, utc_now, write_json


class SourceVerificationError(RuntimeError):
    """Raised when an authoritative source fails its declared identity checks."""


def verify_file(
    path: str | Path, *, expected_size: int, expected_sha256: str
) -> dict[str, Any]:
    """Verify one file and return observed identity fields."""

    file_path = Path(path)
    if not file_path.is_file():
        raise SourceVerificationError(f"Source file is missing: {file_path.name}")
    observed_size = file_path.stat().st_size
    observed_sha256 = sha256_file(file_path)
    if observed_size != int(expected_size):
        raise SourceVerificationError(
            f"Size mismatch for {file_path.name}: expected {expected_size}, "
            f"observed {observed_size}."
        )
    if observed_sha256.lower() != str(expected_sha256).lower():
        raise SourceVerificationError(
            f"SHA-256 mismatch for {file_path.name}: expected {expected_sha256}, "
            f"observed {observed_sha256}."
        )
    return {
        "name": file_path.name,
        "local_relative_path": file_path.name,
        "observed_size_bytes": observed_size,
        "observed_sha256": observed_sha256,
        "verification_status": "verified",
    }


def _default_open(url: str) -> BinaryIO:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "energy-ai-data-onboarding/0.1 (+public-data research)"},
    )
    # Use the system trust store. Deliberately do not disable TLS verification.
    return urllib.request.urlopen(  # noqa: S310 - URLs are pinned by the manifest
        request, timeout=60, context=ssl.create_default_context()
    )


def _download_one(
    record: dict[str, Any],
    target: Path,
    *,
    force: bool,
    opener: Callable[[str], BinaryIO],
) -> dict[str, Any]:
    required = {"name", "url", "expected_size_bytes", "expected_sha256"}
    missing = sorted(required.difference(record))
    if missing:
        raise ValueError(f"Source record is missing fields: {missing}")
    if target.exists() and not force:
        return verify_file(
            target,
            expected_size=int(record["expected_size_bytes"]),
            expected_sha256=str(record["expected_sha256"]),
        )

    temporary = target.with_suffix(target.suffix + ".part")
    digest = hashlib.sha256()
    size = 0
    try:
        with opener(str(record["url"])) as response, temporary.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        expected_size = int(record["expected_size_bytes"])
        expected_sha = str(record["expected_sha256"]).lower()
        if size != expected_size or digest.hexdigest().lower() != expected_sha:
            raise SourceVerificationError(
                f"Downloaded identity mismatch for {target.name}: expected "
                f"{expected_size} bytes/{expected_sha}, observed "
                f"{size} bytes/{digest.hexdigest()}."
            )
        os.replace(temporary, target)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    return verify_file(
        target,
        expected_size=int(record["expected_size_bytes"]),
        expected_sha256=str(record["expected_sha256"]),
    )


def download_sources(
    manifest_path: str | Path,
    data_dir: str | Path,
    *,
    force: bool = False,
    opener: Callable[[str], BinaryIO] | None = None,
) -> Path:
    """Download all files from a pinned source manifest.

    The source declaration is never modified. A separate immutable observation
    manifest is created inside *data_dir*. An existing observation manifest is
    accepted only when it is byte-equivalent to the new verification result.
    """

    source_manifest = read_json(manifest_path)
    records = source_manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("Source manifest must contain a non-empty files list.")
    destination = Path(data_dir)
    destination.mkdir(parents=True, exist_ok=True)
    open_stream = opener or _default_open

    observed: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Every source file declaration must be an object.")
        name = str(record.get("name", ""))
        if not name or Path(name).name != name:
            raise ValueError("Source file names must be simple basenames.")
        target = destination / name
        identity = _download_one(
            record, target, force=force, opener=open_stream
        )
        identity["source_url"] = record["url"]
        observed.append(identity)

    observation = {
        "schema_version": "1.0",
        "source_manifest": Path(manifest_path).name,
        "source_manifest_sha256": sha256_file(manifest_path),
        "dataset_title": source_manifest.get("dataset_title"),
        "version": source_manifest.get("version"),
        "doi": source_manifest.get("doi"),
        "license": source_manifest.get("license"),
        "download_time_utc": utc_now(),
        "verification_status": "verified",
        "observed_files": observed,
    }
    observation_path = destination / "source_download_manifest.json"
    if observation_path.exists() and not force:
        # Preserve the original download timestamp while re-verifying content.
        prior = read_json(observation_path)
        prior_comparable = dict(prior)
        current_comparable = dict(observation)
        prior_comparable.pop("download_time_utc", None)
        current_comparable.pop("download_time_utc", None)
        if prior_comparable != current_comparable:
            raise SourceVerificationError(
                "Existing source_download_manifest.json conflicts with current "
                "verification; use --force only after reviewing the source."
            )
        return observation_path
    write_json(observation_path, observation)
    return observation_path
