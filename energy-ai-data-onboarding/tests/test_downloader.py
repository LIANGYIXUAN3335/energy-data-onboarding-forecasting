from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

from energy_onboarding.downloader import (
    SourceVerificationError,
    download_sources,
    verify_file,
)


def test_verify_file_pass_and_fail(tmp_path: Path) -> None:
    payload = b"pinned public bytes\n"
    path = tmp_path / "source.csv"
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    observed = verify_file(path, expected_size=len(payload), expected_sha256=digest)
    assert observed["verification_status"] == "verified"
    with pytest.raises(SourceVerificationError, match="Size mismatch"):
        verify_file(path, expected_size=len(payload) + 1, expected_sha256=digest)
    with pytest.raises(SourceVerificationError, match="SHA-256 mismatch"):
        verify_file(path, expected_size=len(payload), expected_sha256="0" * 64)


def test_streaming_download_is_verified_and_manifest_is_immutable(tmp_path: Path) -> None:
    payload = b"small deterministic fixture"
    source = {
        "dataset_title": "fixture",
        "version": "1",
        "doi": "fixture",
        "license": "fixture-only",
        "files": [
            {
                "name": "fixture.csv",
                "url": "https://example.invalid/fixture.csv",
                "expected_size_bytes": len(payload),
                "expected_sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    }
    manifest = tmp_path / "source.json"
    manifest.write_text(json.dumps(source), encoding="utf-8")
    output = tmp_path / "raw"
    result = download_sources(
        manifest, output, opener=lambda _url: io.BytesIO(payload)
    )
    assert result.is_file()
    assert (output / "fixture.csv").read_bytes() == payload
    first = json.loads(result.read_text())
    second_path = download_sources(
        manifest, output, opener=lambda _url: io.BytesIO(b"unused")
    )
    assert json.loads(second_path.read_text()) == first
    (output / "fixture.csv").write_bytes(b"tampered")
    with pytest.raises(SourceVerificationError):
        download_sources(manifest, output, opener=lambda _url: io.BytesIO(payload))
