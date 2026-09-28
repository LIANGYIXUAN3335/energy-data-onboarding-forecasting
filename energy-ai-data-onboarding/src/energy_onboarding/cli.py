"""Non-interactive command line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .downloader import SourceVerificationError, download_sources
from .pipeline import run_pipeline, verify_result_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="energy-onboard",
        description="Verify, audit, and publish public energy time-series data.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="download pinned source files")
    download.add_argument("--manifest", required=True)
    download.add_argument("--data-dir", required=True)
    download.add_argument("--force", action="store_true")

    run = commands.add_parser("run", help="run the onboarding pipeline")
    run.add_argument("--config", required=True)
    meter = run.add_mutually_exclusive_group(required=True)
    meter.add_argument("--electricity-csv")
    meter.add_argument("--meter-csv", dest="electricity_csv")
    run.add_argument("--metadata-csv", required=True)
    run.add_argument(
        "--weather-csv",
        help="optional pinned BDG2 weather.csv; publishes weather.csv.gz alongside the conditions",
    )
    run.add_argument("--output-dir", required=True)
    run.add_argument("--fault-cutoff")
    run.add_argument("--seed", type=int)
    run.add_argument("--severity", choices=("low", "medium", "high"))
    run.add_argument(
        "--building-id",
        action="append",
        dest="building_ids",
        help="explicit meter column; repeat for multiple buildings",
    )
    run.add_argument(
        "--fixture-mode",
        action="store_true",
        help="label local test fixtures and skip authoritative source identity checks",
    )

    verify = commands.add_parser("verify", help="verify a completed result directory")
    verify.add_argument("--result-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "download":
            manifest = download_sources(
                args.manifest, args.data_dir, force=bool(args.force)
            )
            print(f"verified source files; observation manifest: {manifest}")
            return 0
        if args.command == "run":
            result = run_pipeline(
                args.config,
                args.electricity_csv,
                args.metadata_csv,
                args.output_dir,
                weather_csv=args.weather_csv,
                fixture_mode=bool(args.fixture_mode),
                fault_cutoff=args.fault_cutoff,
                seed=args.seed,
                severity=args.severity,
                building_ids=args.building_ids,
            )
            gates = result.quality_summary["quality_gates"]
            statuses = ", ".join(
                f"{name}={details['status']}" for name, details in gates.items()
            )
            print(f"published {result.output_dir}; {statuses}")
            print(f"dataset manifest: {result.dataset_manifest}")
            return 0
        if args.command == "verify":
            errors = verify_result_dir(args.result_dir)
            if errors:
                for error in errors:
                    print(f"ERROR: {error}", file=sys.stderr)
                return 2
            print(f"verified result directory: {Path(args.result_dir)}")
            return 0
    except (OSError, ValueError, RuntimeError, AssertionError, SourceVerificationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
