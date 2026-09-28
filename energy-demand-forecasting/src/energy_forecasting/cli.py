from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiment import run_experiment
from .sensitivity import (
    run_sensitivity_experiment,
    verify_sensitivity_result_directory,
)
from .verification import verify_input_directory, verify_result_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="energy-forecast",
        description="Run a reproducible forecast comparison across onboarding conditions.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify_inputs = subparsers.add_parser(
        "verify-inputs", help="verify the complete producer bundle without training"
    )
    verify_inputs.add_argument("--input-dir", required=True, type=Path)

    run = subparsers.add_parser("run", help="run the configured experiment")
    run.add_argument("--reference", required=True, type=Path)
    run.add_argument("--corrupted", required=True, type=Path)
    run.add_argument("--remediated", required=True, type=Path)
    run.add_argument("--producer-manifest", required=True, type=Path)
    run.add_argument("--fault-manifest", required=True, type=Path)
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument(
        "--weather",
        type=Path,
        help="producer weather.csv.gz (defaults to the file next to --reference when the config enables weather features)",
    )

    sensitivity = subparsers.add_parser(
        "run-sensitivity",
        help="run the separate daily-origin direct 1-24h sensitivity analysis",
    )
    sensitivity.add_argument("--reference", required=True, type=Path)
    sensitivity.add_argument("--corrupted", required=True, type=Path)
    sensitivity.add_argument("--remediated", required=True, type=Path)
    sensitivity.add_argument("--producer-manifest", required=True, type=Path)
    sensitivity.add_argument("--fault-manifest", required=True, type=Path)
    sensitivity.add_argument("--config", required=True, type=Path)
    sensitivity.add_argument("--output-dir", required=True, type=Path)
    sensitivity.add_argument("--weather", type=Path, help="producer weather.csv.gz")

    verify_results = subparsers.add_parser(
        "verify-results", help="verify a completed result bundle"
    )
    verify_results.add_argument("--result-dir", required=True, type=Path)
    verify_results.add_argument(
        "--input-dir",
        type=Path,
        help="optionally bind targets, hashes, row counts, and scales to a Repo A bundle",
    )
    verify_sensitivity = subparsers.add_parser(
        "verify-sensitivity-results",
        help="verify a completed direct 1-24h sensitivity bundle",
    )
    verify_sensitivity.add_argument("--result-dir", required=True, type=Path)
    verify_sensitivity.add_argument(
        "--input-dir",
        type=Path,
        help="optionally bind targets, hashes, row counts, and scales to a Repo A bundle",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "verify-inputs":
        verified = verify_input_directory(args.input_dir)
        print(json.dumps(verified.summary, indent=2, sort_keys=True))
        return 0
    if args.command == "run":
        manifest = run_experiment(
            args.reference,
            args.corrupted,
            args.remediated,
            args.config,
            args.output_dir,
            producer_manifest_path=args.producer_manifest,
            fault_manifest_path=args.fault_manifest,
            weather_path=args.weather,
        )
        print(
            "Completed forecast experiment: "
            f"{manifest['common_prediction_count']} common evaluation keys"
        )
        print(f"Results: {args.output_dir.resolve()}")
        return 0
    if args.command == "run-sensitivity":
        manifest = run_sensitivity_experiment(
            args.reference,
            args.corrupted,
            args.remediated,
            args.config,
            args.output_dir,
            producer_manifest_path=args.producer_manifest,
            fault_manifest_path=args.fault_manifest,
            weather_path=args.weather,
        )
        print(
            "Completed separate 1-24h sensitivity: "
            f"{manifest['common_forecast_key_count']} common forecast keys"
        )
        print(f"Sensitivity results: {args.output_dir.resolve()}")
        return 0
    if args.command == "verify-results":
        summary = verify_result_directory(args.result_dir, input_dir=args.input_dir)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "verify-sensitivity-results":
        summary = verify_sensitivity_result_directory(
            args.result_dir, input_dir=args.input_dir
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
