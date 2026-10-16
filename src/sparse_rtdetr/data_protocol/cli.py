"""CLI boundary for the future authorized train/val production conversion."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .converter import (
    ALLOWED_SPLITS,
    AUTHORIZATION_ENV,
    ConversionContractError,
    load_protocol_v2_config,
    run_conversion,
)
from .schema import canonical_json_bytes


def _default_protocol_config() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "visdrone_protocol_v2.json"


def _split_arg(value: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in value.split(",") if part.strip())
    if set(values) != set(ALLOWED_SPLITS) or len(values) != 2:
        raise argparse.ArgumentTypeError("--split must be exactly train,val")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="visdrone-converter")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    subparsers.add_parser("contract-check", help="check the data-free converter contract")
    run = subparsers.add_parser("run", help="authorized train/val conversion")
    run.add_argument("--data-root", required=True, type=Path)
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument("--protocol-config", required=True, type=Path)
    run.add_argument("--split", required=True, type=_split_arg)
    return parser


def contract_check() -> dict[str, object]:
    """Perform no-data, no-output, no-framework contract validation."""

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ConversionContractError("contract-check requires CUDA_VISIBLE_DEVICES=''")
    config_path = _default_protocol_config()
    config = load_protocol_v2_config(config_path)
    if config.get("test_access_allowed") is not False:
        raise ConversionContractError("test access is not disabled")
    return {
        "status": "PASS",
        "mode": "contract-check",
        "protocol_id": config["protocol_id"],
        "cuda_visible_devices": "",
        "real_data_accessed": False,
        "output_directory_created": False,
        "torch_imported": False,
        "torchvision_imported": False,
        "model_constructed": False,
        "dataset_or_dataloader_constructed": False,
        "network_requested": False,
        "project_test_split_historically_observed": True,
        "dataset_test_accessed_by_this_process": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.mode == "contract-check":
            print(canonical_json_bytes(contract_check()).decode("utf-8"))
            return 0
        result = run_conversion(args.data_root, args.output_dir, args.protocol_config, args.split)
        print(canonical_json_bytes(result).decode("utf-8"))
        return 0
    except ConversionContractError as error:
        print(f"visdrone-converter: {type(error).__name__}: {error}", file=os.sys.stderr)
        return 2
    except Exception as error:
        print(f"visdrone-converter: {type(error).__name__}: {error}", file=os.sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
