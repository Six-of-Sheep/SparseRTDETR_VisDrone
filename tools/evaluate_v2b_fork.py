#!/usr/bin/env python3
"""Read-only raw/EMA development evaluation of any v2b checkpoint.

Thin wrapper around ``evaluate_v2b_checkpoint.evaluate_entry``: identical
model construction, development data binding, postprocessing, primary and
COCO evaluators. The only difference is that the checkpoint entry is built
from the checkpoint itself (seed, resolution, epoch from its run binding and
engine) instead of being one of the twelve sealed REV1 contract entries, so
fork/continuation outputs can be scored under the same protocol.

Result files keep the evaluator's schema (its ``evaluation_contract`` label
names the protocol, not a contract entry); a ``.provenance.json`` sidecar
records that the entry was derived here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_v2b_checkpoint import authorities, evaluate_entry  # noqa: E402


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    import torch
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-id", required=True, help="e.g. seed1-r896-e060-lrd48")
    parser.add_argument("--source-campaign", required=True, help="free-form lineage label")
    parser.add_argument("--selector", choices=("raw", "ema", "both"), default="both")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--contract-dir", type=Path, default=root / "contracts/v2b/rev001")
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve(strict=True)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    config = payload["binding"]["config"]
    engine = payload["engine"]
    if engine.get("epoch_active"):
        raise SystemExit("checkpoint is not at an epoch boundary")
    info = {"checkpoint_id": args.checkpoint_id, "sha256": sha_file(checkpoint),
            "epoch": engine["epoch"], "seed": config["seed"], "resolution": config["input_size"],
            "size_bytes": checkpoint.stat().st_size, "source_campaign": args.source_campaign}
    del payload
    args.output_dir.mkdir(parents=True, exist_ok=True)
    auth = authorities(args.contract_dir.resolve(strict=True))
    selectors = ("raw", "ema") if args.selector == "both" else (args.selector,)
    for selector in selectors:
        entry = {"checkpoint": info, "runtime_locator": {"path": str(checkpoint)},
                 "weight_selector": selector}
        started = time.time()
        result = evaluate_entry(args.repo_root.resolve(), auth, entry, args.output_dir, args.device)
        sidecar = args.output_dir / f"{args.checkpoint_id}--{selector}.provenance.json"
        with sidecar.open("x") as stream:
            json.dump({"derived_entry": entry, "tool": str(Path(__file__).resolve()),
                       "tool_sha256": sha_file(Path(__file__).resolve()),
                       "result_sha256": result["result_sha256"],
                       "seconds": round(time.time() - started, 1)}, stream, indent=1, sort_keys=True)
        coco = result["coco_secondary"]
        print(f"{args.checkpoint_id} {selector}: primary AP {result['primary']['AP']:.4f} | "
              f"COCO AP {coco['AP']:.4f} AP50 {coco['AP50']:.4f} AP75 {coco['AP75']:.4f} "
              f"APS {coco['AP_small']:.4f} APM {coco['AP_medium']:.4f} APL {coco['AP_large']:.4f} "
              f"({time.time() - started:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
