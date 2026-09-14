"""Analyze completed SHA-bound 2x2 development outputs on CPU only."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "campaign-sha256", "campaign-result", "campaign-result-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU analysis requires empty CUDA_VISIBLE_DEVICES before imports")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    sys.dont_write_bytecode = True
    from sparse_rtdetr.baseline.training_v2b_campaign import file_reference
    from sparse_rtdetr.baseline.training_v2b_resolution_analysis import run_resolution_analysis
    campaign = file_reference(args.campaign)
    result = file_reference(args.campaign_result)
    if campaign["sha256"] != args.campaign_sha256 or result["sha256"] != args.campaign_result_sha256:
        raise ValueError("analysis input complete-file SHA mismatch")
    report = run_resolution_analysis(campaign_reference=campaign,
                                     campaign_result_reference=result, output_dir=args.output)
    print(json.dumps({"status": report["status"], "seed_gate": report["seed_gate"],
                      "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
