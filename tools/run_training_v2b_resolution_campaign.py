"""Prepare or run the fixed 896 x 8 x 2 campaign against the sealed 640 B."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    prep = actions.add_parser("prepare")
    prep.add_argument("--campaign-id", required=True)
    prep.add_argument("--output-root", required=True)
    prep.add_argument("--matched-completion", required=True)
    prep.add_argument("--matched-completion-sha256", required=True)
    prep.add_argument("--policy-evidence-dir", required=True)
    prep.add_argument("--authorization", required=True)
    prep.add_argument("--authorization-sha256", required=True)
    prep.add_argument("--setter-mode", choices=("external_admin_acknowledged",),
                      default="external_admin_acknowledged")
    prep.add_argument("--external-clock-receipt", required=True)
    prep.add_argument("--external-clock-receipt-sha256", required=True)
    run = actions.add_parser("run")
    run.add_argument("--campaign", required=True)
    run.add_argument("--campaign-sha256", required=True)
    args = parser.parse_args(argv)
    from sparse_rtdetr.baseline.training_v2b_campaign import CampaignError, file_reference
    from sparse_rtdetr.baseline.training_v2b_resolution_campaign import (
        make_resolution_campaign, run_resolution_campaign,
    )
    if args.action == "prepare":
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise CampaignError("CPU preparation requires empty CUDA_VISIBLE_DEVICES")
        references = {}
        for name, path, expected in (
            ("authorization", args.authorization, args.authorization_sha256),
            ("matched", args.matched_completion, args.matched_completion_sha256),
            ("external", args.external_clock_receipt, args.external_clock_receipt_sha256),
        ):
            reference = file_reference(path)
            if reference["sha256"] != expected:
                raise CampaignError(name + " complete-file SHA256 differs")
            references[name] = reference
        from sparse_rtdetr.baseline.training_v2b_admission import build_policy_bundle
        policy = build_policy_bundle(
            args.policy_evidence_dir, setter_mode="external_admin_acknowledged",
            authorization_reference=references["authorization"],
            external_clock_receipt=references["external"],
        )
        prepared = make_resolution_campaign(
            repo_root=ROOT, campaign_id=args.campaign_id, output_root=args.output_root,
            matched_control_reference=references["matched"], policy_bundle=policy,
            authorization_reference=references["authorization"],
        )
        print(json.dumps({"status": "PREPARED_NO_GPU_EXECUTED",
                          "campaign_reference": prepared["campaign_reference"]}, sort_keys=True), flush=True)
        return 0
    reference = file_reference(args.campaign)
    if reference["sha256"] != args.campaign_sha256:
        raise CampaignError("campaign bytes differ from the requested identity")
    print(json.dumps(run_resolution_campaign(reference), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
