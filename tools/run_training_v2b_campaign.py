"""Prepare or run the explicitly authorized fixed 640 A/B campaign."""
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
    prep.add_argument("--input-spec", required=True)
    prep.add_argument("--policy-evidence-dir", required=True)
    prep.add_argument("--authorization", required=True)
    prep.add_argument("--authorization-sha256", required=True)
    prep.add_argument("--setter-mode", choices=("direct", "sudo_n", "external_admin_acknowledged"), required=True)
    prep.add_argument("--external-clock-receipt")
    prep.add_argument("--external-clock-receipt-sha256")
    run = actions.add_parser("run")
    run.add_argument("--campaign", required=True)
    run.add_argument("--campaign-sha256", required=True)
    args = parser.parse_args(argv)
    from sparse_rtdetr.baseline.training_v2b_campaign import (
        CampaignError, file_reference, make_campaign, run_campaign,
    )
    from sparse_rtdetr.baseline.training_v2b_evidence import strict_json_loads
    if args.action == "prepare":
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise CampaignError("CPU preparation requires empty CUDA_VISIBLE_DEVICES")
        from sparse_rtdetr.baseline.training_v2b_admission import build_policy_bundle
        auth = file_reference(args.authorization)
        if auth["sha256"] != args.authorization_sha256:
            raise CampaignError("authorization text identity differs")
        external = None
        if args.setter_mode == "external_admin_acknowledged":
            if not args.external_clock_receipt or not args.external_clock_receipt_sha256:
                parser.error("external administrator mode requires the receipt path and complete-file SHA256")
            external = file_reference(args.external_clock_receipt)
            if external["sha256"] != args.external_clock_receipt_sha256:
                raise CampaignError("external clock receipt bytes differ")
        elif args.external_clock_receipt or args.external_clock_receipt_sha256:
            parser.error("external clock receipt is only valid with external_admin_acknowledged")
        policy = build_policy_bundle(args.policy_evidence_dir, setter_mode=args.setter_mode,
                                     authorization_reference=auth, external_clock_receipt=external)
        prepared = make_campaign(
            repo_root=ROOT, campaign_id=args.campaign_id, output_root=args.output_root,
            input_spec=strict_json_loads(Path(args.input_spec).read_bytes()),
            policy_bundle=policy, authorization_reference=auth,
            sampling_backend="deterministic_gather",
        )
        print(json.dumps({"status": "PREPARED_NO_GPU_EXECUTED",
                          "campaign_reference": prepared["campaign_reference"]}, sort_keys=True), flush=True)
        return 0
    reference = file_reference(args.campaign)
    if reference["sha256"] != args.campaign_sha256:
        raise CampaignError("campaign bytes differ from the requested identity")
    result = run_campaign(reference)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
