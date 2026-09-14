"""Run the explicitly bound development 2x2 evidence campaign once."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--campaign-sha256", required=True)
    args = parser.parse_args(argv)
    path = Path(args.campaign)
    if not path.is_absolute() or path != path.resolve(strict=True):
        raise ValueError("campaign must be an exact canonical absolute path")
    if any(part.casefold() in {"confirmatory", "test"}
           or part.casefold().startswith("confirmatory_") for part in path.parts):
        raise ValueError("campaign crosses a forbidden data role")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        raw = stream.read()
    if hashlib.sha256(raw).hexdigest() != args.campaign_sha256:
        raise ValueError("campaign complete-file SHA differs before imports")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    sys.dont_write_bytecode = True
    from sparse_rtdetr.baseline.training_v2b_evidence import strict_json_loads
    from sparse_rtdetr.baseline.training_v2b_evidence_campaign import run_evidence_campaign
    campaign = strict_json_loads(raw)
    if campaign.get("source", {}).get("repo_root") != str(root):
        raise ValueError("controller imported from a different checkout")
    ref = {"path": str(path), "sha256": args.campaign_sha256,
           "size_bytes": len(raw), "sha256_scope": "complete_file_bytes"}
    try:
        result = run_evidence_campaign(campaign, campaign_reference=ref)
    except BaseException as exc:
        print(json.dumps({"status": "STOP_NO_RETRY", "error_type": type(exc).__name__,
                          "error": str(exc)}, ensure_ascii=False, allow_nan=False), flush=True)
        return 2
    print(json.dumps({"status": result["status"], "campaign_id": result["campaign_id"],
                      "completed_cells": list(result["completed"]), "training_started": False}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
