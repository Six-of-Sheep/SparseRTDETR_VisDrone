#!/usr/bin/env python3
"""Execute one immutable paired-seed campaign; no resume, retry or extension."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from sparse_rtdetr.baseline.training_v2b_campaign import file_reference
from sparse_rtdetr.baseline.training_v2b_replication_campaign import _json, run_replication_campaign


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--freeze-sha256", required=True)
    args = parser.parse_args()
    reference = file_reference(Path(args.freeze).absolute())
    if reference["sha256"] != args.freeze_sha256:
        raise ValueError("freeze SHA differs from the explicit invocation")
    freeze = _json(reference)
    result = run_replication_campaign(freeze, freeze_reference=reference)
    print(json.dumps({"status": result["status"], "campaign_id": result["campaign_id"],
                      "result_reference": file_reference(Path(freeze["output_root"])/"campaign-result.json")}))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(json.dumps({"status": "STOP_NO_RETRY", "type": type(exc).__name__, "message": str(exc)}))
        sys.exit(1)
