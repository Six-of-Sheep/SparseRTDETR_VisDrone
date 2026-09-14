"""Run one immutable development inference contract under native admission."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--contract-sha256", required=True)
    args = parser.parse_args(argv)
    path = Path(args.contract)
    if not path.is_absolute() or path != path.resolve(strict=True):
        raise ValueError("contract must be an exact canonical absolute path")
    if any(part.casefold() in {"confirmatory", "test"}
           or part.casefold().startswith("confirmatory_") for part in path.parts):
        raise ValueError("contract crosses a forbidden data role")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        raw = stream.read()
    if hashlib.sha256(raw).hexdigest() != args.contract_sha256:
        raise ValueError("contract complete-file SHA differs before imports")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    sys.dont_write_bytecode = True
    from sparse_rtdetr.baseline.training_v2b_evidence import strict_json_loads
    from sparse_rtdetr.baseline.training_v2b_cross_eval import run_worker
    contract = strict_json_loads(raw)
    if contract.get("repo_root") != str(root):
        raise ValueError("inference worker imported from a different checkout")
    ref = {"path": str(path), "sha256": args.contract_sha256,
           "size_bytes": len(raw), "sha256_scope": "complete_file_bytes"}
    try:
        result = run_worker(contract, contract_reference=ref)
    except BaseException as exc:
        print(json.dumps({"status": "STOP_NO_RETRY", "error_type": type(exc).__name__,
                          "error": str(exc)}, ensure_ascii=False, allow_nan=False), flush=True)
        return 2
    print(json.dumps({"status": result["status"], "run_id": result["run_id"],
                      "cell_key": contract["cell_key"], "output_dir": contract["output_dir"],
                      "guardian_clearance_required_after_worker_exit": True}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
