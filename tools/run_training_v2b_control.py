"""One frozen worker invocation; authorization and sequence belong to its controller."""
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
    arguments = parser.parse_args(argv)
    # Parse only the explicitly named immutable contract before importing torch.
    path = Path(arguments.contract)
    if not path.is_absolute() or path != path.resolve():
        raise ValueError("contract must be an explicit canonical absolute path")
    if any(part.casefold() in {"confirmatory", "test"} or
           part.casefold().startswith("confirmatory_") for part in path.parts):
        raise ValueError("contract path crosses a forbidden role")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as stream:
        raw = stream.read()
    if hashlib.sha256(raw).hexdigest() != arguments.contract_sha256:
        raise ValueError("contract complete-file SHA-256 mismatch before imports")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    # The process-start environment is validated, never silently repaired.
    sys.dont_write_bytecode = True
    from sparse_rtdetr.baseline.training_v2b_control import run_worker
    from sparse_rtdetr.baseline.training_v2b_evidence import strict_json_loads
    contract = strict_json_loads(raw)
    reference = {"path": str(path), "sha256": arguments.contract_sha256,
                 "size_bytes": len(raw), "sha256_scope": "complete_file_bytes"}
    try:
        result = run_worker(contract, contract_reference=reference)
    except BaseException as exc:
        print(json.dumps({"status": "STOP_NO_RETRY", "failure_type": type(exc).__name__,
                          "failure": str(exc)}, ensure_ascii=False, allow_nan=False), flush=True)
        return 2
    print(json.dumps({"status": result["status"], "run_id": result["run_id"],
                      "stage": result["stage"], "output_dir": contract["output_dir"],
                      "guardian_clearance_required_after_worker_exit": True}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
