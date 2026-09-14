"""Execute exactly one frozen seed/resolution replication invocation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate contract JSON key: " + key)
        value[key] = item
    return value


def _invalid_constant(value):
    raise ValueError("nonfinite JSON constant is forbidden: " + value)


def _load_contract(path_text, expected_sha256):
    # No torch, model, data-loader or native hardware import precedes this read.
    if type(expected_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("contract SHA-256 must be the complete lowercase digest")
    path = Path(path_text)
    if not path.is_absolute() or path != path.resolve():
        raise ValueError("contract must be an explicit canonical absolute path")
    if any(part.casefold() in {"confirmatory", "test"}
           or part.casefold().startswith("confirmatory_") for part in path.parts):
        raise ValueError("contract path crosses a forbidden data role")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("contract must be a regular file")
        raw = stream.read()
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError("contract changed while it was being read")
    if len(raw) != before.st_size or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("contract complete-file SHA-256 mismatch before imports")
    contract = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    if type(contract) is not dict:
        raise ValueError("worker contract must be a JSON object")
    reference = {"path": str(path), "sha256": expected_sha256, "size_bytes": len(raw),
                 "sha256_scope": "complete_file_bytes"}
    return contract, reference


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--contract-sha256", required=True)
    arguments = parser.parse_args(argv)
    try:
        contract, reference = _load_contract(arguments.contract, arguments.contract_sha256)
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "src"))
        # Suppressing bytecode is explicit; numerical/runtime environment remains validated as supplied.
        sys.dont_write_bytecode = True
        from sparse_rtdetr.baseline.training_v2b_replication_worker import run_replication_worker
        result = run_replication_worker(contract, contract_reference=reference)
    except BaseException as exc:
        print(json.dumps({"status": "STOP_NO_RETRY", "failure_type": type(exc).__name__,
                          "failure": str(exc)}, ensure_ascii=False, allow_nan=False), flush=True)
        return 2
    print(json.dumps({
        "status": result["status"], "run_id": result["run_id"], "invocation_id": result["invocation_id"],
        "cell_id": result["cell_id"], "stage": result["stage"], "output_dir": contract["output_dir"],
        "guardian_clearance_required_after_worker_exit": True,
    }, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
