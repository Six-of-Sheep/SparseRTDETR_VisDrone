"""Summarize a completed seed-2/896 cell with the five fixed historical endpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys


def _path(value):
    path = Path(value)
    if not path.is_absolute() or path.resolve() != path or ".." in path.parts:
        raise ValueError("path must be explicit, absolute and canonical")
    if any(part.casefold() in {"test", "confirmatory", "visdrone2019-det-test-dev",
                              "visdrone2019-det-test-challenge"}
           or part.casefold().startswith("confirmatory_") for part in path.parts):
        raise ValueError("path crosses a forbidden data role")
    return path


def _completion_reference(path_text, expected_sha):
    if re.fullmatch("[0-9a-f]{64}", expected_sha) is None:
        raise ValueError("complete lowercase completion SHA-256 required")
    path = _path(path_text)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("completion must be a regular file")
        raw = stream.read()
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_ino, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size, after.st_ino, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError("completion changed while read")
    if len(raw) != before.st_size or hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("completion complete-file SHA mismatch before imports")
    return {"path": str(path), "sha256": expected_sha, "size_bytes": len(raw),
            "sha256_scope": "complete_file_bytes"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--completion", required=True)
    parser.add_argument("--completion-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("CPU summary requires empty CUDA_VISIBLE_DEVICES")
        reference = _completion_reference(args.completion, args.completion_sha256)
        output = _path(args.output_dir)
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[name] = "1"
        sys.dont_write_bytecode = True
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "src"))
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        from sparse_rtdetr.baseline.training_v2b_seed2_completion_summary import build_seed2_completion_summary
        result = build_seed2_completion_summary(completion_reference=reference, output_dir=str(output))
    except BaseException as error:
        print(json.dumps({"status": "STOP_NO_RETRY", "failure_type": type(error).__name__,
                          "failure": str(error)}, ensure_ascii=False, allow_nan=False), flush=True)
        return 2
    print(json.dumps({"status": result["summary"]["status"], "summary_reference": result["summary_reference"],
                      "primary_endpoint_epoch": 30, "historical_campaign_status": "STOP_NO_RETRY",
                      "automatic_experiment_authorization": False}, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
