"""Summarize only the six fixed epoch-30 saved development endpoints on CPU."""
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


def _campaign_reference(path_text, expected_sha):
    if re.fullmatch("[0-9a-f]{64}", expected_sha) is None:
        raise ValueError("complete lowercase campaign-result SHA-256 required")
    path = _path(path_text)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("campaign result must be a regular file")
        raw = stream.read()
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_ino, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size, after.st_ino, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError("campaign result changed while read")
    if hashlib.sha256(raw).hexdigest() != expected_sha or len(raw) != before.st_size:
        raise ValueError("campaign result complete-file SHA mismatch before imports")
    return {"path": str(path), "sha256": expected_sha, "size_bytes": len(raw),
            "sha256_scope": "complete_file_bytes"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--campaign-result", required=True)
    parser.add_argument("--campaign-result-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("CPU summary requires empty CUDA_VISIBLE_DEVICES")
        reference = _campaign_reference(args.campaign_result, args.campaign_result_sha256)
        output = _path(args.output_dir)
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "src"))
        sys.dont_write_bytecode = True
        from sparse_rtdetr.baseline.training_v2b_replication_summary import build_replication_summary
        result = build_replication_summary(campaign_result_reference=reference, output_dir=str(output))
    except BaseException as error:
        print(json.dumps({"status": "STOP_NO_RETRY", "failure_type": type(error).__name__,
                          "failure": str(error)}, ensure_ascii=False, allow_nan=False), flush=True)
        return 2
    print(json.dumps({"status": result["summary"]["status"], "summary_reference": result["summary_reference"],
                      "primary_endpoint_epoch": 30, "automatic_experiment_authorization": False},
                     ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
