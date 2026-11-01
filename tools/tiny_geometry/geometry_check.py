"""Real training-side input check for the tiny-band geometry term (CPU only; CUDA hidden).

One pre-fixed micro-batch (pass-1 window 0, first 8 images, seed0, the production training
transform) is run through the real model and criterion once per mode with gradients enabled:
the native arm (no term), the A1 control and the A2 method.  Each mode records the loss
family, the observed band activity, the draw ledger, the per-family observation rows and
the gradient digest.  ``--compare`` then asserts that A1/A2 change exactly the non-DN
``loss_giou`` keys (loss_bbox and every other loss bitwise equal to the native arm), that
the mechanism is active (different gradient digests) and that the audit rows carry the
declared observation fields.

usage:
  <frozen python> -I tools/geometry_check.py --source-root <release>/source --contract <c.json> --mode A0 --out <dir>/A0.json
  <frozen python> -I tools/geometry_check.py --compare <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch


def verified(ref):
    raw = Path(ref["path"]).read_bytes()
    if len(raw) != ref["bytes"] or hashlib.sha256(raw).hexdigest() != ref["sha256"]:
        raise SystemExit("bound reference changed: " + str(ref["path"]))
    return json.loads(raw)


def build_once(root, contract, mode):
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))
    from sparse_rtdetr.protocol_r18.model import Config, build, fp32_outputs
    from sparse_rtdetr.protocol_r18 import checkpoint as native_state
    from sparse_rtdetr.protocol_r18 import slice_finetune as ft
    from sparse_rtdetr.protocol_r18 import tiny_geometry_arm as arm
    from sparse_rtdetr.protocol_r18.engine import tensor_digest
    from sparse_rtdetr.localization.data import training_loader

    binding = verified(contract["parent_binding"])
    parent = contract["parent_checkpoint"]
    config = Config()
    model, criterion, _ = build(config, synthetic=True)
    payload = native_state.read(parent["path"], parent["sha256"], binding)
    initial = ft.initialize_from_parent(model, payload, expected_binding=binding)
    del payload
    geo_mode = None if mode == "A0" else mode
    criterion.tiny_geometry_declaration = arm.declaration(geo_mode)
    term = None
    if geo_mode is not None:
        term = arm.install(model, criterion, geo_mode)
    data_spec = contract["data"]["train_core"]
    data = ft.Dataset("A2", manifest_path=data_spec["manifest"],
                      images_path=data_spec["roots"]["images"],
                      annotations_path=data_spec["roots"]["annotations"],
                      root_binding=data_spec["roots"])
    loader = training_loader(data, epoch=ft.clock(1), completed_windows=0, window_limit=1)
    x, t, tr = next(iter(loader))
    batch = x[:8]
    subset = t[:8]
    model.train()
    torch.manual_seed(20261101)
    outputs = model(batch, targets=subset)
    losses = criterion(fp32_outputs(outputs), subset)
    rows = [row for row in criterion.last_diagnostics
            if isinstance(row, dict) and "geo_pairs" in row]
    total = sum(losses.values())
    total.backward()
    digest = tensor_digest((name, p.grad) for name, p in model.named_parameters() if p.grad is not None)
    norm = math.sqrt(sum(float((p.grad ** 2).sum()) for p in model.parameters() if p.grad is not None))
    value = {"schema": "r18-tiny-geometry-check-v1", "mode": mode, "geo_mode": geo_mode,
             "image_ids": [row["image_id"] for row in tr[:8]],
             "initial_model_sha256": initial,
             "loss_keys": sorted(losses.keys()),
             "losses": {key: float(val) for key, val in sorted(losses.items())},
             "geo_rows": rows,
             "geo_totals": {"rows": len(rows),
                            "pairs": sum(int(row["geo_pairs"]) for row in rows),
                            "tiny_pairs": sum(int(row["geo_tiny_pairs"]) for row in rows),
                            "tiny_zero_iou": sum(int(row["geo_tiny_zero_iou"]) for row in rows),
                            "native_sum": sum(float(row["geo_tiny_native_sum"]) for row in rows),
                            "applied_sum": sum(float(row["geo_tiny_applied_sum"]) for row in rows),
                            "shifted_calls": sum(1 for row in rows if row.get("geo_shift") is not None)},
             "draws": 0 if term is None else int(term.state_dict()["draws"]),
             "gradient": {"parameters_with_gradient": len([1 for p in model.parameters() if p.grad is not None]),
                          "sha256": digest, "norm": norm},
             "cuda_initialized": bool(torch.cuda.is_initialized()),
             "elapsed_seconds": None}
    return value, losses


def run_mode(args):
    root = Path(args.source_root).resolve()
    if torch.cuda.is_initialized():
        raise SystemExit("CUDA must stay hidden for the check")
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    value, _ = build_once(root, contract, args.mode)
    value["elapsed_seconds"] = time.monotonic() - started
    target = Path(args.out)
    if target.exists():
        raise SystemExit("check output already exists")
    body = json.dumps(value, indent=1, ensure_ascii=False, sort_keys=True)
    target.write_bytes(body.encode("utf-8"))
    print(json.dumps({"status": "CHECK-OK", "mode": args.mode, "loss_keys": len(value["loss_keys"]),
                      "tiny_pairs": value["geo_totals"]["tiny_pairs"], "draws": value["draws"],
                      "grad_norm": value["gradient"]["norm"],
                      "sha256": hashlib.sha256(body.encode()).hexdigest()}))


def compare(args):
    directory = Path(args.compare)
    values = {mode: json.loads((directory / (mode + ".json")).read_text(encoding="utf-8"))
              for mode in ("A0", "A1", "A2")}
    report = {"schema": "r18-tiny-geometry-check-compare-v1"}
    base = values["A0"]
    for mode in ("A1", "A2"):
        other = values[mode]
        if other["loss_keys"] != base["loss_keys"]:
            raise SystemExit("loss key family differs between " + mode + " and A0")
        differing = sorted(key for key in base["losses"]
                           if abs(base["losses"][key] - other["losses"][key]) > 0.0)
        expected = sorted(key for key in base["losses"]
                          if key.startswith("loss_giou") and "_dn_" not in key)
        if differing != expected:
            raise SystemExit(("differing loss keys are not exactly the non-DN geo slots",
                              mode, differing, expected))
        report[mode + "_differing_keys"] = differing
        report[mode + "_geo_totals"] = other["geo_totals"]
        report[mode + "_draws"] = other["draws"]
    diffs = {mode: base["gradient"]["sha256"] != values[mode]["gradient"]["sha256"]
             for mode in ("A1", "A2")}
    if not all(diffs.values()):
        raise SystemExit("the installed arms do not change the gradient digest")
    report["gradient_digests_differ"] = diffs
    report["a0_geo_totals"] = base["geo_totals"]
    if base["geo_totals"]["applied_sum"] != base["geo_totals"]["native_sum"]:
        raise SystemExit("the native arm must record applied == native")
    for mode in ("A1", "A2"):
        totals = values[mode]["geo_totals"]
        if totals["tiny_pairs"] <= 0 or totals["shifted_calls"] != (1 if mode == "A1" else 0):
            raise SystemExit(("unexpected band activity/draws for " + mode, totals))
        if mode == "A1" and values[mode]["draws"] != totals["shifted_calls"]:
            raise SystemExit("A1 draw ledger differs from the shifted-call count")
        if mode == "A2" and values[mode]["draws"] != 0:
            raise SystemExit("A2 must not draw")
        if totals["applied_sum"] == totals["native_sum"]:
            raise SystemExit(mode + " did not change the applied sums")
    body = json.dumps(report, indent=1, ensure_ascii=False, sort_keys=True)
    target = directory / "COMPARE.json"
    if target.exists():
        raise SystemExit("compare output already exists")
    target.write_bytes(body.encode("utf-8"))
    print(json.dumps({"status": "COMPARE-OK", "sha256": hashlib.sha256(body.encode()).hexdigest()}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root")
    parser.add_argument("--contract")
    parser.add_argument("--mode", choices=("A0", "A1", "A2"))
    parser.add_argument("--out")
    parser.add_argument("--compare")
    args = parser.parse_args()
    if args.compare:
        compare(args)
        return
    if not (args.source_root and args.contract and args.mode and args.out):
        raise SystemExit("need --source-root --contract --mode --out, or --compare")
    run_mode(args)


if __name__ == "__main__":
    main()
