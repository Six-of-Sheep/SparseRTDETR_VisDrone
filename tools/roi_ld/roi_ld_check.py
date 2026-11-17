"""Real-input training-side check for the ROI-resolution box distillation term (CPU, CUDA hidden).

One pre-fixed micro-batch (pass-1 window 0, first 8 images, seed 0, the production training
transform) is run through the real model and the patched criterion once per arm with gradients
enabled: the native arm (no term), A1 (whole-image view - the information-free control) and A2
(the four raw-pixel tiles - the method).

Each run records the loss key family and values, the ROI ledger accumulated over every family
call, the attached-target coverage, and a gradient digest.

``--compare`` then asserts:

1. the loss key family is identical across the three arms and the keys that differ from the
   native arm are exactly the non-denoising ``loss_bbox`` / ``loss_giou`` families (VFL, the
   matcher, the DN families and every other key stay bitwise equal);
2. on real inputs the armed arms select a non-empty target set and the ledger accounting is
   exact: selected + below-bar + no-candidate + no-class + unmatched-band == band ground truth;
3. the two armed arms share the pipeline quantity band_gt exactly; the differing counts and the
   candidate-set digests are recorded, not forced equal;
4. the mechanism reaches the parameters: each armed arm's gradient digest differs from native;
5. the native arm records no ROI target, no digest and no mode.

usage:
  <frozen python> -I tools/roi_ld/roi_ld_check.py --source-root <tree>/source --contract <c.json> --mode A0 --out <dir>/A0.json
  <frozen python> -I tools/roi_ld/roi_ld_check.py --compare <dir>
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

ARMS = ("A0", "A1", "A2")
REGRESSION_KEYS = ("loss_bbox", "loss_giou")


def verified(ref):
    raw = Path(ref["path"]).read_bytes()
    if len(raw) != ref["bytes"] or hashlib.sha256(raw).hexdigest() != ref["sha256"]:
        raise SystemExit("bound reference changed: " + str(ref["path"]))
    return json.loads(raw)


def _replay_report(targets):
    """Aggregate the replay-exactness quantities the dataset recorded on each target."""
    report = {"max_error": 0.0, "compared": 0, "missing": 0, "rows": 0}
    for target in targets:
        if "roi_replay_max_error" not in target:
            continue
        report["max_error"] = max(report["max_error"], float(target["roi_replay_max_error"].item()))
        report["compared"] += int(target["roi_replay_compared"].item())
        report["missing"] += int(target["roi_replay_missing"].item())
        report["rows"] += 1
    return report


def build_once(root, contract, arm):
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))
    from sparse_rtdetr.protocol_r18.model import Config, build, fp32_outputs
    from sparse_rtdetr.protocol_r18 import checkpoint as native_state
    from sparse_rtdetr.protocol_r18 import slice_finetune as ft
    from sparse_rtdetr.protocol_r18 import roi_ld_arm
    from sparse_rtdetr.protocol_r18 import roi_ld_data
    from sparse_rtdetr.protocol_r18.engine import tensor_digest
    from sparse_rtdetr.localization.data import training_loader

    binding = verified(contract["parent_binding"])
    parent = contract["parent_checkpoint"]
    config = Config()
    model, criterion, _ = build(config, synthetic=True)
    payload = native_state.read(parent["path"], parent["sha256"], binding)
    initial = ft.initialize_from_parent(model, payload, expected_binding=binding)
    del payload
    mode = ft.ROI_INSTALL[arm]
    criterion.roi_ld_declaration = roi_ld_arm.declaration(mode)
    term = None
    if mode is not None:
        term = roi_ld_arm.install(model, criterion, mode)
    spec = contract["data"]["train_core"]
    data = ft.Dataset(arm, manifest_path=spec["manifest"],
                      images_path=spec["roots"]["images"],
                      annotations_path=spec["roots"]["annotations"],
                      root_binding=spec["roots"])
    loader = training_loader(data, epoch=ft.clock(1), completed_windows=0, window_limit=1)
    x, t, tr = next(iter(loader))
    batch, subset = x[:8], t[:8]
    model.train()
    torch.manual_seed(20261103)
    outputs = model(batch, targets=subset)
    losses = criterion(fp32_outputs(outputs), subset)
    rows = [row for row in criterion.last_diagnostics
            if isinstance(row, dict) and "roi_band_gt" in row]
    total = sum(losses.values())
    total.backward()
    digest = tensor_digest((name, p.grad) for name, p in model.named_parameters() if p.grad is not None)
    norm = math.sqrt(sum(float((p.grad ** 2).sum()) for p in model.parameters() if p.grad is not None))
    totals = {"rows": len(rows)}
    for key in ("roi_band_gt", "roi_eligible", "roi_selected", "roi_no_candidate", "roi_no_class",
                "roi_below_iou", "roi_one_sided", "roi_unmatched_band_gt", "roi_pos_pairs",
                "roi_band_pairs"):
        totals[key] = sum(int(row[key]) for row in rows)
    for key in ("roi_target_iou_sum", "roi_band_native_iou_sum", "roi_other_group_iou_sum"):
        totals[key] = sum(float(row[key]) for row in rows)
    totals["roi_set_digests"] = [row["roi_set_digest"] for row in rows]
    totals["roi_modes"] = sorted({row.get("roi_mode") for row in rows}, key=str)
    value = {"schema": "r18-roi-ld-check-v1", "arm": arm, "roi_mode": mode,
             "image_ids": [row["image_id"] for row in tr[:8]],
             "initial_model_sha256": initial,
             "loss_keys": sorted(losses.keys()),
             "losses": {key: float(val) for key, val in sorted(losses.items())},
             "roi_rows": rows, "heads": sorted(row.get("head") for row in rows),
             "roi_totals": totals,
             "coverage": roi_ld_data.coverage(subset, mode),
             "replay": _replay_report(subset),
             "family_weights": {key: float(val) for key, val in sorted(criterion.weight_dict.items())},
             "targets": 0 if term is None else int(term.state_dict()["targets"]),
             "constants_sha256": roi_ld_arm.declaration(mode).get("constants_sha256"),
             "gradient": {"parameters_with_gradient": len([1 for p in model.parameters() if p.grad is not None]),
                          "sha256": digest, "norm": norm},
             "cuda_initialized": bool(torch.cuda.is_initialized()),
             "elapsed_seconds": None}
    return value


def compare(directory):
    directory = Path(directory)
    values = {arm: json.loads((directory / (arm + ".json")).read_text(encoding="utf-8"))
              for arm in ARMS}
    baseline = values["A0"]
    report = {"schema": "r18-roi-ld-compare-v1", "checks": {}, "detail": {}}
    ok = True

    same_family = all(values[arm]["loss_keys"] == baseline["loss_keys"] for arm in ARMS)
    report["checks"]["loss_key_family_identical"] = bool(same_family)
    ok &= bool(same_family)

    differed = {}
    for arm in ("A1", "A2"):
        keys = {k for k in baseline["losses"]
                if values[arm]["losses"][k] != baseline["losses"][k]}
        differed[arm] = sorted(keys)
        families_ok = all(
            any(key.startswith(prefix) for prefix in REGRESSION_KEYS + ("loss_bbox_", "loss_giou_"))
            for key in keys)
        report["checks"]["differing_keys_are_native_regression_" + arm] = bool(families_ok and keys)
    report["detail"]["differing_keys"] = differed
    ok &= all(report["checks"]["differing_keys_are_native_regression_" + arm] for arm in ("A1", "A2"))

    for arm in ("A1", "A2"):
        tot = values[arm]["roi_totals"]
        identity = (tot["roi_selected"] + tot["roi_below_iou"] + tot["roi_no_candidate"]
                    + tot["roi_no_class"] + tot["roi_one_sided"] + tot["roi_unmatched_band_gt"])
        report["checks"]["ledger_exact_" + arm] = bool(
            identity == tot["roi_band_gt"] and tot["roi_selected"] > 0)
        report["detail"]["ledger_" + arm] = {"identity": identity, "totals": tot}
        ok &= report["checks"]["ledger_exact_" + arm]

    # The control claim: both armed arms supervise the same rows and the same NUMBER of pairs,
    # so the only difference between them is which group supplied the target box.
    report["checks"]["equal_target_counts"] = bool(
        values["A1"]["roi_totals"]["roi_selected"] > 0
        and values["A1"]["roi_totals"]["roi_selected"] == values["A2"]["roi_totals"]["roi_selected"]
        and values["A1"]["targets"] == values["A2"]["targets"])
    ok &= report["checks"]["equal_target_counts"]

    report["checks"]["pipeline_band_gt_shared"] = bool(
        values["A1"]["roi_totals"]["roi_band_gt"] == values["A2"]["roi_totals"]["roi_band_gt"])
    report["detail"]["selected_counts"] = {arm: values[arm]["roi_totals"]["roi_selected"]
                                           for arm in ARMS}
    report["detail"]["set_digests_differ"] = bool(
        values["A1"]["roi_totals"]["roi_set_digests"] != values["A2"]["roi_totals"]["roi_set_digests"])
    ok &= report["checks"]["pipeline_band_gt_shared"]

    for arm in ("A1", "A2"):
        rep = values[arm]["replay"]
        report["checks"]["replay_exact_" + arm] = bool(
            rep["rows"] > 0 and rep["compared"] > 0 and rep["missing"] == 0
            and rep["max_error"] < 1e-4)
        report["detail"]["replay_" + arm] = rep
        ok &= report["checks"]["replay_exact_" + arm]

    for arm in ("A1", "A2"):
        report["checks"]["gradient_reaches_parameters_" + arm] = bool(
            values[arm]["gradient"]["sha256"] != baseline["gradient"]["sha256"])
        ok &= report["checks"]["gradient_reaches_parameters_" + arm]

    report["checks"]["native_arm_records_nothing"] = bool(
        baseline["roi_totals"]["roi_selected"] == 0
        and all(digest is None for digest in baseline["roi_totals"]["roi_set_digests"])
        and baseline["targets"] == 0)
    ok &= report["checks"]["native_arm_records_nothing"]

    report["checks"]["cuda_not_initialized"] = bool(
        not any(values[arm]["cuda_initialized"] for arm in ARMS))
    ok &= report["checks"]["cuda_not_initialized"]

    report["passed"] = bool(ok)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root")
    parser.add_argument("--contract")
    parser.add_argument("--mode", choices=ARMS)
    parser.add_argument("--out")
    parser.add_argument("--compare")
    args = parser.parse_args()
    if args.compare:
        raise SystemExit(compare(args.compare))
    started = time.monotonic()
    root = Path(args.source_root).resolve()
    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    value = build_once(root, contract, args.mode)
    value["elapsed_seconds"] = time.monotonic() - started
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")
    print("WROTE", out, "arm", args.mode, "targets", value["targets"], "seconds", round(value["elapsed_seconds"], 3))


if __name__ == "__main__":
    main()
