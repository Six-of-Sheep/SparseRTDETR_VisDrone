"""Real training-side input check for the tiny-band extra-positive term (CPU only; CUDA hidden).

One pre-fixed micro-batch (pass-1 window 0, first 8 images, seed0, the production training
transform) is run through the real model and criterion once per mode with gradients enabled: the
native arm (no term), the A1 mass control and the A2 method.  Each mode records the loss family,
the band activity (band ground-truth count, native matched band pairs, native band overlap and
band native term sums), the extra-positive ledger (admitted pairs, admissible-sibling misses,
extra overlap and extra term sums) and the gradient digest.

``--compare`` then asserts, against the native arm:

1. the loss key family is identical and the differing keys are exactly the non-denoising
   ``loss_bbox`` and ``loss_giou`` keys (the matcher, VFL, DN families and every other key are
   bitwise equal);
2. the extra positive set is non-empty on real inputs and obeys its own declared admission rule
   (overlap bar, own-best-partner accounting: admitted + no-sibling == band ground truth);
3. the additional regression mass equals the recorded extra term sums divided by the frozen
   normaliser with the criterion's own per-family weights (the augmented-set identity);
4. the mechanism reaches the parameters (gradient digests differ from the native arm);
5. the native arm records no extra pair at all.

usage:
  <frozen python> -I tools/positives_check.py --source-root <release>/source --contract <c.json> --mode A0 --out <dir>/A0.json
  <frozen python> -I tools/positives_check.py --compare <dir>
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
    from sparse_rtdetr.protocol_r18 import tiny_positives_arm as arm
    from sparse_rtdetr.protocol_r18.engine import tensor_digest
    from sparse_rtdetr.localization.data import training_loader

    binding = verified(contract["parent_binding"])
    parent = contract["parent_checkpoint"]
    config = Config()
    model, criterion, _ = build(config, synthetic=True)
    payload = native_state.read(parent["path"], parent["sha256"], binding)
    initial = ft.initialize_from_parent(model, payload, expected_binding=binding)
    del payload
    pos_mode = None if mode == "A0" else mode
    criterion.tiny_positives_declaration = arm.declaration(pos_mode)
    term = None
    if pos_mode is not None:
        term = arm.install(model, criterion, pos_mode)
    data_spec = contract["data"]["train_core"]
    data = ft.Dataset("A0", manifest_path=data_spec["manifest"],
                      images_path=data_spec["roots"]["images"],
                      annotations_path=data_spec["roots"]["annotations"],
                      root_binding=data_spec["roots"])
    loader = training_loader(data, epoch=ft.clock(1), completed_windows=0, window_limit=1)
    x, t, tr = next(iter(loader))
    batch = x[:8]
    subset = t[:8]
    model.train()
    torch.manual_seed(20261102)
    outputs = model(batch, targets=subset)
    losses = criterion(fp32_outputs(outputs), subset)
    rows = [row for row in criterion.last_diagnostics
            if isinstance(row, dict) and "pos_pairs" in row]
    total = sum(losses.values())
    total.backward()
    digest = tensor_digest((name, p.grad) for name, p in model.named_parameters() if p.grad is not None)
    norm = math.sqrt(sum(float((p.grad ** 2).sum()) for p in model.parameters() if p.grad is not None))
    keys = sorted(row.get("head") for row in rows)
    value = {"schema": "r18-tiny-positives-check-v1", "mode": mode, "pos_mode": pos_mode,
             "image_ids": [row["image_id"] for row in tr[:8]],
             "initial_model_sha256": initial,
             "loss_keys": sorted(losses.keys()),
             "losses": {key: float(val) for key, val in sorted(losses.items())},
             "pos_rows": rows,
             "heads": keys,
             "num_boxes": float(rows[0]["denominator"]) if rows and "denominator" in rows[0] else None,
             "family_weights": {key: float(val) for key, val in sorted(criterion.weight_dict.items())},
             "pos_totals": {"rows": len(rows),
                            "pairs": sum(int(row["pos_pairs"]) for row in rows),
                            "band_gt": sum(int(row["pos_band_gt"]) for row in rows),
                            "band_pairs": sum(int(row["pos_band_pairs"]) for row in rows),
                            "band_zero_iou": sum(int(row["pos_band_zero_iou"]) for row in rows),
                            "band_native_iou_sum": sum(float(row["pos_band_native_iou_sum"]) for row in rows),
                            "band_native_l1_sum": sum(float(row["pos_band_native_l1_sum"]) for row in rows),
                            "band_native_giou_sum": sum(float(row["pos_band_native_giou_sum"]) for row in rows),
                            "extra_pairs": sum(int(row["pos_extra_pairs"]) for row in rows),
                            "band_gt_no_sibling": sum(int(row["pos_band_gt_no_sibling"]) for row in rows),
                            "extra_iou_sum": sum(float(row["pos_extra_iou_sum"]) for row in rows),
                            "extra_l1_sum": sum(float(row["pos_extra_l1_sum"]) for row in rows),
                            "extra_giou_sum": sum(float(row["pos_extra_giou_sum"]) for row in rows),
                            "modes": sorted({row.get("pos_extra_mode") for row in rows}, key=str)},
             "extras": 0 if term is None else int(term.state_dict()["extras"]),
             "gradient": {"parameters_with_gradient": len([1 for p in model.parameters() if p.grad is not None]),
                          "sha256": digest, "norm": norm},
             "constants_sha256": arm.constants_sha256(),
             "cuda_initialized": bool(torch.cuda.is_initialized()),
             "elapsed_seconds": None}
    return value


def family_key(head, base):
    """The criterion's weight-dictionary key of one family head for one loss base name."""
    if head == "final":
        return base
    if head.startswith("aux"):
        return "%s_aux_%s" % (base, head[3:])
    if head.startswith("enc"):
        return "%s_enc_%s" % (base, head[3:])
    return None


def run_mode(args):
    root = Path(args.source_root).resolve()
    if torch.cuda.is_initialized():
        raise SystemExit("CUDA must stay hidden for the check")
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    value = build_once(root, contract, args.mode)
    value["elapsed_seconds"] = time.monotonic() - started
    target = Path(args.out)
    if target.exists():
        raise SystemExit("check output already exists")
    body = json.dumps(value, indent=1, ensure_ascii=False, sort_keys=True)
    target.write_bytes(body.encode("utf-8"))
    print(json.dumps({"status": "CHECK-OK", "mode": args.mode, "loss_keys": len(value["loss_keys"]),
                      "band_gt": value["pos_totals"]["band_gt"],
                      "extra_pairs": value["pos_totals"]["extra_pairs"],
                      "band_gt_no_sibling": value["pos_totals"]["band_gt_no_sibling"],
                      "extras": value["extras"], "grad_norm": value["gradient"]["norm"],
                      "sha256": hashlib.sha256(body.encode()).hexdigest()}))


def compare(args):
    directory = Path(args.compare)
    values = {mode: json.loads((directory / (mode + ".json")).read_text(encoding="utf-8"))
              for mode in ("A0", "A1", "A2")}
    report = {"schema": "r18-tiny-positives-check-compare-v1"}
    base = values["A0"]
    if base["constants_sha256"] != values["A2"]["constants_sha256"]:
        raise SystemExit("the frozen constants digest differs between modes")
    expected = sorted(key for key in base["losses"]
                      if (key.startswith("loss_bbox") or key.startswith("loss_giou"))
                      and "_dn_" not in key)
    if not expected:
        raise SystemExit("no non-denoising box loss keys found: " + repr(base["loss_keys"]))
    for mode in ("A1", "A2"):
        other = values[mode]
        if other["loss_keys"] != base["loss_keys"]:
            raise SystemExit("loss key family differs between " + mode + " and A0")
        differing = sorted(key for key in base["losses"]
                           if abs(base["losses"][key] - other["losses"][key]) > 0.0)
        if not set(differing) <= set(expected):
            raise SystemExit(("a loss key outside the non-DN box slots changed", mode,
                              sorted(set(differing) - set(expected))))
        affected = set()
        for row in other["pos_rows"]:
            if int(row["pos_extra_pairs"]) <= 0:
                continue
            for base_name in ("loss_bbox", "loss_giou"):
                key = family_key(row["head"], base_name)
                if key not in base["losses"]:
                    raise SystemExit(("no loss key for family head", row["head"], key,
                                      sorted(base["loss_keys"])))
                affected.add(key)
        if not affected or not affected <= set(differing):
            raise SystemExit(("a family with extra positives did not change its box slots",
                              mode, sorted(affected), differing))
        report[mode + "_differing_keys"] = differing
        report[mode + "_affected_keys"] = sorted(affected)
        report[mode + "_pos_totals"] = other["pos_totals"]
        report[mode + "_extras"] = other["extras"]
        totals = other["pos_totals"]
        if totals["band_gt"] <= 0:
            raise SystemExit(("the fixed window has no tiny-band ground truth for " + mode, totals))
        if totals["modes"] != [mode]:
            raise SystemExit(("the armed arm did not report its own mode", mode, totals["modes"]))
        if totals["extra_pairs"] <= 0:
            raise SystemExit(("the mechanism is inert on the fixed real input window: no admissible "
                              "extra positive was added", mode, totals))
        if totals["extra_pairs"] + totals["band_gt_no_sibling"] != totals["band_gt"]:
            raise SystemExit(("the admission accounting does not close", mode, totals))
        if totals["extra_iou_sum"] < 0.5 * totals["extra_pairs"]:
            raise SystemExit(("an admitted extra positive is below the pre-registered overlap bar", mode, totals))
        if mode == "A1" and totals["extra_pairs"] != totals["band_pairs"]:
            raise SystemExit(("A1 must add exactly one duplicate per native band pair", totals))
        if mode == "A1" and totals["band_gt_no_sibling"] != 0:
            raise SystemExit(("A1 must not report an admission ledger", totals))
        # the augmented-set identity: the added regression mass is the recorded extra sums divided
        # by the frozen normaliser under the criterion's own per-family weights
        # the augmented-set identity, per family key: the added regression mass of one family is the
        # recorded extra sums of that family divided by the frozen normaliser under the criterion's
        # own weight for that loss name (the chain applies one weight per loss name to every
        # family), and must equal that key's loss difference up to float summation order
        weight_keys = values[mode]["family_weights"]
        implied = {}
        for row in other["pos_rows"]:
            for base_name, sum_key in (("loss_bbox", "pos_extra_l1_sum"), ("loss_giou", "pos_extra_giou_sum")):
                key = family_key(row["head"], base_name)
                if key not in base["losses"]:
                    raise SystemExit(("no loss key for family head", row["head"], key))
                weight = weight_keys.get(base_name)
                if weight is None:
                    raise SystemExit(("no criterion weight for loss name", base_name,
                                      sorted(weight_keys)))
                implied[key] = implied.get(key, 0.0) + weight * float(row[sum_key])
        for key, total in sorted(implied.items()):
            expected_delta = total / other["num_boxes"]
            observed = other["losses"][key] - base["losses"][key]
            if abs(observed - expected_delta) > 1e-4 * max(1e-3, abs(expected_delta)):
                raise SystemExit(("the augmented-set identity does not hold", mode, key,
                                  observed, expected_delta))
        report[mode + "_added_mass_total"] = sum(implied.values()) / other["num_boxes"]
        report[mode + "_added_mass_per_key"] = {k: v / other["num_boxes"] for k, v in sorted(implied.items())}
    diffs = {mode: base["gradient"]["sha256"] != values[mode]["gradient"]["sha256"]
             for mode in ("A1", "A2")}
    if not all(diffs.values()):
        raise SystemExit("the installed arms do not change the gradient digest")
    report["gradient_digests_differ"] = diffs
    report["a0_pos_totals"] = base["pos_totals"]
    if base["pos_totals"]["extra_pairs"] != 0 or base["extras"] != 0:
        raise SystemExit("the native arm must not add any extra positive")
    if base["pos_totals"]["modes"] != [None]:
        raise SystemExit(("the native arm must report no positives mode", base["pos_totals"]["modes"]))
    if base["pos_totals"]["band_pairs"] <= 0 or base["pos_totals"]["band_native_iou_sum"] <= 0.0:
        raise SystemExit(("the native arm must record the band matched-pair baseline", base["pos_totals"]))
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
