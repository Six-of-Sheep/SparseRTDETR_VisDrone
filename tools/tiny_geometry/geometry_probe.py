"""Pre-registered tiny-band calibration and opportunity probe (CPU only; CUDA must be hidden).

Fixed inputs (pre-registered, no result-dependent changes): pass-1 windows 0..1 of the
production loader (16 train-side images, epoch clock 109, seed0), the certified parent
state, eval-mode FP32 forward on CPU.  For every matched pair (Hungarian matcher on the
final outputs) with a tiny ground-truth box the probe records the native slot value
v = 1 - giou and s = sqrt(W2); the frozen constant is

    C = median(s) / (-ln(1 - median(v)))

(the pre-registered median-parity rule; the penalty is monotone in s, so the median of the
replaced penalties equals median(v) by construction).  The probe also records the band's
per-micro-batch activation counts for the opportunity registration.  Nothing is trained,
no global RNG is consumed, and the model state must be unchanged afterwards.

usage:
  <frozen python> -I tools/geometry_probe.py --source-root <release>/source \
      --contract <contract.json> --out <probe.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch


def verified(ref):
    raw = Path(ref["path"]).read_bytes()
    if len(raw) != ref["bytes"] or hashlib.sha256(raw).hexdigest() != ref["sha256"]:
        raise SystemExit("bound reference changed: " + str(ref["path"]))
    return json.loads(raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.source_root).resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))
    if torch.cuda.is_initialized():
        raise SystemExit("CUDA must stay hidden for the probe")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()

    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    from sparse_rtdetr.protocol_r18.model import Config, build, fp32_outputs
    from sparse_rtdetr.protocol_r18 import checkpoint as native_state
    from sparse_rtdetr.protocol_r18 import slice_finetune as ft
    from sparse_rtdetr.protocol_r18 import tiny_geometry as tg
    from sparse_rtdetr.protocol_r18.engine import tensor_digest
    from sparse_rtdetr.localization.data import training_loader

    binding = verified(contract["parent_binding"])
    parent = contract["parent_checkpoint"]
    config = Config()
    model, criterion, _ = build(config, synthetic=True)
    payload = native_state.read(parent["path"], parent["sha256"], binding)
    initial = ft.initialize_from_parent(model, payload, expected_binding=binding)
    del payload
    model.eval()
    native_cls = criterion.__class__.__mro__[1]
    module = __import__(native_cls.__module__, fromlist=["generalized_box_iou"])

    data_spec = contract["data"]["train_core"]
    data = ft.Dataset("A2", manifest_path=data_spec["manifest"],
                      images_path=data_spec["roots"]["images"],
                      annotations_path=data_spec["roots"]["annotations"],
                      root_binding=data_spec["roots"])
    before = tensor_digest(model.state_dict().items())
    microbatches = []
    native_values, sqrt_values = [], []
    loader = training_loader(data, epoch=ft.clock(1), completed_windows=0, window_limit=2)
    with torch.no_grad():
        for win, (x, t, tr) in enumerate(loader):
            if len(x) != 16 or len(t) != 16:
                raise SystemExit("unexpected window shape")
            for micro in range(2):
                batch = x[micro * 8:(micro + 1) * 8]
                subset = t[micro * 8:(micro + 1) * 8]
                outputs = fp32_outputs(model(batch))
                matched = criterion.matcher({"pred_logits": outputs["pred_logits"],
                                             "pred_boxes": outputs["pred_boxes"]}, subset)
                indices = matched["indices"]
                batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
                src_idx = torch.cat([src for (src, _) in indices])
                pred = outputs["pred_boxes"][batch_idx, src_idx]
                gt = torch.cat([row["boxes"][j] for row, (_, j) in zip(subset, indices)])
                stats, _, _ = tg.observation(pred, gt, module)
                values = tg.native_penalty(pred, gt, module)
                root_s = tg.wasserstein_sqrt(pred, gt)
                tiny = tg.tiny_mask(gt)
                native_values += values[tiny].double().tolist()
                sqrt_values += root_s[tiny].double().tolist()
                microbatches.append({"pass": 1, "window": win, "microbatch": micro,
                                     "image_ids": [row["image_id"] for row in tr[micro * 8:(micro + 1) * 8]],
                                     **stats})
    after = tensor_digest(model.state_dict().items())
    if before != after:
        raise SystemExit("probe changed the model state")
    if not native_values:
        raise SystemExit("no tiny matched pairs on the fixed probe inputs")
    median_value = statistics.median(native_values)
    median_sqrt = statistics.median(sqrt_values)
    if not (0.0 < median_value < 1.0) or not (median_sqrt > 0.0):
        raise SystemExit(("median-parity preconditions failed", median_value, median_sqrt))
    constant = median_sqrt / (-math.log1p(-median_value))
    median_penalty = statistics.median([1 - math.exp(-s / constant) for s in sqrt_values])
    # For an even sample count statistics.median averages the two middle values, so the
    # monotone image of the median cannot be reproduced exactly in floating point; the
    # observed gap is bounded by half the local sample spacing (measured, not assumed).
    parity_delta = abs(median_penalty - median_value)
    if parity_delta > 1e-6:
        raise SystemExit(("median parity failed", median_penalty, median_value, parity_delta))
    result = {"schema": "r18-tiny-geometry-probe-v1",
              "inputs": {"parent_checkpoint_sha256": parent["sha256"], "parent_binding_sha256": contract["parent_binding"]["sha256"],
                         "initial_model_sha256": initial,
                         "fixed": "pass-1 windows 0..1 (16 train-side images, epoch clock 109, seed0)",
                         "forward_mode": "eval, FP32, CPU, CUDA hidden"},
              "microbatches": microbatches,
              "totals": {"pairs": sum(row["geo_pairs"] for row in microbatches),
                         "tiny_pairs": sum(row["geo_tiny_pairs"] for row in microbatches),
                         "tiny_zero_iou": sum(row["geo_tiny_zero_iou"] for row in microbatches)},
              "rule": "C = median(sqrt(W2)) / -ln(1 - median(1-giou))",
              "medians": {"native_value": median_value, "sqrt_W2": median_sqrt,
                          "penalty_with_C": median_penalty},
              "constant_C": constant, "constant_C_repr": repr(constant),
              "checks": {"cuda_initialized": bool(torch.cuda.is_initialized()),
                         "model_digest_unchanged": before == after,
                         "median_parity_within_1e-6": bool(parity_delta <= 1e-6),
                         "median_parity_delta": parity_delta},
              "elapsed_seconds": time.monotonic() - started}
    body = json.dumps(result, indent=1, ensure_ascii=False, sort_keys=True)
    target = Path(args.out)
    if target.exists():
        raise SystemExit("probe output already exists")
    target.write_bytes(body.encode("utf-8"))
    print(json.dumps({"status": "PROBE-OK", "constant_C_repr": repr(constant),
                      "tiny_pairs": result["totals"]["tiny_pairs"],
                      "pairs": result["totals"]["pairs"],
                      "tiny_zero_iou": result["totals"]["tiny_zero_iou"],
                      "sha256": hashlib.sha256(body.encode()).hexdigest(),
                      "bytes": len(body)}))


if __name__ == "__main__":
    main()
