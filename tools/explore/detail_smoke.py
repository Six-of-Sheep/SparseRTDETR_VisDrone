"""CPU synthetic training smoke of the detail path through the real engine.

Per mode ('detail' or 'control'), mirrors the release's own CPU lifecycle
(``protocol_r18.worker.cpu_state``): synthetic Config -> install -> zero-gate
identity vs native -> one manual criterion + backward gradient pass -> Engine
with a fresh optimizer and EMA -> two train windows -> checkpoint
save/restore/replay exactness -> JSON evidence.  CUDA must be hidden; the
frozen runs and their outputs are untouched.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import detail_path  # noqa: E402
import frozen_r18  # noqa: E402


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fixture(win):
    return [
        {"boxes": torch.tensor([[.4, .4, .15, .2]]), "labels": torch.tensor([1]),
         "gt_ids": ["CPUfixture/w%d/raw0" % win],
         "ignore_union": torch.empty(0, 4, dtype=torch.float64),
         "ignore_class_boxes": torch.empty(0, 4, dtype=torch.float64),
         "ignore_class_labels": torch.empty(0, dtype=torch.int64)},
        {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.int64), "gt_ids": [],
         "ignore_union": torch.empty(0, 4, dtype=torch.float64),
         "ignore_class_boxes": torch.empty(0, 4, dtype=torch.float64),
         "ignore_class_labels": torch.empty(0, dtype=torch.int64)},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--mode", choices=list(detail_path.MODES), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    frozen_r18.require_release_source(args.source_root)
    from sparse_rtdetr.protocol_r18.model import ARM, CommonEMA, Config, build
    from sparse_rtdetr.protocol_r18.engine import Engine
    from sparse_rtdetr.protocol_r18 import checkpoint
    from sparse_rtdetr.baseline.training_v2b import _optimizer

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    require(not torch.cuda.is_initialized(), "CUDA initialized before the CPU smoke")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    conf = Config(seed=0, input_hw=(128, 128), physical_batch_size=1, amp_dtype="float32")

    def construct(mode):
        model, criterion, info = build(conf, pretrained_path=None, synthetic=True)
        if mode is not None:
            detail_path.install_detail_path(model, mode)
        return model, criterion, info

    native_model, _, _ = construct(None)
    model, criterion, info = construct(args.mode)

    probe = torch.rand(1, 3, 128, 128, generator=torch.Generator().manual_seed(713))
    native_model.eval()
    model.eval()
    with torch.no_grad():
        reference = native_model(probe)
        installed = model(probe)
    identical = all(torch.equal(reference[k], installed[k]) for k in ("pred_logits", "pred_boxes"))
    require(identical, "zero-gate model is not bitwise identical to native")
    with torch.no_grad():
        model.decoder.detail_inject.gate.fill_(0.01)
        acted = model(probe)
    acting_diff = max(float((acted[k] - reference[k]).abs().max()) for k in ("pred_logits", "pred_boxes"))
    require(acting_diff > 0.0, "injection has no effect on outputs")
    with torch.no_grad():
        model.decoder.detail_inject.gate.zero_()
    del native_model, reference, installed, acted

    model.train()
    images = torch.rand(2, 3, 128, 128, generator=torch.Generator().manual_seed(21))
    targets = fixture(0)
    losses = criterion(model(images, targets), targets)
    total = sum(losses.values())
    require(bool(torch.isfinite(total)), "training loss is not finite")
    total.backward()
    gate = model.decoder.detail_inject.gate
    conv = model.decoder.detail_inject.conv.weight
    grad_report = {
        "loss_terms": sorted(losses),
        "gate_grad_nonzero": bool(gate.grad is not None and gate.grad.abs().sum() > 0),
        "conv_grad_zero_at_zero_gate": bool(conv.grad is None or float(conv.grad.abs().max()) == 0.0),
        "params_with_grad": int(sum(1 for p in model.parameters() if p.grad is not None)),
        "params_total": int(sum(1 for p in model.parameters())),
    }
    require(grad_report["gate_grad_nonzero"], "gate received no gradient")
    require(grad_report["conv_grad_zero_at_zero_gate"],
            "conv gradient at zero gate differs from the documented behavior")
    model.zero_grad(set_to_none=True)

    engine = Engine(model, criterion, _optimizer(model, conf), CommonEMA(model), conf)
    detail_path.assert_installed(engine.model, args.mode)
    detail_path.assert_installed(engine.ema.module, args.mode)
    binding = {"arm": ARM, "scope": "cpu_R18_synthetic",
               "run_id": "r18-explore-detail-smoke-20261031-001",
               "config": conf.binding(), "model": info}
    gen = torch.Generator().manual_seed(713)
    windows = [(torch.rand(2, 3, 128, 128, generator=gen), fixture(win)) for win in range(2)]
    r0 = engine.train_window(*windows[0], capture=True)
    require(r0["update"] == 1 and engine.completed_updates == engine.ema.updates == 1
            and engine.phase == 0, "update clock differs after window 1")
    cp0 = checkpoint.save(engine, binding, out / "CPU-C1.pth")
    r1 = engine.train_window(*windows[1], capture=True)
    require(r1["update"] == 2 and engine.completed_updates == engine.ema.updates == 2
            and engine.phase == 0, "update clock differs after window 2")
    expected = checkpoint.capture(engine, binding)
    direct = checkpoint.save(engine, binding, out / "CPU-C2-direct.pth")
    clone_model, clone_criterion, clone_info = construct(args.mode)
    clone = Engine(clone_model, clone_criterion, _optimizer(clone_model, conf),
                   CommonEMA(clone_model), conf)
    detail_path.assert_installed(clone.model, args.mode)
    detail_path.assert_installed(clone.ema.module, args.mode)
    restored = checkpoint.restore(clone, binding, cp0["path"], cp0["sha256"])
    r1_replay = clone.train_window(*windows[1], capture=True)
    same_result = ({k: v for k, v in r1.items() if k != "elapsed_seconds"}
                   == {k: v for k, v in r1_replay.items() if k != "elapsed_seconds"})
    state_exact = checkpoint.exact_equal(expected, checkpoint.capture(clone, binding))
    require(same_result, "window-2 replay result differs")
    require(state_exact, "window-2 replay state differs")
    require(bool(restored.get("full_state_exact", False)), "C1 restore is not exact")

    report = {
        "status": "PASS",
        "mode": args.mode,
        "identity": {"zero_gate_bitwise_identical": identical,
                     "nonzero_gate_max_abs_diff": round(acting_diff, 6)},
        "gradients": grad_report,
        "training": {"windows": 2, "updates": engine.completed_updates,
                     "window0_result_keys": sorted(r0),
                     "replay_result_exact": same_result,
                     "replay_state_exact": state_exact,
                     "restore_full_state_exact": bool(restored.get("full_state_exact", False)),
                     "checkpoints": [cp0, direct]},
        "parameter_delta": detail_path.DETAIL_PARAMETERS,
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "cuda_initialized": bool(torch.cuda.is_initialized()),
    }
    (out / "SMOKE.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))
    print("SMOKE_COMPLETE", args.mode, report["elapsed_seconds"], "s")


if __name__ == "__main__":
    main()
