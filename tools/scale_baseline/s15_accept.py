"""S15 limited CUDA engineering acceptance (NOT a formal run; produced state is never formal).

Pre-registered workload (one invocation per memory mode):
  u18241, u18242 from the bound e060 parent with exact output/gradient capture; save after u18241;
  fresh engine restores u18241 and replays u18242 -> complete state and update record must be bitwise equal;
  the replayed engine continues u18243..u18248 through the real two-worker loader for steady timing.
  => 9 actual optimizer updates / 18 training forwards, plus GT-free EMA FP32 batch4 inference on the
  first 8 development images (2 warm-up + 2 timed batches, no prediction file, no scoring).
Production math only: physical batch 8 x accumulation 2, BF16 autocast, FP32 loss, clip 0.1.
--activation-checkpoint is allowed only after a plain-mode CUDA OOM receipt and only because the CPU
check proved it bitwise-equal; it never changes micro-batch/BN/DN.  That mode adds 2 native 768x1344
A0 updates that must reproduce the recorded R25 A0 u18241/u18242 loss/LR/gradient norm exactly (11 updates).

usage:
  python -B tools/scale_baseline/s15_accept.py --contract R25_FORMAL_CONTRACT --out NEW_DIR [--activation-checkpoint --oom-receipt PATH]
"""
from __future__ import annotations

import os
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import gc
import json
import random
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools" / "query_local"), str(ROOT)]

import numpy as np
import torch

from formal_data import PARENT_UPDATE, windows
from sparse_rtdetr.research import scale_baseline as S
from sparse_rtdetr.protocol_r18 import checkpoint as native
from sparse_rtdetr.protocol_r18.geometry import forward_guards

PLAIN_UPDATES, REPLAY_UPDATES, TIMING_UPDATES = 2, 1, 6
EXPECTED_UPDATES = PLAIN_UPDATES + REPLAY_UPDATES + TIMING_UPDATES  # 9
EVAL_IMAGES, EVAL_BATCH = 8, 4


def setup():
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    S.require(torch.cuda.device_count() == 1, "single configured GPU required")
    torch.cuda.set_device(0)


def publish(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=1, allow_nan=False, sort_keys=True); f.write("\n")


def memory():
    return {"peak_allocated": torch.cuda.max_memory_allocated(), "peak_reserved": torch.cuda.max_memory_reserved()}


def register(e, counts):
    e.model.register_forward_hook(lambda *_: counts.update(forwards=1))
    e.optimizer.register_step_post_hook(lambda *_: counts.update(actual_updates=1))


def precision_observers(e, seen):
    def model_hook(_m, args, result):
        seen.append({"input": str(args[0].dtype), "autocast": torch.is_autocast_enabled("cuda"),
                     "autocast_dtype": str(torch.get_autocast_dtype("cuda")),
                     "logits": str(result["pred_logits"].dtype)})
    e.model.register_forward_hook(model_hook)


def update(e, x, t, audit, *, exact):
    torch.cuda.synchronize()
    r = e.train_window(x, t, capture=exact)
    S.require(r["update"] == audit["update"] and r["finite"] and e.ema.updates == e.completed_updates, "S15 clock")
    return r


def inference(e, contract):
    data = S.development(contract["data"]["development"], S.S15_HW)
    S.require([r["stable_image_id"] for r in data.records] == contract["development_ids"], "S15 development identity")
    from infer_seed import load_original
    from torchvision.transforms.v2 import functional as TF
    model = e.ema.module.eval()
    rng = native.capture_rng(e.device)
    torch.cuda.reset_peak_memory_stats()
    timings = []
    with torch.inference_mode():
        for i in range(0, EVAL_IMAGES, EVAL_BATCH):
            rows = data.records[i:i + EVAL_BATCH]
            xs = []
            for row in rows:
                with load_original(data, row) as image:
                    xs.append(TF.pil_to_tensor(data.transform.resize(image)).float() / 255.)
            for repeat in range(2):  # first pass warm-up, second timed
                torch.cuda.synchronize(); t0 = time.perf_counter()
                with torch.autocast("cuda", enabled=False), forward_guards(model):
                    out = model(torch.stack(xs).to(e.device))
                torch.cuda.synchronize()
                timings.append({"batch": i // EVAL_BATCH, "warmup": repeat == 0, "forward_seconds": time.perf_counter() - t0})
                S.require(out["pred_boxes"].shape == (EVAL_BATCH, 300, 4) and out["pred_logits"].dtype == torch.float32,
                          "S15 inference schema")
    S.require(native.exact_equal(native.capture_rng(e.device), rng), "S15 inference changed RNG")
    return {"images": EVAL_IMAGES, "batch": EVAL_BATCH, "fp32": True, "timings": timings, **memory(),
            "predictions_written": False, "scored": False}


def a0_checkpoint_equivalence(contract, counts):
    """CUDA bitwise check of the memory option where the plain path is known to fit: the native
    768x1344 A0 production engine with activation checkpointing must reproduce R25 A0 u18241/u18242."""
    from formal_data import train_dataset
    from sparse_rtdetr.research.query_local_runtime import create as r25_create
    recorded = []
    with (Path(contract["attempt_root"]) / "train-A0" / "UPDATES.jsonl").open() as f:
        for _ in range(2):
            recorded.append(json.loads(f.readline()))
    e, _ = r25_create(contract["parent"]["path"], "A0", "cuda:0")
    S.install_activation_checkpoint(e.model)
    register(e, counts)
    rows = []
    for (x, t, audit), ref in zip(windows(train_dataset(contract["data"]["train_core"]), PARENT_UPDATE, limit=2), recorded):
        r = update(e, x, t, audit, exact=False)
        same = (audit == ref["input"] and r["loss"] == ref["loss"] and r["lr"] == ref["lr"]
                and r["gradient_norm_before_clip"] == ref["gradient_norm_before_clip"])
        rows.append({"update": r["update"], "loss": r["loss"], "gradient_norm_before_clip": r["gradient_norm_before_clip"],
                     "equal_to_R25_A0_record": same, **memory()})
    S.require(all(r["equal_to_R25_A0_record"] for r in rows), "CUDA activation checkpoint differs from R25 A0")
    del e; gc.collect(); torch.cuda.empty_cache()
    return {"rows": rows, "bitwise_equal_to_R25_A0": True}


def run(contract, out, counts, checkpointed):
    parent = contract["parent"]["path"]
    data = S.train_dataset(contract["data"]["train_core"], S.S15_HW)
    e, bridge = S.create(parent, "cuda:0", activation_checkpoint=checkpointed)
    register(e, counts)
    seen = []
    precision_observers(e, seen)
    rows = []
    torch.cuda.reset_peak_memory_stats()
    for i, (x, t, audit) in enumerate(windows(data, PARENT_UPDATE, limit=PLAIN_UPDATES), 1):
        r = update(e, x, t, audit, exact=True)
        rows.append({"input": audit, "update": r, **memory()})
        if i == 1:
            saved = S.save(e, out / "u1.pth")
    expected = S.capture(e)
    S.require(len(seen) == 4 and all(v["autocast"] and v["autocast_dtype"] == "torch.bfloat16"
                                     and v["input"] == "torch.float32" for v in seen), "S15 CUDA precision")
    del e; gc.collect(); torch.cuda.empty_cache()
    clone, _ = S.create(parent, "cuda:0", activation_checkpoint=checkpointed)
    register(clone, counts)
    restored = S.restore(clone, out / "u1.pth", saved["sha256"])
    ((x, t, audit),) = list(windows(data, PARENT_UPDATE + 1, limit=1))
    replay = update(clone, x, t, audit, exact=True)
    after = S.capture(clone)
    after["activation_checkpoint"] = expected["activation_checkpoint"]
    same = {k: v for k, v in replay.items() if k != "elapsed_seconds"} == {
        k: v for k, v in rows[1]["update"].items() if k != "elapsed_seconds"}
    S.require(audit == rows[1]["input"] and same and native.exact_equal(after, expected),
              "S15 CUDA replay differs")
    timing = []
    started = time.perf_counter()
    for x, t, audit in windows(data, clone.completed_updates, limit=TIMING_UPDATES):
        r = update(clone, x, t, audit, exact=False)
        timing.append({"update": r["update"], "update_seconds": r["elapsed_seconds"], "loss": r["loss"], **memory()})
    loader_inclusive = time.perf_counter() - started
    infer = inference(clone, contract)
    equivalence = a0_checkpoint_equivalence(contract, counts) if checkpointed else None
    expected_updates = EXPECTED_UPDATES + (2 if checkpointed else 0)
    S.require(counts["actual_updates"] == expected_updates and counts["forwards"] == 2 * expected_updates,
              "S15 acceptance count")
    return {"bridge": bridge, "first_updates": rows, "saved_u1": saved, "restore": restored,
            "replay_exact": True, "replay_update_seconds": replay["elapsed_seconds"], "timing_updates": timing,
            "timing_loader_inclusive_seconds": loader_inclusive,
            "steady_update_seconds_mean": sum(r["update_seconds"] for r in timing) / len(timing),
            "inference": infer, "precision": seen, "activation_checkpoint": checkpointed,
            "A0_768_checkpoint_equivalence": equivalence,
            "allocator_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "formal_state_usable": False}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--contract", required=True, type=Path); p.add_argument("--out", required=True, type=Path)
    p.add_argument("--activation-checkpoint", action="store_true"); p.add_argument("--oom-receipt", type=Path)
    args = p.parse_args()
    if args.activation_checkpoint:
        receipt = json.loads(args.oom_receipt.read_bytes()) if args.oom_receipt else {}
        S.require(receipt.get("status") == "STOP_CUDA_OOM" and receipt.get("activation_checkpoint") is False,
                  "checkpoint mode requires a plain-mode CUDA OOM receipt")
    args.out.mkdir(parents=True, exist_ok=False)
    contract = json.loads(args.contract.read_bytes())
    counts = Counter(actual_updates=0, forwards=0)
    t0 = time.monotonic()
    try:
        setup()
        result = run(contract, args.out, counts, args.activation_checkpoint)
        publish(args.out / "ACCEPTANCE.json", {"status": "PASS", **result, "counts": dict(counts),
                                                "elapsed_seconds": time.monotonic() - t0, **memory()})
    except torch.cuda.OutOfMemoryError as exc:
        publish(args.out / "STOP.json", {"status": "STOP_CUDA_OOM", "activation_checkpoint": args.activation_checkpoint,
                                         "error": str(exc)[:2000], "counts": dict(counts),
                                         "elapsed_seconds": time.monotonic() - t0, **memory()})
        raise
    except BaseException as exc:
        publish(args.out / "STOP.json", {"status": "STOP", "activation_checkpoint": args.activation_checkpoint,
                                         "error": repr(exc), "traceback": traceback.format_exc(),
                                         "counts": dict(counts), "elapsed_seconds": time.monotonic() - t0,
                                         **(memory() if torch.cuda.is_initialized() else {})})
        raise


if __name__ == "__main__":
    main()
