"""R32-001 runner (TASK_R32_W_SEEDS_001): CLI only.  The round is R28_ROUND; a formal or evaluation process handles the
single unit named by R32_UNIT (W-s1, W-s2); an acceptance process handles the seed named by R32_SEED.

  prepare                 -> r32_control.prepare  (writes <round>/CONTRACT.json, refuses an existing one)
  r32-check               -> r32_control          (CPU: unit tests, references, release difference, reference ledgers)
  r32-wire                -> r32_control          (CUDA, R32_SEED=0: the present code against the stored R28-001 receipts)
  r32-endpoint            -> r32_loop             (CUDA, R32_SEED=0: the stored W seed-0 endpoint through this release)
  r32-accept              -> r32_control          (CUDA, R32_SEED=1|2: reference path / formal path / restore replay)
  formal [--resume state] -> r32_loop             (CUDA, R32_UNIT)
  evaluate                -> r32_loop             (CUDA, R32_UNIT)
  summarize               -> r32_loop             (CPU)

The process is bound once to the seed of its action (s15_formal_train.bind_seed): 0 for the CPU actions and the two
seed-0 checks, R32_SEED for the acceptance, the seed of R32_UNIT for formal / evaluate.  Every stage function returns a
dict that is merged into <out>/CLOSED.json; an exception writes <out>/STOP.json and is re-raised.
"""
import argparse
import importlib
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r28_common as K  # noqa: E402
import r32_control as R32  # noqa: E402

DISPATCH = {"r32-check": "r32_control", "r32-wire": "r32_control", "r32-endpoint": "r32_loop", "r32-accept": "r32_control",
            "formal": "r32_loop", "evaluate": "r32_loop", "summarize": "r32_loop"}
CPU_ACTIONS = ("r32-check", "summarize")
SEED0_ACTIONS = ("r32-wire", "r32-endpoint")


def process_seed(action):
    if action in ("formal", "evaluate"):
        return R32.parse(R32.process_unit())[1]
    K.require("R32_UNIT" not in os.environ, f"R32_UNIT is only for formal / evaluate, not {action}")
    if action in SEED0_ACTIONS:
        return R32.process_seed((0,))
    if action == "r32-accept":
        return R32.process_seed(R32.SEEDS)
    K.require("R32_SEED" not in os.environ, f"R32_SEED is only for r32-wire / r32-endpoint / r32-accept, not {action}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("prepare",) + tuple(DISPATCH))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--resume", type=Path)
    args = ap.parse_args(argv)
    K.require(not any(v in os.environ for v in ("R28_ARM", "R30_UNIT", "R30_SEED", "R30_DELIVERY")),
              "R28_ARM / R30_* variables are not used by the R32 runner")
    if args.action == "prepare":
        K.require(args.out is None and args.resume is None, "prepare takes no --out/--resume")
        return R32.prepare(args)
    K.require(args.out is not None, "--out is required")
    K.require(args.resume is None or args.action == "formal", "--resume is only for formal")
    run = getattr(importlib.import_module(DISPATCH[args.action]), args.action.replace("-", "_"))
    c = R32.load_contract()
    seed = process_seed(args.action)
    unit = os.environ.get("R32_UNIT")
    args.out.mkdir(parents=True, exist_ok=False)
    start, cpu0 = time.monotonic(), time.process_time()
    counts = Counter(actual_updates=0, forwards=0, forward_attempts=0)
    try:
        from r25_run import setup
        from s15_formal_train import bind_seed
        bind_seed(seed)
        if args.action in CPU_ACTIONS:
            K.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", f"the {args.action} action requires CUDA_VISIBLE_DEVICES=\"\"")
            setup("cpu")
        else:
            K.require(os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == "expandable_segments:True", "allocator conf")
            setup("cuda:0")
            K.torch_mod().cuda.reset_peak_memory_stats()
        result = run(c, args.out, counts, args.resume) if args.action == "formal" else run(c, args.out, counts)
        K.require(isinstance(result, dict), f"{args.action} must return a dict")
        K.write_new(args.out / "CLOSED.json", {"status": "PASS", "action": args.action, "unit": unit, "process_seed": seed,
                                               "contract_sha256": c["contract_sha256"], "counts": dict(counts),
                                               "elapsed_seconds": time.monotonic() - start,
                                               "process_cpu_seconds": time.process_time() - cpu0, **result})
    except BaseException as exc:
        K.write_new(args.out / "STOP.json", {"status": "STOP", "phase": K.PHASE["phase"], "action": args.action, "unit": unit,
                                             "process_seed": seed, "error": repr(exc)[:4000], "traceback": traceback.format_exc(),
                                             "counts": dict(counts), "elapsed_seconds": time.monotonic() - start})
        raise


if __name__ == "__main__":
    main()
