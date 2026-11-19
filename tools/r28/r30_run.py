"""R30-002 runner (TASK_R30_SEEDS_002): CLI only.  The round is R28_ROUND; a formal or evaluation process handles the
single unit named by R30_UNIT (L0-s1, FG-s1, L0-s2, FG-s2); an acceptance process handles the seed named by R30_SEED.

  prepare                 -> r30_control.prepare  (writes <round>/CONTRACT.json, refuses an existing one)
  r30-check               -> r30_control          (CPU: unit tests, references, release difference, reference ledgers)
  r30-wire                -> r30_control          (CUDA, R30_SEED=0: the present code against the stored seed-0 receipts)
  r30-accept              -> r30_control          (CUDA, R30_SEED=1|2: reference path / formal path / restore replay of both units)
  formal [--resume state] -> r30_loop             (CUDA, R30_UNIT)
  evaluate                -> r30_loop             (CUDA, R30_UNIT)
  summarize               -> r30_loop             (CPU)
  scorecheck              -> r30_loop             (CPU, dev rounds only, by hand)

The process is bound once to the seed of its action (s15_formal_train.bind_seed): 0 for the CPU actions and the seed-0
wiring, R30_SEED for the acceptance, the seed of R30_UNIT for formal / evaluate.  Every stage function returns a dict
that is merged into <out>/CLOSED.json; an exception writes <out>/STOP.json and is re-raised.
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
import r30_control as R30  # noqa: E402

DISPATCH = {"r30-check": "r30_control", "r30-wire": "r30_control", "r30-accept": "r30_control",
            "formal": "r30_loop", "evaluate": "r30_loop", "summarize": "r30_loop", "scorecheck": "r30_loop"}
CPU_ACTIONS = ("r30-check", "summarize", "scorecheck")


def process_seed(action):
    if action in ("formal", "evaluate"):
        return R30.parse(R30.process_unit())[1]
    K.require("R30_UNIT" not in os.environ, f"R30_UNIT is only for formal / evaluate, not {action}")
    if action == "r30-wire":
        return R30.process_seed((0,))
    if action == "r30-accept":
        return R30.process_seed(R30.SEEDS)
    K.require("R30_SEED" not in os.environ, f"R30_SEED is only for r30-wire / r30-accept, not {action}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("prepare",) + tuple(DISPATCH))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--resume", type=Path)
    args = ap.parse_args(argv)
    K.require("R28_ARM" not in os.environ, "R28_ARM is not used by the R30 runner")
    if args.action == "prepare":
        K.require(args.out is None and args.resume is None, "prepare takes no --out/--resume")
        return R30.prepare(args)
    K.require(args.out is not None, "--out is required")
    K.require(args.resume is None or args.action == "formal", "--resume is only for formal")
    run = getattr(importlib.import_module(DISPATCH[args.action]), args.action.replace("-", "_"))
    c = R30.load_contract()
    seed = process_seed(args.action)
    unit = os.environ.get("R30_UNIT")
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
