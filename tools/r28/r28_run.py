"""R28 runner: CLI only.  The arm (W or K) of a process is the environment variable R28_ARM (default W).

  prepare                         -> r28_common.prepare   (writes <round>/CONTRACT.json, refuses an existing one)
  cpu | zero | accept | evalpath  -> r28_checks
  demand | prior | g0             -> r28_demand
  precost | cost                  -> r28_cost
  formal                          -> r28_loop.formal(c, out, counts, resume)
  evaluate | summarize            -> r28_eval

Every stage function returns a dict that is merged into <out>/CLOSED.json; an exception writes <out>/STOP.json and
is re-raised.
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

DISPATCH = {"cpu": "r28_checks", "zero": "r28_checks", "accept": "r28_checks", "evalpath": "r28_checks",
            "demand": "r28_demand", "prior": "r28_demand", "g0": "r28_demand",
            "precost": "r28_cost", "cost": "r28_cost", "formal": "r28_loop",
            "evaluate": "r28_eval", "summarize": "r28_eval"}
CPU_ACTIONS = ("cpu", "prior", "summarize")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("prepare",) + tuple(DISPATCH))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--resume", type=Path)
    args = ap.parse_args(argv)
    if args.action == "prepare":
        K.require(args.out is None and args.resume is None, "prepare takes no --out/--resume")
        return K.prepare(args)
    K.require(args.out is not None, "--out is required")
    K.require(args.resume is None or args.action == "formal", "--resume is only for formal")
    run = getattr(importlib.import_module(DISPATCH[args.action]), args.action)
    c = K.load_contract()
    args.out.mkdir(parents=True, exist_ok=False)
    start, cpu0 = time.monotonic(), time.process_time()
    counts = Counter(actual_updates=0, forwards=0, forward_attempts=0)
    try:
        from r25_run import setup
        from s15_formal_train import bind_seed
        bind_seed(0)
        if args.action in CPU_ACTIONS:
            K.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", f"the {args.action} action requires CUDA_VISIBLE_DEVICES=\"\"")
            if args.action != "cpu":
                setup("cpu")
        else:
            K.require(os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == "expandable_segments:True", "allocator conf")
            setup("cuda:0")
            K.torch_mod().cuda.reset_peak_memory_stats()
        result = run(c, args.out, counts, args.resume) if args.action == "formal" else run(c, args.out, counts)
        K.require(isinstance(result, dict), f"{args.action} must return a dict")
        K.write_new(args.out / "CLOSED.json", {"status": "PASS", "action": args.action, "arm": K.ARM,
                                               "contract_sha256": c["contract_sha256"], "counts": dict(counts),
                                               "elapsed_seconds": time.monotonic() - start,
                                               "process_cpu_seconds": time.process_time() - cpu0, **result})
    except BaseException as exc:
        K.write_new(args.out / "STOP.json", {"status": "STOP", "phase": K.PHASE["phase"], "action": args.action, "arm": K.ARM,
                                             "error": repr(exc)[:4000], "traceback": traceback.format_exc(),
                                             "counts": dict(counts), "elapsed_seconds": time.monotonic() - start})
        raise


if __name__ == "__main__":
    main()
