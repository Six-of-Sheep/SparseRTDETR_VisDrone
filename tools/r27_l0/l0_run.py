"""R27-002 L0 control runner: CLI only.  Each action lives in its own module and is imported when the action is called.

  prepare                 -> l0_common.prepare            (writes <round>/CONTRACT.json, refuses an existing one)
  cpu | zero | accept     -> l0_checks.<action>(c, out, counts)
  formal                  -> l0_loop.formal(c, out, counts, resume)
  evaluate | summarize    -> l0_eval.<action>(c, out, counts)
  cost4                   -> l0_cost.cost4(c, out, counts)

Every stage function returns a dict that is merged into <out>/CLOSED.json; an exception writes <out>/STOP.json and
is re-raised.  The arm is always L0.  Later steps add modules only; this file is not edited for new actions.
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
import l0_common as K  # noqa: E402

DISPATCH = {"cpu": ("l0_checks", "cpu"), "zero": ("l0_checks", "zero"), "accept": ("l0_checks", "accept"),
            "formal": ("l0_loop", "formal"), "evaluate": ("l0_eval", "evaluate"), "summarize": ("l0_eval", "summarize"),
            "cost4": ("l0_cost", "cost4")}
CUDA_ACTIONS = ("zero", "accept", "formal", "evaluate", "cost4")


def handler(action):
    module, name = DISPATCH[action]
    try:
        m = importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name != module:
            raise
        raise SystemExit(f"action '{action}' is not available in this source tree: tools/r27_l0/{module}.py is missing") from exc
    K.require(callable(getattr(m, name, None)), f"tools/r27_l0/{module}.py does not define {name}()")
    return getattr(m, name)


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
    run = handler(args.action)
    c = K.load_contract()
    args.out.mkdir(parents=True, exist_ok=False)
    start, cpu0 = time.monotonic(), time.process_time()
    counts = Counter(actual_updates=0, forwards=0, forward_attempts=0)
    try:
        from r25_run import setup
        from s15_formal_train import bind_seed
        bind_seed(0)
        if args.action == "cpu":
            K.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "the cpu action requires CUDA_VISIBLE_DEVICES=\"\"")
        if args.action in CUDA_ACTIONS:
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
