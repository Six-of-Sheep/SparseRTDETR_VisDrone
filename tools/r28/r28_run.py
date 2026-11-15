"""R28 runner: CLI only.  The arm (W or K) of a process is the environment variable R28_ARM (default W).

  prepare                         -> r28_common.prepare   (writes <round>/CONTRACT.json, refuses an existing one)
  cpu | zero | accept | evalpath  -> r28_checks
  demand | prior | g0             -> r28_demand
  precost | cost                  -> r28_cost
  formal                          -> r28_loop.formal(c, out, counts, resume)
  evaluate | summarize            -> r28_eval
  dose                            -> r28_dose             (descriptive CPU table, run by hand after eval; not in the chain)
  prepare-followup                -> r28_followup.prepare (TASK_R28_K_FOLLOWUP_003: contract of an independent K round)
  followup-check | medium         -> r28_followup         (CPU: affected-interface checks; descriptive medium tables)
  prepare-fovea                   -> r28_fovea_control.prepare (TASK_R28_FOVEA_MAPPING_CONTROL_004: contract of the F round)
  fovea-check | wire              -> r28_fovea_control    (CPU affected-interface checks; CUDA wiring of evaluation / timing)
  prepare-cal                     -> r28_cal_control.prepare (TASK_R28_CALIBRATED_F_CONTROL_006: contract of the F* round)
  cal-check | cal-fixture         -> r28_cal_control      (CPU: affected interfaces and pure-data pipeline checks; CPU fixture)
  calibrate | cal-wire | pipeline -> r28_cal_control      (CUDA: width calibration with the demand network only; wiring of
                                     evaluation / five-arm timing; serial / overlapped input pipeline acceptance)
  prepare-geo                     -> r28_geo_control.prepare (TASK_R28_STRONG_F_CONTROL_009: contract of the F_geo round)
  geo-check | ledger-means        -> r28_geo_control      (CPU: affected interfaces of arm FG; per-update magnification
                                     means of the stored W / K / F ledgers)
  geo-confirm | geo-wire | geo-accept -> r28_geo_control  (CUDA: validity of the fixed width on epoch-118 windows with the
                                     demand network only; wiring of evaluation / five-arm timing; the 4 + 4 + 2 update
                                     acceptance of the actual width)

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
            "evaluate": "r28_eval", "summarize": "r28_eval", "dose": "r28_dose",
            "followup-check": "r28_followup", "medium": "r28_followup",
            "fovea-check": "r28_fovea_control", "wire": "r28_fovea_control",
            "cal-check": "r28_cal_control", "cal-fixture": "r28_cal_control", "calibrate": "r28_cal_control",
            "cal-wire": "r28_cal_control", "pipeline": "r28_cal_control",
            "geo-check": "r28_geo_control", "geo-confirm": "r28_geo_control", "geo-wire": "r28_geo_control",
            "geo-accept": "r28_geo_control", "ledger-means": "r28_geo_control"}
CPU_ACTIONS = ("cpu", "prior", "summarize", "dose", "followup-check", "medium", "fovea-check", "cal-check", "cal-fixture",
               "geo-check", "ledger-means")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("prepare", "prepare-followup", "prepare-fovea", "prepare-cal", "prepare-geo") + tuple(DISPATCH))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--resume", type=Path)
    args = ap.parse_args(argv)
    if args.action == "prepare":
        K.require(args.out is None and args.resume is None, "prepare takes no --out/--resume")
        return K.prepare(args)
    if args.action == "prepare-followup":
        K.require(args.out is None and args.resume is None, "prepare-followup takes no --out/--resume")
        import r28_followup as FU
        return FU.prepare(args)
    if args.action == "prepare-fovea":
        K.require(args.out is None and args.resume is None, "prepare-fovea takes no --out/--resume")
        import r28_fovea_control as FC
        return FC.prepare(args)
    if args.action == "prepare-cal":
        K.require(args.out is None and args.resume is None, "prepare-cal takes no --out/--resume")
        import r28_cal_control as CC
        return CC.prepare(args)
    if args.action == "prepare-geo":
        K.require(args.out is None and args.resume is None, "prepare-geo takes no --out/--resume")
        import r28_geo_control as GC
        return GC.prepare(args)
    K.require(args.out is not None, "--out is required")
    K.require(args.resume is None or args.action == "formal", "--resume is only for formal")
    run = getattr(importlib.import_module(DISPATCH[args.action]), args.action.replace("-", "_"))
    c = K.load_contract()
    args.out.mkdir(parents=True, exist_ok=False)
    start, cpu0 = time.monotonic(), time.process_time()
    counts = Counter(actual_updates=0, forwards=0, forward_attempts=0)
    try:
        from r25_run import setup
        from s15_formal_train import bind_seed
        bind_seed(0)
        if args.action in CPU_ACTIONS or (args.action in ("demand", "g0") and K.device() == "cpu"):
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
