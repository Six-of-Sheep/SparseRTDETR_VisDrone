"""R30-002: paired seed-1 / seed-2 replication of F_geo against L0 (TASK_R30_SEEDS_002).

A fixed replication of an existing result.  No definition changes: arm L0 is the R27-002 control
(src/sparse_rtdetr/research/r27_l0_control.py), arm FG is the F_geo arm of task 009
(src/sparse_rtdetr/research/r28_fovea_geo.py); both take the seed as an argument whose default 0 is the frozen path.
Four units are trained here, each from the e060 parent of its own seed: L0-s1, FG-s1, L0-s2, FG-s2.  The demand network
is the frozen file of the W round (trained on seed 0); it is not retrained and is shared by the three seeds.

  prepare      contract of the independent round: task file, the four units with their parents and reference ledgers
               (the S15 seed-1 / seed-2 replication ledgers), the seed-0 receipts of 009 and R27-002 (read-only), the
               frozen demand network and kernel width, the release difference against the R29 release, the reading of
               task section 4.
  r30_check    CPU: unit tests; references and identities; the release difference file by file and function by
               function; the S15 reference ledgers of both seeds (row count, update sequence, schedule columns equal to
               the seed-0 schedule, sample identities different from seed 0).
  r30_wire     CUDA, seed 0, not a result: FG 2 updates from the seed-0 e060 parent against the stored accept-FG
               receipts of 009, L0 2 updates against the stored acceptance receipts of R27-002; each on the reference
               path (serial, host tensor) and on the formal path (overlapped input, device window).  8 updates.
  r30_accept   CUDA, one seed per process: for both units of the seed, 4 updates on the reference path, 4 updates on
               the formal path with a full state saved after update 2, then updates 3-4 replayed from that state.
               Every update is compared bit for bit between the two paths and between the run and its replay; every
               window is compared with the S15 ledger of the seed.  20 updates per seed.
  summary      pure arithmetic of task section 4 (the 14 cells per seed, the single-seed rule, the three-seed label,
               mean and sample standard deviation, the APM / APL readings).
Formal training, evaluation and the summary stage are in r30_loop.py; the runner is r30_run.py.
"""
import gc
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import r28_common as K
import r28_cal_control as CC
import r28_geo_control as GC
import r29_input as RI

BASIS = "post_R29_user_forwarded_fgeo_l0_seed_replication"
REGISTRATION = ("R30-002 fixed replication of F_geo - L0 on seeds 1 and 2: four training units, no new method, no changed "
                "definition; TASK_R30_SEEDS_002")
TASK = ("TASK_R30_SEEDS_002.md", "bb17c0776551a89b45e2f81db2bb89115975ae784aeda6e29d68750702ee80e8")
ARMS = ("L0", "FG")
SEEDS = (1, 2)
ORDER = ("L0-s1", "FG-s1", "L0-s2", "FG-s2")           # task section 3
SCHEMA = "r30-state-v1"
SCOPE = "r30-acceptance"
UNIT_RE = re.compile(r"(L0|FG)-s([0-2])")

BASE_ROUND = K.LD_ROOT / "r29-query-unzoom-20261117-002"       # the release this one is compared with
BASE_COMMIT = "6c656518da9c0ea97121bc15d0a147c8bdef3fef"
L0_PINS = {"accept_closed": ("accept/CLOSED.json", "10027b3853314c72b7a21e50be6b5818e8f5b431e9ecb78c97338e920d811170"),
           "accept_u1": ("accept/u1.pth", "4fa9d58e528887dbc193aac92dd26592ac7732ab7f6591881076edb147e6e4d3"),
           "ledger": ("formal-L0/UPDATES.jsonl", "cfded55c71cc722f531f40035486dad8f9b2c4dc3ac33b5de89e12206b737349"),
           "formal_closed": ("formal-L0/CLOSED.json", "3543c94e37436bfc463dce57316e82eb2dc7b4524338246bdc9cc1e87ace939f"),
           "eval_closed": ("eval-L0/CLOSED.json", "82f93b4eed1d7232df1003d7844b90b993ffc3b78dcc2ecb3904349f28095ffe")}
S15_ROUND = K.LD_ROOT / "s15-seed12-replication-20261106-001"
S15_PINS = {"contract_file": "3b4858602267c7ef18f048b6f60928e9345b53aec6d6363e921f6c98e01e7883",
            "contract": "41774c7ad926082d3c5d9bcc07adef9d1cd97707a292a56c4d377bde945c01a0",
            "ledger": {1: "a5ede77df7d3ad7c44bdc61c62440b42d274a1025bbddc0af679c71e0435270a",
                       2: "bcabf3c81066165a246ce4acae12592b2319ddd591cd65a735ec8b679099600f"},
            "closed": {1: "eb9b02b14a96e5d1588cf3ae246b5a66e8dff0d467eabe8557b7b204b3da9013",
                       2: "3dd09058124f381964561191f3680ed0db45cddc602d88bf8fc75b67dfc0f188"}}
DEMAND_SHA = "c9ffc7505e47adbcc45d70949c3d87371c6f3a7795f1aaaab886d98b15ee9e5a"

SCIENCE_CHANGED = ("src/sparse_rtdetr/research/r27_l0_control.py", "src/sparse_rtdetr/research/r28_fovea_geo.py")
SCIENCE_EDITS = {p: ["build_eval_model", "create"] for p in SCIENCE_CHANGED}     # the seed argument, default 0
ALLOWED_ADDED = ("tools/r28/r30_control.py", "tools/r28/r30_loop.py", "tools/r28/r30_run.py", "tools/r28/r30_chain.sh",
                 "tests/test_r30_control.py")
TESTS = ("tests/test_r27_l0_control.py", "tests/test_r28_fovea_geo.py", "tests/test_r28_geo_control.py", "tests/test_r28_pipeline.py",
         "tests/test_r29_input.py", "tests/test_r30_control.py")

WIRE = {"updates_per_path": 2, "paths": [["serial", "host"], ["overlap", RI.FORMAL]]}
ACCEPT = {"steps": 4, "save_at": 2, "reference_path": ["serial", "host"], "formal_pipeline": "overlap"}
_UNIT_ACCEPT = ACCEPT["steps"] * 2 + (ACCEPT["steps"] - ACCEPT["save_at"])
_FORMAL = K.ENDPOINT - K.PARENT_UPDATE
_EVAL = K.EVAL_WARMUP + K.EVAL_BATCHES
WORKLOAD = {
    "r30-check": {"updates": 0, "forwards": 0, "note": "CPU; synthetic unit tests in a subprocess"},
    "wire-s0": {"updates": len(ARMS) * len(WIRE["paths"]) * WIRE["updates_per_path"],
                "forwards": 2 * len(ARMS) * len(WIRE["paths"]) * WIRE["updates_per_path"],
                "note": "seed 0, not a result; states discarded"},
    "accept": {"updates_per_seed": len(ARMS) * _UNIT_ACCEPT, "forwards_per_seed": 2 * len(ARMS) * _UNIT_ACCEPT,
               "updates": len(SEEDS) * len(ARMS) * _UNIT_ACCEPT, "note": "states discarded; never continued as formal"},
    "formal": {"units": list(ORDER), "updates_per_unit": _FORMAL, "forwards_per_unit": 2 * _FORMAL,
               "updates": len(ORDER) * _FORMAL, "forwards": len(ORDER) * 2 * _FORMAL,
               "save_epochs": list(K.SAVE_EPOCHS), "task_save_epochs": list(K.FORMAL_SAVE_EPOCHS)},
    "evaluation": {"endpoint": "e120 EMA of each unit", "view": "A", "images": 548, "batch": K.EVAL_BATCH, "warmup": K.EVAL_WARMUP,
                   "forwards_per_unit": _EVAL, "forwards": len(ORDER) * _EVAL, "budgets": list(K.BUDGETS),
                   "score_budget_calls": 2 * len(ORDER), "per_class_COCOeval_calls": len(ORDER),
                   "per_class_tensors_kept": True, "intermediate_endpoints": "none", "checkpoint_selection": "none"},
    "cost": "none: no timing stage in this task; the cost reading stays the same-process ratio of task 009",
}

METRICS = ("AP", "APS", "AP75_small", "APM", "APL", "AP50", "AP75")
BUDGET_KEYS = ("600", "300")
RULE = (("600", "APS", 1.0), ("600", "AP75_small", 1.0), ("600", "AP", 0.0))     # pp, all three at once
LABELS = ("small_gain_repeated_3of3", "small_gain_repeated_2of3", "small_gain_not_repeated")
TRADEOFF = "存在尺度取舍"
READING = {
    "cells": [f"top{b}_{m}" for b in BUDGET_KEYS for m in METRICS],
    "difference": "F_geo - L0 of the same seed, percentage points, e120 EMA, view A, 548 images; all 14 cells are reported",
    "seed0": "quoted from task 009 (eval-FG) and R27-002 (eval-L0); not rerun",
    "single_seed_rule": [list(r) for r in RULE],
    "single_seed_rule_note": "small-object gain repeated on one seed: APS600 >= +1.0, AP75-small600 >= +1.0 and AP600 >= 0, all three",
    "labels": {"3": LABELS[0], "2": LABELS[1], "0-1": LABELS[2]},
    "aggregate": "mean and sample standard deviation (n - 1) over seeds 0, 1, 2 of each of the 14 paired differences",
    "size_cells": ["top600_APM", "top600_APL"],
    "scale_tradeoff": f"APL600 negative on all three seeds registers {TRADEOFF!r} next to the small-object label; neither cancels the other",
    "threshold_note": "+1.0 is about half of the seed-0 point value, a practical lower bound chosen by the controller in advance; "
                      "not a significance or power test.  Earlier thresholds and decisions are unchanged.",
}
STATEMENTS = {
    "demand": "需求网络是 W 轮在 seed0 上训练并冻结的同一个文件（c9ffc750…9e5a），不重训，三个 seed 共用。",
    "scope": "固定复验：不改 F_geo、L0、核宽、倍率、需求规则、日程；不训练 W、K、F、P、C 的任何 seed；不评中间端点；不测速。",
    "seed0": "seed0 的 F_geo 与 L0 读数引用 009 与 R27-002 的已存结果，本轮不重跑。",
    "development": "548 图 development 集已被重复使用；本结果是该集合上的三 seed 成对读数，不提供独立泛化证明。",
}
INPUT_PIPELINE = {
    "pipeline": "overlap (r28_pipeline, accepted in task 009): workers 2, one prepared window, audit before consumption",
    "window_delivery": RI.RULE,
    "per_unit_condition": "a unit uses the device window delivery only if its own acceptance found every update (outputs, gradients, "
                          "model / optimizer / EMA / RNG state) bit-for-bit equal to the serial host reference path and its save / "
                          "restore replay exact; R30_DELIVERY=host keeps the host tensor and needs an acceptance run in that mode",
    "note": "an engineering choice; batch 8 x 2, BF16, optimizer, LR / EMA / BN clocks, activation checkpoint and the loader are unchanged",
}
LEDGER_COMPARISON = {
    "fields": list(K.SHARED_FIELDS),
    "same_seed_units": "the ledgers of L0-sN and FG-sN are compared row by row on every listed field",
    "against_S15": "every window is compared with the row of the S15 seed-N replication ledger before the update (input record) and "
                   "after it (all listed fields); a difference stops the run before the model is updated with that window",
    "S15_row_keys": ["ema_updates", "gradient_norm_before_clip", "gt_count", "input", "loss", "lr", "update_seconds"],
    "not_comparable": "none of the listed fields; the S15 rows carry no microbatch_losses and there is one S15 ledger per seed (the "
                      "seed-0 rule 'two reference ledgers agree with each other' has no counterpart)",
}


def unit_name(arm, seed):
    return f"{arm}-s{seed}"


def parse(unit):
    m = UNIT_RE.fullmatch(unit or "")
    K.require(m is not None, f"unit must be <L0|FG>-s<seed>: {unit!r}")
    return m.group(1), int(m.group(2))


def process_unit():
    unit = os.environ.get("R30_UNIT")
    K.require(unit in ORDER, f"R30_UNIT must be one of {ORDER}, got {unit!r}")
    return unit


def process_seed(allowed):
    raw = os.environ.get("R30_SEED")
    K.require(raw in tuple(str(s) for s in allowed), f"R30_SEED must be one of {allowed}, got {raw!r}")
    return int(raw)


# ================================================================ pure arithmetic of task section 4
def difference_pp(x, y):
    return {b: {m: 100. * (x[b][m] - y[b][m]) for m in METRICS} for b in BUDGET_KEYS}


def single_seed(diff):
    cells = {f"top{b}_{m}": {"difference_pp": diff[b][m], "minimum_pp": lo, "passed": diff[b][m] >= lo} for b, m, lo in RULE}
    return {"cells": cells, "small_gain_repeated": all(v["passed"] for v in cells.values())}


def label(passed):
    """passed: one bool per seed (three seeds)."""
    K.require(len(passed) == 3 and all(type(p) is bool for p in passed), "the label needs the three seeds")
    n = sum(passed)
    return LABELS[0] if n == 3 else LABELS[1] if n == 2 else LABELS[2]


def aggregate(diffs):
    """diffs: {seed: 14-cell table}.  Mean and sample standard deviation (n - 1) of every cell over the seeds."""
    seeds = sorted(diffs)
    K.require(len(seeds) >= 2, "the sample standard deviation needs two seeds at least")
    out = {}
    for b in BUDGET_KEYS:
        out[b] = {}
        for m in METRICS:
            v = [diffs[s][b][m] for s in seeds]
            mean = math.fsum(v) / len(v)
            out[b][m] = {"mean_pp": mean, "sample_sd_pp": math.sqrt(math.fsum((x - mean) ** 2 for x in v) / (len(v) - 1)),
                         "values_pp": v, "n": len(v)}
    return out


def summary(values):
    """values: {"0" | "1" | "2": {"FG": table, "L0": table}} with table[budget][metric].  Everything of section 4."""
    K.require(sorted(values) == ["0", "1", "2"] and all(sorted(v) == ["FG", "L0"] for v in values.values()), "three seeds, two arms each")
    diffs = {s: difference_pp(values[s]["FG"], values[s]["L0"]) for s in sorted(values)}
    rule = {s: single_seed(d) for s, d in diffs.items()}
    passed = [rule[s]["small_gain_repeated"] for s in ("0", "1", "2")]
    sizes = {s: {m: {"difference_pp": d["600"][m], "sign": "+" if d["600"][m] > 0 else "-" if d["600"][m] < 0 else "0"} for m in ("APM", "APL")}
             for s, d in diffs.items()}
    tradeoff = all(diffs[s]["600"]["APL"] < 0 for s in ("0", "1", "2"))
    return {"F_geo_minus_L0_pp": diffs, "single_seed_rule": rule, "seeds_with_small_gain_repeated": [s for s, p in zip(("0", "1", "2"), passed) if p],
            "label": label(passed), "aggregate_over_seeds": aggregate(diffs), "APM600_APL600_by_seed": sizes,
            "scale_tradeoff": tradeoff, "scale_tradeoff_label": TRADEOFF if tradeoff else None,
            "new_seeds_only_rule": {s: rule[s]["small_gain_repeated"] for s in ("1", "2")}}


# ================================================================ read-only references
def _digest_checked(path, want=None):
    body = K.read_json(path)
    inner = K.sha_bytes(K.canonical({k: v for k, v in body.items() if k != "contract_sha256"}))
    K.require(body["contract_sha256"] == inner and (want is None or inner == want), f"contract digest: {path}")
    return body


def seed0_references():
    """Seed 0 (read-only): the F_geo round of task 009 through r29_control.fg_references (contract, acceptance receipts,
    endpoint, metrics, all pinned) and the L0 round of R27-002 (contract, acceptance receipts, the state after update
    1, the formal ledger, the evaluation)."""
    import r29_control as R29
    from sparse_rtdetr.research import r27_l0_control as L0
    c9, inherited, fg = R29.fg_references()
    ev = K.read_json(inherited["eval-FG"]["path"])
    c0 = K.l0_contract()
    files = {k: K.ref(K.L0_ROUND / rel) for k, (rel, _) in L0_PINS.items()}
    bad = [k for k, (_, sha) in L0_PINS.items() if files[k]["sha256"] != sha]
    K.require(not bad, f"files of the L0 round differ from their pins: {bad}")
    ev0 = K.l0_eval_closed(c0)
    accept0 = K.read_json(files["accept_closed"]["path"])
    K.require(accept0["status"] == "PASS" and accept0["contract_sha256"] == c0["contract_sha256"] and len(accept0["first_updates"]) == 2
              and accept0["saved_u1"]["file"] == files["accept_u1"] and c0["identity_sha256"] == L0.IDENTITY_SHA256
              and c0["parent"] == c9["parent"] and c0["parent"]["sha256"] == K.PARENT_SHA, "L0 round acceptance receipts")
    K.require(ev["GT_counts"] == ev0["GT_counts"] == c9["reference"]["GT_counts"], "GT counts of the seed-0 evaluations")
    return c9, {
        "parent": c9["parent"], "R27_updates": c9["R27_updates"],
        "FG": {**fg, "eval_closed": inherited["eval-FG"], "accept_closed": inherited["accept-FG"], "values": ev["values"],
               "per_class_APL_top600": ev["per_class_APL_top600"], "per_class_APM_top600": ev["per_class_APM_top600"],
               "cocoeval_top600_tensors": ev["cocoeval_top600_tensors"], "demand": ev["demand"], "mapping": ev["mapping"]},
        "L0": {"round": str(K.L0_ROUND), "contract": K.ref(K.L0_FILES["contract"][0]), "contract_sha256": c0["contract_sha256"],
               "identity_sha256": c0["identity_sha256"], **files, "values": ev0["values"],
               "per_class_APL_top600": ev0["per_class_APL_top600"], "endpoint": ev0["endpoint"],
               "per_class_note": "R27-002 kept the per-class APL rows only (no COCOeval tensors): per-class APS / APM of L0 seed 0 "
                                 "are not available without rescoring, which this task does not do"}}


def seed_references():
    """Seeds 1 and 2: the e060 parents named by the S15 seed-1 / seed-2 replication contract and by
    scale_baseline.PARENT_SHAS, and the historical input ledgers of that replication."""
    from sparse_rtdetr.research import scale_baseline as S
    cref = K.ref(S15_ROUND / "formal-001-CONTRACT.json")
    K.require(cref["sha256"] == S15_PINS["contract_file"], "S15 seed-1/2 replication contract file")
    cs = _digest_checked(cref["path"], S15_PINS["contract"])
    seeds = {}
    for seed in SEEDS:
        block = cs["seeds"][str(seed)]
        parent = K.ref(block["parent"]["path"])
        ledger = K.ref(S15_ROUND / f"formal-001/seed{seed}/train-S15/UPDATES.jsonl")
        closed = K.ref(S15_ROUND / f"formal-001/seed{seed}/train-S15/CLOSED.json")
        body = K.read_json(closed["path"])
        K.require(block["seed"] == seed and parent["sha256"] == block["parent"]["sha256"] == S.PARENT_SHAS[seed]
                  and parent["bytes"] == block["parent"]["bytes"] and ledger["sha256"] == S15_PINS["ledger"][seed]
                  and closed["sha256"] == S15_PINS["closed"][seed] and body["status"] == "PASS" and body["seed"] == seed
                  and body["contract_sha256"] == cs["contract_sha256"], f"S15 seed {seed} parent / ledger / closed stage")
        seeds[str(seed)] = {"seed": seed, "parent": parent, "S15_ledger": ledger, "S15_train_closed": closed}
    return cs, {"S15_round": str(S15_ROUND), "S15_contract": {**cref, "contract_sha256": cs["contract_sha256"], "commit": cs["source_commit"]}}, seeds


def base_release():
    """The R29 release (commit 6c65651) this release is compared with."""
    cb = _digest_checked(BASE_ROUND / "CONTRACT.json")
    manifest = K.read_json(K.pinned(cb["source_manifest"]))
    K.require(cb["source_commit"] == manifest["commit"] == BASE_COMMIT and not manifest.get("dirty", False), "base release commit")
    return cb, manifest


def source_delta(old_files, new_files):
    changed = sorted(k for k in old_files if k in new_files and old_files[k] != new_files[k])
    added = sorted(k for k in new_files if k not in old_files)
    removed = sorted(k for k in old_files if k not in new_files)
    K.require(not removed and changed == sorted(SCIENCE_CHANGED) and set(added) <= set(ALLOWED_ADDED),
              f"release differs from the R29 release outside the seed argument and the R30 tools: changed {changed}, added {added}, removed {removed}")
    return {"against": f"release-001 of {BASE_ROUND.name} (commit {BASE_COMMIT})", "changed": changed, "added": added, "removed": removed,
            "unchanged_files": len(old_files) - len(changed),
            "note": "two science files gain a seed argument with default 0 (create, build_eval_model); every other file of src/, vendor/, "
                    "configs/ and every existing tool (loader, engine, checkpoint, input pipeline, window delivery, loops, evaluation) is "
                    "byte-identical; the R30 tools are added files"}


def edits(old_root, new_root):
    out = {}
    for rel, want in SCIENCE_EDITS.items():
        d = GC.function_delta((Path(old_root) / rel).read_text(), (Path(new_root) / rel).read_text())
        K.require(d["changed"] == want and not d["added"] and not d["removed"],
                  f"{rel}: function edits differ from the registered list: {d['changed']} {d['added']} {d['removed']}")
        out[rel] = {"changed": d["changed"], "added": [], "removed": [], "unchanged_functions": len(d["unchanged"])}
    return out


# ================================================================ contract
def prepare(_args=None):
    from sparse_rtdetr.research import r27_l0_control as L0
    from sparse_rtdetr.research import r28_fovea_geo as FG
    round_ = K.ROUND.resolve()
    K.require(K.ROOT.resolve().parent.parent == round_ and K.ROOT.name == "source", "prepare-r30 must run from a frozen release")
    K.require("R28_ROUND" in os.environ and (round_.parent == K.LD_ROOT.resolve() or (K.DEV and round_.parent.parent == K.LD_ROOT.resolve())),
              "the R30 round is an independent directory named by R28_ROUND (a dev round lives inside it)")
    manifest = K.read_json(K.ROOT.parent / "MANIFEST.json")
    K.require(manifest["files"] == K.B.source_manifest(K.ROOT), "release source differs from MANIFEST")
    K.require(not manifest.get("dirty", False) or K.DEV, "a release assembled from a dirty work tree is only accepted in a dev round")
    task = K.ref(K.ROUND / TASK[0])
    K.require(task["sha256"] == TASK[1], "task file SHA")
    c9, seed0 = seed0_references()
    cs, s15, seeds = seed_references()
    cb, base_manifest = base_release()
    delta = source_delta(base_manifest["files"], manifest["files"])
    function_edits = edits(cb["source_root"], K.ROOT)
    fu9 = c9["followup"]
    K.require(fu9["W_demand"]["sha256"] == DEMAND_SHA and seed0["FG"]["demand"]["file"] == fu9["W_demand"]
              and FG.check_selection(fu9["width_selection"])["fwhm_hex"] == FG.WIDTH_HEX and c9["fovea_geo_definition"] == FG.DEFINITION
              and c9["identity_sha256"]["FG"] == FG.IDENTITY_SHA256["FG"] and c9["L0_contract"]["identity_sha256"] == L0.IDENTITY_SHA256,
              "frozen demand network / kernel width / identities")
    for k in ("data", "development_ids", "evaluation_policy", "environment", "python", "gpu_uuid"):
        K.require(cs[k] == c9[k], f"the S15 seed-1/2 contract and the F_geo contract name different {k}")
    units = {}
    for u in ORDER:
        arm, seed = parse(u)
        units[u] = {"arm": arm, "seed": seed, "parent": seeds[str(seed)]["parent"],
                    "identity_sha256": L0.IDENTITY_SHA256 if arm == "L0" else FG.IDENTITY_SHA256["FG"],
                    "reference_ledger": seeds[str(seed)]["S15_ledger"], "formal_directory": f"formal-{u}", "evaluation_directory": f"eval-{u}",
                    **({"demand": fu9["W_demand"], "mapping": seed0["FG"]["mapping"]} if arm == "FG" else {})}
    state_bytes = seed0["L0"]["accept_u1"]["bytes"]
    states = len(ORDER) * len(K.SAVE_EPOCHS)
    plan = {"state_bytes": state_bytes, "formal_states": states, "formal_states_bytes": states * state_bytes,
            "acceptance_states": len(ORDER), "acceptance_states_bytes": len(ORDER) * state_bytes,
            "ledgers_timing_observations_bytes_estimate": 2 * 80 * 1024 ** 2 + 2 * 50 * 1024 ** 2,
            "predictions_metrics_tensors_bytes_estimate": len(ORDER) * 160 * 1024 ** 2,
            "failure_margin_bytes": 4 * 1024 ** 3, "disk_free_bytes_at_prepare": shutil.disk_usage(K.ROUND).free,
            "save_cadence": "task 009: the e060 start state and every 5 epochs (13 full states per unit)",
            "gpu_time_reference": "task estimate 6-10 h per unit, 25-40 h for the four units; a scheduling estimate, not a cap"}
    plan["total_bytes_estimate"] = (plan["formal_states_bytes"] + plan["acceptance_states_bytes"] + plan["ledgers_timing_observations_bytes_estimate"]
                                    + plan["predictions_metrics_tensors_bytes_estimate"])
    K.require(plan["disk_free_bytes_at_prepare"] > 4 * (plan["total_bytes_estimate"] + plan["failure_margin_bytes"]),
              "less free space than four times the planned storage of this round")
    c = {"version": "R30-002-contract-v1", "authorization_basis": BASIS, "registration": REGISTRATION, "task": task,
         "units": units, "training_order": list(ORDER), "arms": list(ARMS), "seeds": seeds, **s15, "seed0": seed0,
         "demand": {"weights": fu9["W_demand"], "binding": seed0["FG"]["demand"], "trained_on_seed": 0, "retrained": False,
                    "shared_by_seeds": [0, 1, 2]},
         "width_selection": fu9["width_selection"],
         "parent_update": K.PARENT_UPDATE, "endpoint_update": K.ENDPOINT,
         "data": c9["data"], "data_receipt": c9["data_receipt"], "development_ids": c9["development_ids"],
         "evaluation_policy": c9["evaluation_policy"], "environment": c9["environment"], "python": c9["python"], "gpu_uuid": c9["gpu_uuid"],
         "recipe": c9["recipe"], "identity": {"L0": L0.IDENTITY, "FG": FG.IDENTITY["FG"]},
         "identity_sha256": {"L0": L0.IDENTITY_SHA256, "FG": FG.IDENTITY_SHA256["FG"]},
         "definition": c9["definition"], "fovea_definition": c9["fovea_definition"], "fovea_geo_definition": c9["fovea_geo_definition"],
         "reference": {"GT_counts": c9["reference"]["GT_counts"],
                       "values": {"FG-s0": seed0["FG"]["values"], "L0-s0": seed0["L0"]["values"]},
                       "seed0_F_geo_minus_L0_pp": difference_pp(seed0["FG"]["values"], seed0["L0"]["values"])},
         "reading": READING, "statements": STATEMENTS, "workload": WORKLOAD, "input_pipeline": INPUT_PIPELINE,
         "ledger_comparison": LEDGER_COMPARISON, "shared_fields": list(K.SHARED_FIELDS),
         "acceptance": {**ACCEPT, "wire": WIRE, "scope": SCOPE}, "resource_plan": plan,
         "base_release": {"round": str(BASE_ROUND), "contract_sha256": cb["contract_sha256"], "commit": cb["source_commit"],
                          "manifest": cb["source_manifest"]},
         "source_delta": delta, "function_edits": function_edits,
         "source_root": str(K.ROOT.resolve()), "source_manifest": K.ref(K.ROOT.parent / "MANIFEST.json"),
         "source_commit": manifest["commit"], "source_dirty": bool(manifest.get("dirty", False)),
         "not_authorised": ["any seed of W, K, F, P, C", "kernel width, magnification, demand rule or schedule changes",
                            "demand network retraining", "extension beyond e120", "intermediate endpoint evaluation", "test-dev",
                            "arms added after seeing results", "timing / cost stages"],
         "outputs": str(K.ROUND), "formal_started": False}
    K.require(single_seed(c["reference"]["seed0_F_geo_minus_L0_pp"])["small_gain_repeated"] is True, "the seed-0 reading does not satisfy its own rule")
    K.require(len(c["development_ids"]) == 548, "development ids")
    c["contract_sha256"] = K.sha_bytes(K.canonical(c))
    K.write_new(K.ROUND / "CONTRACT.json", c)
    print(json.dumps({"contract_sha256": c["contract_sha256"], "identity_sha256": c["identity_sha256"], "authorization_basis": BASIS,
                      "units": {u: v["parent"]["sha256"] for u, v in units.items()}, "source_delta": {k: delta[k] for k in ("changed", "added")},
                      "storage_total_bytes_estimate": plan["total_bytes_estimate"]}))
    return c


def load_contract(path=None):
    from sparse_rtdetr.research import r27_l0_control as L0
    from sparse_rtdetr.research import r28_fovea_geo as FG
    from sparse_rtdetr.research import scale_baseline as S
    c = _digest_checked(path or K.ROUND / "CONTRACT.json")
    K.require(c.get("authorization_basis") == BASIS and c.get("version") == "R30-002-contract-v1", "not the R30-002 contract")
    K.require(Path(c["source_root"]).resolve() == K.ROOT.resolve(), "not running from the bound release")
    K.require(Path(c["outputs"]).resolve() == K.ROUND.resolve(), "contract bound to another round")
    manifest = K.read_json(K.pinned(c["source_manifest"]))
    K.require(manifest["commit"] == c["source_commit"], "MANIFEST commit differs from the contract")
    K.B.release_intact(K.ROOT, manifest["files"])
    K.require(c["identity_sha256"] == {"L0": L0.IDENTITY_SHA256, "FG": FG.IDENTITY_SHA256["FG"]} and c["fovea_geo_definition"] == FG.DEFINITION
              and list(c["units"]) == list(ORDER) == c["training_order"] and c["reading"] == READING,
              "R30 behaviour identity / units / reading differ from the contract")
    for u, v in c["units"].items():
        K.require((v["arm"], v["seed"]) == parse(u) and v["parent"]["sha256"] == S.PARENT_SHAS[v["seed"]]
                  and v["parent"] == c["seeds"][str(v["seed"])]["parent"], f"unit {u} parent")
    K.require(c["seed0"]["parent"]["sha256"] == S.PARENT_SHAS[0] and c["demand"]["weights"]["sha256"] == DEMAND_SHA, "seed-0 parent / demand file")
    return c


def stage_closed(c, name):
    path = K.ROUND / name / "CLOSED.json"
    K.require(path.is_file(), f"stage not closed: {path}")
    closed = K.read_json(path)
    K.require(closed.get("status") == "PASS" and closed.get("contract_sha256") == c["contract_sha256"], f"stage {name} is not PASS under this contract")
    return closed


def demand_source(c):
    """The formal demand source of arm FG on every seed: the frozen network file of the W round and the fixed width."""
    from sparse_rtdetr.research import r28_fovea_geo as FG
    sel = FG.check_selection(c["width_selection"])
    return {"kind": "network", "file": K.pinned(c["demand"]["weights"]) and c["demand"]["weights"], "fwhm": float.fromhex(sel["fwhm_hex"]),
            "selection": sel}


def parent_of(c, seed):
    return c["seed0"]["parent"] if seed == 0 else c["seeds"][str(seed)]["parent"]


def bound_seed():
    import formal_data
    return getattr(formal_data, "_s15_bound_seed", None)


def create(c, arm, seed, device="cuda:0"):
    """Engine of one arm from the e060 parent of one seed.  The process must be bound to that seed (bind_seed)."""
    from sparse_rtdetr.research import r27_l0_control as L0
    from sparse_rtdetr.research import r28_fovea_geo as FG
    K.require(arm in ARMS and seed in (0,) + SEEDS and bound_seed() == seed, f"arm {arm} / seed {seed} / process seed {bound_seed()}")
    parent = K.pinned(parent_of(c, seed))
    if arm == "L0":
        e, checks = L0.create(str(parent), device, seed=seed)
        K.require(getattr(e, "l0_arm", None) == "L0" and not hasattr(e, "r28_arm"), "not an L0 engine")
    else:
        source = demand_source(c)
        e, checks = FG.create(str(parent), device, "FG", source, seed=seed)
        K.require(e.r28_arm == "FG" and e.r28_mapping == FG.mapping(source) == c["seed0"]["FG"]["mapping"]
                  and e.r28_ctx.binding() == c["demand"]["binding"] and FG.is_wrapped(e.model) and FG.is_wrapped(e.ema.module),
                  "not the F_geo engine of task 009 (demand source / mapping)")
    K.require(e.config.seed == seed and e.r25_parent == parent_of(c, seed)["sha256"] and checks["parent_sha256"] == e.r25_parent
              and e.completed_updates == K.PARENT_UPDATE, "engine seed / parent / clock")
    return e, checks


def engine_arm(e):
    if getattr(e, "l0_arm", None) == "L0" and not hasattr(e, "r28_arm"):
        return "L0"
    K.require(getattr(e, "r28_arm", None) == "FG" and not hasattr(e, "l0_arm") and not hasattr(e, "r27_arm"), "engine is neither L0 nor FG")
    return "FG"


# ================================================================ states (schema r30-state-v1)
def binding(c, e, unit, scope, audit=None, role=None):
    from formal_data import cursor
    arm, seed = parse(unit)
    K.require(unit in c["units"] and engine_arm(e) == arm and e.config.seed == seed and bound_seed() == seed
              and e.r25_parent == c["units"][unit]["parent"]["sha256"], f"engine is not the engine of unit {unit}")
    cur = cursor(e.completed_updates)
    K.require(cur["seed"] == seed, "cursor seed")
    return {"schema": SCHEMA, "scope": scope, "unit": unit, "arm": arm, "seed": seed, "contract_sha256": c["contract_sha256"],
            "identity_sha256": c["identity_sha256"][arm], "parent_sha256": c["units"][unit]["parent"]["sha256"],
            "global_update": e.completed_updates, "cursor": cur, "config": e.config.binding(), "audit": audit, "role": role,
            "activation_checkpoint": e.s15_activation_checkpoint, "torch": str(K.torch_mod().__version__),
            **({"demand": e.r28_ctx.binding(), "mapping": e.r28_mapping} if arm == "FG" else {})}


def save(c, e, unit, path, scope, audit=None, role=None):
    from sparse_rtdetr.research import scale_baseline as S
    from sparse_rtdetr.protocol_r18.engine import assert_healthy
    torch = K.torch_mod()
    path = Path(path)
    K.require(e.ema.updates == e.completed_updates, "EMA clock at save")
    K.require(not getattr(e, "r28_check", False), "a check engine cannot write an R30 state")
    assert_healthy(e.model, e.optimizer, e.ema)
    state = S.capture(e)
    b = binding(c, e, unit, scope, audit, role)
    state["r30_binding_sha256"] = K.sha_bytes(K.canonical(b))
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("xb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.link(tmp, path)
    tmp.unlink()
    side = {"status": "COMPLETE", **b, "file": K.ref(path)}
    K.write_new(K.side_of(path), side)
    return side


def read_side(c, path, scope, unit):
    path = Path(path).resolve()
    side = K.read_json(K.side_of(path))
    arm, seed = parse(unit)
    for k, want in (("schema", SCHEMA), ("unit", unit), ("arm", arm), ("seed", seed), ("scope", scope),
                    ("contract_sha256", c["contract_sha256"]), ("parent_sha256", c["units"][unit]["parent"]["sha256"]),
                    ("identity_sha256", c["identity_sha256"][arm])):
        K.require(side.get(k) == want, f"R30 state refused: {k}={side.get(k)!r}, expected {want!r} ({path})")
    K.require(side.get("status") == "COMPLETE", f"R30 state side-car is not COMPLETE ({path})")
    return path, side


def read_state(c, path, scope, unit):
    torch = K.torch_mod()
    path, side = read_side(c, path, scope, unit)
    K.require(side["file"] == K.ref(path), "R30 state file identity")
    s = torch.load(path, map_location="cpu", weights_only=True)
    expect = {k: v for k, v in side.items() if k not in ("status", "file")}
    K.require(s.pop("r30_binding_sha256") == K.sha_bytes(K.canonical(expect)), "R30 binding digest")
    K.require(s["engine"]["completed_updates"] == side["global_update"] == s["ema"]["updates"], "R30 state clock")
    return side, s


def load(c, e, unit, path, scope):
    """Exact restore into a fresh engine of the same unit."""
    from sparse_rtdetr.research import scale_baseline as S
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    arm, seed = parse(unit)
    K.require(engine_arm(e) == arm and e.config.seed == seed and bound_seed() == seed, f"engine is not the engine of unit {unit}")
    side, s = read_state(c, path, scope, unit)
    K.require(side["config"] == e.config.binding() and side["activation_checkpoint"] == e.s15_activation_checkpoint, "R30 state config / memory mode")
    if arm == "FG":
        K.require(side["demand"] == e.r28_ctx.binding() and side["mapping"] == e.r28_mapping, "R30 state demand source / mapping width")
    K.require(s["optimizer_names"] == [g["param_names"] for g in e.optimizer.param_groups], "R30 optimizer names")
    e.model.load_state_dict(s["model"], strict=True)
    e.optimizer.load_state_dict(s["optimizer"])
    e.ema.load_state_dict(s["ema"])
    modes = dict(e.model.named_modules())
    K.require(set(modes) == set(s["model_modes"]), "R30 mode keys")
    for n, m in s["model_modes"].items():
        modes[n].training = m
    e.load_state_dict(s["engine"])
    ck.restore_rng(s["rng"], e.device)
    K.require(ck.exact_equal(S.capture(e), s), "R30 full state did not restore exactly")
    return side


# ================================================================ reference ledgers
def reference_rows(c, seed):
    """Rows of the reference ledger of a seed by update: seed 0 -> the pinned R27 formal-P / formal-C ledgers (the rule of
    R27-002 and 009); seeds 1 / 2 -> the pinned S15 replication ledger of that seed.  Returns (rows, first reading)."""
    import l0_loop as LL
    if seed == 0:
        refs, first = LL.reference_rows({"R27_updates": c["seed0"]["R27_updates"]})
        return refs, {"update": first["update"], "reference": "R27 formal-P", "loss": first["P"]}
    path = K.pinned(c["seeds"][str(seed)]["S15_ledger"])
    rows = []
    with path.open() as f:
        for line in f:
            rows.append({k: v for k, v in json.loads(line).items() if k in ("input", "loss") + K.TOP_FIELDS})
    K.require(len(rows) == K.ENDPOINT - K.PARENT_UPDATE, f"S15 seed {seed} ledger rows: {len(rows)}")
    for i, r in enumerate(rows):
        K.require(r["input"]["update"] == K.PARENT_UPDATE + 1 + i and r["ema_updates"] == r["input"]["update"], "S15 ledger update sequence")
        K.shared(r)
    return {r["input"]["update"]: r for r in rows}, {"update": K.PARENT_UPDATE + 1, "reference": f"S15 seed {seed} replication", "loss": rows[0]["loss"]}


def reference_check(refs, label):
    import l0_loop as LL

    def check(inp):
        K.require(inp["update"] in refs, f"window {inp['update']} is not an update of the schedule")
        bad = LL.input_differences(inp, refs[inp["update"]]["input"])
        K.require(not bad, f"input record differs from the {label} ledger before update {inp['update']} "
                           f"(the model was not updated with this window): {bad}")

    return check


def gpu_compute_processes():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout
    rows = [line.strip() for line in out.splitlines() if line.strip()]
    return {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "compute_processes": rows, "own_pid": os.getpid(),
            "others": [r for r in rows if r.split(",")[0].strip() != str(os.getpid())]}


# ================================================================ r30-check (CPU)
def _own(c, out, name):
    out = Path(out)
    K.require(out.resolve() == (K.ROUND / name).resolve(), f"this action writes <round>/{name}")
    return out


def r30_check(c, out, counts):
    import l0_checks_state as H
    import l0_loop as LL
    import r28_eval  # noqa: F401  (import check of the shared inference)
    import r28_followup  # noqa: F401
    import r30_loop as RL
    from sparse_rtdetr.research import r27_l0_control as L0
    from sparse_rtdetr.research import r28_fovea_geo as FG
    from sparse_rtdetr.research import scale_baseline as S
    torch = K.torch_mod()
    out = _own(c, out, "r30-check")
    K.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_available(), "the R30 check hides CUDA")
    t0 = time.perf_counter()
    res = {"stage": "r30-check", "registration": c["registration"], "authorization_basis": c["authorization_basis"],
           "workload": c["workload"]["r30-check"]}

    K.PHASE["phase"] = "r30-check:pytest"
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *[str(K.ROOT / t) for t in TESTS]],
                       capture_output=True, text=True, cwd=str(K.ROOT),
                       env={k: v for k, v in {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}.items() if not k.startswith(("R28_", "R30_"))})
    res["pytest"] = {"returncode": p.returncode, "tail": p.stdout.strip().splitlines()[-1:], "tests": list(TESTS),
                     "note": "synthetic unit tests in a subprocess; their forwards are not in this stage's counts"}
    K.require(p.returncode == 0, "synthetic pytest failed:\n" + p.stdout[-6000:])

    K.PHASE["phase"] = "r30-check:references"
    c9, seed0 = seed0_references()
    cs, s15, seeds = seed_references()
    cb, base_manifest = base_release()
    manifest = K.read_json(K.pinned(c["source_manifest"]))
    K.require(c["seed0"] == seed0 and c["seeds"] == seeds and all(c[k] == v for k, v in s15.items())
              and K.pinned(c["task"]) == (K.ROUND / TASK[0]).resolve() and c["task"]["sha256"] == TASK[1], "references changed since prepare")
    delta = source_delta(base_manifest["files"], manifest["files"])
    K.B.release_intact(Path(cb["source_root"]), base_manifest["files"])
    function_edits = edits(cb["source_root"], K.ROOT)
    K.require(delta == c["source_delta"] and function_edits == c["function_edits"], "source delta changed since prepare")
    K.require(S.PARENT_SHAS == {0: c["seed0"]["parent"]["sha256"], **{s: c["seeds"][str(s)]["parent"]["sha256"] for s in SEEDS}}
              and len(set(S.PARENT_SHAS.values())) == 3, "parents: scale_baseline.PARENT_SHAS against the contract")
    for k in ("recipe", "data", "evaluation_policy", "environment", "python", "gpu_uuid", "development_ids", "definition", "fovea_definition",
              "fovea_geo_definition"):
        K.require(c[k] == c9[k], f"contract key {k} differs from the F_geo contract")
    K.require(c["identity"]["L0"] == L0.IDENTITY and c["identity"]["FG"] == FG.IDENTITY["FG"] == c9["identity"]["FG"]
              and c["workload"] == WORKLOAD and c["input_pipeline"] == INPUT_PIPELINE and c["ledger_comparison"] == LEDGER_COMPARISON
              and c["acceptance"] == {**ACCEPT, "wire": WIRE, "scope": SCOPE} and c["statements"] == STATEMENTS, "frozen blocks of the contract")
    source = demand_source(c)
    K.require(FG.mapping(source) == c["seed0"]["FG"]["mapping"] and source["file"]["sha256"] == DEMAND_SHA, "demand source / mapping of arm FG")
    res["references"] = {"F_geo_contract_sha256": c9["contract_sha256"], "L0_contract_sha256": c["seed0"]["L0"]["contract_sha256"],
                         "S15_contract_sha256": cs["contract_sha256"], "base_release_commit": cb["source_commit"],
                         "parents": {str(k): v for k, v in S.PARENT_SHAS.items()}, "demand_sha256": DEMAND_SHA,
                         "kernel_width_hex": FG.WIDTH_HEX, "source_delta": delta, "function_edits": function_edits}

    K.PHASE["phase"] = "r30-check:seed-0 defaults"
    import inspect
    sig = {f"{m.__name__.rsplit('.', 1)[-1]}.{fn}": str(inspect.signature(getattr(m, fn))) for m in (L0, FG) for fn in ("create", "build_eval_model")}
    K.require(all(inspect.signature(getattr(m, fn)).parameters["seed"].default == 0 for m in (L0, FG) for fn in ("create", "build_eval_model")),
              "the seed argument of create / build_eval_model must default to 0")
    res["seed_argument"] = {"signatures": sig, "default": 0,
                            "note": "with the default the two modules are the frozen 009 / R27-002 code path; the CUDA stage wire-s0 compares "
                                    "that path with the stored seed-0 receipts update by update"}

    K.PHASE["phase"] = "r30-check:reference ledgers"
    refs0, _ = reference_rows(c, 0)
    ledgers = {}
    for seed in SEEDS:
        refs, first = reference_rows(c, seed)
        K.require(sorted(refs) == sorted(refs0), "update set of the S15 ledger")
        schedule = [k for k in ("epoch", "window", "augmentation_strong") if any(refs[u]["input"][k] != refs0[u]["input"][k] for u in refs)]
        clock = [k for k in ("lr", "ema_updates") if any(refs[u][k] != refs0[u][k] or type(refs[u][k]) is not type(refs0[u][k]) for u in refs)]
        same_ids = sum(refs[u]["input"]["image_ids_sha256"] == refs0[u]["input"]["image_ids_sha256"] for u in refs)
        K.require(not schedule and not clock and same_ids == 0, f"S15 seed {seed} ledger: schedule {schedule}, clocks {clock}, windows equal to seed 0 {same_ids}")
        ledgers[str(seed)] = {"file": c["seeds"][str(seed)]["S15_ledger"], "rows": len(refs), "first_update": min(refs), "last_update": max(refs),
                              "schedule_columns_equal_to_seed0": ["epoch", "window", "augmentation_strong"],
                              "clock_columns_equal_to_seed0": ["lr", "ema_updates"], "windows_with_the_seed0_image_set": same_ids,
                              "first_update_loss_of_S15": first["loss"]}
    r1, r2 = reference_rows(c, 1)[0], reference_rows(c, 2)[0]
    K.require(sum(r1[u]["input"]["image_ids_sha256"] == r2[u]["input"]["image_ids_sha256"] for u in r1) == 0,
              "the seed-1 and seed-2 ledgers share a window")
    res["reference_ledgers"] = ledgers

    K.PHASE["phase"] = "r30-check:refusals"
    refusals = [
        H.refused("a unit name outside the four units", lambda: parse("W-s1")),
        H.refused("seed 3", lambda: parse("FG-s3")),
        H.refused("an engine of seed 1 in a process bound to seed 0", lambda: create(c, "L0", 1, "cpu")),
        H.refused("formal directory of a unit outside the order", lambda: RL.guard(c, K.ROUND / "formal-FG-s0", None, "FG-s0")),
        H.refused("formal run before the checks closed", lambda: RL.guard(c, K.ROUND / "formal-L0-s1", None, "L0-s1")),
        H.refused("the F_geo e120 state of task 009 read as an R30 state",
                  lambda: read_side(c, c["seed0"]["FG"]["FG_endpoint"]["path"], "formal", "FG-s1")),
        H.refused("the L0 acceptance state of R27-002 read as an R30 state",
                  lambda: read_side(c, c["seed0"]["L0"]["accept_u1"]["path"], SCOPE, "L0-s1")),
        H.refused("a three-seed label from two seeds", lambda: label([True, True])),
    ]
    res["refused"] = refusals
    s0 = summary({"0": {"FG": c["seed0"]["FG"]["values"], "L0": c["seed0"]["L0"]["values"]},
                  "1": {"FG": c["seed0"]["FG"]["values"], "L0": c["seed0"]["L0"]["values"]},
                  "2": {"FG": c["seed0"]["L0"]["values"], "L0": c["seed0"]["FG"]["values"]}})
    K.require(s0["label"] == LABELS[1] and s0["F_geo_minus_L0_pp"]["0"] == c["reference"]["seed0_F_geo_minus_L0_pp"]
              and s0["single_seed_rule"]["2"]["small_gain_repeated"] is False, "summary arithmetic self-check")
    res["summary_arithmetic_self_check"] = {"seed0_F_geo_minus_L0_pp": s0["F_geo_minus_L0_pp"]["0"],
                                            "note": "arithmetic exercised on the quoted seed-0 values and their sign-flipped copy; not a result"}
    res["seconds"] = time.perf_counter() - t0
    K.require(counts["actual_updates"] == 0 and counts["forwards"] == 0, "the CPU check ran a model")
    return res


# ================================================================ short CUDA runs shared by wire-s0 and accept-sN
def _drive(c, out, counts, digester, refs, label, *, name, arm, seed, pipeline, mode, first, last, unit=None, load_from=None,
           save_after=None, after=None):
    import l0_loop as LL
    import r28_loop as LP
    import r28_pipeline as PL
    from s15_formal_train import _result_body
    from sparse_rtdetr.research import scale_baseline as S
    torch = K.torch_mod()
    K.PHASE["phase"] = f"{name}: construct"
    e, checks = create(c, arm, seed)
    K.register_counts(e, counts)
    obs = LP.install_observer(e) if arm == "FG" else None
    loaded = None if load_from is None else load(c, e, unit, load_from, SCOPE)
    K.require(e.completed_updates == first, f"{name}: start update")
    start = None if load_from is not None else CC.state_digests(S.capture(e))
    data = S.train_dataset(c["data"]["train_core"], S.S15_HW, seed=seed)
    delivery = RI.Delivery(mode, e.device)
    check = reference_check(refs, label)
    rows, saved = [], None
    state_file = None if unit is None else out / f"{unit}-u{ACCEPT['save_at']:02d}.pth"
    torch.cuda.reset_peak_memory_stats()
    with PL.MemorySampler(0.5) as memory:
        opened = mark = time.perf_counter()
        with delivery.stream(pipeline, data, first, last, check, fine=True) as stream:
            for item in stream:
                x, t, inp = item.x, item.t, item.inp
                K.PHASE["phase"] = f"{name}: update {e.completed_updates + 1}"
                K.require(inp["update"] == e.completed_updates + 1, f"{name}: window {inp['update']} is not the next update")
                bad = LL.input_differences(inp, refs[inp["update"]]["input"])
                K.require(not bad, f"{name}: input record differs from the {label} ledger before update {inp['update']}: {bad}")
                called = time.perf_counter()
                xd, moved = delivery.take(item)
                res = e.train_window(xd, t, capture=True)
                train_call = time.perf_counter() - called
                del xd
                K.require(res["update"] == inp["update"] and res["finite"] and e.ema.updates == e.completed_updates, f"{name}: update clock")
                science = {"input": inp, "gt_count": int(sum(len(v["labels"]) for v in t)), "result": _result_body(res)}
                if obs is not None:
                    science["observer"] = obs.update_row(t)
                bad = LL.shared_differences({"input": inp, "gt_count": science["gt_count"], "lr": res["lr"], "ema_updates": e.ema.updates},
                                            refs[inp["update"]])
                K.require(not bad, f"{name}: shared fields differ from the {label} ledger at update {res['update']}: {bad}")
                digester.submit((name, res["update"]), S.capture(e))
                extra = {}
                if save_after is not None and e.completed_updates == save_after:
                    if pipeline == "overlap":
                        prepared = stream.wait_ready()
                        K.require(prepared == 1, f"{name}: no prepared window at the save ({prepared})")
                    s1 = time.perf_counter()
                    saved = save(c, e, unit, state_file, SCOPE)
                    K.require(pipeline != "overlap" or stream.ready() == 1, f"{name}: the prepared window was consumed during the save")
                    extra = {"save_s": time.perf_counter() - s1}
                now = time.perf_counter()
                rows.append({"science": science,
                             "engineering": {"cycle_s": now - mark, "wait_s": item.timing["wait_s"], "train_call_s": train_call,
                                             "train_s": res["elapsed_seconds"], "prepared_after": stream.ready(),
                                             "ema_clock": e.ema.updates, **moved, **extra}})
                mark = now
                del item, x, t
            account = stream.account()
        closed = time.perf_counter()
    K.require(e.completed_updates == last and len(rows) == last - first, f"{name}: update count")
    if after is not None:
        after(e, unit, state_file)
    info = {"arm": arm, "seed": seed, "unit": unit, "pipeline": pipeline, "first_update": first + 1, "last_update": last, "updates": len(rows),
            "account": account, "window_delivery": delivery.account(), "resumed_from": None if loaded is None else loaded["file"],
            "saved": saved, "start_state": start, "wall_seconds_stream": closed - opened,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "memory": memory.summary(),
            "create_checks": {k: checks[k] for k in ("identity_sha256", "parent_sha256", "new_parameters", "parameters", "optimizer_parameters",
                                                     "old_parameters_sha256", "EMA_updates", "lr_next", "engine_config", "global_model_config")}}
    delivery.close()
    del e, obs, delivery
    gc.collect()
    torch.cuda.empty_cache()
    return rows, info


def _finish(out, digester, paths):
    digests = digester.finish()
    for name, rows in paths.items():
        for r in rows:
            r["science"]["state"] = digests[(name, r["science"]["input"]["update"])]
        with (out / f"STEPS-{name}.jsonl").open("x") as f:
            for r in rows:
                f.write(json.dumps(r, separators=(",", ":"), allow_nan=False) + "\n")
    return {n: K.ref(out / f"STEPS-{n}.jsonl") for n in paths}


def _sci(rows):
    return [r["science"] for r in rows]


L0_RECEIPT_FIELDS = ("input", "gt_count", "loss", "lr", "gradient_norm_before_clip", "ema_updates", "output_sha256", "dn_noise_sha256",
                     "gradient_sha256_before_clip")
L0_LEDGER_FIELDS = ("input", "gt_count", "loss", "lr", "gradient_norm_before_clip", "microbatch_losses", "ema_updates")
L0_CREATE_FIELDS = ("old_parameters_sha256", "lr_next", "parameters", "optimizer_parameters", "new_parameters", "EMA_updates", "engine_config",
                    "global_model_config", "identity_sha256", "parent_sha256")
L0_NOT_COMPARABLE = [
    "model / optimizer / EMA / RNG state after update 2 (R27-002 stored the full state after update 1 only; its acceptance compared the "
    "state after update 2 with its own replay and kept no digest)",
    "result.normalizers, result.coefficients, result.microsteps, result.optimizer_steps, result.finite (not written into the R27-002 "
    "acceptance receipt or ledger)",
    "result.microbatch_losses in the acceptance receipt (compared instead with rows 1-2 of the R27-002 formal ledger, which carry it)",
    "elapsed / update seconds (wall time)",
]


def _l0_seed0_differences(c, rows, info):
    """L0 at seed 0 against everything R27-002 stored for its first two updates."""
    torch = K.torch_mod()
    s0 = c["seed0"]["L0"]
    accept = K.read_json(K.pinned(s0["accept_closed"]))
    theirs = accept["first_updates"]
    mine = []
    for full in rows:
        r, res = full["science"], full["science"]["result"]
        # the stored rows carry the EMA clock of the engine after the update (result.ema_updates is the per-update count)
        mine.append({"input": r["input"], "gt_count": r["gt_count"], "loss": res["loss"], "lr": res["lr"],
                     "gradient_norm_before_clip": res["gradient_norm_before_clip"], "ema_updates": full["engineering"]["ema_clock"],
                     "output_sha256": res["output_sha256"], "dn_noise_sha256": res["dn_noise_sha256"],
                     "gradient_sha256_before_clip": res["gradient_sha256_before_clip"], "microbatch_losses": res["microbatch_losses"]})
    out = {"acceptance_receipt": CC.differences([{k: t[k] for k in L0_RECEIPT_FIELDS} for t in theirs],
                                                [{k: m[k] for k in L0_RECEIPT_FIELDS} for m in mine])}
    ledger = []
    with K.pinned(s0["ledger"]).open() as f:
        for _ in range(len(mine)):
            ledger.append(json.loads(f.readline()))
    out["formal_ledger_rows"] = CC.differences([{k: t[k] for k in L0_LEDGER_FIELDS} for t in ledger],
                                               [{k: m[k] for k in L0_LEDGER_FIELDS} for m in mine])
    out["create_checks"] = CC.differences({k: accept["create_checks"][k] for k in L0_CREATE_FIELDS},
                                          {k: info["create_checks"][k] for k in L0_CREATE_FIELDS})
    stored = torch.load(K.pinned(s0["accept_u1"]), map_location="cpu", weights_only=True)
    stored.pop("l0_binding_sha256")
    K.require(stored["engine"]["completed_updates"] == K.PARENT_UPDATE + 1, "the stored R27-002 acceptance state is not the state after update 1")
    out["state_after_update_1"] = CC.differences(CC.state_digests(stored), rows[0]["science"]["state"])
    del stored
    return out


def r30_wire(c, out, counts):
    """Seed 0, not a result: the present code against the stored receipts of task 009 (FG) and R27-002 (L0)."""
    out = _own(c, out, "wire-s0")
    stage_closed(c, "r30-check")
    seed = process_seed((0,))
    K.require(bound_seed() == 0, "wire-s0 runs in a process bound to seed 0")
    gpu = gpu_compute_processes()
    refs, _ = reference_rows(c, 0)
    p0, n = K.PARENT_UPDATE, WIRE["updates_per_path"]
    digester = CC.Digester()
    paths, info = {}, {}
    for arm in ARMS:
        for pipeline, mode in WIRE["paths"]:
            name = f"{arm}-s0-{pipeline}-{mode}"
            paths[name], info[name] = _drive(c, out, counts, digester, refs, "R27 formal-P", name=name, arm=arm, seed=seed,
                                             pipeline=pipeline, mode=mode, first=p0, last=p0 + n)
    K.PHASE["phase"] = "wire-s0: comparison"
    files = _finish(out, digester, paths)
    stored = []
    with K.pinned(c["seed0"]["FG"]["FG_accept_steps"]).open() as f:
        for line in f:
            stored.append(json.loads(line)["science"])
    K.require(len(stored) >= n and [r["input"]["update"] for r in stored[:n]] == [p0 + 1 + i for i in range(n)], "stored accept-FG receipts")
    fg = {name: CC.differences(stored[:n], _sci(rows)) for name, rows in paths.items() if name.startswith("FG-")}
    l0 = {name: _l0_seed0_differences(c, rows, info[name]) for name, rows in paths.items() if name.startswith("L0-")}
    names = {a: [f"{a}-s0-{p}-{m}" for p, m in WIRE["paths"]] for a in ARMS}
    between = {a: CC.differences(_sci(paths[names[a][0]]), _sci(paths[names[a][1]])) for a in ARMS}
    cross = [a["input"] == b["input"] and a["gt_count"] == b["gt_count"] and a["result"]["lr"] == b["result"]["lr"]
             for a, b in zip(_sci(paths[names["L0"][0]]), _sci(paths[names["FG"][0]]))]
    want = c["workload"]["wire-s0"]
    K.require(counts["actual_updates"] == want["updates"] == 8 and counts["forwards"] == counts["forward_attempts"] == want["forwards"],
              f"wire-s0 counts {dict(counts)}")
    equal = {"FG_equal_to_the_accept_FG_receipts": {k: not v for k, v in fg.items()},
             "L0_equal_to_the_R27_002_receipts": {k: not any(v.values()) for k, v in l0.items()},
             "formal_path_equal_to_the_reference_path": {a: not between[a] for a in ARMS}}
    body = {"schema": "r30-seed0-wiring-v1", "contract_sha256": c["contract_sha256"], "seed": 0, "paths": info, "steps_files": files,
            "FG": {"receipts": c["seed0"]["FG"]["FG_accept_steps"], "steps": n, "differences": fg, "compared": CC.COMPARED, "excluded": CC.EXCLUDED,
                   "not_comparable": []},
            "L0": {"receipts": {k: c["seed0"]["L0"][k] for k in ("accept_closed", "accept_u1", "ledger")}, "steps": n, "differences": l0,
                   "compared": {"acceptance_receipt": list(L0_RECEIPT_FIELDS), "formal_ledger_rows": list(L0_LEDGER_FIELDS),
                                "create_checks": list(L0_CREATE_FIELDS),
                                "state_after_update_1": "every part of the captured state (model, optimizer, EMA, engine clocks, RNG, BN buffers)"},
                   "not_comparable": L0_NOT_COMPARABLE},
            "reference_path_against_formal_path": {"paths": WIRE["paths"], "differences": between,
                                                   "note": "all science fields and the state digests after every update"},
            "L0_and_FG_consume_the_same_windows": cross, "equal": equal, "gpu_at_start": gpu, "result_status": "not a result (seed 0, 2 updates)"}
    K.write_new(out / "WIRE_EQUIVALENCE.json", body)
    K.require(all(equal["FG_equal_to_the_accept_FG_receipts"].values()), f"FG at seed 0 differs from the stored accept-FG receipts: { {k: v[:5] for k, v in fg.items() if v} }")
    K.require(all(equal["L0_equal_to_the_R27_002_receipts"].values()), f"L0 at seed 0 differs from the stored R27-002 receipts: {l0}")
    K.require(all(equal["formal_path_equal_to_the_reference_path"].values()) and all(cross), f"paths differ at seed 0: {between}")
    return {"stage": "wire-s0", "seed": 0, "workload": want, **equal, "L0_not_comparable": L0_NOT_COMPARABLE, "FG_not_comparable": [],
            "equivalence": K.ref(out / "WIRE_EQUIVALENCE.json"), "updates": want["updates"], "complete_detector_forwards": want["forwards"],
            "peak_allocated_bytes": {n_: info[n_]["peak_cuda_allocated_bytes"] for n_ in info}, "gpu_at_start": gpu, "formal_state_usable": False}


def delivery_candidate():
    forced = os.environ.get("R30_DELIVERY")
    K.require(forced in (None, "host"), "R30_DELIVERY can only select the host reference delivery")
    return "host" if forced else RI.FORMAL


def r30_accept(c, out, counts):
    import l0_checks_state as H
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    from sparse_rtdetr.research import scale_baseline as S
    seed = process_seed(SEEDS)
    out = _own(c, out, f"accept-s{seed}")
    stage_closed(c, "r30-check")
    stage_closed(c, "wire-s0")
    K.require(bound_seed() == seed, f"accept-s{seed} runs in a process bound to seed {seed}")
    gpu = gpu_compute_processes()
    label = f"S15 seed {seed}"
    refs, _ = reference_rows(c, seed)
    p0, steps, save_at = K.PARENT_UPDATE, ACCEPT["steps"], ACCEPT["save_at"]
    mode, pipeline = delivery_candidate(), ACCEPT["formal_pipeline"]
    digester = CC.Digester()
    paths, info, refusals = {}, {}, []

    def refuse(e, unit, state_file):
        K.PHASE["phase"] = f"accept:{unit} state refusals"
        arm, _ = parse(unit)
        other_arm = unit_name("FG" if arm == "L0" else "L0", seed)
        other_seed = unit_name(arm, next(s for s in SEEDS if s != seed))
        before = S.capture(e)
        cases = [(f"{unit} acceptance state read as {other_arm}", lambda: read_side(c, state_file, SCOPE, other_arm)),
                 (f"{unit} acceptance state read as {other_seed}", lambda: read_side(c, state_file, SCOPE, other_seed)),
                 (f"{unit} acceptance state read in formal scope", lambda: read_side(c, state_file, "formal", unit)),
                 (f"{unit} acceptance state loaded as a formal state", lambda: load(c, e, unit, state_file, "formal")),
                 (f"the seed-0 F_geo e120 state loaded into the {unit} engine",
                  lambda: load(c, e, unit, c["seed0"]["FG"]["FG_endpoint"]["path"], "formal")),
                 (f"the seed-0 L0 acceptance state loaded into the {unit} engine",
                  lambda: load(c, e, unit, c["seed0"]["L0"]["accept_u1"]["path"], SCOPE))]
        prior = out / f"{other_arm}-u{save_at:02d}.pth"
        if prior.is_file():
            cases.append((f"{other_arm} acceptance state loaded into the {unit} engine", lambda: load(c, e, unit, prior, SCOPE)))
        refusals.extend(H.refused(text, fn) for text, fn in cases)
        K.require(ck.exact_equal(S.capture(e), before), "a refused action changed the engine")

    for arm in ARMS:
        unit = unit_name(arm, seed)
        common = dict(arm=arm, seed=seed, unit=unit)
        paths[f"{unit}-reference"], info[f"{unit}-reference"] = _drive(
            c, out, counts, digester, refs, label, name=f"{unit}-reference", pipeline=ACCEPT["reference_path"][0],
            mode=ACCEPT["reference_path"][1], first=p0, last=p0 + steps, **common)
        info[f"{unit}-reference"]["unit"] = unit
        paths[f"{unit}-full"], info[f"{unit}-full"] = _drive(
            c, out, counts, digester, refs, label, name=f"{unit}-full", pipeline=pipeline, mode=mode, first=p0, last=p0 + steps,
            save_after=p0 + save_at, **common)
        paths[f"{unit}-resumed"], info[f"{unit}-resumed"] = _drive(
            c, out, counts, digester, refs, label, name=f"{unit}-resumed", pipeline=pipeline, mode=mode, first=p0 + save_at, last=p0 + steps,
            load_from=out / f"{unit}-u{save_at:02d}.pth", after=refuse, **common)
    K.PHASE["phase"] = "accept: comparison"
    files = _finish(out, digester, paths)
    units = [unit_name(a, seed) for a in ARMS]
    d_path = {u: CC.differences(_sci(paths[f"{u}-reference"]), _sci(paths[f"{u}-full"])) for u in units}
    d_replay = {u: CC.differences(_sci(paths[f"{u}-full"])[save_at:], _sci(paths[f"{u}-resumed"])) for u in units}
    cross = [a["input"] == b["input"] and a["gt_count"] == b["gt_count"] and a["result"]["lr"] == b["result"]["lr"]
             for a, b in zip(_sci(paths[f"{units[0]}-full"]), _sci(paths[f"{units[1]}-full"]))]
    start = {u: info[f"{u}-full"]["start_state"] for u in units}
    parts = [k for k in start[units[0]] if k != "arm" and start[units[0]][k] != start[units[1]][k]]
    outputs_differ = _sci(paths[f"{units[0]}-full"])[0]["result"]["output_sha256"] != _sci(paths[f"{units[1]}-full"])[0]["result"]["output_sha256"]
    want = c["workload"]["accept"]
    K.require(counts["actual_updates"] == want["updates_per_seed"] == 20 and counts["forwards"] == counts["forward_attempts"] == want["forwards_per_seed"],
              f"acceptance counts {dict(counts)}")
    K.require(all(info[f"{u}-full"]["saved"] is not None and info[f"{u}-full"]["window_delivery"]["mode"] == mode
                  and info[f"{u}-full"]["window_delivery"]["taken"] == steps and info[f"{u}-reference"]["window_delivery"]["mode"] == "host"
                  and info[f"{u}-reference"]["account"]["mode"] == "serial" and info[f"{u}-full"]["account"]["mode"] == pipeline for u in units),
              "acceptance path accounting")
    body = {"schema": "r30-acceptance-equivalence-v1", "contract_sha256": c["contract_sha256"], "seed": seed, "scope": SCOPE, "paths": info,
            "steps_files": files, "reference_ledger": c["seeds"][str(seed)]["S15_ledger"],
            "windows_equal_to_the_S15_ledger": {"rows_per_run": {n: len(r) for n, r in paths.items()}, "fields": list(K.SHARED_FIELDS),
                                                "note": "compared live, before and after every update; a difference would have stopped the stage"},
            "formal_path_against_the_reference_path": {"reference": ACCEPT["reference_path"], "formal": [pipeline, mode], "steps": steps,
                                                       "differences": d_path, "compared": CC.COMPARED, "excluded": CC.EXCLUDED},
            "restore_replay": {u: {"saved_after_update": p0 + save_at, "replayed_updates": [p0 + save_at + 1, p0 + steps],
                                   "differences": d_replay[u], "equal": not d_replay[u]} for u in units},
            "restore": "load() compares the complete restored state with the saved one (exact) before the replay",
            "both_arms_consume_the_same_windows": cross, "start_state_parts_different_between_the_arms": parts,
            "first_update_outputs_differ_between_the_arms": outputs_differ,
            "final_state_digests": {n: r[-1]["science"]["state"] for n, r in paths.items()},
            "loss_per_update": {n: [r["science"]["result"]["loss"] for r in rows] for n, rows in paths.items()},
            "refused": refusals, "gpu_at_start": gpu}
    K.write_new(out / "ACCEPT_EQUIVALENCE.json", body)
    K.require(all(not v for v in d_path.values()),
              f"the formal input path ({pipeline}, {mode}) differs from the serial host reference: { {u: v[:5] for u, v in d_path.items() if v} }")
    K.require(all(not v for v in d_replay.values()), f"save / restore replay differs: { {u: v[:5] for u, v in d_replay.items() if v} }")
    K.require(all(cross) and outputs_differ, f"arms of seed {seed}: same windows {cross}, first outputs differ {outputs_differ}")
    eng = lambda rows: [{"update": r["science"]["input"]["update"], **{k: r["engineering"].get(k) for k in ("cycle_s", "wait_s", "train_call_s", "train_s", "save_s")}}
                        for r in rows]
    K.write_new(out / "ACCEPT_TIMING.json", {"schema": "r30-acceptance-timing-v1", "contract_sha256": c["contract_sha256"],
                                             "per_update": {n: eng(r) for n, r in paths.items()},
                                             "note": "raw wall-clock segments of a few updates with capture digests and a full state capture on "
                                                     "every update; an engineering record, not a throughput or cost measurement"})
    return {"stage": f"accept-s{seed}", "seed": seed, "units": units, "workload": {"updates": want["updates_per_seed"], "forwards": want["forwards_per_seed"]},
            "restore_replay_equal": {u: True for u in units},
            "window_delivery": {u: {"mode": mode, "pipeline": pipeline, "equal_to_the_serial_host_reference": True} for u in units},
            "same_windows_as_the_S15_ledger": True, "both_arms_consume_the_same_windows": True, "formal_training_allowed": True,
            "equivalence": K.ref(out / "ACCEPT_EQUIVALENCE.json"), "timing": K.ref(out / "ACCEPT_TIMING.json"),
            "saved_states": {u: info[f"{u}-full"]["saved"] for u in units}, "refused": refusals,
            "peak_allocated_bytes": {n: info[n]["peak_cuda_allocated_bytes"] for n in info}, "gpu_at_start": gpu, "formal_state_usable": False}


def formal_mode(c, unit):
    """Window delivery of a formal run: the mode the closed acceptance of the unit's seed accepted for that unit."""
    _, seed = parse(unit)
    want = delivery_candidate()
    accepted = stage_closed(c, f"accept-s{seed}")["window_delivery"].get(unit)
    K.require(accepted == {"mode": want, "pipeline": ACCEPT["formal_pipeline"], "equal_to_the_serial_host_reference": True},
              f"the acceptance of seed {seed} did not accept the {want} delivery for unit {unit}: {accepted}")
    return ACCEPT["formal_pipeline"], want
