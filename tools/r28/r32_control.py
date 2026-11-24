"""R32-001: seed-1 / seed-2 replication of arm W, the budgeted separable resampling of R28-001 (TASK_R32_W_SEEDS_001).

A fixed replication of an existing arm.  No definition changes: arm W is src/sparse_rtdetr/research/r28_resample.py
with r28_warp.py, whose create / build_eval_model gain a seed argument (default 0 = the R28-001 path).  Two units are
trained here, each from the e060 parent of its own seed: W-s1, then W-s2.  The demand network is the frozen file of the
W round (trained on seed 0); it is not retrained and is shared by the three seeds.

  prepare      contract of the independent round: the task file (its bytes are bound before anything runs), the two
               units with their parents and reference ledgers (the R30-002 L0 ledger of the seed, with the F_geo and S15
               ledgers of that seed next to it), the seed-0 receipts of R28-001 and the results of R30-002 (read-only),
               the frozen demand network, the release difference against the R30-002 release, the reading of task
               section 5 and this round's own storage plan.
  r32_check    CPU: unit tests; references and identities; the release difference file by file and function by
               function; the three reference ledgers of each seed against each other on the 14 shared fields (18,240
               rows); the W seed-0 ledger; refusals; the reading arithmetic on the quoted seed-0 values.
  r32_wire     CUDA, seed 0, not a result (task section 3, check 1): W from the seed-0 e060 parent, 2 updates on the
               serial host reference path and 2 on the formal path, each against the stored accept-W receipts of
               R28-001 and rows 1-2 of its formal ledger.  4 updates.
  r32_accept   CUDA, one seed per process (check 3): 4 updates on the reference path, 4 on the formal path with a full
               state saved after update 2, then updates 3-4 replayed from that state.  10 updates per seed.
The endpoint reproduction (check 2), formal training, evaluation and the summary are in r32_loop.py; the reading
arithmetic is r32_reading.py; the runner is r32_run.py.
"""
import gc
import inspect
import json
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
import r30_control as R30
import r32_reading as RD

BASIS = "post_R30_user_forwarded_W_seed_replication"
VERSION = "R32-001-contract-v1"
REGISTRATION = ("R32-001 fixed replication of arm W (r28_budgeted_separable_resampling_v1) on seeds 1 and 2: two training "
                "units, no new method, no changed definition; TASK_R32_W_SEEDS_001")
TASK = ("TASK_R32_W_SEEDS_001.md", "fb2bb39802a44c30d219ead4be72d562a910aed93bf4aabb3000ac5745a3b157")
ARM = "W"
SEEDS = (1, 2)
ORDER = ("W-s1", "W-s2")                               # task section 4
SCHEMA = "r32-state-v1"
SCOPE = "r32-acceptance"
UNIT_RE = re.compile(r"W-s([0-2])")
STORAGE_STOP_BYTES = 30 * 1024 ** 3                    # task section 7: above this the executor stops and explains first

# ---- the W round of R28-001 (seed 0; read-only).  SHA-256 of TASK_R32_W_SEEDS_001 section 10 and of the returned receipts.
W_ROUND = K.LD_ROOT / "r28-budgeted-resampling-20261113-001"
W_COMMIT = "59e367de86cb55dbe1fe6606962d4f0bf65458e3"
W_CONTRACT = ("d8d4c8f844161038499ce9da4c1691d1fbfff51ec8f6bf47e44bcf034fac3c30", "a57929d49b88921a9dbd8ade540eb10cb3a2368369c39ba6fdecc370a24288c8")
W_IDENTITY = "3f5d40f55de39a59326e7a2563f7a7de2e55686a7eb83ad4fead060b4f4ff541"
DEMAND_SHA = "c9ffc7505e47adbcc45d70949c3d87371c6f3a7795f1aaaab886d98b15ee9e5a"
DEMAND_STATE_SHA = "f065f83a2c0421a6c91b7152aeab107050377503b2284a2e6aa91a42f882049b"
W_PINS = {"manifest": ("release-001/MANIFEST.json", "a65b4924df3fd1194f9a92f8b10e9da7fa3b7f9e5337682ad5ea48bb6d6f514e"),
          "demand": ("demand/demand-net.pth", DEMAND_SHA),
          "demand_closed": ("demand/CLOSED.json", "47a39fcaadcddfa5fe3206b0bee290786e1fee7e12546e679df3f772f377b59f"),
          "accept_closed": ("accept-W/CLOSED.json", "4c14a2a668a6ff281264283ba9c65226578db34ae9267e9baeda966612515403"),
          "formal_closed": ("formal-W/CLOSED.json", "66928ee850caa812961eda89b9d5ef8e9ec0c41ea1fd164b4ccfc14c8c396cb0"),
          "ledger": ("formal-W/UPDATES.jsonl", "e61682a6866cd14e722a91d4cbcd30522c50c29489b0889dc394de192954d6db"),
          "endpoint": ("formal-W/checkpoint-epoch-120.pth", "a88f2a1541fb911fda7bc7f9426892611ee2b5475aed6abf7df77fa5a6feaf37"),
          "endpoint_side": ("formal-W/checkpoint-epoch-120.pth.json", "f94e012e8cdd82bdf67e9805d69432ef5a2e53c64b964b889341724f1ba112bf"),
          "eval_closed": ("eval-W/CLOSED.json", "2a8ebf03a55bdba479179b7e26b34e84be04a830357b46e3344a33f567f2bd9e"),
          "predictions_600": ("eval-W/top600-predictions.json", "2797c49814aed26df3d030571ceec698513c8f5b9b600fe42ac3beceb0953486"),
          "predictions_300": ("eval-W/top300-predictions.json", "4bf20060f46e9c429040e38044f5b41ba7135b25d6aecafc59a975bb7b947c6d"),
          "metrics_600": ("eval-W/W-EMA-A-top600-metrics.json", "1319e287d8a59ea5be6f567cb9866a162519bb5e0804ff1e72de98af03c1f210"),
          "metrics_300": ("eval-W/W-EMA-A-top300-metrics.json", "4d43d80eb89f4c09ba5931aa38d390e33da271698ded0c6185037f84f2b040de"),
          "summary_closed": ("summary-W/CLOSED.json", "b3a7a1fc35a4080b5e142ed27399036ad94018e6ee0fe7318af41a69fe3f234a"),
          "summary_result": ("summary-W/RESULT.json", "75c4abfe3c6574544820cb1e235ca78ff77b9f20f71488f591461662b518edd0")}
W_ACCEPT_STATE = "accept-W/u1.pth"                     # optional: compared when the file is still the one the receipt names
W_WEIGHTS_SHA = "59becc3e0b68b6d925591c8dbb0300ce417eb6b0184aeb01e86cef2c821948aa"

# ---- the R30-002 round (L0 and F_geo, seeds 1 and 2; read-only)
R30_ROUND = K.LD_ROOT / "r30-fgeo-seeds-20261118-001"
R30_COMMIT = "fa0a381ba9e106a0381590225c48277f796b9c26"
R30_CONTRACT = ("d03d07eaee874f5c326d3c2c6b9dadb93431cb01addbb88fabb270b5ceb5f717", "2afdab274e04e5a8e852089cbce58334cba7cd7e4c0e27932b71d5c82139bcfe")
R30_PINS = {"manifest": ("release-001/MANIFEST.json", "c8f7cf6265c850bcab4578fe324a07eb0eaa777fe694d9be464fe555a0ba4809"),
            "summary_closed": ("summary-R30/CLOSED.json", "9249f6814ac8ec5e6dfbcc1b6b2d0d004c5807f866f5470b8684b7a4d6248970"),
            "summary_result": ("summary-R30/RESULT.json", "a967a5cb0942930aa9be5a024e939a23a765efbfc6b566ca1565451d658f1385")}
R30_LEDGERS = {"L0-s1": "1d4901bcb717b8b7c54072edd84f86fee4543cc760c3c813ef8a201ab126613e",
               "FG-s1": "9ecf7fd1720882b625e084ecbf6bf1619aa7caa971d973d5521f2ac70d0238ba",
               "L0-s2": "004f78b29dc145f91418d069938c36d83aaf406b08ceb109a1fe1c0762937fc6",
               "FG-s2": "329b8ed3ff0d80b7a64fce3c5958e39e05754b154543a81c46d01d1b3f837903"}
EVALUATION_PROTOCOL_SHA = "860285aa5af367b2bf2fef2221c15ac3914b33c57eb765e09f76b9b580004a82"

# ---- S15 top600 metric files of the three seeds (read-only; the seed-0 file is the one the W contract names)
S15_METRICS = {0: (None, "ab9c410a6e36d00b7a148e106e33cc17d3ebcc5fdb3e5a418f7bfbc4fd6e48ff"),
               1: (R30.S15_ROUND / "formal-001/seed1/evaluation/S15-seed1-EMA-A-top600-metrics.json",
                   "401d0679821c6aaf643af06fa3a2f5772b3db093621973d3b718b623275373fc"),
               2: (R30.S15_ROUND / "formal-001/seed2/evaluation/S15-seed2-EMA-A-top600-metrics.json",
                   "70dd828deb9a207dd06a5d3fea9ba3f8bbc8b0e9aefd8ec6ff3d4f48f9874849")}

SCIENCE_FILE = "src/sparse_rtdetr/research/r28_resample.py"
WARP_FILE = "src/sparse_rtdetr/research/r28_warp.py"
SCIENCE_EDITS = ["build_eval_model", "create"]         # the seed argument, default 0
ALLOWED_ADDED = ("tools/r28/r32_reading.py", "tools/r28/r32_control.py", "tools/r28/r32_loop.py", "tools/r28/r32_run.py",
                 "tools/r28/r32_chain.sh", "tests/test_r32_reading.py", "tests/test_r32_control.py")
TESTS = ("tests/test_r28_warp.py", "tests/test_r28_resample.py", "tests/test_r27_l0_control.py", "tests/test_r28_pipeline.py",
         "tests/test_r29_input.py", "tests/test_r30_control.py", "tests/test_r32_reading.py", "tests/test_r32_control.py")

WIRE = {"updates_per_path": 2, "paths": [["serial", "host"], ["overlap", RI.FORMAL]]}
ACCEPT = {"steps": 4, "save_at": 2, "reference_path": ["serial", "host"], "formal_pipeline": "overlap"}
_UNIT_ACCEPT = ACCEPT["steps"] * 2 + (ACCEPT["steps"] - ACCEPT["save_at"])
_FORMAL = K.ENDPOINT - K.PARENT_UPDATE
_EVAL = K.EVAL_WARMUP + K.EVAL_BATCHES
WORKLOAD = {
    "r32-check": {"updates": 0, "forwards": 0, "note": "CPU; synthetic unit tests in a subprocess"},
    "wire-s0": {"updates": len(WIRE["paths"]) * WIRE["updates_per_path"], "forwards": 2 * len(WIRE["paths"]) * WIRE["updates_per_path"],
                "task_limit_updates": 8, "note": "seed 0, not a result; states discarded"},
    "endpoint-s0": {"updates": 0, "forwards": _EVAL, "warmup": K.EVAL_WARMUP, "score_budget_calls": 2, "per_class_COCOeval_calls": 1,
                    "scored": "the stored R28-001 prediction lists, after the new lists were found identical to them",
                    "note": "seed 0, not a result: the stored W seed-0 values stay the quoted ones"},
    "accept": {"updates_per_seed": _UNIT_ACCEPT, "forwards_per_seed": 2 * _UNIT_ACCEPT, "updates": len(SEEDS) * _UNIT_ACCEPT,
               "task_limit_updates_per_seed": 20, "note": "states discarded; never continued as formal"},
    "formal": {"units": list(ORDER), "updates_per_unit": _FORMAL, "forwards_per_unit": 2 * _FORMAL, "updates": len(ORDER) * _FORMAL,
               "forwards": len(ORDER) * 2 * _FORMAL, "save_epochs": list(K.SAVE_EPOCHS), "task_save_epochs": list(K.FORMAL_SAVE_EPOCHS)},
    "evaluation": {"endpoint": "e120 EMA of each unit", "view": "A", "images": 548, "batch": K.EVAL_BATCH, "warmup": K.EVAL_WARMUP,
                   "forwards_per_unit": _EVAL, "forwards": len(ORDER) * _EVAL, "budgets": list(K.BUDGETS),
                   "score_budget_calls": 2 * len(ORDER), "per_class_COCOeval_calls": len(ORDER), "per_class_tensors_kept": True,
                   "intermediate_endpoints": "none", "checkpoint_selection": "none"},
    "cost": "none: no timing stage in this task; the cost reading stays the same-process ratios of task 009",
}
STATEMENTS = {
    "demand": "需求网络是 W 轮在 seed0 上训练并冻结的同一个文件（c9ffc750…9e5a），不重训，三个 seed 共用。",
    "scope": "固定复验：不改 W 的斜率界、ε、网格、需求规则、日程；不训练 K、F、P、C 的任何 seed；不重训或重评 L0、F_geo、S15、W-s0；"
             "不评中间端点；不测速。",
    "seed0": "seed0 的 W 读数引用 R28-001 的已存结果；本轮在 seed0 上只做两项不进结果的检查（接线等价、端点复现）。",
    "quoted": "L0、F_geo 三个 seed 的读数引用 R30-002；S15 引用其三个 seed 的 top600 指标文件。均不重跑。",
    "development": "548 图 development 集已被重复使用；本结果是该集合上的三 seed 成对读数，不提供独立泛化证明。",
    "bound": "+1.0 是论文里已在用的实际意义下界，不是显著性或功效检验；n=3 的样本标准差与 t 半宽只作描述，不据此声称两臂等价。",
}
INPUT_PIPELINE = {
    "pipeline": "overlap (r28_pipeline, accepted in task 009): workers 2, one prepared window, audit before consumption",
    "window_delivery": RI.RULE,
    "per_unit_condition": "a unit uses the overlapped pipeline with the device window delivery only if its own acceptance found every "
                          "update (outputs, gradients, observer readings, model / optimizer / EMA / RNG state) bit-for-bit equal to the "
                          "serial host reference path and its save / restore replay exact; R32_DELIVERY=host selects the serial host "
                          "reference path for the formal run (the acceptance must then have been run with the same setting)",
    "seed0_history": "W seed 0 (R28-001) was trained on the serial host path",
    "note": "an engineering choice allowed by task section 2; batch 8 x 2, BF16, optimizer, LR / EMA / BN clocks, activation "
            "checkpoint, the observer and the loader are unchanged",
}
LEDGER_COMPARISON = {
    "fields": list(K.SHARED_FIELDS),
    "live_reference": "the R30-002 L0 ledger of the unit's seed: every window is compared with its row before the update (the whole "
                      "input record) and after it (all listed fields); a difference stops the run before the model is updated with "
                      "that window",
    "equal_references": "r32-check finds the R30-002 L0 ledger, the R30-002 F_geo ledger and the S15 replication ledger of a seed equal "
                        "row by row on every listed field, so a comparison with one is a comparison with the three",
    "summary": "the finished W ledger of a seed is compared again with each of the three ledgers, row by row",
}
W_RECEIPT_FIELDS = ("input", "gt_count", "loss", "lr", "gradient_norm_before_clip", "ema_updates", "r28", "output_sha256", "dn_noise_sha256",
                    "gradient_sha256_before_clip")
W_LEDGER_FIELDS = ("input", "gt_count", "loss", "lr", "gradient_norm_before_clip", "microbatch_losses", "ema_updates", "r28")
W_CREATE_FIELDS = ("old_parameters_sha256", "lr_next", "parameters", "optimizer_parameters", "new_parameters", "EMA_updates", "engine_config",
                   "global_model_config", "identity_sha256", "parent_sha256", "L0_identity_sha256", "demand", "demand_network_parameters",
                   "wrapped", "arm", "r25_arm")
W_NOT_COMPARABLE = [
    "model / optimizer / EMA / RNG state after update 2 (R28-001 stored the full state after update 1 only; its acceptance compared "
    "the state after update 2 with its own replay and kept no digest)",
    "result.normalizers, result.coefficients, result.microsteps, result.optimizer_steps, result.finite (not written into the R28-001 "
    "acceptance receipt or ledger)",
    "result.microbatch_losses in the acceptance receipt (compared instead with rows 1-2 of the R28-001 formal ledger, which carry it)",
    "elapsed / update seconds (wall time)",
]
W_STATE_GONE = ("model / optimizer / EMA / RNG state after update 1 (accept-W/u1.pth of R28-001 is no longer on the server, or is not "
                "the file its receipt names)")
NOT_AUTHORISED = ["any seed of K, F, P, C", "retraining or re-evaluating L0, F_geo or S15 on any seed", "retraining W seed 0",
                  "slope bounds, epsilon, grid, demand rule or schedule changes", "demand network retraining", "extension beyond e120",
                  "intermediate endpoint evaluation", "timing / cost stages", "recall-by-magnification readings and the three diagnostics",
                  "any UAVDT evaluation", "test-dev", "the 1,602 held-out official-train images", "arms added after seeing results",
                  "merging any pull request", "cleaning any dev round or earlier round"]


def unit_name(seed):
    return f"W-s{seed}"


def parse(unit):
    m = UNIT_RE.fullmatch(unit or "")
    K.require(m is not None, f"unit must be W-s<seed>: {unit!r}")
    return ARM, int(m.group(1))


def process_unit():
    unit = os.environ.get("R32_UNIT")
    K.require(unit in ORDER, f"R32_UNIT must be one of {ORDER}, got {unit!r}")
    return unit


def process_seed(allowed):
    raw = os.environ.get("R32_SEED")
    K.require(raw in tuple(str(s) for s in allowed), f"R32_SEED must be one of {allowed}, got {raw!r}")
    return int(raw)


# ================================================================ read-only references
def _pins(root, pins, what):
    files = {k: K.ref(root / rel) for k, (rel, _) in pins.items()}
    bad = [k for k, (_, sha) in pins.items() if files[k]["sha256"] != sha]
    K.require(not bad, f"files of {what} differ from their pins: {bad}")
    return files


def _closed(ref, contract_sha, what, **want):
    body = K.read_json(ref["path"])
    K.require(body.get("status") == "PASS" and body.get("contract_sha256") == contract_sha
              and all(body.get(k) == v for k, v in want.items()), f"{what} is not PASS under its round's contract")
    return body


def _ledger(path, keys=("input", "loss") + K.TOP_FIELDS):
    rows = []
    with Path(path).open() as f:
        for line in f:
            row = json.loads(line)
            rows.append(row if keys is None else {k: v for k, v in row.items() if k in keys})
    K.require(len(rows) == _FORMAL, f"ledger rows: {len(rows)} ({path})")
    for i, r in enumerate(rows):
        K.require(r["input"]["update"] == K.PARENT_UPDATE + 1 + i and r["ema_updates"] == r["input"]["update"], f"ledger update sequence ({path})")
        K.shared(r)
    return rows


def w0_references():
    """Seed 0 (read-only): the closed W round of R28-001.  Contract (file SHA and inner digest), release manifest,
    demand network, acceptance receipts, formal ledger, e120 state with its side-car, evaluation with its prediction
    and metric files, summary.  Everything pinned; the state after update 1 is referenced when it is still on disk."""
    from sparse_rtdetr.research import r28_resample as R
    cref = K.ref(W_ROUND / "CONTRACT.json")
    K.require(cref["sha256"] == W_CONTRACT[0], "W round contract file")
    c1 = R30._digest_checked(cref["path"], W_CONTRACT[1])
    files = _pins(W_ROUND, W_PINS, "the W round")
    sha = c1["contract_sha256"]
    manifest = K.read_json(files["manifest"]["path"])
    accept = _closed(files["accept_closed"], sha, "accept-W", arm="W")
    formal = _closed(files["formal_closed"], sha, "formal-W", arm="W")
    ev = _closed(files["eval_closed"], sha, "eval-W", arm="W")
    summary = _closed(files["summary_closed"], sha, "summary-W")
    demand = _closed(files["demand_closed"], sha, "demand")
    result = K.read_json(files["summary_result"]["path"])
    side = K.read_json(files["endpoint_side"]["path"])
    K.require(c1["source_commit"] == manifest["commit"] == W_COMMIT and c1["source_manifest"] == files["manifest"] and not manifest.get("dirty", False),
              "W round release")
    K.require(c1["identity_sha256"]["W"] == R.IDENTITY_SHA256["W"] == W_IDENTITY and c1["identity"]["W"] == R.IDENTITY["W"]
              and c1["parent"]["sha256"] == K.PARENT_SHA, "arm W identity / seed-0 parent")
    binding = {"kind": "network", "file": files["demand"], "seed": None, "state_sha256": DEMAND_STATE_SHA}
    K.require(demand["weights"] == files["demand"] and accept["demand"] == ev["demand"] == side["demand"] == binding, "frozen demand network")
    K.require(len(accept["first_updates"]) == 2 and all(set(W_RECEIPT_FIELDS) <= set(r) for r in accept["first_updates"])
              and accept["create_checks"]["identity_sha256"] == W_IDENTITY and accept["saved_u1"]["global_update"] == K.PARENT_UPDATE + 1,
              "accept-W receipts")
    K.require(formal["resumed"] is False and formal["ledger_rows"] == formal["updates"] == _FORMAL and formal["plan"]["parent"] == c1["parent"],
              "formal-W receipt")
    K.require(side["status"] == "COMPLETE" and side["file"] == files["endpoint"] == ev["endpoint"] and side["global_update"] == K.ENDPOINT
              and side["scope"] == "formal" and side["arm"] == "W" and side["contract_sha256"] == sha
              and side["audit"]["ledger"]["sha256"] == files["ledger"]["sha256"] and ev["formal_stage"] == "formal-W"
              and ev["weights_sha256"] == W_WEIGHTS_SHA and ev["global_update"] == K.ENDPOINT and ev["kind"] == "EMA" and ev["view"] == "A",
              "W seed-0 endpoint / evaluation receipt")
    K.require(ev["predictions"] == {b: files[f"predictions_{b}"] for b in RD.BUDGET_KEYS}
              and ev["metrics"] == {b: files[f"metrics_{b}"] for b in RD.BUDGET_KEYS}
              and ev["counts"] == {"actual_updates": 0, "forwards": _EVAL, "forward_attempts": 0, "score_budget_calls": 2, "per_class_COCOeval_calls": 1},
              "W seed-0 prediction / metric files")
    K.require(summary["result"] == files["summary_result"], "summary-W result file")
    K.require(result["values"]["W"] == ev["values"] and result["contract_sha256"] == sha and result["status"] == "PASS"
              and ev["GT_counts"] == c1["reference"]["GT_counts"]
              and [(r["category_id"], r["large_GT"], r["W"]) for r in result["per_class_APL_top600"]]
              == [(r["category_id"], r["GT"], r["APL"]) for r in ev["per_class_APL_top600"]], "summary-W against eval-W")
    state = W_ROUND / W_ACCEPT_STATE
    u1 = K.ref(state) if state.is_file() else None
    if u1 is not None and u1 != accept["saved_u1"]["file"]:
        u1 = None
    s15 = K.ref(K.pinned(c1["S15_metrics"]["600"]))
    K.require(s15["sha256"] == S15_METRICS[0][1], "S15 seed-0 top600 metric file")
    return c1, {
        "round": str(W_ROUND), "contract": cref, "contract_sha256": sha, "commit": W_COMMIT, "identity_sha256": W_IDENTITY,
        "parent": c1["parent"], "R27_updates": c1["R27_updates"], **files, "accept_u1": u1,
        "demand_binding": binding, "weights_sha256": ev["weights_sha256"], "values": ev["values"], "GT_counts": ev["GT_counts"],
        "per_class_APL_top600": ev["per_class_APL_top600"], "S15_metrics_600": s15,
        "training": {"input_pipeline": "serial", "window_delivery": "host", "loop_wall_seconds": formal["loop_wall_seconds"],
                     "peak_cuda_bytes": formal["peak_cuda_bytes"], "resumed": formal["resumed"]},
        "per_class_note": "R28-001 kept the per-class APL rows only (no COCOeval tensors): per-class APS / APM of W seed 0 exist "
                          "only as a by-product of the endpoint-reproduction check of this round, which is not a result"}


def r30_references():
    """Seeds 0-2 of L0 and F_geo (read-only): the closed R30-002 round.  Contract, release manifest, summary result (the
    quoted values and per-class rows) and the four formal ledgers, all pinned."""
    cref = K.ref(R30_ROUND / "CONTRACT.json")
    K.require(cref["sha256"] == R30_CONTRACT[0], "R30-002 contract file")
    c30 = R30._digest_checked(cref["path"], R30_CONTRACT[1])
    files = _pins(R30_ROUND, R30_PINS, "the R30-002 round")
    manifest = K.read_json(files["manifest"]["path"])
    closed = _closed(files["summary_closed"], c30["contract_sha256"], "summary-R30")
    result = K.read_json(files["summary_result"]["path"])
    K.require(c30["source_commit"] == manifest["commit"] == R30_COMMIT and c30["source_manifest"] == files["manifest"] and not manifest.get("dirty", False)
              and c30["authorization_basis"] == R30.BASIS and closed["result"] == files["summary_result"], "R30-002 release / summary")
    ledgers = {}
    for unit, sha in R30_LEDGERS.items():
        ledgers[unit] = result["ledgers"]["units"][unit]["ledger"]
        K.require(ledgers[unit]["sha256"] == sha and K.pinned(ledgers[unit]).parent.parent == R30_ROUND.resolve()
                  and result["ledgers"]["units"][unit]["rows"] == _FORMAL, f"R30-002 ledger of {unit}")
    values = result["values"]
    K.require(sorted(values) == list(RD.SEED_KEYS) and all(sorted(v) == ["FG", "L0"] for v in values.values())
              and all(sorted(values[s][a][b]) == sorted(RD.METRICS) for s in values for a in values[s] for b in RD.BUDGET_KEYS), "R30-002 values")
    apl = {s: result["per_class_top600"][s]["APL"] for s in RD.SEED_KEYS}
    K.require(all([r["category_id"] for r in apl[s]] == list(range(1, 11)) for s in apl), "R30-002 per-class APL rows")
    return c30, {
        "round": str(R30_ROUND), "contract": cref, "contract_sha256": c30["contract_sha256"], "commit": R30_COMMIT, **files,
        "ledgers": ledgers, "values": values, "per_class_APL_top600": apl, "seeds": c30["seeds"],
        "per_class_APS_APM_top600": {s: {k: result["per_class_top600"][s][k] for k in ("APS", "APM")} for s in ("1", "2")},
        "labels": {"label": result["label"], "scale_tradeoff_label": result["scale_tradeoff_label"]}}


def s15_references(w0):
    """The S15 top600 metric files of the three seeds (quoted for S15 - W on the seven top600 cells)."""
    from infer_seed import bridge_values
    refs = {"0": w0["S15_metrics_600"]}
    for seed in SEEDS:
        path, sha = S15_METRICS[seed]
        refs[str(seed)] = K.ref(path)
        K.require(refs[str(seed)]["sha256"] == sha, f"S15 seed {seed} top600 metric file")
    values = {s: bridge_values(K.read_json(r["path"])) for s, r in refs.items()}
    K.require(all(sorted(v) == sorted(RD.METRICS) for v in values.values()), "S15 metric cells")
    return {"files": refs, "values_top600": values}


def base_release(c30):
    manifest = K.read_json(K.pinned(c30["source_manifest"]))
    K.require(manifest["commit"] == R30_COMMIT and not manifest.get("dirty", False), "base release commit")
    return manifest


def source_delta(old_files, new_files, w_files=None):
    changed = sorted(k for k in old_files if k in new_files and old_files[k] != new_files[k])
    added = sorted(k for k in new_files if k not in old_files)
    removed = sorted(k for k in old_files if k not in new_files)
    K.require(not removed and changed == [SCIENCE_FILE] and set(added) <= set(ALLOWED_ADDED),
              f"release differs from the R30-002 release outside the seed argument and the R32 tools: changed {changed}, added {added}, removed {removed}")
    out = {"against": f"release-001 of {R30_ROUND.name} (commit {R30_COMMIT})", "changed": changed, "added": added, "removed": removed,
           "unchanged_files": len(old_files) - len(changed),
           "note": "one science file gains a seed argument with default 0 (create, build_eval_model); every other file of src/, vendor/, "
                   "configs/ and every existing tool (loader, engine, checkpoint, input pipeline, window delivery, loops, evaluation, "
                   "scorer) is byte-identical; the R32 tools are added files"}
    if w_files is not None:
        K.require(w_files[WARP_FILE] == old_files[WARP_FILE] == new_files[WARP_FILE] and w_files[SCIENCE_FILE] == old_files[SCIENCE_FILE],
                  "arm W science files differ between the R28-001 release and the R30-002 release")
        out["arm_W_files_against_R28_001"] = {
            "release": f"release-001 of {W_ROUND.name} (commit {W_COMMIT})", WARP_FILE: "byte-identical in the three releases",
            SCIENCE_FILE: "byte-identical in the R28-001 and R30-002 releases; this release adds the seed argument",
            "files_changed_between_R28_001_and_R30_002": sorted(k for k in w_files if k in old_files and w_files[k] != old_files[k]),
            "note": "the L0 construction and the tools changed between the two earlier releases (seed argument of R30-002, later arms); "
                    "that the present code is the R28-001 path at seed 0 is what the stage wire-s0 measures"}
    return out


def edits(old_root, new_root):
    d = GC.function_delta((Path(old_root) / SCIENCE_FILE).read_text(), (Path(new_root) / SCIENCE_FILE).read_text())
    K.require(d["changed"] == SCIENCE_EDITS and not d["added"] and not d["removed"],
              f"{SCIENCE_FILE}: function edits differ from the registered list: {d['changed']} {d['added']} {d['removed']}")
    return {SCIENCE_FILE: {"changed": d["changed"], "added": [], "removed": [], "unchanged_functions": len(d["unchanged"])}}


def storage_plan(state_bytes, w0, dev_rounds_bytes):
    """This round's own storage budget (task section 7: the executor's own figure is the one that counts)."""
    formal_states = len(ORDER) * len(K.SAVE_EPOCHS)
    predictions = sum(w0[f"predictions_{b}"]["bytes"] for b in RD.BUDGET_KEYS)
    plan = {"state_bytes": state_bytes, "formal_states": formal_states, "formal_states_bytes": formal_states * state_bytes,
            "acceptance_states": len(ORDER), "acceptance_states_bytes": len(ORDER) * state_bytes,
            "new_predictions_bytes_estimate": len(ORDER) * predictions, "seed0_reproduction_predictions_bytes_estimate": predictions,
            "ledgers_observations_timing_bytes_estimate": len(ORDER) * (72 + 24) * 1024 ** 2,
            "metrics_tensors_logs_release_bytes_estimate": 64 * 1024 ** 2,
            "dev_rounds_bytes": dev_rounds_bytes, "failure_margin_bytes": 4 * 1024 ** 3, "stop_line_bytes": STORAGE_STOP_BYTES,
            "disk_free_bytes_at_prepare": shutil.disk_usage(K.ROUND).free,
            "save_cadence": "the e060 start state and every 5 epochs (13 full states per unit), as R28-001 and R30-002",
            "gpu_time_reference": "task estimate 6-7 h per unit on the overlapped device path, about 11 h on the serial host path; a "
                                  "scheduling estimate, not a cap"}
    plan["formal_round_bytes_estimate"] = (plan["formal_states_bytes"] + plan["acceptance_states_bytes"] + plan["new_predictions_bytes_estimate"]
                                           + plan["seed0_reproduction_predictions_bytes_estimate"] + plan["ledgers_observations_timing_bytes_estimate"]
                                           + plan["metrics_tensors_logs_release_bytes_estimate"])
    plan["total_bytes_estimate"] = plan["formal_round_bytes_estimate"] + dev_rounds_bytes
    return plan


def _tree_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file() and not p.is_symlink())


# ================================================================ contract
def prepare(_args=None):
    from sparse_rtdetr.research import r28_resample as R
    from sparse_rtdetr.research import r28_warp as W
    from sparse_rtdetr.research import scale_baseline as S
    round_ = K.ROUND.resolve()
    K.require(K.ROOT.resolve().parent.parent == round_ and K.ROOT.name == "source", "prepare must run from a frozen release")
    K.require("R28_ROUND" in os.environ and (round_.parent == K.LD_ROOT.resolve() or (K.DEV and round_.parent.parent == K.LD_ROOT.resolve())),
              "the R32 round is an independent directory named by R28_ROUND (a dev round lives inside it)")
    manifest = K.read_json(K.ROOT.parent / "MANIFEST.json")
    K.require(manifest["files"] == K.B.source_manifest(K.ROOT), "release source differs from MANIFEST")
    K.require(not manifest.get("dirty", False) or K.DEV, "a release assembled from a dirty work tree is only accepted in a dev round")
    task = K.ref(K.ROUND / TASK[0])
    K.require(task["sha256"] == TASK[1], "task file SHA")
    c1, w0 = w0_references()
    c30, r30 = r30_references()
    cs, s15_round, seeds = R30.seed_references()
    s15 = s15_references(w0)
    base_manifest = base_release(c30)
    delta = source_delta(base_manifest["files"], manifest["files"], K.read_json(w0["manifest"]["path"])["files"])
    function_edits = edits(c30["source_root"], K.ROOT)
    for k in ("data", "data_receipt", "development_ids", "evaluation_policy", "environment", "python", "gpu_uuid", "recipe", "definition",
              "shared_fields"):
        K.require(c1[k] == c30[k], f"the W contract and the R30-002 contract name different {k}")
    K.require(c30["seed0"]["parent"] == c1["parent"] and c30["seed0"]["R27_updates"] == c1["R27_updates"] and c30["seeds"] == seeds
              and c30["demand"]["weights"] == w0["demand"] and c30["demand"]["binding"] == w0["demand_binding"]
              and c30["reference"]["GT_counts"] == w0["GT_counts"], "the R30-002 contract against the W round (parents, demand network, GT counts)")
    K.require(S.PARENT_SHAS == {0: c1["parent"]["sha256"], **{s: seeds[str(s)]["parent"]["sha256"] for s in SEEDS}}, "parents of the three seeds")
    units = {}
    for u in ORDER:
        _, seed = parse(u)
        units[u] = {"arm": ARM, "seed": seed, "parent": seeds[str(seed)]["parent"], "identity_sha256": R.IDENTITY_SHA256[ARM],
                    "reference_ledger": r30["ledgers"][f"L0-s{seed}"],
                    "equal_ledgers": {"R30-002 FG": r30["ledgers"][f"FG-s{seed}"], "S15 replication": seeds[str(seed)]["S15_ledger"]},
                    "demand": w0["demand"], "formal_directory": f"formal-{u}", "evaluation_directory": f"eval-{u}"}
    main_round = round_.parent if K.DEV else round_
    dev_bytes = sum(_tree_bytes(d) for d in sorted(main_round.iterdir()) if d.is_dir() and d.name.startswith("dev-round")) if main_round.is_dir() else 0
    plan = storage_plan(w0["endpoint"]["bytes"], w0, dev_bytes)
    K.require(plan["total_bytes_estimate"] <= STORAGE_STOP_BYTES,
              f"planned storage {plan['total_bytes_estimate']} bytes exceeds the 30 GB line of task section 7: stop and explain first")
    K.require(plan["disk_free_bytes_at_prepare"] > 4 * (plan["formal_round_bytes_estimate"] + plan["failure_margin_bytes"]),
              "less free space than four times the planned storage of this round")
    values0 = {"W": w0["values"], "L0": r30["values"]["0"]["L0"], "FG": r30["values"]["0"]["FG"]}
    c = {"version": VERSION, "authorization_basis": BASIS, "registration": REGISTRATION, "task": task,
         "units": units, "training_order": list(ORDER), "arm": ARM, "seeds": seeds, **s15_round,
         "seed0": w0, "R30": r30, "S15": s15,
         "demand": {"weights": w0["demand"], "binding": w0["demand_binding"], "trained_on_seed": 0, "retrained": False, "shared_by_seeds": [0, 1, 2]},
         "parent_update": K.PARENT_UPDATE, "endpoint_update": K.ENDPOINT,
         "data": c1["data"], "data_receipt": c1["data_receipt"], "development_ids": c1["development_ids"],
         "evaluation_policy": c1["evaluation_policy"], "environment": c1["environment"], "python": c1["python"], "gpu_uuid": c1["gpu_uuid"],
         "recipe": c1["recipe"], "identity": {ARM: R.IDENTITY[ARM]}, "identity_sha256": {ARM: R.IDENTITY_SHA256[ARM]},
         "definition": W.DEFINITION,
         "reference": {"GT_counts": w0["GT_counts"], "values": {"W-s0": w0["values"], **{f"{a}-s{s}": r30["values"][s][a] for s in RD.SEED_KEYS for a in ("L0", "FG")}},
                       "S15_top600": s15["values_top600"],
                       "seed0_W_minus_L0_pp": RD.difference_pp(values0["W"], values0["L0"]),
                       "seed0_W_minus_F_geo_pp": RD.difference_pp(values0["W"], values0["FG"])},
         "reading": RD.READING, "statements": STATEMENTS, "workload": WORKLOAD, "input_pipeline": INPUT_PIPELINE,
         "ledger_comparison": LEDGER_COMPARISON, "shared_fields": list(K.SHARED_FIELDS),
         "acceptance": {**ACCEPT, "wire": WIRE, "scope": SCOPE},
         "seed0_checks": {"wiring": {"receipt_fields": list(W_RECEIPT_FIELDS), "ledger_fields": list(W_LEDGER_FIELDS), "create_fields": list(W_CREATE_FIELDS),
                                     "state_after_update_1": w0["accept_u1"] is not None,
                                     "not_comparable": W_NOT_COMPARABLE + ([] if w0["accept_u1"] is not None else [W_STATE_GONE])},
                          "endpoint": {"predictions": "identical item by item to the stored top600 / top300 lists",
                                       "scoring": "the scoring entry of this release on the stored lists: 14 cells and GT counts bitwise, "
                                                  "per-class rows within 1e-15", "per_class_tolerance": 1e-15,
                                       "on_difference": "stop and hand back to the controller with the largest difference"}},
         "resource_plan": plan,
         "base_release": {"round": str(R30_ROUND), "contract_sha256": c30["contract_sha256"], "commit": R30_COMMIT, "manifest": c30["source_manifest"],
                          "source_root": c30["source_root"]},
         "source_delta": delta, "function_edits": function_edits,
         "source_root": str(K.ROOT.resolve()), "source_manifest": K.ref(K.ROOT.parent / "MANIFEST.json"),
         "source_commit": manifest["commit"], "source_dirty": bool(manifest.get("dirty", False)),
         "not_authorised": NOT_AUTHORISED, "outputs": str(K.ROUND), "formal_started": False}
    K.require(RD.bound_cells(c["reference"]["seed0_W_minus_L0_pp"])["met"] is True and RD.lead(c["reference"]["seed0_W_minus_F_geo_pp"])["reading"] == "neither"
              and c["reference"]["seed0_W_minus_L0_pp"]["600"]["APM"] <= RD.MEDIUM_COST, "the seed-0 readings are not the ones task section 5 starts from")
    K.require(len(c["development_ids"]) == 548 and cs["development_ids"] == c["development_ids"], "development ids")
    c["contract_sha256"] = K.sha_bytes(K.canonical(c))
    K.write_new(K.ROUND / "CONTRACT.json", c)
    print(json.dumps({"contract_sha256": c["contract_sha256"], "identity_sha256": c["identity_sha256"], "authorization_basis": BASIS,
                      "task_sha256": task["sha256"], "units": {u: v["parent"]["sha256"] for u, v in units.items()},
                      "source_delta": {k: delta[k] for k in ("changed", "added")}, "state_after_update_1_available": w0["accept_u1"] is not None,
                      "storage_formal_round_bytes_estimate": plan["formal_round_bytes_estimate"], "storage_dev_rounds_bytes": dev_bytes,
                      "storage_total_bytes_estimate": plan["total_bytes_estimate"]}))
    return c


def load_contract(path=None):
    from sparse_rtdetr.research import r28_resample as R
    from sparse_rtdetr.research import r28_warp as W
    from sparse_rtdetr.research import scale_baseline as S
    c = R30._digest_checked(path or K.ROUND / "CONTRACT.json")
    K.require(c.get("authorization_basis") == BASIS and c.get("version") == VERSION, "not the R32-001 contract")
    K.require(Path(c["source_root"]).resolve() == K.ROOT.resolve(), "not running from the bound release")
    K.require(Path(c["outputs"]).resolve() == K.ROUND.resolve(), "contract bound to another round")
    manifest = K.read_json(K.pinned(c["source_manifest"]))
    K.require(manifest["commit"] == c["source_commit"], "MANIFEST commit differs from the contract")
    K.B.release_intact(K.ROOT, manifest["files"])
    K.require(c["identity_sha256"] == {ARM: R.IDENTITY_SHA256[ARM]} == {ARM: W_IDENTITY} and c["definition"] == W.DEFINITION
              and list(c["units"]) == list(ORDER) == c["training_order"] and c["reading"] == RD.READING and c["task"]["sha256"] == TASK[1],
              "R32 behaviour identity / units / reading / task bytes differ from the contract")
    for u, v in c["units"].items():
        K.require((v["arm"], v["seed"]) == parse(u) and v["parent"]["sha256"] == S.PARENT_SHAS[v["seed"]]
                  and v["parent"] == c["seeds"][str(v["seed"])]["parent"] and v["reference_ledger"]["sha256"] == R30_LEDGERS[f"L0-s{v['seed']}"],
                  f"unit {u} parent / reference ledger")
    K.require(c["seed0"]["parent"]["sha256"] == S.PARENT_SHAS[0] and c["demand"]["weights"]["sha256"] == DEMAND_SHA, "seed-0 parent / demand file")
    return c


def stage_closed(c, name):
    path = K.ROUND / name / "CLOSED.json"
    K.require(path.is_file(), f"stage not closed: {path}")
    closed = K.read_json(path)
    K.require(closed.get("status") == "PASS" and closed.get("contract_sha256") == c["contract_sha256"], f"stage {name} is not PASS under this contract")
    return closed


def demand_source(c):
    """The formal demand source of arm W on every seed: the frozen network file of the W round."""
    K.pinned(c["demand"]["weights"])
    return {"kind": "network", "file": c["demand"]["weights"]}


def parent_of(c, seed):
    return c["seed0"]["parent"] if seed == 0 else c["seeds"][str(seed)]["parent"]


bound_seed = R30.bound_seed
gpu_compute_processes = R30.gpu_compute_processes


def is_w_engine(e):
    return getattr(e, "r28_arm", None) == ARM and not hasattr(e, "l0_arm") and not hasattr(e, "r27_arm") and getattr(e, "r28_mapping", None) is None


def create(c, seed, device="cuda:0"):
    """Arm W engine from the e060 parent of one seed.  The process must be bound to that seed (bind_seed)."""
    from sparse_rtdetr.research import r28_resample as R
    K.require(seed in (0,) + SEEDS and bound_seed() == seed, f"seed {seed} / process seed {bound_seed()}")
    parent = K.pinned(parent_of(c, seed))
    e, checks = R.create(str(parent), device, ARM, demand_source(c), seed=seed)
    K.require(is_w_engine(e) and e.r28_check is False and e.r28_ctx.binding() == c["demand"]["binding"] == checks["demand"]
              and R.is_wrapped(e.model) and R.is_wrapped(e.ema.module) and checks["identity_sha256"] == c["identity_sha256"][ARM],
              "not the arm W engine of R28-001 (demand source / identity)")
    K.require(e.config.seed == seed and e.r25_parent == parent_of(c, seed)["sha256"] and checks["parent_sha256"] == e.r25_parent
              and e.completed_updates == K.PARENT_UPDATE, "engine seed / parent / clock")
    return e, checks


# ================================================================ states (schema r32-state-v1)
def binding(c, e, unit, scope, audit=None, role=None):
    from formal_data import cursor
    _, seed = parse(unit)
    K.require(unit in c["units"] and is_w_engine(e) and e.config.seed == seed and bound_seed() == seed
              and e.r25_parent == c["units"][unit]["parent"]["sha256"], f"engine is not the engine of unit {unit}")
    cur = cursor(e.completed_updates)
    K.require(cur["seed"] == seed, "cursor seed")
    return {"schema": SCHEMA, "scope": scope, "unit": unit, "arm": ARM, "seed": seed, "contract_sha256": c["contract_sha256"],
            "identity_sha256": c["identity_sha256"][ARM], "parent_sha256": c["units"][unit]["parent"]["sha256"],
            "global_update": e.completed_updates, "cursor": cur, "config": e.config.binding(), "audit": audit, "role": role,
            "activation_checkpoint": e.s15_activation_checkpoint, "torch": str(K.torch_mod().__version__), "demand": e.r28_ctx.binding()}


def save(c, e, unit, path, scope, audit=None, role=None):
    from sparse_rtdetr.research import scale_baseline as S
    from sparse_rtdetr.protocol_r18.engine import assert_healthy
    torch = K.torch_mod()
    path = Path(path)
    K.require(e.ema.updates == e.completed_updates, "EMA clock at save")
    K.require(not getattr(e, "r28_check", False), "a check engine cannot write an R32 state")
    assert_healthy(e.model, e.optimizer, e.ema)
    state = S.capture(e)
    b = binding(c, e, unit, scope, audit, role)
    state["r32_binding_sha256"] = K.sha_bytes(K.canonical(b))
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
    _, seed = parse(unit)
    K.require(unit in c["units"], f"R32 state refused: {unit!r} is not a unit of this round")
    for k, want in (("schema", SCHEMA), ("unit", unit), ("arm", ARM), ("seed", seed), ("scope", scope),
                    ("contract_sha256", c["contract_sha256"]), ("parent_sha256", c["units"][unit]["parent"]["sha256"]),
                    ("identity_sha256", c["identity_sha256"][ARM])):
        K.require(side.get(k) == want, f"R32 state refused: {k}={side.get(k)!r}, expected {want!r} ({path})")
    K.require(side.get("status") == "COMPLETE", f"R32 state side-car is not COMPLETE ({path})")
    return path, side


def read_state(c, path, scope, unit):
    torch = K.torch_mod()
    path, side = read_side(c, path, scope, unit)
    K.require(side["file"] == K.ref(path), "R32 state file identity")
    s = torch.load(path, map_location="cpu", weights_only=True)
    expect = {k: v for k, v in side.items() if k not in ("status", "file")}
    K.require(s.pop("r32_binding_sha256") == K.sha_bytes(K.canonical(expect)), "R32 binding digest")
    K.require(s["engine"]["completed_updates"] == side["global_update"] == s["ema"]["updates"], "R32 state clock")
    return side, s


def load(c, e, unit, path, scope):
    """Exact restore into a fresh engine of the same unit."""
    from sparse_rtdetr.research import scale_baseline as S
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    _, seed = parse(unit)
    K.require(is_w_engine(e) and e.config.seed == seed and bound_seed() == seed, f"engine is not the engine of unit {unit}")
    side, s = read_state(c, path, scope, unit)
    K.require(side["config"] == e.config.binding() and side["activation_checkpoint"] == e.s15_activation_checkpoint, "R32 state config / memory mode")
    K.require(side["demand"] == e.r28_ctx.binding(), "R32 state was written with another demand source")
    K.require(s["optimizer_names"] == [g["param_names"] for g in e.optimizer.param_groups], "R32 optimizer names")
    e.model.load_state_dict(s["model"], strict=True)
    e.optimizer.load_state_dict(s["optimizer"])
    e.ema.load_state_dict(s["ema"])
    modes = dict(e.model.named_modules())
    K.require(set(modes) == set(s["model_modes"]), "R32 mode keys")
    for n, m in s["model_modes"].items():
        modes[n].training = m
    e.load_state_dict(s["engine"])
    ck.restore_rng(s["rng"], e.device)
    K.require(ck.exact_equal(S.capture(e), s), "R32 full state did not restore exactly")
    return side


# ================================================================ reference ledgers
def reference_label(seed):
    return "R27 formal-P" if seed == 0 else f"R30-002 L0-s{seed}"


def reference_rows(c, seed):
    """Rows of the live reference ledger of a seed by update: seed 0 -> the pinned R27 formal-P / formal-C ledgers (the
    rule of R28-001); seeds 1 / 2 -> the pinned R30-002 L0 ledger of that seed.  Returns (rows, first reading)."""
    import l0_loop as LL
    if seed == 0:
        refs, first = LL.reference_rows({"R27_updates": c["seed0"]["R27_updates"]})
        return refs, {"update": first["update"], "reference": reference_label(0), "loss": first["P"]}
    K.require(seed in SEEDS, f"seed {seed}")
    rows = _ledger(K.pinned(c["units"][unit_name(seed)]["reference_ledger"]))
    return {r["input"]["update"]: r for r in rows}, {"update": K.PARENT_UPDATE + 1, "reference": reference_label(seed), "loss": rows[0]["loss"]}


reference_check = R30.reference_check


def equal_ledgers(c, seed):
    """The R30-002 L0 ledger, the R30-002 F_geo ledger and the S15 replication ledger of one seed, row by row on the
    14 shared fields and on the whole input record."""
    import l0_loop as LL
    unit = c["units"][unit_name(seed)]
    base = _ledger(K.pinned(unit["reference_ledger"]))
    out = {}
    for name, ref in unit["equal_ledgers"].items():
        other = _ledger(K.pinned(ref))
        for a, b in zip(base, other):
            K.compare_shared(a, b)
            bad = LL.input_differences(a["input"], b["input"])
            K.require(not bad, f"seed {seed}: the {name} ledger differs from the R30-002 L0 ledger at update {a['input']['update']}: {bad}")
        out[name] = {"file": ref, "rows": len(other), "rows_equal_on_the_shared_fields": len(other), "rows_different": 0}
    return base, {"reference": unit["reference_ledger"], "rows": len(base), "fields": list(K.SHARED_FIELDS), "equal_to": out}


# ================================================================ r32-check (CPU)
def _own(c, out, name):
    out = Path(out)
    K.require(out.resolve() == (K.ROUND / name).resolve(), f"this action writes <round>/{name}")
    return out


def reading_self_check(c):
    """The arithmetic of task section 5 on the quoted seed-0 values standing in for every seed (not a result)."""
    v0 = {"W": c["seed0"]["values"], "L0": c["R30"]["values"]["0"]["L0"], "FG": c["R30"]["values"]["0"]["FG"]}
    apl0 = {"W": c["seed0"]["per_class_APL_top600"], **{a: [{"category_id": r["category_id"], "GT": r["GT"], "APL": r[a]}
                                                            for r in c["R30"]["per_class_APL_top600"]["0"]] for a in ("L0", "FG")}}
    s15 = c["S15"]["values_top600"]
    out = RD.reading({s: v0 for s in RD.SEED_KEYS}, {s: s15["0"] for s in RD.SEED_KEYS}, {s: apl0 for s in RD.SEED_KEYS})
    K.require(out["reading_A"]["label"] == RD.LABELS_A[0] and out["reading_B"]["label"] == RD.LABEL_B_NONE and out["reading_C"]["label"] == RD.LABELS_C[0]
              and out["W_minus_L0_pp"]["0"] == c["reference"]["seed0_W_minus_L0_pp"] and out["W_minus_F_geo_pp"]["0"] == c["reference"]["seed0_W_minus_F_geo_pp"]
              and out["seven_cells_W_minus_L0"]["0"]["failed_cells"] == ["top600_APM"] and out["W_unique_advantage"]["status"] == "not_established"
              and len(out["APL"]["0"]["split"]["W_minus_L0"]["rare"]["classes"]) == 5, "reading arithmetic self-check")
    flipped = RD.reading({"0": v0, "1": v0, "2": {"W": v0["L0"], "L0": v0["W"], "FG": v0["FG"]}}, {s: s15["0"] for s in RD.SEED_KEYS},
                         {"0": apl0, "1": apl0, "2": {"W": apl0["L0"], "L0": apl0["W"], "FG": apl0["FG"]}})
    K.require(flipped["reading_A"]["label"] == RD.LABELS_A[1] and flipped["reading_C"]["label"] == RD.LABELS_C[1], "reading arithmetic self-check (flipped seed)")
    return {"seed0_W_minus_L0_pp": out["W_minus_L0_pp"]["0"], "seed0_W_minus_F_geo_pp": out["W_minus_F_geo_pp"]["0"],
            "seed0_S15_minus_W_top600_pp": out["S15_minus_W_top600_pp"]["0"], "seed0_reading_A": out["reading_A"]["per_seed"]["0"]["met"],
            "seed0_reading_B": out["reading_B"]["readings"]["0"], "seed0_seven_cells_failed": out["seven_cells_W_minus_L0"]["0"]["failed_cells"],
            "seed0_APL_split_W_minus_L0": {k: out["APL"]["0"]["split"]["W_minus_L0"][k] for k in ("rare", "other", "macro_difference_pp")},
            "note": "arithmetic exercised on the quoted seed-0 values standing in for every seed, and on a copy with W and L0 exchanged; "
                    "not a result"}


def w0_geometry(c, n=None):
    rows = _ledger(K.pinned(c["seed0"]["ledger"]), keys=None)
    K.require(all(set(W_LEDGER_FIELDS) <= set(r) for r in rows[:2]), "W seed-0 ledger row keys")
    return rows, RD.geometry((r["input"]["epoch"], r["r28"]) for r in (rows if n is None else rows[:n]))


def r32_check(c, out, counts):
    import l0_checks_state as H
    import r28_eval  # noqa: F401  (import check of the shared inference)
    import r32_loop as RL
    from sparse_rtdetr.research import r28_resample as R
    from sparse_rtdetr.research import scale_baseline as S
    torch = K.torch_mod()
    out = _own(c, out, "r32-check")
    K.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_available(), "the R32 check hides CUDA")
    t0 = time.perf_counter()
    res = {"stage": "r32-check", "registration": c["registration"], "authorization_basis": c["authorization_basis"],
           "workload": c["workload"]["r32-check"]}

    K.PHASE["phase"] = "r32-check:pytest"
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *[str(K.ROOT / t) for t in TESTS]],
                       capture_output=True, text=True, cwd=str(K.ROOT),
                       env={k: v for k, v in {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}.items() if not k.startswith(("R28_", "R30_", "R32_"))})
    res["pytest"] = {"returncode": p.returncode, "tail": p.stdout.strip().splitlines()[-1:], "tests": list(TESTS),
                     "note": "synthetic unit tests in a subprocess; their forwards are not in this stage's counts"}
    K.require(p.returncode == 0, "synthetic pytest failed:\n" + p.stdout[-6000:])

    K.PHASE["phase"] = "r32-check:references"
    c1, w0 = w0_references()
    c30, r30 = r30_references()
    cs, s15_round, seeds = R30.seed_references()
    s15 = s15_references(w0)
    manifest = K.read_json(K.pinned(c["source_manifest"]))
    base_manifest = base_release(c30)
    K.require(c["seed0"] == w0 and c["R30"] == r30 and c["S15"] == s15 and c["seeds"] == seeds and all(c[k] == v for k, v in s15_round.items())
              and K.pinned(c["task"]) == (K.ROUND / TASK[0]).resolve() and c["task"]["sha256"] == TASK[1], "references changed since prepare")
    delta = source_delta(base_manifest["files"], manifest["files"], K.read_json(w0["manifest"]["path"])["files"])
    K.B.release_intact(Path(c30["source_root"]), base_manifest["files"])
    function_edits = edits(c30["source_root"], K.ROOT)
    K.require(delta == c["source_delta"] and function_edits == c["function_edits"], "source delta changed since prepare")
    K.require(S.PARENT_SHAS == {0: c["seed0"]["parent"]["sha256"], **{s: c["seeds"][str(s)]["parent"]["sha256"] for s in SEEDS}}
              and len(set(S.PARENT_SHAS.values())) == 3, "parents: scale_baseline.PARENT_SHAS against the contract")
    for k in ("recipe", "data", "evaluation_policy", "environment", "python", "gpu_uuid", "development_ids", "definition"):
        K.require(c[k] == c1[k] == c30[k], f"contract key {k} differs from the W / R30-002 contracts")
    K.require(c["identity"][ARM] == R.IDENTITY[ARM] == c1["identity"][ARM] and c["workload"] == WORKLOAD and c["input_pipeline"] == INPUT_PIPELINE
              and c["ledger_comparison"] == LEDGER_COMPARISON and c["acceptance"] == {**ACCEPT, "wire": WIRE, "scope": SCOPE}
              and c["statements"] == STATEMENTS and c["not_authorised"] == NOT_AUTHORISED, "frozen blocks of the contract")
    K.require(c["evaluation_policy"]["sha256"] == EVALUATION_PROTOCOL_SHA, "the evaluation protocol of task section 10 is not the one the contract names")
    K.pinned(c["evaluation_policy"])
    K.require(demand_source(c)["file"]["sha256"] == DEMAND_SHA, "demand source of arm W")
    res["references"] = {"W_contract_sha256": c1["contract_sha256"], "R30_contract_sha256": c30["contract_sha256"],
                         "S15_contract_sha256": cs["contract_sha256"], "base_release_commit": R30_COMMIT, "W_release_commit": W_COMMIT,
                         "parents": {str(k): v for k, v in S.PARENT_SHAS.items()}, "demand_sha256": DEMAND_SHA,
                         "demand_state_sha256": DEMAND_STATE_SHA, "evaluation_protocol_sha256": EVALUATION_PROTOCOL_SHA,
                         "task_sha256": c["task"]["sha256"], "source_delta": delta, "function_edits": function_edits,
                         "state_after_update_1_of_accept_W": w0["accept_u1"]}

    K.PHASE["phase"] = "r32-check:seed-0 defaults"
    sig = {f"r28_resample.{fn}": str(inspect.signature(getattr(R, fn))) for fn in ("create", "build_eval_model")}
    K.require(all(inspect.signature(getattr(R, fn)).parameters["seed"].default == 0
                  and inspect.signature(getattr(R, fn)).parameters["seed"].kind is inspect.Parameter.KEYWORD_ONLY for fn in ("create", "build_eval_model")),
              "the seed argument of create / build_eval_model must be keyword-only with default 0")
    K.require(R.IDENTITY_SHA256[ARM] == W_IDENTITY and R.IDENTITY[ARM]["parent_sha256"] == S.PARENT_SHAS[0], "arm W identity digest")
    res["seed_argument"] = {"signatures": sig, "default": 0, "identity_sha256": R.IDENTITY_SHA256[ARM],
                            "note": "the identity body of arm W is unchanged (it names the seed-0 parent as R28-001 wrote it); the parent of a "
                                    "unit is bound by the contract, the engine check and every state side-car.  With the default the module "
                                    "is the R28-001 code path; the CUDA stage wire-s0 compares that path with the stored receipts"}

    K.PHASE["phase"] = "r32-check:reference ledgers"
    refs0, _ = reference_rows(c, 0)
    ledgers, bases = {}, {}
    for seed in SEEDS:
        base, info = equal_ledgers(c, seed)
        bases[seed] = refs = {r["input"]["update"]: r for r in base}
        K.require(sorted(refs) == sorted(refs0), "update set of the reference ledger")
        schedule = [k for k in ("epoch", "window", "augmentation_strong") if any(refs[u]["input"][k] != refs0[u]["input"][k] for u in refs)]
        clock = [k for k in ("lr", "ema_updates") if any(refs[u][k] != refs0[u][k] or type(refs[u][k]) is not type(refs0[u][k]) for u in refs)]
        same_ids = sum(refs[u]["input"]["image_ids_sha256"] == refs0[u]["input"]["image_ids_sha256"] for u in refs)
        K.require(not schedule and not clock and same_ids == 0, f"seed {seed} ledger: schedule {schedule}, clocks {clock}, windows equal to seed 0 {same_ids}")
        ledgers[str(seed)] = {**info, "first_update": min(refs), "last_update": max(refs),
                              "schedule_columns_equal_to_seed0": ["epoch", "window", "augmentation_strong"],
                              "clock_columns_equal_to_seed0": ["lr", "ema_updates"], "windows_with_the_seed0_image_set": same_ids}
    K.require(sum(bases[1][u]["input"]["image_ids_sha256"] == bases[2][u]["input"]["image_ids_sha256"] for u in bases[1]) == 0,
              "the seed-1 and seed-2 ledgers share a window")
    res["reference_ledgers"] = ledgers
    rows0, geo0 = w0_geometry(c)
    for r in rows0:
        K.compare_shared(r, refs0[r["input"]["update"]])
    res["W_seed0_ledger"] = {"file": c["seed0"]["ledger"], "rows": len(rows0), "rows_equal_to_R27_formal_P_on_the_shared_fields": len(rows0),
                             "training_geometry": geo0}

    K.PHASE["phase"] = "r32-check:refusals"
    refusals = [
        H.refused("a unit name outside the two units", lambda: parse("FG-s1")),
        H.refused("seed 3", lambda: parse("W-s3")),
        H.refused("an engine of seed 1 in a process bound to seed 0", lambda: create(c, 1, "cpu")),
        H.refused("formal directory of a unit outside the order", lambda: RL.guard(c, K.ROUND / "formal-W-s0", None, "W-s0")),
        H.refused("formal run before the checks closed", lambda: RL.guard(c, K.ROUND / "formal-W-s1", None, "W-s1")),
        H.refused("the W seed-0 e120 state of R28-001 read as an R32 state", lambda: read_side(c, c["seed0"]["endpoint"]["path"], "formal", "W-s1")),
        H.refused("a state read for seed 0, which this round does not train", lambda: read_side(c, c["seed0"]["endpoint"]["path"], "formal", "W-s0")),
        H.refused("a three-seed label from two seeds", lambda: RD.label_a([True, True])),
        H.refused("a three-seed lead label without seed 0 reading neither", lambda: RD.label_b(["W_leads", "neither", "neither"])),
    ]
    res["refused"] = refusals
    res["reading_arithmetic_self_check"] = reading_self_check(c)
    res["storage"] = {"plan": c["resource_plan"], "round_bytes_now": _tree_bytes(K.ROUND), "disk_free_bytes_now": shutil.disk_usage(K.ROUND).free}
    res["seconds"] = time.perf_counter() - t0
    K.require(counts["actual_updates"] == 0 and counts["forwards"] == 0, "the CPU check ran a model")
    return res


# ================================================================ short CUDA runs shared by wire-s0 and accept-sN
def _drive(c, out, counts, digester, refs, label, *, name, seed, pipeline, mode, first, last, unit=None, load_from=None, save_after=None,
           after=None):
    import l0_loop as LL
    import r28_loop as LP
    import r28_pipeline as PL
    from s15_formal_train import _result_body
    from sparse_rtdetr.research import scale_baseline as S
    torch = K.torch_mod()
    K.PHASE["phase"] = f"{name}: construct"
    e, checks = create(c, seed)
    K.register_counts(e, counts)
    obs = LP.install_observer(e)
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
                science = {"input": inp, "gt_count": int(sum(len(v["labels"]) for v in t)), "result": _result_body(res),
                           "observer": obs.update_row(t)}
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
    K.require(e.r28_ctx.binding() == c["demand"]["binding"], f"{name}: the demand source changed during the run")
    if after is not None:
        after(e, unit, state_file)
    info = {"arm": ARM, "seed": seed, "unit": unit, "pipeline": pipeline, "first_update": first + 1, "last_update": last, "updates": len(rows),
            "account": account, "window_delivery": delivery.account(), "resumed_from": None if loaded is None else loaded["file"],
            "saved": saved, "start_state": start, "wall_seconds_stream": closed - opened,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "memory": memory.summary(),
            "create_checks": {k: checks[k] for k in W_CREATE_FIELDS}}
    delivery.close()
    del e, obs, delivery
    gc.collect()
    torch.cuda.empty_cache()
    return rows, info


_finish, _sci = R30._finish, R30._sci


def _w_seed0_differences(c, rows, info):
    """Arm W at seed 0 against everything R28-001 stored for its first two updates."""
    torch = K.torch_mod()
    s0 = c["seed0"]
    accept = K.read_json(K.pinned(s0["accept_closed"]))
    theirs = accept["first_updates"]
    mine = []
    for full in rows:
        r, res = full["science"], full["science"]["result"]
        # the stored rows carry the EMA clock of the engine after the update (result.ema_updates is the per-update count)
        mine.append({"input": r["input"], "gt_count": r["gt_count"], "loss": res["loss"], "lr": res["lr"],
                     "gradient_norm_before_clip": res["gradient_norm_before_clip"], "ema_updates": full["engineering"]["ema_clock"],
                     "r28": r["observer"], "output_sha256": res["output_sha256"], "dn_noise_sha256": res["dn_noise_sha256"],
                     "gradient_sha256_before_clip": res["gradient_sha256_before_clip"], "microbatch_losses": res["microbatch_losses"]})
    out = {"acceptance_receipt": CC.differences([{k: t[k] for k in W_RECEIPT_FIELDS} for t in theirs],
                                                [{k: m[k] for k in W_RECEIPT_FIELDS} for m in mine])}
    ledger = []
    with K.pinned(s0["ledger"]).open() as f:
        for _ in range(len(mine)):
            ledger.append(json.loads(f.readline()))
    out["formal_ledger_rows"] = CC.differences([{k: t[k] for k in W_LEDGER_FIELDS} for t in ledger],
                                               [{k: m[k] for k in W_LEDGER_FIELDS} for m in mine])
    out["create_checks"] = CC.differences({k: accept["create_checks"][k] for k in W_CREATE_FIELDS},
                                          {k: info["create_checks"][k] for k in W_CREATE_FIELDS})
    if s0["accept_u1"] is not None:
        stored = torch.load(K.pinned(s0["accept_u1"]), map_location="cpu", weights_only=True)
        stored.pop("r28_binding_sha256")
        K.require(stored["engine"]["completed_updates"] == K.PARENT_UPDATE + 1, "the stored R28-001 acceptance state is not the state after update 1")
        out["state_after_update_1"] = CC.differences(CC.state_digests(stored), rows[0]["science"]["state"])
        del stored
    return out


def r32_wire(c, out, counts):
    """Seed 0, not a result (task section 3, check 1): the present code against the stored receipts of R28-001."""
    out = _own(c, out, "wire-s0")
    stage_closed(c, "r32-check")
    seed = process_seed((0,))
    K.require(bound_seed() == 0, "wire-s0 runs in a process bound to seed 0")
    gpu = gpu_compute_processes()
    refs, _ = reference_rows(c, 0)
    p0, n = K.PARENT_UPDATE, WIRE["updates_per_path"]
    digester = CC.Digester()
    paths, info = {}, {}
    for pipeline, mode in WIRE["paths"]:
        name = f"W-s0-{pipeline}-{mode}"
        paths[name], info[name] = _drive(c, out, counts, digester, refs, reference_label(0), name=name, seed=seed, pipeline=pipeline, mode=mode,
                                         first=p0, last=p0 + n)
    K.PHASE["phase"] = "wire-s0: comparison"
    files = _finish(out, digester, paths)
    stored = {name: _w_seed0_differences(c, rows, info[name]) for name, rows in paths.items()}
    names = [f"W-s0-{p}-{m}" for p, m in WIRE["paths"]]
    between = CC.differences(_sci(paths[names[0]]), _sci(paths[names[1]]))
    want = c["workload"]["wire-s0"]
    K.require(counts["actual_updates"] == want["updates"] == 4 and counts["forwards"] == counts["forward_attempts"] == want["forwards"],
              f"wire-s0 counts {dict(counts)}")
    not_comparable = c["seed0_checks"]["wiring"]["not_comparable"]
    equal = {"W_equal_to_the_R28_001_receipts": {k: not any(v.values()) for k, v in stored.items()},
             "formal_path_equal_to_the_reference_path": not between}
    body = {"schema": "r32-seed0-wiring-v1", "contract_sha256": c["contract_sha256"], "seed": 0, "paths": info, "steps_files": files,
            "W": {"receipts": {k: c["seed0"][k] for k in ("accept_closed", "accept_u1", "ledger")}, "steps": n, "differences": stored,
                  "compared": {"acceptance_receipt": list(W_RECEIPT_FIELDS), "formal_ledger_rows": list(W_LEDGER_FIELDS),
                               "create_checks": list(W_CREATE_FIELDS),
                               "state_after_update_1": "every part of the captured state (model, optimizer, EMA, engine clocks, RNG, BN buffers)"
                                                       if c["seed0"]["accept_u1"] is not None else None},
                  "not_comparable": not_comparable},
            "reference_path_against_formal_path": {"paths": WIRE["paths"], "differences": between, "compared": CC.COMPARED, "excluded": CC.EXCLUDED,
                                                   "note": "all science fields, the observer rows and the state digests after every update"},
            "equal": equal, "gpu_at_start": gpu, "result_status": "not a result (seed 0, 2 updates per path)"}
    K.write_new(out / "WIRE_EQUIVALENCE.json", body)
    K.require(all(equal["W_equal_to_the_R28_001_receipts"].values()), f"W at seed 0 differs from the stored R28-001 receipts: {stored}")
    K.require(equal["formal_path_equal_to_the_reference_path"], f"paths differ at seed 0: {between[:5]}")
    return {"stage": "wire-s0", "seed": 0, "workload": want, **equal, "W_not_comparable": not_comparable,
            "state_after_update_1_compared": c["seed0"]["accept_u1"] is not None,
            "equivalence": K.ref(out / "WIRE_EQUIVALENCE.json"), "updates": want["updates"], "complete_detector_forwards": want["forwards"],
            "peak_allocated_bytes": {n_: info[n_]["peak_cuda_allocated_bytes"] for n_ in info}, "gpu_at_start": gpu, "formal_state_usable": False}


def delivery_candidate():
    forced = os.environ.get("R32_DELIVERY")
    K.require(forced in (None, "host"), "R32_DELIVERY can only select the serial host reference path")
    return ("serial", "host") if forced else (ACCEPT["formal_pipeline"], RI.FORMAL)


def r32_accept(c, out, counts):
    """One seed (task section 3, check 3): reference path, formal path with a save after update 2, replay of 3-4."""
    import l0_checks_state as H
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    from sparse_rtdetr.research import scale_baseline as S
    seed = process_seed(SEEDS)
    unit = unit_name(seed)
    out = _own(c, out, f"accept-s{seed}")
    stage_closed(c, "r32-check")
    stage_closed(c, "wire-s0")
    stage_closed(c, "endpoint-s0")
    K.require(bound_seed() == seed, f"accept-s{seed} runs in a process bound to seed {seed}")
    gpu = gpu_compute_processes()
    label = reference_label(seed)
    refs, _ = reference_rows(c, seed)
    p0, steps, save_at = K.PARENT_UPDATE, ACCEPT["steps"], ACCEPT["save_at"]
    pipeline, mode = delivery_candidate()
    digester = CC.Digester()
    paths, info, refusals = {}, {}, []

    def refuse(e, unit_, state_file):
        K.PHASE["phase"] = f"accept:{unit_} state refusals"
        other = unit_name(next(s for s in SEEDS if s != seed))
        before = S.capture(e)
        cases = [(f"{unit_} acceptance state read as {other}", lambda: read_side(c, state_file, SCOPE, other)),
                 (f"{unit_} acceptance state read in formal scope", lambda: read_side(c, state_file, "formal", unit_)),
                 (f"{unit_} acceptance state loaded as a formal state", lambda: load(c, e, unit_, state_file, "formal")),
                 (f"the W seed-0 e120 state loaded into the {unit_} engine", lambda: load(c, e, unit_, c["seed0"]["endpoint"]["path"], "formal"))]
        prior = K.ROUND / f"accept-s{next(s for s in SEEDS if s != seed)}" / f"{other}-u{save_at:02d}.pth"
        if prior.is_file():
            cases.append((f"the {other} acceptance state loaded into the {unit_} engine", lambda: load(c, e, unit_, prior, SCOPE)))
        refusals.extend(H.refused(text, fn) for text, fn in cases)
        K.require(ck.exact_equal(S.capture(e), before), "a refused action changed the engine")

    common = dict(seed=seed, unit=unit)
    paths["reference"], info["reference"] = _drive(c, out, counts, digester, refs, label, name="reference", pipeline=ACCEPT["reference_path"][0],
                                                   mode=ACCEPT["reference_path"][1], first=p0, last=p0 + steps, **common)
    paths["full"], info["full"] = _drive(c, out, counts, digester, refs, label, name="full", pipeline=pipeline, mode=mode, first=p0,
                                         last=p0 + steps, save_after=p0 + save_at, **common)
    paths["resumed"], info["resumed"] = _drive(c, out, counts, digester, refs, label, name="resumed", pipeline=pipeline, mode=mode,
                                               first=p0 + save_at, last=p0 + steps, load_from=out / f"{unit}-u{save_at:02d}.pth", after=refuse, **common)
    K.PHASE["phase"] = "accept: comparison"
    files = _finish(out, digester, paths)
    d_path = CC.differences(_sci(paths["reference"]), _sci(paths["full"]))
    d_replay = CC.differences(_sci(paths["full"])[save_at:], _sci(paths["resumed"]))
    want = c["workload"]["accept"]
    K.require(counts["actual_updates"] == want["updates_per_seed"] == 10 and counts["forwards"] == counts["forward_attempts"] == want["forwards_per_seed"],
              f"acceptance counts {dict(counts)}")
    K.require(info["full"]["saved"] is not None and info["full"]["window_delivery"]["mode"] == mode and info["full"]["window_delivery"]["taken"] == steps
              and info["reference"]["window_delivery"]["mode"] == "host" and info["reference"]["account"]["mode"] == "serial"
              and info["full"]["account"]["mode"] == pipeline, "acceptance path accounting")
    observer = [r["observer"] for r in _sci(paths["full"])]
    K.require(all(r["identity_images"] == 0 for r in observer), "the production demand produced an exactly uniform plan")
    body = {"schema": "r32-acceptance-equivalence-v1", "contract_sha256": c["contract_sha256"], "seed": seed, "unit": unit, "scope": SCOPE,
            "paths": info, "steps_files": files, "reference_ledger": c["units"][unit]["reference_ledger"],
            "windows_equal_to_the_reference_ledger": {"rows_per_run": {n: len(r) for n, r in paths.items()}, "fields": list(K.SHARED_FIELDS),
                                                      "note": "compared live, before and after every update; a difference would have stopped the stage"},
            "formal_path_against_the_reference_path": {"reference": ACCEPT["reference_path"], "formal": [pipeline, mode], "steps": steps,
                                                       "differences": d_path, "compared": CC.COMPARED, "excluded": CC.EXCLUDED},
            "restore_replay": {"saved_after_update": p0 + save_at, "replayed_updates": [p0 + save_at + 1, p0 + steps], "differences": d_replay,
                               "equal": not d_replay},
            "restore": "load() compares the complete restored state with the saved one (exact) before the replay",
            "final_state_digests": {n: r[-1]["science"]["state"] for n, r in paths.items()},
            "loss_per_update": {n: [r["science"]["result"]["loss"] for r in rows] for n, rows in paths.items()},
            "observer_per_update": observer, "refused": refusals, "gpu_at_start": gpu}
    K.write_new(out / "ACCEPT_EQUIVALENCE.json", body)
    K.require(not d_path, f"the formal input path ({pipeline}, {mode}) differs from the serial host reference: {d_path[:5]}")
    K.require(not d_replay, f"save / restore replay differs: {d_replay[:5]}")
    eng = lambda rows: [{"update": r["science"]["input"]["update"], **{k: r["engineering"].get(k) for k in ("cycle_s", "wait_s", "train_call_s", "train_s", "save_s")}}
                        for r in rows]
    K.write_new(out / "ACCEPT_TIMING.json", {"schema": "r32-acceptance-timing-v1", "contract_sha256": c["contract_sha256"],
                                             "per_update": {n: eng(r) for n, r in paths.items()},
                                             "note": "raw wall-clock segments of a few updates with capture digests and a full state capture on "
                                                     "every update; an engineering record, not a throughput or cost measurement"})
    return {"stage": f"accept-s{seed}", "seed": seed, "unit": unit, "workload": {"updates": want["updates_per_seed"], "forwards": want["forwards_per_seed"]},
            "restore_replay_equal": True,
            "window_delivery": {"mode": mode, "pipeline": pipeline, "equal_to_the_serial_host_reference": True},
            "same_windows_as_the_reference_ledger": True, "formal_training_allowed": True,
            "equivalence": K.ref(out / "ACCEPT_EQUIVALENCE.json"), "timing": K.ref(out / "ACCEPT_TIMING.json"),
            "saved_state": info["full"]["saved"], "refused": refusals,
            "peak_allocated_bytes": {n: info[n]["peak_cuda_allocated_bytes"] for n in info}, "gpu_at_start": gpu, "formal_state_usable": False}


def formal_mode(c, unit):
    """Input path of a formal run: the one the closed acceptance of the unit's seed accepted."""
    _, seed = parse(unit)
    pipeline, mode = delivery_candidate()
    accepted = stage_closed(c, f"accept-s{seed}")["window_delivery"]
    K.require(accepted == {"mode": mode, "pipeline": pipeline, "equal_to_the_serial_host_reference": True},
              f"the acceptance of seed {seed} did not accept the ({pipeline}, {mode}) path for unit {unit}: {accepted}")
    return pipeline, mode
