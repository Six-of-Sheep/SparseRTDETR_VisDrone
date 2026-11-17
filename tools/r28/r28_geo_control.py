"""R28-F_geo strong mapping control (TASK_R28_STRONG_F_CONTROL_009).  An exploratory control designed after the W, K and
F results and after the geometry readings of task 006 were seen.  Nothing of the closed W round, of the K round, of
the F round or of the 006 round is rerun or written; their decisions, readings and the 006 status
calibration_unmatched (h_star = null) stay.

Arm FG = arm F with the attraction kernel FWHM fixed at 19.03125 nodes (src/sparse_rtdetr/research/r28_fovea_geo.py).
The width is the reading of the stored 006 table with the largest pooled small magnification; it is not a matched
calibration, so arm FS is not built here and its matched guard is untouched.

  prepare       contract of the independent round: authorization basis, task / review files, read-only references into
                the W, K, F and 006 rounds, the selection block, the release difference against the 006 release.
  geo_check     CPU, no model forward: unit tests; identities and refusals; the release difference file by file and
                function by function (pipeline, loader, engine, save / restore and comparison code unchanged); the 006
                calibration file, its 66 readings and its epoch-61 cache re-read at the fixed width (bitwise the stored
                reading; nothing is re-selected); the official rule at the fixed width; the pinned dev receipts of the
                input pipeline (80-update equivalence and timing), which are quoted and not repeated.
  geo_confirm   CUDA for the demand network only: the first 32 windows of epoch 118 (512 occurrences), validity of the
                fixed width, the W and FG magnifications per raw-area class.  Zero detector forwards.  The width is
                never changed by this stage; an invalid geometry closes the stage and ends the chain.
  geo_wire      CUDA wiring of code that otherwise runs only after training: one evaluation batch of arm FG and a
                shortened five-arm staged timing; nothing is scored.  Listed apart from the planned workload.
  geo_accept    CUDA acceptance of the actual width from the e060 parent: 4 serial updates, 4 overlapped updates with
                a full state saved after update 2 while a prepared window exists, updates 3-4 replayed from that
                state.  Bitwise comparison step by step with the comparison code of r28_cal_control (unchanged).
                Cross-arm state refusals (F / FS / FG / W / K / L0).  Not a throughput measurement.
  summarize     summary-FG, RECORD_ONLY: W - FG, FG - F, FG - L0 and the costs; the geometry over the schedule.
  ledger_means  CPU, read-only: arithmetic means over updates of the per-update geometric mean magnification stored in
                the formal W, K and F ledgers (the machine source of the reports' 1.267 / 1.080 / 1.053).
Formal training, evaluation and timing are r28_loop.formal, r28_eval.evaluate and r28_cost.cost with R28_ARM=FG.
"""
import ast
import gc
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import r28_common as K
import r28_followup as FU
import r28_fovea_control as FC
import r28_cal_control as CC

REGISTRATION = "R28-F_geo exploratory fixed-width strong mapping control after the W, K and F results and the 006 geometry readings"
ARM = "FG"
CAL_ROUND = K.LD_ROOT / "r28-calibrated-fovea-control-20261115-001"
TASK = ("TASK_R28_STRONG_F_CONTROL_009.md", "2e4f29f9cace208f339d2779a1179361ba7e188aa4420927c4006bcf22ff1de2")
REVIEW = ("REVIEW_R28_CALIBRATION_009.md", "5e4d6af89335e41c5226131e85bceadbaca3510caffdd9f90daa04d08da551d1")
PIN_CAL = {  # TASK_R28_STRONG_F_CONTROL_009 section 1 and the 006 round's own receipts
    "commit": "f376dc973d8333252d4cf50c9ed37d4fd8b9d244",
    "manifest": "07247d2d2a19fca7daa14273617f178051f0c9e4a94ccd1effcb2000d8540cd3",
    "contract_file": "dbb5eb675e613326fed3078e4a1c3ff80d8782d3c0f8985b87dcba110c3fbddd",
    "contract": "80ac542b24bf52b0d7ec55b0cf3b81c6971f32a1df502241f286e233a67b24fd",
    "calibration": "8dcfcd5cf2cbd7ef96a54b1832d9eabd6923a5b7da860a9e634784ed0fa05c59",
    "calibrate_closed": "31b37f1ff99e253de6c3083cca791c122227c0cbb2eb8602f24dc3bc7b351483",
    "cal_check_closed": "032efc56e506249d171daf749c7dc8f095064982d574e8b272be455207f7f8cd",
    "cache": "2a79360eab5e1d3e8278566a170baed775242ec31392ba22358c12a9777cfe37",
    "dev_contract_file": "33cbf28e3a65809d24bb112689f15b091d62c127cf2e6cad9ecc650963e0cdc1",
    "dev_manifest": "07247d2d2a19fca7daa14273617f178051f0c9e4a94ccd1effcb2000d8540cd3",
    "dev_pipeline_closed": "50a49517c0f37ec6cca8e12acdad4d1456897978fa04fd61b0678de63c4bfa96",
    "dev_equivalence": "a5a411b4f195062a7f64fdde862b4575f51c7ad3a4997ec1754579e8349e54f0",
    "dev_timing": "5ffed79bd8fa914f9a3de6b7d705624b6ec41bc58787346fe61626ac240eea29",
    "dev_FS_state": "6323c91337a84b58b89062247c13c645bacb5777f7ddd5c5cef601288e896483",
}
CAL_FILES = {"contract_file": "CONTRACT.json", "manifest": "release-001/MANIFEST.json", "calibration": "calibrate/CALIBRATION.json",
             "calibrate_closed": "calibrate/CLOSED.json", "cal_check_closed": "cal-check/CLOSED.json", "cache": "calibrate/cache-epoch061.pt",
             "dev_contract_file": "dev-round-2/CONTRACT.json", "dev_manifest": "dev-round-2/release-001/MANIFEST.json",
             "dev_pipeline_closed": "dev-round-2/pipeline/CLOSED.json", "dev_equivalence": "dev-round-2/pipeline/PIPELINE_EQUIVALENCE.json",
             "dev_timing": "dev-round-2/pipeline/PIPELINE_TIMING.json", "dev_FS_state": "dev-round-2/pipeline/overlap-u16.pth"}
OWN_STAGES = ("geo-check", "geo-confirm", "wire-FG", "accept-FG", "formal-FG", "eval-FG", "cost-FG", "summary-FG", "ledger-means")
ALLOWED_CHANGED = ("tools/r28/r28_common.py", "tools/r28/r28_loop.py", "tools/r28/r28_eval.py", "tools/r28/r28_cost.py",
                   "tools/r28/r28_run.py", "tests/test_r28_cal_control.py")
ALLOWED_ADDED = ("src/sparse_rtdetr/research/r28_fovea_geo.py", "tests/test_r28_fovea_geo.py", "tests/test_r28_geo_control.py",
                 "tools/r28/r28_geo_control.py", "tools/r28/r28_geo_chain.sh")
SCIENCE_ADDED = ("src/sparse_rtdetr/research/r28_fovea_geo.py",)
# files of the shared core named by task 009 section 2; each must be byte-identical to the 006 release
CORE_IDENTICAL = ("tools/r28/r28_pipeline.py", "tools/r28/r28_cal_control.py", "tools/r28/r28_fovea_control.py", "tools/r28/r28_followup.py",
                  "tools/r27_l0/l0_loop.py", "tools/r27_l0/l0_common.py", "tools/r27_l0/l0_checks_state.py",
                  "src/sparse_rtdetr/research/r28_fovea.py", "src/sparse_rtdetr/research/r28_fovea_cal.py",
                  "src/sparse_rtdetr/research/r28_warp.py", "src/sparse_rtdetr/research/r28_resample.py",
                  "src/sparse_rtdetr/research/r27_l0_control.py", "src/sparse_rtdetr/research/scale_baseline.py",
                  "src/sparse_rtdetr/protocol_r18/engine.py", "src/sparse_rtdetr/protocol_r18/checkpoint.py")
# the arm dispatch / identity edits of the changed tool files, function by function; every other function is unchanged
EDITS = {
    "tools/r28/r28_common.py": {"changed": ("arm_module", "binding", "demand_source", "endpoint_state", "load_contract", "pipeline_mode"),
                                "added": ("selected_width",)},
    "tools/r28/r28_loop.py": {"changed": ("Observer.update_row", "cal_row", "guard", "train"), "added": ()},
    "tools/r28/r28_eval.py": {"changed": ("evaluate", "summarize"), "added": ()},
    "tools/r28/r28_cost.py": {"changed": ("cost", "instrument"), "added": ()},
    "tools/r28/r28_run.py": {"changed": ("main",), "added": ()},
}
KEPT = {"tools/r28/r28_common.py": ("save", "read_side", "read_state", "load", "stage_closed", "formal_dir", "calibration"),
        "tools/r28/r28_loop.py": ("run_loop", "restore_files", "install_observer", "formal", "Observer.loss_boxes", "Observer.loss_labels_vfl")}
TESTS = CC.TESTS + ("tests/test_r28_fovea_geo.py", "tests/test_r28_geo_control.py")
ARMS5 = ("L0", "K", "F", "FG", "W")
PAIRS = (("W", "FG"), ("FG", "F"), ("FG", "L0"), ("FG", "K"), ("W", "F"), ("W", "L0"), ("W", "K"), ("F", "L0"), ("K", "L0"))
PRIMARY = (("600", "APS"), ("600", "AP75_small"))
COST_PAIRS = ("FG/L0", "W/L0", "F/L0", "K/L0", "W/FG", "FG/W", "FG/F", "W/F", "W/K")
WIRE = {"eval_batches": 1, "cost_warmup": 1, "cost_batches_per_pass": 1, "cost_passes": 1}
ACCEPT = {"steps": 4, "save_at": 2, "ready_limit": 1, "workers": 2}
CONFIRM = {"epoch": 118, "windows": 32, "occurrences": 512}
SCOPE = "fg-acceptance"
_ACCEPT_UPDATES = ACCEPT["steps"] * 2 + (ACCEPT["steps"] - ACCEPT["save_at"])
WORKLOAD = {
    "geo-check": {"device": "cpu", "updates": 0, "forwards": 0, "demand_forwards": 0,
                  "note": "unit tests in a subprocess, identities, the stored 006 readings and cache re-read at the fixed width, "
                          "the official rule at the fixed width; no model"},
    "geo-confirm": {"updates": 0, "forwards": 0, "demand_forward_images": CONFIRM["occurrences"], "detector_forwards": 0,
                    "note": "the frozen demand network once per image occurrence of the first 32 windows of epoch 118"},
    "wire": {"updates": 0, "forwards": WIRE["eval_batches"] + 5 * (WIRE["cost_warmup"] + WIRE["cost_passes"] * WIRE["cost_batches_per_pass"]),
             "scorer_calls": 0, "note": "wiring of the evaluation batch and of the five-arm staged timing; listed apart"},
    "accept": {"updates": _ACCEPT_UPDATES, "forwards": 2 * _ACCEPT_UPDATES,
               "paths": {"serial": ACCEPT["steps"], "overlap": ACCEPT["steps"], "resumed": ACCEPT["steps"] - ACCEPT["save_at"]}},
    "formal": K.WORKLOAD["formal"],
    "evaluation": K.WORKLOAD["evaluation"],
    "cost-FG": K.WORKLOAD["cost-FG"],
    "ledger-means": {"device": "cpu", "updates": 0, "forwards": 0, "note": "reads three stored ledgers"},
    "planned_updates": _ACCEPT_UPDATES + (K.ENDPOINT - K.PARENT_UPDATE),
    "planned_complete_detector_forwards": 2 * _ACCEPT_UPDATES + 2 * (K.ENDPOINT - K.PARENT_UPDATE)
    + (K.EVAL_WARMUP + K.EVAL_BATCHES) + len(K.COST_ENDPOINTS["cost-FG"]) * K.COST_PER_ENDPOINT,
    "listed_apart": ["geo-confirm (at most 512 demand-network image forwards)", "wire (11 forwards)", "dev rounds",
                     "any failure or re-verification"],
    "reused_not_repeated": ["the 80-update serial / overlap / resume equivalence of the 006 dev round (pinned receipt)",
                            "the pure-data boundary and refusal checks of the 006 cal-check on the real loader (pinned receipt)",
                            "the 66 width readings and the epoch-61 demand cache of the 006 calibration (pinned files)",
                            "the CPU fixture and the official-rule / renderer checks of the W, K, F and 006 rounds"],
    "not_run": ["demand network training", "W / K / F / L0 training", "W / K / F / L0 evaluation or rescoring", "a second width or any "
                "width scan", "any seed, schedule, microbatch, gain, slope bound or medium-module variation", "checkpoint selection",
                "holdout access"],
}
PIPELINE_RULE = {
    "version": CC.PIPELINE_RULE["version"], "workers": ACCEPT["workers"], "ready_limit": ACCEPT["ready_limit"],
    "acceptance": {"parent": "e060", "windows": "epoch 61 from its first window", "serial_updates": ACCEPT["steps"],
                   "overlap_updates": ACCEPT["steps"], "save_after_update": ACCEPT["save_at"],
                   "replayed_updates": ACCEPT["steps"] - ACCEPT["save_at"], "compared": CC.PIPELINE_RULE["acceptance"]["compared"]},
    "decision": "the overlapped path is used for the formal run only if every compared field of the three paths is equal and the "
                "restore is exact with the actual width; this short acceptance is not a throughput measurement (task 009 section "
                "3) and no timing enters the decision; otherwise the stage stops and nothing is trained",
    "inherited": "the general 80-update equivalence and the steady-state timing are the pinned dev receipts of the 006 round "
                 "(stand-in width 12.53125, same pipeline / loader / engine / save-restore / comparison bytes)",
    "not_changed": CC.PIPELINE_RULE["not_changed"],
}
STATEMENTS = {
    "registration": "本对照形成于看过 W、K、F 结果以及任务006的几何读数之后，是探索性控制；不是006的继续运行，也不是原 W 的预注册检验。",
    "W_decision": "旧 W 的 closed_fixed_config（APM600 未过）、K_trigger=false、旧成本 C，以及 K、F 的读数原样保留；F_geo 的任何读数都不改写它们。",
    "scope": "F_geo 只把 F 的全局高斯吸引核 FWHM 由 4 改为固定值 19.03125（两轴同值）。该值是任务006已测 66 个核宽中 small 池化倍率"
             "最高、最接近 W 的点，只由训练集几何读数确定，没有用任何核宽的检测结果选择。",
    "not_F_star": "任务006的标定状态仍是 calibration_unmatched、h_star=null；F_geo 不是等幅控制 F*，不是 AP 最优 F，也不是连续参数最优。"
                  "epoch61 上其 small 池化倍率比 W 低约 4.98%，逐实例分布、尾部、各向异性、medium／large 也不同。",
    "not_isolated": "W−F_geo 是两个实际方法配置的比较；平均倍率未匹配，不能据此声称隔离了倍率之外的独有机制，也不能称已排除所有幅度解释。"
                    "F_geo 不是官方 FOVEA 的完整复现。",
    "record_only": "W−F_geo 等比较登记为 RECORD_ONLY：给出全部绝对值、差值、成本与局限，不设新的通过门；差值小不构成统计等效。",
    "contribution": "方法贡献仍为 not_established：单 seed（seed0）、结果后追加、重复使用的 548 图 development 集。",
    "cost": "五端点同进程测速的新比值单列，不覆盖 W 轮登记的 C，也不与 K 轮、F 轮各自的同进程比值混为同一次测量；"
            "训练输入流水线的提速是训练工程项，不计入任何方法的推理收益。",
    "large": "large 读数由少数稀疏类主导，不作整体优劣依据；APL6 只是补充读数。",
    "no_trigger": "本结果不触发自动扩 seed、改倍率界、加 medium 保护或进一步核宽训练。",
}
LABEL = "strong_fovea_control_completed"
CONTRIBUTION = "not_established"
DEV_CONFIRM = "R28_DEV_CONFIRM_WINDOWS"


# ================================================================ release difference
def source_delta(old_files, new_files, *, label="the 006 round release"):
    """File-by-file difference between the 006 release and this one.  The science tree may only gain the arm FG module:
    no file of src/, vendor/ or configs/ is changed or removed, and the named core files are byte-identical."""
    changed = sorted(k for k in old_files if k in new_files and old_files[k] != new_files[k])
    added = sorted(k for k in new_files if k not in old_files)
    removed = sorted(k for k in old_files if k not in new_files)
    K.require(not removed and set(changed) <= set(ALLOWED_CHANGED) and set(added) <= set(ALLOWED_ADDED),
              f"release differs from {label} outside the arm FG adapter: changed {changed}, added {added}, removed {removed}")
    science = ("src/", "vendor/", "configs/")
    K.require(not any(p.startswith(science) for p in changed) and all(p in SCIENCE_ADDED for p in added if p.startswith(science)),
              f"release touches the science tree of {label} beyond the added arm FG module")
    K.require(all(p in old_files and old_files[p] == new_files.get(p) for p in CORE_IDENTICAL),
              f"a core file differs from {label}: {[p for p in CORE_IDENTICAL if old_files.get(p) != new_files.get(p)]}")
    return {"against": label, "changed": changed, "added": added, "removed": removed, "unchanged_files": len(old_files) - len(changed),
            "existing_science_files_identical": True, "science_files_added": [p for p in added if p.startswith(science)],
            "core_files_identical": {p: new_files[p] for p in CORE_IDENTICAL},
            "note": "every existing file of src/, vendor/ and configs/ (arm W / K / F / FS operators and wrappers, detector, "
                    "criterion, data, engine, checkpoint) and the input pipeline, loop base, comparison and control tools of "
                    "the earlier rounds are byte-identical; arm FG is one added module"}


def functions(text):
    """name -> source text of every top-level function and every method of a top-level class."""
    out = {}
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = ast.get_source_segment(text, node)
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out[f"{node.name}.{sub.name}"] = ast.get_source_segment(text, sub)
    return out


def function_delta(old_text, new_text):
    a, b = functions(old_text), functions(new_text)
    return {"changed": sorted(k for k in a if k in b and a[k] != b[k]), "added": sorted(set(b) - set(a)),
            "removed": sorted(set(a) - set(b)), "unchanged": sorted(k for k in a if b.get(k) == a[k])}


def edits(old_root, new_root):
    """The function-level difference of every changed tool file against the 006 release; it must be exactly EDITS."""
    out = {}
    for rel, want in EDITS.items():
        d = function_delta((Path(old_root) / rel).read_text(), (Path(new_root) / rel).read_text())
        K.require(d["changed"] == sorted(want["changed"]) and d["added"] == sorted(want["added"]) and not d["removed"]
                  and set(KEPT.get(rel, ())) <= set(d["unchanged"]),
                  f"{rel}: function edits differ from the registered arm-dispatch list: {d['changed']} {d['added']} {d['removed']}")
        out[rel] = {"changed": d["changed"], "added": d["added"], "removed": d["removed"], "unchanged_functions": len(d["unchanged"]),
                    "kept_named": list(KEPT.get(rel, ()))}
    return out


# ================================================================ references into the 006 round (read-only)
def cal_references(cal_round=None):
    """The 006 round: contract, release, the unmatched calibration with its 66 readings and its cache, the cal-check
    receipt and the dev-round-2 receipts of the input pipeline.  Everything pinned by file SHA; nothing is written."""
    from sparse_rtdetr.research import r28_fovea_cal as FS
    from sparse_rtdetr.research import r28_fovea_geo as FG
    r = Path(cal_round or CAL_ROUND).resolve()
    refs = {k: K.ref(r / v) for k, v in CAL_FILES.items()}
    bad = [k for k in refs if refs[k]["sha256"] != PIN_CAL[k]]
    K.require(not bad, f"files of the 006 round differ from their pins: {bad}")
    c6 = K.read_json(r / CAL_FILES["contract_file"])
    body = {k: v for k, v in c6.items() if k != "contract_sha256"}
    K.require(c6["contract_sha256"] == K.sha_bytes(K.canonical(body)) == PIN_CAL["contract"] and c6["source_commit"] == PIN_CAL["commit"]
              and c6["source_manifest"]["sha256"] == PIN_CAL["manifest"] and c6["authorization_basis"] == K.CAL_BASIS
              and Path(c6["outputs"]).resolve() == r and c6["parent"]["sha256"] == K.PARENT_SHA, "006 round contract identities")
    cal = K.read_json(r / CAL_FILES["calibration"])
    rows = cal["readings"]
    best = max(rows, key=lambda x: x["L"])
    near = min(rows, key=lambda x: (abs(x["relative_error"]), x["h"]))
    K.require(cal["status"] == "calibration_unmatched" and cal["h_star"] is None and cal["h_star_hex"] is None
              and cal["contract_sha256"] == PIN_CAL["contract"] and cal["dev_target_width"] is None and len(rows) == FS.GRID_POINTS + 1
              and all(x["valid"] and x["stage"] == "table" for x in rows) and [x["h"] for x in rows] == FS.candidates()
              and not any(abs(x["relative_error"]) <= cal["tolerance"] for x in rows) and all(x["relative_error"] < 0 for x in rows)
              and best["h_hex"] == near["h_hex"] == cal["closest_if_unmatched"]["h_hex"] == FG.WIDTH_HEX and best["h"] == FG.WIDTH
              and cal["calibration_set"]["cache"] == refs["cache"] and cal["calibration_set"]["small_GT"] == best["n_small"],
              "the stored 006 calibration is not the unmatched table whose largest and closest reading is the fixed width")
    reading = {"W_pooled_small_magnification": math.exp(cal["L_W"]), "FG_pooled_small_magnification": best["pooled_geometric_mean"],
               "FG_over_W_minus_1": best["relative_error"], "tolerance_of_006": cal["tolerance"], "matched": False}
    registered = FG.SELECTION["reading"]
    K.require(set(reading) == set(registered) and reading["matched"] is registered["matched"] is False
              and all(abs(reading[k] - registered[k]) <= 1e-12 for k in reading if k != "matched"),
              f"stored readings differ from the registered selection: {reading}")
    closed = K.read_json(r / CAL_FILES["calibrate_closed"])
    check = K.read_json(r / CAL_FILES["cal_check_closed"])
    K.require(closed["status"] == check["status"] == "PASS" and closed["contract_sha256"] == check["contract_sha256"] == PIN_CAL["contract"]
              and closed["calibration_status"] == "calibration_unmatched" and closed["h_star"] is None
              and closed["formal_training_allowed"] is False and closed["calibration"] == refs["calibration"]
              and not list(r.glob("formal-FS*")) and not (r / "summary-FS").exists(), "006 calibrate / cal-check receipts")
    cdev = K.read_json(r / CAL_FILES["dev_contract_file"])
    dbody = {k: v for k, v in cdev.items() if k != "contract_sha256"}
    pipe = K.read_json(r / CAL_FILES["dev_pipeline_closed"])
    eq = K.read_json(r / CAL_FILES["dev_equivalence"])
    tm = K.read_json(r / CAL_FILES["dev_timing"])
    K.require(cdev["contract_sha256"] == K.sha_bytes(K.canonical(dbody)) and cdev["source_commit"] == PIN_CAL["commit"]
              and cdev["source_manifest"]["sha256"] == PIN_CAL["dev_manifest"] == PIN_CAL["manifest"]
              and pipe["status"] == "PASS" and pipe["contract_sha256"] == cdev["contract_sha256"] == eq["contract_sha256"] == tm["contract_sha256"]
              and pipe["equivalence"] == refs["dev_equivalence"] and pipe["timing"] == refs["dev_timing"]
              and all(eq[k]["equal"] and not eq[k]["differences"] for k in ("serial_vs_overlap", "serial_vs_resumed", "overlap_vs_resumed"))
              and eq["updates"] == 80 and eq["complete_detector_forwards"] == 160 and eq["prepared_windows_at_the_save"] == 1
              and eq["excluded"] == CC.EXCLUDED and eq["compared"] == CC.COMPARED, "006 dev receipts of the input pipeline")
    dev = {"equivalence": refs["dev_equivalence"], "timing": refs["dev_timing"], "closed": refs["dev_pipeline_closed"],
           "contract": refs["dev_contract_file"], "contract_sha256": cdev["contract_sha256"], "manifest": refs["dev_manifest"],
           "width": eq["mapping"]["fwhm"], "updates": eq["updates"], "complete_detector_forwards": eq["complete_detector_forwards"],
           "steps_compared": {k: eq[k]["steps"] for k in ("serial_vs_overlap", "serial_vs_resumed", "overlap_vs_resumed")},
           "steady_cycle_measured_s": {k: tm[k]["steady_cycle_measured_s"] for k in ("serial", "overlap")},
           "steady_cycle_adjusted_s": {k: tm[k]["steady_cycle_s"] for k in ("serial", "overlap")},
           "raw_speedup": tm["serial"]["steady_cycle_measured_s"] / tm["overlap"]["steady_cycle_measured_s"],
           "adjusted_speedup": tm["serial"]["steady_cycle_s"] / tm["overlap"]["steady_cycle_s"],
           "note": "dev round of task 006 with a stand-in width: general wiring evidence of the pipeline code, quoted and not "
                   "renamed into an acceptance of this round; the speed-ups are references, not a promise for this arm"}
    return c6, cal, {"cal_round": str(r), "cal_contract": refs["contract_file"], "cal_contract_sha256": c6["contract_sha256"],
                     "cal_commit": c6["source_commit"], "cal_manifest": refs["manifest"], "cal_calibration": refs["calibration"],
                     "cal_calibrate_closed": refs["calibrate_closed"], "cal_check_closed": refs["cal_check_closed"],
                     "cal_cache": refs["cache"], "cal_status": cal["status"], "cal_h_star": cal["h_star"],
                     "dev_pipeline": dev, "dev_FS_state": refs["dev_FS_state"]}


def selection(cal, cal_ref, task_ref):
    """The contract's selection block: what the fixed width is and which stored reading it was taken from."""
    from sparse_rtdetr.research import r28_fovea_geo as FG
    row = next(x for x in cal["readings"] if x["h_hex"] == FG.WIDTH_HEX)
    return {"kind": FG.SELECTION_KIND, "fwhm": FG.WIDTH, "fwhm_hex": FG.WIDTH_HEX, "calibration": cal_ref,
            "calibration_status": cal["status"], "h_star": cal["h_star"], "task": task_ref,
            "reading": {k: row[k] for k in ("L", "relative_error", "pooled_geometric_mean", "per_window_geometric_mean_mean", "n_small",
                                            "node_slope_min", "node_slope_max")},
            "L_W": cal["L_W"], "rule": FG.SELECTION["rule"], "not_used": FG.SELECTION["not_used"], "scope": FG.SELECTION["scope"]}


# ================================================================ contract of the independent round
def prepare(_args=None):
    from sparse_rtdetr.research import r28_fovea as FV
    from sparse_rtdetr.research import r28_fovea_geo as FG
    from sparse_rtdetr.research import r28_resample as R
    from sparse_rtdetr.research import r28_warp as W
    round_ = K.ROUND.resolve()
    K.require(K.ROOT.resolve().parent.parent == round_ and K.ROOT.name == "source", "prepare-geo must run from a frozen release")
    K.require("R28_ROUND" in os.environ and all(round_ != p and p not in round_.parents
                                                for p in (FU.PRIOR_ROUND.resolve(), FC.K_ROUND.resolve(), CC.F_ROUND.resolve(), CAL_ROUND.resolve())),
              "the F_geo round must be an independent directory named by R28_ROUND")
    K.require(not K.DEV or round_.parent.parent == K.LD_ROOT.resolve(), "a dev round lives inside the round directory")
    manifest = K.read_json(K.ROOT.parent / "MANIFEST.json")
    K.require(manifest["files"] == K.B.source_manifest(K.ROOT), "release source differs from MANIFEST")
    K.require(not manifest.get("dirty", False) or K.DEV, "a release assembled from a dirty work tree is only accepted in a dev round")
    task, review = (K.ref(K.ROUND / f[0]) for f in (TASK, REVIEW))
    K.require(task["sha256"] == TASK[1] and review["sha256"] == REVIEW[1], "task / review file SHA")
    cf, f_inherited, f_refs = CC.f_references()
    c1, ck, fu0 = CC._earlier(cf)
    c6, cal, cal_refs = cal_references()
    K.require(set(fu0["inherited"]).isdisjoint(f_inherited) and set(OWN_STAGES).isdisjoint(set(fu0["inherited"]) | set(f_inherited) | set(CC.OWN_STAGES)),
              "stage names of the rounds overlap")
    K.require(c6["followup"]["F_contract_sha256"] == cf["contract_sha256"] and c6["parent"] == cf["parent"]
              and c6["followup"]["W_demand"] == fu0["W_demand"], "the 006 round and the F round name different inputs")
    delta = source_delta(K.read_json(K.pinned(c6["source_manifest"]))["files"], manifest["files"])
    function_edits = edits(c6["source_root"], K.ROOT)
    K.require(cf["identity_sha256"] == {**R.IDENTITY_SHA256, **FV.IDENTITY_SHA256} and cf["fovea_definition"] == FV.DEFINITION
              and cf["definition"] == W.DEFINITION and cf["identity"]["F"] == FV.IDENTITY["F"],
              "arm W / K / F behaviour identity differs from the F round")
    keep = {k: v for k, v in fu0.items() if k not in ("registration", "authorization_basis", "arm", "task", "review", "source_delta",
                                                      "trained_here", "reused_frozen", "analysis", "statements", "inherited",
                                                      "inherited_contracts")}
    c = {k: v for k, v in cf.items() if k != "contract_sha256"}
    c.update(version="R28-FG-control-009-contract-v1", authorization_basis=K.GEO_BASIS,
             source_root=str(K.ROOT.resolve()), source_manifest=K.ref(K.ROOT.parent / "MANIFEST.json"),
             source_commit=manifest["commit"], source_dirty=bool(manifest.get("dirty", False)),
             identity={**cf["identity"], **FG.IDENTITY}, identity_sha256={**cf["identity_sha256"], **FG.IDENTITY_SHA256},
             fovea_geo_definition=FG.DEFINITION, workload=WORKLOAD, input_pipeline=PIPELINE_RULE,
             cost={**cf["cost"], "cost-FG": {"endpoints": list(K.COST_ENDPOINTS["cost-FG"]),
                                             "rotations": [list(r) for r in K.COST_ROTATIONS["cost-FG"]]}},
             labels={**cf["labels"], "FG_completed": LABEL, "FG_contribution": CONTRIBUTION},
             resource_plan={"state_bytes": cf["parent"]["bytes"], "formal_states": len(K.SAVE_EPOCHS),
                            "disk_free_bytes_at_prepare": shutil.disk_usage(K.ROUND).free,
                            "gpu_time_reference": "serial F measured 11.29 h; the 006 dev steady state suggests about 9-12 h with the "
                                                  "overlapped input path; a wider kernel may change the resampling time; not a cap"},
             outputs=str(K.ROUND), formal_started=False,
             followup={**keep, "registration": REGISTRATION, "authorization_basis": K.GEO_BASIS, "arm": ARM, "task": task,
                       "review": review, **f_refs, **cal_refs, "width_selection": selection(cal, cal_refs["cal_calibration"], task),
                       "inherited": {**fu0["inherited"], **f_inherited},
                       "inherited_contracts": {**fu0["inherited_contracts"], **{name: cf["contract_sha256"] for name in f_inherited}},
                       "source_delta": delta, "function_edits": function_edits,
                       "trained_here": ["F_geo detector, 18,240 updates from the e060 parent", "10 acceptance updates of the actual "
                                        "width (discarded states)"],
                       "reused_frozen": ["W demand network (demand/demand-net.pth of the closed W round): the demand source of F_geo",
                                         "W / K / F e120 endpoints (timing and identity only)", "W / K / F / L0 predictions, "
                                         "metrics and per-class readings (quoted, not rescored)",
                                         "the 006 calibration readings, its epoch-61 cache and its dev pipeline receipts"],
                       "analysis": {"record_only": True, "primary": [f"W-FG top{b} {m}" for b, m in PRIMARY],
                                    "pairs": [f"{x}-{y}" for x, y in PAIRS]},
                       "statements": STATEMENTS})
    c["contract_sha256"] = K.sha_bytes(K.canonical(c))
    K.write_new(K.ROUND / "CONTRACT.json", c)
    print(json.dumps({"contract_sha256": c["contract_sha256"], "identity_sha256_FG": FG.IDENTITY_SHA256["FG"],
                      "authorization_basis": c["authorization_basis"], "fwhm_hex": FG.WIDTH_HEX,
                      "source_delta": {k: delta[k] for k in ("against", "changed", "added")}}))
    return c


def _forced_pipeline(c):
    old = os.environ.get("R28_PIPELINE")
    os.environ["R28_PIPELINE"] = "overlap"
    try:
        K.pipeline_mode(c, "FG")
    finally:
        if old is None:
            del os.environ["R28_PIPELINE"]
        else:
            os.environ["R28_PIPELINE"] = old


# ================================================================ affected-interface check (CPU, no model)
def geo_check(c, out, counts):
    import r28_cost  # noqa: F401  (import check of the timing module)
    import r28_eval as E
    import r28_loop as LP
    import r28_pipeline as PL
    from sparse_rtdetr.research import r28_fovea as FV
    from sparse_rtdetr.research import r28_fovea_cal as FS
    from sparse_rtdetr.research import r28_fovea_geo as FG
    from sparse_rtdetr.research import r28_warp as W
    torch = K.torch_mod()
    out = Path(out)
    K.require(out.resolve() == (K.ROUND / "geo-check").resolve(), "this action writes <round>/geo-check")
    K.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_available(), "the F_geo check hides CUDA")
    fu = K.followup(c)
    K.require(fu is not None and K.followup_arm(c) == "FG" and K.ARM == "FG", "geo-check runs with R28_ARM=FG under the F_geo contract")
    torch.set_num_threads(int(os.environ.get("R28_CPU_THREADS", "16")))
    t0 = time.perf_counter()
    res = {"stage": "geo-check", "registration": fu["registration"], "authorization_basis": c["authorization_basis"],
           "workload": c["workload"]["geo-check"]}

    K.PHASE["phase"] = "geo-check:pytest"
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *[str(K.ROOT / t) for t in TESTS]],
                       capture_output=True, text=True, cwd=str(K.ROOT), env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    res["pytest"] = {"returncode": p.returncode, "tail": p.stdout.strip().splitlines()[-1:], "tests": list(TESTS),
                     "note": "synthetic unit tests in a subprocess; their forwards are not in this stage's counts"}
    K.require(p.returncode == 0, "synthetic pytest failed:\n" + p.stdout[-6000:])

    K.PHASE["phase"] = "geo-check:identities"
    cf, f_inherited, f_refs = CC.f_references(fu["F_round"])
    c1, ck, fu0 = CC._earlier(cf)
    c6, cal, cal_refs = cal_references(fu["cal_round"])
    K.require(all(fu[k] == v for k, v in {**f_refs, **cal_refs}.items()) and fu["inherited"] == {**fu0["inherited"], **f_inherited}
              and all(fu[k] == fu0[k] for k in ("prior_contract", "prior_contract_sha256", "K_contract", "K_contract_sha256", "W_demand",
                                                "W_endpoint", "K_endpoint", "official", "prior_W_decision", "prior_W_result", "K_result")),
              "references changed since prepare")
    K.require(K.other_contract(c, "F")["contract_sha256"] == cf["contract_sha256"] and K.other_contract(c, "K")["contract_sha256"] == ck["contract_sha256"]
              and K.prior_contract(c)["contract_sha256"] == c1["contract_sha256"]
              and len({c["contract_sha256"], cf["contract_sha256"], ck["contract_sha256"], c1["contract_sha256"], c6["contract_sha256"]}) == 5,
              "contract digests of the five rounds")
    for name, f in (("task", TASK), ("review", REVIEW)):
        K.require(K.pinned(fu[name]) == K.ROUND / f[0] and fu[name]["sha256"] == f[1], f"{name} file")
    manifest = K.read_json(K.pinned(c["source_manifest"]))
    old_manifest = K.read_json(K.pinned(fu["cal_manifest"]))
    delta = source_delta(old_manifest["files"], manifest["files"])
    K.B.release_intact(Path(c6["source_root"]), old_manifest["files"])
    function_edits = edits(c6["source_root"], K.ROOT)
    K.require(delta == fu["source_delta"] and function_edits == fu["function_edits"], "source delta changed since prepare")
    f_files = K.read_json(K.pinned(fu["F_manifest"]))["files"]
    science = ("src/", "vendor/", "configs/")
    against_f = sorted(k for k in manifest["files"] if k.startswith(science) and manifest["files"][k] != f_files.get(k))
    K.require(against_f == sorted(("src/sparse_rtdetr/research/r28_fovea_cal.py",) + SCIENCE_ADDED)
              and all(manifest["files"].get(k) == v for k, v in f_files.items() if k.startswith(science)),
              f"science tree against the F release: {against_f}")
    same = [k for k in cf if k not in ("contract_sha256", "version", "authorization_basis", "source_root", "source_manifest", "source_commit",
                                       "source_dirty", "workload", "labels", "resource_plan", "outputs", "formal_started", "identity",
                                       "identity_sha256", "cost", "followup")]
    K.require(all(c[k] == cf[k] for k in same) and all(c["identity_sha256"][a] == cf["identity_sha256"][a] for a in ("W", "K", "F"))
              and c["identity_sha256"]["F"] == FV.IDENTITY_SHA256["F"] == fu["F_identity_sha256"]
              and c["identity_sha256"]["FG"] == FG.IDENTITY_SHA256["FG"] and c["identity"]["FG"] == FG.IDENTITY["FG"]
              and "FS" not in c["identity_sha256"] and FG.IDENTITY_SHA256["FG"] != FS.IDENTITY_SHA256["FS"]
              and c["fovea_geo_definition"] == FG.DEFINITION and c["fovea_definition"] == FV.DEFINITION and c["definition"] == W.DEFINITION
              and c["parent"]["sha256"] == CC.PIN_F["parent"] and len(set(c["identity_sha256"].values())) == 4
              and c["input_pipeline"]["version"] == PL.VERSION,
              "inherited contract fields / identities")
    res["identities"] = {"cal_round_pins": dict(PIN_CAL), "F_round_pins": dict(CC.PIN_F), "W_round_pins": dict(FU.PIN), "K_round_pins": dict(FC.PIN_K),
                         "inherited_stages": {"W_and_K_rounds": list(fu0["inherited"]), "F_round": list(f_inherited)},
                         "contract_fields_equal_to_the_F_round": len(same), "source_delta": delta, "function_edits": function_edits,
                         "science_tree_against_the_F_release": {"added": against_f, "every_F_release_science_file_identical": True},
                         "identity_sha256": c["identity_sha256"], "parent_sha256": c["parent"]["sha256"],
                         "release_006_intact": True}

    K.PHASE["phase"] = "geo-check:authorisation and isolation"
    rounds = [Path(fu[k]) for k in ("prior_round", "K_round", "F_round", "cal_round")]
    for name in fu["inherited"]:
        K.require(not (K.ROUND / name).exists(), f"inherited stage name present in this round: {name}")
    w_summary = K.stage_closed(c, "summary-W", "W")
    K.require(w_summary["K_trigger"] is False and w_summary["label"] == "closed_fixed_config" and fu["prior_W_decision"] == FU.W_DECISION,
              "inherited W summary")
    K.require(K.stage_closed(c, "eval-F", "F")["endpoint"] == fu["F_endpoint"] and K.stage_closed(c, "eval-K", "K")["endpoint"] == fu["K_endpoint"]
              and K.stage_closed(c, "demand")["weights"] == fu["W_demand"] and fu["F_endpoint"]["sha256"] == CC.PIN_F["endpoint"],
              "inherited F / K evaluation and W demand network")
    sel = fu["width_selection"]
    lock = K.selected_width(c)
    K.require(lock == {"fwhm": FG.WIDTH, "selection": sel} and sel == selection(cal, fu["cal_calibration"], fu["task"]),
              "the contract's selection block")
    net = {"kind": "network", "file": fu["W_demand"]}
    good = {**net, "fwhm": FG.WIDTH, "selection": sel}
    K.require(FG.check_source("FG", good, False) == FG.mapping(good) == {"fwhm": FG.WIDTH, "fwhm_hex": FG.WIDTH_HEX, "selection": sel},
              "the formal arm FG source")
    tampered = lambda **kw: {**c, "followup": {**fu, "width_selection": {**sel, **kw}}}
    cases = [
        *[FU._refused(f"formal training of arm {a} in the F_geo round", lambda a=a: LP.guard(c, K.ROUND / f"formal-{a}", None, a))
          for a in ("W", "K", "F", "FS")],
        FU._refused("formal F_geo before the confirmation and the acceptance stages are closed", lambda: LP.guard(c, K.ROUND / "formal-FG", None, "FG")),
        FU._refused("a matched calibration lock in this round (arm FS guard)", lambda: K.calibration(c)),
        FU._refused("arm FS demand source in this round", lambda: K.demand_source(c, "FS")),
        FU._refused("a formal arm FS source without a calibration file (arm FS module guard)",
                    lambda: FS.check_source("FS", {**net, "fwhm": FG.WIDTH}, False)),
        FU._refused("a formal F directory lookup in this round", lambda: K.formal_dir(c, "F")),
        *[FU._refused(f"an evaluation output inside {r.name}", lambda r=r: E.round_stage(r / "eval-FG", "eval-FG")) for r in rounds],
        FU._refused("a follow-up block without the registered authorization basis", lambda: K.followup({**c, "authorization_basis": "none"})),
        FU._refused("the F_geo block under the calibrated-control basis",
                    lambda: K.followup({**c, "authorization_basis": K.CAL_BASIS, "followup": {**fu, "authorization_basis": K.CAL_BASIS}})),
        FU._refused("arm FS built by the arm FG module", lambda: FG.check_source("FS", good, False)),
        FU._refused("arm F built by the arm FG module", lambda: FG.check_source("F", good, False)),
        FU._refused("arm FG with arm F's width 4", lambda: FG.check_source("FG", {**good, "fwhm": 4.0}, False)),
        FU._refused("arm FG with the official default 13", lambda: FG.check_source("FG", {**good, "fwhm": 13.0}, False)),
        FU._refused("arm FG with a neighbouring table width", lambda: FG.check_source("FG", {**good, "fwhm": 19.4375}, False)),
        FU._refused("arm FG with an integer width", lambda: FG.check_source("FG", {**good, "fwhm": 19}, False)),
        FU._refused("arm FG without a width", lambda: FG.check_source("FG", {**net, "selection": sel}, False)),
        FU._refused("a formal arm FG source without the selection reference", lambda: FG.check_source("FG", {**net, "fwhm": FG.WIDTH}, False)),
        FU._refused("arm FG with a calibration lock (posing as F*)",
                    lambda: FG.check_source("FG", {**good, "calibration": sel["calibration"]}, False)),
        FU._refused("arm FG whose selection claims a matched calibration",
                    lambda: FG.check_source("FG", {**good, "selection": {**sel, "calibration_status": "matched", "h_star": FG.WIDTH}}, False)),
        FU._refused("arm FG whose selection names another calibration file",
                    lambda: FG.check_source("FG", {**good, "selection": {**sel, "calibration": {**sel["calibration"], "sha256": "0" * 64}}}, False)),
        FU._refused("arm FG with a fixed (arm K) demand source", lambda: FG.check_source("FG", {**good, "kind": "fixed"}, False)),
        FU._refused("a contract whose selection names another width", lambda: K.selected_width(tampered(fwhm_hex=(13.0).hex()))),
        FU._refused("a contract whose selection claims h_star", lambda: K.selected_width(tampered(h_star=FG.WIDTH))),
        FU._refused("R28_PIPELINE forcing the overlapped path", lambda: _forced_pipeline(c)),
    ]
    K.require(not any(v.startswith("R28_DEV_") for v in os.environ) or K.DEV, "dev flags are refused outside a dev round")
    res["authorisation"] = {"round": str(K.ROUND), "independent_of": [str(r) for r in rounds], "refusals": cases,
                            "W_summary_inherited": {k: w_summary[k] for k in ("label", "label_key", "K_trigger", "M_failed_cells")},
                            "calibration_006": {"status": cal["status"], "h_star": cal["h_star"], "kept": True,
                                                "formal_FS_directory_exists": False}}

    K.PHASE["phase"] = "geo-check:fixed width, kernel and map identity"
    k = FS.kernel_matrix(FS.GRID_SHAPE[0], FG.WIDTH)
    syn = {name: v[None] for name, v in FC.synthetic_demands().items()}
    stacked = torch.cat(list(syn.values()))
    pg, ps, pf = FG.plan_from_demand(stacked), FS.plan_from_demand(stacked, FG.WIDTH), FV.plan_from_demand(stacked)
    K.require(FG.WIDTH == 19.03125 and FG.WIDTH.hex() == FG.WIDTH_HEX == "0x1.3080000000000p+4" and FG.WIDTH in FS.candidates()
              and FS.kernel_key(31, FG.WIDTH) == (31, FS.PAD, FG.WIDTH_HEX, "float64")
              and not torch.equal(k, FV.kernel_matrix(FV.GRID_SHAPE[0]))
              and torch.equal(pg.gx, ps.gx) and torch.equal(pg.gy, ps.gy) and torch.equal(pg.identity, ps.identity) and pg.stats == ps.stats
              and not torch.equal(pg.gx, pf.gx), "fixed width / kernel / map identity")
    res["width"] = {"fwhm": FG.WIDTH, "fwhm_hex": FG.WIDTH_HEX, "both_axes": True, "in_the_006_table": True,
                    "map_equals_the_arm_FS_function_at_this_width_bit_for_bit": True, "differs_from_arm_F": True,
                    "kernel_cache_key": list(FS.kernel_key(31, FG.WIDTH))}

    K.PHASE["phase"] = "geo-check:official rule at the fixed width"
    res["official_rule"] = {"official": fu["official"], "execution": "the pinned official file run unchanged on the CPU with "
                            "attraction_fwhm set to the fixed width (stand-ins only for its unused imports, its .cuda() calls and "
                            "the removed numpy alias np.float)", "tolerance": {k_: CC.TOL[k_] for k_ in ("official_fp32_nodes", "fp64_same_formula")},
                            "widths": [CC.official_check(fu, syn, FG.WIDTH)]}

    K.PHASE["phase"] = "geo-check:stored 006 readings and cache at the fixed width"
    cache = torch.load(K.pinned(fu["cal_cache"]), map_location="cpu", weights_only=True)
    blocks = CC.window_blocks(cache)
    w_logs, w_aniso = CC.geometry_w(blocks)
    w_pool = CC._pool(w_logs, blocks, "small")
    valid, logs, aniso, detail = CC.geometry_h(blocks, FG.WIDTH)
    K.require(valid, f"the fixed width is not valid on the stored epoch-61 cache: {detail}")
    pool = CC._pool(logs, blocks, "small")
    row = next(x for x in cal["readings"] if x["h_hex"] == FG.WIDTH_HEX)
    K.require(len(blocks) == cal["calibration_set"]["windows"] == 304 and len(cache["boxes"]) == cal["calibration_set"]["occurrences"] == 4864
              and w_pool["L"] == cal["L_W"] == cal["L_W_of_arm_W"] and pool["L"] == row["L"] and pool["n"] == row["n_small"] == 137229
              and FS.relative_error(pool["L"], w_pool["L"]) == row["relative_error"]
              and detail["node_slope_min"] == row["node_slope_min"] and detail["node_slope_max"] == row["node_slope_max"],
              "the stored epoch-61 cache does not reproduce the stored 006 readings at the fixed width")
    res["stored_readings"] = {
        "calibration": fu["cal_calibration"], "cache": fu["cal_cache"], "status": cal["status"], "h_star": cal["h_star"],
        "widths_in_the_table": len(cal["readings"]), "all_valid": True, "within_tolerance": 0,
        "largest_and_closest_width": {"h": row["h"], "h_hex": row["h_hex"]},
        "reproduced_bit_for_bit": {"L_W": w_pool["L"], "L_FG": pool["L"], "relative_error": row["relative_error"], "small_GT": pool["n"]},
        "epoch_61": {"W": CC.reading(blocks, w_logs, w_aniso), "FG": CC.reading(blocks, logs, aniso), "detail": detail},
        "note": "one width re-read on the stored cache as an identity check; the table was not recomputed and no width was selected again"}
    del cache, blocks, w_logs, logs

    K.PHASE["phase"] = "geo-check:inherited input-pipeline evidence"
    check6 = K.read_json(K.pinned(fu["cal_check_closed"]))
    pipe6 = check6["input_pipeline"]
    K.require(pipe6["version"] == PL.VERSION and pipe6["model_forwards"] == 0 and len(pipe6["refusals"]) == 4
              and all(r_["refused_before_consumption"] for r_ in pipe6["refusals"]) and pipe6["first_windows"]["pixels_targets_and_records_bitwise_equal"]
              and pipe6["restart"]["rebuilt_bitwise_equal_to_the_uninterrupted_window"] and set(pipe6["boundaries"]) == {"epoch_61_to_62", "epoch_117_to_118"},
              "the pure-data pipeline receipt of the 006 cal-check")
    res["inherited_pipeline_evidence"] = {
        "pure_data_checks_on_the_real_loader": {"receipt": fu["cal_check_closed"], "refusals_before_consumption": len(pipe6["refusals"]),
                                                "boundaries": sorted(pipe6["boundaries"]), "repeated_here": False},
        "cuda_equivalence_80_updates": {**fu["dev_pipeline"], "repeated_here": False},
        "valid_because": "tools/r28/r28_pipeline.py, the loop base, the loader, the engine, the checkpoint code and the comparison code "
                         "are byte-identical to the release those receipts were produced with (source_delta / function_edits)"}
    K.require(counts["forwards"] == 0 and counts["actual_updates"] == 0, f"geo-check counts {dict(counts)}")
    res.update(model_forwards=0, cpu_seconds_wall=time.perf_counter() - t0,
               scope="affected interfaces only: the fixed width, the arm identity, the contract, the references; the general CPU / "
                     "identity / evaluation-path / pipeline evidence of the shared code is inherited from the W, K, F and 006 rounds")
    return res


# ================================================================ validity of the fixed width on epoch 118 (demand network only)
def geo_confirm(c, out, counts):
    import l0_loop as LL
    from sparse_rtdetr.protocol_r18.engine import tensor_digest
    from sparse_rtdetr.research import r28_fovea_cal as FS
    from sparse_rtdetr.research import r28_fovea_geo as FG
    from sparse_rtdetr.research import r28_resample as R
    from sparse_rtdetr.research import scale_baseline as S
    torch = K.torch_mod()
    out = Path(out)
    K.require(out.resolve() == (K.ROUND / "geo-confirm").resolve() and K.ARM == "FG" and K.followup_arm(c) == "FG",
              "this action writes <round>/geo-confirm with R28_ARM=FG")
    K.stage_closed(c, "geo-check", "FG")
    fu = K.followup(c)
    h = K.selected_width(c)["fwhm"]
    n = CONFIRM["windows"]
    if os.environ.get(DEV_CONFIRM):
        K.require(K.DEV, f"{DEV_CONFIRM} is refused outside a dev round")
        n = int(os.environ[DEV_CONFIRM])
    K.require(1 <= n <= CONFIRM["windows"], "confirmation windows")
    t0 = time.perf_counter()
    K.PHASE["phase"] = "geo-confirm:demand cache of the first windows of epoch 118"
    refs, _ = LL.reference_rows(c)
    source = {"kind": "network", "file": K.stage_closed(c, "demand")["weights"]}
    K.require(source["file"] == fu["W_demand"] and source["file"]["sha256"] == CC.PIN_F["W_demand"], "the frozen demand network of arm W")
    ctx = R.Context(source).to("cuda:0")
    data = S.train_dataset(c["data"]["train_core"], S.S15_HW, seed=0)
    cache = CC.collect(c, ctx, data, refs, CC.E118_START, n, counts)
    seconds_cache = time.perf_counter() - t0
    torch.save(cache, out / "cache-epoch118.pt")
    blocks = CC.window_blocks(cache)
    non_positive = sum(b["non_positive"] for b in blocks)
    K.require(non_positive == 0, f"{non_positive} GT boxes of non-positive size in the confirmation windows")
    K.require(all(w["input"]["epoch"] == CONFIRM["epoch"] and w["input"]["augmentation_strong"] is False for w in cache["windows"]),
              "the confirmation windows are the weak-augmentation windows of epoch 118")
    K.PHASE["phase"] = "geo-confirm:geometry"
    w_logs, w_aniso = CC.geometry_w(blocks)
    w_pool = CC._pool(w_logs, blocks, "small")
    updates = [b["update"] for b in blocks]
    w_history = CC.against_ledger(blocks, w_pool, CC.ledger_g_small(K.pinned(fu["W_ledger"]), updates))
    valid, logs, aniso, detail = CC.geometry_h(blocks, h)
    body = {"schema": "r28-fg-confirmation-v1", "contract_sha256": c["contract_sha256"], "task": fu["task"],
            "width": {"fwhm": h, "fwhm_hex": h.hex(), "selection": fu["width_selection"]},
            "validity_rule": FS.CALIBRATION["validity"], "geometry_valid": bool(valid), "detail": detail,
            "confirmation_set": {"epoch": CONFIRM["epoch"], "windows": len(blocks), "occurrences": len(cache["boxes"]),
                                 "first_update": updates[0], "last_update": updates[-1], "cache": K.ref(out / "cache-epoch118.pt"),
                                 "demand_sha256": tensor_digest([("demand", cache["demand"])]),
                                 "every_window_equal_to_the_reference_ledger": True,
                                 "GT": int(sum(int(b["real"].sum()) for b in blocks)), "small_GT": w_pool["n"],
                                 "non_positive_boxes": non_positive, "demand_network": ctx.binding(), "dev_subset": os.environ.get(DEV_CONFIRM)},
            "W": {"reading": CC.reading(blocks, w_logs, w_aniso), "against_the_formal_W_ledger": w_history},
            "FG": None, "FG_over_W_minus_1": None,
            "epoch_61_stored": {"FG_over_W_minus_1": fu["width_selection"]["reading"]["relative_error"],
                                "source": fu["cal_calibration"]},
            "recalibrated": False,
            "note": "validity and description only: the deviation of the mean magnification on another augmentation period is "
                    "recorded as it is and the width is not chosen again from it",
            "seconds": {"demand_cache": seconds_cache}}
    result = {"stage": "geo-confirm", "workload": c["workload"]["geo-confirm"], "geometry_valid": bool(valid), "fwhm": h, "fwhm_hex": h.hex(),
              "windows": len(blocks), "occurrences": len(cache["boxes"]), "small_GT": w_pool["n"]}
    if valid:
        fg = CC.reading(blocks, logs, aniso)
        pool = CC._pool(logs, blocks, "small")
        body["FG"] = {"reading": fg}
        body["FG_over_W_minus_1"] = {k: (None if not fg[k]["n"] else math.expm1(fg[k]["L"] - body["W"]["reading"][k]["L"]))
                                     for k in ("small", "medium", "large")}
        result.update(magnification_pooled={a: {k: v["reading"][k]["pooled_geometric_mean"] for k in ("small", "medium", "large")}
                                            for a, v in (("W", body["W"]), ("FG", body["FG"]))},
                      FG_over_W_minus_1=body["FG_over_W_minus_1"], small_L={"W": w_pool["L"], "FG": pool["L"]},
                      node_slope=[detail["node_slope_min"], detail["node_slope_max"]])
    K.require(w_history["small_counts_equal"] and w_history["max_abs_difference_of_per_window_geometric_mean"] <= CC.TOL["historical_ledger_geomean"],
              f"the recomputed arm W geometry of the confirmation windows differs from its formal ledger: {w_history}")
    K.require(counts["forwards"] == 0 and counts["actual_updates"] == 0 and counts["demand_forward_images"] == len(cache["boxes"]) <= CONFIRM["occurrences"],
              f"the confirmation ran a detector or too many demand forwards: {dict(counts)}")
    K.write_new(out / "CONFIRMATION.json", body)
    result.update(confirmation=K.ref(out / "CONFIRMATION.json"), demand_forward_images=counts["demand_forward_images"], detector_forwards=0,
                  formal_training_allowed=bool(valid),
                  note="an invalid geometry ends the chain here: the map is not repaired and nothing is trained")
    print(json.dumps({"geometry_valid": bool(valid), "fwhm_hex": h.hex(), "FG_over_W_minus_1": body["FG_over_W_minus_1"]}), flush=True)
    return result


# ================================================================ geometry over a ledger (pure)
def schedule_geometry(d):
    """The geometry of a formal FG run: the ledger's per-update sums of log g of the same GT under arm FG and under arm
    W's allocation of the same demand maps, pooled over all updates and per epoch, and the historical caliber (mean
    over updates of the per-update geometric mean).  A description; nothing was fed back."""
    d = Path(d)
    classes, arms = ("small", "medium", "large"), ("FG", "W")
    zero = lambda: {k: {"n": 0, **{a: {"sum_log_g": [], "bins": [0, 0, 0, 0], "ratio": [], "abs_ratio": [], "per_update": []} for a in arms}}
                    for k in classes}
    total, epochs, updates = zero(), {}, 0
    with (d / "UPDATES.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            cal, ep = row["r28"]["cal"], row["input"]["epoch"]
            updates += 1
            for acc in (total, epochs.setdefault(ep, zero())):
                for k in classes:
                    n = cal[k]["n"]
                    acc[k]["n"] += n
                    for a in arms:
                        acc[k][a]["sum_log_g"].append(cal[k][a]["sum_log_g"])
                        acc[k][a]["ratio"].append(cal[k][a]["sum_log_ox_over_oy"])
                        acc[k][a]["abs_ratio"].append(cal[k][a]["sum_abs_log_ox_over_oy"])
                        acc[k][a]["bins"] = [p + q for p, q in zip(acc[k][a]["bins"], cal[k][a]["bins"])]
                        if n:
                            acc[k][a]["per_update"].append(math.exp(cal[k][a]["sum_log_g"] / n))

    def fold(acc):
        out = {}
        for k in classes:
            n = acc[k]["n"]
            cell = {"n": n, "updates_with_objects": len(acc[k]["FG"]["per_update"])}
            for a in arms:
                s = math.fsum(acc[k][a]["sum_log_g"])
                per = acc[k][a]["per_update"]
                cell[a] = {"sum_log_g": s, "L": s / n if n else None, "pooled_geometric_mean": math.exp(s / n) if n else None,
                           "per_update_geometric_mean_mean": statistics.fmean(per) if per else None,
                           "bins": acc[k][a]["bins"], "bin_shares": [b / n for b in acc[k][a]["bins"]] if n else None,
                           "mean_log_ox_over_oy": math.fsum(acc[k][a]["ratio"]) / n if n else None,
                           "mean_abs_log_ox_over_oy": math.fsum(acc[k][a]["abs_ratio"]) / n if n else None}
            cell["FG_over_W_minus_1"] = math.expm1(cell["FG"]["L"] - cell["W"]["L"]) if n else None
            out[k] = cell
        return out

    per_epoch = {ep: fold(acc) for ep, acc in sorted(epochs.items())}
    errors = {ep: v["small"]["FG_over_W_minus_1"] for ep, v in per_epoch.items() if v["small"]["FG_over_W_minus_1"] is not None}
    return {"updates": updates, "overall": fold(total), "bins": ["g<1.0", "1.0<=g<1.2", "1.2<=g<1.4", "g>=1.4"],
            "small_FG_over_W_minus_1_by_epoch": errors,
            "small_FG_over_W_minus_1_range": [min(errors.values()), max(errors.values())] if errors else None,
            "per_epoch": per_epoch,
            "note": "L is the pooled mean log magnification of the training GT of a raw-area class (raw_area is the annotation's "
                    "area on the original image, before augmentation); FG_over_W_minus_1 = exp(L_FG - L_W) - 1; W's geometry is "
                    "computed from the demand maps the F_geo forward already produced (no image rendered, no extra network "
                    "forward); per_update_geometric_mean_mean is the historical caliber of the earlier reports"}


# ================================================================ summary arithmetic (pure)
def summary_fg(c, w_result, values, apl, apm, cost, quoted):
    """values / apl / apm: arm -> stored readings of L0, K, F, FG, W.  Pure arithmetic; no gate."""
    import r28_eval as E
    K.require(set(values) == set(apl) == set(apm) == set(ARMS5), "summary arms")
    diffs = {f"{x}-{y}": E.difference_pp(values[x], values[y]) for x, y in PAIRS}
    primary = {f"W-FG top{b} {m}": diffs["W-FG"][b][m] for b, m in PRIMARY}
    ratios = cost["ratios"]
    return {"arm": ARM, "label": LABEL, "contribution": CONTRIBUTION, "record_only": True,
            "values": {**w_result["values"], "K": values["K"], "F": values["F"], "FG": values["FG"]},
            "value_sources": {"FG": "this round, eval-FG (new)", "F": "F round eval-F (quoted)", "W": "closed W round eval-W (quoted)",
                              "K": "K round eval-K (quoted)", "L0": "R27-002 eval-L0 via the contract reference (quoted)"},
            "primary_W_minus_FG_pp": primary, "differences_pp": diffs,
            "APL6_pp": {f"{x}-{y}": E.apl6(apl[x], apl[y]) for x, y in (("W", "FG"), ("FG", "F"), ("FG", "L0"))},
            "per_class_APL_top600": FC.size_table({a: apl[a] for a in ARMS5}, "APL"),
            "per_class_APM_top600": FC.size_table({a: apm[a] for a in ARMS5}, "APM"),
            "cost_FG": {"model_gpu": {k: ratios["model_gpu"][k] for k in COST_PAIRS},
                        "end_to_end": {k: ratios["end_to_end"][k] for k in COST_PAIRS},
                        "all_ratios": ratios, "ratios_per_pass": cost.get("ratios_per_pass"), "medians": cost["medians"],
                        "stages": cost.get("stages"), "order_executed": cost.get("order_executed"),
                        "source": "cost-FG (L0, K, F, FG, W in one process)"},
            "cost_quoted_from_earlier_rounds": {**quoted, "note": "other processes; quoted, not the same measurement"},
            "W_decision_unchanged": w_result["decision"]}


def _trial_summary(c, cost):
    """summary_fg with the stored W evaluation standing in for FG (wiring only; every W - FG difference is exactly 0)."""
    w_result, ev_w, ev_f, values, apl, apm, quoted = CC._inputs(c)
    if cost is None:
        arms = K.COST_ENDPOINTS["cost-FG"]
        one = {f"{a}/{b}": 1.0 for a in arms for b in arms if a != b}
        cost = {"ratios": {"model_gpu": dict(one), "end_to_end": dict(one)}, "medians": {}}
    trial = summary_fg(c, w_result, {**values, "FG": ev_w["values"]}, {**apl, "FG": apl["W"]}, {**apm, "FG": apm["W"]}, cost, quoted)
    K.require(all(v == 0.0 for v in trial["primary_W_minus_FG_pp"].values()) and trial["APL6_pp"]["W-FG"]["difference_pp"] == 0.0
              and trial["W_decision_unchanged"] == FU.W_DECISION and trial["label"] == LABEL and trial["contribution"] == CONTRIBUTION
              and trial["differences_pp"]["FG-F"] == trial["differences_pp"]["W-F"], "summary arithmetic")
    json.dumps(trial, allow_nan=False)
    return trial


# ================================================================ CUDA wiring of what only runs after training
def geo_wire(c, out, counts):
    import r28_cost as CO
    import r28_eval as E
    from sparse_rtdetr.protocol_r18.engine import tensor_digest
    from sparse_rtdetr.research import r28_fovea_geo as FG
    torch = K.torch_mod()
    out = Path(out)
    K.require(out.resolve() == (K.ROUND / "wire-FG").resolve() and K.ARM == "FG" and K.followup_arm(c) == "FG",
              "this action writes <round>/wire-FG with R28_ARM=FG")
    K.require(K.stage_closed(c, "geo-confirm", "FG")["geometry_valid"] is True, "the fixed width was not confirmed valid")
    source = K.demand_source(c, "FG")
    K.PHASE["phase"] = "wire:model (parent EMA weights behind the arm FG forward)"
    state = torch.load(K.pinned(c["parent"]), map_location="cpu", weights_only=True)["ema"]["module"]
    digest = tensor_digest(state.items())
    model = FG.build_eval_model(state, "cuda:0", "FG", source)
    K.require(FG.is_wrapped(model) and not model.training and model.r28_mapping == FG.mapping(source), "wire model")
    K.PHASE["phase"] = "wire:timing"
    order = K.COST_ENDPOINTS["cost-FG"]
    loaders = {"L0": lambda: CO.load_reference(c, "L0", torch),
               **{a: (lambda a=a: CO.load_arm(c, a, torch)) for a in ("W", "K", "F")},
               "FG": lambda: (model, {"weights": "e060 parent EMA (not trained under F_geo; wiring only)", "demand": model.r28_ctx.binding(),
                                      "mapping": model.r28_mapping})}
    timing = CO._run(c, counts, loaders, order, (order,) * WIRE["cost_passes"], warmup=WIRE["cost_warmup"],
                     batches_per_pass=WIRE["cost_batches_per_pass"], stages=True)
    table = timing["stages"]["table"]
    K.require(set(table) == set(order) and all(abs(sum(table[a]["stage_ms_median_pass"].values()) - table[a]["model_gpu_ms_median_pass"])
                                                <= 1e-6 * max(1.0, table[a]["model_gpu_ms_median_pass"]) for a in order)
              and all(table[a]["stage_ms_median_pass"][s] > 0 for a in ("W", "F", "FG") for s in ("demand", "map", "render", "detector", "unwarp"))
              and all(table[a]["stage_ms_median_pass"]["detector"] > 0 for a in order)
              and "probe" not in vars(model.r28_ctx), ("stage readings", table))
    K.PHASE["phase"] = "wire:evaluation batch"
    sub = Counter(forwards=0)
    data, preds, _ = E.infer(c, model, sub, limit=(0, WIRE["eval_batches"]))
    counts.update(sub)
    K.require(all(len(preds[b]) == WIRE["eval_batches"] * K.EVAL_BATCH * b for b in K.BUDGETS)
              and all(math.isfinite(v) for p in preds[600] for v in (p["score"], *p["bbox_xyxy"]))
              and tensor_digest(model.state_dict().items()) == digest, "wire evaluation batch")
    K.PHASE["phase"] = "wire:summary structure"
    trial = _trial_summary(c, timing)
    K.require(counts["forwards"] == c["workload"]["wire"]["forwards"] and counts["actual_updates"] == 0
              and not counts.get("score_budget_calls"), f"wire counts {dict(counts)}")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {"stage": "wire-FG", "workload": c["workload"]["wire"], "weights": "e060 parent EMA behind the arm FG forward",
            "weights_sha256": digest, "mapping": FG.mapping(source),
            "purpose": "wiring of code that otherwise runs only after the formal training: the evaluation batch path of arm FG and the "
                       "staged five-arm timing with the stored K and F endpoints; these readings are not results (F_geo is not "
                       "trained here, the protocol is shortened)",
            "timing": {k: timing[k] for k in ("ratios", "medians", "stages", "protocol", "forwards_per_endpoint", "order_executed", "weights_sha256")},
            "evaluation_batch": {"images": WIRE["eval_batches"] * K.EVAL_BATCH, "records_top600": len(preds[600]), "scored": False},
            "summary_structure": {"cost_keys": sorted(trial["cost_FG"]), "stage_table_arms": sorted(trial["cost_FG"]["stages"]["table"]),
                                  "pairs": list(trial["differences_pp"])},
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "scorer_calls": 0, "formal_state_usable": False}


# ================================================================ refusals of foreign states
def state_refusals(c, e, state, scope):
    """Cross-arm / cross-width / cross-demand / cross-scope uses of states that must be refused.  `e` is an arm FG
    engine and `state` a state it wrote in `scope`; nothing may change the engine."""
    import l0_checks_state as H
    import r28_loop as LP
    from sparse_rtdetr.research import r28_resample as R
    fu = K.followup(c)
    other = {**c, "identity_sha256": {**c["identity_sha256"], "FG": "0" * 64}}
    as_fs = {**c, "identity_sha256": {**c["identity_sha256"], "FS": c["identity_sha256"]["FG"]}}
    real_ctx, real_map = e.r28_ctx, e.r28_mapping
    synthetic = R.Context({"kind": "synthetic", "seed": 0})
    vars(synthetic)["r28_fwhm"] = real_map["fwhm"]

    def swapped(**attrs):
        def run():
            saved = {k: getattr(e, k) for k in attrs}
            for k, v in attrs.items():
                setattr(e, k, v)
            try:
                K.load(c, e, state, scope)
            finally:
                for k, v in saved.items():
                    setattr(e, k, v)
        return run

    narrow = {**real_map, "fwhm": 4.0, "fwhm_hex": (4.0).hex()}
    locked = {"fwhm": real_map["fwhm"], "fwhm_hex": real_map["fwhm_hex"], "calibration": real_map["selection"]["calibration"]}
    cases = [("arm FG state read as arm F", lambda: K.read_side(c, state, scope, "F")),
             ("arm FG state read as arm FS", lambda: K.read_side(as_fs, state, scope, "FS")),
             ("arm FG state read as arm W", lambda: K.read_side(c, state, scope, "W")),
             ("arm FG state read as arm K", lambda: K.read_side(c, state, scope, "K")),
             ("arm FG state read under another mapping identity", lambda: K.read_side(other, state, scope, "FG")),
             ("arm FG state loaded into an engine bound to another width (4)", swapped(r28_mapping=narrow)),
             ("arm FG state loaded into an engine bound to an arm FS style calibration lock", swapped(r28_mapping=locked)),
             ("arm FG state loaded into an engine bound to another selection file",
              swapped(r28_mapping={**real_map, "selection": {**real_map["selection"], "calibration": {**real_map["selection"]["calibration"], "sha256": "0" * 64}}})),
             ("arm FG state loaded into an engine with another demand source", swapped(r28_ctx=synthetic)),
             ("arm FS acceptance state of the 006 dev round loaded into the arm FG engine",
              lambda: K.load(c, e, K.pinned(fu["dev_FS_state"]), "pipeline-acceptance")),
             ("arm FS acceptance state of the 006 dev round read as arm FG", lambda: K.read_side(c, K.pinned(fu["dev_FS_state"]), "pipeline-acceptance", "FG")),
             ("F formal e120 state loaded into the arm FG engine", lambda: K.load(c, e, fu["F_endpoint"]["path"], "formal")),
             ("W formal e120 state loaded into the arm FG engine", lambda: K.load(c, e, fu["W_endpoint"]["path"], "formal")),
             ("K formal e120 state loaded into the arm FG engine", lambda: K.load(c, e, fu["K_endpoint"]["path"], "formal")),
             ("L0 formal e120 state loaded into the arm FG engine", lambda: K.load(c, e, c["L0_endpoint"]["path"], "formal")),
             ("formal training of arm F in this round", lambda: LP.guard(c, K.ROUND / "formal-F", None, "F")),
             ("formal training of arm FS in this round", lambda: LP.guard(c, K.ROUND / "formal-FS", None, "FS")),
             ("formal training of arm W in this round", lambda: LP.guard(c, K.ROUND / "formal-W", None, "W"))]
    if scope != "formal":
        cases += [(f"{scope} state read in formal scope", lambda: K.read_side(c, state, "formal", "FG")),
                  (f"{scope} state loaded as a formal state", lambda: K.load(c, e, state, "formal"))]
    out = [H.refused(label, fn) for label, fn in cases]
    K.require(e.r28_ctx is real_ctx and e.r28_mapping is real_map, "the engine's demand source / mapping was not restored")
    return out


# ================================================================ acceptance of the actual width (CUDA, 4 + 4 + 2 updates)
def geo_accept(c, out, counts):
    import l0_loop as LL
    import r28_loop as LP
    import r28_pipeline as PL
    from s15_formal_train import _result_body
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    from sparse_rtdetr.research import r28_fovea_geo as FG
    from sparse_rtdetr.research import scale_baseline as S
    torch = K.torch_mod()
    out = Path(out)
    K.require(out.resolve() == (K.ROUND / "accept-FG").resolve() and K.ARM == "FG" and K.followup_arm(c) == "FG",
              "this action writes <round>/accept-FG with R28_ARM=FG")
    for name in ("geo-check", "geo-confirm", "wire-FG"):
        K.stage_closed(c, name, "FG")
    K.require(K.stage_closed(c, "geo-confirm", "FG")["geometry_valid"] is True, "the fixed width was not confirmed valid")
    fu = K.followup(c)
    source = K.demand_source(c, "FG")
    steps, save_at = ACCEPT["steps"], ACCEPT["save_at"]
    refs, _ = LL.reference_rows(c)
    check = CC.reference_check(refs)
    data = S.train_dataset(c["data"]["train_core"], S.S15_HW, seed=0)
    p0 = K.PARENT_UPDATE
    digester = CC.Digester()
    state_file = out / f"overlap-u{save_at:02d}.pth"
    info, refusals = {}, []

    def run(name, mode, first, last, *, load=None, save_after=None, after=None):
        K.PHASE["phase"] = f"accept:{name} construct"
        e, checks = FG.create(c["parent"]["path"], "cuda:0", "FG", source)
        K.register_counts(e, counts)
        obs = LP.install_observer(e)
        loaded = None if load is None else K.load(c, e, load, SCOPE)
        K.require(e.completed_updates == first and e.r28_mapping == FG.mapping(source) and FG.is_wrapped(e.model) and FG.is_wrapped(e.ema.module),
                  f"{name}: start update / mapping")
        rows, saved, window = [], None, None
        torch.cuda.reset_peak_memory_stats()
        with PL.MemorySampler(0.5) as memory:
            opened = mark = time.perf_counter()
            with PL.formal_stream(mode, data, first, last, check, fine=True) as stream:
                for item in stream:
                    x, t, inp = item.x, item.t, item.inp
                    K.PHASE["phase"] = f"accept:{name} update {e.completed_updates + 1}"
                    K.require(inp["update"] == e.completed_updates + 1, f"{name}: window {inp['update']} is not the next update")
                    bad = LL.input_differences(inp, refs[inp["update"]]["input"])
                    K.require(not bad, f"{name}: input record differs from R27 formal-P before update {inp['update']}: {bad}")
                    if window is None:
                        window = PL.window_bytes(x, t)
                    called = time.perf_counter()
                    res = e.train_window(x, t, capture=True)
                    train_call = time.perf_counter() - called
                    K.require(res["update"] == inp["update"] and res["finite"] and e.ema.updates == e.completed_updates, f"{name}: update clock")
                    reading_ = obs.update_row(t)
                    a0 = time.perf_counter()
                    digester.submit((name, res["update"]), S.capture(e))
                    capture_s = time.perf_counter() - a0
                    extra = {}
                    if save_after is not None and e.completed_updates == save_after:
                        s0 = time.perf_counter()
                        prepared = stream.wait_ready()
                        K.require(prepared == 1, f"{name}: no prepared window at the save ({prepared})")
                        s1 = time.perf_counter()
                        saved = K.save(c, e, state_file, SCOPE)
                        K.require(stream.ready() == 1, f"{name}: the prepared window was consumed during the save")
                        extra = {"saved_with_prepared_windows": prepared, "wait_for_prepared_s": s1 - s0, "save_s": time.perf_counter() - s1}
                    now = time.perf_counter()
                    rows.append({"science": {"input": inp, "gt_count": int(sum(len(v["labels"]) for v in t)), "result": _result_body(res),
                                             "observer": reading_},
                                 "engineering": {"cycle_s": now - mark, "wait_s": item.timing["wait_s"], "train_call_s": train_call,
                                                 "train_s": res["elapsed_seconds"], "acceptance_capture_s": capture_s,
                                                 "input": {k: v for k, v in item.timing.items()}, "prepared_after": stream.ready(), **extra}})
                    mark = now
                    del item, x, t
                account = stream.account()
            closed = time.perf_counter()
        K.require(e.completed_updates == last and len(rows) == last - first, f"{name}: update count")
        if after is not None:
            after(e)
        info[name] = {"mode": mode, "first_update": first + 1, "last_update": last, "updates": len(rows), "account": account,
                      "resumed_from": None if loaded is None else loaded["file"], "saved": saved,
                      "wall_seconds_stream": closed - opened, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                      "memory": {"whole_path": memory.summary(), "plan": PL.memory_plan(window, mode)},
                      "create_checks": {k: checks[k] for k in ("arm", "identity_sha256", "demand", "mapping", "new_parameters")}}
        del e, obs
        gc.collect()
        torch.cuda.empty_cache()
        return rows

    def refuse(e):
        K.PHASE["phase"] = "accept:state refusals"
        before = S.capture(e)
        refusals.extend(state_refusals(c, e, state_file, SCOPE))
        K.require(ck.exact_equal(S.capture(e), before), "a refused action changed the engine")

    serial = run("serial", "serial", p0, p0 + steps)
    overlap = run("overlap", "overlap", p0, p0 + steps, save_after=p0 + save_at)
    resumed = run("resumed", "overlap", p0 + save_at, p0 + steps, load=state_file, after=refuse)
    K.PHASE["phase"] = "accept:state digests"
    digests = digester.finish()
    for name, rows in (("serial", serial), ("overlap", overlap), ("resumed", resumed)):
        for r in rows:
            r["science"]["state"] = digests[(name, r["science"]["input"]["update"])]
        with (out / f"STEPS-{name}.jsonl").open("x") as f:
            for r in rows:
                f.write(json.dumps(r, separators=(",", ":"), allow_nan=False) + "\n")
    K.PHASE["phase"] = "accept:comparison"
    sci = lambda rows: [r["science"] for r in rows]
    d_so = CC.differences(sci(serial), sci(overlap))
    d_rs = CC.differences(sci(serial)[save_at:], sci(resumed))
    d_ro = CC.differences(sci(overlap)[save_at:], sci(resumed))
    equal, restore = not d_so, not d_rs and not d_ro
    K.require(info["overlap"]["saved"] is not None and overlap[save_at - 1]["engineering"]["saved_with_prepared_windows"] == 1
              and info["overlap"]["account"]["max_prepared_ahead"] == 1 and info["resumed"]["account"]["max_prepared_ahead"] == 1
              and info["serial"]["account"]["max_prepared_ahead"] == 0, "prepared-window accounting of the three paths")
    want = c["workload"]["accept"]["updates"]
    K.require(want == steps * 2 + (steps - save_at) == 10 and counts["actual_updates"] == want
              and counts["forwards"] == counts["forward_attempts"] == 2 * want == 20, f"acceptance counts {dict(counts)}")
    observers = [r["science"]["observer"] for r in serial]
    K.require(all(o["identity_images"] == 0 and o["fovea"]["clamped_interior_nodes"] == 0 and o["fovea"]["taps_max"] >= 2
                  and set(o["cal"]["small"]) == {"n", "FG", "W"} for o in observers), f"acceptance observer rows: {observers[0]}")
    K.PHASE["phase"] = "accept:ledger readers"
    trial_dir = out / "trial-ledger"
    trial_dir.mkdir()
    with (trial_dir / "UPDATES.jsonl").open("x") as f:
        for r in serial:
            f.write(json.dumps({"input": r["science"]["input"], "update_seconds": r["engineering"]["train_s"], "r28": r["science"]["observer"]},
                               separators=(",", ":"), allow_nan=False) + "\n")
    folded, geometry = FC.formal_observations(trial_dir), schedule_geometry(trial_dir)
    K.require(folded["updates"] == steps and folded["clamped_interior_nodes"] == 0 and geometry["updates"] == steps
              and geometry["overall"]["small"]["n"] == sum(o["cal"]["small"]["n"] for o in observers) == sum(o["n_small"] for o in observers),
              f"folded acceptance readings: {folded} {geometry['overall']['small']}")
    equivalence = {"schema": "r28-fg-acceptance-equivalence-v1", "contract_sha256": c["contract_sha256"], "mapping": FG.mapping(source),
                   "scope": SCOPE, "paths": info,
                   "steps": {"serial": [p0 + 1, p0 + steps], "overlap": [p0 + 1, p0 + steps], "resumed": [p0 + save_at + 1, p0 + steps]},
                   "saved_after_update": p0 + save_at, "prepared_windows_at_the_save": 1,
                   "compared": CC.COMPARED, "excluded": CC.EXCLUDED,
                   "serial_vs_overlap": {"steps": steps, "differences": d_so, "equal": not d_so},
                   "serial_vs_resumed": {"steps": steps - save_at, "differences": d_rs, "equal": not d_rs},
                   "overlap_vs_resumed": {"steps": steps - save_at, "differences": d_ro, "equal": not d_ro},
                   "restore": "K.load compares the complete restored state with the saved one (exact) before the replay",
                   "final_state_digests": {n: r[-1]["science"]["state"] for n, r in (("serial", serial), ("overlap", overlap), ("resumed", resumed))},
                   "steps_files": {n: K.ref(out / f"STEPS-{n}.jsonl") for n in ("serial", "overlap", "resumed")},
                   "updates": want, "complete_detector_forwards": 2 * want,
                   "inherited_general_evidence": fu["dev_pipeline"]}
    K.write_new(out / "ACCEPT_EQUIVALENCE.json", equivalence)
    K.require(equal and restore, f"acceptance of the actual width failed: serial/overlap {d_so[:5]}, serial/resumed {d_rs[:5]}, overlap/resumed {d_ro[:5]}")
    eng = lambda rows: [{"update": r["science"]["input"]["update"], **{k: r["engineering"].get(k) for k in
                                                                       ("cycle_s", "wait_s", "train_call_s", "train_s", "acceptance_capture_s",
                                                                        "wait_for_prepared_s", "save_s")}} for r in rows]
    decision = {"overlap_enabled": True, "equivalence": True, "restore": True, "formal_input_pipeline": "overlap",
                "basis": c["input_pipeline"]["decision"], "timing_used": False}
    timing = {"schema": "r28-fg-acceptance-timing-v1", "contract_sha256": c["contract_sha256"],
              "per_update": {"serial": eng(serial), "overlap": eng(overlap), "resumed": eng(resumed)},
              "wall_seconds_stream": {n: info[n]["wall_seconds_stream"] for n in info},
              "memory": {n: info[n]["memory"] for n in info}, "peak_cuda_allocated_bytes": {n: info[n]["peak_cuda_allocated_bytes"] for n in info},
              "checkpoint": {"save_s": overlap[save_at - 1]["engineering"]["save_s"],
                             "wait_for_prepared_s": overlap[save_at - 1]["engineering"]["wait_for_prepared_s"],
                             "state_bytes": info["overlap"]["saved"]["file"]["bytes"]},
              "note": "raw wall-clock segments of ten updates with capture=True digests, a full state capture per update and the "
                      "start of the loader workers; recorded for the engineering account only: not a throughput measurement, not "
                      "used in the decision and not a method or inference speed-up"}
    K.write_new(out / "ACCEPT_TIMING.json", timing)
    print(json.dumps({"overlap_enabled": True, "serial_vs_overlap_equal": True, "restore_replay_equal": True, "refusals": len(refusals)}), flush=True)
    return {"stage": "accept-FG", "workload": c["workload"]["accept"], "mapping": FG.mapping(source), "overlap_enabled": True,
            "decision": decision, "equivalence": K.ref(out / "ACCEPT_EQUIVALENCE.json"), "timing": K.ref(out / "ACCEPT_TIMING.json"),
            "serial_vs_overlap_equal": True, "restore_replay_equal": True, "compared_steps": {"serial_vs_overlap": steps, "replayed": steps - save_at},
            "saved_state": info["overlap"]["saved"], "updates": want, "complete_detector_forwards": 2 * want, "refused": refusals,
            "ledger_readings_interface": {"rows": steps, "folded": folded, "geometry_overall": geometry["overall"],
                                          "note": "the four serial acceptance rows through the summary readers; an interface check"},
            "first_update_observer": observers[0],
            "peak_allocated_bytes": max(v["peak_cuda_allocated_bytes"] for v in info.values()), "formal_state_usable": False}


# ================================================================ summary-FG
def summarize(c, out, counts):
    import r28_eval as E
    from sparse_rtdetr.research import r28_fovea_geo as FG
    fu = K.followup(c)
    K.require(fu is not None and K.followup_arm(c) == "FG" and K.ARM == "FG", "the F_geo round only summarises arm FG")
    K.PHASE["phase"] = "summarize/read"
    out = E.round_stage(out, "summary-FG")
    w_result, ev_w, ev_f, values, apl, apm, quoted = CC._inputs(c)
    ev, cost = K.stage_closed(c, "eval-FG", "FG"), K.stage_closed(c, "cost-FG", "FG")
    lock = K.selected_width(c)
    confirm, accept = K.stage_closed(c, "geo-confirm", "FG"), K.stage_closed(c, "accept-FG", "FG")
    ws = cost["weights_sha256"]
    K.require(ws["FG"] == ev["weights_sha256"] and ws["W"] == ev_w["weights_sha256"] == fu["W_weights_sha256"]
              and ws["F"] == ev_f["weights_sha256"] and ws["K"] == fu["K_weights_sha256"] and ws["L0"] == c["L0_weights_sha256"],
              "the cost-FG stage timed other models than the evaluated ones")
    K.require(ev["demand"]["file"] == fu["W_demand"] and ev["demand"] == ev_w["demand"] == ev_f["demand"]
              and ev["mapping"] == {"fwhm": lock["fwhm"], "fwhm_hex": FG.WIDTH_HEX, "selection": lock["selection"]},
              "arm FG was not evaluated with arm W's demand network and the fixed width")
    for b in K.BUDGETS:
        K.pinned(ev["metrics"][str(b)]), K.pinned(ev["predictions"][str(b)])
    K.PHASE["phase"] = "summarize/compute"
    res = summary_fg(c, w_result, {**values, "FG": ev["values"]}, {**apl, "FG": ev["per_class_APL_top600"]},
                     {**apm, "FG": ev["per_class_APM_top600"]}, cost, quoted)
    tables = FC.tensor_tables(K.pinned(ev["cocoeval_top600_tensors"]))
    f600 = ev["values"]["600"]
    consistency = {k: (None if tables["macro"][k] is None else tables["macro"][k] - f600[k]) for k in tables["macro"]}
    formal_dir = K.formal_dir(c, "FG")
    formal = K.read_json(formal_dir / "CLOSED.json")
    cal = K.read_json(K.pinned(lock["selection"]["calibration"]))
    confirmation = K.read_json(K.pinned(confirm["confirmation"]))
    geometry = schedule_geometry(formal_dir)
    res.update(status="PASS", registration=fu["registration"], authorization_basis=c["authorization_basis"], task=fu["task"],
               review=fu["review"], contract_sha256=c["contract_sha256"], source_commit=c["source_commit"],
               official=fu["official"], fovea_geo_definition=c["fovea_geo_definition"], identity_sha256_FG=c["identity_sha256"]["FG"],
               width={"fwhm": lock["fwhm"], "fwhm_hex": FG.WIDTH_HEX, "selection": lock["selection"]},
               calibration_006={"file": lock["selection"]["calibration"], "status": cal["status"], "h_star": cal["h_star"],
                                "tolerance": cal["tolerance"], "L_W": cal["L_W"], "closest_if_unmatched": cal["closest_if_unmatched"],
                                "widths_read": len(cal["readings"]), "W": cal["W"], "F_width_4": cal["F_width_4"],
                                "note": "kept as it was: unmatched, no F*; the fixed width is its closest reading, about 4.98 % below "
                                        "arm W on epoch 61"},
               confirmation_epoch_118={"file": confirm["confirmation"], "geometry_valid": confirmation["geometry_valid"],
                                       "W": confirmation["W"]["reading"], "FG": confirmation["FG"]["reading"],
                                       "FG_over_W_minus_1": confirmation["FG_over_W_minus_1"], "detail": confirmation["detail"]},
               W_closed_round={"contract_sha256": fu["prior_contract_sha256"], "result": fu["prior_W_result"], "decision": w_result["decision"],
                               "M_W_minus_L0": w_result["M_W_minus_L0"], "APL6": w_result["APL6"], "C": w_result["C"],
                               "note": "quoted unchanged from the closed round; not recomputed, not overwritten"},
               F_round={"contract_sha256": fu["F_contract_sha256"], "result": fu["F_result"], "note": "quoted unchanged from the F round"},
               FG_per_class_top600={**tables, "source": ev["cocoeval_top600_tensors"], "macro_minus_scored_value": consistency,
                                    "note": "read from the tensors kept by the single per-class COCOeval of eval-FG; no rescoring"},
               FG_demand=ev["demand"], FG_mapping=ev["mapping"], FG_endpoint=ev["endpoint"], FG_weights_sha256=ev["weights_sha256"],
               FG_training_observations=FC.formal_observations(formal_dir),
               FG_geometry_over_the_schedule=geometry,
               input_pipeline={"acceptance": {"equivalence": accept["equivalence"], "timing": accept["timing"], "decision": accept["decision"]},
                               "inherited": fu["dev_pipeline"],
                               "formal": {"mode": formal["plan"]["input_pipeline"], "account": formal["input_pipeline"],
                                          "loop_wall_seconds": formal["loop_wall_seconds"], "update_seconds_sum": formal["update_seconds_sum"],
                                          "seconds_per_update": formal["loop_wall_seconds"] / formal["updates"],
                                          "stage_elapsed_seconds": formal["elapsed_seconds"],
                                          "timing_file": K.ref(formal_dir / formal["timing_file"])},
                               "reference_serial_F": {"seconds_per_update": 40645.923253592104 / 18240, "source": "F round formal-F/CLOSED.json"},
                               "note": STATEMENTS["cost"]},
               stages=E._stage_account(c, ["geo-check", "geo-confirm", "wire-FG", "accept-FG", formal_dir.name, "eval-FG", "cost-FG"]),
               other_directories=E._other_directories(), statements=STATEMENTS)
    K.require(formal["plan"]["mapping"] == ev["mapping"] == accept["mapping"] and confirm["fwhm_hex"] == FG.WIDTH_HEX,
              "one width from the selection to the endpoint")
    K.require(counts["forwards"] == 0 and counts["actual_updates"] == 0 and not counts.get("score_budget_calls"),
              "summary must not run a model or a scorer")
    K.write_new(out / "RESULT.json", res)
    p = res["primary_W_minus_FG_pp"]
    small = geometry["overall"]["small"]
    return {"result": K.ref(out / "RESULT.json"), "registration": fu["registration"], "label": res["label"],
            "contribution": res["contribution"], "record_only": True, "fwhm": lock["fwhm"],
            "W_minus_FG_APS600_pp": p["W-FG top600 APS"], "W_minus_FG_AP75_small600_pp": p["W-FG top600 AP75_small"],
            "FG_minus_F_APS600_pp": res["differences_pp"]["FG-F"]["600"]["APS"], "FG_minus_L0_APS600_pp": res["differences_pp"]["FG-L0"]["600"]["APS"],
            "W_decision_unchanged": w_result["decision"], "model_gpu_FG_over_L0": res["cost_FG"]["model_gpu"]["FG/L0"],
            "model_gpu_W_over_FG": res["cost_FG"]["model_gpu"]["W/FG"],
            "small_FG_over_W_minus_1_over_the_schedule": small["FG_over_W_minus_1"],
            "small_pooled_magnification_over_the_schedule": {a: small[a]["pooled_geometric_mean"] for a in ("FG", "W")}}


# ================================================================ stored W / K / F ledgers: per-update magnification means
LEDGER_FIELDS = ("g_small", "g_medium", "g_large")


def ledger_fold(path):
    """One stored UPDATES.jsonl: for each raw-area class the arithmetic mean over updates of the per-update geometric
    mean magnification (ledger key r28.g_*), the number of updates with a value and without one."""
    acc = {k: [] for k in LEDGER_FIELDS}
    missing = dict.fromkeys(LEDGER_FIELDS, 0)
    rows, first, last, epochs, n_small = 0, None, None, set(), 0
    with Path(path).open() as f:
        for line in f:
            row = json.loads(line)
            r, u = row["r28"], row["input"]["update"]
            rows += 1
            first, last = (u if first is None else first), u
            epochs.add(row["input"]["epoch"])
            n_small += r["n_small"]
            for k in LEDGER_FIELDS:
                if r[k] is None:
                    missing[k] += 1
                else:
                    acc[k].append(r[k])
    return {"rows": rows, "first_update": first, "last_update": last, "epochs": [min(epochs), max(epochs)], "epoch_count": len(epochs),
            "small_GT_observations": n_small,
            "fields": {k: {"field": f"r28.{k}", "mean": statistics.fmean(v) if v else None, "valid_updates": len(v), "missing_updates": missing[k],
                           "min": min(v) if v else None, "max": max(v) if v else None} for k, v in acc.items()}}


def ledger_means(c, out, counts):
    out = Path(out)
    K.require(out.resolve() == (K.ROUND / "ledger-means").resolve() and K.followup_arm(c) == "FG",
              "this action writes <round>/ledger-means in the F_geo round")
    fu = K.followup(c)
    K.PHASE["phase"] = "ledger-means:references"
    k_side_path = K.side_of(K.pinned(fu["K_endpoint"]))
    k_side = K.read_json(k_side_path)
    K.require(k_side["file"] == fu["K_endpoint"] and k_side["contract_sha256"] == fu["K_contract_sha256"] and k_side["arm"] == "K"
              and k_side["global_update"] == K.ENDPOINT, "K endpoint side-car")
    ledgers = {"W": fu["W_ledger"], "K": k_side["audit"]["ledger"], "F": fu["F_ledger"]}
    arms = {}
    for arm, ref in ledgers.items():
        K.PHASE["phase"] = f"ledger-means:{arm}"
        fold = ledger_fold(K.pinned(ref))
        K.require(fold["rows"] == K.ENDPOINT - K.PARENT_UPDATE and fold["first_update"] == K.PARENT_UPDATE + 1 and fold["last_update"] == K.ENDPOINT
                  and fold["epoch_count"] == 60, f"stored {arm} ledger span: {fold['rows']} rows")
        arms[arm] = {"ledger": ref, **fold}
    K.require(counts["forwards"] == 0 and counts["actual_updates"] == 0, "ledger-means must not run a model")
    body = {"schema": "r28-ledger-means-v1", "contract_sha256": c["contract_sha256"], "arms": arms,
            "K_endpoint_side": K.ref(k_side_path),
            "aggregation": "mean_u(g_class(u)) over the updates u = 18241..36480 that have a value, where g_class(u) = "
                           "exp(mean over the GT of that raw-area class in the window of update u of log sqrt(ox oy)); an update "
                           "without a GT of the class stores null and is counted as missing",
            "size_class": "raw_area (annotation area on the original image, before augmentation): small <= 1024, medium 1024..9216, "
                          "large >= 9216, end points inclusive",
            "note": "read from the formal ledgers of the closed rounds (all 60 epochs); not from the epoch-61 calibration cache; "
                    "nothing is rescored, no model runs"}
    K.write_new(out / "LEDGER_MEANS.json", body)
    return {"stage": "ledger-means", "ledger_means": K.ref(out / "LEDGER_MEANS.json"),
            "g_small_mean": {a: v["fields"]["g_small"]["mean"] for a, v in arms.items()},
            "g_small_valid_missing": {a: [v["fields"]["g_small"]["valid_updates"], v["fields"]["g_small"]["missing_updates"]] for a, v in arms.items()},
            "ledgers_sha256": {a: v["ledger"]["sha256"] for a, v in arms.items()}}
