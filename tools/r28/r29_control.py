"""R29: query-wise un-zoomed decoder read on the fixed strong F_geo input (TASK_R29_001, DECISION_R29_001).

An exploratory two-arm method experiment designed after the 009 result.  Nothing of the W, K, F, 006 or F_geo rounds is
rerun or written; their decisions, labels and readings stay.

Arms (internal name / reported name): FG / N (the certified F_geo of task 009, reused: endpoint, metrics, receipts),
QC / C (dense: the projected value tensor is un-zoomed to a source-uniform grid, then read), QP / P (direct: the
source-frame sampling points are mapped onto the warped memory and read).  src/sparse_rtdetr/research/r29_unzoom.py.

  prepare      contract of the independent round: authorization basis, task / decision files, read-only references into
               the F_geo round (and through it into the W, K, F and 006 rounds), the release difference against the
               009 release, the fixed rules of section 6.
  r29_check    CPU: unit tests; identities, references and refusals; the release difference file by file and function
               by function; the real e060 detector on a real window (H fixture, two images): parameter / optimizer /
               EMA identity with the parent, the read of both arms, audit readings, save / restore replay, EMA copy,
               evaluation read without targets, cross-arm state refusals.
  r29_wire     CUDA wiring of code that otherwise runs only after training: one evaluation batch of QP and QC and a
               shortened N / C / P staged timing; nothing is scored.  Listed apart from the planned workload.
  r29_ncheck   CUDA: the stored N endpoint through the present loading path on the first two evaluation batches (8
               images), compared field by field with the stored 009 predictions; N is not scored again.
  r29_accept   CUDA joint acceptance from the e060 parent (8 x 2, BF16): N 2 updates against the stored accept-FG
               receipts (bit for bit); QC and QP 4 updates each on the overlapped input path with a full state saved
               after update 2, then updates 3-4 replayed from that state.  14 updates, 28 detector forwards.  The read
               readings of every acceptance forward are recorded; they are not a gate, except that a read which does
               not move any sampling position closes the stage without opening the formal runs.
  summarize    summary-R29: P - N, C - N, P - C, the same-process costs, the fixed rules of section 6.
Formal training, evaluation and timing are r28_loop.formal, r28_eval.evaluate and r28_cost.cost with R28_ARM=QP / QC.

Release 002 (engineering only, user instruction of 2026-11-17): the window delivery of r29_input.  The science tree is
byte-identical to release-001 of the first R29 round; the contract equals that round's contract outside the named
engineering keys; the joint acceptance compares every update of both arms with that round's stored receipts.
"""
import gc
import json
import math
import os
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
import r28_geo_control as GC
import r29_input as RI

REGISTRATION = ("R29 exploratory query-wise un-zoomed decoder read on the fixed strong F_geo input: two arms, single seed, "
                "designed after the 009 result")
ARMS = K.R29_ARMS                      # ("QP", "QC")
SHORT = {"FG": "N", "QC": "C", "QP": "P"}
ORDER = ("QP", "QC")                   # training order of task section 4: P, then C
FG_ROUND = K.LD_ROOT / "r28-strong-fovea-control-20261116-001"
TASK = ("TASK_R29_001.md", "36adc15921a9a3cd54dd74a064bf64f739b964e46325d1cb87fbadb1f176af4f")
DECISION = ("DECISION_R29_001.md", "2e6380a5c9ba85f3791b136343a10630e2f42293d917d5c822f0a421513c2680")
PIN_FG = {  # TASK_R29_001 section 1 and the F_geo round's own receipts
    "commit": "e1c904f21ad08607871db401be90e2eff47c0465",
    "manifest": "327f596171f50f77e3dad6ca9d50f3be19a6bb0306c1937733434ca7d25310ca",
    "contract_file": "dc39a82b9aab1c0a4a88e19a72a8d2aa73da71e244c6d7e2e959cc22f30e5d1a",
    "contract": "0ae1eca4b3e7231bfad70ab6e7aebb93622cec2143975e41bae273869ed85553",
    "parent": "a2b6412e3c85de7592829f24aff09e355fdd11cb53798bc498bd04b882e1bf8a",
    "demand": "c9ffc7505e47adbcc45d70949c3d87371c6f3a7795f1aaaab886d98b15ee9e5a",
    "endpoint": "f0632fc8ed71e593e5eaab1bf233aafed1e9ec57dcfe44c893721286cdffeb57",
    "weights": "cd62f314f3713d83fd742ff511600902fd990cf48165d9b0a90c7fec4359e219",
    "ledger": "cae8b212e896e504b789a1fec63178819771b641c7d01c5afba47ac4678784f0",
    "accept_closed": "6237f71f0dac2d28aeb26c4a8c69811481222cf5d45f34f00211c68910ab2bdb",
    "accept_steps_serial": "9b64851800ada74b426b294bb58df74e39b7ab7da6a6ac85b87e8a3336df64ea",
    "eval_closed": "9f4dfead07da37cb7520e5978c9c2b32067af7b25df352e2f224a231cf48052d",
    "formal_closed": "f838968fe3ce83f826b85cb4fe1b0c86d6bb3bccc2d9062c2539c6354a7a0506",
}
FG_FILES = {"contract_file": "CONTRACT.json", "manifest": "release-001/MANIFEST.json", "ledger": "formal-FG/UPDATES.jsonl",
            "accept_closed": "accept-FG/CLOSED.json", "accept_steps_serial": "accept-FG/STEPS-serial.jsonl",
            "eval_closed": "eval-FG/CLOSED.json", "formal_closed": "formal-FG/CLOSED.json",
            "endpoint": "formal-FG/checkpoint-epoch-120.pth"}
INHERITED_FG = GC.OWN_STAGES
OWN_STAGES = ("r29-check", "wire-R29", "ncheck-FG", "accept-R29", "formal-QP", "formal-QC", "eval-QP", "eval-QC", "cost-R29", "summary-R29")
ALLOWED_CHANGED = ("tools/r28/r28_common.py", "tools/r28/r28_loop.py", "tools/r28/r28_eval.py", "tools/r28/r28_cost.py",
                   "tools/r28/r28_run.py", "tests/test_r28_cal_control.py", "tests/test_r28_geo_control.py")
ALLOWED_ADDED = ("src/sparse_rtdetr/research/r29_unzoom.py", "tests/test_r29_unzoom.py", "tests/test_r29_control.py",
                 "tools/r28/r29_control.py", "tools/r28/r29_chain.sh", "tools/r28/r29_input.py", "tests/test_r29_input.py")
SCIENCE_ADDED = ("src/sparse_rtdetr/research/r29_unzoom.py",)
# files that must be byte-identical to the 009 release: the arm FG science, the frozen sampler and attention core, the
# vendored decoder, the input pipeline, the loop base, the engine, the checkpoint code and the earlier control tools
CORE_IDENTICAL = GC.CORE_IDENTICAL + (
    "tools/r28/r28_geo_control.py", "src/sparse_rtdetr/research/r28_fovea_geo.py",
    "src/sparse_rtdetr/baseline/training_v2b_deterministic_sampling.py", "src/sparse_rtdetr/baseline/training_v2b.py",
    "src/sparse_rtdetr/protocol_r18/model.py", "src/sparse_rtdetr/protocol_r18/geometry.py",
    "vendor/rtdetrv2_pytorch/src/zoo/rtdetr/rtdetrv2_decoder.py", "vendor/rtdetrv2_pytorch/src/zoo/rtdetr/utils.py",
    "vendor/rtdetrv2_pytorch/src/zoo/rtdetr/denoising.py")
EDITS = {
    "tools/r28/r28_common.py": {"changed": ("arm_module", "binding", "demand_source", "endpoint_state", "followup_arm", "formal_dir",
                                            "load_contract", "other_contract", "pipeline_mode", "selected_width"), "added": ("r29",)},
    "tools/r28/r28_loop.py": {"changed": ("Observer.update_row", "guard", "run_loop", "train"), "added": ()},
    "tools/r28/r28_eval.py": {"changed": ("evaluate", "summarize"), "added": ()},
    "tools/r28/r28_cost.py": {"changed": ("Probe.__init__", "Probe.stages", "cost", "instrument", "stage_table", "timing"), "added": ()},
    "tools/r28/r28_run.py": {"changed": ("main",), "added": ()},
}
KEPT = {"tools/r28/r28_common.py": ("save", "read_side", "read_state", "load", "stage_closed", "calibration", "prepare"),
        "tools/r28/r28_loop.py": ("restore_files", "install_observer", "formal", "cal_row", "Observer.loss_boxes", "Observer.loss_labels_vfl"),
        "tools/r28/r28_eval.py": ("infer", "difference_pp", "apl6"),
        "tools/r28/r28_cost.py": ("aggregate", "_run", "load_arm", "load_reference", "Probe.mark", "Probe.start")}
TESTS = GC.TESTS + ("tests/test_r29_unzoom.py", "tests/test_r29_control.py", "tests/test_r29_input.py")
# release-001 of the first R29 round: the accepted host delivery.  Read-only reference of the window delivery.
HOST_ROUND = K.LD_ROOT / "r29-query-unzoom-20261116-001"
PIN_HOST = {
    "commit": "efc113dcd8324b32c7a862e3db4ff28985c813a6",
    "manifest": "5607ec6cb86da2872305e67b046e55e6a92b124c11a084320626aa6d8e1c60bc",
    "contract_file": "db772c99e53614bb721c4595fd31b1aad2550de804d526f8a6a30f9d4087ccff",
    "contract": "17a2ca0cf3eb87870006b7879b49b3f188b23324e41b11faacdba8241e7550f8",
    "accept_closed": "9d3ea07e81b60927c4d30c406486d71445ed59ebd264c2f6c7d2842602360af7",
    "steps": {"QP-full": "e928c0d9c0fd9e2f40362a1aaa991e8e4acf8c27c1e88b32e1ff0426c2e731aa",
              "QP-resumed": "26d186c4b90302c5f9be2c0dd2f273c40fd0dba5c44f60ba117914253fa57c80",
              "QC-full": "9e3863fc4710aad284d059b8f07cb1f83ec5bf98d61d4258e638d9c72b2c64c9",
              "QC-resumed": "9bec764092af9d5bd4d8e8b5a5a3bd4248dfc0cf216a975a39c2b592be87f3e6"},
}
HOST_CHANGED = ("tools/r28/r28_loop.py", "tools/r28/r28_run.py", "tools/r28/r29_control.py", "tools/r28/r29_chain.sh",
                "tests/test_r29_control.py")
HOST_ADDED = ("tools/r28/r29_input.py", "tests/test_r29_input.py")
# keys in which the contract of this release may differ from the contract of release-001; every other key is equal
HOST_CONTRACT_DIFFERENT = ("contract_sha256", "version", "source_root", "source_manifest", "source_commit", "source_dirty",
                           "input_pipeline", "resource_plan", "outputs", "followup")
HOST_FOLLOWUP_DIFFERENT = ("task", "decision", "source_delta", "function_edits")
SCOPE = "r29-acceptance"
ACCEPT = {"N_updates": 2, "steps": 4, "save_at": 2, "ready_limit": 1, "workers": 2}
_ARM_UPDATES = ACCEPT["steps"] + (ACCEPT["steps"] - ACCEPT["save_at"])
_ACCEPT_UPDATES = ACCEPT["N_updates"] + len(ARMS) * _ARM_UPDATES
WIRE = {"eval_batches": 1, "cost_warmup": 1, "cost_batches_per_pass": 1, "cost_passes": 1}
NCHECK = {"batches": 2, "images": 2 * K.EVAL_BATCH}
_FORMAL = K.ENDPOINT - K.PARENT_UPDATE
_EVAL = K.EVAL_WARMUP + K.EVAL_BATCHES
WORKLOAD = {
    "r29-check": {"device": "cpu", "updates": 3 * len(ARMS), "forwards": 6 * len(ARMS) + 6,
                  "note": "listed apart: unit tests in a subprocess and the two-image H fixture of each arm (two updates, one "
                          "restore replay) plus evaluation-mode forwards of the fixture checks; no CUDA"},
    "wire": {"updates": 0, "forwards": len(ARMS) * WIRE["eval_batches"] + 3 * (WIRE["cost_warmup"] + WIRE["cost_passes"] * WIRE["cost_batches_per_pass"]),
             "scorer_calls": 0, "note": "listed apart: wiring of the evaluation batch and of the staged N / C / P timing"},
    "ncheck": {"updates": 0, "forwards": NCHECK["batches"], "scorer_calls": 0},
    "accept": {"updates": _ACCEPT_UPDATES, "forwards": 2 * _ACCEPT_UPDATES,
               "paths": {"N": ACCEPT["N_updates"], **{a: {"full": ACCEPT["steps"], "resumed": ACCEPT["steps"] - ACCEPT["save_at"]} for a in ARMS}}},
    "formal": {a: K.WORKLOAD["formal"] for a in ARMS},
    "evaluation": {a: K.WORKLOAD["evaluation"] for a in ARMS},
    "cost-R29": K.WORKLOAD["cost-R29"],
    "planned_updates": _ACCEPT_UPDATES + len(ARMS) * _FORMAL,
    "planned_complete_detector_forwards": 2 * _ACCEPT_UPDATES + len(ARMS) * 2 * _FORMAL + len(ARMS) * _EVAL + NCHECK["batches"]
    + len(K.COST_ENDPOINTS["cost-R29"]) * K.COST_PER_ENDPOINT,
    "score_budget_calls": 2 * len(ARMS), "per_class_COCOeval_calls": len(ARMS),
    "listed_apart": ["r29-check (CPU fixture)", "wire (8 forwards)", "the frozen demand network inside every forward", "dev rounds",
                     "any failure or re-verification"],
    "not_run": ["N training or rescoring", "a second seed", "any kernel width / point count / scale scan", "loss or head changes",
                "128-step learning screens", "checkpoint selection", "holdout access"],
}
PROTECTION = (("600", "AP", -0.10), ("300", "AP", -0.20), ("300", "APS", -0.20), ("600", "APM", -0.50), ("600", "APL", -0.50))
RULES = {  # TASK_R29_001 section 6; differences in pp; fixed before any arm is trained
    "N_gate": {"pair": ["QP", "FG"], "reported": "P - N", "primary": [["600", "APS", 0.30], ["600", "AP75_small", 0.0]],
               "protection": [list(x) for x in PROTECTION], "model_gpu_ratio": ["QP/FG", 1.10]},
    "M_gate": {"pair": ["QP", "QC"], "reported": "P - C", "primary": [["600", "APS", 0.20], ["600", "AP75_small", 0.0]],
               "protection": [list(x) for x in PROTECTION], "model_gpu_ratio": ["QP/QC", 1.05]},
    "tradeoff_reading": {"pair": ["QP", "QC"], "model_gpu_ratio": ["QP/QC", 0.90],
                         "cells": [["600", "AP", -0.10], ["600", "APS", -0.10], ["600", "AP75_small", -0.10]],
                         "label": "pointwise_accuracy_cost_tradeoff",
                         "note": "an independent cost reading: not a non-inferiority proof and not a replacement of N_gate"},
    "branches": {"both": "single_seed_method_signal", "N_only": "unzoom_adaptation_preliminary_gain_direct_read_increment_not_established",
                 "N_failed": "net_increment_target_not_reached"},
    "nature": "practical increment targets of a single-seed screen on the reused 548-image development set; not significance tests, "
              "equivalence tests, power calibration or changes of earlier thresholds",
}
STATEMENTS = {
    "registration": "R29 是读取 R28 强 F 结果之后的新探索性方法实验：固定强 F_geo 输入，只改三层 decoder cross-attention 的读取算子；"
                    "不是 W 重开、F_geo 调参或旧轮继续运行。",
    "arms": "N＝已认证 F_geo（复用 009 端点与指标，不重训不重评）；C＝先把投影后的值张量整幅反变形到同尺寸源均匀网格再读取；"
            "P＝同一源坐标查询点直接映到 warped memory 读取。两臂零新增参数，原有参数照旧训练。",
    "C_is_adaptation": "C 是 LZU 思路的匹配适配控制，不是完整官方 LZU 复现；C 的中间重建会改变带宽、平滑、混叠、边界与梯度，"
                       "P−C 不能单独归因于少一次插值。",
    "two_questions": "P−N 检验新的读取归纳偏置，不证明原生坐标有错误；P−C 检验直接读取相对整幅重建的差别。只胜 C 而不胜 N 不立项为检测收益。",
    "single_seed": "单 seed（seed0）、重复使用的 548 图 development 集；阈值是事先选定的实用增量目标，不是显著性、等效或功效检验。"
                   "两者接近不称等效，单 seed 不称稳定。",
    "previous": "旧 W 的 closed_fixed_config、F_geo 的 RECORD_ONLY 与全部既有读数、成本原样保留。",
    "cost": "成本以 N/C/P 同进程、GPU 独占、同一设置的完整模型 GPU 时间为准；不用共用 GPU 或旧锁频墙钟作方法成本，"
            "不以“省去 C 的重建”代替相对 N 的成本说明；训练输入流水线提速是工程项。",
    "audit": "训练期几何读数只取每个 epoch 的第一个窗口与最后一次更新（sampled windows），不外推为每一步成立；它们不是准入或选择依据。",
    "no_trigger": "结果不自动触发扩 seed、扫参数、加新头或延程；原创性未建立，后续由总控决定。",
}


# ================================================================ small pure helpers (used by the loop and the cost stage)
BASE_FIELDS = ("input", "gt_count", "lr", "r28")


def _digest(value):
    return K.sha_bytes(K.canonical(value))


def base_ledger(c):
    """Per update, the digests of the arm FG ledger fields that do not depend on the detector weights: the input record
    (samples, augmentation draws, tensors), the GT count, the learning rate and the demand / map / render readings."""
    fu = K.followup(c)
    rows = {}
    with K.pinned(fu["FG_ledger"]).open() as f:
        for line in f:
            row = json.loads(line)
            rows[row["input"]["update"]] = {k: _digest(row[k]) for k in BASE_FIELDS}
    K.require(len(rows) == K.ENDPOINT - K.PARENT_UPDATE and min(rows) == K.PARENT_UPDATE + 1 and max(rows) == K.ENDPOINT,
              f"arm FG ledger span: {len(rows)} rows")
    return rows


def base_differences(row, base):
    return [k for k in BASE_FIELDS if _digest(row[k]) != base[k]]


def audit_updates(start, stop):
    """Task section 4: the first window of every 304 updates and the last update of the run."""
    return frozenset(u for u in range(start + 1, stop + 1) if (u - 1) % K.WINDOWS_PER_EPOCH == 0) | {stop}


def audit_row(sink):
    """Ledger entry of one update: None when the update was not audited, else the read readings of its two forwards.
    A sampling-position gradient that is missing or not finite stops the run."""
    if sink is None:
        return None
    K.require(len(sink) == 2, f"R29 audit: {len(sink)} forward records in one update")
    for fwd in sink:
        K.require(fwd["identity_batch"] or (len(fwd["layers"]) == 3 and all(r.get("location_gradient", {}).get("finite") is True
                                                                              for r in fwd["layers"])),
                  f"R29 audit: sampling-position gradient missing or not finite: {[r.get('location_gradient') for r in fwd['layers']]}")
    json.dumps(sink, allow_nan=False)
    return {"forwards": sink}


def gpu_compute_processes():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def exclusive_gpu(label):
    rows = gpu_compute_processes()
    others = [r for r in rows if r.split(",")[0].strip() != str(os.getpid())]
    K.require(not others, f"the GPU holds another compute process {label}: {others}")
    return {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "compute_processes": rows, "own_pid": os.getpid(), "others": others}


# ================================================================ release difference
def source_delta(old_files, new_files, *, label="the 009 round release"):
    changed = sorted(k for k in old_files if k in new_files and old_files[k] != new_files[k])
    added = sorted(k for k in new_files if k not in old_files)
    removed = sorted(k for k in old_files if k not in new_files)
    K.require(not removed and set(changed) <= set(ALLOWED_CHANGED) and set(added) <= set(ALLOWED_ADDED),
              f"release differs from {label} outside the R29 adapter: changed {changed}, added {added}, removed {removed}")
    science = ("src/", "vendor/", "configs/")
    K.require(not any(p.startswith(science) for p in changed) and all(p in SCIENCE_ADDED for p in added if p.startswith(science)),
              f"release touches the science tree of {label} beyond the added R29 read module")
    K.require(all(p in old_files and old_files[p] == new_files.get(p) for p in CORE_IDENTICAL),
              f"a core file differs from {label}: {[p for p in CORE_IDENTICAL if old_files.get(p) != new_files.get(p)]}")
    return {"against": label, "changed": changed, "added": added, "removed": removed, "unchanged_files": len(old_files) - len(changed),
            "existing_science_files_identical": True, "science_files_added": [p for p in added if p.startswith(science)],
            "core_files_identical": {p: new_files[p] for p in CORE_IDENTICAL},
            "note": "every existing file of src/, vendor/ and configs/ (arm FG and the earlier arms, the frozen deterministic sampler "
                    "and attention core, the vendored decoder, criterion, data, engine, checkpoint), the input pipeline, the loop "
                    "base and the control tools of the earlier rounds are byte-identical; the read operator is one added module"}


def edits(old_root, new_root):
    """The function-level difference of every changed tool file against the 009 release; it must be exactly EDITS."""
    out = {}
    for rel, want in EDITS.items():
        d = GC.function_delta((Path(old_root) / rel).read_text(), (Path(new_root) / rel).read_text())
        K.require(d["changed"] == sorted(want["changed"]) and d["added"] == sorted(want["added"]) and not d["removed"]
                  and set(KEPT.get(rel, ())) <= set(d["unchanged"]),
                  f"{rel}: function edits differ from the registered R29 list: {d['changed']} {d['added']} {d['removed']}")
        out[rel] = {"changed": d["changed"], "added": d["added"], "removed": d["removed"], "unchanged_functions": len(d["unchanged"]),
                    "kept_named": list(KEPT.get(rel, ()))}
    return out


# ================================================================ references into the F_geo round (read-only)
def fg_references(fg_round=None):
    """The F_geo round of task 009: contract, release, the e120 endpoint with its weights digest, predictions, metrics,
    the formal ledger and the acceptance receipts.  Everything pinned by file SHA; nothing is written."""
    from sparse_rtdetr.research import r28_fovea_geo as FG
    r = Path(fg_round or FG_ROUND).resolve()
    refs = {k: K.ref(r / v) for k, v in FG_FILES.items()}
    bad = [k for k in refs if refs[k]["sha256"] != PIN_FG[k]]
    K.require(not bad, f"files of the F_geo round differ from their pins: {bad}")
    c9 = K.read_json(r / FG_FILES["contract_file"])
    body = {k: v for k, v in c9.items() if k != "contract_sha256"}
    K.require(c9["contract_sha256"] == K.sha_bytes(K.canonical(body)) == PIN_FG["contract"] and c9["source_commit"] == PIN_FG["commit"]
              and c9["source_manifest"]["sha256"] == PIN_FG["manifest"] and c9["authorization_basis"] == K.GEO_BASIS
              and Path(c9["outputs"]).resolve() == r and c9["parent"]["sha256"] == K.PARENT_SHA == PIN_FG["parent"]
              and c9["followup"]["W_demand"]["sha256"] == PIN_FG["demand"] and c9["identity_sha256"]["FG"] == FG.IDENTITY_SHA256["FG"]
              and c9["followup"]["width_selection"]["fwhm_hex"] == FG.WIDTH_HEX, "F_geo round contract identities")
    K.pinned(c9["source_manifest"])
    inherited = {}
    for name in INHERITED_FG:
        ref = K.ref(r / name / "CLOSED.json")
        closed = K.read_json(ref["path"])
        K.require(closed.get("status") == "PASS" and closed.get("contract_sha256") == c9["contract_sha256"],
                  f"F_geo round stage {name} is not PASS under the F_geo contract")
        inherited[name] = ref
    ev, summary = K.read_json(inherited["eval-FG"]["path"]), K.read_json(inherited["summary-FG"]["path"])
    accept, formal = K.read_json(inherited["accept-FG"]["path"]), K.read_json(inherited["formal-FG"]["path"])
    K.require(ev["arm"] == "FG" and ev["endpoint"] == refs["endpoint"] and ev["weights_sha256"] == PIN_FG["weights"]
              and ev["demand"]["file"]["sha256"] == PIN_FG["demand"] and ev["global_update"] == K.ENDPOINT
              and formal["updates"] == K.ENDPOINT - K.PARENT_UPDATE and formal["resumed"] is False
              and accept["overlap_enabled"] is True and accept["serial_vs_overlap_equal"] is True and accept["restore_replay_equal"] is True,
              "F_geo evaluation / formal / acceptance receipts")
    for ref in (summary["result"], ev["cocoeval_top600_tensors"], *ev["predictions"].values(), *ev["metrics"].values(), accept["saved_state"]["file"]):
        K.pinned(ref)
    ledger = K.read_json(K.side_of(ev["endpoint"]["path"]))["audit"]["ledger"]
    K.require(ledger == refs["ledger"], "F_geo ledger reference of the endpoint side-car")
    return c9, inherited, {"FG_round": str(r), "FG_contract": refs["contract_file"], "FG_contract_sha256": c9["contract_sha256"],
                           "FG_commit": c9["source_commit"], "FG_manifest": c9["source_manifest"], "FG_result": summary["result"],
                           "FG_endpoint": ev["endpoint"], "FG_weights_sha256": ev["weights_sha256"], "FG_predictions": ev["predictions"],
                           "FG_metrics": ev["metrics"], "FG_identity_sha256": c9["identity_sha256"]["FG"], "FG_ledger": ledger,
                           "FG_accept_steps": refs["accept_steps_serial"], "FG_accept_state": accept["saved_state"]["file"],
                           "FG_sampling_backend": "deterministic_gather"}


# ================================================================ references into release-001 of the first R29 round
def host_references(new_files):
    """The accepted host-delivery release: its contract, its closed joint acceptance and the per-update receipts of both
    arms, pinned by file SHA; the file difference of this release against it.  Nothing is written."""
    r = HOST_ROUND.resolve()
    refs = {"contract_file": K.ref(r / "CONTRACT.json"), "manifest": K.ref(r / "release-001/MANIFEST.json"),
            "accept_closed": K.ref(r / "accept-R29/CLOSED.json")}
    steps = {n: K.ref(r / f"accept-R29/STEPS-{n}.jsonl") for n in PIN_HOST["steps"]}
    bad = [k for k in refs if refs[k]["sha256"] != PIN_HOST[k]] + [n for n in steps if steps[n]["sha256"] != PIN_HOST["steps"][n]]
    K.require(not bad, f"files of the first R29 round differ from their pins: {bad}")
    ch = K.read_json(refs["contract_file"]["path"])
    K.require(ch["contract_sha256"] == K.sha_bytes(K.canonical({k: v for k, v in ch.items() if k != "contract_sha256"})) == PIN_HOST["contract"]
              and ch["source_commit"] == PIN_HOST["commit"] and ch["source_manifest"]["sha256"] == PIN_HOST["manifest"]
              and ch["authorization_basis"] == K.R29_BASIS and Path(ch["outputs"]).resolve() == r, "contract of the first R29 round")
    closed = K.read_json(refs["accept_closed"]["path"])
    K.require(closed.get("status") == "PASS" and closed.get("contract_sha256") == PIN_HOST["contract"] and closed["N_equal_to_the_FG_receipts"] is True
              and all(closed["restore_replay_equal"][a] is True and closed["read_active"][a] is True for a in ARMS),
              "the joint acceptance of release-001 is not PASS under its contract")
    old = K.read_json(refs["manifest"]["path"])["files"]
    changed = sorted(k for k in old if k in new_files and old[k] != new_files[k])
    added = sorted(k for k in new_files if k not in old)
    removed = sorted(k for k in old if k not in new_files)
    science = ("src/", "vendor/", "configs/")
    K.require(not removed and set(changed) <= set(HOST_CHANGED) and set(added) <= set(HOST_ADDED)
              and not any(p.startswith(science) for p in changed + added),
              f"release differs from release-001 outside the window delivery: changed {changed}, added {added}, removed {removed}")
    return ch, {"host_round": str(r), "host_contract": refs["contract_file"], "host_contract_sha256": ch["contract_sha256"],
                "host_commit": ch["source_commit"], "host_manifest": refs["manifest"], "host_accept": refs["accept_closed"],
                "host_accept_steps": steps,
                "host_release_delta": {"against": "release-001 of the first R29 round", "changed": changed, "added": added, "removed": removed,
                                       "identical_files": len(old) - len(changed), "science_tree_identical": True,
                                       "note": "every file of src/, vendor/ and configs/ (the read operator of both arms, arm FG, the "
                                               "detector, criterion, data, loader, engine, checkpoint) and the input pipeline module "
                                               "are byte-identical to release-001; the change is the window delivery in the loop tools"}}


def _forced_delivery(c):
    old = os.environ.get("R29_DELIVERY")
    os.environ["R29_DELIVERY"] = "device"
    try:
        RI.formal_mode(c, "QP")
    finally:
        if old is None:
            del os.environ["R29_DELIVERY"]
        else:
            os.environ["R29_DELIVERY"] = old


# ================================================================ contract of the independent round
def prepare(_args=None):
    from sparse_rtdetr.research import r29_unzoom as UZ
    round_ = K.ROUND.resolve()
    K.require(K.ROOT.resolve().parent.parent == round_ and K.ROOT.name == "source", "prepare-r29 must run from a frozen release")
    K.require("R28_ROUND" in os.environ and round_ != FG_ROUND.resolve() and FG_ROUND.resolve() not in round_.parents
              and round_ != HOST_ROUND.resolve() and HOST_ROUND.resolve() not in round_.parents,
              "the R29 round must be an independent directory named by R28_ROUND")
    K.require(not K.DEV or round_.parent.parent == K.LD_ROOT.resolve(), "a dev round lives inside the round directory")
    manifest = K.read_json(K.ROOT.parent / "MANIFEST.json")
    K.require(manifest["files"] == K.B.source_manifest(K.ROOT), "release source differs from MANIFEST")
    K.require(not manifest.get("dirty", False) or K.DEV, "a release assembled from a dirty work tree is only accepted in a dev round")
    task, decision = (K.ref(K.ROUND / f[0]) for f in (TASK, DECISION))
    K.require(task["sha256"] == TASK[1] and decision["sha256"] == DECISION[1], "task / decision file SHA")
    c9, fg_inherited, fg_refs = fg_references()
    ch, host_refs = host_references(manifest["files"])
    fu9 = c9["followup"]
    K.require(set(fu9["inherited"]).isdisjoint(fg_inherited) and set(OWN_STAGES).isdisjoint(set(fu9["inherited"]) | set(fg_inherited)),
              "stage names of the rounds overlap")
    delta = source_delta(K.read_json(K.pinned(c9["source_manifest"]))["files"], manifest["files"])
    function_edits = edits(c9["source_root"], K.ROOT)
    keep = {k: v for k, v in fu9.items() if k not in ("registration", "authorization_basis", "arm", "task", "review", "source_delta",
                                                      "function_edits", "trained_here", "reused_frozen", "analysis", "statements",
                                                      "inherited", "inherited_contracts")}
    states = len(ARMS) * len(K.SAVE_EPOCHS)
    c = {k: v for k, v in c9.items() if k != "contract_sha256"}
    c.update(version="R29-001-contract-v2", authorization_basis=K.R29_BASIS,
             source_root=str(K.ROOT.resolve()), source_manifest=K.ref(K.ROOT.parent / "MANIFEST.json"),
             source_commit=manifest["commit"], source_dirty=bool(manifest.get("dirty", False)),
             identity={**c9["identity"], **UZ.IDENTITY}, identity_sha256={**c9["identity_sha256"], **UZ.IDENTITY_SHA256},
             r29_read_definition=UZ.DEFINITION, r29_rules=RULES, workload=WORKLOAD,
             input_pipeline={**c9["input_pipeline"], "r29": "the overlapped path accepted in task 009 (workers 2, one prepared window, "
                             "audit before consumption, restart from the committed cursor), unchanged; each arm uses it only if "
                             "its own save / restore replay of the joint acceptance was exact",
                             "r29_window_delivery": RI.RULE},
             cost={**c9["cost"], "cost-R29": {"endpoints": list(K.COST_ENDPOINTS["cost-R29"]),
                                              "rotations": [list(r) for r in K.COST_ROTATIONS["cost-R29"]], "exclusive_gpu": True,
                                              "stages": ["demand", "map", "render", "detector", "read", "unwarp", "other"]}},
             labels={**c9["labels"], **{f"R29_{k}": v for k, v in RULES["branches"].items()}},
             resource_plan={"state_bytes": c9["parent"]["bytes"], "formal_states": states,
                            "formal_states_bytes": states * c9["parent"]["bytes"],
                            "acceptance_states_bytes": len(ARMS) * c9["parent"]["bytes"],
                            "ledgers_predictions_audits_bytes_estimate": len(ARMS) * 400 * 1024 ** 2,
                            "failure_margin_bytes": 4 * 1024 ** 3, "disk_free_bytes_at_prepare": __import__("shutil").disk_usage(K.ROUND).free,
                            "gpu_time_reference": "F_geo measured 11.42 h wall for 18,240 updates (about 9.6 h exclusive); the task "
                                                  "estimate for both arms is 20-28 h; not a cap"},
             outputs=str(K.ROUND), formal_started=False,
             followup={**keep, "registration": REGISTRATION, "authorization_basis": K.R29_BASIS, "arm": None, "arms": list(ARMS),
                       "short_names": SHORT, "training_order": list(ORDER), "task": task, "decision": decision, **fg_refs,
                       "inherited": {**fu9["inherited"], **fg_inherited},
                       "inherited_contracts": {**fu9["inherited_contracts"], **{name: c9["contract_sha256"] for name in fg_inherited}},
                       "source_delta": delta, "function_edits": function_edits,
                       "trained_here": ["QP (P) detector, 18,240 updates from the e060 parent", "QC (C) detector, 18,240 updates from the "
                                        "e060 parent", "14 acceptance updates (discarded states)"],
                       "reused_frozen": ["W demand network (the demand source of F_geo, QP and QC)", "the fixed kernel width 19.03125 and "
                                         "its selection reference", "F_geo e120 endpoint, predictions, metrics, per-class tensors (N; "
                                         "quoted, not retrained, not rescored)", "the F_geo formal ledger (input / plan identity per "
                                         "update) and its acceptance receipts"],
                       "analysis": {"pairs": ["QP-FG", "QC-FG", "QP-QC"], "reported_pairs": ["P-N", "C-N", "P-C"], "rules": "r29_rules"},
                       "statements": STATEMENTS, **host_refs})
    # the science of this contract is the science of release-001: every key outside the named engineering keys is equal
    fu, fuh = c["followup"], ch["followup"]
    K.require(set(c) == set(ch) and all(c[k] == ch[k] for k in ch if k not in HOST_CONTRACT_DIFFERENT)
              and set(fu) == set(fuh) | set(host_refs) and all(fu[k] == fuh[k] for k in fuh if k not in HOST_FOLLOWUP_DIFFERENT)
              and all(fu[k]["sha256"] == fuh[k]["sha256"] for k in ("task", "decision"))
              and fu["function_edits"] == fuh["function_edits"]
              and {k: v for k, v in c["input_pipeline"].items() if k != "r29_window_delivery"} == ch["input_pipeline"]
              and {k: v for k, v in c["resource_plan"].items() if k != "disk_free_bytes_at_prepare"}
              == {k: v for k, v in ch["resource_plan"].items() if k != "disk_free_bytes_at_prepare"},
              "the contract differs from the contract of release-001 outside the window delivery")
    c["followup"]["host_contract_equal"] = {
        "equal_keys": sorted(k for k in ch if k not in HOST_CONTRACT_DIFFERENT),
        "different_keys": {"version": "contract version", "source_root / source_manifest / source_commit / source_dirty": "the release",
                           "input_pipeline": "one added key, r29_window_delivery; every other key equal",
                           "resource_plan": "disk_free_bytes_at_prepare only", "outputs": "the round directory",
                           "followup": "task / decision (same SHA, other path), source_delta (two added tool / test files), the "
                                       "host_* references; function_edits and every other key equal"},
        "note": "parent, data, recipe (8 x 2, BF16, optimizer, LR, EMA, BN, DN, schedule), identities of N / C / P, the read "
                "definition, the rules of section 6, the evaluation policy and the workload are the equal keys"}
    K.require(c["resource_plan"]["disk_free_bytes_at_prepare"] > 4 * (c["resource_plan"]["formal_states_bytes"] + c["resource_plan"]["failure_margin_bytes"]),
              "less free space than four times the planned storage of this round")
    c["contract_sha256"] = K.sha_bytes(K.canonical(c))
    K.write_new(K.ROUND / "CONTRACT.json", c)
    print(json.dumps({"contract_sha256": c["contract_sha256"], "identity_sha256": UZ.IDENTITY_SHA256,
                      "authorization_basis": c["authorization_basis"], "source_delta": {k: delta[k] for k in ("against", "changed", "added")}}))
    return c


def _own(c, out, name):
    out = Path(out)
    K.require(out.resolve() == (K.ROUND / name).resolve() and K.r29(c) and K.ARM in ARMS,
              f"this action writes <round>/{name} under the R29 contract with R28_ARM=QP or QC")
    return out


# ================================================================ CPU: affected interfaces and the real-model fixture
def _fixture(c, arm, x2, t2, out, counts, n_rows):
    """Two updates, a restore replay of the second, and the read checks of one arm on the two-image H fixture."""
    import copy
    import l0_checks_state as H
    import r28_loop as LP
    from s15_formal_train import _bn_counts, _differences, _result_body
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    from sparse_rtdetr.research import r29_unzoom as UZ
    from sparse_rtdetr.research import scale_baseline as S
    torch = K.torch_mod()
    source, parent = K.demand_source(c, arm), K.pinned(c["parent"])
    K.PHASE["phase"] = f"r29-check:{arm} construct"
    e, checks = UZ.create(parent, "cpu", arm, source, fixture=True, check=True)
    ps = torch.load(parent, map_location="cpu", weights_only=True)
    sd, esd = e.model.state_dict(), e.ema.module.state_dict()
    keys = {"online_equal_parent": list(sd) == list(ps["model"]), "EMA_equal_parent": list(esd) == list(ps["ema"]["module"]),
            "state_keys": len(sd), "r29_keys": sum("r29" in k or "unzoom" in k for k in sd),
            "parameter_groups": [len(g["params"]) for g in e.optimizer.param_groups]}
    K.require(keys["online_equal_parent"] and keys["EMA_equal_parent"] and keys["r29_keys"] == 0
              and all(checks[k] for k in ("online_exact", "EMA_exact", "optimizer_exact", "optimizer_groups_exact", "parameter_steps_exact"))
              and checks["new_parameters"] == 0, ("native key set / parent identity", keys))
    del ps
    before = [tt["boxes"].clone() for tt in t2]
    fx = {"arm": arm, "short_name": SHORT[arm], "read_mode": e.r29_read, "native_keys": keys,
          "create_checks": {k: v for k, v in checks.items() if k != "new_parameter_names"},
          "images": [tt["image_id"] for tt in t2], "physical_batch": 1, "accumulation": 2, "precision": "float32"}
    # ---- evaluation read: no targets enter, the copy of the EMA reads alike, nothing stays installed
    K.PHASE["phase"] = f"r29-check:{arm} evaluation read"
    ema = e.ema.module
    with torch.no_grad():
        a = ema(x2[:1])
        b = ema(x2[:1], t2[:1])
        twin = copy.deepcopy(ema)
        d = twin(x2[:1])
    counts.update(forwards=3)
    K.require(not ema.training and set(a) == {"pred_logits", "pred_boxes"} and all(torch.equal(a[k], b[k]) and torch.equal(a[k], d[k]) for k in a)
              and UZ.is_wrapped(ema, arm) and UZ.is_wrapped(twin, arm) and twin.r28_ctx is not ema.r28_ctx
              and all(p is not q for p, q in zip(twin.parameters(), ema.parameters())), "evaluation read / EMA copy")
    fx["evaluation_read"] = {"targets_change_the_output": False, "deep_copy_output_equal": True, "copy_has_its_own_context": True,
                             "read_context_left_installed": False}
    del twin
    K.register_counts(e, counts)
    obs = LP.install_observer(e)
    bn0 = _bn_counts(e.model)
    rows, readings, audits = [], [], []
    for i in (1, 2):
        K.PHASE["phase"] = f"r29-check:{arm} u{i}"
        sink = vars(e.r28_ctx)["r29_audit"] = []
        r = e.train_window(x2, t2, capture=True)
        K.require(vars(e.r28_ctx).pop("r29_audit") is sink, "audit sink")
        K.require(r["update"] == K.PARENT_UPDATE + i and r["finite"] and e.ema.updates == e.completed_updates, "fixture clock")
        rows.append(r)
        readings.append(obs.update_row(t2))
        audits.append(audit_row(sink))
        if i == 1:
            side = K.save(c, e, out / f"cpu-fixture-{arm}-u1.pth", "cpu-fixture")
    K.require(all(torch.equal(a_, tt["boxes"]) for a_, tt in zip(before, t2)), "the forward modified the caller's targets")
    K.require(UZ.is_wrapped(e.model, arm) and side["arm"] == arm and side["identity_sha256"] == c["identity_sha256"][arm]
              and side["mapping"] == e.r28_mapping == UZ.mapping(source), "the saved state does not carry the arm / mapping identity")
    first = audits[0]["forwards"][0]["layers"]
    moved = max(layer[g]["displacement_feature_px"]["max"] for layer in first for g in ("normal", "dn") if g in layer)
    K.require(len(first) == 3 and all({"normal", "dn"} <= set(layer) and layer["mode"] == e.r29_read for layer in first) and moved > 0,
              f"fixture read readings: {first[0]}")
    fx.update(BN_delta_two_updates=sorted(set(v - bn0[k] for k, v in _bn_counts(e.model).items())), loss=[r["loss"] for r in rows],
              observer_first_update=readings[0], read_first_forward=first, max_displacement_feature_px_first_forward=moved)
    K.require(fx["BN_delta_two_updates"] == [4], f"fixture BN delta {fx['BN_delta_two_updates']}")
    expected = S.capture(e)
    del e
    gc.collect()
    K.PHASE["phase"] = f"r29-check:{arm} restore+replay"
    clone, _ = UZ.create(parent, "cpu", arm, source, fixture=True, check=True)
    K.register_counts(clone, counts)
    obs2 = LP.install_observer(clone)
    K.load(c, clone, out / f"cpu-fixture-{arm}-u1.pth", "cpu-fixture")
    sink = vars(clone.r28_ctx)["r29_audit"] = []
    replay = clone.train_window(x2, t2, capture=True)
    vars(clone.r28_ctx).pop("r29_audit")
    got = S.capture(clone)
    diff = _differences(expected, got) + _differences(_result_body(rows[1]), _result_body(replay))
    K.require(not diff and ck.exact_equal(expected, got) and obs2.update_row(t2) == readings[1] and audit_row(sink) == audits[1],
              f"fixture replay differs {diff[:5]}")
    fx.update(saved_u1=side, full_state_exact=True, replay_result_exact=True, observer_replay_exact=True, read_readings_replay_exact=True,
              replay_loss=replay["loss"], elapsed_seconds=[r["elapsed_seconds"] for r in rows + [replay]])
    return fx, clone, got, {"result": [_result_body(r) for r in rows], "observer": readings}


def r29_check(c, out, counts):
    import l0_checks_state as H
    import r28_checks as CK
    import r28_cost  # noqa: F401  (import check of the timing module)
    import r28_eval as E
    import r28_loop as LP
    import r28_pipeline as PL
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    from sparse_rtdetr.research import r28_fovea_geo as FG
    from sparse_rtdetr.research import r29_unzoom as UZ
    from sparse_rtdetr.research import scale_baseline as S
    torch = K.torch_mod()
    out = _own(c, out, "r29-check")
    K.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_available(), "the R29 check hides CUDA")
    fu = K.followup(c)
    torch.set_num_threads(int(os.environ.get("R28_CPU_THREADS", "16")))
    t0 = time.perf_counter()
    res = {"stage": "r29-check", "registration": fu["registration"], "authorization_basis": c["authorization_basis"],
           "workload": c["workload"]["r29-check"]}

    K.PHASE["phase"] = "r29-check:pytest"
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *[str(K.ROOT / t) for t in TESTS]],
                       capture_output=True, text=True, cwd=str(K.ROOT), env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    res["pytest"] = {"returncode": p.returncode, "tail": p.stdout.strip().splitlines()[-1:], "tests": list(TESTS),
                     "note": "synthetic unit tests in a subprocess; their forwards are not in this stage's counts"}
    K.require(p.returncode == 0, "synthetic pytest failed:\n" + p.stdout[-6000:])

    K.PHASE["phase"] = "r29-check:identities"
    c9, fg_inherited, fg_refs = fg_references(fu["FG_round"])
    fu9 = c9["followup"]
    K.require(all(fu[k] == v for k, v in fg_refs.items()) and fu["inherited"] == {**fu9["inherited"], **fg_inherited}
              and K.other_contract(c, "FG")["contract_sha256"] == c9["contract_sha256"] and fu["arms"] == list(ARMS) and fu["arm"] is None,
              "references changed since prepare")
    for name, f in (("task", TASK), ("decision", DECISION)):
        K.require(K.pinned(fu[name]) == K.ROUND / f[0] and fu[name]["sha256"] == f[1], f"{name} file")
    manifest, old_manifest = K.read_json(K.pinned(c["source_manifest"])), K.read_json(K.pinned(fu["FG_manifest"]))
    delta = source_delta(old_manifest["files"], manifest["files"])
    K.B.release_intact(Path(c9["source_root"]), old_manifest["files"])
    function_edits = edits(c9["source_root"], K.ROOT)
    K.require(delta == fu["source_delta"] and function_edits == fu["function_edits"], "source delta changed since prepare")
    _, host_refs = host_references(manifest["files"])
    K.require(all(fu[k] == v for k, v in host_refs.items()) and c["input_pipeline"]["r29_window_delivery"] == RI.RULE
              and RI.FORMAL in RI.MODES and RI.RULE["workers"] == PL.WORKERS == ACCEPT["workers"] and RI.RULE["ready_limit"] == PL.READY_LIMIT,
              "references into release-001 / window delivery rule changed since prepare")
    same = [k for k in c9 if k not in ("contract_sha256", "version", "authorization_basis", "source_root", "source_manifest", "source_commit",
                                       "source_dirty", "workload", "labels", "resource_plan", "outputs", "formal_started", "identity",
                                       "identity_sha256", "cost", "followup", "input_pipeline")]
    K.require(all(c[k] == c9[k] for k in same) and all(c["identity_sha256"][a] == c9["identity_sha256"][a] for a in c9["identity_sha256"])
              and all(c["identity_sha256"][a] == UZ.IDENTITY_SHA256[a] and c["identity"][a] == UZ.IDENTITY[a] for a in ARMS)
              and len(set(c["identity_sha256"].values())) == len(c9["identity_sha256"]) + 2 and c["r29_read_definition"] == UZ.DEFINITION
              and c["r29_rules"] == RULES and c["parent"]["sha256"] == PIN_FG["parent"]
              and all(fu[k] == fu9[k] for k in ("width_selection", "W_demand", "cal_calibration", "prior_contract", "prior_contract_sha256")),
              "inherited contract fields / identities")
    res["identities"] = {"FG_round_pins": dict(PIN_FG), "inherited_stages_of_the_FG_round": list(fg_inherited),
                         "contract_fields_equal_to_the_FG_round": len(same), "source_delta": delta, "function_edits": function_edits,
                         "identity_sha256": c["identity_sha256"], "parent_sha256": c["parent"]["sha256"], "release_009_intact": True,
                         "sampling_backend": {"accept-FG saved_state.config": K.stage_closed(c, "accept-FG", "FG")["saved_state"]["config"]["sampling_backend"],
                                              "formal-FG install.engine_config": K.stage_closed(c, "formal-FG", "FG")["install"]["engine_config"]["sampling_backend"],
                                              "engine_config_of_this_release": S.production_config(0).binding()["sampling_backend"]}}
    K.require(set(res["identities"]["sampling_backend"].values()) == {"deterministic_gather"},
              f"sampling backend of the 009 receipts / the production recipe: {res['identities']['sampling_backend']}")

    K.PHASE["phase"] = "r29-check:authorisation and isolation"
    for name in fu["inherited"]:
        K.require(not (K.ROUND / name).exists(), f"inherited stage name present in this round: {name}")
    lock = K.selected_width(c)
    source = K.demand_source(c, "QP")
    K.require(source == K.demand_source(c, "QC") == K.demand_source(c, "FG") and source["file"] == fu["W_demand"]
              and lock["fwhm"] == FG.WIDTH and all(UZ.check_source(a, source, False) == FG.mapping(source) for a in ARMS),
              "the three arms share one demand source and one fixed width")
    ev_n = K.stage_closed(c, "eval-FG", "FG")
    K.require(ev_n["endpoint"] == fu["FG_endpoint"] and ev_n["weights_sha256"] == fu["FG_weights_sha256"], "inherited N evaluation")
    rounds = [Path(fu[k]) for k in ("prior_round", "K_round", "F_round", "cal_round", "FG_round")]
    cases = [
        *[FU._refused(f"formal training of arm {a} in the R29 round", lambda a=a: LP.guard(c, K.ROUND / f"formal-{a}", None, a))
          for a in ("W", "K", "F", "FS", "FG")],
        *[FU._refused(f"formal {a} before the check and the acceptance are closed", lambda a=a: LP.guard(c, K.ROUND / f"formal-{a}", None, a))
          for a in ARMS],
        FU._refused("a formal FG directory lookup in this round", lambda: K.formal_dir(c, "FG")),
        FU._refused("an endpoint of arm W in this round", lambda: K.endpoint_state(c, "W")),
        *[FU._refused(f"an evaluation output inside {r.name}", lambda r=r: E.round_stage(r / "eval-QP", "eval-QP")) for r in rounds],
        FU._refused("an R29 arm under the F_geo basis",
                    lambda: K.demand_source({**c, "authorization_basis": K.GEO_BASIS, "followup": {**fu, "authorization_basis": K.GEO_BASIS}}, "QP")),
        FU._refused("arm QP built by the arm FG module", lambda: FG.check_source("QP", source, False)),
        FU._refused("arm FG built by the R29 module", lambda: UZ.check_source("FG", source, False)),
        FU._refused("arm QP with arm F's width 4", lambda: UZ.check_source("QP", {**source, "fwhm": 4.0}, False)),
        FU._refused("arm QC without the selection reference", lambda: UZ.check_source("QC", {k: v for k, v in source.items() if k != "selection"}, False)),
        FU._refused("arm QP with a fixed (arm K) demand source", lambda: UZ.check_source("QP", {**source, "kind": "fixed"}, False)),
        FU._refused("R28_PIPELINE forcing the overlapped path", lambda: GC._forced_pipeline(c)),
        FU._refused("the overlapped path of arm QP before the acceptance", lambda: K.pipeline_mode(c, "QP")),
        FU._refused("R29_DELIVERY forcing another delivery than the host reference", lambda: _forced_delivery(c)),
        FU._refused("window delivery of arm FG", lambda: RI.formal_mode(c, "FG")),
        *([] if RI.FORMAL == "host" else [FU._refused(f"the {RI.FORMAL} delivery of arm QP before the acceptance", lambda: RI.formal_mode(c, "QP"))]),
    ]
    K.require(not any(v.startswith("R28_DEV_") for v in os.environ) or K.DEV, "dev flags are refused outside a dev round")
    res["authorisation"] = {"round": str(K.ROUND), "independent_of": [str(r) for r in rounds], "refusals": cases}

    K.PHASE["phase"] = "r29-check:base ledger and rules"
    base = base_ledger(c)
    row1 = json.loads(K.pinned(fu["FG_ledger"]).open().readline())
    K.require(not base_differences(row1, base[K.PARENT_UPDATE + 1]) and base_differences({**row1, "lr": row1["lr"] * 2}, base[K.PARENT_UPDATE + 1]) == ["lr"]
              and sorted(audit_updates(K.PARENT_UPDATE, K.ENDPOINT))[:2] == [K.PARENT_UPDATE + 1, K.PARENT_UPDATE + 1 + K.WINDOWS_PER_EPOCH]
              and len(audit_updates(K.PARENT_UPDATE, K.ENDPOINT)) == 61 and K.ENDPOINT in audit_updates(K.PARENT_UPDATE, K.ENDPOINT),
              "base ledger / audit windows")
    trial = _trial_summary(c)
    res["rules"] = {"r29_rules": RULES, "audit_windows_per_arm": 61, "base_ledger_rows": len(base),
                    "trial_summary": {"branch": trial["decision"]["branch"], "pairs": list(trial["differences_pp"])}}
    del base

    K.PHASE["phase"] = "r29-check:real first window"
    (x, t, inp), = CK._first_windows(c, 1)
    theirs = H.r27_rows(c, "P", 1)[0]["input"]
    K.require(all(type(inp[k]) is type(theirs[k]) and inp[k] == theirs[k] for k in K.INPUT_FIELDS) and set(inp) == set(K.INPUT_FIELDS)
              and inp == row1["input"], "input identity differs from R27 formal-P row 1 / the arm FG ledger row 1")
    x2, t2 = x[:2], t[:2]
    res["real_window"] = {"input": inp, "window_shape": list(x.shape), "input_identity_equal_to_the_FG_ledger_row_1": True}
    fixtures, engines = {}, {}
    for arm in ARMS:
        fixtures[arm], engines[arm], state, trace = _fixture(c, arm, x2, t2, out, counts, 2)
        fixtures[arm]["trace"] = trace
        engines[arm] = (engines[arm], state)
    K.require(fixtures["QP"]["observer_first_update"] == fixtures["QC"]["observer_first_update"]
              and fixtures["QP"]["trace"]["observer"] == fixtures["QC"]["trace"]["observer"]
              and fixtures["QP"]["trace"]["result"][0]["microbatch_losses"] != fixtures["QC"]["trace"]["result"][0]["microbatch_losses"],
              "the two arms must see the same plan and differ in the read")
    K.PHASE["phase"] = "r29-check:state refusals"
    refused = []
    for arm, other in (("QP", "QC"), ("QC", "QP")):
        e, state = engines[arm]
        mine, theirs_ = out / f"cpu-fixture-{arm}-u1.pth", out / f"cpu-fixture-{other}-u1.pth"
        refused += [
            H.refused(f"arm {other} state loaded into the arm {arm} engine", lambda e=e, p=theirs_: K.load(c, e, p, "cpu-fixture")),
            H.refused(f"arm {arm} state read as arm {other}", lambda p=mine, o=other: K.read_side(c, p, "cpu-fixture", o)),
            H.refused(f"arm {arm} state read as arm FG", lambda p=mine: K.read_side(c, p, "cpu-fixture", "FG")),
            H.refused(f"arm {arm} fixture state read in formal scope", lambda p=mine, a=arm: K.read_side(c, p, "formal", a)),
            H.refused(f"FG formal e120 state loaded into the arm {arm} engine", lambda e=e: K.load(c, e, fu["FG_endpoint"]["path"], "formal")),
            H.refused(f"FG acceptance state loaded into the arm {arm} engine", lambda e=e: K.load(c, e, fu["FG_accept_state"]["path"], "fg-acceptance")),
            H.refused(f"check engine of arm {arm} writing a formal state", lambda e=e, a=arm: K.save(c, e, out / f"refused-{a}.pth", "formal")),
            H.refused(f"endpoint_state of arm {arm} without an endpoint", lambda a=arm: K.endpoint_state(c, a)),
        ]
        K.require(ck.exact_equal(S.capture(e), state) and not (out / f"refused-{arm}.pth").exists(), "a refused action changed the engine or wrote a file")
    res["refused_states"] = refused
    for arm in ARMS:
        fixtures[arm].pop("trace")
    res["fixture"] = fixtures
    K.require(counts["actual_updates"] == 3 * len(ARMS) and counts["forwards"] == 6 * len(ARMS) + 3 * len(ARMS), f"r29-check counts {dict(counts)}")
    res.update(fixture_updates=counts["actual_updates"], full_forwards=counts["forwards"], formal_state_usable=False,
               input_pipeline_version=PL.VERSION, window_delivery=RI.RULE, host_release_delta=fu["host_release_delta"],
               cpu_seconds_wall=time.perf_counter() - t0,
               scope="the read operator, its wiring into the real detector, the arm identities, the contract and the references; the "
                     "data, mapping, rendering, pipeline and restore evidence of the shared code is inherited from the 009 round")
    return res


# ================================================================ CUDA wiring of what only runs after training
def r29_wire(c, out, counts):
    import r28_cost as CO
    import r28_eval as E
    from sparse_rtdetr.protocol_r18.engine import tensor_digest
    from sparse_rtdetr.research import r29_unzoom as UZ
    torch = K.torch_mod()
    out = _own(c, out, "wire-R29")
    K.stage_closed(c, "r29-check")
    K.PHASE["phase"] = "wire:models (parent EMA weights behind the QP / QC forwards)"
    state = torch.load(K.pinned(c["parent"]), map_location="cpu", weights_only=True)["ema"]["module"]
    digest = tensor_digest(state.items())
    models = {a: UZ.build_eval_model(state, "cuda:0", a, K.demand_source(c, a)) for a in ARMS}
    K.require(all(UZ.is_wrapped(m, a) and not m.training for a, m in models.items()), "wire models")
    K.PHASE["phase"] = "wire:timing"
    order = K.COST_ENDPOINTS["cost-R29"]
    loaders = {"FG": lambda: CO.load_arm(c, "FG", torch),
               **{a: (lambda a=a: (models[a], {"weights": "e060 parent EMA (not trained under this read; wiring only)",
                                               "demand": models[a].r28_ctx.binding(), "mapping": models[a].r28_mapping}))
                  for a in ARMS}}
    timing = CO._run(c, counts, loaders, order, (order,) * WIRE["cost_passes"], warmup=WIRE["cost_warmup"],
                     batches_per_pass=WIRE["cost_batches_per_pass"], stages=True, names=CO.READ_STAGES)
    table = timing["stages"]["table"]
    K.require(set(table) == set(order) and all(abs(sum(table[a]["stage_ms_median_pass"].values()) - table[a]["model_gpu_ms_median_pass"])
                                                <= 1e-6 * max(1.0, table[a]["model_gpu_ms_median_pass"]) for a in order)
              and all(table[a]["stage_ms_median_pass"][s] > 0 for a in order for s in ("demand", "map", "render", "detector", "read", "unwarp"))
              and all("probe" not in vars(m.r28_ctx) and not any(layer.cross_attn._forward_hooks or layer.cross_attn._forward_pre_hooks
                                                                 for layer in m.decoder.decoder.layers) for m in models.values()),
              ("stage readings", table))
    K.PHASE["phase"] = "wire:evaluation batches"
    batches = {}
    for a in ARMS:
        sub = Counter(forwards=0)
        _, preds, _ = E.infer(c, models[a], sub, limit=(0, WIRE["eval_batches"]))
        counts.update(sub)
        K.require(all(len(preds[b]) == WIRE["eval_batches"] * K.EVAL_BATCH * b for b in K.BUDGETS)
                  and all(math.isfinite(v) for p in preds[600] for v in (p["score"], *p["bbox_xyxy"]))
                  and tensor_digest(models[a].state_dict().items()) == digest, f"wire evaluation batch of arm {a}")
        batches[a] = {"images": WIRE["eval_batches"] * K.EVAL_BATCH, "records_top600": len(preds[600]), "scored": False}
    K.PHASE["phase"] = "wire:summary structure"
    trial = _trial_summary(c, timing)
    K.require(counts["forwards"] == c["workload"]["wire"]["forwards"] and counts["actual_updates"] == 0
              and not counts.get("score_budget_calls"), f"wire counts {dict(counts)}")
    del models
    gc.collect()
    torch.cuda.empty_cache()
    return {"stage": "wire-R29", "workload": c["workload"]["wire"], "weights": "e060 parent EMA behind the QP / QC forwards; the stored N endpoint",
            "weights_sha256": digest,
            "purpose": "wiring of code that otherwise runs only after the formal training: the evaluation batch path of QP and QC and "
                       "the staged N / C / P timing with the decoder read split off; these readings are not results (the arms are "
                       "not trained here, the protocol is shortened, the GPU is not required to be exclusive)",
            "timing": {k: timing[k] for k in ("ratios", "medians", "stages", "protocol", "forwards_per_endpoint", "order_executed", "weights_sha256")},
            "evaluation_batch": batches,
            "summary_structure": {"branch": trial["decision"]["branch"], "pairs": list(trial["differences_pp"]),
                                  "cost_pairs": sorted(trial["cost"]["model_gpu"])},
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "scorer_calls": 0, "formal_state_usable": False}


# ================================================================ N through the present loading path (two batches, not scored)
def r29_ncheck(c, out, counts):
    import r28_eval as E
    from sparse_rtdetr.protocol_r18.engine import tensor_digest
    from sparse_rtdetr.research import r28_fovea_geo as FG
    torch = K.torch_mod()
    out = _own(c, out, "ncheck-FG")
    K.stage_closed(c, "r29-check")
    fu = K.followup(c)
    K.PHASE["phase"] = "ncheck:model"
    side, state = K.endpoint_state(c, "FG")
    source = K.demand_source(c, "FG")
    weights = tensor_digest(state.items())
    K.require(side["file"] == fu["FG_endpoint"] and weights == fu["FG_weights_sha256"] == PIN_FG["weights"]
              and side.get("mapping") == FG.mapping(source), "stored N endpoint")
    model = FG.build_eval_model(state, "cuda:0", "FG", source)
    K.require(FG.is_wrapped(model) and not model.training, "N evaluation model is the unchanged arm FG forward")
    K.PHASE["phase"] = "ncheck:inference"
    _, preds, timing = E.infer(c, model, counts, limit=(0, NCHECK["batches"]))
    K.PHASE["phase"] = "ncheck:comparison"
    compared = {}
    for b in K.BUDGETS:
        stored = K.read_json(K.pinned(fu["FG_predictions"][str(b)]))
        n = NCHECK["images"] * b
        K.require(len(stored) == 548 * b and len(preds[b]) == n, "prediction record counts")
        diff = CC.differences(stored[:n], preds[b], limit=20)
        compared[str(b)] = {"records": n, "fields": sorted(preds[b][0]), "differences": diff, "equal": not diff,
                            "stored": fu["FG_predictions"][str(b)]}
    K.require(counts["forwards"] == c["workload"]["ncheck"]["forwards"] and not counts.get("score_budget_calls"), f"ncheck counts {dict(counts)}")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    K.require(all(v["equal"] for v in compared.values()),
              f"N through the present loading path differs from the stored 009 predictions: {[v['differences'][:3] for v in compared.values()]}")
    return {"stage": "ncheck-FG", "workload": c["workload"]["ncheck"], "endpoint": side["file"], "weights_sha256": weights,
            "images": NCHECK["images"], "batches": NCHECK["batches"], "compared": compared, "field_by_field_equal": True, "scored": False,
            "note": "N is quoted from the 009 round; these two batches only confirm that the present release loads and runs it "
                    "as the stored predictions were produced", **timing}


# ================================================================ joint CUDA acceptance (N 2, QC 4 + 2, QP 4 + 2 updates)
def _displacement(audit, key="max"):
    """Largest displacement (feature pixels) of the read of one audited update, over forwards, layers and query groups."""
    vals = [layer[g]["displacement_feature_px"][key] for fwd in audit["forwards"] for layer in fwd["layers"]
            for g in ("normal", "dn") if g in layer and layer[g]["displacement_feature_px"][key] is not None]
    return max(vals) if vals else 0.0


def _dtypes(audits):
    """The distinct dtype records of the audited reads (one per arm is expected)."""
    seen = {json.dumps(layer["dtypes"], sort_keys=True) for a in audits for fwd in a["forwards"] for layer in fwd["layers"]}
    return [json.loads(s) for s in sorted(seen)]


def _read_summary(audits):
    """Compact per-layer means over the audited acceptance forwards (ordinary / dn apart); a description, not a gate."""
    out = {}
    for group in ("normal", "dn"):
        layers = {}
        for i in range(3):
            recs = [fwd["layers"][i][group] for a in audits for fwd in a["forwards"] if fwd["layers"] and group in fwd["layers"][i]]
            if not recs:
                continue
            pick = lambda f: [v for v in (f(r) for r in recs) if v is not None]
            mean = lambda f: (statistics.fmean(pick(f)) if pick(f) else None)
            layers[str(i)] = {
                "forwards": len(recs), "queries": sum(r["queries"] for r in recs),
                "displacement_normalised_mean": mean(lambda r: r["displacement_normalised"]["mean"]),
                "displacement_feature_px_mean": mean(lambda r: r["displacement_feature_px"]["mean"]),
                "displacement_feature_px_max": max(pick(lambda r: r["displacement_feature_px"]["max"]), default=None),
                "share_above_half_px": mean(lambda r: r["displacement_feature_px"]["share_above_half_px"]),
                "attention_weighted_displacement_feature_px_mean": mean(lambda r: r["attention_weighted_displacement_feature_px"]["mean"]),
                "reference_edges_in_different_segments": mean(lambda r: r["reference_edges_in_different_segments"]["any"]),
                "border": {w: {k: mean(lambda r, w=w, k=k: r["border"][w][k]) for k in ("interior", "edge_half_pixel", "outside")}
                           for w in ("N", "P", "C")},
                "read_difference_P_vs_C": {"samples_relative_l2": mean(lambda r: r["read_difference_P_vs_C"]["samples_relative_l2"]),
                                           "attention_output_relative_l2": mean(lambda r: r["read_difference_P_vs_C"]["attention_output_relative_l2"]),
                                           **{k: mean(lambda r, k=k: r["read_difference_P_vs_C"]["by_border_class_of_the_source_point"][k]["relative_l2"])
                                              for k in ("interior", "edge_half_pixel", "outside")}}}
        out[group] = layers
    return out


def r29_accept(c, out, counts):
    import l0_checks_state as H
    import l0_loop as LL
    import r28_loop as LP
    import r28_pipeline as PL
    from s15_formal_train import _result_body
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    from sparse_rtdetr.research import r28_fovea_geo as FG
    from sparse_rtdetr.research import r29_unzoom as UZ
    from sparse_rtdetr.research import scale_baseline as S
    torch = K.torch_mod()
    out = _own(c, out, "accept-R29")
    for name in ("r29-check", "wire-R29", "ncheck-FG"):
        K.stage_closed(c, name)
    fu = K.followup(c)
    steps, save_at, n_steps = ACCEPT["steps"], ACCEPT["save_at"], ACCEPT["N_updates"]
    refs, _ = LL.reference_rows(c)
    check = CC.reference_check(refs)
    base = base_ledger(c)
    data = S.train_dataset(c["data"]["train_core"], S.S15_HW, seed=0)
    p0 = K.PARENT_UPDATE
    digester = CC.Digester()
    info, refusals, starts = {}, [], {}
    stored = []
    with K.pinned(fu["FG_accept_steps"]).open() as f:
        for line in f:
            stored.append(json.loads(line)["science"])
    K.require(len(stored) == GC.ACCEPT["steps"] >= steps and [r["input"]["update"] for r in stored] == [p0 + 1 + i for i in range(len(stored))],
              "stored accept-FG serial receipts")

    def run(name, arm, mode, first, last, *, load=None, save_after=None, after=None):
        K.PHASE["phase"] = f"accept:{name} construct"
        M = FG if arm == "FG" else UZ
        source = K.demand_source(c, arm)
        e, checks = M.create(c["parent"]["path"], "cuda:0", arm, source)
        K.register_counts(e, counts)
        obs = LP.install_observer(e)
        loaded = None if load is None else K.load(c, e, load, SCOPE)
        K.require(e.completed_updates == first and e.r28_mapping == FG.mapping(source) and e.r28_arm == arm
                  and (M.is_wrapped(e.model, arm) and M.is_wrapped(e.ema.module, arm) if arm != "FG" else FG.is_wrapped(e.model) and FG.is_wrapped(e.ema.module)),
                  f"{name}: start update / mapping / arm")
        if load is None:
            starts[name] = CC.state_digests(S.capture(e))
        rows, saved, state_file = [], None, out / f"{arm}-u{save_at:02d}.pth"
        delivery = RI.Delivery("host" if arm == "FG" else RI.FORMAL, e.device)   # N: the reference delivery of the stored receipts
        torch.cuda.reset_peak_memory_stats()
        with PL.MemorySampler(0.5) as memory:
            opened = mark = time.perf_counter()
            with delivery.stream(mode, data, first, last, check, fine=True) as stream:
                for item in stream:
                    x, t, inp = item.x, item.t, item.inp
                    K.PHASE["phase"] = f"accept:{name} update {e.completed_updates + 1}"
                    K.require(inp["update"] == e.completed_updates + 1, f"{name}: window {inp['update']} is not the next update")
                    bad = LL.input_differences(inp, refs[inp["update"]]["input"])
                    K.require(not bad, f"{name}: input record differs from R27 formal-P before update {inp['update']}: {bad}")
                    sink = None
                    if arm != "FG":
                        sink = vars(e.r28_ctx)["r29_audit"] = []
                    called = time.perf_counter()
                    xd, moved = delivery.take(item)
                    res = e.train_window(xd, t, capture=True)
                    train_call = time.perf_counter() - called
                    del xd
                    K.require(res["update"] == inp["update"] and res["finite"] and e.ema.updates == e.completed_updates, f"{name}: update clock")
                    science = {"input": inp, "gt_count": int(sum(len(v["labels"]) for v in t)), "result": _result_body(res),
                               "observer": obs.update_row(t)}
                    if arm != "FG":
                        K.require(vars(e.r28_ctx).pop("r29_audit") is sink, f"{name}: audit sink")
                        science["r29"] = audit_row(sink)
                        ledger_row = {"input": inp, "gt_count": science["gt_count"], "lr": res["lr"], "r28": science["observer"]}
                        bad = base_differences(ledger_row, base[res["update"]])
                        K.require(not bad, f"{name}: input / demand / map readings differ from the arm FG ledger at update {res['update']}: {bad}")
                    digester.submit((name, res["update"]), S.capture(e))
                    extra = {}
                    if save_after is not None and e.completed_updates == save_after:
                        prepared = stream.wait_ready()
                        K.require(prepared == 1, f"{name}: no prepared window at the save ({prepared})")
                        s1 = time.perf_counter()
                        saved = K.save(c, e, state_file, SCOPE)
                        K.require(stream.ready() == 1, f"{name}: the prepared window was consumed during the save")
                        extra = {"saved_with_prepared_windows": prepared, "save_s": time.perf_counter() - s1}
                    now = time.perf_counter()
                    rows.append({"science": science,
                                 "engineering": {"cycle_s": now - mark, "wait_s": item.timing["wait_s"], "train_call_s": train_call,
                                                 "train_s": res["elapsed_seconds"], "prepared_after": stream.ready(), **moved, **extra}})
                    mark = now
                    del item, x, t
                account = stream.account()
            closed = time.perf_counter()
        K.require(e.completed_updates == last and len(rows) == last - first, f"{name}: update count")
        if after is not None:
            after(e, arm, state_file)
        info[name] = {"arm": arm, "short_name": SHORT[arm], "mode": mode, "first_update": first + 1, "last_update": last, "updates": len(rows),
                      "account": account, "window_delivery": delivery.account(),
                      "resumed_from": None if loaded is None else loaded["file"], "saved": saved,
                      "wall_seconds_stream": closed - opened, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                      "memory": memory.summary(),
                      "create_checks": {k: checks[k] for k in ("arm", "identity_sha256", "demand", "mapping", "new_parameters", "parameters",
                                                               "optimizer_parameters", "old_parameters_sha256", "EMA_updates")}}
        delivery.close()
        del e, obs, delivery
        gc.collect()
        torch.cuda.empty_cache()
        return rows

    def refuse(e, arm, state_file):
        K.PHASE["phase"] = f"accept:{arm} state refusals"
        other = next(a for a in ARMS if a != arm)
        before = S.capture(e)
        cases = [(f"arm {arm} acceptance state read as arm {other}", lambda: K.read_side(c, state_file, SCOPE, other)),
                 (f"arm {arm} acceptance state read as arm FG", lambda: K.read_side(c, state_file, SCOPE, "FG")),
                 (f"arm {arm} acceptance state read in formal scope", lambda: K.read_side(c, state_file, "formal", arm)),
                 (f"arm {arm} acceptance state loaded as a formal state", lambda: K.load(c, e, state_file, "formal")),
                 (f"FG formal e120 state loaded into the arm {arm} engine", lambda: K.load(c, e, fu["FG_endpoint"]["path"], "formal")),
                 (f"FG acceptance state of the 009 round loaded into the arm {arm} engine",
                  lambda: K.load(c, e, fu["FG_accept_state"]["path"], "fg-acceptance")),
                 (f"formal training of arm FG in this round", lambda: LP.guard(c, K.ROUND / "formal-FG", None, "FG"))]
        prior = out / f"{other}-u{save_at:02d}.pth"
        if prior.is_file():
            cases.append((f"arm {other} acceptance state loaded into the arm {arm} engine", lambda: K.load(c, e, prior, SCOPE)))
        refusals.extend(H.refused(label, fn) for label, fn in cases)
        K.require(ck.exact_equal(S.capture(e), before), "a refused action changed the engine")

    # N first: the unchanged arm FG engine of this release against the stored receipts of the 009 acceptance
    paths = {"N": run("N", "FG", "serial", p0, p0 + n_steps)}
    for arm in ("QC", "QP"):
        paths[f"{arm}-full"] = run(f"{arm}-full", arm, "overlap", p0, p0 + steps, save_after=p0 + save_at)
        paths[f"{arm}-resumed"] = run(f"{arm}-resumed", arm, "overlap", p0 + save_at, p0 + steps, load=out / f"{arm}-u{save_at:02d}.pth", after=refuse)
    K.PHASE["phase"] = "accept:state digests"
    digests = digester.finish()
    for name, rows in paths.items():
        for r in rows:
            r["science"]["state"] = digests[(name, r["science"]["input"]["update"])]
        with (out / f"STEPS-{name}.jsonl").open("x") as f:
            for r in rows:
                f.write(json.dumps(r, separators=(",", ":"), allow_nan=False) + "\n")
    K.PHASE["phase"] = "accept:comparison"
    sci = lambda rows: [r["science"] for r in rows]
    d_n = CC.differences(stored[:n_steps], sci(paths["N"]))
    parts = [k for k in starts["N"] if k not in ("arm",)]
    start_equal = {a: [k for k in parts if starts[f"{a}-full"][k] != starts["N"][k]] for a in ARMS}
    arm_part = {a: starts[f"{a}-full"]["arm"] != starts["N"]["arm"] for a in ARMS}
    replay, plan_equal, active, reads, first_loss = {}, {}, {}, {}, {}
    for arm in ARMS:
        full, resumed = sci(paths[f"{arm}-full"]), sci(paths[f"{arm}-resumed"])
        replay[arm] = CC.differences(full[save_at:], resumed)
        plan_equal[arm] = all(a["input"] == b["input"] and a["observer"] == b["observer"] and a["gt_count"] == b["gt_count"]
                              for a, b in zip(full, stored[:steps]))
        moved = _displacement(full[0]["r29"])
        differs = (full[0]["result"]["output_sha256"] != stored[0]["result"]["output_sha256"]
                   and full[0]["result"]["microbatch_losses"] != stored[0]["result"]["microbatch_losses"])
        active[arm] = bool(moved > 0 and differs)
        reads[arm] = {"max_displacement_feature_px_first_update": moved, "first_update_outputs_differ_from_N": differs,
                      "dtypes": _dtypes([r["r29"] for r in full + resumed]),
                      "summary_over_the_acceptance_forwards": _read_summary([r["r29"] for r in full])}
        first_loss[arm] = [r["result"]["loss"] for r in full]
    cross = sci(paths["QP-full"])[0]["result"]["output_sha256"] != sci(paths["QC-full"])[0]["result"]["output_sha256"]
    # window delivery: every update of both arms against the stored receipts of release-001 (host delivery), bit for bit
    host = {}
    for name in fu["host_accept_steps"]:
        with K.pinned(fu["host_accept_steps"][name]).open() as f:
            host[name] = CC.differences([json.loads(line)["science"] for line in f], sci(paths[name]))
    delivery_equal = {a: not host[f"{a}-full"] and not host[f"{a}-resumed"] for a in ARMS}
    K.require(sorted(host) == sorted(f"{a}-{p}" for a in ARMS for p in ("full", "resumed"))
              and all(info[f"{a}-{p}"]["window_delivery"]["mode"] == RI.FORMAL and info[f"{a}-{p}"]["window_delivery"]["taken"] == info[f"{a}-{p}"]["updates"]
                      for a in ARMS for p in ("full", "resumed")) and info["N"]["window_delivery"]["mode"] == "host", "window delivery accounting")
    want = c["workload"]["accept"]["updates"]
    K.require(want == _ACCEPT_UPDATES == 14 and counts["actual_updates"] == want and counts["forwards"] == counts["forward_attempts"] == 2 * want == 28,
              f"acceptance counts {dict(counts)}")
    K.require(all(info[f"{a}-full"]["saved"] is not None and info[f"{a}-full"]["account"]["max_prepared_ahead"] == 1 for a in ARMS)
              and info["N"]["account"]["max_prepared_ahead"] == 0, "prepared-window accounting")
    equivalence = {"schema": "r29-acceptance-equivalence-v1", "contract_sha256": c["contract_sha256"], "scope": SCOPE, "paths": info,
                   "N_against_the_stored_FG_receipts": {"receipts": fu["FG_accept_steps"], "steps": n_steps, "differences": d_n, "equal": not d_n,
                                                        "compared": CC.COMPARED, "excluded": CC.EXCLUDED},
                   "start_state_parts_different_from_N": start_equal, "start_arm_field_differs_from_N": arm_part,
                   "restore_replay": {a: {"steps": steps - save_at, "differences": replay[a], "equal": not replay[a]} for a in ARMS},
                   "input_and_plan_equal_to_the_FG_receipts": plan_equal, "read_active": active, "QP_and_QC_outputs_differ": cross,
                   "window_delivery": {"rule": RI.RULE, "host_receipts": fu["host_accept_steps"], "host_contract_sha256": fu["host_contract_sha256"],
                                       "differences": host, "equal": delivery_equal,
                                       "compared": CC.COMPARED + ["science.r29 (the read readings of every acceptance forward)"],
                                       "steps": {a: {"full": steps, "resumed": steps - save_at} for a in ARMS}},
                   "restore": "K.load compares the complete restored state with the saved one (exact) before the replay",
                   "final_state_digests": {n: r[-1]["science"]["state"] for n, r in paths.items()},
                   "steps_files": {n: K.ref(out / f"STEPS-{n}.jsonl") for n in paths},
                   "updates": want, "complete_detector_forwards": 2 * want, "loss_per_update": {"N": [r["science"]["result"]["loss"] for r in paths["N"]],
                                                                                                 **first_loss},
                   "refused": refusals}
    K.write_new(out / "ACCEPT_EQUIVALENCE.json", equivalence)
    K.write_new(out / "READ_READINGS.json", {"schema": "r29-acceptance-read-readings-v1", "contract_sha256": c["contract_sha256"],
                                             "arms": reads, "note": "recorded from the acceptance forwards themselves (no extra model forward); a description of the "
                                                                    "sampling change, not a gate and not a reason to change the centre rule, the point count or the map"})
    K.require(not d_n, f"N (the unchanged arm FG engine of this release) differs from the stored accept-FG receipts: {d_n[:5]}")
    K.require(all(not v for v in start_equal.values()) and all(arm_part.values()), f"start states differ from the parent state of N: {start_equal}")
    K.require(all(not v for v in replay.values()), f"save / restore replay differs: { {a: v[:5] for a, v in replay.items()} }")
    K.require(all(delivery_equal.values()),
              f"the {RI.FORMAL} window delivery differs from the stored host receipts of release-001: { {n: v[:5] for n, v in host.items() if v} }")
    K.require(all(plan_equal.values()), f"input / plan readings differ from the arm FG receipts: {plan_equal}")
    K.require(all(active.values()) and cross,
              f"the read of an arm did not act (read_active {active}, QP and QC outputs differ {cross}): check the wiring before any formal run")
    # task section 2.4: geometry FP64, sampling coordinates FP32, one dtype pattern per arm, the same in both arms up to the rebuilt memory
    kinds = {a: reads[a]["dtypes"] for a in ARMS}
    K.require(all(len(v) == 1 and v[0]["geometry"] == "torch.float64" and v[0]["sampling_positions"] == v[0]["native_positions"] == "torch.float32"
                  and v[0]["autocast"] == "torch.bfloat16" for v in kinds.values())
              and kinds["QC"][0]["rebuilt_value"] == "torch.float32" and kinds["QP"][0]["rebuilt_value"] is None
              and {k: v for k, v in kinds["QP"][0].items() if k != "rebuilt_value"} == {k: v for k, v in kinds["QC"][0].items() if k != "rebuilt_value"},
              f"dtypes of the read: {kinds}")
    eng = lambda rows: [{"update": r["science"]["input"]["update"], **{k: r["engineering"].get(k) for k in ("cycle_s", "wait_s", "train_call_s", "train_s", "save_s")}}
                        for r in rows]
    timing = {"schema": "r29-acceptance-timing-v1", "contract_sha256": c["contract_sha256"], "per_update": {n: eng(r) for n, r in paths.items()},
              "peak_cuda_allocated_bytes": {n: info[n]["peak_cuda_allocated_bytes"] for n in info},
              "note": "raw wall-clock segments of a few updates with capture=True digests, a full state capture and the read audit on every "
                      "update; an engineering record only: not a throughput measurement and not a method or inference cost"}
    K.write_new(out / "ACCEPT_TIMING.json", timing)
    print(json.dumps({"N_equal_to_the_FG_receipts": True, "restore_replay_equal": True, "read_active": active, "refusals": len(refusals)}), flush=True)
    return {"stage": "accept-R29", "workload": c["workload"]["accept"], "N_equal_to_the_FG_receipts": True,
            "restore_replay_equal": {a: True for a in ARMS}, "read_active": active, "overlap_enabled": {a: True for a in ARMS},
            "window_delivery": {a: {"mode": RI.FORMAL, "equal_to_the_host_receipts": True} for a in ARMS},
            "formal_training_allowed": True, "equivalence": K.ref(out / "ACCEPT_EQUIVALENCE.json"), "timing": K.ref(out / "ACCEPT_TIMING.json"),
            "read_readings": K.ref(out / "READ_READINGS.json"), "read_dtypes": {a: kinds[a][0] for a in ARMS},
            "saved_states": {a: info[f"{a}-full"]["saved"] for a in ARMS},
            "updates": want, "complete_detector_forwards": 2 * want, "refused": refusals,
            "max_displacement_feature_px_first_update": {a: reads[a]["max_displacement_feature_px_first_update"] for a in ARMS},
            "peak_allocated_bytes": {n: info[n]["peak_cuda_allocated_bytes"] for n in info}, "formal_state_usable": False}


# ================================================================ rules of section 6 (pure)
def gate(rule, values, ratios):
    """One gate: primary cells, protection cells and the model-GPU ratio of its pair."""
    x, y = rule["pair"]
    primary = K.cells(values[x], values[y], [tuple(v) for v in rule["primary"]])
    protection = K.cells(values[x], values[y], [tuple(v) for v in rule["protection"]])
    key, limit = rule["model_gpu_ratio"]
    ratio = ratios[key]
    failed = [k for k, v in {**primary, **protection}.items() if not v["passed"]] + ([] if ratio <= limit else ["model_gpu_ratio"])
    return {"pair": rule["reported"], "primary": primary, "protection": protection,
            "model_gpu_ratio": {"pair": key, "value": ratio, "maximum": limit, "passed": ratio <= limit},
            "failed": failed, "passed": not failed,
            "only_protection_or_cost_failed": bool(failed) and all(v["passed"] for v in primary.values())}


def decide(values, ratios):
    """The fixed readings and branch of TASK_R29_001 section 6.  values: arm -> {"600": {...}, "300": {...}};
    ratios: model GPU time ratios keyed "A/B"."""
    n, m = gate(RULES["N_gate"], values, ratios), gate(RULES["M_gate"], values, ratios)
    t = RULES["tradeoff_reading"]
    cells = K.cells(values["QP"], values["QC"], [tuple(v) for v in t["cells"]])
    key, limit = t["model_gpu_ratio"]
    tradeoff = {"model_gpu_ratio": {"pair": key, "value": ratios[key], "maximum": limit, "passed": ratios[key] <= limit}, "cells": cells,
                "label": t["label"] if ratios[key] <= limit and all(v["passed"] for v in cells.values()) else None, "note": t["note"]}
    if n["passed"] and m["passed"]:
        branch, text = "both", "N_gate and M_gate reached: a single-seed method signal; paired seeds and external validation are the controller's decision"
    elif n["passed"]:
        branch, text = "N_only", ("N_gate reached, M_gate not: a preliminary gain of the source-frame / un-zoom adaptation; a precision "
                                  "contribution of the direct read of its own is not established")
    else:
        branch, text = "N_failed", "N_gate not reached: this configuration did not reach the registered net increment; nothing is added automatically"
    return {"N_gate": n, "M_gate": m, "tradeoff_reading": tradeoff, "branch": branch, "label": RULES["branches"][branch], "reading": text,
            "beats_C_but_not_N": bool(m["passed"] and not n["passed"]),
            "notes": [s for s, on in (
                ("only protection cells or the cost cell of N_gate failed: registered as not reached, with the actual trade-off shown",
                 n["only_protection_or_cost_failed"]),
                ("P is above C but not above N: not a detection gain", m["passed"] and not n["passed"])) if on]}


def summary_r29(c, values, apl, apm, cost):
    """values / apl / apm: arm -> stored readings of FG, QC, QP.  Pure arithmetic."""
    import r28_eval as E
    K.require(set(values) == set(apl) == set(apm) == {"FG", "QC", "QP"}, "summary arms")
    pairs = (("QP", "FG"), ("QC", "FG"), ("QP", "QC"))
    name = lambda x, y: f"{SHORT[x]}-{SHORT[y]}"
    ratios = cost["ratios"]
    keys = ("QP/FG", "QC/FG", "QP/QC")
    return {"arms": {"N": "FG", "C": "QC", "P": "QP"}, "values": {SHORT[a]: values[a] for a in ("FG", "QC", "QP")},
            "value_sources": {"N": "009 round eval-FG (quoted, not rescored)", "C": "this round, eval-QC (new)", "P": "this round, eval-QP (new)"},
            "differences_pp": {name(x, y): E.difference_pp(values[x], values[y]) for x, y in pairs},
            "decision": decide(values, ratios["model_gpu"]),
            "APL6_pp": {name(x, y): E.apl6(apl[x], apl[y]) for x, y in pairs},
            "per_class_APL_top600": FC.size_table({SHORT[a]: apl[a] for a in ("FG", "QC", "QP")}, "APL"),
            "per_class_APM_top600": FC.size_table({SHORT[a]: apm[a] for a in ("FG", "QC", "QP")}, "APM"),
            "cost": {"model_gpu": {f"{SHORT[k.split('/')[0]]}/{SHORT[k.split('/')[1]]}": ratios["model_gpu"][k] for k in keys},
                     "end_to_end": {f"{SHORT[k.split('/')[0]]}/{SHORT[k.split('/')[1]]}": ratios["end_to_end"][k] for k in keys},
                     "all_ratios": ratios, "ratios_per_pass": cost.get("ratios_per_pass"), "medians": cost["medians"],
                     "stages": cost.get("stages"), "order_executed": cost.get("order_executed"), "exclusive_gpu": cost.get("exclusive_gpu"),
                     "source": "cost-R29 (N, C, P in one process)"}}


def _trial_summary(c, cost=None):
    """summary_r29 with the stored N evaluation standing in for both new arms (wiring only: every difference is 0)."""
    ev = K.stage_closed(c, "eval-FG", "FG")
    if cost is None:
        arms = K.COST_ENDPOINTS["cost-R29"]
        one = {f"{a}/{b}": 1.0 for a in arms for b in arms if a != b}
        cost = {"ratios": {"model_gpu": dict(one), "end_to_end": dict(one)}, "medians": {}}
    values = {a: ev["values"] for a in ("FG", "QC", "QP")}
    trial = summary_r29(c, values, {a: ev["per_class_APL_top600"] for a in values}, {a: ev["per_class_APM_top600"] for a in values}, cost)
    K.require(all(v == 0.0 for d in trial["differences_pp"].values() for b in d.values() for v in b.values())
              and trial["decision"]["branch"] == "N_failed" and trial["decision"]["N_gate"]["failed"][0] == "top600_APS", "summary arithmetic")
    json.dumps(trial, allow_nan=False)
    return trial


def audit_fold(d):
    """The sampled read readings of a formal run: per audited update the per-layer means of both forwards."""
    rows = []
    with (Path(d) / "UPDATES.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("r29") is not None:
                rows.append({"update": row["input"]["update"], "epoch": row["input"]["epoch"], "window": row["input"]["window"],
                             "max_displacement_feature_px": _displacement(row["r29"]), "summary": _read_summary([row["r29"]]),
                             "location_gradient": [[layer.get("location_gradient") for layer in fwd["layers"]] for fwd in row["r29"]["forwards"]]})
    return {"audited_updates": len(rows), "rule": "the first window of every epoch and the last update of the run (sampled windows; "
                                                 "not extrapolated to every update)", "rows": rows}


# ================================================================ summary-R29
def summarize(c, out, counts):
    import r28_eval as E
    from sparse_rtdetr.research import r29_unzoom as UZ
    fu = K.followup(c)
    K.require(K.r29(c) and K.ARM in ARMS, "the R29 round is summarised with R28_ARM=QP or QC")
    K.PHASE["phase"] = "summarize/read"
    out = E.round_stage(out, "summary-R29")
    ev = {"FG": K.stage_closed(c, "eval-FG", "FG"), **{a: K.stage_closed(c, f"eval-{a}", a) for a in ARMS}}
    cost, accept, ncheck = K.stage_closed(c, "cost-R29"), K.stage_closed(c, "accept-R29"), K.stage_closed(c, "ncheck-FG")
    ws = cost["weights_sha256"]
    K.require(all(ws[a] == ev[a]["weights_sha256"] for a in ev) and ws["FG"] == fu["FG_weights_sha256"] == ncheck["weights_sha256"],
              "the cost-R29 stage timed other models than the evaluated ones")
    lock = K.selected_width(c)
    K.require(all(ev[a]["demand"] == ev["FG"]["demand"] and ev[a]["mapping"] == ev["FG"]["mapping"] and ev[a]["read_mode"] == UZ.READ[a]
                  and ev[a]["GT_counts"] == ev["FG"]["GT_counts"] and ev[a]["global_update"] == K.ENDPOINT and ev[a]["kind"] == "EMA" for a in ARMS)
              and ev["FG"]["mapping"]["fwhm"] == lock["fwhm"], "the three arms were not evaluated with one demand source, width and GT set")
    for a in ARMS:
        for b in K.BUDGETS:
            K.pinned(ev[a]["metrics"][str(b)]), K.pinned(ev[a]["predictions"][str(b)])
    K.PHASE["phase"] = "summarize/compute"
    res = summary_r29(c, {a: ev[a]["values"] for a in ev}, {a: ev[a]["per_class_APL_top600"] for a in ev},
                      {a: ev[a]["per_class_APM_top600"] for a in ev}, cost)
    tables, formal, observations, geometry, audits = {}, {}, {}, {}, {}
    for a in ("FG",) + ARMS:
        t = FC.tensor_tables(K.pinned(ev[a]["cocoeval_top600_tensors"]))
        f600 = ev[a]["values"]["600"]
        tables[SHORT[a]] = {**t, "source": ev[a]["cocoeval_top600_tensors"],
                            "macro_minus_scored_value": {k: (None if t["macro"][k] is None else t["macro"][k] - f600[k]) for k in t["macro"]}}
    for a in ARMS:
        d = K.formal_dir(c, a)
        closed = K.read_json(d / "CLOSED.json")
        K.require(closed["plan"]["mapping"] == ev[a]["mapping"] and closed["plan"]["read_mode"] == UZ.READ[a]
                  and closed["rows_equal_to_the_arm_FG_ledger"] + closed["ledger_rows_copied_on_resume"] == K.ENDPOINT - K.PARENT_UPDATE,
                  f"formal run of arm {a}: mapping / read mode / rows compared with the arm FG ledger")
        formal[SHORT[a]] = {"directory": d.name, "resumed": closed["resumed"], "updates": closed["updates"], "input_pipeline": closed["plan"]["input_pipeline"],
                            "window_delivery": closed["plan"]["window_delivery"],
                            "loop_wall_seconds": closed["loop_wall_seconds"], "update_seconds_sum": closed["update_seconds_sum"],
                            "seconds_per_update": closed["loop_wall_seconds"] / closed["updates"], "peak_cuda_bytes": closed["peak_cuda_bytes"],
                            "rows_equal_to_the_arm_FG_ledger": closed["rows_equal_to_the_arm_FG_ledger"],
                            "audited_updates": closed["audited_updates"], "endpoint": ev[a]["endpoint"], "weights_sha256": ev[a]["weights_sha256"]}
        observations[SHORT[a]] = FC.formal_observations(d)
        geometry[SHORT[a]] = GC.schedule_geometry(d)["overall"]
        audits[SHORT[a]] = audit_fold(d)
    stages = ["r29-check", "wire-R29", "ncheck-FG", "accept-R29", *[K.formal_dir(c, a).name for a in ORDER], *[f"eval-{a}" for a in ORDER], "cost-R29"]
    res.update(status="PASS", registration=fu["registration"], authorization_basis=c["authorization_basis"], task=fu["task"],
               decision_file=fu["decision"], contract_sha256=c["contract_sha256"], source_commit=c["source_commit"],
               r29_read_definition=c["r29_read_definition"], identity_sha256={SHORT[a]: c["identity_sha256"][a] for a in ("FG",) + ARMS},
               rules=c["r29_rules"], N_reference={"round": fu["FG_round"], "contract_sha256": fu["FG_contract_sha256"], "result": fu["FG_result"],
                                                  "endpoint": fu["FG_endpoint"], "weights_sha256": fu["FG_weights_sha256"],
                                                  "predictions": fu["FG_predictions"], "metrics": fu["FG_metrics"],
                                                  "present_loading_path_check": {k: ncheck[k] for k in ("images", "batches", "field_by_field_equal", "scored")},
                                                  "note": "quoted unchanged; not retrained, not rescored"},
               per_class_top600=tables, formal=formal, training_observations=observations, map_geometry_over_the_schedule=geometry,
               read_readings={"acceptance": accept["read_readings"], "formal_sampled_windows": audits},
               acceptance={"equivalence": accept["equivalence"], "timing": accept["timing"], "read_active": accept["read_active"],
                           "N_equal_to_the_FG_receipts": accept["N_equal_to_the_FG_receipts"]},
               stages=E._stage_account(c, stages), other_directories=E._other_directories(), statements=STATEMENTS)
    K.require(counts["forwards"] == 0 and counts["actual_updates"] == 0 and not counts.get("score_budget_calls"),
              "summary must not run a model or a scorer")
    K.write_new(out / "RESULT.json", res)
    d = res["decision"]
    diff = res["differences_pp"]
    return {"result": K.ref(out / "RESULT.json"), "registration": fu["registration"], "branch": d["branch"], "label": d["label"],
            "N_gate_passed": d["N_gate"]["passed"], "N_gate_failed": d["N_gate"]["failed"], "M_gate_passed": d["M_gate"]["passed"],
            "M_gate_failed": d["M_gate"]["failed"], "tradeoff_label": d["tradeoff_reading"]["label"],
            "P_minus_N_APS600_pp": diff["P-N"]["600"]["APS"], "P_minus_N_AP75_small600_pp": diff["P-N"]["600"]["AP75_small"],
            "C_minus_N_APS600_pp": diff["C-N"]["600"]["APS"], "P_minus_C_APS600_pp": diff["P-C"]["600"]["APS"],
            "model_gpu": res["cost"]["model_gpu"]}
