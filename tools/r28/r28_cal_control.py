"""R28-F* calibrated mapping control (TASK_R28_CALIBRATED_F_CONTROL_006) with the input-pipeline acceptance of
ENGINEERING_INPUT_OVERLAP_NEXT_005.  An exploratory control designed after the W, K and F results were seen; nothing
of the closed W round, of the K round or of the F round is rerun or written, and their decisions and readings stay.

Arm FS = arm F (same frozen demand network, parent, canvas, detector, data, schedule, map rule) with the attraction
kernel FWHM changed from 4 to one global scalar h*, calibrated once on training-set geometry
(src/sparse_rtdetr/research/r28_fovea_cal.py holds the map and the selection rule).

  prepare       contract of the independent round: authorization basis, task / review / engineering files, read-only
                references into the W, K and F rounds, the release difference (the science tree gains one module).
  cal_check     CPU, no model forward: unit tests; identities, authorisation and refusals; the official rule executed
                from the pinned official file at several widths; the serial and the overlapped input path on the real
                loader (bitwise equal windows, epoch boundary, the 117 -> 118 augmentation switch, restart from the
                committed cursor with a prepared window dropped, wrong digest / identity refused before consumption).
  calibrate     CUDA for the demand network only: epoch 61 (all 304 windows) cached once, every width of the fixed
                table read on the cache, the selection rule, the lock; epoch 118 (32 windows) confirms validity only.
                Zero detector forwards.  A calibration that is not matched closes the stage and ends the chain.
  cal_fixture   CPU: geometry of the locked width on the real first window and the fixture u1 / u2 / restore / replay.
  cal_wire      CUDA wiring of code that otherwise runs only after training: one evaluation batch of arm FS and a
                shortened five-arm staged timing; nothing is scored.
  pipeline      CUDA acceptance of the input pipeline from the e060 parent: 32 serial updates, 32 overlapped updates
                with a full state saved after update 16 while a prepared window exists, 16 updates replayed from that
                state.  Bitwise comparison step by step; wall-clock segments and memory; the fixed decision rule.
  summarize     summary-FS, RECORD_ONLY: W - F*, F* - F, F* - L0 and the costs; the calibration and its match over
                the complete schedule.  No pass / fail gate.
Formal training, evaluation and timing are r28_loop.formal, r28_eval.evaluate and r28_cost.cost with R28_ARM=FS.
"""
import gc
import hashlib
import json
import math
import os
import queue
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import r28_common as K
import r28_followup as FU
import r28_fovea_control as FC

REGISTRATION = "R28-F* exploratory calibrated-width mapping control after the W, K and F results"
ARM = "FS"
F_ROUND = K.LD_ROOT / "r28-fovea-map-control-20261114-001"
TASK = ("TASK_R28_CALIBRATED_F_CONTROL_006.md", "a627a7e2119f9946ea8e4c31d470fdb544f955e4a7b01de6d6453a154465acf2")
REVIEW = ("REVIEW_R28_L0_W_K_F_006.md", "54864e4ebd91b3599b72a4a517b81a0bf1ef51063bedc60acf295c67f1b834b7")
ENGINEERING = ("ENGINEERING_INPUT_OVERLAP_NEXT_005.md", "23b93f183a5ee14c2ac8f95b80f69e30302cf3aa81f8265382481a2b2c027840")
PIN_F = {  # TASK_R28_CALIBRATED_F_CONTROL_006 section 1 and the F round's own receipts
    "commit": "e78e8cbe310084da5523466833086796c91ea554",
    "manifest": "ca0fc9cba3a7d1666a98318119b356365bd499d6d6caad7a2d2ac3c2a5d3e6c8",
    "contract_file": "5e3d847257857ec09d81cc09659c5af8ac5434f83ab0b5cdd515526feae8a513",
    "contract": "a0b44e15d0ac0d89eb91c9840640214927f67690efc892035aacc8ce4c0988bf",
    "result": "db8219c5a63f32c30178e961e354b7ac2a4551acab72242d3165ddfe77f236a9",
    "endpoint": "a0dc7d6dec0bd3ed95490c4a20ceb9b8d2926336c2036575febe98a0f8386798",
    "parent": "a2b6412e3c85de7592829f24aff09e355fdd11cb53798bc498bd04b882e1bf8a",
    "W_demand": "c9ffc7505e47adbcc45d70949c3d87371c6f3a7795f1aaaab886d98b15ee9e5a",
    "official_commit": "c8707bd2e4f10678f6636d24c2915a43b44a59ed",
}
INHERITED_F = FC.OWN_STAGES
OWN_STAGES = ("cal-check", "calibrate", "cal-fixture", "wire-FS", "pipeline", "formal-FS", "eval-FS", "cost-FS", "summary-FS")
ALLOWED_CHANGED = ("tools/r28/r28_common.py", "tools/r28/r28_loop.py", "tools/r28/r28_eval.py", "tools/r28/r28_cost.py",
                   "tools/r28/r28_run.py", "tests/test_r28_fovea_control.py")
ALLOWED_ADDED = ("src/sparse_rtdetr/research/r28_fovea_cal.py", "tests/test_r28_fovea_cal.py", "tests/test_r28_pipeline.py",
                 "tests/test_r28_cal_control.py", "tools/r28/r28_cal_control.py", "tools/r28/r28_pipeline.py",
                 "tools/r28/r28_cal_chain.sh")
SCIENCE_ADDED = ("src/sparse_rtdetr/research/r28_fovea_cal.py",)
TESTS = FC.TESTS + ("tests/test_r28_fovea_cal.py", "tests/test_r28_pipeline.py", "tests/test_r28_cal_control.py")
ARMS5 = ("L0", "K", "F", "FS", "W")
PAIRS = (("W", "FS"), ("FS", "F"), ("FS", "L0"), ("FS", "K"), ("W", "F"), ("W", "L0"), ("W", "K"), ("F", "L0"), ("K", "L0"))
PRIMARY = (("600", "APS"), ("600", "AP75_small"))
COST_PAIRS = ("FS/L0", "W/L0", "F/L0", "K/L0", "W/FS", "FS/W", "FS/F", "W/F", "W/K")
WIRE = {"eval_batches": 1, "cost_warmup": 1, "cost_batches_per_pass": 1, "cost_passes": 1}
PIPE = {"steps": 32, "save_at": 16, "steady_from": 5, "ready_limit": 1, "workers": 2}
OFFICIAL_WIDTHS = (4.0, 13.0, 30.0)
TOL = {"fp64_same_formula": 1e-10, "official_fp32_nodes": 1e-5, "round_trip": 1e-9, "render_fp32_vs_fp64": 1e-5,
       "historical_ledger_geomean": 1e-6}
E118_START = 117 * K.WINDOWS_PER_EPOCH
WORKLOAD = {
    "cal-check": {"device": "cpu", "updates": 0, "forwards": 0, "demand_forwards": 0,
                  "note": "unit tests in a subprocess, identities, the official rule at fixed widths, pure-data checks of the "
                          "input pipeline on the real loader; no model"},
    "calibrate": {"updates": 0, "forwards": 0, "demand_forward_images": 4864 + 512, "detector_forwards": 0,
                  "note": "the frozen demand network once per image occurrence; widths are read on the cache"},
    "cal-fixture": {"device": "cpu", "updates": 3, "forwards": 6,
                    "note": "CPU fixture (2 real images, FP32, u1 / u2 / replay) with the locked width; listed apart"},
    "wire": {"updates": 0, "forwards": WIRE["eval_batches"] + 5 * (WIRE["cost_warmup"] + WIRE["cost_passes"] * WIRE["cost_batches_per_pass"]),
             "scorer_calls": 0, "note": "wiring of the evaluation batch and of the five-arm staged timing; listed apart"},
    "pipeline": {"updates": PIPE["steps"] * 2 + (PIPE["steps"] - PIPE["save_at"]),
                 "forwards": 2 * (PIPE["steps"] * 2 + (PIPE["steps"] - PIPE["save_at"])),
                 "paths": {"serial": PIPE["steps"], "overlap": PIPE["steps"], "resumed": PIPE["steps"] - PIPE["save_at"]}},
    "formal": K.WORKLOAD["formal"],
    "evaluation": K.WORKLOAD["evaluation"],
    "cost-FS": K.WORKLOAD["cost-FS"],
    "planned_updates": PIPE["steps"] * 2 + (PIPE["steps"] - PIPE["save_at"]) + (K.ENDPOINT - K.PARENT_UPDATE),
    "planned_complete_detector_forwards": 2 * (PIPE["steps"] * 2 + (PIPE["steps"] - PIPE["save_at"])) + 2 * (K.ENDPOINT - K.PARENT_UPDATE)
    + (K.EVAL_WARMUP + K.EVAL_BATCHES) + len(K.COST_ENDPOINTS["cost-FS"]) * K.COST_PER_ENDPOINT,
    "listed_apart": ["cal-fixture (3 updates, 6 forwards, CPU)", "wire (11 forwards)", "dev rounds", "any failure or re-verification"],
    "not_run": ["demand network training", "W / K / F / L0 training", "W / K / F / L0 evaluation or rescoring", "a second width",
                "any seed, schedule, microbatch or medium-module variation", "holdout access"],
}
PIPELINE_RULE = {
    "version": "r28-input-pipeline-v1", "workers": PIPE["workers"], "ready_limit": PIPE["ready_limit"],
    "acceptance": {"parent": "e060", "windows": "epoch 61 from its first window", "serial_updates": PIPE["steps"],
                   "overlap_updates": PIPE["steps"], "save_after_update": PIPE["save_at"], "replayed_updates": PIPE["steps"] - PIPE["save_at"],
                   "compared": "input record, augmentation digests, outputs / DN noise / loss / gradient digests, observer row, "
                               "model with BN buffers and modes, optimizer, EMA, engine clock, CPU and CUDA RNG, step by step",
                   "steady_state": f"updates {PIPE['steady_from']}..{PIPE['steps']} of each path"},
    "decision": "the overlapped path is used for the formal run only if every compared field of the three paths is equal, the "
                "restore is exact, the measured steady-state cycle of the overlapped path (acceptance-only work removed) is "
                "shorter than the serial one, and the conservative bound max(consumer work, producer work) is shorter too; "
                "otherwise the serial reference path runs the same schedule",
    "not_changed": ["workers=2, spawn, prefetch_factor=2, the loader generator", "augmentation inside the worker processes",
                    "every per-image and whole-window hash and its format", "the reference ledger comparison before consumption",
                    "host-to-device copy, pinned memory, precision, microbatch, BN, DN"],
}
STATEMENTS = {
    "registration": "本对照形成于看过 W、K、F 结果之后，是探索性控制；不是原 W 的预注册检验，也不追认原 W。",
    "W_decision": "旧 W 的 closed_fixed_config（APM600 未过）、K_trigger=false、旧成本 C，以及 K、F 的读数原样保留；F* 的任何读数都不改写它们。",
    "scope": "F* 只把 F 的全局高斯吸引核 FWHM 由 4 改为一个标量 h*（两轴同值），h* 仅由训练集 epoch61 的需求图与真值几何标定，"
             "使 small 真值的池化平均对数倍率接近 W；不用检测器预测或 AP 选核宽，训练中不再调整。",
    "not_isolated": "平均倍率匹配不等于逐实例分布、位置、各向异性、尾部、medium／large、梯度或采样足迹匹配。"
                    "这是训练集单参数标定后的更强映射控制，不是官方 FOVEA 原样复现，也不是硬界的纯单因素消融。",
    "posthoc": "h* 的标定方案是在读过 F 结果之后设计的。",
    "record_only": "W−F* 等比较登记为 RECORD_ONLY：给出全部绝对值、差值、成本与局限，不设新的通过门；差值小不构成统计等效。",
    "contribution": "方法贡献仍为 not_established：单 seed（seed0）、结果后追加、重复使用的 548 图 development 集。",
    "cost": "五端点同进程测速的新比值单列，不覆盖 W 轮登记的 C，也不与 K 轮、F 轮各自的同进程比值混为同一次测量；"
            "训练输入流水线的提速是训练工程项，不计入任何方法的推理收益。",
    "large": "large 读数由少数稀疏类主导，不作整体优劣依据；APL6 只是补充读数。",
}
LABEL = "calibrated_mapping_control_completed"
CONTRIBUTION = "not_established"
DEV_CAL = "R28_DEV_CAL_WINDOWS"
DEV_PIPE = "R28_DEV_PIPE"
DEV_TARGET = "R28_DEV_CAL_TARGET_WIDTH"   # dev rounds only: the target is arm F's rule at this width (wiring, never a result)


# ================================================================ release difference
def source_delta(old_files, new_files, *, label="the F round release"):
    """File-by-file difference between the F round's release and this one.  The science tree may only gain the arm FS
    module: no file of src/, vendor/ or configs/ is changed or removed (arm F's module stays byte-identical)."""
    changed = sorted(k for k in old_files if k in new_files and old_files[k] != new_files[k])
    added = sorted(k for k in new_files if k not in old_files)
    removed = sorted(k for k in old_files if k not in new_files)
    K.require(not removed and set(changed) <= set(ALLOWED_CHANGED) and set(added) <= set(ALLOWED_ADDED),
              f"release differs from {label} outside the arm FS adapter: changed {changed}, added {added}, removed {removed}")
    science = ("src/", "vendor/", "configs/")
    K.require(not any(p.startswith(science) for p in changed) and all(p in SCIENCE_ADDED for p in added if p.startswith(science)),
              f"release touches the science tree of {label} beyond the added arm FS module")
    return {"against": label, "changed": changed, "added": added, "removed": removed, "unchanged_files": len(old_files) - len(changed),
            "existing_science_files_identical": True, "science_files_added": [p for p in added if p.startswith(science)],
            "note": "every existing file of src/, vendor/ and configs/ (arm W / K / F operators and wrappers, detector, criterion, "
                    "data) is byte-identical; arm FS is one added module"}


# ================================================================ references into the F round (read-only)
def f_references(f_round=None):
    f_round = Path(f_round or F_ROUND).resolve()
    contract = K.ref(f_round / "CONTRACT.json")
    K.require(contract["sha256"] == PIN_F["contract_file"], "F round CONTRACT.json file SHA")
    cf = K.read_json(f_round / "CONTRACT.json")
    body = {k: v for k, v in cf.items() if k != "contract_sha256"}
    K.require(cf["contract_sha256"] == K.sha_bytes(K.canonical(body)) == PIN_F["contract"], "F round contract digest")
    K.require(cf["source_commit"] == PIN_F["commit"] and cf["source_manifest"]["sha256"] == PIN_F["manifest"]
              and cf["authorization_basis"] == K.FOVEA_BASIS and Path(cf["outputs"]).resolve() == f_round
              and cf["parent"]["sha256"] == K.PARENT_SHA == PIN_F["parent"]
              and cf["followup"]["official"]["commit"] == PIN_F["official_commit"], "F round contract identities")
    K.pinned(cf["source_manifest"])
    inherited = {}
    for name in INHERITED_F:
        r = K.ref(f_round / name / "CLOSED.json")
        closed = K.read_json(r["path"])
        K.require(closed.get("status") == "PASS" and closed.get("contract_sha256") == cf["contract_sha256"],
                  f"F round stage {name} is not PASS under the F contract")
        inherited[name] = r
    summary, ev = K.read_json(inherited["summary-F"]["path"]), K.read_json(inherited["eval-F"]["path"])
    K.require(summary["result"]["sha256"] == PIN_F["result"] and ev["endpoint"]["sha256"] == PIN_F["endpoint"] and ev["arm"] == "F"
              and Path(ev["endpoint"]["path"]).parent == f_round / "formal-F" and ev["demand"]["file"]["sha256"] == PIN_F["W_demand"]
              and cf["followup"]["W_demand"]["sha256"] == PIN_F["W_demand"], "F result / endpoint / demand network SHA")
    for r in (summary["result"], ev["endpoint"], ev["cocoeval_top600_tensors"], *ev["predictions"].values(), *ev["metrics"].values()):
        K.pinned(r)
    ledger = K.read_json(K.side_of(ev["endpoint"]["path"]))["audit"]["ledger"]
    w_ledger = K.read_json(K.pinned(cf["followup"]["W_endpoint_side"]))["audit"]["ledger"]
    K.require(Path(ledger["path"]) == f_round / "formal-F" / "UPDATES.jsonl", "F ledger location")
    K.pinned(ledger), K.pinned(w_ledger)
    return cf, inherited, {"F_round": str(f_round), "F_contract": contract, "F_contract_sha256": cf["contract_sha256"],
                           "F_commit": cf["source_commit"], "F_manifest": cf["source_manifest"], "F_result": summary["result"],
                           "F_endpoint": ev["endpoint"], "F_weights_sha256": ev["weights_sha256"],
                           "F_predictions": ev["predictions"], "F_metrics": ev["metrics"],
                           "F_identity_sha256": cf["identity_sha256"]["F"], "F_ledger": ledger, "W_ledger": w_ledger}


def _earlier(cf):
    """The W and K round references of the F contract, verified again through those rounds' own pins."""
    fu0 = cf["followup"]
    c1, refs = FU.prior_references(fu0["prior_round"])
    ck, k_inherited, k_refs = FC.k_references(fu0["K_round"])
    K.require(all(fu0[k] == v for k, v in refs.items() if k != "inherited") and all(fu0[k] == v for k, v in k_refs.items())
              and fu0["inherited"] == {**refs["inherited"], **k_inherited}, "the F contract's W / K references changed")
    return c1, ck, fu0


# ================================================================ contract of the independent round
def prepare(_args=None):
    from sparse_rtdetr.research import r28_fovea as FV
    from sparse_rtdetr.research import r28_fovea_cal as FS
    from sparse_rtdetr.research import r28_resample as R
    from sparse_rtdetr.research import r28_warp as W
    round_ = K.ROUND.resolve()
    K.require(K.ROOT.resolve().parent.parent == round_ and K.ROOT.name == "source", "prepare-cal must run from a frozen release")
    K.require("R28_ROUND" in os.environ and all(round_ != p and p not in round_.parents
                                                for p in (FU.PRIOR_ROUND.resolve(), FC.K_ROUND.resolve(), F_ROUND.resolve())),
              "the calibrated mapping-control round must be an independent directory named by R28_ROUND")
    K.require(not K.DEV or round_.parent.parent == K.LD_ROOT.resolve(), "a dev round lives inside the round directory")
    manifest = K.read_json(K.ROOT.parent / "MANIFEST.json")
    K.require(manifest["files"] == K.B.source_manifest(K.ROOT), "release source differs from MANIFEST")
    K.require(not manifest.get("dirty", False) or K.DEV, "a release assembled from a dirty work tree is only accepted in a dev round")
    task, review, engineering = (K.ref(K.ROUND / f[0]) for f in (TASK, REVIEW, ENGINEERING))
    K.require(task["sha256"] == TASK[1] and review["sha256"] == REVIEW[1] and engineering["sha256"] == ENGINEERING[1],
              "task / review / engineering file SHA")
    cf, f_inherited, f_refs = f_references()
    c1, ck, fu0 = _earlier(cf)
    K.require(set(fu0["inherited"]).isdisjoint(f_inherited) and set(OWN_STAGES).isdisjoint(set(fu0["inherited"]) | set(f_inherited)),
              "stage names of the four rounds overlap")
    delta = source_delta(K.read_json(cf["source_manifest"]["path"])["files"], manifest["files"])
    K.require(cf["identity_sha256"] == {**R.IDENTITY_SHA256, **FV.IDENTITY_SHA256} and cf["fovea_definition"] == FV.DEFINITION
              and cf["definition"] == W.DEFINITION and cf["identity"]["F"] == FV.IDENTITY["F"],
              "arm W / K / F behaviour identity differs from the F round")
    keep = {k: v for k, v in fu0.items() if k not in ("registration", "authorization_basis", "arm", "task", "review", "source_delta",
                                                      "trained_here", "reused_frozen", "analysis", "statements", "inherited",
                                                      "inherited_contracts")}
    c = {k: v for k, v in cf.items() if k != "contract_sha256"}
    c.update(version="R28-FS-control-006-contract-v1", authorization_basis=K.CAL_BASIS,
             source_root=str(K.ROOT.resolve()), source_manifest=K.ref(K.ROOT.parent / "MANIFEST.json"),
             source_commit=manifest["commit"], source_dirty=bool(manifest.get("dirty", False)),
             identity={**cf["identity"], **FS.IDENTITY}, identity_sha256={**cf["identity_sha256"], **FS.IDENTITY_SHA256},
             fovea_cal_definition=FS.DEFINITION, workload=WORKLOAD, input_pipeline=PIPELINE_RULE,
             cost={**cf["cost"], "cost-FS": {"endpoints": list(K.COST_ENDPOINTS["cost-FS"]),
                                             "rotations": [list(r) for r in K.COST_ROTATIONS["cost-FS"]]}},
             labels={**cf["labels"], "FS_completed": LABEL, "FS_contribution": CONTRIBUTION,
                     "FS_calibration_statuses": list(FS.STATUSES)},
             resource_plan={"state_bytes": cf["parent"]["bytes"], "formal_states": len(K.SAVE_EPOCHS),
                            "disk_free_bytes_at_prepare": shutil.disk_usage(K.ROUND).free,
                            "gpu_time_reference": "serial F measured 11.29 h; 6-8 h with the overlapped input path is an "
                                                  "unverified estimate; a wider kernel may change the resampling time; not a cap"},
             outputs=str(K.ROUND), formal_started=False,
             followup={**keep, "registration": REGISTRATION, "authorization_basis": K.CAL_BASIS, "arm": ARM, "task": task,
                       "review": review, "engineering": engineering, **f_refs,
                       "inherited": {**fu0["inherited"], **f_inherited},
                       "inherited_contracts": {**fu0["inherited_contracts"], **{name: cf["contract_sha256"] for name in f_inherited}},
                       "source_delta": delta,
                       "trained_here": ["F* detector, 18,240 updates from the e060 parent", "80 acceptance updates of the input "
                                        "pipeline (discarded states)"],
                       "reused_frozen": ["W demand network (demand/demand-net.pth of the closed W round): the demand source of F*",
                                         "W / K / F e120 endpoints (timing and identity only)", "W / K / F / L0 predictions, "
                                         "metrics and per-class readings (quoted, not rescored)"],
                       "analysis": {"record_only": True, "primary": [f"W-FS top{b} {m}" for b, m in PRIMARY],
                                    "pairs": [f"{x}-{y}" for x, y in PAIRS]},
                       "statements": STATEMENTS})
    c["contract_sha256"] = K.sha_bytes(K.canonical(c))
    K.write_new(K.ROUND / "CONTRACT.json", c)
    print(json.dumps({"contract_sha256": c["contract_sha256"], "identity_sha256_FS": FS.IDENTITY_SHA256["FS"],
                      "authorization_basis": c["authorization_basis"],
                      "source_delta": {k: delta[k] for k in ("against", "changed", "added")}}))
    return c


# ================================================================ the official rule at a given width
def official_check(fu, demands, h):
    """Production nodes of width h against the pinned official file run unchanged (its own FP32) and against the FP64
    loop reference written apart from the production code."""
    from sparse_rtdetr.research import r28_fovea as FV
    from sparse_rtdetr.research import r28_fovea_cal as FS
    from sparse_rtdetr.research import r28_warp as W
    torch = K.torch_mod()
    with FC.official_module(K.pinned(fu["official"]["grid_generator"])) as m:
        gen = m.RecasensSaliencyToGridMixin(output_shape=tuple(FS.GRID_SHAPE), separable=True, attraction_fwhm=h, anti_crop=True)
    K.require(tuple(gen.grid_shape) == FS.GRID_SHAPE and gen.padding_size == FS.PAD and gen.padding_mode == "reflect"
              and tuple(gen.filter.shape) == (1, 1, 2 * FS.PAD + 1), "official generator constants")
    kernel = FS.kernel_matrix(FS.GRID_SHAPE[0], h)[0, :FS.PAD + 1].flip(0)
    kernel_error = float((gen.filter[0, 0, FS.PAD:].double() - kernel).abs().max())
    K.require(kernel_error <= 1e-7, f"attraction kernel of width {h} differs from the official filter: {kernel_error}")
    worst = {"official_fp32_nodes": 0.0, "fp64_same_formula": 0.0}
    cases = {}
    for name, demand in demands.items():
        d = demand.to(torch.float64) + W.EPS
        b = d.shape[0]
        sx, sy = FV.node_saliency(d.sum(1), FS.GRID_SHAPE[1]), FV.node_saliency(d.sum(2), FS.GRID_SHAPE[0])
        (raw_x, gx), (raw_y, gy) = FS.attraction_nodes(sx, h), FS.attraction_nodes(sy, h)
        grid = gen.separable_saliency_to_grid(torch.zeros(b, 1, 1, 1), None, sx.float()[:, None], sy.float()[:, None], "cpu")
        K.require(tuple(grid.shape) == (b, *FS.GRID_SHAPE, 2), "official grid shape")
        off = {"x": (grid[:, 0, :, 0].double() + 1) / 2, "y": (grid[:, :, 0, 1].double() + 1) / 2}
        row = {"images": b}
        for axis, g, raw, s in (("x", gx, raw_x, sx), ("y", gy, raw_y, sy)):
            ref = torch.tensor([FC.reference_nodes(s[i].tolist(), h, FS.PAD) for i in range(b)], dtype=torch.float64)
            e_nodes, e_ref = float((off[axis] - g).abs().max()), float((ref - raw).abs().max())
            step = (g[:, 1:] - g[:, :-1]) * (g.shape[1] - 1)
            row[axis] = {"official_fp32_nodes_max": e_nodes, "fp64_same_formula_max": e_ref,
                         "node_slope": [float(step.min()), float(step.max())],
                         "endpoint_raw": [float(raw[:, 0].abs().max()), float((raw[:, -1] - 1).abs().max())]}
            worst["official_fp32_nodes"], worst["fp64_same_formula"] = max(worst["official_fp32_nodes"], e_nodes), max(worst["fp64_same_formula"], e_ref)
        cases[name] = row
    K.require(worst["official_fp32_nodes"] <= TOL["official_fp32_nodes"] and worst["fp64_same_formula"] <= TOL["fp64_same_formula"],
              f"official / reference comparison outside tolerance at width {h}: {worst}")
    return {"width": h, "width_hex": h.hex(), "kernel_max_abs_difference_to_official_fp32_filter": kernel_error, "cases": cases,
            "worst": worst}


# ================================================================ pure-data checks of the input pipeline (real loader)
def reference_check(refs):
    import l0_loop as LL

    def check(inp):
        K.require(inp["update"] in refs, f"window {inp['update']} is not an update of the schedule")
        bad = LL.input_differences(inp, refs[inp["update"]]["input"])
        K.require(not bad, f"input record differs from R27 formal-P before update {inp['update']} "
                           f"(the model was not updated with this window): {bad}")

    return check


def _same_window(a, b):
    torch = K.torch_mod()
    if a.inp != b.inp or not torch.equal(a.x, b.x) or len(a.t) != len(b.t):
        return False
    for ta, tb in zip(a.t, b.t):
        if set(ta) != set(tb):
            return False
        for k in ta:
            same = torch.equal(ta[k], tb[k]) if isinstance(ta[k], torch.Tensor) else ta[k] == tb[k]
            if not same:
                return False
    return True


def data_checks(c, refs):
    """The serial and the overlapped path on the real spawn loader, without any model: equal windows, the epoch
    boundary, the 117 -> 118 augmentation switch, a restart from the committed cursor, refusals before consumption."""
    import random
    import numpy as np
    import r28_pipeline as PL
    from sparse_rtdetr.research import scale_baseline as S
    torch = K.torch_mod()
    K.require(not torch.cuda.is_initialized(), "the pure-data pipeline checks run before any CUDA initialisation")
    data = S.train_dataset(c["data"]["train_core"], S.S15_HW, seed=0)
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    p0, per = K.PARENT_UPDATE, K.WINDOWS_PER_EPOCH
    res = {}

    def stream(mode, start, stop, table=refs):
        return PL.formal_stream(mode, data, start, stop, reference_check(table))

    K.PHASE["phase"] = "cal-check:pipeline serial and overlap, first windows of epoch 61"
    with stream("serial", p0, p0 + 3) as st:
        serial = list(st)
        serial_account = st.account()
    same = []
    with stream("overlap", p0, p0 + 3) as st:
        for i, item in enumerate(st):
            same.append(_same_window(item, serial[i]))
            del item
        overlap_account = st.account()
    K.require(same == [True] * 3 and [s.inp["update"] for s in serial] == [p0 + 1, p0 + 2, p0 + 3]
              and overlap_account["max_prepared_ahead"] == 1 and overlap_account["taken"] == 3, f"serial / overlap windows: {same} {overlap_account}")
    window = PL.window_bytes(serial[0].x, serial[0].t)
    res["first_windows"] = {"updates": [s.inp["update"] for s in serial], "pixels_targets_and_records_bitwise_equal": True,
                            "equal_to_the_reference_ledger": True, "serial": serial_account, "overlap": overlap_account,
                            "window_bytes": window, "memory_plan": {m: PL.memory_plan(window, m) for m in PL.MODES}}

    K.PHASE["phase"] = "cal-check:pipeline restart from the committed cursor"
    with stream("overlap", p0, p0 + 4) as st:
        it = iter(st)
        first, second = next(it), next(it)
        prepared = st.wait_ready()
        K.require(prepared == 1 and [first.inp["update"], second.inp["update"]] == [p0 + 1, p0 + 2], "a prepared window after two consumed")
        dropped_account = st.account()
        del first, second
    with stream("overlap", p0 + 2, p0 + 3) as st:
        (again,) = list(st)
    K.require(_same_window(again, serial[2]) and again.inp == refs[p0 + 3]["input"], "the window rebuilt from the committed cursor differs")
    res["restart"] = {"consumed_before_close": 2, "prepared_window_dropped": prepared, "account_at_close": dropped_account,
                      "committed_cursor": p0 + 2, "rebuilt_update": again.inp["update"],
                      "rebuilt_bitwise_equal_to_the_uninterrupted_window": True}
    del again

    K.PHASE["phase"] = "cal-check:pipeline refusals before consumption"
    refusals = []
    for mode, field in (("overlap", "input_tensor_sha256"), ("overlap", "image_ids_sha256"), ("overlap", "augmentation_trace_sha256"),
                        ("serial", "input_tensor_sha256")):
        bad = dict(refs)
        bad[p0 + 2] = {**refs[p0 + 2], "input": {**refs[p0 + 2]["input"], field: "0" * 64}}
        consumed, error = [], None
        try:
            with stream(mode, p0, p0 + 3, bad) as st:
                for item in st:
                    consumed.append(item.inp["update"])      # stands for the forward: only delivered windows reach it
        except ValueError as exc:
            error = str(exc)
        K.require(consumed == [p0 + 1] and error is not None and f"before update {p0 + 2}" in error and field in error,
                  f"a window with a wrong {field} was not refused before consumption ({mode}): {consumed} {error}")
        refusals.append({"mode": mode, "tampered_reference_field": field, "tampered_update": p0 + 2, "consumed_updates": consumed,
                         "refused_before_consumption": True, "message": error[:160]})
    res["refusals"] = refusals
    del serial

    K.PHASE["phase"] = "cal-check:pipeline epoch boundary and the 117 -> 118 switch"
    boundaries = {}
    for name, edge in (("epoch_61_to_62", 61 * per), ("epoch_117_to_118", 117 * per)):
        with stream("overlap", edge - 2, edge + 2) as st:
            got = [(item.inp["update"], item.inp["epoch"], item.inp["window"], item.inp["augmentation_strong"]) for item in st]
            account = st.account()
        ep = edge // per
        strong = (ep < 118, ep + 1 < 118)
        want = [(edge - 1, ep, per - 2, strong[0]), (edge, ep, per - 1, strong[0]), (edge + 1, ep + 1, 0, strong[1]), (edge + 2, ep + 1, 1, strong[1])]
        K.require(got == want and account["max_prepared_ahead"] == 1, f"{name}: {got}")
        boundaries[name] = {"windows": [list(g) for g in got], "equal_to_the_reference_ledger": True, "account": account}
    K.require(boundaries["epoch_117_to_118"]["windows"][1][3] is True and boundaries["epoch_117_to_118"]["windows"][2][3] is False,
              "strong augmentation switch")
    res["boundaries"] = boundaries
    after = np.random.get_state()
    K.require(random.getstate() == rng[0] and torch.equal(torch.get_rng_state(), rng[2]) and rng[1][0] == after[0]
              and np.array_equal(rng[1][1], after[1]) and rng[1][2:] == after[2:] and not torch.cuda.is_initialized(),
              "the input pipeline checks changed the parent RNG or initialised CUDA")
    res.update(parent_rng_unchanged=True, cuda_initialized=False, model_forwards=0)
    return res


# ================================================================ affected-interface check (CPU, no model)
def cal_check(c, out, counts):
    import l0_loop as LL
    import r28_cost  # noqa: F401  (import check of the timing module)
    import r28_eval as E
    import r28_loop as LP
    import r28_pipeline as PL
    from sparse_rtdetr.research import r28_fovea as FV
    from sparse_rtdetr.research import r28_fovea_cal as FS
    from sparse_rtdetr.research import r28_resample as R
    from sparse_rtdetr.research import r28_warp as W
    torch = K.torch_mod()
    out = Path(out)
    K.require(out.resolve() == (K.ROUND / "cal-check").resolve(), "this action writes <round>/cal-check")
    K.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_available(), "the calibration check hides CUDA")
    fu = K.followup(c)
    K.require(fu is not None and K.followup_arm(c) == "FS" and K.ARM == "FS", "cal-check runs with R28_ARM=FS under the calibrated-control contract")
    torch.set_num_threads(int(os.environ.get("R28_CPU_THREADS", "16")))
    t0 = time.perf_counter()
    res = {"stage": "cal-check", "registration": fu["registration"], "authorization_basis": c["authorization_basis"],
           "workload": c["workload"]["cal-check"]}

    K.PHASE["phase"] = "cal-check:pytest"
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *[str(K.ROOT / t) for t in TESTS]],
                       capture_output=True, text=True, cwd=str(K.ROOT), env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    res["pytest"] = {"returncode": p.returncode, "tail": p.stdout.strip().splitlines()[-1:], "tests": list(TESTS),
                     "note": "synthetic unit tests in a subprocess; their forwards are not in this stage's counts"}
    K.require(p.returncode == 0, "synthetic pytest failed:\n" + p.stdout[-6000:])

    K.PHASE["phase"] = "cal-check:identities"
    cf, f_inherited, f_refs = f_references(fu["F_round"])
    c1, ck, fu0 = _earlier(cf)
    K.require(all(fu[k] == v for k, v in f_refs.items()) and fu["inherited"] == {**fu0["inherited"], **f_inherited}
              and all(fu[k] == fu0[k] for k in ("prior_contract", "prior_contract_sha256", "K_contract", "K_contract_sha256", "W_demand",
                                                "W_endpoint", "K_endpoint", "official", "prior_W_decision", "prior_W_result", "K_result")),
              "references changed since prepare")
    K.require(K.other_contract(c, "F")["contract_sha256"] == cf["contract_sha256"] and K.other_contract(c, "K")["contract_sha256"] == ck["contract_sha256"]
              and K.prior_contract(c)["contract_sha256"] == c1["contract_sha256"]
              and len({c["contract_sha256"], cf["contract_sha256"], ck["contract_sha256"], c1["contract_sha256"]}) == 4,
              "contract digests of the four rounds")
    for name, f in (("task", TASK), ("review", REVIEW), ("engineering", ENGINEERING)):
        K.require(K.pinned(fu[name]) == K.ROUND / f[0] and fu[name]["sha256"] == f[1], f"{name} file")
    manifest = K.read_json(K.pinned(c["source_manifest"]))
    delta = source_delta(K.read_json(K.pinned(fu["F_manifest"]))["files"], manifest["files"])
    K.require(delta == fu["source_delta"], "source delta changed since prepare")
    same = [k for k in cf if k not in ("contract_sha256", "version", "authorization_basis", "source_root", "source_manifest", "source_commit",
                                       "source_dirty", "workload", "labels", "resource_plan", "outputs", "formal_started", "identity",
                                       "identity_sha256", "cost", "followup")]
    K.require(all(c[k] == cf[k] for k in same) and all(c["identity_sha256"][a] == cf["identity_sha256"][a] for a in ("W", "K", "F"))
              and c["identity_sha256"]["F"] == FV.IDENTITY_SHA256["F"] == fu["F_identity_sha256"]
              and c["identity_sha256"]["FS"] == FS.IDENTITY_SHA256["FS"] and c["identity"]["FS"] == FS.IDENTITY["FS"]
              and c["fovea_cal_definition"] == FS.DEFINITION and c["fovea_definition"] == FV.DEFINITION and c["definition"] == W.DEFINITION
              and c["parent"]["sha256"] == PIN_F["parent"] and len(set(c["identity_sha256"].values())) == 4,
              "inherited contract fields / identities")
    res["identities"] = {"F_round_pins": dict(PIN_F), "W_round_pins": dict(FU.PIN), "K_round_pins": dict(FC.PIN_K),
                         "inherited_stages": {"W_and_K_rounds": list(fu0["inherited"]), "F_round": list(f_inherited)},
                         "contract_fields_equal_to_the_F_round": len(same), "source_delta": delta,
                         "identity_sha256": c["identity_sha256"], "parent_sha256": c["parent"]["sha256"],
                         "arm_F_module_sha256_unchanged": manifest["files"]["src/sparse_rtdetr/research/r28_fovea.py"]
                         == K.read_json(K.pinned(fu["F_manifest"]))["files"]["src/sparse_rtdetr/research/r28_fovea.py"]}
    K.require(res["identities"]["arm_F_module_sha256_unchanged"], "arm F's module changed")

    K.PHASE["phase"] = "cal-check:authorisation and isolation"
    rounds = [Path(fu[k]) for k in ("prior_round", "K_round", "F_round")]
    for name in fu["inherited"]:
        K.require(not (K.ROUND / name).exists(), f"inherited stage name present in this round: {name}")
    w_summary = K.stage_closed(c, "summary-W", "W")
    K.require(w_summary["K_trigger"] is False and w_summary["label"] == "closed_fixed_config" and fu["prior_W_decision"] == FU.W_DECISION,
              "inherited W summary")
    K.require(K.stage_closed(c, "eval-F", "F")["endpoint"] == fu["F_endpoint"] and K.stage_closed(c, "eval-K", "K")["endpoint"] == fu["K_endpoint"]
              and K.stage_closed(c, "demand")["weights"] == fu["W_demand"], "inherited F / K evaluation and W demand network")
    net = {"kind": "network", "file": fu["W_demand"]}
    cases = [
        FU._refused("formal training of arm W in the calibrated-control round", lambda: LP.guard(c, K.ROUND / "formal-W", None, "W")),
        FU._refused("formal training of arm K in the calibrated-control round", lambda: LP.guard(c, K.ROUND / "formal-K", None, "K")),
        FU._refused("formal training of arm F in the calibrated-control round", lambda: LP.guard(c, K.ROUND / "formal-F", None, "F")),
        FU._refused("formal F* before the calibration and the acceptance stages are closed", lambda: LP.guard(c, K.ROUND / "formal-FS", None, "FS")),
        FU._refused("the locked width before the calibration is closed", lambda: K.calibration(c)),
        FU._refused("arm FS demand source before the calibration is closed", lambda: K.demand_source(c, "FS")),
        FU._refused("a formal F directory lookup in this round", lambda: K.formal_dir(c, "F")),
        *[FU._refused(f"an evaluation output inside {r.name}", lambda r=r: E.round_stage(r / "eval-FS", "eval-FS")) for r in rounds],
        FU._refused("a follow-up block without the registered authorization basis", lambda: K.followup({**c, "authorization_basis": "none"})),
        FU._refused("the calibrated-control block under the F mapping-control basis",
                    lambda: K.followup({**c, "authorization_basis": K.FOVEA_BASIS, "followup": {**fu, "authorization_basis": K.FOVEA_BASIS}})),
        FU._refused("arm F built by the arm FS module", lambda: FS.check_source("F", {**net, "fwhm": 4.0}, True)),
        FU._refused("arm FS without a width", lambda: FS.check_source("FS", net, True)),
        FU._refused("arm FS with an integer width", lambda: FS.check_source("FS", {**net, "fwhm": 13}, True)),
        FU._refused("arm FS with a width outside the calibration range", lambda: FS.check_source("FS", {**net, "fwhm": 31.0}, True)),
        FU._refused("a formal arm FS source without the calibration file", lambda: FS.check_source("FS", {**net, "fwhm": 13.0}, False)),
        FU._refused("arm FS with a fixed (arm K) demand source", lambda: FS.check_source("FS", {"kind": "fixed", "fwhm": 13.0, "calibration": net["file"]}, False)),
        FU._refused("R28_PIPELINE forcing the overlapped path", lambda: _forced_pipeline(c)),
    ]
    K.require(not any(v.startswith("R28_DEV_") for v in os.environ) or K.DEV, "dev flags are refused outside a dev round")
    res["authorisation"] = {"round": str(K.ROUND), "independent_of": [str(r) for r in rounds], "refusals": cases,
                            "W_summary_inherited": {k: w_summary[k] for k in ("label", "label_key", "K_trigger", "M_failed_cells")}}

    K.PHASE["phase"] = "cal-check:width table and kernel identity"
    table = FS.candidates()
    k4 = FS.kernel_matrix(FS.GRID_SHAPE[0], 4.0)
    K.require(len(table) == FS.GRID_POINTS + 1 and table[0] == 4.0 and table[-1] == 30.0 and 13.0 in table and table == sorted(set(table))
              and torch.equal(k4, FV.kernel_matrix(FV.GRID_SHAPE[0])) and FS.kernel_key(31, 4.0) != FS.kernel_key(31, 13.0)
              and not torch.equal(k4, FS.kernel_matrix(FS.GRID_SHAPE[0], 13.0)), "width table / kernel cache identity")
    syn = {k: v[None] for k, v in FC.synthetic_demands().items()}
    stacked = torch.cat(list(syn.values()))
    pf, ps = FV.plan_from_demand(stacked), FS.plan_from_demand(stacked, 4.0)
    K.require(torch.equal(pf.gx, ps.gx) and torch.equal(pf.gy, ps.gy) and torch.equal(pf.identity, ps.identity) and pf.stats == ps.stats,
              "arm FS with width 4 is not arm F bit for bit")
    res["width_table"] = {"widths": len(table), "first": table[0], "last": table[-1], "step": table[1] - table[0], "extra": list(FS.EXTRA_WIDTHS),
                          "width_4_equals_arm_F_bit_for_bit": True, "kernel_cache_key_example": list(FS.kernel_key(31, 13.0))}

    K.PHASE["phase"] = "cal-check:official rule at fixed widths"
    res["official_rule"] = {"official": fu["official"], "execution": "the pinned official file run unchanged on the CPU with "
                            "attraction_fwhm set to the width (stand-ins only for its unused imports, its .cuda() calls and the "
                            "removed numpy alias np.float)", "tolerance": {k: TOL[k] for k in ("official_fp32_nodes", "fp64_same_formula")},
                            "widths": [official_check(fu, syn, h) for h in OFFICIAL_WIDTHS]}

    refs, _ = LL.reference_rows(c)
    res["input_pipeline"] = {"version": PL.VERSION, "rule": c["input_pipeline"], **data_checks(c, refs)}
    K.require(counts["forwards"] == 0 and counts["actual_updates"] == 0, f"cal-check counts {dict(counts)}")
    res.update(model_forwards=0, cpu_seconds_wall=time.perf_counter() - t0,
               scope="affected interfaces only; the general CPU / identity / evaluation-path evidence of the shared code is the "
                     "closed W round's cpu, zero and evalpath stages and the K and F rounds' chains (inherited, pinned)")
    return res


def _forced_pipeline(c):
    old = os.environ.get("R28_PIPELINE")
    os.environ["R28_PIPELINE"] = "overlap"
    try:
        K.pipeline_mode(c, "FS")
    finally:
        if old is None:
            del os.environ["R28_PIPELINE"]
        else:
            os.environ["R28_PIPELINE"] = old


# ================================================================ calibration geometry (pure, CPU FP64)
def window_blocks(cache):
    """Per calibration window: demand maps, padded source boxes, the small / medium / large masks of the real GT."""
    from sparse_rtdetr.research import r28_resample as R
    torch = K.torch_mod()
    blocks, at = [], 0
    for w in cache["windows"]:
        n = w["images"]
        targets = [{"boxes": cache["boxes"][i]} for i in range(at, at + n)]
        pad, counts = R._padded_xyxy(targets)
        real = torch.zeros(pad.shape[:2], dtype=torch.bool)
        area = torch.zeros(pad.shape[:2], dtype=torch.float64)
        for i, k in enumerate(counts):
            real[i, :k] = True
            area[i, :k] = cache["raw_area"][at + i].to(torch.float64)
        size = pad[..., 2:] - pad[..., :2]
        blocks.append({"update": w["input"]["update"], "demand": cache["demand"][at:at + n], "pad": pad, "real": real,
                       "small": real & (area <= K.SMALL_AREA), "medium": real & (area >= K.SMALL_AREA) & (area <= K.LARGE_AREA),
                       "large": real & (area >= K.LARGE_AREA), "non_positive": int((real & (size <= 0).any(-1)).sum())})
        at += n
    K.require(at == len(cache["boxes"]) == len(cache["demand"]), "calibration cache layout")
    return blocks


def _pool(logs, blocks, name):
    """Pooled and per-window statistics of log g over one size class.  logs: per window the (B, M) log g tensor."""
    sums, counts = [], []
    for lg, b in zip(logs, blocks):
        v = lg[b[name]]
        sums.append(math.fsum(v.tolist()))
        counts.append(int(v.numel()))
    n = sum(counts)
    per = [math.exp(s / k) for s, k in zip(sums, counts) if k]
    return {"n": n, "sum_log_g": math.fsum(sums), "L": math.fsum(sums) / n if n else None,
            "pooled_geometric_mean": math.exp(math.fsum(sums) / n) if n else None,
            "per_window_geometric_mean_mean": statistics.fmean(per) if per else None, "windows_with_objects": len(per),
            "per_window": [None if not k else math.exp(s / k) for s, k in zip(sums, counts)], "per_window_n": counts}


def _bins(logs, blocks, name):
    torch = K.torch_mod()
    g = torch.exp(torch.cat([lg[b[name]] for lg, b in zip(logs, blocks)]))
    return {"g<1.0": int((g < 1.0).sum()), "1.0<=g<1.2": int(((g >= 1.0) & (g < 1.2)).sum()), "1.2<=g<1.4": int(((g >= 1.2) & (g < 1.4)).sum()),
            "g>=1.4": int((g >= 1.4).sum()), "min": float(g.min()) if len(g) else None, "max": float(g.max()) if len(g) else None}


def geometry_w(blocks):
    """Arm W on the cached demand maps and GT: log g per window with arm W's own allocation and box map."""
    from sparse_rtdetr.research import r28_warp as W
    torch = K.torch_mod()
    logs, aniso = [], []
    for b in blocks:
        ox, oy = W.magnification(W.plan_from_demand(b["demand"]), b["pad"])
        K.require(bool((torch.isfinite(ox) & torch.isfinite(oy) & (ox > 0) & (oy > 0))[b["real"]].all()), "arm W magnification on the calibration set")
        logs.append(torch.log(torch.sqrt(ox * oy)))
        aniso.append(torch.log(ox / oy))
    return logs, aniso


def geometry_h(blocks, h):
    """Arm F's rule with width h on the cached demand maps and GT.  Returns (valid, logs or None, detail): valid means
    every image of every window passes the fixed validity test; nothing is repaired."""
    from sparse_rtdetr.research import r28_fovea as FV
    from sparse_rtdetr.research import r28_fovea_cal as FS
    torch = K.torch_mod()
    logs, aniso, bad, flags = [], [], 0, Counter()
    lo, hi, end, clamp, identity = math.inf, 0.0, 0.0, 0.0, 0
    for b in blocks:
        nodes, ident = FS.nodes_from_demand(b["demand"], h)
        ok, f = FS.valid_images(nodes, ident)
        bad += int((~ok).sum())
        identity += int(ident.sum())
        for k, v in f.items():
            flags[k] += int(v.sum())
        if not bool(ok.all()):
            continue
        for axis in ("y", "x"):
            r = FS.node_readings(*nodes[axis], ident)
            n = nodes[axis][1].shape[1] - 1
            lo, hi = min(lo, float(r["step"].min()) * n), max(hi, float(r["step"].max()) * n)
            if bool(r["live"].any()):
                end, clamp = max(end, float(r["end"][r["live"]].max())), max(clamp, float(r["over"][r["live"]].max()))
        plan = FV.Plan(nodes["x"][1], nodes["y"][1], ident, {})
        ox, oy = FV.magnification(plan, b["pad"])
        if not bool((torch.isfinite(ox) & torch.isfinite(oy) & (ox > 0) & (oy > 0))[b["real"]].all()):
            bad += 1
            flags["magnification_not_finite_positive"] += 1
            continue
        logs.append(torch.log(torch.sqrt(ox * oy)))
        aniso.append(torch.log(ox / oy))
    valid = bad == 0
    detail = {"invalid_images": bad, "invalid_flags": {k: v for k, v in flags.items() if v}, "identity_images": identity}
    if valid:
        detail.update(node_slope_min=lo, node_slope_max=hi, endpoint_error_max=end, clamp_max=clamp)
    return valid, (logs if valid else None), (aniso if valid else None), detail


def reading(blocks, logs, aniso):
    """The size-class readings of one arm on one window set (description next to the calibrated statistic)."""
    torch = K.torch_mod()
    out = {}
    for name in ("small", "medium", "large"):
        pool = _pool(logs, blocks, name)
        a = torch.cat([v[b[name]] for v, b in zip(aniso, blocks)])
        out[name] = {k: pool[k] for k in ("n", "L", "pooled_geometric_mean", "per_window_geometric_mean_mean", "windows_with_objects")}
        out[name].update(bins=_bins(logs, blocks, name),
                         mean_log_ox_over_oy=float(a.mean()) if len(a) else None, mean_abs_log_ox_over_oy=float(a.abs().mean()) if len(a) else None)
    return out


def ledger_g_small(path, updates):
    """update -> (g_small, n_small) of a stored formal ledger for the given updates."""
    want, out = set(updates), {}
    with Path(path).open() as f:
        for line in f:
            row = json.loads(line)
            u = row["input"]["update"]
            if u in want:
                out[u] = (row["r28"]["g_small"], row["r28"]["n_small"])
            if u > max(want):
                break
    K.require(set(out) == want, "stored ledger rows of the calibration windows")
    return out


def against_ledger(blocks, pool, stored):
    """The per-window geometric mean of small g recomputed here against the per-update value a formal run recorded."""
    worst, counts_equal = 0.0, True
    for b, g, n in zip(blocks, pool["per_window"], pool["per_window_n"]):
        sg, sn = stored[b["update"]]
        counts_equal &= sn == n
        if g is None or sg is None:
            counts_equal &= g is None and sg is None
        else:
            worst = max(worst, abs(g - sg))
    return {"windows": len(blocks), "small_counts_equal": bool(counts_equal), "max_abs_difference_of_per_window_geometric_mean": worst,
            "tolerance": TOL["historical_ledger_geomean"],
            "historical_caliber_recomputed": pool["per_window_geometric_mean_mean"],
            "historical_caliber_stored": statistics.fmean(v[0] for v in stored.values() if v[0] is not None)}


def collect(c, ctx, data, refs, start, windows_, counts):
    """Cache of one window range: the production demand forward (microbatches of 8 on the model device, FP32, eval,
    no gradient, autocast off) and the transformed GT.  Every window is first compared with the reference ledger."""
    from formal_data import windows
    torch = K.torch_mod()
    check = reference_check(refs)
    cache = {"demand": [], "boxes": [], "raw_area": [], "image_ids": [], "windows": []}
    for x, t, inp in windows(data, start, limit=windows_):
        check(inp)
        with torch.autocast("cuda", enabled=False), torch.no_grad():
            d = torch.cat([ctx.demand(x[i:i + 8].to(ctx.device)) for i in range(0, len(x), 8)])
        counts.update(demand_forward_images=len(x))
        K.require(tuple(d.shape) == (len(x), 12, 21) and bool(torch.isfinite(d).all()) and bool((d >= 0).all()), "demand map")
        cache["demand"].append(d)
        for tt in t:
            cache["boxes"].append(tt["boxes"].clone())
            cache["raw_area"].append(tt["raw_area"].clone())
            cache["image_ids"].append(tt["image_id"])
        cache["windows"].append({"input": inp, "images": len(x)})
    cache["demand"] = torch.cat(cache["demand"])
    K.require(len(cache["windows"]) == windows_ and [w["input"]["update"] for w in cache["windows"]] == list(range(start + 1, start + windows_ + 1)),
              "calibration windows")
    return cache


def calibrate(c, out, counts):
    import l0_loop as LL
    from sparse_rtdetr.protocol_r18.engine import tensor_digest
    from sparse_rtdetr.research import r28_fovea_cal as FS
    from sparse_rtdetr.research import r28_resample as R
    from sparse_rtdetr.research import scale_baseline as S
    torch = K.torch_mod()
    out = Path(out)
    K.require(out.resolve() == (K.ROUND / "calibrate").resolve() and K.ARM == "FS" and K.followup_arm(c) == "FS",
              "this action writes <round>/calibrate with R28_ARM=FS")
    K.stage_closed(c, "cal-check", "FS")
    fu = K.followup(c)
    n61, n118 = FS.CALIBRATION["data"]["windows"], FS.CALIBRATION["confirmation"]["windows"]
    if os.environ.get(DEV_CAL):
        K.require(K.DEV, f"{DEV_CAL} is refused outside a dev round")
        n61, n118 = (int(v) for v in os.environ[DEV_CAL].split(","))
    t0 = time.perf_counter()
    K.PHASE["phase"] = "calibrate:demand cache of epoch 61"
    refs, _ = LL.reference_rows(c)
    source = {"kind": "network", "file": K.stage_closed(c, "demand")["weights"]}
    K.require(source["file"] == fu["W_demand"] and source["file"]["sha256"] == PIN_F["W_demand"], "the frozen demand network of arm W")
    ctx = R.Context(source).to("cuda:0")
    data = S.train_dataset(c["data"]["train_core"], S.S15_HW, seed=0)
    cache = collect(c, ctx, data, refs, K.PARENT_UPDATE, n61, counts)
    seconds_cache = time.perf_counter() - t0
    torch.save(cache, out / "cache-epoch061.pt")
    cache_ref = K.ref(out / "cache-epoch061.pt")
    blocks = window_blocks(cache)
    non_positive = sum(b["non_positive"] for b in blocks)
    K.require(non_positive == 0, f"{non_positive} GT boxes of non-positive size in the calibration windows (the engine refuses such a window)")

    K.PHASE["phase"] = "calibrate:arm W target"
    w_logs, w_aniso = geometry_w(blocks)
    w_pool = _pool(w_logs, blocks, "small")
    K.require(w_pool["n"] > 0, "no small GT in the calibration windows")
    l_w = w_pool["L"]
    dev_target = None
    if os.environ.get(DEV_TARGET):
        # A dev round shakes out the matched branch and the later stages with a target the rule can reach: the reading
        # of arm F's rule at the given width on the same cache.  Such a calibration is wiring, never a result.
        K.require(K.DEV, f"{DEV_TARGET} is refused outside a dev round")
        dev_target = float(os.environ[DEV_TARGET])
        ok, dev_logs, _, _ = geometry_h(blocks, dev_target)
        K.require(ok, f"the dev target width {dev_target} is not valid on the dev calibration set")
        l_w = _pool(dev_logs, blocks, "small")["L"]
        del dev_logs
    updates = [b["update"] for b in blocks]
    w_history = against_ledger(blocks, w_pool, ledger_g_small(K.pinned(fu["W_ledger"]), updates))

    K.PHASE["phase"] = "calibrate:width table"
    kept = {}

    def evaluate(h):
        valid, logs, aniso, detail = geometry_h(blocks, h)
        if not valid:
            return False, None, detail
        pool = _pool(logs, blocks, "small")
        kept[h.hex()] = (logs, aniso, pool) if h == 4.0 else (None, None, pool)     # per-box tensors only for arm F's width
        return True, pool["L"], {**detail, "n_small": pool["n"], "pooled_geometric_mean": pool["pooled_geometric_mean"],
                                 "per_window_geometric_mean_mean": pool["per_window_geometric_mean_mean"]}

    t1 = time.perf_counter()
    sel = FS.select(evaluate, l_w)
    seconds_rule = time.perf_counter() - t1
    f4 = kept.get((4.0).hex())
    f_history = None if f4 is None else against_ledger(blocks, f4[2], ledger_g_small(K.pinned(fu["F_ledger"]), updates))
    for name, hist in (("W", w_history), ("F", f_history)):
        K.require(hist is None or (hist["small_counts_equal"] and hist["max_abs_difference_of_per_window_geometric_mean"] <= TOL["historical_ledger_geomean"]),
                  f"the recomputed arm {name} geometry of the calibration windows differs from its formal ledger: {hist}")
    status = sel["status"]
    body = {"schema": "r28-fs-calibration-v1", "contract_sha256": c["contract_sha256"], "task": fu["task"], "rule": FS.CALIBRATION,
            "status": status, "rule_applied": sel["rule_applied"], "h_star": sel["h_star"], "h_star_hex": sel["h_star_hex"],
            "L_W": l_w, "L_h_star": sel["L_h_star"], "relative_error": sel["relative_error"], "tolerance": sel["tolerance"],
            "interval": sel["interval"], "closest_if_unmatched": sel.get("closest"), "readings": sel["readings"],
            "calibration_set": {"epoch": 61, "windows": len(blocks), "occurrences": len(cache["boxes"]), "first_update": updates[0],
                                "last_update": updates[-1], "cache": cache_ref, "demand_sha256": tensor_digest([("demand", cache["demand"])]),
                                "every_window_equal_to_the_reference_ledger": True, "GT": int(sum(int(b["real"].sum()) for b in blocks)),
                                "small_GT": w_pool["n"], "non_positive_boxes": non_positive, "demand_network": ctx.binding(),
                                "dev_subset": os.environ.get(DEV_CAL)},
            "L_W_of_arm_W": w_pool["L"], "dev_target_width": dev_target,
            "W": {"reading": reading(blocks, w_logs, w_aniso), "against_the_formal_W_ledger": w_history},
            "F_width_4": None if f4 is None else {"reading": reading(blocks, f4[0], f4[1]), "against_the_formal_F_ledger": f_history},
            "historical_caliber_note": "the reports' 1.267 (W) and 1.053 (F) are means over updates of per-update geometric means over "
                                       "all 60 epochs; the calibrated statistic pools every small observation of epoch 61",
            "confirmation": None, "seconds": {"demand_cache_epoch_61": seconds_cache, "rule": seconds_rule}}
    result = {"stage": "calibrate", "workload": c["workload"]["calibrate"], "calibration_status": status, "h_star": sel["h_star"],
              "h_star_hex": sel["h_star_hex"], "L_W": l_w, "L_h_star": sel["L_h_star"], "relative_error": sel["relative_error"],
              "widths_read": len(sel["readings"]), "invalid_widths": [r["h"] for r in sel["readings"] if not r["valid"]],
              "interval": sel["interval"], "rule_applied": sel["rule_applied"], "dev_target_width": dev_target,
              "closest_if_unmatched": sel.get("closest")}
    if status == "matched":
        h = sel["h_star"]
        valid, logs, aniso, _ = geometry_h(blocks, h)                               # the same pure computation, repeated
        K.require(valid and _pool(logs, blocks, "small")["L"] == sel["L_h_star"] == kept[h.hex()][2]["L"],
                  "the locked width does not reproduce its reading")
        body["F_star"] = {"reading": reading(blocks, logs, aniso)}
        result["magnification_small_pooled"] = {"W": math.exp(l_w), "FS": math.exp(sel["L_h_star"]),
                                                "F_width_4": None if f4 is None else f4[2]["pooled_geometric_mean"]}
        K.PHASE["phase"] = "calibrate:confirmation on epoch 118"
        t2 = time.perf_counter()
        cache2 = collect(c, ctx, data, refs, E118_START, n118, counts)
        torch.save(cache2, out / "cache-epoch118.pt")
        blocks2 = window_blocks(cache2)
        np2 = sum(b["non_positive"] for b in blocks2)
        K.require(np2 == 0, f"{np2} GT boxes of non-positive size in the confirmation windows")
        w2_logs, w2_aniso = geometry_w(blocks2)
        w2 = _pool(w2_logs, blocks2, "small")
        w2_history = against_ledger(blocks2, w2, ledger_g_small(K.pinned(fu["W_ledger"]), [b["update"] for b in blocks2]))
        K.require(w2_history["small_counts_equal"] and w2_history["max_abs_difference_of_per_window_geometric_mean"] <= TOL["historical_ledger_geomean"],
                  f"the recomputed arm W geometry of the confirmation windows differs from its formal ledger: {w2_history}")
        valid2, logs2, aniso2, detail2 = geometry_h(blocks2, h)
        conf = {"epoch": 118, "windows": len(blocks2), "occurrences": len(cache2["boxes"]), "first_update": blocks2[0]["update"],
                "last_update": blocks2[-1]["update"], "cache": K.ref(out / "cache-epoch118.pt"), "valid": bool(valid2), "detail": detail2,
                "every_window_equal_to_the_reference_ledger": True, "recalibrated": False, "small_GT": w2["n"],
                "W": {"reading": reading(blocks2, w2_logs, w2_aniso), "against_the_formal_W_ledger": w2_history},
                "seconds": time.perf_counter() - t2}
        if valid2 and w2["n"]:
            f2 = _pool(logs2, blocks2, "small")
            conf.update(L_W=w2["L"], L_h_star=f2["L"], relative_error=FS.relative_error(f2["L"], w2["L"]),
                        F_star={"reading": reading(blocks2, logs2, aniso2)},
                        note="the error on another augmentation period is reported as it is; the width is not changed by it")
        else:
            status = body["status"] = result["calibration_status"] = "confirmation_invalid"
        body["confirmation"] = conf
        result["confirmation"] = {k: conf.get(k) for k in ("valid", "windows", "small_GT", "L_W", "L_h_star", "relative_error")}
    K.require(status in FS.STATUSES, f"calibration status {status}")
    K.require(counts["forwards"] == 0 and counts["actual_updates"] == 0, f"the calibration ran a detector: {dict(counts)}")
    K.write_new(out / "CALIBRATION.json", body)
    result.update(calibration=K.ref(out / "CALIBRATION.json"), cache=cache_ref, demand_forward_images=counts["demand_forward_images"],
                  detector_forwards=0, formal_training_allowed=status == "matched",
                  note="a status other than matched ends the chain here: no formal training, no other width, range, target or parameter")
    print(json.dumps({"calibration_status": status, "h_star": sel["h_star"], "h_star_hex": sel["h_star_hex"], "L_W": l_w,
                      "relative_error": sel["relative_error"]}), flush=True)
    return result


# ================================================================ CPU fixture with the locked width
def real_geometry(c, x, t, source):
    """The locked width on one real training window (CPU): demand equal to arm W's, the map, GT forward / inverse,
    the complete tent support against a dense reference."""
    import numpy as np
    from sparse_rtdetr.research import r28_fovea as FV
    from sparse_rtdetr.research import r28_fovea_cal as FS
    from sparse_rtdetr.research import r28_resample as R
    from sparse_rtdetr.research import r28_warp as W
    from sparse_rtdetr.research.r27_local_memory import H_HW, L_HW
    torch = K.torch_mod()
    F64 = torch.float64
    h = source["fwhm"]
    ctx_f, ctx_w = R.Context({"kind": "network", "file": source["file"]}), R.Context(K.demand_source(c, "W"))
    with torch.no_grad():
        d_f, d_w = ctx_f.demand(x), ctx_w.demand(x)
    K.require(torch.equal(d_f, d_w) and ctx_f.binding() == ctx_w.binding(), "F* and W demand maps differ on the same input")
    plan, plan4, wplan = FS.plan_from_demand(d_f, h), FV.plan_from_demand(d_f), W.plan_from_demand(d_w)
    K.require(not bool(plan.identity.any()), "identity flags on real input")
    slope = lambda p: [min(p.stats[a]["node_slope_min"] for a in "xy"), max(p.stats[a]["node_slope_max"] for a in "xy")]
    res = {"images": len(x), "width": h, "width_hex": h.hex(), "demand_equal_to_W_bit_for_bit": True,
           "map": {"stats": plan.stats, "node_slope_range": slope(plan), "node_slope_range_at_width_4": slope(plan4),
                   "strictly_increasing": True}}
    K.require(all(plan.stats[a]["endpoint_error_max"] <= FS.END_TOLERANCE and plan.stats[a]["clamp_max"] <= FS.END_TOLERANCE
                  and plan.stats[a]["clamped_interior_nodes"] == 0 for a in "xy"), f"anti-crop end points: {plan.stats}")
    pad, n = R._padded_xyxy(t)
    tw = FV.warp_targets(plan, t)
    worst = {"dn_forward_vs_numpy": 0.0, "gt_round_trip": 0.0}
    for i, (a, w) in enumerate(zip(t, tw)):
        if not n[i]:
            continue
        e = W.cxcywh_to_xyxy(a["boxes"].to(F64)).numpy()
        gx, gy = plan.gx[i].numpy(), plan.gy[i].numpy()
        ref = np.stack([FC._np_warp(gx, e[:, 0]), FC._np_warp(gy, e[:, 1]), FC._np_warp(gx, e[:, 2]), FC._np_warp(gy, e[:, 3])], -1)
        one = plan.select(slice(i, i + 1))
        mine = W.cxcywh_to_xyxy(FV.map_boxes(one, a["boxes"][None].to(F64), inverse=False))[0]
        worst["dn_forward_vs_numpy"] = max(worst["dn_forward_vs_numpy"], float(np.abs(mine.numpy() - ref).max()))
        back = W.cxcywh_to_xyxy(FV.map_boxes(one, W.xyxy_to_cxcywh(mine)[None], inverse=True))[0]
        worst["gt_round_trip"] = max(worst["gt_round_trip"], float(np.abs(back.numpy() - e).max()))
        K.require(w["boxes"].dtype == a["boxes"].dtype and bool((w["boxes"][:, 2:] > 0).all())
                  and float((w["boxes"].to(F64) - W.xyxy_to_cxcywh(mine)).abs().max()) <= 1e-6, "denoising targets are not the inverse of G")
    v = torch.linspace(-0.25, 1.25, 3001, dtype=F64).expand(len(x), -1)
    worst["round_trip_warped_source_warped"] = max(float((FV.warp(g, FV.unwarp(g, v)) - v).abs().max()) for g in (plan.gx, plan.gy))
    worst["round_trip_source_warped_source"] = max(float((FV.unwarp(g, FV.warp(g, v)) - v).abs().max()) for g in (plan.gx, plan.gy))
    K.require(all(e <= TOL["round_trip"] for e in worst.values()), f"map round trips on real input: {worst}")
    ox, oy = FV.magnification(plan, pad)
    ox4, oy4 = FV.magnification(plan4, pad)
    wx, wy = W.magnification(wplan, pad)
    area = torch.cat([tt["raw_area"] for tt in t]).to(F64)
    flat = lambda a, b: torch.cat([torch.sqrt(a * b)[i, :k] for i, k in enumerate(n)])
    small = area <= K.SMALL_AREA
    geo = lambda g: float(torch.exp(torch.log(g[small]).mean())) if bool(small.any()) else None
    res["gt"] = {"boxes": int(sum(n)), "errors": worst, "tolerance": TOL["round_trip"],
                 "small_magnification_first_window": {"n": int(small.sum()), "FS": geo(flat(ox, oy)), "F_width_4": geo(flat(ox4, oy4)),
                                                      "W": geo(flat(wx, wy)), "note": "description of one window; not a gate"}}
    t0 = time.perf_counter()
    xw, info = FV.resample(x, plan)
    seconds = time.perf_counter() - t0
    K.require(tuple(xw.shape) == (len(x), 3, *L_HW) and xw.dtype == torch.float32 and bool(torch.isfinite(xw).all())
              and float(xw.min()) >= float(x.min()) - 1e-5 and float(xw.max()) <= float(x.max()) + 1e-5, "rendered image")
    dense = []
    for i in range(min(2, len(x))):
        ax, sx, dx0, dx1 = FC.reference_axis(plan.gx[i].numpy(), L_HW[1], H_HW[1])
        ay, sy, dy0, dy1 = FC.reference_axis(plan.gy[i].numpy(), L_HW[0], H_HW[0])
        ref = np.einsum("ih,chw,jw->cij", ay, x[i].double().numpy(), ax, optimize=True)
        dense.append({"image": i, "max_abs_difference": float(np.abs(xw[i].double().numpy() - ref).max()),
                      "reference_support_x": sx, "reference_support_y": sy, "footprint_px": [min(dx0, dy0), max(dx1, dy1)]})
    K.require(all(r["max_abs_difference"] <= TOL["render_fp32_vs_fp64"] and info["taps_x"] >= r["reference_support_x"]
                  and info["taps_y"] >= r["reference_support_y"] for r in dense), f"complete tent support: {dense} {info}")
    res["render"] = {"readings": info, "cpu_seconds_16_images": seconds, "dense_reference": dense, "tolerance": TOL["render_fp32_vs_fp64"],
                     "needs_more_than_4_taps_on_this_window": max(info["taps_x"], info["taps_y"]) > W.TAPS}
    return res, d_f


def state_refusals(c, e, state, scope):
    """Cross-arm / cross-width / cross-demand / cross-scope uses of states that must be refused.  `e` is an arm FS
    engine and `state` a state it wrote in `scope`; nothing may change the engine."""
    import l0_checks_state as H
    import r28_loop as LP
    from sparse_rtdetr.research import r28_resample as R
    fu = K.followup(c)
    other = {**c, "identity_sha256": {**c["identity_sha256"], "FS": "0" * 64}}
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

    wider = {**real_map, "fwhm": 4.0, "fwhm_hex": (4.0).hex()}
    cases = [("arm FS state read as arm F", lambda: K.read_side(c, state, scope, "F")),
             ("arm FS state read as arm W", lambda: K.read_side(c, state, scope, "W")),
             ("arm FS state read as arm K", lambda: K.read_side(c, state, scope, "K")),
             ("arm FS state read under another mapping identity", lambda: K.read_side(other, state, scope, "FS")),
             ("arm FS state loaded into an engine bound to another width (4)", swapped(r28_mapping=wider)),
             ("arm FS state loaded into an engine bound to another calibration file",
              swapped(r28_mapping={**real_map, "calibration": {**(real_map["calibration"] or {}), "sha256": "0" * 64}})),
             ("arm FS state loaded into an engine with another demand source", swapped(r28_ctx=synthetic)),
             ("F formal e120 state loaded into the arm FS engine", lambda: K.load(c, e, fu["F_endpoint"]["path"], "formal")),
             ("W formal e120 state loaded into the arm FS engine", lambda: K.load(c, e, fu["W_endpoint"]["path"], "formal")),
             ("K formal e120 state loaded into the arm FS engine", lambda: K.load(c, e, fu["K_endpoint"]["path"], "formal")),
             ("L0 formal e120 state loaded into the arm FS engine", lambda: K.load(c, e, c["L0_endpoint"]["path"], "formal")),
             ("formal training of arm F in this round", lambda: LP.guard(c, K.ROUND / "formal-F", None, "F")),
             ("formal training of arm W in this round", lambda: LP.guard(c, K.ROUND / "formal-W", None, "W"))]
    if scope != "formal":
        cases += [(f"{scope} state read in formal scope", lambda: K.read_side(c, state, "formal", "FS")),
                  (f"{scope} state loaded as a formal state", lambda: K.load(c, e, state, "formal"))]
    out = [H.refused(label, fn) for label, fn in cases]
    K.require(e.r28_ctx is real_ctx and e.r28_mapping is real_map, "the engine's demand source / mapping was not restored")
    return out


def cal_fixture(c, out, counts):
    import l0_checks_state as H
    import r28_checks as CK
    import r28_loop as LP
    from s15_formal_train import _bn_counts, _differences, _result_body
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    from sparse_rtdetr.research import r28_fovea_cal as FS
    from sparse_rtdetr.research import scale_baseline as S
    torch = K.torch_mod()
    out = Path(out)
    K.require(out.resolve() == (K.ROUND / "cal-fixture").resolve() and K.ARM == "FS" and K.followup_arm(c) == "FS",
              "this action writes <round>/cal-fixture with R28_ARM=FS")
    K.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_available(), "the fixture hides CUDA")
    torch.set_num_threads(int(os.environ.get("R28_CPU_THREADS", "16")))
    t0 = time.perf_counter()
    fu = K.followup(c)
    source = K.demand_source(c, "FS")
    lock = K.calibration(c)
    h = lock["fwhm"]
    res = {"stage": "cal-fixture", "workload": c["workload"]["cal-fixture"], "mapping": FS.mapping(source)}
    K.PHASE["phase"] = "cal-fixture:real first window"
    (x, t, inp), = CK._first_windows(c, 1)
    theirs = H.r27_rows(c, "P", 1)[0]["input"]
    K.require(all(type(inp[k]) is type(theirs[k]) and inp[k] == theirs[k] for k in K.INPUT_FIELDS) and set(inp) == set(K.INPUT_FIELDS),
              "input identity differs from R27 formal-P row 1")
    geometry, real_d = real_geometry(c, x, t, source)
    res["real_window"] = {"input": inp, "window_shape": list(x.shape), "input_identity_equal_to_R27_formal_P_row_1": True, **geometry}
    K.PHASE["phase"] = "cal-fixture:official rule at the locked width"
    res["official_rule_at_the_locked_width"] = official_check(fu, {**{k: v[None] for k, v in FC.synthetic_demands().items()},
                                                                 "real_first_window": real_d}, h)

    K.PHASE["phase"] = "cal-fixture:construct"
    parent = K.pinned(c["parent"])
    e, checks = FS.create(parent, "cpu", "FS", source, fixture=True, check=True)
    ps = torch.load(parent, map_location="cpu", weights_only=True)
    sd, esd = e.model.state_dict(), e.ema.module.state_dict()
    keys = {"online_equal_parent": list(sd) == list(ps["model"]), "EMA_equal_parent": list(esd) == list(ps["ema"]["module"]),
            "state_keys": len(sd), "demand_or_r28_keys": sum("r28" in k or "demand" in k or "fovea" in k for k in sd)}
    K.require(keys["online_equal_parent"] and keys["EMA_equal_parent"] and keys["demand_or_r28_keys"] == 0, ("native key set", keys))
    del ps
    res.update(native_keys=keys, create_checks={k: v for k, v in checks.items() if k != "new_parameter_names"})
    x2, t2 = x[:2], t[:2]
    before = [tt["boxes"].clone() for tt in t2]
    res["fixture"] = fx = {"images": [tt["image_id"] for tt in t2], "physical_batch": 1, "accumulation": 2, "precision": "float32",
                           "demand": e.r28_ctx.binding(), "mapping": e.r28_mapping, "non_identity": True}
    K.register_counts(e, counts)
    obs = LP.install_observer(e)
    bn0 = _bn_counts(e.model)
    rows, readings = [], []
    for i in (1, 2):
        K.PHASE["phase"] = f"cal-fixture:u{i}"
        r = e.train_window(x2, t2, capture=True)
        K.require(r["update"] == K.PARENT_UPDATE + i and r["finite"] and e.ema.updates == e.completed_updates, "fixture clock")
        rows.append(r)
        readings.append(obs.update_row(t2))
        if i == 1:
            side = K.save(c, e, out / "cpu-fixture-u1.pth", "cpu-fixture")
    K.require(all(torch.equal(a, tt["boxes"]) for a, tt in zip(before, t2)), "the forward modified the caller's targets")
    K.require(all(r["identity_images"] == 0 and "fovea" in r and r["fovea"]["taps_max"] >= 2 and "cal" in r
                  and sum(r["cal"][k]["n"] for k in ("small", "medium", "large")) >= sum(len(tt["labels"]) for tt in t2)
                  for r in readings), f"fixture readings: {readings}")
    K.require(side["mapping"] == e.r28_mapping == FS.mapping(source) and side["mapping"]["fwhm_hex"] == h.hex()
              and side["mapping"]["calibration"] == lock["calibration"], "the saved state does not carry the locked width")
    json.dumps(readings, allow_nan=False)
    expected = S.capture(e)
    erow = obs.epoch_row(61)
    fx["BN_delta_two_updates"] = sorted(set(v - bn0[k] for k, v in _bn_counts(e.model).items()))
    K.require(fx["BN_delta_two_updates"] == [4], f"fixture BN delta {fx['BN_delta_two_updates']}")
    del e
    gc.collect()
    K.PHASE["phase"] = "cal-fixture:restore+replay"
    clone, _ = FS.create(parent, "cpu", "FS", source, fixture=True, check=True)
    K.register_counts(clone, counts)
    obs2 = LP.install_observer(clone)
    K.load(c, clone, out / "cpu-fixture-u1.pth", "cpu-fixture")
    replay = clone.train_window(x2, t2, capture=True)
    reading2 = obs2.update_row(t2)
    got = S.capture(clone)
    diff = _differences(expected, got) + _differences(_result_body(rows[1]), _result_body(replay))
    K.require(not diff and ck.exact_equal(expected, got) and ck.exact_equal(expected["rng"], got["rng"]) and reading2 == readings[1],
              f"fixture replay differs {diff[:5]}")
    K.require(counts["actual_updates"] == 3 and counts["forwards"] == counts["forward_attempts"] == 6, f"fixture counts {dict(counts)}")
    fx.update(saved_u1=side, full_state_exact=True, rng_exact=True, replay_result_exact=True, observer_replay_exact=True,
              loss=[r["loss"] for r in rows], replay_loss=replay["loss"], observer_updates=readings, observer_epoch=erow,
              elapsed_seconds=[r["elapsed_seconds"] for r in rows + [replay]])
    K.PHASE["phase"] = "cal-fixture:state refusals"
    st = out / "cpu-fixture-u1.pth"
    res["refused"] = state_refusals(c, clone, st, "cpu-fixture") + [
        H.refused("check engine writing a formal state", lambda: K.save(c, clone, out / "refused.pth", "formal")),
        H.refused("formal engine with a synthetic demand", lambda: FS.check_source("FS", {"kind": "synthetic", "seed": 0, "fwhm": h}, False)),
        H.refused("endpoint_state without an endpoint", lambda: K.endpoint_state(c, "FS"))]
    K.require(ck.exact_equal(S.capture(clone), got) and not (out / "refused.pth").exists(), "a refused action changed the engine or wrote a file")

    K.PHASE["phase"] = "cal-fixture:ledger readings"
    trial_dir = out / "trial-ledger"
    trial_dir.mkdir()
    with (trial_dir / "UPDATES.jsonl").open("x") as f:
        for i, (r, rd) in enumerate(zip(rows, readings)):
            f.write(json.dumps({"input": {"update": K.PARENT_UPDATE + 1 + i, "epoch": 61}, "update_seconds": r["elapsed_seconds"], "r28": rd},
                               separators=(",", ":"), allow_nan=False) + "\n")
    folded, match = FC.formal_observations(trial_dir), formal_calibration(trial_dir)
    K.require(folded["updates"] == 2 and folded["taps_max"] >= 2 and folded["clamped_interior_nodes"] == 0
              and match["updates"] == 2 and match["overall"]["small"]["n"] == sum(rd["cal"]["small"]["n"] for rd in readings),
              f"folded fixture readings: {folded} {match}")
    res["ledger_readings_interface"] = {"rows": 2, "folded": folded, "calibration_match": match,
                                        "note": "CPU fixture rows; interface check of the summary readers"}
    res.update(fixture_updates=3, full_forwards=6, formal_state_usable=False, cpu_seconds_wall=time.perf_counter() - t0)
    return res


# ================================================================ summary arithmetic (pure)
def summary_fs(c, w_result, values, apl, apm, cost, quoted):
    """values / apl / apm: arm -> stored readings of L0, K, F, FS, W.  Pure arithmetic; no gate."""
    import r28_eval as E
    K.require(set(values) == set(apl) == set(apm) == set(ARMS5), "summary arms")
    diffs = {f"{x}-{y}": E.difference_pp(values[x], values[y]) for x, y in PAIRS}
    primary = {f"W-FS top{b} {m}": diffs["W-FS"][b][m] for b, m in PRIMARY}
    ratios = cost["ratios"]
    return {"arm": ARM, "label": LABEL, "contribution": CONTRIBUTION, "record_only": True,
            "values": {**w_result["values"], "K": values["K"], "F": values["F"], "FS": values["FS"]},
            "value_sources": {"FS": "this round, eval-FS (new)", "F": "F round eval-F (quoted)", "W": "closed W round eval-W (quoted)",
                              "K": "K round eval-K (quoted)", "L0": "R27-002 eval-L0 via the contract reference (quoted)"},
            "primary_W_minus_FS_pp": primary, "differences_pp": diffs,
            "APL6_pp": {f"{x}-{y}": E.apl6(apl[x], apl[y]) for x, y in (("W", "FS"), ("FS", "F"), ("FS", "L0"))},
            "per_class_APL_top600": FC.size_table({a: apl[a] for a in ARMS5}, "APL"),
            "per_class_APM_top600": FC.size_table({a: apm[a] for a in ARMS5}, "APM"),
            "cost_FS": {"model_gpu": {k: ratios["model_gpu"][k] for k in COST_PAIRS},
                        "end_to_end": {k: ratios["end_to_end"][k] for k in COST_PAIRS},
                        "all_ratios": ratios, "ratios_per_pass": cost.get("ratios_per_pass"), "medians": cost["medians"],
                        "stages": cost.get("stages"), "order_executed": cost.get("order_executed"),
                        "source": "cost-FS (L0, K, F, FS, W in one process)"},
            "cost_quoted_from_earlier_rounds": {**quoted, "note": "other processes; quoted, not the same measurement"},
            "W_decision_unchanged": w_result["decision"]}


def _inputs(c):
    fu = K.followup(c)
    w_result, k_result, ev_w, values, apl, apm = FC._inputs(c)
    ev_f = K.stage_closed(c, "eval-F", "F")
    f_result = K.read_json(K.pinned(fu["F_result"]))
    K.require(f_result["values"]["F"] == ev_f["values"] and f_result["contract_sha256"] == fu["F_contract_sha256"]
              and ev_f["weights_sha256"] == fu["F_weights_sha256"], "stored F result")
    values, apl, apm = {**values, "F": ev_f["values"]}, {**apl, "F": ev_f["per_class_APL_top600"]}, {**apm, "F": ev_f["per_class_APM_top600"]}
    quoted = {"F_round_L0_W_F": f_result["cost_F"]["model_gpu"], "K_round_L0_W_K": k_result["cost_K"]["model_gpu"]}
    return w_result, ev_w, ev_f, values, apl, apm, quoted


def _trial_summary(c, cost):
    """summary_fs with the stored W evaluation standing in for FS (wiring only; every W - FS difference is exactly 0)."""
    w_result, ev_w, ev_f, values, apl, apm, quoted = _inputs(c)
    if cost is None:
        arms = K.COST_ENDPOINTS["cost-FS"]
        one = {f"{a}/{b}": 1.0 for a in arms for b in arms if a != b}
        cost = {"ratios": {"model_gpu": dict(one), "end_to_end": dict(one)}, "medians": {}}
    trial = summary_fs(c, w_result, {**values, "FS": ev_w["values"]}, {**apl, "FS": apl["W"]}, {**apm, "FS": apm["W"]}, cost, quoted)
    K.require(all(v == 0.0 for v in trial["primary_W_minus_FS_pp"].values()) and trial["APL6_pp"]["W-FS"]["difference_pp"] == 0.0
              and trial["W_decision_unchanged"] == FU.W_DECISION and trial["label"] == LABEL and trial["contribution"] == CONTRIBUTION
              and trial["differences_pp"]["FS-F"] == trial["differences_pp"]["W-F"], "summary arithmetic")
    json.dumps(trial, allow_nan=False)
    return trial


def formal_calibration(d):
    """How the calibrated match held over a formal run: the ledger's per-update sums of log g of the same GT under arm
    FS and under arm W's allocation, pooled over all updates and per epoch.  A description; nothing was fed back."""
    d = Path(d)
    classes, arms = ("small", "medium", "large"), ("FS", "W")
    zero = lambda: {k: {"n": 0, **{a: {"sum_log_g": [], "bins": [0, 0, 0, 0], "ratio": [], "abs_ratio": []} for a in arms}} for k in classes}
    total, epochs, updates = zero(), {}, 0
    with (d / "UPDATES.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            cal, ep = row["r28"]["cal"], row["input"]["epoch"]
            updates += 1
            for acc in (total, epochs.setdefault(ep, zero())):
                for k in classes:
                    acc[k]["n"] += cal[k]["n"]
                    for a in arms:
                        acc[k][a]["sum_log_g"].append(cal[k][a]["sum_log_g"])
                        acc[k][a]["ratio"].append(cal[k][a]["sum_log_ox_over_oy"])
                        acc[k][a]["abs_ratio"].append(cal[k][a]["sum_abs_log_ox_over_oy"])
                        acc[k][a]["bins"] = [p + q for p, q in zip(acc[k][a]["bins"], cal[k][a]["bins"])]

    def fold(acc):
        out = {}
        for k in classes:
            n = acc[k]["n"]
            cell = {"n": n}
            for a in arms:
                s = math.fsum(acc[k][a]["sum_log_g"])
                cell[a] = {"L": s / n if n else None, "pooled_geometric_mean": math.exp(s / n) if n else None,
                           "bins": acc[k][a]["bins"], "bin_shares": [b / n for b in acc[k][a]["bins"]] if n else None,
                           "mean_log_ox_over_oy": math.fsum(acc[k][a]["ratio"]) / n if n else None,
                           "mean_abs_log_ox_over_oy": math.fsum(acc[k][a]["abs_ratio"]) / n if n else None}
            cell["FS_over_W_minus_1"] = math.expm1(cell["FS"]["L"] - cell["W"]["L"]) if n else None
            out[k] = cell
        return out

    per_epoch = {ep: fold(acc) for ep, acc in sorted(epochs.items())}
    errors = {ep: v["small"]["FS_over_W_minus_1"] for ep, v in per_epoch.items() if v["small"]["FS_over_W_minus_1"] is not None}
    return {"updates": updates, "overall": fold(total), "bins": ["g<1.0", "1.0<=g<1.2", "1.2<=g<1.4", "g>=1.4"],
            "small_FS_over_W_minus_1_by_epoch": errors,
            "small_FS_over_W_minus_1_range": [min(errors.values()), max(errors.values())] if errors else None,
            "per_epoch": per_epoch,
            "note": "L is the pooled mean log magnification of the training GT of a raw-area class; FS_over_W_minus_1 = exp(L_FS - L_W) - 1 "
                    "is the quantity the calibration matched on epoch 61; W's geometry is computed from the demand maps the F* forward "
                    "already produced (no image rendered, no extra network forward)"}


# ================================================================ CUDA wiring of what only runs after training
def cal_wire(c, out, counts):
    import r28_cost as CO
    import r28_eval as E
    from sparse_rtdetr.protocol_r18.engine import tensor_digest
    from sparse_rtdetr.research import r28_fovea_cal as FS
    torch = K.torch_mod()
    out = Path(out)
    K.require(out.resolve() == (K.ROUND / "wire-FS").resolve() and K.ARM == "FS" and K.followup_arm(c) == "FS",
              "this action writes <round>/wire-FS with R28_ARM=FS")
    K.stage_closed(c, "cal-fixture", "FS")
    source = K.demand_source(c, "FS")
    K.PHASE["phase"] = "wire:model (parent EMA weights behind the arm FS forward)"
    state = torch.load(K.pinned(c["parent"]), map_location="cpu", weights_only=True)["ema"]["module"]
    digest = tensor_digest(state.items())
    model = FS.build_eval_model(state, "cuda:0", "FS", source)
    K.require(FS.is_wrapped(model) and not model.training and model.r28_mapping == FS.mapping(source), "wire model")
    K.PHASE["phase"] = "wire:timing"
    order = K.COST_ENDPOINTS["cost-FS"]
    loaders = {"L0": lambda: CO.load_reference(c, "L0", torch),
               **{a: (lambda a=a: CO.load_arm(c, a, torch)) for a in ("W", "K", "F")},
               "FS": lambda: (model, {"weights": "e060 parent EMA (not trained under F*; wiring only)", "demand": model.r28_ctx.binding(),
                                      "mapping": model.r28_mapping})}
    timing = CO._run(c, counts, loaders, order, (order,) * WIRE["cost_passes"], warmup=WIRE["cost_warmup"],
                     batches_per_pass=WIRE["cost_batches_per_pass"], stages=True)
    table = timing["stages"]["table"]
    K.require(set(table) == set(order) and all(abs(sum(table[a]["stage_ms_median_pass"].values()) - table[a]["model_gpu_ms_median_pass"])
                                                <= 1e-6 * max(1.0, table[a]["model_gpu_ms_median_pass"]) for a in order)
              and all(table[a]["stage_ms_median_pass"][s] > 0 for a in ("W", "F", "FS") for s in ("demand", "map", "render", "detector", "unwarp"))
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
    return {"stage": "wire-FS", "workload": c["workload"]["wire"], "weights": "e060 parent EMA behind the arm FS forward",
            "weights_sha256": digest, "mapping": FS.mapping(source),
            "purpose": "wiring of code that otherwise runs only after the formal training: the evaluation batch path of arm FS and the "
                       "staged five-arm timing with the stored K and F endpoints; these readings are not results (F* is not trained "
                       "here, the protocol is shortened)",
            "timing": {k: timing[k] for k in ("ratios", "medians", "stages", "protocol", "forwards_per_endpoint", "order_executed", "weights_sha256")},
            "evaluation_batch": {"images": WIRE["eval_batches"] * K.EVAL_BATCH, "records_top600": len(preds[600]), "scored": False},
            "summary_structure": {"cost_keys": sorted(trial["cost_FS"]), "stage_table_arms": sorted(trial["cost_FS"]["stages"]["table"]),
                                  "pairs": list(trial["differences_pp"])},
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "scorer_calls": 0, "formal_state_usable": False}


# ================================================================ input pipeline acceptance (CUDA)
def deep_digest(value):
    """SHA-256 over a nested state value: tensors by dtype / shape / bytes, containers by type, length and keys."""
    from sparse_rtdetr.protocol_r18.engine import tensor_digest
    torch = K.torch_mod()
    h = hashlib.sha256()

    def walk(item, path):
        if isinstance(item, torch.Tensor):
            h.update((path + ":tensor:" + tensor_digest([(path, item)])).encode())
        elif isinstance(item, dict):
            keys = sorted(item, key=lambda k: (type(k).__name__, str(k)))
            h.update((path + ":dict:" + json.dumps([[type(k).__name__, str(k)] for k in keys])).encode())
            for k in keys:
                walk(item[k], f"{path}/{type(k).__name__}:{k}")
        elif isinstance(item, (tuple, list)):
            h.update((path + ":" + type(item).__name__ + ":" + str(len(item))).encode())
            for i, child in enumerate(item):
                walk(child, f"{path}/{i}")
        elif item is None or type(item) in (bool, int, float, str):
            h.update((path + ":" + type(item).__name__ + ":" + json.dumps(item, allow_nan=False)).encode())
        else:
            raise TypeError("unrecorded state value type: " + path + "/" + type(item).__name__)

    walk(value, "state")
    return h.hexdigest()


STATE_PARTS = ("version", "arm", "parent_sha256", "config", "activation_checkpoint", "model", "model_modes", "optimizer", "optimizer_names",
               "ema", "engine", "rng")


def state_digests(state):
    K.require(set(state) == set(STATE_PARTS), f"captured state keys: {sorted(state)}")
    out = {k: deep_digest(state[k]) for k in STATE_PARTS}
    out["rng_cpu"] = deep_digest({k: state["rng"][k] for k in ("python", "numpy", "torch_cpu")})
    out["rng_cuda"] = deep_digest(state["rng"]["cuda"])
    out["bn_buffers"] = deep_digest({k: v for k, v in state["model"].items()
                                     if k.endswith(("running_mean", "running_var", "num_batches_tracked"))})
    return out


class Digester:
    """Hashes captured CPU states in a thread, so the acceptance-only hashing does not sit in the timed cycle."""

    def __init__(self):
        self.q, self.out, self.error = queue.Queue(maxsize=2), {}, None
        self.thread = threading.Thread(target=self._run, name="r28-acceptance-digester", daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            job = self.q.get()
            if job is None:
                return
            try:
                self.out[job[0]] = state_digests(job[1])
            except BaseException as exc:
                self.error = exc

    def submit(self, key, state):
        K.require(self.error is None, f"state digest failed: {self.error!r}")
        self.q.put((key, state))

    def finish(self):
        self.q.put(None)
        self.thread.join()
        K.require(self.error is None, f"state digest failed: {self.error!r}")
        return self.out


def differences(a, b, path="", out=None, limit=40):
    """Every differing leaf between two JSON-like values (value or type)."""
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append({"path": f"{path}/{k}", "kind": "missing"})
            else:
                differences(a[k], b[k], f"{path}/{k}", out, limit)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append({"path": path, "kind": "length", "a": len(a), "b": len(b)})
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                differences(x, y, f"{path}/{i}", out, limit)
    elif type(a) is not type(b) or a != b:
        out.append({"path": path, "kind": "value", "a": a if not isinstance(a, (dict, list)) else "...", "b": b if not isinstance(b, (dict, list)) else "..."})
    return out


EXCLUDED = ["result.elapsed_seconds (wall time inside train_window)",
            "engineering.cycle_s", "engineering.wait_s", "engineering.train_call_s", "engineering.train_s",
            "engineering.acceptance_capture_s", "engineering.input.next_s", "engineering.input.audit_s", "engineering.input.fetch_s",
            "engineering.input.reference_check_s", "engineering.input.per_image_hash_s", "engineering.input.window_pixel_digest_s",
            "engineering.input.window_target_digest_s", "engineering.input.ready_at", "engineering.input.taken_at",
            "engineering.input.wait_s", "engineering.input.pipeline (the path name)", "engineering.prepared_after",
            "engineering.saved_with_prepared_windows", "engineering.wait_for_prepared_s", "engineering.save_s",
            "worker process ids (never part of the input record)"]
COMPARED = ["science.input (update, epoch, window, image / GT identities, augmentation seeds and trace, per-sample digests, "
            "whole-window pixel and target digests, augmentation strength)",
            "science.gt_count",
            "science.result (update, epoch, window, microsteps, lr, coefficients, microbatch losses, loss, normalizers, gradient norm "
            "and gradient digest before clipping, output digests and DN noise digests of both microbatches, finite, optimizer_steps, "
            "ema_updates)",
            "science.observer (the ledger's r28 row: magnification readings, map / tap readings, the calibration description)",
            "science.state.model (all parameters and buffers)", "science.state.bn_buffers", "science.state.model_modes",
            "science.state.optimizer", "science.state.optimizer_names", "science.state.ema", "science.state.engine (clocks, cursor, "
            "per-parameter optimizer steps)", "science.state.rng / rng_cpu (Python, NumPy, torch CPU) / rng_cuda",
            "science.state.version / arm / parent_sha256 / config / activation_checkpoint"]


def steady(rows, key, start):
    return [r["engineering"][key] for r in rows if r["science"]["input"]["update"] - K.PARENT_UPDATE >= start]


def path_timing(rows, start):
    """Wall-clock readings of one path.  cycle = commit to commit.  The acceptance-only work (state capture, the wait
    for a prepared window and the save of the overlapped path) is removed from the compared cycle."""
    eng = [r["engineering"] for r in rows]
    sel = [e for r, e in zip(rows, eng) if r["science"]["input"]["update"] - K.PARENT_UPDATE >= start]
    extra = lambda e: e["acceptance_capture_s"] + e.get("wait_for_prepared_s", 0.0) + e.get("save_s", 0.0)
    mean = lambda f: statistics.fmean(f(e) for e in sel)
    segments = ("wait_s", "train_call_s", "train_s", "acceptance_capture_s")
    inputs = ("next_s", "fetch_s", "audit_s", "per_image_hash_s", "window_pixel_digest_s", "window_target_digest_s", "reference_check_s")
    out = {"updates": len(rows), "steady_updates": len(sel),
           "first_update": {"cycle_s": eng[0]["cycle_s"], "wait_for_first_window_s": eng[0]["wait_s"], "train_call_s": eng[0]["train_call_s"],
                            "note": "includes the spawn of the loader workers and the first CUDA call"},
           "steady_cycle_measured_s": mean(lambda e: e["cycle_s"]), "steady_cycle_s": mean(lambda e: e["cycle_s"] - extra(e)),
           "steady_cycle_median_s": statistics.median(e["cycle_s"] - extra(e) for e in sel),
           "steady_segments_mean_s": {**{k: mean(lambda e, k=k: e[k]) for k in segments},
                                      "input_check_before_the_train_timer_s": mean(lambda e: e["train_call_s"] - e["train_s"]),
                                      "consumer_other_s": mean(lambda e: e["cycle_s"] - extra(e) - e["wait_s"] - e["train_call_s"]),
                                      **{f"input_{k}": mean(lambda e, k=k: e["input"][k]) for k in inputs}},
           "note": "segments of the two threads of the overlapped path run at the same time: they do not add up to the cycle"}
    work = [e["cycle_s"] - extra(e) - e["wait_s"] for e in sel]                       # consumer work of each steady cycle
    produce = [e["input"]["next_s"] + e["input"]["reference_check_s"] for e in sel]   # producer work of each steady window
    out["steady_consumer_work_s"], out["steady_producer_work_s"] = statistics.fmean(work), statistics.fmean(produce)
    out["steady_bound_s"] = statistics.fmean(max(w, p) for w, p in zip(work, produce))
    return out


def decide(serial, overlap, equal, restore):
    measured = overlap["steady_cycle_s"] < serial["steady_cycle_s"]
    bound = overlap["steady_bound_s"] < serial["steady_cycle_s"]
    enabled = bool(equal and restore and measured and bound)
    return {"overlap_enabled": enabled, "equivalence": bool(equal), "restore": bool(restore),
            "measured_steady_cycle_shorter": bool(measured), "conservative_bound_shorter": bool(bound),
            "serial_steady_cycle_s": serial["steady_cycle_s"], "overlap_steady_cycle_s": overlap["steady_cycle_s"],
            "overlap_steady_bound_s": overlap["steady_bound_s"],
            "measured_speedup": serial["steady_cycle_s"] / overlap["steady_cycle_s"],
            "bound_speedup": serial["steady_cycle_s"] / overlap["steady_bound_s"],
            "bound": "max(consumer work, producer work) per steady cycle: the cycle of the overlapped path if the acceptance-only "
                     "work had given the producer no extra time; the serial cycle already benefits from that extra time through "
                     "the loader's own prefetch, so both adjustments are against the overlapped path",
            "formal_input_pipeline": "overlap" if enabled else "serial"}


def pipeline(c, out, counts):
    import l0_loop as LL
    import r28_loop as LP
    import r28_pipeline as PL
    from s15_formal_train import _result_body
    from sparse_rtdetr.research import r28_fovea_cal as FS
    from sparse_rtdetr.research import scale_baseline as S
    torch = K.torch_mod()
    out = Path(out)
    K.require(out.resolve() == (K.ROUND / "pipeline").resolve() and K.ARM == "FS" and K.followup_arm(c) == "FS",
              "this action writes <round>/pipeline with R28_ARM=FS")
    for name in ("cal-check", "calibrate", "cal-fixture", "wire-FS"):
        K.stage_closed(c, name, "FS")
    source = K.demand_source(c, "FS")
    steps, save_at, steady_from = PIPE["steps"], PIPE["save_at"], PIPE["steady_from"]
    if os.environ.get(DEV_PIPE):
        K.require(K.DEV, f"{DEV_PIPE} is refused outside a dev round")
        steps, save_at = (int(v) for v in os.environ[DEV_PIPE].split(","))
        steady_from = min(steady_from, max(2, steps - 1))
    K.require(1 <= save_at < steps, "pipeline acceptance span")
    refs, _ = LL.reference_rows(c)
    check = reference_check(refs)
    data = S.train_dataset(c["data"]["train_core"], S.S15_HW, seed=0)
    p0 = K.PARENT_UPDATE
    digester = Digester()
    state_file = out / f"overlap-u{save_at:02d}.pth"
    info = {}

    def run(name, mode, first, last, *, load=None, save_after=None):
        K.PHASE["phase"] = f"pipeline:{name} construct"
        e, checks = FS.create(c["parent"]["path"], "cuda:0", "FS", source)
        K.register_counts(e, counts)
        obs = LP.install_observer(e)
        loaded = None if load is None else K.load(c, e, load, "pipeline-acceptance")
        K.require(e.completed_updates == first and e.r28_mapping == FS.mapping(source), f"{name}: start update / mapping")
        rows, saved, window = [], None, None
        torch.cuda.reset_peak_memory_stats()
        with PL.MemorySampler(0.5) as memory:
            opened = mark = time.perf_counter()
            steady_at = None
            with PL.formal_stream(mode, data, first, last, check, fine=True) as stream:
                for item in stream:
                    x, t, inp = item.x, item.t, item.inp
                    K.PHASE["phase"] = f"pipeline:{name} update {e.completed_updates + 1}"
                    K.require(inp["update"] == e.completed_updates + 1, f"{name}: window {inp['update']} is not the next update")
                    bad = LL.input_differences(inp, refs[inp["update"]]["input"])
                    K.require(not bad, f"{name}: input record differs from R27 formal-P before update {inp['update']}: {bad}")
                    if window is None:
                        window = PL.window_bytes(x, t)
                    if inp["update"] - p0 == steady_from:
                        steady_at = time.perf_counter()
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
                        saved = K.save(c, e, state_file, "pipeline-acceptance")
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
        info[name] = {"mode": mode, "first_update": first + 1, "last_update": last, "updates": len(rows), "account": account,
                      "resumed_from": None if loaded is None else loaded["file"], "saved": saved,
                      "wall_seconds_stream": closed - opened, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                      "memory": {"whole_path": memory.summary(), "steady_state": memory.summary(since=steady_at),
                                 "plan": PL.memory_plan(window, mode)},
                      "create_checks": {k: checks[k] for k in ("arm", "identity_sha256", "demand", "mapping", "new_parameters")}}
        del e, obs
        gc.collect()
        torch.cuda.empty_cache()
        return rows

    serial = run("serial", "serial", p0, p0 + steps)
    overlap = run("overlap", "overlap", p0, p0 + steps, save_after=p0 + save_at)
    resumed = run("resumed", "overlap", p0 + save_at, p0 + steps, load=state_file)
    K.PHASE["phase"] = "pipeline:state digests"
    digests = digester.finish()
    for name, rows in (("serial", serial), ("overlap", overlap), ("resumed", resumed)):
        for r in rows:
            r["science"]["state"] = digests[(name, r["science"]["input"]["update"])]
        with (out / f"STEPS-{name}.jsonl").open("x") as f:
            for r in rows:
                f.write(json.dumps(r, separators=(",", ":"), allow_nan=False) + "\n")
    K.PHASE["phase"] = "pipeline:comparison"
    sci = lambda rows: [r["science"] for r in rows]
    d_so = differences(sci(serial), sci(overlap))
    d_rs = differences(sci(serial)[save_at:], sci(resumed))
    d_ro = differences(sci(overlap)[save_at:], sci(resumed))
    equal, restore = not d_so, not d_rs and not d_ro
    K.require(info["overlap"]["saved"] is not None and overlap[save_at - 1]["engineering"]["saved_with_prepared_windows"] == 1
              and info["overlap"]["account"]["max_prepared_ahead"] == 1 and info["resumed"]["account"]["max_prepared_ahead"] == 1
              and info["serial"]["account"]["max_prepared_ahead"] == 0, "prepared-window accounting of the three paths")
    want = steps * 2 + (steps - save_at)
    K.require(counts["actual_updates"] == want and counts["forwards"] == counts["forward_attempts"] == 2 * want, f"pipeline counts {dict(counts)}")
    equivalence = {"schema": "r28-pipeline-equivalence-v1", "contract_sha256": c["contract_sha256"], "mapping": FS.mapping(source),
                   "paths": info, "steps": {"serial": [p0 + 1, p0 + steps], "overlap": [p0 + 1, p0 + steps], "resumed": [p0 + save_at + 1, p0 + steps]},
                   "saved_after_update": p0 + save_at, "prepared_windows_at_the_save": 1,
                   "compared": COMPARED, "excluded": EXCLUDED,
                   "serial_vs_overlap": {"steps": steps, "differences": d_so, "equal": not d_so},
                   "serial_vs_resumed": {"steps": steps - save_at, "differences": d_rs, "equal": not d_rs},
                   "overlap_vs_resumed": {"steps": steps - save_at, "differences": d_ro, "equal": not d_ro},
                   "restore": "K.load compares the complete restored state with the saved one (exact) before the replay",
                   "final_state_digests": {n: r[-1]["science"]["state"] for n, r in (("serial", serial), ("overlap", overlap), ("resumed", resumed))},
                   "steps_files": {n: K.ref(out / f"STEPS-{n}.jsonl") for n in ("serial", "overlap", "resumed")},
                   "updates": want, "complete_detector_forwards": 2 * want}
    K.write_new(out / "PIPELINE_EQUIVALENCE.json", equivalence)
    K.require(equal and restore, f"input pipeline equivalence failed: serial/overlap {d_so[:5]}, serial/resumed {d_rs[:5]}, overlap/resumed {d_ro[:5]}")
    t_serial, t_overlap, t_resumed = path_timing(serial, steady_from), path_timing(overlap, steady_from), path_timing(resumed, save_at + 2)
    decision = decide(t_serial, t_overlap, equal, restore)
    timing = {"schema": "r28-pipeline-timing-v1", "contract_sha256": c["contract_sha256"], "steady_state_from_update": steady_from,
              "serial": t_serial, "overlap": t_overlap, "resumed": t_resumed, "decision": decision, "rule": c["input_pipeline"]["decision"],
              "memory": {n: info[n]["memory"] for n in info}, "peak_cuda_allocated_bytes": {n: info[n]["peak_cuda_allocated_bytes"] for n in info},
              "checkpoint": {"save_s": overlap[save_at - 1]["engineering"]["save_s"], "wait_for_prepared_s": overlap[save_at - 1]["engineering"]["wait_for_prepared_s"],
                             "restore_and_first_window": {"wait_for_first_window_s": resumed[0]["engineering"]["wait_s"],
                                                          "first_cycle_s": resumed[0]["engineering"]["cycle_s"]},
                             "state_bytes": info["overlap"]["saved"]["file"]["bytes"]},
              "reference": {"formal_F_serial_seconds_per_update": 40645.923253592104 / 18240, "formal_F_train_window_seconds": 20918.401087376755 / 18240,
                            "source": "F round formal-F/CLOSED.json (loop_wall_seconds, update_seconds_sum); another process and day"},
              "dev": os.environ.get(DEV_PIPE),
              "note": "training engineering only: not a method or inference speed-up; acceptance updates carry capture=True digests "
                      "and a full state capture per update, which the formal loop does not"}
    K.write_new(out / "PIPELINE_TIMING.json", timing)
    print(json.dumps({"overlap_enabled": decision["overlap_enabled"], "serial_steady_cycle_s": t_serial["steady_cycle_s"],
                      "overlap_steady_cycle_s": t_overlap["steady_cycle_s"], "overlap_steady_bound_s": t_overlap["steady_bound_s"]}), flush=True)
    return {"stage": "pipeline", "workload": c["workload"]["pipeline"], "mapping": FS.mapping(source), "overlap_enabled": decision["overlap_enabled"],
            "decision": decision, "equivalence": K.ref(out / "PIPELINE_EQUIVALENCE.json"), "timing": K.ref(out / "PIPELINE_TIMING.json"),
            "serial_vs_overlap_equal": True, "restore_replay_equal": True, "compared_steps": {"serial_vs_overlap": steps, "replayed": steps - save_at},
            "saved_state": info["overlap"]["saved"], "updates": want, "complete_detector_forwards": 2 * want,
            "peak_allocated_bytes": max(v["peak_cuda_allocated_bytes"] for v in info.values()), "formal_state_usable": False}


# ================================================================ summary-FS
def summarize(c, out, counts):
    import r28_eval as E
    fu = K.followup(c)
    K.require(fu is not None and K.followup_arm(c) == "FS" and K.ARM == "FS", "the calibrated-control round only summarises arm FS")
    K.PHASE["phase"] = "summarize/read"
    out = E.round_stage(out, "summary-FS")
    w_result, ev_w, ev_f, values, apl, apm, quoted = _inputs(c)
    ev, cost = K.stage_closed(c, "eval-FS", "FS"), K.stage_closed(c, "cost-FS", "FS")
    lock = K.calibration(c)
    cal_closed, pipe = K.stage_closed(c, "calibrate", "FS"), K.stage_closed(c, "pipeline", "FS")
    ws = cost["weights_sha256"]
    K.require(ws["FS"] == ev["weights_sha256"] and ws["W"] == ev_w["weights_sha256"] == fu["W_weights_sha256"]
              and ws["F"] == ev_f["weights_sha256"] and ws["K"] == fu["K_weights_sha256"] and ws["L0"] == c["L0_weights_sha256"],
              "the cost-FS stage timed other models than the evaluated ones")
    K.require(ev["demand"]["file"] == fu["W_demand"] and ev["demand"] == ev_w["demand"] == ev_f["demand"]
              and ev["mapping"]["fwhm_hex"] == lock["fwhm"].hex() and ev["mapping"]["calibration"] == lock["calibration"],
              "arm FS was not evaluated with arm W's demand network and the locked width")
    for b in K.BUDGETS:
        K.pinned(ev["metrics"][str(b)]), K.pinned(ev["predictions"][str(b)])
    K.PHASE["phase"] = "summarize/compute"
    res = summary_fs(c, w_result, {**values, "FS": ev["values"]}, {**apl, "FS": ev["per_class_APL_top600"]},
                     {**apm, "FS": ev["per_class_APM_top600"]}, cost, quoted)
    tables = FC.tensor_tables(K.pinned(ev["cocoeval_top600_tensors"]))
    f600 = ev["values"]["600"]
    consistency = {k: (None if tables["macro"][k] is None else tables["macro"][k] - f600[k]) for k in tables["macro"]}
    formal_dir = K.formal_dir(c, "FS")
    formal = K.read_json(formal_dir / "CLOSED.json")
    calibration = K.read_json(K.pinned(lock["calibration"]))
    res.update(status="PASS", registration=fu["registration"], authorization_basis=c["authorization_basis"], task=fu["task"],
               review=fu["review"], engineering=fu["engineering"], contract_sha256=c["contract_sha256"], source_commit=c["source_commit"],
               official=fu["official"], fovea_cal_definition=c["fovea_cal_definition"], identity_sha256_FS=c["identity_sha256"]["FS"],
               calibration={"file": lock["calibration"], "status": calibration["status"], "h_star": calibration["h_star"],
                            "h_star_hex": calibration["h_star_hex"], "L_W": calibration["L_W"], "L_h_star": calibration["L_h_star"],
                            "relative_error": calibration["relative_error"], "rule_applied": calibration["rule_applied"],
                            "interval": calibration["interval"], "widths_read": len(calibration["readings"]),
                            "invalid_widths": [r["h"] for r in calibration["readings"] if not r["valid"]],
                            "calibration_set": calibration["calibration_set"], "W": calibration["W"], "F_width_4": calibration["F_width_4"],
                            "F_star": calibration.get("F_star"), "confirmation": calibration["confirmation"]},
               W_closed_round={"contract_sha256": fu["prior_contract_sha256"], "result": fu["prior_W_result"], "decision": w_result["decision"],
                               "M_W_minus_L0": w_result["M_W_minus_L0"], "APL6": w_result["APL6"], "C": w_result["C"],
                               "note": "quoted unchanged from the closed round; not recomputed, not overwritten"},
               F_round={"contract_sha256": fu["F_contract_sha256"], "result": fu["F_result"], "note": "quoted unchanged from the F round"},
               FS_per_class_top600={**tables, "source": ev["cocoeval_top600_tensors"], "macro_minus_scored_value": consistency,
                                    "note": "read from the tensors kept by the single per-class COCOeval of eval-FS; no rescoring"},
               FS_demand=ev["demand"], FS_mapping=ev["mapping"], FS_endpoint=ev["endpoint"], FS_weights_sha256=ev["weights_sha256"],
               FS_training_observations=FC.formal_observations(formal_dir),
               FS_calibration_over_the_schedule=formal_calibration(formal_dir),
               input_pipeline={"acceptance": {"equivalence": pipe["equivalence"], "timing": pipe["timing"], "decision": pipe["decision"]},
                               "formal": {"mode": formal["plan"]["input_pipeline"], "account": formal["input_pipeline"],
                                          "loop_wall_seconds": formal["loop_wall_seconds"], "update_seconds_sum": formal["update_seconds_sum"],
                                          "seconds_per_update": formal["loop_wall_seconds"] / formal["updates"],
                                          "timing_file": K.ref(formal_dir / formal["timing_file"])},
                               "reference_serial_F": {"seconds_per_update": 40645.923253592104 / 18240, "source": "F round formal-F/CLOSED.json"},
                               "note": STATEMENTS["cost"]},
               stages=E._stage_account(c, ["cal-check", "calibrate", "cal-fixture", "wire-FS", "pipeline", formal_dir.name, "eval-FS", "cost-FS"]),
               other_directories=E._other_directories(), statements=STATEMENTS)
    K.require(cal_closed["h_star_hex"] == calibration["h_star_hex"] == formal["plan"]["mapping"]["fwhm_hex"], "one width from the calibration to the endpoint")
    K.require(counts["forwards"] == 0 and counts["actual_updates"] == 0 and not counts.get("score_budget_calls"),
              "summary must not run a model or a scorer")
    K.write_new(out / "RESULT.json", res)
    p = res["primary_W_minus_FS_pp"]
    return {"result": K.ref(out / "RESULT.json"), "registration": fu["registration"], "label": res["label"],
            "contribution": res["contribution"], "record_only": True, "h_star": calibration["h_star"],
            "W_minus_FS_APS600_pp": p["W-FS top600 APS"], "W_minus_FS_AP75_small600_pp": p["W-FS top600 AP75_small"],
            "FS_minus_F_APS600_pp": res["differences_pp"]["FS-F"]["600"]["APS"], "FS_minus_L0_APS600_pp": res["differences_pp"]["FS-L0"]["600"]["APS"],
            "W_decision_unchanged": w_result["decision"], "model_gpu_FS_over_L0": res["cost_FS"]["model_gpu"]["FS/L0"],
            "model_gpu_W_over_FS": res["cost_FS"]["model_gpu"]["W/FS"],
            "small_FS_over_W_minus_1_over_the_schedule": res["FS_calibration_over_the_schedule"]["overall"]["small"]["FS_over_W_minus_1"]}
