"""R28-K exploratory follow-up after the W closure (TASK_R28_K_FOLLOWUP_003).

Registration: a fixed-prior control K added AFTER the W result was seen and the fixed W configuration was closed
(closed_fixed_config, K_trigger=false).  It is authorised by the follow-up task, not by the original chain; the W
decision of the closed round is never changed and never consulted as a trigger.

This module only adds what the independent round needs; the K science is the code of the closed round, unchanged:
  prepare          contract of a new round: authorization basis, the task file, read-only references into the closed
                   R28 round (contract, release manifest, W result, frozen prior, W endpoint and demand network, the
                   PASS receipts of its stages), and the file-by-file difference between this release and that one.
  followup_check   CPU, affected interfaces only: prior-round identities, authorisation and output isolation, the
                   frozen prior as arm K's demand source, the summary keeping the W decision, and per_class_sizes
                   against the stored per-class large AP of W and L0 (one COCOeval of each stored top600 list).
  per_class_sizes  the single per-class COCOeval of an evaluation, reading medium next to large.
  medium           CPU, descriptive (task section 5): per-class APM600 of W / L0 / K and same-class candidate coverage
                   of the certified medium (and small) GT grouped by the stored magnification.  No model forward.
  summarize        summary-K of a follow-up round: the reused K arithmetic plus the unchanged W decision.
Accept, formal training, evaluation and timing are the original r28_checks.accept, r28_loop.formal, r28_eval.evaluate
and r28_cost.cost with R28_ARM=K; r28_common resolves the inherited stages and the prior W endpoint.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import r28_common as K

REGISTRATION = "R28-K exploratory follow-up after W closure"
PRIOR_ROUND = K.LD_ROOT / "r28-budgeted-resampling-20261113-001"
TASK = ("TASK_R28_K_FOLLOWUP_003.md", "814ce55113dcf22da524f084ed9e700d2888502cf9ed96a8499ee49849f40cf6")
PIN = {  # TASK_R28_K_FOLLOWUP_003 section 2
    "commit": "59e367de86cb55dbe1fe6606962d4f0bf65458e3",
    "manifest": "a65b4924df3fd1194f9a92f8b10e9da7fa3b7f9e5337682ad5ea48bb6d6f514e",
    "contract_file": "d8d4c8f844161038499ce9da4c1691d1fbfff51ec8f6bf47e44bcf034fac3c30",
    "W_result": "75c4abfe3c6574544820cb1e235ca78ff77b9f20f71488f591461662b518edd0",
    "prior": "e6afe56facc31536867a489af1a16a8977f7cb39ac27cf38012a0cbc166fda9c",
    "parent": "a2b6412e3c85de7592829f24aff09e355fdd11cb53798bc498bd04b882e1bf8a",
    "W_endpoint": "a88f2a1541fb911fda7bc7f9426892611ee2b5475aed6abf7df77fa5a6feaf37",
    "W_demand": "c9ffc7505e47adbcc45d70949c3d87371c6f3a7795f1aaaab886d98b15ee9e5a",
}
INHERITED = ("cpu", "demand", "prior", "g0", "zero", "accept-W", "evalpath", "precost", "formal-W", "eval-W", "cost", "summary-W")
W_DECISION = {"label_key": "closed", "K_trigger": False, "exploratory": False, "label": "closed_fixed_config"}
ALLOWED_CHANGED = ("tools/r28/r28_common.py", "tools/r28/r28_eval.py", "tools/r28/r28_loop.py", "tools/r28/r28_run.py")
ALLOWED_ADDED = ("tests/test_r28_followup.py", "tools/r28/r28_followup.py", "tools/r28/r28_k_followup_chain.sh")
TESTS = ("tests/test_r28_warp.py", "tests/test_r28_resample.py", "tests/test_r28_summary.py", "tests/test_r28_followup.py")
THRESHOLDS = (0.5, 0.75)
ARMS3 = ("L0", "W", "K")
PAIRS = (("W", "L0"), ("K", "L0"), ("W", "K"))
SIZES = {"medium": ("APM", 4), "large": ("APL", 5)}
WORKLOAD = {
    "followup-check": {"updates": 0, "forwards": 0, "stored_list_COCOeval_calls": 2},
    "accept": K.WORKLOAD["accept"],
    "formal": K.WORKLOAD["formal"],
    "evaluation": K.WORKLOAD["evaluation"],
    "cost-K": K.WORKLOAD["cost-K"],
    "medium": {"updates": 0, "forwards": 0, "COCOeval_calls": 0},
    "planned_complete_detector_forwards": 6 + 2 * (K.ENDPOINT - K.PARENT_UPDATE) + (K.EVAL_WARMUP + K.EVAL_BATCHES)
    + len(K.COST_ENDPOINTS["cost-K"]) * K.COST_PER_ENDPOINT,
    "planned_updates": 3 + (K.ENDPOINT - K.PARENT_UPDATE),
    "not_run": ["demand network training", "G0", "W training", "W pre-launch cost gate", "W evaluation"],
}
STATEMENTS = {
    "registration": "本对照是看过 W 结果、W 固定配置已关闭之后追加的探索性控制（post-hoc），不是原链自动触发，也不是未看 W 结果的预注册检验。",
    "W_decision": "W 的 closed_fixed_config（APM600 未过）与 K_trigger=false 原样保留；K 的任何读数都不追认或改写 W 的判定。",
    "A": "A 沿用原定义：W−K 的 APS600 ≥ +0.30 pp（未舍入值）。达到记 adaptive_increment_reached_in_seed0；"
         "未达到记 adaptive_increment_not_established，不是等效性证明。",
    "A_scope": "A 达到也不说明自适应整体更优：须连同 W−K 的 APM／APL 与实际时间一并解释。",
    "K_minus_L0": "把原七格阈值逐格应用于 K−L0 只是对新 K 基线的展示，不以其过／不过追认 W。",
    "contribution": "W−K 不等于有界分配相对已有采样器（FOVEA／LZU／ZoomDet）的独有贡献；发表主张与跨 seed 稳定性均未建立。",
    "single_seed": "单 seed（seed0）、重复使用的 548 图 development 集上的读数。",
    "cost": "L0／W／K 同进程测速的新比值单列，不覆盖 W 轮已登记的 C。",
    "medium": "medium 分解是描述：APM 下降不必来自几何覆盖下降，也可能涉及排序、误检或类别构成；分组关联不作因果归因，"
              "也不据此修改 ω、ε、倍率界或增加保护模块。",
}


# ================================================================ contract of the independent round
def source_delta(old_files, new_files):
    """File-by-file difference between the closed round's release and this one; the science tree must be identical."""
    changed = sorted(k for k in old_files if k in new_files and old_files[k] != new_files[k])
    added = sorted(k for k in new_files if k not in old_files)
    removed = sorted(k for k in old_files if k not in new_files)
    K.require(not removed and set(changed) <= set(ALLOWED_CHANGED) and set(added) <= set(ALLOWED_ADDED),
              f"release differs from the closed round outside the follow-up adapter: changed {changed}, added {added}, removed {removed}")
    K.require(not any(p.startswith(("src/", "vendor/", "configs/")) for p in changed + added), "follow-up release touches the science tree")
    return {"changed": changed, "added": added, "removed": removed, "unchanged_files": len(old_files) - len(changed),
            "science_tree_identical": True,
            "note": "src/ (operator, wrapper, detector, criterion, data) and every other tool are byte-identical to the "
                    "release of the closed round; only the listed adapter files differ"}


def prior_references(prior=None):
    """Read-only references into the closed R28 round, each checked against the task's pinned identities."""
    prior = Path(prior or PRIOR_ROUND).resolve()
    contract = K.ref(prior / "CONTRACT.json")
    K.require(contract["sha256"] == PIN["contract_file"], "prior CONTRACT.json file SHA")
    c1 = K.read_json(prior / "CONTRACT.json")
    body = {k: v for k, v in c1.items() if k != "contract_sha256"}
    K.require(c1["contract_sha256"] == K.sha_bytes(K.canonical(body)), "prior contract inner digest")
    K.require(c1["source_commit"] == PIN["commit"] and c1["source_manifest"]["sha256"] == PIN["manifest"]
              and c1["parent"]["sha256"] == PIN["parent"] == K.PARENT_SHA and Path(c1["outputs"]).resolve() == prior
              and c1.get("followup") is None, "prior contract identities")
    K.pinned(c1["source_manifest"]), K.pinned(c1["parent"])
    inherited = {}
    for name in INHERITED:
        r = K.ref(prior / name / "CLOSED.json")
        closed = K.read_json(r["path"])
        K.require(closed.get("status") == "PASS" and closed.get("contract_sha256") == c1["contract_sha256"],
                  f"prior stage {name} is not PASS under the prior contract")
        inherited[name] = r
    closed = {n: K.read_json(inherited[n]["path"]) for n in ("summary-W", "prior", "demand", "eval-W", "formal-W", "g0")}
    w_result = closed["summary-W"]["result"]
    K.require(w_result["sha256"] == PIN["W_result"], "prior summary-W/RESULT.json SHA")
    decision = K.read_json(K.pinned(w_result))["decision"]
    K.require(decision == W_DECISION and closed["summary-W"]["K_trigger"] is False and closed["summary-W"]["label_key"] == "closed",
              f"prior W decision is not the closed one: {decision}")
    prior_file, w_demand, w_endpoint = closed["prior"]["file"], closed["demand"]["weights"], closed["eval-W"]["endpoint"]
    K.require(prior_file["sha256"] == PIN["prior"] and w_demand["sha256"] == PIN["W_demand"] and w_endpoint["sha256"] == PIN["W_endpoint"],
              "prior / W demand network / W endpoint SHA")
    K.require(closed["formal-W"]["arm"] == "W" and closed["eval-W"]["formal_stage"] == "formal-W"
              and Path(w_endpoint["path"]).parent == prior / "formal-W", "prior W endpoint location")
    for r in (prior_file, w_demand, w_endpoint, closed["g0"]["files"]["per_gt"], *closed["eval-W"]["predictions"].values(),
              *closed["eval-W"]["metrics"].values()):
        K.pinned(r)
    return c1, {"prior_round": str(prior), "prior_contract": contract, "prior_contract_sha256": c1["contract_sha256"],
                "prior_commit": c1["source_commit"], "prior_manifest": c1["source_manifest"], "prior_W_result": w_result,
                "prior_W_decision": decision, "inherited": inherited, "prior_file": prior_file,
                "prior_state_sha256": closed["prior"]["state_sha256"], "W_demand": w_demand, "W_endpoint": w_endpoint,
                "W_endpoint_side": K.ref(K.side_of(w_endpoint["path"])), "W_weights_sha256": closed["eval-W"]["weights_sha256"],
                "W_predictions": closed["eval-W"]["predictions"], "W_metrics": closed["eval-W"]["metrics"],
                "per_gt": closed["g0"]["files"]["per_gt"]}


def prepare(_args=None):
    from sparse_rtdetr.research import r28_resample as R
    from sparse_rtdetr.research import r28_warp as W
    round_, prior = K.ROUND.resolve(), PRIOR_ROUND.resolve()
    K.require(K.ROOT.resolve().parent.parent == round_ and K.ROOT.name == "source", "prepare-followup must run from a frozen release")
    K.require(round_ != prior and round_.parent != prior and "R28_ROUND" in os.environ,
              "the follow-up round must be an independent directory named by R28_ROUND")
    K.require(not K.DEV or round_.parent.parent == K.LD_ROOT.resolve(), "a follow-up dev round lives inside the follow-up round")
    manifest = K.read_json(K.ROOT.parent / "MANIFEST.json")
    K.require(manifest["files"] == K.B.source_manifest(K.ROOT), "release source differs from MANIFEST")
    K.require(not manifest.get("dirty", False) or K.DEV, "a release assembled from a dirty work tree is only accepted in a dev round")
    task = K.ref(K.ROUND / TASK[0])
    K.require(task["sha256"] == TASK[1], f"task file SHA: {TASK[0]}")
    c1, refs = prior_references()
    delta = source_delta(K.read_json(c1["source_manifest"]["path"])["files"], manifest["files"])
    K.require(c1["identity_sha256"] == R.IDENTITY_SHA256 and c1["identity"] == R.IDENTITY and c1["definition"] == W.DEFINITION,
              "R28 behaviour identity differs from the closed round")
    c = {k: v for k, v in c1.items() if k != "contract_sha256"}
    c.update(version="R28-K-followup-003-contract-v1", authorization_basis=K.FOLLOWUP_BASIS,
             source_root=str(K.ROOT.resolve()), source_manifest=K.ref(K.ROOT.parent / "MANIFEST.json"),
             source_commit=manifest["commit"], source_dirty=bool(manifest.get("dirty", False)),
             workload=WORKLOAD,
             labels={**c1["labels"], "A_passed": "adaptive_increment_reached_in_seed0", "A_failed": "adaptive_increment_not_established"},
             resource_plan={"state_bytes": c1["parent"]["bytes"], "formal_states": len(K.SAVE_EPOCHS),
                            "disk_free_bytes_at_prepare": shutil.disk_usage(K.ROUND).free,
                            "gpu_time_reference": "task estimate 11-13 h (W measured 11.31 h, L0 11.95 h); not a cap"},
             outputs=str(K.ROUND), formal_started=False,
             followup={"registration": REGISTRATION, "authorization_basis": K.FOLLOWUP_BASIS, "task": task, **refs,
                       "source_delta": delta,
                       "arm": "K", "trained_here": ["K detector, 18,240 updates from the e060 parent"],
                       "reused_frozen": ["K prior (prior/prior.json of the closed round)", "W e120 endpoint and demand "
                                         "network (timing and identity only)", "W / L0 predictions and metrics"],
                       "statements": STATEMENTS})
    c["contract_sha256"] = K.sha_bytes(K.canonical(c))
    K.write_new(K.ROUND / "CONTRACT.json", c)
    print(json.dumps({"contract_sha256": c["contract_sha256"], "identity_sha256": R.IDENTITY_SHA256,
                      "authorization_basis": c["authorization_basis"], "source_delta": {k: delta[k] for k in ("changed", "added")}}))
    return c


# ================================================================ per-class COCOeval (medium next to large)
def per_class_sizes(rows, parsed, predictions, bridge, keep=None):
    """One COCOeval identical to s15_formal_eval.per_class_large; returns (large rows, medium rows).
    The large rows have the format and the values of per_class_large; the medium rows are read from the same tensor.
    keep: a path that receives the accumulated tensors of this single COCOeval (so later tables need no rescoring)."""
    import contextlib
    import io
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from s15_formal_eval import LARGE_NAMES
    from sparse_rtdetr.r0.evaluation import bridge_inputs
    dataset, det, _ = bridge_inputs(rows, parsed, predictions)
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(); coco.dataset = dataset; coco.createIndex()
        dt = coco.loadRes(det)
        e = COCOeval(coco, dt, "bbox")
        e.params.imgIds = sorted(r["id"] for r in dataset["images"]); e.params.catIds = list(range(1, 11))
        e.params.maxDets = [1, 10, 100]
        e.evaluate(); e.accumulate(); e.summarize()
    K.require(all(float(v) == bridge["metrics"][k] for k, v in zip(LARGE_NAMES, e.stats)),
              "per-class recomputation differs from the scored COCO bridge")
    out = {}
    for label, (key, stat) in SIZES.items():
        ix = e.params.areaRngLbl.index(label)
        lo, hi = e.params.areaRng[ix]
        values = []
        for k, cat in enumerate(e.params.catIds):
            p = e.eval["precision"][:, :, k, ix, -1]
            v = p[p > -1]
            n = sum(1 for a in dataset["annotations"] if a["category_id"] == cat and not a.get("iscrowd") and lo <= a["area"] <= hi)
            values.append({"category_id": int(cat), "GT": n, key: float(v.mean()) if len(v) else None})
        mean = float(np.mean([v[key] for v in values if v[key] is not None]))
        K.require(abs(mean - e.stats[stat]) < 1e-14, f"per-class {key} macro differs")
        out[key] = values
    if keep is not None:
        keep_tensors(e, dataset, keep)
    return out["APL"], out["APM"]


def keep_tensors(e, dataset, path):
    """The accumulated arrays of a finished COCOeval, with the axes needed to read them, as one .npz file."""
    import numpy as np
    path = Path(path)
    K.require(not path.exists(), f"refusing to overwrite {path}")
    p = e.params
    gt = [[sum(1 for a in dataset["annotations"] if a["category_id"] == cat and not a.get("iscrowd") and lo <= a["area"] <= hi)
           for lo, hi in p.areaRng] for cat in p.catIds]
    with path.open("xb") as f:
        np.savez(f, precision=e.eval["precision"], recall=e.eval["recall"], scores=e.eval["scores"], stats=np.asarray(e.stats),
                 iouThrs=np.asarray(p.iouThrs), recThrs=np.asarray(p.recThrs), catIds=np.asarray(p.catIds),
                 areaRng=np.asarray(p.areaRng, dtype=np.float64), areaRngLbl=np.asarray(p.areaRngLbl), maxDets=np.asarray(p.maxDets),
                 gt_count=np.asarray(gt), axes=np.asarray(["iou", "recall", "category", "area", "maxDets"]))
    return path


def _development(c):
    from sparse_rtdetr.research import scale_baseline as S
    data = S.development(c["data"]["development"], S.S15_HW)
    rows = data.records
    K.require([r["stable_image_id"] for r in rows] == c["development_ids"] and len(rows) == 548, "development members/order")
    return rows, [data.raw(row) for row in rows]


def stored_list_sizes(c, rows, parsed, predictions, metrics, stored_apl):
    """per_class_sizes of one stored top600 list; its large rows must equal the per-class large AP stored by the
    formal evaluation of that list (which used per_class_large)."""
    t = time.perf_counter()
    preds = K.read_json(K.pinned(predictions))
    bridge = K.read_json(K.pinned(metrics))["COCO_bridge"]
    apl, apm = per_class_sizes(rows, parsed, preds, bridge)
    K.require(apl == stored_apl, "per_class_sizes large rows differ from the stored per_class_large rows")
    return {"per_class_APL_top600": apl, "per_class_APM_top600": apm, "APL_rows_equal_stored_per_class_large": True,
            "predictions": predictions, "metrics": metrics, "COCOeval_calls": 1, "seconds": time.perf_counter() - t}


# ================================================================ affected-interface check (CPU)
def _refused(label, fn):
    try:
        fn()
    except ValueError as exc:
        return {"case": label, "refused": True, "message": str(exc)[:200]}
    raise ValueError(f"follow-up refusal case was accepted: {label}")


def followup_check(c, out, counts):
    import r28_cost  # noqa: F401  (import check of the reused timing module)
    import r28_eval as E
    import r28_loop as LP
    from sparse_rtdetr.research import r28_resample as R
    from sparse_rtdetr.research import r28_warp as W
    torch = K.torch_mod()
    out = Path(out)
    K.require(out.resolve() == (K.ROUND / "followup-check").resolve(), "this action writes <round>/followup-check")
    K.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_available(), "the follow-up check hides CUDA")
    fu = K.followup(c)
    K.require(fu is not None and K.ARM == "K", "followup-check runs with R28_ARM=K under a follow-up contract")
    torch.set_num_threads(int(os.environ.get("R28_CPU_THREADS", "16")))
    res = {"stage": "followup-check", "registration": fu["registration"], "authorization_basis": c["authorization_basis"]}

    K.PHASE["phase"] = "followup-check:pytest"
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *[str(K.ROOT / t) for t in TESTS]],
                       capture_output=True, text=True, cwd=str(K.ROOT), env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    res["pytest"] = {"returncode": p.returncode, "tail": p.stdout.strip().splitlines()[-1:], "tests": list(TESTS)}
    K.require(p.returncode == 0, "synthetic pytest failed:\n" + p.stdout[-6000:])

    K.PHASE["phase"] = "followup-check:prior identities"
    c1, refs = prior_references(fu["prior_round"])
    K.require(all(fu[k] == v for k, v in refs.items()), "prior-round references changed since prepare")
    K.require(K.prior_contract(c)["contract_sha256"] == c1["contract_sha256"] != c["contract_sha256"], "prior contract digest")
    K.require(K.pinned(fu["task"]) == K.ROUND / TASK[0] and fu["task"]["sha256"] == TASK[1], "follow-up task file")
    manifest = K.read_json(K.pinned(c["source_manifest"]))
    delta = source_delta(K.read_json(K.pinned(fu["prior_manifest"]))["files"], manifest["files"])
    K.require(delta == fu["source_delta"], "source delta changed since prepare")
    same = [k for k in c1 if k not in ("contract_sha256", "version", "source_root", "source_manifest", "source_commit", "source_dirty",
                                       "workload", "labels", "resource_plan", "outputs", "formal_started")]
    K.require(all(c[k] == c1[k] for k in same) and c["identity_sha256"] == R.IDENTITY_SHA256 and c["definition"] == W.DEFINITION
              and (W.EPS, W.LO, W.HI, W.GRID) == (0.2, 2.0 / 3.0, 1.5, (12, 21)), "inherited contract fields / R28 definition")
    res["prior_identities"] = {"pinned": {k: PIN[k] for k in PIN}, "inherited_stages": list(fu["inherited"]),
                               "contract_fields_equal_to_prior": len(same), "source_delta": delta,
                               "identity_sha256_K": R.IDENTITY_SHA256["K"], "definition_equal": True}

    K.PHASE["phase"] = "followup-check:authorisation and isolation"
    prior = Path(fu["prior_round"])
    K.require(K.ROUND.resolve() != prior and prior not in K.ROUND.resolve().parents, "follow-up round is not independent")
    for name in fu["inherited"]:
        K.require(not (K.ROUND / name).exists(), f"inherited stage name present in the follow-up round: {name}")
    w_summary = K.stage_closed(c, "summary-W", "W")
    K.require(w_summary["K_trigger"] is False and w_summary["label"] == "closed_fixed_config", "inherited W summary")
    wrong = {**c, "authorization_basis": "none"}
    cases = [
        _refused("formal training of arm W in a follow-up round", lambda: LP.guard(c, K.ROUND / "formal-W", None, "W")),
        _refused("formal K before followup-check and accept-K are closed", lambda: LP.guard(c, K.ROUND / "formal-K", None, "K")),
        _refused("a formal W directory lookup in a follow-up round", lambda: K.formal_dir(c, "W")),
        _refused("an evaluation output inside the closed round", lambda: E.round_stage(prior / "eval-K", "eval-K")),
        _refused("a follow-up block without the registered authorization basis", lambda: K.followup(wrong)),
        _refused("the original summary rule (K_trigger must be true) on this W result",
                 lambda: K.require(E.decide_m(K.read_json(K.pinned(fu["prior_W_result"]))["M_W_minus_L0"]["cells"],
                                              K.read_json(K.pinned(fu["prior_W_result"]))["APL6"]["difference_pp"],
                                              K.read_json(K.pinned(fu["prior_W_result"]))["C"]["model_gpu_W_over_L0"])["K_trigger"],
                                   "arm K was not triggered")),
    ]
    K.require(os.environ.get("R28_DEV_FORCE_K") is None and os.environ.get("R28_DEV_IGNORE_G0") is None
              and os.environ.get("R28_DEV_IGNORE_COST") is None, "dev override flags are not used by a follow-up round")
    res["authorisation"] = {"round": str(K.ROUND), "prior_round": str(prior), "independent": True, "refusals": cases,
                            "W_summary_inherited": {k: w_summary[k] for k in ("label", "label_key", "K_trigger", "M_failed_cells")},
                            "dev_override_flags_absent": True}

    K.PHASE["phase"] = "followup-check:frozen prior"
    source = K.demand_source(c, "K")
    ctx = R.Context(source)
    g0 = K.stage_closed(c, "g0")
    K.require(source == {"kind": "fixed", "file": fu["prior_file"]} and ctx.binding()["state_sha256"] == fu["prior_state_sha256"]
              == g0["demand"]["prior"]["state_sha256"] and ctx.net is None and tuple(ctx.fixed.shape) == W.GRID,
              "arm K demand source is not the frozen prior of the closed round")
    side_path, side = K.read_side(K.prior_contract(c), fu["W_endpoint"]["path"], "formal", "W")
    K.require(side["file"] == fu["W_endpoint"] and side["global_update"] == K.ENDPOINT
              and side["demand"]["file"] == K.demand_source(c, "W")["file"] == fu["W_demand"], "prior W endpoint side-car")
    res["frozen_sources"] = {"K_demand": ctx.binding(), "prior_regenerated": False,
                             "W_endpoint_side": {"path": str(side_path), "global_update": side["global_update"], "arm": side["arm"]},
                             "W_demand": fu["W_demand"]}

    K.PHASE["phase"] = "followup-check:summary keeps the W decision"
    ev_w = K.stage_closed(c, "eval-W", "W")
    w_result = K.read_json(K.pinned(fu["prior_W_result"]))
    fake_k = {**ev_w, "arm": "K"}
    fake_cost = {"ratios": {"model_gpu": {"W/K": 1.0}, "end_to_end": {"W/K": 1.0}}}
    trial = E.summary_k(c, w_result, ev_w, fake_k, fake_cost)
    K.require(trial["M_W_minus_L0_decision_unchanged"] == W_DECISION and trial["A_W_minus_K"]["difference_pp"] == 0.0
              and trial["A_W_minus_K"]["label"] == "adaptive_increment_not_established"
              and c["labels"]["A_passed"] == "adaptive_increment_reached_in_seed0", "follow-up summary arithmetic / labels")
    res["summary_interface"] = {"W_decision_kept": trial["M_W_minus_L0_decision_unchanged"],
                                "trial": "summary_k with the W evaluation in the K slot (difference 0, label not_established)",
                                "labels": {k: c["labels"][k] for k in ("A_passed", "A_failed")}}

    K.PHASE["phase"] = "followup-check:per-class sizes of the stored W and L0 lists"
    rows, parsed = _development(c)
    res["per_class"] = {
        "W": stored_list_sizes(c, rows, parsed, ev_w["predictions"]["600"], ev_w["metrics"]["600"], ev_w["per_class_APL_top600"]),
        "L0": stored_list_sizes(c, rows, parsed, c["L0_predictions"]["600"], c["L0_metrics"]["600"],
                                c["reference"]["per_class_APL_top600"]["L0"])}
    counts.update(stored_list_COCOeval_calls=2)
    K.require(counts["forwards"] == 0 and counts["actual_updates"] == 0 and not counts.get("score_budget_calls"),
              "the follow-up check must not run a model or score_budget")
    res["scope"] = ("affected interfaces only; the general CPU / identity / box-mapping / evaluation-path evidence is the closed "
                    "round's cpu, zero and evalpath stages (inherited, pinned)")
    return res


# ================================================================ medium description (CPU, no model)
def apm_decomposition(tables, aggregate):
    """tables: arm -> ten per-class rows with APM; aggregate: arm -> APM600.  Class contributions of each pair in pp
    (difference / 10); their sum must be the aggregate difference."""
    import math
    for a in ARMS3:
        K.require([r["category_id"] for r in tables[a]] == list(range(1, 11)) and all(type(r["APM"]) is float for r in tables[a])
                  and [r["GT"] for r in tables[a]] == [r["GT"] for r in tables["L0"]], f"per-class APM table of {a}")
    rows = []
    for i in range(10):
        row = {"category_id": i + 1, "medium_GT": tables["L0"][i]["GT"], **{a: tables[a][i]["APM"] for a in ARMS3}}
        for x, y in PAIRS:
            d = 100. * (tables[x][i]["APM"] - tables[y][i]["APM"])
            row[f"{x}-{y}"] = {"difference_pp": d, "contribution_pp": d / 10.}
        rows.append(row)
    totals = {}
    for x, y in PAIRS:
        s = math.fsum(r[f"{x}-{y}"]["contribution_pp"] for r in rows)
        agg = 100. * (aggregate[x] - aggregate[y])
        K.require(abs(s - agg) < 1e-9, f"class contributions of {x}-{y} do not sum to the aggregate APM difference")
        totals[f"{x}-{y}"] = {"APM600_difference_pp": agg, "sum_of_contributions_pp": s,
                              "negative_contribution_classes": [r["category_id"] for r in rows if r[f"{x}-{y}"]["contribution_pp"] < 0]}
    return {"classes": rows, "totals": totals, "medium_GT_total": sum(r["medium_GT"] for r in rows)}


def size_rows(per_gt, size):
    """Certified GT rows of a size class on the raw area, COCOeval ranges with inclusive end points."""
    lo, hi = {"small": (0.0, K.SMALL_AREA), "medium": (K.SMALL_AREA, K.LARGE_AREA)}[size]
    return [r for r in per_gt if lo <= r["area"] <= hi]


def best_ious(rows, indexes):
    import numpy as np
    import r28_dose as D
    out = {a: np.empty(len(rows)) for a in indexes}
    for n, r in enumerate(rows):
        box = np.array(r["bbox_xyxy"], dtype=np.float64)
        for a, index in indexes.items():
            out[a][n] = D._best_iou(box, index.get(r["image"], {}).get(r["category_id"]))
    return out


def coverage_table(rows, iou, key):
    """Same-class candidate coverage of the same GT in the three lists, grouped by the stored magnification `key`.
    Coverage = some candidate of the image with the GT's class has IoU >= t (no one-to-one assignment; not TP, not AP)."""
    import numpy as np
    import r28_loop as LP
    bins = np.array([LP._bin(r[key]) for r in rows], dtype=np.int64)
    out = []
    for b, label in enumerate(LP.BIN_NAMES + ("all",)):
        sel = np.ones(len(rows), dtype=bool) if label == "all" else bins == b
        row = {"bin": label, "n": int(sel.sum())}
        for t in THRESHOLDS:
            hit = {a: iou[a][sel] >= t for a in ARMS3}
            cell = {"covered": {a: int(hit[a].sum()) for a in ARMS3},
                    "coverage": {a: (float(hit[a].mean()) if sel.any() else None) for a in ARMS3}}
            for x, y in PAIRS:
                cell[f"{x}-{y}"] = {"difference_pp": (100. * (float(hit[x].mean()) - float(hit[y].mean())) if sel.any() else None),
                                    "gained": int((hit[x] & ~hit[y]).sum()), "lost": int((~hit[x] & hit[y]).sum())}
            row[f"coverage@{t}"] = cell
        out.append(row)
    return out


def medium(c, out, counts):
    import r28_dose as D
    import r28_loop as LP
    out = Path(out)
    K.require(out.resolve() == (K.ROUND / "medium").resolve(), "this action writes <round>/medium")
    fu = K.followup(c)
    K.require(fu is not None and K.ARM == "K", "the medium description belongs to a follow-up round (R28_ARM=K)")
    K.PHASE["phase"] = "medium:per-class APM"
    chk, ev_k, ev_w = K.stage_closed(c, "followup-check"), K.stage_closed(c, "eval-K", "K"), K.stage_closed(c, "eval-W", "W")
    tables = {"L0": chk["per_class"]["L0"]["per_class_APM_top600"], "W": chk["per_class"]["W"]["per_class_APM_top600"],
              "K": ev_k["per_class_APM_top600"]}
    aggregate = {"L0": c["reference"]["values"]["L0"]["600"]["APM"], "W": ev_w["values"]["600"]["APM"], "K": ev_k["values"]["600"]["APM"]}
    apm = apm_decomposition(tables, aggregate)
    K.PHASE["phase"] = "medium:coverage"
    per_gt = K.read_json(K.pinned(fu["per_gt"]))
    lists = {"L0": c["L0_predictions"]["600"], "W": ev_w["predictions"]["600"], "K": ev_k["predictions"]["600"]}
    indexes = {}
    for a in ARMS3:
        indexes[a], n = D._by_image(K.pinned(lists[a]))
        K.require(n == 548 * 600, f"top600 record count of {a}")
    tables_cov, sets = {}, {}
    for size in ("medium", "small"):
        rows = size_rows(per_gt, size)
        iou = best_ious(rows, indexes)
        sets[size] = len(rows)
        tables_cov[size] = {"grouped_by_W_predicted_g": coverage_table(rows, iou, "predicted"),
                            "grouped_by_K_prior_g": coverage_table(rows, iou, "prior")}
    g0 = K.stage_closed(c, "g0")
    K.require(sets["medium"] == g0["development"]["predicted"]["medium"]["n"] == apm["medium_GT_total"]
              and sets["small"] == g0["development"]["predicted"]["small"]["n"], f"size-class GT counts {sets}")
    K.require(counts["forwards"] == 0 and counts["actual_updates"] == 0 and not counts.get("score_budget_calls"),
              "the medium description must not run a model or a scorer")
    return {"stage": "medium", "per_class_APM600": apm, "coverage": tables_cov, "GT": sets,
            "set": {"definition": g0["set"]["definition"], "medium_area_range": [K.SMALL_AREA, K.LARGE_AREA],
                    "small_area_range": [0.0, K.SMALL_AREA], "ignore": "raw-ignore raster filter of the certified evaluation; "
                    "crowd excluded (none present)", "boundary": "inclusive end points as in COCOeval: a GT with raw area exactly "
                    "1024 belongs to both small and medium"},
            "grouping": {"main": "W's stored per-GT magnification of the frozen demand network (g0/per_gt.json `predicted`), the same "
                                 "GT groups for all three lists", "secondary": "K's own per-GT magnification of the frozen prior "
                                 "(g0/per_gt.json `prior`, pure geometry already stored; the demand network was not rerun)",
                         "bins": list(LP.BIN_NAMES)},
            "coverage_rule": "same class, best IoU over the image's 600 candidates >= threshold; no one-to-one assignment; "
                             "gained / lost count GT covered by the first list only / by the second list only",
            "per_class_sources": {"W": "followup-check (one COCOeval of the stored W top600 list)",
                                  "L0": "followup-check (one COCOeval of the stored L0 top600 list)",
                                  "K": "eval-K (the evaluation's single per-class COCOeval)"},
            "inputs": {"per_gt": fu["per_gt"], "predictions": lists},
            "new_model_forwards": 0, "new_COCOeval_calls_in_this_stage": 0, "reading": STATEMENTS["medium"]}


# ================================================================ summary-K of a follow-up round
def summarize(c, out, counts):
    import r28_eval as E
    fu = K.followup(c)
    K.require(fu is not None and K.ARM == "K", "a follow-up round only summarises arm K")
    K.PHASE["phase"] = "summarize/read"
    out = E.round_stage(out, "summary-K")
    w_closed = K.stage_closed(c, "summary-W", "W")
    K.require(w_closed["result"] == fu["prior_W_result"], "inherited summary-W result reference")
    w_result = K.read_json(K.pinned(fu["prior_W_result"]))
    K.require(w_result["decision"] == fu["prior_W_decision"] == W_DECISION and w_result["contract_sha256"] == fu["prior_contract_sha256"],
              "the W decision of the closed round changed")
    ev_w, ev_k, cost_k = K.stage_closed(c, "eval-W", "W"), K.stage_closed(c, "eval-K", "K"), K.stage_closed(c, "cost-K")
    K.require(cost_k["weights_sha256"]["K"] == ev_k["weights_sha256"] and cost_k["weights_sha256"]["W"] == ev_w["weights_sha256"]
              == fu["W_weights_sha256"] and cost_k["weights_sha256"]["L0"] == c["L0_weights_sha256"],
              "the cost-K stage timed other models than the evaluated ones")
    K.require(ev_k["demand"]["file"] == fu["prior_file"] and ev_k["demand"]["state_sha256"] == fu["prior_state_sha256"],
              "arm K was not evaluated with the frozen prior")
    for b in K.BUDGETS:
        K.pinned(ev_k["metrics"][str(b)]), K.pinned(ev_k["predictions"][str(b)])
    med = K.stage_closed(c, "medium")
    K.PHASE["phase"] = "summarize/compute"
    res = E.summary_k(c, w_result, ev_w, ev_k, cost_k)
    a = res["A_W_minus_K"]
    formal = K.formal_dir(c, "K").name
    res.update(status="PASS", registration=fu["registration"], authorization_basis=c["authorization_basis"], task=fu["task"],
               contract_sha256=c["contract_sha256"], source_commit=c["source_commit"],
               W_closed_round={"contract_sha256": fu["prior_contract_sha256"], "result": fu["prior_W_result"],
                               "decision": w_result["decision"], "M_W_minus_L0": w_result["M_W_minus_L0"], "APL6": w_result["APL6"],
                               "C": w_result["C"], "note": "quoted unchanged from the closed round; not recomputed, not overwritten"},
               K_minus_L0_cells_note=STATEMENTS["K_minus_L0"],
               K_minus_L0_failed_cells=[k for k, v in res["K_minus_L0_cells"].items() if not v["passed"]],
               cost_K_timing={"medians": cost_k["medians"], "ratios": cost_k["ratios"], "ratios_per_pass": cost_k["ratios_per_pass"],
                              "order_executed": cost_k["order_executed"], "protocol": cost_k["protocol"],
                              "note": STATEMENTS["cost"]},
               per_class_APM_top600=med["per_class_APM600"], medium_description=K.ref(K.ROUND / "medium" / "CLOSED.json"),
               K_demand=ev_k["demand"], K_endpoint=ev_k["endpoint"], K_weights_sha256=ev_k["weights_sha256"],
               stages=E._stage_account(c, ["followup-check", "accept-K", formal, "eval-K", "cost-K", "medium"]),
               other_directories=E._other_directories(), statements=STATEMENTS)
    K.require(counts["forwards"] == 0 and counts["actual_updates"] == 0 and not counts.get("score_budget_calls"),
              "summary must not run a model or a scorer")
    K.write_new(out / "RESULT.json", res)
    return {"result": K.ref(out / "RESULT.json"), "registration": fu["registration"], "label": a["label"], "A_passed": a["passed"],
            "A_difference_pp": a["difference_pp"], "W_decision_unchanged": w_result["decision"],
            "K_minus_L0_failed_cells": res["K_minus_L0_failed_cells"]}
