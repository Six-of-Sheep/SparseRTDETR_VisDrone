"""R30-001 (TASK_R30_DIAG_001): residual-error diagnostic of the F_geo endpoint.  No training, RECORD_ONLY, no gate.

usage: r30_diag.py <prepare|check3|infer|score|tables> --round <round> [--release <name>]

The round is an independent directory; the closed rounds it reads (F_geo 009, L0, S15 seed0, W) are pinned by file SHA
in <round>/CONTRACT.json and are never written.  Every stage writes <round>/<stage>/CLOSED.json (or STOP.json).

  prepare  contract of the round: the release (git archive of a commit; only this file and its test may differ from the
           R29 release the branch starts from), the pinned inputs.
  check3   CPU.  Task check 3: the recall rule (r28_dose.table, unchanged) is recomputed on arm W and must equal the
           stored dose-W table cell for cell before it is used on F_geo.
  infer    CUDA.  Task check 1 (the first 8 images through this entry under the predicted demand must equal the stored
           009 top600 / top300 records field by field), the per-GT geometry of the F_geo map under the predicted demand
           and under the label demand (demand network and map only), then three inferences of 4 warm-up + 137 batches:
             A2    F_geo e120 EMA, demand = label demand of each development image (oracle_demand_diagnostic)
             A3a   F_geo e120 EMA weights, uniform input (the L0 forward: to_low inside the model)
             A3b   L0 e120 EMA weights, F_geo forward (predicted demand, boxes mapped back by the F_geo inverse map)
  score    CPU.  Six score_budget calls (three endpoints x top600 / top300), each asserting the certified GT counts,
           and the one per-class COCOeval (A2, top600) with its tensors kept.
  tables   CPU.  A1 / A2 / A3 tables from the stored and the new predictions; no model, no scorer.

A dev round (a directory dev-round* inside the round) runs every inference on 1 warm-up + 3 batches; its numbers are
wiring evidence only.
"""
import argparse
import gc
import json
import os
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VERSION = "R30-001-fgeo-diagnostic-v1"
LABEL = "oracle_demand_diagnostic"
TASK = ("TASK_R30_DIAG_001.md", "a121ca4dc004ab8e5378cd0bd528d6b30ff040e97c0135dbe117a715d230878b")
BASE_COMMIT = "6c656518da9c0ea97121bc15d0a147c8bdef3fef"          # R29 release (round r29-query-unzoom-20261117-002)
ALLOWED_ADDED = ("tools/r28/r30_diag.py", "tests/test_r30_diag.py")
PIN = {
    "FG_contract_file": "dc39a82b9aab1c0a4a88e19a72a8d2aa73da71e244c6d7e2e959cc22f30e5d1a",
    "FG_contract": "0ae1eca4b3e7231bfad70ab6e7aebb93622cec2143975e41bae273869ed85553",
    "FG_endpoint": "f0632fc8ed71e593e5eaab1bf233aafed1e9ec57dcfe44c893721286cdffeb57",
    "FG_weights": "cd62f314f3713d83fd742ff511600902fd990cf48165d9b0a90c7fec4359e219",
    "L0_endpoint": "966dac4afe28bdd154cd1b8d5054719a541c3ad02d57258e9636d2d5454c1701",
    "L0_predictions_600": "25fcf2fd2ca30e866c623a4aa26c7dcca3207a9e16e9583ea28905ac91b3c760",
    "S15_endpoint": "656f7be2357f20dce50a718f1093a639c1a32a7faad69d03184e36e43f32871f",
    "demand_net": "c9ffc7505e47adbcc45d70949c3d87371c6f3a7795f1aaaab886d98b15ee9e5a",
    "evaluation_policy": "860285aa5af367b2bf2fef2221c15ac3914b33c57eb765e09f76b9b580004a82",
}
GT_COUNTS = {"GT_before": 38759, "GT_after": 38599}
SMALL_GT = 26441
# REPORT_R28_001 section 6.1 (the cells named by the task): bin -> (n, L0 recall@0.5 %, W recall@0.5 %)
REPORT_W = {"g<1.0": (3039, 79.53, 72.36), "all": (26441, 82.73, 83.23)}
BIN_NAMES = ("g<1.0", "1.0<=g<1.2", "1.2<=g<1.4", "g>=1.4")
THRESHOLDS = (0.5, 0.75)
SMALL_AREA, SPLIT_AREA, LARGE_AREA = 32.0 ** 2, 16.0 ** 2, 96.0 ** 2
AREA_GROUPS = ("small", "area<16^2", "16^2<=area<=32^2")
METRICS = ("AP", "APS", "AP75_small", "APM", "APL", "AP50", "AP75")
ENDPOINTS = ("A2", "A3a", "A3b")
ENDPOINT_TEXT = {
    "A2": "F_geo e120 EMA weights; demand = label demand of the image; F_geo map and inverse map",
    "A3a": "F_geo e120 EMA weights; uniform input (L0 forward, to_low inside the model); no map",
    "A3b": "L0 e120 EMA weights; F_geo forward with the predicted demand; boxes mapped back by the F_geo inverse map",
}
DEV_LIMIT = (1, 3)
STATEMENTS = {
    "record_only": "RECORD_ONLY：不设通过门，不触发任何自动后续；任何结果都不自动启动训练。",
    "oracle": "A2 在推理时使用了 development 标注（oracle_demand_diagnostic），不是方法结果；检测器是在预测需求下训练的，"
              "所以它既不是可达上限，也不是下限。",
    "single_seed": "单 seed（seed0）、重复使用的 548 图 development 集上的读数。",
    "recall": "召回＝同图同类 top600 候选中最佳 IoU ≥ 阈值，无一对一匹配、无分数阈值；是描述量，不是 AP。",
    "a3": "A3 的两格是权重与输入不匹配的读数（权重没有在该输入下训练），只作 2×2 交叉读出。",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


# ================================================================ pure helpers (no torch, no project import)
def bin_index(g):
    return 0 if g < 1.0 else 1 if g < 1.2 else 2 if g < 1.4 else 3


def area_masks(area):
    """Masks of the small GT rows by raw area: every small row, area < 16^2, 16^2 <= area <= 32^2."""
    import numpy as np
    area = np.asarray(area, dtype=np.float64)
    require(bool((area <= SMALL_AREA).all()), "area_masks takes small GT only")
    return {"small": np.ones(len(area), dtype=bool), "area<16^2": area < SPLIT_AREA, "16^2<=area<=32^2": area >= SPLIT_AREA}


def bin_table(g, area, iou, pairs=()):
    """g, area: (n,) of the small GT; iou: {endpoint: (n,) best IoU}.  One row per (bin + all) x area group with the
    recall of every endpoint at both thresholds and the differences (pp) of the named pairs."""
    import numpy as np
    g = np.asarray(g, dtype=np.float64)
    bins = np.array([bin_index(v) for v in g], dtype=np.int64)
    masks = area_masks(area)
    rows = []
    for b, label in enumerate(BIN_NAMES + ("all",)):
        in_bin = np.ones(len(g), dtype=bool) if label == "all" else bins == b
        for group in AREA_GROUPS:
            sel = in_bin & masks[group]
            row = {"bin": label, "area_group": group, "n": int(sel.sum())}
            for t in THRESHOLDS:
                r = {k: (float((v[sel] >= t).mean()) if sel.any() else None) for k, v in iou.items()}
                row[f"recall@{t}"] = r
                row[f"difference_pp@{t}"] = {f"{a}-{c}": (None if r[a] is None else 100. * (r[a] - r[c])) for a, c in pairs}
            rows.append(row)
    return rows


def miss_table(g, area, hit_other, hit_fg):
    """Counts of small GT the other endpoint recalls and F_geo does not, per bin and area group (and the reverse)."""
    import numpy as np
    g = np.asarray(g, dtype=np.float64)
    bins = np.array([bin_index(v) for v in g], dtype=np.int64)
    masks = area_masks(area)
    out = {}
    for t in THRESHOLDS:
        a, f = np.asarray(hit_other[t], dtype=bool), np.asarray(hit_fg[t], dtype=bool)
        lost, won = a & ~f, f & ~a
        rows = []
        for group in AREA_GROUPS:
            m = masks[group]
            total = int((lost & m).sum())
            for b, label in enumerate(BIN_NAMES + ("all",)):
                sel = m if label == "all" else m & (bins == b)
                n, k = int(sel.sum()), int((lost & sel).sum())
                rows.append({"bin": label, "area_group": group, "n": n, "other_hit_FG_miss": k,
                             "share_of_all_such_GT": None if not total else k / total,
                             "share_of_bin": None if not n else k / n,
                             "FG_hit_other_miss": int((won & sel).sum()),
                             "bin_share_of_GT": None if not m.sum() else n / int(m.sum())})
        out[f"IoU>={t}"] = rows
    return out


def geometry_reading(g, area):
    """Per size class (COCO ranges on the raw area, inclusive end points as in COCOeval): pooled magnification."""
    import numpy as np
    g, area = np.asarray(g, dtype=np.float64), np.asarray(area, dtype=np.float64)
    classes = {"small": area <= SMALL_AREA, "medium": (area >= SMALL_AREA) & (area <= LARGE_AREA), "large": area >= LARGE_AREA,
               "small_area<16^2": area < SPLIT_AREA, "small_16^2<=area<=32^2": (area >= SPLIT_AREA) & (area <= SMALL_AREA)}
    out = {}
    for name, sel in classes.items():
        v = g[sel]
        out[name] = {"n": int(sel.sum())}
        if len(v):
            out[name].update(g_geomean=float(np.exp(np.log(v).mean())), g_mean=float(v.mean()),
                             area_ratio_mean=float((v * v).mean()),
                             g_q10_50_90=[float(q) for q in np.quantile(v, [0.1, 0.5, 0.9])],
                             frac_g_lt_1=float((v < 1.0).mean()), frac_g_ge_1p2=float((v >= 1.2).mean()),
                             frac_g_ge_1p4=float((v >= 1.4).mean()),
                             bin_counts=[int(((np.array([bin_index(x) for x in v])) == b).sum()) for b in range(4)])
    return out


def cross_bins(g_a, g_b):
    """4 x 4 counts: rows = bin under g_a, columns = bin under g_b."""
    out = [[0] * 4 for _ in range(4)]
    for a, b in zip(g_a, g_b):
        out[bin_index(a)][bin_index(b)] += 1
    return out


def records_equal(mine, stored):
    """Field-for-field comparison of prediction records; returns the first difference or None."""
    if len(mine) != len(stored):
        return {"index": None, "lengths": [len(mine), len(stored)]}
    for i, (a, b) in enumerate(zip(mine, stored)):
        if a != b:
            return {"index": i, "mine": a, "stored": b}
    return None


def difference_pp(x, y):
    return {b: {m: 100. * (x[b][m] - y[b][m]) for m in METRICS} for b in ("600", "300")}


def batch_order(ids, warm, scored, batch=4):
    """Image ids of every model call of r28_eval.infer in call order: `warm` warm-up batches, then `scored` batches,
    both starting at the first image."""
    rows = [ids[i * batch:(i + 1) * batch] for i in range(warm)]
    rows += [ids[i * batch:(i + 1) * batch] for i in range(scored)]
    require(all(len(r) == batch for r in rows), "the image list does not fill the batches")
    return rows


def make_oracle_context(base, maps, order, fwhm):
    """A demand context (subclass of the given r28_resample.Context class) whose demand(x) returns the stored
    per-image maps of the next batch of `order`.  Nothing of it enters a detector state."""
    import torch

    class OracleDemandContext(base):
        def __init__(self):                                   # the network / file source of the base class is not built
            self.kind, self.source, self.net, self.fixed = LABEL, {"kind": LABEL}, None, None
            self.observe, self.records = False, []
            self.device = torch.device("cpu")
            self.maps, self.order, self.calls = maps, [list(o) for o in order], 0

        def to(self, device):
            self.device = torch.device(device)
            return self

        def binding(self):
            return {"kind": LABEL, "file": None, "seed": None, "images": len(self.maps)}

        def demand(self, x):
            require(self.calls < len(self.order), "oracle demand context called more often than the planned batches")
            ids = self.order[self.calls]
            self.calls += 1
            require(x.shape[0] == len(ids), "oracle demand batch size differs from the planned batch")
            return torch.stack([self.maps[i] for i in ids]).to(torch.float64).cpu()

    ctx = OracleDemandContext()
    vars(ctx)["r28_fwhm"] = fwhm
    return ctx


# ================================================================ project layer
_M = {}


def mods(round_dir=None):
    """Project modules, imported once; r28_common binds its round from the environment at import time."""
    if not _M:
        require(round_dir is not None, "the round must be known before the project modules are imported")
        os.environ["R28_ROUND"] = str(Path(round_dir).resolve())
        sys.path.insert(0, str(HERE))
        import r28_common as K
        import r28_dose as DS
        import l0_common as B
        require(K.ROOT.resolve() == ROOT.resolve() and K.ROUND.resolve() == Path(round_dir).resolve(), "module roots")
        require(K.SMALL_AREA == SMALL_AREA and K.LARGE_AREA == LARGE_AREA and tuple(K.BUDGETS) == (600, 300)
                and (K.EVAL_BATCH, K.EVAL_WARMUP, K.EVAL_BATCHES) == (4, 4, 137), "evaluation constants")
        import r28_loop as LP
        require(tuple(LP.BIN_NAMES) == BIN_NAMES and all(LP._bin(v) == bin_index(v) for v in (0.5, 1.0, 1.1, 1.2, 1.3, 1.4, 2.0))
                and tuple(DS.THRESHOLDS) == THRESHOLDS, "bins / thresholds differ from the R28 dose table")
        _M.update(K=K, DS=DS, B=B)
    return _M["K"], _M["DS"], _M["B"]


def is_dev(round_dir):
    return Path(round_dir).resolve().name.startswith("dev-round")


def fg_contract(K, ref):
    c = K.read_json(K.pinned(ref))
    body = {k: v for k, v in c.items() if k != "contract_sha256"}
    require(ref["sha256"] == PIN["FG_contract_file"] and c["contract_sha256"] == K.sha_bytes(K.canonical(body)) == PIN["FG_contract"],
            "F_geo (009) contract identity")
    require(K.followup_arm(c) == "FG", "the pinned contract is not the F_geo round's")
    return c


def prepare(round_dir, release):
    K, DS, B = mods(round_dir)
    R = Path(round_dir).resolve()
    dev = is_dev(R)
    require(not (R / "CONTRACT.json").exists(), "CONTRACT.json exists")
    require((R / release / "source").resolve() == ROOT.resolve(), "prepare must run from <round>/<release>/source")
    task = (R.parent if dev else R) / TASK[0]
    require(task.is_file() and K.file_sha(task) == TASK[1], f"task file missing or different: {task}")
    man_ref = K.ref(R / release / "MANIFEST.json")
    manifest = K.read_json(man_ref["path"])
    require(manifest["dirty"] is False or dev, "a formal release must come from a clean committed tree")
    B.release_intact(ROOT, manifest["files"])
    ld = K.LD_ROOT
    base_ref = K.ref(ld / "r29-query-unzoom-20261117-002/release-001/MANIFEST.json")
    base = K.read_json(base_ref["path"])
    require(base["commit"] == BASE_COMMIT, "base release commit")
    new, old = manifest["files"], base["files"]
    delta = {"added": sorted(set(new) - set(old)), "removed": sorted(set(old) - set(new)),
             "changed": sorted(k for k in new if k in old and new[k] != old[k])}
    require(set(delta["added"]) == set(ALLOWED_ADDED) and not delta["removed"] and not delta["changed"],
            f"the release may differ from the R29 release only by the two added files: {delta}")
    fg_round = ld / "r28-strong-fovea-control-20261116-001"
    fgc_ref = K.ref(fg_round / "CONTRACT.json")
    cf = fg_contract(K, fgc_ref)
    fu = cf["followup"]
    ev_ref = K.ref(fg_round / "eval-FG/CLOSED.json")
    ev = K.read_json(ev_ref["path"])
    require(ev["status"] == "PASS" and ev["arm"] == "FG" and ev["contract_sha256"] == PIN["FG_contract"]
            and ev["weights_sha256"] == PIN["FG_weights"] and ev["endpoint"]["sha256"] == PIN["FG_endpoint"]
            and ev["demand"]["file"]["sha256"] == PIN["demand_net"] and ev["GT_counts"]["600"] == GT_COUNTS, "eval-FG identity")
    s15_dir = ld / "s15-scale-baseline-20261105-001/formal-002/evaluation"
    s15_res_ref = K.ref(s15_dir / "S15-EMA/RESULT.json")
    s15_res = K.read_json(s15_res_ref["path"])
    require(s15_res["status"] == "PASS" and s15_res["arm"] == "S15" and s15_res["kind"] == "EMA" and s15_res["epoch"] == 120
            and s15_res["view"] == "A" and s15_res["images"] == 548 and s15_res["checkpoint"]["sha256"] == PIN["S15_endpoint"]
            and s15_res["checkpoint"] == cf["S15_endpoint"], "stored S15 seed0 inference identity")
    s15_pred = K.ref(s15_res["predictions"]["600"]["path"])
    require(s15_pred["sha256"] == s15_res["predictions"]["600"]["sha256"], "stored S15 top600 predictions")
    w_round = ld / "r28-budgeted-resampling-20261113-001"
    dose_ref = K.ref(w_round / "dose-W/CLOSED.json")
    dose = K.read_json(dose_ref["path"])
    require(dose["status"] == "PASS" and dose["arm"] == "W" and dose["inputs"]["per_gt"] == fu["per_gt"]
            and dose["inputs"]["arm_predictions"] == fu["W_predictions"]["600"]
            and dose["inputs"]["L0_predictions"] == cf["L0_predictions"]["600"], "stored dose-W inputs")
    require(cf["L0_endpoint"]["sha256"] == PIN["L0_endpoint"] and cf["L0_predictions"]["600"]["sha256"] == PIN["L0_predictions_600"]
            and fu["W_demand"]["sha256"] == PIN["demand_net"] and cf["evaluation_policy"]["sha256"] == PIN["evaluation_policy"]
            and cf["reference"]["GT_counts"]["600"] == GT_COUNTS, "F_geo contract references")
    inputs = {"FG_contract": fgc_ref, "FG_eval_closed": ev_ref, "FG_endpoint": ev["endpoint"], "FG_predictions": ev["predictions"],
              "FG_metrics": ev["metrics"], "FG_cocoeval_tensors": ev["cocoeval_top600_tensors"],
              "L0_endpoint": cf["L0_endpoint"], "L0_predictions": cf["L0_predictions"], "L0_metrics": cf["L0_metrics"],
              "S15_endpoint": cf["S15_endpoint"], "S15_inference_result": s15_res_ref, "S15_predictions_600": s15_pred,
              "S15_metrics": cf["S15_metrics"], "demand_net": fu["W_demand"], "evaluation_policy": cf["evaluation_policy"],
              "W_per_gt": fu["per_gt"], "W_predictions_600": fu["W_predictions"]["600"], "dose_W_closed": dose_ref,
              "base_manifest": base_ref}
    for r in (inputs["FG_endpoint"], inputs["L0_endpoint"], inputs["FG_predictions"]["600"], inputs["FG_predictions"]["300"],
              inputs["L0_predictions"]["600"], inputs["W_per_gt"], inputs["W_predictions_600"], inputs["demand_net"],
              inputs["FG_cocoeval_tensors"], inputs["FG_metrics"]["600"], inputs["evaluation_policy"]):
        K.pinned(r)
    c = {"version": VERSION, "task": {"file": TASK[0], "sha256": TASK[1]}, "label": LABEL, "record_only": True,
         "source_root": str(ROOT), "source_commit": manifest["commit"], "source_branch": manifest["branch"],
         "source_dirty": manifest["dirty"], "source_manifest": man_ref, "base_commit": BASE_COMMIT, "source_delta": delta,
         "outputs": str(R), "dev": dev, "inputs": inputs, "pins": PIN, "GT_counts": GT_COUNTS,
         "reference_values": {"L0": cf["reference"]["values"]["L0"], "S15": cf["reference"]["values"]["S15"], "FG": ev["values"]},
         "mapping": {"fwhm_hex": cf["followup"]["width_selection"]["fwhm_hex"], "grid_shape": cf["fovea_geo_definition"]["grid_shape"],
                     "padding_size": cf["fovea_geo_definition"]["padding_size"], "eps": cf["definition"]["eps"],
                     "note": "the 009 definition, read from the pinned F_geo contract; nothing is changed"},
         "endpoints": ENDPOINT_TEXT,
         "gt_demand": "r28_warp.demand_from_boxes(r28_demand.deployment_xyxy(parsed, width, height)[0]): the label demand "
                      "of the g0 'gt_demand' reading (REPORT_R28_001 section 5), FP64, without eps",
         "workload": {"detector_forwards": {"check1": 2, "A2": 141, "A3a": 141, "A3b": 141}, "score_budget_calls": 6,
                      "per_class_COCOeval_calls": 1, "S15_reinference": 0, "updates": 0},
         "statements": STATEMENTS, "python": sys.executable}
    c["contract_sha256"] = K.sha_bytes(K.canonical(c))
    K.write_new(R / "CONTRACT.json", c)
    print(json.dumps({"contract_sha256": c["contract_sha256"], "contract_file": K.ref(R / "CONTRACT.json"),
                      "commit": c["source_commit"], "delta": delta}))


def load_contract(round_dir):
    K, DS, B = mods(round_dir)
    R = Path(round_dir).resolve()
    c = K.read_json(R / "CONTRACT.json")
    body = {k: v for k, v in c.items() if k != "contract_sha256"}
    require(c["contract_sha256"] == K.sha_bytes(K.canonical(body)) and c["version"] == VERSION, "contract digest")
    require(Path(c["source_root"]).resolve() == ROOT.resolve(), "not running from the bound release")
    require(Path(c["outputs"]).resolve() == R and c["dev"] == is_dev(R), "contract bound to another round")
    manifest = K.read_json(K.pinned(c["source_manifest"]))
    require(manifest["commit"] == c["source_commit"], "MANIFEST commit differs from the contract")
    B.release_intact(ROOT, manifest["files"])
    require(c["pins"] == PIN and c["GT_counts"] == GT_COUNTS, "pins differ from the release")
    return c


def stage_closed(K, c, name):
    path = Path(c["outputs"]) / name / "CLOSED.json"
    require(path.is_file(), f"stage not closed: {path}")
    closed = K.read_json(path)
    require(closed.get("status") == "PASS" and closed.get("contract_sha256") == c["contract_sha256"], f"stage {name} is not PASS")
    return closed


def gpu_compute_processes():
    p = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                       capture_output=True, text=True)
    return [line.strip() for line in p.stdout.splitlines() if line.strip()]


# ================================================================ check3
def check3(c, out, counts):
    K, DS, B = mods()
    i = c["inputs"]
    stored = K.read_json(K.pinned(i["dose_W_closed"]))
    per_gt = K.read_json(K.pinned(i["W_per_gt"]))
    mine, n_mine = DS._by_image(K.pinned(i["W_predictions_600"]))
    theirs, n_theirs = DS._by_image(K.pinned(i["L0_predictions"]["600"]))
    require(n_mine == n_theirs == 548 * 600, "top600 prediction record counts")
    res = DS.table(per_gt, mine, theirs)
    require(res["small_GT"] == stored["small_GT"] == SMALL_GT, "small GT count")
    cells = []
    for a, b in zip(res["bins"], stored["bins"]):
        cells.append({"bin": a["bin"], "n": a["n"], "equal_to_stored_dose_W": a == b,
                      "recall@0.5": a["recall@0.5"], "recall@0.75": a["recall@0.75"]})
    require(len(res["bins"]) == len(stored["bins"]) == 5 and all(x["equal_to_stored_dose_W"] for x in cells),
            f"the recomputed W table differs from the stored dose-W table: {cells}")
    by = {r["bin"]: r for r in res["bins"]}
    report = {}
    for label, (n, l0, w) in REPORT_W.items():
        r = by[label]
        got = (r["n"], round(100. * r["recall@0.5"]["L0"], 2), round(100. * r["recall@0.5"]["arm"], 2))
        report[label] = {"REPORT_R28_001_6.1": [n, l0, w], "recomputed": list(got), "equal": got == (n, l0, w)}
    require(all(v["equal"] for v in report.values()), f"the recomputed W cells differ from REPORT_R28_001 section 6.1: {report}")
    return {"stage": "check3", "passed": True, "rule": "r28_dose.table (unchanged file of the release)", "cells": cells,
            "report_cells": report, "all_five_rows_equal_stored_dose_W": True}


# ================================================================ infer
def _development(c, cf):
    from sparse_rtdetr.research import scale_baseline as S
    K, DS, B = mods()
    data = S.development(cf["data"]["development"], S.S15_HW)
    rows = data.records
    require([r["stable_image_id"] for r in rows] == cf["development_ids"] and len(rows) == 548, "development members/order")
    return data, rows, [data.raw(row) for row in rows]


def _models(c, cf):
    """(FG side, FG EMA state, FG demand source, L0 side, L0 EMA state), every file read under its own contract."""
    K, DS, B = mods()
    from sparse_rtdetr.protocol_r18.engine import tensor_digest
    i = c["inputs"]
    side, s = K.read_state(cf, K.pinned(i["FG_endpoint"]), "formal", "FG")
    require(side["file"] == i["FG_endpoint"] and side["global_update"] == K.ENDPOINT
            and s["engine"]["completed_updates"] == s["ema"]["updates"] == K.ENDPOINT, "stored FG endpoint clocks")
    fg_state = s["ema"]["module"]
    require(tensor_digest(fg_state.items()) == PIN["FG_weights"], "F_geo e120 EMA digest")
    source = K.demand_source(cf, "FG")
    require(source["file"] == i["demand_net"] and side["demand"]["file"] == source["file"], "demand network file")
    c0 = K.l0_contract(cf)
    side0, l0_state = B.endpoint_state(c0)
    require(side0["file"] == i["L0_endpoint"] and tensor_digest(l0_state.items()) == cf["L0_weights_sha256"], "L0 e120 EMA")
    return side, fg_state, source, side0, l0_state


def _geometry(c, cf, ctx, data, rows, parsed, counts, n_images):
    """Per-GT g of the certified development GT under the F_geo map: predicted demand (the frozen network, batches of
    4 as in the evaluation) and label demand.  Demand network and map only; no detector forward."""
    import numpy as np
    K, DS, B = mods()
    torch = K.torch_mod()
    import r28_demand as DM
    from infer_seed import load_original, pixels
    from sparse_rtdetr.r0.evaluation import bridge_inputs
    from sparse_rtdetr.research import r28_fovea as FV
    from sparse_rtdetr.research import r28_fovea_geo as FG
    from sparse_rtdetr.research import r28_warp as W
    from sparse_rtdetr.research import scale_baseline as S
    DM._torch()
    dataset, _, gt_counts = bridge_inputs(rows, parsed, [])
    require({k: gt_counts[k] for k in GT_COUNTS} == GT_COUNTS, f"certified GT set differs: {gt_counts}")
    by_image = {}
    for a in dataset["annotations"]:
        by_image.setdefault(a["image_id"], []).append(a)

    def magnify(plan, boxes):
        if not len(boxes):
            return torch.zeros(0, dtype=torch.float64)
        ox, oy = FV.magnification(plan, boxes[None])
        return torch.sqrt(ox * oy)[0]

    per_gt, per_image, d_gt_all = [], [], {}
    dev = ctx.device
    with torch.inference_mode():
        for off in range(0, len(rows), K.EVAL_BATCH):
            batch = rows[off:off + K.EVAL_BATCH]
            d_gt = [W.demand_from_boxes(DM.deployment_xyxy(parsed[off + j], row["width"], row["height"])[0]) for j, row in enumerate(batch)]
            for row, d in zip(batch, d_gt):
                require(tuple(d.shape) == tuple(W.GRID) and d.dtype == torch.float64 and bool(torch.isfinite(d).all())
                        and bool((d >= 0).all()), "label demand map")
                d_gt_all[row["stable_image_id"]] = d
            if off >= n_images:
                continue
            imgs = []
            for row in batch:
                with load_original(data, row) as image:
                    imgs.append(pixels(data, image))
            x = torch.stack(imgs).to(dev)
            require(tuple(x.shape[-2:]) == tuple(S.S15_HW) and x.dtype == torch.float32, "geometry input is not the FP32 H canvas")
            with torch.autocast(dev.type, enabled=False):
                d_pred = ctx.demand(x)
            counts.update(demand_only_forwards=1)
            plans_pred = FG.plan_from_demand(d_pred)
            plans_gt = FG.plan_from_demand(torch.stack(d_gt))
            for j, row in enumerate(batch):
                anns = [a for a in by_image.get(off + j + 1, []) if not a["iscrowd"]]
                wh = np.array([row["width"], row["height"]] * 2, dtype=np.float64)
                b = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in anns],
                             dtype=np.float64).reshape(-1, 4)
                b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, row["width"])
                b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, row["height"])
                require(bool(np.all(b[:, 2:] > b[:, :2])), "certified GT with a non-positive clipped side")
                boxes = torch.from_numpy(b / wh)
                g = {"predicted": magnify(plans_pred.select(slice(j, j + 1)), boxes),
                     "gt_demand": magnify(plans_gt.select(slice(j, j + 1)), boxes)}
                for n, a in enumerate(anns):
                    per_gt.append({"image": row["stable_image_id"], "raw_annotation_id": a["raw_annotation_id"],
                                   "category_id": a["category_id"], "area": a["area"],
                                   "bbox_xyxy": [a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]],
                                   **{k: float(v[n]) for k, v in g.items()}})
                per_image.append({"image": row["stable_image_id"], "gt": len(anns),
                                  "demand_sum_predicted": float(d_pred[j].sum()), "demand_sum_gt": float(d_gt[j].sum()),
                                  "identity_predicted": bool(plans_pred.identity[j]), "identity_gt": bool(plans_gt.identity[j]),
                                  "demand_l1_difference": float((d_pred[j] - d_gt[j]).abs().sum()),
                                  "demand_correlation": _corr(d_pred[j], d_gt[j])})
    return per_gt, per_image, d_gt_all


def _corr(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    den = float(a.norm() * b.norm())
    return None if den == 0.0 else float((a * b).sum()) / den


def infer(c, out, counts):
    K, DS, B = mods()
    import r28_eval as EV
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    from sparse_rtdetr.protocol_r18.engine import tensor_digest
    from sparse_rtdetr.research import r27_l0_control as L0
    from sparse_rtdetr.research import r28_fovea_geo as FG
    from sparse_rtdetr.research import r28_resample as RS
    torch = K.torch_mod()
    dev_round = c["dev"]
    i = c["inputs"]
    K.PHASE["phase"] = "infer/load"
    others = gpu_compute_processes()
    cf = fg_contract(K, i["FG_contract"])
    side, fg_state, source, side0, l0_state = _models(c, cf)
    device = torch.device("cuda:0")
    data, rows, parsed = _development(c, cf)
    ids = [r["stable_image_id"] for r in rows]
    limit = DEV_LIMIT if dev_round else None
    warm, scored = (K.EVAL_WARMUP, K.EVAL_BATCHES) if limit is None else limit
    per_run = warm + scored
    model = FG.build_eval_model(fg_state, device, "FG", source)
    require(FG.is_wrapped(model) and not model.training and model.r28_ctx.binding() == side["demand"]
            and model.r28_mapping == side["mapping"] and tensor_digest(model.state_dict().items()) == PIN["FG_weights"],
            "F_geo evaluation model")
    network_ctx = model.r28_ctx

    # ---- check 1: this entry under the predicted demand, first 8 images, against the stored 009 predictions
    K.PHASE["phase"] = "infer/check1"
    _, p1, _ = EV.infer(cf, model, counts, limit=(0, 2))
    require(counts["forwards"] == 2, "check 1 forward count")
    check1 = {"images": 8, "forwards": 2}
    for b in K.BUDGETS:
        stored = K.read_json(K.pinned(i["FG_predictions"][str(b)]))
        require(len(stored) == 548 * b, "stored F_geo prediction record count")
        diff = records_equal(p1[b], stored[:8 * b])
        check1[f"top{b}"] = {"records": len(p1[b]), "equal": diff is None, "first_difference": diff,
                             "fields": sorted(p1[b][0])}
        del stored
    require(check1["top600"]["equal"] and check1["top300"]["equal"],
            f"check 1: this entry does not reproduce the stored 009 predictions: {check1}")
    check1["passed"] = True
    K.write_new(out / "CHECK1.json", check1)

    # ---- geometry (demand network and map only)
    K.PHASE["phase"] = "infer/geometry"
    n_geo = len(rows) if not dev_round else scored * K.EVAL_BATCH
    per_gt, per_image, d_gt = _geometry(c, cf, network_ctx, data, rows, parsed, counts, n_geo)
    K.write_new(out / "per_gt.json", per_gt)
    K.write_new(out / "per_image.json", per_image)

    def run(name, m, extra=None):
        K.PHASE["phase"] = f"infer/{name}"
        before, digest, rng = counts["forwards"], tensor_digest(m.state_dict().items()), ck.capture_rng(device)
        _, preds, timing = EV.infer(cf, m, counts, extra=extra, limit=limit)
        require(counts["forwards"] - before == per_run, f"{name} forward count")
        require(tensor_digest(m.state_dict().items()) == digest and ck.exact_equal(ck.capture_rng(device), rng),
                f"{name} inference changed the model or the RNG")
        d = out / name
        d.mkdir()
        refs = {}
        for b in K.BUDGETS:
            K.write_new(d / f"top{b}-predictions.json", preds[b])
            refs[str(b)] = K.ref(d / f"top{b}-predictions.json")
        return {"endpoint": ENDPOINT_TEXT[name], "weights_sha256": digest, "predictions": refs, "forwards": per_run,
                "images_scored": scored * K.EVAL_BATCH, **timing}

    # ---- A2: the same model object, demand source replaced by the label demand
    order = batch_order(ids, warm, scored, K.EVAL_BATCH)
    oracle = make_oracle_context(RS.Context, d_gt, order, vars(network_ctx)["r28_fwhm"]).to(device)
    seen = []
    vars(model)["r28_ctx"] = oracle
    require(FG.is_wrapped(model), "F_geo model with the label-demand context")
    res = {"A2": run("A2", model, extra=lambda x, rws, index: seen.append([r["stable_image_id"] for r in rws]))}
    require(oracle.calls == per_run and seen == order[warm:], "label-demand batches differ from the evaluated batches")
    res["A2"].update(demand={"kind": LABEL, "calls": oracle.calls, "images": len(d_gt),
                             "state_sha256": tensor_digest([(k, d_gt[k]) for k in ids])},
                     mapping=side["mapping"], weights="F_geo e120 EMA", endpoint_file=side["file"])
    vars(model)["r28_ctx"] = network_ctx
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ---- A3a: F_geo weights on the uniform input (the L0 forward)
    m = L0.build_eval_model(fg_state, device)
    require(L0.is_wrapped(m) and not FG.is_wrapped(m) and not hasattr(m, "r28_ctx")
            and tensor_digest(m.state_dict().items()) == PIN["FG_weights"], "A3a model")
    res["A3a"] = run("A3a", m)
    res["A3a"].update(weights="F_geo e120 EMA", endpoint_file=side["file"], forward="r27_l0_control (to_low, native network)")
    del m
    gc.collect()
    torch.cuda.empty_cache()

    # ---- A3b: L0 weights on the F_geo forward with the predicted demand
    m = FG.build_eval_model(l0_state, device, "FG", source)
    require(FG.is_wrapped(m) and m.r28_ctx.binding() == side["demand"] and m.r28_mapping == side["mapping"]
            and tensor_digest(m.state_dict().items()) == cf["L0_weights_sha256"], "A3b model")
    res["A3b"] = run("A3b", m)
    res["A3b"].update(weights="L0 e120 EMA", endpoint_file=side0["file"], demand=side["demand"], mapping=side["mapping"],
                      forward="r28_fovea_geo (predicted demand, F_geo map, boxes mapped back by its inverse)")
    del m
    gc.collect()
    torch.cuda.empty_cache()
    require(counts["forwards"] == 2 + 3 * per_run and counts["actual_updates"] == 0, "inference workload")
    require(dev_round or counts["forwards"] == 425, "formal detector forward count is 2 + 3 x 141")
    return {"stage": "infer", "label": LABEL, "check1": check1, "endpoints": res,
            "geometry": {"per_gt": K.ref(out / "per_gt.json"), "per_image": K.ref(out / "per_image.json"),
                         "GT_measured": len(per_gt), "images": len(per_image)},
            "limit": None if limit is None else list(limit), "other_gpu_compute_processes_at_start": others,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(), "device": torch.cuda.get_device_name(0)}


# ================================================================ score
def score(c, out, counts):
    K, DS, B = mods()
    import r28_followup as FU
    from infer_seed import bridge_values, _group, _prefix
    from sparse_rtdetr.r0.evaluation import score_budget
    i = c["inputs"]
    K.PHASE["phase"] = "score/load"
    inf = stage_closed(K, c, "infer")
    cf = fg_contract(K, i["FG_contract"])
    data, rows, parsed = _development(c, cf)
    ids = [r["stable_image_id"] for r in rows]
    res = {}
    for name in ENDPOINTS:
        e = inf["endpoints"][name]
        preds = {b: K.read_json(K.pinned(e["predictions"][str(b)])) for b in K.BUDGETS}
        if not c["dev"]:
            grouped = {b: _group(preds[b], ids, b, whole=True, exact=True) for b in K.BUDGETS}
            _prefix(grouped[600], grouped[300], ids, whole=True)
            del grouped
        values, gt, files, seconds, per_class = {}, {}, {}, {}, None
        for b in K.BUDGETS:
            K.PHASE["phase"] = f"score/{name}/top{b}"
            t = time.perf_counter()
            metrics = score_budget(rows, parsed, preds[b], c["source_root"])
            counts.update(score_budget_calls=1)
            gt[str(b)] = {k: metrics["spatial_filter_counts"][k] for k in GT_COUNTS}
            require(gt[str(b)] == GT_COUNTS, f"{name} top{b}: scored GT set is {gt[str(b)]}, expected {GT_COUNTS}")
            metrics["R30_binding"] = {"contract_sha256": c["contract_sha256"], "endpoint": name, "text": ENDPOINT_TEXT[name],
                                      "label": LABEL if name == "A2" else "weights_input_cross_readout", "kind": "EMA",
                                      "view": "A", "epoch": 120, "budget": b, "weights_sha256": e["weights_sha256"],
                                      "predictions": e["predictions"][str(b)], "dev_partial": c["dev"]}
            values[str(b)] = bridge_values(metrics)
            p = out / f"{name}-EMA-A-top{b}-metrics.json"
            K.write_new(p, metrics)
            files[str(b)] = K.ref(p)
            seconds[str(b)] = time.perf_counter() - t
            if name == "A2" and b == 600:
                keep = out / "A2-cocoeval-top600.npz"
                apl, apm = FU.per_class_sizes(rows, parsed, preds[b], metrics["COCO_bridge"], keep=keep)
                counts.update(per_class_COCOeval_calls=1)
                per_class = {"APL": apl, "APM": apm, "tensors": K.ref(keep)}
        res[name] = {"values": values, "GT_counts": gt, "metrics": files, "seconds": seconds,
                     **({} if per_class is None else {"per_class_top600": per_class})}
        del preds
        gc.collect()
    require(counts["score_budget_calls"] == 6 and counts["per_class_COCOeval_calls"] == 1 and counts["forwards"] == 0,
            "scorer call counts")
    return {"stage": "score", "endpoints": res, "GT_counts_asserted_every_call": True, "dev_partial_predictions": c["dev"]}


# ================================================================ tables
def _per_class(npz_path):
    """Per-class APS / APM / APL of one stored COCOeval (maxDets 100)."""
    import numpy as np
    z = np.load(npz_path, allow_pickle=False)
    labels = [str(x) for x in z["areaRngLbl"]]
    out = []
    for k, cat in enumerate(z["catIds"].tolist()):
        row = {"category_id": int(cat)}
        for key, label in (("APS", "small"), ("APM", "medium"), ("APL", "large")):
            ix = labels.index(label)
            p = z["precision"][:, :, k, ix, -1]
            v = p[p > -1]
            row[key] = float(v.mean()) if len(v) else None
            row[f"GT_{label}"] = int(z["gt_count"][k][ix])
        out.append(row)
    return out


def tables(c, out, counts):
    import numpy as np
    K, DS, B = mods()
    i = c["inputs"]
    dev = c["dev"]
    K.PHASE["phase"] = "tables/read"
    stage_closed(K, c, "check3")
    inf, sc = stage_closed(K, c, "infer"), stage_closed(K, c, "score")
    require(inf["check1"]["passed"] is True, "check 1")
    per_gt = K.read_json(K.pinned(inf["geometry"]["per_gt"]))
    if not dev:
        w_gt = K.read_json(K.pinned(i["W_per_gt"]))
        key = ("image", "raw_annotation_id", "category_id", "area", "bbox_xyxy")
        require(len(per_gt) == len(w_gt) and all(all(a[k] == b[k] for k in key) for a, b in zip(per_gt, w_gt)),
                "the measured GT set differs from the g0 set of the W round")
        require(max(abs(a["gt_demand"] - 0.0) for a in per_gt) > 0, "label-demand magnification missing")
        del w_gt
    small = [r for r in per_gt if r["area"] <= SMALL_AREA]
    require(dev or len(small) == SMALL_GT, f"small GT count {len(small)}")
    area_all = np.array([r["area"] for r in per_gt])
    area = np.array([r["area"] for r in small])
    g_pred = np.array([r["predicted"] for r in small])
    g_gt = np.array([r["gt_demand"] for r in small])
    sources = {"L0": i["L0_predictions"]["600"], "FG": i["FG_predictions"]["600"], "S15": i["S15_predictions_600"],
               "A2": inf["endpoints"]["A2"]["predictions"]["600"]}
    iou = {}
    for name, ref in sources.items():
        index, n = DS._by_image(K.pinned(ref))
        require(n == 548 * 600 or (dev and name == "A2"), f"{name} top600 prediction record count {n}")
        iou[name] = np.array([DS._best_iou(np.array(r["bbox_xyxy"], dtype=np.float64), index.get(r["image"], {}).get(r["category_id"]))
                              for r in small])
        del index
        gc.collect()
    hit = {name: {t: v >= t for t in THRESHOLDS} for name, v in iou.items()}
    K.PHASE["phase"] = "tables/compute"
    three = {k: iou[k] for k in ("L0", "FG", "S15")}
    a1 = {"small_GT": len(small), "g": "F_geo deployment map under the predicted demand (frozen demand network), per GT",
          "bins": bin_table(g_pred, area, three, pairs=(("FG", "L0"), ("S15", "FG"), ("S15", "L0"))),
          "S15_hit_FG_miss": miss_table(g_pred, area, hit["S15"], hit["FG"]),
          "L0_hit_FG_miss": miss_table(g_pred, area, hit["L0"], hit["FG"])}
    ref_values = c["reference_values"]
    v = {name: sc["endpoints"][name]["values"] for name in ENDPOINTS}
    geo_all = {k: geometry_reading(np.array([r[k] for r in per_gt]), area_all) for k in ("predicted", "gt_demand")}
    fg_classes = _per_class(K.pinned(i["FG_cocoeval_tensors"]))
    a2_tensors = sc["endpoints"]["A2"]["per_class_top600"]["tensors"]
    a2_classes = _per_class(K.pinned(a2_tensors))
    per_class = [{"category_id": a["category_id"], **{f"GT_{s}": a[f"GT_{s}"] for s in ("small", "medium", "large")},
                  **{f"{m}_A2": a[m] for m in ("APS", "APM", "APL")}, **{f"{m}_FG": f[m] for m in ("APS", "APM", "APL")},
                  **{f"{m}_difference_pp": (None if a[m] is None or f[m] is None else 100. * (a[m] - f[m])) for m in ("APS", "APM", "APL")}}
                 for a, f in zip(a2_classes, fg_classes)]
    require([a["category_id"] for a in a2_classes] == [f["category_id"] for f in fg_classes] == list(range(1, 11)), "per-class tables")
    d_fg = difference_pp(v["A2"], ref_values["FG"])
    aps = d_fg["600"]["APS"]
    a2 = {"label": LABEL, "values": v["A2"], "GT_counts": sc["endpoints"]["A2"]["GT_counts"],
          "differences_pp": {"A2-FG": d_fg, "A2-L0": difference_pp(v["A2"], ref_values["L0"]),
                             "A2-S15": difference_pp(v["A2"], ref_values["S15"])},
          "controller_reading": {"A2_minus_FG_APS600_pp": aps,
                                 "band": ">=+0.8" if aps >= 0.8 else "<=+0.3" if aps <= 0.3 else "between",
                                 "note": "the controller's pre-written practical reading; not a gate, not a significance test"},
          "geometry": geo_all,
          "bins_by_gt_demand_g": {"g": "F_geo deployment map under the label demand, per GT",
                                  "bins": bin_table(g_gt, area, {k: iou[k] for k in ("A2", "FG", "L0", "S15")},
                                                    pairs=(("A2", "FG"), ("A2", "L0"), ("S15", "A2"))),
                                  "note": "the task asks this table for the A2 endpoint; the FG / L0 / S15 columns are the "
                                          "same stored predictions regrouped by the label-demand g, added for reading only"},
          "bins_by_predicted_g": {"bins": bin_table(g_pred, area, {k: iou[k] for k in ("A2", "FG")}, pairs=(("A2", "FG"),)),
                                  "note": "A2 against F_geo on the predicted-demand bins of A1 (same GT in every bin)"},
          "bin_transition_predicted_to_gt_demand": {"rows": "predicted-demand bin", "columns": "label-demand bin",
                                                    "bins": list(BIN_NAMES), "counts": cross_bins(g_pred, g_gt)},
          "A2_hit_FG_miss": miss_table(g_pred, area, hit["A2"], hit["FG"]),
          "per_class_top600": per_class, "per_class_tensors": a2_tensors}
    cellsx = {"L0_weights/uniform_input": ref_values["L0"], "FG_weights/FG_input": ref_values["FG"],
              "FG_weights/uniform_input": v["A3a"], "L0_weights/FG_input": v["A3b"]}
    a3 = {"cells": cellsx,
          "sources": {"L0_weights/uniform_input": "stored L0 evaluation (R27-002)", "FG_weights/FG_input": "stored F_geo evaluation (009)",
                      "FG_weights/uniform_input": "this round, A3a", "L0_weights/FG_input": "this round, A3b"},
          "GT_counts": {k: sc["endpoints"][k]["GT_counts"] for k in ("A3a", "A3b")},
          "differences_pp": {
              "FG_minus_L0 (both matched)": difference_pp(ref_values["FG"], ref_values["L0"]),
              "input_only (L0 weights: FG input - uniform input)": difference_pp(v["A3b"], ref_values["L0"]),
              "weights_only (uniform input: FG weights - L0 weights)": difference_pp(v["A3a"], ref_values["L0"]),
              "input_given_FG_weights (FG weights: FG input - uniform input)": difference_pp(ref_values["FG"], v["A3a"]),
              "weights_given_FG_input (FG input: FG weights - L0 weights)": difference_pp(ref_values["FG"], v["A3b"])},
          "values_S15": ref_values["S15"]}
    res = {"status": "PASS", "version": VERSION, "contract_sha256": c["contract_sha256"], "source_commit": c["source_commit"],
           "record_only": True, "dev_partial": dev, "A1": a1, "A2": a2, "A3": a3,
           "checks": {"check1": inf["check1"], "check2_GT_counts": {k: sc["endpoints"][k]["GT_counts"] for k in ENDPOINTS},
                      "check3": {k: stage_closed(K, c, "check3")[k] for k in ("passed", "report_cells", "all_five_rows_equal_stored_dose_W")}},
           "workload": {name: stage_closed(K, c, name)["counts"] for name in ("check3", "infer", "score")},
           "stage_seconds": {name: stage_closed(K, c, name)["elapsed_seconds"] for name in ("check3", "infer", "score")},
           "inputs": {k: i[k] for k in ("FG_predictions", "L0_predictions", "S15_predictions_600", "FG_endpoint", "L0_endpoint",
                                        "S15_endpoint", "demand_net", "evaluation_policy")},
           "new_predictions": {k: inf["endpoints"][k]["predictions"] for k in ENDPOINTS},
           "statements": STATEMENTS}
    require(counts["forwards"] == 0 and not counts.get("score_budget_calls"), "tables must not run a model or a scorer")
    K.write_new(out / "RESULT.json", res)
    return {"stage": "tables", "result": K.ref(out / "RESULT.json"), "A2_minus_FG_APS600_pp": aps,
            "band": a2["controller_reading"]["band"]}


# ================================================================ CLI
STAGES = {"check3": (check3, False), "infer": (infer, True), "score": (score, False), "tables": (tables, False)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("prepare",) + tuple(STAGES))
    ap.add_argument("--round", required=True, type=Path)
    ap.add_argument("--release")
    args = ap.parse_args(argv)
    if args.action == "prepare":
        require(args.release, "prepare needs --release")
        return prepare(args.round, args.release)
    K, DS, B = mods(args.round)
    c = load_contract(args.round)
    run, cuda = STAGES[args.action]
    out = Path(c["outputs"]) / args.action
    out.mkdir(exist_ok=False)
    start, cpu0 = time.monotonic(), time.process_time()
    counts = Counter(actual_updates=0, forwards=0, forward_attempts=0)
    try:
        from r25_run import setup
        if cuda:
            require(os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == "expandable_segments:True", "allocator conf")
            setup("cuda:0")
            K.torch_mod().cuda.reset_peak_memory_stats()
        else:
            require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", f"the {args.action} stage requires CUDA_VISIBLE_DEVICES=\"\"")
            setup("cpu")
        result = run(c, out, counts)
        K.write_new(out / "CLOSED.json", {"status": "PASS", "action": args.action, "contract_sha256": c["contract_sha256"],
                                          "counts": dict(counts), "elapsed_seconds": time.monotonic() - start,
                                          "process_cpu_seconds": time.process_time() - cpu0, **result})
    except BaseException as exc:
        K.write_new(out / "STOP.json", {"status": "STOP", "phase": K.PHASE["phase"], "action": args.action,
                                        "error": repr(exc)[:4000], "traceback": traceback.format_exc(),
                                        "counts": dict(counts), "elapsed_seconds": time.monotonic() - start})
        raise


if __name__ == "__main__":
    main()
