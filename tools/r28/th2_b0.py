"""TH2 B0 (TASK_TH2_B0_001): the geometry gate of a two-dimensional (non-separable) attraction map.
Geometry only: no detector forward, no training, no scoring, no test-dev, no image outside train_core.

usage: th2_b0.py <action> --round <round> [--release <name>]

  prepare    contract: release, pinned phase-1 files, widths, definitions, selection rule, verdict rule, figure ids.
  wire       CPU: SEP(19.03125) on development must reproduce the phase-1 readings and per-GT magnifications.
  demand_tc  CUDA: the frozen demand network on every train_core image of the round, once (unaugmented H canvas).
  tc         CPU: train_core GT table, label demand, T1 (form x width x source) and the T4 part of train_core.
  select     CPU: the width rule on the NS rows of T1 with source P -> SELECTION.json.
  dev        CPU: development readings T2, T3, the per-GT file (refuses to run before SELECTION.json exists).
  figs       CPU: at most 12 train_core images, SEP(19.03125) and NS(w*) canvases side by side.
  tables     CPU: verdict and RESULT.json.

Every stage writes <round>/<name>/CLOSED.json (PASS) or STOP.json.  A dev round (dev-round* inside the round) uses the
first TH2B0_DEV_N train_core images by stable id and never computes a non-separable map on development: its `dev`
stage runs the same code with separable stand-ins.  Its numbers are not results.
"""
import argparse
import datetime
import gc
import io
import json
import os
import shutil
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "tools" / "r27_l0")]

import numpy as np  # noqa: E402

import l0_common as B  # noqa: E402
import r31_external as X  # noqa: E402
import th2b0_core as C  # noqa: E402

VERSION = "TH2-B0-001-v2"
TASK = ("TASK_TH2_B0_001.md", "03e62a4aea680f6ab6b787a1a6db57bc4a1e2da463feeffd54c33109527181c2")
P1 = B.WS / "experiments/thesis-study2/th2-phase1-20261122-001"
P1_CONTRACT_SHA = "2471ad9271ebbae0c645cc283de137118e9571d1aefadf7d321028414ae8788d"
P1_MANIFEST_SHA = "bdfe93fb0c58868581d4d1914084ac2cf1826daf8f51c8cfd8e0991949d02695"
P1_RESULT_SHA = "1ae92152e027d54cbbe93ac82c497ffbe8441ec9d88f2b5973afe131d870ba68"
P1_COMMIT = "b24dec791b72812c8d9a1e6342ff637497bef770"
ALLOWED_ADDED = ("tools/r28/th2b0_core.py", "tools/r28/th2_b0.py", "tools/r28/th2b0_chain.sh", "tests/test_th2_b0.py")
ALLOWED_CHANGED = ("tools/r28/th2_phase1.py",)          # the erratum commit bc29d03 of the phase-1 branch
N_TC, N_DEV, N_GT, N_SMALL = 4869, 548, 38599, 26441
WIRE = {"P": {"small_g_geomean": 1.217172830834459, "small_share_g_lt_1.2": 0.47490639537082563},
        "Lb": {"small_g_geomean": 1.2824692214016544, "small_share_g_lt_1.2": 0.35255852653076664},
        "Lb_collateral": {"pooled_share": 0.8005455838603095, "share_of_canvas_ge_1.2": 0.1914805462125306}}
WIRE_TOL = 1e-12
SEEDS = (0, 1, 2)
FIGURES = 12
WORKERS = 8
CHUNK = 12000
OFFICIAL = {
    "repository": "https://github.com/tchittesh/fovea", "commit": "c8707bd2e4f10678f6636d24c2915a43b44a59ed",
    "file": "fovea/models/grid_generator.py", "sha256": "4d36cec017d3acf888bb9643a5746507feb817b5fcce640bc8528bcf40c1596b",
    "function": "RecasensSaliencyToGridMixin.nonseparable_saliency_to_grid", "function_present": True,
    "agrees_with_task_4.2": ["reflect padding by min(grid_shape) - 1 = 30 nodes on the four sides (anti_crop)",
                             "kernel make2DGaussian(61, fwhm): exp(-4 ln2 (di^2 + dj^2) / fwhm^2) on the full 61 x 61 square",
                             "node position = conv2d(P * saliency) / conv2d(saliency) for x and for y with the same weights; "
                             "P_x = (j - 30) / 50, P_y = (i - 30) / 30",
                             "clamp of 2G - 1 to [-1, 1]", "bilinear between the 31 x 51 nodes (F.interpolate, align_corners=True)"],
    "differences": [
        "saliency: the official generators pass a 31 x 51 KDE of boxes (or a dataset prior); here it is D + 0.2 on 12 x 21 read "
        "to the 31 x 51 nodes by the r28_fovea.node_saliency rule in both directions (task 4.2) and divided by its sum",
        "arithmetic: FP64 numpy on the CPU, the 2-D kernel applied as two 1-D sums; the official code is FP32 conv2d on CUDA",
        "frame: G is used as a continuous piecewise bilinear map on normalised canvas coordinates with nodes at u = j/50, "
        "v = i/30 (the convention of arm F / F_geo); the official code resamples the node grid to the output pixels and "
        "renders with grid_sample.  No image is rendered here except the illustrations (plain bilinear sampling)",
        "boxes: the official unwarp_bboxes maps the two corners of a canvas box through the grid (grid_sample, border "
        "padding) and has no source -> canvas direction; here source -> canvas is the numerical inverse of G and the "
        "round trip takes the bounding rectangle of 32 mapped boundary points (task 4.4)",
        "kernel width: the released configuration uses 4 and the constructor default is 13; here nine fixed widths",
        "validity: the official code does not test the grid for folds; here every cell corner determinant must be positive",
        "constant demand: SEP keeps arm F's exact-uniform bypass for a constant demand map; NS has no bypass (a constant "
        "saliency gives the uniform nodes up to rounding)",
        "order of operations: SEP reads the marginal of D + 0.2 to the nodes; NS reads D + 0.2 itself to the nodes.  For a "
        "rank-one D + 0.2 the two maps coincide exactly; otherwise 'no marginalisation' is the only difference",
    ],
}
DEFINITIONS = {
    "TC": "train_core, 4,869 images, unaugmented; boxes clipped to the image and normalised (r28_demand.deployment_xyxy)",
    "TC_GT": "the valid GT of the annotation file kept by deployment_xyxy that pass the evaluation filter of the development "
             "table (r0.evaluation.bridge_inputs per image: raster ignore prefilter >= 0.5 removed, non-crowd).  Annotation rows "
             "of non-positive extent (role nonpositive_area; they cover no pixel) are left out before the filter, which "
             "refuses them; the tc receipt counts them",
    "DEV": "development 548 images; the per-GT table of phase 1 (geom/gt.npz): 38,759 deployment rows, 38,599 scored, 26,441 small",
    "P": "the frozen demand network's 12 x 21 map (without eps): development from phase 1 demand/demand.npy; train_core from "
         "one forward per image through r28_resample.Context.demand on the FP32 H canvas (CUDA, batch 4, as phase 1)",
    "Lb": "r28_warp.demand_from_boxes of the deployment boxes of the image",
    "size_bins": "annotation area: <16^2, [16^2, 32^2], (32^2, 64^2), [64^2, 96^2), >=96^2; small = bins 1-2, medium = bins 3-4",
    "SEP": "r28_fovea_cal.nodes_from_demand(demand, w) (the code of arm F / F_geo, width w); SEP(19.03125) = F_geo, SEP(4) = F",
    "NS": "th2b0_core.ns_nodes(node_saliency_2d(D + 0.2), w): task 4.2",
    "fold": "NS: an image whose 30 x 50 cells have a non-positive (or non-finite) Jacobian determinant at any of the four "
            "corners.  SEP: an image whose nodes are not finite and strictly increasing on either axis.  Nothing is repaired; "
            "the GT of a folded image are not measured and the row reports the count",
    "inverse": "source -> canvas by damped Newton on the piecewise bilinear G with an exhaustive per-cell search for the "
               "points that are left; every point is mapped back with G and must agree within 1e-9",
    "g": "sqrt(area of the canvas polygon of 32 boundary points / source box area), normalised coordinates; SEP rows report "
         "r28_fovea.magnification sqrt(ox * oy) and the polygon value must agree with it within 1e-9",
    "stretch": "|log(sigma_1 / sigma_2)| of the Jacobian at the box centre",
    "expand": "area of the axis-aligned bounding rectangle of the canvas polygon / polygon area",
    "round_trip_iou": "R = bounding rectangle of the canvas polygon; 32 boundary points of R through G; bounding rectangle "
                      "in the source; IoU with the original box",
    "local_magnification": "1 / sqrt(det of the canvas -> source Jacobian, normalised) at the 768 x 1344 canvas pixel centres",
    "collateral": "share of canvas pixels with local magnification >= 1.2, and the share of those not covered by any scored "
                  "small GT (canvas polygon dilated by 8 pixels, square element; a pixel is covered when it meets the shape)",
    "compression": "share of canvas pixels with local magnification <= 0.9, and the share of those not covered by any GT of "
                   "the deployment table (scored or not), same dilation",
    "nonseparability": "1 - sigma_1^2 / sum sigma_i^2 of the 12 x 21 matrix D + 0.2",
    "bootstrap": "images resampled with replacement, 1,000 times, seed 20261122 (T2 only)",
    "T3_hit": "phase-1 e0/outcomes.npz: best same-class IoU >= 0.75 of the stored top600 of S15 and F_geo, per seed",
    "dlogg": "log g under NS(w*) - log g under SEP(19.03125), source P",
    "figures": "train_core ids sorted; positions round(k (n - 1) / 11), k = 0..11; source P; plain bilinear rendering at "
               "672 x 384 per canvas; every deployment GT polygon drawn",
}
NOT_AUTHORISED = ["any detector forward, training or scoring", "test-dev; the 1,602 held-out train images",
                  "installing or upgrading packages; edits under experiments/localization-distribution",
                  "widths, sources or map forms outside task section 4; extra scans", "merging pull requests"]


def is_dev(r):
    return Path(r).resolve().name.startswith("dev-round")


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def hexes():
    return [w.hex() for w in C.WIDTHS]


# ================================================================ contract
def prepare(round_dir, release):
    R = Path(round_dir).resolve()
    dev = is_dev(R)
    B.require(not (R / "CONTRACT.json").exists(), "CONTRACT.json exists")
    B.require((R / release / "source").resolve() == ROOT.resolve(), "prepare must run from <round>/<release>/source")
    B.require("localization-distribution" not in str(R), "outputs must not be under experiments/localization-distribution")
    home = R.parent if dev else R
    task = home / TASK[0]
    B.require(task.is_file() and B.file_sha(task) == TASK[1], f"task file missing or different: {task}")
    n = os.environ.get("TH2B0_DEV_N")
    B.require((n is not None) == dev, "TH2B0_DEV_N is required in a dev round and refused in a formal round")
    man_ref = B.ref(R / release / "MANIFEST.json")
    manifest = B.read_json(man_ref["path"])
    B.require(manifest["dirty"] is False or dev, "a formal release must come from a clean committed tree")
    B.release_intact(ROOT, manifest["files"])
    base_ref = B.ref(P1 / "release-001/MANIFEST.json")
    base = B.read_json(base_ref["path"])
    B.require(base_ref["sha256"] == P1_MANIFEST_SHA and base["commit"] == P1_COMMIT, "phase-1 release identity")
    new, old = manifest["files"], base["files"]
    delta = {"added": sorted(set(new) - set(old)), "removed": sorted(set(old) - set(new)),
             "changed": sorted(f for f in new if f in old and new[f] != old[f])}
    B.require(set(delta["added"]) == set(ALLOWED_ADDED) and not delta["removed"] and set(delta["changed"]) <= set(ALLOWED_CHANGED),
              f"the release may differ from the phase-1 release only by the B0 files and the erratum: {delta}")
    p1_ref = B.ref(P1 / "CONTRACT.json")
    B.require(p1_ref["sha256"] == P1_CONTRACT_SHA, "phase-1 contract file")
    p1 = B.read_json(p1_ref["path"])
    B.require(os.path.samefile(sys.executable, p1["python"]), f"python differs from phase 1: {sys.executable}")
    res_ref = B.ref(P1 / "tables/RESULT.json")
    B.require(res_ref["sha256"] == P1_RESULT_SHA, "phase-1 RESULT.json")
    res = B.read_json(res_ref["path"])
    geo_a, e3 = res["E2"]["geometry"]["geometry"]["a"]["small"], res["E3"]
    B.require(geo_a["g_geomean"] == WIRE["P"]["small_g_geomean"] and geo_a["share_g_lt_1.2"] == WIRE["P"]["small_share_g_lt_1.2"]
              and e3["ideal_2d_allocation"]["small_GT_under_b_actual"]["g_geomean"] == WIRE["Lb"]["small_g_geomean"]
              and e3["ideal_2d_allocation"]["small_GT_under_b_actual"]["share_g_lt_1.2"] == WIRE["Lb"]["small_share_g_lt_1.2"]
              and e3["collateral_magnification"]["pooled_share"] == WIRE["Lb_collateral"]["pooled_share"]
              and e3["collateral_magnification"]["share_of_canvas_ge_1.2"] == WIRE["Lb_collateral"]["share_of_canvas_ge_1.2"],
              "the wiring targets of task 4.6 are not the values of the phase-1 RESULT.json")
    stored = {}
    for stage, keys in (("demand", ("demand",)), ("geom", ("gt", "g", "d_star")), ("e0", ("outcomes",))):
        closed = B.read_json(P1 / stage / "CLOSED.json")
        B.require(closed["status"] == "PASS" and closed["contract_sha256"] == p1["contract_sha256"], f"phase-1 stage {stage}")
        for k in keys:
            B.pinned(closed["files"][k])
            stored[f"{stage}/{k}"] = closed["files"][k]
    r30 = B.read_json(B.pinned({k: p1["R30_contract"][k] for k in ("path", "bytes", "sha256")}))
    B.require(r30["contract_sha256"] == p1["R30_contract"]["contract_sha256"], "R30 contract")
    B.pinned(p1["demand"]["weights"])
    from sparse_rtdetr.research import r28_fovea as FV
    from sparse_rtdetr.research import r28_fovea_geo as FG
    from sparse_rtdetr.research import r28_warp as W
    B.require(FV.OFFICIAL["commit"] == OFFICIAL["commit"] and FV.OFFICIAL["files"][OFFICIAL["file"]] == OFFICIAL["sha256"],
              "the official file identity differs from the one bound by arm F")
    B.require(FG.WIDTH == C.FG_WIDTH and FV.GRID_SHAPE == C.NODES and FV.PAD == C.PAD and W.EPS == C.EPS and W.GRID == C.GRID
              and W.L_HW == C.L_HW, "map constants")
    tc_binding = r30["data"]["train_core"]
    ids = sorted(r["stable_image_id"] for r in B.read_json(tc_binding["manifest"])["records"])
    B.require(len(ids) == N_TC == len(set(ids)), "train_core members")
    if dev:
        ids = ids[:int(n)]
    figure_ids = [ids[round(k * (len(ids) - 1) / (FIGURES - 1))] for k in range(FIGURES)]
    B.require(len(set(figure_ids)) == FIGURES, "figure ids")
    est = {"tc_per_gt_metrics": 36 * 350_000 * 24, "demand_maps": N_TC * 252 * 8 * 2, "dev_per_gt": 60_000_000,
           "release_source": 15_000_000, "figures_json_logs": 15_000_000}
    c = {"version": VERSION, "task": {"file": TASK[0], "sha256": TASK[1]}, "zero_training": True, "zero_detector_forwards": True,
         "source_root": str(ROOT), "source_commit": manifest["commit"], "source_branch": manifest["branch"],
         "source_dirty": manifest["dirty"], "source_manifest": man_ref, "base_manifest": base_ref, "source_delta": delta,
         "outputs": str(R), "dev": dev, "phase1": {"contract": p1_ref, "contract_sha256": p1["contract_sha256"], "result": res_ref,
                                                     "files": stored},
         "demand": p1["demand"], "data": {"train_core": tc_binding}, "development_ids": p1["development_ids"],
         "tc_ids": ids, "figure_ids": figure_ids, "gpu_uuid": p1["gpu_uuid"], "python": sys.executable,
         "official": OFFICIAL, "widths": list(C.WIDTHS), "widths_hex": hexes(), "forms": ["SEP", "NS"], "sources": ["P", "Lb"],
         "map": {"grid": list(C.GRID), "nodes": list(C.NODES), "padding": C.PAD, "eps": C.EPS, "canvas": list(C.L_HW)},
         "definitions": DEFINITIONS, "wire_targets": WIRE, "wire_tolerance": WIRE_TOL,
         "selection_rule": {"rows": "T1, form NS, source P", "order": ["folded", "medium_mean_logg", "small_round_trip_iou"],
                            **C.SELECT, "pick": "largest small mean log g; ties to the smaller width",
                            "no_width_left": "STOP; NS(w*) rows of T2-T4 report NS(19.03125), record only"},
         "verdict_rule": C.VERDICT, "inverse_tolerance": C.INVERSE_TOLERANCE,
         "dev_round_rule": "first TH2B0_DEV_N train_core ids; the dev stage replaces every NS map on development by a "
                           "separable stand-in (NS(w) -> SEP(w), and SEP(13) when that is already a row)",
         "limits": {"cpu_hours": 6.0, "gpu_minutes": 10.0, "storage_bytes": 1_000_000_000},
         "resource_plan": {"storage_estimate_bytes": est, "storage_estimate_total": sum(est.values()),
                           "disk_free_bytes_at_prepare": shutil.disk_usage(R).free},
         "not_authorised": NOT_AUTHORISED}
    c["contract_sha256"] = B.sha_bytes(B.canonical(c))
    B.write_new(R / "CONTRACT.json", c)
    print(json.dumps({"contract_sha256": c["contract_sha256"], "contract_file": B.ref(R / "CONTRACT.json"),
                      "commit": c["source_commit"], "delta": delta, "tc_images": len(ids)}))


def load_contract(round_dir):
    R = Path(round_dir).resolve()
    c = B.read_json(R / "CONTRACT.json")
    body = {k: v for k, v in c.items() if k != "contract_sha256"}
    B.require(c["contract_sha256"] == B.sha_bytes(B.canonical(body)) and c["version"] == VERSION, "contract digest")
    B.require(Path(c["source_root"]).resolve() == ROOT.resolve(), "not running from the bound release")
    B.require(Path(c["outputs"]).resolve() == R and c["dev"] == is_dev(R), "contract bound to another round")
    manifest = B.read_json(B.pinned(c["source_manifest"]))
    B.require(manifest["commit"] == c["source_commit"], "MANIFEST commit differs from the contract")
    B.release_intact(ROOT, manifest["files"])
    B.require(c["task"]["sha256"] == TASK[1] and c["definitions"] == DEFINITIONS and c["verdict_rule"] == C.VERDICT
              and c["widths_hex"] == hexes() and c["official"] == OFFICIAL, "contract differs from the release")
    return c


def stage_closed(c, name):
    path = Path(c["outputs"]) / name / "CLOSED.json"
    B.require(path.is_file(), f"stage not closed: {path}")
    closed = B.read_json(path)
    B.require(closed.get("status") == "PASS" and closed.get("contract_sha256") == c["contract_sha256"], f"stage {name} is not PASS")
    return closed


def p1_file(c, key):
    return np.load(B.pinned(c["phase1"]["files"][key]), allow_pickle=False)


# ================================================================ maps
def sep_nodes(maps, w):
    """(B, 12, 21) demand -> (gx (B, 51), gy (B, 31), identity, bad): the nodes of arm F's code at width w; bad marks the
    images whose nodes are not finite and strictly increasing (nothing raises, nothing is repaired)."""
    from sparse_rtdetr.research import r28_fovea_cal as FS
    torch = B.torch_mod()
    nodes, identity = FS.nodes_from_demand(torch.from_numpy(np.ascontiguousarray(maps)), w)
    gx, gy = nodes["x"][1].numpy(), nodes["y"][1].numpy()
    bad = ~(np.isfinite(gx).all(1) & np.isfinite(gy).all(1) & (np.diff(gx, axis=1) > 0).all(1) & (np.diff(gy, axis=1) > 0).all(1))
    return gx, gy, identity.numpy(), bad


def sep_closed_g(gx, gy, bad, image, norm):
    """sqrt(ox * oy) of r28_fovea.magnification per row (NaN on a bad image)."""
    from sparse_rtdetr.research import r28_fovea as FV
    torch = B.torch_mod()
    out = np.full(len(image), np.nan)
    ok = ~bad[image]
    order = np.flatnonzero(ok)
    for lo in range(0, len(order), 200000):
        idx = order[lo:lo + 200000]
        plan = FV.Plan(torch.from_numpy(gx[image[idx]]), torch.from_numpy(gy[image[idx]]), None, {})
        ox, oy = FV.magnification(plan, torch.from_numpy(norm[idx])[:, None, :])
        out[idx] = torch.sqrt(ox * oy)[:, 0].numpy()
    return out


def build(form, w, maps):
    """-> (G (B, 31, 51, 2), folded (B,), extra)."""
    if form == "SEP":
        gx, gy, identity, bad = sep_nodes(maps, w)
        G = C.sep_grid(gx, gy)
        B.require(bool((C.folded(G) == bad).all()), "the determinant test and the monotonicity test of a separable map disagree")
        return G, bad, {"gx": gx, "gy": gy, "identity_images": int(identity.sum())}
    raw, G = C.ns_nodes(C.node_saliency_2d(maps + C.EPS), w)
    over = np.abs(raw - np.clip(raw, 0, 1))
    return G, C.folded(G), {"clamp_max": float(over.max()), "clamped_interior_nodes": int((over[:, 1:-1, 1:-1] > 0).sum())}


_JOB = {}


def _work(task):
    """Worker (forked): box metrics of the rows task = (config key, lo, hi) of the measured order; optionally the
    image-level pixel counts of the images task[3]."""
    key, lo, hi, images = task
    j = _JOB[key]
    idx = j["order"][lo:hi]
    m = C.box_metrics(j["G"], j["image"][idx], j["norm"][idx])
    out = {k: m[k] for k in ("g", "stretch", "expand", "iou", "residual")}
    out["brute"] = m["brute"]
    level = None
    if images is not None:
        img = j["image"][idx]
        level = {}
        for i in images:
            sel = img == i
            level[int(i)] = C.image_level(j["G"][i], m["poly"][sel & j["small_scored"][idx]], m["poly"][sel])
    return key, lo, hi, out, level


def measure(configs, image, norm, small_scored=None, n_images=None, level=False):
    """configs: {key: (form, w, maps)}.  Returns {key: result} with per-row arrays (NaN on folded images), the fold
    flags, the SEP closed-form g and, with level=True, the per-image pixel counts (n_images, 5)."""
    import multiprocessing as mp
    res, tasks = {}, []
    _JOB.clear()
    for key, (form, w, maps) in configs.items():
        G, bad, extra = build(form, w, maps)
        ok = ~bad[image]
        order = np.flatnonzero(ok)
        r = {"form": form, "width": w, "folded": bad, "ok": ok, "extra": {k: v for k, v in extra.items() if k not in ("gx", "gy")},
             **{k: np.full(len(image), np.nan) for k in ("g", "stretch", "expand", "iou", "residual")}, "brute": 0, "G": G}
        if form == "SEP":
            r["g_closed"] = sep_closed_g(extra["gx"], extra["gy"], bad, image, norm)
        if level:
            r["level"] = np.zeros((n_images, 5), dtype=np.int64)
        res[key] = r
        _JOB[key] = {"G": G, "image": image, "norm": norm, "order": order, "small_scored": small_scored}
        if level:                                             # chunks follow image boundaries
            imgs = np.flatnonzero(~bad)
            for part in np.array_split(imgs, max(1, len(imgs) // 8)):
                lo, hi = np.searchsorted(image[order], [part[0], part[-1] + 1])
                tasks.append((key, int(lo), int(hi), part))
        else:
            tasks += [(key, lo, min(lo + CHUNK, len(order)), None) for lo in range(0, len(order), CHUNK)]
    with mp.get_context("fork").Pool(WORKERS) as pool:
        for key, lo, hi, out, lv in pool.imap_unordered(_work, tasks):
            r = res[key]
            idx = _JOB[key]["order"][lo:hi]
            for k in ("g", "stretch", "expand", "iou", "residual"):
                r[k][idx] = out[k]
            r["brute"] += out["brute"]
            if lv is not None:
                for i, v in lv.items():
                    r["level"][i] = v
    _JOB.clear()
    for key, r in res.items():
        worst = float(np.nanmax(r["residual"])) if r["ok"].any() else 0.0
        B.require(worst <= C.INVERSE_TOLERANCE, f"{key}: the numerical inverse leaves a residual of {worst:.3e} on an unfolded image")
        r["residual_max"] = worst
        if r["form"] == "SEP":
            d = float(np.nanmax(np.abs(r["g"] - r["g_closed"]))) if r["ok"].any() else 0.0
            io = float(np.nanmin(r["iou"])) if r["ok"].any() else 1.0
            B.require(d <= 1e-9 and io >= 1 - 1e-9, f"{key}: the general path differs from the separable closed form (g {d:.3e}, IoU {io})")
            r["closed_form_check"] = {"g_max_abs_difference": d, "round_trip_iou_min": io}
            r["g_polygon"], r["g"] = r["g"], r["g_closed"]
    return res


def summary(r):
    return {"form": r["form"], "width": r["width"], "width_hex": r["width"].hex(), "folded_images": int(r["folded"].sum()),
            "inverse_residual_max": r["residual_max"], "points_sent_to_exhaustive_search": int(r["brute"]),
            **({"closed_form_check": r["closed_form_check"]} if r["form"] == "SEP" else {}), **r["extra"]}


# ================================================================ wire (CPU)
def _e3_replica(d_star, norm, image, small):
    """The collateral reading of phase 1 (th2_phase1.e3), statement for statement."""
    from sparse_rtdetr.research import r28_fovea as FV
    from sparse_rtdetr.research import r28_fovea_geo as FG
    torch = B.torch_mod()
    plan = FG.plan_from_demand(torch.from_numpy(d_star))
    H, Wd = C.L_HW
    tot, empty = 0.0, 0.0
    bounds = np.searchsorted(image, np.arange(len(d_star) + 1))
    for i in range(len(d_star)):
        gx, gy = plan.gx[i].numpy(), plan.gy[i].numpy()
        ox = 1.0 / (np.diff(gx) * (len(gx) - 1))[np.minimum((np.arange(Wd) + 0.5) / Wd * (len(gx) - 1), len(gx) - 2).astype(int)]
        oy = 1.0 / (np.diff(gy) * (len(gy) - 1))[np.minimum((np.arange(H) + 0.5) / H * (len(gy) - 1), len(gy) - 2).astype(int)]
        hot = np.sqrt(oy[:, None] * ox[None, :]) >= 1.2
        mask = np.zeros((H, Wd), dtype=bool)
        s = slice(bounds[i], bounds[i + 1])
        b = norm[s][small[s]]
        if len(b):
            t = torch.from_numpy(b)
            x1, x2 = (FV.warp(plan.gx[i:i + 1], t[None, :, k])[0].numpy() * Wd for k in (0, 2))
            y1, y2 = (FV.warp(plan.gy[i:i + 1], t[None, :, k])[0].numpy() * H for k in (1, 3))
            for a_, b_, c_, d_ in zip(x1, y1, x2, y2):
                mask[max(0, int(np.floor(b_ - 8))):int(np.ceil(d_ + 8)), max(0, int(np.floor(a_ - 8))):int(np.ceil(c_ + 8))] = True
        tot += hot.sum()
        empty += (hot & ~mask).sum()
    return {"pooled_share": float(empty / tot), "share_of_canvas_ge_1.2": float(tot / (len(d_star) * H * Wd))}


def dev_table(c):
    z = p1_file(c, "geom/gt")
    z = {k: z[k] for k in ("image", "norm", "area", "scored", "g0", "g_b")}
    sc = z["scored"]
    B.require(len(sc) == 38759 and int(sc.sum()) == N_GT and int((z["area"][sc] <= 32.0 ** 2).sum()) == N_SMALL
              and bool((np.diff(z["image"]) >= 0).all()), "phase-1 GT table")
    z["size"] = C.size_bin(z["area"])
    maps = {"P": p1_file(c, "demand/demand"), "Lb": p1_file(c, "geom/d_star")}
    B.require(all(m.shape == (N_DEV, 12, 21) and m.dtype == np.float64 for m in maps.values()), "phase-1 demand maps")
    return z, maps


def wire(c, out, counts):
    from sparse_rtdetr.research import r28_fovea as FV
    from sparse_rtdetr.research import r28_fovea_geo as FG
    torch = B.torch_mod()
    B.PHASE["phase"] = "wire/closed form"
    z, maps = dev_table(c)
    sc, small = z["scored"], z["size"] <= 1
    g_file = p1_file(c, "geom/g")[:, 0]
    checks, closed = {}, {}
    for src, stored in (("P", z["g0"]), ("Lb", z["g_b"])):
        gx, gy, _, bad = sep_nodes(maps[src], C.FG_WIDTH)
        plan = FG.plan_from_demand(torch.from_numpy(maps[src]))               # the phase-1 call; raises on an invalid map
        B.require(not bad.any() and bool(np.array_equal(plan.gx.numpy(), gx)) and bool(np.array_equal(plan.gy.numpy(), gy)),
                  "sep_nodes is not the F_geo plan")
        g = sep_closed_g(gx, gy, bad, z["image"], z["norm"])
        closed[src] = g
        v = g[sc & small]
        mine = {"small_g_geomean": float(np.exp(np.log(v).mean())), "small_share_g_lt_1.2": float((v < 1.2).mean())}
        checks[src] = {"mine": mine, "phase1": WIRE[src], "per_gt_max_abs_difference_to_gt_npz": float(np.abs(g - stored).max()),
                       "n_small": int(len(v))}
        B.require(all(abs(mine[k] - WIRE[src][k]) <= WIRE_TOL for k in mine) and checks[src]["per_gt_max_abs_difference_to_gt_npz"] == 0.0,
                  f"wiring check failed for source {src}: {checks[src]}")
    d = float(np.abs(closed["P"][sc] - g_file).max())
    B.require(d == 0.0, f"per-GT magnification differs from phase-1 geom/g.npy: {d}")
    checks["P"]["per_gt_max_abs_difference_to_g_npy"] = d
    B.PHASE["phase"] = "wire/collateral"
    rep = _e3_replica(maps["Lb"], z["norm"][sc], z["image"][sc], small[sc])
    checks["Lb_collateral"] = {"mine": rep, "phase1": WIRE["Lb_collateral"]}
    B.require(all(abs(rep[k] - WIRE["Lb_collateral"][k]) <= WIRE_TOL for k in rep), f"collateral wiring check failed: {checks['Lb_collateral']}")
    B.PHASE["phase"] = "wire/general path"
    res = measure({"SEP|Lb": ("SEP", C.FG_WIDTH, maps["Lb"])}, z["image"], z["norm"], small_scored=small & sc, n_images=N_DEV, level=True)["SEP|Lb"]
    lv = res["level"].sum(0)
    general = {"pooled_share": float(lv[2] / lv[1]), "share_of_canvas_ge_1.2": float(lv[1] / lv[0])}
    checks["Lb_collateral"]["general_path"] = general
    checks["Lb_collateral"]["general_minus_phase1"] = {k: general[k] - WIRE["Lb_collateral"][k] for k in general}
    B.require(all(abs(v) <= 2e-3 for v in checks["Lb_collateral"]["general_minus_phase1"].values()),
              f"the general image-level path is not the phase-1 reading: {checks['Lb_collateral']}")
    checks["general_path_SEP_Lb"] = summary(res)
    B.require(counts["forwards"] == 0, "wire must not run a model")
    return {"stage": "wire", "passed": True, "checks": checks, "NS_maps_computed_on_development": 0}


# ================================================================ demand_tc (CUDA)
def tc_data(c):
    from sparse_rtdetr.research import scale_baseline as S
    data = S.train_dataset(c["data"]["train_core"], S.S15_HW, seed=0)
    B.require(len(data.records) == N_TC, "train_core members")
    by = {r["stable_image_id"]: r for r in data.records}
    return data, [by[i] for i in c["tc_ids"]]


def _decode(data, row):
    from PIL import Image
    from torchvision.transforms.v2 import functional as TF
    image_bytes, _ = data.bound_bytes(row)
    with Image.open(io.BytesIO(image_bytes)) as im:
        return TF.pil_to_tensor(data.transform.resize(im.convert("RGB")))


def demand_device():
    d = os.environ.get("TH2B0_DEMAND_DEVICE", "cuda")
    B.require(d in ("cuda", "cpu"), "TH2B0_DEMAND_DEVICE")
    return d


def demand_tc(c, out, counts):
    from concurrent.futures import ThreadPoolExecutor
    from sparse_rtdetr.research import r28_resample as RS
    B.PHASE["phase"] = "demand_tc/load"
    torch = B.torch_mod()
    prior = os.environ.get("TH2B0_DEMAND_ARCHIVE")
    if prior:                                   # task section 2: an archive of the same convention is not recomputed
        pc = B.read_json(Path(prior) / "CONTRACT.json")
        closed = B.read_json(Path(prior) / "demand_tc/CLOSED.json")
        B.require(closed["status"] == "PASS" and closed["contract_sha256"] == pc["contract_sha256"] and pc["tc_ids"] == c["tc_ids"]
                  and pc["demand"] == c["demand"] and pc["dev"] is False and closed["images"] == len(c["tc_ids"]), "prior demand archive")
        mine, theirs = B.read_json(c["source_manifest"]["path"])["files"], B.read_json(pc["source_manifest"]["path"])["files"]
        same = ("src/sparse_rtdetr/research/r28_resample.py", "src/sparse_rtdetr/research/r28_warp.py", "tools/r28/r28_demand.py",
                "src/sparse_rtdetr/research/scale_baseline.py")
        B.require(all(mine[f] == theirs[f] for f in same), "the demand code differs from the archive's release")
        d = np.load(B.pinned(closed["files"]["demand"]), allow_pickle=False)
        B.require(d.shape == (len(c["tc_ids"]), 12, 21) and d.dtype == np.float64, "prior demand maps")
        keep = ("images", "batch", "device", "device_note", "gpu_forward_seconds", "forward_seconds", "stage_wall_seconds",
                "demand_min_max_mean", "other_gpu_compute_processes_at_start", "other_gpu_compute_processes_at_end", "peak_cuda_bytes")
        counts.update(demand_forwards=0, demand_images=0)
        return {"stage": "demand_tc", "files": {"demand": X.save_npy(out / "demand_tc.npy", d)}, "each_image_forwarded_once": True,
                "adopted_from": closed["files"]["demand"], "adopted_stage_counts": closed["counts"], "forwards_in_this_stage": 0,
                **{k: closed[k] for k in keep}}
    on_gpu = demand_device() == "cuda"
    B.require(not on_gpu or X.nvidia("gpu=uuid") == [c["gpu_uuid"]], "GPU differs from the contract")
    device = torch.device("cuda:0" if on_gpu else "cpu")
    others = X.other_gpu_processes() if on_gpu else X.nvidia("compute-apps=pid,process_name,used_memory")
    ctx = RS.Context({"kind": "network", "file": c["demand"]["weights"]}).to(device)
    B.require(ctx.binding() == c["demand"]["binding"], "demand network binding")
    data, rows = tc_data(c)
    maps, seconds, t0 = [], 0.0, time.monotonic()
    B.PHASE["phase"] = "demand_tc/forward"
    with ThreadPoolExecutor(8) as pool, torch.inference_mode():
        for off in range(0, len(rows), 64):
            images = list(pool.map(lambda r: _decode(data, r), rows[off:off + 64]))
            for b in range(0, len(images), X.BATCH):
                t1 = time.monotonic()
                x = (torch.stack(images[b:b + X.BATCH]).float() / 255.).to(device)
                B.require(tuple(x.shape[1:]) == (3, 1152, 2016) and x.dtype == torch.float32, "the input is not the FP32 H canvas")
                with torch.autocast(device.type, enabled=False), torch.no_grad():
                    d = ctx.demand(x)
                if on_gpu:
                    torch.cuda.synchronize()
                seconds += time.monotonic() - t1
                counts.update(demand_forwards=1, demand_images=len(x))
                maps.append(d.clone())
    d = torch.cat(maps).numpy()
    B.require(d.shape == (len(rows), 12, 21) and d.dtype == np.float64 and bool(np.isfinite(d).all()) and bool((d >= 0).all())
              and counts["demand_images"] == len(rows), "demand maps")
    B.require(ctx.binding() == c["demand"]["binding"], "the forwards changed the demand network")
    B.require(counts["forwards"] == 0, "no detector forward")
    return {"stage": "demand_tc", "files": {"demand": X.save_npy(out / "demand_tc.npy", d)}, "images": len(rows),
            "each_image_forwarded_once": True, "batch": X.BATCH, "device": device.type,
            "device_note": "phase 1 ran this network on CUDA; on the CPU the FP32 result may differ in the last bits",
            "gpu_forward_seconds": seconds if on_gpu else 0.0, "forward_seconds": seconds, "stage_wall_seconds": time.monotonic() - t0,
            "demand_min_max_mean": [float(d.min()), float(d.max()), float(d.mean())],
            "other_gpu_compute_processes_at_start": others,
            "other_gpu_compute_processes_at_end": X.other_gpu_processes() if on_gpu else X.nvidia("compute-apps=pid,process_name,used_memory"),
            "peak_cuda_bytes": torch.cuda.max_memory_allocated() if on_gpu else 0}


# ================================================================ tc (CPU)
def tc_table(c):
    """GT table of the round's train_core images (labels only) and the label demand."""
    import r28_demand as DM
    from sparse_rtdetr.r0.evaluation import bridge_inputs
    from sparse_rtdetr.research import r28_warp as W
    data, rows = tc_data(c)
    image, norm, area, scored, d_star, raw_ids = [], [], [], [], [], []
    dropped = 0
    for i, row in enumerate(rows):
        p = DM._annotation(data, row)
        nb, keep = DM.deployment_xyxy(p, row["width"], row["height"])
        d_star.append(W.demand_from_boxes(nb).numpy())
        objs = [o for o, k in zip(p["gt"], keep) if k]
        if objs:
            usable = {**p, "rows": [o for o in p["rows"] if o["role"] != "nonpositive_area"]}
            dropped += len(p["rows"]) - len(usable["rows"])
            dataset, _, _ = bridge_inputs([row], [usable], [])
            ids = {a["raw_annotation_id"] for a in dataset["annotations"] if not a["iscrowd"]}
            image += [i] * len(objs)
            norm.append(nb.numpy().reshape(-1, 4))
            area += [float(o["raw_area"]) for o in objs]
            scored += [o["id"] in ids for o in objs]
            raw_ids += [o["id"] for o in objs]
    z = {"image": np.array(image, dtype=np.int64), "norm": np.concatenate(norm), "area": np.array(area), "scored": np.array(scored)}
    z["size"] = C.size_bin(z["area"])
    return z, np.stack(d_star), raw_ids, dropped


def tc(c, out, counts):
    B.PHASE["phase"] = "tc/table"
    z, d_star, raw_ids, dropped = tc_table(c)
    sc = z["scored"]
    maps = {"P": np.load(B.pinned(stage_closed(c, "demand_tc")["files"]["demand"]), allow_pickle=False), "Lb": d_star}
    B.require(maps["P"].shape == maps["Lb"].shape == (len(c["tc_ids"]), 12, 21), "train_core demand maps")
    image, norm, size = z["image"][sc], z["norm"][sc], z["size"][sc]
    t1, arrays, t4 = [], {}, {}
    for src in ("P", "Lb"):
        t4[src] = C.describe(C.nonseparability(maps[src] + C.EPS))
    B.PHASE["phase"] = "tc/measure"
    for src in ("P", "Lb"):                                    # 18 configurations at a time
        configs = {f"{form}|{w.hex()}|{src}": (form, w, maps[src]) for form in ("SEP", "NS") for w in C.WIDTHS}
        res = measure(configs, image, norm)
        for key, r in res.items():
            m = {k: r[k] for k in ("g", "stretch", "expand", "iou")}
            t1.append({"key": key, "source": src, **summary(r), "gt_measured": int(r["ok"].sum()), "gt_not_measured_folded": int((~r["ok"]).sum()),
                       **C.row(size, m, r["ok"])})
            arrays[f"{key}|g"] = r["g"]
            for k in ("stretch", "expand", "iou"):
                arrays[f"{key}|{k}"] = r[k].astype(np.float32)
        del res
        gc.collect()
    np.savez(out / "metrics.npz", **arrays)
    np.savez(out / "gt.npz", **z, d_star=d_star)
    B.write_new(out / "raw_ids.json", raw_ids)
    B.write_new(out / "T1.json", {"rows": t1, "nonseparability": t4, "gt": {"deployment": int(len(sc)), "scored": int(sc.sum()),
                                                                           "small": int((size <= 1).sum()), "medium": int(((size == 2) | (size == 3)).sum()),
                                                                           "large96": int((size == 4).sum())}})
    B.require(counts["forwards"] == 0, "tc must not run a model")
    return {"stage": "tc", "files": {"metrics": B.ref(out / "metrics.npz"), "gt": B.ref(out / "gt.npz"), "raw_ids": B.ref(out / "raw_ids.json")},
            "result": B.ref(out / "T1.json"), "images": len(c["tc_ids"]),
            "annotation_rows_of_non_positive_extent_left_out_of_the_filter": dropped, "rows": len(t1), "development_read": False}


# ================================================================ select (CPU)
def t1_rows(c):
    return B.read_json(B.pinned(stage_closed(c, "tc")["result"]))


def select(c, out, counts):
    B.PHASE["phase"] = "select"
    rows = {r["key"]: r for r in t1_rows(c)["rows"]}
    ref = rows[f"SEP|{C.FG_WIDTH.hex()}|P"]
    B.require(ref["folded_images"] == 0, "SEP(19.03125) folds on train_core")
    ns = {w.hex(): rows[f"NS|{w.hex()}|P"] for w in C.WIDTHS}
    sel = C.select(ns, ref["medium_mean_logg"])
    for h, r in sel["removed"].items():
        r["readings"] = {k: ns[h][k] for k in ("folded_images", "medium_mean_logg", "small_share_iou_ge_0.90", "small_mean_logg")}
    doc = {**sel, "candidates": {h: {k: ns[h][k] for k in ("folded_images", "medium_mean_logg", "small_share_iou_ge_0.90", "small_mean_logg")} for h in ns},
           "SEP_reference": {k: ref[k] for k in ("small_mean_logg", "medium_mean_logg")},
           "contract_sha256": c["contract_sha256"], "T1": stage_closed(c, "tc")["result"], "written_at": now(), "written_at_unix": time.time(),
           "development_NS_readings_before_this_file": 0, "dev_round_numbers_are_not_results": c["dev"]}
    B.write_new(out / "SELECTION.json", doc)
    return {"stage": "select", "selection": B.ref(out / "SELECTION.json"), "w_star": sel["w_star"], "w_star_hex": sel["w_star_hex"],
            "removed": {h: r["first_reason"] for h, r in sel["removed"].items()}, "written_at": doc["written_at"]}


def selection(c):
    closed = stage_closed(c, "select")
    return B.read_json(B.pinned(closed["selection"]))


def mappings(c, sel):
    """The four mappings of T2 as (name, form, width); in a dev round every NS is replaced by a separable stand-in."""
    w = sel["w_star"] if sel["w_star"] is not None else C.FG_WIDTH
    out = [("SEP(19.03125)", "SEP", C.FG_WIDTH), ("SEP(4)", "SEP", C.F_WIDTH), ("NS(w*)", "NS", w), ("NS(19.03125)", "NS", C.FG_WIDTH)]
    if c["dev"]:
        taken = {C.FG_WIDTH, C.F_WIDTH}
        swap = []
        for name, form, width in out:
            if form == "NS":
                width = width if width not in taken else next(x for x in (13.0, 8.0, 24.0) if x not in taken)
                taken.add(width)
                form = "SEP"
            swap.append((name, form, width))
        out = swap
    return out, w


# ================================================================ dev (CPU)
def dev(c, out, counts):
    B.PHASE["phase"] = "dev/start"
    sel = selection(c)
    maps4, w_star = mappings(c, sel)
    B.require(c["dev"] or sel["written_at_unix"] < time.time(), "SELECTION.json is not older than this stage")
    start = {"first_development_NS_computation_started_at": now(), "unix": time.time(), "SELECTION_written_at": sel["written_at"],
             "SELECTION": stage_closed(c, "select")["selection"], "NS_replaced_by_separable_stand_ins": c["dev"]}
    B.write_new(out / "DEV_NS_START.json", start)
    z, maps = dev_table(c)
    sc = z["scored"]
    small_sc = (z["size"] <= 1) & sc
    w = C.boot_weights(N_DEV)
    B.PHASE["phase"] = "dev/measure"
    res, same = {}, {}
    for src in ("P", "Lb"):
        configs = {}
        for name, form, width in maps4:
            k = (form, width)
            if k in same:
                continue
            same[k] = name
            configs[f"{name}|{src}"] = (form, width, maps[src])
        same.clear()
        res.update(measure(configs, z["image"], z["norm"], small_scored=small_sc, n_images=N_DEV, level=True))
    alias = {}
    for name, form, width in maps4:                            # NS(w*) may be NS(19.03125)
        first = next(n for n, f, x in maps4 if (f, x) == (form, width))
        alias[name] = first
    B.PHASE["phase"] = "dev/T2"
    image, size = z["image"][sc], z["size"][sc]
    t2 = []
    ar = np.arange(N_DEV)
    for name, form, width in maps4:
        for src in ("P", "Lb"):
            r = res[f"{alias[name]}|{src}"]
            m = {k: r[k][sc] for k in ("g", "stretch", "expand", "iou")}
            ok = r["ok"][sc]
            lv = r["level"].astype(np.float64)
            t2.append({"mapping": name, "source": src, "same_map_as": alias[name], **summary(r), "form_computed": form,
                       "gt_measured": int(ok.sum()), **C.row_ci(size, m, ok, image, N_DEV, w),
                       "hot_share_of_canvas": C.ratio_ci(lv[:, 1], lv[:, 0], ar, N_DEV, w),
                       "hot_share_outside_small_gt": C.ratio_ci(lv[:, 2], lv[:, 1], ar, N_DEV, w),
                       "cold_share_of_canvas": C.ratio_ci(lv[:, 3], lv[:, 0], ar, N_DEV, w),
                       "cold_share_outside_any_gt": C.ratio_ci(lv[:, 4], lv[:, 3], ar, N_DEV, w)})
    B.PHASE["phase"] = "dev/T3"
    a, b = res["SEP(19.03125)|P"], res[f"{alias['NS(w*)']}|P"]
    ok = (a["ok"] & b["ok"])[sc]
    ga, gb = a["g"][sc], b["g"][sc]
    B.require(float(np.abs(ga - p1_file(c, "geom/g")[:, 0]).max()) == 0.0, "SEP(19.03125) g differs from the phase-1 per-GT file")
    small = size <= 1
    with np.errstate(invalid="ignore", divide="ignore"):
        dl = np.log(gb) - np.log(ga)
    cross = np.zeros((4, 4), dtype=np.int64)
    np.add.at(cross, (C.g_bin(ga[small & ok]), C.g_bin(gb[small & ok])), 1)
    o = p1_file(c, "e0/outcomes")
    hits = {}
    for s in SEEDS:
        s15, fg = o[f"iou_S15-s{s}"] >= 0.75, o[f"iou_FG-s{s}"] >= 0.75
        rows = {}
        for gname, m in (("S15_only", s15 & ~fg), ("FG_only", fg & ~s15), ("both", s15 & fg), ("neither", ~s15 & ~fg)):
            m = m & small & ok
            rows[gname] = {"n": int(m.sum()), "g_geomean_SEP": float(np.exp(np.log(ga[m]).mean())) if m.any() else None,
                           "g_geomean_NS": float(np.exp(np.log(gb[m]).mean())) if m.any() else None,
                           "mean_dlogg": float(dl[m].mean()) if m.any() else None,
                           "share_dlogg_ge_0.10": float((dl[m] >= C.DLOGG).mean()) if m.any() else None}
        hits[f"s{s}"] = rows
    reg = np.zeros((3, 5), dtype=np.int64)
    code = np.where(dl >= C.DLOGG, 0, np.where(dl <= -C.DLOGG, 2, 1))
    np.add.at(reg, (code[ok], size[ok]), 1)
    t3 = {"compared": ["SEP(19.03125)", "NS(w*)"], "source": "P", "cross_table_small": {"rows_SEP_bins": list(C.G_NAMES), "cols_NS_bins": list(C.G_NAMES),
                                                                                      "counts": cross.tolist(), "n": int(cross.sum())},
          "hit75_groups_small": hits,
          "registration_dlogg_by_size": {"rows": ["dlogg>=+0.10", "between", "dlogg<=-0.10"], "cols": list(C.SIZE_NAMES), "counts": reg.tolist(),
                                         "gt_not_measured": int((~ok).sum())}}
    B.PHASE["phase"] = "dev/files"
    np.savez(out / "per_gt.npz", size_bin=size.astype(np.int8), image=image.astype(np.int16), g_SEP=ga, g_NS=gb, dlogg=dl,
             stretch_SEP=a["stretch"][sc], stretch_NS=b["stretch"][sc], iou_SEP=a["iou"][sc], iou_NS=b["iou"][sc], measured=ok)
    raw_ids = B.read_json(P1 / "geom/raw_ids.json")
    B.require(len(raw_ids) == N_GT, "phase-1 raw ids")
    f = lambda v: None if not np.isfinite(v) else float(v)
    with (out / "per_gt.tsv").open("x") as fh:
        fh.write("raw_annotation_id\timage\tsize_bin\tg_SEP\tg_NS\tdlogg\tstretch_SEP\tstretch_NS\tiou_SEP\tiou_NS\n")
        for i in range(N_GT):
            vals = [ga[i], gb[i], dl[i], a["stretch"][sc][i], b["stretch"][sc][i], a["iou"][sc][i], b["iou"][sc][i]]
            fh.write(f"{raw_ids[i]}\t{c['development_ids'][image[i]]}\t{int(size[i])}\t" + "\t".join("" if f(v) is None else repr(f(v)) for v in vals) + "\n")
    t4 = {src: C.describe(C.nonseparability(maps[src] + C.EPS)) for src in ("P", "Lb")}
    B.write_new(out / "DEV.json", {"T2": t2, "T3": t3, "T4_development": t4, "w_star": sel["w_star"], "w_used_for_NS(w*)": w_star,
                                   "mappings": [{"name": n, "form": fo, "width": x} for n, fo, x in maps4], "start": start,
                                   "dev_round_numbers_are_not_results": c["dev"]})
    B.require(counts["forwards"] == 0, "dev must not run a model")
    return {"stage": "dev", "result": B.ref(out / "DEV.json"), "files": {"per_gt": B.ref(out / "per_gt.npz"), "per_gt_tsv": B.ref(out / "per_gt.tsv")},
            "start": B.ref(out / "DEV_NS_START.json"), "first_development_NS_computation_started_at": start["first_development_NS_computation_started_at"]}


# ================================================================ figs (CPU)
def _render(img, Gi, hw):
    """Plain bilinear rendering of the canvas (illustration only)."""
    h, w = hw
    vv, uu = np.meshgrid((np.arange(h) + 0.5) / h, (np.arange(w) + 0.5) / w, indexing="ij")
    q = C.ev(Gi[None], np.zeros(h * w, dtype=np.int64), uu.reshape(-1), vv.reshape(-1))
    sh, sw = img.shape[:2]
    x, y = np.clip(q[0] * sw - 0.5, 0, sw - 1), np.clip(q[1] * sh - 0.5, 0, sh - 1)
    x0, y0 = np.minimum(x.astype(int), sw - 2), np.minimum(y.astype(int), sh - 2)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    f = img.astype(np.float64)
    o = f[y0, x0] * (1 - fx) * (1 - fy) + f[y0, x0 + 1] * fx * (1 - fy) + f[y0 + 1, x0] * (1 - fx) * fy + f[y0 + 1, x0 + 1] * fx * fy
    return np.clip(o + 0.5, 0, 255).astype(np.uint8).reshape(h, w, 3)


def figs(c, out, counts):
    from PIL import Image, ImageDraw
    B.PHASE["phase"] = "figs"
    sel = selection(c)
    w_star = sel["w_star"] if sel["w_star"] is not None else C.FG_WIDTH
    data, rows = tc_data(c)
    pos = {i: n for n, i in enumerate(c["tc_ids"])}
    pick = [pos[i] for i in c["figure_ids"]]
    z = np.load(B.pinned(stage_closed(c, "tc")["files"]["gt"]), allow_pickle=False)
    P = np.load(B.pinned(stage_closed(c, "demand_tc")["files"]["demand"]), allow_pickle=False)[pick]
    hw = (384, 672)
    files, notes = {}, []
    for n, i in enumerate(pick):
        image_bytes, _ = data.bound_bytes(rows[i])
        with Image.open(io.BytesIO(image_bytes)) as im:
            src = np.asarray(im.convert("RGB"))
        boxes = z["norm"][z["image"] == i]
        small = z["size"][z["image"] == i] <= 1
        panels = []
        for form, width in (("SEP", C.FG_WIDTH), ("NS", w_star)):
            G, bad, _ = build(form, width, P[n:n + 1])
            canvas = Image.fromarray(_render(src, G[0], hw))
            if not bad[0] and len(boxes):
                poly = C.box_metrics(G, np.zeros(len(boxes), dtype=np.int64), boxes)["poly"]
                draw = ImageDraw.Draw(canvas)
                for p, s in zip(poly, small):
                    pts = [(float(a * hw[1]), float(b * hw[0])) for a, b in p]
                    draw.line(pts + pts[:1], fill=(255, 64, 64) if s else (64, 200, 255), width=1)
            panels.append(canvas)
            notes.append({"figure": n, "image": c["figure_ids"][n], "form": form, "width": width, "folded": bool(bad[0]), "boxes": int(len(boxes))})
        sheet = Image.new("RGB", (hw[1] * 2 + 8, hw[0]), (255, 255, 255))
        sheet.paste(panels[0], (0, 0))
        sheet.paste(panels[1], (hw[1] + 8, 0))
        path = out / f"fig-{n:02d}.jpg"
        sheet.save(path, quality=88)
        files[path.name] = B.ref(path)
    B.require(len(files) <= FIGURES and counts["forwards"] == 0, "figures")
    return {"stage": "figs", "files": files, "panels": notes, "left": "SEP(19.03125)", "right": f"NS({w_star!r})", "source": "P",
            "colours": "red = small GT, blue = other GT", "train_core_images_read": len(pick)}


# ================================================================ tables (CPU)
def tables(c, out, counts):
    B.PHASE["phase"] = "tables"
    sel = selection(c)
    t1 = t1_rows(c)
    d = B.read_json(B.pinned(stage_closed(c, "dev")["result"]))
    t2 = {(r["mapping"], r["source"]): r for r in d["T2"]}
    ns, sep = t2[("NS(w*)", "P")], t2[("SEP(19.03125)", "P")]
    w_used = d["w_used_for_NS(w*)"]
    tc_rows = {r["key"]: r for r in t1["rows"]}
    folds = {"TC_P": tc_rows[f"NS|{w_used.hex()}|P"]["folded_images"], "TC_Lb": tc_rows[f"NS|{w_used.hex()}|Lb"]["folded_images"],
             "DEV_P": ns["folded_images"], "DEV_Lb": t2[("NS(w*)", "Lb")]["folded_images"]}
    g1 = 100.0 * (sep["small_share_g_lt_1.2"]["value"] - ns["small_share_g_lt_1.2"]["value"])
    g2 = 100.0 * (ns["medium_share_g_lt_1"]["value"] - sep["medium_share_g_lt_1"]["value"])
    g3 = ns["small_share_iou_ge_0.90"]["value"]
    readings = {"selected": sel["w_star"] is not None, "G1_pp": g1, "G2_pp": g2, "G3_share": g3, "folds": folds}
    verdict = {**C.verdict(readings), "readings": readings, "w_star": sel["w_star"], "w_used_for_NS(w*)": w_used,
               "dev_round_numbers_are_not_results": c["dev"]}
    stages = {}
    for name in ("wire", "demand_tc", "tc", "select", "dev", "figs"):
        st = stage_closed(c, name)
        stages[name] = {"elapsed_seconds": st["elapsed_seconds"], "process_cpu_seconds": st["process_cpu_seconds"], "counts": st["counts"]}
    dm = stage_closed(c, "demand_tc")
    used = sum(p.stat().st_size for p in Path(c["outputs"]).rglob("*") if p.is_file() and not p.is_symlink())
    result = {"status": "PASS", "version": VERSION, "contract_sha256": c["contract_sha256"], "source_commit": c["source_commit"],
              "source_branch": c["source_branch"], "task": c["task"], "dev": c["dev"], "verdict": verdict, "selection": sel,
              "T1": t1["rows"], "T2": d["T2"], "T3": d["T3"],
              "T4": {"train_core": t1["nonseparability"], "development": d["T4_development"]}, "gt_counts_train_core": t1["gt"],
              "wire": stage_closed(c, "wire")["checks"], "official": c["official"],
              "timing": {"SELECTION_written_at": sel["written_at"],
                         "first_development_NS_computation_started_at": d["start"]["first_development_NS_computation_started_at"]},
              "workload": {"stages": stages, "cpu_wall_seconds": sum(v["elapsed_seconds"] for k, v in stages.items() if k != "demand_tc" or dm["device"] == "cpu"),
                           "demand_network_images_forwarded": dm["images"], "demand_network_forward_calls": dm.get("adopted_stage_counts", dm["counts"])["demand_forwards"],
                           "demand_archive_adopted_from": dm.get("adopted_from"),
                           "gpu_forward_seconds": dm["gpu_forward_seconds"], "demand_device": dm["device"],
                           "demand_forward_seconds": dm["forward_seconds"], "demand_stage_wall_seconds": dm["stage_wall_seconds"],
                           "detector_forwards": 0, "updates": 0, "scoring_calls": 0,
                           "other_gpu_compute_processes_seen": sorted(set(dm["other_gpu_compute_processes_at_start"] + dm["other_gpu_compute_processes_at_end"])),
                           "bytes_in_round_directory_before_this_file": used},
              "figures": stage_closed(c, "figs")["files"], "definitions": c["definitions"]}
    B.write_new(out / "RESULT.json", result)
    return {"stage": "tables", "result": B.ref(out / "RESULT.json"), "label": verdict["label"], "readings": readings}


# ================================================================ CLI
STAGES = {"wire": (wire, "cpu"), "demand_tc": (demand_tc, "cuda"), "tc": (tc, "cpu"), "select": (select, "cpu"), "dev": (dev, "cpu"),
          "figs": (figs, "cpu"), "tables": (tables, "cpu")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("prepare",) + tuple(STAGES))
    ap.add_argument("--round", required=True, type=Path)
    ap.add_argument("--release")
    args = ap.parse_args(argv)
    if args.action == "prepare":
        B.require(args.release, "prepare needs --release")
        return prepare(args.round, args.release)
    run, mode = STAGES[args.action]
    if args.action == "demand_tc":
        mode = demand_device()
    B.require(args.release is None, "--release is for prepare")
    c = load_contract(args.round)
    out = Path(c["outputs"]) / args.action
    out.mkdir(exist_ok=False)
    start, cpu0 = time.monotonic(), time.process_time()
    counts = Counter(actual_updates=0, forwards=0, forward_attempts=0)
    try:
        from r25_run import setup
        if mode == "cuda":
            B.require(os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == "expandable_segments:True", "allocator conf")
            from s15_formal_train import bind_seed
            bind_seed(0)
            setup("cuda:0")
            B.torch_mod().cuda.reset_peak_memory_stats()
        else:
            B.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", f"the {args.action} stage requires CUDA_VISIBLE_DEVICES=\"\"")
            setup("cpu")
        result = run(c, out, counts)
        B.require(isinstance(result, dict) and counts["actual_updates"] == 0 and counts["forwards"] == 0, f"{args.action}: stage result")
        children = os.times()
        B.write_new(out / "CLOSED.json", {"status": "PASS", "action": args.action, "contract_sha256": c["contract_sha256"],
                                          "counts": dict(counts), "started_at_unix": time.time() - (time.monotonic() - start),
                                          "elapsed_seconds": time.monotonic() - start,
                                          "process_cpu_seconds": time.process_time() - cpu0 + children.children_user + children.children_system,
                                          "closed_at": now(), **result})
    except BaseException as exc:
        B.write_new(out / "STOP.json", {"status": "STOP", "phase": B.PHASE["phase"], "action": args.action,
                                        "error": repr(exc)[:4000], "traceback": traceback.format_exc(),
                                        "counts": dict(counts), "elapsed_seconds": time.monotonic() - start})
        raise
    finally:
        gc.collect()


if __name__ == "__main__":
    main()
