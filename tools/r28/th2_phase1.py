"""TH2 phase 1 (TASK_TH2_PHASE1_001): intervention measurement of sampling value on the frozen F_geo endpoints.
Inference, matching, CPU statistics and geometry only: no update, no BN re-estimation, no demand-network training,
no test-dev, no image outside the 548 development images.

usage: th2_phase1.py <action> --round <round> [--release <name>] [--unit <unit>]

  prepare  contract: release, pinned endpoints / predictions, folds, perturbation rule, slope models, verdict rule.
  wire     CUDA, per F_geo unit: the first 8 development images must reproduce the stored top600 records (r31 entry).
  demand   CUDA: the frozen demand network's map of every development image, recorded from the F_geo seed-0
           evaluation forward itself; the 548 x 600 records of that forward must equal the stored file.
  geom     CPU: GT table and features, magnification under the predicted and the label demand (checked against the
           R30-001 per-GT file), amplitude choice on fold A (geometry only), every perturbed map and its magnification.
  e0       CPU: per-GT outcomes of the stored predictions; noise floor, S15 response, table 15 on three seeds.
  e1       CUDA, per seed: one unperturbed and K perturbed inferences; the demand context replays the recorded map
           times the perturbation.  The unperturbed pass must equal the stored predictions record for record.
  r14      CPU: R1-R4, the slope models, the value maps (c), (d) and their geometry.
  e2       CUDA, per unit: one inference with demand source b / c / d.
  score    CPU, per unit: the frozen scorer on the predictions of e2.
  e3       CPU: the two geometry readings for the alternative direction.
  tables   CPU: E2 readings, predicted net change, verdict, RESULT.json.

Every stage writes <round>/<name>/CLOSED.json (PASS) or STOP.json.  A dev round (dev-round* inside the round, with
TH2_DEV_K perturbations) is an engineering shake-out; its numbers are not results.
"""
import argparse
import gc
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
import th2_core as T  # noqa: E402

VERSION = "TH2-PHASE1-001-v1"
TASK = ("TASK_TH2_PHASE1_001.md", "89e85810d20d6a68bb8c9745f280258d64799287e66e480706d273a0bbbcc3a4")
BASE_COMMIT = "3c491df1a4765d9826f69e247cb295ecb00dc04c"          # R31 release (journal baseline)
BASE_MANIFEST = ("r31-external-uavdt-20261120-001/release-001/MANIFEST.json",
                 "7efcb10a793d5177c2c41fdad4f886236c9d22202780c734bcbbf7b3f562a41d")
ALLOWED_ADDED = ("tools/r28/r30_diag.py", "tests/test_r30_diag.py", "tools/r28/th2_core.py", "tools/r28/th2_phase1.py",
                 "tools/r28/th2_chain.sh", "tests/test_th2_phase1.py")
SEEDS = (0, 1, 2)
FG_UNITS = tuple(f"FG-s{s}" for s in SEEDS)
STORED = {"W-s0": ("r28-budgeted-resampling-20261113-001/eval-W/top600-predictions.json", 87894197,
                   "2797c49814aed26df3d030571ceec698513c8f5b9b600fe42ac3beceb0953486"),
          "K-s0": ("r28-fixed-prior-control-20261114-001/eval-K/top600-predictions.json", 87894126,
                   "e32cbc5052784cbc05f3fffd29ac19ab6e0fb981c0c7479d03269532034ee141"),
          "F-s0": ("r28-fovea-map-control-20261114-001/eval-F/top600-predictions.json", 87892733,
                   "685c929227b52627e3134c7eafd0731c2253018d9d45868d60cf164bafeef3cc"),
          "A2-s0": ("r30-fgeo-diagnostic-20261118-001/infer/A2/top600-predictions.json", 87893751,
                    "5fd93f965c9d6605412b0c0770907b81efb21debf84a42fd748e01817f30b465")}
A2_SCORE = ("r30-fgeo-diagnostic-20261118-001/score/CLOSED.json", "815effc90aeffea270cafbd59afdd6fcb1c4381fa43d9dd4a733e43808847d31")
E0_UNITS = X.UNITS + tuple(STORED)
E2_UNITS = ("b-s1", "b-s2", "c-s0", "c-s1", "c-s2", "d-s0", "d-s1", "d-s2")
N_GT, N_SMALL, GT_COUNTS = 38599, 26441, {"GT_before": 38759, "GT_after": 38599}
METRICS = ("AP", "APS", "AP75_small", "APM", "APL")
M2_KIND = "linear_with_interactions"
DEFINITIONS = {
    "evaluation_set": "development = official val, 548 images; the readings use the development labels (as R30-001)",
    "folds": "images sorted by stable image id; positions 1, 3, 5, ... are fold A, positions 2, 4, ... fold B",
    "draw": "sha256('TH2-PHASE1-001|<image id>|<k>'): bytes 0-7 mod 12 = centre row, bytes 8-15 mod 21 = centre column; "
            "direction by k: multiply for (k-1) mod 4 in {0, 3}, divide otherwise; the same draws for the three seeds",
    "perturbation": "the predicted demand (12 x 21, without eps) times a (or 1/a) on the block centred on the drawn cell, clipped at the border",
    "amplitude": "geometry only, fold A: a in {2,3,4,6,8} whose median |dlog g| of the small GT with centre cell in the block is "
                 "closest to 0.15; if no candidate reaches 0.08 the 5 x 9 block is tried once; otherwise the chain stops",
    "invalid_map": "a perturbed demand whose map is not finite / strictly increasing stops the geom stage; nothing is repaired",
    "hit": "same image, same class, best IoU over the 600 candidates >= 0.5 / 0.75; no one-to-one matching, no score threshold",
    "q50": "highest score among the same-class candidates with IoU >= 0.5, 0 if none",
    "g": "sqrt(ox * oy) of the map of that inference on the GT box (r28_fovea.magnification)",
    "GT": "the 38,599 scored development GT (raster ignore prefilter, non-crowd); small = annotation area <= 32^2 (26,441)",
    "size_bins": "annotation area: <16^2, [16^2, 32^2], (32^2, 64^2), [64^2, 96^2), >=96^2; medium = bins 3 and 4",
    "features": "s = sqrt(wh) on canvas L, class, occlusion and truncation columns of the annotation file, log aspect ratio, number "
                "of other GT of the image with centre within 64 canvas-L pixels, unperturbed g",
    "slope": "through-origin regression of dy on dlog g; unit pp per 0.1 log g; pairs with dlog g = 0 are in no sign group",
    "M0": "constant slope", "M1": "piecewise linear in s (hinges at 16, 32, 64, 96; s capped at 160)",
    "M2": M2_KIND + ": the M1 basis, the other features as main effects and each of them times omega(s); least squares, "
          "fitted on y = hit75 pooled over the three seeds, no model comparison",
    "M2_increment": "pair-level correlation of the cross-fitted prediction beta(x) * dlog g with the three-seed mean of d hit75; "
                    "corr(M2) - corr(M1), image bootstrap; 'no resolvable increment' = the 95% interval contains 0",
    "M1_shape": "binned slope per size bin and the bin mean of omega(s), each divided by its sum over the five bins; 'agrees' = "
                "omega lies inside the 95% image-bootstrap interval of the slope share in every bin",
    "per_gt_slope_reliability": "per-GT slope over odd k against even k (GT whose sum of dlog g^2 is >= 0.02 in both halves)",
    "V*": "the label-demand accumulation with omega(s) replaced by max(beta_M2(x), 0), every valid GT of the image, same grid and "
          "3 x 3 mean; fold-B images use the fold-A fit and vice versa",
    "rank_remap": "(d): per image the 252 values of D* placed by the rank of V*; ties of V* ordered by D* (descending), then by cell index",
    "divergent_set": "small GT whose magnification bin (g<1, 1-1.2, 1.2-1.4, >=1.4) differs between (d) and (b)",
    "bootstrap": "images resampled with replacement, 1,000 times, seed 20261122; per seed",
    "verdict": "task section 5; main reading small-GT hit75, (d) - (b), three-seed mean",
}
NOT_AUTHORISED = ["any training, fine-tuning, BN re-estimation or demand-network training", "VisDrone test-dev; the 1,602 held-out train images",
                  "a formal training after a GO verdict", "extra scans of amplitude, block size or model type",
                  "merging pull requests; cleaning old rounds"]


def is_dev(r):
    return Path(r).resolve().name.startswith("dev-round")


def ld(rel):
    return B.LD_ROOT / rel


# ================================================================ contract
def prepare(round_dir, release):
    R = Path(round_dir).resolve()
    dev = is_dev(R)
    B.require(not (R / "CONTRACT.json").exists(), "CONTRACT.json exists")
    B.require((R / release / "source").resolve() == ROOT.resolve(), "prepare must run from <round>/<release>/source")
    home = R.parent if dev else R
    B.require("localization-distribution" not in str(R), "outputs must not be under experiments/localization-distribution")
    task = home / TASK[0]
    B.require(task.is_file() and B.file_sha(task) == TASK[1], f"task file missing or different: {task}")
    k = os.environ.get("TH2_DEV_K")
    B.require((k is not None) == dev, "TH2_DEV_K is required in a dev round and refused in a formal round")
    k = T.K_PERTURB if k is None else int(k)
    B.require(k % 4 == 0 and 4 <= k <= T.K_PERTURB, "number of perturbations: a multiple of 4")
    man_ref = B.ref(R / release / "MANIFEST.json")
    manifest = B.read_json(man_ref["path"])
    B.require(manifest["dirty"] is False or dev, "a formal release must come from a clean committed tree")
    B.release_intact(ROOT, manifest["files"])
    base_ref = B.ref(ld(BASE_MANIFEST[0]))
    base = B.read_json(base_ref["path"])
    B.require(base_ref["sha256"] == BASE_MANIFEST[1] and base["commit"] == BASE_COMMIT, "base release (R31) identity")
    new, old = manifest["files"], base["files"]
    delta = {"added": sorted(set(new) - set(old)), "removed": sorted(set(old) - set(new)),
             "changed": sorted(f for f in new if f in old and new[f] != old[f])}
    B.require(set(delta["added"]) == set(ALLOWED_ADDED) and not delta["removed"] and not delta["changed"],
              f"the release may differ from the R31 release only by the R30-001 diagnostic files and the TH2 files: {delta}")
    r30_ref = B.ref(ld(X.R30_CONTRACT[0]))
    r30 = B.read_json(r30_ref["path"])
    B.require(r30_ref["sha256"] == X.R30_CONTRACT[1] and r30["contract_sha256"] == X.R30_CONTRACT[2], "R30 contract identity")
    B.require(os.path.samefile(sys.executable, r30["python"]), f"python differs from the R30 round: {sys.executable}")
    units = {}
    for unit in X.UNITS:
        arm, seed = X.parse(unit)
        pin = X.PIN[unit]
        rec_ref = B.ref(ld(pin["receipt"][0]))
        B.require(rec_ref["sha256"] == pin["receipt"][1], f"{unit}: stored evaluation receipt file")
        rec = B.read_json(rec_ref["path"])
        endpoint = X.checked_ref(ld(pin["endpoint"][0]), pin["endpoint"][1], pin["endpoint"][2], f"{unit} endpoint")
        stored = X.checked_ref(ld(pin["predictions"][0]), pin["predictions"][1], pin["predictions"][2], f"{unit} stored top600")
        B.require(rec["status"] == "PASS" and rec["weights_sha256"] == pin["weights"] and rec.get("seed", 0) == seed, f"{unit}: receipt")
        units[unit] = {"arm": arm, "seed": seed, "kind": "EMA", "epoch": 120, "receipt": rec_ref, "endpoint": endpoint,
                       "side": B.ref(B.side_of(endpoint["path"])), "weights_sha256": pin["weights"], "predictions_600": stored,
                       "values_600": rec["values"]["600"] if arm == "FG" else None}
    stored = {u: X.checked_ref(ld(p), n, s, f"{u} stored top600") for u, (p, n, s) in STORED.items()}
    a2_ref = B.ref(ld(A2_SCORE[0]))
    B.require(a2_ref["sha256"] == A2_SCORE[1], "R30-001 score receipt")
    a2 = B.read_json(a2_ref["path"])["endpoints"]["A2"]
    B.require(a2["GT_counts"]["600"] == GT_COUNTS, "R30-001 A2 GT counts")
    per_gt = X.checked_ref(ld(X.DEVELOPMENT_PER_GT[0]), X.DEVELOPMENT_PER_GT[1], X.DEVELOPMENT_PER_GT[2], "R30-001 per-GT geometry")
    B.pinned(r30["demand"]["weights"])
    ids = r30["development_ids"]
    fold = T.folds(ids)
    passes = k + 1
    c = {"version": VERSION, "task": {"file": TASK[0], "sha256": TASK[1]}, "zero_training": True,
         "source_root": str(ROOT), "source_commit": manifest["commit"], "source_branch": manifest["branch"],
         "source_dirty": manifest["dirty"], "source_manifest": man_ref, "base_commit": BASE_COMMIT, "base_manifest": base_ref,
         "source_delta": delta, "outputs": str(R), "dev": dev, "perturbations": k,
         "R30_contract": {**r30_ref, "contract_sha256": r30["contract_sha256"]},
         "units": units, "stored_predictions": stored, "label_demand_seed0": {"receipt": a2_ref, "values_600": a2["values"]["600"]},
         "development_geometry": {"per_gt": per_gt},
         "demand": r30["demand"], "width_selection": r30["width_selection"], "mapping": X.MAPPING,
         "data": {"development": r30["data"]["development"]}, "development_ids": ids,
         "gpu_uuid": r30["gpu_uuid"], "environment": r30["environment"], "python": sys.executable,
         "folds": {"A": [i for i, f in zip(ids, fold) if f == 0], "B": [i for i, f in zip(ids, fold) if f == 1]},
         "design": {"blocks": [list(b) for b in T.BLOCKS], "amplitudes": list(T.AMPLITUDES), "amplitude_target": T.AMP_TARGET,
                    "amplitude_floor": T.AMP_FLOOR, "draw_tag": T.DRAW_TAG, "M2": M2_KIND, "M2_extra_features": list(T.M2_EXTRA),
                    "s_knots": list(T.S_KNOTS), "s_cap": T.S_CAP, "size_bins": list(T.SIZE_NAMES), "g_bins": list(T.G_NAMES),
                    "crowd_bins": list(T.CROWD_NAMES), "crowd_radius": T.CROWD_RADIUS, "amplified_dlogg": T.AMPLIFIED,
                    "min_ssx": T.MIN_SSX, "bootstrap": [T.BOOT, T.BOOT_SEED], "E2_units": list(E2_UNITS)},
         "definitions": DEFINITIONS, "verdict_rule": T.VERDICT,
         "workload": {"wire_forwards": 3 * 2, "demand_forwards": 137, "e1_forwards": 3 * passes * 137, "e2_forwards": len(E2_UNITS) * 137,
                      "score_budget_calls": len(E2_UNITS), "updates": 0, "gpu_cap_hours": 4.0},
         "resource_plan": {"storage_estimate_bytes": {"e2_top600_predictions": len(E2_UNITS) * 88_000_000,
                                                      "per_gt_tables": 3 * N_GT * passes * 8 + N_GT * passes * 8 + 40_000_000,
                                                      "demand_and_value_maps": 548 * 252 * 8 * 5, "release_logs_small": 100_000_000},
                           "disk_free_bytes_at_prepare": shutil.disk_usage(R).free,
                           "kept_on_server": "per-GT tables, maps, E2 predictions; only small JSON / report files are returned"},
         "not_authorised": NOT_AUTHORISED}
    c["contract_sha256"] = B.sha_bytes(B.canonical(c))
    B.write_new(R / "CONTRACT.json", c)
    print(json.dumps({"contract_sha256": c["contract_sha256"], "contract_file": B.ref(R / "CONTRACT.json"),
                      "commit": c["source_commit"], "delta": delta, "perturbations": k}))


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
    B.require(c["task"]["sha256"] == TASK[1] and c["design"]["M2"] == M2_KIND and c["definitions"] == DEFINITIONS
              and c["verdict_rule"] == T.VERDICT, "contract differs from the release")
    return c


def stage_closed(c, name):
    path = Path(c["outputs"]) / name / "CLOSED.json"
    B.require(path.is_file(), f"stage not closed: {path}")
    closed = B.read_json(path)
    B.require(closed.get("status") == "PASS" and closed.get("contract_sha256") == c["contract_sha256"], f"stage {name} is not PASS")
    return closed


def load(c, stage, key):
    return np.load(B.pinned(stage_closed(c, stage)["files"][key]), allow_pickle=False)


# ================================================================ shared pieces
def development(c):
    from sparse_rtdetr.research import scale_baseline as S
    data = S.development(c["data"]["development"], S.S15_HW)
    B.require([r["stable_image_id"] for r in data.records] == c["development_ids"] and len(data.records) == 548, "development members/order")
    return data


def decode_all(data):
    """uint8 (3, 1152, 2016) tensors of the 548 images: infer_seed.pixels without its last step (.float() / 255)."""
    from concurrent.futures import ThreadPoolExecutor
    from infer_seed import load_original
    from torchvision.transforms.v2 import functional as TF

    def one(row):
        with load_original(data, row) as image:
            return TF.pil_to_tensor(data.transform.resize(image))

    with ThreadPoolExecutor(8) as pool:
        return list(pool.map(one, data.records))


def replay_context(base, fwhm):
    """A demand context that returns the planned maps batch by batch; the forward is not touched."""
    torch = B.torch_mod()

    class ReplayDemandContext(base):
        def __init__(self):
            self.kind, self.source, self.net, self.fixed = "replay", {"kind": "replay"}, None, None
            self.observe, self.records = False, []
            self.device = torch.device("cpu")
            self.maps, self.cursor = None, 0

        def to(self, device):
            self.device = torch.device(device)
            return self

        def binding(self):
            return {"kind": "replay", "file": None, "seed": None}

        def start(self, maps):
            self.maps, self.cursor = maps, 0

        def demand(self, x):
            b = x.shape[0]
            B.require(self.maps is not None and self.cursor + b <= len(self.maps), "replay demand context called beyond its plan")
            d = self.maps[self.cursor:self.cursor + b]
            self.cursor += b
            return d.to(torch.float64).cpu()

    ctx = ReplayDemandContext()
    vars(ctx)["r28_fwhm"] = fwhm
    return ctx


def forward_pass(model, images, device, counts, sink):
    """548 images in batches of 4 on the evaluation path (FP32, autocast off, forward guards, inference mode)."""
    from sparse_rtdetr.protocol_r18.geometry import forward_guards
    torch = B.torch_mod()
    with torch.inference_mode():
        for off in range(0, len(images), X.BATCH):
            x = (torch.stack(images[off:off + X.BATCH]).float() / 255.).to(device)
            B.require(tuple(x.shape) == (X.BATCH, 3, 1152, 2016) and x.dtype == torch.float32, "the input is not the FP32 H canvas")
            counts.update(forward_attempts=1)
            with torch.autocast("cuda", enabled=False), forward_guards(model):
                o = model(x)
            counts.update(forwards=1)
            B.require(set(o) == {"pred_logits", "pred_boxes"} and o["pred_boxes"].shape == (X.BATCH, 300, 4)
                      and o["pred_boxes"].dtype == o["pred_logits"].dtype == torch.float32, "endpoint output schema")
            sink(off, o["pred_boxes"].cpu(), o["pred_logits"].cpu())


def gpu_stage(c):
    torch = B.torch_mod()
    have = X.nvidia("gpu=uuid")
    B.require(have == [c["gpu_uuid"]], f"GPU differs from the contract: {have}")
    return torch, torch.device("cuda:0"), X.other_gpu_processes()


def stored_index(path):
    """Stored top600 records -> per image (category, score, corners) in stored order."""
    with Path(path).open() as f:
        records = json.load(f)
    B.require(len(records) == 548 * 600, "stored top600 record count")
    out = {}
    for i in range(0, len(records), 600):
        part = records[i:i + 600]
        B.require(len({p["image_id"] for p in part}) == 1, "stored records are not grouped by image")
        out[part[0]["image_id"]] = (np.array([p["category_id"] for p in part]), np.array([p["score"] for p in part], dtype=np.float64),
                                    np.array([p["bbox_xyxy"] for p in part], dtype=np.float64))
    return out


class GT:
    """The scored GT rows (image order, annotation order) with per-image slices."""

    def __init__(self, c):
        z = load(c, "geom", "gt")
        self.z = {k: z[k] for k in z.files}
        sc = self.z["scored"]
        self.image, self.cat, self.box = self.z["image"][sc], self.z["category"][sc], self.z["xyxy"][sc]
        self.area = self.z["area"][sc]
        self.n = int(sc.sum())
        self.bounds = np.searchsorted(self.image, np.arange(549))
        self.size = T.size_bin(self.area)
        self.small = self.area <= T.SMALL_AREA
        self.medium = (self.size == 2) | (self.size == 3)
        self.fold = T.folds(c["development_ids"])[self.image]

    def sl(self, i):
        return slice(self.bounds[i], self.bounds[i + 1])

    def features(self, scored_only=True):
        m = self.z["scored"] if scored_only else np.ones(len(self.z["scored"]), dtype=bool)
        return {k: self.z[k][m] for k in ("s", "category", "occlusion", "truncation", "log_aspect", "crowd", "g0")}


def outcomes_of(gt, index, ids):
    iou, q = np.zeros(gt.n), np.zeros(gt.n)
    for i, image_id in enumerate(ids):
        s = gt.sl(i)
        cat, score, box = index[image_id]
        iou[s], q[s] = T.outcomes(gt.box[s], gt.cat[s], cat, score, box)
    return iou, q


# ================================================================ demand (CUDA)
def demand(c, out, counts):
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    from sparse_rtdetr.protocol_r18.engine import tensor_digest
    from sparse_rtdetr.r0.evaluation import rank_candidates
    B.PHASE["phase"] = "demand/load"
    for u in FG_UNITS:
        B.require(stage_closed(c, f"wire-{u}")["passed"] is True, "wiring check")
    torch, device, others = gpu_stage(c)
    model, side, weights, _ = X.build_model(c, "FG-s0", device)
    data = development(c)
    images = decode_all(data)
    store, remove = X.record_demand(model)
    mine = []

    def sink(off, boxes, logits):
        for j in range(X.BATCH):
            mine.extend(rank_candidates(logits[j], boxes[j], data.records[off + j], 600))

    rng = ck.capture_rng(device)
    B.PHASE["phase"] = "demand/forward"
    t0 = time.monotonic()
    forward_pass(model, images, device, counts, sink)
    seconds = time.monotonic() - t0
    remove()
    B.require(counts["forwards"] == 137 and len(store) == 137, "forward count")
    B.require(tensor_digest(model.state_dict().items()) == weights and ck.exact_equal(ck.capture_rng(device), rng)
              and model.r28_ctx.binding() == side["demand"], "the forwards changed the model, the RNG or the demand source")
    d = torch.cat(store).numpy()
    B.require(d.shape == (548, 12, 21) and d.dtype == np.float64 and bool(np.isfinite(d).all()) and bool((d >= 0).all()), "demand maps")
    stored = B.read_json(B.pinned(c["units"]["FG-s0"]["predictions_600"]))
    difference = X.C.records_equal(mine, stored)
    B.require(difference is None, f"the full forward does not reproduce the stored F_geo seed-0 predictions: {difference}")
    return {"stage": "demand", "files": {"demand": X.save_npy(out / "demand.npy", d)}, "images": 548, "records_equal_to_stored": True,
            "records": len(mine), "pass_seconds": seconds, "demand_min_max_mean": [float(d.min()), float(d.max()), float(d.mean())],
            "weights_sha256": weights, "other_gpu_compute_processes_at_start": others,
            "other_gpu_compute_processes_at_end": X.other_gpu_processes(), "peak_cuda_bytes": torch.cuda.max_memory_allocated()}


# ================================================================ geom (CPU)
def _pad_boxes(boxes):
    m = max(1, max(len(b) for b in boxes))
    pad = np.tile(np.array([0.25, 0.25, 0.5, 0.5]), (len(boxes), m, 1))
    for i, b in enumerate(boxes):
        pad[i, :len(b)] = b
    return pad


def magnify(demand_maps, pad, lengths, select=None):
    """Demand maps (B, 12, 21) -> list of per-image g arrays under the F_geo map.  Raises if a map is invalid."""
    from sparse_rtdetr.research import r28_fovea as FV
    from sparse_rtdetr.research import r28_fovea_geo as FG
    torch = B.torch_mod()
    plan = FG.plan_from_demand(torch.from_numpy(np.ascontiguousarray(demand_maps)))
    ox, oy = FV.magnification(plan, torch.from_numpy(pad))
    g = torch.sqrt(ox * oy).numpy()
    return np.concatenate([g[i, :n] for i, n in enumerate(lengths)]), plan


def geom(c, out, counts):
    import r28_demand as DM
    from sparse_rtdetr.r0.evaluation import bridge_inputs
    from sparse_rtdetr.research import r28_warp as W
    torch = B.torch_mod()
    B.PHASE["phase"] = "geom/table"
    d_hat = np.load(B.pinned(stage_closed(c, "demand")["files"]["demand"]), allow_pickle=False)
    data = development(c)
    rows = data.records
    ids = c["development_ids"]
    parsed = [data.raw(r) for r in rows]
    dataset, _, gt_counts = bridge_inputs(rows, parsed, [])
    B.require({k: gt_counts[k] for k in GT_COUNTS} == GT_COUNTS, "certified GT counts")
    scored_ids = {a["raw_annotation_id"] for a in dataset["annotations"] if not a["iscrowd"]}
    cols = {k: [] for k in ("image", "category", "xyxy", "norm", "area", "occlusion", "truncation", "scored")}
    raw_ids, boxes, d_star = [], [], []
    for i, (row, p) in enumerate(zip(rows, parsed)):
        norm, keep = DM.deployment_xyxy(p, row["width"], row["height"])
        d_star.append(W.demand_from_boxes(norm).numpy())
        norm = norm.numpy()
        wh = np.array([row["width"], row["height"]] * 2, dtype=np.float64)
        objs = [o for o, k in zip(p["gt"], keep) if k]
        boxes.append(norm)
        cols["image"] += [i] * len(objs)
        cols["norm"].append(norm.reshape(-1, 4))
        cols["xyxy"].append((norm * wh).reshape(-1, 4))
        for o in objs:
            f = [int(float(v)) for v in o["raw_fields"]]
            B.require(len(f) == 8 and f[5] == o["category"] and f[4] == o["score"], "raw annotation columns")
            cols["category"].append(o["category"])
            cols["area"].append(float(o["raw_area"]))
            cols["truncation"].append(f[6])
            cols["occlusion"].append(f[7])
            cols["scored"].append(o["id"] in scored_ids)
            raw_ids.append(o["id"])
    z = {k: np.array(v) if k not in ("xyxy", "norm") else np.concatenate(v) for k, v in cols.items()}
    B.require(int(z["scored"].sum()) == N_GT == len(scored_ids), f"scored GT rows: {int(z['scored'].sum())}")
    B.require(int((z["area"][z["scored"]] <= T.SMALL_AREA).sum()) == N_SMALL, "small GT count")
    w = (z["norm"][:, 2] - z["norm"][:, 0]) * T.L_HW[1]
    h = (z["norm"][:, 3] - z["norm"][:, 1]) * T.L_HW[0]
    z["s"] = np.sqrt(w * h)
    z["log_aspect"] = np.log(w / h)
    cx, cy = (z["norm"][:, 0] + z["norm"][:, 2]) / 2 * T.L_HW[1], (z["norm"][:, 1] + z["norm"][:, 3]) / 2 * T.L_HW[0]
    z["crowd"] = T.crowd_counts(z["image"], cx, cy)
    z["row"], z["col"] = T.cell_of(z["norm"])
    d_star = np.stack(d_star)
    mine = np.stack([T.blur3(T.weighted_demand(b, T.omega(np.sqrt((b[:, 2] - b[:, 0]) * T.L_HW[1] * (b[:, 3] - b[:, 1]) * T.L_HW[0])))) for b in boxes])
    B.require(bool(np.allclose(mine, d_star, rtol=0, atol=1e-12)), "the numpy label-demand accumulation differs from r28_warp.demand_from_boxes")

    B.PHASE["phase"] = "geom/baseline"
    pad, lengths = _pad_boxes(boxes), [len(b) for b in boxes]
    g0, _ = magnify(d_hat, pad, lengths)
    gb, _ = magnify(d_star, pad, lengths)
    z["g0"], z["g_b"] = g0, gb
    ref = B.read_json(B.pinned(c["development_geometry"]["per_gt"]))
    pos = {r: n for n, r in enumerate(raw_ids)}
    idx = np.array([pos[r["raw_annotation_id"]] for r in ref])
    B.require(len(set(idx.tolist())) == N_GT and bool(z["scored"][idx].all()), "R30-001 per-GT rows are not the scored GT")
    err = {"predicted": float(np.abs(g0[idx] - np.array([r["predicted"] for r in ref])).max()),
           "label": float(np.abs(gb[idx] - np.array([r["gt_demand"] for r in ref])).max())}
    B.require(err["predicted"] < 1e-9 and err["label"] < 1e-9, f"magnification differs from the R30-001 per-GT file: {err}")

    B.PHASE["phase"] = "geom/amplitude"
    fold = T.folds(ids)
    small = z["area"] <= T.SMALL_AREA
    lg0 = np.log(g0)
    k_all = c["perturbations"]

    def perturbed(a, block, k, images):
        maps, inblock = np.empty((len(images), 12, 21)), np.zeros(len(z["image"]), dtype=bool)
        for n, i in enumerate(images):
            maps[n], m = T.perturb(d_hat[i], ids[i], k, a, block)
            sel = z["image"] == i
            inblock[sel] = m[z["row"][sel], z["col"][sel]]
        return maps, inblock

    by_image = [np.flatnonzero(z["image"] == i) for i in range(548)]
    img_a = [i for i in range(548) if fold[i] == 0]
    rows_a = np.concatenate([by_image[i] for i in img_a])
    pad_a, len_a = pad[img_a], [lengths[i] for i in img_a]
    calibration, chosen = [], None
    for block in T.BLOCKS:
        med = {}
        for a in T.AMPLITUDES:
            vals = []
            for k in range(1, T.K_PERTURB + 1):          # the calibration always uses the 48 draws
                maps = np.stack([T.perturb(d_hat[i], ids[i], k, a, block)[0] for i in img_a])
                g, _ = magnify(maps, pad_a, len_a)
                r, cc, _ = zip(*[T.draw(ids[i], k) for i in img_a])
                inb = np.concatenate([T.block_mask(r[n], cc[n], block)[z["row"][by_image[i]], z["col"][by_image[i]]] for n, i in enumerate(img_a)])
                sel = inb & small[rows_a] & z["scored"][rows_a]
                vals.append(np.abs(np.log(g[sel]) - lg0[rows_a][sel]))
            med[a] = float(np.median(np.concatenate(vals)))
        a, reached = T.choose_amplitude(med)
        calibration.append({"block": list(block), "median_abs_dlogg": {str(k): v for k, v in med.items()}, "reached_floor": reached, "chosen": a})
        if reached:
            chosen = (a, block)
            break
    B.require(chosen is not None, f"no amplitude reaches the floor with either block; hand back: {calibration}")
    a, block = chosen

    B.PHASE["phase"] = "geom/perturbed"
    sc = z["scored"]
    g = np.empty((int(sc.sum()), k_all + 1))
    g[:, 0] = g0[sc]
    inblock = np.zeros((int(sc.sum()), k_all), dtype=bool)
    maps_all = np.empty((k_all + 1, 548, 12, 21))
    maps_all[0] = d_hat
    centre = np.zeros((548, k_all, 3), dtype=np.int16)
    for k in range(1, k_all + 1):
        maps, inb = perturbed(a, block, k, range(548))
        gk, _ = magnify(maps, pad, lengths)              # raises on an invalid map: nothing is repaired
        g[:, k], inblock[:, k - 1], maps_all[k] = gk[sc], inb[sc], maps
        centre[:, k - 1] = [T.draw(i, k) for i in ids]
    B.require(bool(np.isfinite(g).all()) and bool((g > 0).all()), "perturbed magnification")
    dl = np.abs(np.log(g[:, 1:]) - np.log(g[:, :1]))
    sm = small[sc]
    annex = {"amplitude": a, "block": list(block), "fallback_block_used": list(block) != list(T.BLOCKS[0]), "calibration": calibration,
             "fold": "A", "contract_sha256": c["contract_sha256"], "written_before_any_detector_inference_of_E1": True}
    B.write_new(out / "AMPLITUDE.json", annex)
    files = {"gt": None, "g": X.save_npy(out / "g.npy", g), "inblock": X.save_npy(out / "inblock.npy", inblock),
             "maps": X.save_npy(out / "maps.npy", maps_all), "d_star": X.save_npy(out / "d_star.npy", d_star),
             "centre": X.save_npy(out / "centre.npy", centre)}
    tmp = out / "gt.npz"
    np.savez(tmp, **z)
    files["gt"] = B.ref(tmp)
    B.write_new(out / "raw_ids.json", [r for r, s in zip(raw_ids, sc) if s])
    B.require(counts["forwards"] == 0, "geom must not run a model")
    return {"stage": "geom", "files": files, "amplitude_annex": B.ref(out / "AMPLITUDE.json"), "raw_ids": B.ref(out / "raw_ids.json"),
            "amplitude": a, "block": list(block), "calibration": calibration, "rows": {"deployment": int(len(sc)), "scored": int(sc.sum()), "small": int(sm.sum())},
            "check_against_R30_001_per_gt": err, "occlusion_levels": {str(k): int(v) for k, v in zip(*np.unique(z["occlusion"][sc], return_counts=True))},
            "truncation_levels": {str(k): int(v) for k, v in zip(*np.unique(z["truncation"][sc], return_counts=True))},
            "realised": {"median_abs_dlogg_small_in_block": float(np.median(dl[sm][inblock[sm]])) if inblock[sm].any() else None,
                         "median_abs_dlogg_all_pairs": float(np.median(dl)), "share_pairs_abs_dlogg_ge_0.10": float((dl >= 0.10).mean())}}


# ================================================================ e0 (CPU)
def _flip(gt, a, b, im, w):
    rows = {}
    for name, sel in _size_groups(gt).items():
        rows[name] = T.ratio((a != b) & sel, sel, im, 548, w, scale=100.0)
    return rows


def _size_groups(gt):
    out = {n: gt.size == i for i, n in enumerate(T.SIZE_NAMES)}
    out.update(small=gt.small, medium=gt.medium, all=np.ones(gt.n, dtype=bool))
    return out


def _pearson(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or x[ok].std() == 0 or y[ok].std() == 0:
        return None
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def e0(c, out, counts):
    B.PHASE["phase"] = "e0/outcomes"
    gt = GT(c)
    ids = c["development_ids"]
    w = T.boot_weights(548)
    arrays, refs = {}, {}
    for u in E0_UNITS:
        ref = c["units"][u]["predictions_600"] if u in c["units"] else c["stored_predictions"][u]
        iou, q = outcomes_of(gt, stored_index(B.pinned(ref)), ids)
        arrays[f"iou_{u}"], arrays[f"q_{u}"] = iou, q
        refs[u] = ref
    np.savez(out / "outcomes.npz", **arrays)
    hit = {t: {u: arrays[f"iou_{u}"] >= t for u in E0_UNITS} for t in (0.5, 0.75)}
    names = {0.5: "hit50", 0.75: "hit75"}
    B.PHASE["phase"] = "e0/tables"
    coverage = {u: {names[t]: {g: float(100 * hit[t][u][s].mean()) for g, s in _size_groups(gt).items()} for t in names} for u in E0_UNITS}
    pairs = ((0, 1), (0, 2), (1, 2))
    noise = {arm: {f"s{a}-s{b}": {names[t]: _flip(gt, hit[t][f"{arm}-s{a}"], hit[t][f"{arm}-s{b}"], gt.image, w) for t in names}
                   for a, b in pairs} for arm in ("L0", "FG", "S15")}
    cross = {f"{x}-{y}": {f"s{s}": {names[t]: _flip(gt, hit[t][f"{x}-s{s}"], hit[t][f"{y}-s{s}"], gt.image, w) for t in names} for s in SEEDS}
             for x, y in (("FG", "L0"), ("S15", "L0"), ("S15", "FG"))}
    f = gt.features()
    groups = {"size": (gt.size, T.SIZE_NAMES), "class": (gt.cat - 1, tuple(str(k) for k in range(1, 11))),
              "occlusion": (np.minimum(f["occlusion"], 2), ("0", "1", "2")), "crowd": (T.crowd_bin(f["crowd"]), T.CROWD_NAMES)}
    cell = gt.size * 1000 + (gt.cat - 1) * 100 + np.minimum(f["occlusion"], 2) * 10 + T.crowd_bin(f["crowd"])
    response = {}
    for t in names:
        r = {s: hit[t][f"S15-s{s}"].astype(np.float64) - hit[t][f"L0-s{s}"] for s in SEEDS}
        means = {}
        for gname, (code, labels) in groups.items():
            means[gname] = [{"group": labels[k], "n": int((code == k).sum()),
                             **{f"s{s}": T.ratio(r[s] * (code == k), code == k, gt.image, 548, w, scale=100.0) for s in SEEDS}}
                            for k in range(len(labels))]
        cells = [k for k, n in zip(*np.unique(cell, return_counts=True)) if n >= 50]
        cm = {s: np.array([r[s][cell == k].mean() for k in cells]) for s in SEEDS}
        response[names[t]] = {"per_gt_seed_correlation": {f"s{a}-s{b}": _pearson(r[a], r[b]) for a, b in pairs},
                              "per_gt_seed_correlation_small": {f"s{a}-s{b}": _pearson(r[a][gt.small], r[b][gt.small]) for a, b in pairs},
                              "group_mean_seed_correlation": {f"s{a}-s{b}": _pearson(cm[a], cm[b]) for a, b in pairs},
                              "group_cells": {"definition": "size bin x class x occlusion x crowd bin, cells with n >= 50", "cells": len(cells)},
                              "group_means_pp": means}
    gb = T.g_bin(f["g0"])
    table15 = {}
    for s in SEEDS:
        rows = []
        tot = {t: None for t in names}
        for t in names:
            fg, s15 = hit[t][f"FG-s{s}"], hit[t][f"S15-s{s}"]
            tot[t] = int((s15 & ~fg & gt.small).sum()) - int((fg & ~s15 & gt.small).sum())
        for k, name in enumerate(T.G_NAMES + ("all",)):
            sel = gt.small & ((gb == k) if name != "all" else True)
            row = {"bin": name, "n": int(sel.sum())}
            for t in names:
                fg, l0, s15 = hit[t][f"FG-s{s}"], hit[t][f"L0-s{s}"], hit[t][f"S15-s{s}"]
                net = int((s15 & ~fg & sel).sum()) - int((fg & ~s15 & sel).sum())
                row[names[t]] = {"FG": float(100 * fg[sel].mean()), "L0": float(100 * l0[sel].mean()), "S15": float(100 * s15[sel].mean()),
                                 "FG-L0_pp": float(100 * (fg[sel].mean() - l0[sel].mean())), "S15-FG_pp": float(100 * (s15[sel].mean() - fg[sel].mean())),
                                 "S15_only": int((s15 & ~fg & sel).sum()), "FG_only": int((fg & ~s15 & sel).sum()), "net": net,
                                 "share_of_net_pct": None if not tot[t] else float(100 * net / tot[t])}
            rows.append(row)
        table15[f"s{s}"] = rows
    res = {"coverage_pp": coverage, "noise_floor_flip_rate_pp": noise, "cross_arm_flip_rate_pp": cross, "S15_response": response,
           "table15_by_Fgeo_magnification": {"bins_by": "g under the predicted demand (same map for the three seeds)", "per_seed": table15}}
    B.write_new(out / "E0.json", res)
    B.require(counts["forwards"] == 0, "e0 must not run a model")
    return {"stage": "e0", "files": {"outcomes": B.ref(out / "outcomes.npz")}, "result": B.ref(out / "E0.json"), "inputs": refs,
            "small_hit75_pp": {u: coverage[u]["hit75"]["small"] for u in E0_UNITS}}


# ================================================================ e1 / e2 (CUDA)
def _replay_model(c, unit, device):
    from sparse_rtdetr.research import r28_fovea_geo as FG
    from sparse_rtdetr.research import r28_resample as RS
    model, side, weights, _ = X.build_model(c, unit, device)
    network = model.r28_ctx
    ctx = replay_context(RS.Context, vars(network)["r28_fwhm"]).to(device)
    vars(model)["r28_ctx"] = ctx
    B.require(FG.is_wrapped(model), "the model with the replay context is not an F_geo evaluation model")
    return model, ctx, network, side, weights


def _pass(model, ctx, maps, images, device, counts, gt, rows, keep=None):
    """One inference of the 548 images under the given maps -> per-GT (iou_max, q50)."""
    from sparse_rtdetr.r0.evaluation import rank_candidates
    torch = B.torch_mod()
    iou, q = np.zeros(gt.n), np.zeros(gt.n)
    ctx.start(torch.from_numpy(np.ascontiguousarray(maps)))

    def sink(off, boxes, logits):
        scores = logits.sigmoid().numpy()
        bx = boxes.numpy()
        for j in range(X.BATCH):
            i = off + j
            cat, score, box, _ = T.top600(scores[j], bx[j], rows[i]["width"], rows[i]["height"])
            s = gt.sl(i)
            iou[s], q[s] = T.outcomes(gt.box[s], gt.cat[s], cat, score, box)
            if keep is not None:
                keep.extend(rank_candidates(logits[j], boxes[j], rows[i], 600))

    forward_pass(model, images, device, counts, sink)
    B.require(ctx.cursor == 548, "the replay context did not serve every image once")
    return iou, q


def e1(c, out, counts, unit):
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    from sparse_rtdetr.protocol_r18.engine import tensor_digest
    seed = int(unit[1:])
    B.PHASE["phase"] = f"e1/{unit}/load"
    geo = stage_closed(c, "geom")
    annex = B.read_json(B.pinned(geo["amplitude_annex"]))
    B.require(annex["contract_sha256"] == c["contract_sha256"] and annex["amplitude"] == geo["amplitude"], "amplitude annex")
    stage_closed(c, "e0")
    torch, device, others = gpu_stage(c)
    gt = GT(c)
    maps = load(c, "geom", "maps")
    k_all = c["perturbations"]
    B.require(maps.shape == (k_all + 1, 548, 12, 21), "perturbed maps")
    model, ctx, network, side, weights = _replay_model(c, f"FG-s{seed}", device)
    data = development(c)
    images = decode_all(data)
    rng = ck.capture_rng(device)
    iou, q = np.zeros((gt.n, k_all + 1), dtype=np.float32), np.zeros((gt.n, k_all + 1), dtype=np.float32)
    B.PHASE["phase"] = f"e1/{unit}/unperturbed"
    keep = []
    t0 = time.monotonic()
    a, b = _pass(model, ctx, maps[0], images, device, counts, gt, data.records, keep)
    seconds = [time.monotonic() - t0]
    stored = B.read_json(B.pinned(c["units"][f"FG-s{seed}"]["predictions_600"]))
    difference = X.C.records_equal(keep, stored)
    B.require(difference is None, f"the unperturbed replay pass does not reproduce the stored predictions: {difference}")
    z = load(c, "e0", "outcomes")
    B.require(bool(np.array_equal(a, z[f"iou_FG-s{seed}"])) and bool(np.array_equal(b, z[f"q_FG-s{seed}"])),
              "per-GT outcomes of the unperturbed pass differ from those of the stored predictions")
    del keep, stored
    iou[:, 0], q[:, 0] = a, b
    for k in range(1, k_all + 1):
        B.PHASE["phase"] = f"e1/{unit}/k{k}"
        t0 = time.monotonic()
        a, b = _pass(model, ctx, maps[k], images, device, counts, gt, data.records)
        iou[:, k], q[:, k] = a, b
        seconds.append(time.monotonic() - t0)
    vars(model)["r28_ctx"] = network
    B.require(counts["forwards"] == counts["forward_attempts"] == (k_all + 1) * 137 and counts["actual_updates"] == 0, "forward count")
    B.require(tensor_digest(model.state_dict().items()) == weights and ck.exact_equal(ck.capture_rng(device), rng)
              and model.r28_ctx.binding() == side["demand"], "the passes changed the model, the RNG or the demand source")
    return {"stage": "e1", "unit": unit, "seed": seed, "passes": k_all + 1, "forwards": counts["forwards"],
            "files": {"iou": X.save_npy(out / "iou.npy", iou), "q50": X.save_npy(out / "q50.npy", q)},
            "unperturbed_equals_stored": True, "pass_seconds_median": float(np.median(seconds)), "pass_seconds_total": float(np.sum(seconds)),
            "weights_sha256": weights, "other_gpu_compute_processes_at_start": others,
            "other_gpu_compute_processes_at_end": X.other_gpu_processes(), "peak_cuda_bytes": torch.cuda.max_memory_allocated()}


def e2(c, out, counts, unit):
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    from sparse_rtdetr.protocol_r18.engine import tensor_digest
    source, seed = unit.split("-s")
    seed = int(seed)
    B.PHASE["phase"] = f"e2/{unit}/load"
    maps = load(c, "r14", f"map_{source}")
    B.require(maps.shape == (548, 12, 21) and bool(np.isfinite(maps).all()) and bool((maps >= 0).all()), "demand maps of the source")
    torch, device, others = gpu_stage(c)
    gt = GT(c)
    model, ctx, network, side, weights = _replay_model(c, f"FG-s{seed}", device)
    data = development(c)
    images = decode_all(data)
    rng = ck.capture_rng(device)
    keep = []
    B.PHASE["phase"] = f"e2/{unit}/forward"
    t0 = time.monotonic()
    iou, q = _pass(model, ctx, maps, images, device, counts, gt, data.records, keep)
    seconds = time.monotonic() - t0
    vars(model)["r28_ctx"] = network
    B.require(counts["forwards"] == 137 and len(keep) == 548 * 600, "forward / record count")
    B.require(tensor_digest(model.state_dict().items()) == weights and ck.exact_equal(ck.capture_rng(device), rng), "model / RNG changed")
    B.write_new(out / "top600-predictions.json", keep)
    return {"stage": "e2", "unit": unit, "source": source, "seed": seed, "forwards": 137, "pass_seconds": seconds,
            "predictions": B.ref(out / "top600-predictions.json"),
            "files": {"iou": X.save_npy(out / "iou.npy", iou), "q50": X.save_npy(out / "q50.npy", q)}, "weights_sha256": weights,
            "other_gpu_compute_processes_at_start": others, "other_gpu_compute_processes_at_end": X.other_gpu_processes()}


def score(c, out, counts, unit):
    from infer_seed import bridge_values
    from sparse_rtdetr.r0.evaluation import score_budget
    B.PHASE["phase"] = f"score/{unit}"
    ev = stage_closed(c, f"e2-{unit}")
    preds = B.read_json(B.pinned(ev["predictions"]))
    data = development(c)
    rows = data.records
    parsed = [data.raw(r) for r in rows]
    metrics = score_budget(rows, parsed, preds, c["source_root"])
    counts.update(score_budget_calls=1)
    B.require({k: metrics["spatial_filter_counts"][k] for k in GT_COUNTS} == GT_COUNTS, "scored GT set")
    metrics["TH2_binding"] = {"contract_sha256": c["contract_sha256"], "unit": unit, "predictions": ev["predictions"], "weights_sha256": ev["weights_sha256"]}
    B.write_new(out / "top600-metrics.json", metrics)
    return {"stage": "score", "unit": unit, "values_600": bridge_values(metrics), "metrics": B.ref(out / "top600-metrics.json")}


# ================================================================ r14 (CPU)
def _e1(c, gt):
    iou = np.stack([load(c, f"e1-s{s}", "iou") for s in SEEDS]).astype(np.float64)
    q = np.stack([load(c, f"e1-s{s}", "q50") for s in SEEDS]).astype(np.float64)
    g = load(c, "geom", "g")
    B.require(iou.shape == (3, gt.n, g.shape[1]), "E1 tables")
    y = {"hit75": (iou >= 0.75).astype(np.float64), "hit50": (iou >= 0.5).astype(np.float64), "q50": q}
    dy = {k: v[:, :, 1:] - v[:, :, :1] for k, v in y.items()}
    dx = np.log(g[:, 1:]) - np.log(g[:, :1])
    return iou, q, g, y, dy, dx


def _slope_rows(dx, dy_seeds, gt, w, groups):
    rows = []
    for name, sel in groups:
        for sign, sm in (("magnified", dx > 0), ("compressed", dx < 0)):
            m = sel[:, None] & sm
            row = {"group": name, "side": sign, "pairs": int(m.sum())}
            for s in SEEDS:
                row[f"s{s}"] = T.slope(dx, dy_seeds[s], gt.image, 548, w, m)
            row["mean_of_seeds"] = T.slope(dx, dy_seeds.mean(0), gt.image, 548, w, m)
            rows.append(row)
    return rows


def r14(c, out, counts):
    B.PHASE["phase"] = "r14/load"
    gt = GT(c)
    w = T.boot_weights(548)
    iou, q, g, y, dy, dx = _e1(c, gt)
    inblock = load(c, "geom", "inblock")
    k_all = dx.shape[1]
    f = gt.features()
    g0b = T.g_bin(f["g0"])
    im = gt.image
    res = {"pairs": {"total": int(dx.size), "magnified": int((dx > 0).sum()), "compressed": int((dx < 0).sum()), "zero": int((dx == 0).sum()),
                     "abs_dlogg_quantiles_10_50_90": [float(v) for v in np.quantile(np.abs(dx), [0.1, 0.5, 0.9])],
                     "in_block_share": float(inblock.mean())}}

    # ---------------- R1
    B.PHASE["phase"] = "r14/R1"
    size_groups = [(n, gt.size == i) for i, n in enumerate(T.SIZE_NAMES)] + [("small", gt.small), ("medium", gt.medium)]
    g_groups = [(f"small, {n}", gt.small & (g0b == i)) for i, n in enumerate(T.G_NAMES)]
    full = [(f"{sn} | {gn}", (gt.size == i) & (g0b == j)) for i, sn in enumerate(T.SIZE_NAMES) for j, gn in enumerate(T.G_NAMES)]
    res["R1"] = {"unit": "pp per 0.1 log g (q50: score points x 100)",
                 **{k: {"by_size": _slope_rows(dx, dy[k], gt, w, size_groups), "small_by_g0": _slope_rows(dx, dy[k], gt, w, g_groups)} for k in dy},
                 "hit75_by_size_and_g0": _slope_rows(dx, dy["hit75"], gt, w, full)}

    # ---------------- R2
    B.PHASE["phase"] = "r14/R2"
    base = y["hit75"][:, :, 0].sum(0)
    diff = np.where(base == 0, 0, np.where(base == 3, 2, 1))
    amp = dx >= T.AMPLIFIED
    hits2 = y["hit75"][:, :, 1:].sum(0) >= 2
    z = load(c, "e0", "outcomes")
    s15 = np.stack([z[f"iou_S15-s{s}"] >= 0.75 for s in SEEDS])
    r2 = {"groups": {}}
    for k, name in enumerate(("hard (0 of 3)", "middle (1-2)", "easy (3 of 3)")):
        for scope, sc in (("all", np.ones(gt.n, dtype=bool)), ("small", gt.small)):
            sel = (diff == k) & sc
            m = sel[:, None] & amp
            r2["groups"][f"{name} / {scope}"] = {
                "n": int(sel.sum()), "amplified_pairs": int(m.sum()),
                "pairs_hit_in_ge2_seeds_pct": T.ratio((hits2 & m).sum(1), m.sum(1), im, 548, w, scale=100.0),
                "gt_with_an_amplified_pair": int((sel & amp.any(1)).sum()),
                "gt_hit_in_ge2_seeds_under_some_amplified_pair_pct": T.ratio(sel & (hits2 & amp).any(1), sel & amp.any(1), im, 548, w, scale=100.0),
                "S15_hit75_pct_per_seed": [float(100 * s15[s][sel].mean()) if sel.any() else None for s in SEEDS],
                "S15_hit75_in_ge2_seeds_pct": float(100 * (s15[:, sel].sum(0) >= 2).mean()) if sel.any() else None}
    hard = (diff == 0) & amp.any(1)
    label = (hits2 & amp).any(1)
    phi2 = T.m2_basis(f)
    pred_all = np.full(gt.n, np.nan)
    for fold in (0, 1):
        tr, te = hard & (gt.fold == fold), hard & (gt.fold != fold)
        theta = np.linalg.lstsq(phi2[tr], label[tr].astype(np.float64), rcond=None)[0]
        pred_all[te] = phi2[te] @ theta
    r2["auc_hard_hit_after_magnification"] = {
        "population": "hard GT with at least one pair of dlog g >= 0.10; label = hit75 in >= 2 seeds under some such pair",
        "n": int(hard.sum()), "positives": int(label[hard].sum()),
        "unperturbed_q50_mean_of_seeds": T.auc(q[:, hard, 0].mean(0), label[hard]),
        "unperturbed_iou_max_mean_of_seeds": T.auc(iou[:, hard, 0].mean(0), label[hard]),
        "size_s_only": T.auc(f["s"][hard], label[hard]),
        "all_features_cross_fitted_linear": T.auc(pred_all[hard], label[hard]),
        "note": "an AUC below 0.5 means the predictor ranks in the other direction"}
    odd, even = np.arange(0, k_all, 2), np.arange(1, k_all, 2)       # columns of k = 1, 3, ... and k = 2, 4, ...
    rel = {}
    for name, sel in (("all", np.ones(gt.n, dtype=bool)), ("small", gt.small)):
        rows = {}
        for s in SEEDS:
            a, b = T.per_gt_slope(dx, dy["hit75"][s], odd), T.per_gt_slope(dx, dy["hit75"][s], even)
            rows[f"s{s}_odd_even"] = T.corr(T.corr_sums(a, b, im, 548, sel), w)
        dm = dy["hit75"].mean(0)
        a, b = T.per_gt_slope(dx, dm, odd), T.per_gt_slope(dx, dm, even)
        rows["mean_of_seeds_odd_even"] = T.corr(T.corr_sums(a, b, im, 548, sel), w)
        sl = [T.per_gt_slope(dx, dy["hit75"][s]) for s in SEEDS]
        for a_, b_ in ((0, 1), (0, 2), (1, 2)):
            rows[f"seed_s{a_}-s{b_}"] = T.corr(T.corr_sums(sl[a_], sl[b_], im, 548, sel), w)
        rel[name] = rows
    r2["per_gt_slope_reliability"] = rel
    res["R2"] = r2

    # ---------------- R3
    B.PHASE["phase"] = "r14/R3"
    dm = dy["hit75"].mean(0)
    bases = {"M0": np.ones((gt.n, 1)), "M1": T.m1_basis(f["s"]), "M2": phi2}
    beta, thetas = {}, {}
    for name, phi in bases.items():
        beta[name] = np.empty(gt.n)
        for fold in (0, 1):
            theta = T.fit_slope_model(phi[gt.fold == fold], dx[gt.fold == fold], dm[gt.fold == fold])
            beta[name][gt.fold != fold] = phi[gt.fold != fold] @ theta
            thetas[f"{name}_fit_on_{'AB'[fold]}"] = [float(v) for v in theta]
    slope_gt = T.per_gt_slope(dx, dm)
    r3 = {"coefficients": thetas, "models": {}}
    sums = {}
    for name in bases:
        pred = beta[name][:, None] * dx
        sums[name] = T.corr_sums(pred, dm, im, 548)
        r3["models"][name] = {
            "pair_correlation_mean_of_seeds": T.corr(sums[name], w),
            "pair_correlation_per_seed": [T.corr(T.corr_sums(pred, dy["hit75"][s], im, 548), w) for s in SEEDS],
            "pair_correlation_small": T.corr(T.corr_sums(pred, dm, im, 548, np.broadcast_to(gt.small[:, None], dx.shape)), w),
            "gt_correlation_with_measured_slope": T.corr(T.corr_sums(beta[name], slope_gt, im, 548), w),
            "deciles": T.deciles(pred, dm),
            "beta_pp_per_0.1_quantiles_5_50_95": [float(10 * v) for v in np.quantile(beta[name], [0.05, 0.5, 0.95])]}
    inc = T.corr_difference(sums["M2"], sums["M1"], w)
    inc_small = T.corr_difference(T.corr_sums(beta["M2"][:, None] * dx, dm, im, 548, np.broadcast_to(gt.small[:, None], dx.shape)),
                                  T.corr_sums(beta["M1"][:, None] * dx, dm, im, 548, np.broadcast_to(gt.small[:, None], dx.shape)), w)
    gt_inc = T.corr_difference(T.corr_sums(beta["M2"], slope_gt, im, 548), T.corr_sums(beta["M1"], slope_gt, im, 548), w)
    r3["M2_minus_M1"] = {"pair_correlation": inc, "pair_correlation_small": inc_small, "gt_correlation": gt_inc,
                         "split_half_reliability_mean_of_seeds": rel["all"]["mean_of_seeds_odd_even"],
                         "note": "the split-half correlation bounds how much of the per-GT slope any model can explain"}
    x, yv = np.where(np.isfinite(dx), dx, 0.0), dm
    num = np.stack([T.by_image(((x * yv).sum(1)) * (gt.size == i), im, 548) for i in range(5)], 1)
    den = np.stack([T.by_image(((x * x).sum(1)) * (gt.size == i), im, 548) for i in range(5)], 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        point = num.sum(0) / den.sum(0)
        samples = (w @ num) / (w @ den)
        share, share_s = point / point.sum(), samples / samples.sum(1, keepdims=True)
    om = np.array([T.omega(f["s"][gt.size == i]).mean() for i in range(5)])
    om_share = om / om.sum()
    ci = [T.interval(share_s[:, i]) for i in range(5)]
    inside = [bool(ci[i][0] is not None and ci[i][0] <= om_share[i] <= ci[i][1]) for i in range(5)]
    grid = np.array([4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160], dtype=np.float64)
    r3["M1_shape"] = {"bins": list(T.SIZE_NAMES), "slope_pp_per_0.1": [float(10 * v) for v in point], "slope_share": [float(v) for v in share],
                      "slope_share_ci95": ci, "omega_bin_mean": [float(v) for v in om], "omega_share": [float(v) for v in om_share],
                      "omega_inside_ci": inside, "agrees": bool(all(inside)),
                      "curve": {"s": grid.tolist(), "omega": T.omega(grid).tolist(),
                                **{f"M1_fit_on_{'AB'[fold]}_pp_per_0.1": [float(10 * v) for v in T.m1_basis(grid) @ np.array(thetas[f"M1_fit_on_{'AB'[fold]}"])]
                                   for fold in (0, 1)}}}
    res["R3"] = r3

    # ---------------- R4
    B.PHASE["phase"] = "r14/R4"
    centre = load(c, "geom", "centre")
    up = centre[:, :k_all, 2].astype(bool)
    r4 = {}
    n_in_small = np.zeros((548, k_all))
    n_in_any = np.zeros((548, k_all))
    for k in range(k_all):
        np.add.at(n_in_small[:, k], im, (inblock[:, k] & gt.small).astype(np.float64))
        np.add.at(n_in_any[:, k], im, inblock[:, k].astype(np.float64))
    content = np.where(n_in_any == 0, 0, np.where(n_in_small == 0, 1, np.where(n_in_small <= 5, 2, 3)))
    content_names = ("no GT in block", "GT but no small GT", "1-5 small GT", ">5 small GT")
    per_seed = {}
    for s in list(SEEDS) + ["mean"]:
        d = dy["hit75"].mean(0) if s == "mean" else dy["hit75"][s]

        def tot(mask):
            o = np.zeros((548, k_all))
            for k in range(k_all):
                np.add.at(o[:, k], im, d[:, k] * mask[:, k])
            return o

        parts = {"in_block": tot(inblock), "out_of_block": tot(~inblock), "magnified": tot(dx > 0), "compressed": tot(dx < 0),
                 "small": tot(np.broadcast_to(gt.small[:, None], dx.shape)), "medium_large": tot(np.broadcast_to(~gt.small[:, None], dx.shape))}
        net = parts["in_block"] + parts["out_of_block"]
        rows = {}
        for dname, dm_ in (("multiply", up), ("divide", ~up), ("both", np.ones_like(up))):
            row = {"perturbations": int(dm_.sum()), **{k: float(v[dm_].mean()) for k, v in parts.items()}, "net": float(net[dm_].mean()),
                   "share_net_negative_pct": float(100 * (net[dm_] < 0).mean()), "share_net_zero_pct": float(100 * (net[dm_] == 0).mean()),
                   "share_net_positive_pct": float(100 * (net[dm_] > 0).mean()),
                   "out_of_block_loss_per_in_block_gain": (None if parts["in_block"][dm_].sum() == 0
                                                           else float(-parts["out_of_block"][dm_].sum() / parts["in_block"][dm_].sum())),
                   "by_block_content": [{"content": content_names[k], "perturbations": int((dm_ & (content == k)).sum()),
                                         "in_block": float(parts["in_block"][dm_ & (content == k)].mean()) if (dm_ & (content == k)).any() else None,
                                         "out_of_block": float(parts["out_of_block"][dm_ & (content == k)].mean()) if (dm_ & (content == k)).any() else None,
                                         "net": float(net[dm_ & (content == k)].mean()) if (dm_ & (content == k)).any() else None,
                                         "share_net_negative_pct": float(100 * (net[dm_ & (content == k)] < 0).mean()) if (dm_ & (content == k)).any() else None}
                                        for k in range(4)]}
            rows[dname] = row
        per_seed[f"s{s}" if s != "mean" else "mean_of_seeds"] = rows
    r4 = {"unit": "sum of d hit75 over the GT of one image per perturbation (GT counts, not pp)", "per_seed": per_seed}
    res["R4"] = r4

    # ---------------- value maps and geometry of the four sources
    B.PHASE["phase"] = "r14/maps"
    zz = gt.z
    fa = gt.features(scored_only=False)
    phi_all = T.m2_basis(fa)
    fold_all = T.folds(c["development_ids"])[zz["image"]]
    beta_all = np.empty(len(fold_all))
    m1_all = np.empty(len(fold_all))
    for fold in (0, 1):
        sel = fold_all != fold
        beta_all[sel] = phi_all[sel] @ np.array(thetas[f"M2_fit_on_{'AB'[fold]}"])
        m1_all[sel] = T.m1_basis(fa["s"][sel]) @ np.array(thetas[f"M1_fit_on_{'AB'[fold]}"])
    B.require(bool(np.allclose(beta_all[zz["scored"]], beta["M2"])), "cross-fitted slopes of the scored rows")
    d_star = load(c, "geom", "d_star")
    d_hat = load(c, "geom", "maps")[0]
    wgt = np.clip(beta_all, 0, None)
    v_star = np.stack([T.blur3(T.weighted_demand(zz["norm"][zz["image"] == i], wgt[zz["image"] == i])) for i in range(548)])
    d_map = np.stack([T.rank_remap(v_star[i], d_star[i]) for i in range(548)])
    B.require(all(np.array_equal(np.sort(d_map[i].ravel()), np.sort(d_star[i].ravel())) for i in range(548)), "rank remap value sets")
    boxes = [zz["norm"][zz["image"] == i] for i in range(548)]
    pad, lengths = _pad_boxes(boxes), [len(b) for b in boxes]
    sc = zz["scored"]
    gs = {}
    for name, m in (("a", d_hat), ("b", d_star), ("c", v_star), ("d", d_map)):
        gs[name] = magnify(m, pad, lengths)[0][sc]
    B.require(bool(np.allclose(gs["a"], g[:, 0])) and bool(np.allclose(gs["b"], zz["g_b"][sc])), "geometry of sources a / b")
    geometry = {}
    for name, v in gs.items():
        geometry[name] = {n: {"n": int(sel.sum()), "g_geomean": float(np.exp(np.log(v[sel]).mean())), "share_g_lt_1": float((v[sel] < 1).mean()),
                              "share_g_lt_1.2": float((v[sel] < 1.2).mean()), "share_g_ge_1.4": float((v[sel] >= 1.4).mean())}
                          for n, sel in size_groups + [("all", np.ones(gt.n, dtype=bool))]}
    div = T.g_bin(gs["d"]) != T.g_bin(gs["b"])
    dl = np.log(gs["d"]) - np.log(gs["b"])
    r1 = {(r["group"], r["side"]): r["mean_of_seeds"]["value"] for r in res["R1"]["hit75"]["by_size"]}
    binned = np.array([(r1[(T.SIZE_NAMES[b], "magnified" if v > 0 else "compressed")] or 0.0) / 10.0 for b, v in zip(gt.size, dl)]) * dl
    predicted = {n: {"M2": float(100 * (beta["M2"] * dl)[sel].mean()), "M1": float(100 * (beta["M1"] * dl)[sel].mean()),
                     "R1_size_by_side_slopes": float(binned[sel].mean())} for n, sel in (("small", gt.small), ("medium", gt.medium))}
    res["sources"] = {
        "geometry": geometry,
        "divergent_set": {"share_of_small": T.ratio(div & gt.small, gt.small, im, 548, w), "small_GT": int((div & gt.small).sum()),
                          "abs_dlogg_d_vs_b_small_quantiles_10_50_90": [float(v) for v in np.quantile(np.abs(dl[gt.small]), [0.1, 0.5, 0.9])],
                          "mean_dlogg_d_vs_b_small": float(dl[gt.small].mean())},
        "predicted_small_hit75_change_d_minus_b_pp": predicted,
        "value_weights": {"share_negative_clipped": float((beta_all < 0).mean()), "share_negative_clipped_small": float((beta_all[zz["area"] <= T.SMALL_AREA] < 0).mean()),
                          "quantiles_5_50_95": [float(v) for v in np.quantile(wgt, [0.05, 0.5, 0.95])],
                          "correlation_with_omega": _pearson(wgt, T.omega(fa["s"])), "boxes": int(len(wgt))},
        "maps": {"cells_changed_d_vs_b_share": float((d_map != d_star).mean()),
                 "per_image_rank_correlation_V_vs_Dstar_median": float(np.median([_pearson(np.argsort(np.argsort(v_star[i].ravel())), np.argsort(np.argsort(d_star[i].ravel()))) or 0.0 for i in range(548)])),
                 "V_sum_over_Dstar_sum": float(v_star.sum() / d_star.sum())}}
    B.write_new(out / "R14.json", res)
    files = {"map_b": X.save_npy(out / "map_b.npy", d_star), "map_c": X.save_npy(out / "map_c.npy", v_star), "map_d": X.save_npy(out / "map_d.npy", d_map),
             "g_sources": X.save_npy(out / "g_sources.npy", np.stack([gs[k] for k in "abcd"], 1)),
             "beta": X.save_npy(out / "beta.npy", np.stack([beta["M1"], beta["M2"], slope_gt], 1)),
             "beta_all": X.save_npy(out / "beta_all.npy", np.stack([m1_all, beta_all], 1))}
    B.require(counts["forwards"] == 0, "r14 must not run a model")
    return {"stage": "r14", "result": B.ref(out / "R14.json"), "files": files, "divergent_share_small": res["sources"]["divergent_set"]["share_of_small"]["value"],
            "M2_minus_M1_pair_correlation": inc, "M1_shape_agrees": r3["M1_shape"]["agrees"]}


# ================================================================ e3 (CPU)
def e3(c, out, counts):
    from sparse_rtdetr.research import r28_fovea as FV
    from sparse_rtdetr.research import r28_fovea_geo as FG
    torch = B.torch_mod()
    B.PHASE["phase"] = "e3"
    gt = GT(c)
    zz = gt.z
    sc = zz["scored"]
    d_star = load(c, "geom", "d_star")
    plan = FG.plan_from_demand(torch.from_numpy(d_star))
    H, Wd = T.L_HW
    tot, empty, per_image = 0.0, 0.0, []
    norm, small = zz["norm"][sc], gt.small
    for i in range(548):
        gx, gy = plan.gx[i].numpy(), plan.gy[i].numpy()
        ox = 1.0 / (np.diff(gx) * (len(gx) - 1))[np.minimum((np.arange(Wd) + 0.5) / Wd * (len(gx) - 1), len(gx) - 2).astype(int)]
        oy = 1.0 / (np.diff(gy) * (len(gy) - 1))[np.minimum((np.arange(H) + 0.5) / H * (len(gy) - 1), len(gy) - 2).astype(int)]
        hot = np.sqrt(oy[:, None] * ox[None, :]) >= 1.2
        mask = np.zeros((H, Wd), dtype=bool)
        b = norm[gt.sl(i)][small[gt.sl(i)]]
        if len(b):
            t = torch.from_numpy(b)
            x1, x2 = (FV.warp(plan.gx[i:i + 1], t[None, :, k])[0].numpy() * Wd for k in (0, 2))
            y1, y2 = (FV.warp(plan.gy[i:i + 1], t[None, :, k])[0].numpy() * H for k in (1, 3))
            for a_, b_, c_, d_ in zip(x1, y1, x2, y2):
                mask[max(0, int(np.floor(b_ - 8))):int(np.ceil(d_ + 8)), max(0, int(np.floor(a_ - 8))):int(np.ceil(c_ + 8))] = True
        tot += hot.sum()
        empty += (hot & ~mask).sum()
        if hot.any():
            per_image.append(float((hot & ~mask).sum() / hot.sum()))
    g_b = zz["g_b"][sc]
    cells = zz["row"][sc] * 21 + zz["col"][sc]
    bound = np.empty(gt.n)
    for i in range(548):
        s = gt.sl(i)
        count = np.bincount(cells[s][small[s]], minlength=252)
        area = T.waterfill_area(count, (2 / 3) ** 2, 1.5 ** 2)
        bound[s] = np.sqrt(area[cells[s]])

    def reading(v):
        v = v[small]
        return {"g_geomean": float(np.exp(np.log(v).mean())), "share_g_lt_1.2": float((v < 1.2).mean()), "share_g_lt_1": float((v < 1).mean())}

    res = {"collateral_magnification": {"definition": "canvas-L pixels whose local magnification under (b) is >= 1.2; share not covered by any "
                                                      "small GT box dilated by 8 canvas pixels",
                                        "pooled_share": float(empty / tot) if tot else None, "per_image_mean": float(np.mean(per_image)) if per_image else None,
                                        "images_with_such_area": len(per_image), "share_of_canvas_ge_1.2": float(tot / (548 * H * Wd))},
           "ideal_2d_allocation": {"definition": "per image, cell area scale a = clip(n_small / lam, 4/9, 9/4) with mean 1 over the 252 cells "
                                                 "(maximises the sum of log g of the small GT; no map is constructed); g = sqrt(a) of the centre cell",
                                   "small_GT_under_b_actual": reading(g_b), "small_GT_upper_bound": reading(bound)}}
    B.write_new(out / "E3.json", res)
    B.require(counts["forwards"] == 0, "e3 must not run a model")
    return {"stage": "e3", "result": B.ref(out / "E3.json"), **res}


# ================================================================ tables (CPU)
def tables(c, out, counts):
    B.PHASE["phase"] = "tables"
    gt = GT(c)
    w = T.boot_weights(548)
    im = gt.image
    z = load(c, "e0", "outcomes")
    r14c = stage_closed(c, "r14")
    r = B.read_json(B.pinned(r14c["result"]))
    e0r = B.read_json(B.pinned(stage_closed(c, "e0")["result"]))
    e3r = B.read_json(B.pinned(stage_closed(c, "e3")["result"]))
    iou, values, stages = {}, {}, {}
    for s in SEEDS:
        iou[("a", s)] = z[f"iou_FG-s{s}"]
        values[("a", s)] = c["units"][f"FG-s{s}"]["values_600"]
    iou[("b", 0)] = z["iou_A2-s0"]
    values[("b", 0)] = c["label_demand_seed0"]["values_600"]
    for u in E2_UNITS:
        src, s = u.split("-s")
        iou[(src, int(s))] = load(c, f"e2-{u}", "iou")
        sc = stage_closed(c, f"score-{u}")
        values[(src, int(s))] = sc["values_600"]
        stages[u] = {"predictions": stage_closed(c, f"e2-{u}")["predictions"], "metrics": sc["metrics"]}
    groups = _size_groups(gt)
    names = {0.5: "hit50", 0.75: "hit75"}
    cover = {src: {f"s{s}": {names[t]: {g: float(100 * (iou[(src, s)] >= t)[sel].mean()) for g, sel in groups.items()} for t in names}
                   for s in SEEDS} for src in "abcd"}
    ap = {src: {f"s{s}": {m: 100.0 * values[(src, s)][m] for m in METRICS} for s in SEEDS} for src in "abcd"}

    def diff(x, y):
        rows = {}
        for s in SEEDS:
            rows[f"s{s}"] = {names[t]: {g: T.ratio(((iou[(x, s)] >= t).astype(float) - (iou[(y, s)] >= t)) * sel, sel, im, 548, w, scale=100.0)
                                        for g, sel in groups.items()} for t in names}
            rows[f"s{s}"]["AP_pp"] = {m: ap[x][f"s{s}"][m] - ap[y][f"s{s}"][m] for m in METRICS}
        mean = lambda fn: float(np.mean([fn(rows[f"s{s}"]) for s in SEEDS]))
        d3 = np.mean([(iou[(x, s)] >= 0.75).astype(float) - (iou[(y, s)] >= 0.75) for s in SEEDS], 0)
        rows["mean_of_seeds"] = {**{names[t]: {g: mean(lambda q_, t=t, g=g: q_[names[t]][g]["value"]) for g in groups} for t in names},
                                 "AP_pp": {m: mean(lambda q_, m=m: q_["AP_pp"][m]) for m in METRICS},
                                 "hit75_small_ci95": T.ratio(d3 * gt.small, gt.small, im, 548, w, scale=100.0)["ci95"],
                                 "hit75_medium_ci95": T.ratio(d3 * gt.medium, gt.medium, im, 548, w, scale=100.0)["ci95"]}
        return rows

    main, c_d, b_a = diff("d", "b"), diff("c", "d"), diff("b", "a")
    g_src = load(c, "r14", "g_sources")
    div = (T.g_bin(g_src[:, 3]) != T.g_bin(g_src[:, 1])) & gt.small
    same = gt.small & ~div
    split = {f"s{s}": {name: T.ratio(((iou[("d", s)] >= 0.75).astype(float) - (iou[("b", s)] >= 0.75)) * sel, sel, im, 548, w, scale=100.0)
                       for name, sel in (("divergent small GT", div), ("other small GT", same))} for s in SEEDS}
    readings = {"small_hit75_pp": [main[f"s{s}"]["hit75"]["small"]["value"] for s in SEEDS],
                "medium_hit75_pp": [main[f"s{s}"]["hit75"]["medium"]["value"] for s in SEEDS],
                "aps_pp": [main[f"s{s}"]["AP_pp"]["APS"] for s in SEEDS], "ap75s_pp": [main[f"s{s}"]["AP_pp"]["AP75_small"] for s in SEEDS],
                "m2_increment_ci_contains_zero": r["R3"]["M2_minus_M1"]["pair_correlation"]["ci_contains_zero"],
                "m1_shape_agrees": r["R3"]["M1_shape"]["agrees"], "divergent_share": r["sources"]["divergent_set"]["share_of_small"]["value"]}
    verdict = {**T.verdict(readings), "readings": readings, "dev_round_numbers_are_not_results": c["dev"]}
    gpu = {}
    for name in [f"wire-{u}" for u in FG_UNITS] + ["demand"] + [f"e1-s{s}" for s in SEEDS] + [f"e2-{u}" for u in E2_UNITS]:
        st = stage_closed(c, name)
        gpu[name] = {"forwards": st["counts"]["forwards"], "stage_seconds": st["elapsed_seconds"],
                     "others": sorted(set(st.get("other_gpu_compute_processes_at_start", []) + st.get("other_gpu_compute_processes_at_end", [])))}
    geo = stage_closed(c, "geom")
    result = {"status": "PASS", "version": VERSION, "contract_sha256": c["contract_sha256"], "source_commit": c["source_commit"],
              "source_branch": c["source_branch"], "task": c["task"], "dev": c["dev"], "perturbations": c["perturbations"],
              "verdict": verdict,
              "E2": {"main_d_minus_b": main, "c_minus_d_record_only": c_d, "b_minus_a_reference_only": b_a, "coverage_pp": cover, "AP_pp": ap,
                     "d_minus_b_small_hit75_by_divergence": split, "geometry": r["sources"], "files": stages,
                     "note": "(b), (c), (d) are all mismatched configurations (weights not trained under that demand source); only (d) - (b) is read"},
              "amplitude": {k: geo[k] for k in ("amplitude", "block", "calibration", "realised")},
              "R1": r["R1"], "R2": r["R2"], "R3": r["R3"], "R4": r["R4"], "pairs": r["pairs"], "E0": e0r, "E3": e3r,
              "geometry_checks": {"per_gt_file_max_abs_error": geo["check_against_R30_001_per_gt"], "rows": geo["rows"],
                                  "occlusion_levels": geo["occlusion_levels"], "truncation_levels": geo["truncation_levels"]},
              "workload": {"stages": gpu, "forwards": sum(v["forwards"] for v in gpu.values()),
                           "gpu_stage_seconds": sum(v["stage_seconds"] for v in gpu.values()), "updates": 0,
                           "other_gpu_compute_processes_seen": sorted({p for v in gpu.values() for p in v["others"]}),
                           "score_budget_calls": len(E2_UNITS)},
              "identities": {"units": {u: {k: c["units"][u][k] for k in ("endpoint", "weights_sha256", "predictions_600")} for u in c["units"]},
                             "stored_predictions": c["stored_predictions"], "demand": c["demand"], "mapping": c["mapping"],
                             "R30_contract": c["R30_contract"], "source_manifest": c["source_manifest"]},
              "definitions": c["definitions"]}
    B.require(counts["forwards"] == 0, "tables must not run a model")
    B.write_new(out / "RESULT.json", result)
    return {"stage": "tables", "result": B.ref(out / "RESULT.json"), "label": verdict["label"], "means_pp": verdict["means_pp"],
            "readings": readings}


# ================================================================ CLI
def _wire(c, out, counts, unit):
    return X.wire(c, out, counts, unit)


STAGES = {"wire": (_wire, "cuda", FG_UNITS), "demand": (demand, "cuda", None), "geom": (geom, "cpu", None), "e0": (e0, "cpu", None),
          "e1": (e1, "cuda", tuple(f"s{s}" for s in SEEDS)), "r14": (r14, "cpu", None), "e2": (e2, "cuda", E2_UNITS),
          "score": (score, "cpu", E2_UNITS), "e3": (e3, "cpu", None), "tables": (tables, "cpu", None)}


def unit_seed(unit):
    return 0 if unit is None else int(unit.rsplit("s", 1)[1])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("prepare",) + tuple(STAGES))
    ap.add_argument("--round", required=True, type=Path)
    ap.add_argument("--release")
    ap.add_argument("--unit")
    args = ap.parse_args(argv)
    if args.action == "prepare":
        B.require(args.release and args.unit is None, "prepare needs --release and no --unit")
        return prepare(args.round, args.release)
    run, mode, units = STAGES[args.action]
    B.require((args.unit in units) if units else args.unit is None, f"{args.action}: unit must be one of {units}")
    B.require(args.release is None, "--release is for prepare")
    c = load_contract(args.round)
    name = f"{args.action}-{args.unit}" if units else args.action
    out = Path(c["outputs"]) / name
    out.mkdir(exist_ok=False)
    start, cpu0 = time.monotonic(), time.process_time()
    counts = Counter(actual_updates=0, forwards=0, forward_attempts=0)
    try:
        from r25_run import setup
        if mode == "cuda":
            B.require(os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == "expandable_segments:True", "allocator conf")
            from s15_formal_train import bind_seed
            bind_seed(unit_seed(args.unit))
            setup("cuda:0")
            B.torch_mod().cuda.reset_peak_memory_stats()
        else:
            B.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", f"the {args.action} stage requires CUDA_VISIBLE_DEVICES=\"\"")
            setup("cpu")
        result = run(c, out, counts, args.unit) if units else run(c, out, counts)
        B.require(isinstance(result, dict) and counts["actual_updates"] == 0, f"{args.action}: stage result")
        B.write_new(out / "CLOSED.json", {"status": "PASS", "action": args.action, "contract_sha256": c["contract_sha256"],
                                          "counts": dict(counts), "elapsed_seconds": time.monotonic() - start,
                                          "process_cpu_seconds": time.process_time() - cpu0, **result})
    except BaseException as exc:
        B.write_new(out / "STOP.json", {"status": "STOP", "phase": B.PHASE["phase"], "action": args.action, "unit": args.unit,
                                        "error": repr(exc)[:4000], "traceback": traceback.format_exc(),
                                        "counts": dict(counts), "elapsed_seconds": time.monotonic() - start})
        raise
    finally:
        gc.collect()


if __name__ == "__main__":
    main()
