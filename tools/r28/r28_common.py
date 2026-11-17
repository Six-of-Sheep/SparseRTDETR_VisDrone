"""R28 runner, shared layer: constants, contract, demand-source references, state guards.

Arms W (frozen demand network), K (fixed train_core mean demand), F (arm W's demand network with the FOVEA-style
map, TASK_R28_FOVEA_MAPPING_CONTROL_004), FS (arm F with the calibrated kernel width,
TASK_R28_CALIBRATED_F_CONTROL_006), FG (arm F with the fixed geometry-selected kernel width,
TASK_R28_STRONG_F_CONTROL_009) and QP / QC (TASK_R29_001: arm FG with the query-wise un-zoomed decoder read, direct
and dense; reported as P and C); the arm of a process comes from R28_ARM.
Pure helpers (hashing, pinned references, ledger comparison) are the L0 ones, imported from tools/r27_l0/l0_common.py;
nothing here writes into the L0 round.  The L0 round stays reachable through l0_common with its own default round,
so the environment variable L0_ROUND must not be set in an R28 process.
"""
import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tools" / "r27_l0"), str(ROOT / "tools" / "r27")]
import l0_common as B  # noqa: E402  (adds src, tools/query_local, tools/scale_baseline and ROOT to sys.path)

require, canonical, sha_bytes, file_sha, ref, pinned = B.require, B.canonical, B.sha_bytes, B.file_sha, B.ref, B.pinned
write_new, read_json, torch_mod, cells, side_of = B.write_new, B.read_json, B.torch_mod, B.cells, B.side_of
shared, compare_shared, register_counts = B.shared, B.compare_shared, B.register_counts
INPUT_FIELDS, TOP_FIELDS, SHARED_FIELDS = B.INPUT_FIELDS, B.TOP_FIELDS, B.SHARED_FIELDS
PARENT, PARENT_SHA, PARENT_UPDATE, ENDPOINT, PYTHON = B.PARENT, B.PARENT_SHA, B.PARENT_UPDATE, B.ENDPOINT, B.PYTHON
WINDOWS_PER_EPOCH, SAVE_EPOCHS, FORMAL_SAVE_EPOCHS = B.WINDOWS_PER_EPOCH, B.SAVE_EPOCHS, B.FORMAL_SAVE_EPOCHS
EVAL_BATCH, EVAL_WARMUP, EVAL_BATCHES, BUDGETS = B.EVAL_BATCH, B.EVAL_WARMUP, B.EVAL_BATCHES, B.BUDGETS
COST_BATCH, COST_WARMUP, COST_PASSES, COST_BATCHES_PER_PASS = B.COST_BATCH, B.COST_WARMUP, B.COST_PASSES, B.COST_BATCHES_PER_PASS
COST_PER_ENDPOINT = B.COST_PER_ENDPOINT
PHASE = B.PHASE

require("L0_ROUND" not in os.environ, "L0_ROUND must not be set in an R28 process (the L0 round is read-only input)")
LD_ROOT = B.LD_ROOT
ROUND = Path(os.environ.get("R28_ROUND", str(LD_ROOT / "r28-budgeted-resampling-20261113-001")))
DEV = ROUND.resolve().name.startswith("dev-round")
ARMS = ("W", "K", "F", "FS", "FG", "QP", "QC")
R29_ARMS = ("QP", "QC")                 # TASK_R29_001: P (direct read) and C (dense un-zoom control) on the arm FG input
ARM = os.environ.get("R28_ARM", "W")
require(ARM in ARMS, f"R28_ARM must be W, K, F, FS, FG, QP or QC, got {ARM!r}")
SCHEMA = "r28-state-v1"

L0_ROUND = LD_ROOT / "r27-l0-control-20261111-001"
L0_FILES = {"contract": (L0_ROUND / "CONTRACT.json", "4c4eeba108eb58e3f75690d2a32f3ac2673cbc7d8ccf58101e16d9d392ad87f8"),
            "result": (L0_ROUND / "summary/RESULT.json", "eece80d88a276df91427223ef4d42ee1fd02d5e7699ee69f91addfa4ac08052c"),
            "manifest": (L0_ROUND / "release-001/MANIFEST.json", "41135d848308deda9ea61ce7c2722a4635fba435201cf7782124dd00bde06ef6")}
TASKS = {"TASK_R28_001.md": "903db45fcb0b2f3ddf0aa610389ed3f3f90872b0576bd65a7e84d053bb1584e2",
         "CANDIDATE_R28_001.md": "d7b6704e298b40df247a816187b187a110530c9c9f2814a635f9066bc24a1249",
         "TASK_R28_CODEX_AMENDMENT_002.md": "b53a7adaf93119eaca56b02c12f84eeb40c2fce6333321a8759d958330d12e95"}

# ---- fixed rules (TASK_R28_001 section 7, AMENDMENT_002 sections 3 and 5)
GATE_M = B.GATE_D                       # W - L0, pp: the seven cells of the L0 D condition
GATE_A = ("600", "APS", 0.30)           # W - K, pp
APL6_CLASSES = (1, 4, 5, 6, 7, 9)
APL6_MINIMUM = -0.50
COST_LIMIT = 1.15                       # model GPU time W / L0
G0_MINIMUM = 1.208                      # geometric-mean magnification of certified small GT on development
SMALL_AREA, LARGE_AREA = 32.0 ** 2, 96.0 ** 2   # COCO ranges on the raw area, inclusive end points as in COCOeval

DEMAND = {"epochs": 30, "strong_epochs": 24, "pipeline_epochs": list(range(1, 25)) + [118, 119, 120, 118, 119, 120],
          "batch": 16, "windows_per_epoch": WINDOWS_PER_EPOCH, "optimizer": "AdamW", "lr": 1e-3, "lr_end": 1e-5,
          "weight_decay": 1e-4, "schedule": "cosine per step from lr to lr_end over all steps", "precision": "float32",
          "seed": 0, "loss": "smooth_l1(log1p(pred), log1p(target))", "endpoint": "last step, no selection",
          "data": "train_core through the R27/S15 H augmentation pipeline (seed 0); demand target from the augmented "
                  "valid GT; image reduced to 384x672 in the loader worker", "workers": 12,
          "note": "pipeline epochs 1-24 carry the strong augmentation (epoch < 118), 118-120 do not; the three "
                  "non-strong epochs are used twice"}
DEMAND["steps"] = DEMAND["epochs"] * WINDOWS_PER_EPOCH

PRECOST_ENDPOINTS = ("L0", "Wpre")
PRECOST_ROTATIONS = (("L0", "Wpre"), ("Wpre", "L0"), ("L0", "Wpre"))
COST_ENDPOINTS = {"cost": ("A0", "L0", "W", "S15"), "cost-K": ("L0", "W", "K"), "cost-F": ("L0", "W", "F"),
                  "cost-FS": ("L0", "K", "F", "FS", "W"), "cost-FG": ("L0", "K", "F", "FG", "W"),
                  "cost-R29": ("FG", "QC", "QP")}
COST_ROTATIONS = {"cost": (("A0", "L0", "W", "S15"), ("W", "S15", "A0", "L0"), ("L0", "A0", "S15", "W")),
                  "cost-K": (("L0", "W", "K"), ("K", "L0", "W"), ("W", "K", "L0")),
                  "cost-F": (("L0", "W", "F"), ("F", "L0", "W"), ("W", "F", "L0")),
                  # task 006 section 4: the fixed order, its reverse, the fixed order rotated left by two
                  "cost-FS": (("L0", "K", "F", "FS", "W"), ("W", "FS", "F", "K", "L0"), ("F", "FS", "W", "L0", "K")),
                  # task 009 section 4: the same three orders with FG in the place of FS
                  "cost-FG": (("L0", "K", "F", "FG", "W"), ("W", "FG", "F", "K", "L0"), ("F", "FG", "W", "L0", "K")),
                  # TASK_R29_001 section 5: N-C-P, P-C-N, C-P-N (N is arm FG)
                  "cost-R29": (("FG", "QC", "QP"), ("QP", "QC", "FG"), ("QC", "QP", "FG"))}
EVALPATH_EXTRA = 2                      # non-identity deployment batches: one wrapped and one manual forward each

WORKLOAD = {
    "cpu": {"updates": 3, "forwards": 6},
    "demand": {"updates": 0, "forwards": 0, "demand_steps": DEMAND["steps"], "demand_forwards": DEMAND["steps"]},
    "prior": {"updates": 0, "forwards": 0},
    "g0": {"updates": 0, "forwards": 0, "demand_forwards_development": 137},
    "zero": {"updates": 0, "forwards": 4},
    "accept": {"updates": 3, "forwards": 6},
    "evalpath": {"updates": 0, "forwards": EVAL_WARMUP + EVAL_BATCHES + 2 * EVALPATH_EXTRA},
    "precost": {"updates": 0, "forwards": len(PRECOST_ENDPOINTS) * COST_PER_ENDPOINT},
    "formal": {"updates": ENDPOINT - PARENT_UPDATE, "forwards": 2 * (ENDPOINT - PARENT_UPDATE),
               "save_epochs": list(SAVE_EPOCHS), "task_save_epochs": list(FORMAL_SAVE_EPOCHS)},
    "evaluation": {"endpoint": "e120 EMA", "view": "A", "images": 548, "batch": EVAL_BATCH, "warmup": EVAL_WARMUP,
                   "forwards": EVAL_WARMUP + EVAL_BATCHES, "budgets": list(BUDGETS), "score_budget_calls": 2,
                   "per_class_APL_COCOeval_calls": 1},
    "cost": {"forwards": len(COST_ENDPOINTS["cost"]) * COST_PER_ENDPOINT},
    "cost-K": {"forwards": len(COST_ENDPOINTS["cost-K"]) * COST_PER_ENDPOINT},
    "cost-F": {"forwards": len(COST_ENDPOINTS["cost-F"]) * COST_PER_ENDPOINT},
    "cost-FS": {"forwards": len(COST_ENDPOINTS["cost-FS"]) * COST_PER_ENDPOINT},
    "cost-FG": {"forwards": len(COST_ENDPOINTS["cost-FG"]) * COST_PER_ENDPOINT},
    "cost-R29": {"forwards": len(COST_ENDPOINTS["cost-R29"]) * COST_PER_ENDPOINT},
}
LABELS = {
    "G0_failed": "本固定需求预测未达到投入要求",
    "M_passed": "M_passed",
    "M_failed_APL": "M_failed_APL; continue_K_exploratory",
    "closed": "closed_fixed_config",
    "A_passed": "adaptive_increment_established",
    "A_failed": "adaptive_increment_not_established",
}


# ================================================================ follow-up rounds (TASK_R28_K_FOLLOWUP_003)
FOLLOWUP_BASIS = "post_W_user_approved_K_followup"
FOVEA_BASIS = "post_K_user_forwarded_mapping_control"       # TASK_R28_FOVEA_MAPPING_CONTROL_004
CAL_BASIS = "post_F_user_forwarded_calibrated_mapping_control"   # TASK_R28_CALIBRATED_F_CONTROL_006
GEO_BASIS = "post_006_user_forwarded_fixed_geometry_selected_width_control"   # TASK_R28_STRONG_F_CONTROL_009
R29_BASIS = "post_009_user_forwarded_query_unzoom_read"                          # TASK_R29_001
# the only arm a round of each basis trains; the R29 round trains two (R29_ARMS) and is marked by a name no arm has
FOLLOWUP_ARMS = {FOLLOWUP_BASIS: "K", FOVEA_BASIS: "F", CAL_BASIS: "FS", GEO_BASIS: "FG", R29_BASIS: "R29"}


def r29(c):
    """True for the contract of the R29 round (arms QP and QC trained, arm FG read as N from its closed round)."""
    return c.get("authorization_basis") == R29_BASIS and followup(c) is not None


def followup(c):
    """The follow-up block of a contract written by r28_followup.prepare or r28_fovea_control.prepare, or None for
    the original R28 contract."""
    fu = c.get("followup")
    require(fu is None or (fu.get("authorization_basis") == c.get("authorization_basis") and c.get("authorization_basis") in FOLLOWUP_ARMS
                           and fu.get("arm") in (None, FOLLOWUP_ARMS[c["authorization_basis"]])),
            "follow-up contract without the registered authorization basis")
    return fu


def followup_arm(c):
    """The arm a follow-up round trains (K, F, FS or FG; "R29" for the two-arm R29 round), or None for the original
    R28 contract."""
    return None if followup(c) is None else FOLLOWUP_ARMS[c["authorization_basis"]]


def arm_module(arm):
    """The science module of an arm: r28_resample for W / K, r28_fovea for F, r28_fovea_cal for FS, r28_fovea_geo for
    FG (same create / build_eval_model interface)."""
    require(arm in ARMS, f"arm {arm}")
    if arm in R29_ARMS:
        from sparse_rtdetr.research import r29_unzoom as M
    elif arm == "FG":
        from sparse_rtdetr.research import r28_fovea_geo as M
    elif arm == "FS":
        from sparse_rtdetr.research import r28_fovea_cal as M
    elif arm == "F":
        from sparse_rtdetr.research import r28_fovea as M
    else:
        from sparse_rtdetr.research import r28_resample as M
    return M


def prior_contract(c):
    """The contract of the closed R28 round a follow-up refers to (read-only; pinned by file SHA and inner digest)."""
    fu = followup(c)
    require(fu is not None, "not a follow-up contract")
    c1 = read_json(pinned(fu["prior_contract"]))
    body = {k: v for k, v in c1.items() if k != "contract_sha256"}
    require(c1["contract_sha256"] == sha_bytes(canonical(body)) == fu["prior_contract_sha256"], "prior R28 contract digest")
    return c1


def other_contract(c, which):
    """The contract of the K round (which="K"), of the F round (which="F") or of the F_geo round (which="FG", R29 only)
    a later control refers to: read-only, pinned by file SHA and inner digest."""
    fu = followup(c)
    require(fu is not None and which in ("K", "F", "FG") and f"{which}_contract" in fu, f"this contract has no reference to the {which} round")
    cx = read_json(pinned(fu[f"{which}_contract"]))
    body = {k: v for k, v in cx.items() if k != "contract_sha256"}
    require(cx["contract_sha256"] == sha_bytes(canonical(body)) == fu[f"{which}_contract_sha256"], f"{which} round contract digest")
    return cx


# ================================================================ L0 round (read-only input)
def l0_contract(c=None):
    """The L0 contract, verified by file SHA and inner digest (it is not `loaded`: this is not the L0 release)."""
    path, sha = L0_FILES["contract"]
    r = ref(path)
    require(r["sha256"] == sha and (c is None or c["L0_contract"]["sha256"] == sha), "L0 contract file SHA")
    c0 = read_json(path)
    body = {k: v for k, v in c0.items() if k != "contract_sha256"}
    require(c0["contract_sha256"] == sha_bytes(canonical(body)), "L0 contract inner digest")
    return c0


def l0_eval_closed(c0):
    closed = read_json(L0_ROUND / "eval-L0/CLOSED.json")
    require(closed["status"] == "PASS" and closed["arm"] == "L0" and closed["contract_sha256"] == c0["contract_sha256"],
            "L0 eval-L0 CLOSED.json")
    return closed


# ================================================================ contract
def prepare(_args=None):
    from sparse_rtdetr.research import r28_resample as R
    from sparse_rtdetr.research import r28_warp as W
    require(ROOT.resolve().parent.parent == ROUND.resolve() and ROOT.name == "source", "prepare must run from a frozen release")
    manifest = read_json(ROOT.parent / "MANIFEST.json")
    require(manifest["files"] == B.source_manifest(ROOT), "release source differs from MANIFEST")
    require(not manifest.get("dirty", False) or DEV, "a release assembled from a dirty work tree is only accepted in a dev round")
    tasks = {}
    for name, sha in TASKS.items():
        tasks[name] = ref(ROUND / name)
        require(tasks[name]["sha256"] == sha, f"task file SHA: {name}")
    files = {k: ref(p) for k, (p, _) in L0_FILES.items()}
    require(all(files[k]["sha256"] == sha for k, (_, sha) in L0_FILES.items()), "L0 contract / result / manifest SHA")
    c0, result = l0_contract(), read_json(L0_FILES["result"][0])
    require(result["contract_sha256"] == c0["contract_sha256"] and result["status"] == "PASS", "L0 RESULT.json")
    require(c0["parent"]["sha256"] == PARENT_SHA and c0["python"] == PYTHON, "L0 contract parent/python")
    ev = l0_eval_closed(c0)
    require(ev["values"] == result["values"]["L0"] and ev["endpoint"] == result["endpoint"]["file"], "L0 evaluation readings")
    for k in ("A0_endpoint", "S15_endpoint"):
        pinned(c0[k])
    for k in ("A0_metrics", "S15_metrics"):
        for b in ("600", "300"):
            pinned(c0[k][b])
    for b in ("600", "300"):
        pinned(ev["predictions"][b]), pinned(ev["metrics"][b])
    pinned(ev["endpoint"])
    reference = {"values": {**c0["reference"]["values"], "L0": ev["values"]},
                 "per_class_APL_top600": {**c0["reference"]["per_class_APL_top600"], "L0": ev["per_class_APL_top600"]},
                 "GT_counts": ev["GT_counts"],
                 "sources": {"P/C/S15/A0": "R27-001 certified readings via the L0 contract (reused)",
                             "L0": "R27-002 eval-L0 CLOSED.json (reused)"}}
    require(set(reference["values"]) == {"P", "C", "S15", "A0", "L0"}, "reference arms")
    parent = ref(PARENT)
    c = {"version": "R28-001-contract-v1", "tasks": tasks, "parent": parent, "parent_update": PARENT_UPDATE,
         "L0_contract": {**files["contract"], "contract_sha256": c0["contract_sha256"], "identity_sha256": c0["identity_sha256"]},
         "L0_result": files["result"], "L0_manifest": files["manifest"],
         "L0_endpoint": ev["endpoint"], "L0_endpoint_side": ref(side_of(ev["endpoint"]["path"])),
         "L0_eval_closed": ref(L0_ROUND / "eval-L0/CLOSED.json"), "L0_weights_sha256": ev["weights_sha256"],
         "L0_predictions": ev["predictions"], "L0_metrics": ev["metrics"],
         "R27_updates": c0["R27_updates"],
         "A0_endpoint": c0["A0_endpoint"], "A0_metrics": c0["A0_metrics"],
         "S15_endpoint": c0["S15_endpoint"], "S15_metrics": c0["S15_metrics"],
         "data": c0["data"], "data_receipt": c0["data_receipt"], "development_ids": c0["development_ids"],
         "monitor_rule": c0["monitor_rule"], "monitor_ids": c0["monitor_ids"], "monitor_sha256": c0["monitor_sha256"],
         "evaluation_policy": c0["evaluation_policy"], "environment": c0["environment"], "python": c0["python"],
         "gpu_uuid": c0["gpu_uuid"], "reference": reference,
         "identity": R.IDENTITY, "identity_sha256": R.IDENTITY_SHA256, "definition": W.DEFINITION,
         "source_root": str(ROOT.resolve()), "source_manifest": ref(ROOT.parent / "MANIFEST.json"),
         "source_commit": manifest["commit"], "source_dirty": bool(manifest.get("dirty", False)),
         "demand_training": DEMAND, "workload": WORKLOAD,
         "gates": {"G0_small_geomean_minimum": G0_MINIMUM, "M_W_minus_L0": GATE_M, "C_model_gpu_W_over_L0_maximum": COST_LIMIT,
                   "APL6_classes": list(APL6_CLASSES), "APL6_minimum_pp": APL6_MINIMUM, "A_W_minus_K": list(GATE_A)},
         "cost": {"precost": {"endpoints": list(PRECOST_ENDPOINTS), "rotations": [list(r) for r in PRECOST_ROTATIONS]},
                  **{k: {"endpoints": list(COST_ENDPOINTS[k]), "rotations": [list(r) for r in COST_ROTATIONS[k]]} for k in COST_ENDPOINTS},
                  "batch": COST_BATCH, "warmup_per_endpoint": COST_WARMUP, "passes": COST_PASSES,
                  "batches_per_pass": COST_BATCHES_PER_PASS},
         "shared_fields": list(SHARED_FIELDS),
         "recipe": {**c0["recipe"]},
         "labels": LABELS,
         "resource_plan": {"state_bytes": parent["bytes"], "formal_states_per_arm": len(SAVE_EPOCHS),
                           "disk_free_bytes_at_prepare": shutil.disk_usage(ROUND).free,
                           "gpu_time_reference": "task estimate 15-19 h without K, 27-32 h with K (not measured, not a cap)"},
         "outputs": str(ROUND), "formal_started": False}
    require(c["monitor_sha256"] == sha_bytes(canonical(c["monitor_ids"])) and len(c["monitor_ids"]) == 64, "monitor ids")
    require(len(c["development_ids"]) == 548, "development ids")
    c["contract_sha256"] = sha_bytes(canonical(c))
    write_new(ROUND / "CONTRACT.json", c)
    print(json.dumps({"contract_sha256": c["contract_sha256"], "identity_sha256": R.IDENTITY_SHA256}))
    return c


def load_contract(path=None):
    c = read_json(path or ROUND / "CONTRACT.json")
    body = {k: v for k, v in c.items() if k != "contract_sha256"}
    require(c["contract_sha256"] == sha_bytes(canonical(body)), "contract digest")
    require(Path(c["source_root"]).resolve() == ROOT.resolve(), "not running from the bound release")
    require(Path(c["outputs"]).resolve() == ROUND.resolve(), "contract bound to another round")
    manifest = read_json(pinned(c["source_manifest"]))
    require(manifest["commit"] == c["source_commit"], "MANIFEST commit differs from the contract")
    B.release_intact(ROOT, manifest["files"])
    pinned(c["parent"])
    from sparse_rtdetr.research import r28_resample as R
    from sparse_rtdetr.research import r28_warp as W
    want = dict(R.IDENTITY_SHA256)
    if "F" in c["identity_sha256"]:
        from sparse_rtdetr.research import r28_fovea as FV
        want.update(FV.IDENTITY_SHA256)
        require(c["fovea_definition"] == FV.DEFINITION and followup_arm(c) in ("F", "FS", "FG", "R29"), "R28-F mapping definition differs from the contract")
    if "FS" in c["identity_sha256"]:
        from sparse_rtdetr.research import r28_fovea_cal as FS
        want.update(FS.IDENTITY_SHA256)
        require(c["fovea_cal_definition"] == FS.DEFINITION and followup_arm(c) == "FS",
                "R28-FS calibrated mapping definition differs from the contract")
    if "FG" in c["identity_sha256"]:
        from sparse_rtdetr.research import r28_fovea_geo as FG
        want.update(FG.IDENTITY_SHA256)
        require(c["fovea_geo_definition"] == FG.DEFINITION and followup_arm(c) in ("FG", "R29"),
                "R28-FG fixed-width mapping definition differs from the contract")
    if any(a in c["identity_sha256"] for a in R29_ARMS):
        from sparse_rtdetr.research import r29_unzoom as UZ
        want.update(UZ.IDENTITY_SHA256)
        require(c["r29_read_definition"] == UZ.DEFINITION and r29(c) and "FG" in c["identity_sha256"],
                "R29 read definition differs from the contract")
    require(c["identity_sha256"] == want and c["definition"] == W.DEFINITION,
            "R28 behaviour identity differs from the contract")
    return c


def stage_closed(c, name, arm=None):
    """<round>/<name>/CLOSED.json must be PASS under this contract (and of the given arm, when one is given).
    In a follow-up round the stages of the closed R28 round are read from that round, each CLOSED.json pinned by the
    follow-up contract and bound to the prior contract; such a name is never read from (or written to) the new round."""
    fu = followup(c)
    if fu is not None and name in fu["inherited"]:
        require(not (ROUND / name).exists(), f"a follow-up round must not contain the inherited stage {name}")
        closed = read_json(pinned(fu["inherited"][name]))
        # a stage inherited from another closed round than the W round names that round's contract explicitly
        bound = fu.get("inherited_contracts", {}).get(name, fu["prior_contract_sha256"])
        require(closed.get("status") == "PASS" and closed.get("contract_sha256") == bound
                and (arm is None or closed.get("arm") == arm), f"inherited stage {name} is not PASS under the prior contract")
        return closed
    path = ROUND / name / "CLOSED.json"
    require(path.is_file(), f"stage not closed: {path}")
    closed = read_json(path)
    require(closed.get("status") == "PASS" and closed.get("contract_sha256") == c["contract_sha256"]
            and (arm is None or closed.get("arm") == arm), f"stage {name} is not PASS under this contract")
    return closed


# ================================================================ demand sources
def calibration(c):
    """The locked calibration of the FS round: its closed calibrate stage must have matched.  Returns the width (FP64,
    from the stored hex string) and the pinned CALIBRATION.json reference."""
    require(followup_arm(c) == "FS", "only the calibrated mapping-control round has a calibration")
    closed = stage_closed(c, "calibrate", "FS")
    require(closed.get("calibration_status") == "matched",
            f"the calibration did not produce a matched width ({closed.get('calibration_status')!r}): arm FS cannot be built")
    body = read_json(pinned(closed["calibration"]))
    h = float.fromhex(closed["h_star_hex"])
    require(body["status"] == "matched" and body["h_star_hex"] == closed["h_star_hex"] == h.hex()
            and body["contract_sha256"] == c["contract_sha256"], "locked calibration file differs from the closed stage")
    return {"fwhm": h, "calibration": closed["calibration"]}


def selected_width(c):
    """The fixed kernel width of the FG round (TASK_R28_STRONG_F_CONTROL_009): the contract's selection block, checked
    against the pinned calibration file of task 006, which must still read calibration_unmatched with h_star = null
    and name this width as its closest reading.  Returns the width (FP64, from the stored hex string) and the block."""
    require(followup_arm(c) in ("FG", "R29"), "only the fixed-width strong mapping-control round and the R29 round have a selected width")
    from sparse_rtdetr.research import r28_fovea_geo as FG
    sel = FG.check_selection(followup(c).get("width_selection"))
    body = read_json(pinned(sel["calibration"]))
    best = body.get("closest_if_unmatched") or {}
    require(body.get("status") == "calibration_unmatched" and body.get("h_star") is None and body.get("h_star_hex") is None
            and best.get("h_hex") == sel["fwhm_hex"] == FG.WIDTH_HEX and sel.get("fwhm") == FG.WIDTH,
            "the 006 calibration file does not carry the unmatched reading the fixed width was selected from")
    return {"fwhm": float.fromhex(sel["fwhm_hex"]), "selection": sel}


def pipeline_mode(c, arm):
    """Input pipeline of a formal run: arms FS, FG, QP and QC use the overlapped path only if their closed acceptance
    enabled it; every other arm, and R28_PIPELINE=serial, use the serial reference path."""
    forced = os.environ.get("R28_PIPELINE")
    require(forced in (None, "serial"), "R28_PIPELINE can only select the serial reference path")
    if arm not in ("FS", "FG") + R29_ARMS or forced:
        return "serial"
    if arm in R29_ARMS:                   # one acceptance stage covers both arms; each arm has its own decision
        require(r29(c), "arms QP and QC are trained only in the R29 round")
        return "overlap" if stage_closed(c, "accept-R29")["overlap_enabled"][arm] is True else "serial"
    stage = "pipeline" if arm == "FS" else "accept-FG"
    return "overlap" if stage_closed(c, stage, arm)["overlap_enabled"] is True else "serial"


def demand_source(c, arm):
    """The formal demand source of an arm, read from the closed stage that produced it."""
    if arm == "FG" or arm in R29_ARMS:    # arm F's demand source plus the fixed geometry-selected kernel width
        require(arm == "FG" or r29(c), "arms QP and QC exist only in the R29 round")
        return {"kind": "network", "file": stage_closed(c, "demand")["weights"], **selected_width(c)}
    if arm == "FS":                       # arm F's demand source plus the locked kernel width
        return {"kind": "network", "file": stage_closed(c, "demand")["weights"], **calibration(c)}
    if arm in ("W", "F"):                 # arm F reads the same frozen demand network file as arm W
        return {"kind": "network", "file": stage_closed(c, "demand")["weights"]}
    require(arm == "K", f"arm {arm}")
    return {"kind": "fixed", "file": stage_closed(c, "prior")["file"]}


def device():
    """cuda:0, or cpu for the light demand-side stages of a dev round (R28_DEV_CPU=1; refused elsewhere)."""
    if os.environ.get("R28_DEV_CPU") == "1":
        require(DEV, "R28_DEV_CPU is refused outside a dev round")
        return "cpu"
    return "cuda:0"


def g0_passed(c):
    closed = stage_closed(c, "g0")
    require(closed["G0_passed"] is True or (DEV and os.environ.get("R28_DEV_IGNORE_G0") == "1"),
            "G0 did not pass: the formal W path is not opened")
    return closed


# ================================================================ states (schema r28-state-v1)
def binding(c, e, scope, audit=None, role=None):
    from formal_data import cursor
    arm = getattr(e, "r28_arm", None)
    require(arm in ARMS and not hasattr(e, "l0_arm") and not hasattr(e, "r27_arm"), "engine is not an R28 engine")
    mapping = getattr(e, "r28_mapping", None)
    require((arm in ("FS", "FG") + R29_ARMS) == (mapping is not None), "only an arm FS / FG / QP / QC engine carries a mapping width binding")
    return {"schema": SCHEMA, "scope": scope, "arm": arm, "contract_sha256": c["contract_sha256"],
            "identity_sha256": c["identity_sha256"][arm], "parent_sha256": c["parent"]["sha256"],
            "global_update": e.completed_updates, "cursor": cursor(e.completed_updates), "config": e.config.binding(),
            "audit": audit, "role": role, "activation_checkpoint": e.s15_activation_checkpoint,
            "demand": e.r28_ctx.binding(), "torch": str(torch_mod().__version__),
            **({} if mapping is None else {"mapping": mapping})}      # arm FS / FG: kernel width and its file reference


def save(c, e, path, scope, audit=None, role=None):
    from sparse_rtdetr.research import scale_baseline as S
    from sparse_rtdetr.protocol_r18.engine import assert_healthy
    torch = torch_mod()
    path = Path(path)
    require(e.ema.updates == e.completed_updates, "EMA clock at save")
    require(scope != "formal" or not e.r28_check, "a check engine cannot write a formal state")
    assert_healthy(e.model, e.optimizer, e.ema)
    state = S.capture(e)
    b = binding(c, e, scope, audit, role)
    state["r28_binding_sha256"] = sha_bytes(canonical(b))
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("xb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.link(tmp, path)
    tmp.unlink()
    side = {"status": "COMPLETE", **b, "file": ref(path)}
    write_new(side_of(path), side)
    return side


def read_side(c, path, scope, arm):
    path = Path(path).resolve()
    side = read_json(side_of(path))
    for k, want in (("schema", SCHEMA), ("arm", arm), ("scope", scope), ("contract_sha256", c["contract_sha256"]),
                    ("parent_sha256", c["parent"]["sha256"]), ("identity_sha256", c["identity_sha256"][arm])):
        require(side.get(k) == want, f"R28 state refused: {k}={side.get(k)!r}, expected {want!r} ({path})")
    require(side.get("status") == "COMPLETE", f"R28 state side-car is not COMPLETE ({path})")
    return path, side


def read_state(c, path, scope, arm):
    torch = torch_mod()
    path, side = read_side(c, path, scope, arm)
    require(side["file"] == ref(path), "R28 state file identity")
    s = torch.load(path, map_location="cpu", weights_only=True)
    expect = {k: v for k, v in side.items() if k not in ("status", "file")}
    require(s.pop("r28_binding_sha256") == sha_bytes(canonical(expect)), "R28 binding digest")
    require(s["engine"]["completed_updates"] == side["global_update"] == s["ema"]["updates"], "R28 state clock")
    return side, s


def load(c, e, path, scope):
    """Exact restore into a fresh engine of the same arm and demand source."""
    from sparse_rtdetr.research import scale_baseline as S
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    arm = getattr(e, "r28_arm", None)
    require(arm in ARMS, "engine is not an R28 engine")
    side, s = read_state(c, path, scope, arm)
    require(side["config"] == e.config.binding() and side["activation_checkpoint"] == e.s15_activation_checkpoint,
            "R28 state config / memory mode")
    require(side["demand"] == e.r28_ctx.binding(), "R28 state was written with another demand source")
    require(side.get("mapping") == getattr(e, "r28_mapping", None), "R28 state was written with another mapping width / calibration")
    require(s["optimizer_names"] == [g["param_names"] for g in e.optimizer.param_groups], "R28 optimizer names")
    e.model.load_state_dict(s["model"], strict=True)
    e.optimizer.load_state_dict(s["optimizer"])
    e.ema.load_state_dict(s["ema"])
    modes = dict(e.model.named_modules())
    require(set(modes) == set(s["model_modes"]), "R28 mode keys")
    for n, m in s["model_modes"].items():
        modes[n].training = m
    e.load_state_dict(s["engine"])
    ck.restore_rng(s["rng"], e.device)
    require(ck.exact_equal(S.capture(e), s), "R28 full state did not restore exactly")
    return side


def formal_re(arm):
    return re.compile(rf"formal-{arm}(-r([2-9]|[1-9][0-9]+))?")


def formal_dir(c, arm):
    require(followup(c) is None or arm == followup_arm(c) or (r29(c) and arm in R29_ARMS),
            f"this follow-up round has no formal directory of arm {arm}")
    found = []
    for d in sorted(ROUND.iterdir()) if ROUND.is_dir() else ():
        if not (d.is_dir() and formal_re(arm).fullmatch(d.name) and (d / "CLOSED.json").is_file()):
            continue
        closed = read_json(d / "CLOSED.json")
        if closed.get("status") == "PASS" and closed.get("arm") == arm and closed.get("contract_sha256") == c["contract_sha256"]:
            found.append(d)
    require(len(found) == 1, f"exactly one closed PASS formal-{arm} directory is required under {ROUND}: {[d.name for d in found]}")
    return found[0]


def formal_stop():
    """Stop update of a formal run: the endpoint, or a short dev span (dev rounds only)."""
    span = os.environ.get("R28_DEV_SPAN")
    if span is None:
        return ENDPOINT, None
    require(DEV, "R28_DEV_SPAN is refused outside a dev round")
    n = int(span)
    require(2 <= n <= 64 and n % 2 == 0, "R28_DEV_SPAN must be an even number of updates in 2..64")
    return PARENT_UPDATE + n, (PARENT_UPDATE + n // 2, PARENT_UPDATE + n)


def endpoint_name(stop):
    ep, rest = divmod(stop, WINDOWS_PER_EPOCH)
    return f"checkpoint-update-{stop}.pth" if rest else f"checkpoint-epoch-{ep:03d}.pth"


def endpoint_state(c, arm):
    """The only evaluation source of an arm: the endpoint state of its closed formal directory, EMA module only.
    In a follow-up round arm W is the pinned e120 state of the closed R28 round, read under the prior contract."""
    fu = followup(c)
    require(not (fu is not None and r29(c)) or arm in R29_ARMS + ("FG",), f"the R29 round has no endpoint of arm {arm}")
    if fu is not None and arm == "W":
        side, s = read_state(prior_contract(c), pinned(fu["W_endpoint"]), "formal", "W")
        require(side["file"] == fu["W_endpoint"] and side["global_update"] == ENDPOINT
                and s["engine"]["completed_updates"] == s["ema"]["updates"] == ENDPOINT, "prior W endpoint clocks")
        return side, s["ema"]["module"]
    if fu is not None and r29(c) and arm == "FG":
        # R29: N is the stored e120 state of the closed F_geo round, read under that round's contract
        side, s = read_state(other_contract(c, "FG"), pinned(fu["FG_endpoint"]), "formal", "FG")
        require(side["file"] == fu["FG_endpoint"] and side["global_update"] == ENDPOINT
                and s["engine"]["completed_updates"] == s["ema"]["updates"] == ENDPOINT, "stored FG endpoint clocks")
        return side, s["ema"]["module"]
    if fu is not None and followup_arm(c) in ("FS", "FG") and arm in ("K", "F"):
        # the FS / FG controls time the stored K and F endpoints: each is read under its own round's contract
        side, s = read_state(other_contract(c, arm), pinned(fu[f"{arm}_endpoint"]), "formal", arm)
        require(side["file"] == fu[f"{arm}_endpoint"] and side["global_update"] == ENDPOINT
                and s["engine"]["completed_updates"] == s["ema"]["updates"] == ENDPOINT, f"stored {arm} endpoint clocks")
        return side, s["ema"]["module"]
    stop, _ = formal_stop()
    path = formal_dir(c, arm) / endpoint_name(stop)
    require(path.is_file() and side_of(path).is_file(), f"R28 endpoint missing: {path}")
    side, s = read_state(c, path, "formal", arm)
    require(side["global_update"] == stop and s["engine"]["completed_updates"] == s["ema"]["updates"] == stop, "R28 endpoint clocks")
    return side, s["ema"]["module"]
