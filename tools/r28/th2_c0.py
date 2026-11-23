"""TH2 C0 (TASK_TH2_C0_001): per-image routing between two stored endpoints of the same seed (L0 on canvas L, S15 on
canvas H) at a fixed mean pixel budget.  Recombination and scoring of stored results only: no model forward, no
training, no GPU, no test-dev, no train image or label.  The reading and verdict functions are c0_ref.py (the
controller's file, byte-identical copy).

usage: th2_c0.py storage --home <home>
       th2_c0.py prepare --round <round> --release <name>
       th2_c0.py <stage> --round <round> [--unit s0|s1|s2]

  storage  the first file of the home directory: STORAGE_ESTIMATE.json (refuses a directory that is not empty).
  prepare  contract: release, pinned inputs and predictions, registered values, orders, k, verdict rule.
  wire     CPU: task 4.7 items 1, 2, 3, 5, 6 and the agreement of the task text with c0_ref.py.
  ends     CPU, per seed: task 4.7 item 4.  The splice function with R = empty and R = all, scored; both must equal
           the registered values within 1e-9.
  hits     CPU, formal round only: orders, assignments, T1-T4, T6, the per-image file.
  routed   CPU, per seed, formal round only: 11 scorings (order P at k = 158, 219, 341; order N at k = 158; the 7
           balanced subsets).
  tables   CPU, formal round only: T5, verdict, RESULT.json.

The home directory is the formal round; a dev round is <home>/dev-round*.  A dev round runs wire and ends only (they
produce no new reading).  Every stage writes <round>/<name>/CLOSED.json (PASS) or STOP.json.
"""
import argparse
import csv
import datetime
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "tools" / "r27_l0")]

import numpy as np  # noqa: E402

import c0_ref as R  # noqa: E402
import l0_common as B  # noqa: E402
import th2_core as T  # noqa: E402

VERSION = "TH2-C0-001-v1"
TASK = ("TASK_TH2_C0_001.md", "ef52a60ddc38f5360be8f4639ce5f5e93b0c4f3374ae427cb8f903abc5cffa53")
REF = ("tools/r28/c0_ref.py", "c55f9dfab5e4804f53b7aa87fd5aaebe7d08c0d7e50698f07c6a45498ffb3653")
P1 = B.WS / "experiments/thesis-study2/th2-phase1-20261122-001"
P1_CONTRACT_SHA = "2471ad9271ebbae0c645cc283de137118e9571d1aefadf7d321028414ae8788d"
P1_MANIFEST_SHA = "bdfe93fb0c58868581d4d1914084ac2cf1826daf8f51c8cfd8e0991949d02695"
P1_COMMIT = "b24dec791b72812c8d9a1e6342ff637497bef770"
ALLOWED_ADDED = ("tools/r28/c0_ref.py", "tools/r28/th2_c0.py", "tools/r28/th2c0_chain.sh", "tests/test_th2_c0.py")
ALLOWED_CHANGED = ("tools/r28/th2_phase1.py",)          # the erratum commit bc29d03 of the phase-1 branch
M, N_GT, GT_COUNTS = 548, 38599, {"GT_before": 38759, "GT_after": 38599}
GROUP_COUNTS = {"small": 26441, "medium": 11090, "large": 1068}
SEEDS = (0, 1, 2)
ARMS = ("L0", "S15")
UNITS = tuple(f"{a}-s{s}" for a in ("L0", "FG", "S15") for s in SEEDS)
METRICS = ("AP", "APS", "AP75_small", "APM", "APL")
K_SCORED = {"P": (158, 219, 341), "N": (158,)}
SCORED = tuple(f"{o}-k{k}" for o in ("P", "N") for k in K_SCORED[o]) + tuple(f"RAND-r{r}" for r in range(R.RESIDUES))
SCORINGS = len(SEEDS) * (len(ARMS) + len(SCORED))
T4_ROWS = (("medium_hit75", "medium", 0.75), ("large_hit75", "large", 0.75), ("small_hit50", "small", 0.5))
TOL = 1e-9
CAP_BYTES, CAP_SECONDS, MAX_PARALLEL = 200_000_000, 4 * 3600, 4
# task 4.1: file -> (phase-1 stage, key in its receipt, bytes, sha256)
INPUT_PINS = {
    "outcomes": ("e0", "outcomes", 8035062, "72f4b1fee92055a595983c49de33310a354748b537030465b1782673cd98fc52"),
    "gt": ("geom", "gt", 6243855, "11dab33f84ef440b2f68f02651b89fda644279d582e9223c5a1d08bcfe0716c1"),
    "demand": ("demand", "demand", 1104896, "48f9bbcede4cd94ec6fc8123f4ad2a84be0a3dd89494262c4fd27907787c80f6"),
    "d_star": ("geom", "d_star", 1104896, "4f4112d853070965ad5a536cf65b412d0ff773304ac2815192329aed46d0dca0"),
}
PRED_PINS = {
    "L0-s0": (87892832, "25fcf2fd2ca30e866c623a4aa26c7dcca3207a9e16e9583ea28905ac91b3c760"),
    "L0-s1": (87888074, "466e253d099a40b5b6b5d0bf39bc172b7362e06d31f3955ceb322bc395745b45"),
    "L0-s2": (87888655, "25a49e19d4f4410ce83c557fb92a363417314beabd46b52a9604196933c928e9"),
    "S15-s0": (96433942, "f8e89847c4ebcc67f8ed08b8f794099f7459b9051b096517656afcb6b6af0ac5"),
    "S15-s1": (96437040, "87f15dbdce49f28b3444ea1ae2310af4dfe4832f71067d012abe3a92eaa0a0dd"),
    "S15-s2": (96435494, "2baf949ae8a25d6aabc093a0cb0c1bb128977bca5baa5400c7ea89c2d8f30ca9"),
}
# task 4.7 item 2: small hit75 (pp) recomputed from outcomes.npz
REG_HIT = {"L0": (43.515752051737834, 43.38338186906698, 43.33421580121781),
           "S15": (48.526908967134375, 48.59120305586021, 48.76517529594191),
           "FG": (45.73957112060815, 45.327332551718925, 45.81899323021066)}
# task 4.7 item 3
REG_DEMAND = {"min": 8.867765245668124e-06, "max": 7.389012813568115, "mean": 0.21952297445568622}
# task 4.7 item 4: top600 bridge metrics (AP, APS, AP75_small, APM, APL)
REG_AP = {
    "L0-s0": (0.30054005272689926, 0.2238294714480465, 0.20231585685439496, 0.39561519581083376, 0.4951035581025013),
    "L0-s1": (0.30290162806340576, 0.2254072810959945, 0.20096192156510068, 0.40102132976280724, 0.4488479214760187),
    "L0-s2": (0.30150564214328784, 0.22422677959481147, 0.19754164165515165, 0.3956624409754658, 0.4788377695020645),
    "S15-s0": (0.3200866660864559, 0.2513747771125344, 0.23720953090636165, 0.40495283253729625, 0.4446219718630393),
    "S15-s1": (0.32275410693313805, 0.2518638980699198, 0.2336804215936788, 0.41066023104790234, 0.42352199717772643),
    "S15-s2": (0.3226902485312341, 0.2531481646852, 0.24237263525556532, 0.4098810662004296, 0.42790301920128576),
}
# task 4.5: k -> (p, B, l) to six decimals
TEXT_TABLE = {158: (0.288321, 1.360401, 0.379539), 219: (0.399635, 1.499544, 0.499625), 341: (0.622263, 1.777828, 0.709546)}
STORAGE_PLAN = {"release_source_formal_round": 11_500_000, "release_source_dev_round": 11_500_000,
                "release_source_of_one_permitted_rerun": 11_500_000, "metric_files_45_x_70kB": 45 * 70_000,
                "assignments_per_image_file_tables": 1_500_000, "contracts_receipts_logs": 1_500_000}
DEFINITIONS = {
    "evaluation_set": "development = official val, 548 images; the labels are read only as the input of the hits / the scorer",
    "GT": "the 38,599 scored development GT of phase 1 (geom/gt.npz rows with scored = True), image order",
    "groups": "annotation area: small <= 32^2 (26,441); medium = (32^2, 96^2) (11,090); large >= 96^2 (1,068)",
    "hit": "phase-1 e0/outcomes.npz iou_<unit> (best same-image same-class IoU over the 600 stored candidates) >= 0.75 / 0.5",
    "arms": "cheap = L0-s on canvas L 768 x 1344 (cost 1); expensive = S15-s on canvas H 1152 x 2016 (cost 2.25); same seed index",
    "budget": "p = |R| / 548; B = 1 + 1.25 p in units of the L canvas; the routing statistic itself is not charged",
    "order_P": "T_i = sum over the 12 x 21 cells of phase-1 demand/demand.npy[i] (numpy sum over axes 1, 2 of the FP64 array); "
               "no development label is used; the same order for the three seeds",
    "order_N": "number of small GT of the image (labels)", "order_Dstar": "sum over the cells of phase-1 geom/d_star.npy[i] (labels)",
    "order_X": "d_i = b_i - a_i (small hit75) of seed t applied to seed s != t", "order_O": "d_i of the same seed (upper bound, uses the noise)",
    "ties": "c0_ref.order_desc: largest first, ties by image index ascending",
    "readings": "c0_ref.readings: routed value, chord, log-linear reference (l = ln B / ln 2.25), capture, Q = routed - reference (pp)",
    "interval": "c0_ref.q_bootstrap with c0_ref.boot_weights(548) (= th2_core.boot_weights(548): 1,000 image resamples, seed "
                "20261122), R held fixed.  The interval of the three-seed mean uses the mean of a and of b over the seeds "
                "(Q is linear in the hits for a fixed order and a fixed n)",
    "rank_correlation": "Pearson correlation of average ranks over the 548 images",
    "T3_groups": "c0_ref.groups5 (np.array_split of the order into 110, 110, 110, 109, 109 images); a group without small GT has "
                 "null rates",
    "splice": "development image order; image i in R takes the 600 stored records of S15-s, otherwise those of L0-s; records "
              "unchanged, no rescoring, no reordering, nothing dropped.  The list is rebuilt from the stored bytes for every scoring",
    "scoring": "the call path of the phase-1 score stage: th2_phase1.development, score_budget(rows, parsed, preds, source_root), "
               "infer_seed.bridge_values; GT_before 38759, GT_after 38599",
    "random_rows": "c0_ref.balanced_subsets: R_r = {i : i mod 7 in {r, (r + 3) mod 7}}, r = 0..6",
    "mixing": "c0_ref.mixing: mean of the 7 subsets minus [M_L + (2/7) lead], pp",
    "Q_metric": "c0_ref.q_metric with k = number of routed images of the row; pure-endpoint values of this round",
    "three_seed_mean": "arithmetic mean of the per-seed value of every numeric field",
    "std": "sample standard deviation (ddof = 1) of the 7 random rows, pp",
    "verdict": "c0_ref.verdict; order P, k = 158, L0 against S15, s = t; task section 6",
}
VERDICT_RULE = {"reads": "order P, k = 158, cheap L0, expensive S15, same seed index; means over the three seeds",
                "G1": "mean Q_hit >= 0.50 + 0.10 * mean lead_hit and every seed Q_hit > 0", "G2": "mean Q_APS >= 0.20 + 0.10 * mean lead_APS and min Q_APS >= -0.10",
                "G3": "mean Q_AP >= 0", "STOP": "mean Q_hit <= 0 or the three signs of Q_hit differ",
                "labels": "STOP column -> STOP; G1 and G2 and G3 -> GO-CANDIDATE (starts nothing); otherwise UNDECIDED",
                "registered_with_the_label": ["order N clears G1", "mixing cost of APS and of AP"]}
BOUNDARIES = ["development is a development set; these readings are not a final evaluation of a routing method",
              "k is taken by rank inside development; a deployable rule needs a threshold fixed in advance and its share would differ",
              "the routed system uses two separately trained models: two training costs, not one model",
              "the budget counts detector input pixels only",
              "the log-linear reference is an assumption: no uniform canvas between L and H has been trained",
              "GO-CANDIDATE does not mean that per-image allocation beats a uniform canvas"]
NOT_AUTHORISED = ["any model forward (detector or demand network), training, GPU use", "test-dev; any image or label of official train",
                  "routing statistics, k values, endpoint pairs, score calibration, reordering or analyses outside task sections 4-5",
                  "installing or upgrading packages; edits under experiments/localization-distribution, phase-1 or B0 outputs",
                  "merging pull requests; changing the state of PR #43, #44, #45"]


def is_dev(r):
    return Path(r).resolve().name.startswith("dev-round")


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


# ================================================================ limits
def chain_seconds(home):
    """Wall seconds of the chain runs logged in <home>/chain-runs.log (closed runs plus the open one)."""
    log = Path(home) / "chain-runs.log"
    if not log.is_file():
        return 0.0
    total, opened = 0.0, None
    for line in log.read_text().split("\n"):
        f = line.split()
        if len(f) != 3:
            continue
        if f[1] == "start":
            opened = float(f[2])
        elif f[1] == "end" and opened is not None:
            total, opened = total + float(f[2]) - opened, None
    return total + (time.time() - opened if opened is not None else 0.0)


def limits_guard(home):
    """Refuses to go on without the storage estimate, above the storage cap or above the time cap."""
    home = Path(home)
    path = home / "STORAGE_ESTIMATE.json"
    B.require(path.is_file(), f"no STORAGE_ESTIMATE.json in {home}: nothing may run")
    est = B.read_json(path)
    B.require(est["cap_bytes"] == CAP_BYTES and 0 < est["estimate_bytes"] <= CAP_BYTES, "storage estimate above the cap")
    used = sum(p.stat().st_size for p in home.rglob("*") if p.is_file() and not p.is_symlink())
    B.require(used <= CAP_BYTES, f"storage cap reached: {used} bytes in {home}")
    seconds = chain_seconds(home)
    B.require(seconds <= CAP_SECONDS, f"time cap reached: {seconds:.0f} s of chain runs")
    return {"bytes_used": used, "chain_seconds": seconds, "estimate": B.ref(path)}


def storage(home):
    home = Path(home).resolve()
    B.require(home.is_dir() and not any(home.iterdir()), f"storage: {home} must exist and be empty")
    B.require("localization-distribution" not in str(home) and home.parent == B.WS / "experiments/thesis-study2",
              "the home directory belongs under experiments/thesis-study2")
    total = sum(STORAGE_PLAN.values())
    B.require(total <= CAP_BYTES, "estimate above the cap")
    B.write_new(home / "STORAGE_ESTIMATE.json", {
        "version": VERSION, "task": {"file": TASK[0], "sha256": TASK[1]}, "cap_bytes": CAP_BYTES, "estimate_bytes": total,
        "breakdown_bytes": STORAGE_PLAN, "expected_without_rerun_bytes": total - STORAGE_PLAN["release_source_of_one_permitted_rerun"],
        "not_stored": "the spliced prediction lists (fixed by the assignment arrays and the stored files)",
        "written_at": now(), "written_at_unix": time.time(), "disk_free_bytes": shutil.disk_usage(home).free})
    B.require(sorted(p.name for p in home.iterdir()) == ["STORAGE_ESTIMATE.json"], "STORAGE_ESTIMATE.json is not the only file")
    print(json.dumps({"storage_estimate_bytes": total, "cap_bytes": CAP_BYTES, "file": B.ref(home / "STORAGE_ESTIMATE.json")}))


# ================================================================ contract
def text_against_reference():
    """The numbers written in the task text against c0_ref.py."""
    B.require(R.M_IMAGES == M and R.K_GATE == 158 and R.K_CURVE == (55, 110, 158, 219, 274, 341, 438) and R.RESIDUES == 7
              and R.H_COST == 2.25 and (R.BOOT, R.BOOT_SEED) == (T.BOOT, T.BOOT_SEED) == (1000, 20261122), "constants of c0_ref.py")
    for k, (p, b, ell) in TEXT_TABLE.items():
        B.require(round(k / M, 6) == p and round(R.budget(k / M), 6) == b and round(float(R.loglinear(k / M)), 6) == ell,
                  f"task 4.5 table differs from c0_ref.py at k = {k}")
    B.require(R.budget(158 / M) <= 49 / 36 < R.budget(159 / M) and round(R.budget(158 / M), 4) == 1.3604
              and round(R.budget(159 / M), 4) == 1.3627, "k = 158 is not the last k under the 896 x 1568 canvas")
    B.require(round(R.budget(341 / M), 5) == 1.77783 and round(R.budget(341 / M) - 16 / 9, 5) == 0.00005
              and round(R.budget(219 / M), 4) == 1.4995, "k = 341 / k = 219 budgets")
    sizes = sorted(len(s) for s in R.balanced_subsets(M))
    B.require(sizes[0] == 156 and sizes[-1] == 157 and sum(sizes) == 2 * M, "balanced subsets")
    return {"table_4.5": {str(k): [k / M, R.budget(k / M), float(R.loglinear(k / M))] for k in TEXT_TABLE}, "subset_sizes": sizes}


def prepare(round_dir, release):
    R_ = Path(round_dir).resolve()
    dev = is_dev(R_)
    home = R_.parent if dev else R_
    lim = limits_guard(home)
    B.require(not (R_ / "CONTRACT.json").exists(), "CONTRACT.json exists")
    B.require((R_ / release / "source").resolve() == ROOT.resolve(), "prepare must run from <round>/<release>/source")
    B.require("localization-distribution" not in str(R_) and "th2-phase1" not in str(R_) and "th2-b0" not in str(R_),
              "outputs must not be under the old output directories")
    task = home / TASK[0]
    B.require(task.is_file() and B.file_sha(task) == TASK[1], f"task file missing or different: {task}")
    ref_file = B.ref(ROOT / REF[0])
    B.require(ref_file["sha256"] == REF[1], "c0_ref.py is not the controller's file")
    text = text_against_reference()
    man_ref = B.ref(R_ / release / "MANIFEST.json")
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
              f"the release may differ from the phase-1 release only by the C0 files and the erratum: {delta}")
    p1_ref = B.ref(P1 / "CONTRACT.json")
    B.require(p1_ref["sha256"] == P1_CONTRACT_SHA, "phase-1 contract file")
    p1 = B.read_json(p1_ref["path"])
    B.require(os.path.samefile(sys.executable, p1["python"]), f"python differs from phase 1: {sys.executable}")
    inputs = {}
    for name, (stage, key, size, sha) in INPUT_PINS.items():
        closed = B.read_json(P1 / stage / "CLOSED.json")
        B.require(closed["status"] == "PASS" and closed["contract_sha256"] == p1["contract_sha256"], f"phase-1 stage {stage}")
        r = closed["files"][key]
        B.require(r["bytes"] == size and r["sha256"] == sha, f"{name}: the phase-1 receipt differs from task 4.1")
        B.pinned(r)
        inputs[name] = r
    predictions, receipts = {}, {}
    for unit, (size, sha) in PRED_PINS.items():
        r = p1["units"][unit]["predictions_600"]
        B.require(r["bytes"] == size and r["sha256"] == sha, f"{unit}: the phase-1 contract differs from task 4.1")
        B.pinned(r)
        predictions[unit] = r
        if unit.startswith("L0"):           # the L0 receipts carry values["600"]; the S15 values are the task's table
            rec = B.read_json(B.pinned(p1["units"][unit]["receipt"]))
            B.require(tuple(rec["values"]["600"][m] for m in METRICS) == REG_AP[unit], f"{unit}: receipt values differ from task 4.7")
            receipts[unit] = p1["units"][unit]["receipt"]
    c = {"version": VERSION, "task": {"file": TASK[0], "sha256": TASK[1]}, "reference": ref_file,
         "zero_training": True, "zero_forwards": True, "cpu_only": True,
         "source_root": str(ROOT), "source_commit": manifest["commit"], "source_branch": manifest["branch"],
         "source_dirty": manifest["dirty"], "source_manifest": man_ref, "base_manifest": base_ref, "source_delta": delta,
         "outputs": str(R_), "home": str(home), "dev": dev,
         "phase1": {"contract": p1_ref, "contract_sha256": p1["contract_sha256"]},
         "inputs": inputs, "predictions": predictions, "registered_receipts_L0": receipts,
         "registered": {"small_hit75_pp": {k: list(v) for k, v in REG_HIT.items()}, "demand": REG_DEMAND,
                        "top600": {k: dict(zip(METRICS, v)) for k, v in REG_AP.items()}, "tolerance": TOL},
         "data": {"development": p1["data"]["development"]}, "development_ids": p1["development_ids"], "python": sys.executable,
         "design": {"canvas_L": [768, 1344], "canvas_H": [1152, 2016], "cost_H": R.H_COST, "k_gate": R.K_GATE, "k_curve": list(R.K_CURVE),
                    "k_scored": {k: list(v) for k, v in K_SCORED.items()}, "random_subsets": R.RESIDUES, "seeds": list(SEEDS),
                    "scored_rows_per_seed": list(ARMS) + list(SCORED), "scorings": SCORINGS, "metrics": list(METRICS),
                    "bootstrap": [R.BOOT, R.BOOT_SEED], "text_table": text},
         "definitions": DEFINITIONS, "verdict_rule": VERDICT_RULE, "boundaries": BOUNDARIES,
         "dev_round_rule": "a dev round runs wire and ends only (task 4.7); hits, routed and tables refuse it",
         "limits": {"cpu_wall_seconds": CAP_SECONDS, "storage_bytes": CAP_BYTES, "parallel_scoring_processes": MAX_PARALLEL},
         "storage_estimate": lim["estimate"], "not_authorised": NOT_AUTHORISED}
    c["contract_sha256"] = B.sha_bytes(B.canonical(c))
    B.write_new(R_ / "CONTRACT.json", c)
    print(json.dumps({"contract_sha256": c["contract_sha256"], "contract_file": B.ref(R_ / "CONTRACT.json"),
                      "commit": c["source_commit"], "delta": delta, "dev": dev}))


def load_contract(round_dir):
    R_ = Path(round_dir).resolve()
    c = B.read_json(R_ / "CONTRACT.json")
    body = {k: v for k, v in c.items() if k != "contract_sha256"}
    B.require(c["contract_sha256"] == B.sha_bytes(B.canonical(body)) and c["version"] == VERSION, "contract digest")
    B.require(Path(c["source_root"]).resolve() == ROOT.resolve(), "not running from the bound release")
    B.require(Path(c["outputs"]).resolve() == R_ and c["dev"] == is_dev(R_), "contract bound to another round")
    manifest = B.read_json(B.pinned(c["source_manifest"]))
    B.require(manifest["commit"] == c["source_commit"], "MANIFEST commit differs from the contract")
    B.release_intact(ROOT, manifest["files"])
    B.require(c["task"]["sha256"] == TASK[1] and c["reference"]["sha256"] == REF[1] and B.file_sha(ROOT / REF[0]) == REF[1]
              and c["definitions"] == DEFINITIONS and c["verdict_rule"] == VERDICT_RULE, "contract differs from the release")
    limits_guard(c["home"])
    return c


def stage_closed(c, name):
    path = Path(c["outputs"]) / name / "CLOSED.json"
    B.require(path.is_file(), f"stage not closed: {path}")
    closed = B.read_json(path)
    B.require(closed.get("status") == "PASS" and closed.get("contract_sha256") == c["contract_sha256"], f"stage {name} is not PASS")
    return closed


# ================================================================ pure pieces (tested on synthetic arrays)
def group_masks(area):
    size = T.size_bin(area)
    return {"small": np.asarray(area) <= T.SMALL_AREA, "medium": (size == 2) | (size == 3), "large": size == 4}


def clean(x):
    """Plain JSON values; a non-finite number (a ratio whose denominator is 0) becomes None."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    return x


def mean_of(rows):
    """Field-wise mean of dicts of numbers (None if any is None); 'k' stays as it is."""
    rows = list(rows)
    out = {}
    for key in rows[0]:
        v = [r[key] for r in rows]
        out[key] = v[0] if key in ("k", "group", "images") else None if any(x is None or not np.isfinite(x) for x in v) else float(np.mean(v))
    return out


def avg_rank(x):
    _, inv, cnt = np.unique(np.asarray(x, dtype=np.float64), return_inverse=True, return_counts=True)
    end = np.cumsum(cnt)
    return ((end - cnt + end - 1) / 2.0 + 1.0)[inv]


def rank_correlation(x, y):
    a, b = avg_rank(x), avg_rank(y)
    a, b = a - a.mean(), b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return None if den == 0 else float((a * b).sum() / den)


def by_image(records, ids):
    """Stored top600 records -> {image id: its 600 records, stored order}."""
    B.require(len(records) == len(ids) * 600, "stored top600 record count")
    out = {}
    for i in range(0, len(records), 600):
        part = records[i:i + 600]
        iid = part[0]["image_id"]
        B.require(all(p["image_id"] == iid for p in part) and iid not in out, "stored records are not grouped by image")
        out[iid] = part
    B.require(set(out) == set(ids), "stored records: image set")
    return out


def splice(ids, route, cheap, expensive):
    """Task 4.6: development order; image i in R takes the expensive arm's 600 records, otherwise the cheap arm's."""
    B.require(len(route) == len(ids), "assignment length")
    preds = []
    for i, iid in enumerate(ids):
        preds.extend(expensive[iid] if route[i] else cheap[iid])
    return preds


def assignment(order, k, m=M):
    r = np.zeros(m, dtype=bool)
    r[np.asarray(order)[:k]] = True
    return r


def assignments_of(order_p, order_n, m=M):
    out = {f"P-k{k}": assignment(order_p, k, m) for k in R.K_CURVE}
    out.update({f"N-k{k}": assignment(order_n, k, m) for k in K_SCORED["N"]})
    for r, sub in enumerate(R.balanced_subsets(m)):
        a = np.zeros(m, dtype=bool)
        a[sub] = True
        out[f"RAND-r{r}"] = a
    return out


def stat_note(stat):
    s = np.sort(np.asarray(stat, dtype=np.float64))
    gaps = np.diff(s)
    return {"min": float(s[0]), "max": float(s[-1]), "exact_ties_between_neighbours": int((gaps == 0).sum()),
            "smallest_positive_gap": float(gaps[gaps > 0].min()) if bool((gaps > 0).any()) else None}


def hit_tables(inp, m=M):
    """T1-T4, T6, the per-image columns and the assignments from the per-GT arrays.  inp: image (n,), area (n,),
    iou {unit: (n,)}, demand (m, 12, 21), d_star (m, 12, 21)."""
    image, iou = inp["image"], inp["iou"]
    masks = group_masks(inp["area"])
    n = {g: R.per_image(masks[g], image, m) for g in masks}

    def hits(arm, s, group="small", thr=0.75):
        return R.per_image((iou[f"{arm}-s{s}"] >= thr) & masks[group], image, m)

    a, b, f = ({s: hits(arm, s) for s in SEEDS} for arm in ("L0", "S15", "FG"))
    d = {s: b[s] - a[s] for s in SEEDS}
    stat = {"P": inp["demand"].sum(axis=(1, 2)), "N": n["small"], "Dstar": inp["d_star"].sum(axis=(1, 2))}
    order = {k: R.order_desc(v) for k, v in stat.items()}
    order_o = {s: R.order_desc(d[s]) for s in SEEDS}

    def rows(cheap, expensive, count, order_of, ks):
        out = {}
        for k in ks:
            per = {f"s{s}": R.readings(cheap[s], expensive[s], count, order_of(s), k) for s in SEEDS}
            per["mean"] = mean_of(per.values())
            out[f"k{k}"] = per
        return out

    t1 = {name: rows(a, b, n["small"], lambda s, o=order[name]: o, R.K_CURVE) for name in ("P", "N", "Dstar")}
    w = R.boot_weights(m)
    kg = R.K_GATE
    t1["P_k158_Q_ci95"] = {**{f"s{s}": R.q_bootstrap(a[s], b[s], n["small"], order["P"], kg, w) for s in SEEDS},
                           "mean": R.q_bootstrap(np.mean([a[s] for s in SEEDS], 0), np.mean([b[s] for s in SEEDS], 0), n["small"], order["P"], kg, w)}
    pairs = ((0, 1), (0, 2), (1, 2))
    cross = {}
    for t in SEEDS:
        for s in SEEDS:
            if s != t:
                cross[f"s{t}->s{s}"] = {f"k{k}": R.readings(a[s], b[s], n["small"], order_o[t], k) for k in R.K_CURVE}
    cross["mean"] = {f"k{k}": mean_of(v[f"k{k}"] for v in cross.values()) for k in R.K_CURVE}
    t2 = {"O": rows(a, b, n["small"], lambda s: order_o[s], R.K_CURVE), "X": cross,
          "rank_correlation": {
              "d_s_d_t": {f"s{x}-s{y}": rank_correlation(d[x], d[y]) for x, y in pairs},
              "T_N": rank_correlation(stat["P"], stat["N"]), "T_Dstar": rank_correlation(stat["P"], stat["Dstar"]),
              "T_d": {f"s{s}": rank_correlation(stat["P"], d[s]) for s in SEEDS},
              "N_d": {f"s{s}": rank_correlation(stat["N"], d[s]) for s in SEEDS}},
          "P_k158_Q_pairings": {f"L0-s{s}|S15-s{t}": {k: v for k, v in R.readings(a[s], b[t], n["small"], order["P"], kg).items()
                                                      if k in ("Q_pp", "lead_pp", "capture", "routed_pp")} for s in SEEDS for t in SEEDS}}

    def group_rows(order_, s):
        out = []
        tot_n, tot_d = n["small"].sum(), d[s].sum()
        for j, g in enumerate(R.groups5(order_)):
            gn, ga, gb = n["small"][g].sum(), a[s][g].sum(), b[s][g].sum()
            out.append({"group": j + 1, "images": int(len(g)), "small_gt": float(gn), "small_gt_share": float(gn / tot_n),
                        "L0_hit75_pp": None if gn == 0 else float(100 * ga / gn), "S15_hit75_pp": None if gn == 0 else float(100 * gb / gn),
                        "lead_per_gt_pp": None if gn == 0 else float(100 * (gb - ga) / gn), "share_of_total_lead": float((gb - ga) / tot_d)})
        return out

    t3 = {}
    for name in ("N", "P"):
        per = {f"s{s}": group_rows(order[name], s) for s in SEEDS}
        per["mean"] = [mean_of(per[f"s{s}"][j] for s in SEEDS) for j in range(5)]
        t3[name] = per
    t4 = {}
    for name in ("P", "N"):
        t4[name] = {}
        for label, group, thr in T4_ROWS:
            ca, cb = ({s: hits(arm, s, group, thr) for s in SEEDS} for arm in ("L0", "S15"))
            for k, v in rows(ca, cb, n[group], lambda s, o=order[name]: o, (158, 219, 341)).items():
                t4[name].setdefault(k, {})[label] = v
    t6 = {name: rows(f, b, n["small"], lambda s, o=order[name]: o, R.K_CURVE) for name in ("P", "N")}
    assign = assignments_of(order["P"], order["N"], m)
    columns = {"T": stat["P"], "N": stat["N"], "Dstar": stat["Dstar"]}
    for arm, src in (("L0", a), ("FG", f), ("S15", b)):
        columns.update({f"hit75_small_{arm}_s{s}": src[s] for s in SEEDS})
    columns.update({f"P_k{k}": assign[f"P-k{k}"].astype(np.float64) for k in R.K_CURVE})
    gate = {"q_hit": [t1["P"][f"k{kg}"][f"s{s}"]["Q_pp"] for s in SEEDS], "lead_hit": [t1["P"][f"k{kg}"][f"s{s}"]["lead_pp"] for s in SEEDS],
            "q_hit_count": [t1["N"][f"k{kg}"][f"s{s}"]["Q_pp"] for s in SEEDS]}
    mixing = {}
    for label, group, thr in (("small_hit75", "small", 0.75),) + T4_ROWS:
        for s in SEEDS:
            ca, cb = hits("L0", s, group, thr), hits("S15", s, group, thr)
            routed = [100 * (ca.sum() + (cb - ca)[assign[f"RAND-r{r}"]].sum()) / n[group].sum() for r in range(R.RESIDUES)]
            mixing[f"{label}|s{s}"] = float(np.mean(routed) - 100 * (ca.sum() + (2.0 / R.RESIDUES) * (cb - ca).sum()) / n[group].sum())
    res = {"T1": t1, "T2": t2, "T3": t3, "T4": t4, "T6": t6, "gate": gate, "hit_level_mixing_pp": mixing,
           "group_counts": {g: int(masks[g].sum()) for g in masks}, "statistics": {k: stat_note(v) for k, v in stat.items()},
           "small_hit75_pp": {arm: [float(100 * src[s].sum() / n["small"].sum()) for s in SEEDS] for arm, src in (("L0", a), ("S15", b), ("FG", f))}}
    return {**clean(res), "_assignments": assign, "_columns": columns}


def ap_tables(values, k_of):
    """T5 from the scorer values.  values[f's{s}'][row][metric] (0..1); rows: L0, S15 and SCORED; k_of[row] = |R|."""
    t5 = {}
    for s in SEEDS:
        v = values[f"s{s}"]
        lo, hi = v["L0"], v["S15"]
        routed = {}
        for name in SCORED:
            k = k_of[name]
            routed[name] = {"k": int(k), "p": k / M, "budget": R.budget(k / M), "values": {m: v[name][m] for m in METRICS},
                            "Q_pp": {m: R.q_metric(v[name][m], lo[m], hi[m], k) for m in METRICS},
                            "minus_chord_pp": {m: 100.0 * (v[name][m] - (lo[m] + k / M * (hi[m] - lo[m]))) for m in METRICS}}
        rnd = {m: [v[f"RAND-r{r}"][m] for r in range(R.RESIDUES)] for m in METRICS}
        t5[f"s{s}"] = {"endpoints": {"L0": {m: lo[m] for m in METRICS}, "S15": {m: hi[m] for m in METRICS}},
                       "lead_pp": {m: 100.0 * (hi[m] - lo[m]) for m in METRICS}, "routed": routed,
                       "random": {"mean": {m: float(np.mean(rnd[m])) for m in METRICS},
                                  "std_pp": {m: float(100.0 * np.std(rnd[m], ddof=1)) for m in METRICS},
                                  "mixing_pp": {m: R.mixing(rnd[m], lo[m], hi[m]) for m in METRICS}}}
    per = [t5[f"s{s}"] for s in SEEDS]
    t5["mean"] = {"endpoints": {arm: mean_of(p["endpoints"][arm] for p in per) for arm in ARMS}, "lead_pp": mean_of(p["lead_pp"] for p in per),
                  "routed": {name: {"k": k_of[name], "values": mean_of(p["routed"][name]["values"] for p in per),
                                    "Q_pp": mean_of(p["routed"][name]["Q_pp"] for p in per),
                                    "minus_chord_pp": mean_of(p["routed"][name]["minus_chord_pp"] for p in per)} for name in SCORED},
                  "random": {key: mean_of(p["random"][key] for p in per) for key in ("mean", "std_pp", "mixing_pp")}}
    return clean(t5)


def verdict_of(gate, t5):
    key = f"P-k{R.K_GATE}"
    q_aps = [t5[f"s{s}"]["routed"][key]["Q_pp"]["APS"] for s in SEEDS]
    q_ap = [t5[f"s{s}"]["routed"][key]["Q_pp"]["AP"] for s in SEEDS]
    lead_aps = [t5[f"s{s}"]["lead_pp"]["APS"] for s in SEEDS]
    v = R.verdict(gate["q_hit"], gate["lead_hit"], q_aps, lead_aps, q_ap, gate["q_hit_count"])
    v["readings_pp"] = {"Q_hit": gate["q_hit"], "lead_hit": gate["lead_hit"], "Q_APS": q_aps, "lead_APS": lead_aps, "Q_AP": q_ap,
                        "Q_hit_order_N": gate["q_hit_count"],
                        "means": {"Q_hit": float(np.mean(gate["q_hit"])), "lead_hit": float(np.mean(gate["lead_hit"])),
                                  "Q_APS": float(np.mean(q_aps)), "lead_APS": float(np.mean(lead_aps)), "Q_AP": float(np.mean(q_ap)),
                                  "Q_hit_order_N": float(np.mean(gate["q_hit_count"]))}}
    v["registered_with_the_label"] = {"order_N_clears_G1": v["count_order_clears_G1"],
                                      "mixing_pp": {m: {**{f"s{s}": t5[f"s{s}"]["random"]["mixing_pp"][m] for s in SEEDS},
                                                        "mean": t5["mean"]["random"]["mixing_pp"][m]} for m in ("APS", "AP")}}
    return clean(v)


# ================================================================ stages
def load_inputs(c):
    f = c["inputs"]
    z = np.load(B.pinned(f["gt"]), allow_pickle=False)
    sc = z["scored"]
    image, area = z["image"][sc].astype(np.int64), z["area"][sc].astype(np.float64)
    o = np.load(B.pinned(f["outcomes"]), allow_pickle=False)
    iou = {u: o[f"iou_{u}"] for u in UNITS}
    demand, d_star = (np.load(B.pinned(f[k]), allow_pickle=False) for k in ("demand", "d_star"))
    B.require(len(image) == N_GT and bool((np.diff(image) >= 0).all()) and image[0] >= 0 and image[-1] < M
              and all(v.shape == (N_GT,) for v in iou.values()), "per-GT arrays")
    B.require(demand.shape == d_star.shape == (M, 12, 21) and demand.dtype == d_star.dtype == np.float64
              and bool(np.isfinite(demand).all()) and bool(np.isfinite(d_star).all()), "demand maps")
    return {"image": image, "area": area, "iou": iou, "demand": demand, "d_star": d_star}


def wire(c, out, counts):
    B.PHASE["phase"] = "wire/files"
    files = {**{k: B.ref(B.pinned(v)) for k, v in c["inputs"].items()}, **{k: B.ref(B.pinned(v)) for k, v in c["predictions"].items()}}
    for name, (_, _, size, sha) in INPUT_PINS.items():
        B.require((files[name]["bytes"], files[name]["sha256"]) == (size, sha), f"4.7 item 1: {name}")
    for unit, pin in PRED_PINS.items():
        B.require((files[unit]["bytes"], files[unit]["sha256"]) == pin, f"4.7 item 1: {unit}")
    B.PHASE["phase"] = "wire/selftest"
    R.selftest()
    text = text_against_reference()
    B.require(bool(np.array_equal(R.boot_weights(M), T.boot_weights(M))), "c0_ref.boot_weights differs from the phase-1 bootstrap")
    B.PHASE["phase"] = "wire/arrays"
    inp = load_inputs(c)
    masks = group_masks(inp["area"])
    got = {g: int(masks[g].sum()) for g in masks}
    B.require(got == GROUP_COUNTS, f"group counts: {got}")
    hit = {}
    for arm, reg in REG_HIT.items():
        hit[arm] = [float(100 * (inp["iou"][f"{arm}-s{s}"][masks["small"]] >= 0.75).mean()) for s in SEEDS]
        B.require(max(abs(x - y) for x, y in zip(hit[arm], reg)) <= TOL, f"4.7 item 2: small hit75 of {arm}: {hit[arm]}")
    d = inp["demand"]
    dm = {"min": float(d.min()), "max": float(d.max()), "mean": float(d.mean())}
    B.require(all(abs(dm[k] - REG_DEMAND[k]) <= 1e-12 for k in dm), f"4.7 item 3: demand statistics: {dm}")
    B.PHASE["phase"] = "wire/mixing"
    image = inp["image"]
    subsets = R.balanced_subsets(M)
    mixing = {}
    for label, group, thr in (("small_hit75", "small", 0.75),) + T4_ROWS:
        n = R.per_image(masks[group], image, M)
        for s in SEEDS:
            a, b = (R.per_image((inp["iou"][f"{arm}-s{s}"] >= thr) & masks[group], image, M) for arm in ARMS)
            routed = [100 * (a.sum() + (b - a)[sub].sum()) / n.sum() for sub in subsets]
            mixing[f"{label}|s{s}"] = float(np.mean(routed) - 100 * (a.sum() + (2.0 / R.RESIDUES) * (b - a).sum()) / n.sum())
    B.require(max(abs(v) for v in mixing.values()) <= TOL, f"4.7 item 5: hit-level mixing: {mixing}")
    B.require(counts["forwards"] == 0, "wire must not run a model")
    return {"stage": "wire", "passed": True,
            "checks": {"item1_files": files, "item2_small_hit75_pp": hit, "item3_demand": dm, "item5_hit_level_mixing_pp": mixing,
                       "item6_c0_ref_selftest": "PASS", "text_against_c0_ref": text, "group_counts": got,
                       "bootstrap_weights_equal_to_phase1": True, "tolerance": TOL}}


class Scorer:
    """The phase-1 score stage's call path on a spliced list of the stored records of one seed."""

    def __init__(self, c, seed, counts):
        from infer_seed import bridge_values
        from sparse_rtdetr.r0.evaluation import score_budget
        import th2_phase1 as P
        self.c, self.seed, self.counts = c, seed, counts
        self.score_budget, self.bridge_values = score_budget, bridge_values
        self.ids = c["development_ids"]
        self.refs = {arm: c["predictions"][f"{arm}-s{seed}"] for arm in ARMS}
        self.text = {arm: B.pinned(self.refs[arm]).read_bytes() for arm in ARMS}
        data = P.development(c)
        self.rows = data.records
        self.parsed = [data.raw(r) for r in self.rows]

    def stored(self, arm):
        return json.loads(self.text[arm])

    def score(self, out, name, route):
        route = np.asarray(route)
        B.require(route.shape == (M,) and route.dtype == np.bool_, "assignment array")
        lists = {arm: self.stored(arm) for arm in ARMS}
        preds = splice(self.ids, route, by_image(lists["L0"], self.ids), by_image(lists["S15"], self.ids))
        B.require(len(preds) == M * 600, "spliced record count")
        same = {arm: preds == lists[arm] for arm in ARMS}
        metrics = self.score_budget(self.rows, self.parsed, preds, self.c["source_root"])
        self.counts.update(score_budget_calls=1)
        B.require({k: metrics["spatial_filter_counts"][k] for k in GT_COUNTS} == GT_COUNTS, "scored GT set")
        metrics["TH2C0_binding"] = {"contract_sha256": self.c["contract_sha256"], "seed": self.seed, "row": name,
                                    "routed_images": int(route.sum()), "assignment_sha256": B.sha_bytes(route.astype(np.uint8).tobytes()),
                                    "predictions": self.refs, "spliced_list_equals_stored_list": same}
        path = out / f"{name}.top600-metrics.json"
        B.write_new(path, metrics)
        return self.bridge_values(metrics), B.ref(path), same


def ends(c, out, counts, unit):
    seed = int(unit[1:])
    B.PHASE["phase"] = f"ends/{unit}"
    B.require(stage_closed(c, "wire")["passed"] is True, "wiring check")
    sc = Scorer(c, seed, counts)
    values, files, diff, same = {}, {}, {}, {}
    for arm, route in (("L0", np.zeros(M, dtype=bool)), ("S15", np.ones(M, dtype=bool))):
        v, ref, eq = sc.score(out, arm, route)
        reg = c["registered"]["top600"][f"{arm}-s{seed}"]
        values[arm], files[arm], same[arm] = v, ref, eq[arm]
        diff[arm] = max(abs(v[m] - reg[m]) for m in METRICS)
        B.require(diff[arm] <= TOL, f"4.7 item 4: {arm}-s{seed} scored {v}, registered {reg}")
    B.require(counts["forwards"] == 0 and counts["score_budget_calls"] == 2, "ends: counts")
    return {"stage": "ends", "unit": unit, "passed": True, "values_600": values, "files": files,
            "max_abs_difference_from_registered": diff, "spliced_list_equals_stored_file": same}


def hits(c, out, counts):
    B.PHASE["phase"] = "hits"
    B.require(not c["dev"], "a dev round runs the wiring checks only (task section 8)")
    for s in SEEDS:
        B.require(stage_closed(c, f"ends-s{s}")["passed"] is True, "4.7 item 4 must pass before any reading")
    res = hit_tables(load_inputs(c))
    assign, columns = res.pop("_assignments"), res.pop("_columns")
    B.require(res["group_counts"] == GROUP_COUNTS and max(abs(v) for v in res["hit_level_mixing_pp"].values()) <= TOL, "hits: wiring")
    with (out / "assignments.npz").open("xb") as f:
        np.savez(f, **assign)
    names = list(columns)
    with (out / "per_image.csv").open("x", newline="") as f:
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(["index", "image_id"] + names)
        for i, iid in enumerate(c["development_ids"]):
            wr.writerow([i, iid] + [repr(float(columns[k][i])) if k in ("T", "Dstar") else int(columns[k][i]) for k in names])
    B.write_new(out / "HITS.json", res)
    B.require(counts["forwards"] == 0, "hits must not run a model")
    return {"stage": "hits", "result": B.ref(out / "HITS.json"),
            "files": {"assignments": B.ref(out / "assignments.npz"), "per_image": B.ref(out / "per_image.csv")},
            "routed_images": {k: int(v.sum()) for k, v in assign.items()}, "gate": res["gate"]}


def routed(c, out, counts, unit):
    seed = int(unit[1:])
    B.PHASE["phase"] = f"routed/{unit}"
    B.require(not c["dev"], "a dev round runs the wiring checks only (task section 8)")
    h = stage_closed(c, "hits")
    assign = np.load(B.pinned(h["files"]["assignments"]), allow_pickle=False)
    sc = Scorer(c, seed, counts)
    values, files = {}, {}
    for name in SCORED:
        B.PHASE["phase"] = f"routed/{unit}/{name}"
        route = assign[name]
        B.require(int(route.sum()) == h["routed_images"][name], f"assignment {name}")
        values[name], files[name], _ = sc.score(out, name, route)
    B.require(counts["forwards"] == 0 and counts["score_budget_calls"] == len(SCORED), "routed: counts")
    return {"stage": "routed", "unit": unit, "values_600": values, "files": files}


def tables(c, out, counts):
    B.PHASE["phase"] = "tables"
    B.require(not c["dev"], "a dev round runs the wiring checks only (task section 8)")
    h = stage_closed(c, "hits")
    res = B.read_json(B.pinned(h["result"]))
    values, metric_files, stages = {}, {}, {}
    names = ["wire", "hits"] + [f"{a}-s{s}" for a in ("ends", "routed") for s in SEEDS]
    for name in names:
        st = stage_closed(c, name)
        stages[name] = {"elapsed_seconds": st["elapsed_seconds"], "process_cpu_seconds": st["process_cpu_seconds"], "counts": st["counts"],
                        "started_at_unix": st["started_at_unix"], "closed_at": st["closed_at"]}
        if "-s" in name:
            values.setdefault(name.split("-")[1], {}).update(st["values_600"])
            metric_files.update({f"{name.split('-')[1]}/{k}": v for k, v in st["files"].items()})
    k_of = h["routed_images"]
    t5 = ap_tables(values, k_of)
    verdict = verdict_of(res["gate"], t5)
    calls = sum(v["counts"].get("score_budget_calls", 0) for v in stages.values())
    B.require(calls == SCORINGS == len(metric_files) == 39, f"scoring count: {calls}")
    first, last = min(v["started_at_unix"] for v in stages.values()), time.time()
    used = sum(p.stat().st_size for p in Path(c["home"]).rglob("*") if p.is_file() and not p.is_symlink())
    result = {"status": "PASS", "version": VERSION, "contract_sha256": c["contract_sha256"], "source_commit": c["source_commit"],
              "source_branch": c["source_branch"], "task": c["task"], "reference": c["reference"], "dev": c["dev"],
              "verdict": verdict, "T1": res["T1"], "T2": res["T2"], "T3": res["T3"], "T4": res["T4"], "T5": t5, "T6": res["T6"],
              "statistics": res["statistics"], "group_counts": res["group_counts"], "small_hit75_pp": res["small_hit75_pp"],
              "hit_level_mixing_pp": res["hit_level_mixing_pp"], "routed_images": k_of,
              "wire": stage_closed(c, "wire")["checks"],
              "ends": {f"s{s}": {k: stage_closed(c, f"ends-s{s}")[k] for k in ("max_abs_difference_from_registered", "spliced_list_equals_stored_file")}
                       for s in SEEDS},
              "metric_files": metric_files, "per_image_file": h["files"]["per_image"], "assignments_file": h["files"]["assignments"],
              "workload": {"stages": stages, "score_budget_calls": calls, "model_forwards": 0, "updates": 0, "gpu_seconds": 0,
                           "sum_of_stage_wall_seconds": sum(v["elapsed_seconds"] for v in stages.values()),
                           "sum_of_stage_cpu_seconds": sum(v["process_cpu_seconds"] for v in stages.values()),
                           "wall_seconds_first_stage_start_to_tables": last - first,
                           "chain_run_seconds_in_home_including_dev_rounds": chain_seconds(c["home"]),
                           "bytes_in_home_before_this_file": used},
              "boundaries": c["boundaries"], "definitions": c["definitions"], "verdict_rule": c["verdict_rule"]}
    B.write_new(out / "RESULT.json", result)
    return {"stage": "tables", "result": B.ref(out / "RESULT.json"), "label": verdict["label"], "readings_pp": verdict["readings_pp"]}


# ================================================================ CLI
UNIT_NAMES = tuple(f"s{s}" for s in SEEDS)
STAGES = {"wire": (wire, None), "ends": (ends, UNIT_NAMES), "hits": (hits, None), "routed": (routed, UNIT_NAMES), "tables": (tables, None)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("storage", "prepare") + tuple(STAGES))
    ap.add_argument("--home", type=Path)
    ap.add_argument("--round", type=Path)
    ap.add_argument("--release")
    ap.add_argument("--unit")
    args = ap.parse_args(argv)
    B.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "every action of this tool requires CUDA_VISIBLE_DEVICES=\"\"")
    if args.action == "storage":
        B.require(args.home and not (args.round or args.release or args.unit), "storage needs --home only")
        return storage(args.home)
    B.require(args.round and args.home is None, "--round is required; --home is for storage")
    if args.action == "prepare":
        B.require(args.release and args.unit is None, "prepare needs --release and no --unit")
        return prepare(args.round, args.release)
    run, units = STAGES[args.action]
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
        setup("cpu")
        result = run(c, out, counts, args.unit) if units else run(c, out, counts)
        B.require(isinstance(result, dict) and counts["actual_updates"] == 0 and counts["forwards"] == 0 and counts["forward_attempts"] == 0,
                  f"{args.action}: stage result")
        children = os.times()
        B.write_new(out / "CLOSED.json", {"status": "PASS", "action": args.action, "contract_sha256": c["contract_sha256"],
                                          "counts": dict(counts), "started_at_unix": time.time() - (time.monotonic() - start),
                                          "elapsed_seconds": time.monotonic() - start,
                                          "process_cpu_seconds": time.process_time() - cpu0 + children.children_user + children.children_system,
                                          "closed_at": now(), **result})
    except BaseException as exc:
        B.write_new(out / "STOP.json", {"status": "STOP", "phase": B.PHASE["phase"], "action": args.action, "unit": args.unit,
                                        "error": repr(exc)[:4000], "traceback": traceback.format_exc(),
                                        "counts": dict(counts), "elapsed_seconds": time.monotonic() - start})
        raise
    finally:
        gc.collect()


if __name__ == "__main__":
    main()
