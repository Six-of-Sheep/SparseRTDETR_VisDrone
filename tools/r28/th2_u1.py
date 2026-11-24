"""TH2 U1 (TASK_TH2_U1_001): one uniform intermediate canvas, U896 = 896 x 1568, seed 0, trained once from the certified
e060 state with the S15 recipe and evaluated once at the e120 EMA end point.  The reading functions are u1_ref.py and
c0_ref.py (the controller's files, byte-identical copies).  The accepted S15 code (bridge, activation checkpointing,
save / restore, formal loop, acceptance, evaluation forward) is reused through the arm registry of scale_baseline.py;
nothing of it is rewritten here.

usage: th2_u1.py prepare --round <round> --release <name> [--home <formal round>]
       th2_u1.py run <action> --round <round> [--unit k] [--rerun]         one stage under the supervisor
       th2_u1.py chain --round <round> [--resume-train]                     train -> align -> infer -> score -> routed -> readings
       th2_u1.py status --round <round>
       th2_u1.py stage <action> --round <round> --name <directory> ...       (internal: the stage process itself)

  t0-wire     CPU: task 4.7 items 1, 2, 3, 5, 6 and the agreement of the task text with u1_ref.py / c0_ref.py.
  t0-ends     CPU: task 4.7 item 4.  The splice function with R = empty and R = all, scored (2 of the 5 scorings).
  t0-predict  CPU: per-GT outcomes of A0-s0 and PREDICTION.json (written once, before any training).
  a0-count    CPU, 6 units: the native 768 loader replayed over the 18,240 windows; every receipt field must equal the
              stored A0 trajectory; the supervised targets are counted (the stored trajectory has no such field).
  cpu         CPU gate of the trainer on the 896 x 1568 canvas (task 4.4 items 1-3) and the refusal cases.
  eval-check  CPU checks before the evaluation (GT-free pixels, weight selection, refusals, one 4-image forward).
  accept      CUDA minimal acceptance (3 updates, 6 training forwards); at most 3 rounds.
  train       CUDA: the formal training, exactly once (exact restores of an interrupted run: at most 3).
  align       CPU: the 18,240 input rows against the native 768 trajectory.
  infer       CUDA: the e120 EMA end point, 548 images, top600 / top300.
  score       CPU: 2 scorings (U896-s0 top600, top300).
  routed      CPU: 1 scoring (splice P-k158 of {A0-s0, S15-s0}).
  readings    CPU: U1-U6, per_gt.npz, per_image.csv, RESULT.json.

The home directory is the formal round.  A dev round (directory name dev-round*) runs t0-wire, cpu and eval-check only:
they produce no reading and no scoring.  Every stage writes <round>/<name>/CLOSED.json (PASS) or STOP.json.
"""
import argparse
import csv
import datetime
import fcntl
import gc
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
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
import th2_c0 as C0  # noqa: E402
import th2_core as T  # noqa: E402
import u1_ref as U  # noqa: E402

VERSION = "TH2-U1-001-v1"
TASK = ("TASK_TH2_U1_001.md", "fbc818ea21ad5f79c4f0e286b236029831b719f331d405c8327dc35d217c9e7f")
REFS = {"u1_ref": ("tools/r28/u1_ref.py", "0cf6f33a34027aab0077c0c67eb3b8445f2f3f648379c31de19ad964f812eb30"),
        "c0_ref": ("tools/r28/c0_ref.py", "c55f9dfab5e4804f53b7aa87fd5aaebe7d08c0d7e50698f07c6a45498ffb3653")}
E = B.WS / "experiments"
LD = B.LD_ROOT
STUDY = E / "thesis-study2"
BASE_MANIFEST = (STUDY / "th2-c0-20261123-001/release-001/MANIFEST.json",
                 "189c582c9ac181c09f88c35110abfffe1142410ebd05c9596be3c9904552f9da", "7e8f9615b4486ef00af14e86221eec92fa22fd48")
ALLOWED_ADDED = ("tools/r28/u1_ref.py", "tools/r28/th2_u1.py", "tests/test_th2_u1.py")
ALLOWED_CHANGED = ("src/sparse_rtdetr/research/scale_baseline.py", "tools/scale_baseline/s15_formal_train.py",
                   "tools/scale_baseline/s15_formal_eval.py")       # the arm registry (task 4.3)
ARM, SEED, HW = "U896", 0, (896, 1568)
M, N_GT, GT_COUNTS = 548, 38599, {"GT_before": 38759, "GT_after": 38599}
GROUP_COUNTS = {"small": 26441, "medium": 11090, "large": 1068}
NATIVE = ((1920, 1080), (1360, 765), (960, 540))               # (width, height) of the development images
NATIVE_COUNTS = ((19, 829), (408, 19380), (121, 6232))        # images, small GT (task 4.5)
NATIVE_SCALE_4DP = (0.7055, 0.9960, 1.4111)
PARENT_UPDATE, END_UPDATE, PER_EPOCH = 18240, 36480, 304
SAVE_EPOCHS = (60, 75, 90, 105, 120)
METRICS = ("AP", "APS", "AP75_small", "APM", "APL")
TOL = 1e-9
CAP_BYTES = 4_000_000_000
NO_PROGRESS_SECONDS = 1800
ACCEPT_ROUNDS, RESTORES = 3, 3
ALLOCATOR = "expandable_segments:True"
S15_ROUND = LD / "s15-scale-baseline-20261105-001"
S15_CONTRACT = (S15_ROUND / "formal-002-CONTRACT.json", "14e502a69a55496b53be8a89d078fba20322185731ebbe8f46d519b8fea852a1", 60064)
S15_E060 = S15_ROUND / "formal-002/train-S15/checkpoint-epoch-060.pth"     # a stored S15 state: cross-arm refusal, S15 regression
S15_GT_TOTAL = 13312115                                                    # task section 6, U6
P1 = STUDY / "th2-phase1-20261122-001"
P1_CONTRACT_SHA = "2471ad9271ebbae0c645cc283de137118e9571d1aefadf7d321028414ae8788d"
OTHER_SEED_PARENTS = {1: LD / "r18-sahi-seed12-20261016-001/seed1/formal/checkpoint-epoch-060.pth",
                      2: LD / "r18-sahi-seed12-20261016-001/protocol-resume-20261017-001/seed2/formal/checkpoint-epoch-060.pth"}
# task 4.1: name -> (path, bytes, sha256)
PINS = {
    "parent": (LD / "rtdetrv2-r18-common-protocol-seed0-e120-20261013-001/epoch-recovery-20261014-001/formal/checkpoint-epoch-060.pth",
               323460289, "a2b6412e3c85de7592829f24aff09e355fdd11cb53798bc498bd04b882e1bf8a"),
    "a0_trajectory": (LD / "r25-query-local-regression-20261104-001/formal-001/train-A0/UPDATES.jsonl",
                      15640076, "efc1606d336cbbccfa3b0abd53d8e24bf3c536ed054062c64dc962e6aaaf5172"),
    "equivalence": (S15_ROUND / "formal-002/accept/A0_768_EQUIVALENCE.json",
                    4428, "c000130d8d2b690a78d7b031556296b37565ed3ce50b9b1f56d9d685dadd87fe"),
    "data_receipt": (LD / "r18-tiny-positives-20261102-001/production-contract-002.json",
                     60184, "b0fe33198d8abc7f8b89a1cc9824de9abdba8721580932aa07be565ce40d896b"),
    "evaluation_protocol": (LD / "rtdetrv2-r18-common-protocol-seed0-e120-20261013-001/release-002/EVALUATION_PROTOCOL.json",
                            941, "860285aa5af367b2bf2fef2221c15ac3914b33c57eb765e09f76b9b580004a82"),
    "A0-s0": (LD / "r25-query-local-regression-20261104-001/formal-001/evaluation/A0-EMA/top600-predictions.json",
              96444850, "95eb697cd4bd30a1bebdebd052b88d4cd71d22bd3ae8fd96a018bcef8c91c0e0"),
    "S15-s0": (S15_ROUND / "formal-002/evaluation/S15-EMA/top600-predictions.json",
               96433942, "f8e89847c4ebcc67f8ed08b8f794099f7459b9051b096517656afcb6b6af0ac5"),
    "outcomes": (P1 / "e0/outcomes.npz", 8035062, "72f4b1fee92055a595983c49de33310a354748b537030465b1782673cd98fc52"),
    "gt": (P1 / "geom/gt.npz", 6243855, "11dab33f84ef440b2f68f02651b89fda644279d582e9223c5a1d08bcfe0716c1"),
    "assignments": (STUDY / "th2-c0-20261123-001/hits/assignments.npz", 11914,
                    "2cc5ef36237931a3099c6edf82c4c040a0ff53784f52b21577739aea0bb18698"),
}
# task 4.1, registered values: AP, APS, AP75_small, APM, APL
REG_AP = {
    "A0-s0": {"600": (0.2978928000787488, 0.2217618575337924, 0.19968104528445318, 0.39344481665007297, 0.4725232045664882),
              "300": (0.29347668234713176, 0.2177929184225555, 0.19712874220974094, 0.3883243218218308, 0.4686511018043208)},
    "S15-s0": {"600": (0.3200866660864559, 0.2513747771125344, 0.23720953090636165, 0.40495283253729625, 0.4446219718630393),
               "300": (0.3158812870411125, 0.24734999661182375, 0.234330056498393, 0.4013240711944936, 0.44069524292638923)},
    "L0-s0": {"600": (0.30054005272689926, 0.2238294714480465, 0.20231585685439496, 0.39561519581083376, 0.4951035581025013)},
    "FG-s0": {"600": (0.3132075511405779, 0.24417841447853297, 0.22563462872819284, 0.39927912432216967, 0.48974805105438635)},
    "routed_P-k158_L0_S15": {"600": (0.30996226302599766, 0.24036121897417648, 0.22416930799576062, 0.39666851523483976,
                                     0.49759205515048016)},
}
REG_HIT = {"L0-s0": 43.515752051737834, "S15-s0": 48.526908967134375, "FG-s0": 45.73957112060815,
           "routed_P-k158_L0_S15": 46.02700351726485}
REG_SOURCES = {"A0-s0": {"600": (LD / "r25-query-local-regression-20261104-001/formal-001/evaluation/A0-EMA-A-top600-metrics.json",
                                 "2b4556491834cbac2099caaa22979092614c9c5e649dde5aec85765d6358652a"),
                         "300": (LD / "r25-query-local-regression-20261104-001/formal-001/evaluation/A0-EMA-A-top300-metrics.json",
                                 "89ebae18c3bde3ceb32927792a0e01950b5badaeb7dd36ba9d9f3dd0fde5a217")}}
SLOTS = ("t0-ends/A0", "t0-ends/S15", "score/top600", "score/top300", "routed/P-k158")     # the 5 scorings of the task
WORKLOAD = {
    "train": {"arm": ARM, "seed": SEED, "epochs": [61, 120], "initial_update": PARENT_UPDATE, "updates_per_epoch": PER_EPOCH,
              "formal_updates": END_UPDATE - PARENT_UPDATE, "training_forwards": 2 * (END_UPDATE - PARENT_UPDATE),
              "stop_update": END_UPDATE, "save_epochs": list(SAVE_EPOCHS), "train_batch": 8, "accumulation": 2,
              "precision": "bfloat16", "clip_norm": 0.1, "input_hw": list(HW), "activation_checkpoint": True,
              "checkpoint_segments": ["backbone.res_layers", "encoder.fpn_blocks", "encoder.pan_blocks"],
              "allocator_conf": ALLOCATOR,
              # the two keys below are read by the reused acceptance function of the S15 trainer (its own names)
              "acceptance": {"S15_updates_per_seed": 3, "S15_training_forwards_per_seed": 6}},
    "acceptance_rounds_at_most": ACCEPT_ROUNDS, "exact_restores_at_most": RESTORES,
    "cpu_gate": {"fixture": "FP32, 1 x 2 per window, 896 x 1568, the first two images of each real window",
                 "updates": 5, "note": "2 + 1 (resume replay) with activation checkpointing, 2 without"},
    "evaluation": {"epoch": 120, "weights": "ema", "view": "A", "images": M, "batch": 4, "precision": "float32",
                   "input_hw": list(HW), "warmup_batches": 4, "steady_batches": 137, "budgets": [600, 300]},
    "scorings": list(SLOTS),
}
DEFINITIONS = {
    "evaluation_set": "development = official val, 548 images; the labels are read only for scoring and per-GT outcomes",
    "GT": "the 38,599 scored development GT of phase 1 (geom/gt.npz rows with scored = True), image order",
    "groups": "annotation area: small <= 32^2 (26,441); medium = (32^2, 96^2) (11,090); large >= 96^2 (1,068)",
    "iou": "best same-image same-class IoU over the 600 stored candidates: th2_phase1.stored_index + outcomes_of; hit75 = iou >= 0.75, "
           "hit50 = iou >= 0.5",
    "endpoints": "a = A0-s0 (768 x 1344), u = U896-s0 (896 x 1568), b = S15-s0 (1152 x 2016), l = L0-s0, fg = FG-s0.  a and u from "
                 "their own top600 lists; b, l, fg from phase-1 e0/outcomes.npz",
    "per_image": "c0_ref.per_image(values, image, 548); n = small GT per image",
    "f": "u1_ref.fraction(u, a, b) = (U896 - A0) / (S15 - A0); interval u1_ref.fraction_bootstrap with c0_ref.boot_weights(548)",
    "routed": "R = C0 assignments.npz P-k158 (158 images): the images of R read S15-s0, the others A0-s0; "
              "u1_ref.delta_hit / delta_hit_bootstrap / stage1_reading",
    "native_groups": "development images by (width, height): 1920x1080, 1360x765, 960x540; scale_l = sqrt((1344 / W) * (768 / H))",
    "prediction_1": "u1_ref.CHORD", "prediction_2": "u1_ref.LOGLINEAR",
    "prediction_3": "u1_ref.group_concave(lead, count, scale_l)['f']; lead[g] = 100 * sum over the images of g of (b - a) / small GT "
                    "of g; count[g] = small GT of g",
    "prediction_4": "u1_ref.shift_prediction(s, category, hit75 of A0, small) over all 38,599 GT, then shift_summary(pred, small "
                    "hit75 of A0, of S15)['f_law'].  Per native group: the same curve, target = small and in the group, observed = the "
                    "group's A0 and S15 rates",
    "implied_hit": "A0 + f * (S15 - A0), small hit75, pp",
    "prediction_group_values": "U5, native groups: prediction 3 = group_concave predicted_lead_U of the group; prediction 4 = f_law "
                               "of the group * the observed lead of the group, with law_gain_U_pp of the group beside it",
    "equivalent_budget": "u1_ref.equivalent_budget(gain, gain_u, gain_h) in pp: gain = FG-s0 - L0-s0, gain_u = U896-s0 - A0-s0, "
                         "gain_h = S15-s0 - A0-s0",
    "U5_groups": "phase-1 E0 groupings: size bin (th2_core.size_bin, all GT); on small GT only: class, occlusion (min(occ, 2)), "
                 "crowd (crowd_bin), F_geo magnification (g_bin of g0), native size",
    "splice": "development image order; image i in R takes the 600 stored records of S15-s0, otherwise those of A0-s0 (th2_c0."
              "by_image + splice); the list is rebuilt from the stored bytes for every scoring",
    "scoring": "development rows and parsed labels, score_budget(rows, parsed, predictions, source_root), infer_seed.bridge_values; "
               "U896 lists pass infer_seed._group (exact) and _prefix first, as in the S15 evaluation tool's scoring stage",
    "input_alignment": "row i of the U896 ledger against row i of the native 768 trajectory: update, epoch, window, image_ids_sha256, "
                       "augmentation_seeds_sha256, augmentation_strong must be equal; the canvas-dependent fields are counted",
    "A0_supervised_targets": "the stored 768 trajectory has no GT count; the native 768 loader is replayed on CPU over the 18,240 "
                             "windows, every receipt field must equal the stored row, and the targets are counted",
    "start_condition": "free GPU memory at launch >= peak reserved of the acceptance * 1.25 + 1.0e9 bytes",
}
BOUNDARIES = ["development is a development set; the U896 readings are not a final evaluation of any method",
              "one seed: the interval of f holds image resampling only, not training randomness; two trainings of the same arm differ "
              "by about 0.22-0.26 pp (resampling s.d.) in small hit75",
              "A0, U896 and S15 are the direct-resize family; L0 and F_geo are the through-the-H-canvas family; differences across "
              "the families are recorded only",
              "all three end points continue the 768 e060 state for 60 epochs; training from scratch on the canvas was not done",
              "the budget counts detector input pixels; the 896 image is interpolated from the original, more pixels are not more information",
              "no GO / STOP: stage1_reading is an exploratory reading on development and one seed and starts nothing"]
NOT_AUTHORISED = ["seed 1, seed 2; any canvas other than 896 x 1568; a second formal training",
                  "evaluating online weights, scoring during training, choosing a checkpoint by development, scoring e075 / e090 / e105",
                  "any change of the recipe; 4 x 4 instead of 8 x 2",
                  "test-dev; images or labels of the 1,602 held-out official-train images",
                  "readings outside task section 6; changing a definition after seeing a result",
                  "installing or upgrading packages; writing under experiments/localization-distribution or the phase-1, B0, C0 outputs",
                  "merging pull requests; changing the state of PR #43-#46; touching other people's processes"]
CPU_STAGES = ("t0-wire", "t0-ends", "t0-predict", "a0-count", "cpu", "eval-check", "align", "score", "routed", "readings")
CUDA_STAGES = ("accept", "train", "infer")
DEV_STAGES = ("t0-wire", "cpu", "eval-check")
ATTEMPT_KINDS = {"cpu": "cpu-gate", "accept": "acceptance", "train": "formal"}
PRODUCTS = ("PREDICTION.json", "RESULT.json")
A0_UNITS = tuple(str(k) for k in range(6))


def is_dev(r):
    return Path(r).resolve().name.startswith("dev-round")


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


# ================================================================ limits, scoring ledger
def bytes_under(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file() and not p.is_symlink())


def limits_guard(home):
    """Refuses to go on without the storage estimate or at / above the storage cap (task section 2)."""
    home = Path(home)
    path = home / "STORAGE_ESTIMATE.json"
    B.require(path.is_file(), f"no STORAGE_ESTIMATE.json in {home}: nothing may run")
    est = B.read_json(path)
    B.require(est["cap_bytes"] == CAP_BYTES and 0 < est["estimate_bytes"] <= CAP_BYTES, "storage estimate above the cap")
    used = bytes_under(home)
    B.require(used <= CAP_BYTES, f"storage cap reached: {used} bytes in {home}")
    return {"bytes_used": used, "estimate": B.ref(path), "estimate_bytes": est["estimate_bytes"]}


def scorings(home):
    path = Path(home) / "SCORINGS.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def spend(c, slot, counts):
    """One of the five scorings of the task.  The slot is entered in the ledger before the scorer is called, so a
    scoring that fails half-way is still counted and is never repeated."""
    B.require(not c["dev"], "a dev round never scores")
    used = [r["slot"] for r in scorings(c["home"])]
    B.require(slot in SLOTS and slot not in used and len(used) < len(SLOTS), f"scoring {slot}: already used or not registered ({used})")
    with (Path(c["home"]) / "SCORINGS.jsonl").open("a") as f:
        f.write(json.dumps({"slot": slot, "at": now(), "pid": os.getpid(), "contract_sha256": c["contract_sha256"]}) + "\n")
        f.flush()
        os.fsync(f.fileno())
    counts.update(score_budget_calls=1)


# ================================================================ contract
def native_scale():
    return [math.sqrt((1344.0 / w) * (768.0 / h)) for w, h in NATIVE]


def text_against_reference():
    """The numbers written in the task text against u1_ref.py and c0_ref.py."""
    B.require(U.L_HW == (768, 1344) and U.U_HW == HW and U.H_HW == (1152, 2016) and HW[1] / HW[0] == 1.75
              and U.tokens(U.U_HW) == 28812 == 112 * 196 + 56 * 98 + 28 * 49 and U.tokens(U.L_HW) == 21168, "canvases of u1_ref.py")
    B.require(abs(U.B_U - 49.0 / 36.0) < 1e-15 and round(U.B_U, 4) == 1.3611 and U.B_H == 2.25 and abs(U.M_U - 7.0 / 6.0) < 1e-15
              and U.M_H == 1.5, "budgets of u1_ref.py")
    B.require(round(U.CHORD, 6) == 0.288889 and round(U.LOGLINEAR, 6) == 0.380182, "predictions 1 and 2 of u1_ref.py")
    B.require(U.M_IMAGES == M == R.M_IMAGES and U.K_GATE == 158 == R.K_GATE and U.LINE_HIT == 0.5
              and (U.BOOT, U.BOOT_SEED) == (R.BOOT, R.BOOT_SEED) == (T.BOOT, T.BOOT_SEED) == (1000, 20261122), "constants")
    B.require(round(R.budget(158 / M), 4) == 1.3604 and R.budget(158 / M) <= U.B_U, "the routed system spends more than U896")
    scale = native_scale()
    B.require(tuple(round(x, 4) for x in scale) == NATIVE_SCALE_4DP, f"scale_l of the native groups: {scale}")
    B.require(U.stage1_reading(0.5) == "ROUTING_AHEAD_ON_SEED0" and U.stage1_reading(0.0) == "UNIFORM_NOT_BEHIND_ON_SEED0"
              and U.stage1_reading(0.25) == "IN_BETWEEN", "stage1_reading lines")
    return {"B_U": U.B_U, "B_H": U.B_H, "M_U": U.M_U, "M_H": U.M_H, "CHORD": U.CHORD, "LOGLINEAR": U.LOGLINEAR,
            "routed_budget_k158": R.budget(158 / M), "scale_l": scale, "tokens": {"L": U.tokens(U.L_HW), "U": U.tokens(U.U_HW), "H": U.tokens(U.H_HW)}}


def registered_in_text(text):
    """Every registered number of this file must be written, digit for digit, in the task text."""
    values = [v for unit in REG_AP.values() for row in unit.values() for v in row] + list(REG_HIT.values())
    missing = [repr(v) for v in values if repr(v) not in text]
    B.require(not missing, f"registered values not found in the task text: {missing}")
    for name, (path, size, sha) in PINS.items():
        B.require(sha in text and f"{size:,}" in text, f"pin of {name} not found in the task text")
    return len(values)


def prepare(round_dir, release, home_arg):
    R_ = Path(round_dir).resolve()
    dev = is_dev(R_)
    home = Path(home_arg).resolve() if home_arg else R_
    B.require(dev == (home != R_) and not is_dev(home), "a dev round needs --home <formal round>; the formal round is its own home")
    lim = limits_guard(home)
    B.require(not (R_ / "CONTRACT.json").exists(), "CONTRACT.json exists")
    B.require((R_ / release / "source").resolve() == ROOT.resolve(), "prepare must run from <round>/<release>/source")
    B.require(home.parent == STUDY and "localization-distribution" not in str(R_) and "th2-phase1" not in str(R_)
              and "th2-b0" not in str(R_) and "th2-c0" not in str(R_), "outputs must not be under the old output directories")
    task = home / TASK[0]
    B.require(task.is_file() and B.file_sha(task) == TASK[1], f"task file missing or different: {task}")
    refs = {}
    for name, (rel, sha) in REFS.items():
        refs[name] = B.ref(ROOT / rel)
        B.require(refs[name]["sha256"] == sha, f"{rel} is not the controller's file")
    text = text_against_reference()
    registered = registered_in_text(task.read_text(encoding="utf-8"))
    man_ref = B.ref(R_ / release / "MANIFEST.json")
    manifest = B.read_json(man_ref["path"])
    B.require(manifest["dirty"] is False or dev, "a formal release must come from a clean committed tree")
    B.release_intact(ROOT, manifest["files"])
    base_ref = B.ref(BASE_MANIFEST[0])
    base = B.read_json(base_ref["path"])
    B.require(base_ref["sha256"] == BASE_MANIFEST[1] and base["commit"] == BASE_MANIFEST[2], "C0 release identity")
    new, old = manifest["files"], base["files"]
    delta = {"added": sorted(set(new) - set(old)), "removed": sorted(set(old) - set(new)),
             "changed": sorted(f for f in new if f in old and new[f] != old[f])}
    B.require(set(delta["added"]) == set(ALLOWED_ADDED) and not delta["removed"] and set(delta["changed"]) == set(ALLOWED_CHANGED),
              f"the release may differ from the C0 release only by the U1 files and the arm registry: {delta}")
    inputs = {}
    for name, (path, size, sha) in PINS.items():
        inputs[name] = B.ref(path)
        B.require((inputs[name]["bytes"], inputs[name]["sha256"]) == (size, sha), f"task 4.1: {name} differs")
    for unit, rows in REG_SOURCES.items():                       # the registered A0 values against their source files
        for budget, (path, sha) in rows.items():
            B.require(B.file_sha(path) == sha, f"registered source of {unit} top{budget}")
            got = B.read_json(path)["COCO_bridge"]["metrics"]
            B.require(tuple(got[m] for m in METRICS) == REG_AP[unit][budget], f"registered values of {unit} top{budget}")
    s15_ref = B.ref(S15_CONTRACT[0])
    B.require((s15_ref["sha256"], s15_ref["bytes"]) == S15_CONTRACT[1:], "S15 seed0 contract file")
    s15 = B.read_json(s15_ref["path"])
    same = lambda a, b: (a["sha256"], a["bytes"]) == (b["sha256"], b["bytes"]) and Path(a["path"]).resolve() == Path(b["path"]).resolve()
    B.require(same(s15["data_receipt"], inputs["data_receipt"]) and same(s15["evaluation_policy"], inputs["evaluation_protocol"])
              and same(s15["parent"], inputs["parent"]), "the S15 seed0 contract differs from task 4.1 (data receipt, evaluation protocol, parent)")
    native = s15["scientific_binding"]["native_computation"]
    B.require(len(native) == 14 and all(new.get(k) == v for k, v in native.items()),
              "a native computation file differs from the S15 seed0 contract")
    p1_ref = B.ref(P1 / "CONTRACT.json")
    B.require(p1_ref["sha256"] == P1_CONTRACT_SHA, "phase-1 contract file")
    p1 = B.read_json(p1_ref["path"])
    B.require(p1["development_ids"] == s15["development_ids"] and len(p1["development_ids"]) == M
              and p1["data"]["development"] == s15["data"]["development"], "development binding: phase 1 against the S15 seed0 contract")
    B.require(os.path.samefile(sys.executable, s15["python"]) and os.path.samefile(sys.executable, p1["python"]),
              f"python differs from the frozen environment: {sys.executable}")
    torch = B.torch_mod()
    import torchvision
    env = {"torch": torch.__version__.split("+")[0], "torchvision": torchvision.__version__.split("+")[0],
           "cuda_build": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
    B.require(env == s15["environment"], f"environment differs from the S15 seed0 contract: {env}")
    identity = {"task": TASK[1], "arm": ARM, "seed": SEED, "parent": inputs["parent"], "data": s15["data"],
                "evaluation_policy": inputs["evaluation_protocol"], "native_computation": native,
                "implementation": {f: new[f] for f in ALLOWED_CHANGED}, "workload": WORKLOAD["train"]}
    c = {"version": VERSION, "task": {"file": TASK[0], "sha256": TASK[1]}, "references": refs, "arm": ARM, "seed": SEED, "input_hw": list(HW),
         "source_root": str(ROOT), "source_commit": manifest["commit"], "source_branch": manifest["branch"],
         "source_dirty": manifest["dirty"], "source_manifest": man_ref, "base_manifest": base_ref, "source_delta": delta,
         "outputs": str(R_), "home": str(home), "dev": dev,
         "formal_identity": B.sha_bytes(B.canonical(identity)), "scientific_binding": identity,
         "parent": inputs["parent"], "inputs": inputs, "S15_seed0_contract": {"file": s15_ref, "contract_sha256": s15["contract_sha256"]},
         "phase1_contract": {"file": p1_ref, "contract_sha256": p1["contract_sha256"]},
         "data": s15["data"], "development_ids": p1["development_ids"], "evaluation_policy": inputs["evaluation_protocol"],
         "registered": {"top": {u: {b: dict(zip(METRICS, v)) for b, v in rows.items()} for u, rows in REG_AP.items()},
                        "small_hit75_pp": REG_HIT, "S15_supervised_targets": S15_GT_TOTAL, "tolerance": TOL,
                        "values_found_in_task_text": registered},
         "design": {"canvas_L": list(U.L_HW), "canvas_U": list(U.U_HW), "canvas_H": list(U.H_HW), "text_against_reference": text,
                    "native_groups": [list(g) for g in NATIVE], "bootstrap": [U.BOOT, U.BOOT_SEED], "metrics": list(METRICS)},
         "workload": WORKLOAD, "reused_evidence": {"A0_768_plain_vs_checkpoint_CUDA": inputs["equivalence"]},
         "adapter_A0_list": "none needed: the stored A0-s0 list has the record layout th2_phase1.stored_index reads",
         "environment": env, "python": sys.executable, "gpu_uuid": s15["gpu_uuid"],
         "resource_policy": {"storage_cap_bytes": CAP_BYTES, "no_progress_seconds": NO_PROGRESS_SECONDS,
                             "training_time": "estimate 9-12 h, no hard cap; a note in CURRENT at 20 h"},
         "storage_estimate": lim["estimate"], "definitions": DEFINITIONS, "boundaries": BOUNDARIES, "not_authorised": NOT_AUTHORISED,
         "dev_round_rule": "a dev round runs t0-wire, cpu and eval-check only; its states live outside the formal round"}
    c["contract_sha256"] = B.sha_bytes(B.canonical(c))
    B.write_new(R_ / "CONTRACT.json", c)
    print(json.dumps({"contract_sha256": c["contract_sha256"], "contract_file": B.ref(R_ / "CONTRACT.json"),
                      "commit": c["source_commit"], "formal_identity": c["formal_identity"], "delta": delta, "dev": dev}))


def load_contract(round_dir, *, check_source=True):
    R_ = Path(round_dir).resolve()
    c = B.read_json(R_ / "CONTRACT.json")
    body = {k: v for k, v in c.items() if k != "contract_sha256"}
    B.require(c["contract_sha256"] == B.sha_bytes(B.canonical(body)) and c["version"] == VERSION, "contract digest")
    B.require(Path(c["outputs"]).resolve() == R_ and c["dev"] == is_dev(R_), "contract bound to another round")
    B.require(c["task"]["sha256"] == TASK[1] and c["definitions"] == DEFINITIONS and c["workload"] == WORKLOAD
              and c["formal_identity"] == B.sha_bytes(B.canonical(c["scientific_binding"])), "contract differs from the release")
    if check_source:
        B.require(Path(c["source_root"]).resolve() == ROOT.resolve(), "not running from the bound release")
        manifest = B.read_json(B.pinned(c["source_manifest"]))
        B.require(manifest["commit"] == c["source_commit"], "MANIFEST commit differs from the contract")
        B.release_intact(ROOT, manifest["files"])
        for rel, sha in REFS.values():
            B.require(B.file_sha(ROOT / rel) == sha, f"{rel} changed")
    limits_guard(c["home"])
    return c


def siblings(c, base):
    """<base>, <base>-r2, <base>-r3 ... that exist, in order."""
    root = Path(c["outputs"])
    names = [base] + [f"{base}-r{k}" for k in range(2, 20)]
    return [n for n in names if (root / n).exists()]


def stage_closed(c, base):
    """The one closed directory of a stage (a failed first directory may have a rerun beside it)."""
    root = Path(c["outputs"])
    done = [n for n in siblings(c, base) if (root / n / "CLOSED.json").is_file()]
    B.require(len(done) == 1, f"stage not closed (or closed twice): {base} {done}")
    closed = B.read_json(root / done[0] / "CLOSED.json")
    B.require(closed.get("status") == "PASS" and closed.get("contract_sha256") == c["contract_sha256"], f"stage {base} is not PASS")
    closed["_name"] = done[0]
    return closed


# ================================================================ attempts (views for the reused S15 trainer)
def attempt_file(c, name):
    return Path(c["outputs"]) / f"{name}.ATTEMPT.json"


def new_attempt(c, name, kind, prior=None):
    """Written once, before the stage directory: the identity a state of this attempt is bound to."""
    body = {"version": VERSION, "kind": kind, "attempt": name, "round_contract_sha256": c["contract_sha256"],
            "formal_identity": c["formal_identity"], "arm": ARM, "seed": SEED, "train": str(Path(c["outputs"]) / name),
            "prior": prior, "created_at": now()}
    body["attempt_sha256"] = B.sha_bytes(B.canonical(body))
    B.write_new(attempt_file(c, name), body)
    return body


def view(c, name):
    """The flat mapping the S15 trainer functions read (_save, _load, _formal, _accept, read_endpoint)."""
    a = B.read_json(attempt_file(c, name))
    body = {k: v for k, v in a.items() if k != "attempt_sha256"}
    B.require(a["attempt_sha256"] == B.sha_bytes(B.canonical(body)) and a["round_contract_sha256"] == c["contract_sha256"]
              and a["formal_identity"] == c["formal_identity"] and (a["arm"], a["seed"], a["attempt"]) == (ARM, SEED, name)
              and a["train"] == str(Path(c["outputs"]) / name), f"attempt record differs: {name}")
    return {"seed": SEED, "formal_identity": c["formal_identity"], "contract_sha256": a["attempt_sha256"],
            "contract_path": str(attempt_file(c, name)), "parent": c["parent"], "data": c["data"],
            "development_ids": c["development_ids"], "workload": c["workload"]["train"], "reused_evidence": c["reused_evidence"],
            "prior": a["prior"], "outputs": {"train": a["train"]}, "source_root": c["source_root"], "attempt": name, "kind": a["kind"]}


def eval_view(c):
    return {"seed": SEED, "formal_identity": c["formal_identity"], "contract_sha256": c["contract_sha256"], "parent": c["parent"],
            "data": c["data"], "development_ids": c["development_ids"], "workload": c["workload"]["train"],
            "source_root": c["source_root"]}


def train_attempts(c):
    root = Path(c["outputs"])
    return [n for n in ["train-U896"] + [f"train-U896-r{k}" for k in range(1, RESTORES + 2)] if (root / n).exists() or attempt_file(c, n).exists()]


def final_train(c):
    names = train_attempts(c)
    B.require(names and (Path(c["outputs"]) / names[-1] / "CLOSED.json").is_file(), "the formal training is not closed")
    closed = B.read_json(Path(c["outputs"]) / names[-1] / "CLOSED.json")
    B.require(closed["status"] == "PASS" and closed["contract_sha256"] == c["contract_sha256"] and closed["global_update"] == END_UPDATE
              and closed["arm"] == ARM, "formal training closure")
    return names, closed


# ================================================================ pure pieces (tested on synthetic arrays)
class GTView:
    """The scored rows of gt.npz in the shape th2_phase1.outcomes_of reads (n, box, cat, sl)."""

    def __init__(self, z):
        sc = np.asarray(z["scored"])
        self.image = np.asarray(z["image"])[sc].astype(np.int64)
        self.cat, self.box = np.asarray(z["category"])[sc], np.asarray(z["xyxy"])[sc]
        self.area = np.asarray(z["area"])[sc].astype(np.float64)
        self.s = np.asarray(z["s"])[sc].astype(np.float64)
        self.occlusion, self.crowd, self.g0 = (np.asarray(z[k])[sc] for k in ("occlusion", "crowd", "g0"))
        self.n = int(sc.sum())
        self.bounds = np.searchsorted(self.image, np.arange(M + 1))
        self.size = T.size_bin(self.area)
        masks = C0.group_masks(self.area)
        self.small, self.medium, self.large = masks["small"], masks["medium"], masks["large"]

    def sl(self, i):
        return slice(self.bounds[i], self.bounds[i + 1])

    def masks(self):
        return {"small": self.small, "medium": self.medium, "large": self.large}


def native_group(width, height):
    """Index of the native-size group of every image; an unknown size is refused."""
    width, height = np.asarray(width), np.asarray(height)
    g = np.full(len(width), -1, dtype=np.int64)
    for k, (w, h) in enumerate(NATIVE):
        g[(width == w) & (height == h)] = k
    B.require(bool((g >= 0).all()), "a development image has a size outside the three native groups")
    return g


def native_label(k):
    return f"{NATIVE[k][0]}x{NATIVE[k][1]}"


def _factors(pred):
    return {"1": pred[1.0], "M_U": pred[float(U.M_U)], "M_H": pred[float(U.M_H)]}


def prediction_tables(image, small, s, cls, iou_a, iou_b, group):
    """Task 4.6.  image, small, s, cls, iou_a, iou_b: per scored GT (all sizes); group: native group per image."""
    hit_a, hit_b = iou_a >= 0.75, iou_b >= 0.75
    a, b, n = (R.per_image(x, image, M) for x in (hit_a & small, hit_b & small, small))
    rate = lambda x, m=slice(None): 100.0 * float(x[m].sum()) / float(n[m].sum())
    rate_a, rate_b = rate(a), rate(b)
    scale = native_scale()
    groups, lead, count = [], [], []
    for k in range(len(NATIVE)):
        m = group == k
        lead.append(100.0 * float((b[m] - a[m]).sum()) / float(n[m].sum()))
        count.append(float(n[m].sum()))
        groups.append({"native": native_label(k), "images": int(m.sum()), "small_gt": int(n[m].sum()), "scale_l": scale[k],
                       "A0_small_hit75_pp": rate(a, m), "S15_small_hit75_pp": rate(b, m), "lead_pp": lead[-1]})
    concave = U.group_concave(lead, count, scale)
    hit = hit_a.astype(np.float64)
    pred = U.shift_prediction(s, cls, hit, small)
    summary = U.shift_summary(pred, rate_a, rate_b)
    shift_groups = []
    for k in range(len(NATIVE)):
        target = small & (group[image] == k)
        pg = U.shift_prediction(s, cls, hit, target)
        shift_groups.append({"native": native_label(k), "predicted_rate_pp": _factors(pg),
                             "summary": U.shift_summary(pg, groups[k]["A0_small_hit75_pp"], groups[k]["S15_small_hit75_pp"])})
    f = {"1_linear": U.CHORD, "2_loglinear": U.LOGLINEAR, "3_group_concave": concave["f"], "4_size_shift": summary["f_law"]}
    lead_all = rate_b - rate_a
    return C0.clean({
        "endpoints": {"A0_small_hit75_pp": rate_a, "S15_small_hit75_pp": rate_b, "lead_pp": lead_all, "small_gt": int(n.sum()),
                      "A0_small_hits": int(a.sum()), "S15_small_hits": int(b.sum())},
        "predictions": {k: {"f": v, "implied_U896_small_hit75_pp": None if v is None else rate_a + v * lead_all} for k, v in f.items()},
        "prediction_3": {"groups": groups, "group_concave": concave},
        "prediction_4": {"factors": {"1": 1.0, "M_U": U.M_U, "M_H": U.M_H}, "predicted_rate_pp": _factors(pred), "summary": summary,
                         "groups": shift_groups}})


def _rate(hit, sel):
    return 100.0 * float(hit[sel].mean())


def _cell(sel, hit):
    n = int(sel.sum())
    if n == 0:
        return {"n": 0, "A0_hit75_pp": None, "U896_minus_A0_pp": None, "S15_minus_A0_pp": None, "f": None, "FG_minus_L0_pp": None}
    ha, hu, hb, hl, hf = (int(hit[k][sel].sum()) for k in ("a", "u", "b", "l", "fg"))
    return {"n": n, "A0_hit75_pp": 100.0 * ha / n, "U896_minus_A0_pp": 100.0 * (hu - ha) / n, "S15_minus_A0_pp": 100.0 * (hb - ha) / n,
            "f": None if hb == ha else (hu - ha) / (hb - ha), "FG_minus_L0_pp": 100.0 * (hf - hl) / n}


def _ap_fraction(u, a, b):
    return {m: (None if b[m] == a[m] else U.fraction(u[m], a[m], b[m])) for m in ("APS", "AP75_small", "AP", "APM")}


def reading_tables(inp, ap, prediction):
    """Task section 6, U1-U5.  inp: image, area, s, cat, occlusion, crowd, g0 (per scored GT), iou {a, u, b, l, fg},
    routed (548 bool), group (native group per image).  ap: {'U896': {'600', '300'}, 'A0': {...}, 'S15': {...}, 'L0': {'600'},
    'FG': {'600'}, 'routed_A0_S15': {'600'}, 'routed_L0_S15': {'600'}} of metric dicts (0..1); 'routed_L0_S15_small_hit75_pp'.
    prediction: the content of PREDICTION.json."""
    image, iou, routed, group = inp["image"], inp["iou"], np.asarray(inp["routed"]), np.asarray(inp["group"])
    B.require(routed.dtype == np.bool_ and routed.shape == (M,) and int(routed.sum()) == U.K_GATE, "routed set")
    masks = C0.group_masks(inp["area"])
    small = masks["small"]
    names = {"a": "A0-s0", "u": "U896-s0", "b": "S15-s0", "l": "L0-s0", "fg": "FG-s0"}
    w = R.boot_weights(M)
    hit = {t: {k: iou[k] >= t for k in names} for t in (0.75, 0.5)}

    def img(k, g="small", t=0.75):
        return R.per_image(hit[t][k] & masks[g], image, M)

    n = {g: R.per_image(masks[g], image, M) for g in masks}
    pp = lambda v: 100.0 * v
    # ---------------- U1
    u1 = {"AP_level": {"U896-s0": ap["U896"], "A0-s0": ap["A0"], "S15-s0": ap["S15"], "L0-s0": ap["L0"], "FG-s0": ap["FG"]},
          "hit_level_pp": {names[k]: {f"{g}_{label}": _rate(hit[t][k], masks[g]) for g in masks for t, label in ((0.75, "hit75"), (0.5, "hit50"))}
                           for k in names}}
    # ---------------- U2
    a, u, b = img("a"), img("u"), img("b")
    tot = float(n["small"].sum())
    rate = {k: 100.0 * float(img(k).sum()) / tot for k in names}
    f = U.fraction(u, a, b)
    ci = U.fraction_bootstrap(u, a, b, w)
    preds = {k: v["f"] for k, v in prediction["predictions"].items()}
    u2 = {"small_hit75": {"f": f, "ci95": ci, "U896_minus_A0_pp": rate["u"] - rate["a"], "S15_minus_A0_pp": rate["b"] - rate["a"],
                          "A0_pp": rate["a"], "U896_pp": rate["u"], "S15_pp": rate["b"],
                          "predictions": {k: {"f": v, "inside_ci95": None if v is None else bool(ci[0] <= v <= ci[1]),
                                              "measured_minus_predicted": None if v is None else f - v} for k, v in preds.items()}},
          "AP_level": {}}
    for label, g, t in (("medium_hit75", "medium", 0.75), ("small_hit50", "small", 0.5)):
        xa, xu, xb = img("a", g, t), img("u", g, t), img("b", g, t)
        den = float(n[g].sum())
        u2[label] = {"f": None if xb.sum() == xa.sum() else U.fraction(xu, xa, xb),
                     "ci95": None if xb.sum() == xa.sum() else U.fraction_bootstrap(xu, xa, xb, w),
                     "A0_pp": pp(xa.sum() / den), "U896_pp": pp(xu.sum() / den), "S15_pp": pp(xb.sum() / den),
                     "U896_minus_A0_pp": pp((xu.sum() - xa.sum()) / den), "S15_minus_A0_pp": pp((xb.sum() - xa.sum()) / den)}
    for budget in ("600", "300"):
        va, vu, vb = ap["A0"][budget], ap["U896"][budget], ap["S15"][budget]
        u2["AP_level"][f"top{budget}"] = {"f": _ap_fraction(vu, va, vb),
                                          "APL_values": {"A0": va["APL"], "U896": vu["APL"], "S15": vb["APL"]},
                                          "U896_minus_A0_pp": {m: pp(vu[m] - va[m]) for m in METRICS},
                                          "S15_minus_A0_pp": {m: pp(vb[m] - va[m]) for m in METRICS}}
    # ---------------- U3
    delta = U.delta_hit(a, b, u, n["small"], routed)
    u6 = ap["U896"]["600"]
    u3 = {"small_hit75": {"routed_A0_S15_pp": 100.0 * float(U.routed_hits(a, b, routed).sum()) / tot, "U896_pp": rate["u"],
                          "routed_minus_U896_pp": delta, "ci95": U.delta_hit_bootstrap(a, b, u, n["small"], routed, w),
                          "stage1_reading": U.stage1_reading(delta), "routed_images": int(routed.sum()),
                          "routed_budget": R.budget(int(routed.sum()) / M), "U896_budget": U.B_U},
          "AP_level_top600": {"routed_A0_S15": {m: ap["routed_A0_S15"]["600"][m] for m in METRICS},
                              "routed_minus_U896_pp": {m: pp(ap["routed_A0_S15"]["600"][m] - u6[m]) for m in METRICS}},
          "cheap_side_L0_record_only": {
              "small_hit75": {"routed_L0_S15_pp": ap["routed_L0_S15_small_hit75_pp"], "routed_minus_U896_pp": ap["routed_L0_S15_small_hit75_pp"] - rate["u"]},
              "AP_level_top600": {"routed_L0_S15": {m: ap["routed_L0_S15"]["600"][m] for m in METRICS},
                                  "routed_minus_U896_pp": {m: pp(ap["routed_L0_S15"]["600"][m] - u6[m]) for m in METRICS}}}}
    # ---------------- U4
    fg, lo = img("fg"), img("l")
    none = np.zeros(M, dtype=bool)
    gains = {"small_hit75": (rate["fg"] - rate["l"], rate["u"] - rate["a"], rate["b"] - rate["a"])}
    for m in ("APS", "AP75_small", "AP"):
        gains[m] = (pp(ap["FG"]["600"][m] - ap["L0"]["600"][m]), pp(u6[m] - ap["A0"]["600"][m]), pp(ap["S15"]["600"][m] - ap["A0"]["600"][m]))
    u4 = {"equivalent_budget": {k: {"gain_pp": g, "gain_u_pp": gu, "gain_h_pp": gh, **U.equivalent_budget(g, gu, gh)} for k, (g, gu, gh) in gains.items()},
          "FG_minus_U896": {"small_hit75_pp": U.delta_hit(fg, fg, u, n["small"], none),
                            "small_hit75_ci95": U.delta_hit_bootstrap(fg, fg, u, n["small"], none, w),
                            "AP_level_top600_pp": {m: pp(ap["FG"]["600"][m] - u6[m]) for m in METRICS}}}
    # ---------------- U5
    h75 = hit[0.75]
    size = T.size_bin(inp["area"])
    gt_group = group[image]
    codes = {"class": (np.asarray(inp["cat"]) - 1, tuple(str(k) for k in range(1, 11))),
             "occlusion": (np.minimum(np.asarray(inp["occlusion"]), 2), ("0", "1", "2")),
             "crowd": (T.crowd_bin(inp["crowd"]), T.CROWD_NAMES), "F_geo_magnification": (T.g_bin(inp["g0"]), T.G_NAMES),
             "native_size": (gt_group, tuple(native_label(k) for k in range(len(NATIVE))))}
    u5 = {"size_bin_all_GT": [{"group": name, **_cell(size == k, h75)} for k, name in enumerate(T.SIZE_NAMES)]}
    for key, (code, labels) in codes.items():
        u5[f"{key}_small_GT"] = [{"group": labels[k], **_cell(small & (code == k), h75)} for k in range(len(labels))]
    p3, p4 = prediction["prediction_3"], prediction["prediction_4"]
    rows = []
    for k in range(len(NATIVE)):
        cell = _cell(small & (gt_group == k), h75)
        s4 = p4["groups"][k]["summary"]
        rows.append({"group": native_label(k), "n": cell["n"], "measured_U896_minus_A0_pp": cell["U896_minus_A0_pp"],
                     "S15_minus_A0_pp": cell["S15_minus_A0_pp"], "measured_f": cell["f"],
                     "prediction_3_lead_U_pp": p3["group_concave"]["predicted_lead_U"][k],
                     "prediction_4_f_law": s4["f_law"],
                     "prediction_4_lead_U_pp": None if s4["f_law"] is None else s4["f_law"] * s4["observed_lead_H_pp"],
                     "prediction_4_law_gain_U_pp": s4["law_gain_U_pp"]})
    u5["native_size_against_predictions"] = rows
    columns = {"n_small": n["small"], **{f"hit75_small_{names[k].split('-')[0]}": img(k) for k in names}, "P_k158": routed.astype(np.float64)}
    return C0.clean({"U1": u1, "U2": u2, "U3": u3, "U4": u4, "U5": u5}), columns


def align_rows(mine, native):
    """Task 4.4 item 6 over two ledgers (iterables of JSON lines)."""
    same = ("update", "epoch", "window", "image_ids_sha256", "augmentation_seeds_sha256", "augmentation_strong")
    differ = ("gt_ids_sha256", "augmentation_trace_sha256", "augmented_samples_sha256", "input_tensor_sha256", "target_sha256")
    counts = {k: 0 for k in differ}
    clocks = {"lr": 0, "ema_updates": 0}
    digest, gt, rows = hashlib.sha256(), 0, 0
    native = iter(native)
    for line in mine:
        row = json.loads(line)
        ref = json.loads(next(native, "null"))
        B.require(ref is not None, "the U896 ledger is longer than the native trajectory")
        rows += 1
        a, r = row["input"], ref["input"]
        B.require(a["update"] == PARENT_UPDATE + rows and all(a[k] == r[k] for k in same),
                  f"order / draws differ from the native 768 trajectory at update {PARENT_UPDATE + rows}")
        for k in differ:
            counts[k] += a[k] != r[k]
        for k in clocks:
            clocks[k] += row[k] == ref[k]
        gt += row["gt_count"]
        digest.update(bytes.fromhex(B.sha_bytes(B.canonical(a))))
    B.require(next(native, None) is None, "the native trajectory is longer than the U896 ledger")
    return {"updates": rows, "equal_fields": list(same), "rows_with_canvas_dependent_difference": counts,
            "rows_with_equal_clock": clocks, "U896_positive_GT_total_after_sanitize": gt, "input_sha256": digest.hexdigest()}


# ================================================================ shared loaders
def development(c, hw):
    from sparse_rtdetr.research import scale_baseline as S
    data = S.development(c["data"]["development"], hw)
    B.require([r["stable_image_id"] for r in data.records] == c["development_ids"] and len(data.records) == M, "development members/order")
    return data


def load_arrays(c):
    z = np.load(B.pinned(c["inputs"]["gt"]), allow_pickle=False)
    gt = GTView({k: z[k] for k in z.files})
    o = np.load(B.pinned(c["inputs"]["outcomes"]), allow_pickle=False)
    iou = {k: o[f"iou_{u}"] for k, u in (("b", "S15-s0"), ("l", "L0-s0"), ("fg", "FG-s0"))}
    routed = np.load(B.pinned(c["inputs"]["assignments"]), allow_pickle=False)["P-k158"]
    B.require(gt.n == N_GT and bool((np.diff(gt.image) >= 0).all()) and gt.image[0] >= 0 and gt.image[-1] < M
              and all(v.shape == (N_GT,) for v in iou.values()), "per-GT arrays")
    return gt, iou, routed


def list_outcomes(c, gt, path):
    import th2_phase1 as P
    return P.outcomes_of(gt, P.stored_index(path), c["development_ids"])[0]


# ================================================================ T0
def t0_wire(c, out, counts, args):
    B.PHASE["phase"] = "t0-wire/files"
    files = {k: B.ref(B.pinned(v)) for k, v in c["inputs"].items()}
    for name, (_, size, sha) in PINS.items():
        B.require((files[name]["bytes"], files[name]["sha256"]) == (size, sha), f"4.7 item 1: {name}")
    for rel, sha in REFS.values():
        B.require(B.file_sha(ROOT / rel) == sha, f"4.7 item 1: {rel}")
    B.PHASE["phase"] = "t0-wire/selftest"
    U.selftest()
    R.selftest()
    text = text_against_reference()
    B.require(bool(np.array_equal(R.boot_weights(M), T.boot_weights(M))), "c0_ref.boot_weights differs from the phase-1 bootstrap")
    B.PHASE["phase"] = "t0-wire/arrays"
    gt, iou, routed = load_arrays(c)
    got = {g: int(m.sum()) for g, m in gt.masks().items()}
    B.require(got == GROUP_COUNTS, f"group counts: {got}")
    n = R.per_image(gt.small, gt.image, M)
    img = {k: R.per_image((v >= 0.75) & gt.small, gt.image, M) for k, v in iou.items()}
    hit = {u: float(100 * (iou[k][gt.small] >= 0.75).mean()) for k, u in (("l", "L0-s0"), ("b", "S15-s0"), ("fg", "FG-s0"))}
    for u, v in hit.items():
        B.require(abs(v - REG_HIT[u]) <= TOL, f"4.7 item 2: small hit75 of {u}: {v}")
    B.PHASE["phase"] = "t0-wire/item3"
    again = list_outcomes(c, gt, B.pinned(c["inputs"]["S15-s0"]))
    B.require(again.shape == iou["b"].shape and bool(np.array_equal(again, iou["b"])), "4.7 item 3: iou of S15-s0 differs from outcomes.npz")
    B.PHASE["phase"] = "t0-wire/item5"
    B.require(routed.dtype == np.bool_ and routed.shape == (M,) and int(routed.sum()) == U.K_GATE, "4.7 item 5: P-k158")
    routed_hit = 100.0 * float(U.routed_hits(img["l"], img["b"], routed).sum()) / float(n.sum())
    B.require(abs(routed_hit - REG_HIT["routed_P-k158_L0_S15"]) <= TOL, f"4.7 item 5: routed small hit75 {routed_hit}")
    B.PHASE["phase"] = "t0-wire/item6"
    rows = development(c, tuple(U.H_HW)).records
    group = native_group([r["width"] for r in rows], [r["height"] for r in rows])
    sizes = tuple((int((group == k).sum()), int(n[group == k].sum())) for k in range(len(NATIVE)))
    B.require(sizes == NATIVE_COUNTS, f"4.7 item 6: native groups {sizes}")
    B.require(counts["forwards"] == 0, "t0-wire must not run a model")
    return {"stage": "t0-wire", "passed": True,
            "checks": {"item1_files": files, "item1_selftests": {"u1_ref": "PASS", "c0_ref": "PASS"}, "item2_small_hit75_pp": hit,
                       "item3_iou_S15_s0_equal_to_outcomes": True, "item5_routed_L0_S15_small_hit75_pp": routed_hit,
                       "item6_native_groups": [{"native": native_label(k), "images": sizes[k][0], "small_gt": sizes[k][1]} for k in range(len(NATIVE))],
                       "group_counts": got, "text_against_reference": text, "bootstrap_weights_equal_to_phase1": True, "tolerance": TOL}}


class Scorer:
    """The C0 Scorer path (th2_c0.Scorer) with the cheap side A0-s0: splice of the stored records, score_budget, bridge_values."""

    def __init__(self, c, counts):
        from infer_seed import bridge_values
        from sparse_rtdetr.r0.evaluation import score_budget
        self.c, self.counts = c, counts
        self.score_budget, self.bridge_values = score_budget, bridge_values
        self.ids = c["development_ids"]
        self.refs = {"A0": c["inputs"]["A0-s0"], "S15": c["inputs"]["S15-s0"]}
        self.text = {arm: B.pinned(self.refs[arm]).read_bytes() for arm in self.refs}
        data = development(c, tuple(U.H_HW))
        self.rows = data.records
        self.parsed = [data.raw(r) for r in self.rows]

    def score(self, out, slot, name, route):
        route = np.asarray(route)
        B.require(route.shape == (M,) and route.dtype == np.bool_, "assignment array")
        lists = {arm: json.loads(self.text[arm]) for arm in self.refs}
        preds = C0.splice(self.ids, route, C0.by_image(lists["A0"], self.ids), C0.by_image(lists["S15"], self.ids))
        B.require(len(preds) == M * 600, "spliced record count")
        same = {arm: preds == lists[arm] for arm in self.refs}
        del lists
        spend(self.c, slot, self.counts)
        metrics = self.score_budget(self.rows, self.parsed, preds, self.c["source_root"])
        B.require({k: metrics["spatial_filter_counts"][k] for k in GT_COUNTS} == GT_COUNTS, "scored GT set")
        metrics["TH2U1_binding"] = {"contract_sha256": self.c["contract_sha256"], "slot": slot, "row": name, "routed_images": int(route.sum()),
                                    "assignment_sha256": B.sha_bytes(route.astype(np.uint8).tobytes()), "predictions": self.refs,
                                    "spliced_list_equals_stored_list": same}
        path = out / f"{name}.top600-metrics.json"
        B.write_new(path, metrics)
        return self.bridge_values(metrics), B.ref(path), same


def t0_ends(c, out, counts, args):
    B.PHASE["phase"] = "t0-ends"
    B.require(stage_closed(c, "t0-wire")["passed"] is True, "wiring check")
    sc = Scorer(c, counts)
    values, files, diff, same = {}, {}, {}, {}
    for arm, route in (("A0", np.zeros(M, dtype=bool)), ("S15", np.ones(M, dtype=bool))):
        B.PHASE["phase"] = f"t0-ends/{arm}"
        v, ref, eq = sc.score(out, f"t0-ends/{arm}", arm, route)
        reg = c["registered"]["top"][f"{arm}-s0"]["600"]
        values[arm], files[arm], same[arm] = v, ref, eq[arm]
        diff[arm] = max(abs(v[m] - reg[m]) for m in METRICS)
        B.require(eq[arm] is True, f"4.7 item 4: the spliced list of {arm} differs from the stored file")
        B.require(diff[arm] <= TOL, f"4.7 item 4: {arm}-s0 scored {v}, registered {reg}")
    B.require(counts["forwards"] == 0 and counts["score_budget_calls"] == 2, "t0-ends: counts")
    return {"stage": "t0-ends", "passed": True, "values_600": values, "files": files, "max_abs_difference_from_registered": diff,
            "spliced_list_equals_stored_file": same, "GT_counts": GT_COUNTS,
            "two_scoring_paths_on_S15_s0": "the registered S15-s0 values are those of the S15 evaluation tool (COMPARISON.json); "
                                           "this scoring is the C0 Scorer path with R = all"}


def t0_predict(c, out, counts, args):
    B.PHASE["phase"] = "t0-predict"
    B.require(not c["dev"], "a dev round registers no prediction")
    B.require(stage_closed(c, "t0-wire")["passed"] is True and stage_closed(c, "t0-ends")["passed"] is True, "4.7 items 1-6 must pass first")
    B.require(not list(Path(c["outputs"]).glob("*/PREDICTION.json")), "PREDICTION.json exists")
    gt, iou, _ = load_arrays(c)
    a = list_outcomes(c, gt, B.pinned(c["inputs"]["A0-s0"]))
    B.require(a.shape == (N_GT,) and bool(np.isfinite(a).all()) and float(a.min()) >= 0.0 and float(a.max()) <= 1.0, "iou of A0-s0")
    rows = development(c, tuple(U.H_HW)).records
    group = native_group([r["width"] for r in rows], [r["height"] for r in rows])
    tables = prediction_tables(gt.image, gt.small, gt.s, gt.cat, a, iou["b"], group)
    B.require(abs(tables["endpoints"]["S15_small_hit75_pp"] - REG_HIT["S15-s0"]) <= TOL and tables["endpoints"]["small_gt"] == GROUP_COUNTS["small"],
              "t0-predict: S15 end point")
    with (out / "per_gt_A0.npz").open("xb") as f:
        np.savez(f, **{"iou_A0-s0": a})
    per_gt = B.ref(out / "per_gt_A0.npz")
    doc = {"version": VERSION, "task": c["task"], "contract_sha256": c["contract_sha256"], "source_commit": c["source_commit"],
           "written_at": now(), "written_before_any_training": True,
           "quantity": "f = (U896 - A0) / (S15 - A0), small hit75, seed 0", **tables,
           "controller_reference": {"prediction_3_on_the_L0_family_seed0": 0.4771,
                                    "note": "task 4.6: a reference only; the registered value is the one above, on the A0 family"},
           "inputs": {k: c["inputs"][k] for k in ("A0-s0", "outcomes", "gt")}, "per_gt_A0": per_gt, "definitions": {
               k: c["definitions"][k] for k in ("f", "prediction_1", "prediction_2", "prediction_3", "prediction_4", "implied_hit", "native_groups")}}
    B.write_new(out / "PREDICTION.json", doc)
    B.require(counts["forwards"] == 0, "t0-predict must not run a model")
    return {"stage": "t0-predict", "prediction": B.ref(out / "PREDICTION.json"), "per_gt_A0": per_gt,
            "f": {k: v["f"] for k, v in tables["predictions"].items()}, "endpoints": tables["endpoints"]}


# ================================================================ A0 supervised targets (native 768 replay, CPU)
def a0_count(c, out, counts, args):
    import formal_data
    torch = B.torch_mod()
    B.require(not c["dev"] and args.unit in A0_UNITS, "a0-count: unit 0..5, formal round")
    B.require(formal_data.SEED == SEED, "native loader seed")
    torch.set_num_threads(1)
    lines = B.pinned(c["inputs"]["a0_trajectory"]).read_text().splitlines()
    B.require(len(lines) == END_UPDATE - PARENT_UPDATE, "native trajectory length")
    data = formal_data.train_dataset(c["data"]["train_core"])
    first = 61 + 10 * int(args.unit)
    epochs = {}
    for epoch in range(first, first + 10):
        B.PHASE["phase"] = f"a0-count/e{epoch}"
        done = (epoch - 1) * PER_EPOCH
        gt = seen = 0
        for _, targets, receipt in formal_data.windows(data, done):
            ref = json.loads(lines[done + seen - PARENT_UPDATE])["input"]
            B.require(receipt == ref, f"the native 768 replay differs from the stored A0 trajectory at update {done + seen + 1}")
            gt += int(sum(len(t["labels"]) for t in targets))
            seen += 1
        B.require(seen == PER_EPOCH, f"epoch {epoch}: {seen} windows")
        epochs[str(epoch)] = gt
        print(json.dumps({"epoch": epoch, "positive_gt": gt}), flush=True)
    B.require(not torch.cuda.is_initialized() and counts["forwards"] == 0, "a0-count must not run a model")
    return {"stage": "a0-count", "unit": args.unit, "epochs": epochs, "windows": 10 * PER_EPOCH, "positive_gt": sum(epochs.values()),
            "all_receipt_fields_equal_to_the_stored_trajectory": True}


def a0_total(c):
    """Sum of the six units, or the reason why the number is not available."""
    root, total, epochs = Path(c["outputs"]), 0, 0
    for k in A0_UNITS:
        try:
            closed = stage_closed(c, f"a0-count-{k}")
        except ValueError as exc:
            return {"available": False, "reason": str(exc)[:300]}
        total += closed["positive_gt"]
        epochs += len(closed["epochs"])
    B.require(epochs == 60 and root.is_dir(), "a0-count units")
    return {"available": True, "A0_positive_GT_total_after_sanitize": total,
            "source": "native 768 loader replayed on CPU over the 18,240 windows; all 11 receipt fields of every window equal the "
                      "stored A0 trajectory (the stored trajectory itself holds no GT count)"}


# ================================================================ CPU gate of the trainer
def trainer(c=None):
    """The accepted S15 trainer bound to this arm and seed (process-wide, once)."""
    import s15_formal_train as Tr
    from sparse_rtdetr.research import scale_baseline as S
    Tr.bind_arm(ARM)
    Tr.bind_seed(SEED)
    B.require(Tr.ARM == ARM and S.arm_hw(ARM) == HW and S.ARM_SEEDS[ARM] == (SEED,) and Tr.SAVE_EPOCHS == SAVE_EPOCHS, "arm registry")
    return Tr, S


def _refused(label, call, excs, needle=None):
    try:
        call()
    except excs as exc:
        B.require(needle is None or needle in str(exc), f"{label}: refused for another reason: {exc!r}")
        return {"case": label, "reason": repr(exc)[:300]}
    raise AssertionError("accepted: " + label)


def _loader_check(c, Tr, v):
    import formal_data
    torch = B.torch_mod()
    B.require(not torch.cuda.is_initialized(), "loader check before CUDA")
    lines = B.pinned(c["inputs"]["a0_trajectory"]).read_text().splitlines()
    native = formal_data.train_dataset(c["data"]["train_core"])
    scaled = Tr._data(v)
    same = ("update", "epoch", "window", "image_ids_sha256", "augmentation_seeds_sha256", "augmentation_strong")
    differs = ("gt_ids_sha256", "augmentation_trace_sha256", "augmented_samples_sha256", "input_tensor_sha256", "target_sha256")
    rows = []
    for completed in Tr.LOADER_CHECK_UPDATES:
        (_, t768, a768), = list(formal_data.windows(native, completed, limit=1))
        (x, t896, a896), = list(formal_data.windows(scaled, completed, limit=1))
        stored = json.loads(lines[completed - PARENT_UPDATE])["input"]
        exact = {k: a768[k] == stored.get(k) for k in a768}
        B.require(set(a768) == set(stored) == set(same) | set(differs) and all(exact.values()),
                  f"the native 768 loader differs from the stored A0 trajectory at update {completed + 1}")
        B.require(all(a896[k] == a768[k] for k in same), "U896 order / draws differ from the native trajectory")
        B.require(tuple(x.shape) == (16, 3, *HW), "U896 window shape")
        rows.append({"update": completed + 1, "native_768_equal_stored_trajectory": exact,
                     "U896_order_draw_equal_768": {k: a896[k] == a768[k] for k in same},
                     "U896_canvas_dependent_differs": {k: a896[k] != a768[k] for k in differs},
                     "positive_gt_768": int(sum(len(t["labels"]) for t in t768)), "positive_gt_896": int(sum(len(t["labels"]) for t in t896)),
                     "receipt_768": a768, "receipt_896": a896})
    B.require(not torch.cuda.is_initialized(), "loader check initialized CUDA")
    return rows


def cpu(c, out, counts, args):
    import formal_data
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    from sparse_rtdetr.protocol_r18.model import Config
    torch = B.torch_mod()
    Tr, S = trainer()
    v = view(c, args.name)
    wl = v["workload"]
    B.require(not torch.cuda.is_initialized(), "CPU gate initialized CUDA")
    t0, c0 = time.perf_counter(), time.process_time()
    cost = lambda a, b: {"wall_seconds": time.perf_counter() - a, "cpu_seconds": time.process_time() - b}
    # ---- contract projection
    B.PHASE["phase"] = "cpu/projection"
    mine, native, s15 = S.production_config(SEED, ARM).binding(), Config(seed=SEED).binding(), S.production_config(SEED).binding()
    vs_native, vs_s15 = sorted(k for k in mine if mine[k] != native[k]), sorted(k for k in mine if mine[k] != s15[k])
    B.require(vs_native == ["input_hw"] == vs_s15 and tuple(mine["input_hw"]) == HW and tuple(s15["input_hw"]) == (1152, 2016),
              f"U896 recipe differs beyond the canvas: {vs_native} {vs_s15}")
    B.require(wl["checkpoint_segments"] == list(S.CHECKPOINT_SEGMENTS) and wl["input_hw"] == list(S.arm_hw(ARM))
              and (mine["physical_batch_size"], mine["amp_dtype"]) == (wl["train_batch"], wl["precision"]), "workload against the recipe")
    B.require(S.binding(S.production_config(SEED, ARM), activation_checkpoint=True, arm=ARM)["arm"] == ARM
              and S.binding(S.production_config(SEED))["arm"] == "S15" and S.production_config(SEED) == Config(seed=SEED, input_hw=S.S15_HW),
              "arm names / S15 defaults")
    projection = {"differs_from_native_768_recipe": vs_native, "differs_from_S15_recipe": vs_s15, "config": mine}
    # ---- 768 plain / checkpoint CUDA equivalence: reused, SHA only
    eq = B.read_json(B.pinned(c["inputs"]["equivalence"]))
    B.require(eq["status"] == "PASS" and [u["update"] for u in eq["updates"]] == [18241, 18242]
              and all(u["equivalent"] and u["state_exact_equal"] and not u["gradient_differences"] for u in eq["updates"]), "768 equivalence evidence")
    # ---- loader
    B.PHASE["phase"] = "cpu/loader"
    loader = _loader_check(c, Tr, v)
    loader_cost = cost(t0, c0)
    # ---- production bridge, formal-binding save / load / exact restore
    B.PHASE["phase"] = "cpu/bridge"
    e, bridge = Tr._make(v, "cpu")
    B.require(e.config == S.production_config(SEED, ARM) and e.s15_activation_checkpoint is True and e.r25_arm == ARM
              and e.r25_parent == c["parent"]["sha256"] and len(bridge["activation_checkpoint_segments"]) == Tr.BLOCKS
              and bridge["online_size_cache"] == bridge["EMA_size_cache"]
              and bridge["online_size_cache"]["s15_anchor_tokens"] == U.tokens(HW)
              and bridge["online_size_cache"]["parent_anchor_tokens"] == U.tokens(U.L_HW) and bridge["new_parameters"] == 0,
              "U896 production engine: arm, parent, memory mode or size caches")
    path = out / "cpu-production-state.pth"
    Tr._save(e, v, path, "cpu-fixture")
    side, state = Tr._load(v, path, "cpu-fixture")
    B.require(state["arm"] == ARM and tuple(state["config"]["input_hw"]) == HW and side["arm"] == ARM, "a complete state registers the arm and the canvas")
    restored = Tr._restore_engine(e, path, side, state)
    del state
    refused = []
    other = {**v, "contract_sha256": "0" * 64}
    for label, vw, scope, fixture in (("formal scope", v, "formal", False), ("acceptance scope", v, "acceptance", False),
                                      ("fixture config", v, "cpu-fixture", True), ("state of another attempt", other, "cpu-fixture", False)):
        refused.append(_refused(label, lambda: Tr._load(vw, path, scope, fixture=fixture), (ValueError, KeyError)))
    # ---- arms: a U896 state into an S15 engine, an S15 state into the U896 engine; the S15 path is unchanged
    B.PHASE["phase"] = "cpu/arms"
    s15e, s15_bridge = S.create(c["parent"]["path"], "cpu", seed=SEED, activation_checkpoint=True)
    B.require(s15e.r25_arm == "S15" and s15e.config == Config(seed=SEED, input_hw=(1152, 2016)), "default arm is no longer S15")
    cross = "cross-version/arm/canvas/recipe"
    refused.append(_refused("U896 complete state into an S15 engine", lambda: S.restore(s15e, path, side["file"]["sha256"]), ValueError, cross))
    s15_sha = ck.file_sha(S15_E060)
    refused.append(_refused("stored S15 complete state into the U896 engine", lambda: S.restore(e, S15_E060, s15_sha), ValueError, cross))
    refused.append(_refused("stored S15 formal checkpoint through the U896 trainer", lambda: Tr._load(v, S15_E060, "formal"), (ValueError, KeyError)))
    stored = torch.load(S15_E060, map_location="cpu", weights_only=True)
    stored.pop("formal_binding_sha256", None)
    fresh = S.capture(s15e)
    diffs = Tr._differences(stored, fresh)
    outside = [d for d in diffs if not d["path"].startswith("/rng")]
    B.require(not outside, f"the S15 engine built by this release differs from the stored S15 e060 state: {outside[:5]}")
    s15_regression = {"stored_state": {"path": str(S15_E060), "sha256": s15_sha, "bytes": S15_E060.stat().st_size},
                      "fresh_S15_engine_equals_stored_state_except": sorted({d["path"] for d in diffs}),
                      "note": "the stored state was captured on CUDA; a CPU engine holds no CUDA generator state",
                      "S15_anchor_tokens": s15_bridge["online_size_cache"]["s15_anchor_tokens"]}
    del stored, fresh, s15e
    gc.collect()
    # ---- parents of other seeds, other canvases
    for seed, parent in OTHER_SEED_PARENTS.items():
        refused.append(_refused(f"U896 with seed {seed}", lambda: S.create(c["parent"]["path"], "cpu", seed=seed, activation_checkpoint=True, arm=ARM),
                                ValueError, "no certified parent"))
        refused.append(_refused(f"seed{seed} parent file bound as seed 0", lambda: S.create(parent, "cpu", seed=SEED, activation_checkpoint=True, arm=ARM),
                                ValueError, "parent identity"))
    for hw in ((1024, 1792), (768, 1344), (1152, 2016), (896, 1600)):
        refused.append(_refused(f"U896 production engine on canvas {hw[0]}x{hw[1]}",
                                lambda: S.create(c["parent"]["path"], "cpu", Config(seed=SEED, input_hw=hw), seed=SEED, activation_checkpoint=True, arm=ARM),
                                ValueError, "production recipe differs"))
    refused.append(_refused("S15 production engine on canvas 896x1568",
                            lambda: S.create(c["parent"]["path"], "cpu", Config(seed=SEED, input_hw=HW), seed=SEED, activation_checkpoint=True),
                            ValueError, "production recipe differs"))
    refused.append(_refused("unregistered arm", lambda: S.arm_hw("U1024"), ValueError, "unknown scale-baseline arm"))
    refused.append(_refused("a second arm in one process", lambda: Tr.bind_arm("S15"), ValueError, "already bound"))
    # ---- audit prefix
    history = out / "fixture-UPDATES.jsonl"
    with history.open("xb") as stream:
        stream.write(b'{"input":{"update":18241}}\n')
    audit = Tr._file(history)
    with history.open("ab") as stream:
        stream.write(b'{"input":{"update":18242}}\n')
    Tr._verify_audit(audit, history)
    refused.append(_refused("audit prefix with a wrong SHA", lambda: Tr._verify_audit({**audit, "sha256": "0" * 64}, history), ValueError))
    del e
    gc.collect()
    bridge_cost = cost(t0, c0)
    # ---- the formal loop on the fixture: 2 updates, then an exact resume replaying the second
    B.PHASE["phase"] = "cpu/fixture-loop"
    t1, c1 = time.perf_counter(), time.process_time()
    first, second = out / "fixture-run", out / "fixture-resume"
    first.mkdir()
    second.mkdir()
    counts1 = Counter(actual_updates=0, forwards=0, forward_attempts=0)
    run1 = Tr._formal(v, first, counts1, None, None, fixture=True)
    gc.collect()
    counts2 = Counter(actual_updates=0, forwards=0, forward_attempts=0)
    run2 = Tr._formal(v, second, counts2, first / f"fixture-update-{PARENT_UPDATE + 1}.pth", v, fixture=True)
    end1 = torch.load(first / f"fixture-update-{Tr.FIXTURE_STOP}.pth", map_location="cpu", weights_only=True)
    end2 = torch.load(second / f"fixture-update-{Tr.FIXTURE_STOP}.pth", map_location="cpu", weights_only=True)
    end1.pop("formal_binding_sha256")
    end2.pop("formal_binding_sha256")
    B.require(ck.exact_equal(end1, end2), "fixture: the resumed end state differs")
    B.require(run2["prior_recorded_updates_replayed_exact"] == 1 and counts2["actual_updates"] == 1 and counts1["actual_updates"] == 2, "fixture resume count")
    rows1 = (first / "UPDATES.jsonl").read_text().splitlines()
    rows2 = (second / "UPDATES.jsonl").read_text().splitlines()
    B.require(len(rows1) == 2 and rows2[0] == rows1[0] and json.loads(rows2[1])["loss"] == json.loads(rows1[1])["loss"], "fixture ledgers differ")
    B.require(json.loads(rows1[0])["input"] == loader[0]["receipt_896"], "fixture loop window differs from the loader check")
    del end1, end2
    loop_cost = cost(t1, c1)
    # ---- the same two updates without activation checkpointing: complete states bitwise equal (compared in memory)
    B.PHASE["phase"] = "cpu/off-on"
    t2, c2 = time.perf_counter(), time.process_time()
    before = S.RECOMPUTE["calls"]
    plain, _ = S.create(c["parent"]["path"], "cpu", Tr.fixture_config(SEED), seed=SEED, fixture=True, activation_checkpoint=False, arm=ARM)
    B.require(plain.s15_activation_checkpoint is False and not any(getattr(m, "_s15_checkpointed", False) for m in plain.model.modules()),
              "the plain fixture engine has checkpointing installed")
    data = Tr._data(v)
    off_on = []
    for i, (x, t, inp) in enumerate(formal_data.windows(data, PARENT_UPDATE, limit=2), 1):
        result = plain.train_window(x[:2], t[:2])
        counts.update(actual_updates=1, forwards=2, forward_attempts=2)
        mine_state = S.capture(plain)
        kept = torch.load(first / f"fixture-update-{PARENT_UPDATE + i}.pth", map_location="cpu", weights_only=True)
        kept.pop("formal_binding_sha256")
        B.require(mine_state["activation_checkpoint"] is False and kept["activation_checkpoint"] is True, "memory-mode flags of the two runs")
        mine_state["activation_checkpoint"] = True              # the implementation flag is not state
        d = Tr._differences(kept, mine_state)
        row = json.loads(rows1[i - 1])
        B.require(not d and ck.exact_equal(kept, mine_state), f"activation checkpointing changes the U896 update {PARENT_UPDATE + i}: {d[:5]}")
        B.require(row["input"] == inp and row["loss"] == result["loss"] and row["gradient_norm_before_clip"] == result["gradient_norm_before_clip"],
                  "plain fixture update differs from the checkpointed ledger row")
        off_on.append({"update": PARENT_UPDATE + i, "complete_state_bitwise_equal": True, "loss": result["loss"],
                       "gradient_norm_before_clip": result["gradient_norm_before_clip"], "seconds": result["elapsed_seconds"]})
        del mine_state, kept
    B.require(S.RECOMPUTE["calls"] == before, "the plain engine recomputed blocks")
    del plain
    gc.collect()
    B.require(not torch.cuda.is_initialized(), "CPU gate initialized CUDA")
    counts.update({k: counts1[k] + counts2[k] for k in counts1})
    return {"stage": "cpu", "science_projection": projection, "equivalence_768_reused": c["inputs"]["equivalence"], "loader_check": loader,
            "production_bridge": bridge, "production_restore": restored, "refused": refused, "S15_regression": s15_regression,
            "audit_prefix_after_append_exact": True,
            "fixture_formal_loop": {"config": Tr.fixture_config(SEED).binding(), "first": run1, "first_counts": dict(counts1), "resume": run2,
                                    "resume_counts": dict(counts2), "resumed_endpoint_complete_state_exact": True},
            "plain_against_activation_checkpoint": {"updates": off_on, "compared": "S.capture of the plain engine against the states "
                                                    "the checkpointed fixture loop saved; in memory, no extra file"},
            "cpu_fixture_updates": counts["actual_updates"], "cpu_fixture_training_forwards": counts["forwards"],
            "threads": torch.get_num_threads(),
            "cost": {"loader_and_projection": loader_cost, "bridge_and_state_cumulative": bridge_cost, "fixture_loop": loop_cost,
                     "plain_fixture": cost(t2, c2)},
            "CUDA_initialized": torch.cuda.is_initialized()}


# ================================================================ evaluation: CPU checks, inference, scoring
def evaluator():
    import s15_formal_eval as Ev
    Ev.bind_arm(ARM)
    B.require(Ev.ARM == ARM and Ev.KIND == "EMA" and (Ev.BATCH, Ev.WARMUPS, Ev.ENDPOINT) == (4, 4, END_UPDATE) and tuple(Ev.BUDGETS) == (600, 300),
              "evaluation constants")
    return Ev


def validate_policy(c):
    from sparse_rtdetr.localization.runio import read
    from sparse_rtdetr.r0.evaluation import POLICY
    B.require(read(B.pinned(c["evaluation_policy"])) == POLICY, "the certified view-A evaluation policy differs")


def eval_check(c, out, counts, args):
    from sparse_rtdetr.baseline.primary_evaluator import primary_evaluator_contract_binding
    from torchvision.transforms.v2 import functional as TF
    torch = B.torch_mod()
    Ev = evaluator()
    _, S = trainer()
    B.require(not torch.cuda.is_initialized(), "CPU evaluation checks initialized CUDA")
    validate_policy(c)
    ev = eval_view(c)
    data = Ev.development(ev)
    row = data.records[0]
    x = Ev._image(data, row)
    with Ev.load_original(data, row) as image:
        direct = TF.pil_to_tensor(TF.resize(image, list(HW))).float() / 255.
    B.require(x.dtype == torch.float32 and tuple(x.shape) == (3, *HW) and bool(torch.equal(x, direct)),
              "GT-free pixels differ from one direct 896 x 1568 resize")
    primary_evaluator_contract_binding(c["source_root"])
    flag = ev["workload"]["activation_checkpoint"]
    e, _ = S.create(c["parent"]["path"], "cpu", seed=SEED, activation_checkpoint=flag, arm=ARM)
    payload = S.capture(e)
    del e
    gc.collect()
    payload["engine"]["completed_updates"] = END_UPDATE
    payload["ema"]["updates"] = END_UPDATE
    key = next(k for k, v in payload["model"].items() if v.is_floating_point() and not S._is_size_cache(k))
    payload["ema"]["module"][key] = payload["model"][key] + 0.001
    model, sha = Ev._load_model(payload, ev, "EMA", "cpu", flag)
    B.require(bool(torch.equal(model.state_dict()[key], payload["ema"]["module"][key]))
              and not bool(torch.equal(model.state_dict()[key], payload["model"][key])), "the EMA end point selected the wrong state")
    refused = []
    for label, broken in (("wrong clock", {**payload, "engine": {**payload["engine"], "completed_updates": END_UPDATE - 1}}),
                          ("768 recipe", {**payload, "config": S.Config(seed=SEED).binding()}),
                          ("S15 recipe (1152 x 2016)", {**payload, "config": S.production_config(SEED).binding()}),
                          ("S15 arm name", {**payload, "arm": "S15"}),
                          ("memory mode", {**payload, "activation_checkpoint": not flag}),
                          ("seed1 recipe", {**payload, "config": S.production_config(1, ARM).binding()}),
                          ("seed1 parent", {**payload, "parent_sha256": S.PARENT_SHAS[1]})):
        refused.append(_refused(label, lambda: Ev._load_model(broken, ev, "EMA", "cpu", flag), (ValueError, RuntimeError))["case"])
    with torch.inference_mode():
        counts.update(forward_attempts=1)
        preds, timing = Ev.forward_batch(model, data, data.records[:Ev.BATCH], "cpu")
        counts.update(forwards=1)
    ids = [r["stable_image_id"] for r in data.records[:Ev.BATCH]]
    grouped = {b: Ev._group(preds[b], ids, b, whole=True, exact=True) for b in Ev.BUDGETS}
    Ev._prefix(grouped[600], grouped[300], ids, whole=True)
    B.require(set(preds[600][0]) == {"image_id", "query_id", "category_id", "score", "bbox_xyxy"}, f"record keys: {sorted(preds[600][0])}")
    B.require(not torch.cuda.is_initialized(), "CPU evaluation checks initialized CUDA")
    return {"stage": "eval-check", "development_images": len(data.records), "one_actual_development_image_pixels_equal_direct_resize": ids[0],
            "input_hw": list(HW), "EMA_weight_selection_sha256": sha, "refused": refused,
            "one_cpu_endpoint_forward": {"images": Ev.BATCH, "timing": timing, "ranked_per_image": len(preds[600]) // Ev.BATCH},
            "ranking_prefix_exact": True, "record_keys": sorted(preds[600][0]), "detector_forwards": 1, "score_calls": 0,
            "evaluation_policy": c["evaluation_policy"]}


def accept(c, out, counts, args):
    Tr, _ = trainer()
    B.require(not c["dev"], "a dev round has no CUDA acceptance")
    v = view(c, args.name)
    r = Tr._accept(v, out, counts)
    body = r.pop("S15")                                        # the reused function's own key for the arm it runs
    seconds = [row["update"]["elapsed_seconds"] for row in body["first_updates"]] + [body["replay_update_seconds"]]
    peaks = [body["first_phase_peak"], body["replay_peak"]]
    return {"stage": "accept", "arm": ARM, "U896": body, **r,
            "summary": {"update_seconds": seconds, "peak_allocated_bytes": max(p["peak_allocated_bytes"] for p in peaks),
                        "peak_reserved_bytes": max(p["peak_reserved_bytes"] for p in peaks),
                        "replay_complete_state_exact": body["replay_complete_state_and_record_exact"]}}


def accepted(c):
    """The closed acceptance round (exactly one may pass)."""
    root = Path(c["outputs"])
    names = [f"accept-{k}" for k in range(1, ACCEPT_ROUNDS + 1) if (root / f"accept-{k}" / "CLOSED.json").is_file()]
    B.require(len(names) == 1, f"CUDA acceptance not closed: {names}")
    closed = B.read_json(root / names[0] / "CLOSED.json")
    B.require(closed["status"] == "PASS" and closed["contract_sha256"] == c["contract_sha256"]
              and closed["summary"]["replay_complete_state_exact"] is True, "acceptance closure")
    closed["_name"] = names[0]
    return closed


def gates(c):
    """Everything the formal training waits for (task 4.4, 4.6)."""
    t0 = stage_closed(c, "t0-predict")
    B.pinned(t0["prediction"])
    B.require(stage_closed(c, "t0-wire")["passed"] is True and stage_closed(c, "t0-ends")["passed"] is True, "T0 wiring")
    return {"prediction": t0["prediction"], "cpu": stage_closed(c, "cpu")["_name"], "eval_check": stage_closed(c, "eval-check")["_name"],
            "accept": accepted(c)["_name"]}


def train(c, out, counts, args):
    Tr, _ = trainer()
    B.require(not c["dev"], "a dev round never trains")
    passed = gates(c)
    v = view(c, args.name)
    prior = view(c, args.prior) if args.prior else None
    B.require((args.resume is None) == (prior is None) == (v["prior"] is None), "resume needs the checkpoint and the prior attempt")
    r = Tr._formal(v, out, counts, args.resume, prior)
    return {"stage": "train", "gates": passed, **r}


def align(c, out, counts, args):
    names, closed = final_train(c)
    ledger = Path(c["outputs"]) / names[-1] / "UPDATES.jsonl"
    with ledger.open() as mine, B.pinned(c["inputs"]["a0_trajectory"]).open() as native:
        res = align_rows(mine, native)
    B.require(res["updates"] == END_UPDATE - PARENT_UPDATE, "input audit length")
    return {"stage": "align", "attempt": names[-1], "ledger": B.ref(ledger), "native_trajectory": c["inputs"]["a0_trajectory"], **res}


def infer(c, out, counts, args):
    from sparse_rtdetr.localization.runio import read
    import th2_phase1 as P
    torch = B.torch_mod()
    Tr, _ = trainer()
    Ev = evaluator()
    B.require(not c["dev"], "a dev round never infers")
    validate_policy(c)
    names, closed = final_train(c)
    B.require(stage_closed(c, "align")["updates"] == END_UPDATE - PARENT_UPDATE and stage_closed(c, "eval-check")["detector_forwards"] == 1,
              "input alignment and the evaluation checks come first")
    tv = view(c, names[-1])
    path = Path(tv["outputs"]["train"]) / "checkpoint-epoch-120.pth"
    side = read(path.with_suffix(".pth.json"))
    B.require((side["status"], side["arm"], side["seed"], side["contract_sha256"], side["formal_identity"], side["global_update"], side["scope"])
              == ("COMPLETE", ARM, SEED, tv["contract_sha256"], c["formal_identity"], END_UPDATE, "formal")
              and Path(side["file"]["path"]).resolve() == path.resolve(), "end point sidecar identity")
    stage = out / "U896-EMA"
    stage.mkdir()
    counts.update(forward_attempts=Ev.WARMUPS + 137)
    result = Ev.infer(eval_view(c), side["file"], stage, lambda p: Tr.read_endpoint(tv, p))
    counts.update(forwards=result["completed_forwards"])
    B.require((result["status"], result["arm"], result["kind"], result["images"], result["steady_forwards"], result["warmup_forwards"],
               tuple(result["input_hw"])) == ("PASS", ARM, "EMA", M, 137, Ev.WARMUPS, HW), "inference result")
    # task 4.7 item 7
    index = P.stored_index(result["predictions"]["600"]["path"])
    three = read(result["predictions"]["300"]["path"])
    B.require(list(index) == c["development_ids"] and all(len(v[0]) == 600 for v in index.values()) and len(three) == M * 300,
              "4.7 item 7: the top600 list is not 548 x 600 records grouped by image in development order")
    del index, three
    return {"stage": "infer", "attempt": names[-1], "checkpoint": side["file"], "result": B.ref(stage / "RESULT.json"),
            "predictions": result["predictions"], "weights_sha256": result["weights_sha256"],
            "item7_top600_is_548x600_grouped_and_readable": True, "steady_timing": result["steady_timing"],
            "steady_forward_seconds_per_batch": result["steady_timing"]["forward"] / 137,
            "inference_peak_allocated_bytes": result["peak_allocated_bytes"], "inference_peak_reserved_bytes": result["peak_reserved_bytes"]}


def score(c, out, counts, args):
    """The scoring stage of the S15 evaluation tool (s15_formal_eval.score) without its S15-only parts (A0 reference,
    per-class APL, decision): development rows, parsed labels, _group (exact), _prefix, score_budget, bridge_values."""
    from infer_seed import BUDGETS, _group, _prefix, bridge_values
    from sparse_rtdetr.localization.runio import read
    from sparse_rtdetr.r0.evaluation import score_budget
    B.require(not c["dev"], "a dev round never scores")
    inf = stage_closed(c, "infer")
    result = read(B.pinned(inf["result"]))
    B.require((result["status"], result["arm"], result["seed"], result["kind"], result["contract_sha256"], result["images"],
               result["steady_forwards"], result["warmup_forwards"]) == ("PASS", ARM, SEED, "EMA", c["contract_sha256"], M, 137, 4),
              "scoring: inference identity")
    data = development(c, HW)
    rows, ids = data.records, c["development_ids"]
    parsed = [data.raw(row) for row in rows]
    predictions, grouped = {}, {}
    for budget in BUDGETS:
        ref = result["predictions"][str(budget)]
        predictions[budget] = read(B.pinned({"path": str(Path(ref["path"]).resolve()), "bytes": ref["bytes"], "sha256": ref["sha256"]}))
        grouped[budget] = _group(predictions[budget], ids, budget, whole=True, exact=True)
    _prefix(grouped[600], grouped[300], ids, whole=True)
    del grouped
    values, files, seconds = {}, {}, {}
    for budget in BUDGETS:
        B.PHASE["phase"] = f"score/top{budget}"
        t = time.perf_counter()
        spend(c, f"score/top{budget}", counts)
        metrics = score_budget(rows, parsed, predictions[budget], c["source_root"])
        B.require({k: metrics["spatial_filter_counts"][k] for k in GT_COUNTS} == GT_COUNTS, "scored GT set")
        metrics["U896_binding"] = {"contract_sha256": c["contract_sha256"], "formal_identity": c["formal_identity"], "arm": ARM, "seed": SEED,
                                   "kind": "EMA", "view": "A", "input_hw": list(HW), "epoch": 120, "budget": budget,
                                   "checkpoint": result["checkpoint"], "weights_sha256": result["weights_sha256"],
                                   "predictions": result["predictions"][str(budget)]}
        values[str(budget)] = bridge_values(metrics)
        path = out / f"U896-seed0-EMA-A-top{budget}-metrics.json"
        B.write_new(path, metrics)
        files[str(budget)] = B.ref(path)
        seconds[str(budget)] = time.perf_counter() - t
    B.require(counts["score_budget_calls"] == 2 and counts["forwards"] == 0, "score: counts")
    return {"stage": "score", "values": values, "files": files, "GT_counts": GT_COUNTS, "score_budget_seconds": seconds}


def routed(c, out, counts, args):
    B.require(not c["dev"], "a dev round never scores")
    B.require(stage_closed(c, "score")["GT_counts"] == GT_COUNTS, "the U896 scores come first")
    route = np.load(B.pinned(c["inputs"]["assignments"]), allow_pickle=False)["P-k158"]
    B.require(int(route.sum()) == U.K_GATE, "P-k158")
    v, ref, same = Scorer(c, counts).score(out, "routed/P-k158", "routed-P-k158-A0-S15", route)
    B.require(counts["score_budget_calls"] == 1 and counts["forwards"] == 0 and not any(same.values()), "routed: counts")
    return {"stage": "routed", "values_600": v, "file": ref, "routed_images": int(route.sum()), "cheap": "A0-s0", "expensive": "S15-s0"}


def readings(c, out, counts, args):
    B.PHASE["phase"] = "readings"
    B.require(not c["dev"], "a dev round produces no reading")
    B.require(not list(Path(c["outputs"]).glob("readings*/RESULT.json")), "RESULT.json exists: the readings are produced once")
    t0 = stage_closed(c, "t0-predict")
    prediction = B.read_json(B.pinned(t0["prediction"]))
    ends, inf, sc, rt, al = (stage_closed(c, n) for n in ("t0-ends", "infer", "score", "routed", "align"))
    names, trained = final_train(c)
    acc = accepted(c)
    gt, iou, route = load_arrays(c)
    iou["a"] = np.load(B.pinned(t0["per_gt_A0"]), allow_pickle=False)["iou_A0-s0"]
    ref = inf["predictions"]["600"]
    iou["u"] = list_outcomes(c, gt, B.pinned({"path": str(Path(ref["path"]).resolve()), "bytes": ref["bytes"], "sha256": ref["sha256"]}))
    B.require(iou["u"].shape == (N_GT,) and bool(np.isfinite(iou["u"]).all()), "iou of U896-s0")
    rows = development(c, tuple(U.H_HW)).records
    width, height = np.array([r["width"] for r in rows]), np.array([r["height"] for r in rows])
    group = native_group(width, height)
    reg = c["registered"]["top"]
    five = lambda v: {m: v[m] for m in METRICS}
    ap = {"U896": {b: five(sc["values"][b]) for b in ("600", "300")}, "A0": reg["A0-s0"], "S15": reg["S15-s0"], "L0": reg["L0-s0"], "FG": reg["FG-s0"],
          "routed_A0_S15": {"600": five(rt["values_600"])}, "routed_L0_S15": reg["routed_P-k158_L0_S15"],
          "routed_L0_S15_small_hit75_pp": c["registered"]["small_hit75_pp"]["routed_P-k158_L0_S15"]}
    inp = {"image": gt.image, "area": gt.area, "s": gt.s, "cat": gt.cat, "occlusion": gt.occlusion, "crowd": gt.crowd, "g0": gt.g0,
           "iou": iou, "routed": route, "group": group}
    tables, columns = reading_tables(inp, ap, prediction)
    B.require(abs(tables["U2"]["small_hit75"]["A0_pp"] - prediction["endpoints"]["A0_small_hit75_pp"]) <= TOL
              and abs(tables["U2"]["small_hit75"]["S15_pp"] - REG_HIT["S15-s0"]) <= TOL, "readings: end points differ from T0")
    with (out / "per_gt.npz").open("xb") as f:
        np.savez(f, **{"iou_A0-s0": iou["a"], "iou_U896-s0": iou["u"]})
    with (out / "per_image.csv").open("x", newline="") as f:
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(["index", "image_id", "width", "height"] + list(columns))
        for i, iid in enumerate(c["development_ids"]):
            wr.writerow([i, iid, int(width[i]), int(height[i])] + [int(columns[k][i]) for k in columns])
    # ---------------- U6
    root = Path(c["outputs"])
    journal = [json.loads(line) for line in (root / "STAGES.jsonl").read_text().splitlines()] if (root / "STAGES.jsonl").is_file() else []
    finish = {r["name"]: r for r in journal if r.get("event") == "FINISH"}
    begun = {r["name"]: r for r in journal if r.get("event") == "START"}
    attempts, total_actual, wall = [], 0, 0.0
    for n in names:
        d = root / n
        ledger = d / "UPDATES.jsonl"
        lines = sum(1 for line in ledger.open("rb") if line.endswith(b"\n")) if ledger.is_file() else 0
        first = B.read_json(d / "RESTORE.json")["global_update"] if (d / "RESTORE.json").is_file() else PARENT_UPDATE
        spent = finish.get(n, {}).get("seconds")
        if spent is None and ledger.is_file() and n in begun:      # the supervisor died with the run: last ledger write - start
            spent = ledger.stat().st_mtime - begun[n]["started_epoch_seconds"]
        row = {"attempt": n, "started_at_update": first, "ledger_rows": lines, "updates_recorded_by_this_attempt": lines - (first - PARENT_UPDATE),
               "wall_seconds": spent, "supervisor": {k: finish.get(n, {}).get(k) for k in ("seconds", "status", "gpu_watch")}}
        for name in ("CLOSED.json", "STOP.json"):
            if (d / name).is_file():
                doc = B.read_json(d / name)
                row[name] = {k: doc.get(k) for k in ("status", "elapsed_seconds", "started_at_update", "actual_updates", "peak_allocated_bytes",
                                                     "peak_reserved_bytes", "prior_recorded_updates_replayed_exact", "block_recomputations",
                                                     "phase", "error")}
        total_actual += row["updates_recorded_by_this_attempt"]
        wall += spent or 0.0
        attempts.append(row)
    seconds = np.array([json.loads(line)["update_seconds"] for line in (root / names[-1] / "UPDATES.jsonl").open()])
    gpu_stages = [n for n in finish if n.startswith(("accept-", "train-U896", "infer"))]
    u6 = {"training": {"attempts": attempts, "exact_restores": len(names) - 1, "restores_at_most": RESTORES,
                       "updates_recorded_over_all_attempts": total_actual,
                       "updates_replayed_beyond_18240": total_actual - (END_UPDATE - PARENT_UPDATE),
                       "wall_seconds_of_training_stages": wall,
                       "update_seconds_in_window": {"mean": float(seconds.mean()), "median": float(np.median(seconds)),
                                                    "p95": float(np.quantile(seconds, 0.95)), "rows": int(len(seconds))},
                       "wall_seconds_per_update_final_attempt": trained["elapsed_seconds"] / max(trained["actual_updates"], 1),
                       "peak_allocated_bytes_final_attempt": trained["peak_allocated_bytes"], "peak_reserved_bytes_final_attempt": trained["peak_reserved_bytes"],
                       "block_recomputations_final_attempt": trained["block_recomputations"], "block_recomputations_per_update": 16},
          "acceptance": {"round": acc["_name"], **acc["summary"], "rounds_started": len(list(root.glob("accept-[0-9]")))},
          "inference": {"steady_timing": inf["steady_timing"], "steady_forward_seconds_per_batch": inf["steady_forward_seconds_per_batch"],
                        "peak_allocated_bytes": inf["inference_peak_allocated_bytes"], "peak_reserved_bytes": inf["inference_peak_reserved_bytes"]},
          "supervised_targets": {"U896": al["U896_positive_GT_total_after_sanitize"], "S15": c["registered"]["S15_supervised_targets"], "A0": a0_total(c),
                                 "rows_with_canvas_dependent_difference": al["rows_with_canvas_dependent_difference"], "rows": al["updates"]},
          "gpu_other_processes": {n: finish[n].get("gpu_watch") for n in gpu_stages},
          "gpu_seconds_of_cuda_stages": {n: finish[n].get("seconds") for n in gpu_stages}}
    used = scorings(c["home"])
    B.require([r["slot"] for r in used] == list(SLOTS), f"scoring ledger: {[r['slot'] for r in used]}")
    f = tables["U2"]["small_hit75"]
    result = {"status": "PASS", "version": VERSION, "contract_sha256": c["contract_sha256"], "source_commit": c["source_commit"],
              "source_branch": c["source_branch"], "task": c["task"], "references": c["references"], "arm": ARM, "seed": SEED, "input_hw": list(HW),
              "summary": {"f_small_hit75": f["f"], "f_ci95": f["ci95"], "predictions": f["predictions"],
                          "stage1_reading": tables["U3"]["small_hit75"]["stage1_reading"],
                          "routed_minus_U896_pp": tables["U3"]["small_hit75"]["routed_minus_U896_pp"],
                          "note": "no GO / STOP; one seed; stage1_reading is exploratory and starts nothing"},
              **tables, "U6": u6, "prediction": t0["prediction"],
              "wire": {"t0_wire": stage_closed(c, "t0-wire")["checks"], "t0_ends": {k: ends[k] for k in ("values_600", "max_abs_difference_from_registered",
                                                                                                         "spliced_list_equals_stored_file")},
                       "item7": inf["item7_top600_is_548x600_grouped_and_readable"]},
              "identity": {"endpoint_checkpoint": inf["checkpoint"], "weights_sha256": inf["weights_sha256"], "predictions": inf["predictions"],
                           "metric_files": {"U896_top600": sc["files"]["600"], "U896_top300": sc["files"]["300"], "routed_P-k158_A0_S15": rt["file"],
                                            "A0_R_empty": ends["files"]["A0"], "S15_R_all": ends["files"]["S15"]},
                           "final_attempt": names[-1], "formal_identity": c["formal_identity"]},
              "files": {"per_gt": B.ref(out / "per_gt.npz"), "per_image": B.ref(out / "per_image.csv")},
              "scorings": used, "bytes_in_home_before_this_file": bytes_under(c["home"]),
              "boundaries": c["boundaries"], "definitions": c["definitions"]}
    B.write_new(out / "RESULT.json", C0.clean(result))
    return {"stage": "readings", "result": B.ref(out / "RESULT.json"), "files": result["files"], "summary": C0.clean(result["summary"])}


# ================================================================ the stage process
STAGES = {"t0-wire": t0_wire, "t0-ends": t0_ends, "t0-predict": t0_predict, "a0-count": a0_count, "cpu": cpu, "eval-check": eval_check,
          "accept": accept, "train": train, "align": align, "infer": infer, "score": score, "routed": routed, "readings": readings}


def stage_main(args):
    c = load_contract(args.round)
    action, name = args.action, args.name
    device = "cuda:0" if action in CUDA_STAGES else "cpu"
    B.require(not c["dev"] or action in DEV_STAGES, f"a dev round runs {DEV_STAGES} only")
    if device == "cpu":
        B.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "a CPU stage requires CUDA_VISIBLE_DEVICES=\"\"")
    else:
        B.require(os.environ.get("CUDA_VISIBLE_DEVICES") == c["gpu_uuid"] and os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == ALLOCATOR
                  and os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "CUDA stage environment differs from the contract")
    out = Path(c["outputs"]) / name
    B.require(not out.exists(), f"stage directory exists: {out}")
    if action in ATTEMPT_KINDS:
        prior = None
        if args.prior:
            prior = {"contract": B.ref(attempt_file(c, args.prior)), "attempt": args.prior, "checkpoint": str(Path(args.resume).resolve())}
        new_attempt(c, name, ATTEMPT_KINDS[action], prior)
    out.mkdir()
    start, cpu0 = time.monotonic(), time.process_time()
    counts = Counter(actual_updates=0, forwards=0, forward_attempts=0)
    torch = None
    try:
        from r25_run import setup
        torch = B.torch_mod()
        setup(device)
        if device == "cpu":
            torch.set_num_threads(int(os.environ.get("TH2U1_CPU_THREADS", "8")))
        else:
            torch.cuda.reset_peak_memory_stats()
        result = STAGES[action](c, out, counts, args)
        B.require(isinstance(result, dict), f"{action}: stage result")
        memory = {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
        if device != "cpu":
            memory = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        children = os.times()
        B.write_new(out / "CLOSED.json", C0.clean({
            "status": "PASS", "action": action, "contract_sha256": c["contract_sha256"], "counts": dict(counts),
            "started_at_unix": time.time() - (time.monotonic() - start), "elapsed_seconds": time.monotonic() - start,
            "process_cpu_seconds": time.process_time() - cpu0 + children.children_user + children.children_system,
            "closed_at": now(), **memory, **result}))
    except BaseException as exc:
        oom = torch is not None and isinstance(exc, torch.cuda.OutOfMemoryError)
        memory = {}
        try:
            if torch is not None and torch.cuda.is_initialized():
                memory = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        except Exception:
            pass
        if not (out / "STOP.json").exists():
            B.write_new(out / "STOP.json", {"status": "STOP_CUDA_OOM" if oom else "STOP", "phase": B.PHASE["phase"], "action": action,
                                            "unit": args.unit, "error": repr(exc)[:4000], "traceback": traceback.format_exc(),
                                            "counts": dict(counts), "actual_updates": counts["actual_updates"],
                                            "elapsed_seconds": time.monotonic() - start, **memory})
        raise
    finally:
        gc.collect()


# ================================================================ supervisor (no torch in this process)
def execution_environment(device, tmp):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update(CUDA_VISIBLE_DEVICES=device, CUBLAS_WORKSPACE_CONFIG=":4096:8", MKL_THREADING_LAYER="GNU", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF=ALLOCATOR)
    if tmp:
        env.update(TMPDIR=str(tmp), TEMP=str(tmp), TMP=str(tmp))
    return env


def smi(*query):
    out = subprocess.check_output(["nvidia-smi", *query, "--format=csv,noheader,nounits"], text=True, stderr=subprocess.STDOUT, timeout=30)
    return [[x.strip() for x in line.split(",")] for line in out.splitlines() if line.strip()]


def gpu_memory(uuid):
    rows = [r for r in smi("--query-gpu=uuid,memory.free,memory.used,memory.total") if r[0] == uuid]
    B.require(len(rows) == 1, "the bound GPU is not present")
    return {k: int(v) * 1024 * 1024 for k, v in zip(("free_bytes", "used_bytes", "total_bytes"), rows[0][1:])}


def gpu_apps(uuid):
    return [{"pid": int(r[0]), "used_mib": r[2]} for r in smi("--query-compute-apps=pid,gpu_uuid,used_gpu_memory") if r[1] == uuid]


def journal(path, value):
    with Path(path).open("a") as f:
        f.write(json.dumps(value, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def group_alive(pid):
    try:
        os.killpg(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def stop_child(child):
    if group_alive(child.pid):
        os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    if group_alive(child.pid):
        os.killpg(child.pid, signal.SIGKILL)
    child.wait(timeout=10)


def progress_time(directory, log):
    files = [log, directory / "UPDATES.jsonl", directory / "progress.jsonl", *directory.glob("*/progress.jsonl")]
    return max((p.stat().st_mtime_ns for p in files if p.exists()), default=0)


def run_stage(c, action, name, extra=()):
    """One stage as a child process in the contract's environment, with the no-progress watchdog and, on CUDA, a
    read-only watch of other processes on the GPU."""
    root = Path(c["outputs"])
    directory, log = root / name, root / f"{name}.log"
    cuda = action in CUDA_STAGES
    B.require(not directory.exists() and not log.exists(), f"stage {name} was already started")
    limits_guard(c["home"])
    tmp = tempfile.mkdtemp(prefix="th2u1-", dir="/tmp")
    env = execution_environment(c["gpu_uuid"] if cuda else "", tmp)
    argv = [c["python"], "-I", "-B", "-u", str(ROOT / "tools/r28/th2_u1.py"), "stage", action, "--round", str(root), "--name", name, *extra]
    if action == "a0-count":
        argv = ["nice", "-n", "19", *argv]
        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TH2U1_CPU_THREADS="1")
    row = {"name": name, "action": action, "argv": argv, "started_at": now(), "started_epoch_seconds": time.time(), "temporary_directory": tmp}
    if cuda:
        row["gpu_at_start"] = {**gpu_memory(c["gpu_uuid"]), "compute_apps": gpu_apps(c["gpu_uuid"])}
    journal(root / "STAGES.jsonl", {"event": "START", **row})
    began, child = time.monotonic(), None
    watch = {"samples": 0, "samples_with_other_processes": 0, "other_pids": [], "query_errors": 0}
    try:
        with log.open("x") as stream:
            child = subprocess.Popen(argv, cwd=ROOT, env=env, start_new_session=True, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
        (root / f"{name}.pid").write_text(f"{child.pid}\n")
        stamp, last, sampled = progress_time(directory, log), time.monotonic(), 0.0
        while child.poll() is None:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                updated = progress_time(directory, log)
                if updated != stamp:
                    stamp, last = updated, time.monotonic()
                B.require(time.monotonic() - last < NO_PROGRESS_SECONDS, f"no progress in {name} for {NO_PROGRESS_SECONDS} s")
                if cuda and time.monotonic() - sampled >= 60:
                    sampled = time.monotonic()
                    try:
                        others = [a for a in gpu_apps(c["gpu_uuid"]) if a["pid"] != child.pid]
                        watch["samples"] += 1
                        if others:
                            watch["samples_with_other_processes"] += 1
                            watch["other_pids"] = sorted(set(watch["other_pids"]) | {a["pid"] for a in others})
                            journal(root / "gpu-watch.jsonl", {"at": now(), "stage": name, "own_pid": child.pid, "others": others})
                    except (subprocess.SubprocessError, ValueError, OSError):
                        watch["query_errors"] += 1
        B.require(child.returncode == 0, f"{name} exited {child.returncode}; see {log}")
        deadline = time.monotonic() + 5
        while group_alive(child.pid) and time.monotonic() < deadline:
            time.sleep(0.1)
        B.require(not group_alive(child.pid), f"{name} left a live worker process")
        closed = B.read_json(directory / "CLOSED.json")
        B.require(closed["status"] == "PASS" and closed["contract_sha256"] == c["contract_sha256"], f"{name}: closure")
        row.update(status="PASS", result=B.ref(directory / "CLOSED.json"))
        return closed
    except BaseException as exc:
        row.update(status="STOP", error=repr(exc)[:1000])
        if (directory / "STOP.json").is_file():
            row["result"] = B.ref(directory / "STOP.json")
        if child is not None:
            stop_child(child)
        raise
    finally:
        row.update(seconds=time.monotonic() - began, returncode=child.returncode if child is not None else None, finished_at=now())
        if cuda:
            row["gpu_watch"] = watch
        journal(root / "STAGES.jsonl", {"event": "FINISH", **row})
        shutil.rmtree(tmp, ignore_errors=True)


def lock(c):
    handle = (Path(c["outputs"]) / ".th2u1.lock").open("a")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError("another th2_u1 supervisor holds the round lock")
    return handle


def pid_alive(c, name):
    path = Path(c["outputs"]) / f"{name}.pid"
    return path.is_file() and group_alive(int(path.read_text().strip()))


def next_name(c, action, unit, rerun):
    root = Path(c["outputs"])
    if action == "a0-count":
        B.require(unit in A0_UNITS, "a0-count needs --unit 0..5")
        return f"a0-count-{unit}", ["--unit", unit]
    B.require(unit is None, "--unit is for a0-count")
    if action == "accept":
        started = [k for k in range(1, ACCEPT_ROUNDS + 1) if (root / f"accept-{k}").exists() or attempt_file(c, f"accept-{k}").exists()]
        B.require(not any((root / f"accept-{k}" / "CLOSED.json").is_file() for k in started), "the acceptance is already closed")
        B.require(len(started) < ACCEPT_ROUNDS, f"{ACCEPT_ROUNDS} acceptance rounds were started: stop and report (task section 10)")
        B.require(stage_closed(c, "cpu")["CUDA_initialized"] is False and stage_closed(c, "eval-check")["detector_forwards"] == 1, "CPU gates first")
        return f"accept-{len(started) + 1}", []
    existing = siblings(c, action)
    if not existing:
        return action, []
    B.require(rerun, f"stage {action} exists: {existing} (a rerun after an engineering failure needs --rerun)")
    for n in existing:
        d = root / n
        B.require((d / "STOP.json").is_file() and not (d / "CLOSED.json").exists() and not any((d / p).exists() for p in PRODUCTS)
                  and not pid_alive(c, n), f"{n}: not a stopped stage without products; a rerun is refused")
    return f"{action}-r{len(existing) + 1}", []


def run(args):
    c = load_contract(args.round)
    B.require(args.action in STAGES and args.action != "train", "run: train is started by chain only")
    B.require(not c["dev"] or args.action in DEV_STAGES, f"a dev round runs {DEV_STAGES} only")
    name, extra = next_name(c, args.action, args.unit, args.rerun)
    if args.action == "a0-count":                           # the six units run side by side: no round lock
        closed = run_stage(c, args.action, name, extra)
    else:
        with lock(c):
            closed = run_stage(c, args.action, name, extra)
    print(json.dumps({"stage": name, "status": closed["status"], "elapsed_seconds": closed["elapsed_seconds"]}))


def best_checkpoint(c, names):
    """The latest complete state this task saved (epoch, then attempt order)."""
    root, best = Path(c["outputs"]), None
    for i, n in enumerate(names):
        for ep in SAVE_EPOCHS[:-1]:
            p = root / n / f"checkpoint-epoch-{ep:03d}.pth"
            if p.with_suffix(".pth.json").is_file() and p.is_file() and (best is None or (ep, i) > best[:2]):
                best = (ep, i, n, p)
    return best


def chain(args):
    c = load_contract(args.round)
    B.require(not c["dev"], "a dev round has no chain")
    root = Path(c["outputs"])
    with lock(c):
        passed = gates(c)
        acc = accepted(c)
        names = train_attempts(c)
        if not (names and (root / names[-1] / "CLOSED.json").is_file()):
            extra, name = [], "train-U896"
            if names:
                last = names[-1]
                B.require(args.resume_train, f"{last} is not closed; an exact restore needs --resume-train")
                B.require(not pid_alive(c, last), f"{last} is still running")
                B.require(not (root / last / "STOP.json").exists(),
                          f"{last} stopped by itself (STOP.json): not an external interruption; stop and report (task section 2)")
                B.require(len(names) - 1 < RESTORES, f"{RESTORES} exact restores were used: stop and report")
                best = best_checkpoint(c, names)
                B.require(best is not None, "no complete state of this task to restore from")
                name = f"train-U896-r{len(names)}"
                extra = ["--resume", str(best[3]), "--prior", best[2]]
            memory = gpu_memory(c["gpu_uuid"])
            need = acc["summary"]["peak_reserved_bytes"] * 1.25 + 1.0e9
            start = {"at": now(), "attempt": name, "gates": passed, "free_bytes": memory["free_bytes"], "needed_bytes": need,
                     "acceptance_peak_reserved_bytes": acc["summary"]["peak_reserved_bytes"], "compute_apps": gpu_apps(c["gpu_uuid"]),
                     "passed": memory["free_bytes"] >= need, "resume": extra}
            journal(root / "START_CONDITION.jsonl", start)
            B.require(start["passed"], f"start condition not met: free {memory['free_bytes']} < needed {need:.0f} bytes; not launched")
            run_stage(c, "train", name, extra)
        for action in ("align", "infer", "score", "routed", "readings"):
            existing = siblings(c, action)
            if any((root / n / "CLOSED.json").is_file() for n in existing):
                continue
            B.require(not existing, f"stage {action} exists without CLOSED.json: {existing}")
            run_stage(c, action, action)
    print(json.dumps({"chain": "DONE", "result": B.ref(root / "readings" / "RESULT.json")}))


def status(args):
    R_ = Path(args.round).resolve()
    out = {"round": str(R_), "at": now(), "stages": {}, "scorings": [r["slot"] for r in scorings(R_)], "bytes_used": bytes_under(R_)}
    for d in sorted(p for p in R_.iterdir() if p.is_dir() and not p.name.startswith("release")):
        state = "PASS" if (d / "CLOSED.json").is_file() else "STOP" if (d / "STOP.json").is_file() else "OPEN"
        row = {"state": state}
        pid = R_ / f"{d.name}.pid"
        if state == "OPEN" and pid.is_file():
            row["alive"] = group_alive(int(pid.read_text().strip()))
        ledger = d / "UPDATES.jsonl"
        if ledger.is_file():
            with ledger.open("rb") as f:
                rows = sum(1 for _ in f)
            row.update(ledger_rows=rows, ledger_age_seconds=time.time() - ledger.stat().st_mtime,
                       checkpoints=sorted(p.name for p in d.glob("checkpoint-epoch-*.pth")))
        out["stages"][d.name] = row
    print(json.dumps(out, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("prepare", "run", "chain", "status", "stage"))
    ap.add_argument("action", nargs="?")
    ap.add_argument("--round", type=Path, required=True)
    ap.add_argument("--release")
    ap.add_argument("--home", type=Path)
    ap.add_argument("--name")
    ap.add_argument("--unit")
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--prior")
    ap.add_argument("--resume-train", action="store_true")
    args = ap.parse_args(argv)
    if args.mode == "prepare":
        B.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and args.release and args.action is None, "prepare: CPU, --release")
        return prepare(args.round, args.release, args.home)
    if args.mode == "status":
        return status(args)
    if args.mode == "chain":
        B.require(args.action is None, "chain takes no action")
        return chain(args)
    B.require(args.action in STAGES, f"action must be one of {tuple(STAGES)}")
    if args.mode == "run":
        return run(args)
    B.require(bool(args.name) and (args.resume is None) == (args.prior is None) and (args.action == "train" or args.resume is None),
              "stage: --name; --resume / --prior for train only")
    return stage_main(args)


if __name__ == "__main__":
    main()
