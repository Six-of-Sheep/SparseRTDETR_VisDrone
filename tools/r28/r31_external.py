"""R31-001 (TASK_R31_EXTERNAL_UAVDT_001): the frozen L0 / F_geo e120 EMA endpoints of seeds 0, 1, 2 on the UAVDT-M
test set.  Inference and scoring only: no update, no fine-tuning, no BN re-estimation, no demand-network training, no
score threshold / NMS / TTA, no gate.  The S15 endpoints of the three seeds are the reference arm of task section 6
(recorded, not part of the label).

usage: r31_external.py <action> --round <round> [--release <name>] [--unit <unit>]

  prepare   contract of the round: the release (git archive of a commit; only the four R31 files may differ from the
            R30 release the branch starts from), the nine pinned endpoints with their stored development predictions,
            the three UAVDT packages, the frozen demand network and the mapping of task 009.
  data      CPU.  Package identity, the image package directory against the selection evidence, the 20 test sequences
            unpacked (every image asserted 1024 x 540), GT / ignore / attribute parsing against the selection census,
            the ignore rule on the toolkit's own det_FRCNN results (task check 4).
  gtcheck   CPU.  Task check 2, first half: the GT boxes as detections through the scoring path.
  wire      CUDA, one unit.  Task check 1: the first 8 development images through this entry must give the stored
            top600 records field by field; only then does the chain run that unit on UAVDT.
  infer     CUDA, one unit.  Every selected UAVDT frame, batches of 4, FP32; the raw 300 x (4 + 10) outputs are kept.
  geometry  CPU.  The realised magnification of the F_geo map per GT (from the demand maps the F_geo inference itself
            used), the development column, box sizes on canvas L, the map round trip of task check 2 (second half).
  score     CPU, one unit.  Vehicle candidates, the ignore rule, the single-class COCOeval and its image subsets, the
            three-class side table, the best-IoU coverage of the small GT.
  dhash     CPU.  64-bit dHash of every 10th UAVDT test frame against the VisDrone official train and val images.
  tables    CPU.  Every table of the report from the closed stages; no model, no scorer.

Per-unit stages write <round>/<action>-<unit>/, the others <round>/<action>/; every stage ends with CLOSED.json
(PASS) or STOP.json.  A dev round (a directory dev-round* inside the round; R31_DEV_LIMIT frames per sequence, fixed
at prepare) is an engineering shake-out of the chain: its numbers are not results.

The image path is the development evaluation's: infer_seed.load_original and infer_seed.pixels (the original image
resized directly to 1152 x 2016), torch.stack, FP32 with autocast disabled under forward_guards, inside
torch.inference_mode.  The only difference is that the images of the next batches are decoded ahead by worker threads
(the same two functions per image); the wire stage proves the records equal.
"""
import argparse
import gc
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import types
import zipfile
from collections import Counter, deque
from pathlib import Path

os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "tools" / "r27_l0")]

import numpy as np  # noqa: E402

import l0_common as B  # noqa: E402
import r31_core as C  # noqa: E402

VERSION = "R31-001-external-uavdt-v1"
TASK = ("TASK_R31_EXTERNAL_UAVDT_001.md", "b9216d72d07e0640ec9fcebeb7e3aa8ebf25c4884031593bc5f2c6a686673e74")
INPUTS = {"uavdt_census.json": "1652c839482d0b4e41838bf11a9d6d8113672192ceec72bc554f82266e17ee03",
          "uavdt_image_zip_directory.json": "b2eb86b15c72743ef6861211681af3c871d89c824506f55fa868a70d17bfc2e7"}
BASE_COMMIT = "fa0a381ba9e106a0381590225c48277f796b9c26"           # R30 release (round r30-fgeo-seeds-20261118-001)
BASE_MANIFEST = ("r30-fgeo-seeds-20261118-001/release-001/MANIFEST.json",
                 "c8f7cf6265c850bcab4578fe324a07eb0eaa777fe694d9be464fe555a0ba4809")
ALLOWED_ADDED = ("tools/r28/r31_core.py", "tools/r28/r31_external.py", "tools/r28/r31_chain.sh", "tests/test_r31_external.py")
R30_CONTRACT = ("r30-fgeo-seeds-20261118-001/CONTRACT.json",
                "d03d07eaee874f5c326d3c2c6b9dadb93431cb01addbb88fabb270b5ceb5f717",
                "2afdab274e04e5a8e852089cbce58334cba7cd7e4c0e27932b71d5c82139bcfe")
DATA_ROOT = Path("/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_VisDrone/artifacts/data/external/uavdt-m")
# package -> (path under DATA_ROOT, bytes, SHA-256).  The image package SHA is registered by this task after the
# download (the task book fixes its Google Drive id, size and file count); the other two are the task book's.
PACKAGES = {
    "images": ("download/UAV-benchmark-M.zip", 6790452113, "e82675c938e4e6bd65b4533dba2fb6a028e3e60bd3577cc2f3031504bd585b08"),
    "toolkit": ("download/UAV-benchmark-MOTD_v1.0.zip", 245719325, "1e565da8c2a035bf1e56b1322de76c74b73702ea3b0f918f45debf69c2b26799"),
    "attributes": ("download/attributes.zip", 9503, "ac6f85e355db3f4808cd64148b1f3c33933906ebd47c63939bb7eba6974824d1"),
}
IMAGE_PACKAGE_SOURCE = {"google_drive_id": "1m8KA6oPIRK_Iwt9TYFquC87vBc_8wRVc", "member_count_jpg": 40735}
VISDRONE_CONVERSION = Path("/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_VisDrone/artifacts/data/visdrone_protocol_v2_conversion_r3")
VISDRONE_MANIFESTS = (
    ("raw_val_manifest.jsonl", 548, "d28e87e61812159eb914befd181529a915ed4120d2b70f2be8d76700f01583c1"),
    ("raw_train_manifest.part-00001.jsonl", 1729, "5215844b8e950a83040e4364c977ba059ae385b46ec0a1d9aa0e743fdc8a98e0"),
    ("raw_train_manifest.part-00002.jsonl", 1728, "58e3b7b7963d21415822f21e288e6d586e1fdab33676996b704e7ecdca0ead69"),
    ("raw_train_manifest.part-00003.jsonl", 1726, "6e3da6466e585c64fe8c9c97f99c9a916890a11f27d774c8aaea6f05fc03be10"),
    ("raw_train_manifest.part-00004.jsonl", 1288, "dfbe60ca529c1a673d8da5ec43fdf5e5bf577082b40d642332932d958a92b215"),
)
VISDRONE_IMAGES = {"train": "/media/lyy/Data/JupyterLab/LiuZhiyang/p3_data_roots/train_core/VisDrone2019-DET-train/images",
                   "val": "/media/lyy/Data/JupyterLab/LiuZhiyang/p3_data_roots/development/VisDrone2019-DET-val/images"}
# R30-001 (closed diagnostic round): per-GT magnification of the F_geo map on the development set, predicted demand
DEVELOPMENT_PER_GT = ("r30-fgeo-diagnostic-20261118-001/infer/per_gt.json", 13270127,
                      "a28113d2dd614f253b236cbd91b964b669d3f90fbe82f488824e2c082adb606c")
DEVELOPMENT_READING = {"GT": 38599, "small_GT": 26441, "small_bin_counts": [3143, 9414, 7840, 6044], "small_g_geomean_4dp": 1.2172}
DEVELOPMENT_VEHICLE_CATEGORIES = (4, 5, 6, 9)                      # VisDrone car, van, truck, bus

MAIN_UNITS = ("L0-s0", "FG-s0", "L0-s1", "FG-s1", "L0-s2", "FG-s2")
REFERENCE_UNITS = ("S15-s0", "S15-s1", "S15-s2")
UNITS = MAIN_UNITS + REFERENCE_UNITS
SEEDS = (0, 1, 2)
ENDPOINT = 36480
BATCH = 4
WIRE_IMAGES = 8
ROUND_TRIP_FRAMES = 200
DECODE_THREADS, DECODE_AHEAD = 8, 6
MAPPING = {"fwhm": 19.03125, "fwhm_hex": "0x1.3080000000000p+4", "grid_shape": [31, 51], "padding_size": 30, "eps": 0.2}
BIN_METRICS = ("AP_lt16", "AP75_lt16", "AP_16to32", "AP75_16to32")
IMAGE_MEMBER = re.compile(r"UAV-benchmark-M/(M\d{4})/img(\d{6})\.jpg")
PIN = {
    "L0-s0": {"receipt": ("r27-l0-control-20261111-001/eval-L0/CLOSED.json",
                          "82f93b4eed1d7232df1003d7844b90b993ffc3b78dcc2ecb3904349f28095ffe"),
              "endpoint": ("r27-l0-control-20261111-001/formal-L0/checkpoint-epoch-120.pth", 323123009,
                           "966dac4afe28bdd154cd1b8d5054719a541c3ad02d57258e9636d2d5454c1701"),
              "weights": "429dc1e6fb881a9ce4cfd14c5913431b75bbcb75cf82d00fd1fd115d23a86dc2",
              "predictions": ("r27-l0-control-20261111-001/eval-L0/top600-predictions.json", 87892832,
                              "25fcf2fd2ca30e866c623a4aa26c7dcca3207a9e16e9583ea28905ac91b3c760")},
    "FG-s0": {"receipt": ("r28-strong-fovea-control-20261116-001/eval-FG/CLOSED.json",
                          "9f4dfead07da37cb7520e5978c9c2b32067af7b25df352e2f224a231cf48052d"),
              "endpoint": ("r28-strong-fovea-control-20261116-001/formal-FG/checkpoint-epoch-120.pth", 323123009,
                           "f0632fc8ed71e593e5eaab1bf233aafed1e9ec57dcfe44c893721286cdffeb57"),
              "weights": "cd62f314f3713d83fd742ff511600902fd990cf48165d9b0a90c7fec4359e219",
              "predictions": ("r28-strong-fovea-control-20261116-001/eval-FG/top600-predictions.json", 87894853,
                              "02e555afce410321c0f1ee6414787a75082a276042eaa9326e439ff2961e31ba")},
    "L0-s1": {"receipt": ("r30-fgeo-seeds-20261118-001/eval-L0-s1/CLOSED.json",
                          "c9b130458943b960d79fa0a7ec2b0bc0e7653c8a3ad33fe0ccf1137432f15898"),
              "endpoint": ("r30-fgeo-seeds-20261118-001/formal-L0-s1-r2/checkpoint-epoch-120.pth", 323123073,
                           "e5df408273bd9254f469fc1cb2ba62a05cde0ab06a72e534409ee0039209037c"),
              "weights": "f9f5218110eb71b45f99fc1142c50728ce36fdb1be41c895921543d009f9d870",
              "predictions": ("r30-fgeo-seeds-20261118-001/eval-L0-s1/top600-predictions.json", 87888074,
                              "466e253d099a40b5b6b5d0bf39bc172b7362e06d31f3955ceb322bc395745b45")},
    "FG-s1": {"receipt": ("r30-fgeo-seeds-20261118-001/eval-FG-s1/CLOSED.json",
                          "939608f2b373f7dbbc4514573b5e6ee93c72378b37cfbce748a37d54dfdc18dd"),
              "endpoint": ("r30-fgeo-seeds-20261118-001/formal-FG-s1/checkpoint-epoch-120.pth", 323123073,
                           "7dc9bdb2f4bbc93c831fcb5f76abba0689aef19d48ad3be136393a8a2a3747c6"),
              "weights": "7d2f2a0c2ce57ccf116a3577d13a53ccbb0b1fa7b4393c28b71d47cd353d64ea",
              "predictions": ("r30-fgeo-seeds-20261118-001/eval-FG-s1/top600-predictions.json", 87892656,
                              "b0dc80249a3eb547231a32c6faf9c3fa68221d5844220cb1a64fe1b3e277c242")},
    "L0-s2": {"receipt": ("r30-fgeo-seeds-20261118-001/eval-L0-s2/CLOSED.json",
                          "67e5ae972095ff41538f4f825fb515dc9c1840e98856b4acf54cd1184aaee378"),
              "endpoint": ("r30-fgeo-seeds-20261118-001/formal-L0-s2/checkpoint-epoch-120.pth", 323123073,
                           "670871fcb11f4dc042afde371d65d145e350eb2af2225dbcf17b702e2d55c352"),
              "weights": "d3b38979ae7336f9c3e9bafdc3093accf94bb6a58236e77cd99305d682e318f6",
              "predictions": ("r30-fgeo-seeds-20261118-001/eval-L0-s2/top600-predictions.json", 87888655,
                              "25a49e19d4f4410ce83c557fb92a363417314beabd46b52a9604196933c928e9")},
    "FG-s2": {"receipt": ("r30-fgeo-seeds-20261118-001/eval-FG-s2/CLOSED.json",
                          "e39ee5644d6e237d39b2a8054e1c6f6ea3872a91782f1bdcbb70bc0d2687fdbb"),
              "endpoint": ("r30-fgeo-seeds-20261118-001/formal-FG-s2/checkpoint-epoch-120.pth", 323123073,
                           "30e0db2c398a4be96d5cc8e027196ab8fbffea168b7e2a158d1ae8e617242648"),
              "weights": "3efecbf08453995f0f0dae0db1d08d67e57e64f440ccb229bf6fb5735588ae04",
              "predictions": ("r30-fgeo-seeds-20261118-001/eval-FG-s2/top600-predictions.json", 87894598,
                              "e173a599039423e00e0b9aaff6128d0974c84478c0a3d88bacbdbe4c94777d5d")},
    "S15-s0": {"receipt": ("s15-scale-baseline-20261105-001/formal-002/evaluation/S15-EMA/RESULT.json",
                           "3302dbd2424f8e014969d8aad572fc2b0c81f9a01b780f25b0d0f7bf1ddfdaaf"),
               "endpoint": ("s15-scale-baseline-20261105-001/formal-002/train-S15/checkpoint-epoch-120.pth", 324022785,
                            "656f7be2357f20dce50a718f1093a639c1a32a7faad69d03184e36e43f32871f"),
               "weights": "072616ff442dec05693f8f4f98974213f59a35b72edfbf4a1fb87bb7fd49ac68",
               "predictions": ("s15-scale-baseline-20261105-001/formal-002/evaluation/S15-EMA/top600-predictions.json", 96433942,
                               "f8e89847c4ebcc67f8ed08b8f794099f7459b9051b096517656afcb6b6af0ac5")},
    "S15-s1": {"receipt": ("s15-seed12-replication-20261106-001/formal-001/seed1/evaluation/S15-EMA/RESULT.json",
                           "55bf1a8ea9e0e764a34f613264cb6c5ab599f71d06830db242535c4d56f599f5"),
               "endpoint": ("s15-seed12-replication-20261106-001/formal-001/seed1/train-S15/checkpoint-epoch-120.pth", 324022785,
                            "5b8b6e8b68c6450586313983651f716b9e9af7e0c30d40d0e881c958b7c78196"),
               "weights": "0dc59a9f7f2bedd1e1fbbc9ccadd0cd3b1e55250cac2c7c46729ad36085471cc",
               "predictions": ("s15-seed12-replication-20261106-001/formal-001/seed1/evaluation/S15-EMA/top600-predictions.json", 96437040,
                               "87f15dbdce49f28b3444ea1ae2310af4dfe4832f71067d012abe3a92eaa0a0dd")},
    "S15-s2": {"receipt": ("s15-seed12-replication-20261106-001/formal-001/seed2/evaluation/S15-EMA/RESULT.json",
                           "4fa09e52f1913457924ab540974c0424530f4ce27846a44ad19c4e1a53c7e889"),
               "endpoint": ("s15-seed12-replication-20261106-001/formal-001/seed2/train-S15/checkpoint-epoch-120.pth", 324022785,
                            "d5a3afbf4c26eed4d18d0dee5f88d92d1ad58d596530923c3b98b51443fc7936"),
               "weights": "e324442a732e3608c36702a9ad0ab18855ec8d0eb6fb3d643b739af7124c9022",
               "predictions": ("s15-seed12-replication-20261106-001/formal-001/seed2/evaluation/S15-EMA/top600-predictions.json", 96435494,
                               "2baf949ae8a25d6aabc093a0cb0c1bb128977bca5baa5400c7ea89c2d8f30ca9")},
}
PROTOCOL = {
    "data": "UAVDT-Benchmark-M official test split: 20 sequences, every frame (16,592; the frames without GT are inferred "
            "and evaluated), GT = every row of <seq>_gt_whole.txt (occlusion / out-of-view not used), no subsampling",
    "input": "original 1024 x 540 image resized directly to 1152 x 2016 (canvas H; no padding, no aspect preservation, "
             "no augmentation), then the endpoint's own frozen pipeline; FP32, batches of 4",
    "candidates": "300 queries per frame, one vehicle candidate per query: the query's box, score = the largest sigmoid "
                  "score of car, van, truck, bus; all 300 submitted in score order (ties by query index); no threshold, "
                  "no NMS, no TTA, no box clipping; the other six classes are not used; one budget (300 per frame)",
    "ignore": "CalculateDetectionPR_overall.m of the UAVDT toolkit: a candidate whose four edges lie strictly inside one "
              "ignore rectangle of its frame is deleted; nothing is refilled; GT is never deleted",
    "metrics": "pycocotools COCOeval, one category (vehicle), IoU 0.50:0.05:0.95, 101 recall points, maxDets [1, 10, 100] "
               "(the 100 column is read), area = annotation width x height with the bounds 32^2 and 96^2 (COCOeval's "
               "closed end points), predictions in original image coordinates",
    "main": list(C.MAIN), "extra": list(C.EXTRA),
    "labels": {"external_gain_repeated_3of3": "every seed: APS >= +1.0 pp, AP75-small >= +1.0 pp and AP >= 0 (F_geo - L0)",
               "external_gain_reduced": "the three APS differences are positive, the first label is not met",
               "external_gain_reversed": "the three APS differences are negative",
               "external_gain_mixed": "every other case"},
    "gate": None,
}
NOT_AUTHORISED = [
    "any training, fine-tuning, BN re-estimation or demand-network retraining",
    "any endpoint of the arms W, K, F, P, C; intermediate-epoch endpoints",
    "score threshold, NMS, TTA, multi-scale or sliced inference",
    "VisDrone test-dev (sealed); AU-AIR or another dataset; a second canvas budget point",
    "class / area / frame / sequence selections added after a result is seen; timing",
    "merging pull requests; cleaning old rounds",
]
RESOLUTION_CAVEAT = ("UAVDT 原图 1024×540 的像素数少于画布 L；均匀臂已是上采样。本任务检验的是冻结端点的增益是否带到另一个"
                     "数据集，不是画布不足时的信息损失，结论不外推到后者。")


def parse(unit):
    B.require(unit in UNITS, f"unit: {unit}")
    arm, seed = unit.rsplit("-s", 1)
    return arm, int(seed)


def is_dev(round_dir):
    return Path(round_dir).resolve().name.startswith("dev-round")


def ld(rel):
    return B.LD_ROOT / rel


def nvidia(query):
    p = subprocess.run(["nvidia-smi", f"--query-{query}", "--format=csv,noheader"], capture_output=True, text=True)
    B.require(p.returncode == 0, f"nvidia-smi failed: {p.stderr[:300]}")
    return [line.strip() for line in p.stdout.splitlines() if line.strip()]


def other_gpu_processes():
    """Compute processes on the GPU that are not this process (pid, name, memory); nothing is ever stopped."""
    me = f"{os.getpid()},"
    return [line for line in nvidia("compute-apps=pid,process_name,used_memory") if not line.startswith(me)]


def link_new(tmp, path):
    os.link(tmp, path)
    Path(tmp).unlink()


def save_npy(path, array):
    tmp = Path(str(path) + ".partial")
    with tmp.open("xb") as f:
        np.save(f, array, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    link_new(tmp, path)
    return B.ref(path)


def checked_ref(path, size, sha, label):
    r = B.ref(path)
    B.require(r["bytes"] == size and r["sha256"] == sha, f"{label} differs from its registered identity: {r}")
    return r


# ================================================================ contract
def prepare(round_dir, release):
    R = Path(round_dir).resolve()
    dev = is_dev(R)
    B.require(not (R / "CONTRACT.json").exists(), "CONTRACT.json exists")
    B.require((R / release / "source").resolve() == ROOT.resolve(), "prepare must run from <round>/<release>/source")
    home = R.parent if dev else R
    task = home / TASK[0]
    B.require(task.is_file() and B.file_sha(task) == TASK[1], f"task file missing or different: {task}")
    inputs = {}
    for name, sha in INPUTS.items():
        inputs[name] = B.ref(home / "inputs" / name)
        B.require(inputs[name]["sha256"] == sha, f"selection evidence missing or different: {name}")
    limit = os.environ.get("R31_DEV_LIMIT")
    B.require((limit is not None) == dev, "R31_DEV_LIMIT is required in a dev round and refused in a formal round")
    limit = None if limit is None else int(limit)
    B.require(limit is None or (limit > 0 and limit % BATCH == 0 and limit <= min(C.SEQUENCE_FRAMES)), "R31_DEV_LIMIT: a positive multiple of 4")
    man_ref = B.ref(R / release / "MANIFEST.json")
    manifest = B.read_json(man_ref["path"])
    B.require(manifest["dirty"] is False or dev, "a formal release must come from a clean committed tree")
    B.release_intact(ROOT, manifest["files"])
    base_ref = B.ref(ld(BASE_MANIFEST[0]))
    base = B.read_json(base_ref["path"])
    B.require(base_ref["sha256"] == BASE_MANIFEST[1] and base["commit"] == BASE_COMMIT, "base release (R30) identity")
    new, old = manifest["files"], base["files"]
    delta = {"added": sorted(set(new) - set(old)), "removed": sorted(set(old) - set(new)),
             "changed": sorted(k for k in new if k in old and new[k] != old[k])}
    B.require(set(delta["added"]) == set(ALLOWED_ADDED) and not delta["removed"] and not delta["changed"],
              f"the release may differ from the R30 release only by the four added R31 files: {delta}")
    r30_ref = B.ref(ld(R30_CONTRACT[0]))
    r30 = B.read_json(r30_ref["path"])
    body = {k: v for k, v in r30.items() if k != "contract_sha256"}
    B.require(r30_ref["sha256"] == R30_CONTRACT[1] and r30["contract_sha256"] == B.sha_bytes(B.canonical(body)) == R30_CONTRACT[2]
              and r30["source_commit"] == BASE_COMMIT, "R30 contract identity")
    B.require(os.path.samefile(sys.executable, r30["python"]), f"python differs from the R30 round: {sys.executable}")
    units = {}
    for unit in UNITS:
        arm, seed = parse(unit)
        pin = PIN[unit]
        rec_ref = B.ref(ld(pin["receipt"][0]))
        B.require(rec_ref["sha256"] == pin["receipt"][1], f"{unit}: stored evaluation receipt file")
        rec = B.read_json(rec_ref["path"])
        endpoint = checked_ref(ld(pin["endpoint"][0]), pin["endpoint"][1], pin["endpoint"][2], f"{unit} endpoint")
        stored = checked_ref(ld(pin["predictions"][0]), pin["predictions"][1], pin["predictions"][2], f"{unit} stored top600 predictions")
        ep, pr = rec.get("endpoint") or rec.get("checkpoint"), rec["predictions"]["600"]
        B.require(rec["status"] == "PASS" and rec["arm"] == arm and rec["kind"] == "EMA" and rec["epoch"] == 120 and rec["view"] == "A"
                  and rec.get("seed", 0) == seed and rec["weights_sha256"] == pin["weights"]
                  and {k: ep[k] for k in endpoint} == endpoint and {k: pr[k] for k in stored} == stored,
                  f"{unit}: the stored evaluation receipt does not name this e120 EMA endpoint and these predictions")
        B.require(arm != "FG" or rec["demand"] == r30["demand"]["binding"], f"{unit}: the stored evaluation used another demand source")
        units[unit] = {"arm": arm, "seed": seed, "kind": "EMA", "epoch": 120, "receipt": rec_ref, "endpoint": endpoint,
                       "side": B.ref(B.side_of(endpoint["path"])), "weights_sha256": pin["weights"], "predictions_600": stored}
    B.pinned(r30["demand"]["weights"])
    B.require(r30["demand"]["retrained"] is False and r30["width_selection"]["fwhm_hex"] == MAPPING["fwhm_hex"], "demand network / width of task 009")
    packages = {name: checked_ref(DATA_ROOT / rel, size, sha, f"UAVDT package {name}") for name, (rel, size, sha) in PACKAGES.items()}
    visdrone = {"manifests": [checked_ref(VISDRONE_CONVERSION / name, (VISDRONE_CONVERSION / name).stat().st_size, sha, name)
                              for name, _, sha in VISDRONE_MANIFESTS],
                "rows": [n for _, n, _ in VISDRONE_MANIFESTS], "images": dict(VISDRONE_IMAGES)}
    per_gt = checked_ref(ld(DEVELOPMENT_PER_GT[0]), DEVELOPMENT_PER_GT[1], DEVELOPMENT_PER_GT[2], "R30-001 per-GT geometry")
    frames = C.FRAMES if limit is None else limit * len(C.TEST_SEQUENCES)
    per_unit = {"wire_forwards": WIRE_IMAGES // BATCH, "uavdt_forwards": frames // BATCH, "updates": 0}
    c = {"version": VERSION, "task": {"file": TASK[0], "sha256": TASK[1]}, "record_only": True, "gate": None,
         "source_root": str(ROOT), "source_commit": manifest["commit"], "source_branch": manifest["branch"],
         "source_dirty": manifest["dirty"], "source_manifest": man_ref, "base_commit": BASE_COMMIT, "base_manifest": base_ref,
         "source_delta": delta, "outputs": str(R), "dev": dev, "dev_limit": limit, "inputs": inputs,
         "R30_contract": {**r30_ref, "contract_sha256": r30["contract_sha256"]},
         "units": units, "main_units": list(MAIN_UNITS), "reference_units": list(REFERENCE_UNITS),
         "demand": r30["demand"], "width_selection": r30["width_selection"], "mapping": MAPPING,
         "data": {"development": r30["data"]["development"]}, "development_ids": r30["development_ids"],
         "gpu_uuid": r30["gpu_uuid"], "environment": r30["environment"], "python": sys.executable,
         "uavdt": {"root": str(DATA_ROOT), "packages": packages, "image_package_source": IMAGE_PACKAGE_SOURCE,
                   "images": str(DATA_ROOT / "test-images"), "sequences": list(C.TEST_SEQUENCES),
                   "sequence_frames": list(C.SEQUENCE_FRAMES), "frames": C.FRAMES, "image_wh": list(C.IMAGE_WH)},
         "visdrone": visdrone, "development_geometry": {"per_gt": per_gt, "reading": DEVELOPMENT_READING},
         "protocol": PROTOCOL, "rule_pp": [list(r) for r in C.RULE], "resolution_caveat": RESOLUTION_CAVEAT,
         "workload": {"frames": frames, "batch": BATCH, "per_unit": per_unit,
                      "main_units_uavdt_forwards": len(MAIN_UNITS) * per_unit["uavdt_forwards"],
                      "reference_units_uavdt_forwards": len(REFERENCE_UNITS) * per_unit["uavdt_forwards"],
                      "wire_forwards": len(UNITS) * per_unit["wire_forwards"], "updates": 0,
                      "reference_arm": "task section 6 (added by the controller, strikable by the user): run after the six "
                                       "main units, recorded only, not part of the label"},
         "resource_plan": {
             "storage_estimate_bytes": {"image_package": PACKAGES["images"][1], "unpacked_test_images": 3_100_000_000,
                                        "raw_outputs_per_unit": frames * C.QUERIES * (4 + C.CLASSES) * 4,
                                        "raw_outputs_all_units": len(UNITS) * frames * C.QUERIES * (4 + C.CLASSES) * 4,
                                        "demand_maps_three_units": 3 * frames * 12 * 21 * 8, "tables_logs_and_small": 100_000_000},
             "disk_free_bytes_at_prepare": shutil.disk_usage(R).free,
             "gpu_time_reference": "task: about 1-2 h planning reference (not a cap or a promise)",
             "kept_on_server": "images, raw outputs; only the small JSON files are returned"},
         "not_authorised": NOT_AUTHORISED}
    c["contract_sha256"] = B.sha_bytes(B.canonical(c))
    B.write_new(R / "CONTRACT.json", c)
    print(json.dumps({"contract_sha256": c["contract_sha256"], "contract_file": B.ref(R / "CONTRACT.json"),
                      "commit": c["source_commit"], "delta": delta, "dev_limit": limit}))


def load_contract(round_dir):
    R = Path(round_dir).resolve()
    c = B.read_json(R / "CONTRACT.json")
    body = {k: v for k, v in c.items() if k != "contract_sha256"}
    B.require(c["contract_sha256"] == B.sha_bytes(B.canonical(body)) and c["version"] == VERSION, "contract digest")
    B.require(Path(c["source_root"]).resolve() == ROOT.resolve(), "not running from the bound release")
    B.require(Path(c["outputs"]).resolve() == R and c["dev"] == is_dev(R) and (c["dev_limit"] is not None) == c["dev"],
              "contract bound to another round")
    manifest = B.read_json(B.pinned(c["source_manifest"]))
    B.require(manifest["commit"] == c["source_commit"], "MANIFEST commit differs from the contract")
    B.release_intact(ROOT, manifest["files"])
    B.require(c["task"]["sha256"] == TASK[1] and sorted(c["units"]) == sorted(UNITS)
              and all(c["units"][u]["weights_sha256"] == PIN[u]["weights"] and c["units"][u]["endpoint"]["sha256"] == PIN[u]["endpoint"][2]
                      for u in UNITS)
              and all(c["uavdt"]["packages"][k]["sha256"] == v[2] for k, v in PACKAGES.items()) and c["mapping"] == MAPPING,
              "pins differ from the release")
    return c


def stage_closed(c, name):
    path = Path(c["outputs"]) / name / "CLOSED.json"
    B.require(path.is_file(), f"stage not closed: {path}")
    closed = B.read_json(path)
    B.require(closed.get("status") == "PASS" and closed.get("contract_sha256") == c["contract_sha256"],
              f"stage {name} is not PASS under this contract")
    return closed


# ================================================================ data
def _offsets():
    return np.concatenate([[0], np.cumsum(C.SEQUENCE_FRAMES)]).astype(np.int64)


def _pack(tables, label, drop_outside):
    """{sequence: (n, k) int rows with the frame number in column 0} -> (global frame index, rows, offsets), sorted
    stably by frame.  Rows of a frame number outside the sequence are an error for GT and are dropped (and counted)
    for ignore rectangles."""
    off = _offsets()
    frame, rows, dropped = [], [], {}
    for k, s in enumerate(C.TEST_SEQUENCES):
        t = tables[s]
        ok = (t[:, 0] >= 1) & (t[:, 0] <= C.SEQUENCE_FRAMES[k])
        if not bool(ok.all()):
            B.require(drop_outside, f"{s}: {label} rows outside frames 1..{C.SEQUENCE_FRAMES[k]}: {int((~ok).sum())}")
            dropped[s] = int((~ok).sum())
            t = t[ok]
        frame.append(off[k] + t[:, 0] - 1)
        rows.append(t)
    frame, rows = np.concatenate(frame), np.concatenate(rows)
    order = np.argsort(frame, kind="stable")
    frame, rows = frame[order], rows[order]
    return frame, rows, np.searchsorted(frame, np.arange(C.FRAMES + 1)).astype(np.int64), dropped


def data(c, out, counts):
    from PIL import Image
    B.PHASE["phase"] = "data/packages"
    u = c["uavdt"]
    zips = {k: B.pinned(v) for k, v in u["packages"].items()}
    evidence = B.read_json(B.pinned(c["inputs"]["uavdt_census.json"]))
    directory = B.read_json(B.pinned(c["inputs"]["uavdt_image_zip_directory.json"]))
    B.require(evidence["toolkit_sha256"] == u["packages"]["toolkit"]["sha256"] and evidence["attributes_sha256"] == u["packages"]["attributes"]["sha256"]
              and tuple(evidence["test_sequences"]) == C.TEST_SEQUENCES and directory["zip_bytes"] == u["packages"]["images"]["bytes"],
              "the selection evidence was taken from other packages")

    B.PHASE["phase"] = "data/images"
    root = Path(u["images"])
    root.mkdir(parents=True, exist_ok=True)
    frames = []
    with zipfile.ZipFile(zips["images"]) as z:
        infos = z.infolist()
        files = [i for i in infos if not i.is_dir()]
        members = {}
        for i in files:
            m = IMAGE_MEMBER.fullmatch(i.filename)
            B.require(m is not None, f"unexpected member of the image package: {i.filename!r}")
            B.require(int(m.group(2)) not in members.setdefault(m.group(1), {}), f"duplicate member: {i.filename!r}")
            members[m.group(1)][int(m.group(2))] = i
        listing = {"entries": len(infos), "files": len(files), "uncompressed_bytes": sum(i.file_size for i in files),
                   "per_sequence_files": {s: len(v) for s, v in members.items()},
                   "per_sequence_index_range": {s: [min(v), max(v)] for s, v in members.items()}}
        differ = {k: [listing[k], directory[k]] for k in listing if listing[k] != directory[k]}
        B.require(not differ and len(files) == IMAGE_PACKAGE_SOURCE["member_count_jpg"],
                  f"the image package directory differs from the selection evidence: { {k: str(v)[:300] for k, v in differ.items()} }")
        for s, n in zip(C.TEST_SEQUENCES, C.SEQUENCE_FRAMES):
            B.require(sorted(members[s]) == list(range(1, n + 1)), f"{s}: the frame numbers are not 1..{n}")
            d = root / s
            d.mkdir(exist_ok=True)
            for f in range(1, n + 1):
                raw = z.read(members[s][f])                       # CRC checked by zipfile
                name = "img%06d.jpg" % f
                p = d / name
                if p.exists() or p.is_symlink():
                    B.require(p.is_file() and not p.is_symlink() and p.read_bytes() == raw, f"an unpacked image differs from its member: {p}")
                    counts.update(images_already_unpacked=1)
                else:
                    with p.open("xb") as fh:
                        fh.write(raw)
                    counts.update(images_unpacked=1)
                with Image.open(io.BytesIO(raw)) as im:
                    size, kind = im.size, im.format
                B.require(size == C.IMAGE_WH and kind == "JPEG",
                          f"STOP AND REPORT: {s}/{name} is {size} {kind}, the task fixes 1024 x 540 JPEG")
                frames.append({"sequence": s, "frame": f, "relative_path": f"{s}/{name}", "image_size_bytes": len(raw),
                               "image_sha256": hashlib.sha256(raw).hexdigest(), "width": size[0], "height": size[1]})
            B.require(sorted(x.name for x in d.iterdir()) == ["img%06d.jpg" % f for f in range(1, n + 1)], f"{s}: other files in {d}")
    B.require(sorted(x.name for x in root.iterdir()) == sorted(C.TEST_SEQUENCES) and len(frames) == C.FRAMES, f"other entries in {root}")
    B.write_new(out / "frames.json", frames)

    B.PHASE["phase"] = "data/toolkit"
    with zipfile.ZipFile(zips["toolkit"]) as z:
        names = z.namelist()

        def member(suffix):
            hits = [n for n in names if n.endswith(suffix)]
            B.require(len(hits) == 1, f"toolkit member {suffix}: {hits}")
            return z.read(hits[0])

        gt = {s: C.parse_gt(member(f"/GT/{s}_gt_whole.txt").decode("ascii")) for s in C.TEST_SEQUENCES}
        ign = {s: C.parse_ignore(member(f"/GT/{s}_gt_ignore.txt").decode("ascii")) for s in C.TEST_SEQUENCES}
        det = {s: C.parse_detections(member(f"/RES_DET/det_FRCNN/{s}.txt").decode("ascii")) for s in ("M0203", "M0701")}
        rule_source = member("/utils/CalculateDetectionPR_overall.m")
    mine = C.census(gt, ign)
    want = evidence["splits"]["test"]
    pairs = {"gt": (mine["gt"], want["gt"]), "area<16^2": (mine["lt16"], want["bin"]["lt16"]),
             "16^2<=area<32^2": (mine["16to32"], want["bin"]["16to32"]), "32^2<=area<96^2": (mine["32to96"], want["bin"]["32to96"]),
             "area>=96^2": (mine["ge96"], want["bin"]["ge96"]), "small": (mine["small"], want["small_total"]),
             "categories": (mine["cat"], want["cat"]), "ignore_boxes": (mine["ignore_boxes"], want["ignore_boxes"]),
             "frames_with_gt": (mine["frames_with_gt"], want["frames_with_gt"]),
             "gt_center_in_ignore": (mine["gt_center_in_ignore"], want["gt_center_in_ignore"]),
             "gt_center_in_ignore_small": (mine["gt_center_in_ignore_small"], want["gt_center_in_ignore_small"]),
             "max_gt_per_frame": (mine["max_gt_per_frame"], want["per_frame_gt"]["max"]),
             "frames_gt_over_100": (mine["frames_gt_over_100"], want["per_frame_gt"]["frames_gt_over_100"]),
             **{k: (mine[k], want[k]) for k in ("max_x2", "max_y2", "min_x", "min_y")}}
    for s in C.TEST_SEQUENCES:
        for k, v in mine["per_sequence"][s].items():
            pairs[f"{s}.{k}"] = (v, evidence["per_sequence"][s][k])
    differ = {k: list(v) for k, v in pairs.items() if v[0] != v[1]}
    B.require(not differ, f"parsed counts differ from the selection census (check the parsing; the definitions are not changed): {differ}")

    B.PHASE["phase"] = "data/attributes"
    attrs = {}
    with zipfile.ZipFile(zips["attributes"]) as z:
        for n in z.namelist():
            key = C.attribute_sequence(n)
            if key is not None and key[0] == "test":
                B.require(key[1] not in attrs, f"two attribute files for {key[1]}")
                attrs[key[1]] = C.parse_attributes(z.read(n).decode("ascii"))
    B.require(set(attrs) == set(C.TEST_SEQUENCES) and all(attrs[s] == evidence["per_sequence"][s]["attrs"] for s in C.TEST_SEQUENCES),
              "sequence attributes differ from the selection census")
    groups = C.sequence_groups(attrs)
    seq = mine["per_sequence"]
    alt = {k: {a: sum(seq[s][field] for s in groups[a]) for a in C.ALTITUDE}
           for k, field in (("alt_frames", "frames_with_gt"), ("alt_gt", "gt"), ("alt_small", "small"))}
    B.require(all(alt[k] == want[k] for k in alt), f"altitude groups differ from the selection census: {alt}")

    B.PHASE["phase"] = "data/ignore-rule"
    text = rule_source.decode("latin-1")
    lines = text.splitlines()
    at = [i for i, line in enumerate(lines) if "ignore the detections in the ignored regions" in line]
    passed = re.search(r"pass\s*=\s*\{([^}]*)\}", text)
    B.require(len(at) == 1 and passed is not None, "the toolkit rule was not found in CalculateDetectionPR_overall.m")
    pass_list = re.findall(r"M\d{4}", passed.group(1))
    outside = [s for s in C.TEST_SEQUENCES if s not in pass_list]
    B.require(all(len(ign[s]) == 0 for s in outside) and set(pass_list) <= set(C.TEST_SEQUENCES),
              f"STOP AND HAND BACK: the toolkit skips the rule on sequences that carry ignore rectangles: {outside}")
    rule = {}
    for s, d in det.items():
        mask = C.deleted_by_frame(d, ign[s].astype(np.float64))
        independent = np.array(C.toolkit_deleted(d.tolist(), ign[s].tolist()), dtype=bool)
        rule[s] = {"detections_before": int(len(d)), "deleted": int(mask.sum()), "detections_after": int((~mask).sum()),
                   "independent_deleted": int(independent.sum()), "independent_after": int((~independent).sum()),
                   "same_rows": bool((mask == independent).all()), "ignore_rectangles": int(len(ign[s]))}
    B.require(all(v["same_rows"] and v["detections_before"] > 0 for v in rule.values()),
              f"the ignore rule of the scoring path differs from the independent judgement: {rule}")

    B.PHASE["phase"] = "data/annotations"
    gt_frame, gt_rows, gt_off, _ = _pack(gt, "GT", False)
    ign_frame, ign_rows, ign_off, dropped = _pack(ign, "ignore", True)
    B.require(len(gt_rows) == mine["gt"] and len(ign_rows) + sum(dropped.values()) == mine["ignore_boxes"], "packed annotation counts")
    tmp = out / "annotations.partial.npz"
    np.savez(tmp, frame_seq=np.repeat(np.arange(len(C.TEST_SEQUENCES)), C.SEQUENCE_FRAMES).astype(np.int64),
             frame_no=np.concatenate([np.arange(1, n + 1) for n in C.SEQUENCE_FRAMES]).astype(np.int64),
             gt_frame=gt_frame, gt_xywh=gt_rows[:, 2:6], gt_cat=gt_rows[:, 8], gt_offsets=gt_off,
             ign_frame=ign_frame, ign_xywh=ign_rows[:, 1:5], ign_offsets=ign_off)
    link_new(tmp, out / "annotations.npz")
    return {"stage": "data", "packages": u["packages"], "image_package_directory": {"equal_to_selection_evidence": True, **{
                k: listing[k] for k in ("entries", "files", "uncompressed_bytes")}},
            "images": {"root": str(root), "sequences": len(C.TEST_SEQUENCES), "frames": len(frames), "every_image": "1024 x 540 JPEG",
                       "bytes": sum(f["image_size_bytes"] for f in frames)},
            "frames": B.ref(out / "frames.json"), "annotations": B.ref(out / "annotations.npz"),
            "census": {k: v for k, v in mine.items() if k != "per_sequence"}, "per_sequence": mine["per_sequence"],
            "census_equal_to_selection_evidence": True, "census_cells_compared": len(pairs),
            "attributes": attrs, "attribute_groups": groups, "altitude_census": alt,
            "ignore_rows_outside_frames": dropped,
            "ignore_rule": {"source": "utils/CalculateDetectionPR_overall.m", "source_sha256": hashlib.sha256(rule_source).hexdigest(),
                            "source_lines": {str(i + 1): lines[i].strip() for i in range(max(0, at[0] - 1), min(len(lines), at[0] + 7))},
                            "pass_list": pass_list, "test_sequences_not_in_pass_list": outside,
                            "their_ignore_rectangles": {s: int(len(ign[s])) for s in outside},
                            "note": "the toolkit applies the rule to the sequences of its pass list; the test sequences missing "
                                    "from it carry no ignore rectangle, so applying the rule to all 20 sequences is the same",
                            "det_FRCNN": rule}}


def load_data(c):
    closed = stage_closed(c, "data")
    frames = B.read_json(B.pinned(closed["frames"]))
    with np.load(B.pinned(closed["annotations"]), allow_pickle=False) as z:
        ann = {k: z[k] for k in z.files}
    B.require(len(frames) == C.FRAMES == len(ann["frame_seq"]) and len(ann["gt_offsets"]) == C.FRAMES + 1, "frame table")
    return closed, frames, ann


def selection(c):
    """Global indices of the frames this round handles: every test frame, or the first dev_limit frames of every
    sequence in a dev round."""
    if c["dev_limit"] is None:
        return np.arange(C.FRAMES, dtype=np.int64)
    off = _offsets()
    return np.concatenate([np.arange(o, o + min(c["dev_limit"], n), dtype=np.int64) for o, n in zip(off, C.SEQUENCE_FRAMES)])


def selected_rows(offsets, sel):
    """Row indices of the annotations of the selected frames and, per row, the position of its frame in `sel`."""
    idx = [np.arange(offsets[f], offsets[f + 1], dtype=np.int64) for f in sel]
    pos = [np.full(len(r), p, dtype=np.int64) for p, r in enumerate(idx)]
    return (np.concatenate(idx) if idx else np.zeros(0, np.int64)), (np.concatenate(pos) if pos else np.zeros(0, np.int64))


def xyxy_of(xywh):
    b = np.asarray(xywh, dtype=np.float64).reshape(-1, 4)
    return np.stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]], 1)


# ================================================================ gtcheck (task check 2, first half)
def gtcheck(c, out, counts):
    B.PHASE["phase"] = "gtcheck/read"
    _, _, ann = load_data(c)
    sel = selection(c)
    n = len(sel)
    idx, pos = selected_rows(ann["gt_offsets"], sel)
    xywh = ann["gt_xywh"][idx]
    width, height = C.IMAGE_WH
    emitted = C.normalised_cxcywh(xywh, width, height)            # FP32 cxcywh, the form a detector emits
    px = C.pixel_boxes(emitted, width, height)                    # the conversion of the scoring path
    error = float(np.abs(px.astype(np.float64) - xyxy_of(xywh)).max()) if len(xywh) else 0.0
    inside = np.zeros(len(xywh), dtype=bool)
    starts = np.searchsorted(pos, np.arange(n + 1))
    for p, f in enumerate(sel):
        a, b = starts[p], starts[p + 1]
        if b > a:
            inside[a:b] = C.inside_ignore(px[a:b], ann["ign_xywh"][ann["ign_offsets"][f]:ann["ign_offsets"][f + 1]])
    per_frame = np.diff(starts)
    dataset = C.coco_dataset(n, pos, xywh, np.ones(len(xywh), dtype=np.int64), {1: "vehicle"})
    variants = {}
    for name, delete, max_dets in (("coordinates_only", False, (1, 10, 300)), ("with_maxDets_100", False, C.MAX_DETS),
                                   ("with_ignore_rule", True, (1, 10, 300)), ("formal_scoring_path", True, C.MAX_DETS)):
        B.PHASE["phase"] = f"gtcheck/{name}"
        keep = []
        for p in range(n):
            a, b = starts[p], starts[p + 1]
            if b > a:
                gone = inside[a:b] if delete else np.zeros(b - a, dtype=bool)
                keep.append(a + C.submission(gone, np.ones(b - a, dtype=np.float32)))
        keep = np.concatenate(keep) if keep else np.zeros(0, np.int64)
        dets = C.detection_records(pos[keep], np.ones(len(keep), dtype=np.int64), px[keep], np.ones(len(keep)))
        e = C.cocoeval(dataset, dets, area_ranges=C.SIDE_AREA_RANGES, max_dets=max_dets)
        counts.update(cocoeval_calls=1)
        cells = C.read_cells(*C.accumulate(e), [r[0] for r in C.SIDE_AREA_RANGES])
        variants[name] = {"ignore_rule": delete, "maxDets": list(max_dets), "detections": int(len(keep)),
                          "AP": cells["all"]["AP"], "AP50": cells["all"]["AP50"], "AP75": cells["all"]["AP75"], "AR": cells["all"]["AR"],
                          "APS": cells["small"]["AP"], "AP75_small": cells["small"]["AP75"], "AR_small": cells["small"]["AR"]}
        del e, dets
        gc.collect()
    ap = variants["coordinates_only"]["AP"]
    B.require(ap is not None and ap > 1.0 - 1e-9, f"GT boxes as detections do not give vehicle AP 1.000 through the scoring path: {variants}")
    return {"stage": "gtcheck", "passed": True, "frames": int(n), "gt": int(len(xywh)),
            "max_coordinate_error_px": error, "variants": variants,
            "explains_the_gap": {"gt_strictly_inside_an_ignore_rectangle": int(inside.sum()),
                                 "frames_with_more_than_100_gt": int((per_frame > 100).sum()),
                                 "gt_beyond_the_first_100_of_their_frame": int(np.clip(per_frame - 100, 0, None).sum()),
                                 "max_gt_per_frame": int(per_frame.max()) if n else 0},
            "note": "coordinates_only (no deletion, maxDets 300 so that no frame is truncated) is the check of the coordinate "
                    "path and must be AP 1.000 (COCOeval's precision carries an eps of 2.2e-16, so the bound is 1 - 1e-9). "
                    "Under the formal settings a GT copy inside an ignore rectangle is deleted as a detection while the GT "
                    "stays, and maxDets 100 truncates the frames with more than 100 GT: the last two variants are below 1 by "
                    "construction and are reported, not required. APS of a GT copy set can also fall below 1 because "
                    "COCOeval lets a copy of an out-of-range GT take an in-range GT."}


# ================================================================ models
def build_model(c, unit, device):
    """(model, side-car, EMA digest, extra record) of one pinned e120 EMA endpoint, built by the frozen code of its arm."""
    torch = B.torch_mod()
    from sparse_rtdetr.protocol_r18.engine import tensor_digest
    u = c["units"][unit]
    arm, seed = u["arm"], u["seed"]
    path = B.pinned(u["endpoint"])
    side = B.read_json(B.pinned(u["side"]))
    B.require(side["file"] == u["endpoint"] and side["status"] == "COMPLETE" and side["arm"] == arm and side["scope"] == "formal"
              and side["global_update"] == ENDPOINT and side.get("seed", 0) == seed and side["config"]["seed"] == seed
              and side.get("unit", unit) == unit, f"{unit}: endpoint side-car")
    s = torch.load(path, map_location="cpu", weights_only=True)
    B.require(s["engine"]["completed_updates"] == s["ema"]["updates"] == ENDPOINT, f"{unit}: e120 clocks")
    state = s["ema"]["module"]
    B.require(all(not v.is_floating_point() or v.dtype == torch.float32 for v in state.values()), f"{unit}: EMA state is not FP32")
    B.require(tensor_digest(state.items()) == u["weights_sha256"], f"{unit}: EMA digest differs from the stored evaluation receipt")
    extra = {}
    if arm == "S15":
        from sparse_rtdetr.protocol_r18.geometry import validate_geometry
        from sparse_rtdetr.protocol_r18.model import build
        from sparse_rtdetr.research import scale_baseline as S
        B.require((s["version"], s["arm"], s["config"]) == (S.VERSION, "S15", S.production_config(seed).binding()), f"{unit}: S15 state identity")
        model, _, _ = build(S.production_config(seed), synthetic=True)
        model.load_state_dict(state, strict=True)
        validate_geometry(model)
        model.eval().requires_grad_(False).to(device)
        B.require(not any(getattr(m, "_s15_checkpointed", False) for m in model.modules()), "activation checkpoint installed for evaluation")
    else:
        keys = [k for k in s if k.endswith("_binding_sha256")]
        expect = {k: v for k, v in side.items() if k not in ("status", "file")}
        B.require(len(keys) == 1 and s[keys[0]] == B.sha_bytes(B.canonical(expect)), f"{unit}: state binding digest")
        from sparse_rtdetr.research import r27_l0_control as L0
        if arm == "L0":
            model = L0.build_eval_model(state, device, seed)
            B.require(getattr(model, "l0_arm", None) == "L0" and L0.is_wrapped(model), f"{unit}: not an L0 evaluation model")
        else:
            from sparse_rtdetr.research import r28_fovea_geo as FG
            from sparse_rtdetr.research import r28_warp as W
            sel = FG.check_selection(c["width_selection"])
            B.pinned(c["demand"]["weights"])
            source = {"kind": "network", "file": c["demand"]["weights"], "fwhm": float.fromhex(sel["fwhm_hex"]), "selection": sel}
            mapping = FG.mapping(source)
            B.require(FG.WIDTH_HEX == MAPPING["fwhm_hex"] and FG.WIDTH == MAPPING["fwhm"] and list(FG.GRID_SHAPE) == MAPPING["grid_shape"]
                      and FG.PAD == MAPPING["padding_size"] and W.EPS == MAPPING["eps"], "the mapping constants are not those of task 009")
            B.require(side["demand"] == c["demand"]["binding"] and side["mapping"] == mapping,
                      f"{unit}: the endpoint was trained with another demand source / width")
            model = FG.build_eval_model(state, device, "FG", source, seed=seed)
            B.require(FG.is_wrapped(model) and model.r28_ctx.binding() == side["demand"] and model.r28_mapping == mapping,
                      f"{unit}: not an F_geo evaluation model")
            extra = {"demand": side["demand"], "mapping": {**mapping, "grid_shape": MAPPING["grid_shape"],
                                                           "padding_size": MAPPING["padding_size"], "eps": MAPPING["eps"]}}
    del s
    weights = tensor_digest(model.state_dict().items())
    B.require(weights == u["weights_sha256"] and all(not m.training for m in model.modules())
              and not any(p.requires_grad for p in model.parameters()), f"{unit}: evaluation model state / mode")
    return model, side, weights, extra


def record_demand(model):
    """F_geo only: keep a copy of every demand map the forward computes.  The model's own context computes it; the
    forward is not changed.  Returns (list of (B, 12, 21) FP64 CPU tensors, remover)."""
    from sparse_rtdetr.research import r28_resample as RS
    ctx = model.r28_ctx
    store = []

    def demand(x):
        d = RS.Context.demand(ctx, x)
        store.append(d.clone())
        return d

    B.require("demand" not in vars(ctx), "the demand context is already instrumented")
    vars(ctx)["demand"] = demand
    return store, lambda: vars(ctx).pop("demand")


def frame_scores(logits):
    """(300, 10) FP32 logits of one frame -> FP32 sigmoid scores: the expression of r0.evaluation.rank_candidates."""
    torch = B.torch_mod()
    return torch.from_numpy(np.array(logits, dtype=np.float32)).sigmoid().numpy()


def run_frames(model, jobs, device, counts, sink, hw):
    """Batches of 4 through the model on the development evaluation path.  jobs: (data, row) per image, handled by
    infer_seed.load_original / infer_seed.pixels; sink(offset, boxes, logits) receives the FP32 CPU outputs."""
    from concurrent.futures import ThreadPoolExecutor
    from infer_seed import load_original, pixels
    from sparse_rtdetr.protocol_r18.geometry import forward_guards
    torch = B.torch_mod()
    B.require(len(jobs) % BATCH == 0, "the image list does not fill the batches")

    def decode(job):
        with load_original(job[0], job[1]) as image:
            return pixels(job[0], image)

    pending, submitted = deque(), 0
    with ThreadPoolExecutor(DECODE_THREADS) as pool, torch.inference_mode():
        for off in range(0, len(jobs), BATCH):
            while submitted < len(jobs) and len(pending) < DECODE_AHEAD * BATCH:
                pending.append(pool.submit(decode, jobs[submitted]))
                submitted += 1
            x = torch.stack([pending.popleft().result() for _ in range(BATCH)]).to(device)
            B.require(tuple(x.shape) == (BATCH, 3, *hw) and x.dtype == torch.float32, "the input is not the FP32 H canvas")
            counts.update(forward_attempts=1)
            with torch.autocast("cuda", enabled=False), forward_guards(model):
                o = model(x)
            counts.update(forwards=1)
            B.require(set(o) == {"pred_logits", "pred_boxes"} and o["pred_boxes"].shape == (BATCH, C.QUERIES, 4)
                      and o["pred_logits"].shape == (BATCH, C.QUERIES, C.CLASSES)
                      and o["pred_boxes"].dtype == o["pred_logits"].dtype == torch.float32, "endpoint output schema")
            sink(off, o["pred_boxes"].cpu(), o["pred_logits"].cpu())


def _gpu_stage(c, unit):
    torch = B.torch_mod()
    have = nvidia("gpu=uuid")
    B.require(have == [c["gpu_uuid"]], f"GPU differs from the contract: {have}")
    return torch, torch.device("cuda:0"), other_gpu_processes()


# ================================================================ wire (task check 1)
def wire(c, out, counts, unit):
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    from sparse_rtdetr.protocol_r18.engine import tensor_digest
    from sparse_rtdetr.r0.evaluation import rank_candidates
    from sparse_rtdetr.research import scale_baseline as S
    B.PHASE["phase"] = f"wire/{unit}/load"
    torch, device, others = _gpu_stage(c, unit)
    model, side, weights, extra = build_model(c, unit, device)
    data = S.development(c["data"]["development"], S.S15_HW)
    B.require([r["stable_image_id"] for r in data.records] == c["development_ids"] and len(data.records) == 548, "development members/order")
    rows = data.records[:WIRE_IMAGES]
    store, remove = record_demand(model) if c["units"][unit]["arm"] == "FG" else (None, None)
    mine, same = [], {"records": 0, "boxes": 0, "scores": 0}

    def sink(off, boxes, logits):
        for j in range(BATCH):
            row = rows[off + j]
            records = rank_candidates(logits[j], boxes[j], row, 600)
            mine.extend(records)
            px = C.pixel_boxes(boxes[j].numpy(), row["width"], row["height"])
            sc = frame_scores(logits[j].numpy())
            for r in records:                                      # this entry's own conversions against the frozen ranking
                same["records"] += 1
                same["boxes"] += [float(v) for v in px[r["query_id"]]] == r["bbox_xyxy"]
                same["scores"] += float(sc[r["query_id"], r["category_id"] - 1]) == r["score"]

    rng = ck.capture_rng(device)
    B.PHASE["phase"] = f"wire/{unit}/forward"
    run_frames(model, [(data, r) for r in rows], device, counts, sink, S.S15_HW)
    if remove is not None:
        remove()
        B.require(len(store) == WIRE_IMAGES // BATCH and model.r28_ctx.binding() == side["demand"], "demand source after the wiring forwards")
    B.require(counts["forwards"] == counts["forward_attempts"] == WIRE_IMAGES // BATCH and counts["actual_updates"] == 0, "wiring forward count")
    B.require(tensor_digest(model.state_dict().items()) == weights and ck.exact_equal(ck.capture_rng(device), rng),
              "the wiring forwards changed the model or the RNG")
    B.PHASE["phase"] = f"wire/{unit}/compare"
    stored = B.read_json(B.pinned(c["units"][unit]["predictions_600"]))
    B.require(len(stored) == 548 * 600, "stored top600 prediction record count")
    difference = C.records_equal(mine, stored[:WIRE_IMAGES * 600])
    res = {"stage": "wire", "unit": unit, "images": WIRE_IMAGES, "forwards": counts["forwards"], "records": len(mine),
           "fields": sorted(mine[0]), "equal_to_stored_top600": difference is None, "first_difference": difference,
           "own_conversions": {**same, "equal": same["boxes"] == same["scores"] == same["records"] == len(mine)},
           "stored_predictions": c["units"][unit]["predictions_600"], "endpoint": side["file"], "weights_sha256": weights, **extra,
           "other_gpu_compute_processes_at_start": others, "other_gpu_compute_processes_at_end": other_gpu_processes(),
           "peak_cuda_bytes": torch.cuda.max_memory_allocated(), "device": torch.cuda.get_device_name(0)}
    B.require(res["equal_to_stored_top600"] and res["own_conversions"]["equal"],
              f"wiring check failed: this entry does not reproduce the stored development predictions: "
              f"{ {k: res[k] for k in ('first_difference', 'own_conversions')} }")
    res["passed"] = True
    return res


# ================================================================ infer
def infer(c, out, counts, unit):
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    from sparse_rtdetr.protocol_r18.engine import tensor_digest
    from sparse_rtdetr.r0.evaluation import rank_candidates
    from sparse_rtdetr.research import scale_baseline as S
    B.PHASE["phase"] = f"infer/{unit}/guards"
    wired = stage_closed(c, f"wire-{unit}")
    B.require(wired["passed"] is True and wired["unit"] == unit and wired["weights_sha256"] == c["units"][unit]["weights_sha256"],
              f"{unit}: the wiring check has not passed")
    closed, frames, _ = load_data(c)
    sel = selection(c)
    n = len(sel)
    B.require(n % BATCH == 0 and (c["dev"] or n == C.FRAMES), "frame selection")
    torch, device, others = _gpu_stage(c, unit)
    B.PHASE["phase"] = f"infer/{unit}/load"
    model, side, weights, extra = build_model(c, unit, device)
    arm = c["units"][unit]["arm"]
    transform = S.RawAugmentation(S.S15_HW)                        # only its resize is used, as in S.development
    roots = {s: types.SimpleNamespace(spec={"roots": {"images": str(Path(c["uavdt"]["images"]) / s)}}, transform=transform)
             for s in C.TEST_SEQUENCES}
    jobs = [(roots[frames[i]["sequence"]], frames[i]) for i in sel]
    rows = [{"stable_image_id": frames[i]["relative_path"], "width": frames[i]["width"], "height": frames[i]["height"]} for i in sel]
    B.require(all((r["width"], r["height"]) == C.IMAGE_WH for r in rows), "frame size table")
    store, remove = record_demand(model) if arm == "FG" else (None, None)
    open_memmap = np.lib.format.open_memmap
    boxes_mm = open_memmap(out / "boxes.npy.partial", mode="w+", dtype=np.float32, shape=(n, C.QUERIES, 4))
    logits_mm = open_memmap(out / "logits.npy.partial", mode="w+", dtype=np.float32, shape=(n, C.QUERIES, C.CLASSES))
    demand_mm = open_memmap(out / "demand.npy.partial", mode="w+", dtype=np.float64, shape=(n, 12, 21)) if arm == "FG" else None
    started = time.perf_counter()
    refused = []

    def sink(off, boxes, logits):
        B.require(bool(torch.isfinite(boxes).all()) and bool(torch.isfinite(logits).all()),
                  f"STOP AND HAND BACK: non-finite endpoint output in the batch at frame position {off}")
        for j in range(BATCH):
            # the ranking function of the development evaluation also refuses a frame with a box whose FP32 corners are
            # not a positive interval.  Nothing is repaired or dropped here: the frame is kept as emitted (the task
            # submits all 300 candidates) and the refusal is counted for the report.
            try:
                rank_candidates(logits[j], boxes[j], rows[off + j], 300)
            except Exception as exc:
                counts.update(frames_refused_by_the_frozen_ranking=1)
                if len(refused) < 50:
                    refused.append({"position": off + j, "frame": rows[off + j]["stable_image_id"], "error": repr(exc)[:300]})
        boxes_mm[off:off + BATCH] = boxes.numpy()
        logits_mm[off:off + BATCH] = logits.numpy()
        if store is not None:
            B.require(len(store) == 1 and tuple(store[0].shape) == (BATCH, 12, 21) and store[0].dtype == torch.float64,
                      "one demand map per forward")
            demand_mm[off:off + BATCH] = store.pop().numpy()
        counts.update(frames=BATCH)
        if off % (250 * BATCH) == 0:
            print(f"{unit}: {off + BATCH}/{n} frames, {time.perf_counter() - started:.0f} s", flush=True)

    rng = ck.capture_rng(device)
    B.PHASE["phase"] = f"infer/{unit}/forward"
    run_frames(model, jobs, device, counts, sink, S.S15_HW)
    seconds = time.perf_counter() - started
    if remove is not None:
        remove()
    B.PHASE["phase"] = f"infer/{unit}/close"
    B.require(counts["forwards"] == counts["forward_attempts"] == n // BATCH and counts["frames"] == n and counts["actual_updates"] == 0
              and (c["dev"] or counts["forwards"] == 4148), "inference forward count")
    B.require(tensor_digest(model.state_dict().items()) == weights and ck.exact_equal(ck.capture_rng(device), rng)
              and (arm != "FG" or model.r28_ctx.binding() == side["demand"]), "the inference changed the model, the RNG or the demand source")
    refs = {}
    for name, mm in (("boxes", boxes_mm), ("logits", logits_mm), ("demand", demand_mm)):
        if mm is not None:
            mm.flush()
            link_new(out / f"{name}.npy.partial", out / f"{name}.npy")
            refs["demand_maps" if name == "demand" else name] = B.ref(out / f"{name}.npy")
    return {"stage": "infer", "unit": unit, "arm": arm, "seed": c["units"][unit]["seed"], "kind": "EMA", "epoch": 120,
            "endpoint": side["file"], "weights_sha256": weights, **extra, "frames": int(n), "forwards": counts["forwards"],
            "batch": BATCH, "input_hw": list(S.S15_HW), "FP32": True,
            "frame_selection_sha256": hashlib.sha256(sel.astype("<i8").tobytes()).hexdigest(), **refs,
            "frozen_ranking_guard": {"function": "r0.evaluation.rank_candidates(logits, boxes, row, 300) on every frame",
                                     "frames_refused": counts["frames_refused_by_the_frozen_ranking"], "first_refusals": refused,
                                     "note": "a refused frame is kept as emitted and scored like every other frame"},
            "inference_seconds": seconds, "frames_per_second": n / seconds,
            "other_gpu_compute_processes_at_start": others, "other_gpu_compute_processes_at_end": other_gpu_processes(),
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(), "device": torch.cuda.get_device_name(0)}


# ================================================================ geometry
def _development_geometry(c):
    """The development column: per-GT g of R30-001 (predicted demand), sizes on canvas L, original image sizes."""
    from sparse_rtdetr.research import scale_baseline as S
    per_gt = B.read_json(B.pinned(c["development_geometry"]["per_gt"]))
    want = c["development_geometry"]["reading"]
    g = np.array([r["predicted"] for r in per_gt], dtype=np.float64)
    area = np.array([r["area"] for r in per_gt], dtype=np.float64)
    reading = C.geometry_reading(g, area)
    B.require(len(per_gt) == want["GT"] and reading["small"]["n"] == want["small_GT"] and reading["small"]["bin_counts"] == want["small_bin_counts"]
              and round(reading["small"]["g_geomean"], 4) == want["small_g_geomean_4dp"],
              f"the development geometry of R30-001 is not reproduced: {reading['small']}")
    records = S.development(c["data"]["development"], S.S15_HW).records
    B.require([r["stable_image_id"] for r in records] == c["development_ids"], "development members/order")
    wh = {r["stable_image_id"]: (r["width"], r["height"]) for r in records}
    box = np.array([r["bbox_xyxy"] for r in per_gt], dtype=np.float64)
    bw, bh = box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]
    iw = np.array([wh[r["image"]][0] for r in per_gt], dtype=np.float64)
    ih = np.array([wh[r["image"]][1] for r in per_gt], dtype=np.float64)
    vehicle = np.isin([r["category_id"] for r in per_gt], DEVELOPMENT_VEHICLE_CATEGORIES)
    small = area <= C.SMALL_AREA
    sizes = Counter((r["width"], r["height"]) for r in records)
    w_all, h_all = np.array([r["width"] for r in records], dtype=np.float64), np.array([r["height"] for r in records], dtype=np.float64)
    return {"source": c["development_geometry"]["per_gt"], "reproduces_R30_001": True,
            "all_classes": reading, "vehicle_classes_only": C.geometry_reading(g[vehicle], area[vehicle]),
            "vehicle_classes_note": "VisDrone car / van / truck / bus only; a descriptive column added by the executor before "
                                    "any UAVDT result was read (the task asks for the all-class column)",
            "canvas_L_sizes": {"all_classes": C.canvas_sizes(bw, bh, iw, ih, S.PARENT_HW),
                               "all_classes_small": C.canvas_sizes(bw[small], bh[small], iw[small], ih[small], S.PARENT_HW),
                               "vehicle_classes_only": C.canvas_sizes(bw[vehicle], bh[vehicle], iw[vehicle], ih[vehicle], S.PARENT_HW),
                               "vehicle_classes_only_small": C.canvas_sizes(bw[vehicle & small], bh[vehicle & small], iw[vehicle & small],
                                                                            ih[vehicle & small], S.PARENT_HW)},
            "annotation_sizes": {"width": C.quantiles(bw), "height": C.quantiles(bh), "area": C.quantiles(area)},
            "original_image_sizes": {"images": len(records), "width": C.quantiles(w_all), "height": C.quantiles(h_all),
                                     "pixels": C.quantiles(w_all * h_all),
                                     "uniform_scale_to_L_x": C.quantiles(S.PARENT_HW[1] / w_all),
                                     "uniform_scale_to_L_y": C.quantiles(S.PARENT_HW[0] / h_all),
                                     "counts": [{"width": w, "height": h, "images": k} for (w, h), k in sizes.most_common()]}}


def geometry(c, out, counts):
    from sparse_rtdetr.research import r28_fovea as FV
    from sparse_rtdetr.research import r28_fovea_geo as FG
    from sparse_rtdetr.research import scale_baseline as S
    torch = B.torch_mod()
    B.PHASE["phase"] = "geometry/read"
    _, _, ann = load_data(c)
    sel = selection(c)
    n = len(sel)
    units = [u for u in MAIN_UNITS if parse(u)[0] == "FG"]
    inf = {u: stage_closed(c, f"infer-{u}") for u in units}
    demand = [np.load(B.pinned(inf[u]["demand_maps"]), allow_pickle=False) for u in units]
    B.require(all(d.shape == (n, 12, 21) and d.dtype == np.float64 for d in demand), "demand map tables")
    identical = all(np.array_equal(demand[0], d) for d in demand[1:])
    idx, pos = selected_rows(ann["gt_offsets"], sel)
    xywh = ann["gt_xywh"][idx]
    area = (xywh[:, 2] * xywh[:, 3]).astype(np.float64)
    width, height = C.IMAGE_WH
    starts = np.searchsorted(pos, np.arange(n + 1))
    clipped = xyxy_of(xywh)
    clipped[:, [0, 2]] = np.clip(clipped[:, [0, 2]], 0, width)
    clipped[:, [1, 3]] = np.clip(clipped[:, [1, 3]], 0, height)
    measurable = (clipped[:, 2] > clipped[:, 0]) & (clipped[:, 3] > clipped[:, 1])
    unit_box = clipped / np.array([width, height, width, height], dtype=np.float64)

    def per_gt(d):
        g = np.full(len(xywh), np.nan)
        identity = np.zeros(n, dtype=bool)
        for p0 in range(0, n, BATCH):                             # the batches of the inference: the plan of the forward
            plan = FG.plan_from_demand(torch.from_numpy(d[p0:p0 + BATCH]))
            identity[p0:p0 + BATCH] = plan.identity.numpy()
            for j in range(BATCH):
                rows = np.arange(starts[p0 + j], starts[p0 + j + 1])
                rows = rows[measurable[rows]]
                if len(rows):
                    ox, oy = FV.magnification(plan.select(slice(j, j + 1)), torch.from_numpy(unit_box[rows])[None])
                    g[rows] = torch.sqrt(ox * oy)[0].numpy()
        return g, identity

    B.PHASE["phase"] = "geometry/magnification"
    first, identity = per_gt(demand[0])
    table = [first] + [first if identical else per_gt(d)[0] for d in demand[1:]]
    B.require(bool(np.isfinite(first[measurable]).all()) and bool((first[measurable] > 0).all()), "per-GT magnification")
    readings = {u: C.geometry_reading(g[measurable], area[measurable]) for u, g in zip(units, table)}

    B.PHASE["phase"] = "geometry/round-trip"
    with_gt = np.nonzero(np.diff(starts) > 0)[0]
    pick = np.unique(with_gt[np.round(np.linspace(0, len(with_gt) - 1, min(ROUND_TRIP_FRAMES, len(with_gt)))).astype(np.int64)])
    worst = {"float32": 0.0, "float64": 0.0}
    boxes_checked = 0
    for p in pick:
        p0 = (p // BATCH) * BATCH
        plan = FG.plan_from_demand(torch.from_numpy(demand[0][p0:p0 + BATCH])).select(slice(p - p0, p - p0 + 1))
        rows = np.arange(starts[p], starts[p + 1])
        truth = xyxy_of(xywh[rows])
        b = xywh[rows].astype(np.float64)
        exact = np.stack([(b[:, 0] + b[:, 2] / 2) / width, (b[:, 1] + b[:, 3] / 2) / height, b[:, 2] / width, b[:, 3] / height], 1)
        for name, start in (("float32", C.normalised_cxcywh(xywh[rows], width, height)), ("float64", exact)):
            warped = FV.map_boxes(plan, torch.from_numpy(start)[None], inverse=False)   # source frame -> warped canvas
            back = FV.map_boxes(plan, warped, inverse=True)[0].numpy()                  # warped canvas -> source frame
            if name == "float32":
                corners = C.pixel_boxes(back, width, height).astype(np.float64)        # the conversion of the scoring path
            else:
                half = back[:, 2:] / 2
                corners = np.concatenate((back[:, :2] - half, back[:, :2] + half), 1) * np.array([width, height] * 2, dtype=np.float64)
            worst[name] = max(worst[name], float(np.abs(corners - truth).max()))
        boxes_checked += len(rows)
    B.require(worst["float32"] < 0.5 and worst["float64"] < 0.5, f"F_geo map round trip exceeds 0.5 px: {worst}")

    B.PHASE["phase"] = "geometry/development"
    development = _development_geometry(c)
    g_ref = save_npy(out / "per_gt_g.npy", np.stack(table))
    small = area <= C.SMALL_AREA
    return {"stage": "geometry", "units": units, "frames": int(n), "gt": int(len(xywh)),
            "demand_maps_identical_across_the_three_F_geo_units": bool(identical),
            "demand_maps": {u: inf[u]["demand_maps"] for u in units}, "per_gt_g": g_ref,
            "per_gt_g_rows": units, "gt_without_g": {"count": int((~measurable).sum()),
                                                      "reason": "the box clipped to the image has a non-positive side"},
            "identity_frames": int(identity.sum()),
            "g": "sqrt(ox * oy) of r28_fovea.magnification on the GT box clipped to the image, plan = the map of the F_geo "
                 "forward (demand of the frozen network, fwhm 19.03125, 31 x 51 nodes, padding 30, eps 0.2)",
            "uavdt": readings[units[0]], "uavdt_by_unit": readings if not identical else None,
            "development": development,
            "canvas_L_sizes": {"uavdt": C.canvas_sizes(xywh[:, 2], xywh[:, 3], width, height, S.PARENT_HW),
                               "uavdt_small": C.canvas_sizes(xywh[small, 2], xywh[small, 3], width, height, S.PARENT_HW),
                               "canvas_hw": list(S.PARENT_HW), "uavdt_image_wh": [width, height],
                               "uavdt_uniform_scale": {"x": S.PARENT_HW[1] / width, "y": S.PARENT_HW[0] / height}},
            "uavdt_annotation_sizes": {"width": C.quantiles(xywh[:, 2]), "height": C.quantiles(xywh[:, 3]), "area": C.quantiles(area)},
            "round_trip": {"passed": True, "frames": int(len(pick)), "boxes": int(boxes_checked), "max_error_px": worst,
                           "bound_px": 0.5, "path": "GT as normalised cxcywh -> r28_fovea.map_boxes(inverse=False) -> "
                                                    "map_boxes(inverse=True) -> pixel corners, against the integer GT corners; "
                                                    "float32: FP32 boxes and the FP32 conversion of the scoring path, "
                                                    "float64: the same maps on FP64 boxes"}}


# ================================================================ score
def score(c, out, counts, unit):
    B.PHASE["phase"] = f"score/{unit}/read"
    dat, _, ann = load_data(c)
    inf = stage_closed(c, f"infer-{unit}")
    B.require(inf["unit"] == unit and inf["weights_sha256"] == c["units"][unit]["weights_sha256"], "inference receipt")
    sel = selection(c)
    n = len(sel)
    boxes = np.load(B.pinned(inf["boxes"]), mmap_mode="r", allow_pickle=False)
    logits = np.load(B.pinned(inf["logits"]), mmap_mode="r", allow_pickle=False)
    B.require(boxes.shape == (n, C.QUERIES, 4) and logits.shape == (n, C.QUERIES, C.CLASSES)
              and boxes.dtype == logits.dtype == np.float32, "raw output tables")
    width, height = C.IMAGE_WH
    idx, pos = selected_rows(ann["gt_offsets"], sel)
    xywh, cat = ann["gt_xywh"][idx], ann["gt_cat"][idx]
    area = (xywh[:, 2] * xywh[:, 3]).astype(np.float64)
    starts = np.searchsorted(pos, np.arange(n + 1))
    truth = xyxy_of(xywh)
    small = area <= C.SMALL_AREA
    coverage = np.zeros((2, len(xywh)), dtype=np.float64)          # best IoU: all 300 candidates / after the ignore rule
    main = {"image": [], "xyxy": [], "score": []}
    side = {"image": [], "category": [], "xyxy": [], "score": []}
    deleted = np.zeros(n, dtype=np.int64)
    B.PHASE["phase"] = f"score/{unit}/candidates"
    for p, f in enumerate(sel):
        sc = frame_scores(logits[p])
        px = C.pixel_boxes(np.asarray(boxes[p]), width, height)
        inside = C.inside_ignore(px, ann["ign_xywh"][ann["ign_offsets"][f]:ann["ign_offsets"][f + 1]])
        deleted[p] = inside.sum()
        vehicle = C.class_scores(sc, C.VEHICLE_COLUMNS)
        kept = C.submission(inside, vehicle)
        main["image"].append(np.full(len(kept), p, dtype=np.int64))
        main["xyxy"].append(px[kept])
        main["score"].append(vehicle[kept])
        for category, columns in C.THREE_CLASS_COLUMNS.items():
            s3 = C.class_scores(sc, columns)
            k3 = C.submission(inside, s3, limit=C.MAX_DETS[-1])   # COCOeval reads at most maxDets per image and category
            side["image"].append(np.full(len(k3), p, dtype=np.int64))
            side["category"].append(np.full(len(k3), category, dtype=np.int64))
            side["xyxy"].append(px[k3])
            side["score"].append(s3[k3])
        a, b = starts[p], starts[p + 1]
        if b > a:
            coverage[0, a:b] = C.best_iou(truth[a:b], px)
            coverage[1, a:b] = C.best_iou(truth[a:b], px[~inside])
    main = {k: np.concatenate(v) for k, v in main.items()}
    side = {k: np.concatenate(v) for k, v in side.items()}
    B.require(len(main["image"]) == n * C.QUERIES - int(deleted.sum()), "submitted candidate count")

    B.PHASE["phase"] = f"score/{unit}/cocoeval"
    labels = [r[0] for r in C.AREA_RANGES]
    dataset = C.coco_dataset(n, pos, xywh, np.ones(len(xywh), dtype=np.int64), {1: "vehicle"})
    e = C.cocoeval(dataset, C.detection_records(main["image"], np.ones(len(main["image"]), dtype=np.int64), main["xyxy"], main["score"]))
    counts.update(cocoeval_calls=1)
    seq_of = ann["frame_seq"][sel]

    def reading(positions=None):
        precision, recall = C.accumulate(e, positions)
        cells = C.read_cells(precision, recall, labels)
        return precision, recall, cells, C.named_metrics(cells)

    def subset(mask):
        positions = np.nonzero(mask)[0]
        rows = np.isin(pos, positions)
        return {"frames": int(len(positions)), "gt": int(rows.sum()), "small_gt": int((rows & small).sum()),
                "metrics": reading(positions)[3] if len(positions) else None}

    precision, recall, cells, overall = reading()
    per_sequence = {s: subset(seq_of == k) for k, s in enumerate(C.TEST_SEQUENCES)}
    without = subset(seq_of != C.TEST_SEQUENCES.index(C.DOMINANT_SEQUENCE))
    by_attribute = {g: {"sequences": seqs, **subset(np.isin(seq_of, [C.TEST_SEQUENCES.index(s) for s in seqs]))}
                    for g, seqs in dat["attribute_groups"].items()}
    tmp = out / "cocoeval.partial.npz"
    np.savez(tmp, precision=precision, recall=recall, areaRngLbl=np.array(labels), maxDets=np.array(C.MAX_DETS))
    link_new(tmp, out / "cocoeval.npz")
    e = dataset = None                                             # released before the three-class evaluation
    gc.collect()

    B.PHASE["phase"] = f"score/{unit}/three-class"
    dataset3 = C.coco_dataset(n, pos, xywh, cat, C.UAVDT_CLASSES)
    e3 = C.cocoeval(dataset3, C.detection_records(side["image"], side["category"], side["xyxy"], side["score"]),
                    area_ranges=C.SIDE_AREA_RANGES)
    counts.update(cocoeval_calls=1)
    p3, r3 = C.accumulate(e3)
    three = {}
    for k, (category, name) in enumerate(sorted(C.UAVDT_CLASSES.items())):
        cells3 = C.read_cells(p3, r3, [r[0] for r in C.SIDE_AREA_RANGES], category=k)
        three[name] = {"AP": cells3["all"]["AP"], "APS": cells3["small"]["AP"], "gt": int((cat == category).sum()),
                       "small_gt": int(((cat == category) & small).sum()), "detector_classes": [col + 1 for col in C.THREE_CLASS_COLUMNS[category]]}
    del e3, dataset3
    gc.collect()

    B.PHASE["phase"] = f"score/{unit}/write"
    cov_ref = save_npy(out / "coverage.npy", coverage[:, small])
    metrics = {"unit": unit, "arm": c["units"][unit]["arm"], "seed": c["units"][unit]["seed"], "weights_sha256": inf["weights_sha256"],
               "contract_sha256": c["contract_sha256"], "dev_partial": c["dev"], "frames": int(n), "gt": int(len(xywh)),
               "gt_by_area_range": {name: int(((area >= lo) & (area <= hi)).sum()) for name, lo, hi in C.AREA_RANGES},
               "candidates": int(n * C.QUERIES), "deleted_by_ignore_rule": int(deleted.sum()),
               "frames_with_a_deleted_candidate": int((deleted > 0).sum()), "submitted": int(len(main["image"])),
               "frames_refused_by_the_frozen_ranking": inf["frozen_ranking_guard"]["frames_refused"],
               "overall": overall, "cells": cells, "per_sequence": per_sequence, "without_" + C.DOMINANT_SEQUENCE: without,
               "by_attribute": by_attribute,
               "three_class": {"header": C.THREE_CLASS_HEADER, "classes": three, "submitted": int(len(side["image"])),
                               "note": "per class and frame the first 100 candidates after the ignore rule are submitted: "
                                       "COCOeval reads no more than maxDets 100 per image and category"}}
    B.write_new(out / "metrics.json", metrics)
    return {"stage": "score", "unit": unit, "metrics": B.ref(out / "metrics.json"), "coverage": cov_ref,
            "coverage_rows": ["best IoU over the 300 candidates of the frame", "best IoU over the candidates left by the ignore rule"],
            "coverage_gt": "the small GT (annotation area <= 32^2) of the selected frames, in annotation order",
            "cocoeval": B.ref(out / "cocoeval.npz"), "overall": overall, "frames": int(n),
            "deleted_by_ignore_rule": int(deleted.sum()), "submitted": int(len(main["image"]))}


# ================================================================ dhash (task reading 7)
def dhash(c, out, counts):
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    B.PHASE["phase"] = "dhash/read"
    _, frames, _ = load_data(c)
    off = _offsets()
    ours = [frames[off[k] + f - 1] for k, n in enumerate(C.SEQUENCE_FRAMES) for f in C.every_tenth(n)]
    root = Path(c["uavdt"]["images"])
    theirs = []
    for ref, rows in zip(c["visdrone"]["manifests"], c["visdrone"]["rows"]):
        lines = [json.loads(line) for line in B.pinned(ref).read_text().splitlines() if line.strip()]
        B.require(len(lines) == rows, f"VisDrone manifest rows: {ref['path']}")
        theirs.extend(lines)
    B.require(len({r["relative_path"] for r in theirs}) == len(theirs) == sum(c["visdrone"]["rows"]), "VisDrone official train + val members")

    def path_of(row):
        split = Path(row["relative_path"]).parts[0]
        return Path(c["visdrone"]["images"][split]) / Path(row["relative_path"]).name

    def one(job):
        path, row = job
        B.require(path.is_file() and not path.is_symlink(), f"image missing: {path}")
        raw = path.read_bytes()
        B.require(len(raw) == row["image_size_bytes"] and hashlib.sha256(raw).hexdigest() == row["image_sha256"], f"image differs: {path}")
        with Image.open(io.BytesIO(raw)) as image:
            return C.dhash(np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)))

    B.PHASE["phase"] = "dhash/hash"
    with ThreadPoolExecutor(DECODE_THREADS) as pool:
        a = np.array(list(pool.map(one, [(root / r["relative_path"], r) for r in ours])), dtype=np.uint64)
        b = np.array(list(pool.map(one, [(path_of(r), r) for r in theirs])), dtype=np.uint64)
    counts.update(uavdt_frames_hashed=len(a), visdrone_images_hashed=len(b))
    B.PHASE["phase"] = "dhash/compare"
    distance = C.hamming(a, b)
    seq = np.array([C.TEST_SEQUENCES.index(r["sequence"]) for r in ours])
    per_sequence = {}
    for k, s in enumerate(C.TEST_SEQUENCES):
        d = distance[seq == k]
        i, j = np.unravel_index(int(d.argmin()), d.shape)
        rows = np.nonzero(seq == k)[0]
        per_sequence[s] = {"frames_hashed": int(len(rows)), "min_hamming": int(d.min()),
                           "pair": [ours[rows[i]]["relative_path"], theirs[j]["relative_path"]],
                           "pairs_le_6": int((d <= C.DHASH_NEAR).sum())}
    near = [{"hamming": int(distance[i, j]), "uavdt": ours[i]["relative_path"], "visdrone": theirs[j]["relative_path"]}
            for i, j in zip(*np.nonzero(distance <= C.DHASH_NEAR))]
    B.write_new(out / "pairs_le_6.json", near)
    splits = Counter(Path(r["relative_path"]).parts[0] for r in theirs)
    return {"stage": "dhash", "definition": "64-bit dHash: image -> PIL 'L' -> resize to 9 x 8 (LANCZOS) -> bit = pixel brighter "
                                            "than its left neighbour, row major; Hamming distance between the 64-bit values",
            "uavdt_frames": "frames 1, 11, 21, ... of each of the 20 test sequences", "uavdt_frames_hashed": int(len(a)),
            "visdrone_images_hashed": int(len(b)), "visdrone_splits": dict(splits),
            "visdrone_note": "the VisDrone official train and val images (raw manifests of the project's conversion); "
                             "test-dev is not read", "threshold": C.DHASH_NEAR,
            "per_sequence": per_sequence, "min_hamming": int(distance.min()), "pairs_le_6": len(near),
            "pairs_le_6_file": B.ref(out / "pairs_le_6.json"), "record_only": "pairs are listed by path; no image was viewed",
            "distance_histogram_0_to_16": [int((distance == k).sum()) for k in range(17)]}


# ================================================================ tables
def tables(c, out, counts):
    B.PHASE["phase"] = "tables/read"
    R = Path(c["outputs"])
    dat, _, ann = load_data(c)
    gtc, geo, dh = stage_closed(c, "gtcheck"), stage_closed(c, "geometry"), stage_closed(c, "dhash")
    reference = [u for u in REFERENCE_UNITS if (R / f"score-{u}").exists() or (R / f"infer-{u}").exists()]
    B.require(len(reference) in (0, len(REFERENCE_UNITS)), f"the reference arm is partly present: {reference}")
    units = MAIN_UNITS + tuple(reference)
    wired = {u: stage_closed(c, f"wire-{u}") for u in units}
    inf = {u: stage_closed(c, f"infer-{u}") for u in units}
    sc = {u: stage_closed(c, f"score-{u}") for u in units}
    m = {u: B.read_json(B.pinned(sc[u]["metrics"])) for u in units}
    sel = selection(c)
    B.require(all(m[u]["frames"] == len(sel) and m[u]["weights_sha256"] == c["units"][u]["weights_sha256"] for u in units), "scored units")

    def values(arm, pick=lambda x: x["overall"]):
        return {s: pick(m[f"{arm}-s{s}"]) for s in SEEDS}

    B.PHASE["phase"] = "tables/main"
    fg, l0 = values("FG"), values("L0")
    main = C.paired(fg, l0)
    label = C.label([main["per_seed"][s] for s in SEEDS])
    result = {"status": "PASS", "version": VERSION, "contract_sha256": c["contract_sha256"], "source_commit": c["source_commit"],
              "record_only": True, "gate": None, "dev_partial": c["dev"], "frames": int(len(sel)),
              "main": {"difference": "F_geo - L0, percentage points, per seed; mean and sample standard deviation (n - 1) of the three",
                       "differences_pp": main, "label": label, "labels": PROTOCOL["labels"],
                       "rule": {s: C.rule_cells(main["per_seed"][s]) for s in SEEDS},
                       "endpoints": {u: m[u]["overall"] for u in MAIN_UNITS}, "cells": {u: m[u]["cells"] for u in MAIN_UNITS},
                       "gt_by_area_range": m[MAIN_UNITS[0]]["gt_by_area_range"],
                       "candidates": {u: {k: m[u][k] for k in ("candidates", "deleted_by_ignore_rule", "frames_with_a_deleted_candidate", "submitted",
                                                          "frames_refused_by_the_frozen_ranking")}
                                      for u in units},
                       "no_per_frame_interval": "frames of one sequence are strongly correlated; no per-frame confidence interval is computed",
                       "resolution_caveat": RESOLUTION_CAVEAT}}

    B.PHASE["phase"] = "tables/readings"
    per_sequence = {}
    for q in C.TEST_SEQUENCES:
        a, b = values("FG", lambda x: x["per_sequence"][q]["metrics"]), values("L0", lambda x: x["per_sequence"][q]["metrics"])
        info = m[MAIN_UNITS[0]]["per_sequence"][q]
        per_sequence[q] = {"frames": info["frames"], "gt": info["gt"], "small_gt": info["small_gt"],
                           "in_the_16_with_ge_1000_small_gt": q in C.SEQUENCES_GE_1000_SMALL, "attributes": dat["attributes"][q],
                           "FG": {s: {k: a[s][k] for k in ("AP", "APS")} for s in SEEDS},
                           "L0": {s: {k: b[s][k] for k in ("AP", "APS")} for s in SEEDS},
                           "differences_pp": C.paired(a, b, metrics=("AP", "APS"))}
    signs = {}
    for s in SEEDS:
        d = [per_sequence[q]["differences_pp"]["per_seed"][s]["APS"] for q in C.SEQUENCES_GE_1000_SMALL]
        signs[s] = {"of": len(d), "positive": sum(v is not None and v > 0 for v in d), "negative": sum(v is not None and v < 0 for v in d),
                    "zero": sum(v is not None and v == 0 for v in d), "without_value": sum(v is None for v in d)}
    key = "without_" + C.DOMINANT_SEQUENCE
    without = {"frames": m[MAIN_UNITS[0]][key]["frames"], "gt": m[MAIN_UNITS[0]][key]["gt"], "small_gt": m[MAIN_UNITS[0]][key]["small_gt"],
               "endpoints": {u: {k: m[u][key]["metrics"][k] for k in C.MAIN} for u in MAIN_UNITS},
               "differences_pp": C.paired(values("FG", lambda x: x[key]["metrics"]), values("L0", lambda x: x[key]["metrics"]), metrics=C.MAIN)}
    attributes = {}
    for g in C.ALTITUDE + C.WEATHER:
        info = m[MAIN_UNITS[0]]["by_attribute"][g]
        attributes[g] = {"sequences": info["sequences"], "frames": info["frames"], "gt": info["gt"], "small_gt": info["small_gt"],
                         "endpoints": {u: (None if m[u]["by_attribute"][g]["metrics"] is None else
                                           {k: m[u]["by_attribute"][g]["metrics"][k] for k in C.MAIN}) for u in MAIN_UNITS},
                         "differences_pp": None if info["metrics"] is None else C.paired(
                             values("FG", lambda x: x["by_attribute"][g]["metrics"]), values("L0", lambda x: x["by_attribute"][g]["metrics"]),
                             metrics=C.MAIN)}

    B.PHASE["phase"] = "tables/coverage"
    idx, _ = selected_rows(ann["gt_offsets"], sel)
    xywh = ann["gt_xywh"][idx]
    area = (xywh[:, 2] * xywh[:, 3]).astype(np.float64)
    small = area <= C.SMALL_AREA
    g_table = np.load(B.pinned(geo["per_gt_g"]), allow_pickle=False)
    B.require(g_table.shape == (len(SEEDS), len(xywh)) and geo["per_gt_g_rows"] == [f"FG-s{s}" for s in SEEDS], "per-GT magnification table")
    coverage = {}
    for k, s in enumerate(SEEDS):
        g = g_table[k][small]
        ok = np.isfinite(g)
        iou = {arm: np.load(B.pinned(sc[f"{arm}-s{s}"]["coverage"]), allow_pickle=False) for arm in ("L0", "FG")}
        B.require(all(v.shape == (2, int(small.sum())) for v in iou.values()), "coverage tables")
        coverage[s] = {"small_GT": int(small.sum()), "small_GT_without_g": int((~ok).sum()),
                       "bins": C.bin_table(g[ok], area[small][ok], {arm: v[0][ok] for arm, v in iou.items()}, pairs=(("FG", "L0"),)),
                       "bins_after_ignore_rule": C.bin_table(g[ok], area[small][ok], {arm: v[1][ok] for arm, v in iou.items()},
                                                             pairs=(("FG", "L0"),))}
    three = {"header": C.THREE_CLASS_HEADER, "not_in_the_label": True, "classes": {}}
    for name in C.UAVDT_CLASSES.values():
        a, b = values("FG", lambda x: x["three_class"]["classes"][name]), values("L0", lambda x: x["three_class"]["classes"][name])
        three["classes"][name] = {"gt": a[0]["gt"], "small_gt": a[0]["small_gt"], "detector_classes": a[0]["detector_classes"],
                                  "FG": {s: {k: a[s][k] for k in ("AP", "APS")} for s in SEEDS},
                                  "L0": {s: {k: b[s][k] for k in ("AP", "APS")} for s in SEEDS},
                                  "differences_pp": C.paired(a, b, metrics=("AP", "APS"))}
    result.update(
        per_sequence={"sequences": per_sequence, "APS_difference_signs_among_the_16": signs,
                      "the_16": list(C.SEQUENCES_GE_1000_SMALL)},
        without_dominant_sequence={"sequence": C.DOMINANT_SEQUENCE, **without}, by_attribute=attributes,
        geometry={k: geo[k] for k in ("g", "uavdt", "uavdt_by_unit", "development", "canvas_L_sizes", "uavdt_annotation_sizes",
                                      "demand_maps_identical_across_the_three_F_geo_units", "gt_without_g", "identity_frames")},
        coverage={"definition": "recall of the small GT: best IoU over the vehicle candidates of the frame >= threshold, no "
                                "one-to-one matching, no score threshold; bins of R30-001 A1 on the magnification g of the F_geo map",
                  "candidates": "bins: the 300 vehicle candidates of the frame; bins_after_ignore_rule: those left by the ignore rule",
                  "per_seed": coverage},
        three_class=three, near_duplicates={k: dh[k] for k in ("definition", "uavdt_frames_hashed", "visdrone_images_hashed", "threshold",
                                                              "per_sequence", "min_hamming", "pairs_le_6", "pairs_le_6_file")})
    if reference:
        s15 = values("S15")
        cols = C.MAIN + BIN_METRICS
        result["reference_S15"] = {"record_only": "task section 6: not part of the label",
                                   "endpoints": {u: m[u]["overall"] for u in reference},
                                   "S15 - L0": C.paired(s15, l0, metrics=cols), "S15 - F_geo": C.paired(s15, fg, metrics=cols)}
    else:
        result["reference_S15"] = None

    B.PHASE["phase"] = "tables/checks"
    shared = sorted({p for u in units for st in (wired[u], inf[u])
                     for p in st["other_gpu_compute_processes_at_start"] + st["other_gpu_compute_processes_at_end"]})
    result["checks"] = {
        "1_wiring": {u: {k: wired[u][k] for k in ("passed", "images", "forwards", "records", "equal_to_stored_top600", "own_conversions")}
                     for u in units},
        "2_coordinates": {"gt_as_detections": {k: gtc[k] for k in ("passed", "max_coordinate_error_px", "variants", "explains_the_gap")},
                          "F_geo_round_trip": geo["round_trip"]},
        "3_data_identity": {"packages": dat["packages"], "image_package_directory": dat["image_package_directory"],
                            "images": dat["images"], "census_equal_to_selection_evidence": dat["census_equal_to_selection_evidence"],
                            "census": dat["census"]},
        "4_ignore_rule": dat["ignore_rule"]}
    result["workload"] = {
        "units": {u: {"wire_forwards": wired[u]["forwards"], "uavdt_forwards": inf[u]["forwards"], "frames": inf[u]["frames"],
                      "inference_seconds": inf[u]["inference_seconds"], "infer_stage_seconds": inf[u]["elapsed_seconds"],
                      "wire_stage_seconds": wired[u]["elapsed_seconds"], "peak_cuda_bytes": inf[u]["peak_cuda_bytes"]} for u in units},
        "uavdt_forwards": sum(inf[u]["forwards"] for u in units), "wire_forwards": sum(wired[u]["forwards"] for u in units),
        "updates": 0, "gpu_stage_seconds": sum(inf[u]["elapsed_seconds"] + wired[u]["elapsed_seconds"] for u in units),
        "other_gpu_compute_processes_seen": shared, "gpu_shared": bool(shared),
        "cpu_stage_seconds": {**{name: stage_closed(c, name)["elapsed_seconds"] for name in ("data", "gtcheck", "geometry", "dhash")},
                              **{f"score-{u}": sc[u]["elapsed_seconds"] for u in units}}}
    result["endpoints"] = {u: {k: c["units"][u][k] for k in ("arm", "seed", "kind", "epoch", "endpoint", "weights_sha256")} for u in units}
    result["demand"], result["mapping"] = c["demand"], c["mapping"]
    B.require(counts["forwards"] == 0, "tables must not run a model")
    B.write_new(out / "RESULT.json", result)
    return {"stage": "tables", "result": B.ref(out / "RESULT.json"), "label": label, "reference_arm": bool(reference),
            "main_differences_pp": {s: {k: main["per_seed"][s][k] for k in C.MAIN} for s in SEEDS}}


# ================================================================ CLI
STAGES = {"data": (data, "plain", False), "gtcheck": (gtcheck, "plain", False), "wire": (wire, "cuda", True),
          "infer": (infer, "cuda", True), "geometry": (geometry, "cpu", False), "score": (score, "cpu", True),
          "dhash": (dhash, "plain", False), "tables": (tables, "plain", False)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("prepare",) + tuple(STAGES))
    ap.add_argument("--round", required=True, type=Path)
    ap.add_argument("--release")
    ap.add_argument("--unit", choices=UNITS)
    args = ap.parse_args(argv)
    if args.action == "prepare":
        B.require(args.release and args.unit is None, "prepare needs --release and no --unit")
        return prepare(args.round, args.release)
    run, mode, per_unit = STAGES[args.action]
    B.require((args.unit is not None) == per_unit and args.release is None, f"{args.action}: --unit is {'required' if per_unit else 'not used'}")
    c = load_contract(args.round)
    name = f"{args.action}-{args.unit}" if per_unit else args.action
    out = Path(c["outputs"]) / name
    out.mkdir(exist_ok=False)
    start, cpu0 = time.monotonic(), time.process_time()
    counts = Counter(actual_updates=0, forwards=0, forward_attempts=0)
    try:
        if mode == "cuda":
            B.require(os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == "expandable_segments:True", "allocator conf")
            from r25_run import setup
            from s15_formal_train import bind_seed
            bind_seed(parse(args.unit)[1])
            setup("cuda:0")
            B.torch_mod().cuda.reset_peak_memory_stats()
        elif mode == "cpu":
            B.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", f"the {args.action} stage requires CUDA_VISIBLE_DEVICES=\"\"")
            from r25_run import setup
            setup("cpu")
        result = run(c, out, counts, args.unit) if per_unit else run(c, out, counts)
        B.require(isinstance(result, dict) and (mode != "plain" or "torch" not in sys.modules), f"{args.action}: stage result / torch in a plain stage")
        B.write_new(out / "CLOSED.json", {"status": "PASS", "action": args.action, "contract_sha256": c["contract_sha256"],
                                          "counts": dict(counts), "elapsed_seconds": time.monotonic() - start,
                                          "process_cpu_seconds": time.process_time() - cpu0, **result})
    except BaseException as exc:
        B.write_new(out / "STOP.json", {"status": "STOP", "phase": B.PHASE["phase"], "action": args.action, "unit": args.unit,
                                        "error": repr(exc)[:4000], "traceback": traceback.format_exc(),
                                        "counts": dict(counts), "elapsed_seconds": time.monotonic() - start})
        raise


if __name__ == "__main__":
    main()
