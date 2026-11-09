"""TASK_SQC_SEED12_REPLICATION_002 (SQC-002): the fixed SQC-001 recipe on the existing S15 seed1/seed2 e120 EMA detectors.

Adapter over sqc_run.py and shared_query_cls.py, both byte-identical to SQC-001. Only the detector identity changes:
checkpoint, EMA digest, parent contract and the same-seed B0/A0 references. Head initialisation (CPU generator seed 0),
head split, schedule, labels rule and all hyperparameters stay those of SQC-001. The old seed0 loader guards in
s15_cross_scale are not relaxed or edited; seed12 states are admitted only through the explicit checks below.

  freeze    --release release-NNN
  contract  --release release-NNN --seed 1|2
  unit | guard-check | cpu-check | gpu-check  --contract C --out DIR
  extract --role train | labels | fit | dev | score | judge   as in sqc_run (judge: fixed condition R, group contributions)
  run       --contract C --gates DIR...   one seed, sequential stages
  run-all                                 seed1 then seed2 unconditionally, then aggregate; never reads a metric to branch
  aggregate | collect
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools" / "query_local"), str(ROOT / "tools" / "scale_baseline"),
                str(ROOT)]

import sqc_run as B
from s15_formal_run import canonical_sha, file_sha, journal, pinned, read, ref, suffix, verify_source, write_new

require = B.require
EXP = B.EXP
ROUND = EXP / "shared-query-classification-seed12-20261109-001"
SQC1 = B.ROUND
S12 = EXP / "s15-seed12-replication-20261106-001"
FORMAL = S12 / "formal-001"
TASK_NAME = "TASK_SQC_SEED12_REPLICATION_002.md"
TASK_SHA = "5a37553e3867f7022f65f2ed9123702698468eb9e02a9fa91a604f2cfefcd489"
THIS_FILES = ("tools/scale_baseline/sqc_seed12_run.py", "tests/test_sqc_seed12.py")
RELEASE_VERSION = "SQC-002-release-v1"
CONTRACT_VERSION = "SQC-002-contract-v1"
SQC1_MANIFEST_SHA = "ae3d26caa9c0e25e27d4e7d6dfa8214c0d2039689f6c0ffe5020e27e2776211a"
SQC1_CONTRACT_SHA = "2090fc182ea76eed9312609ef4b18f22d853b16ad87d05aace4593f64f840018"
SQC1_SPLIT_SHA = "e0b5dbdf7d165cfa7d713d0ae631b5fefac0d655f69a3d380c8b9bec30ad7412"
SQC1_SCORE_SHA = "68c20c8a66a7b656be430c0ed22676e82f8e3ae438b06b28daf5ddbc8c44c0cd"
HEAD_INIT_SHA = "6eb7512314253c8cc6de113597ce3c1bef54b44ca8dd194ab142feeb6e00ec39"
HEAD_INIT_SEED = 0
S12_CONTRACT = S12 / "formal-001-CONTRACT.json"
S12_CONTRACT_FILE_SHA = "3b4858602267c7ef18f048b6f60928e9345b53aec6d6363e921f6c98e01e7883"
S12_CONTRACT_SHA = "41774c7ad926082d3c5d9bcc07adef9d1cd97707a292a56c4d377bde945c01a0"
SCHEMA = "s15-seed-complete-state-v1"
END = B.END
SEED_ORDER = (1, 2)
SEEDS = {
    1: {"checkpoint_sha256": "5b8b6e8b68c6450586313983651f716b9e9af7e0c30d40d0e881c958b7c78196",
        "ema_weights_sha256": "0dc59a9f7f2bedd1e1fbbc9ccadd0cd3b1e55250cac2c7c46729ad36085471cc",
        "formal_identity": "8362abca01aa0bb8f1196cadbe9ce5980eefc0483d06cae627430f8171cdc09a",
        "sidecar_sha256": "81732506e65a33f18857d212526763f01910bd1f6f65fee677e36548c22620f5",
        "parent_sha256": "6f8fd316a4e508b9cc8a76be2a18f7c5aaa0beb1b5bd809fd27cb83a1de05234",
        "comparison_sha256": "0e7b94fa422159bae3a908c9459c28463145db585d25be5250627149bd771dfb",
        "top600_metrics_sha256": "401d0679821c6aaf643af06fa3a2f5772b3db093621973d3b718b623275373fc",
        "top300_metrics_sha256": "95cb81c67f01904908a355349fe67bb69d8e53bd8fa928fb27679b87d1f5bce4",
        "top600_predictions_sha256": "87f15dbdce49f28b3444ea1ae2310af4dfe4832f71067d012abe3a92eaa0a0dd",
        "top300_predictions_sha256": "de65f69b7482eec1367da7ad3a84474d2358e96ce0323bb96f75bafaef3ac037"},
    2: {"checkpoint_sha256": "d5a3afbf4c26eed4d18d0dee5f88d92d1ad58d596530923c3b98b51443fc7936",
        "ema_weights_sha256": "e324442a732e3608c36702a9ad0ab18855ec8d0eb6fb3d643b739af7124c9022",
        "formal_identity": "fe07bbc611e7bc5ae508b47a921bbb1d984c1154ac79335452353e7b9f2d1487",
        "sidecar_sha256": "099899f96ecb3cdff90eb5cb0b22b8b3c50991f6840e7c9770145a1a94ecd72b",
        "parent_sha256": "c36e3c1fd4e5ad05cbbf8367d3e2b2c465858e0d69a233839c833ef29a200dd6",
        "comparison_sha256": "eafc37364535833acdfd5d36b8fbacc17425d2fe73859bbe6a4c839010d00149",
        "top600_metrics_sha256": "70dd828deb9a207dd06a5d3fea9ba3f8bbc8b0e9aefd8ec6ff3d4f48f9874849",
        "top300_metrics_sha256": "058ac044c1b8bf555cb76705bad073e923317991a63ddeb1dc607ab39a2ba969",
        "top600_predictions_sha256": "2baf949ae8a25d6aabc093a0cb0c1bb128977bca5baa5400c7ea89c2d8f30ca9",
        "top300_predictions_sha256": "8c834f4c4f63f563b5f0427104f2f694862d1edd4301f07e4e24544180cc46a6"},
}
# Condition R, frozen in the task before any seed1/2 endpoint exists; B2 minus the same seed's B0=S15@H.
R_CELLS = ((600, "APL", 0.50), (600, "AP", -0.10), (600, "APS", -0.20), (600, "AP75_small", -0.20), (600, "APM", -0.50),
           (300, "AP", -0.20), (300, "APS", -0.20))
# Group fixed by SQC-001's large-GT counts (GT<=12); never regrouped from new results.
SPARSE_CLASS_IDS = (2, 3, 7, 8, 10)
NAMED_CLASS_IDS = (4, 5, 6, 7)
OUTCOME_REPLICATED = "跨检测器训练seed的有限增量重复出现"
OUTCOME_NOT_ESTABLISHED = "本次所要求的稳定增量未建立"
BAD = (ValueError, RuntimeError, KeyError)


def workload(seed):
    w = json.loads(json.dumps(B.WORKLOAD))
    w["detector"] = f"seed{seed} S15 formal-001 e120 EMA; eval, FP32, frozen (0 updates)"
    w["timing"] = {"not_run": "deployment cost cites the SQC-001 seed0 measurement; not re-measured here (task section 4)"}
    return w


def decision_rule():
    return {"G": [list(x) for x in B.G_CELLS],
            "retention": {"cells": [list(x) for x in B.RETENTION], "fraction": B.RETENTION_FRACTION,
                          "denominator": "this seed's unrounded S15@H minus this seed's A0@L"},
            "I_B2_minus_B1": [list(x) for x in B.I_CELLS], "labels": B.DECISIONS,
            "R": {"requires_I": True, "B2_minus_B0": [list(x) for x in R_CELLS],
                  "both_new_seeds_must_pass": True, "replicated": OUTCOME_REPLICATED,
                  "not_established": OUTCOME_NOT_ESTABLISHED,
                  "scope": "research-investment condition for seeds 1 and 2 only; does not replace G/I, not applied to seed0"},
            "net_reference": "same seed native A0@L e120 EMA", "cross_scale_reference": "same seed B1",
            "scale_reference": "same seed B0=S15@H e120 EMA"}


# ================================================================ release / contract
def freeze(release_name):
    ROUND.mkdir(exist_ok=True)
    release = ROUND / suffix(release_name, "release")
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip(), "worktree must be clean")
    task = pinned(ROUND / TASK_NAME, TASK_SHA)
    base = pinned(SQC1 / "release-001/MANIFEST.json", SQC1_MANIFEST_SHA)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    release.mkdir()
    source = release / "source"
    source.mkdir()
    archive = subprocess.check_output(["git", "archive", commit], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for entry in tar.getmembers():
            p = source / entry.name
            require(p.resolve().is_relative_to(source) and (entry.isdir() or entry.isfile()), f"unsafe member {entry.name}")
        tar.extractall(source)
    rows = []
    for path in sorted(x for x in source.rglob("*") if x.is_file()):
        relative = path.relative_to(source).as_posix()
        actual = ref(path)
        blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        require(hashlib.sha256(blob).hexdigest() == actual["sha256"], f"archive blob differs: {relative}")
        rows.append({"path": relative, "sha256": actual["sha256"], "bytes": actual["bytes"]})
    current = {row["path"]: row["sha256"] for row in rows}
    old = {row["path"]: row["sha256"] for row in read(base["path"])["source_files"]}
    changed = sorted(p for p in set(old) | set(current) if old.get(p) != current.get(p))
    require(set(changed) == set(THIS_FILES), f"changes beyond the SQC-002 adapter: {changed}")
    path = release / "MANIFEST.json"
    write_new(path, {"version": RELEASE_VERSION, "task": task, "commit": commit, "SQC001_release_manifest": base,
                     "changed_vs_SQC001_release": changed, "source_files": rows})
    print(json.dumps(ref(path)), flush=True)


def seed_references(seed):
    """The same seed's certified S15@H outputs and native A0@L metrics; every link is pinned or cross-checked."""
    pin = SEEDS[seed]
    ev = FORMAL / f"seed{seed}/evaluation"
    comp_ref = pinned(ev / "COMPARISON.json", pin["comparison_sha256"])
    comp = read(comp_ref["path"])
    require((comp["seed"], comp["status"], comp["contract_sha256"], comp["formal_identity"]) ==
            (seed, "PASS", S12_CONTRACT_SHA, pin["formal_identity"]), f"seed{seed} COMPARISON identity")
    a0 = comp["A0_reference"]
    require(a0["seed"] == seed, f"seed{seed} A0 reference belongs to another seed")
    refs = {"S15_EMA_result": ref(ev / "S15-EMA/RESULT.json"),
            "S15_EMA_top600": pinned(ev / "S15-EMA/top600-predictions.json", pin["top600_predictions_sha256"]),
            "S15_EMA_top300": pinned(ev / "S15-EMA/top300-predictions.json", pin["top300_predictions_sha256"]),
            "S15@H_600": pinned(ev / f"S15-seed{seed}-EMA-A-top600-metrics.json", pin["top600_metrics_sha256"]),
            "S15@H_300": pinned(ev / f"S15-seed{seed}-EMA-A-top300-metrics.json", pin["top300_metrics_sha256"]),
            "A0@L_600": pinned(a0["A0_metrics"]["600"]["path"], a0["A0_metrics"]["600"]["sha256"]),
            "A0@L_300": pinned(a0["A0_metrics"]["300"]["path"], a0["A0_metrics"]["300"]["sha256"]),
            "S15_comparison": comp_ref}
    require(comp["metrics"] == {"600": refs["S15@H_600"], "300": refs["S15@H_300"]}, f"seed{seed} COMPARISON metrics binding")
    res = read(refs["S15_EMA_result"]["path"])
    require((res["seed"], res["kind"], res["epoch"], res["weights_sha256"], res["images"], res["batch_size"],
             res["contract_sha256"], res["checkpoint"]["sha256"]) ==
            (seed, "EMA", 120, pin["ema_weights_sha256"], 548, B.BATCH, S12_CONTRACT_SHA, pin["checkpoint_sha256"]),
            f"seed{seed} certified EMA readout binding")
    require(res["predictions"] == {"600": refs["S15_EMA_top600"], "300": refs["S15_EMA_top300"]},
            f"seed{seed} certified prediction binding")
    return refs


def verify_references(c, expect=None):
    require(c["references"] == (expect or seed_references(c["detector_seed"])), "references are not this seed's certified set")


def verify_split(c):
    require(c["split"] == pinned(SQC1 / "sqc-001-SPLIT.json", SQC1_SPLIT_SHA), "head split is not the SQC-001 file")


def verify_head_init(state=None):
    from sparse_rtdetr.research import shared_query_cls as Q
    require(Q.INIT_SEED == HEAD_INIT_SEED, "head initialisation seed changed")
    digest = Q.head_digest(Q.new_head(Q.initial_state() if state is None else state))
    require(digest == HEAD_INIT_SHA, f"head initial bytes differ from SQC-001: {digest}")
    return digest


def verify_head_contract(c):
    require(c["head_init"] == {"cpu_generator_seed": HEAD_INIT_SEED, "sha256": HEAD_INIT_SHA}, "contract head initialisation")


def verify_endpoint(c):
    seed, ep = c["detector_seed"], c["endpoint"]
    pin = SEEDS[seed]
    require((ep["file"]["sha256"], ep["sidecar"]["sha256"], ep["ema_weights_sha256"], ep["formal_identity"], ep["parent_sha256"],
             ep["schema"], ep["contract_sha256"], ep["train_canvas"]) ==
            (pin["checkpoint_sha256"], pin["sidecar_sha256"], pin["ema_weights_sha256"], pin["formal_identity"], pin["parent_sha256"],
             SCHEMA, S12_CONTRACT_SHA, "H"), f"seed{seed} endpoint pin")


def contract(release_name, seed):
    from sparse_rtdetr.research import shared_query_cls as Q
    require(seed in SEEDS, "only detector seeds 1 and 2 belong to SQC-002")
    release = ROUND / suffix(release_name, "release")
    manifest_ref = ref(release / "MANIFEST.json")
    manifest = read(manifest_ref["path"])
    task = pinned(ROUND / TASK_NAME, TASK_SHA)
    require(manifest["version"] == RELEASE_VERSION and manifest["task"] == task, "wrong release")
    verify_source(release / "source", manifest)
    s12_ref = pinned(S12_CONTRACT, S12_CONTRACT_FILE_SHA)
    s12 = read(s12_ref["path"])
    require(s12["contract_sha256"] == S12_CONTRACT_SHA and list(s12["seed_order"]) == list(SEED_ORDER), "S15 seed12 contract")
    spec, pin = s12["seeds"][str(seed)], SEEDS[seed]
    require((spec["seed"], spec["formal_identity"], spec["parent"]["sha256"]) ==
            (seed, pin["formal_identity"], pin["parent_sha256"]), f"seed{seed} entry of the S15 seed12 contract")
    ckpt = FORMAL / f"seed{seed}/train-S15/checkpoint-epoch-120.pth"
    endpoint = {"file": pinned(ckpt, pin["checkpoint_sha256"]),
                "sidecar": pinned(ckpt.with_suffix(".pth.json"), pin["sidecar_sha256"]), "contract": s12_ref,
                "contract_sha256": S12_CONTRACT_SHA, "formal_identity": pin["formal_identity"], "schema": SCHEMA,
                "ema_weights_sha256": pin["ema_weights_sha256"], "train_canvas": "H", "parent_sha256": pin["parent_sha256"]}
    old_ref = pinned(SQC1 / "sqc-001-CONTRACT.json", SQC1_CONTRACT_SHA)
    old = read(old_ref["path"])
    for key in ("data", "development_ids", "evaluation_policy", "environment", "python", "gpu_uuid"):
        require(old[key] == s12[key], f"{key} differs between SQC-001 and the S15 seed12 contract")
    references = seed_references(seed)
    a0 = read(references["S15_comparison"]["path"])["A0_reference"]
    require(a0 == spec["reference"], f"seed{seed} A0 reference differs from its S15 seed12 contract entry")
    split_ref = pinned(SQC1 / "sqc-001-SPLIT.json", SQC1_SPLIT_SHA)
    verify_head_init()
    budget = B.byte_budget()
    free = shutil.disk_usage(ROUND).free
    require(free > 2 * budget["total_estimate_bytes"] + budget["reserve_bytes"], f"disk free {free} below budget")
    root = ROUND / f"seed{seed}"
    value = {"version": CONTRACT_VERSION, "attempt": f"seed{seed}", "contract_path": str(ROUND / f"seed{seed}-CONTRACT.json"),
             "attempt_root": str(root), "task": task, "inherited_task_sha256": B.TASK_SHA, "SQC001_contract": old_ref,
             "SQC001_score": pinned(SQC1 / "sqc-001/score/RESULT.json", SQC1_SCORE_SHA),
             "workload": workload(seed), "library_version": Q.VERSION, "source_root": str(release / "source"),
             "source_manifest": manifest_ref, "source_commit": manifest["commit"], "detector_seed": seed,
             "head_init": {"cpu_generator_seed": HEAD_INIT_SEED, "sha256": HEAD_INIT_SHA}, "endpoint": endpoint,
             "S15_activation_checkpoint_flag": s12["workload"]["activation_checkpoint"], "references": references,
             "A0_reference": a0, "split": split_ref, "data": s12["data"], "development_ids": s12["development_ids"],
             "evaluation_policy": s12["evaluation_policy"], "environment": s12["environment"],
             "decision_rule": decision_rule(), "byte_budget": budget, "disk_free_at_contract": free, "python": B.PYTHON,
             "gpu_uuid": B.GPU}
    value["contract_sha256"] = canonical_sha(value)
    path = Path(value["contract_path"])
    write_new(path, value)
    print(json.dumps({"file": ref(path), "contract_sha256": value["contract_sha256"], "split": split_ref}), flush=True)


def load_contract(path):
    path = Path(path).absolute()
    c = read(path)
    body = dict(c)
    require(body.pop("contract_sha256") == canonical_sha(body), "contract canonical digest")
    seed = c["detector_seed"]
    require(seed in SEEDS and c["version"] == CONTRACT_VERSION and c["contract_path"] == str(path)
            and path == ROUND / f"seed{seed}-CONTRACT.json" and c["workload"] == workload(seed)
            and c["decision_rule"] == decision_rule(), "contract version/seed/workload/path/decision rule")
    require(c["task"] == pinned(ROUND / TASK_NAME, TASK_SHA), "task changed")
    manifest = read(pinned(c["source_manifest"]["path"], c["source_manifest"]["sha256"])["path"])
    require(manifest["commit"] == c["source_commit"] and Path(c["source_root"]) == ROOT, "stage must run from the release")
    verify_endpoint(c)
    verify_references(c)
    verify_split(c)
    verify_head_contract(c)
    return c


# ================================================================ explicit seed12 endpoint admission
def binding_of(side):
    return {k: v for k, v in side.items() if k not in ("status", "file")}


def binding_sha(side):
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    return hashlib.sha256(ck.canonical(binding_of(side))).hexdigest()


def check_side(side, c):
    ep, seed = c["endpoint"], c["detector_seed"]
    cur = side["cursor"]
    require((side["status"], side["schema"], side["scope"], side["arm"], side["seed"], side["global_update"],
             side["contract_sha256"], side["formal_identity"], side["parent_sha256"], cur["seed"], cur["end"],
             cur["completed_updates"], cur["completed_epochs"]) ==
            ("COMPLETE", ep["schema"], "formal", "S15", seed, END, ep["contract_sha256"], ep["formal_identity"],
             ep["parent_sha256"], seed, True, END, 120) and side["file"] == ep["file"], f"seed{seed} e120 sidecar identity")


def check_payload(payload, c, expected_binding_sha):
    from sparse_rtdetr.research import scale_baseline as S
    seed, ep = c["detector_seed"], c["endpoint"]
    require(payload.get("formal_binding_sha256") == expected_binding_sha, f"seed{seed} payload/sidecar binding digest")
    require((payload["version"], payload["arm"], payload["parent_sha256"], payload["config"], payload["activation_checkpoint"]) ==
            (S.VERSION, "S15", ep["parent_sha256"], S.production_config(seed).binding(), c["S15_activation_checkpoint_flag"]),
            f"payload is not the seed{seed} 1152x2016 S15 state")
    require(payload["engine"]["completed_updates"] == payload["ema"]["updates"] == END, f"seed{seed} payload is not e120")


def endpoint_payload(c, path=None):
    """Verified e120 complete state of this contract's seed; any other file, seed, clock or recipe is refused."""
    torch = B._torch()
    ep = c["endpoint"]
    path = Path(path or ep["file"]["path"]).absolute()
    require(ref(path) == ep["file"], f"endpoint file identity (wrong endpoint, seed or altered file): {path}")
    sidecar = path.with_suffix(".pth.json")
    require(ref(sidecar) == ep["sidecar"], "endpoint sidecar identity")
    side = read(sidecar)
    check_side(side, c)
    audit = side["audit"]
    h = hashlib.sha256()
    with Path(audit["path"]).open("rb") as f:
        h.update(f.read(audit["bytes"]))
    require(h.hexdigest() == audit["sha256"], "formal audit prefix")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    check_payload(payload, c, binding_sha(side))
    return payload


def load_models(c, device):
    """Same construction as sqc_run.load_models; only the admission of the complete state is seed specific."""
    import gc
    import s15_cross_scale as X
    xc = B._xc(c)
    payload = endpoint_payload(c)
    state = X.ema_state(payload, xc, "S15")
    mh, rh = X.build_cell(xc, "HH", state)
    ml, rl = X.build_cell(xc, "HL", state)
    require(rh["loaded_model_sha256"] == c["endpoint"]["ema_weights_sha256"], "H model is not the certified EMA")
    require(rh["non_size_tensor_sha256"] == rl["non_size_tensor_sha256"], "H/L instances hold different learned tensors")
    del payload, state
    gc.collect()
    for m in (mh, ml):
        m.eval().requires_grad_(False)
        m.to(device)
    return mh, ml, {"H": rh, "L": rl}


def bind():
    B.load_models = load_models


# ================================================================ affected checks
@B.stage
def unit(c, out):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run([B.PYTHON, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_sqc_seed12.py"],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
    (out / "pytest.log").write_text(proc.stdout + proc.stderr)
    require(proc.returncode == 0, f"unit tests failed:\n{proc.stdout[-3000:]}")
    tail = [l for l in proc.stdout.splitlines() if "passed" in l]
    return {"pytest": tail[-1] if tail else proc.stdout[-200:], "log": ref(out / "pytest.log"),
            "file": ref(ROOT / "tests/test_sqc_seed12.py")}


@B.stage
def guard_check(c, out):
    """Real files: the admitted state, and every wrong identity named by the task refused."""
    import s15_cross_scale as X
    B.setup("cpu")
    seed = c["detector_seed"]
    other = 3 - seed
    opin = SEEDS[other]
    xc = B._xc(c)
    payload = endpoint_payload(c)
    state = X.ema_state(payload, xc, "S15")
    require(X.SEED0_PARENT == "a2b6412e3c85de7592829f24aff09e355fdd11cb53798bc498bd04b882e1bf8a", "seed0 parent pin in the old loader")
    side = read(c["endpoint"]["sidecar"]["path"])
    good_binding = binding_sha(side)
    refused = []

    def refuse(label, fn):
        try:
            fn()
        except BAD:
            refused.append(label)
            return
        raise AssertionError(f"guard accepted: {label}")

    own = FORMAL / f"seed{seed}/train-S15"
    a0 = c["A0_reference"]
    refuse(f"seed{other} e120 in the seed{seed} slot",
           lambda: endpoint_payload(c, FORMAL / f"seed{other}/train-S15/checkpoint-epoch-120.pth"))
    refuse("seed0 S15 e120 in this slot", lambda: endpoint_payload(c, B.S15_ENDPOINT["file"]))
    refuse("own e105", lambda: endpoint_payload(c, own / "checkpoint-epoch-105.pth"))
    refuse("own A0 e120", lambda: endpoint_payload(c, a0["A0_e120_file"]["path"]))
    refuse("unmodified seed0 loader on this seed's e120", lambda: X.endpoint_payload(xc, "S15"))

    tampered = (("other seed", {"seed": other}), ("schema", {"schema": "s15-complete-state-v1"}),
                ("scope", {"scope": "fixture"}), ("arm", {"arm": "A0"}), ("status", {"status": "PARTIAL"}),
                ("e105 clock", {"global_update": 31920}), ("other parent", {"parent_sha256": opin["parent_sha256"]}),
                ("other identity", {"formal_identity": opin["formal_identity"]}),
                ("parent contract", {"contract_sha256": B.S15_ENDPOINT["contract_sha256"]}),
                ("cursor seed", {"cursor": {**side["cursor"], "seed": other}}),
                ("cursor not at end", {"cursor": {**side["cursor"], "end": False}}),
                ("cursor epochs", {"cursor": {**side["cursor"], "completed_epochs": 105}}))
    for label, change in tampered:
        refuse(f"sidecar: {label}", lambda: check_side({**side, **change}, c))

    from sparse_rtdetr.research import scale_baseline as S
    broken = (("other seed recipe", {"config": S.production_config(other).binding()}),
              ("seed0 recipe", {"config": S.production_config(0).binding()}),
              ("768 recipe", {"config": S.Config(seed=seed).binding()}),
              ("other parent", {"parent_sha256": opin["parent_sha256"]}),
              ("memory mode", {"activation_checkpoint": not c["S15_activation_checkpoint_flag"]}),
              ("arm", {"arm": "A0"}), ("version", {"version": "other"}),
              ("binding digest", {"formal_binding_sha256": "0" * 64}),
              ("engine clock", {"engine": {**payload["engine"], "completed_updates": 31920}}),
              ("EMA clock", {"ema": {**payload["ema"], "updates": 36479}}))
    for label, change in broken:
        refuse(f"payload: {label}", lambda: check_payload({**payload, **change}, c, good_binding))

    refuse("online requested", lambda: X.ema_state(payload, xc, "S15", kind="online"))
    refuse("online weights in the EMA slot",
           lambda: X.ema_state({**payload, "ema": {**payload["ema"], "module": payload["model"]}}, xc, "S15"))
    refuse("other seed's EMA digest", lambda: X.ema_state(
        payload, {**xc, "endpoints": {"S15": {**c["endpoint"], "ema_weights_sha256": opin["ema_weights_sha256"]}}}, "S15"))

    expect = seed_references(seed)
    other_refs = seed_references(other)
    for key in ("S15_comparison", "S15_EMA_top600", "S15_EMA_top300", "S15@H_600", "S15@H_300", "A0@L_600", "A0@L_300"):
        refuse(f"reference {key} of seed{other}",
               lambda: verify_references({**c, "references": {**c["references"], key: other_refs[key]}}, expect))
    seed0_a0 = B.REFERENCES["A0@L_600"]
    refuse("R25 seed0 A0 as net reference", lambda: verify_references(
        {**c, "references": {**c["references"], "A0@L_600": pinned(seed0_a0[0], seed0_a0[1])}}, expect))

    refuse("seed0 endpoint pin", lambda: verify_endpoint(
        {**c, "endpoint": {**c["endpoint"], "file": {**c["endpoint"]["file"], "sha256": B.S15_ENDPOINT["sha256"]}}}))
    refuse("split digest", lambda: verify_split({**c, "split": {**c["split"], "sha256": "0" * 64}}))
    refuse("another split file", lambda: verify_split({**c, "split": c["SQC001_contract"]}))
    refuse("head init seed in the contract", lambda: verify_head_contract(
        {**c, "head_init": {**c["head_init"], "cpu_generator_seed": seed}}))
    from sparse_rtdetr.research import shared_query_cls as Q
    init = {k: v.clone() for k, v in Q.initial_state().items()}
    init["fc1.weight"][0, 0] += 1e-3
    refuse("head init bytes", lambda: verify_head_init(init))
    digest = verify_head_init()

    refuse("seed0 SQC-001 extraction receipt", lambda: B.receipt(SQC1 / "sqc-001/extract-train", c))
    foreign = out / "foreign"
    foreign.mkdir()
    other_contract = read(ROUND / f"seed{other}-CONTRACT.json")["contract_sha256"]
    write_new(foreign / "RESULT.json", {"status": "PASS", "contract_sha256": other_contract})
    refuse(f"seed{other} cache receipt", lambda: B.receipt(foreign, c))
    return {"detector_seed": seed, "accepted": {"checkpoint": c["endpoint"]["file"], "schema": side["schema"],
                                                 "epoch": side["cursor"]["completed_epochs"], "global_update": side["global_update"],
                                                 "EMA_updates": payload["ema"]["updates"],
                                                 "EMA_tensors": len(state), "ema_weights_sha256": c["endpoint"]["ema_weights_sha256"]},
            "initial_head_sha256": digest, "refused": refused, "refused_count": len(refused), "foreign_receipt": ref(foreign / "RESULT.json")}


# ================================================================ fixed condition R and group contributions
def gate_r(diff_b2_b0, i_passed):
    cells = {f"top{b}_{m}": {"difference_pp": diff_b2_b0[str(b)][m], "minimum_pp": lim, "passed": diff_b2_b0[str(b)][m] >= lim}
             for b, m, lim in R_CELLS}
    failed = ([] if i_passed else ["I_not_passed"]) + [k for k, v in cells.items() if not v["passed"]]
    return {"requires_I": True, "I_passed": bool(i_passed), "cells": cells, "passed": not failed, "failed": failed}


def contributions(rows):
    """Ten-class macro contribution of the fixed groups; arithmetic only, the macro mean keeps every class."""
    by = {x["category_id"]: x for x in rows}
    require(sorted(by) == list(range(1, 11)), "ten category ids")
    require(all(by[k]["GT_large"] <= 12 for k in SPARSE_CLASS_IDS)
            and all(by[k]["GT_large"] > 12 for k in by if k not in SPARSE_CLASS_IDS), "fixed GT<=12 group differs from SQC-001")
    out = {}
    for key in ("B2-B0_pp", "B2-B1_pp", "B2-A0_pp", "B1-B0_pp"):
        require(all(by[k][key] is not None for k in by), f"{key} undefined for a class")
        c = {k: by[k][key] / 10. for k in by}
        out[key] = {"macro_total_pp": sum(c.values()), "GT_le_12_classes_pp": sum(c[k] for k in SPARSE_CLASS_IDS),
                    "other_classes_pp": sum(c[k] for k in c if k not in SPARSE_CLASS_IDS),
                    "per_class_macro_pp": {B.CLASS_NAMES[k]: c[k] for k in sorted(c)},
                    "named_class_difference_pp": {B.CLASS_NAMES[k]: by[k][key] for k in NAMED_CLASS_IDS}}
    return {"sparse_class_ids": list(SPARSE_CLASS_IDS), "GT_large": {B.CLASS_NAMES[k]: by[k]["GT_large"] for k in sorted(by)},
            "comparisons": out}


@B.stage
def judge(c, out):
    a = Path(c["attempt_root"])
    sr, sr_ref = B.receipt(a / "score", c)
    dv, dv_ref = B.receipt(a / "dev", c)
    fits = {}
    for arm in ("B1", "B2"):
        r, r_ref = B.receipt(a / f"fit-{arm}", c)
        require((r["arm"], r["updates"], r["detector_updates"], r["initial_head_sha256"]) == (arm, 2184, 0, HEAD_INIT_SHA),
                f"{arm} fit invariants")
        fits[arm] = {"fit": r_ref, "initial_head_sha256": r["initial_head_sha256"], "final_head_sha256": r["final_head_sha256"],
                     "updates": r["updates"], "validation_initial": r["validation_initial"]["global_loss"],
                     "validation_final": r["validation_final"]["global_loss"]}
    require(all(v["exact"] for v in dv["B0_vs_certified"].values()) and dv["detector_updates"] == 0, "dev invariants")
    require((sr["score_budget_calls"], sr["extra_COCOeval_calls"]) == (4, 2), "score counts")
    r_gate = gate_r(sr["differences_pp"]["B2-B0"], sr["I"]["passed"])
    macro = contributions(sr["per_class_APL_top600"])
    errors = {key: macro["comparisons"][key]["macro_total_pp"] - sr["differences_pp"][dk]["600"]["APL"]
              for key, dk in (("B2-B0_pp", "B2-B0"), ("B2-B1_pp", "B2-B1"), ("B2-A0_pp", "B2-A0@L"), ("B1-B0_pp", "B1-B0"))}
    return {"detector_seed": c["detector_seed"], "score": sr_ref, "dev": dv_ref, "fits": fits, "G_B2_passed": sr["G"]["B2"]["passed"],
            "G_B1_passed": sr["G"]["B1"]["passed"], "I_passed": sr["I"]["passed"], "R": r_gate,
            "per_seed_SQC001_label": sr["scientific_decision"], "contributions": macro, "macro_vs_bridge_error_pp": errors}


# ================================================================ supervisor
def chain(c):
    a = Path(c["attempt_root"])
    con = ["--contract", c["contract_path"]]
    return [
        ("gpu-check", True, ["gpu-check", *con, "--out", str(a / "gpu-check")]),
        ("extract-train", True, ["extract", *con, "--role", "train", "--out", str(a / "extract-train")]),
        ("labels", False, ["labels", *con, "--extract", str(a / "extract-train"), "--out", str(a / "labels")]),
        ("fit-B1", True, ["fit", *con, "--arm", "B1", "--extract", str(a / "extract-train"), "--labels", str(a / "labels"),
                          "--out", str(a / "fit-B1")]),
        ("fit-B2", True, ["fit", *con, "--arm", "B2", "--extract", str(a / "extract-train"), "--labels", str(a / "labels"),
                          "--out", str(a / "fit-B2")]),
        ("dev", True, ["dev", *con, "--fit-B1", str(a / "fit-B1"), "--fit-B2", str(a / "fit-B2"), "--out", str(a / "dev")]),
        ("score", False, ["score", *con, "--dev", str(a / "dev"), "--out", str(a / "score")]),
        ("judge", False, ["judge", *con, "--out", str(a / "judge")]),
    ]


def run(c, gates, start_at=None):
    a = Path(c["attempt_root"])
    a.mkdir(exist_ok=True)
    for g in gates:
        B.receipt(g, c)
    log = a / "STAGES.jsonl"
    steps = chain(c)
    names = [s[0] for s in steps]
    begin = names.index(start_at) if start_at else 0
    for name, _, _ in steps[:begin]:
        B.receipt(a / name, c)
    journal(log, {"event": "supervisor_start", "pid": os.getpid(), "time": time.time(), "start_at": names[begin],
                  "gates": [str(g) for g in gates]})
    for name, gpu, args in steps[begin:]:
        t = time.time()
        journal(log, {"event": "stage_start", "stage": name, "time": t})
        with (a / f"{name}.log").open("x") as f:
            proc = subprocess.run([B.PYTHON, "-B", str(ROOT / "tools/scale_baseline/sqc_seed12_run.py"), *args], cwd=ROOT,
                                  env=B._env(gpu), stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        journal(log, {"event": "stage_end", "stage": name, "returncode": proc.returncode, "seconds": time.time() - t})
        if proc.returncode != 0:
            write_new(a / "STOP.json", {"status": "STOP", "stage": name, "returncode": proc.returncode, "time": time.time()})
            return 1
    write_new(a / "CLOSED.json", {"status": "CLOSED_PASS", "stages": names[begin:], "time": time.time()})
    journal(log, {"event": "supervisor_end", "time": time.time()})
    return 0


def run_all():
    """Fixed order; the only branch is on a stage's return code, never on a metric."""
    for seed in SEED_ORDER:
        c = load_contract(ROUND / f"seed{seed}-CONTRACT.json")
        a = Path(c["attempt_root"])
        rc = run(c, [a / "unit", a / "guard-check", a / "cpu-check"])
        if rc != 0:
            return rc
    return subprocess.run([B.PYTHON, "-B", str(ROOT / "tools/scale_baseline/sqc_seed12_run.py"), "aggregate"], cwd=ROOT,
                          env=B._env(False), stdin=subprocess.DEVNULL).returncode


# ================================================================ three-seed table and return whitelist
COMPARISONS = ("B2-A0@L", "B1-A0@L", "B2-B0", "B1-B0", "B2-B1", "B0-A0@L (reference)")


def describe(tables):
    """Mean and sample SD of seeds 0/1/2 on the same 548 images: descriptive only, never a gate."""
    out = {}
    for comp in COMPARISONS:
        out[comp] = {}
        for budget in ("600", "300"):
            out[comp][budget] = {}
            for m in B.METRICS:
                v = [t[comp][budget][m] for t in tables]
                out[comp][budget][m] = {"mean_pp": statistics.fmean(v), "sample_sd_pp": statistics.stdev(v), "values_pp": v}
    return out


def aggregate():
    seeds = {}
    for seed in SEED_ORDER:
        c = load_contract(ROUND / f"seed{seed}-CONTRACT.json")
        a = Path(c["attempt_root"])
        require(read(a / "CLOSED.json")["status"] == "CLOSED_PASS", f"seed{seed} chain is not closed")
        sr, sr_ref = B.receipt(a / "score", c)
        jr, jr_ref = B.receipt(a / "judge", c)
        seeds[str(seed)] = {"contract_sha256": c["contract_sha256"], "score": sr_ref, "judge": jr_ref, "scores": sr["scores"],
                            "differences_pp": sr["differences_pp"], "G": sr["G"], "I": sr["I"], "R": jr["R"],
                            "per_seed_SQC001_label": sr["scientific_decision"], "S15_gain_pp": sr["S15_gain_pp"],
                            "per_class_APL_top600": sr["per_class_APL_top600"], "contributions": jr["contributions"]}
    s0 = read(pinned(SQC1 / "sqc-001/score/RESULT.json", SQC1_SCORE_SHA)["path"])
    seed0 = {"role": "observed discovery round, read-only; closed_fixed_config, G FAIL and I PASS unchanged; R not applied",
             "scores": s0["scores"], "differences_pp": s0["differences_pp"], "G": s0["G"], "I": s0["I"],
             "per_seed_SQC001_label": s0["scientific_decision"], "S15_gain_pp": s0["S15_gain_pp"],
             "per_class_APL_top600": s0["per_class_APL_top600"], "contributions": contributions(s0["per_class_APL_top600"])}
    both = all(seeds[str(s)]["R"]["passed"] for s in SEED_ORDER)
    summary = {"task": TASK_NAME, "seed0": seed0, "seeds": seeds,
               "R_by_seed": {k: v["R"]["passed"] for k, v in seeds.items()},
               "G_B2_by_seed": {"0": s0["G"]["B2"]["passed"], **{k: v["G"]["B2"]["passed"] for k, v in seeds.items()}},
               "I_by_seed": {"0": s0["I"]["passed"], **{k: v["I"]["passed"] for k, v in seeds.items()}},
               "both_new_seeds_pass_R": both, "registration": OUTCOME_REPLICATED if both else OUTCOME_NOT_ESTABLISHED,
               "continue_this_fixed_recipe": both,
               "descriptive_three_seed": describe([s0["differences_pp"], seeds["1"]["differences_pp"], seeds["2"]["differences_pp"]]),
               "descriptive_note": "same 548 development images; not independent data; head init and split held fixed"}
    write_new(ROUND / "SUMMARY.json", summary)
    print(json.dumps({"summary": ref(ROUND / "SUMMARY.json"), "registration": summary["registration"],
                      "R_by_seed": summary["R_by_seed"]}, ensure_ascii=False), flush=True)


def collect():
    """Small whitelisted files to ROUND/return-002 with SHA lists; large caches stay on the server (indexed)."""
    target = ROUND / "return-002"
    target.mkdir()
    picked = [ROUND / TASK_NAME, ROUND / "SUMMARY.json", ROUND / "release-001/MANIFEST.json"]
    picked += [ROUND / f"seed{s}-CONTRACT.json" for s in SEED_ORDER]
    small = ("RESULT.json", "IDENTITY.json", "PER_IMAGE_LIST_SHA256.json", "STEPS.jsonl", "HEAD-e8.pth", "pytest.log")
    for seed in SEED_ORDER:
        a = ROUND / f"seed{seed}"
        picked += [p for p in a.iterdir() if p.is_file()]
        for d in sorted(x for x in a.iterdir() if x.is_dir()):
            picked += [p for p in d.iterdir() if p.is_file() and (p.name in small or p.name.endswith("-metrics.json"))]
    rows = []
    for p in sorted(picked):
        rel = p.relative_to(ROUND)
        dst = target / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        require(file_sha(dst) == file_sha(p), f"copy differs: {rel}")
        rows.append((file_sha(dst), rel.as_posix(), dst.stat().st_size))
    (target / "RETURN_SHA256.txt").write_text("".join(f"{s}  {r}\n" for s, r, _ in rows))
    returned = {r for _, r, _ in rows}
    server = []
    for p in sorted(x for x in ROUND.rglob("*") if x.is_file()):
        rel = p.relative_to(ROUND)
        if rel.parts[0] in ("return-002", "release-001") or rel.as_posix() in returned:
            continue
        server.append((file_sha(p), rel.as_posix(), p.stat().st_size))
    (target / "SERVER_ONLY_SHA256.txt").write_text("".join(f"{s}  {n}  {r}\n" for s, r, n in server))
    print(json.dumps({"returned_files": len(rows), "returned_bytes": sum(n for _, _, n in rows), "server_only_files": len(server),
                      "server_only_bytes": sum(n for _, _, n in server)}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command")
    for name in ("--release", "--contract", "--out", "--role", "--arm", "--extract", "--labels", "--fit-B1", "--fit-B2", "--dev", "--start-at"):
        p.add_argument(name)
    p.add_argument("--seed", type=int)
    p.add_argument("--gates", nargs="*", default=[])
    a = p.parse_args()
    simple = {"freeze": lambda: freeze(a.release), "contract": lambda: contract(a.release, a.seed), "aggregate": aggregate,
              "collect": collect, "run-all": lambda: sys.exit(run_all())}
    if a.command in simple:
        return simple[a.command]()
    c = load_contract(a.contract)
    bind()
    fits = {"B1": a.fit_B1, "B2": a.fit_B2}
    commands = {"unit": lambda: unit(c, a.out), "guard-check": lambda: guard_check(c, a.out),
                "cpu-check": lambda: B.cpu_check(c, a.out), "gpu-check": lambda: B.gpu_check(c, a.out),
                "extract": lambda: B.extract(c, a.out, a.role), "labels": lambda: B.labels(c, a.out, a.extract),
                "fit": lambda: B.fit(c, a.out, a.arm, a.extract, a.labels), "dev": lambda: B.dev(c, a.out, fits),
                "score": lambda: B.score(c, a.out, a.dev), "judge": lambda: judge(c, a.out),
                "run": lambda: sys.exit(run(c, a.gates, a.start_at))}
    require(a.command in commands, f"unknown command {a.command}")
    commands[a.command]()


if __name__ == "__main__":
    main()
