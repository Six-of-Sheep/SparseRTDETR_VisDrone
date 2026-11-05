"""Immutable TASK_S15_EXECUTE_001 preparation and detached, journaled single-arm S15 execution.

  freeze     --release release-NNN                 frozen git-archive release + manifest
  contract   --release release-NNN --attempt formal-NNN [--prior formal-MMM]
  cpu-gate   --contract C --kind train|eval --out DIR   CUDA-hidden CPU gate from the frozen release
  preflight / launch --train-gate G --eval-gate G / status --contract C
Stages run by the detached supervisor: accept -> train-S15 -> input-alignment -> evaluation.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
W = Path("/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace")
EXPERIMENTS = W / "experiments/localization-distribution"
ROUND = EXPERIMENTS / "s15-scale-baseline-20261105-001"
PYTHON = "/home/lyy/.cache/r0-seed0-e120-20261010-001/environment/bin/python"
GPU = "GPU-1faee6f0-1da7-4ede-2475-67a5a00274a8"
TASK_NAME = "TASK_S15_EXECUTE_001.md"
TASK_SHA = "e557ecf2209a4e5de42e35ec820edde37051100add577af64bba3305eac584b5"
PREP_MANIFEST = ROUND / "PREP_MANIFEST.json"
DATA_REF = EXPERIMENTS / "r18-tiny-positives-20261102-001/production-contract-002.json"
DATA_REF_SHA = "b0fe33198d8abc7f8b89a1cc9824de9abdba8721580932aa07be565ce40d896b"
HISTORY = EXPERIMENTS / "rtdetrv2-r18-common-protocol-seed0-e120-20261013-001/epoch-recovery-20261014-001"
PARENT = HISTORY / "formal/checkpoint-epoch-060.pth"
PARENT_SHA = "a2b6412e3c85de7592829f24aff09e355fdd11cb53798bc498bd04b882e1bf8a"
PARENT_BYTES = 323_460_289
ORIGINAL_SOURCE = HISTORY / "release-006/source-manifest.json"
ORIGINAL_SOURCE_SHA = "7cf1d34d38e536f9e11e76047b4d733db085c85624c9a6f4b22d0a10c2ff773d"
R25 = EXPERIMENTS / "r25-query-local-regression-20261104-001"
R25_CONTRACT = R25 / "formal-001-CONTRACT.json"
R25_CONTRACT_SHA256 = "178eeb821a2049e12c0ba0c21ebd2f4c628715f8c1358218aa60c9b240e32ece"  # canonical contract digest
R25_RELEASE_MANIFEST = R25 / "release-002/MANIFEST.json"
R25_RELEASE_MANIFEST_SHA = "31e6fb65ba0ef883b03afdeae459def5a25c528ef3264007d1501b21dc0ddb4c"
R25_COMPARISON = R25 / "formal-001/evaluation/COMPARISON.json"
R25_COMPARISON_SHA = "882dcfb0a312e23c4ea05bb63133043b176eafd10d6a09e4ebfe23af37848a9b"
R25_A0_ACCEPTANCE = R25 / "formal-001/accept/A0/ACCEPTANCE.json"
R25_A0_ACCEPTANCE_SHA = "ce1d9d85e1d5648c8769d36a94f0cb00e2319b8a1a5e0f64cff65213f8b43fda"
R25_A0_UPDATES = R25 / "formal-001/train-A0/UPDATES.jsonl"
R25_A0_UPDATES_SHA = "efc1606d336cbbccfa3b0abd53d8e24bf3c536ed054062c64dc962e6aaaf5172"
R25_A0_PER_CLASS = EXPERIMENTS / "r25-cached-followup-002/run-A-003/RESULT_R25_CHANNELS_002.json"
R25_A0_PER_CLASS_SHA = "bfcbeb49b0b67e887172662b6ae4ff8e64b4d829972444f8aba6acb02f7acf9c"
ALLOCATOR = "expandable_segments:True"
WORKLOAD = {
    "arm": "S15", "seed": 0, "epochs": [61, 120], "initial_update": 18240, "updates_per_epoch": 304,
    "formal_updates": 18240, "stop_update": 36480, "save_epochs": [60, 75, 90, 105, 120],
    "train_batch": 8, "accumulation": 2, "precision": "bfloat16", "clip_norm": 0.1, "input_hw": [1152, 2016],
    "activation_checkpoint": True,
    "checkpoint_segments": ["backbone.res_layers", "encoder.fpn_blocks", "encoder.pan_blocks"],
    "allocator_conf": ALLOCATOR,
    "acceptance": {"A0_768_plain_updates": 2, "A0_768_checkpoint_updates": 2, "S15_updates": 9,
                   "S15_training_forwards": 18, "EMA_inference_forwards": 4,
                   "success_path_updates": 13, "success_path_full_forwards": 30},
    "evaluation": {
        "epoch": 120, "weights": ["ema", "online"], "primary_weights": "ema", "view": "A", "images": 548,
        "batch": 4, "precision": "float32", "input_hw": [1152, 2016], "warmup_batches_per_endpoint": 4,
        "budgets": [600, 300], "score_calls": 4, "per_class_APL_budget": 600,
        "per_class_APL_extra_COCOeval_calls": 2,
    },
}
DECISION_RULE = {
    "comparison": "S15 EMA e120 minus R25 A0 EMA e120 (certified, pinned)",
    "primary": {"top600_APS": 0.50, "top600_AP75_small": 0.30},
    "protections": {"top600_AP": -0.10, "top300_AP": -0.20, "top300_APS": -0.20, "top600_APM": -0.50,
                    "top600_APL": -0.50},
    "pass": "single_seed_preliminary_net_benefit", "fail": "closed_fixed_config",
    "online": "reported in full, does not decide", "mechanism_gate": None, "originality_claim": None,
}
RESOURCE = {
    "reference": {"gpu_hours": "15-20 (estimate, not a cap)", "additional_bytes_estimate": 2_500_000_000},
    "artificial_resource_caps": None,
    "no_progress_seconds": 1800,
    "reserve_disk_bytes": 4 * 1024**3,
    "remaining_save_estimate_bytes": 3 * 1024**3,
}
GATE_MODES = ("train", "eval")
MODEL_FILES = (
    "src/sparse_rtdetr/protocol_r18/model.py",
    "src/sparse_rtdetr/protocol_r18/engine.py",
    "src/sparse_rtdetr/protocol_r18/geometry.py",
    "src/sparse_rtdetr/protocol_r18/data.py",
    "src/sparse_rtdetr/protocol_r18/checkpoint.py",
    "src/sparse_rtdetr/localization/data.py",
    "src/sparse_rtdetr/baseline/training_v2b.py",
    "src/sparse_rtdetr/baseline/training_v2b_data.py",
    "src/sparse_rtdetr/baseline/training_v2b_engine.py",
    "src/sparse_rtdetr/research/query_local_runtime.py",
    "src/sparse_rtdetr/r0/evaluation.py",
    "src/sparse_rtdetr/r0/evaluation_data.py",
    "tools/query_local/formal_data.py",
    "infer_seed.py",
)
S15_FILES = ("src/sparse_rtdetr/research/scale_baseline.py",)


def require(condition, error):
    if not condition:
        raise ValueError(error)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())


def journal(path, value):
    with Path(path).open("a", encoding="utf-8") as file:
        file.write(json.dumps(value, allow_nan=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def ref(path):
    path = Path(path).absolute()
    require(path.is_file() and not path.is_symlink(), f"not a regular file: {path}")
    return {"path": str(path), "sha256": file_sha(path), "bytes": path.stat().st_size}


def checked_ref(reference):
    actual = ref(reference["path"])
    require(actual == reference, f"material reference differs: {actual['path']}")
    return Path(actual["path"])


def pinned(path, sha, size=None):
    result = ref(path)
    require(result["sha256"] == sha and (size is None or result["bytes"] == size), f"pinned file mismatch: {path}")
    return result


def suffix(value, prefix):
    require(re.fullmatch(prefix + r"-[0-9]{3}", value), f"invalid {prefix} identity")
    return value


def source_paths(source):
    paths = [x for x in sorted(source.rglob("*")) if x.is_file() or x.is_symlink()]
    require(not any(p.is_symlink() for p in paths), "release contains a link")
    return paths


def freeze(release_name):
    release = ROUND / suffix(release_name, "release")
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip(),
            "worktree must be clean to freeze")
    task = pinned(ROUND / TASK_NAME, TASK_SHA)
    r25_manifest = pinned(R25_RELEASE_MANIFEST, R25_RELEASE_MANIFEST_SHA)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    release.mkdir()
    source = release / "source"
    source.mkdir()
    archive = subprocess.check_output(["git", "archive", commit], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for entry in tar.getmembers():
            p = source / entry.name
            require(p.resolve().is_relative_to(source) and (entry.isdir() or entry.isfile()),
                    f"unsafe archive member: {entry.name}")
        tar.extractall(source)
    rows = []
    for path in source_paths(source):
        relative = path.relative_to(source).as_posix()
        actual = ref(path)
        blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        require(hashlib.sha256(blob).hexdigest() == actual["sha256"], f"archive blob differs: {relative}")
        rows.append({"path": relative, "sha256": actual["sha256"], "bytes": actual["bytes"]})
    base = {row["path"]: row["sha256"] for row in read(r25_manifest["path"])["source_files"]}
    current = {row["path"]: row["sha256"] for row in rows}
    same = {p: base.get(p) == current.get(p) and p in current for p in MODEL_FILES}
    require(all(same.values()), f"native model/engine/data/evaluation computation differs from R25 A0 release: {same}")
    path = release / "MANIFEST.json"
    write_new(path, {"version": "S15-EXEC-001-release-v1", "task": task, "commit": commit,
                     "R25_A0_release_manifest": r25_manifest,
                     "native_files_identical_to_R25_A0_release": sorted(MODEL_FILES), "source_files": rows})
    print(json.dumps(ref(path)), flush=True)


def verify_source(source, manifest):
    indexed = {row["path"]: row for row in manifest["source_files"]}
    found = {p.relative_to(source).as_posix(): p for p in source_paths(source)}
    require(len(indexed) == len(manifest["source_files"]) and indexed.keys() == found.keys(),
            "source inventory differs from release manifest")
    for name, path in found.items():
        row = indexed[name]
        require(path.stat().st_size == row["bytes"] and file_sha(path) == row["sha256"], f"source byte identity differs: {name}")


def contract_name(attempt):
    return ROUND / (suffix(attempt, "formal") + "-CONTRACT.json")


def load_contract(path, *, check_source=True):
    path = Path(path).absolute()
    contract = read(path)
    binding = dict(contract)
    supplied = binding.pop("contract_sha256", None)
    require(supplied == canonical_sha(binding), "canonical contract digest differs")
    require(contract["version"] == "S15-EXEC-001-contract-v1" and contract["workload"] == WORKLOAD
            and contract["decision_rule"] == DECISION_RULE, "S15 formal workload or decision rule differs")
    require(contract["formal_identity"] == canonical_sha(contract["scientific_binding"]), "formal identity differs")
    require(contract["source_root"] == str(Path(contract["source_manifest"]["path"]).parent / "source"),
            "source manifest/root binding differs")
    require(contract["contract_path"] == str(path), "contract relocated")
    require(contract["attempt_root"] == str(ROUND / suffix(contract["attempt"], "formal")), "attempt path differs")
    require(contract["parent"]["path"] == str(PARENT) and contract["parent"]["sha256"] == PARENT_SHA,
            "parent path/digest differs")
    require(contract["resource_policy"] == RESOURCE and path == contract_name(contract["attempt"]),
            "resource policy or actual contract path differs")
    if check_source:
        manifest = read(checked_ref(contract["source_manifest"]))
        require(manifest["version"] == "S15-EXEC-001-release-v1" and manifest["commit"] == contract["source_commit"]
                and manifest["task"] == contract["task"], "contract release identity differs")
    return contract


def identity_for(manifest, task, parent, data, evaluation_policy, reference):
    sources = {row["path"]: row["sha256"] for row in manifest["source_files"]}
    return {"task": task, "parent": parent, "data": data, "evaluation_policy": evaluation_policy,
            "native_computation": {name: sources[name] for name in MODEL_FILES},
            "s15_canvas_and_memory_implementation": {name: sources[name] for name in S15_FILES},
            "workload": WORKLOAD, "decision_rule": DECISION_RULE, "A0_reference": reference}


def a0_reference():
    comparison_ref = pinned(R25_COMPARISON, R25_COMPARISON_SHA)
    comparison = read(comparison_ref["path"])
    require(comparison["contract_sha256"] == R25_CONTRACT_SHA256, "R25 comparison contract differs")
    metrics = {kind: {b: comparison["metrics"][kind]["A0"][b] for b in ("600", "300")} for kind in ("EMA", "online")}
    for kind in metrics:
        for b, r in metrics[kind].items():
            checked_ref(r)
    endpoint = comparison["inference"]["EMA"]["A0"]["checkpoint"]
    require(endpoint == comparison["inference"]["online"]["A0"]["checkpoint"], "R25 A0 endpoint differs across kinds")
    checked_ref(endpoint)
    return {"R25_contract": ref(R25_CONTRACT), "R25_contract_sha256": R25_CONTRACT_SHA256,
            "A0_comparison": comparison_ref, "A0_metrics": metrics, "A0_endpoint": endpoint,
            "A0_acceptance": pinned(R25_A0_ACCEPTANCE, R25_A0_ACCEPTANCE_SHA),
            "A0_updates": pinned(R25_A0_UPDATES, R25_A0_UPDATES_SHA),
            "A0_per_class": pinned(R25_A0_PER_CLASS, R25_A0_PER_CLASS_SHA)}


def contract(release_name, attempt, prior_name):
    task = pinned(ROUND / TASK_NAME, TASK_SHA)
    release = ROUND / suffix(release_name, "release")
    manifest_ref = ref(release / "MANIFEST.json")
    manifest = read(manifest_ref["path"])
    require(manifest["version"] == "S15-EXEC-001-release-v1" and manifest["task"] == task, "wrong release for S15")
    data_ref = pinned(DATA_REF, DATA_REF_SHA)
    old = read(data_ref["path"])
    data = old["data"]
    eval_policy = old["evaluation_policy"]
    parent = pinned(PARENT, PARENT_SHA, PARENT_BYTES)
    reference = a0_reference()
    scientific = identity_for(manifest, task, parent, data, eval_policy, reference)
    formal_identity = canonical_sha(scientific)
    path = contract_name(attempt)
    prior = None
    if prior_name:
        require(prior_name != attempt, "cannot continue the same attempt")
        old_contract = load_contract(contract_name(prior_name))
        require(old_contract["formal_identity"] == formal_identity, "prior science identity differs")
        prior_path = Path(old_contract["attempt_root"])
        require((prior_path / "STOP.json").is_file() and not (prior_path / "CLOSED.json").exists(),
                "continuation requires a stopped, nonclosed prior attempt")
        require(old_contract["source_manifest"]["sha256"] != manifest_ref["sha256"],
                "continuation requires the specific repaired release")
        prior = {"contract": ref(contract_name(prior_name)), "stop": ref(prior_path / "STOP.json")}
    value = {
        "version": "S15-EXEC-001-contract-v1", "attempt": attempt, "contract_path": str(path),
        "attempt_root": str(ROUND / attempt), "prior": prior, "task": task, "formal_identity": formal_identity,
        "scientific_binding": scientific, "workload": WORKLOAD, "decision_rule": DECISION_RULE,
        "resource_policy": RESOURCE, "parent": parent, "parent_source": pinned(ORIGINAL_SOURCE, ORIGINAL_SOURCE_SHA),
        "prep_manifest": ref(PREP_MANIFEST), "source_root": str(release / "source"), "source_manifest": manifest_ref,
        "source_commit": manifest["commit"], "data_receipt": data_ref, "data": data,
        "development_ids": old["development_ids"], "evaluation_policy": eval_policy, "reference": reference,
        "environment": old["environment"], "python": PYTHON, "gpu_uuid": GPU,
        "outputs": {"accept": str(ROUND / attempt / "accept"), "train": str(ROUND / attempt / "train-S15"),
                    "evaluation": str(ROUND / attempt / "evaluation")},
    }
    value["contract_sha256"] = canonical_sha(value)
    write_new(path, value)
    print(json.dumps({"file": ref(path), "contract_sha256": value["contract_sha256"],
                      "formal_identity": formal_identity}), flush=True)


def validate_materials(c):
    started = time.monotonic()
    task = pinned(ROUND / TASK_NAME, TASK_SHA)
    parent = pinned(PARENT, PARENT_SHA, PARENT_BYTES)
    require(c["task"] == task and c["parent"] == parent, "task or parent changed")
    require(c["parent_source"] == pinned(ORIGINAL_SOURCE, ORIGINAL_SOURCE_SHA), "historical source identity changed")
    checked_ref(c["prep_manifest"])
    data_receipt = pinned(DATA_REF, DATA_REF_SHA)
    old = read(data_receipt["path"])
    require(c["data_receipt"] == data_receipt and c["data"] == old["data"]
            and c["development_ids"] == old["development_ids"] and c["evaluation_policy"] == old["evaluation_policy"]
            and c["environment"] == old["environment"] and c["python"] == PYTHON and c["gpu_uuid"] == GPU,
            "admitted materials or environment changed")
    checked_ref(c["evaluation_policy"])
    require(c["reference"] == a0_reference(), "A0 reference materials changed")
    manifest = read(checked_ref(c["source_manifest"]))
    require(manifest["version"] == "S15-EXEC-001-release-v1" and manifest["commit"] == c["source_commit"]
            and manifest["task"] == task, "release binding differs")
    require(c["scientific_binding"] == identity_for(manifest, task, parent, c["data"], c["evaluation_policy"],
                                                    c["reference"]), "science projection differs from bound materials")
    verify_source(Path(c["source_root"]), manifest)
    source = Path(c["source_root"])
    sys.path.insert(0, str(source / "src"))
    from sparse_rtdetr.localization.material_paths import MANIFESTS, metadata_path
    from sparse_rtdetr.data_protocol.schema import stable_image_id
    counts, data_bytes = {}, 0
    for role, (digest, count, split) in MANIFESTS.items():
        entry = c["data"][role]
        path = metadata_path(role, entry["manifest"], digest)
        rows = read(pinned(path, digest)["path"])["records"]
        ids = [row["stable_image_id"] for row in rows]
        require(len(ids) == len(set(ids)) == count, f"{role} member count or uniqueness mismatch")
        if role == "development":
            require(ids == c["development_ids"], "development order differs")
        roots = entry["roots"]
        require(roots["role"] == role, "data root role differs")
        for kind in ("images", "annotations"):
            directory = Path(roots[kind])
            require(directory.is_dir() and directory.resolve() == directory and directory.name == kind
                    and directory.parent.name == role and not any(p.is_symlink() for p in (directory, *directory.parents)),
                    "data root differs from staged role directory")
        for row in rows:
            require(row["split"] == split and row["stable_image_id"] == stable_image_id(split, row["relative_path"]),
                    "stable data identity mismatch")
            for kind, key, field in (("images", "relative_path", "image"),
                                     ("annotations", "annotation_relative_path", "annotation")):
                relative = Path(row[key])
                require(len(relative.parts) == 3 and relative.parts[:2] == (split, kind)
                        and relative.name not in {".", ".."}, "data path is not a bound role member")
                bound = pinned(Path(roots[kind]) / relative.name, row[field + "_sha256"], row[field + "_size_bytes"])
                data_bytes += bound["bytes"]
        counts[role] = count
    return {"source_files": len(manifest["source_files"]), "verified_role_counts": counts,
            "verified_data_bytes": data_bytes, "seconds": time.monotonic() - started}


def execution_environment(device, tmp):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update(CUDA_VISIBLE_DEVICES=device, CUBLAS_WORKSPACE_CONFIG=":4096:8", MKL_THREADING_LAYER="GNU",
               PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF=ALLOCATOR)
    if tmp:
        env.update(TMPDIR=str(tmp), TEMP=str(tmp), TMP=str(tmp))
    return env


def hidden_environment(c):
    code = """
import importlib.metadata as md, json, sys, torch, torchvision
print(json.dumps({'torch': torch.__version__.split('+')[0], 'torchvision': torchvision.__version__.split('+')[0],
 'cuda_build': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(), 'python': sys.executable,
 'pycocotools': md.version('pycocotools'), 'cuda_available': torch.cuda.is_available()}))
"""
    observed = json.loads(subprocess.check_output([c["python"], "-B", "-c", code], cwd=c["source_root"],
                                                  env=execution_environment("", ""), text=True))
    require(all(observed[key] == value for key, value in c["environment"].items()) and observed["python"] == c["python"]
            and not observed["cuda_available"], f"runtime environment mismatch: {observed}")
    return observed


def command_output(argv):
    return subprocess.check_output(argv, text=True, stderr=subprocess.STDOUT, timeout=15).strip()


def protect_disk(tmp=None, estimate=True):
    free = {"round": shutil.disk_usage(ROUND).free, "temporary": shutil.disk_usage(tmp or "/tmp").free,
            "shared_memory": shutil.disk_usage("/dev/shm").free}
    reserve = RESOURCE["reserve_disk_bytes"]
    required = reserve + (RESOURCE["remaining_save_estimate_bytes"] if estimate else 0)
    require(free["round"] > required, f"insufficient storage for safe next-stage write: {free}")
    require(free["temporary"] > reserve and free["shared_memory"] > 2 * 1024**3,
            f"insufficient temporary or DataLoader shared memory: {free}")
    return free


def resources():
    devices = command_output(["nvidia-smi", "--query-gpu=uuid,name,memory.total,memory.free,driver_version",
                              "--format=csv,noheader,nounits"])
    target = [line for line in devices.splitlines() if line.split(",")[0].strip() == GPU]
    require(len(target) == 1, "bound GPU not present")
    # Admission only: S15 needs essentially the whole device.  No allocator fraction or VRAM cap is set.
    require(int(target[0].split(",")[3].strip()) > 23000, f"bound GPU is not free: {target[0]}")
    apps = command_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                           "--format=csv,noheader,nounits"])
    require(not any(GPU in line for line in apps.splitlines()), f"bound GPU already in use: {apps}")
    return {"gpu": target[0], "compute_apps": apps, "free_bytes": protect_disk(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def preflight(path):
    c = load_contract(path)
    return {"status": "PASS", "contract_sha256": c["contract_sha256"], "source_commit": c["source_commit"],
            "materials": validate_materials(c), "environment": hidden_environment(c)}


def cpu_gate(path, kind, out):
    """Run one CUDA-hidden CPU gate from the frozen release (lowest priority)."""
    c = load_contract(path)
    out = Path(out).absolute()
    require(out.parent == ROUND and not out.exists(), "CPU gate output must be a new directory in the round")
    program, action = {"train": ("s15_formal_train.py", "cpu"), "eval": ("s15_formal_eval.py", "cpu-check")}[kind]
    argv = ["nice", "-n", "19", c["python"], "-B", str(Path(c["source_root"]) / "tools/scale_baseline" / program),
            action, "--contract", c["contract_path"], "--out", str(out)]
    env = execution_environment("", "")
    env["S15_CPU_THREADS"] = "8"
    started = time.monotonic()
    with (ROUND / (out.name + ".log")).open("x") as log:
        code = subprocess.call(argv, cwd=c["source_root"], env=env, stdin=subprocess.DEVNULL, stdout=log,
                               stderr=subprocess.STDOUT)
    print(json.dumps({"kind": kind, "returncode": code, "wall_seconds": time.monotonic() - started,
                      "out": str(out)}), flush=True)
    require(code == 0, f"CPU gate {kind} failed; see {out}")


def check_gate(path, c, mode):
    reference = ref(path)
    gate = read(path)
    require(gate["status"] == "PASS" and gate["contract_sha256"] == c["contract_sha256"]
            and (gate.get("mode") == mode or (mode == "train" and gate.get("action") == "cpu")),
            f"{mode} CPU gate identity or status mismatch")
    return reference


def process_alive(pid):
    try:
        stat = (Path("/proc") / str(pid) / "stat").read_text()
        return stat.rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return False


def latest_active():
    for directory in sorted(ROUND.glob("formal-[0-9][0-9][0-9]")):
        if not directory.is_dir() or (directory / "STOP.json").is_file() or (directory / "CLOSED.json").is_file():
            continue
        launch = directory / "LAUNCH.json"
        state = ("active" if process_alive(read(launch)["pid"]) else "uncertain terminal") if launch.is_file() \
            else "missing launch receipt"
        raise RuntimeError(f"existing unclosed formal attempt ({state}): {directory}")


def lock_round():
    handle = (ROUND / ".s15-formal.lock").open("a")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError("S15 formal run lock is held")
    return handle


def launch(path, train_gate, eval_gate):
    c = load_contract(path)
    attempt = Path(c["attempt_root"])
    with lock_round() as handle:
        latest_active()
        if not c["prior"]:
            require(not list(ROUND.glob("formal-[0-9][0-9][0-9]")),
                    "existing formal attempt; continuation requires its STOP and a new release")
        require(not attempt.exists(), "duplicate attempt")
        gates = {"train": check_gate(train_gate, c, "train"), "eval": check_gate(eval_gate, c, "eval")}
        verified = preflight(path)
        current = resources()
        attempt.mkdir()
        tmp = Path(tempfile.mkdtemp(prefix="s15-", dir="/tmp"))
        request = attempt / "REQUEST.json"
        write_new(request, {"contract": ref(path), "gates": gates, "preflight": verified, "resources": current,
                            "temporary_directory": str(tmp), "prior": c["prior"], "epoch_seconds": time.time()})
        argv = [c["python"], "-B", str(Path(c["source_root"]) / "tools/scale_baseline/s15_formal_run.py"),
                "_supervise", "--contract", c["contract_path"], "--lock-fd", str(handle.fileno()),
                "--request-sha", file_sha(request)]
        try:
            with (attempt / "supervisor.log").open("x") as log:
                child = subprocess.Popen(argv, cwd=c["source_root"], start_new_session=True, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, env=execution_environment(GPU, tmp),
                                         pass_fds=(handle.fileno(),))
        except BaseException as exc:
            write_new(attempt / "STOP.json", {"status": "STOP", "stage": "supervisor-spawn", "error": repr(exc),
                                              "contract_sha256": c["contract_sha256"]})
            raise
        receipt = {"status": "STARTED", "pid": child.pid, "argv": argv, "cwd": c["source_root"],
                   "contract_sha256": c["contract_sha256"], "request": ref(request), "epoch_seconds": time.time()}
        write_new(attempt / "LAUNCH.json", receipt)
        print(json.dumps(receipt), flush=True)


def close_stage(stage, c, action):
    path = stage / "CLOSED.json"
    row = read(path)
    require(row["status"] == "PASS" and row["contract_sha256"] == c["contract_sha256"]
            and (row.get("action") == action or (action == "evaluate" and row.get("mode") == "eval")),
            f"stage closure differs: {path}")
    return row


def endpoint_receipt(directory, c, epoch):
    path = directory / f"checkpoint-epoch-{epoch:03d}.pth"
    side = read(path.with_suffix(".pth.json"))
    require((side["status"], side["scope"], side["arm"], side["formal_identity"], side["contract_sha256"],
             side["global_update"]) == ("COMPLETE", "formal", "S15", c["formal_identity"], c["contract_sha256"], epoch * 304)
            and side["file"]["path"] == str(path), "checkpoint binding differs")
    checked_ref(side["file"])
    return path


def carry_from(c):
    """Accepted stages of stopped prior attempts with the same scientific identity."""
    if not c["prior"]:
        return None, None, None
    prior_path = checked_ref(c["prior"]["contract"])
    previous = load_contract(prior_path)
    stop = read(checked_ref(c["prior"]["stop"]))
    require(stop["contract_sha256"] == previous["contract_sha256"] and c["formal_identity"] == previous["formal_identity"],
            "prior STOP binding differs")
    path = Path(previous["attempt_root"])
    accept, completed, resume = carry_from(previous)
    if (path / "accept/CLOSED.json").exists():
        close_stage(path / "accept", previous, "accept")
        accept = ref(path / "accept/CLOSED.json")
    directory = path / "train-S15"
    if (directory / "CLOSED.json").exists():
        closed = close_stage(directory, previous, "train")
        require(closed["global_update"] == WORKLOAD["stop_update"], "prior completed S15 identity differs")
        endpoint_receipt(directory, previous, 120)
        completed, resume = {"path": str(directory), "contract": str(prior_path)}, None
        require(not list((path / "evaluation").glob("*-metrics.json")),
                "prior evaluation already scored; continuation must account those score calls")
    elif directory.exists() and any(directory.glob("checkpoint-epoch-*.pth.json")):
        eligible = [ep for ep in WORKLOAD["save_epochs"] if (directory / f"checkpoint-epoch-{ep:03d}.pth.json").exists()]
        resume = {"checkpoint": str(endpoint_receipt(directory, previous, max(eligible))), "contract": str(prior_path)}
    elif directory.exists() and (directory / "UPDATES.jsonl").exists():
        require((directory / "UPDATES.jsonl").stat().st_size == 0, "S15 had updates without a recoverable formal state")
    return accept, completed, resume


def input_alignment(train_root):
    """S15 order/augmentation draws equal R25 A0 row by row; canvas-dependent GT/target identities are counted."""
    same_fields = ("update", "epoch", "window", "image_ids_sha256", "augmentation_seeds_sha256", "augmentation_strong")
    differ_fields = ("gt_ids_sha256", "augmentation_trace_sha256", "augmented_samples_sha256", "input_tensor_sha256",
                     "target_sha256")
    start = WORKLOAD["initial_update"]
    counts = {k: 0 for k in differ_fields}
    digest, gt, n = hashlib.sha256(), 0, 0
    with (Path(train_root["path"]) / "UPDATES.jsonl").open() as mine, checked_ref(
            {"path": str(R25_A0_UPDATES), "sha256": R25_A0_UPDATES_SHA, "bytes": R25_A0_UPDATES.stat().st_size}).open() as a0:
        for line, other in zip(mine, a0):
            row, ref_row = json.loads(line), json.loads(other)["input"]
            n += 1
            require(row["input"]["update"] == start + n, "S15 audit cursor mismatch")
            require(all(row["input"][k] == ref_row[k] for k in same_fields), f"S15 order/draw differs from A0 at {start + n}")
            for k in differ_fields:
                counts[k] += row["input"][k] != ref_row[k]
            gt += row["gt_count"]
            digest.update(bytes.fromhex(canonical_sha(row["input"])))
    require(n == WORKLOAD["formal_updates"], "S15 input audit length differs")
    return {"updates": n, "order_and_draws_equal_R25_A0": list(same_fields), "rows_with_canvas_dependent_difference": counts,
            "S15_positive_GT_total_after_sanitize": gt, "input_sha256": digest.hexdigest(),
            "full_audit": ref(Path(train_root["path"]) / "UPDATES.jsonl"),
            "A0_audit": {"path": str(R25_A0_UPDATES), "sha256": R25_A0_UPDATES_SHA}}


def group_alive(pid):
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
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


def progress_time(path, log):
    files = [log, path / "UPDATES.jsonl", path / "progress.jsonl"]
    files.extend(path.glob("*/progress.jsonl"))
    return max(p.stat().st_mtime_ns for p in files if p.exists())


def run_stage(c, attempt, tmp, stage, program, action, extra, ledger):
    protect_disk(tmp)
    directory = attempt / stage
    log = attempt / f"{stage}.log"
    argv = [c["python"], "-B", str(ROOT / "tools/scale_baseline" / program), action,
            "--contract", c["contract_path"], "--out", str(directory), *extra]
    began = time.monotonic()
    row = {"stage": stage, "started_epoch_seconds": time.time(), "argv": argv}
    ledger.append(row)
    journal(attempt / "STAGES.jsonl", {"event": "START", **row})
    child = None
    try:
        with log.open("x") as stream:
            child = subprocess.Popen(argv, cwd=ROOT, env=execution_environment(GPU, tmp), start_new_session=True,
                                     stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
        stamp = progress_time(directory, log)
        last = time.monotonic()
        while child.poll() is None:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                protect_disk(tmp, estimate=False)
                updated = progress_time(directory, log)
                if updated != stamp:
                    stamp, last = updated, time.monotonic()
                require(time.monotonic() - last < RESOURCE["no_progress_seconds"], f"no progress in {stage}")
        require(child.returncode == 0, f"{stage} subprocess exited {child.returncode}; see {log}")
        deadline = time.monotonic() + 5
        while group_alive(child.pid) and time.monotonic() < deadline:
            time.sleep(.1)
        require(not group_alive(child.pid), f"{stage} left a live worker process")
        result = close_stage(directory, c, action)
        row.update({"status": "PASS", "result": ref(directory / "CLOSED.json")})
        return result
    except BaseException:
        row["status"] = "STOP"
        if (directory / "STOP.json").is_file():
            row["result"] = ref(directory / "STOP.json")
        if child is not None:
            stop_child(child)
        raise
    finally:
        row["seconds"] = time.monotonic() - began
        row["returncode"] = child.returncode if child is not None else None
        journal(attempt / "STAGES.jsonl", {"event": "FINISH", **row})


def byte_usage(path):
    return int(command_output(["du", "-sb", str(path)]).split()[0]) if Path(path).exists() else 0


def measurements(tmp):
    paths = {"experiment": ROUND, "temporary": tmp, "torch_cache": Path.home() / ".cache/torch",
             "torch_extensions": Path.home() / ".cache/torch_extensions", "triton_cache": Path.home() / ".triton"}
    result = {}
    for kind, path in paths.items():
        try:
            result[kind] = byte_usage(path)
        except (ValueError, OSError, subprocess.SubprocessError):
            result[kind] = None
    return result


def _supervise(path, lock_fd, request_sha):
    c = load_contract(path)
    attempt = Path(c["attempt_root"])
    require(ROOT == Path(c["source_root"]), "supervisor did not run from frozen source")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == GPU and os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == ALLOCATOR,
            "wrong visible GPU or allocator")
    os.fstat(lock_fd)
    req_path = attempt / "REQUEST.json"
    require(file_sha(req_path) == request_sha, "launch request bytes differ")
    req = read(req_path)
    checked_ref(req["contract"])
    for mode in GATE_MODES:
        checked_ref(req["gates"][mode])
    tmp = Path(req["temporary_directory"])
    require(os.environ.get("TMPDIR") == str(tmp), "short temporary path is not active")
    receipt = attempt / "LAUNCH.json"
    deadline = time.monotonic() + 10
    while not receipt.exists() and time.monotonic() < deadline:
        time.sleep(.1)
    require(read(receipt)["pid"] == os.getpid(), "attempt launcher PID differs")
    started = time.monotonic()
    ledger, stage = [], "initialize"
    before = measurements(tmp)

    def interrupt(signum, _):
        raise RuntimeError(f"supervisor signal {signum}")

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupt)
    result = {"contract_sha256": c["contract_sha256"], "source_commit": c["source_commit"],
              "formal_identity": c["formal_identity"]}
    try:
        accept, completed, resume = carry_from(c)
        write_new(attempt / "INHERITED.json", {"accept": accept, "train_root": completed, "resume": resume,
                                              "prior": c["prior"]})
        if accept is None:
            stage = "accept"
            row = run_stage(c, attempt, tmp, stage, "s15_formal_train.py", "accept", [], ledger)
            require(row["totals"]["optimizer_updates"] == WORKLOAD["acceptance"]["success_path_updates"]
                    and row["totals"]["full_model_forwards"] == WORKLOAD["acceptance"]["success_path_full_forwards"]
                    and row["A0_768_equivalence"]["status"] == "PASS" and row["formal_state_usable"] is False,
                    "acceptance work count or status differs")
        if completed is None:
            stage = "train-S15"
            args = ["--resume", resume["checkpoint"], "--prior-contract", resume["contract"]] if resume else []
            row = run_stage(c, attempt, tmp, stage, "s15_formal_train.py", "train", args, ledger)
            require(row["arm"] == "S15" and row["effective_updates"] == WORKLOAD["formal_updates"]
                    and row["global_update"] == WORKLOAD["stop_update"], "formal training terminal count differs")
            completed = {"path": str(attempt / stage), "contract": str(path)}
        stage = "input-alignment"
        write_new(attempt / "TRAIN_ROOT.json", completed)
        write_new(attempt / "INPUT_ALIGNMENT.json", input_alignment(completed))
        stage = "evaluation"
        row = run_stage(c, attempt, tmp, stage, "s15_formal_eval.py", "evaluate",
                        ["--train-root", str(attempt / "TRAIN_ROOT.json")], ledger)
        comparison = read(checked_ref(row["result"]))
        require(comparison["score_calls"] == WORKLOAD["evaluation"]["score_calls"], "formal evaluation score count differs")
        result.update(status="PASS", scientific_decision=comparison["scientific_decision"], comparison=row["result"])
    except BaseException as exc:
        result.update(status="STOP", stage=stage, error=repr(exc))
        raise
    finally:
        storage = measurements(tmp)
        result.update(wall_seconds=time.monotonic() - started, stages=ledger, storage_before_bytes=before,
                      storage_after_bytes=storage, prior=c["prior"],
                      storage_delta_bytes={k: storage[k] - before[k] if storage[k] is not None and before[k] is not None
                                           else None for k in storage})
        write_new(attempt / ("CLOSED.json" if result["status"] == "PASS" else "STOP.json"), result)


def supervise(path, lock_fd, request_sha):
    try:
        _supervise(path, lock_fd, request_sha)
    except BaseException as exc:
        c = load_contract(path, check_source=False)
        attempt = Path(c["attempt_root"])
        if not (attempt / "STOP.json").exists() and not (attempt / "CLOSED.json").exists():
            write_new(attempt / "STOP.json", {"status": "STOP", "stage": "supervisor-initialization",
                                              "contract_sha256": c["contract_sha256"], "error": repr(exc)})
        raise


def status(path):
    c = load_contract(path)
    attempt = Path(c["attempt_root"])
    for name in ("CLOSED.json", "STOP.json"):
        if (attempt / name).is_file():
            return read(attempt / name)
    if not attempt.exists():
        return {"status": "NOT_STARTED"}
    if not (attempt / "LAUNCH.json").is_file():
        return {"status": "STARTUP_UNCERTAIN"}
    launch_receipt = read(attempt / "LAUNCH.json")
    return {"status": "RUNNING" if process_alive(launch_receipt["pid"]) else "TERMINAL_UNKNOWN", "launch": launch_receipt}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    for name in ("freeze", "contract"):
        prep = actions.add_parser(name)
        prep.add_argument("--release", required=True)
        if name == "contract":
            prep.add_argument("--attempt", required=True)
            prep.add_argument("--prior")
    gate = actions.add_parser("cpu-gate")
    gate.add_argument("--contract", type=Path, required=True)
    gate.add_argument("--kind", choices=GATE_MODES, required=True)
    gate.add_argument("--out", type=Path, required=True)
    for name in ("preflight", "launch", "status", "_supervise"):
        call = actions.add_parser(name)
        call.add_argument("--contract", type=Path, required=True)
        if name == "launch":
            call.add_argument("--train-gate", type=Path, required=True)
            call.add_argument("--eval-gate", type=Path, required=True)
        elif name == "_supervise":
            call.add_argument("--lock-fd", type=int, required=True)
            call.add_argument("--request-sha", required=True)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.release)
    elif args.action == "contract":
        contract(args.release, args.attempt, args.prior)
    elif args.action == "cpu-gate":
        cpu_gate(args.contract, args.kind, args.out)
    elif args.action == "preflight":
        print(json.dumps(preflight(args.contract)), flush=True)
    elif args.action == "launch":
        launch(args.contract.absolute(), args.train_gate, args.eval_gate)
    elif args.action == "status":
        print(json.dumps(status(args.contract)), flush=True)
    else:
        supervise(args.contract, args.lock_fd, args.request_sha)


if __name__ == "__main__":
    main()
