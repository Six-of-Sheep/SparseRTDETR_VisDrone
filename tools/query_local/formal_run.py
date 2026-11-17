"""Immutable TASK_R25_002 preparation and detached, journaled execution."""
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
ROUND = EXPERIMENTS / "r25-query-local-regression-20261104-001"
PYTHON = "/home/lyy/.cache/r0-seed0-e120-20261010-001/environment/bin/python"
GPU = "GPU-1faee6f0-1da7-4ede-2475-67a5a00274a8"
TASK_NAME = "TASK_R25_002.md"
TASK_SHA = "af6627780980869326b3b29bdbd17c32f3539bc90c87d8f43b197c796259e4b7"
DATA_REF = EXPERIMENTS / "r18-tiny-positives-20261102-001/production-contract-002.json"
DATA_REF_SHA = "b0fe33198d8abc7f8b89a1cc9824de9abdba8721580932aa07be565ce40d896b"
HISTORY = EXPERIMENTS / "rtdetrv2-r18-common-protocol-seed0-e120-20261013-001/epoch-recovery-20261014-001"
PARENT = HISTORY / "formal/checkpoint-epoch-060.pth"
PARENT_SHA = "a2b6412e3c85de7592829f24aff09e355fdd11cb53798bc498bd04b882e1bf8a"
ORIGINAL_SOURCE = HISTORY / "release-006/source-manifest.json"
ORIGINAL_SOURCE_SHA = "7cf1d34d38e536f9e11e76047b4d733db085c85624c9a6f4b22d0a10c2ff773d"
DIAGNOSTIC_MANIFEST = ROUND / "release-001/MANIFEST.json"
DIAGNOSTIC_SHA = "8e232c2d0e7e263c92956452b6206ade95a8e93015f99e508f600bd0aab76101"
WORKLOAD = {
    "arms": ["A0", "A1", "A2"],
    "seed": 0,
    "epochs": [61, 120],
    "initial_update": 18240,
    "updates_per_epoch": 304,
    "updates_per_arm": 18240,
    "total_updates": 54720,
    "stop_update": 36480,
    "save_epochs": [60, 75, 90, 105, 120],
    "acceptance_optimizer_updates": 9,
    "train_batch": 8,
    "accumulation": 2,
    "precision": "bfloat16",
    "clip_norm": 0.1,
    "evaluation": {
        "epoch": 120, "weights": ["ema", "online"], "primary_weights": "ema",
        "view": "A", "images": 548, "batch": 4, "precision": "float32",
        "warmup_batches_per_endpoint": 4, "budgets": [600, 300], "score_calls": 12,
    },
}
RESOURCE = {
    "reference": {"gpu_hours": 30, "wall_hours": 38, "additional_bytes": 6_100_000_000},
    "artificial_resource_caps": None,
    "no_progress_seconds": 1800,
    "reserve_disk_bytes": 4 * 1024**3,
    "remaining_save_estimate_bytes": 7 * 1024**3,
}
GATE_MODES = ("train", "eval")
MODEL_FILES = (
    "src/sparse_rtdetr/research/query_local.py",
    "src/sparse_rtdetr/research/query_local_runtime.py",
    "src/sparse_rtdetr/protocol_r18/model.py",
    "src/sparse_rtdetr/protocol_r18/engine.py",
    "src/sparse_rtdetr/localization/data.py",
    "src/sparse_rtdetr/baseline/training_v2b.py",
    "src/sparse_rtdetr/baseline/training_v2b_data.py",
)


def require(condition, error):
    if not condition:
        raise ValueError(error)


def file_sha(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hash_chunks(file)


def hash_chunks(file):
    digest = hashlib.sha256()
    for block in iter(lambda: file.read(1024**2), b""):
        digest.update(block)
    return digest.hexdigest()


def canonical_sha(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


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
    require(result["sha256"] == sha and (size is None or result["bytes"] == size),
            f"pinned file mismatch: {path}")
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
    diagnostic = pinned(DIAGNOSTIC_MANIFEST, DIAGNOSTIC_SHA)
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
    base = {row["path"]: row["sha256"] for row in read(diagnostic["path"])["source_files"]}
    current = {row["path"]: row["sha256"] for row in rows}
    require(all(base.get(p) == current.get(p) and p in current for p in MODEL_FILES),
            "R25 model/engine/data computation changed relative to diagnostic release")
    path = release / "MANIFEST.json"
    write_new(path, {"version": "R25-002-release-v1", "task": task, "commit": commit,
                     "R25_001_manifest": diagnostic, "source_files": rows})
    print(json.dumps(ref(path)), flush=True)


def verify_source(source, manifest):
    indexed = {row["path"]: row for row in manifest["source_files"]}
    found = {p.relative_to(source).as_posix(): p for p in source_paths(source)}
    require(len(indexed) == len(manifest["source_files"]) and indexed.keys() == found.keys(),
            "source inventory differs from release manifest")
    for name, path in found.items():
        row = indexed[name]
        require(path.stat().st_size == row["bytes"] and file_sha(path) == row["sha256"],
                f"source byte identity differs: {name}")


def contract_name(attempt):
    return ROUND / (suffix(attempt, "formal") + "-CONTRACT.json")


def load_contract(path, *, check_source=True):
    path = Path(path).absolute()
    contract = read(path)
    binding = dict(contract)
    supplied = binding.pop("contract_sha256", None)
    require(supplied == canonical_sha(binding), "canonical contract digest differs")
    require(contract["version"] == "R25-002-contract-v1" and contract["workload"] == WORKLOAD,
            "formal workload differs")
    require(contract["formal_identity"] == canonical_sha(contract["scientific_binding"]),
            "formal identity differs")
    require(contract["source_root"] == str(Path(contract["source_manifest"]["path"]).parent / "source"),
            "source manifest/root binding differs")
    require(contract["contract_path"] == str(path), "contract relocated")
    require(contract["attempt_root"] == str(ROUND / suffix(contract["attempt"], "formal")),
            "attempt path differs")
    require(contract["parent"]["path"] == str(PARENT) and contract["parent"]["sha256"] == PARENT_SHA,
            "parent path/digest differs")
    require(contract["resource_policy"] == RESOURCE and path == contract_name(contract["attempt"]),
            "resource policy or actual contract path differs")
    if check_source:
        manifest = read(checked_ref(contract["source_manifest"]))
        require(manifest["version"] == "R25-002-release-v1" and
                manifest["commit"] == contract["source_commit"] and manifest["task"] == contract["task"],
                "contract release identity differs")
    return contract


def identity_for(manifest, task, parent, data, evaluation_policy):
    sources = {row["path"]: row["sha256"] for row in manifest["source_files"]}
    return {
        "task": task, "parent": parent, "data": data, "evaluation_policy": evaluation_policy,
        "model_engine_and_augmentation": {name: sources[name] for name in MODEL_FILES},
        "workload": WORKLOAD,
    }


def contract(release_name, attempt, prior_name):
    task = pinned(ROUND / TASK_NAME, TASK_SHA)
    release = ROUND / suffix(release_name, "release")
    manifest_ref = ref(release / "MANIFEST.json")
    manifest = read(manifest_ref["path"])
    require(manifest["version"] == "R25-002-release-v1" and manifest["task"] == task,
            "wrong release for R25-002")
    data_ref = pinned(DATA_REF, DATA_REF_SHA)
    old = read(data_ref["path"])
    data = old["data"]
    eval_policy = old["evaluation_policy"]
    parent = pinned(PARENT, PARENT_SHA, 323_460_289)
    scientific = identity_for(manifest, task, parent, data, eval_policy)
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
        "version": "R25-002-contract-v1", "attempt": attempt,
        "contract_path": str(path), "attempt_root": str(ROUND / attempt), "prior": prior,
        "task": task, "formal_identity": formal_identity, "scientific_binding": scientific,
        "workload": WORKLOAD, "resource_policy": RESOURCE,
        "parent": parent, "parent_source": pinned(ORIGINAL_SOURCE, ORIGINAL_SOURCE_SHA),
        "source_root": str(release / "source"), "source_manifest": manifest_ref,
        "source_commit": manifest["commit"], "data_receipt": data_ref, "data": data,
        "development_ids": old["development_ids"], "evaluation_policy": eval_policy,
        "environment": old["environment"], "python": PYTHON, "gpu_uuid": GPU,
    }
    value["contract_sha256"] = canonical_sha(value)
    write_new(path, value)
    print(json.dumps({"file": ref(path), "contract_sha256": value["contract_sha256"],
                      "formal_identity": formal_identity}), flush=True)

def validate_materials(c):
    started = time.monotonic()
    task = pinned(ROUND / TASK_NAME, TASK_SHA)
    parent = pinned(PARENT, PARENT_SHA, 323_460_289)
    require(c["task"] == task and c["parent"] == parent, "task or parent changed")
    require(c["parent_source"] == pinned(ORIGINAL_SOURCE, ORIGINAL_SOURCE_SHA),
            "historical source identity changed")
    data_receipt = pinned(DATA_REF, DATA_REF_SHA)
    old = read(data_receipt["path"])
    require(c["data_receipt"] == data_receipt and c["data"] == old["data"] and
            c["development_ids"] == old["development_ids"] and
            c["evaluation_policy"] == old["evaluation_policy"] and
            c["environment"] == old["environment"] and c["python"] == PYTHON and c["gpu_uuid"] == GPU,
            "admitted materials or environment changed")
    checked_ref(c["evaluation_policy"])
    manifest = read(checked_ref(c["source_manifest"]))
    require(manifest["version"] == "R25-002-release-v1" and manifest["commit"] == c["source_commit"] and
            manifest["task"] == task, "release binding differs")
    require(c["scientific_binding"] == identity_for(manifest, task, parent, c["data"], c["evaluation_policy"]),
            "science projection differs from bound materials")
    verify_source(Path(c["source_root"]), manifest)
    source = Path(c["source_root"])
    sys.path.insert(0, str(source / "src"))
    from sparse_rtdetr.localization.material_paths import MANIFESTS, metadata_path
    from sparse_rtdetr.data_protocol.schema import stable_image_id
    counts = {}
    data_bytes = 0
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
            require(directory.is_dir() and directory.resolve() == directory and
                    directory.name == kind and directory.parent.name == role and
                    not any(p.is_symlink() for p in (directory, *directory.parents)),
                    "data root differs from staged role directory")
        for row in rows:
            require(row["split"] == split and
                    row["stable_image_id"] == stable_image_id(split, row["relative_path"]),
                    "stable data identity mismatch")
            for kind, key, field in (("images", "relative_path", "image"),
                                     ("annotations", "annotation_relative_path", "annotation")):
                relative = Path(row[key])
                require(len(relative.parts) == 3 and relative.parts[:2] == (split, kind) and
                        relative.name not in {".", ".."}, "data path is not a bound role member")
                bound = pinned(Path(roots[kind]) / relative.name, row[field + "_sha256"],
                               row[field + "_size_bytes"])
                data_bytes += bound["bytes"]
        counts[role] = count
    return {"source_files": len(manifest["source_files"]), "verified_role_counts": counts,
            "verified_data_bytes": data_bytes, "seconds": time.monotonic() - started}


def hidden_environment(c):
    env = execution_environment("", "")
    code = """
import importlib.metadata as md
import json, sys
import torch, torchvision
print(json.dumps({
    'torch': torch.__version__.split('+')[0],
    'torchvision': torchvision.__version__.split('+')[0],
    'cuda_build': torch.version.cuda,
    'cudnn': torch.backends.cudnn.version(),
    'python': sys.executable,
    'pycocotools': md.version('pycocotools'),
    'cuda_available': torch.cuda.is_available(),
}))
"""
    result = subprocess.check_output([c["python"], "-B", "-c", code], cwd=c["source_root"],
                                     env=env, text=True)
    observed = json.loads(result)
    require(all(observed[key] == value for key, value in c["environment"].items()) and
            observed["python"] == c["python"] and not observed["cuda_available"],
            f"runtime environment mismatch: {observed}")
    return observed


def command_output(argv):
    return subprocess.check_output(argv, text=True, stderr=subprocess.STDOUT, timeout=15).strip()


def storage_snapshot(tmp=None):
    free = {"round": shutil.disk_usage(ROUND).free,
            "temporary": shutil.disk_usage(tmp or "/tmp").free,
            "shared_memory": shutil.disk_usage("/dev/shm").free}
    return free


def protect_disk(tmp=None, estimate=True):
    free = storage_snapshot(tmp)
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
    # Admission headroom based on observed previous CUDA peaks and realistic overhead.
    # This neither sets an allocator fraction nor imposes a running VRAM cap.
    free_mib = int(target[0].split(",")[3].strip())
    require(free_mib > 16 * 1024, f"bound GPU has insufficient available memory: {target[0]}")
    apps = command_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                           "--format=csv,noheader,nounits"])
    require(not any(GPU in line for line in apps.splitlines()), f"bound GPU already in use: {apps}")
    return {"gpu": target[0], "compute_apps": apps, "free_bytes": protect_disk(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def execution_environment(device, tmp):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update(CUDA_VISIBLE_DEVICES=device, CUBLAS_WORKSPACE_CONFIG=":4096:8",
               MKL_THREADING_LAYER="GNU", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               PYTHONUNBUFFERED="1")
    if tmp:
        env.update(TMPDIR=str(tmp), TEMP=str(tmp), TMP=str(tmp))
    return env


def preflight(path):
    c = load_contract(path)
    return {"status": "PASS", "contract_sha256": c["contract_sha256"],
            "source_commit": c["source_commit"], "materials": validate_materials(c),
            "environment": hidden_environment(c)}


def check_gate(path, c, mode):
    reference = ref(path)
    gate = read(path)
    require(gate["status"] == "PASS" and gate["contract_sha256"] == c["contract_sha256"] and
            (gate.get("mode") == mode or (mode == "train" and gate.get("action") == "cpu")),
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
        if not directory.is_dir():
            continue
        terminal = (directory / "STOP.json").is_file() or (directory / "CLOSED.json").is_file()
        if terminal:
            continue
        launch = directory / "LAUNCH.json"
        if launch.is_file():
            pid = read(launch)["pid"]
            state = "active" if process_alive(pid) else "uncertain terminal"
        else:
            state = "missing launch receipt"
        raise RuntimeError(f"existing unclosed formal attempt ({state}): {directory}")


def lock_round():
    handle = (ROUND / ".r25-formal.lock").open("a")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError("R25 formal run lock is held")
    return handle


def launch(path, train_gate, eval_gate):
    c = load_contract(path)
    attempt = Path(c["attempt_root"])
    with lock_round() as handle:
        latest_active()
        if not c["prior"]:
            require(not list(ROUND.glob("formal-[0-9][0-9][0-9]")),
                    "existing formal attempt; continuation requires its STOP and new release")
        require(not attempt.exists(), "duplicate attempt")
        gates = {"train": check_gate(train_gate, c, "train"), "eval": check_gate(eval_gate, c, "eval")}
        verified = preflight(path)
        current_resources = resources()
        attempt.mkdir()
        tmp = Path(tempfile.mkdtemp(prefix="r25-", dir="/tmp"))
        request = attempt / "REQUEST.json"
        write_new(request, {"contract": ref(path), "gates": gates, "preflight": verified,
                            "resources": current_resources, "temporary_directory": str(tmp),
                            "prior": c["prior"], "epoch_seconds": time.time()})
        argv = [c["python"], "-B", str(Path(c["source_root"]) / "tools/query_local/formal_run.py"),
                "_supervise", "--contract", c["contract_path"], "--lock-fd", str(handle.fileno()),
                "--request-sha", file_sha(request)]
        try:
            with (attempt / "supervisor.log").open("x") as log:
                child = subprocess.Popen(argv, cwd=c["source_root"], start_new_session=True,
                                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                         env=execution_environment(GPU, tmp), pass_fds=(handle.fileno(),))
        except BaseException as exc:
            write_new(attempt / "STOP.json", {"status": "STOP", "stage": "supervisor-spawn",
                                               "error": repr(exc), "contract_sha256": c["contract_sha256"]})
            raise
        receipt = {"status": "STARTED", "pid": child.pid, "argv": argv,
                   "cwd": c["source_root"], "contract_sha256": c["contract_sha256"],
                   "request": ref(request), "epoch_seconds": time.time()}
        write_new(attempt / "LAUNCH.json", receipt)
        print(json.dumps(receipt), flush=True)

def close_stage(stage, c, action):
    path = stage / "CLOSED.json"
    row = read(path)
    require(row["status"] == "PASS" and row["contract_sha256"] == c["contract_sha256"] and
            (row.get("action") == action or (action == "evaluate" and row.get("mode") == "eval")),
            f"stage closure differs: {path}")
    return row


def endpoint_receipt(directory, c, arm, epoch):
    path = directory / f"checkpoint-epoch-{epoch:03d}.pth"
    side = read(path.with_suffix(".pth.json"))
    require((side["status"], side["scope"], side["arm"], side["formal_identity"],
             side["contract_sha256"], side["global_update"]) ==
            ("COMPLETE", "formal", arm, c["formal_identity"], c["contract_sha256"], epoch * 304) and
            side["file"]["path"] == str(path), "checkpoint binding differs")
    checked_ref(side["file"])
    return path


def carry_from(c):
    if not c["prior"]:
        return None, {}, None
    prior_path = checked_ref(c["prior"]["contract"])
    previous = load_contract(prior_path)
    stop = read(checked_ref(c["prior"]["stop"]))
    require(stop["contract_sha256"] == previous["contract_sha256"] and
            c["formal_identity"] == previous["formal_identity"], "prior STOP binding differs")
    require(stop["stage"] != "evaluation",
            "preserve completed training; evaluation continuation requires accounting for prior score calls")
    path = Path(previous["attempt_root"])
    accept, completed, resume = carry_from(previous)
    if (path / "accept/CLOSED.json").exists():
        close_stage(path / "accept", previous, "accept")
        accept = ref(path / "accept/CLOSED.json")
    for arm in WORKLOAD["arms"]:
        directory = path / ("train-" + arm)
        if (directory / "CLOSED.json").exists():
            closed = close_stage(directory, previous, "train")
            require(closed["arm"] == arm and closed["global_update"] == WORKLOAD["stop_update"],
                    "prior completed arm identity differs")
            endpoint_receipt(directory, previous, arm, 120)
            completed[arm] = {"path": str(directory), "contract": str(prior_path)}
            resume = None
        elif directory.exists() and any(directory.glob("checkpoint-epoch-*.pth.json")):
            eligible = [epoch for epoch in WORKLOAD["save_epochs"]
                        if (directory / f"checkpoint-epoch-{epoch:03d}.pth.json").exists()]
            checkpoint = endpoint_receipt(directory, previous, arm, max(eligible))
            resume = {"arm": arm, "checkpoint": str(checkpoint), "contract": str(prior_path)}
        elif directory.exists() and (directory / "UPDATES.jsonl").exists():
            require((directory / "UPDATES.jsonl").stat().st_size == 0,
                    f"{arm} had updates without a recoverable formal state")
    return accept, completed, resume


def input_stream(train_roots):
    streams = {}
    start = WORKLOAD["initial_update"]
    for arm in WORKLOAD["arms"]:
        path = Path(train_roots[arm]["path"]) / "UPDATES.jsonl"
        digest = hashlib.sha256()
        count = 0
        with path.open(encoding="utf-8") as file:
            for line in file:
                row = json.loads(line)["input"]
                count += 1
                require(row["update"] == start + count, f"{arm} audit cursor mismatch")
                digest.update(bytes.fromhex(canonical_sha(row)))
        require(count == WORKLOAD["updates_per_arm"], f"{arm} input audit length differs")
        streams[arm] = {"updates": count, "input_sha256": digest.hexdigest(), "full_audit": ref(path)}
    require(len({v["input_sha256"] for v in streams.values()}) == 1,
            "across-arm input, GT or augmentation audit differs")
    return streams


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
    argv = [c["python"], "-B", str(ROOT / "tools/query_local" / program), action,
            "--contract", c["contract_path"], "--out", str(directory), *extra]
    began = time.monotonic()
    row = {"stage": stage, "started_epoch_seconds": time.time(), "argv": argv}
    ledger.append(row)
    journal(attempt / "STAGES.jsonl", {"event": "START", **row})
    child = None
    try:
        with log.open("x") as stream:
            child = subprocess.Popen(argv, cwd=ROOT, env=execution_environment(GPU, tmp),
                                     start_new_session=True, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT)
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
                require(time.monotonic() - last < RESOURCE["no_progress_seconds"],
                        f"no progress in {stage}")
        require(child.returncode == 0, f"{stage} subprocess exited {child.returncode}; see {log}")
        workers_deadline = time.monotonic() + 5
        while group_alive(child.pid) and time.monotonic() < workers_deadline:
            time.sleep(.1)
        require(not group_alive(child.pid), f"{stage} left a live worker process")
        result = close_stage(directory, c, action)
        row.update({"status": "PASS", "result": ref(directory / "CLOSED.json")})
        return result
    except BaseException:
        row["status"] = "STOP"
        stopped = directory / "STOP.json"
        if stopped.is_file():
            row["result"] = ref(stopped)
        if child is not None:
            stop_child(child)
        raise
    finally:
        row["seconds"] = time.monotonic() - began
        row["returncode"] = child.returncode if child is not None else None
        journal(attempt / "STAGES.jsonl", {"event": "FINISH", **row})


def byte_usage(path):
    if not Path(path).exists():
        return 0
    return int(command_output(["du", "-sb", str(path)]).split()[0])


def measurements(tmp):
    paths = {"experiment": ROUND, "temporary": tmp, "torch_cache": Path.home() / ".cache/torch",
             "torch_extensions": Path.home() / ".cache/torch_extensions",
             "triton_cache": Path.home() / ".triton"}
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
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == GPU, "wrong visible GPU")
    os.fstat(lock_fd)  # The detached supervisor holds the inherited round lock until exit.
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
    ledger = []
    stage = "initialize"
    before = measurements(tmp)

    def interrupt(signum, _):
        raise RuntimeError(f"supervisor signal {signum}")

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupt)
    result = {"contract_sha256": c["contract_sha256"], "source_commit": c["source_commit"],
              "formal_identity": c["formal_identity"]}
    try:
        accept, roots, resume = carry_from(c)
        write_new(attempt / "INHERITED.json", {"accept": accept, "train_roots": roots,
                                              "resume": resume, "prior": c["prior"]})
        if accept is None:
            stage = "accept"
            row = run_stage(c, attempt, tmp, stage, "formal_train.py", "accept", [], ledger)
            require(row["counts"]["actual_updates"] == WORKLOAD["acceptance_optimizer_updates"]
                    and row["counts"]["forwards"] == 18, "acceptance work count differs")
        for arm in WORKLOAD["arms"]:
            if arm in roots:
                continue
            stage = "train-" + arm
            args = ["--arm", arm]
            if resume and resume["arm"] == arm:
                args += ["--resume", resume["checkpoint"], "--prior-contract", resume["contract"]]
            row = run_stage(c, attempt, tmp, stage, "formal_train.py", "train", args, ledger)
            require(row["arm"] == arm and row["effective_updates"] == WORKLOAD["updates_per_arm"] and
                    row["global_update"] == WORKLOAD["stop_update"], "formal training terminal count differs")
            roots[arm] = {"path": str(attempt / stage), "contract": str(path)}
        stage = "input-alignment"
        write_new(attempt / "TRAIN_ROOTS.json", roots)
        write_new(attempt / "INPUT_ALIGNMENT.json", input_stream(roots))
        stage = "evaluation"
        row = run_stage(c, attempt, tmp, stage, "formal_eval.py", "evaluate",
                        ["--train-roots", str(attempt / "TRAIN_ROOTS.json")], ledger)
        comparison = read(checked_ref(row["result"]))
        require(comparison["score_calls"] == WORKLOAD["evaluation"]["score_calls"],
                "formal evaluation score count differs")
        result.update(status="PASS", scientific_decision=comparison["scientific_decision"],
                      comparison=row["result"])
    except BaseException as exc:
        result.update(status="STOP", stage=stage, error=repr(exc))
        raise
    finally:
        storage = measurements(tmp)
        storage_delta = {k: storage[k]-before[k] if storage[k] is not None and before[k] is not None else None
                         for k in storage}
        result.update(wall_seconds=time.monotonic()-started, stages=ledger, storage_before_bytes=before,
                      storage_after_bytes=storage, storage_delta_bytes=storage_delta, prior=c["prior"])
        write_new(attempt / ("CLOSED.json" if result["status"] == "PASS" else "STOP.json"), result)


def supervise(path, lock_fd, request_sha):
    try:
        _supervise(path, lock_fd, request_sha)
    except BaseException as exc:
        c = load_contract(path, check_source=False)
        attempt = Path(c["attempt_root"])
        if not (attempt / "STOP.json").exists() and not (attempt / "CLOSED.json").exists():
            write_new(attempt / "STOP.json", {
                "status": "STOP", "stage": "supervisor-initialization",
                "contract_sha256": c["contract_sha256"], "error": repr(exc),
            })
        raise


def status(path):
    c = load_contract(path)
    attempt = Path(c["attempt_root"])
    for name in ("CLOSED.json", "STOP.json"):
        terminal = attempt / name
        if terminal.is_file():
            return read(terminal)
    if not attempt.exists():
        return {"status": "NOT_STARTED"}
    receipt = attempt / "LAUNCH.json"
    if not receipt.is_file():
        return {"status": "STARTUP_UNCERTAIN"}
    launch = read(receipt)
    return {"status": "RUNNING" if process_alive(launch["pid"]) else "TERMINAL_UNKNOWN",
            "launch": launch}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    for name in ("freeze", "contract"):
        prep = actions.add_parser(name)
        prep.add_argument("--release", required=True)
        if name == "contract":
            prep.add_argument("--attempt", required=True)
            prep.add_argument("--prior")
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
