"""R27-002 L0 control: shared layer of the runner (constants, contract, state guards, shared-field comparison).

L0 = same H input as R27/S15, FP32 bilinear to L inside the model, native L network, no local branch.
This module never imports r27_run (that module binds its own ROOT/ROUND at import time).
The round is the L0 round; a dev round is selected only through the L0_ROUND environment variable.
"""
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools" / "query_local"), str(ROOT / "tools" / "scale_baseline"), str(ROOT)]

WS = Path("/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace")
LD_ROOT = WS / "experiments/localization-distribution"
ROUND = Path(os.environ.get("L0_ROUND", str(LD_ROOT / "r27-l0-control-20261111-001")))  # dev rounds only via env
PARENT = LD_ROOT / ("rtdetrv2-r18-common-protocol-seed0-e120-20261013-001/epoch-recovery-20261014-001/formal/"
                    "checkpoint-epoch-060.pth")
PARENT_SHA = "a2b6412e3c85de7592829f24aff09e355fdd11cb53798bc498bd04b882e1bf8a"
PYTHON = "/home/lyy/.cache/r0-seed0-e120-20261010-001/environment/bin/python"
TASK = ROUND / "TASK_R27_L0_CONTROL_002.md"
TASK_SHA = "ea1d5b17afdab687f8d8b0bacc69877e81cb45bf5359dbc5f7faf0b405a7a6d1"

# original R27 round: read-only bindings
R27_ROUND = LD_ROOT / "r27-budgeted-local-memory-20261109-001"
R27_CONTRACT = R27_ROUND / "CONTRACT.json"
R27_CONTRACT_FILE_SHA = "e47400497049ef3493b09a2260d8d7be180d8bbd96c0b28e4c2e7e1f0ab0a121"
R27_CONTRACT_INNER_SHA = "2cff7f62b1a28a95956dba1940a94f209fefd9b451b5fb1be09ff061336772d7"
R27_RESULT = R27_ROUND / "summary/RESULT.json"
R27_RESULT_SHA = "b3825839491dc6b6552b4d1fb9ea131d1cfcad1749ee5ec35bd492cbc5cabecd"
R27_P_ENDPOINT = R27_ROUND / "formal-P/checkpoint-epoch-120.pth"
R27_P_ENDPOINT_SHA = "56f287f8bec09d6b4d1363de3ed7e38d9a348ba358213705b92fccee26d019a7"
R27_P_E060 = R27_ROUND / "formal-P/checkpoint-epoch-060.pth"
R27_UPDATES = {a: R27_ROUND / f"formal-{a}/UPDATES.jsonl" for a in ("P", "C")}
R27_METRICS = {a: {str(b): R27_ROUND / f"eval-{a}/R27-{a}-EMA-A-top{b}-metrics.json" for b in (600, 300)} for a in ("P", "C")}
R27_EVAL_CLOSED = {a: R27_ROUND / f"eval-{a}/CLOSED.json" for a in ("P", "C")}

SCHEMA = "r27-l0-state-v1"
ARM = "L0"
ENDPOINT = 36480
PARENT_UPDATE = 18240
WINDOWS_PER_EPOCH = 304
RECOVERY_EVERY = 5
SAVE_EPOCHS = tuple(range(60, 121, RECOVERY_EVERY))  # 13 full states, e060 included
FORMAL_SAVE_EPOCHS = (60, 75, 90, 105, 120)  # task-fixed subset of SAVE_EPOCHS (same files, not stored twice)
EVAL_BATCH, EVAL_WARMUP, EVAL_BATCHES = 4, 4, 137
BUDGETS = (600, 300)
COST_BATCH, COST_WARMUP, COST_PASSES, COST_BATCHES_PER_PASS = 4, 2, 3, 16
COST_ENDPOINTS = ("A0", "L0", "P", "S15")
COST_PER_ENDPOINT = COST_WARMUP + COST_PASSES * COST_BATCHES_PER_PASS
ROTATIONS = (("A0", "L0", "P", "S15"), ("P", "S15", "A0", "L0"), ("L0", "A0", "S15", "P"))
GATE_D = (("600", "APS", 0.50), ("600", "AP75_small", 0.30), ("600", "AP", -0.10), ("300", "AP", -0.20),
          ("300", "APS", -0.20), ("600", "APM", -0.50), ("600", "APL", -0.50))  # P - L0, pp
GATE_G = GATE_D  # the original R27 G rule (P - A0) has the same cells; used only for the prepare self-check
LABELS = {"D_passed": "同输入/日程下完整局部支路的单 seed 增量达到继续研究条件",
          "D_failed": "该局部支路增量未达到继续投入条件"}

INPUT_FIELDS = ("update", "epoch", "window", "image_ids_sha256", "gt_ids_sha256", "augmentation_seeds_sha256",
                "augmentation_trace_sha256", "augmented_samples_sha256", "input_tensor_sha256", "target_sha256",
                "augmentation_strong")
TOP_FIELDS = ("lr", "gt_count", "ema_updates")
SHARED_FIELDS = tuple(f"input.{k}" for k in INPUT_FIELDS) + TOP_FIELDS  # loss, gradient norm, seconds, monitor: not shared

WORKLOAD = {
    "cpu_fixture": {"updates": 3, "forwards": 6},
    "cuda_zero_step": {"updates": 0, "forwards": 4},
    "cuda_accept": {"updates": 3, "forwards": 6},
    "formal": {"updates": ENDPOINT - PARENT_UPDATE, "forwards": 2 * (ENDPOINT - PARENT_UPDATE),
               "save_epochs": list(SAVE_EPOCHS), "task_save_epochs": list(FORMAL_SAVE_EPOCHS)},
    "evaluation": {"endpoint": "L0 e120 EMA", "view": "A", "images": 548, "batch": EVAL_BATCH, "warmup": EVAL_WARMUP,
                   "forwards": EVAL_WARMUP + EVAL_BATCHES, "budgets": list(BUDGETS), "score_budget_calls": 2,
                   "per_class_APL_COCOeval_calls": 1, "online": "saved, not inferred or scored"},
    "cost4": {"endpoints": list(COST_ENDPOINTS), "batch": COST_BATCH, "warmup_per_endpoint": COST_WARMUP,
              "passes": COST_PASSES, "batches_per_pass": COST_BATCHES_PER_PASS,
              "forwards_per_endpoint": COST_PER_ENDPOINT, "forwards": len(COST_ENDPOINTS) * COST_PER_ENDPOINT,
              "rotations": [list(r) for r in ROTATIONS]},
}
WORKLOAD["normal_gpu_total"] = {
    "updates": WORKLOAD["cuda_zero_step"]["updates"] + WORKLOAD["cuda_accept"]["updates"] + WORKLOAD["formal"]["updates"],
    "forwards": sum(WORKLOAD[k]["forwards"] for k in ("cuda_zero_step", "cuda_accept", "formal", "evaluation", "cost4"))}
PHASE = {"phase": "initialize"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ref(path):
    p = Path(path).resolve()
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": file_sha(p)}


def pinned(r):
    require(ref(r["path"]) == r, f"pinned file differs: {r['path']}")
    return Path(r["path"])


def write_new(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("x") as f:
        f.write(json.dumps(data, indent=1, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.link(tmp, path)
    tmp.unlink()


def read_json(path):
    return json.loads(Path(path).read_text())


def torch_mod():
    import torch
    return torch


# ================================================================ shared audit fields (L0 vs R27 UPDATES.jsonl)
def shared(row):
    """The fields of one UPDATES.jsonl row that L0 and R27 must have in common, keyed as in SHARED_FIELDS."""
    require(set(row["input"]) == set(INPUT_FIELDS), ("input record keys differ from the declared shared set",
                                                     sorted(set(row["input"]) ^ set(INPUT_FIELDS))))
    out = {f"input.{k}": row["input"][k] for k in INPUT_FIELDS}
    out.update({k: row[k] for k in TOP_FIELDS})
    return out


def compare_shared(row, ref_row):
    a, b = shared(row), shared(ref_row)
    for k in SHARED_FIELDS:
        same = type(a[k]) is type(b[k]) and a[k] == b[k]
        require(same, f"shared field differs: {k} at update {a['input.update']}: {a[k]!r} != {b[k]!r}")
    return a


def cells(minuend, subtrahend, rule):
    out = {}
    for b, m, minimum in rule:
        d = 100. * (minuend[b][m] - subtrahend[b][m])
        out[f"top{b}_{m}"] = {"difference_pp": d, "minimum_pp": minimum, "passed": d >= minimum,
                              "minuend": minuend[b][m], "subtrahend": subtrahend[b][m]}
    return out


# ================================================================ contract
def source_manifest(root):
    files = {}
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts:
            files[str(p.relative_to(root))] = file_sha(p)
    return files


def side_of(path):
    return Path(path).with_suffix(".pth.json")


def reference_values(old, result, metrics):
    """P/C/S15/A0 reference readings with their provenance checks and the P-A0 self-check against RESULT.json."""
    values = result["values"]
    require(set(values) == {"P", "C", "S15", "A0"}, "RESULT.json values arms")
    for a in ("P", "C"):
        for b in ("600", "300"):
            bridge = read_json(metrics[a][b]["path"])["COCO_bridge"]["metrics"]
            require(all(bridge[k] == v for k, v in values[a][b].items()), f"RESULT values differ from {a} top{b} metrics file")
    require(values["S15"] == old["S15_EMA_values"] and values["A0"] == old["A0_EMA_values"],
            "RESULT S15/A0 values differ from the original contract")
    check = cells(values["P"], values["A0"], GATE_G)
    recorded = result["G_P_minus_A0"]["cells"]
    require(set(check) == set(recorded) and all(check[k]["difference_pp"] == recorded[k]["difference_pp"] for k in check),
            "P-A0 self-check differs from RESULT.json G_P_minus_A0")
    apl = result["per_class_APL_top600"]
    require(set(apl) == {"P", "C", "S15", "A0"} and all(len(v) == 10 for v in apl.values()), "per-class APL arms/classes")
    return {"values": values, "per_class_APL_top600": apl,
            "P_minus_A0_self_check": {"equal_to_RESULT_G_P_minus_A0": True,
                                      "difference_pp": {k: v["difference_pp"] for k, v in check.items()}}}


def prepare(_args=None):
    from sparse_rtdetr.research import r27_l0_control as L
    require(ROOT.resolve().parent.parent == ROUND.resolve() and ROOT.name == "source", "prepare must run from a frozen release")
    manifest = read_json(ROOT.parent / "MANIFEST.json")
    require(manifest["files"] == source_manifest(ROOT), "release source differs from MANIFEST")
    require(not manifest.get("dirty", False) or ROUND.resolve().name.startswith("dev-round"),
            "a release assembled from a dirty work tree is only accepted in a dev round")
    task = ref(TASK)
    require(task["sha256"] == TASK_SHA, "task file SHA")
    old_ref, result_ref = ref(R27_CONTRACT), ref(R27_RESULT)
    require(old_ref["sha256"] == R27_CONTRACT_FILE_SHA and result_ref["sha256"] == R27_RESULT_SHA, "R27 contract/result file SHA")
    old, result = read_json(R27_CONTRACT), read_json(R27_RESULT)
    require(old["contract_sha256"] == R27_CONTRACT_INNER_SHA == result["contract_sha256"], "R27 contract inner digest")
    require(old["parent"]["sha256"] == PARENT_SHA and old["python"] == PYTHON, "R27 contract parent/python")
    metrics = {a: {b: ref(p) for b, p in R27_METRICS[a].items()} for a in ("P", "C")}
    for k in ("A0_endpoint", "S15_endpoint"):
        pinned(old[k])
    for k in ("A0_metrics", "S15_metrics"):
        for b in ("600", "300"):
            pinned(old[k][b])
    parent = ref(PARENT)
    state_bytes = parent["bytes"]
    c = {"version": "R27-002-L0-contract-v1", "task": task, "parent": parent, "parent_update": PARENT_UPDATE,
         "R27_contract": {**old_ref, "contract_sha256": old["contract_sha256"], "identity_sha256": old["identity_sha256"]},
         "R27_result": result_ref,
         "P_endpoint": ref(R27_P_ENDPOINT), "P_endpoint_side": ref(side_of(R27_P_ENDPOINT)),
         "R27_P_e060": ref(R27_P_E060), "R27_P_e060_side": ref(side_of(R27_P_E060)),
         "R27_updates": {a: ref(p) for a, p in R27_UPDATES.items()},
         "R27_metrics": metrics, "R27_eval_closed": {a: ref(p) for a, p in R27_EVAL_CLOSED.items()},
         "A0_endpoint": old["A0_endpoint"], "A0_metrics": old["A0_metrics"],
         "S15_endpoint": old["S15_endpoint"], "S15_metrics": old["S15_metrics"],
         "data": old["data"], "data_receipt": old["data_receipt"], "development_ids": old["development_ids"],
         "monitor_rule": old["monitor_rule"], "monitor_ids": old["monitor_ids"], "monitor_sha256": old["monitor_sha256"],
         "evaluation_policy": old["evaluation_policy"], "environment": old["environment"], "python": old["python"],
         "gpu_uuid": old["gpu_uuid"],
         "reference": reference_values(old, result, metrics),
         "identity": L.IDENTITY, "identity_sha256": L.IDENTITY_SHA256,
         "source_root": str(ROOT.resolve()), "source_manifest": ref(ROOT.parent / "MANIFEST.json"),
         "source_commit": manifest["commit"], "source_dirty": bool(manifest.get("dirty", False)),
         "workload": WORKLOAD, "gates": {"D_P_minus_L0": GATE_D}, "rotations": [list(r) for r in ROTATIONS],
         "shared_fields": list(SHARED_FIELDS),
         "recipe": {"input_hw": [1152, 2016], "global_hw": [768, 1344], "train_batch": 8, "accumulation": 2,
                    "precision": "bfloat16", "clip": 0.1, "optimizer": "native AdamW groups, global LR clock",
                    "activation_checkpoint": True,
                    "checkpoint_segments": ["backbone.res_layers", "encoder.fpn_blocks", "encoder.pan_blocks"],
                    "allocator_conf": "expandable_segments:True"},
         "labels": LABELS,
         "resource_plan": {
             "storage_estimate_bytes": {"formal_states": len(SAVE_EPOCHS) * state_bytes, "acceptance_states": 2 * state_bytes,
                                        "cpu_fixture_states": 2 * state_bytes, "formal_audit": 200_000_000,
                                        "predictions_and_metrics": 160_000_000, "source_logs_and_small": 100_000_000},
             "state_bytes_basis": "parent e060 full state size (L0 has the native key set, no local branch)",
             "disk_free_bytes_at_prepare": shutil.disk_usage(ROUND).free,
             "gpu_time_reference": "task: about 8-12 h planning reference (not a cap or a promise)"},
         "outputs": str(ROUND), "formal_started": False}
    require(c["P_endpoint"]["sha256"] == R27_P_ENDPOINT_SHA, "P e120 endpoint SHA")
    require(c["monitor_sha256"] == sha_bytes(canonical(c["monitor_ids"])) and len(c["monitor_ids"]) == 64, "monitor ids")
    require(len(c["development_ids"]) == 548, "development ids")
    c["contract_sha256"] = sha_bytes(canonical(c))
    write_new(ROUND / "CONTRACT.json", c)
    print(json.dumps({"contract_sha256": c["contract_sha256"], "identity_sha256": L.IDENTITY_SHA256}))
    return c


def load_contract(path=None):
    c = read_json(path or ROUND / "CONTRACT.json")
    body = {k: v for k, v in c.items() if k != "contract_sha256"}
    require(c["contract_sha256"] == sha_bytes(canonical(body)), "contract digest")
    require(Path(c["source_root"]).resolve() == ROOT.resolve(), "not running from the bound release")
    require(Path(c["outputs"]).resolve() == ROUND.resolve(), "contract bound to another round")
    pinned(c["source_manifest"])
    pinned(c["parent"])
    from sparse_rtdetr.research import r27_l0_control as L
    require(c["identity_sha256"] == L.IDENTITY_SHA256, "L0 behaviour identity differs from the contract")
    return c


def stage_closed(c, name):
    """Precondition guard for later stages: <round>/<name>/CLOSED.json must be PASS under this contract."""
    path = ROUND / name / "CLOSED.json"
    require(path.is_file(), f"stage not closed: {path}")
    closed = read_json(path)
    require(closed.get("status") == "PASS" and closed.get("contract_sha256") == c["contract_sha256"]
            and closed.get("arm") == ARM, f"stage {name} is not PASS under this contract")
    return closed


# ================================================================ states (schema r27-l0-state-v1, arm L0 only)
def binding(c, e, scope, audit=None, role=None):
    from formal_data import cursor
    require(getattr(e, "l0_arm", None) == ARM and not hasattr(e, "r27_arm"), "engine is not an L0 engine")
    return {"schema": SCHEMA, "scope": scope, "arm": ARM, "contract_sha256": c["contract_sha256"],
            "identity_sha256": c["identity_sha256"], "parent_sha256": c["parent"]["sha256"],
            "global_update": e.completed_updates, "cursor": cursor(e.completed_updates), "config": e.config.binding(),
            "audit": audit, "role": role, "activation_checkpoint": e.s15_activation_checkpoint,
            "torch": str(torch_mod().__version__)}


def save(c, e, path, scope, audit=None, role=None):
    from sparse_rtdetr.research import scale_baseline as S
    from sparse_rtdetr.protocol_r18.engine import assert_healthy
    torch = torch_mod()
    path = Path(path)
    require(e.ema.updates == e.completed_updates, "EMA clock at save")
    assert_healthy(e.model, e.optimizer, e.ema)
    state = S.capture(e)
    b = binding(c, e, scope, audit, role)
    state["l0_binding_sha256"] = sha_bytes(canonical(b))
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


def read_side(c, path, scope):
    """Side-car guard only (no tensors loaded): refuses R27 N/C/P states, other contracts, scopes and parents."""
    path = Path(path).resolve()
    side = read_json(side_of(path))
    for k, want in (("schema", SCHEMA), ("arm", ARM), ("scope", scope), ("contract_sha256", c["contract_sha256"]),
                    ("parent_sha256", c["parent"]["sha256"]), ("identity_sha256", c["identity_sha256"])):
        require(side.get(k) == want, f"L0 state refused: {k}={side.get(k)!r}, expected {want!r} ({path})")
    require(side.get("status") == "COMPLETE", f"L0 state side-car is not COMPLETE ({path})")
    return path, side


def read_state(c, path, scope):
    torch = torch_mod()
    path, side = read_side(c, path, scope)
    require(side["file"] == ref(path), "L0 state file identity")
    s = torch.load(path, map_location="cpu", weights_only=True)
    expect = {k: v for k, v in side.items() if k not in ("status", "file")}
    require(s.pop("l0_binding_sha256") == sha_bytes(canonical(expect)), "L0 binding digest")
    require(s["engine"]["completed_updates"] == side["global_update"] == s["ema"]["updates"], "L0 state clock")
    return side, s


def load(c, e, path, scope):
    """Exact restore into a fresh L0 engine; another scope/arm/schema/contract/clock is refused."""
    from sparse_rtdetr.research import scale_baseline as S
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    require(getattr(e, "l0_arm", None) == ARM and not hasattr(e, "r27_arm"), "engine is not an L0 engine")
    side, s = read_state(c, path, scope)
    require(side["config"] == e.config.binding() and side["activation_checkpoint"] == e.s15_activation_checkpoint,
            "L0 state config / memory mode")
    require(s["optimizer_names"] == [g["param_names"] for g in e.optimizer.param_groups], "L0 optimizer names")
    e.model.load_state_dict(s["model"], strict=True)
    e.optimizer.load_state_dict(s["optimizer"])
    e.ema.load_state_dict(s["ema"])
    modes = dict(e.model.named_modules())
    require(set(modes) == set(s["model_modes"]), "L0 mode keys")
    for n, m in s["model_modes"].items():
        modes[n].training = m
    e.load_state_dict(s["engine"])
    ck.restore_rng(s["rng"], e.device)
    require(ck.exact_equal(S.capture(e), s), "L0 full state did not restore exactly")
    return side


FORMAL_DIR = re.compile(r"formal-L0(-r([2-9]|[1-9][0-9]+))?")


def formal_dir(c=None):
    """The one closed formal directory of this round: formal-L0 or (after a mid-run recovery) formal-L0-rN, whose
    CLOSED.json is PASS, arm L0, under this contract.  Zero or several such directories are refused."""
    digest = (c if c is not None else read_json(ROUND / "CONTRACT.json"))["contract_sha256"]
    found = []
    for d in sorted(ROUND.iterdir()) if ROUND.is_dir() else ():
        if not (d.is_dir() and FORMAL_DIR.fullmatch(d.name) and (d / "CLOSED.json").is_file()):
            continue
        closed = read_json(d / "CLOSED.json")
        if closed.get("status") == "PASS" and closed.get("arm") == ARM and closed.get("contract_sha256") == digest:
            found.append(d)
    require(len(found) == 1, f"exactly one closed PASS formal-L0 directory is required under {ROUND}: "
                             f"{[d.name for d in found]}")
    return found[0]


def endpoint_path(c=None):
    return formal_dir(c) / f"checkpoint-epoch-{ENDPOINT // WINDOWS_PER_EPOCH:03d}.pth"


def endpoint_state(c):
    """The only evaluation source: the e120 state of this round's closed formal-L0[-rN] directory, EMA module only
    (online weights are not returned)."""
    path = endpoint_path(c)
    require(path.is_file() and side_of(path).is_file(), f"L0 endpoint missing: {path}")
    side, s = read_state(c, path, "formal")
    require(side["global_update"] == ENDPOINT and s["engine"]["completed_updates"] == s["ema"]["updates"] == ENDPOINT,
            "L0 e120 clocks")
    require(side["cursor"]["completed_epochs"] == ENDPOINT // WINDOWS_PER_EPOCH, "L0 endpoint epoch")
    return side, s["ema"]["module"]


def register_counts(e, counts):
    e.model.register_forward_pre_hook(lambda *_: counts.update(forward_attempts=1))
    e.model.register_forward_hook(lambda *_: counts.update(forwards=1))
    e.optimizer.register_step_post_hook(lambda *_: counts.update(actual_updates=1))
