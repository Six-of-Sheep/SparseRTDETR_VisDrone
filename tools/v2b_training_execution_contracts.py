#!/usr/bin/env python3
"""Seal the minimal REV1 training/execution continuation closure."""
from __future__ import annotations
import argparse, hashlib, json, stat
from pathlib import Path
from typing import Any

RUNTIME_KEYS = {"runtime_locator", "runtime_file_count", "runtime_locators", "state_runtime_locator"}

def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")

def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()

def canonical_sha(value: Any) -> str:
    return sha_bytes(canonical_bytes(value))

def without_runtime(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: without_runtime(v) for k, v in value.items() if k not in RUNTIME_KEYS}
    if isinstance(value, list):
        return [without_runtime(v) for v in value]
    return value

def load_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes(); value = json.loads(raw.decode("utf-8"))
    if canonical_bytes(value) != raw: raise ValueError(f"non-canonical JSON: {path}")
    if not isinstance(value, dict): raise ValueError(f"JSON object required: {path}")
    return value

def regular(path: Path) -> None:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink(): raise ValueError(f"regular non-symlink file required: {path}")

def file_row(root: Path, relative: str) -> dict[str, Any]:
    if not isinstance(relative, str) or not relative or "\\" in relative or ".." in relative.split("/"):
        raise ValueError(f"invalid logical path: {relative!r}")
    path = root.joinpath(*relative.split("/")); regular(path); raw = path.read_bytes()
    return {"relative_path": relative, "size_bytes": len(raw), "sha256": sha_bytes(raw)}

def checkpoint_identity(checkpoint: Path, state_path: Path) -> dict[str, Any]:
    import importlib
    torch = importlib.import_module("torch")
    regular(checkpoint); regular(state_path)
    cp = torch.load(checkpoint, map_location="cpu", weights_only=False); state = json.loads(state_path.read_text(encoding="utf-8"))
    if cp.get("format") != "sparse_rtdetr.training_v2b_checkpoint" or cp.get("schema_version") != 2: raise ValueError("unexpected checkpoint schema")
    engine = cp.get("engine") or {}; clocks = state.get("clocks") or {}; eclock = clocks.get("engine") or {}; lclock = clocks.get("loader") or {}
    if engine.get("epoch") != 53 or engine.get("epoch_active") is not False or engine.get("failed") is not False: raise ValueError("not a completed epoch-53 boundary")
    if eclock.get("epoch") != 53 or eclock.get("epoch_active") is not False or lclock.get("epoch") != 53 or lclock.get("completed_batches") != lclock.get("batches_per_epoch"): raise ValueError("state is not a completed epoch-53 boundary")
    binding = cp.get("binding") or {}; cfg = binding.get("config") or {}; data = binding.get("data") or {}; loader = data.get("loader") or {}
    schema = {"format": cp["format"], "schema_version": cp["schema_version"], "checkpoint_keys": sorted(cp), "state_keys": sorted(state)}
    result = {
        "checkpoint_id": "s2-r896-e053-r35", "epoch": 53, "source_campaign": "v2bepoch60-r35-20260918t200000z-97a78f7", "resolution": 896, "seed": 2,
        "sha256": sha_bytes(checkpoint.read_bytes()), "size_bytes": checkpoint.stat().st_size, "state_file_sha256": sha_bytes(state_path.read_bytes()), "state_file_size_bytes": state_path.stat().st_size,
        "schema": {"format": cp["format"], "version": cp["schema_version"], "identity_sha256": canonical_sha(schema)},
        "model_state_sha256": state.get("raw_state_sha256"), "optimizer_state_sha256": state.get("optimizer_state_sha256"), "ema_state_sha256": state.get("ema_state_sha256"), "rng_state_sha256": canonical_sha(state["rng"]), "loader_state_sha256": canonical_sha(lclock), "engine_state_sha256": canonical_sha(eclock), "sample_order_sha256": lclock.get("order_sha256"),
        "optimizer_updates": eclock.get("optimizer_updates"), "microsteps": eclock.get("microsteps"), "loader_cursor": {"epoch": 53, "completed_batches": lclock.get("completed_batches"), "next_epoch": 54, "next_window": 1}, "binding_sha256": binding.get("binding_sha256"),
        "runtime_locator": {"checkpoint": str(checkpoint.resolve()), "state": str(state_path.resolve())},
        "training_config": {"physical_batch_size": cfg.get("physical_batch_size"), "accumulation_steps": cfg.get("accumulation_steps"), "logical_batch_size": cfg.get("logical_batch_size"), "input_size": cfg.get("input_size"), "amp_dtype": cfg.get("amp_dtype"), "optimizer": cfg.get("optimizer"), "learning_rate": cfg.get("learning_rate"), "weight_decay": cfg.get("weight_decay"), "betas": cfg.get("betas"), "num_queries": cfg.get("num_queries"), "num_denoising": cfg.get("num_denoising"), "ema_decay": cfg.get("ema_decay"), "ema_warmups": cfg.get("ema_warmups"), "warmup_steps": cfg.get("warmup_steps"), "clip_max_norm": cfg.get("clip_max_norm"), "bn_statistics": cfg.get("bn_statistics"), "dn_policy": cfg.get("dn_policy"), "loss_normalization": cfg.get("loss_normalization")},
        "data_binding": {"loader_binding_sha256": data.get("loader_binding_sha256"), "annotation_sha256": loader.get("annotation", {}).get("sha256"), "manifest_sha256": loader.get("manifest", {}).get("sha256"), "sample_count": loader.get("sample_count"), "deterministic_sample_rng": loader.get("config", {}).get("sample_rng")},
    }
    for key in ("model_state_sha256", "optimizer_state_sha256", "ema_state_sha256"):
        if not isinstance(result[key], str) or len(result[key]) != 64: raise ValueError(f"missing {key}")
    result["augmentation"] = loader.get("transforms")
    return result

def make_training(scientific: dict[str, Any], authorities: dict[str, Any], cp: dict[str, Any]) -> dict[str, Any]:
    body = {"schema_version": 1, "training_contract_id": "v2b-training-001", "revision_id": "P3-V2B-CONTRACT-REV1", "mode": "revision_aware_continuation", "external_authority_ids": sorted(authorities), "scientific_source_id": scientific["source_id"], "scientific_source_sha256": scientific["scientific_source_sha256"], "model": {"architecture": "RT-DETRv2-R18-vd", "backbone": "PResNet-18-vd", "num_classes": 10, "num_queries": 300, "num_denoising": 100}, "data": {"resolution": 896, "seed": 2, "physical_batch_size": 8, "accumulation_steps": 2, "effective_batch_size": 16, "deterministic_sampler": True, "tail_policy": "drop_incomplete_logical_batch", "logical_window_normalization": "logical_window_gt_sum_clamped_once", "augmentation_seed_rule": "sha256(seed,logged_epoch,order_position,stable_image_id)"}, "optimization": {"optimizer": "AdamW", "learning_rate": 0.0001, "weight_decay": 0.0001, "betas": [0.9, 0.999], "scheduler": {"type": "MultiStepLR", "milestones": [1000], "gamma": 0.1}, "warmup": {"type": "LinearWarmup", "steps": 2000}, "clip_max_norm": 0.1, "amp": {"enabled": True, "dtype": "bfloat16", "grad_scaler": False, "scope": "model_and_criterion_only"}}, "stateful_semantics": {"batch_norm": "train", "dn": "native_per_physical_microbatch", "ema": {"enabled": True, "decay": 0.9999, "warmups": 2000, "float_buffers": "vendor_average", "integer_buffers": "vendor_unchanged"}, "matcher": "HungarianMatcher", "loss": "RTDETRCriterionv2", "checkpoint_schema": "sparse_rtdetr.training_v2b_checkpoint/v2"}, "source_checkpoint": cp, "authorization": {"optimizer_update": True, "resume": True, "formal_training": False, "target_epoch": 60}}
    body["data"]["augmentation"] = cp.get("augmentation")
    body["training_contract_sha256"] = canonical_sha(without_runtime(body)); return body

def make_execution(repo_root: Path, scientific: dict[str, Any]) -> dict[str, Any]:
    paths = ["tools/repository_contract_check.py", "tools/run_training_v2b_epoch60_continuation_campaign.py", "tools/run_training_v2b_epoch60_continuation_worker.py", "tools/v2b_training_execution_contracts.py", "tools/seal_v2b_training_execution.py", "tools/verify_v2b_training_execution.py", "src/sparse_rtdetr/baseline/training_v2b_admission.py", "src/sparse_rtdetr/baseline/training_v2b_device.py", "src/sparse_rtdetr/baseline/training_v2b_campaign.py", "src/sparse_rtdetr/baseline/training_v2b_checkpoint.py", "src/sparse_rtdetr/baseline/training_v2b_evidence.py", "src/sparse_rtdetr/baseline/training_v2b_hardware.py", "src/sparse_rtdetr/baseline/training_v2b_runtime.py", "src/sparse_rtdetr/baseline/training_v2b_epoch60_continuation.py"]
    body = {"schema_version": 1, "execution_source_id": "v2b-exec-001", "scope": "rev1_continuation_execution", "source_commit": "71c16cf881eaacd58f678ee64f1a5df2ccf3ad9d", "scientific_source_excluded": scientific["scientific_source_sha256"], "files": sorted([file_row(repo_root, p) for p in paths], key=lambda r: r["relative_path"])}
    body["execution_source_sha256"] = canonical_sha(without_runtime(body)); return body

def make_lineage(cp: dict[str, Any], scientific: dict[str, Any]) -> dict[str, Any]:
    body = {"schema_version": 1, "lineage_id": "v2b-s2-r896-r36-continuation-001", "kind": "revision_aware_continuation", "parent_campaign": "v2bepoch60-r35-20260918t200000z-97a78f7", "parent_epoch": 53, "parent_checkpoint_sha256": cp["sha256"], "old_scientific_revision": {"label": "historical-R35", "checkpoint_binding_sha256": cp["binding_sha256"], "source": "R35 checkpoint binding"}, "new_scientific_revision": {"source_id": scientific["source_id"], "scientific_source_sha256": scientific["scientific_source_sha256"]}, "semantic_delta": "ownership_boundary_hardening", "preserved_failure": {"campaign": "v2bepoch60-r35-20260918t200000z-97a78f7", "epoch": 54, "window": 54, "exception": "TrainCoreDataError: augmented batch changed after yield", "preserved": True, "replayed": False, "reinterpreted": False}, "resume_boundary": {"from_epoch": 53, "start_epoch": 54, "start_window": 1, "mid_window_resume": False, "optimizer_updates": cp["optimizer_updates"], "microsteps": cp["microsteps"], "loader_cursor_sha256": cp["loader_state_sha256"], "rng_state_sha256": cp["rng_state_sha256"], "sample_order_sha256": cp["sample_order_sha256"]}}
    body["lineage_sha256"] = canonical_sha(without_runtime(body)); return body

def make_execution_contract(repo_root: Path, training: dict[str, Any], execution: dict[str, Any], lineage: dict[str, Any], cp: dict[str, Any]) -> dict[str, Any]:
    campaign_root = str(repo_root.parent / "SparseRTDETR_VisDrone_v2b_rev1_s2_r896_continuation_001")
    body = {"schema_version": 1, "execution_contract_id": "v2b-execution-contract-001", "revision_id": "P3-V2B-CONTRACT-REV1", "mode": "revision_aware_continuation", "training_contract_sha256": training["training_contract_sha256"], "execution_source_sha256": execution["execution_source_sha256"], "lineage_sha256": lineage["lineage_sha256"], "checkpoint_sha256": cp["sha256"], "campaign": {"campaign_id": "v2b-rev1-s2-r896-continuation-001", "parent_campaign": lineage["parent_campaign"], "target_epoch": 60, "resume_epoch": 54, "first_window": 1}, "permissions": {"read_only_verification": True, "smoke_allowed": True, "formal_continuation_launch": False, "owner_authorization_required": True}, "runtime_locator": {"campaign_root": campaign_root}}
    body["execution_contract_sha256"] = canonical_sha(without_runtime(body)); return body

def make_authorization(execution_contract: dict[str, Any], cp: dict[str, Any]) -> dict[str, Any]:
    body = {"schema_version": 1, "authorization_id": "auth-v2b-rev1-s2-r896-continuation-001", "campaign_id": execution_contract["campaign"]["campaign_id"], "execution_contract_sha256": execution_contract["execution_contract_sha256"], "training_contract_sha256": execution_contract["training_contract_sha256"], "lineage_sha256": execution_contract["lineage_sha256"], "checkpoint_sha256": cp["sha256"], "parent_epoch": 53, "target_epoch": 60, "consumed": False, "formal_launch_permitted": False, "runtime_locator": {"campaign_root": execution_contract["runtime_locator"]["campaign_root"]}}
    body["authorization_sha256"] = canonical_sha(without_runtime(body)); return body

def make_smoke_auth(execution_contract: dict[str, Any], cp: dict[str, Any]) -> dict[str, Any]:
    body = {"schema_version": 1, "smoke_authorization_id": "smoke-v2b-rev1-s2-r896-ownership-001", "execution_contract_sha256": execution_contract["execution_contract_sha256"], "checkpoint_sha256": cp["sha256"], "scope": {"max_epochs": 0, "max_windows": 1, "gpu_training": False, "formal_continuation": False}, "status": "PENDING_NOT_EXECUTED", "reason": "contract-only preparation; no GPU smoke launched in this transaction"}
    body["smoke_authorization_sha256"] = canonical_sha(without_runtime(body)); return body

def write_new(path: Path, value: Any) -> str:
    if path.exists() or path.is_symlink(): raise ValueError(f"refusing to overwrite {path}")
    raw = canonical_bytes(value); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw); return sha_bytes(raw)

def seal(repo_root: Path, contract_dir: Path, checkpoint: Path, state_path: Path) -> dict[str, Any]:
    auth_doc = load_json(contract_dir / "external_authorities.json"); sci = load_json(contract_dir / "scientific_source.json"); authorities = {a["authority_id"]: a for a in auth_doc["authorities"]}; cp = checkpoint_identity(checkpoint, state_path)
    training = make_training(sci, authorities, cp); execution = make_execution(repo_root, sci); lineage = make_lineage(cp, sci); ec = make_execution_contract(repo_root, training, execution, lineage, cp); auth = make_authorization(ec, cp); smoke = make_smoke_auth(ec, cp)
    outputs = {}
    for name, value in (("training_contract.json", training), ("execution_source.json", execution), ("lineage.json", lineage), ("execution_contract.json", ec), ("campaign_authorization.json", auth), ("smoke_authorization.json", smoke)): outputs[name] = write_new(contract_dir / name, value)
    return {"status": "PASS", "checkpoint": cp, "training_contract_sha256": training["training_contract_sha256"], "execution_source_sha256": execution["execution_source_sha256"], "lineage_sha256": lineage["lineage_sha256"], "execution_contract_sha256": ec["execution_contract_sha256"], "campaign_authorization_sha256": auth["authorization_sha256"], "campaign_id": ec["campaign"]["campaign_id"], "smoke_status": smoke["status"], "files": outputs}

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--repo-root",type=Path,required=True); ap.add_argument("--contract-dir",type=Path,required=True); ap.add_argument("--checkpoint",type=Path,required=True); ap.add_argument("--state",type=Path,required=True); a=ap.parse_args()
    try: print(json.dumps(seal(a.repo_root.resolve(strict=True),a.contract_dir.resolve(strict=True),a.checkpoint.resolve(strict=True),a.state.resolve(strict=True)),sort_keys=True,indent=2)); return 0
    except Exception as exc: print(json.dumps({"status":"FAIL","error":f"{type(exc).__name__}: {exc}"},sort_keys=True)); return 2

if __name__ == "__main__": raise SystemExit(main())
