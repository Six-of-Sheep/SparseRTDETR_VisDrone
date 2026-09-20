#!/usr/bin/env python3
"""Create an explicit, immutable R35 -> REV1 checkpoint bridge.

This is deliberately not a migration of restore_checkpoint. It authenticates
the completed R35 epoch-53 artifact, reconstructs the current REV1 run binding
on CPU, and publishes a new checkpoint whose only payload changes are binding
and binding_sha256. The parent is never replaced.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any


PARENT_CAMPAIGN = "v2bepoch60-r35-20260918t200000z-97a78f7"
PARENT_EPOCH = 53
GPU_UUID = "GPU-1faee6f0-1da7-4ede-2475-67a5a00274a8"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha(value: Any) -> str:
    return sha_bytes(canonical_bytes(value))


def read_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if canonical_bytes(value) != raw or type(value) is not dict:
        raise ValueError(f"non-canonical JSON object: {path}")
    return value


def read_state_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"state JSON object required: {path}")
    return value


def write_json(path: Path, value: dict[str, Any], *, overwrite: bool = False) -> str:
    if path.exists() and not overwrite:
        raise ValueError(f"refusing to overwrite {path}")
    raw = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    temporary.write_bytes(raw)
    os.replace(temporary, path)
    return sha_bytes(raw)


def payload_descriptor(value: Any) -> Any:
    """Return a deterministic, tensor-aware descriptor for equality checks."""
    import torch

    if isinstance(value, torch.Tensor):
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        return {
            "type": "tensor",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": sha_bytes(raw.numpy().tobytes()),
        }
    if type(value) in (str, int, bool) or value is None:
        return value
    if type(value) is float:
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("non-finite payload scalar")
        return value
    if type(value) is dict:
        return {
            str(key): payload_descriptor(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if type(value) in (list, tuple):
        return {
            "type": type(value).__name__,
            "items": [payload_descriptor(item) for item in value],
        }
    raise TypeError(f"unsupported payload type: {type(value).__name__}")


def state_digest(value: Any) -> str:
    return canonical_sha(payload_descriptor(value))


def _authenticated_parent(parent_path: Path, parent_sha: str) -> tuple[bytes, dict[str, Any]]:
    import torch

    raw = parent_path.read_bytes()
    if sha_bytes(raw) != parent_sha:
        raise ValueError("parent checkpoint SHA mismatch")
    payload = torch.load(parent_path, map_location="cpu", weights_only=True)
    if type(payload) is not dict or payload.get("format") != "sparse_rtdetr.training_v2b_checkpoint":
        raise ValueError("parent checkpoint format mismatch")
    if payload.get("schema_version") != 2:
        raise ValueError("parent checkpoint schema is not v2")
    required = {
        "format", "schema_version", "torch_version", "binding", "binding_sha256",
        "model", "model_layout", "model_training", "optimizer", "optimizer_layout",
        "ema", "ema_layout", "engine", "engine_type", "scheduler", "warmup",
        "rng", "runtime", "sampler",
    }
    if set(payload) != required:
        raise ValueError("parent checkpoint payload keys differ")
    engine = payload["engine"]
    if (engine.get("epoch") != PARENT_EPOCH or engine.get("epoch_active") is not False
            or engine.get("failed") is not False
            or engine.get("optimizer_updates") != 16112
            or engine.get("microsteps") != 32224):
        raise ValueError("parent is not the certified completed epoch-53 boundary")
    if payload["binding_sha256"] != payload["binding"].get("binding_sha256"):
        raise ValueError("parent binding digest mismatch")
    return raw, payload


def _current_binding(repo_root: Path, parent: dict[str, Any],
                     pretrained_path: Path, pretrained_sha256: str) -> dict[str, Any]:
    """Construct the exact current R36 binding without initializing CUDA."""
    from sparse_rtdetr.baseline.training_v2b import V2BConfig, build_v2b_components
    from sparse_rtdetr.baseline.training_v2b_data import (
        TrainCoreDataConfig, build_train_core_loader,
    )
    from sparse_rtdetr.baseline.training_v2b_device import prepare_runtime
    from sparse_rtdetr.baseline.training_v2b_runtime import build_train_core_run_binding

    old = parent["binding"]
    old_config = old["config"]
    fields = (
        "seed", "input_size", "physical_batch_size", "accumulation_steps",
        "amp_dtype", "bn_statistics", "pretrained_required", "num_denoising",
        "learning_rate", "weight_decay", "clip_max_norm", "warmup_steps",
        "ema_decay", "ema_warmups", "sampling_backend",
    )
    config = V2BConfig(**{name: old_config[name] for name in fields})
    runtime = prepare_runtime(device="cpu", seed=config.seed)
    components = build_v2b_components(
        config, repo_root=repo_root, pretrained_path=pretrained_path,
        pretrained_sha256=pretrained_sha256, runtime=runtime,
    )
    data = old["data"]["loader"]
    loader_config = data["config"]
    loader = build_train_core_loader(
        TrainCoreDataConfig(
            seed=loader_config["seed"],
            input_size=loader_config["input_size"],
            logical_batch_size=loader_config["logical_batch_size"],
            num_workers=0,
            prefetch_factor=2,
            augmentation_stop_internal_epoch=loader_config[
                "augmentation_stop_internal_epoch"
            ],
        ),
        annotation_file=data["annotation"]["path"],
        annotation_sha256=data["annotation"]["sha256"],
        manifest_file=data["manifest"]["path"],
        manifest_sha256=data["manifest"]["sha256"],
        image_root=data["image_root"],
        repo_root=repo_root,
    )
    binding = build_train_core_run_binding(
        components, loader, run_id="v2b-rev1-s2-r896-bridge-001",
        repo_root=repo_root, requested_device="cuda:0",
        cuda_gpu_uuid=old_config["cuda_gpu_uuid"],
    )
    if binding["initial_weights"] != old["initial_weights"]:
        raise ValueError("current initial model identity differs from parent")
    if binding["data"] != old["data"]:
        raise ValueError("current train_core data identity differs from parent")
    if binding["config"] != old["config"]:
        raise ValueError("current training config differs from parent")
    return binding


def _bridge_binding(current: dict[str, Any], parent: dict[str, Any],
                    contracts: dict[str, str]) -> dict[str, Any]:
    binding = copy.deepcopy(current)
    binding.pop("binding_sha256", None)
    bridge_record = {
        "kind": "revision_bridge",
        "bridge_id": "v2b-r35-e053-to-rev1-s2-r896-002",
        "parent_campaign": PARENT_CAMPAIGN,
        "parent_epoch": PARENT_EPOCH,
        "parent_checkpoint_sha256": contracts["parent_checkpoint_sha256"],
        "old_binding_sha256": parent["binding_sha256"],
        "new_scientific_revision": {
            "source_id": "v2b-sci-001",
            "scientific_source_sha256": contracts["scientific_source_sha256"],
        },
        "training_contract_sha256": contracts["training_contract_sha256"],
        "execution_source_sha256": contracts["execution_source_sha256"],
        "execution_contract_sha256": contracts["execution_contract_sha256"],
        "lineage_sha256": contracts["lineage_sha256"],
        "semantic_delta": "ownership_boundary_hardening",
        "preserved_failure": {
            "campaign": PARENT_CAMPAIGN,
            "epoch": 54,
            "window": 54,
            "exception": "TrainCoreDataError: augmented batch changed after yield",
            "preserved": True,
            "replayed": False,
            "reinterpreted": False,
        },
        "state_policy": "model_optimizer_ema_rng_loader_scheduler_warmup_engine_unchanged",
    }
    # Keep the ordinary run-binding top-level schema intact.  The explicit
    # bridge metadata is part of the bound config, so normal validators still
    # authenticate it without a generic bypass.
    binding["config"] = copy.deepcopy(binding["config"])
    binding["config"]["revision_bridge"] = bridge_record
    binding["binding_sha256"] = canonical_sha(binding)
    return binding


def _publish_derived(parent: dict[str, Any], binding: dict[str, Any],
                     output_root: Path) -> tuple[Path, str]:
    import torch

    output_root.mkdir(parents=True, exist_ok=False)
    destination = output_root / "checkpoint-epoch-053-rev1-bridge.pt"
    derived = dict(parent)
    derived["binding"] = binding
    derived["binding_sha256"] = binding["binding_sha256"]
    for key in parent:
        if key not in {"binding", "binding_sha256"}:
            if state_digest(parent[key]) != state_digest(derived[key]):
                raise ValueError("derived payload changed outside binding: " + key)
    temporary = output_root / ".checkpoint-epoch-053-rev1-bridge.pt.tmp"
    with temporary.open("wb") as stream:
        torch.save(derived, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, destination)
    temporary.unlink()
    directory = os.open(output_root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    derived_raw = destination.read_bytes()
    return destination, sha_bytes(derived_raw)


def _state_identity(parent: dict[str, Any], state_doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": state_digest(parent["model"]),
        "optimizer": state_digest(parent["optimizer"]),
        "ema": state_digest(parent["ema"]),
        "rng": state_digest(parent["rng"]),
        "sampler": state_digest(parent["sampler"]),
        "scheduler": state_digest(parent["scheduler"]),
        "warmup": state_digest(parent["warmup"]),
        "engine": state_digest(parent["engine"]),
        "declared_state_file": {
            key: state_doc.get(key)
            for key in (
                "raw_state_sha256", "optimizer_state_sha256", "ema_state_sha256",
                "rng_state_sha256", "loader_state_sha256", "engine_state_sha256",
                "sample_order_sha256",
            )
        },
    }


def _gpu_smoke_auth(contract_dir: Path, bridge_manifest_sha: str,
                    derived_sha: str, contracts: dict[str, str]) -> str:
    path = contract_dir / "gpu_smoke_authorization_r2.json"
    if path.exists():
        raise ValueError("GPU smoke authorization already exists")
    body = {
        "schema_version": 1,
        "kind": "pre_formal_gpu_smoke",
        "smoke_authorization_id": "smoke-v2b-rev1-s2-r896-bridge-002",
        "checkpoint_sha256": derived_sha,
        "bridge_manifest_sha256": bridge_manifest_sha,
        "training_contract_sha256": contracts["training_contract_sha256"],
        "execution_source_sha256": contracts["execution_source_sha256"],
        "execution_contract_sha256": contracts["execution_contract_sha256"],
        "lineage_sha256": contracts["lineage_sha256"],
        "scope": {
            "gpu": True,
            "max_windows": 1,
            "max_optimizer_updates": 1,
            "formal_campaign": False,
        },
        "status": "PENDING_NOT_EXECUTED",
        "consumed": False,
        "formal_launch_permitted": False,
        "reason": "bridge-only preparation after explicit binding-schema correction; no GPU smoke or formal continuation launched",
    }
    body["smoke_authorization_sha256"] = canonical_sha(body)
    return write_json(path, body)


def bridge(repo_root: Path, contract_dir: Path, parent_path: Path,
           parent_state_path: Path, output_root: Path,
           parent_sha: str) -> dict[str, Any]:
    raw_parent, parent = _authenticated_parent(parent_path, parent_sha)
    state_doc = read_state_json(parent_state_path)
    contracts = {
        "parent_checkpoint_sha256": parent_sha,
        "scientific_source_sha256": read_json(contract_dir / "scientific_source.json")[
            "scientific_source_sha256"
        ],
        "training_contract_sha256": read_json(contract_dir / "training_contract.json")[
            "training_contract_sha256"
        ],
        "execution_source_sha256": read_json(contract_dir / "execution_source.json")[
            "execution_source_sha256"
        ],
        "execution_contract_sha256": read_json(contract_dir / "execution_contract.json")[
            "execution_contract_sha256"
        ],
        "lineage_sha256": read_json(contract_dir / "lineage.json")["lineage_sha256"],
    }
    expected_state = {
        "raw_state_sha256": "b74fd28452eff6964066a9d2d6622ac0e96fa30f4d8d4af7236edb6fe776f512",
        "optimizer_state_sha256": "11d6928319d39cc18640ddea7ab73168bb0bcbd419e00812ae199066e9557294",
        "ema_state_sha256": "a04bf814fe12acc3bd13ce6503b6a6c16d9d794b2254952ed4421f3a793e4f6e",
    }
    for key, value in expected_state.items():
        if state_doc.get(key) != value:
            raise ValueError("parent state declaration mismatch: " + key)
    if canonical_sha(state_doc["rng"]) != (
            "b7366e02278b5d81066913d1c97daedfcaf5cd50eefb331562ba7bc9ae578b0a"
    ):
        raise ValueError("parent RNG state declaration mismatch")
    clocks = state_doc.get("clocks", {})
    if canonical_sha(clocks.get("loader")) != (
            "343fe1b7f92274a2e2df6bfc6098ebef6b24edabf29c5124f96dffed245131fe"
    ) or canonical_sha(clocks.get("engine")) != (
            "d4547f426fef8e562abe0a5120e598c3b1ea42853a0805e3be23486121beccd6"
    ) or clocks.get("loader", {}).get("order_sha256") != (
            "25b1e626a194afd71f32da7a62b26155f921900a0a9ef020691374f264d6294a"
    ):
        raise ValueError("parent clock identity declaration mismatch")
    authority_doc = read_json(contract_dir / "external_authorities.json")
    weights = [row for authority in authority_doc["authorities"]
               if authority.get("logical_type") == "presnet18_vd_pretrained"
               for row in authority["inventory"]["rows"]
               if row["relative_path"].endswith(".pth")]
    if len(weights) != 1:
        raise ValueError("pretrained authority weight row is not unique")
    weights_authority = next(authority for authority in authority_doc["authorities"]
                             if authority.get("logical_type") == "presnet18_vd_pretrained")
    pretrained_path = Path(weights_authority["runtime_locator"]["path"]) / weights[0]["relative_path"]
    current = _current_binding(repo_root, parent, pretrained_path, weights[0]["sha256"])
    binding = _bridge_binding(current, parent, contracts)
    derived_path, derived_sha = _publish_derived(parent, binding, output_root)
    state_identity = _state_identity(parent, state_doc)
    manifest_body = {
        "schema_version": 1,
        "kind": "revision_bridge",
        "bridge_id": binding["config"]["revision_bridge"]["bridge_id"],
        "parent": {
            "campaign": PARENT_CAMPAIGN,
            "epoch": PARENT_EPOCH,
            "checkpoint_sha256": parent_sha,
            "binding_sha256": parent["binding_sha256"],
            "size_bytes": len(raw_parent),
        },
        "derived": {
            "checkpoint_sha256": derived_sha,
            "size_bytes": derived_path.stat().st_size,
            "relative_path": derived_path.name,
        },
        "binding_transition": {
            "old_binding_sha256": parent["binding_sha256"],
            "new_binding_sha256": binding["binding_sha256"],
        },
        "new_revision": contracts,
        "semantic_delta": "ownership_boundary_hardening",
        "resume_boundary": {
            "parent_epoch": 53, "start_epoch": 54, "start_window": 1,
            "optimizer_updates": 16112, "microsteps": 32224,
        },
        "state_identity": state_identity,
        "preserved_failure": binding["config"]["revision_bridge"]["preserved_failure"],
        "runtime_locator": {
            "output_root": str(output_root.resolve()),
            "checkpoint": str(derived_path.resolve()),
        },
    }
    manifest_path = output_root / "checkpoint-epoch-053-rev1-bridge.json"
    manifest_sha = write_json(manifest_path, manifest_body)
    auth_sha = _gpu_smoke_auth(contract_dir, manifest_sha, derived_sha, contracts)
    return {
        "status": "PASS",
        "classification": "REV1_CHECKPOINT_BRIDGE_READY",
        "parent_checkpoint_sha256": parent_sha,
        "derived_checkpoint_sha256": derived_sha,
        "derived_checkpoint": str(derived_path),
        "bridge_manifest": str(manifest_path),
        "bridge_manifest_sha256": manifest_sha,
        "new_binding_sha256": binding["binding_sha256"],
        "gpu_smoke_authorization_sha256": auth_sha,
        "gpu_smoke_authorization": str(contract_dir / "gpu_smoke_authorization_r2.json"),
        "state_identity": state_identity,
        "optimizer_updates": 16112,
        "microsteps": 32224,
        "resume_boundary": "epoch54/window1",
        "gpu_smoke_executed": False,
        "formal_continuation_executed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-state", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--parent-sha256", required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(bridge(
            args.repo_root.resolve(strict=True), args.contract_dir.resolve(strict=True),
            args.parent_checkpoint.resolve(strict=True),
            args.parent_state.resolve(strict=True), args.output_root.resolve(),
            args.parent_sha256,
        ), sort_keys=True, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({
            "status": "FAIL",
            "classification": "REV1_CHECKPOINT_BRIDGE_BLOCKED",
            "error": f"{type(exc).__name__}: {exc}",
        }, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
