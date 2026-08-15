"""Frozen Baseline Smoke V1 contract and injectable one-batch harness."""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable

from .artifacts import resolve_runtime_image_path, resolve_runtime_paths, verify_r3_binding
from .config import build_r18_cpu_model
from .contract import (
    BaselineContractError,
    R3_ARTIFACT_INVENTORY_SHA256,
    R3_ARTIFACT_RELATIVE,
    R3_ENTRY_CANONICAL_INVENTORY_SHA256,
)
from .smoke_evidence import (
    SmokeEvidence,
    build_scientific_context,
    build_cuda_runtime_identity,
    build_input_batch_audit,
    build_model_identity,
    build_source_identity,
    canonical_json_bytes,
    describe_preprocessing_pipeline,
    tensor_audit,
)


class SmokeContractError(BaselineContractError):
    """Raised when a smoke contract or harness invariant fails."""


SMOKE_SCHEMA_VERSION = 1
SMOKE_ID = "rtdetrv2_r18_visdrone_baseline_smoke_v1"
SMOKE_V2_ID = "rtdetrv2_r18_visdrone_baseline_smoke_v2"
SMOKE_V3_ID = "rtdetrv2_r18_visdrone_baseline_smoke_v3"
SMOKE_V4_ID = "rtdetrv2_r18_visdrone_baseline_smoke_v4"
SMOKE_V5_ID = "rtdetrv2_r18_visdrone_baseline_smoke_v5"
SMOKE_V6_ID = "rtdetrv2_r18_visdrone_baseline_smoke_v6"
SMOKE_V7_ID = "rtdetrv2_r18_visdrone_baseline_smoke_v7"
SMOKE_CONFIG_RELATIVE = "configs/baseline/rtdetrv2_r18_visdrone_smoke_v1.json"
SMOKE_V2_CONFIG_RELATIVE = "configs/baseline/rtdetrv2_r18_visdrone_smoke_v2.json"
SMOKE_V3_CONFIG_RELATIVE = "configs/baseline/rtdetrv2_r18_visdrone_smoke_v3.json"
SMOKE_V4_CONFIG_RELATIVE = "configs/baseline/rtdetrv2_r18_visdrone_smoke_v4.json"
SMOKE_V5_CONFIG_RELATIVE = "configs/baseline/rtdetrv2_r18_visdrone_smoke_v5.json"
SMOKE_V6_CONFIG_RELATIVE = "configs/baseline/rtdetrv2_r18_visdrone_smoke_v6.json"
SMOKE_V7_CONFIG_RELATIVE = "configs/baseline/rtdetrv2_r18_visdrone_smoke_v7.json"
SMOKE_OUTPUT_RELATIVE = "artifacts/runs/rtdetrv2_r18_visdrone_baseline_smoke_r1"
SMOKE_PROCESS_RELATIVE = "artifacts/process_evidence/rtdetrv2_r18_visdrone_baseline_smoke_r1"
SMOKE_V2_OUTPUT_RELATIVE = "artifacts/runs/rtdetrv2_r18_visdrone_baseline_smoke_r2"
SMOKE_V2_PROCESS_RELATIVE = "artifacts/process_evidence/rtdetrv2_r18_visdrone_baseline_smoke_r2"
SMOKE_V2_OUTER_RELATIVE = "artifacts/outer_launch_evidence/rtdetrv2_r18_visdrone_baseline_smoke_r2"
SMOKE_V2_TMUX_SESSION = "p3_rtdetrv2_r18_visdrone_baseline_smoke_r2"
SMOKE_V2_TMUX_TIMEOUT_SECONDS = 10
SMOKE_V3_OUTPUT_RELATIVE = "artifacts/runs/rtdetrv2_r18_visdrone_baseline_smoke_r3"
SMOKE_V3_PROCESS_RELATIVE = "artifacts/process_evidence/rtdetrv2_r18_visdrone_baseline_smoke_r3"
SMOKE_V3_OUTER_RELATIVE = "artifacts/outer_launch_evidence/rtdetrv2_r18_visdrone_baseline_smoke_r3"
SMOKE_V3_TMUX_SESSION = "p3_rtdetrv2_r18_visdrone_baseline_smoke_r3"
SMOKE_V3_TMUX_TIMEOUT_SECONDS = 10
SMOKE_V4_OUTPUT_RELATIVE = "artifacts/runs/rtdetrv2_r18_visdrone_baseline_smoke_r4"
SMOKE_V4_PROCESS_RELATIVE = "artifacts/process_evidence/rtdetrv2_r18_visdrone_baseline_smoke_r4"
SMOKE_V4_OUTER_RELATIVE = "artifacts/outer_launch_evidence/rtdetrv2_r18_visdrone_baseline_smoke_r4"
SMOKE_V4_TMUX_SESSION = "p3_rtdetrv2_r18_visdrone_baseline_smoke_r4"
SMOKE_V4_TMUX_TIMEOUT_SECONDS = 10
SMOKE_V5_OUTPUT_RELATIVE = "artifacts/runs/rtdetrv2_r18_visdrone_baseline_smoke_r5"
SMOKE_V5_PROCESS_RELATIVE = "artifacts/process_evidence/rtdetrv2_r18_visdrone_baseline_smoke_r5"
SMOKE_V5_OUTER_RELATIVE = "artifacts/outer_launch_evidence/rtdetrv2_r18_visdrone_baseline_smoke_r5"
SMOKE_V5_TMUX_SESSION = "p3_rtdetrv2_r18_visdrone_baseline_smoke_r5"
SMOKE_V5_TMUX_TIMEOUT_SECONDS = 10
SMOKE_V6_OUTPUT_RELATIVE = "artifacts/runs/rtdetrv2_r18_visdrone_baseline_smoke_r6"
SMOKE_V6_PROCESS_RELATIVE = "artifacts/process_evidence/rtdetrv2_r18_visdrone_baseline_smoke_r6"
SMOKE_V6_OUTER_RELATIVE = "artifacts/outer_launch_evidence/rtdetrv2_r18_visdrone_baseline_smoke_r6"
SMOKE_V6_TMUX_SESSION = "p3_rtdetrv2_r18_visdrone_baseline_smoke_r6"
SMOKE_V6_TMUX_TIMEOUT_SECONDS = 10
SMOKE_V7_OUTPUT_RELATIVE = "artifacts/runs/rtdetrv2_r18_visdrone_baseline_smoke_r7"
SMOKE_V7_PROCESS_RELATIVE = "artifacts/process_evidence/rtdetrv2_r18_visdrone_baseline_smoke_r7"
SMOKE_V7_OUTER_RELATIVE = "artifacts/outer_launch_evidence/rtdetrv2_r18_visdrone_baseline_smoke_r7"
SMOKE_V7_TMUX_SESSION = "p3_rtdetrv2_r18_visdrone_baseline_smoke_r7"
SMOKE_V7_TMUX_TIMEOUT_SECONDS = 10
SMOKE_V7_SNAPSHOT_MAX_DELAY_SECONDS = 1.0
SMOKE_NONCE_ENV = "P3_RTDETR_BASELINE_SMOKE_NONCE"
SMOKE_AUTH_ENV = "P3_RTDETR_BASELINE_SMOKE_AUTHORIZED"


@dataclass(frozen=True)
class SmokeRuntimeSpec:
    """Immutable runtime identity for one versioned smoke contract."""

    smoke_id: str
    config_relative_path: str
    output_relative_path: str
    process_relative_path: str
    outer_relative_path: str | None
    tmux_session: str | None
    tmux_timeout_seconds: int | None
    allow_outer_launch: bool


SMOKE_RUNTIME_SPECS = MappingProxyType({
    SMOKE_ID: SmokeRuntimeSpec(
        smoke_id=SMOKE_ID,
        config_relative_path=SMOKE_CONFIG_RELATIVE,
        output_relative_path=SMOKE_OUTPUT_RELATIVE,
        process_relative_path=SMOKE_PROCESS_RELATIVE,
        outer_relative_path=None,
        tmux_session=None,
        tmux_timeout_seconds=None,
        allow_outer_launch=False,
    ),
    SMOKE_V2_ID: SmokeRuntimeSpec(
        smoke_id=SMOKE_V2_ID,
        config_relative_path=SMOKE_V2_CONFIG_RELATIVE,
        output_relative_path=SMOKE_V2_OUTPUT_RELATIVE,
        process_relative_path=SMOKE_V2_PROCESS_RELATIVE,
        outer_relative_path=SMOKE_V2_OUTER_RELATIVE,
        tmux_session=SMOKE_V2_TMUX_SESSION,
        tmux_timeout_seconds=SMOKE_V2_TMUX_TIMEOUT_SECONDS,
        allow_outer_launch=True,
    ),
    SMOKE_V3_ID: SmokeRuntimeSpec(
        smoke_id=SMOKE_V3_ID,
        config_relative_path=SMOKE_V3_CONFIG_RELATIVE,
        output_relative_path=SMOKE_V3_OUTPUT_RELATIVE,
        process_relative_path=SMOKE_V3_PROCESS_RELATIVE,
        outer_relative_path=SMOKE_V3_OUTER_RELATIVE,
        tmux_session=SMOKE_V3_TMUX_SESSION,
        tmux_timeout_seconds=SMOKE_V3_TMUX_TIMEOUT_SECONDS,
        allow_outer_launch=True,
    ),
    SMOKE_V4_ID: SmokeRuntimeSpec(
        smoke_id=SMOKE_V4_ID,
        config_relative_path=SMOKE_V4_CONFIG_RELATIVE,
        output_relative_path=SMOKE_V4_OUTPUT_RELATIVE,
        process_relative_path=SMOKE_V4_PROCESS_RELATIVE,
        outer_relative_path=SMOKE_V4_OUTER_RELATIVE,
        tmux_session=SMOKE_V4_TMUX_SESSION,
        tmux_timeout_seconds=SMOKE_V4_TMUX_TIMEOUT_SECONDS,
        allow_outer_launch=True,
    ),
    SMOKE_V5_ID: SmokeRuntimeSpec(
        smoke_id=SMOKE_V5_ID,
        config_relative_path=SMOKE_V5_CONFIG_RELATIVE,
        output_relative_path=SMOKE_V5_OUTPUT_RELATIVE,
        process_relative_path=SMOKE_V5_PROCESS_RELATIVE,
        outer_relative_path=SMOKE_V5_OUTER_RELATIVE,
        tmux_session=SMOKE_V5_TMUX_SESSION,
        tmux_timeout_seconds=SMOKE_V5_TMUX_TIMEOUT_SECONDS,
        allow_outer_launch=True,
    ),
    SMOKE_V6_ID: SmokeRuntimeSpec(
        smoke_id=SMOKE_V6_ID,
        config_relative_path=SMOKE_V6_CONFIG_RELATIVE,
        output_relative_path=SMOKE_V6_OUTPUT_RELATIVE,
        process_relative_path=SMOKE_V6_PROCESS_RELATIVE,
        outer_relative_path=SMOKE_V6_OUTER_RELATIVE,
        tmux_session=SMOKE_V6_TMUX_SESSION,
        tmux_timeout_seconds=SMOKE_V6_TMUX_TIMEOUT_SECONDS,
        allow_outer_launch=True,
    ),
    SMOKE_V7_ID: SmokeRuntimeSpec(
        smoke_id=SMOKE_V7_ID,
        config_relative_path=SMOKE_V7_CONFIG_RELATIVE,
        output_relative_path=SMOKE_V7_OUTPUT_RELATIVE,
        process_relative_path=SMOKE_V7_PROCESS_RELATIVE,
        outer_relative_path=SMOKE_V7_OUTER_RELATIVE,
        tmux_session=SMOKE_V7_TMUX_SESSION,
        tmux_timeout_seconds=SMOKE_V7_TMUX_TIMEOUT_SECONDS,
        allow_outer_launch=True,
    ),
})


def get_smoke_runtime_spec(smoke_id: Any) -> SmokeRuntimeSpec:
    """Return exactly one frozen runtime spec, rejecting unknown identities."""

    if type(smoke_id) is not str:
        raise SmokeContractError("smoke_id must be a string")
    try:
        return SMOKE_RUNTIME_SPECS[smoke_id]
    except KeyError as exc:
        raise SmokeContractError("unknown smoke_id") from exc


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = child
    return value


def _runtime_schema(spec: SmokeRuntimeSpec) -> dict[str, Any]:
    runtime = {
        "authorized_env": "P3_RTDETR_BASELINE_SMOKE_AUTHORIZED=1",
        "visible_devices": "0",
        "device_count": 1,
        "cpu_fallback": False,
        "output_relative_path": spec.output_relative_path,
        "process_evidence_relative_path": spec.process_relative_path,
    }
    if spec.allow_outer_launch:
        if spec.outer_relative_path is None or spec.tmux_session is None or spec.tmux_timeout_seconds is None:
            raise SmokeContractError("outer-enabled runtime spec is incomplete")
        runtime.update({
            "config_relative_path": spec.config_relative_path,
            "outer_launch_evidence_relative_path": spec.outer_relative_path,
            "tmux_session_name": spec.tmux_session,
            "tmux_client_timeout_seconds": spec.tmux_timeout_seconds,
        })
    return runtime

FROZEN_IMAGE_RECORDS = (
    {
        "coco_image_id": 1,
        "relative_path": "train/images/0000002_00005_d_0000014.jpg",
        "stable_image_id": "5129422a3f13f97e398e6d5cf77efe3fa2398f20f2fc06ba9d8fc8d76b2bfd96",
        "image_size_bytes": 98691,
        "image_sha256": "17532f304f390479d8b7998863af7d2c51549702977c1e76b82ed8700dea7085",
        "annotation_relative_path": "train/annotations/0000002_00005_d_0000014.txt",
        "annotation_size_bytes": 1982,
        "annotation_sha256": "598f421aafd38aa19348ac922a97bc3a297deb96b4243f1961f30662332e45c3",
        "width": 960,
        "height": 540,
        "raw_annotation_rows": 88,
        "keep_rows": 82,
        "ignore_rows": 6,
        "filtered_rows": 0,
        "sequence_key": "0000002",
        "split": "train",
    },
    {
        "coco_image_id": 2,
        "relative_path": "train/images/0000002_00448_d_0000015.jpg",
        "stable_image_id": "4e912e6a969c3e0b4c8ef02ba7512318f0a4351cbd8e74951846bca66421a3a4",
        "image_size_bytes": 100183,
        "image_sha256": "699587aa61bdcd0a0fd49dd001df3cbce002dd78c08586788462c2cf486d0b88",
        "annotation_relative_path": "train/annotations/0000002_00448_d_0000015.txt",
        "annotation_size_bytes": 2188,
        "annotation_sha256": "07fcae5c681eb1e154b532eecb59994a0d219ba1f0dcb4b7c2fba3018858b6fd",
        "width": 960,
        "height": 540,
        "raw_annotation_rows": 95,
        "keep_rows": 85,
        "ignore_rows": 10,
        "filtered_rows": 0,
        "sequence_key": "0000002",
        "split": "train",
    },
)

EXECUTION_CONTRACT = {
    "image_count": 2,
    "batch_size": 2,
    "workers": 0,
    "expected_batches": 1,
    "expected_dataset_next_calls": 1,
    "expected_model_forward_calls": 1,
    "expected_postprocessor_calls": 1,
    "expected_evaluator_calls": 0,
    "expected_criterion_calls": 0,
    "expected_backward_calls": 0,
    "expected_optimizer_steps": 0,
    "expected_scheduler_steps": 0,
    "expected_checkpoint_loads": 0,
    "expected_network_calls": 0,
}


def _strict_int(value: Any, field: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise SmokeContractError(f"{field} must be a strict integer >= {minimum}")


def _strict_bool(value: Any, field: str) -> None:
    if type(value) is not bool:
        raise SmokeContractError(f"{field} must be a strict boolean")


def _manifest_path(repo_root: str | Path) -> Path:
    return Path(repo_root).resolve() / "artifacts/data/visdrone_protocol_v2_conversion_r3/train_core_manifest.json"


def load_frozen_image_selection(repo_root: str | Path, *, runtime_mode: str = "real", config_path: str | Path | None = None) -> tuple[dict[str, Any], ...]:
    """Read and validate only the two explicitly frozen manifest records."""

    if runtime_mode == "synthetic":
        selected_config_path = Path(repo_root).resolve() / SMOKE_CONFIG_RELATIVE if config_path is None else Path(config_path)
        if not selected_config_path.is_absolute():
            selected_config_path = Path(repo_root).resolve() / selected_config_path
        try:
            config = json.loads(selected_config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SmokeContractError("synthetic smoke config is invalid") from exc
        records = config.get("image_selection", {}).get("records") if isinstance(config, dict) else None
        if not isinstance(records, list) or len(records) != len(FROZEN_IMAGE_RECORDS):
            raise SmokeContractError("synthetic smoke selection is missing")
        return tuple(dict(record) for record in records)
    if runtime_mode != "real":
        raise SmokeContractError("unknown smoke runtime mode")
    path = _manifest_path(repo_root)
    if path.is_symlink() or not path.is_file():
        raise SmokeContractError("certified train_core_manifest.json is missing or not regular")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SmokeContractError("train_core manifest is invalid") from exc
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        raise SmokeContractError("train_core manifest records are missing")
    by_id: dict[int, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or type(record.get("coco_image_id")) is not int:
            raise SmokeContractError("train_core manifest contains an invalid record")
        if record["coco_image_id"] in by_id:
            raise SmokeContractError("train_core manifest contains duplicate image IDs")
        by_id[record["coco_image_id"]] = record
    selected: list[dict[str, Any]] = []
    for expected in FROZEN_IMAGE_RECORDS:
        actual = by_id.get(expected["coco_image_id"])
        if actual is None:
            raise SmokeContractError(f"frozen image is missing: {expected['coco_image_id']}")
        for field, value in expected.items():
            if actual.get(field) != value:
                raise SmokeContractError(f"frozen image field drift: {expected['coco_image_id']}:{field}")
        selected.append(dict(actual))
    return tuple(selected)


def validate_smoke_config(config: dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise SmokeContractError("smoke config must be a JSON object")
    smoke_id = config.get("smoke_id")
    expected_top = {"schema_version", "smoke_id", "mode", "split_role", "execution", "model", "image_selection", "runtime", "evidence"}
    if smoke_id == SMOKE_V7_ID:
        expected_top.add("evidence_policy")
    if set(config) != expected_top:
        raise SmokeContractError("smoke config top-level schema drift")
    schema_version = config.get("schema_version")
    if type(schema_version) is not int or schema_version != SMOKE_SCHEMA_VERSION:
        raise SmokeContractError("smoke config identity drift")
    spec = get_smoke_runtime_spec(smoke_id)
    if config["mode"] != "smoke" or config["split_role"] != "train_core":
        raise SmokeContractError("smoke mode or split role drift")
    execution = config["execution"]
    if execution != EXECUTION_CONTRACT:
        raise SmokeContractError("smoke execution contract drift")
    for field, value in execution.items():
        _strict_int(value, f"execution.{field}")
    model = config["model"]
    if not isinstance(model, dict):
        raise SmokeContractError("smoke model contract is missing")
    model_expected = {
        "baseline_id": "rtdetrv2_r18_visdrone_baseline_v1",
        "num_classes": 10,
        "parameters": 20094584,
        "input_size": [640, 640],
        "pretrained": False,
        "checkpoint": None,
        "seed": 0,
        "num_queries": 300,
        "num_top_queries": 300,
        "nms": False,
    }
    if model != model_expected:
        raise SmokeContractError("smoke model contract drift")
    _strict_int(model["num_classes"], "model.num_classes", 1)
    _strict_int(model["parameters"], "model.parameters", 1)
    _strict_int(model["num_queries"], "model.num_queries", 1)
    _strict_int(model["num_top_queries"], "model.num_top_queries", 1)
    _strict_int(model["seed"], "model.seed", 0)
    _strict_bool(model["pretrained"], "model.pretrained")
    _strict_bool(model["nms"], "model.nms")
    if config["image_selection"] != {
        "manifest_relative_path": "artifacts/data/visdrone_protocol_v2_conversion_r3/train_core_manifest.json",
        "selection_policy": "explicit_coco_image_ids_only",
        "records": list(FROZEN_IMAGE_RECORDS),
    }:
        raise SmokeContractError("smoke image selection contract drift")
    runtime = config["runtime"]
    runtime_expected = _runtime_schema(spec)
    if runtime != runtime_expected:
        raise SmokeContractError("smoke runtime contract drift")
    _strict_int(runtime["device_count"], "runtime.device_count", 1)
    _strict_bool(runtime["cpu_fallback"], "runtime.cpu_fallback")
    if spec.allow_outer_launch:
        _strict_int(runtime["tmux_client_timeout_seconds"], "runtime.tmux_client_timeout_seconds", 1)
    evidence = config["evidence"]
    if evidence != {
        "completion_status": "COMPLETED",
        "entry_inventory_excludes": ["artifact_inventory.json", "completion.json"],
        "process_inventory_excludes": ["process_inventory.json", "process_completion.json"],
        "non_reportable": True,
    }:
        raise SmokeContractError("smoke evidence contract drift")
    _strict_bool(evidence["non_reportable"], "evidence.non_reportable")
    if smoke_id == SMOKE_V7_ID:
        policy = config["evidence_policy"]
        if policy != {
            "policy_version": 1,
            "certification_policy": "DUAL_GATE",
            "durable_terminal_evidence_required": True,
            "immediate_snapshot": {
                "required": True,
                "owner": "outer_launcher",
                "capture_count": 1,
                "max_delay_seconds": SMOKE_V7_SNAPSHOT_MAX_DELAY_SECONDS,
                "response_binding_required": True,
            },
        }:
            raise SmokeContractError("smoke V7 evidence policy drift")
    if "data_root" in json.dumps(config, ensure_ascii=True, sort_keys=True):
        raise SmokeContractError("portable smoke config contains data_root")


def load_smoke_config(repo_root: str | Path, config_path: str | Path | None = None, *, runtime_mode: str = "real") -> dict[str, Any]:
    root = Path(repo_root).resolve()
    path = root / SMOKE_CONFIG_RELATIVE if config_path is None else Path(config_path)
    if not path.is_absolute():
        path = root / path
    if path.is_symlink() or not path.is_file():
        raise SmokeContractError("smoke config is missing or not regular")
    path = path.resolve()
    try:
        config = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_strict_json_object)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise SmokeContractError("smoke config is invalid") from exc
    validate_smoke_config(config)
    spec = get_smoke_runtime_spec(config["smoke_id"])
    if path != root / spec.config_relative_path:
        raise SmokeContractError("smoke config path is not bound to its config identity")
    selected = load_frozen_image_selection(repo_root, runtime_mode=runtime_mode, config_path=path)
    if list(selected) != config["image_selection"]["records"]:
        raise SmokeContractError("smoke config and manifest selection differ")
    return config


def _build_frozen_preprocessing_pipeline(repo_root: str | Path) -> tuple[Any, dict[str, Any]]:
    """Construct and describe the one production preprocessing pipeline."""

    from .config import _vendor_path

    with _vendor_path(Path(repo_root).resolve() / "vendor/rtdetrv2_pytorch"):
        from src.data.transforms import ConvertPILImage, Resize
        from src.data.transforms.container import Compose

    pipeline = Compose([Resize([640, 640]), ConvertPILImage(dtype="float32", scale=True)])
    return pipeline, describe_preprocessing_pipeline(pipeline)


def contract_check(repo_root: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    """Data-free smoke contract check. This function must not import torch."""

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise SmokeContractError("contract-check requires CUDA_VISIBLE_DEVICES=''")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise SmokeContractError("contract-check requires PYTHONNOUSERSITE=1")
    config = load_smoke_config(repo_root, config_path)
    binding = verify_r3_binding(repo_root)
    return {
        "status": "PASS",
        "mode": "contract-check",
        "smoke_id": config["smoke_id"],
        "r3_artifact_root": str(binding.artifact_root),
        "cuda_visible_devices": "",
        "python_no_user_site": "1",
        "real_data_accessed": False,
        "real_image_accessed": False,
        "model_constructed": False,
        "dataset_or_dataloader_constructed": False,
        "cuda_object_created": False,
        "network_requested": False,
        "output_directory_created": False,
        "process_evidence_created": False,
        "confirmatory_metrics_accessed": False,
        "dataset_test_accessed_by_this_process": False,
        "torch_imported": "torch" in sys.modules,
    }


def _tensor_sha(tensor: Any) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def tensor_audit(tensor: Any) -> dict[str, Any]:
    import torch

    if not torch.is_tensor(tensor):
        raise SmokeContractError("smoke tensor output is not a tensor")
    finite = bool(torch.isfinite(tensor.detach()).all())
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "finite": finite,
        "logical_sha256": _tensor_sha(tensor) if finite else None,
    }


@dataclass
class SmokeCounters:
    images_requested: int = 2
    images_loaded: int = 0
    batches_requested: int = 0
    batches_processed: int = 0
    dataset_next_calls: int = 0
    model_forward_calls: int = 0
    postprocessor_calls: int = 0
    evaluator_calls: int = 0
    criterion_calls: int = 0
    backward_calls: int = 0
    optimizer_steps: int = 0
    scheduler_steps: int = 0
    checkpoint_loads: int = 0
    network_calls: int = 0

    def as_dict(self) -> dict[str, int]:
        return self.__dict__.copy()


def _rng_snapshot(torch: Any) -> dict[str, str | None]:
    import numpy as np

    snapshot: dict[str, str | None] = {
        "python": hashlib.sha256(repr(random.getstate()).encode("utf-8")).hexdigest(),
        "torch": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        "numpy": hashlib.sha256(np.random.get_state()[1].tobytes()).hexdigest(),
        "cuda": None,
    }
    if torch.cuda.is_available():
        snapshot["cuda"] = hashlib.sha256(
            b"".join(state.cpu().contiguous().numpy().tobytes() for state in torch.cuda.get_rng_state_all())
        ).hexdigest()
    return snapshot


def _rng_state(torch: Any) -> tuple[object, object, tuple[Any, ...], list[Any] | None]:
    import numpy as np

    cuda_state = None
    if torch.cuda.is_available():
        cuda_state = [state.clone() for state in torch.cuda.get_rng_state_all()]
    return random.getstate(), torch.get_rng_state().clone(), np.random.get_state(), cuda_state


def _restore_rng(torch: Any, state: tuple[object, object, tuple[Any, ...], list[Any] | None]) -> None:
    import numpy as np

    random.setstate(state[0])
    torch.set_rng_state(state[1])
    np.random.set_state(state[2])
    if state[3] is not None:
        torch.cuda.set_rng_state_all(state[3])


def _validate_batch(batch: dict[str, Any], torch: Any) -> None:
    if set(batch) != {"images", "targets", "orig_target_sizes", "stable_image_ids"}:
        raise SmokeContractError("smoke batch schema drift")
    images = batch["images"]
    if not torch.is_tensor(images) or images.dtype != torch.float32 or tuple(images.shape) != (2, 3, 640, 640) or not bool(torch.isfinite(images).all()):
        raise SmokeContractError("smoke input batch shape or finite contract failed")
    targets = batch["targets"]
    if not isinstance(targets, list) or len(targets) != 2:
        raise SmokeContractError("smoke target batch count drift")
    for target in targets:
        labels = target.get("labels") if isinstance(target, dict) else None
        if not torch.is_tensor(labels) or labels.dtype != torch.int64 or labels.ndim != 1 or not bool(torch.isfinite(labels).all()) or labels.numel() and (bool((labels < 0).any()) or bool((labels > 9).any())):
            raise SmokeContractError("smoke model labels are not 0..9")
        if isinstance(target, dict) and "boxes" in target:
            boxes = target["boxes"]
            if not torch.is_tensor(boxes) or not torch.is_floating_point(boxes) or tuple(boxes.shape) != (labels.numel(), 4) or not bool(torch.isfinite(boxes).all()):
                raise SmokeContractError("smoke target boxes schema drift")
    sizes = batch["orig_target_sizes"]
    if not torch.is_tensor(sizes) or sizes.dtype != torch.int64 or tuple(sizes.shape) != (2, 2) or not bool((sizes > 0).all()):
        raise SmokeContractError("smoke original size shape drift")
    stable_ids = batch["stable_image_ids"]
    expected_ids = [record["stable_image_id"] for record in FROZEN_IMAGE_RECORDS]
    if stable_ids != expected_ids:
        raise SmokeContractError("smoke stable image ID order drift")


def _base_completion(counters: SmokeCounters, *, success: bool) -> dict[str, Any]:
    values = counters.as_dict()
    values.update({
        "schema_version": 1,
        "status": "COMPLETED" if success else "FAILED",
        "mode": "smoke",
        "smoke_pass_candidate": success,
        "non_training": True,
        "inference_only": True,
        "model_eval": True,
        "torch_no_grad": True,
        "non_selection": True,
        "non_reportable": True,
        "formal_resume_eligible": False,
        "formal_training_eligible": False,
        "baseline_training_ready": False,
        "confirmatory_metrics_accessed": False,
        "dataset_test_accessed_by_this_process": False,
        "speed_measurement": False,
    })
    return values


def _seed_smoke(torch: Any, seed: int, *, cuda: bool = False) -> None:
    import numpy as np

    if type(seed) is not int or seed < 0:
        raise SmokeContractError("smoke seed must be a non-negative strict integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def run_synthetic_smoke(
    repo_root: str | Path,
    output_dir: Path,
    *,
    loader_factory: Callable[[], Iterable[dict[str, Any]]] | None = None,
    model_factory: Callable[[], Any] | None = None,
    postprocessor_factory: Callable[[], Any] | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the exact one-batch state machine with injected CPU components."""

    import torch

    config = load_smoke_config(repo_root, config_path, runtime_mode="synthetic")
    evidence = SmokeEvidence(output_dir, repo_root=Path(repo_root))
    config_sha = evidence.write_config(config)
    counters = SmokeCounters()
    rng_before = _rng_state(torch)
    rng_before_audit = _rng_snapshot(torch)
    _seed_smoke(torch, config["model"]["seed"])
    if loader_factory is None:
        loader_factory = lambda: SyntheticOneBatchLoader(synthetic_batch())
    if model_factory is None:
        model_factory = SyntheticSmokeModel
    if postprocessor_factory is None:
        postprocessor_factory = SyntheticSmokePostProcessor
    completion = _base_completion(counters, success=False)
    try:
        model = model_factory()
        loader = loader_factory()
        postprocessor = postprocessor_factory()
        if getattr(postprocessor, "nms", False) is not False or hasattr(postprocessor, "threshold"):
            raise SmokeContractError("smoke postprocessor selection contract drift")
        if not hasattr(model, "eval"):
            raise SmokeContractError("smoke model lacks eval")
        model.eval()
        iterator = iter(loader)
        counters.batches_requested += 1
        counters.dataset_next_calls += 1
        batch = next(iterator)
        _validate_batch(batch, torch)
        counters.images_loaded = 2
        images = batch["images"]
        with torch.no_grad():
            counters.model_forward_calls += 1
            outputs = model(images)
            if not isinstance(outputs, dict) or set(outputs) != {"pred_logits", "pred_boxes"}:
                raise SmokeContractError("smoke model output schema drift")
            logits = outputs["pred_logits"]
            boxes = outputs["pred_boxes"]
            if tuple(logits.shape) != (2, 300, 10) or tuple(boxes.shape) != (2, 300, 4):
                raise SmokeContractError("smoke model output shape drift")
            if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(boxes).all()):
                raise SmokeContractError("smoke model output is nonfinite")
            counters.postprocessor_calls += 1
            results = postprocessor(outputs, batch["orig_target_sizes"])
        if not isinstance(results, list) or len(results) != 2:
            raise SmokeContractError("smoke postprocessor image count drift")
        total_predictions = 0
        postprocess_audit: list[dict[str, Any]] = []
        for result in results:
            if not isinstance(result, dict) or set(result) != {"labels", "boxes", "scores"}:
                raise SmokeContractError("smoke postprocessor result schema drift")
            labels = result["labels"]
            result_boxes = result["boxes"]
            scores = result["scores"]
            if not all(torch.is_tensor(value) for value in (labels, result_boxes, scores)):
                raise SmokeContractError("smoke postprocessor tensor schema drift")
            if labels.dtype != torch.int64 or tuple(labels.shape) != (300,) or not torch.is_floating_point(result_boxes) or tuple(result_boxes.shape) != (300, 4) or not torch.is_floating_point(scores) or tuple(scores.shape) != (300,):
                raise SmokeContractError("smoke prediction shape drift")
            if not bool(torch.isfinite(labels).all()) or bool((labels < 1).any()) or bool((labels > 10).any()) or not bool(torch.isfinite(result_boxes).all()) or not bool(torch.isfinite(scores).all()):
                raise SmokeContractError("smoke prediction finite/category contract failed")
            total_predictions += len(labels)
            postprocess_audit.append({
                "stable_image_id": batch["stable_image_ids"][len(postprocess_audit)],
                "labels_min": int(labels.min().item()),
                "labels_max": int(labels.max().item()),
                "prediction_count": len(labels),
                "labels": tensor_audit(labels),
                "boxes": tensor_audit(result_boxes),
                "scores": tensor_audit(scores),
            })
        if total_predictions != 600:
            raise SmokeContractError("smoke total prediction count drift")
        counters.batches_processed = 1
        counters_dict = counters.as_dict()
        if counters_dict != {
            **counters_dict,
            "images_requested": 2,
            "images_loaded": 2,
            "batches_requested": 1,
            "batches_processed": 1,
            "dataset_next_calls": 1,
            "model_forward_calls": 1,
            "postprocessor_calls": 1,
            "evaluator_calls": 0,
            "criterion_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "checkpoint_loads": 0,
            "network_calls": 0,
        }:
            raise SmokeContractError("smoke call counters drift")
        completion = _base_completion(counters, success=True)
        completion.update({"config_sha256": config_sha, "total_predictions": total_predictions})
        invocation = {"schema_version": 1, "mode": "synthetic", "smoke_id": config["smoke_id"], "config_sha256": config_sha}
        if config_path is not None:
            selected_config_path = Path(config_path).resolve()
            invocation.update({
                "config_relative_path": selected_config_path.relative_to(Path(repo_root).resolve()).as_posix(),
                "config_size_bytes": (output_dir / "config.json").stat().st_size,
            })
        evidence.write_json("invocation.json", invocation)
        _, preprocessing = _build_frozen_preprocessing_pipeline(repo_root)
        data_binding = {
            "schema_version": 3,
            "context": {},
            "role": "train_core",
            "manifest_relative_path": None,
            "records": list(config["image_selection"]["records"]),
            "metadata": [],
            "portable": True,
            "nonportable": False,
            "real_data_accessed": False,
            "runtime_artifacts_accessed": False,
            "artifact_validation_mode": "synthetic_tracked_contract_only",
        }
        context = build_scientific_context(repo_root, runtime_mode="synthetic", expected_device="cpu", preprocessing=preprocessing, data_binding=data_binding)
        data_binding["context"] = context
        input_batch_audit = build_input_batch_audit(
            batch,
            images,
            batch["orig_target_sizes"],
            runtime_mode="synthetic",
            preprocessing=preprocessing,
            context=context,
        )
        model_output_audit = {"pred_logits": tensor_audit(logits), "pred_boxes": tensor_audit(boxes)}
        evidence.write_json("source_identity.json", build_source_identity(repo_root, config, runtime_mode="synthetic", context=context))
        evidence.write_json("data_binding_audit.json", data_binding)
        evidence.write_json("image_selection_audit.json", {
            "records": list(FROZEN_IMAGE_RECORDS),
            "verified_from_manifest": False,
            "selection_policy": "explicit_coco_image_ids_only",
        })
        evidence.write_json("cuda_runtime_identity.json", build_cuda_runtime_identity({}, runtime_mode="synthetic", context=context))
        evidence.write_json("model_identity.json", build_model_identity(repo_root, config, model, postprocessor, model_output_audit, runtime_mode="synthetic", device="cpu", context=context))
        evidence.write_json("call_audit.json", counters.as_dict())
        evidence.write_json("input_batch_audit.json", input_batch_audit)
        evidence.write_json("model_output_audit.json", model_output_audit)
        evidence.write_json("postprocess_audit.json", {"images": postprocess_audit, "total_predictions": total_predictions, "nms": False, "threshold": None})
        rng_after = _rng_snapshot(torch)
        _restore_rng(torch, rng_before)
        rng_restored = _rng_snapshot(torch)
        evidence.write_json("rng_audit.json", {"before": rng_before_audit, "after": rng_after, "restored": rng_restored, "restored_equal": rng_before_audit == rng_restored})
        evidence.finalize_success(completion)
        return completion
    except BaseException as error:
        _restore_rng(torch, rng_before)
        evidence.finalize_failure(error, completion)
        raise


def synthetic_batch() -> dict[str, Any]:
    import torch

    return {
        "images": torch.zeros((2, 3, 640, 640), dtype=torch.float32),
        "targets": [
            {"labels": torch.tensor([0], dtype=torch.int64), "boxes": torch.full((1, 4), 0.5, dtype=torch.float32)},
            {"labels": torch.tensor([9], dtype=torch.int64), "boxes": torch.full((1, 4), 0.5, dtype=torch.float32)},
        ],
        "orig_target_sizes": torch.tensor([[960, 540], [960, 540]], dtype=torch.int64),
        "stable_image_ids": [record["stable_image_id"] for record in FROZEN_IMAGE_RECORDS],
    }


class SyntheticOneBatchLoader:
    def __init__(self, batch: dict[str, Any]) -> None:
        self.batch = batch
        self.next_calls = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.next_calls += 1
        if self.next_calls != 1:
            raise SmokeContractError("synthetic loader requested a second batch")
        return self.batch


class SyntheticSmokeModel:
    def __init__(self) -> None:
        self.training = True
        self.forward_calls = 0

    def eval(self):
        self.training = False
        return self

    def __call__(self, images):
        import torch

        self.forward_calls += 1
        if self.training or tuple(images.shape) != (2, 3, 640, 640):
            raise SmokeContractError("synthetic model eval/input contract failed")
        return {
            "pred_logits": torch.zeros((2, 300, 10), dtype=torch.float32),
            "pred_boxes": torch.full((2, 300, 4), 0.5, dtype=torch.float32),
        }


class SyntheticSmokePostProcessor:
    nms = False

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, outputs, orig_target_sizes):
        import torch

        self.calls += 1
        return [
            {"labels": torch.ones((300,), dtype=torch.int64), "boxes": torch.full((300, 4), 0.5), "scores": torch.full((300,), 0.5)},
            {"labels": torch.full((300,), 10, dtype=torch.int64), "boxes": torch.full((300, 4), 0.5), "scores": torch.full((300,), 0.5)},
        ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_sha256(value: Any, field: str) -> None:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise SmokeContractError(f"{field} must be a lowercase SHA-256 string")


def validate_real_smoke_environment() -> dict[str, Any]:
    if os.environ.get(SMOKE_AUTH_ENV) != "1":
        raise SmokeContractError(f"{SMOKE_AUTH_ENV}=1 is required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise SmokeContractError("real smoke requires CUDA_VISIBLE_DEVICES='0'")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise SmokeContractError("real smoke requires PYTHONNOUSERSITE=1")
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SmokeContractError("real smoke requires exactly one visible CUDA device")
    return {
        "cuda_visible_devices": "0",
        "device_count": 1,
        "device_name": torch.cuda.get_device_name(0),
        "cpu_fallback": False,
    }


def run_authorized_smoke(
    repo_root: str | Path,
    data_root: Path,
    output_dir: Path,
    *,
    handoff_receipt: dict[str, Any],
    handoff_receipt_sha256: str,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the future real one-batch smoke entrypoint under its exact gates."""

    if not isinstance(handoff_receipt, dict):
        raise SmokeContractError("real smoke handoff receipt is invalid")
    _strict_sha256(handoff_receipt_sha256, "real smoke handoff receipt SHA")
    config = load_smoke_config(repo_root, config_path)
    spec = get_smoke_runtime_spec(config["smoke_id"])
    if spec.allow_outer_launch:
        if config_path is None:
            raise SmokeContractError("versioned outer smoke requires an explicit config path")
        bound_config_path = Path(config_path).resolve()
        canonical_sha = hashlib.sha256(canonical_json_bytes(config)).hexdigest()
        expected_receipt = {
            "smoke_id": config["smoke_id"],
            "config_relative_path": spec.config_relative_path,
            "config_path": str(bound_config_path),
            "config_size_bytes": bound_config_path.stat().st_size,
            "config_file_sha256": _sha256_file(bound_config_path),
            "config_canonical_sha256": canonical_sha,
        }
        for field, expected in expected_receipt.items():
            if handoff_receipt.get(field) != expected:
                raise SmokeContractError(f"versioned handoff {field} binding mismatch")
        _strict_sha256(handoff_receipt.get("config_file_sha256"), "versioned handoff config_file_sha256")
        _strict_sha256(handoff_receipt.get("config_canonical_sha256"), "versioned handoff config_canonical_sha256")
    if type(handoff_receipt.get("child_pid")) is not int or type(handoff_receipt.get("child_ppid")) is not int:
        raise SmokeContractError("real smoke handoff receipt is invalid")
    paths = resolve_runtime_paths(repo_root, data_root, "train_core")
    if output_dir.exists() or output_dir.is_symlink():
        raise SmokeContractError("real smoke output directory already exists")
    evidence = SmokeEvidence(output_dir, repo_root=Path(repo_root))
    config_sha = evidence.write_config(config)
    entry_invocation = {
        "schema_version": 1,
        "mode": "real",
        "smoke_id": config["smoke_id"],
        "config_sha256": config_sha,
        "data_role": "train_core",
        "cuda_visible_devices": "0",
        "nonce": handoff_receipt["nonce"],
        "launcher_pid": handoff_receipt["launcher_pid"],
        "child_pid": handoff_receipt["child_pid"],
        "child_ppid": handoff_receipt["child_ppid"],
        "child_argv_sha256": handoff_receipt["child_argv_sha256"],
        "handoff_receipt_sha256": handoff_receipt_sha256,
        "handoff_receipt_relative_path": "handoff_receipt.json",
    }
    if spec.allow_outer_launch:
        entry_invocation.update({
            "config_relative_path": config["runtime"]["config_relative_path"],
            "config_size_bytes": len(canonical_json_bytes(config)),
        })
    evidence.write_json("invocation.json", entry_invocation)
    try:
        selected = load_frozen_image_selection(repo_root)
        for record in selected:
            image_path = resolve_runtime_image_path(paths.data_root, "train_core", record["relative_path"])
            if _sha256_file(image_path) != record["image_sha256"]:
                raise SmokeContractError(f"frozen image SHA drift: {record['coco_image_id']}")
        runtime = validate_real_smoke_environment()
        import torch

        rng_before = _rng_state(torch)
        rng_before_audit = _rng_snapshot(torch)
        _seed_smoke(torch, config["model"]["seed"], cuda=True)
        from torch.utils.data import DataLoader
        from .config import _vendor_path
        from .dataset import VisDroneCocoDetection
        from .postprocessor import VisDronePostProcessor

        transforms, preprocessing = _build_frozen_preprocessing_pipeline(repo_root)
        dataset = VisDroneCocoDetection(
            paths.data_root,
            paths.annotation_file,
            transforms=transforms,
            vendor_root=Path(repo_root).resolve() / "vendor/rtdetrv2_pytorch",
            role="train_core",
        )
        selected_indices = [dataset.ids.index(record["coco_image_id"]) for record in selected]

        class SelectedDataset:
            def __len__(self):
                return len(selected_indices)

            def __getitem__(self, index):
                image, target = dataset[selected_indices[index]]
                target["stable_image_id"] = selected[index]["stable_image_id"]
                return image, target

        def collate(items):
            images = torch.stack([item[0] for item in items])
            targets = [item[1] for item in items]
            return {
                "images": images,
                "targets": targets,
                "orig_target_sizes": torch.stack([target["orig_size"] for target in targets]),
                "stable_image_ids": [target["stable_image_id"] for target in targets],
            }

        loader = DataLoader(
            SelectedDataset(),
            batch_size=2,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            collate_fn=collate,
        )
        model = build_r18_cpu_model(repo_root)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count != config["model"]["parameters"]:
            raise SmokeContractError("real smoke model parameter count drift")
        model = model.to("cuda:0")
        postprocessor = VisDronePostProcessor(
            vendor_root=Path(repo_root).resolve() / "vendor/rtdetrv2_pytorch"
        ).to("cuda:0")
        return _run_prepared_smoke(
            Path(repo_root),
            config,
            evidence,
            loader,
            model,
            postprocessor,
            runtime,
            config_sha,
            data_binding={
                "schema_version": 3,
                "context": {},
                "role": "train_core",
                "manifest_relative_path": "artifacts/data/visdrone_protocol_v2_conversion_r3/train_core_manifest.json",
                "records": list(selected),
                "metadata": [],
                "portable": False,
                "nonportable": True,
                "real_data_accessed": True,
                "runtime_artifacts_accessed": True,
                "artifact_validation_mode": "real_r3_metadata",
            },
            preprocessing=preprocessing,
            rng_before=rng_before,
            rng_before_audit=rng_before_audit,
        )
    except BaseException as error:
        if "torch" in locals() and "rng_before" in locals():
            _restore_rng(torch, rng_before)
        if not (output_dir / "completion.json").exists():
            evidence.finalize_failure(error, _base_completion(SmokeCounters(), success=False))
        raise


def _run_prepared_smoke(
    repo_root: Path,
    config: dict[str, Any],
    evidence: SmokeEvidence,
    loader: Iterable[dict[str, Any]],
    model: Any,
    postprocessor: Any,
    runtime: dict[str, Any],
    config_sha: str,
    *,
    data_binding: dict[str, Any],
    preprocessing: dict[str, Any],
    rng_before: tuple[object, object, tuple[Any, ...], list[Any] | None],
    rng_before_audit: dict[str, str | None],
) -> dict[str, Any]:
    """Execute exactly one prepared real batch and close the entry evidence."""

    import torch

    counters = SmokeCounters()
    completion = _base_completion(counters, success=False)
    try:
        if getattr(postprocessor, "nms", False) is not False or hasattr(postprocessor, "threshold"):
            raise SmokeContractError("smoke postprocessor selection contract drift")
        if not hasattr(model, "eval"):
            raise SmokeContractError("smoke model lacks eval")
        model.eval()
        iterator = iter(loader)
        counters.batches_requested += 1
        counters.dataset_next_calls += 1
        batch = next(iterator)
        _validate_batch(batch, torch)
        counters.images_loaded = 2
        images = batch["images"].to("cuda:0")
        orig_target_sizes = batch["orig_target_sizes"].to("cuda:0")
        if images.device.type != "cuda" or orig_target_sizes.device.type != "cuda":
            raise SmokeContractError("smoke tensors did not reach CUDA")
        with torch.no_grad():
            counters.model_forward_calls += 1
            outputs = model(images)
            if not isinstance(outputs, dict) or set(outputs) != {"pred_logits", "pred_boxes"}:
                raise SmokeContractError("smoke model output schema drift")
            logits = outputs["pred_logits"]
            boxes = outputs["pred_boxes"]
            if tuple(logits.shape) != (2, 300, 10) or tuple(boxes.shape) != (2, 300, 4):
                raise SmokeContractError("smoke model output shape drift")
            if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(boxes).all()):
                raise SmokeContractError("smoke model output is nonfinite")
            counters.postprocessor_calls += 1
            results = postprocessor(outputs, orig_target_sizes)
        if not isinstance(results, list) or len(results) != 2:
            raise SmokeContractError("smoke postprocessor image count drift")
        total_predictions = 0
        postprocess_audit: list[dict[str, Any]] = []
        for result in results:
            if not isinstance(result, dict) or set(result) != {"labels", "boxes", "scores"}:
                raise SmokeContractError("smoke postprocessor result schema drift")
            labels = result["labels"]
            result_boxes = result["boxes"]
            scores = result["scores"]
            if not all(torch.is_tensor(value) for value in (labels, result_boxes, scores)):
                raise SmokeContractError("smoke postprocessor tensor schema drift")
            if labels.dtype != torch.int64 or tuple(labels.shape) != (300,) or not torch.is_floating_point(result_boxes) or tuple(result_boxes.shape) != (300, 4) or not torch.is_floating_point(scores) or tuple(scores.shape) != (300,):
                raise SmokeContractError("smoke prediction shape drift")
            if not bool(torch.isfinite(labels).all()) or bool((labels < 1).any()) or bool((labels > 10).any()) or not bool(torch.isfinite(result_boxes).all()) or not bool(torch.isfinite(scores).all()):
                raise SmokeContractError("smoke prediction finite/category contract failed")
            total_predictions += len(labels)
            postprocess_audit.append({
                "stable_image_id": batch["stable_image_ids"][len(postprocess_audit)],
                "labels_min": int(labels.min().item()),
                "labels_max": int(labels.max().item()),
                "prediction_count": len(labels),
                "labels": tensor_audit(labels),
                "boxes": tensor_audit(result_boxes),
                "scores": tensor_audit(scores),
            })
        if total_predictions != 600:
            raise SmokeContractError("smoke total prediction count drift")
        counters.batches_processed = 1
        expected_counters = {
            "images_requested": 2,
            "images_loaded": 2,
            "batches_requested": 1,
            "batches_processed": 1,
            "dataset_next_calls": 1,
            "model_forward_calls": 1,
            "postprocessor_calls": 1,
            "evaluator_calls": 0,
            "criterion_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "checkpoint_loads": 0,
            "network_calls": 0,
        }
        if counters.as_dict() != expected_counters:
            raise SmokeContractError("smoke call counters drift")
        completion = _base_completion(counters, success=True)
        completion.update({"config_sha256": config_sha, "total_predictions": total_predictions})
        model_output_audit = {"pred_logits": tensor_audit(logits), "pred_boxes": tensor_audit(boxes)}
        metadata = []
        r3_root = Path(repo_root) / R3_ARTIFACT_RELATIVE
        for relative in (
            "completion.json", "artifact_inventory.json", "config.json", "category_contract.json", "source_identity.json",
        ):
            path = r3_root / relative
            metadata.append({"relative_path": (Path(R3_ARTIFACT_RELATIVE) / relative).as_posix(), "size_bytes": path.stat().st_size, "sha256": _sha256_file(path)})
        manifest_path = _manifest_path(repo_root)
        metadata.append({"relative_path": manifest_path.relative_to(Path(repo_root)).as_posix(), "size_bytes": manifest_path.stat().st_size, "sha256": _sha256_file(manifest_path)})
        data_binding.update({"schema_version": 3, "manifest_relative_path": metadata[-1]["relative_path"], "metadata": metadata, "portable": False, "nonportable": True, "real_data_accessed": True, "runtime_artifacts_accessed": True, "artifact_validation_mode": "real_r3_metadata"})
        context = build_scientific_context(repo_root, runtime_mode="real", expected_device="cuda:0", preprocessing=preprocessing, data_binding=data_binding)
        data_binding["context"] = context
        input_batch_audit = build_input_batch_audit(
            batch,
            images,
            orig_target_sizes,
            runtime_mode="real",
            preprocessing=preprocessing,
            context=context,
        )
        evidence.write_json("source_identity.json", build_source_identity(repo_root, config, runtime_mode="real", context=context))
        evidence.write_json("data_binding_audit.json", data_binding)
        evidence.write_json("image_selection_audit.json", {
            "records": list(FROZEN_IMAGE_RECORDS),
            "verified_from_manifest": True,
            "selection_policy": "explicit_coco_image_ids_only",
        })
        evidence.write_json("cuda_runtime_identity.json", build_cuda_runtime_identity(runtime, runtime_mode="real", context=context))
        evidence.write_json("model_identity.json", build_model_identity(repo_root, config, model, postprocessor, model_output_audit, runtime_mode="real", device="cuda:0", context=context))
        evidence.write_json("call_audit.json", counters.as_dict())
        evidence.write_json("input_batch_audit.json", input_batch_audit)
        evidence.write_json("model_output_audit.json", model_output_audit)
        evidence.write_json("postprocess_audit.json", {
            "images": postprocess_audit,
            "total_predictions": total_predictions,
            "nms": False,
            "threshold": None,
        })
        rng_after = _rng_snapshot(torch)
        _restore_rng(torch, rng_before)
        rng_restored = _rng_snapshot(torch)
        evidence.write_json("rng_audit.json", {
            "before": rng_before_audit,
            "after": rng_after,
            "restored": rng_restored,
            "restored_equal": rng_before_audit == rng_restored,
        })
        evidence.finalize_success(completion)
        return completion
    except BaseException as error:
        _restore_rng(torch, rng_before)
        evidence.finalize_failure(error, completion)
        raise
