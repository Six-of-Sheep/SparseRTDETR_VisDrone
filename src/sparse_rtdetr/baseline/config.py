"""Portable baseline configuration and isolated vendor model construction."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .artifacts import resolve_runtime_paths
from .contract import (
    BaselineContractError,
    CLASS_DEPENDENT_PARAMETER_DELTA,
    PARAMETER_COUNT_RECONCILIATION_STATUS,
    PARAMETER_COUNT_SCOPE,
    UPSTREAM_R18_DEFAULT_NUM_CLASSES,
    UPSTREAM_R18_DEFAULT_PARAMETER_COUNT,
    VENDOR_CONFIG_RELATIVE,
    VENDOR_CONFIG_SHA256,
    VISDRONE_BASELINE_NUM_CLASSES,
    VISDRONE_BASELINE_PARAMETER_COUNT,
)


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def canonical_config_bytes(value: Any) -> bytes:
    """Return deterministic bytes for a JSON-compatible config object."""

    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BaselineContractError("baseline config is not canonical JSON data") from exc


def _config_path(repo_root: str | Path | None) -> Path:
    root = _default_repo_root() if repo_root is None else Path(repo_root).resolve()
    return root / "configs" / "baseline" / "rtdetrv2_r18_visdrone_baseline_v1.json"


def _assert_portable(value: Any, key: str = "config") -> None:
    if isinstance(value, dict):
        for child_key, child in value.items():
            if child_key in {"data_root", "img_folder"} and child not in (None, ""):
                if isinstance(child, str) and (child.startswith("/") or ":\\" in child):
                    raise BaselineContractError(f"portable config contains an absolute {child_key}")
            _assert_portable(child, child_key)
    elif isinstance(value, list):
        for child in value:
            _assert_portable(child, key)
    elif isinstance(value, str) and value.startswith("/"):
        raise BaselineContractError("portable config contains an absolute path")


def _validate_contract(value: dict[str, Any]) -> None:
    required = {
        "baseline_id",
        "upstream_implementation",
        "backbone",
        "encoder",
        "decoder",
        "decoder_layers",
        "num_queries",
        "num_classes",
        "feature_strides",
        "hidden_dim",
        "num_feature_levels",
        "deformable_sampling_points",
        "engineering_input_size",
        "pretrained",
        "checkpoint",
        "postprocess",
        "num_top_queries",
        "nms",
        "roles",
        "evaluators",
        "formal_training_configuration_frozen",
        "model_selection_certified",
        "speed_measurement_ready",
    }
    if set(value) != {"schema_version", "baseline_contract", "vendor", "r3_binding", "model", "roles", "evaluators", "notes"}:
        raise BaselineContractError("baseline config top-level schema drift")
    contract = value["baseline_contract"]
    if not isinstance(contract, dict) or not required.issubset(contract):
        raise BaselineContractError("baseline contract fields are incomplete")
    integer_fields = {
        "decoder_layers": 1,
        "num_queries": 1,
        "num_classes": 1,
        "hidden_dim": 1,
        "num_feature_levels": 1,
        "num_top_queries": 1,
        "upstream_default_num_classes": 1,
        "upstream_default_parameter_count": 1,
        "visdrone_num_classes": 1,
        "visdrone_parameter_count": 1,
        "class_dependent_parameter_delta": 0,
    }
    for field, minimum in integer_fields.items():
        current = contract.get(field)
        if type(current) is not int or current < minimum:
            raise BaselineContractError(f"{field} must be a non-negative strict integer")
    for field in ("pretrained", "nms", "formal_training_configuration_frozen", "model_selection_certified", "speed_measurement_ready"):
        if type(contract.get(field)) is not bool:
            raise BaselineContractError(f"{field} must be a strict boolean")
    if type(contract.get("checkpoint")) is not type(None):
        raise BaselineContractError("checkpoint must be null in the frozen baseline contract")
    for field in ("feature_strides", "deformable_sampling_points", "engineering_input_size"):
        current = contract.get(field)
        if not isinstance(current, list) or not current or any(type(item) is not int or item <= 0 for item in current):
            raise BaselineContractError(f"{field} must be a non-empty list of positive integers")
    if contract["baseline_id"] != "rtdetrv2_r18_visdrone_baseline_v1":
        raise BaselineContractError("baseline ID drift")
    if contract["num_classes"] != VISDRONE_BASELINE_NUM_CLASSES or contract["num_queries"] != 300 or contract["num_top_queries"] != 300:
        raise BaselineContractError("baseline class/query contract drift")
    if contract["pretrained"] is not False or contract["checkpoint"] is not None or contract["nms"] is not False:
        raise BaselineContractError("baseline initialization/postprocess contract drift")
    if contract["feature_strides"] != [8, 16, 32] or contract["deformable_sampling_points"] != [4, 4, 4]:
        raise BaselineContractError("baseline feature contract drift")
    if contract["upstream_default_num_classes"] != UPSTREAM_R18_DEFAULT_NUM_CLASSES:
        raise BaselineContractError("upstream default class contract drift")
    if contract["upstream_default_parameter_count"] != UPSTREAM_R18_DEFAULT_PARAMETER_COUNT:
        raise BaselineContractError("upstream default parameter contract drift")
    if contract["visdrone_num_classes"] != VISDRONE_BASELINE_NUM_CLASSES:
        raise BaselineContractError("VisDrone class contract drift")
    if contract["visdrone_parameter_count"] != VISDRONE_BASELINE_PARAMETER_COUNT:
        raise BaselineContractError("VisDrone parameter contract drift")
    if contract["class_dependent_parameter_delta"] != CLASS_DEPENDENT_PARAMETER_DELTA:
        raise BaselineContractError("class-dependent parameter delta drift")
    if contract["upstream_default_parameter_count"] - contract["visdrone_parameter_count"] != contract["class_dependent_parameter_delta"]:
        raise BaselineContractError("parameter count fields contradict one another")
    if contract.get("parameter_count_scope") != PARAMETER_COUNT_SCOPE:
        raise BaselineContractError("parameter count scope drift")
    if contract.get("parameter_count_reconciliation_status") != PARAMETER_COUNT_RECONCILIATION_STATUS:
        raise BaselineContractError("parameter count reconciliation status drift")
    if value["schema_version"] != 1:
        raise BaselineContractError("baseline config schema version drift")
    _assert_portable(value)


def build_baseline_config(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Load a fresh portable config object with no runtime data root."""

    path = _config_path(repo_root)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BaselineContractError(f"cannot load baseline config: {path}") from exc
    if not isinstance(value, dict):
        raise BaselineContractError("baseline config must be a JSON object")
    _validate_contract(value)
    return copy.deepcopy(value)


def build_runtime_config(
    repo_root: str | Path,
    data_root: str | Path,
    role: str,
) -> dict[str, Any]:
    """Bind one explicit allowed role to a fresh runtime config object."""

    config = build_baseline_config(repo_root)
    paths = resolve_runtime_paths(repo_root, data_root, role)
    config["runtime"] = {
        "role": paths.role,
        "data_root": str(paths.data_root),
        "annotation_file": str(paths.annotation_file),
    }
    config["dataloader"] = {
        "dataset_type": "VisDroneCocoDetection",
        "role": paths.role,
        "annotation_file": str(paths.annotation_file),
        "img_folder": str(paths.data_root),
        "remap_mscoco_category": False,
    }
    return config


def _vendor_root(repo_root: str | Path | None) -> Path:
    root = _default_repo_root() if repo_root is None else Path(repo_root).resolve()
    return root / "vendor" / "rtdetrv2_pytorch"


def _vendor_config_path(repo_root: str | Path | None) -> Path:
    root = _default_repo_root() if repo_root is None else Path(repo_root).resolve()
    path = root / VENDOR_CONFIG_RELATIVE
    if not path.is_file():
        raise BaselineContractError(f"missing frozen vendor config: {path}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != VENDOR_CONFIG_SHA256:
        raise BaselineContractError("frozen vendor R18 config SHA drift")
    return path


@contextmanager
def _vendor_path(vendor_root: Path) -> Iterator[None]:
    value = str(vendor_root)
    inserted = value not in sys.path
    namespace_module = sys.modules.get("src")
    original_namespace_path = None
    vendor_src = str(vendor_root / "src")
    if namespace_module is not None and hasattr(namespace_module, "__path__"):
        original_namespace_path = list(namespace_module.__path__)
        if vendor_src not in namespace_module.__path__:
            namespace_module.__path__.insert(0, vendor_src)
    if inserted:
        sys.path.insert(0, value)
    try:
        yield
    finally:
        if inserted:
            sys.path.remove(value)
        if original_namespace_path is not None:
            namespace_module.__path__[:] = original_namespace_path


def load_isolated_vendor_config_dict(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Load vendor YAML with a fresh dict instead of its mutable default dict."""

    path = _vendor_config_path(repo_root)
    vendor_root = _vendor_root(repo_root)
    with _vendor_path(vendor_root):
        from src.core.yaml_utils import load_config

        return copy.deepcopy(load_config(str(path), cfg={}))


def _clone_registry_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _clone_registry_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_clone_registry_value(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_clone_registry_value(child) for child in value)
    return value


def _snapshot_registry(registry: Any) -> dict[str, Any]:
    return {key: _clone_registry_value(value) for key, value in registry.items()}


def _restore_registry(registry: Any, snapshot: dict[str, Any]) -> None:
    registry.clear()
    registry.update(snapshot)


def _build_r18_cpu_model(repo_root: str | Path | None, num_classes: int):
    """Construct one unfrozen-weight R18 identity on CPU, without forward/backward."""

    path = _vendor_config_path(repo_root)
    vendor_root = _vendor_root(repo_root)
    with _vendor_path(vendor_root):
        import importlib

        importlib.import_module("src.solver")
        importlib.import_module("src.data")
        importlib.import_module("src.nn")
        importlib.import_module("src.zoo.rtdetr")
        from src.core import GLOBAL_CONFIG
        from src.core._config import BaseConfig
        from src.core.yaml_config import YAMLConfig as VendorYAMLConfig
        from src.core.yaml_utils import load_config, merge_dict

        registry_snapshot = _snapshot_registry(GLOBAL_CONFIG)

        class IsolatedYAMLConfig(VendorYAMLConfig):
            def __init__(self, cfg_path: str, **kwargs: Any) -> None:
                BaseConfig.__init__(self)
                cfg = load_config(cfg_path, cfg={})
                cfg = merge_dict(cfg, copy.deepcopy(kwargs), inplace=True)
                self.yaml_cfg = copy.deepcopy(cfg)
                for key in tuple(self.__dict__):
                    if not key.startswith("_") and key in cfg:
                        self.__dict__[key] = cfg[key]

        try:
            config = IsolatedYAMLConfig(
                str(path),
                PResNet={"pretrained": False},
                num_classes=num_classes,
                remap_mscoco_category=False,
            )
            model = config.model
        finally:
            _restore_registry(GLOBAL_CONFIG, registry_snapshot)
    model.to(device="cpu")
    return model


def build_r18_cpu_model(repo_root: str | Path | None = None):
    """Construct the 10-class VisDrone R18 baseline on CPU."""

    return _build_r18_cpu_model(repo_root, VISDRONE_BASELINE_NUM_CLASSES)


def build_upstream_r18_cpu_model(repo_root: str | Path | None = None):
    """Construct the upstream default 80-class R18 identity on CPU."""

    return _build_r18_cpu_model(repo_root, UPSTREAM_R18_DEFAULT_NUM_CLASSES)
