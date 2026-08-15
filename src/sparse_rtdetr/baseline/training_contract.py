"""Portable, standard-library-only validation for the frozen T1 contract."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any


TRAINING_CONFIG_RELATIVE_PATH = "configs/baseline/rtdetrv2_r18_visdrone_training_v1.json"
BASELINE_CONFIG_RELATIVE_PATH = "configs/baseline/rtdetrv2_r18_visdrone_baseline_v1.json"
TRAINING_CONFIG_SIZE_BYTES = 8058
TRAINING_CONFIG_RAW_SHA256 = "8ce30e636caad84ee6e3e0ec10d384730a00861ccf179c1e765c095465c6cd5d"
TRAINING_CONTRACT_CANONICAL_SHA256 = "753991ca118267f571efda6a602c1347323707d074d9d7b2e4ed64fb7a17c4f6"
BASELINE_CONFIG_SIZE_BYTES = 4316
BASELINE_CONFIG_RAW_SHA256 = "38702c3483efcd3c3855552d087bbfd0fcfe3628fa9b1582847039f17eb083dd"
BASELINE_CONFIG_CANONICAL_SHA256 = "c392efd44de7738401c1136261c8ca628dea3d3b0fe355b79d0d69d6ed91bfe0"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_EXPECTED_TOP_LEVEL = {
    "schema_version", "training_contract_id", "baseline_id", "owner_decision",
    "source_bindings", "model", "data_roles", "initialization", "topology",
    "schedule", "optimizer", "learning_rate", "amp", "ema", "augmentation",
    "evaluation_and_selection", "checkpoint_policy", "acceptance",
}


class TrainingContractError(ValueError):
    """Raised when the portable formal-training contract fails closed."""


def canonical_training_contract_bytes(config: Any) -> bytes:
    """Return deterministic compact JSON bytes without a trailing newline."""

    try:
        return json.dumps(
            config, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TrainingContractError("training contract is not finite JSON data") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TrainingContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise TrainingContractError(f"non-finite JSON number: {value}")


def _parse_portable_json(raw: bytes, *, label: str) -> dict[str, Any]:
    if raw.startswith(b"\xef\xbb\xbf"):
        raise TrainingContractError(f"{label} must not contain a UTF-8 BOM")
    if b"\r" in raw or not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise TrainingContractError(f"{label} must use LF and exactly one trailing LF")
    try:
        text = raw.decode("utf-8")
        value = json.loads(
            text, object_pairs_hook=_strict_pairs, parse_constant=_reject_constant
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise TrainingContractError(f"cannot parse {label}") from exc
    if type(value) is not dict:
        raise TrainingContractError(f"{label} must be a JSON object")
    return value


def _assert_json_types(value: Any, field: str = "contract") -> None:
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                raise TrainingContractError(f"{field} contains a non-string key")
            _assert_json_types(child, f"{field}.{key}")
    elif type(value) is list:
        for index, child in enumerate(value):
            _assert_json_types(child, f"{field}[{index}]")
    elif type(value) is float:
        if not math.isfinite(value):
            raise TrainingContractError(f"{field} must be finite")
    elif type(value) not in {str, int, bool, type(None)}:
        raise TrainingContractError(f"{field} has a non-JSON or non-strict scalar type")


def _assert_relative_path(value: Any, field: str) -> str:
    if type(value) is not str or not value or "\x00" in value or "\\" in value:
        raise TrainingContractError(f"{field} is not a portable POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise TrainingContractError(f"{field} is not a repository-relative POSIX path")
    return value


def _assert_sha(value: Any, field: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise TrainingContractError(f"{field} must be 64 lowercase hex characters")
    return value


def _validate_path_and_sha_fields(config: dict[str, Any]) -> None:
    sources = config["source_bindings"]
    baseline = sources["baseline_config"]
    _assert_relative_path(baseline["relative_path"], "baseline_config.relative_path")
    _assert_sha(baseline["raw_sha256"], "baseline_config.raw_sha256")
    _assert_sha(baseline["canonical_sha256"], "baseline_config.canonical_sha256")
    recipe = sources["vendor_recipe"]
    _assert_relative_path(recipe["relative_path"], "vendor_recipe.relative_path")
    _assert_sha(recipe["sha256"], "vendor_recipe.sha256")
    for index, item in enumerate(sources["vendor_includes"]):
        _assert_relative_path(item["relative_path"], f"vendor_includes[{index}].relative_path")
        _assert_sha(item["sha256"], f"vendor_includes[{index}].sha256")
    conversion = sources["conversion_r3"]
    _assert_relative_path(conversion["artifact_root"], "conversion_r3.artifact_root")
    for key, value in conversion.items():
        if key.endswith("sha256"):
            _assert_sha(value, f"conversion_r3.{key}")


def _validate_cross_fields(config: dict[str, Any]) -> None:
    topology = config["topology"]
    if topology["train_micro_batch"] * topology["gradient_accumulation_steps"] != topology["effective_train_batch"]:
        raise TrainingContractError("effective train batch arithmetic drift")
    schedule = config["schedule"]
    if schedule["augmentation_stop_epoch"] >= schedule["epochs"]:
        raise TrainingContractError("augmentation stop must precede completion")
    lr = config["learning_rate"]
    expected_events = sum(item < schedule["epochs"] for item in lr["milestones"])
    if lr["scheduler_step_unit"] != "epoch" or expected_events != lr["expected_decay_events_within_120_epochs"]:
        raise TrainingContractError("scheduler decay semantics drift")
    checkpoint = config["checkpoint_policy"]
    if schedule["epochs"] % checkpoint["periodic_checkpoint_frequency_epochs"] != 0 or not checkpoint["final_epoch_checkpoint"]:
        raise TrainingContractError("periodic/final checkpoint policy drift")
    if config["augmentation"]["stop_epoch"] != schedule["augmentation_stop_epoch"]:
        raise TrainingContractError("augmentation stop fields disagree")
    evaluation = config["evaluation_and_selection"]
    if evaluation["evaluation_frequency_epochs"] != schedule["development_evaluation_frequency_epochs"]:
        raise TrainingContractError("development evaluation frequency drift")
    if evaluation["primary_evaluator_independently_certified"] or not evaluation["training_launch_blocked"]:
        raise TrainingContractError("primary evaluator launch gate must remain closed")


def _validate_baseline_binding(config: dict[str, Any], baseline_config: dict[str, Any]) -> None:
    _assert_json_types(baseline_config, "baseline_config")
    source = config["source_bindings"]["baseline_config"]
    if source != {
        "relative_path": BASELINE_CONFIG_RELATIVE_PATH,
        "size_bytes": BASELINE_CONFIG_SIZE_BYTES,
        "raw_sha256": BASELINE_CONFIG_RAW_SHA256,
        "canonical_sha256": BASELINE_CONFIG_CANONICAL_SHA256,
    }:
        raise TrainingContractError("baseline source binding drift")
    if _sha256_bytes(canonical_training_contract_bytes(baseline_config)) != BASELINE_CONFIG_CANONICAL_SHA256:
        raise TrainingContractError("baseline canonical identity drift")
    contract = baseline_config.get("baseline_contract", {})
    model = baseline_config.get("model", {})
    roles = baseline_config.get("roles", {})
    if contract.get("baseline_id") != config["baseline_id"] or model.get("num_classes") != 10:
        raise TrainingContractError("baseline ID/class binding drift")
    if model.get("PResNet") != {"depth": 18, "pretrained": False} or model.get("num_queries") != 300:
        raise TrainingContractError("baseline architecture/initialization binding drift")
    if config["initialization"]["pretrained"] is not False or config["initialization"]["checkpoint"] is not None:
        raise TrainingContractError("T1 random initialization drift")
    if roles.get("train_core", {}).get("status") != "allowed" or roles.get("development", {}).get("status") != "allowed":
        raise TrainingContractError("baseline allowed-role binding drift")
    if roles.get("confirmatory", {}).get("status") != "sealed_and_forbidden" or roles.get("test", {}).get("status") != "forbidden":
        raise TrainingContractError("baseline forbidden-role binding drift")


def validate_training_contract(config: Any, baseline_config: Any) -> dict[str, Any]:
    """Validate exact T1 content and return a detached portable object."""

    if type(config) is not dict or set(config) != _EXPECTED_TOP_LEVEL:
        raise TrainingContractError("training contract top-level exact-key schema drift")
    if type(baseline_config) is not dict:
        raise TrainingContractError("baseline config must be an exact dict")
    _assert_json_types(config)
    digest = _sha256_bytes(canonical_training_contract_bytes(config))
    if digest != TRAINING_CONTRACT_CANONICAL_SHA256:
        raise TrainingContractError("training contract frozen content drift")
    _validate_path_and_sha_fields(config)
    _validate_cross_fields(config)
    _validate_baseline_binding(config, baseline_config)
    return copy.deepcopy(config)


def load_training_contract(repo_root: str | Path, config_path: str | Path = TRAINING_CONFIG_RELATIVE_PATH) -> dict[str, Any]:
    """Load and validate the checked-in contract without runtime side effects."""

    root = Path(repo_root).resolve()
    relative = _assert_relative_path(str(config_path), "config_path")
    if relative != TRAINING_CONFIG_RELATIVE_PATH:
        raise TrainingContractError("training config path identity drift")
    training_raw = (root / relative).read_bytes()
    baseline_raw = (root / BASELINE_CONFIG_RELATIVE_PATH).read_bytes()
    baseline = _parse_portable_json(baseline_raw, label="baseline config")
    if len(baseline_raw) != BASELINE_CONFIG_SIZE_BYTES or _sha256_bytes(baseline_raw) != BASELINE_CONFIG_RAW_SHA256:
        raise TrainingContractError("baseline raw identity drift")
    if len(training_raw) != TRAINING_CONFIG_SIZE_BYTES or _sha256_bytes(training_raw) != TRAINING_CONFIG_RAW_SHA256:
        raise TrainingContractError("training config raw identity drift")
    config = _parse_portable_json(training_raw, label="training config")
    return validate_training_contract(config, baseline)


def _validate_vendor_source_files(root: Path, config: dict[str, Any]) -> None:
    bindings = [config["source_bindings"]["vendor_recipe"]]
    bindings.extend(config["source_bindings"]["vendor_includes"])
    for binding in bindings:
        relative = _assert_relative_path(binding["relative_path"], "vendor source path")
        path = root / relative
        if not path.is_file() or path.is_symlink() or _sha256_bytes(path.read_bytes()) != binding["sha256"]:
            raise TrainingContractError(f"vendor source identity drift: {relative}")


def training_contract_binding(repo_root: str | Path, config_path: str | Path = TRAINING_CONFIG_RELATIVE_PATH) -> dict[str, Any]:
    """Return raw/canonical identities after complete validation."""

    root = Path(repo_root).resolve()
    config = load_training_contract(root, config_path)
    _validate_vendor_source_files(root, config)
    raw = (root / TRAINING_CONFIG_RELATIVE_PATH).read_bytes()
    canonical = canonical_training_contract_bytes(config)
    return {
        "schema_version": 1,
        "training_contract_id": config["training_contract_id"],
        "baseline_id": config["baseline_id"],
        "relative_path": TRAINING_CONFIG_RELATIVE_PATH,
        "raw_size_bytes": len(raw),
        "raw_sha256": _sha256_bytes(raw),
        "canonical_size_bytes": len(canonical),
        "canonical_sha256": _sha256_bytes(canonical),
    }
