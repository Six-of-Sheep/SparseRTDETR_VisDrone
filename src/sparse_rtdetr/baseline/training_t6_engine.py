"""Production training engine boundary for the detached T6B launch path.

The module is import-safe.  Runtime libraries, models, data sources and the
primary evaluator are resolved only after the T6B entry has accepted a
detached authorization.  CPU tests use the same loop through narrow injected
ports and never need a vendor import.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import os
import pathlib
import random
import stat
from typing import Any, Callable, Iterable, Mapping


ENGINE_SCHEMA_VERSION = 1
ENGINE_ID = "rtdetrv2_r18_visdrone_training_t6_engine_v1"
ENGINE_MODULE_RELATIVE_PATH = "src/sparse_rtdetr/baseline/training_t6_engine.py"
ENGINE_CONTEXT_SCHEMA_VERSION = 1
ENGINE_CONTEXT_KIND = "T6B_PRODUCTION_EXECUTION_CONTEXT"
ENGINE_CLAIM_KIND = "T6B_ENGINE_EXECUTION_CLAIM"
ENGINE_CLAIM_FILE_NAME = "engine_execution_claim.json"


__all__ = (
    "TrainingEngineError",
    "validate_training_policy",
    "resolve_production_ports",
    "run_training_engine",
)


class TrainingEngineError(RuntimeError):
    """Raised when the frozen production training boundary is violated."""


def _fail(message: str) -> None:
    raise TrainingEngineError(message)


def _assert_builtin(value: Any, field: str = "value") -> None:
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                _fail(f"{field} has a non-string key")
            _assert_builtin(child, f"{field}.{key}")
        return
    if type(value) is list:
        for index, child in enumerate(value):
            _assert_builtin(child, f"{field}[{index}]")
        return
    if type(value) is float and not math.isfinite(value):
        _fail(f"{field} is non-finite")
    if type(value) not in {str, int, float, bool, type(None)}:
        _fail(f"{field} is not a builtin JSON value")


def _exact(value: Any, keys: set[str], field: str) -> dict[str, Any]:
    if type(value) is not dict:
        _fail(f"{field} must be a builtin dict")
    actual = set(value)
    if actual != keys:
        _fail(f"{field} keys drift: missing={sorted(keys - actual)} extra={sorted(actual - keys)}")
    return value


def _strict_equal(actual: Any, expected: Any, field: str) -> None:
    if type(actual) is not type(expected):
        _fail(f"{field} type drift")
    if type(actual) is dict:
        if set(actual) != set(expected):
            _fail(f"{field} keys drift")
        for key in expected:
            _strict_equal(actual[key], expected[key], f"{field}.{key}")
        return
    if type(actual) is list:
        if len(actual) != len(expected):
            _fail(f"{field} length drift")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _strict_equal(left, right, f"{field}[{index}]")
        return
    if actual != expected:
        _fail(f"{field} value drift")


def _finite_number(value: Any, field: str) -> float:
    if type(value) not in {int, float} or isinstance(value, bool):
        _fail(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{field} must be finite")
    return result


def _finite_observation(value: Any, field: str) -> None:
    if type(value) in {int, float} and not isinstance(value, bool):
        _finite_number(value, field)
    elif type(value) is dict:
        for key, child in value.items():
            _finite_observation(child, f"{field}.{key}")
    elif type(value) is list:
        for index, child in enumerate(value):
            _finite_observation(child, f"{field}[{index}]")


def _canonical(value: Any) -> bytes:
    _assert_builtin(value, "canonical value")
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise TrainingEngineError("value is not canonical JSON") from exc


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_string(value: Any, field: str) -> str:
    if type(value) is not str or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        _fail(f"{field} is not a lowercase SHA-256")
    return value


def _digest(value: Any) -> str:
    return _sha_bytes(_canonical(value))


def _copy(value: Any) -> Any:
    return copy.deepcopy(value)


def _read_context_file(path: pathlib.Path, field: str) -> tuple[dict[str, Any], bytes]:
    """Read one canonical evidence JSON file through the entry's stable reader."""

    from sparse_rtdetr.baseline import training_t6_entry as entry

    try:
        return entry._read_json(path, field)
    except entry.TrainingEntryError as exc:
        raise TrainingEngineError(str(exc)) from exc


def _read_context_bytes(path: pathlib.Path, field: str, *, mode: int = 0o600) -> bytes:
    from sparse_rtdetr.baseline import training_t6_entry as entry

    try:
        entry._regular_file(path, field, expected_mode=mode, expected_nlink=1)
        return entry._read_stable_file(path, field, mode=mode)
    except entry.TrainingEntryError as exc:
        raise TrainingEngineError(str(exc)) from exc


def _hex_bytes(value: Any, field: str) -> bytes:
    if type(value) is not str or not value or len(value) % 2:
        _fail(f"{field} must be an even-length hexadecimal string")
    try:
        return bytes.fromhex(value)
    except ValueError as exc:
        raise TrainingEngineError(f"{field} is not hexadecimal") from exc


def _validate_execution_context(value: Any) -> dict[str, Any]:
    """Revalidate the durable entry chain before any production factory runs."""

    keys = {
        "schema_version",
        "kind",
        "descriptor",
        "descriptor_sha256",
        "training_run_id",
        "nonce",
        "authorization",
        "evidence",
        "entry",
        "repository",
        "t6a_contract_binding",
        "source_bindings",
        "config_identity",
        "environment",
        "data_roles",
        "training_policy",
        "targets",
        "cwd",
        "argv",
        "child_environment",
        "evidence_root",
        "process_evidence_root",
        "outer_evidence_root",
        "engine_claim_path",
    }
    context = _exact(value, keys, "authorization context")
    _assert_builtin(context, "authorization context")
    if context["schema_version"] != ENGINE_CONTEXT_SCHEMA_VERSION or context["kind"] != ENGINE_CONTEXT_KIND:
        _fail("authorization context schema drift")

    from sparse_rtdetr.baseline import training_t6_entry as entry

    try:
        descriptor = entry.validate_entry_descriptor(context["descriptor"])
    except entry.TrainingEntryError as exc:
        raise TrainingEngineError(str(exc)) from exc
    if context["descriptor_sha256"] != descriptor["aggregate_sha256"]:
        _fail("authorization context descriptor digest drift")
    if context["training_run_id"] != descriptor["training_run_id"] or context["nonce"] != descriptor["nonce"]:
        _fail("authorization context run identity drift")
    for field in (
        "repository",
        "t6a_contract_binding",
        "source_bindings",
        "config_identity",
        "environment",
        "data_roles",
        "training_policy",
        "targets",
        "cwd",
        "argv",
        "child_environment",
        "evidence_root",
        "process_evidence_root",
        "outer_evidence_root",
    ):
        if context[field] != descriptor[field]:
            _fail(f"authorization context {field} drift")
    if context["engine_claim_path"] != str(pathlib.Path(descriptor["evidence_root"]) / ENGINE_CLAIM_FILE_NAME):
        _fail("authorization context engine claim path drift")

    authorization = _exact(
        context["authorization"],
        {"path", "binding", "binding_sha256", "receipt_path", "receipt_size_bytes", "receipt_sha256", "receipt_bytes_hex"},
        "authorization context.authorization",
    )
    if authorization["path"] != descriptor["authorization_path"] or authorization["binding"] != descriptor["authorization_binding"] or authorization["binding_sha256"] != descriptor["authorization_binding_sha256"] or authorization["receipt_path"] != descriptor["authorization_receipt_path"]:
        _fail("authorization context authorization binding drift")
    authorization_artifact_path = pathlib.Path(authorization["path"])
    authorization_artifact_raw = _read_context_bytes(authorization_artifact_path, "detached authorization artifact")
    binding = descriptor["authorization_binding"]
    if len(authorization_artifact_raw) != binding["raw_size_bytes"] or _sha_bytes(authorization_artifact_raw) != binding["raw_sha256"]:
        _fail("authorization context detached artifact identity drift")
    try:
        authorization_value, authorization_parsed_raw = entry._read_json(authorization_artifact_path, "detached authorization artifact", trailing_lf=True)
    except entry.TrainingEntryError as exc:
        raise TrainingEngineError(str(exc)) from exc
    if authorization_parsed_raw != authorization_artifact_raw or authorization_value != binding["authorization"] or _sha_bytes(entry._canonical(authorization_value)) != binding["canonical_sha256"]:
        _fail("authorization context detached artifact bytes drift")
    authorization_receipt_path = pathlib.Path(authorization["receipt_path"])
    authorization_raw = _read_context_bytes(authorization_receipt_path, "authorization consumption receipt")
    if authorization["receipt_size_bytes"] != len(authorization_raw) or authorization["receipt_sha256"] != _sha_bytes(authorization_raw) or _hex_bytes(authorization["receipt_bytes_hex"], "authorization context receipt_bytes_hex") != authorization_raw:
        _fail("authorization context receipt identity drift")
    try:
        consumed = entry.validate_consumed_authorization_receipt(
            str(authorization_receipt_path),
            authorization_binding=descriptor["authorization_binding"],
            authorization_path=descriptor["authorization_path"],
        )
    except entry.TrainingEntryError as exc:
        raise TrainingEngineError(str(exc)) from exc
    if consumed["receipt_sha256"] != authorization["receipt_sha256"]:
        _fail("authorization context consumed receipt drift")

    evidence = _exact(
        context["evidence"],
        {"root", "receipt_path", "receipt_size_bytes", "receipt_sha256", "receipt_bytes_hex"},
        "authorization context.evidence",
    )
    if evidence["root"] != descriptor["evidence_root"] or evidence["receipt_path"] != descriptor["evidence_receipt_path"]:
        _fail("authorization context evidence binding drift")
    evidence_path = pathlib.Path(evidence["receipt_path"])
    evidence_raw = _read_context_bytes(evidence_path, "training evidence claim receipt")
    if evidence["receipt_size_bytes"] != len(evidence_raw) or evidence["receipt_sha256"] != _sha_bytes(evidence_raw) or _hex_bytes(evidence["receipt_bytes_hex"], "authorization context evidence_bytes_hex") != evidence_raw:
        _fail("authorization context evidence receipt identity drift")
    claim, claim_raw = _read_context_file(evidence_path, "training evidence claim receipt")
    expected_claim = {
        "schema_version": 1,
        "kind": "T6B_ENTRY_EVIDENCE_CLAIM",
        "descriptor_sha256": descriptor["aggregate_sha256"],
        "descriptor": descriptor,
        "training_run_id": descriptor["training_run_id"],
        "nonce": descriptor["nonce"],
        "status": "CLAIMED",
    }
    if claim_raw != _canonical(claim) or claim != expected_claim or _sha_bytes(claim_raw) != evidence["receipt_sha256"]:
        _fail("authorization context evidence claim drift")

    entry_evidence = _exact(
        context["entry"],
        {
            "consumption_path",
            "consumption_size_bytes",
            "consumption_sha256",
            "consumption_bytes_hex",
            "invocation_path",
            "invocation_size_bytes",
            "invocation_sha256",
            "invocation_bytes_hex",
        },
        "authorization context.entry",
    )
    root = pathlib.Path(descriptor["evidence_root"])
    expected_consumption_path = root / "entry_consumption.json"
    expected_invocation_path = root / "entry_invocation.json"
    if entry_evidence["consumption_path"] != str(expected_consumption_path) or entry_evidence["invocation_path"] != str(expected_invocation_path):
        _fail("authorization context entry path drift")
    consumption_raw = _read_context_bytes(expected_consumption_path, "entry consumption")
    invocation_raw = _read_context_bytes(expected_invocation_path, "entry invocation")
    if entry_evidence["consumption_size_bytes"] != len(consumption_raw) or entry_evidence["consumption_sha256"] != _sha_bytes(consumption_raw) or _hex_bytes(entry_evidence["consumption_bytes_hex"], "authorization context consumption_bytes_hex") != consumption_raw:
        _fail("authorization context consumption identity drift")
    if entry_evidence["invocation_size_bytes"] != len(invocation_raw) or entry_evidence["invocation_sha256"] != _sha_bytes(invocation_raw) or _hex_bytes(entry_evidence["invocation_bytes_hex"], "authorization context invocation_bytes_hex") != invocation_raw:
        _fail("authorization context invocation identity drift")
    consumption, parsed_consumption_raw = _read_context_file(expected_consumption_path, "entry consumption")
    invocation, parsed_invocation_raw = _read_context_file(expected_invocation_path, "entry invocation")
    if parsed_consumption_raw != consumption_raw or parsed_invocation_raw != invocation_raw:
        _fail("authorization context entry readback drift")
    consumption_expected = {
        "schema_version": 1,
        "kind": "T6B_ENTRY_INVOCATION",
        "descriptor_sha256": descriptor["aggregate_sha256"],
        "authorization_binding_sha256": descriptor["authorization_binding_sha256"],
        "training_run_id": descriptor["training_run_id"],
        "nonce": descriptor["nonce"],
        "status": "CLAIMED",
        "receipt_path": str(expected_consumption_path),
        "consumed_authorization_receipt_sha256": authorization["receipt_sha256"],
        "evidence_claim_receipt_sha256": evidence["receipt_sha256"],
        "process_pid": consumption.get("process_pid"),
    }
    if consumption != consumption_expected or type(consumption["process_pid"]) is not int or consumption["process_pid"] < 0:
        _fail("authorization context entry consumption drift")
    invocation_expected = {
        "schema_version": 1,
        "status": "RUNNING",
        "descriptor_sha256": descriptor["aggregate_sha256"],
        "process_pid": consumption["process_pid"],
    }
    if invocation != invocation_expected:
        _fail("authorization context entry invocation drift")
    return _copy(context)


def _claim_engine_execution(context: dict[str, Any]) -> dict[str, Any]:
    """Exclusively publish the one engine-consumption claim before factories."""

    from sparse_rtdetr.baseline import training_t6_entry as entry

    root = pathlib.Path(context["evidence_root"])
    claim_path = pathlib.Path(context["engine_claim_path"])
    if claim_path != root / ENGINE_CLAIM_FILE_NAME or not root.is_dir() or root.is_symlink() or claim_path.exists() or claim_path.is_symlink():
        _fail("engine execution claim target already exists or is unsafe")
    try:
        root_stat = root.lstat()
        parent_stat = root.parent.lstat()
        if root.resolve(strict=True) != root or not stat.S_ISDIR(root_stat.st_mode) or root_stat.st_uid != os.getuid() or root_stat.st_gid != os.getgid() or root_stat.st_dev != parent_stat.st_dev:
            _fail("engine execution evidence root metadata drift")
    except OSError as exc:
        raise TrainingEngineError("engine execution evidence root is unavailable") from exc
    payload = {
        "schema_version": ENGINE_CONTEXT_SCHEMA_VERSION,
        "kind": ENGINE_CLAIM_KIND,
        "engine_id": ENGINE_ID,
        "context_sha256": _digest(context),
        "descriptor_sha256": context["descriptor_sha256"],
        "training_run_id": context["training_run_id"],
        "nonce": context["nonce"],
        "evidence_root": context["evidence_root"],
        "status": "CLAIMED",
        "context": _copy(context),
    }
    raw = _canonical(payload)
    # _write_new uses O_EXCL and the final pathname is the irreversible
    # reservation. Once it exists, every later failure must retain it so a
    # replay cannot regain access to the production factories.
    entry._write_new(claim_path, raw, "engine execution claim")
    entry._fsync_directory(root, "engine execution claim parent")
    observed = claim_path.lstat()
    root_observed = root.lstat()
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1 or stat.S_IMODE(observed.st_mode) != 0o600 or observed.st_uid != os.getuid() or observed.st_gid != os.getgid() or observed.st_dev != root_observed.st_dev:
        _fail("engine execution claim metadata drift")
    if _read_context_bytes(claim_path, "engine execution claim") != raw:
        _fail("engine execution claim readback drift")
    _validate_engine_claim(
        claim_path,
        expected_context_sha256=payload["context_sha256"],
        expected_descriptor_sha256=context["descriptor_sha256"],
        expected_root=context["evidence_root"],
    )
    return {
        "path": str(claim_path),
        "size_bytes": len(raw),
        "sha256": _sha_bytes(raw),
        "context_sha256": payload["context_sha256"],
    }


def _validate_engine_claim(path: pathlib.Path, *, expected_context_sha256: str, expected_descriptor_sha256: str, expected_root: str) -> dict[str, Any]:
    claim, raw = _read_context_file(path, "engine execution claim")
    expected = {"schema_version", "kind", "engine_id", "context_sha256", "descriptor_sha256", "training_run_id", "nonce", "evidence_root", "status", "context"}
    claim = _exact(claim, expected, "engine execution claim")
    observed = _read_context_bytes(path, "engine execution claim")
    if observed != raw or claim["schema_version"] != ENGINE_CONTEXT_SCHEMA_VERSION or claim["kind"] != ENGINE_CLAIM_KIND or claim["engine_id"] != ENGINE_ID or claim["context_sha256"] != expected_context_sha256 or claim["descriptor_sha256"] != expected_descriptor_sha256 or claim["evidence_root"] != expected_root or claim["status"] != "CLAIMED" or _digest(claim["context"]) != expected_context_sha256:
        _fail("engine execution claim identity drift")
    _validate_execution_context(claim["context"])
    return {"claim": claim, "raw": raw, "sha256": _sha_bytes(raw)}


def validate_training_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the frozen T1/T2 policy consumed by the real engine."""

    if type(policy) is not dict:
        _fail("training policy must be a builtin dict")
    required = {
        "initialization",
        "topology",
        "schedule",
        "optimizer",
        "learning_rate",
        "amp",
        "ema",
        "evaluator",
        "checkpoint",
        "execution",
    }
    value = _exact(policy, required, "training policy")
    _assert_builtin(value, "training policy")
    initialization = _exact(
        value["initialization"],
        {"random_initialization", "pretrained", "checkpoint", "seed"},
        "initialization",
    )
    _strict_equal(
        initialization,
        {"random_initialization": True, "pretrained": False, "checkpoint": None, "seed": 0},
        "initialization",
    )
    topology = _exact(
        value["topology"],
        {"world_size", "device", "train_batch_size", "development_batch_size"},
        "topology",
    )
    _strict_equal(
        topology,
        {"world_size": 1, "device": "cuda:0", "train_batch_size": 16, "development_batch_size": 32},
        "topology",
    )
    schedule = _exact(
        value["schedule"],
        {"epochs", "development_evaluation_every_epoch", "checkpoint_every_epoch"},
        "schedule",
    )
    _strict_equal(
        schedule,
        {"epochs": 120, "development_evaluation_every_epoch": True, "checkpoint_every_epoch": True},
        "schedule",
    )
    optimizer = _exact(
        value["optimizer"],
        {
            "type",
            "default_lr",
            "backbone_non_norm_lr",
            "betas",
            "default_weight_decay",
            "norm_bn_weight_decay",
            "clip_max_norm",
        },
        "optimizer",
    )
    _strict_equal(
        optimizer,
        {
            "type": "AdamW",
            "default_lr": 0.0001,
            "backbone_non_norm_lr": 0.00001,
            "betas": [0.9, 0.999],
            "default_weight_decay": 0.0001,
            "norm_bn_weight_decay": 0.0,
            "clip_max_norm": 0.1,
        },
        "optimizer",
    )
    learning_rate = _exact(
        value["learning_rate"],
        {
            "warmup_type",
            "warmup_optimizer_steps",
            "scheduler_type",
            "scheduler_step_unit",
            "milestones",
            "gamma",
            "expected_decay_events_within_120_epochs",
        },
        "learning_rate",
    )
    _strict_equal(
        learning_rate,
        {
            "warmup_type": "LinearWarmup",
            "warmup_optimizer_steps": 2000,
            "scheduler_type": "MultiStepLR",
            "scheduler_step_unit": "epoch",
            "milestones": [1000],
            "gamma": 0.1,
            "expected_decay_events_within_120_epochs": 0,
        },
        "learning_rate",
    )
    amp = _exact(
        value["amp"],
        {
            "enabled",
            "scaler_type",
            "init_scale",
            "growth_factor",
            "backoff_factor",
            "growth_interval",
            "nonfinite_events_allowed",
            "skipped_optimizer_steps_allowed",
            "overflow_events_allowed",
        },
        "amp",
    )
    _strict_equal(
        amp,
        {
            "enabled": True,
            "scaler_type": "GradScaler",
            "init_scale": 65536.0,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": 2000,
            "nonfinite_events_allowed": 0,
            "skipped_optimizer_steps_allowed": 0,
            "overflow_events_allowed": 0,
        },
        "amp",
    )
    ema = _exact(
        value["ema"],
        {"enabled", "decay", "warmup_optimizer_updates", "development_weights", "selection_weights"},
        "ema",
    )
    _strict_equal(
        ema,
        {
            "enabled": True,
            "decay": 0.9999,
            "warmup_optimizer_updates": 2000,
            "development_weights": "ema",
            "selection_weights": "ema",
        },
        "ema",
    )
    evaluator = _exact(
        value["evaluator"],
        {"primary_id", "protocol_id", "role", "secondary_can_certify"},
        "evaluator",
    )
    _strict_equal(
        evaluator,
        {
            "primary_id": "visdrone_official_primary_evaluator_v1",
            "protocol_id": "visdrone_official_style_v1",
            "role": "development_only",
            "secondary_can_certify": False,
        },
        "evaluator",
    )
    checkpoint = _exact(
        value["checkpoint"],
        {"roles", "required_atomicity", "required_loadability", "append_only"},
        "checkpoint",
    )
    _strict_equal(
        checkpoint,
        {"roles": ["last", "best", "periodic", "final"], "required_atomicity": True, "required_loadability": True, "append_only": True},
        "checkpoint",
    )
    execution = _exact(value["execution"], {"overwrite", "resume", "retry", "fallback"}, "execution")
    _strict_equal(execution, {"overwrite": False, "resume": False, "retry": False, "fallback": False}, "execution")
    return copy.deepcopy(value)


def _resolve_factory(ports: Mapping[str, Any], name: str) -> Any:
    value = ports.get(name)
    if callable(value):
        return value()
    if value is None:
        _fail(f"ports.{name} is required")
    return value


def _iter_batches(value: Any, field: str) -> Iterable[Any]:
    if callable(value):
        value = value()
    try:
        return iter(value)
    except TypeError as exc:
        raise TrainingEngineError(f"ports.{field} must provide an iterable") from exc


def _validate_runtime_data_roles(
    repo_root: str,
    data_roles: Mapping[str, Any] | None,
    baseline_config: Any,
) -> dict[str, dict[str, Any]]:
    """Bind only the two owner-authorized runtime roles to frozen manifests."""

    if type(data_roles) is not dict or set(data_roles) != {"train_core", "development", "test", "confirmatory"}:
        _fail("production data roles must be a closed object")
    if data_roles["test"] != {"role": "test", "access": "forbidden", "identity_read": "forbidden"}:
        _fail("test data role is not sealed")
    if data_roles["confirmatory"] != {"role": "confirmatory", "access": "sealed_and_forbidden", "identity_read": "forbidden"}:
        _fail("confirmatory data role is not sealed")

    pathlib = importlib.import_module("pathlib")
    stat = importlib.import_module("stat")
    root = pathlib.Path(repo_root)
    if not root.is_absolute() or root != root.resolve(strict=True):
        _fail("repository root is not canonical")
    forbidden = {"test", "confirmatory", "raw", "raw_annotation", "annotations"}
    checked: dict[str, dict[str, Any]] = {}
    for role in ("train_core", "development"):
        binding = data_roles[role]
        if type(binding) is not dict or set(binding) != {"role", "root", "manifest_sha256"}:
            _fail(f"production {role} binding is not closed")
        if binding["role"] != role or type(binding["root"]) is not str:
            _fail(f"production {role} binding is invalid")
        if not binding["root"].startswith("/") or "\\" in binding["root"] or "//" in binding["root"]:
            _fail(f"production {role} root is not canonical")
        if any(part in {"", ".", ".."} for part in binding["root"].split("/")[1:]):
            _fail(f"production {role} root has unsafe components")
        if type(binding["manifest_sha256"]) is not str or len(binding["manifest_sha256"]) != 64 or any(char not in "0123456789abcdef" for char in binding["manifest_sha256"]):
            _fail(f"production {role} manifest SHA is invalid")
        path = pathlib.Path(binding["root"])
        try:
            observed = path.lstat()
            canonical = path.resolve(strict=True)
        except OSError as exc:
            raise TrainingEngineError(f"production {role} root is unavailable") from exc
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode) or canonical != path:
            _fail(f"production {role} root is not a canonical directory")
        if any(part.casefold() in forbidden or "annotation" in part.casefold() for part in path.parts):
            _fail(f"production {role} root points into sealed data")
        checked[role] = {"binding": copy.deepcopy(binding), "root": path}
    train_root = checked["train_core"]["root"]
    development_root = checked["development"]["root"]
    if train_root == development_root or train_root in development_root.parents or development_root in train_root.parents:
        _fail("train_core and development roots overlap")

    artifact_root = root / "artifacts" / "data" / "visdrone_protocol_v2_conversion_r3"
    for role in ("train_core", "development"):
        runtime = baseline_config.build_runtime_config(str(root), str(checked[role]["root"]), role)
        if type(runtime) is not dict or type(runtime.get("dataloader")) is not dict:
            _fail(f"production {role} runtime binding is invalid")
        annotation = pathlib.Path(runtime["dataloader"].get("annotation_file", ""))
        manifest = artifact_root / f"{role}_manifest.json"
        try:
            annotation_stat = annotation.lstat()
            manifest_stat = manifest.lstat()
            annotation_sha = _sha_bytes(annotation.read_bytes())
            manifest_sha = _sha_bytes(manifest.read_bytes())
        except OSError as exc:
            raise TrainingEngineError(f"production {role} manifest is unavailable") from exc
        if stat.S_ISLNK(annotation_stat.st_mode) or not stat.S_ISREG(annotation_stat.st_mode) or annotation_stat.st_nlink != 1:
            _fail(f"production {role} annotation is not a regular unique file")
        if stat.S_ISLNK(manifest_stat.st_mode) or not stat.S_ISREG(manifest_stat.st_mode) or manifest_stat.st_nlink != 1:
            _fail(f"production {role} manifest is not a regular unique file")
        if manifest_sha != checked[role]["binding"]["manifest_sha256"]:
            _fail(f"production {role} manifest identity drift")
        checked[role].update({"runtime": runtime, "annotation": annotation, "annotation_sha256": annotation_sha, "manifest": manifest})
    return checked


def _supports_keyword(callable_value: Any, keyword: str) -> bool:
    try:
        inspect = importlib.import_module("inspect")
        parameters = inspect.signature(callable_value).parameters.values()
        return any(parameter.kind is inspect.Parameter.VAR_KEYWORD or parameter.name == keyword for parameter in parameters)
    except (TypeError, ValueError):
        return False


def _move_to_device(value: Any, device: Any) -> Any:
    if type(value) is dict:
        return {key: _move_to_device(child, device) for key, child in value.items()}
    if type(value) is list:
        return [_move_to_device(child, device) for child in value]
    mover = getattr(value, "to", None)
    if callable(mover):
        return mover(device)
    return value


def _split_training_batch(batch: Any) -> tuple[Any, Any] | None:
    if type(batch) in {tuple, list} and len(batch) == 2:
        return batch[0], batch[1]
    if type(batch) is dict and set(batch).issuperset({"images", "targets"}):
        return batch["images"], batch["targets"]
    return None


def _loss_total(loss: Any) -> Any:
    if type(loss) is dict:
        if not loss:
            _fail("criterion returned an empty loss dictionary")
        values = list(loss.values())
        total = values[0]
        for value in values[1:]:
            total = total + value
        return total
    return loss


def resolve_production_ports(
    repo_root: str | None = None,
    *,
    data_roles: Mapping[str, Any] | None = None,
    output_root: str | None = None,
) -> dict[str, Any]:
    """Resolve the certified runtime graph only after entry authorization.

    The repository vendors the upstream project as the ``src`` namespace;
    there is no importable ``vendor.rtdetrv2_pytorch`` package.  Keep all of
    these imports in this post-authorization resolver and use the project's
    own isolation helper so registrations do not leak into the caller.
    """

    try:
        torch = importlib.import_module("torch")
        pathlib_module = importlib.import_module("pathlib")
        baseline_config = importlib.import_module("sparse_rtdetr.baseline.config")
        baseline_dataset = importlib.import_module("sparse_rtdetr.baseline.dataset")
        baseline_postprocessor = importlib.import_module("sparse_rtdetr.baseline.postprocessor")
        primary_evaluator = importlib.import_module("sparse_rtdetr.baseline.primary_evaluator")
        root = str(pathlib_module.Path(repo_root if repo_root is not None else baseline_config._default_repo_root()).resolve(strict=True))
        vendor_root = baseline_config._vendor_root(root)
        vendor_config_path = baseline_config._vendor_config_path(root)
        with baseline_config._vendor_path(vendor_root):
            yaml_config_module = importlib.import_module("src.core.yaml_config")
            workspace_module = importlib.import_module("src.core.workspace")
            dataloader_module = importlib.import_module("src.data.dataloader")
            solver_module = importlib.import_module("src.solver")
            optim_module = importlib.import_module("src.optim")
            importlib.import_module("src.data")
            importlib.import_module("src.nn")
            importlib.import_module("src.zoo.rtdetr")
    except TrainingEngineError:
        raise
    except Exception as exc:
        raise TrainingEngineError("certified production runtime is unavailable") from exc

    YAMLConfig = getattr(yaml_config_module, "YAMLConfig", None)
    tasks = getattr(solver_module, "TASKS", None)
    grad_scaler = getattr(optim_module, "GradScaler", None)
    model_ema = getattr(optim_module, "ModelEMA", None)
    if not callable(YAMLConfig) or type(tasks) is not dict or not callable(grad_scaler) or not callable(model_ema):
        _fail("certified vendor API surface is incomplete")
    if "detection" not in tasks:
        _fail("certified vendor TASKS does not expose detection")
    visdrone_dataset = getattr(baseline_dataset, "VisDroneCocoDetection", None)
    visdrone_postprocessor = getattr(baseline_postprocessor, "VisDronePostProcessor", None)
    evaluate_primary = getattr(primary_evaluator, "evaluate_primary_v1", None)
    if not callable(visdrone_dataset) or not callable(visdrone_postprocessor) or not callable(evaluate_primary):
        _fail("certified baseline runtime API surface is incomplete")

    role_bindings = _validate_runtime_data_roles(root, data_roles, baseline_config)

    state: dict[str, Any] = {}

    def runtime_config() -> Any:
        config = state.get("config")
        if config is None:
            with baseline_config._vendor_path(vendor_root):
                config = YAMLConfig(
                    str(vendor_config_path),
                    PResNet={"pretrained": False},
                    num_classes=10,
                    remap_mscoco_category=False,
                    device="cuda:0",
                    output_dir=output_root,
                )
            state["config"] = config
        return config

    def model_factory() -> Any:
        model = runtime_config().model
        to = getattr(model, "to", None)
        if not callable(to):
            _fail("certified model does not expose to")
        return to(device="cuda:0")

    def criterion_factory() -> Any:
        return runtime_config().criterion

    def optimizer_factory() -> Any:
        return runtime_config().optimizer

    def scheduler_factory() -> Any:
        return runtime_config().lr_scheduler

    def warmup_scheduler_factory() -> Any:
        return runtime_config().lr_warmup_scheduler

    def scaler_factory() -> Any:
        return grad_scaler(
            init_scale=65536.0,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000,
        )

    def ema_factory() -> Any:
        return model_ema(runtime_config().model, decay=0.9999, warmups=2000)

    def _configured_component(value: Any, field: str) -> Any:
        if type(value) is not dict:
            _fail(f"production {field} configuration is invalid")
        payload = copy.deepcopy(value)
        component_type = payload.pop("type", None)
        if type(component_type) is not str or not component_type:
            _fail(f"production {field} type is missing")
        return workspace_module.create(component_type, runtime_config().global_cfg, **payload)

    def dataset_factory(role: str) -> Any:
        if role not in {"train_core", "development"}:
            _fail("production dataset factory requires detached train/development bindings")
        cached = state.setdefault("datasets", {}).get(role)
        if cached is not None:
            return cached
        loader_name = "train_dataloader" if role == "train_core" else "val_dataloader"
        loader_config = runtime_config().yaml_cfg.get(loader_name)
        if type(loader_config) is not dict or type(loader_config.get("dataset")) is not dict:
            _fail(f"production {role} dataloader configuration is invalid")
        dataset_config = loader_config["dataset"]
        transform = _configured_component(dataset_config.get("transforms"), f"{role} transforms")
        runtime = role_bindings[role]["runtime"]
        if runtime["dataloader"]["img_folder"] != str(role_bindings[role]["root"]):
            _fail(f"production {role} runtime root binding drift")
        dataset = visdrone_dataset(
            img_folder=runtime["dataloader"]["img_folder"],
            ann_file=runtime["dataloader"]["annotation_file"],
            transforms=transform,
            return_masks=False,
            remap_mscoco_category=False,
            vendor_root=vendor_root,
            role=role,
        )
        state.setdefault("datasets", {})[role] = dataset
        return dataset

    def _loader_factory(role: str) -> Any:
        cached = state.setdefault("loaders", {}).get(role)
        if cached is not None:
            return cached
        loader_name = "train_dataloader" if role == "train_core" else "val_dataloader"
        loader_config = runtime_config().yaml_cfg.get(loader_name)
        if type(loader_config) is not dict:
            _fail(f"production {role} dataloader configuration is invalid")
        collate = _configured_component(loader_config.get("collate_fn"), f"{role} collate")
        batch_size = runtime_config().get_rank_batch_size(loader_config)
        loader_class = getattr(dataloader_module, "DataLoader", None)
        if not callable(loader_class):
            _fail("certified vendor DataLoader is unavailable")
        loader = loader_class(
            dataset=dataset_factory(role),
            batch_size=batch_size,
            num_workers=loader_config.get("num_workers", 0),
            drop_last=loader_config.get("drop_last", role == "train_core"),
            collate_fn=collate,
            shuffle=loader_config.get("shuffle", role == "train_core"),
        )
        loader.shuffle = bool(loader_config.get("shuffle", role == "train_core"))
        state.setdefault("loaders", {})[role] = loader
        return loader

    def train_batches() -> Any:
        return _loader_factory("train_core")

    def development_batches() -> Any:
        loader = _loader_factory("development")
        state["development_loader"] = loader
        return loader

    def postprocessor_factory() -> Any:
        postprocessor = state.get("postprocessor")
        if postprocessor is None:
            postprocessor = visdrone_postprocessor(vendor_root=vendor_root)
            state["postprocessor"] = postprocessor
        return postprocessor

    class PrimaryEvaluator:
        role = "development_only"

        def evaluate(self, weights: Any, batches: Any, epoch: int) -> dict[str, Any]:
            del epoch
            contract = state.get("primary_contract")
            if contract is None:
                contract = primary_evaluator.load_primary_evaluator_contract(root)
                state["primary_contract"] = contract
            torch_module = torch
            model = getattr(weights, "module", weights)
            if not callable(getattr(model, "__call__", None)):
                _fail("EMA evaluation weights are not callable")
            evaluate_mode = getattr(model, "eval", None)
            if callable(evaluate_mode):
                evaluate_mode()
            loader = state.get("development_loader")
            dataset = getattr(loader, "dataset", None)
            protocol = importlib.import_module("sparse_rtdetr.data_protocol.evaluation")
            images = []
            detections = []
            ground_truth = []
            seen_images: set[str] = set()

            def _scalar(value: Any) -> int:
                item = getattr(value, "item", None)
                if callable(item):
                    value = item()
                if type(value) is not int:
                    _fail("development image identity is not an integer")
                return value

            def _image_identity(target: Any) -> tuple[str, int, int, int]:
                if type(target) is not dict:
                    _fail("development target must be a builtin dict")
                image_value = target.get("image_id")
                numeric_id = _scalar(image_value)
                stable = target.get("stable_image_id")
                image_record = None
                if type(stable) is not str or not stable:
                    vendor = getattr(dataset, "_vendor", None)
                    coco = getattr(vendor, "coco", None)
                    if coco is not None and hasattr(coco, "loadImgs"):
                        loaded = coco.loadImgs(numeric_id)
                        if type(loaded) is list and len(loaded) == 1 and type(loaded[0]) is dict:
                            image_record = loaded[0]
                            stable = image_record.get("stable_image_id")
                    if type(stable) is not str or not stable:
                        stable = str(numeric_id)
                sizes = target.get("orig_size")
                if sizes is None or len(sizes) != 2:
                    _fail("development target orig_size is invalid")
                width = _scalar(sizes[0])
                height = _scalar(sizes[1])
                if width <= 0 or height <= 0:
                    _fail("development target dimensions are invalid")
                if image_record is not None:
                    if image_record.get("stable_image_id") not in {None, stable}:
                        _fail("development stable image identity drift")
                return stable, numeric_id, width, height

            def _raw_ground_truth(numeric_id: int, stable: str, target: Any) -> list[Any]:
                vendor = getattr(dataset, "_vendor", None)
                coco = getattr(vendor, "coco", None)
                raw_annotations = getattr(coco, "imgToAnns", {}).get(numeric_id, []) if coco is not None else []
                rows = []
                if type(raw_annotations) is list and raw_annotations:
                    for index, annotation in enumerate(raw_annotations):
                        if type(annotation) is not dict:
                            _fail("development annotation row is invalid")
                        bbox = annotation.get("bbox")
                        if type(bbox) is not list or len(bbox) != 4:
                            _fail("development annotation box is invalid")
                        x, y, width, height = (float(item) for item in bbox)
                        if width <= 0 or height <= 0:
                            continue
                        category = annotation.get("category_id")
                        if type(category) is not int or category < 0 or category > 10:
                            _fail("development annotation category is invalid")
                        annotation_id = annotation.get("stable_annotation_id", f"{stable}:{index}")
                        rows.append(
                            protocol.PrimaryGroundTruth(
                                annotation_id=str(annotation_id),
                                image_id=stable,
                                category_id=category,
                                bbox_xyxy=(x, y, x + width, y + height),
                                area=float(annotation.get("area", width * height)),
                                ignore_region=category == 0,
                                ignored=bool(annotation.get("ignore", False) or annotation.get("iscrowd", 0)),
                            )
                        )
                    return rows
                boxes = target.get("boxes")
                labels = target.get("labels")
                areas = target.get("area")
                if boxes is None or labels is None:
                    _fail("development target lacks evaluator ground truth")
                for index, (box, label) in enumerate(zip(boxes, labels)):
                    values = tuple(float(item) for item in box)
                    if len(values) != 4 or values[2] <= values[0] or values[3] <= values[1]:
                        _fail("development target box is invalid")
                    label_value = _scalar(label)
                    category = label_value + 1 if 0 <= label_value <= 9 else label_value
                    area_value = float(areas[index].item() if areas is not None and hasattr(areas[index], "item") else (areas[index] if areas is not None else (values[2] - values[0]) * (values[3] - values[1])))
                    rows.append(
                        protocol.PrimaryGroundTruth(
                            annotation_id=f"{stable}:{index}",
                            image_id=stable,
                            category_id=category,
                            bbox_xyxy=values,
                            area=area_value,
                            ignore_region=category == 0,
                            ignored=False,
                        )
                    )
                return rows

            with torch_module.no_grad():
                for batch in batches:
                    split = _split_training_batch(batch)
                    if split is None:
                        _fail("development loader yielded a vendor-shaped batch pair")
                    samples, targets = split
                    if type(targets) is not list:
                        _fail("development targets must be a list")
                    device = getattr(samples, "device", None)
                    if device is None:
                        device = "cuda:0"
                    samples_for_model = samples.to(device) if callable(getattr(samples, "to", None)) else samples
                    outputs = model(samples_for_model)
                    sizes = torch_module.stack([target["orig_size"] for target in targets], dim=0)
                    output_values = outputs.values() if type(outputs) is dict else ()
                    first_output = next(iter(output_values), None)
                    if first_output is not None and callable(getattr(sizes, "to", None)):
                        sizes = sizes.to(getattr(first_output, "device", sizes.device))
                    processed = postprocessor_factory()(outputs, sizes)
                    stable_ids = []
                    for target in targets:
                        stable, numeric_id, width, height = _image_identity(target)
                        if stable in seen_images:
                            _fail("development image IDs are duplicated")
                        seen_images.add(stable)
                        stable_ids.append(stable)
                        images.append(protocol.PrimaryImageV2(stable, width, height))
                        ground_truth.extend(_raw_ground_truth(numeric_id, stable, target))
                    detections.extend(postprocessor_factory().to_detections(processed, stable_ids))
            if not images:
                _fail("development loader yielded no images")
            input_value = protocol.PrimaryEvaluatorInputV2(tuple(images), tuple(detections), tuple(ground_truth))
            result = evaluate_primary(input_value, contract)
            primary_evaluator.validate_primary_evaluator_result(result, input_value, contract)
            return {
                "primary_score": result.AP,
                "evaluator_id": primary_evaluator.PRIMARY_EVALUATOR_ID,
                "protocol_id": primary_evaluator.PRIMARY_PROTOCOL_ID,
                "role": "development_only",
            }

    class CheckpointWriter:
        def __init__(self) -> None:
            if type(output_root) is not str or not output_root:
                _fail("production checkpoint writer requires an evidence root")
            pathlib = importlib.import_module("pathlib")
            stat = importlib.import_module("stat")
            self._root = pathlib.Path(output_root)
            try:
                root_stat = self._root.lstat()
            except OSError as exc:
                raise TrainingEngineError("production checkpoint root is unavailable") from exc
            if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode) or self._root != self._root.resolve(strict=True):
                _fail("production checkpoint root is not canonical")
            self._checkpoint_root = self._root / "checkpoints"
            if self._checkpoint_root.exists() or self._checkpoint_root.is_symlink():
                checkpoint_stat = self._checkpoint_root.lstat()
                if stat.S_ISLNK(checkpoint_stat.st_mode) or not stat.S_ISDIR(checkpoint_stat.st_mode):
                    _fail("production checkpoint directory is invalid")
            else:
                os_module = importlib.import_module("os")
                os_module.mkdir(self._checkpoint_root, 0o700)
                fsync = getattr(os_module, "fsync", None)
                if not callable(fsync):
                    _fail("production checkpoint fsync is unavailable")
                directory_fd = os_module.open(self._root, os_module.O_RDONLY | getattr(os_module, "O_DIRECTORY", 0))
                try:
                    fsync(directory_fd)
                finally:
                    os_module.close(directory_fd)
            self._records: list[dict[str, Any]] = []

        def _write_exclusive(self, path: Any, payload: bytes, mode: int = 0o600) -> dict[str, Any]:
            os_module = importlib.import_module("os")
            stat = importlib.import_module("stat")
            flags = os_module.O_RDWR | os_module.O_CREAT | os_module.O_EXCL | getattr(os_module, "O_CLOEXEC", 0) | getattr(os_module, "O_NOFOLLOW", 0)
            fd = None
            try:
                fd = os_module.open(path, flags, mode)
                offset = 0
                while offset < len(payload):
                    try:
                        count = os_module.write(fd, payload[offset:])
                    except InterruptedError:
                        continue
                    if count <= 0:
                        _fail("checkpoint short write")
                    offset += count
                while True:
                    try:
                        os_module.fsync(fd)
                        break
                    except InterruptedError:
                        continue
                os_module.lseek(fd, 0, os_module.SEEK_SET)
                readback = b""
                while len(readback) < len(payload):
                    try:
                        chunk = os_module.read(fd, len(payload) - len(readback))
                    except InterruptedError:
                        continue
                    if not chunk:
                        _fail("checkpoint readback was truncated")
                    readback += chunk
                observed = os_module.fstat(fd)
                if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1 or stat.S_IMODE(observed.st_mode) != mode or observed.st_size != len(payload) or readback != payload:
                    _fail("checkpoint publication metadata drift")
                return {"size_bytes": len(payload), "sha256": _sha_bytes(payload), "mode": stat.S_IMODE(observed.st_mode), "nlink": observed.st_nlink}
            except OSError as exc:
                raise TrainingEngineError("checkpoint publication failed") from exc
            finally:
                if fd is not None:
                    os_module.close(fd)

        def save_atomic(self, role: str, epoch: int, state_value: dict[str, Any]) -> dict[str, Any]:
            if role not in {"last", "best", "periodic", "final"} or type(epoch) is not int or epoch <= 0:
                _fail("unknown checkpoint role")
            os_module = importlib.import_module("os")
            tempfile_module = importlib.import_module("tempfile")
            target = self._checkpoint_root / f"checkpoint-{role}-{epoch:04d}.pth"
            if target.exists() or target.is_symlink():
                _fail("checkpoint target already exists")
            fd, temporary = tempfile_module.mkstemp(prefix=f".{target.name}.", dir=str(self._checkpoint_root))
            try:
                with os_module.fdopen(fd, "wb") as stream:
                    torch.save(state_value, stream)
                    stream.flush()
                    os_module.fsync(stream.fileno())
                temporary_path = importlib.import_module("pathlib").Path(temporary)
                payload = temporary_path.read_bytes()
                if target.exists() or target.is_symlink():
                    _fail("checkpoint target appeared during publication")
                os_module.link(temporary, target)
                os_module.unlink(temporary)
                directory_fd = os_module.open(self._checkpoint_root, os_module.O_RDONLY | getattr(os_module, "O_DIRECTORY", 0))
                try:
                    os_module.fsync(directory_fd)
                finally:
                    os_module.close(directory_fd)
            except Exception:
                try:
                    os_module.unlink(temporary)
                except OSError:
                    pass
                raise
            target_raw = target.read_bytes()
            target_stat = target.lstat()
            if not target.is_file() or target.is_symlink() or target_stat.st_nlink != 1 or target_stat.st_size != len(target_raw):
                _fail("checkpoint target readback metadata drift")
            reference = {
                "role": role,
                "epoch": epoch,
                "path": str(target),
                "state_sha256": _digest(state_value),
                "file_size_bytes": len(target_raw),
                "file_sha256": _sha_bytes(target_raw),
            }
            self._records.append(copy.deepcopy(reference))
            inventory = {
                "schema_version": ENGINE_SCHEMA_VERSION,
                "records": copy.deepcopy(self._records),
                "inventory_sha256": _digest(self._records),
            }
            inventory_raw = _canonical(inventory)
            inventory_path = self._checkpoint_root / f"checkpoint-inventory-{len(self._records):04d}.json"
            self._write_exclusive(inventory_path, inventory_raw)
            directory_fd = os_module.open(self._checkpoint_root, os_module.O_RDONLY | getattr(os_module, "O_DIRECTORY", 0))
            try:
                os_module.fsync(directory_fd)
            finally:
                os_module.close(directory_fd)
            reference["inventory_path"] = str(inventory_path)
            reference["inventory_sha256"] = _sha_bytes(inventory_raw)
            return reference

        def is_loadable(self, reference: dict[str, Any]) -> bool:
            pathlib = importlib.import_module("pathlib")
            stat = importlib.import_module("stat")
            path = pathlib.Path(reference.get("path", ""))
            try:
                relative = path.resolve(strict=True).relative_to(self._checkpoint_root.resolve(strict=True))
                observed = path.lstat()
            except (OSError, ValueError):
                return False
            if len(relative.parts) != 1 or stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
                return False
            try:
                raw = path.read_bytes()
                if reference.get("file_size_bytes") != len(raw) or reference.get("file_sha256") != _sha_bytes(raw):
                    return False
                torch.load(path, map_location="cpu", weights_only=False)
                inventory_path = pathlib.Path(reference.get("inventory_path", ""))
                inventory_relative = inventory_path.resolve(strict=True).relative_to(self._checkpoint_root.resolve(strict=True))
                inventory_stat = inventory_path.lstat()
                if len(inventory_relative.parts) != 1 or stat.S_ISLNK(inventory_stat.st_mode) or not stat.S_ISREG(inventory_stat.st_mode) or inventory_stat.st_nlink != 1:
                    return False
                inventory_raw = inventory_path.read_bytes()
                if reference.get("inventory_sha256") != _sha_bytes(inventory_raw):
                    return False
                inventory = json.loads(inventory_raw.decode("utf-8"))
                records = inventory.get("records") if type(inventory) is dict else None
                if type(inventory) is not dict or type(records) is not list or inventory.get("inventory_sha256") != _digest(records):
                    return False
                if not any(type(record) is dict and record.get("path") == str(path) and record.get("file_sha256") == reference.get("file_sha256") for record in records):
                    return False
            except Exception:
                return False
            return True

    resolved: dict[str, Any] = {
        "torch": torch,
        "YAMLConfig": YAMLConfig,
        "TASKS": tasks,
        "GradScaler": grad_scaler,
        "ModelEMA": model_ema,
        "baseline_config": baseline_config,
        "VisDroneCocoDetection": visdrone_dataset,
        "VisDronePostProcessor": visdrone_postprocessor,
        "primary_evaluator": primary_evaluator,
        "model_factory": model_factory,
        "criterion_factory": criterion_factory,
        "optimizer_factory": optimizer_factory,
        "scheduler_factory": scheduler_factory,
        "warmup_scheduler_factory": warmup_scheduler_factory,
        "scaler_factory": scaler_factory,
        "ema_factory": ema_factory,
        "train_batches": train_batches,
        "development_batches": development_batches,
        "checkpoint_writer": CheckpointWriter,
        "evaluator": PrimaryEvaluator,
        "dataset_factory": dataset_factory,
        "postprocessor_factory": postprocessor_factory,
        "seed": lambda seed: torch.manual_seed(seed),
        "autocast_context": lambda enabled=True: torch.autocast(device_type="cuda", enabled=enabled, cache_enabled=True),
        "data_bindings": {
            role: {
                "root": str(value["root"]),
                "manifest_sha256": value["binding"]["manifest_sha256"],
                "annotation_sha256": value["annotation_sha256"],
            }
            for role, value in role_bindings.items()
        },
    }
    return resolved


def _loss_value(loss: Any) -> float:
    if type(loss) is dict:
        if not loss:
            _fail("criterion returned an empty loss dictionary")
        return _loss_value(_loss_total(loss))
    if type(loss) in {int, float} and not isinstance(loss, bool):
        return _finite_number(loss, "loss")
    item = getattr(loss, "item", None)
    if callable(item):
        return _finite_number(item(), "loss.item()")
    _fail("criterion must return a numeric or scalar-like loss")


def _backward(ports: Mapping[str, Any], scaled_loss: Any) -> None:
    backward = ports.get("backward")
    if callable(backward):
        backward(scaled_loss)
        return
    method = getattr(scaled_loss, "backward", None)
    if callable(method):
        method()
        return
    _fail("no backward port is available")


def _snapshot_value(value: Any) -> Any:
    if type(value) in {str, int, float, bool, type(None)}:
        return value
    if type(value) is dict:
        return {str(key): _snapshot_value(child) for key, child in value.items()}
    if type(value) is list:
        return [_snapshot_value(child) for child in value]
    return {"type": type(value).__name__}


def _checkpoint(writer: Any, role: str, epoch: int, state: Mapping[str, Any]) -> dict[str, Any]:
    save = getattr(writer, "save_atomic", None)
    if not callable(save):
        _fail("checkpoint writer must expose save_atomic")
    reference = save(role, epoch, copy.deepcopy(dict(state)))
    if type(reference) is not dict:
        _fail("checkpoint writer returned a non-dict reference")
    _assert_builtin(reference, "checkpoint reference")
    loadable = getattr(writer, "is_loadable", None)
    if not callable(loadable) or loadable(reference) is not True:
        _fail(f"checkpoint {role} is not loadable")
    return copy.deepcopy(reference)


def _validate_engine_result(value: Any) -> dict[str, Any]:
    """Validate the engine-owned terminal result before another layer binds it."""

    keys = {
        "schema_version",
        "engine_id",
        "mode",
        "seed",
        "epochs",
        "optimizer_type",
        "scheduler_type",
        "warmup_optimizer_steps",
        "optimizer_steps",
        "nonfinite_events",
        "skipped_optimizer_steps",
        "overflow_events",
        "evaluator_role",
        "test_accessed",
        "confirmatory_accessed",
        "speed_measured",
        "epoch_reports",
        "checkpoint_references",
        "terminal",
        "scientific_training_certified",
        "execution_context_sha256",
        "engine_claim_path",
        "engine_claim_size_bytes",
        "engine_claim_sha256",
        "engine_execution_count",
        "engine_result_sha256",
    }
    result = _exact(value, keys, "engine result")
    _assert_builtin(result, "engine result")
    if result["schema_version"] != ENGINE_SCHEMA_VERSION or result["engine_id"] != ENGINE_ID or result["mode"] not in {"production", "cpu_fake"}:
        _fail("engine result identity drift")
    if result["seed"] != 0 or result["epochs"] != 120 or result["optimizer_type"] != "AdamW" or result["scheduler_type"] != "MultiStepLR" or result["warmup_optimizer_steps"] != 2000:
        _fail("engine result policy drift")
    for field in ("optimizer_steps", "nonfinite_events", "skipped_optimizer_steps", "overflow_events"):
        if type(result[field]) is not int or result[field] < 0:
            _fail(f"engine result {field} drift")
    for field in ("test_accessed", "confirmatory_accessed", "speed_measured", "scientific_training_certified"):
        if result[field] is not False:
            _fail(f"engine result {field} must remain false")
    if result["evaluator_role"] != "development_only" or result["terminal"] != "TERMINAL_COMPLETE" or type(result["epoch_reports"]) is not list or type(result["checkpoint_references"]) is not list:
        _fail("engine result terminal evidence drift")
    if result["mode"] == "production":
        _sha_string(result["execution_context_sha256"], "engine result execution_context_sha256")
        if type(result["engine_claim_path"]) is not str or not result["engine_claim_path"].startswith("/"):
            _fail("engine result claim path drift")
        _sha_string(result["engine_claim_sha256"], "engine result engine_claim_sha256")
        if type(result["engine_claim_size_bytes"]) is not int or result["engine_claim_size_bytes"] <= 0 or result["engine_execution_count"] != 1:
            _fail("engine result execution claim drift")
    else:
        if result["execution_context_sha256"] is not None or result["engine_claim_path"] is not None or result["engine_claim_sha256"] is not None or result["engine_claim_size_bytes"] != 0 or result["engine_execution_count"] != 0:
            _fail("cpu fake mode published production evidence")
    _sha = result["engine_result_sha256"]
    if type(_sha) is not str or len(_sha) != 64 or any(char not in "0123456789abcdef" for char in _sha):
        _fail("engine result digest drift")
    body = _copy(result)
    del body["engine_result_sha256"]
    if _digest(body) != _sha:
        _fail("engine result aggregate drift")
    return _copy(result)


def run_training_engine(
    policy: Mapping[str, Any],
    ports: Mapping[str, Any] | None = None,
    *,
    mode: str = "production",
    authorization_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute the frozen 120-epoch loop through explicit runtime ports.

    ``cpu_fake`` is only a test mode.  It does not grant authorization and is
    never selected by the production entry descriptor.
    """

    checked_policy = validate_training_policy(policy)
    if mode not in {"production", "cpu_fake"}:
        _fail("unknown engine mode")
    checked_context: dict[str, Any] | None = None
    execution_claim: dict[str, Any] | None = None
    if mode == "production":
        if authorization_context is None:
            _fail("production engine requires a bound execution context")
        checked_context = _validate_execution_context(authorization_context)
        if ports is None:
            ports = resolve_production_ports(
                repo_root=checked_context["repository"]["repo_root"],
                data_roles=checked_context["data_roles"],
                output_root=checked_context["evidence_root"],
            )
        if type(ports) is not dict:
            _fail("production resolver must return a builtin dict")
        execution_claim = _claim_engine_execution(checked_context)
    elif authorization_context is not None:
        _fail("cpu fake mode rejects a production authorization context")
    if ports is None:
        _fail("cpu fake mode requires injected CPU ports")
    if type(ports) is not dict:
        _fail("ports must be a builtin dict")
    _assert_builtin({key: _snapshot_value(value) for key, value in ports.items()}, "ports")

    seed = ports.get("seed")
    if seed is None:
        random.seed(0)
    elif callable(seed):
        seed(0)
    else:
        _fail("ports.seed must be callable")

    model = _resolve_factory(ports, "model_factory")
    criterion = _resolve_factory(ports, "criterion_factory")
    optimizer = _resolve_factory(ports, "optimizer_factory")
    scheduler = _resolve_factory(ports, "scheduler_factory")
    warmup_value = ports.get("warmup_scheduler_factory")
    warmup = warmup_value() if callable(warmup_value) else warmup_value
    scaler = _resolve_factory(ports, "scaler_factory")
    ema = _resolve_factory(ports, "ema_factory")
    writer = _resolve_factory(ports, "checkpoint_writer")
    evaluator = _resolve_factory(ports, "evaluator")
    train_source = ports.get("train_batches")
    development_source = ports.get("development_batches")
    if train_source is None or development_source is None:
        _fail("train_core and development ports are required")
    if any(name in ports for name in ("test_batches", "confirmatory_batches", "speed_measurement")):
        _fail("test, confirmatory and speed ports are forbidden")
    if not callable(getattr(evaluator, "evaluate", None)):
        _fail("primary evaluator must expose evaluate")
    if getattr(evaluator, "role", "development_only") != "development_only":
        _fail("evaluator is not development-only")

    epochs = checked_policy["schedule"]["epochs"]
    optimizer_steps = 0
    nonfinite_events = 0
    skipped_steps = 0
    overflow_events = 0
    epoch_reports: list[dict[str, Any]] = []
    checkpoint_references: list[dict[str, Any]] = []
    best_score: float | None = None
    warmup_finished = warmup is None

    def checkpoint_binding(epoch_value: int) -> dict[str, Any]:
        runtime_identity = ports.get("checkpoint_identity")
        if runtime_identity is None:
            runtime_identity = {"data_bindings": ports.get("data_bindings", {})}
        return {
            "runtime_identity": _snapshot_value(runtime_identity),
            "training_policy": copy.deepcopy(checked_policy),
            "epoch": epoch_value,
            "optimizer_type": type(optimizer).__name__,
            "scheduler_type": type(scheduler).__name__,
            "warmup_optimizer_updates": optimizer_steps,
            "scaler_type": type(scaler).__name__,
            "ema_type": type(ema).__name__,
            "evaluator_role": "development_only",
        }

    for epoch in range(1, epochs + 1):
        batches = _iter_batches(train_source, "train_batches")
        batch_count = 0
        epoch_loss = 0.0
        for batch in batches:
            batch_count += 1
            zero_grad = getattr(optimizer, "zero_grad", None)
            if callable(zero_grad):
                zero_grad()
            forward = model if callable(model) else getattr(model, "__call__", None)
            if not callable(forward):
                _fail("model must be callable")
            split = _split_training_batch(batch)
            if split is None:
                model_input = batch
                criterion_input = batch
                output = forward(batch)
            else:
                samples, targets = split
                device = getattr(samples, "device", None)
                if device is None:
                    parameters = getattr(model, "parameters", None)
                    first_parameter = next(iter(parameters()), None) if callable(parameters) else None
                    device = getattr(first_parameter, "device", "cuda:0")
                model_input = _move_to_device(samples, device)
                criterion_input = _move_to_device(targets, device)
                autocast_factory = ports.get("autocast_context")
                contextlib = importlib.import_module("contextlib")
                context = autocast_factory(True) if mode == "production" and callable(autocast_factory) else contextlib.nullcontext()
                with context:
                    if _supports_keyword(forward, "targets"):
                        output = forward(model_input, targets=criterion_input)
                    else:
                        output = forward(model_input)
            loss_fn = criterion if callable(criterion) else getattr(criterion, "__call__", None)
            if not callable(loss_fn):
                _fail("criterion must be callable")
            if _supports_keyword(loss_fn, "epoch"):
                loss = loss_fn(output, criterion_input, epoch=epoch, step=batch_count - 1, global_step=optimizer_steps)
            else:
                loss = loss_fn(output, criterion_input)
            total_loss = _loss_total(loss)
            value = _loss_value(total_loss)
            if not math.isfinite(value):
                nonfinite_events += 1
                _fail("non-finite training loss")
            epoch_loss += value
            scale = getattr(scaler, "scale", None)
            scaled = scale(total_loss) if callable(scale) else total_loss
            _backward(ports, scaled)
            if callable(getattr(scaler, "unscale_", None)):
                scaler.unscale_(optimizer)
            clip_norm = checked_policy["optimizer"]["clip_max_norm"]
            if clip_norm > 0:
                torch_module = ports.get("torch")
                parameters = getattr(model, "parameters", None)
                if torch_module is not None and callable(parameters):
                    torch_module.nn.utils.clip_grad_norm_(parameters(), clip_norm)
            step = getattr(scaler, "step", None)
            scale_before = scaler.get_scale() if callable(getattr(scaler, "get_scale", None)) else None
            if callable(step):
                step_result = step(optimizer)
                if step_result is False:
                    skipped_steps += 1
            else:
                optimizer_step = getattr(optimizer, "step", None)
                if not callable(optimizer_step):
                    _fail("optimizer must expose step")
                optimizer_step()
            update = getattr(scaler, "update", None)
            if callable(update):
                update()
            scale_after = scaler.get_scale() if callable(getattr(scaler, "get_scale", None)) else None
            if scale_before is not None and scale_after is not None and scale_after < scale_before:
                overflow_events += 1
            optimizer_steps += 1
            ema_update = getattr(ema, "update", None)
            if not callable(ema_update):
                _fail("EMA port must expose update")
            ema_update(model)
            if warmup is not None:
                warmup_step = getattr(warmup, "step", None)
                if not callable(warmup_step):
                    _fail("warmup scheduler must expose step")
                warmup_step()
                finished = getattr(warmup, "finished", None)
                warmup_finished = bool(finished()) if callable(finished) else optimizer_steps >= checked_policy["learning_rate"]["warmup_optimizer_steps"]
        if batch_count == 0:
            _fail("train_core yielded no batches")

        scheduler_step = getattr(scheduler, "step", None)
        if not callable(scheduler_step):
            _fail("scheduler must expose step")
        if warmup_finished:
            scheduler_step()

        development_batches = _iter_batches(development_source, "development_batches")
        evaluation = evaluator.evaluate(ema, development_batches, epoch)
        _assert_builtin(evaluation, "evaluation")
        _finite_observation(evaluation, "evaluation")
        score = evaluation.get("primary_score") if type(evaluation) is dict else None
        if score is not None:
            score = _finite_number(score, "evaluation.primary_score")
        state = {"epoch": epoch, "loss": epoch_loss / batch_count, "evaluation": evaluation}
        state["training_binding"] = checkpoint_binding(epoch)
        last = _checkpoint(writer, "last", epoch, state)
        checkpoint_references.append({"role": "last", "epoch": epoch, "reference": last})
        if epoch % 10 == 0:
            periodic = _checkpoint(writer, "periodic", epoch, state)
            checkpoint_references.append({"role": "periodic", "epoch": epoch, "reference": periodic})
        if best_score is None or (score is not None and score > best_score):
            best_score = score if score is not None else best_score
            best = _checkpoint(writer, "best", epoch, state)
            checkpoint_references.append({"role": "best", "epoch": epoch, "reference": best})
        epoch_reports.append(
            {"epoch": epoch, "batch_count": batch_count, "mean_loss": epoch_loss / batch_count, "evaluation": evaluation}
        )

    final_state = {"epoch": epochs, "epochs": epochs, "best_score": best_score}
    final_state["training_binding"] = checkpoint_binding(epochs)
    final = _checkpoint(writer, "final", epochs, final_state)
    checkpoint_references.append({"role": "final", "epoch": epochs, "reference": final})

    counters = ports.get("amp_counters")
    if callable(counters):
        observed = counters()
        if type(observed) is not dict:
            _fail("amp_counters must return a builtin dict")
        if set(observed) != {"overflow_events", "skipped_optimizer_steps"}:
            _fail("amp_counters key set drift")
        if type(observed["overflow_events"]) is not int or type(observed["skipped_optimizer_steps"]) is not int:
            _fail("amp counters must be builtin ints")
        if observed["overflow_events"] < 0 or observed["skipped_optimizer_steps"] < 0:
            _fail("amp counters cannot be negative")
        overflow_events = observed["overflow_events"]
        skipped_steps = observed["skipped_optimizer_steps"]
    if nonfinite_events != 0 or skipped_steps != 0 or overflow_events != 0:
        _fail("AMP/non-finite policy was violated")

    body = {
        "schema_version": ENGINE_SCHEMA_VERSION,
        "engine_id": ENGINE_ID,
        "mode": mode,
        "seed": 0,
        "epochs": epochs,
        "optimizer_type": checked_policy["optimizer"]["type"],
        "scheduler_type": checked_policy["learning_rate"]["scheduler_type"],
        "warmup_optimizer_steps": checked_policy["learning_rate"]["warmup_optimizer_steps"],
        "optimizer_steps": optimizer_steps,
        "nonfinite_events": nonfinite_events,
        "skipped_optimizer_steps": skipped_steps,
        "overflow_events": overflow_events,
        "evaluator_role": "development_only",
        "test_accessed": False,
        "confirmatory_accessed": False,
        "speed_measured": False,
        "epoch_reports": epoch_reports,
        "checkpoint_references": checkpoint_references,
        "terminal": "TERMINAL_COMPLETE",
        "scientific_training_certified": False,
        "execution_context_sha256": _digest(checked_context) if checked_context is not None else None,
        "engine_claim_path": execution_claim["path"] if execution_claim is not None else None,
        "engine_claim_size_bytes": execution_claim["size_bytes"] if execution_claim is not None else 0,
        "engine_claim_sha256": execution_claim["sha256"] if execution_claim is not None else None,
        "engine_execution_count": 1 if execution_claim is not None else 0,
    }
    _assert_builtin(body, "engine result")
    result = {**body, "engine_result_sha256": _digest(body)}
    return _validate_engine_result(result)
