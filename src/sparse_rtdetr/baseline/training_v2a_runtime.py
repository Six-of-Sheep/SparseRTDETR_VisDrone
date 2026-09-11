"""CPU/fake runtime integration for the frozen baseline-v2a contract.

The module is intentionally inert at import time.  It binds every v2a policy
value through :func:`training_v2a_contract_binding`, and imports PyTorch and
the vendor backbone only inside the explicit CPU model factory.  The fake
runner exercises the same ordering and evidence boundaries without importing
PyTorch, reading a dataset, probing a GPU, or launching formal training.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib
import inspect
import json
import math
import os
import pathlib
import stat
import sys
from typing import Any, Iterable, Mapping

from sparse_rtdetr.baseline import training_v2a_contract as _contract


RUNTIME_SCHEMA_VERSION = 1
RUNTIME_ID = "rtdetrv2_r18_visdrone_baseline_v2a_runtime_t7c"
TARGET_RELATIVE_PATH = f"artifacts/training/{RUNTIME_ID}"
PROCESS_TERMINAL_KIND = "T7C_V2A_PROCESS_TERMINAL_RESULT"
CHECKPOINT_KIND = "T7C_V2A_CHECKPOINT"
PROGRESS_FILE_NAME = "epoch_progress.jsonl"
TERMINAL_FILE_NAME = "terminal_result.json"
CHECKPOINT_FILE_PREFIX = "checkpoint_"

__all__ = (
    "V2ARuntimeError",
    "build_v2a_runtime_policy",
    "bind_training_parent_identity",
    "bind_authorized_data_roots",
    "bind_environment_identity",
    "load_v2a_pretrained_backbone",
    "audit_v2a_optimizer_parameters",
    "build_v2a_optimizer",
    "build_v2a_runtime_components",
    "v2a_bf16_autocast_context",
    "select_v2a_development_candidate",
    "build_v2a_checkpoint",
    "validate_v2a_checkpoint",
    "EpochProgressWriter",
    "parse_v2a_terminal_stdout",
    "validate_v2a_runtime_closure",
    "run_v2a_fake_runtime",
)


class V2ARuntimeError(ValueError):
    """Raised when a v2a runtime input or evidence boundary is invalid."""


def _fail(message: str) -> None:
    raise V2ARuntimeError(message)


def _copy(value: Any) -> Any:
    return copy.deepcopy(value)


def _assert_builtin(value: Any, field: str = "value") -> None:
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                _fail(f"{field} contains a non-string key")
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


def _canonical(value: Any) -> bytes:
    _assert_builtin(value, "canonical value")
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise V2ARuntimeError("value is not canonical JSON") from exc


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_value(value: Any) -> str:
    return _sha_bytes(_canonical(value))


def _exact(value: Any, keys: set[str], field: str) -> dict[str, Any]:
    if type(value) is not dict:
        _fail(f"{field} must be a builtin dict")
    actual = set(value)
    if actual != keys:
        _fail(f"{field} key set drift: missing={sorted(keys - actual)} extra={sorted(actual - keys)}")
    return value


def _string(value: Any, field: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        _fail(f"{field} must be a builtin string")
    return value


def _integer(value: Any, field: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        _fail(f"{field} must be a builtin int")
    if minimum is not None and value < minimum:
        _fail(f"{field} is below its minimum")
    return value


def _finite_float(value: Any, field: str) -> float:
    if type(value) not in {int, float} or type(value) is bool or not math.isfinite(float(value)):
        _fail(f"{field} must be a finite number")
    return float(value)


def _absolute_path(value: Any, field: str) -> pathlib.Path:
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise V2ARuntimeError(f"{field} is not an absolute lexical path") from exc
    if type(raw) is not str:
        _fail(f"{field} is not an absolute lexical path")
    if not raw.startswith("/") or "\\" in raw or "\x00" in raw or "//" in raw:
        _fail(f"{field} is not an absolute lexical path")
    if any(part in {"", ".", ".."} for part in raw.split("/")[1:]):
        _fail(f"{field} has unsafe path components")
    return pathlib.Path(raw)


def _regular_identity(path: pathlib.Path, field: str, *, mode: int | None = None) -> dict[str, int]:
    try:
        observed = path.lstat()
    except OSError as exc:
        raise V2ARuntimeError(f"{field} is unavailable") from exc
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
        _fail(f"{field} is not a regular non-symlink file")
    if observed.st_nlink != 1:
        _fail(f"{field} nlink drift")
    if mode is not None and stat.S_IMODE(observed.st_mode) != mode:
        _fail(f"{field} mode drift")
    return {
        "device": observed.st_dev,
        "inode": observed.st_ino,
        "mode": stat.S_IMODE(observed.st_mode),
        "nlink": observed.st_nlink,
        "size_bytes": observed.st_size,
        "uid": observed.st_uid,
        "gid": observed.st_gid,
    }


def _directory_identity(path: pathlib.Path, field: str) -> dict[str, int]:
    try:
        observed = path.lstat()
    except OSError as exc:
        raise V2ARuntimeError(f"{field} is unavailable") from exc
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
        _fail(f"{field} is not a directory")
    return {
        "device": observed.st_dev,
        "inode": observed.st_ino,
        "mode": stat.S_IMODE(observed.st_mode),
        "nlink": observed.st_nlink,
        "uid": observed.st_uid,
        "gid": observed.st_gid,
    }


def _canonical_directory(value: Any, field: str) -> pathlib.Path:
    path = _absolute_path(value, field)
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise V2ARuntimeError(f"{field} is unavailable") from exc
    if resolved != path:
        _fail(f"{field} is not a canonical path")
    _directory_identity(path, field)
    return path


def _contract_from(value: Mapping[str, Any]) -> dict[str, Any]:
    if type(value) is not dict:
        _fail("contract binding must be a builtin dict")
    contract = value.get("contract")
    if type(contract) is not dict:
        _fail("contract binding has no detached contract")
    if value.get("contract_id") != _contract.V2A_CONTRACT_ID or contract.get("contract_id") != _contract.V2A_CONTRACT_ID:
        _fail("v2a contract binding identity drift")
    if value.get("baseline_id") != _contract.V2A_BASELINE_ID or contract.get("baseline_id") != _contract.V2A_BASELINE_ID:
        _fail("v2a baseline binding identity drift")
    return contract


def _binding_contract_sha(binding: Mapping[str, Any]) -> str:
    contract = _contract_from(binding)
    return _sha_value(contract)


def _target_path(repo_root: pathlib.Path, target_root: Any) -> pathlib.Path:
    if target_root is None:
        path = repo_root / TARGET_RELATIVE_PATH
    else:
        path = _absolute_path(target_root, "target_root")
    if any(token in str(path).casefold() for token in ("training_t6", "baseline_v1", "training_v1")):
        _fail("v2a target path overlaps a v1 or t6 target")
    return path


def bind_training_parent_identity(
    repo_root: str | os.PathLike[str],
    target_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Bind the existing training parent while requiring the v2a target absent."""

    root = _canonical_directory(os.fspath(repo_root), "repo_root")
    parent = root / "artifacts" / "training"
    before = _directory_identity(parent, "existing artifacts/training parent")
    target = _target_path(root, target_root)
    if target.exists() or target.is_symlink():
        _fail("v2a runtime target must be absent")
    try:
        parent_for_target = target.parent.resolve(strict=True)
    except OSError as exc:
        raise V2ARuntimeError("v2a target parent is unavailable") from exc
    if parent_for_target != target.parent or not parent_for_target.is_dir():
        _fail("v2a target parent is not canonical")
    return {
        "relative_parent": "artifacts/training",
        "parent_path": str(parent),
        "parent_identity": before,
        "target_path": str(target),
        "target_relative_path": TARGET_RELATIVE_PATH if target == root / TARGET_RELATIVE_PATH else None,
        "target_absent_before": True,
    }


def _same_directory_identity(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(left.get(key) == right.get(key) for key in ("device", "inode", "mode", "nlink", "uid", "gid"))


def _validate_parent_after(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "relative_parent",
        "parent_path",
        "parent_identity",
        "target_path",
        "target_relative_path",
        "target_absent_before",
    }
    _exact(snapshot, required, "training parent identity")
    parent = pathlib.Path(_string(snapshot["parent_path"], "training parent identity.parent_path"))
    target = pathlib.Path(_string(snapshot["target_path"], "training parent identity.target_path"))
    observed = _directory_identity(parent, "existing artifacts/training parent after fake run")
    if not _same_directory_identity(snapshot["parent_identity"], observed):
        _fail("existing artifacts/training parent identity changed")
    if target.is_symlink() or not target.is_dir():
        _fail("v2a fake target is not a regular directory")
    return {"before_after_identity_pass": True, "parent_identity": _copy(snapshot["parent_identity"]), "target_path": str(target)}


def bind_authorized_data_roots(
    binding: Mapping[str, Any],
    data_roots: Mapping[str, str | os.PathLike[str]],
) -> dict[str, Any]:
    """Bind only the independent train-core and development layouts."""

    contract = _contract_from(binding)
    if type(data_roots) is not dict or set(data_roots) != {"train_core", "development"}:
        _fail("authorized data roots must contain only train_core and development")
    declared = contract["source_bindings"]["data_roles"]
    result: dict[str, Any] = {}
    seen: set[tuple[int, int]] = set()
    for role in ("train_core", "development"):
        root = _canonical_directory(data_roots[role], f"{role} data root")
        identity = _directory_identity(root, f"{role} data root")
        key = (identity["device"], identity["inode"])
        if key in seen:
            _fail("train_core and development data roots overlap")
        seen.add(key)
        annotation = declared[role]["annotation"]
        if type(annotation) is not str or not annotation or "/" in annotation or annotation in {".", ".."}:
            _fail(f"{role} annotation declaration drift")
        annotation_path = root / annotation
        annotation_identity = _regular_identity(annotation_path, f"{role} annotation")
        result[role] = {
            "role": role,
            "root": str(root),
            "identity": identity,
            "annotation": annotation,
            "annotation_identity": annotation_identity,
        }
    return {"roles": result, "confirmatory_read": False, "test_read": False}


def bind_environment_identity(
    binding: Mapping[str, Any],
    observed: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an externally supplied environment observation without probing hardware."""

    contract = _contract_from(binding)
    if type(observed) is not dict:
        _fail("environment observation must be a builtin dict")
    required = {"gpu_name", "cuda_version", "graphics_clock_mhz", "exclusive_host", "other_training_load"}
    if not required <= set(observed):
        _fail(f"environment observation missing={sorted(required - set(observed))}")
    environment = contract["environment_identity"]
    hardware = contract["hardware_launch_gate"]
    gpu_name = _string(observed["gpu_name"], "environment.gpu_name")
    cuda_version = _string(observed["cuda_version"], "environment.cuda_version")
    clock = _finite_float(observed["graphics_clock_mhz"], "environment.graphics_clock_mhz")
    exclusive = observed["exclusive_host"]
    load = observed["other_training_load"]
    if type(exclusive) is not bool or exclusive is not True:
        _fail("environment is not exclusive")
    if type(load) is not int or load != 0:
        _fail("other training load is nonzero")
    if gpu_name != environment["gpu_name"] or gpu_name != hardware["gpu_name"]:
        _fail("environment GPU identity drift")
    if cuda_version != environment["cuda_version"] or cuda_version != hardware["cuda_version"]:
        _fail("environment CUDA identity drift")
    if clock > float(environment["graphics_clock_max_mhz"]) or clock > float(hardware["graphics_clock_upper_limit_mhz"]):
        _fail("environment graphics clock exceeds v2a cap")
    result = {
        "observation_mode": "supplied_cpu_fake_observation",
        "gpu_name": gpu_name,
        "cuda_version": cuda_version,
        "graphics_clock_mhz": clock,
        "graphics_clock_cap_mhz": environment["graphics_clock_max_mhz"],
        "exclusive_host": True,
        "other_training_load": 0,
        "hardware_probe_executed": False,
    }
    if "observation_id" in observed:
        result["observation_id"] = _string(observed["observation_id"], "environment.observation_id")
    return result


def _policy_parts(binding: Mapping[str, Any]) -> dict[str, Any]:
    contract = _contract_from(binding)
    topology = contract["topology"]
    optimizer = contract["optimizer"]
    amp = contract["amp"]
    ema = contract["ema"]
    selection = contract["evaluation_and_selection"]
    checkpoint = contract["checkpoint_policy"]
    if topology["train_micro_batch"] * topology["gradient_accumulation_steps"] != topology["effective_train_batch"]:
        _fail("v2a effective batch arithmetic drift")
    if topology["gradient_accumulation_steps"] != 1 or topology["runtime_batch_adaptation_forbidden"] is not True:
        _fail("v2a accumulation or batch-adaptation policy drift")
    if amp["autocast_dtype"] != "bfloat16" or amp["grad_scaler_enabled"] is not False or amp["scaler_policy"] != "disabled_absent":
        _fail("v2a bf16/no-scaler policy drift")
    if any(group["learning_rate"] != optimizer["default_lr"] for group in optimizer["parameter_groups"]):
        _fail("v2a optimizer group learning-rate replay drift")
    return {
        "model": _copy(contract["model"]),
        "topology": {
            "train_micro_batch": topology["train_micro_batch"],
            "gradient_accumulation_steps": topology["gradient_accumulation_steps"],
            "effective_train_batch": topology["effective_train_batch"],
            "development_batch": topology["development_batch"],
            "runtime_batch_adaptation_forbidden": topology["runtime_batch_adaptation_forbidden"],
        },
        "optimizer": _copy(optimizer),
        "amp": _copy(amp),
        "ema": _copy(ema),
        "selection": _copy(selection),
        "checkpoint": _copy(checkpoint),
        "data_roles": _copy(contract["data_roles"]),
    }


def build_v2a_runtime_policy(
    repo_root: str | os.PathLike[str],
    *,
    data_roots: Mapping[str, str | os.PathLike[str]] | None = None,
    environment_identity: Mapping[str, Any] | None = None,
    target_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Bind the frozen contract and all five runtime closure inputs."""

    binding = _contract.training_v2a_contract_binding(repo_root)
    parts = _policy_parts(binding)
    parent = bind_training_parent_identity(repo_root, target_root)
    data_binding = bind_authorized_data_roots(binding, data_roots) if data_roots is not None else None
    environment_binding = bind_environment_identity(binding, environment_identity) if environment_identity is not None else None
    return {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "runtime_id": RUNTIME_ID,
        "contract_id": binding["contract_id"],
        "baseline_id": binding["baseline_id"],
        "contract_binding": _copy(binding),
        "contract_sha256": _binding_contract_sha(binding),
        "run_identity": {
            "run_id": RUNTIME_ID,
            "target_relative_path": TARGET_RELATIVE_PATH,
            "progress_file_name": PROGRESS_FILE_NAME,
            "terminal_file_name": TERMINAL_FILE_NAME,
            "checkpoint_file_prefix": CHECKPOINT_FILE_PREFIX,
        },
        "parent_identity": parent,
        "data_roots": data_binding,
        "environment_identity": environment_binding,
        **parts,
        "runtime_closure": {
            "process_terminal_json": {"required": True, "bound": False},
            "training_parent_identity": {"required": True, "bound": True},
            "authorization_data_roots": {"required": True, "bound": data_binding is not None},
            "environment_identity": {"required": True, "bound": environment_binding is not None},
            "epoch_progress_evidence": {"required": True, "bound": False},
        },
        "production": {
            "owner_authorization_consumed": False,
            "gpu_probe_executed": False,
            "formal_training_executed": False,
            "training_ready": False,
            "independent_audit_pass": False,
        },
    }


def _load_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except Exception as exc:
        raise V2ARuntimeError("PyTorch is required only for the explicit CPU model factory") from exc


def _vendor_presnet_class(source_path: pathlib.Path) -> type[Any]:
    vendor_root = source_path.parents[3]
    old_path = list(sys.path)
    sys.path.insert(0, str(vendor_root))
    try:
        module = importlib.import_module("src.nn.backbone.presnet")
    except Exception as exc:
        raise V2ARuntimeError("vendor PResNet import failed") from exc
    finally:
        sys.path[:] = old_path
    loaded = pathlib.Path(getattr(module, "__file__", "")).resolve()
    if loaded != source_path.resolve() or not hasattr(module, "PResNet"):
        _fail("vendor PResNet source identity drift")
    return module.PResNet


def _authority_weight_path(repo_root: str | os.PathLike[str], binding: Mapping[str, Any]) -> pathlib.Path:
    root = _canonical_directory(os.fspath(repo_root), "repo_root")
    contract = _contract_from(binding)
    authority = binding["authority_binding"]
    weight = authority["weight"]
    relative = _string(
        contract["source_bindings"]["pretrained_authority"]["weight"]["relative_path"],
        "authority weight.relative_path",
    )
    path = root / relative
    observed = _regular_identity(path, "pretrained authority weight", mode=weight["mode"])
    raw = path.read_bytes()
    if len(raw) != weight["size_bytes"] or _sha_bytes(raw) != weight["sha256"]:
        _fail("pretrained authority weight content identity drift")
    if observed["nlink"] != weight["nlink"]:
        _fail("pretrained authority weight nlink drift")
    return path


def load_v2a_pretrained_backbone(
    repo_root: str | os.PathLike[str],
    *,
    binding: Mapping[str, Any] | None = None,
    torch_module: Any | None = None,
    presnet_class: type[Any] | None = None,
) -> dict[str, Any]:
    """Load the local PResNet authority on CPU with a strict key match."""

    bound = binding if binding is not None else _contract.training_v2a_contract_binding(repo_root)
    contract = _contract_from(bound)
    weight_path = _authority_weight_path(repo_root, bound)
    authority = bound["authority_binding"]
    validation = authority["manifest_load_validation"]
    constructor = _copy(validation["constructor"])
    if constructor.get("pretrained") is not False:
        _fail("vendor PResNet constructor must disable network pretrained loading")
    torch = torch_module if torch_module is not None else _load_torch()
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and callable(getattr(cuda, "is_initialized", None)) and cuda.is_initialized():
        _fail("CUDA was initialized before v2a CPU backbone load")
    try:
        state = torch.load(str(weight_path), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise V2ARuntimeError("pretrained authority CPU load failed") from exc
    if not isinstance(state, Mapping):
        _fail("pretrained authority state is not a mapping")
    key_count = len(state)
    tensor_numel = 0
    for key, value in state.items():
        if type(key) is not str or not callable(getattr(value, "numel", None)):
            _fail("pretrained authority state inventory is not tensor-like")
        tensor_numel += int(value.numel())
    if key_count != validation["state_key_count"] or tensor_numel != validation["tensor_numel"]:
        _fail("pretrained authority state inventory drift")
    cls = presnet_class if presnet_class is not None else _vendor_presnet_class(
        pathlib.Path(repo_root) / contract["source_bindings"]["vendor_r18"]["presnet_source"]["relative_path"]
    )
    try:
        model = cls(**constructor)
        loaded = model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise V2ARuntimeError("pretrained authority strict load failed") from exc
    missing = list(getattr(loaded, "missing_keys", ()))
    unexpected = list(getattr(loaded, "unexpected_keys", ()))
    if missing or unexpected:
        _fail(f"pretrained authority strict load key drift: missing={missing} unexpected={unexpected}")
    if callable(getattr(cuda, "is_initialized", None)) and cuda.is_initialized():
        _fail("CUDA initialized during v2a CPU backbone load")
    return {
        "model": model,
        "architecture": validation["architecture"],
        "constructor": constructor,
        "weight": {
            "relative_path": contract["source_bindings"]["pretrained_authority"]["weight"]["relative_path"],
            "size_bytes": authority["weight"]["size_bytes"],
            "sha256": authority["weight"]["sha256"],
        },
        "state_key_count": key_count,
        "tensor_numel": tensor_numel,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "strict_load_pass": True,
        "cuda_initialized": False,
    }


def _named_parameters(model_or_named: Any) -> list[tuple[str, Any]]:
    source = model_or_named.named_parameters() if callable(getattr(model_or_named, "named_parameters", None)) else model_or_named
    try:
        rows = list(source)
    except TypeError as exc:
        raise V2ARuntimeError("model named_parameters is not iterable") from exc
    result: list[tuple[str, Any]] = []
    seen_names: set[str] = set()
    seen_params: set[int] = set()
    for index, row in enumerate(rows):
        if type(row) not in {tuple, list} or len(row) != 2:
            _fail(f"parameter row {index} is not a name/value pair")
        name, parameter = row
        if type(name) is not str or not name or name in seen_names:
            _fail("parameter names are not unique non-empty strings")
        seen_names.add(name)
        if getattr(parameter, "requires_grad", True) is False:
            continue
        identity = id(parameter)
        if identity in seen_params:
            _fail("trainable parameter identities overlap")
        seen_params.add(identity)
        result.append((name, parameter))
    if not result:
        _fail("model has no trainable parameters")
    return result


def _bound_group_for_name(name: str, optimizer: Mapping[str, Any]) -> str:
    groups = optimizer["parameter_groups"]
    matches = []
    for group in groups:
        includes = group["include_substrings"]
        excludes = group["exclude_substrings"]
        if any(token in name for token in includes) and not any(token in name for token in excludes):
            matches.append(group["name"])
    if len(matches) > 1:
        _fail(f"optimizer parameter selector overlap: {name}")
    if matches:
        return matches[0]
    default = [group for group in groups if group["name"] == "default"]
    if len(default) != 1 or default[0]["include_substrings"] or default[0]["exclude_substrings"]:
        _fail("optimizer default remainder selector drift")
    return "default"


def audit_v2a_optimizer_parameters(
    model_or_named_parameters: Any,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit complete, mutually exclusive v2a trainable parameter ownership."""

    contract = _contract_from(binding)
    optimizer = contract["optimizer"]
    groups = optimizer["parameter_groups"]
    if [group["name"] for group in groups] != ["norm_or_bn", "default"]:
        _fail("v2a optimizer group order drift")
    if groups[0]["include_substrings"] != ["norm", "bn"] or groups[0]["exclude_substrings"] != []:
        _fail("v2a norm/bn selector drift")
    if groups[1]["parameter_scope"] != "remaining_trainable" or groups[1]["include_substrings"] != [] or groups[1]["exclude_substrings"] != []:
        _fail("v2a default remainder selector drift")
    names = _named_parameters(model_or_named_parameters)
    rows = []
    grouped: dict[str, list[Any]] = {group["name"]: [] for group in groups}
    for name, parameter in names:
        group_name = _bound_group_for_name(name, optimizer)
        grouped[group_name].append(parameter)
        rows.append({"parameter_name": name, "group": group_name})
    if sum(len(values) for values in grouped.values()) != len(names):
        _fail("optimizer groups do not cover all trainable parameters")
    default_lr = optimizer["default_lr"]
    for group in groups:
        if group["learning_rate"] != default_lr:
            _fail("v2a backbone learning-rate override detected")
    return {
        "groups": _copy(groups),
        "parameter_rows": rows,
        "group_counts": {name: len(values) for name, values in grouped.items()},
        "parameter_count": len(names),
        "mutually_exclusive": True,
        "covered_all_trainable_parameters": True,
        "backbone_non_norm_lr_override": False,
        "identity_key": "parameter_name",
        "_parameter_objects": grouped,
    }


def build_v2a_optimizer(
    model: Any,
    binding: Mapping[str, Any],
    *,
    optimizer_factory: Any | None = None,
) -> dict[str, Any]:
    """Create AdamW only after the strict authority load has completed."""

    audit = audit_v2a_optimizer_parameters(model, binding)
    contract = _contract_from(binding)
    optimizer_spec = contract["optimizer"]
    parameter_groups = []
    for group in optimizer_spec["parameter_groups"]:
        parameter_groups.append(
            {
                "name": group["name"],
                "params": audit["_parameter_objects"][group["name"]],
                "lr": group["learning_rate"],
                "weight_decay": group["weight_decay"],
            }
        )
    if optimizer_factory is None:
        torch = _load_torch()
        optimizer_factory = torch.optim.AdamW
    try:
        optimizer = optimizer_factory(
            parameter_groups,
            betas=tuple(optimizer_spec["betas"]),
            eps=1e-8,
        )
    except Exception as exc:
        raise V2ARuntimeError("v2a AdamW construction failed") from exc
    return {"optimizer": optimizer, "audit": {key: value for key, value in audit.items() if key != "_parameter_objects"}, "parameter_groups": parameter_groups}


def build_v2a_runtime_components(
    repo_root: str | os.PathLike[str],
    *,
    data_roots: Mapping[str, str | os.PathLike[str]],
    environment_identity: Mapping[str, Any],
    target_root: str | os.PathLike[str] | None = None,
    torch_module: Any | None = None,
    presnet_class: type[Any] | None = None,
    optimizer_factory: Any | None = None,
) -> dict[str, Any]:
    """Execute the detached positive chain: bind, strict-load, then optimize."""

    policy = build_v2a_runtime_policy(
        repo_root,
        data_roots=data_roots,
        environment_identity=environment_identity,
        target_root=target_root,
    )
    loaded = load_v2a_pretrained_backbone(
        repo_root,
        binding=policy["contract_binding"],
        torch_module=torch_module,
        presnet_class=presnet_class,
    )
    optimizer = build_v2a_optimizer(loaded["model"], policy["contract_binding"], optimizer_factory=optimizer_factory)
    return {"policy": policy, "pretrained": {key: value for key, value in loaded.items() if key != "model"}, "model": loaded["model"], **optimizer}


def v2a_bf16_autocast_context(
    binding: Mapping[str, Any],
    *,
    torch_module: Any | None = None,
) -> Any:
    """Return CUDA bfloat16 autocast; no scaler object exists in this path."""

    contract = _contract_from(binding)
    amp = contract["amp"]
    if amp["autocast_dtype"] != "bfloat16" or amp["grad_scaler_enabled"] is not False:
        _fail("v2a autocast/no-scaler policy drift")
    torch = torch_module if torch_module is not None else _load_torch()
    dtype = getattr(torch, "bfloat16", None)
    autocast = getattr(torch, "autocast", None)
    if dtype is None or not callable(autocast):
        _fail("torch bfloat16 autocast is unavailable")
    return autocast(device_type="cuda", dtype=dtype, enabled=True)


def select_v2a_development_candidate(
    candidates: Iterable[Mapping[str, Any]],
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Select EMA-backed development output using unrounded float64 ordering."""

    contract = _contract_from(binding)
    selection = contract["evaluation_and_selection"]
    if selection["selection_direction"] != "maximize" or selection["metric_comparison_precision"] != "unrounded_float64":
        _fail("v2a selection policy drift")
    rows = list(candidates)
    if not rows:
        _fail("development candidate list is empty")
    normalized = []
    for index, candidate in enumerate(rows):
        if type(candidate) is not dict:
            _fail(f"development candidate {index} is not a builtin dict")
        required = {"epoch", "AP", "AP50", "AR500", "weights"}
        if not required <= set(candidate):
            _fail(f"development candidate {index} is missing fields")
        epoch = _integer(candidate["epoch"], f"development candidate {index}.epoch", minimum=1)
        values = {field: _finite_float(candidate[field], f"development candidate {index}.{field}") for field in ("AP", "AP50", "AR500")}
        if candidate["weights"] != contract["ema"]["model_selection_weights"] or candidate["weights"] != "ema":
            _fail("development selection candidate is not EMA-backed")
        normalized.append({**_copy(candidate), "epoch": epoch, **values})
    return _copy(max(normalized, key=lambda row: (row["AP"], row["AP50"], row["AR500"], -row["epoch"])))


def _state_payload(value: Any, field: str) -> Any:
    if callable(value):
        value = value()
    _assert_builtin(value, field)
    return _copy(value)


def build_v2a_checkpoint(
    policy: Mapping[str, Any],
    *,
    epoch: int,
    optimizer_updates: int,
    raw_model_state: Any,
    ema_state: Any,
    environment_identity: Mapping[str, Any] | None = None,
    source_identity: Mapping[str, Any] | None = None,
    optimizer_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the independent v2a checkpoint identity without scaler state."""

    _exact(policy, {
        "schema_version", "runtime_id", "contract_id", "baseline_id", "contract_binding", "contract_sha256",
        "run_identity", "parent_identity", "data_roots", "environment_identity", "model", "topology", "optimizer",
        "amp", "ema", "selection", "checkpoint", "data_roles", "runtime_closure", "production",
    }, "v2a runtime policy")
    binding = policy["contract_binding"]
    contract = _contract_from(binding)
    epoch = _integer(epoch, "checkpoint.epoch", minimum=1)
    optimizer_updates = _integer(optimizer_updates, "checkpoint.optimizer_updates", minimum=1)
    raw_state = _state_payload(raw_model_state, "checkpoint.raw_model")
    ema_payload = _state_payload(ema_state, "checkpoint.ema")
    if raw_state == ema_payload:
        _fail("checkpoint raw and EMA states are not distinct")
    environment = environment_identity if environment_identity is not None else policy.get("environment_identity")
    if type(environment) is not dict:
        _fail("checkpoint environment identity is unbound")
    source = source_identity if source_identity is not None else {
        "runtime_module": "src/sparse_rtdetr/baseline/training_v2a_runtime.py",
        "runtime_module_sha256": _sha_bytes(pathlib.Path(__file__).read_bytes()),
    }
    _assert_builtin(source, "checkpoint.source_identity")
    audit = optimizer_audit if optimizer_audit is not None else {"parameter_rows": [], "group_counts": {"norm_or_bn": 0, "default": 0}}
    _assert_builtin(audit, "checkpoint.optimizer_parameter_groups")
    checkpoint = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "kind": CHECKPOINT_KIND,
        "runtime_id": policy["runtime_id"],
        "contract_id": contract["contract_id"],
        "baseline_id": contract["baseline_id"],
        "contract_identity": {
            "canonical_sha256": policy["contract_sha256"],
            "config_raw_sha256": binding["raw_sha256"],
            "config_canonical_sha256": binding["canonical_sha256"],
        },
        "authority_identity": {
            "root_relative_path": binding["authority_binding"]["root_relative_path"],
            "weight": _copy(binding["authority_binding"]["weight"]),
            "manifest": _copy(binding["authority_binding"]["manifest"]),
        },
        "model_identity": {"backbone": contract["model"]["backbone"], "input_size": _copy(contract["model"]["input_size"])},
        "optimizer_parameter_groups": _copy(audit),
        "amp_state": {
            "enabled": contract["amp"]["enabled"],
            "autocast_dtype": contract["amp"]["autocast_dtype"],
            "grad_scaler_enabled": contract["amp"]["grad_scaler_enabled"],
            "scaler_policy": contract["amp"]["scaler_policy"],
        },
        "environment_identity": _copy(environment),
        "source_identity": _copy(source),
        "epoch": epoch,
        "global_optimizer_step": optimizer_updates,
        "raw_model": raw_state,
        "ema": ema_payload,
        "evaluation_weights": "ema",
    }
    _assert_builtin(checkpoint, "checkpoint")
    checkpoint["checkpoint_sha256"] = _sha_value(checkpoint)
    return checkpoint


def validate_v2a_checkpoint(checkpoint: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version", "kind", "runtime_id", "contract_id", "baseline_id", "contract_identity", "authority_identity",
        "model_identity", "optimizer_parameter_groups", "amp_state", "environment_identity", "source_identity", "epoch",
        "global_optimizer_step", "raw_model", "ema", "evaluation_weights", "checkpoint_sha256",
    }
    value = _exact(checkpoint, required, "v2a checkpoint")
    _assert_builtin(value, "v2a checkpoint")
    if value["schema_version"] != RUNTIME_SCHEMA_VERSION or value["kind"] != CHECKPOINT_KIND or value["runtime_id"] != policy["runtime_id"]:
        _fail("v2a checkpoint runtime identity drift")
    binding = policy["contract_binding"]
    if value["contract_id"] != binding["contract_id"] or value["baseline_id"] != binding["baseline_id"]:
        _fail("v2a checkpoint contract identity drift")
    if value["contract_identity"]["canonical_sha256"] != policy["contract_sha256"]:
        _fail("v2a checkpoint contract digest drift")
    amp = value["amp_state"]
    if set(amp) != {"enabled", "autocast_dtype", "grad_scaler_enabled", "scaler_policy"} or amp["autocast_dtype"] != "bfloat16" or amp["grad_scaler_enabled"] is not False or amp["scaler_policy"] != "disabled_absent":
        _fail("v2a checkpoint AMP/no-scaler state drift")
    if value["evaluation_weights"] != "ema" or value["raw_model"] == value["ema"]:
        _fail("v2a checkpoint EMA/raw state drift")
    if _sha_value({key: value[key] for key in value if key != "checkpoint_sha256"}) != value["checkpoint_sha256"]:
        _fail("v2a checkpoint digest drift")
    return _copy(value)


class EpochProgressWriter:
    """Append one canonical, fsynced JSONL record for each epoch."""

    def __init__(self, path: str | os.PathLike[str], *, run_id: str, expected_epochs: int) -> None:
        self.path = _absolute_path(os.fspath(path), "progress_path")
        self.run_id = _string(run_id, "progress.run_id")
        self.expected_epochs = _integer(expected_epochs, "progress.expected_epochs", minimum=1)
        if self.path.exists() or self.path.is_symlink():
            _fail("epoch progress target already exists")
        if not self.path.parent.is_dir() or self.path.parent.is_symlink():
            _fail("epoch progress parent is unsafe")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        try:
            fd = os.open(self.path, flags, 0o600)
            self._handle = os.fdopen(fd, "ab", buffering=0)
        except OSError as exc:
            raise V2ARuntimeError("epoch progress file creation failed") from exc
        self._epochs: set[int] = set()
        self._closed = False

    def append(self, epoch: int, report: Mapping[str, Any]) -> dict[str, Any]:
        if self._closed:
            _fail("epoch progress writer is closed")
        epoch = _integer(epoch, "progress.epoch", minimum=1)
        if epoch > self.expected_epochs or epoch in self._epochs or epoch != len(self._epochs) + 1:
            _fail("epoch progress sequence drift")
        if type(report) is not dict:
            _fail("epoch progress report must be a builtin dict")
        row = {"schema_version": RUNTIME_SCHEMA_VERSION, "run_id": self.run_id, "epoch": epoch, **_copy(report)}
        _assert_builtin(row, "epoch progress row")
        raw = _canonical(row) + b"\n"
        try:
            self._handle.write(raw)
            self._handle.flush()
            os.fsync(self._handle.fileno())
        except OSError as exc:
            raise V2ARuntimeError("epoch progress fsync failed") from exc
        self._epochs.add(epoch)
        return row

    def finalize(self) -> dict[str, Any]:
        if self._closed:
            _fail("epoch progress writer is closed")
        if self._epochs != set(range(1, self.expected_epochs + 1)):
            _fail("epoch progress does not contain exactly one row per epoch")
        try:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
            self._closed = True
            observed = _regular_identity(self.path, "epoch progress", mode=0o600)
            raw = self.path.read_bytes()
        except OSError as exc:
            raise V2ARuntimeError("epoch progress finalization failed") from exc
        lines = raw.splitlines(keepends=True)
        rows = []
        for index, line in enumerate(lines):
            if not line.endswith(b"\n") or line[:-1] != line[:-1].rstrip(b"\r"):
                _fail(f"epoch progress line {index} is not canonical JSONL")
            try:
                row = json.loads(line[:-1].decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise V2ARuntimeError("epoch progress contains invalid JSON") from exc
            if type(row) is not dict or _canonical(row) != line[:-1]:
                _fail("epoch progress contains noncanonical JSON")
            rows.append(row)
        return {
            "relative_name": self.path.name,
            "path": str(self.path),
            "size_bytes": len(raw),
            "sha256": _sha_bytes(raw),
            "mode": observed["mode"],
            "line_count": len(rows),
            "rows_sha256": _sha_value(rows),
            "first_epoch": rows[0]["epoch"],
            "last_epoch": rows[-1]["epoch"],
        }


def _write_exclusive_json(path: pathlib.Path, value: Mapping[str, Any]) -> dict[str, Any]:
    raw = _canonical(value)
    if path.exists() or path.is_symlink():
        _fail(f"evidence target already exists: {path.name}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o600)
        with os.fdopen(fd, "wb", buffering=0) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        observed = _regular_identity(path, f"evidence file {path.name}", mode=0o600)
    except OSError as exc:
        raise V2ARuntimeError(f"evidence write failed: {path.name}") from exc
    return {"path": str(path), "size_bytes": len(raw), "sha256": _sha_bytes(raw), "mode": observed["mode"]}


def parse_v2a_terminal_stdout(stdout: bytes) -> dict[str, Any]:
    """Preserve raw process bytes and select the last canonical terminal line."""

    if type(stdout) is not bytes:
        _fail("process stdout must be raw bytes")
    candidates = []
    for line_number, line in enumerate(stdout.splitlines(keepends=True), start=1):
        if not line.endswith(b"\n"):
            continue
        payload = line[:-1]
        try:
            decoded = payload.decode("utf-8")
            value = json.loads(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if type(value) is not dict or value.get("kind") != PROCESS_TERMINAL_KIND:
            continue
        if _canonical(value) != payload:
            continue
        candidates.append((line_number, value, payload))
    if not candidates:
        _fail("process stdout has no canonical terminal JSON line")
    line_number, value, payload = candidates[-1]
    return {
        "value": _copy(value),
        "line_number": line_number,
        "line_size_bytes": len(payload) + 1,
        "stdout_size_bytes": len(stdout),
        "stdout_sha256": _sha_bytes(stdout),
        "terminal_line_sha256": _sha_bytes(payload),
        "stdout_bytes": stdout,
    }


def _get_port(ports: Any, name: str, *, required: bool = True) -> Any:
    if isinstance(ports, Mapping):
        value = ports.get(name)
    else:
        value = getattr(ports, name, None)
    if value is None and required:
        _fail(f"fake runtime port is required: {name}")
    return value


def _call_evaluator(evaluator: Any, ema: Any, epoch: int) -> Any:
    function = getattr(evaluator, "evaluate", None)
    if function is None and callable(evaluator):
        function = evaluator
    if not callable(function):
        _fail("primary evaluator is not callable")
    try:
        parameters = list(inspect.signature(function).parameters.values())
    except (TypeError, ValueError) as exc:
        raise V2ARuntimeError("primary evaluator signature is unavailable") from exc
    positional = [item for item in parameters if item.kind in {item.POSITIONAL_ONLY, item.POSITIONAL_OR_KEYWORD}]
    if any(item.kind == item.VAR_KEYWORD for item in parameters):
        return function(ema_model=ema, epoch=epoch)
    if len(positional) >= 2:
        first = positional[0].name.casefold()
        if first in {"epoch", "epoch_index", "step"}:
            return function(epoch, ema)
        return function(ema, epoch)
    if len(positional) == 1:
        if positional[0].name.casefold() in {"epoch", "epoch_index", "step"}:
            return function(epoch)
        return function(ema)
    return function()


def _normalize_metrics(metrics: Any, binding: Mapping[str, Any]) -> dict[str, Any]:
    if type(metrics) is not dict:
        _fail("primary evaluator metrics must be a builtin dict")
    result = {field: _finite_float(metrics[field], f"development.{field}") for field in ("AP", "AP50", "AR500") if field in metrics}
    if set(result) != {"AP", "AP50", "AR500"}:
        _fail("primary evaluator did not return AP/AP50/AR500")
    result["weights"] = "ema"
    return result


def _model_state(model: Any, port: Any, epoch: int, label: str) -> Any:
    if port is not None:
        return _state_payload(port, label)
    state_dict = getattr(model, "state_dict", None)
    if callable(state_dict):
        value = state_dict()
        if isinstance(value, Mapping):
            converted = {}
            for key, item in value.items():
                if callable(getattr(item, "tolist", None)):
                    converted[str(key)] = item.tolist()
                elif type(item) in {str, int, float, bool, type(None), list, dict}:
                    converted[str(key)] = item
                else:
                    converted[str(key)] = str(item)
            return _state_payload(converted, label)
    return {"epoch": epoch, "label": label}


def _safe_mkdir(path: pathlib.Path) -> None:
    if path.exists() or path.is_symlink():
        _fail("v2a fake target was created or occupied before run")
    try:
        path.mkdir(mode=0o700, parents=False, exist_ok=False)
    except OSError as exc:
        raise V2ARuntimeError("v2a fake target creation failed") from exc


def _fake_batches(value: Any) -> list[Any]:
    if callable(value):
        value = value()
    try:
        rows = list(value)
    except TypeError as exc:
        raise V2ARuntimeError("fake batches are not iterable") from exc
    if not rows:
        _fail("fake runtime requires at least one valid batch")
    return rows


def _loss_is_finite(loss: Any) -> bool:
    if type(loss) in {int, float} and type(loss) is not bool:
        return math.isfinite(float(loss))
    item = getattr(loss, "item", None)
    if callable(item):
        value = item()
        return type(value) in {int, float} and type(value) is not bool and math.isfinite(float(value))
    return True


def _call_backward(loss: Any, backward: Any) -> None:
    function = backward if callable(backward) else getattr(loss, "backward", None)
    if not callable(function):
        return
    if backward is not None:
        function(loss)
    else:
        function()


def _checkpoint_source_identity() -> dict[str, Any]:
    path = pathlib.Path(__file__)
    return {"runtime_module": "src/sparse_rtdetr/baseline/training_v2a_runtime.py", "runtime_module_sha256": _sha_bytes(path.read_bytes())}


def _read_canonical_json_file(path: pathlib.Path, field: str) -> tuple[dict[str, Any], bytes]:
    _regular_identity(path, field, mode=0o600)
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V2ARuntimeError(f"{field} is not canonical JSON") from exc
    if type(value) is not dict or _canonical(value) != raw:
        _fail(f"{field} is not canonical JSON")
    return value, raw


def validate_v2a_runtime_closure(result: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version", "kind", "status", "runtime_id", "run_id", "contract_sha256", "parent_identity", "data_roots",
        "environment_identity", "progress_inventory", "checkpoint_inventory", "process_identity", "terminal_stdout",
        "optimizer_updates", "epochs", "selection", "closure", "production", "terminal_file",
    }
    value = _exact(result, required, "v2a runtime result")
    if value["schema_version"] != RUNTIME_SCHEMA_VERSION or value["kind"] != PROCESS_TERMINAL_KIND or value["status"] != "FAKE_TERMINAL_COMPLETE":
        _fail("v2a fake terminal status drift")
    if value["runtime_id"] != policy["runtime_id"] or value["run_id"] != policy["run_identity"]["run_id"] or value["contract_sha256"] != policy["contract_sha256"]:
        _fail("v2a fake terminal run identity drift")
    if value["data_roots"] != policy["data_roots"] or value["environment_identity"] != policy["environment_identity"]:
        _fail("v2a fake terminal input binding drift")
    parent = value["parent_identity"]
    if (
        type(parent) is not dict
        or parent.get("before_after_identity_pass") is not True
        or parent.get("target_path") != policy["parent_identity"]["target_path"]
        or parent.get("parent_identity") != policy["parent_identity"]["parent_identity"]
    ):
        _fail("v2a fake terminal parent identity drift")
    target = pathlib.Path(policy["parent_identity"]["target_path"])
    progress = value["progress_inventory"]
    if type(progress) is not dict or set(progress) != {"relative_name", "path", "size_bytes", "sha256", "mode", "line_count", "rows_sha256", "first_epoch", "last_epoch"}:
        _fail("v2a fake terminal progress inventory shape drift")
    progress_path = pathlib.Path(progress["path"])
    if progress_path != target / policy["run_identity"]["progress_file_name"] or progress_path.is_symlink() or not progress_path.is_file():
        _fail("v2a fake terminal progress path drift")
    try:
        progress_raw = progress_path.read_bytes()
    except OSError as exc:
        raise V2ARuntimeError("v2a fake terminal progress is unavailable") from exc
    if progress["size_bytes"] != len(progress_raw) or progress["sha256"] != _sha_bytes(progress_raw) or progress["mode"] != 0o600:
        _fail("v2a fake terminal progress identity drift")
    progress_rows = []
    for line in progress_raw.splitlines(keepends=True):
        try:
            row = json.loads(line[:-1].decode("utf-8")) if line.endswith(b"\n") else None
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise V2ARuntimeError("v2a fake terminal progress JSON drift") from exc
        if type(row) is not dict or _canonical(row) != line[:-1]:
            _fail("v2a fake terminal progress bytes drift")
        progress_rows.append(row)
    if progress["line_count"] != value["epochs"] or progress["rows_sha256"] != _sha_value(progress_rows) or progress["first_epoch"] != 1 or progress["last_epoch"] != value["epochs"]:
        _fail("v2a fake terminal progress completeness drift")
    checkpoint_rows = value["checkpoint_inventory"]
    if type(checkpoint_rows) is not list or not checkpoint_rows:
        _fail("v2a fake terminal checkpoint inventory is empty")
    for row in checkpoint_rows:
        if type(row) is not dict or set(row) != {"role", "epoch", "path", "size_bytes", "sha256", "mode"}:
            _fail("v2a fake terminal checkpoint inventory shape drift")
        checkpoint_path = pathlib.Path(row["path"])
        if checkpoint_path.parent != target or checkpoint_path.is_symlink() or not checkpoint_path.is_file():
            _fail("v2a fake terminal checkpoint path drift")
        checkpoint, checkpoint_raw = _read_canonical_json_file(checkpoint_path, "v2a fake terminal checkpoint")
        if row["size_bytes"] != len(checkpoint_raw) or row["sha256"] != _sha_bytes(checkpoint_raw) or row["mode"] != 0o600:
            _fail("v2a fake terminal checkpoint identity drift")
        validate_v2a_checkpoint(checkpoint, policy)
    terminal_file = value["terminal_file"]
    if type(terminal_file) is not dict or set(terminal_file) != {"path", "size_bytes", "sha256", "mode"}:
        _fail("v2a fake terminal receipt shape drift")
    terminal_path = pathlib.Path(terminal_file["path"])
    if terminal_path != target / policy["run_identity"]["terminal_file_name"] or terminal_path.is_symlink() or not terminal_path.is_file():
        _fail("v2a fake terminal receipt path drift")
    terminal_value, terminal_raw = _read_canonical_json_file(terminal_path, "v2a fake terminal receipt")
    if terminal_file["size_bytes"] != len(terminal_raw) or terminal_file["sha256"] != _sha_bytes(terminal_raw) or terminal_file["mode"] != 0o600:
        _fail("v2a fake terminal receipt identity drift")
    terminal_base = {key: value[key] for key in required if key not in {"terminal_stdout", "terminal_file"}}
    if terminal_value != terminal_base:
        _fail("v2a fake terminal receipt content drift")
    stdout = value["terminal_stdout"]
    if type(stdout) is not dict or set(stdout) != {"value", "line_number", "line_size_bytes", "stdout_size_bytes", "stdout_sha256", "terminal_line_sha256", "stdout_bytes"}:
        _fail("v2a fake terminal stdout identity shape drift")
    parsed = parse_v2a_terminal_stdout(stdout["stdout_bytes"])
    if parsed != stdout or parsed["value"] != terminal_base:
        _fail("v2a fake terminal stdout content drift")
    closure = value["closure"]
    if closure != {
        "process_terminal_json": True,
        "training_parent_identity": True,
        "authorization_data_roots": True,
        "environment_identity": True,
        "epoch_progress_evidence": True,
    }:
        _fail("v2a runtime closure is incomplete")
    if value["production"] != {
        "owner_authorization_consumed": False,
        "gpu_probe_executed": False,
        "formal_training_executed": False,
        "training_ready": False,
        "independent_audit_pass": False,
    }:
        _fail("v2a fake runtime production state drift")
    return _copy(value)


def run_v2a_fake_runtime(
    policy: Mapping[str, Any],
    ports: Any,
    *,
    epochs: int = 1,
) -> dict[str, Any]:
    """Run a CPU-only fake epoch sequence through all five runtime closures."""

    _exact(policy, {
        "schema_version", "runtime_id", "contract_id", "baseline_id", "contract_binding", "contract_sha256",
        "run_identity", "parent_identity", "data_roots", "environment_identity", "model", "topology", "optimizer",
        "amp", "ema", "selection", "checkpoint", "data_roles", "runtime_closure", "production",
    }, "v2a runtime policy")
    epochs = _integer(epochs, "fake epochs", minimum=1)
    if policy["data_roots"] is None or policy["environment_identity"] is None:
        _fail("fake runtime requires authorized data roots and environment identity")
    model = _get_port(ports, "model")
    optimizer = _get_port(ports, "optimizer")
    ema = _get_port(ports, "ema")
    evaluator = _get_port(ports, "evaluator")
    evaluator_id = getattr(evaluator, "evaluator_id", None)
    if isinstance(evaluator, Mapping):
        evaluator_id = evaluator.get("evaluator_id", evaluator_id)
    if evaluator_id is None:
        evaluator_id = _get_port(ports, "evaluator_id", required=False)
    if evaluator_id != policy["selection"]["primary_evaluator"]:
        _fail("fake evaluator is not the certified primary evaluator")
    batches_port = _get_port(ports, "batches")
    loss_port = _get_port(ports, "compute_loss", required=False)
    if loss_port is None:
        loss_port = _get_port(ports, "loss", required=False)
    if loss_port is None or not callable(loss_port):
        _fail("fake compute_loss port is required")
    backward = _get_port(ports, "backward", required=False)
    scheduler = _get_port(ports, "scheduler", required=False)
    target = pathlib.Path(policy["parent_identity"]["target_path"])
    _safe_mkdir(target)
    progress = EpochProgressWriter(target / policy["run_identity"]["progress_file_name"], run_id=policy["run_identity"]["run_id"], expected_epochs=epochs)
    candidates = []
    checkpoint_inventory = []
    optimizer_updates = 0
    selected = None
    try:
        for epoch in range(1, epochs + 1):
            batch_count = 0
            epoch_loss = 0.0
            for batch in _fake_batches(batches_port):
                if isinstance(batch, Mapping) and "batch_size" in batch and batch["batch_size"] != policy["topology"]["train_micro_batch"]:
                    _fail("fake runtime batch adaptation detected")
                zero_grad = getattr(optimizer, "zero_grad", None)
                if callable(zero_grad):
                    zero_grad()
                loss = loss_port(batch)
                if not _loss_is_finite(loss):
                    _fail("fake runtime loss is non-finite")
                _call_backward(loss, backward)
                step = getattr(optimizer, "step", None)
                if not callable(step):
                    _fail("fake optimizer has no direct step")
                step_result = step()
                if step_result is False or (isinstance(step_result, Mapping) and step_result.get("skipped") is True):
                    _fail("fake optimizer step was skipped")
                update = getattr(ema, "update", None)
                if callable(update):
                    update(model)
                elif callable(ema):
                    ema(model)
                else:
                    _fail("fake EMA update port is required")
                optimizer_updates += 1
                batch_count += 1
                if type(loss) in {int, float} and type(loss) is not bool:
                    epoch_loss += float(loss)
            if callable(scheduler):
                scheduler()
            metrics = _normalize_metrics(_call_evaluator(evaluator, ema, epoch), policy["contract_binding"])
            candidate = {"epoch": epoch, **metrics}
            candidates.append(candidate)
            selected = select_v2a_development_candidate(candidates, policy["contract_binding"])
            checkpoint = build_v2a_checkpoint(
                policy,
                epoch=epoch,
                optimizer_updates=optimizer_updates,
                raw_model_state=_model_state(model, _get_port(ports, "raw_model_state", required=False), epoch, "raw"),
                ema_state=_model_state(ema, _get_port(ports, "ema_state", required=False), epoch, "ema"),
                optimizer_audit=_get_port(ports, "optimizer_audit", required=False),
                source_identity=_checkpoint_source_identity(),
            )
            checkpoint_bytes = _canonical(checkpoint)
            checkpoint_path = target / f"{CHECKPOINT_FILE_PREFIX}last_epoch_{epoch}.json"
            checkpoint_ref = _write_exclusive_json(checkpoint_path, checkpoint)
            checkpoint_inventory.append({"role": "last", "epoch": epoch, **checkpoint_ref})
            if selected["epoch"] == epoch:
                best_ref = _write_exclusive_json(target / f"{CHECKPOINT_FILE_PREFIX}best_epoch_{epoch}.json", checkpoint)
                checkpoint_inventory.append({"role": "best", "epoch": epoch, **best_ref})
            progress.append(epoch, {
                "batch_count": batch_count,
                "mean_loss": epoch_loss / batch_count,
                "optimizer_updates": optimizer_updates,
                "evaluation": metrics,
                "selected_epoch": selected["epoch"],
                "checkpoint_sha256": _sha_bytes(checkpoint_bytes),
            })
        final_ref = _write_exclusive_json(target / f"{CHECKPOINT_FILE_PREFIX}final_epoch_{epochs}.json", checkpoint)
        checkpoint_inventory.append({"role": "final", "epoch": epochs, **final_ref})
        progress_inventory = progress.finalize()
    except BaseException:
        if not progress._closed:
            try:
                progress._handle.flush()
                os.fsync(progress._handle.fileno())
                progress._handle.close()
                progress._closed = True
            except OSError:
                pass
        raise
    parent_after = _validate_parent_after(policy["parent_identity"])
    terminal = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "kind": PROCESS_TERMINAL_KIND,
        "status": "FAKE_TERMINAL_COMPLETE",
        "runtime_id": policy["runtime_id"],
        "run_id": policy["run_identity"]["run_id"],
        "contract_sha256": policy["contract_sha256"],
        "parent_identity": parent_after,
        "data_roots": _copy(policy["data_roots"]),
        "environment_identity": _copy(policy["environment_identity"]),
        "progress_inventory": progress_inventory,
        "checkpoint_inventory": checkpoint_inventory,
        "process_identity": {"mode": "cpu_fake", "pid": os.getpid(), "parent_pid": os.getppid()},
        "optimizer_updates": optimizer_updates,
        "epochs": epochs,
        "selection": _copy(selected),
        "closure": {
            "process_terminal_json": True,
            "training_parent_identity": True,
            "authorization_data_roots": True,
            "environment_identity": True,
            "epoch_progress_evidence": True,
        },
        "production": {
            "owner_authorization_consumed": False,
            "gpu_probe_executed": False,
            "formal_training_executed": False,
            "training_ready": False,
            "independent_audit_pass": False,
        },
    }
    terminal_ref = _write_exclusive_json(target / policy["run_identity"]["terminal_file_name"], terminal)
    stdout = b"fake-vendor-log\n" + _canonical(terminal) + b"\n"
    parsed = parse_v2a_terminal_stdout(stdout)
    if parsed["value"] != terminal:
        _fail("terminal stdout result differs from published result")
    result = {
        **terminal,
        "terminal_stdout": parsed,
        "terminal_file": terminal_ref,
    }
    return validate_v2a_runtime_closure(result, policy)
