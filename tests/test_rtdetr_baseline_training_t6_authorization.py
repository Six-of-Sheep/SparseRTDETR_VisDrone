"""CPU and synthetic mutation coverage for the T6A detached contract."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path

import pytest

from sparse_rtdetr.baseline import training_t6_authorization as contract


ROOT = Path(__file__).resolve().parents[1]
GOOD_NONCE = "0123456789abcdefdeadbeefcafebabe"
OTHER_NONCE = "fedcba9876543210beadfeedfacecafe"


def _binding() -> dict[str, object]:
    return contract.training_launch_contract_binding(ROOT)


def _observed(binding: dict[str, object], nonce: str = GOOD_NONCE) -> dict[str, object]:
    return {
        "repository": {
            "repo_root": str(ROOT.resolve()),
            "branch": "codex/p3-rtdetrv2-baseline-training-t6-authorization-contract-r1",
            "head": "1" * 40,
            "tree": "2" * 40,
            "parent": "3" * 40,
            "upstream": "origin/codex/p3-rtdetrv2-baseline-training-t6-authorization-contract-r1",
            "upstream_sha": "4" * 40,
        },
        "targets": copy.deepcopy(binding["targets"]),
        "target_absence": {
            "training_evidence_root": True,
            "process_evidence_root": True,
            "outer_evidence_root": True,
            "tmux_session": True,
            "receipt": True,
            "lock": True,
        },
        "contract_identity": copy.deepcopy(binding["contract_identity"]),
        "source_bindings": copy.deepcopy(binding["source_bindings"]),
        "environment_identity": {
            "python": {
                "path": "/opt/t6a/python",
                "realpath": "/opt/t6a/python3.10",
                "version": "3.10.16",
                "sha256": "a" * 64,
            },
            "tmux": {
                "path": "/opt/t6a/tmux",
                "realpath": "/opt/t6a/tmux.real",
                "version": "3.4",
                "sha256": "b" * 64,
            },
            "gpu": {
                "index": 0,
                "name": "NVIDIA GeForce RTX 4090 D",
                "uuid": "GPU-1faee6f0-1da7-4ede-2475-67a5a00274a8",
                "driver_version": "580.00",
                "cuda_version": "12.4",
                "power_limit_watts": 425.0,
            },
            "filesystem": {"device": 12345, "mount": "/" + "mnt" + "/t6a-training"},
        },
        "data_roles": {
            "train_core": {
                "role": "train_core",
                "root": "/" + "mnt" + "/t6a-training/train-core",
                "manifest_sha256": "c" * 64,
            },
            "development": {
                "role": "development",
                "root": "/" + "mnt" + "/t6a-training/development",
                "manifest_sha256": "d" * 64,
            },
            "confirmatory": {
                "role": "confirmatory",
                "access": "sealed_and_forbidden",
                "identity_read": "forbidden",
            },
            "test": {
                "role": "test",
                "access": "forbidden",
                "identity_read": "forbidden",
            },
        },
        "training_policy": copy.deepcopy(binding["contract"]["training_policy"]),
        "state_machine": copy.deepcopy(binding["contract"]["state_machine"]),
        "training_run_id": "rtdetrv2_r18_visdrone_baseline_training_v1",
        "tmux_session_name": "p3_rtdetrv2_r18_visdrone_baseline_training_v1",
        "nonce": nonce,
    }


def _authorization(binding: dict[str, object], observed: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "authorization_id": "rtdetrv2_r18_visdrone_training_owner_authorization_v1",
        "authorized": True,
        "contract_id": contract.LAUNCH_CONTRACT_ID,
        "training_run_id": observed["training_run_id"],
        "nonce": observed["nonce"],
        "tmux_session_name": observed["tmux_session_name"],
        "targets": copy.deepcopy(observed["targets"]),
        "target_absence": copy.deepcopy(observed["target_absence"]),
        "repository": copy.deepcopy(observed["repository"]),
        "contract_identity": copy.deepcopy(observed["contract_identity"]),
        "source_bindings": copy.deepcopy(observed["source_bindings"]),
        "environment_identity": copy.deepcopy(observed["environment_identity"]),
        "data_roles": copy.deepcopy(observed["data_roles"]),
        "training_policy": copy.deepcopy(observed["training_policy"]),
        "state_machine": copy.deepcopy(observed["state_machine"]),
        "owner": {
            "owner_id": "project_owner",
            "uid": 1000,
            "gid": 1000,
            "detached_external": True,
            "self_issued": False,
        },
    }


def _valid_case(nonce: str = GOOD_NONCE) -> tuple[dict[str, object], dict[str, object], dict[str, object], bytes, dict[str, object]]:
    binding = _binding()
    observed = _observed(binding, nonce)
    authorization = _authorization(binding, observed)
    canonical = contract.canonical_owner_authorization_bytes(authorization)
    raw = canonical + b"\n"
    file_identity = {
        "regular": True,
        "symlink": False,
        "mode": 384,
        "uid": 1000,
        "gid": 1000,
        "nlink": 1,
    }
    return binding, observed, authorization, raw, file_identity


def _validate(authorization: object, binding: dict[str, object], observed: dict[str, object], raw: bytes, file_identity: dict[str, object]) -> dict[str, object]:
    return contract.validate_owner_authorization(
        authorization,
        expected_raw_size_bytes=len(raw),
        expected_raw_sha256=hashlib.sha256(raw).hexdigest(),
        contract_binding=binding,
        observed_binding=observed,
        authorization_file_identity=file_identity,
    )


def _reject(callable_object) -> None:
    with pytest.raises(contract.TrainingLaunchContractError):
        callable_object()


def test_three_canonical_controls_are_deterministic_and_lf_free() -> None:
    loaded = contract.load_training_launch_contract(ROOT)
    first = contract.canonical_training_launch_contract_bytes(loaded)
    second = contract.canonical_training_launch_contract_bytes(copy.deepcopy(loaded))
    assert first == second
    assert not first.endswith(b"\n")
    assert len(first) == contract.LAUNCH_CONTRACT_CANONICAL_SIZE_BYTES
    assert hashlib.sha256(first).hexdigest() == contract.LAUNCH_CONTRACT_CANONICAL_SHA256

    binding, observed, authorization, raw, file_identity = _valid_case()
    auth_canonical = contract.canonical_owner_authorization_bytes(authorization)
    assert auth_canonical == contract.canonical_owner_authorization_bytes(copy.deepcopy(authorization))
    assert _validate(authorization, binding, observed, raw, file_identity)["authorized"] is True

    binding2, observed2, authorization2, raw2, file_identity2 = _valid_case(OTHER_NONCE)
    assert binding["contract_identity"] == binding2["contract_identity"]
    assert authorization["nonce"] != authorization2["nonce"]
    assert raw != raw2
    assert _validate(raw2, binding2, observed2, raw2, file_identity2)["nonce"] == OTHER_NONCE


@pytest.mark.parametrize("mutation", ["missing", "extra", "rename"])
def test_static_contract_is_closed_at_top_level(mutation: str) -> None:
    value = contract.load_training_launch_contract(ROOT)
    if mutation == "missing":
        value.pop("status")
    elif mutation == "extra":
        value["unexpected"] = False
    else:
        value["stage_name"] = value.pop("stage")
    _reject(lambda: contract.validate_training_launch_contract(value))


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("schema_version",), True),
        (("contract_id",), 1),
        (("repository", "head"), True),
        (("targets", "all_targets_must_be_absent"), 1),
        (("training_policy", "topology", "train_batch_size"), False),
        (("training_policy", "amp", "enabled"), 1),
        (("training_policy", "amp", "init_scale"), math.nan),
        (("training_policy", "amp", "growth_factor"), math.inf),
        (("readiness", "static_config_authorizes_production"), True),
    ],
)
def test_static_contract_rejects_scalar_and_numeric_mutations(path: tuple[str, ...], replacement: object) -> None:
    value = contract.load_training_launch_contract(ROOT)
    target = value
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    _reject(lambda: contract.validate_training_launch_contract(value))


@pytest.mark.parametrize("raw", [
    b'{"a":1,"a":2}\n',
    b"\xef\xbb\xbf{}\n",
    b"{\"a\":\x00}\n",
    b'{"a":1}\r\n',
    b'{"a":NaN}\n',
    b'{"a":Infinity}\n',
    b'{"a":-Infinity}\n',
    b"{}",
    b"{}\n\n",
])
def test_strict_json_byte_controls_are_rejected(raw: bytes) -> None:
    _reject(lambda: contract._parse_json(raw, "synthetic", trailing_lf=True))


def test_input_objects_are_not_mutated() -> None:
    binding, observed, authorization, raw, file_identity = _valid_case()
    before_auth = copy.deepcopy(authorization)
    before_observed = copy.deepcopy(observed)
    before_binding = copy.deepcopy(binding)
    before_file_identity = copy.deepcopy(file_identity)
    _validate(authorization, binding, observed, raw, file_identity)
    assert authorization == before_auth
    assert observed == before_observed
    assert binding == before_binding
    assert file_identity == before_file_identity


@pytest.mark.parametrize("field", [
    "schema_version",
    "authorization_id",
    "authorized",
    "contract_id",
    "training_run_id",
    "nonce",
    "tmux_session_name",
    "targets",
    "target_absence",
    "repository",
    "contract_identity",
    "source_bindings",
    "environment_identity",
    "data_roles",
    "training_policy",
    "state_machine",
    "owner",
])
def test_owner_authorization_missing_extra_and_rename_are_rejected(field: str) -> None:
    binding, observed, authorization, raw, file_identity = _valid_case()
    missing = copy.deepcopy(authorization)
    missing.pop(field)
    _reject(lambda: contract.canonical_owner_authorization_bytes(missing))

    extra = copy.deepcopy(authorization)
    extra["unexpected"] = False
    _reject(lambda: contract.canonical_owner_authorization_bytes(extra))

    renamed = copy.deepcopy(authorization)
    renamed[f"{field}_renamed"] = renamed.pop(field)
    _reject(lambda: contract.canonical_owner_authorization_bytes(renamed))
    _validate(authorization, binding, observed, raw, file_identity)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("schema_version", True),
        ("authorization_id", 1),
        ("authorized", 1),
        ("contract_id", False),
        ("training_run_id", 1),
        ("nonce", 1),
        ("tmux_session_name", False),
        ("targets", []),
        ("target_absence", []),
        ("repository", []),
        ("contract_identity", []),
        ("source_bindings", []),
        ("environment_identity", []),
        ("data_roles", []),
        ("training_policy", []),
        ("state_machine", []),
        ("owner", []),
    ],
)
def test_owner_authorization_wrong_types_and_bool_as_int_are_rejected(field: str, replacement: object) -> None:
    _, _, authorization, _, _ = _valid_case()
    mutated = copy.deepcopy(authorization)
    mutated[field] = replacement
    _reject(lambda: contract.canonical_owner_authorization_bytes(mutated))


@pytest.mark.parametrize("field", [
    "training_evidence_root",
    "process_evidence_root",
    "outer_evidence_root",
    "tmux_session",
    "receipt",
    "lock",
])
def test_target_absence_is_closed_and_must_be_true(field: str) -> None:
    binding, observed, authorization, raw, file_identity = _valid_case()
    mutated = copy.deepcopy(authorization)
    mutated["target_absence"][field] = False
    _reject(lambda: contract.canonical_owner_authorization_bytes(mutated))
    mutated = copy.deepcopy(observed)
    mutated["target_absence"][field] = 1
    _reject(lambda: contract.validate_owner_authorization(
        authorization,
        expected_raw_size_bytes=len(raw),
        expected_raw_sha256=hashlib.sha256(raw).hexdigest(),
        contract_binding=binding,
        observed_binding=mutated,
        authorization_file_identity=file_identity,
    ))


@pytest.mark.parametrize("nonce", [
    "0" * 32,
    "a" * 32,
    "0123456789abcdef" * 2,
    "ABCDEF0123456789abcdef0123456789",
    "1234",
    "0" * 31,
    "not-a-nonce-00000000000000000000",
])
def test_nonce_placeholder_entropy_and_format_controls(nonce: str) -> None:
    binding = _binding()
    observed = _observed(binding, nonce)
    authorization = _authorization(binding, observed)
    _reject(lambda: contract.canonical_owner_authorization_bytes(authorization))


@pytest.mark.parametrize("field", ["mode", "uid", "gid", "nlink"])
def test_authorization_file_object_identity_is_strict(field: str) -> None:
    binding, observed, authorization, raw, file_identity = _valid_case()
    mutated = copy.deepcopy(file_identity)
    mutated[field] = 0
    _reject(lambda: _validate(authorization, binding, observed, raw, mutated))
    mutated = copy.deepcopy(file_identity)
    mutated["regular"] = False
    _reject(lambda: _validate(authorization, binding, observed, raw, mutated))
    mutated = copy.deepcopy(file_identity)
    mutated["symlink"] = True
    _reject(lambda: _validate(authorization, binding, observed, raw, mutated))


def test_special_files_and_hardlink_objects_are_not_accepted_as_authorization_files(tmp_path: Path) -> None:
    regular = tmp_path / "regular"
    regular.write_bytes(b"x")
    assert contract._regular_file(tmp_path, "regular", "fixture") == regular
    directory = tmp_path / "directory"
    directory.mkdir()
    hardlink = tmp_path / "hardlink"
    hardlink.hardlink_to(regular)
    symlink = tmp_path / "symlink"
    symlink.symlink_to(regular)
    fifo = tmp_path / "fifo"
    if hasattr(os, "mkfifo"):
        os.mkfifo(fifo, 0o600)

    for relative in ("directory", "hardlink", "symlink"):
        _reject(lambda relative=relative: contract._regular_file(tmp_path, relative, "fixture"))
    if fifo.exists():
        _reject(lambda: contract._regular_file(tmp_path, "fifo", "fixture"))


@pytest.mark.parametrize("field", ["repository", "contract_identity", "source_bindings", "environment_identity", "data_roles"])
def test_cross_file_identity_mutations_fail_closed(field: str) -> None:
    binding, observed, authorization, raw, file_identity = _valid_case()
    mutated = copy.deepcopy(authorization)
    if field == "repository":
        mutated[field]["tree"] = "9" * 40
    elif field == "contract_identity":
        mutated[field]["canonical_sha256"] = "9" * 64
    elif field == "source_bindings":
        mutated[field]["process"]["entry_module"]["sha256"] = "9" * 64
    elif field == "environment_identity":
        mutated[field]["gpu"]["power_limit_watts"] = 424.0
    else:
        mutated[field]["development"]["manifest_sha256"] = "9" * 64
    _reject(lambda: _validate(mutated, binding, observed, raw, file_identity))
    _validate(authorization, binding, observed, raw, file_identity)


@pytest.mark.parametrize("field", ["repository", "environment_identity", "data_roles"])
def test_launch_time_observation_mutations_fail_closed(field: str) -> None:
    binding, observed, authorization, raw, file_identity = _valid_case()
    mutated = copy.deepcopy(observed)
    if field == "repository":
        mutated[field]["head"] = "9" * 40
    elif field == "environment_identity":
        mutated[field]["filesystem"]["device"] = True
    else:
        mutated[field]["train_core"]["root"] = "/" + "mnt" + "/t6a-training/test"
    _reject(lambda: _validate(authorization, binding, mutated, raw, file_identity))


def test_pre_t6_repository_reference_is_not_a_launch_observation() -> None:
    binding, observed, authorization, raw, file_identity = _valid_case()
    for field in ("branch", "head", "tree"):
        mutated = copy.deepcopy(observed)
        if field == "branch":
            mutated["repository"][field] = contract._FROZEN_REPOSITORY_REFERENCE[field]
        else:
            mutated["repository"][field] = contract._FROZEN_REPOSITORY_REFERENCE[field]
        _reject(lambda mutated=mutated: _validate(authorization, binding, mutated, raw, file_identity))


def test_external_raw_identity_is_owned_by_caller() -> None:
    binding, observed, authorization, raw, file_identity = _valid_case()
    expected_sha = hashlib.sha256(raw).hexdigest()
    _reject(lambda: contract.validate_owner_authorization(
        authorization,
        expected_raw_size_bytes=len(raw) + 1,
        expected_raw_sha256=expected_sha,
        contract_binding=binding,
        observed_binding=observed,
        authorization_file_identity=file_identity,
    ))
    _reject(lambda: contract.validate_owner_authorization(
        authorization,
        expected_raw_size_bytes=len(raw),
        expected_raw_sha256="e" * 64,
        contract_binding=binding,
        observed_binding=observed,
        authorization_file_identity=file_identity,
    ))
    _reject(lambda: contract.validate_owner_authorization(
        authorization,
        expected_raw_size_bytes=True,
        expected_raw_sha256=expected_sha,
        contract_binding=binding,
        observed_binding=observed,
        authorization_file_identity=file_identity,
    ))


@pytest.mark.parametrize("raw", [
    lambda valid: valid[:-1],
    lambda valid: valid + b"\n",
    lambda valid: valid[:-1] + b" \n",
    lambda valid: b"\xef\xbb\xbf" + valid,
    lambda valid: valid[:-1] + b"\r\n",
])
def test_authorization_raw_encoding_and_trailing_lf_are_strict(raw) -> None:
    binding, observed, authorization, valid, file_identity = _valid_case()
    mutated = raw(valid)
    _reject(lambda: _validate(mutated, binding, observed, valid, file_identity))


def test_owner_authorization_binding_returns_only_verified_identity() -> None:
    binding, observed, authorization, raw, file_identity = _valid_case()
    result = contract.owner_authorization_binding(
        raw,
        expected_raw_size_bytes=len(raw),
        expected_raw_sha256=hashlib.sha256(raw).hexdigest(),
        contract_binding=binding,
        observed_binding=observed,
        authorization_file_identity=file_identity,
    )
    assert result["authorized"] is True
    assert result["raw_size_bytes"] == len(raw)
    assert result["raw_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["canonical_size_bytes"] == len(raw) - 1
    assert result["canonical_sha256"] == hashlib.sha256(raw[:-1]).hexdigest()
    assert result["authorization"]["nonce"] == GOOD_NONCE


def test_state_machine_and_execution_policy_are_immutable() -> None:
    binding, observed, authorization, raw, file_identity = _valid_case()
    for key in ("resume", "retry", "overwrite", "fallback"):
        mutated = copy.deepcopy(authorization)
        mutated["training_policy"]["execution"][key] = True
        _reject(lambda mutated=mutated: contract.canonical_owner_authorization_bytes(mutated))
    mutated = copy.deepcopy(authorization)
    mutated["state_machine"]["states"] = ["DESIGN_ONLY", "RUNNING"]
    _reject(lambda: contract.canonical_owner_authorization_bytes(mutated))
    mutated = copy.deepcopy(observed)
    mutated["state_machine"]["no_retry_after"] = "RUNNING"
    _reject(lambda: _validate(authorization, binding, mutated, raw, file_identity))


def test_contract_binding_reads_only_declared_source_identities() -> None:
    result = _binding()
    assert result["contract_id"] == contract.LAUNCH_CONTRACT_ID
    assert result["contract_identity"]["raw_sha256"] == contract.LAUNCH_CONTRACT_RAW_SHA256
    assert set(result["source_bindings"]) == {
        "training_contract",
        "runtime_plan",
        "training_evidence",
        "prepared_adapter",
        "process",
        "primary_evaluator",
        "vendor_runtime",
        "conversion_r3",
    }
    assert result["source_bindings"]["conversion_r3"]["file_count"] == 25


def test_module_import_is_stdlib_only_and_has_no_production_operation_surface() -> None:
    source_path = ROOT / "src/sparse_rtdetr/baseline/training_t6_authorization.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
            imports.add(node.module or "")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            calls.add(node.func.attr.casefold())
    assert imports == {"copy", "hashlib", "json", "math", "os", "re", "stat", "pathlib", "typing"}
    assert not {"torch", "subprocess", "tmux", "dataloader", "dataset", "evaluator"} & imports
    assert not {"write", "write_bytes", "write_text", "mkdir", "unlink", "remove", "run", "popen", "system", "sleep"} & calls
    assert contract.__all__ == (
        "TrainingLaunchContractError",
        "canonical_training_launch_contract_bytes",
        "load_training_launch_contract",
        "validate_training_launch_contract",
        "training_launch_contract_binding",
        "canonical_owner_authorization_bytes",
        "validate_owner_authorization",
        "owner_authorization_binding",
    )


def test_static_config_has_no_authorization_or_nonce_value() -> None:
    raw = (ROOT / contract.LAUNCH_CONTRACT_CONFIG_RELATIVE_PATH).read_bytes()
    assert raw.count(b"\r") == 0
    assert raw.count(b"\x00") == 0
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    value = json.loads(raw.decode("utf-8"))
    assert "authorized" not in value
    assert "nonce" not in value
    assert value["status"] == "DETACHED_AUTHORIZATION_REQUIRED"
    assert value["readiness"]["static_config_authorizes_production"] is False
