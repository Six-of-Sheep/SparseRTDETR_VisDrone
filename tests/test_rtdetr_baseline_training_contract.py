from __future__ import annotations

import ast
import copy
from collections import OrderedDict, defaultdict
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import stat
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/sparse_rtdetr/baseline/training_contract.py"
SPEC = importlib.util.spec_from_file_location("p3_training_contract_tests", SOURCE)
assert SPEC is not None and SPEC.loader is not None
contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contract)


def _baseline() -> dict:
    return json.loads((ROOT / contract.BASELINE_CONFIG_RELATIVE_PATH).read_text(encoding="utf-8"))


def _training() -> dict:
    return json.loads((ROOT / contract.TRAINING_CONFIG_RELATIVE_PATH).read_text(encoding="utf-8"))


def _set_path(value, path, replacement):
    current = value
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = replacement


def _leaves(value, path=()):
    if type(value) is dict:
        for key, child in value.items():
            yield from _leaves(child, path + (key,))
    elif type(value) is list:
        for index, child in enumerate(value):
            yield from _leaves(child, path + (index,))
    else:
        yield path, value


def _different(value):
    if value is None:
        return "not-null"
    if type(value) is bool:
        return not value
    if type(value) is int:
        return value + 1
    if type(value) is float:
        return value + 0.125
    if type(value) is str:
        return value + "_drift"
    raise AssertionError(type(value))


def _reorder_builtin_dicts(value):
    if type(value) is dict:
        return {key: _reorder_builtin_dicts(value[key]) for key in reversed(list(value))}
    if type(value) is list:
        return [_reorder_builtin_dicts(item) for item in value]
    return copy.deepcopy(value)


def _closed_object_roles(value, schema=contract._TRAINING_CONTRACT_SCHEMA, pointer=""):
    kind, detail = schema
    if kind == "object":
        yield pointer or "/", value, schema
        for key, child_schema in detail.items():
            child_pointer = (pointer + "/" + key) if pointer else ("/" + key)
            yield from _closed_object_roles(value[key], child_schema, child_pointer)
    elif kind == "list":
        for index, child_schema in enumerate(detail):
            child_pointer = pointer + "/" + str(index)
            yield from _closed_object_roles(value[index], child_schema, child_pointer)


def _numeric_leaves(value, pointer="", excluded=()):
    if type(value) is dict:
        for key, child in value.items():
            child_pointer = (pointer + "/" + key) if pointer else ("/" + key)
            yield from _numeric_leaves(child, child_pointer, excluded)
    elif type(value) is list:
        for index, child in enumerate(value):
            yield from _numeric_leaves(child, pointer + "/" + str(index), excluded)
    elif type(value) in {int, float} and pointer not in excluded:
        yield pointer, value


def _portable_root(tmp_path: Path, training_bytes: bytes, *, include_conversion: bool = False) -> Path:
    for relative in (contract.TRAINING_CONFIG_RELATIVE_PATH, contract.BASELINE_CONFIG_RELATIVE_PATH):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        source = ROOT / relative
        target.write_bytes(training_bytes if relative == contract.TRAINING_CONFIG_RELATIVE_PATH else source.read_bytes())
    shutil.copytree(ROOT / "vendor/rtdetrv2_pytorch", tmp_path / "vendor/rtdetrv2_pytorch")
    manifest_target = tmp_path / "manifests/rtdetrv2_upstream.json"
    manifest_target.parent.mkdir(parents=True, exist_ok=True)
    manifest_target.write_bytes((ROOT / "manifests/rtdetrv2_upstream.json").read_bytes())
    for relative in (
        contract.PRIMARY_EVALUATOR_CONFIG_RELATIVE_PATH,
        contract.PRIMARY_EVALUATOR_MANIFEST_RELATIVE_PATH,
        *(relative for _, relative, _ in contract.PRIMARY_EVALUATOR_SOURCE_FILES),
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    if include_conversion:
        shutil.copytree(
            ROOT / contract.CONVERSION_R3_RELATIVE_PATH,
            tmp_path / contract.CONVERSION_R3_RELATIVE_PATH,
        )
    return tmp_path


def _portable_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "portable-repo"
    root.mkdir(parents=True)
    return _portable_root(root, (ROOT / contract.TRAINING_CONFIG_RELATIVE_PATH).read_bytes())


@pytest.fixture(scope="session")
def full_portable_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("portable-r3")
    return _portable_root(
        root,
        (ROOT / contract.TRAINING_CONFIG_RELATIVE_PATH).read_bytes(),
        include_conversion=True,
    )


def _replace_with_fifo(path: Path) -> None:
    path.unlink()
    os.mkfifo(path)


def _replace_with_hardlink(path: Path, target: Path) -> None:
    target.write_bytes(path.read_bytes())
    path.unlink()
    path.hardlink_to(target)


def _replace_with_symlink(path: Path, target: Path) -> None:
    path.unlink()
    path.symlink_to(target)


def _vendor_source_paths(training: dict) -> list[str]:
    sources = [training["source_bindings"]["vendor_recipe"]]
    sources.extend(training["source_bindings"]["vendor_includes"])
    return [item["relative_path"] for item in sources]


def test_positive_load_validate_canonical_and_binding():
    loaded = contract.load_training_contract(ROOT)
    assert loaded == contract.validate_training_contract(_training(), _baseline())
    canonical = contract.canonical_training_contract_bytes(loaded)
    assert hashlib.sha256(canonical).hexdigest() == contract.TRAINING_CONTRACT_CANONICAL_SHA256
    binding = contract.training_contract_binding(ROOT)
    assert binding == {
        "schema_version": 1,
        "training_contract_id": "rtdetrv2_r18_visdrone_baseline_training_v1",
        "baseline_id": "rtdetrv2_r18_visdrone_baseline_v1",
        "relative_path": contract.TRAINING_CONFIG_RELATIVE_PATH,
        "raw_size_bytes": 10907,
        "raw_sha256": "0f9ddb2e6d8ec6d2b21e42f6c419511f5bd70df287457c2c2b6109eaf8a93297",
        "canonical_size_bytes": 9117,
        "canonical_sha256": contract.TRAINING_CONTRACT_CANONICAL_SHA256,
        "vendor_runtime_binding": {
            "relative_path": "vendor/rtdetrv2_pytorch",
            "file_count": 124,
            "directory_count_excluding_root": 25,
            "total_size_bytes": 373735,
            "compact_inventory_sha256": "0fc6803665bc4b5720e983345b2cacb0147eceead9f882588880b6f8a0e68051",
            "manifest_relative_path": "manifests/rtdetrv2_upstream.json",
            "manifest_size_bytes": 33264,
            "manifest_raw_sha256": "f65a2d475365346a5dd5ce4f46b022a135b187e21421e922cc41eee4d28d20ae",
            "manifest_inventory_sha256": "2312c80d5b0fba88d43ffc6807c3fc150ae74b77740f2ab65072f044e033d6d7",
        },
        "conversion_r3_runtime_binding": {
            "relative_path": "artifacts/data/visdrone_protocol_v2_conversion_r3",
            "file_count": 25,
            "directory_count_excluding_root": 0,
            "total_size_bytes": 304418794,
            "completion_sha256": "46734010937168ac65bf27c3a6f4bf3f234554d5d4a99407903dec4b3ae9e817",
            "artifact_inventory_sha256": "aaee01ce4e00749b9db8ac5e6e49cf35875a8b266e9c3efc237bdf0b8b109ae7",
            "entry_canonical_inventory_sha256": "ddf32745302d6095e1c3dde89b1a85f05db53e6b4cc8ee85638e0d45bfa8d982",
            "config_sha256": "58be8659d6d7ae6faace82b36d45523f80cd48bae1bbcc6da4586c53056ec0d0",
            "category_contract_sha256": "b4b309f357cbe130a505a610dff340cc498dc74f766384c4559b2acd900728a0",
            "source_identity_sha256": "f0f16ba4438b51a09a6203f78884b199f8e2a2d3fb46309c357368a47a552bb9",
        },
        "primary_evaluator_runtime_binding": {
            "evaluator": {
                "evaluator_id": "visdrone_official_primary_evaluator_v1",
                "protocol_id": "visdrone_official_style_v1",
            },
            "implementation": {
                "commit": "036cca4d127ddd9e10e3cc7900c3eb759b55f59f",
                "tree": "fbe931976f6bac7d8e4b3bb319ff1905e99c5444",
            },
            "config": {
                "relative_path": "configs/baseline/visdrone_official_evaluator_v1.json",
                "raw_size_bytes": 3857,
                "raw_sha256": "36cfa69b0ff645c47c871283f917577ca27c760daf424e9544ab363dbf080ff5",
                "canonical_size_bytes": 3295,
                "canonical_sha256": "355de90bdb6007ed42ed65b3f653a1b8ea57f2184ab4fd48a9561b692b993998",
            },
            "authority_manifest": {
                "relative_path": "manifests/visdrone_det_toolkit_005445.json",
                "raw_size_bytes": 4166,
                "raw_sha256": "71168baf15d6d945fd5a4a6c5605b5ba533524efeede2a9d4020127576c1b36f",
                "canonical_size_bytes": 3351,
                "canonical_sha256": "5bad9faf7622fe4542aa3b46561d6d41fb2e4ee34f35551ecfd821c9577159e3",
            },
            "authority_archive_inventory": {
                "archive_size_bytes": 40960,
                "archive_sha256": "bf19dd9477210adf106c7cbf2a72370ed4af22dedb577f361f3dc9e77e99baa4",
                "file_count": 11,
                "inventory_sha256": "35a14a021509b82f1238912e5c77ebb3559f9ee6daa3cc6c92db810b5dce5da0",
            },
            "source_files": [
                {
                    "role": "primary_evaluator",
                    "relative_path": "src/sparse_rtdetr/baseline/primary_evaluator.py",
                    "sha256": "d831fc641ac930822e693f99fbbcdd48abbe76738a618525135dd099963901bd",
                },
                {
                    "role": "evaluation_protocol",
                    "relative_path": "src/sparse_rtdetr/data_protocol/evaluation.py",
                    "sha256": "e70ad71bb834b4cfc5d25441a78d2caa5982a86ae782918488924f67a832d216",
                },
            ],
            "independent_audit": {
                "stage": "P3_BASELINE_PRIMARY_EVALUATOR_V1_INDEPENDENT_AUDIT_R1",
                "classification": "PRIMARY_EVALUATOR_V1_CERTIFIED",
                "formal_return_code": 0,
                "script_size_bytes": 51244,
                "script_sha256": "4182f4082756fb2e52d6969e93889f24ae9af398cc98d67a1b9d990041b5524a",
                "mutation_cases_rejected": 129,
                "independent_oracle_cases": 100,
                "evaluator_tests_passed": 46,
                "full_cpu_tests_passed": 1116,
                "clean_archive_tests_passed": 46,
            },
        },
    }


def test_primary_evaluator_certification_document_and_historical_policy_are_frozen():
    training = _training()
    certification = training["source_bindings"]["primary_evaluator_certification"]
    assert set(certification) == {
        "evaluator",
        "implementation",
        "config",
        "authority_manifest",
        "authority_archive_inventory",
        "source_files",
        "independent_audit",
    }
    evaluator_config = json.loads(
        (ROOT / contract.PRIMARY_EVALUATOR_CONFIG_RELATIVE_PATH).read_text(encoding="utf-8")
    )
    assert evaluator_config["evaluator_id"] == certification["evaluator"]["evaluator_id"]
    assert evaluator_config["protocol_id"] == certification["evaluator"]["protocol_id"]
    assert evaluator_config["policy"]["independent_audit_pass"] is False
    assert evaluator_config["policy"]["training_gate_open"] is False
    assert certification["independent_audit"] == {
        "stage": "P3_BASELINE_PRIMARY_EVALUATOR_V1_INDEPENDENT_AUDIT_R1",
        "classification": "PRIMARY_EVALUATOR_V1_CERTIFIED",
        "formal_return_code": 0,
        "script_size_bytes": 51244,
        "script_sha256": "4182f4082756fb2e52d6969e93889f24ae9af398cc98d67a1b9d990041b5524a",
        "mutation_cases_rejected": 129,
        "independent_oracle_cases": 100,
        "evaluator_tests_passed": 46,
        "full_cpu_tests_passed": 1116,
        "clean_archive_tests_passed": 46,
    }
    contract.validate_training_contract(training, _baseline())


@pytest.mark.parametrize(
    "path",
    [
        ("source_bindings", "primary_evaluator_certification", "evaluator"),
        ("source_bindings", "primary_evaluator_certification", "implementation"),
        ("source_bindings", "primary_evaluator_certification", "config"),
        ("source_bindings", "primary_evaluator_certification", "authority_manifest"),
        ("source_bindings", "primary_evaluator_certification", "authority_archive_inventory"),
        ("source_bindings", "primary_evaluator_certification", "source_files", 0),
        ("source_bindings", "primary_evaluator_certification", "source_files", 1),
        ("source_bindings", "primary_evaluator_certification", "independent_audit"),
    ],
)
def test_primary_evaluator_certification_nested_objects_are_closed(path):
    candidate = _training()
    target = candidate
    for token in path:
        target = target[token] if type(target) is dict else target[token]
    first = next(iter(target))
    target["unexpected_key"] = "value"
    with pytest.raises(contract.TrainingContractError, match="closed schema extra keys"):
        contract._validate_closed_schema(candidate)

    candidate = _training()
    target = candidate
    for token in path:
        target = target[token] if type(target) is dict else target[token]
    target.pop(first)
    with pytest.raises(contract.TrainingContractError, match="closed schema missing keys"):
        contract._validate_closed_schema(candidate)


@pytest.mark.parametrize(
    ("required", "certified", "blocked"),
    [
        (False, False, False),
        (False, False, True),
        (False, True, False),
        (False, True, True),
        (True, False, False),
        (True, False, True),
        (True, True, True),
        (True, True, False),
    ],
)
def test_primary_evaluator_gate_truth_table_accepts_only_certified_open(required, certified, blocked):
    candidate = _training()
    evaluation = candidate["evaluation_and_selection"]
    evaluation["primary_evaluator_required_before_training"] = required
    evaluation["primary_evaluator_independently_certified"] = certified
    evaluation["training_launch_blocked"] = blocked
    if (required, certified, blocked) == (True, True, False):
        contract._validate_cross_fields(candidate)
    else:
        with pytest.raises(contract.TrainingContractError):
            contract._validate_cross_fields(candidate)


@pytest.mark.parametrize(
    ("pointer", "bad"),
    [
        ("/source_bindings/primary_evaluator_certification/implementation/commit", "0" * 40),
        ("/source_bindings/primary_evaluator_certification/implementation/tree", "0" * 40),
        ("/source_bindings/primary_evaluator_certification/config/relative_path", "../evaluator.json"),
        ("/source_bindings/primary_evaluator_certification/config/raw_sha256", "0" * 64),
        ("/source_bindings/primary_evaluator_certification/config/canonical_sha256", "0" * 64),
        ("/source_bindings/primary_evaluator_certification/authority_manifest/relative_path", "/manifest.json"),
        ("/source_bindings/primary_evaluator_certification/authority_manifest/raw_sha256", "0" * 64),
        ("/source_bindings/primary_evaluator_certification/authority_manifest/canonical_sha256", "0" * 64),
        ("/source_bindings/primary_evaluator_certification/authority_archive_inventory/archive_sha256", "0" * 64),
        ("/source_bindings/primary_evaluator_certification/authority_archive_inventory/inventory_sha256", "0" * 64),
        ("/source_bindings/primary_evaluator_certification/source_files/0/relative_path", "../primary.py"),
        ("/source_bindings/primary_evaluator_certification/source_files/0/sha256", "0" * 64),
        ("/source_bindings/primary_evaluator_certification/source_files/1/relative_path", "absolute/path.py"),
        ("/source_bindings/primary_evaluator_certification/source_files/1/sha256", "0" * 64),
        ("/source_bindings/primary_evaluator_certification/independent_audit/script_sha256", "0" * 64),
    ],
)
def test_primary_evaluator_certification_path_sha_and_git_identity_mutations_reject(pointer, bad):
    candidate = _training()
    _set_json_pointer(candidate, pointer, bad)
    with pytest.raises(contract.TrainingContractError):
        contract.validate_training_contract(candidate, _baseline())


@pytest.mark.parametrize(
    ("pointer", "bad"),
    [
        ("/source_bindings/primary_evaluator_certification/evaluator/evaluator_id", None),
        ("/source_bindings/primary_evaluator_certification/implementation/commit", None),
        ("/source_bindings/primary_evaluator_certification/config/relative_path", 1),
        ("/source_bindings/primary_evaluator_certification/authority_manifest/raw_sha256", False),
        ("/source_bindings/primary_evaluator_certification/authority_archive_inventory/archive_sha256", 1),
        ("/source_bindings/primary_evaluator_certification/source_files/0/role", 1),
        ("/source_bindings/primary_evaluator_certification/source_files/0/sha256", None),
        ("/source_bindings/primary_evaluator_certification/independent_audit/classification", 1),
        ("/source_bindings/primary_evaluator_certification/independent_audit/script_sha256", None),
    ],
)
def test_primary_evaluator_certification_wrong_builtin_types_reject(pointer, bad):
    candidate = _training()
    _set_json_pointer(candidate, pointer, bad)
    with pytest.raises(contract.TrainingContractError):
        contract._validate_closed_schema(candidate)


@pytest.mark.parametrize(
    ("relative", "kind"),
    [
        (contract.PRIMARY_EVALUATOR_CONFIG_RELATIVE_PATH, "missing"),
        (contract.PRIMARY_EVALUATOR_CONFIG_RELATIVE_PATH, "append"),
        (contract.PRIMARY_EVALUATOR_CONFIG_RELATIVE_PATH, "semantic"),
        (contract.PRIMARY_EVALUATOR_CONFIG_RELATIVE_PATH, "repack"),
        (contract.PRIMARY_EVALUATOR_MANIFEST_RELATIVE_PATH, "missing"),
        (contract.PRIMARY_EVALUATOR_MANIFEST_RELATIVE_PATH, "append"),
        (contract.PRIMARY_EVALUATOR_MANIFEST_RELATIVE_PATH, "semantic"),
        (contract.PRIMARY_EVALUATOR_MANIFEST_RELATIVE_PATH, "repack"),
    ],
)
def test_primary_evaluator_config_and_manifest_runtime_identity_mutations_reject(full_portable_fixture, relative, kind):
    path = full_portable_fixture / relative
    original = path.read_bytes()
    original_mode = stat.S_IMODE(path.stat().st_mode)
    try:
        if kind == "missing":
            path.unlink()
        elif kind == "append":
            path.write_bytes(original + b" ")
        else:
            value = json.loads(original.decode("utf-8"))
            if kind == "semantic":
                if relative == contract.PRIMARY_EVALUATOR_CONFIG_RELATIVE_PATH:
                    value["evaluator_id"] = "other_evaluator"
                else:
                    value["toolkit_version"] = "other"
            elif kind != "repack":
                raise AssertionError(kind)
            path.write_bytes(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2).encode("utf-8") + b"\n")
        with pytest.raises(contract.TrainingContractError):
            contract.training_contract_binding(full_portable_fixture)
    finally:
        path.write_bytes(original)
        path.chmod(original_mode)


def test_primary_evaluator_config_and_manifest_coordinated_repack_rejects(full_portable_fixture):
    config_path = full_portable_fixture / contract.PRIMARY_EVALUATOR_CONFIG_RELATIVE_PATH
    manifest_path = full_portable_fixture / contract.PRIMARY_EVALUATOR_MANIFEST_RELATIVE_PATH
    config_original = config_path.read_bytes()
    manifest_original = manifest_path.read_bytes()
    try:
        config_path.write_bytes(config_original + b" ")
        manifest_path.write_bytes(manifest_original + b" ")
        with pytest.raises(contract.TrainingContractError):
            contract.training_contract_binding(full_portable_fixture)
    finally:
        config_path.write_bytes(config_original)
        manifest_path.write_bytes(manifest_original)


def _remove_test_object(path: Path) -> None:
    try:
        observed = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISDIR(observed.st_mode):
        path.rmdir()
    else:
        path.unlink()


@pytest.mark.parametrize("source_index", [0, 1])
@pytest.mark.parametrize("kind", ["missing", "rename", "bytes", "symlink", "hardlink", "directory", "fifo"])
def test_primary_evaluator_source_file_object_matrix_rejects(full_portable_fixture, source_index, kind, tmp_path):
    relative = contract.PRIMARY_EVALUATOR_SOURCE_FILES[source_index][1]
    path = full_portable_fixture / relative
    original = path.read_bytes()
    original_mode = stat.S_IMODE(path.stat().st_mode)
    renamed = path.with_name(path.name + ".renamed")
    hardlink_source = tmp_path / "source-hardlink"
    try:
        if kind == "missing":
            path.unlink()
        elif kind == "rename":
            path.rename(renamed)
        elif kind == "bytes":
            path.write_bytes(original + b"source mutation")
        elif kind == "symlink":
            path.unlink()
            path.symlink_to(ROOT / relative)
        elif kind == "hardlink":
            _replace_with_hardlink(path, hardlink_source)
        elif kind == "directory":
            path.unlink()
            path.mkdir()
        elif kind == "fifo":
            _replace_with_fifo(path)
        else:
            raise AssertionError(kind)
        with pytest.raises(contract.TrainingContractError):
            contract.training_contract_binding(full_portable_fixture)
    finally:
        _remove_test_object(path)
        _remove_test_object(renamed)
        path.write_bytes(original)
        path.chmod(original_mode)
        _remove_test_object(hardlink_source)


@pytest.mark.parametrize("source_index", [0, 1])
def test_primary_evaluator_source_file_changed_during_read_rejects(full_portable_fixture, source_index, monkeypatch):
    relative = contract.PRIMARY_EVALUATOR_SOURCE_FILES[source_index][1]
    basename = Path(relative).name
    original_lstat = contract._lstat_at
    seen = 0

    def drifting_lstat(name, parent_fd):
        nonlocal seen
        observed = original_lstat(name, parent_fd)
        if name == basename:
            seen += 1
            if seen >= 2:
                return _stat_variant(observed, metadata=True)
        return observed

    monkeypatch.setattr(contract, "_lstat_at", drifting_lstat)
    with pytest.raises(contract.TrainingContractError, match="changed while reading|metadata drift|identity drift"):
        contract.training_contract_binding(full_portable_fixture)
    assert seen >= 2


def test_primary_evaluator_runtime_binding_is_detached_and_does_not_read_external_provenance(full_portable_fixture, monkeypatch):
    seen_paths = []
    original_read_file = contract._VerifiedRepository.read_file

    def recording_read_file(repository, relative):
        seen_paths.append(relative)
        return original_read_file(repository, relative)

    monkeypatch.setattr(contract._VerifiedRepository, "read_file", recording_read_file)
    binding = contract.training_contract_binding(full_portable_fixture)
    snapshot = copy.deepcopy(binding["primary_evaluator_runtime_binding"])
    binding["primary_evaluator_runtime_binding"]["source_files"][0]["sha256"] = "0" * 64
    binding["primary_evaluator_runtime_binding"]["config"]["raw_size_bytes"] = 1
    assert binding["primary_evaluator_runtime_binding"] != snapshot
    second = contract.training_contract_binding(full_portable_fixture)
    assert second["primary_evaluator_runtime_binding"] == snapshot
    assert all(type(relative) is str and not relative.startswith("/") for relative in seen_paths)
    assert "visdrone_det_toolkit_005445782213e20c.tar" in SOURCE.read_text(encoding="utf-8")


def test_primary_evaluator_config_authority_file_declarations_bind_to_frozen_manifest():
    training = _training()
    certification = training["source_bindings"]["primary_evaluator_certification"]
    raw = (ROOT / contract.PRIMARY_EVALUATOR_CONFIG_RELATIVE_PATH).read_bytes()
    document = json.loads(raw.decode("utf-8"))
    document["authority"]["authority_file_sha256"]["utils/evalRes.m"] = "0" * 64
    mutated = json.dumps(document, ensure_ascii=True, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    with pytest.raises(contract.TrainingContractError, match="authority file identity drift"):
        contract._validate_primary_evaluator_config_document(mutated, certification)


def test_closed_schema_exploit_replay_rejects_before_frozen_digest():
    candidate = _training()
    candidate["source_bindings"]["unexpected_nested_key"] = "value"
    with pytest.raises(contract.TrainingContractError, match=r"closed schema extra keys at /source_bindings"):
        contract._validate_closed_schema(candidate)
    with pytest.raises(contract.TrainingContractError, match=r"closed schema extra keys at /source_bindings"):
        contract.validate_training_contract(candidate, _baseline())


def test_closed_schema_all_object_roles_extra_missing_and_rename():
    valid = _training()
    roles = list(_closed_object_roles(valid))
    assert tuple(pointer for pointer, _, _ in roles) == contract._closed_schema_object_roles()
    assert len(roles) == 42
    cases = 0
    for pointer, _, schema in roles:
        _, required = schema
        first = next(iter(required))

        extra = copy.deepcopy(valid)
        target = _value_at_pointer(extra, pointer)
        target["unexpected_key"] = "value"
        with pytest.raises(contract.TrainingContractError, match="closed schema extra keys"):
            contract._validate_closed_schema(extra)
        cases += 1

        missing = copy.deepcopy(valid)
        target = _value_at_pointer(missing, pointer)
        target.pop(first)
        with pytest.raises(contract.TrainingContractError, match="closed schema missing keys"):
            contract._validate_closed_schema(missing)
        cases += 1

        renamed = copy.deepcopy(valid)
        target = _value_at_pointer(renamed, pointer)
        target[first + "_renamed"] = target.pop(first)
        with pytest.raises(contract.TrainingContractError, match="closed schema renamed keys"):
            contract._validate_closed_schema(renamed)
        cases += 1
    assert cases == 126


def _value_at_pointer(value, pointer):
    if pointer == "/":
        return value
    current = value
    for token in pointer.lstrip("/").split("/"):
        current = current[int(token)] if type(current) is list else current[token]
    return current


def test_closed_schema_builtin_dict_reorder_and_roundtrip_controls():
    valid = _training()
    candidates = [copy.deepcopy(valid), _reorder_builtin_dicts(valid), json.loads(json.dumps(valid))]
    expected = contract.canonical_training_contract_bytes(valid)
    for candidate in candidates:
        assert type(candidate) is dict
        assert candidate == valid
        assert contract.canonical_training_contract_bytes(candidate) == expected
        contract._assert_json_types(candidate)
        contract._validate_closed_schema(candidate)
        contract._validate_path_and_sha_fields(candidate)
        contract._validate_cross_fields(candidate)
        assert contract.validate_training_contract(candidate, _baseline()) == valid
    assert list(candidates[1]) == list(reversed(list(valid)))


def test_closed_schema_rejects_non_builtin_containers():
    class DictSubclass(dict):
        pass

    class ListSubclass(list):
        pass

    candidates = [OrderedDict(_training()), defaultdict(int, _training()), DictSubclass(_training())]
    for candidate in candidates:
        with pytest.raises(contract.TrainingContractError, match="closed schema object type mismatch"):
            contract._validate_closed_schema(candidate)
    candidate = _training()
    candidate["model"]["input_size"] = ListSubclass(candidate["model"]["input_size"])
    with pytest.raises(contract.TrainingContractError, match=r"closed schema list type mismatch at /model/input_size"):
        contract._validate_closed_schema(candidate)


def test_closed_schema_strict_scalar_and_amp_executable_parameter_types():
    mutations = [
        (("schema_version",), True, "scalar type mismatch"),
        (("initialization", "seed"), False, "scalar type mismatch"),
        (("topology", "world_size"), True, "scalar type mismatch"),
        (("model", "nms"), 0, "scalar type mismatch"),
        (("source_bindings", "vendor_recipe", "relative_path"), np.str_("vendor/x"), "scalar type mismatch"),
        (("source_bindings", "vendor_recipe", "sha256"), np.str_("a" * 64), "scalar type mismatch"),
        (("amp", "init_scale"), None, "scalar type mismatch"),
        (("amp", "growth_factor"), True, "scalar type mismatch"),
        (("amp", "backoff_factor"), 1, "scalar type mismatch"),
        (("amp", "growth_interval"), 1.0, "scalar type mismatch"),
        (("amp", "init_scale"), "other", "scalar type mismatch"),
    ]
    for path, bad, message in mutations:
        candidate = _training()
        _set_path(candidate, path, bad)
        with pytest.raises(contract.TrainingContractError, match=message):
            contract._validate_closed_schema(candidate)


def test_numeric_registry_exactly_covers_all_builtin_numeric_leaves():
    numeric = dict(_numeric_leaves(_training(), excluded=contract._EXTERNAL_BINDING_NUMERIC_POINTERS))
    registry = contract._TRAINING_CONTRACT_NUMERIC_CONSTRAINTS
    assert len(numeric) == 67
    assert set(numeric) == set(registry)
    for pointer, value in numeric.items():
        constraint = registry[pointer]
        assert constraint["expected_type"] is type(value)
        assert constraint["finite"] is True
        assert constraint["exact"] == value
        assert type(constraint["role"]) is str and constraint["role"]
    contract._validate_numeric_constraints(_training())


def test_amp_executable_defaults_and_scalar_partition_are_frozen():
    training = _training()
    amp = training["amp"]
    assert amp["init_scale"] == 65536.0
    assert amp["growth_factor"] == 2.0
    assert amp["backoff_factor"] == 0.5
    assert amp["growth_interval"] == 2000
    assert type(amp["init_scale"]) is float
    assert type(amp["growth_factor"]) is float
    assert type(amp["backoff_factor"]) is float
    assert type(amp["growth_interval"]) is int
    assert "IMPLEMENTATION_MUST_FREEZE_EXPLICITLY" not in json.dumps(training)

    numeric = {
        pointer for pointer, _ in _numeric_leaves(
            training, excluded=contract._EXTERNAL_BINDING_NUMERIC_POINTERS
        )
    }
    semantic = set(contract.semantic_contract_inventory(training)["leaf_pointers"])
    syntax_or_binding = set(contract._SEMANTIC_SYNTAX_OR_BINDING_POINTERS)
    external_binding = set(contract._EXTERNAL_BINDING_POINTERS)
    assert len(numeric) == 67
    assert len(semantic) == 129
    assert len(syntax_or_binding) == 36
    assert len(external_binding) == 9
    assert len(numeric | semantic | syntax_or_binding) == 232
    assert len(numeric | semantic | syntax_or_binding | external_binding) == 241
    assert not numeric & semantic
    assert not numeric & syntax_or_binding
    assert not semantic & syntax_or_binding
    assert not external_binding & numeric
    assert not external_binding & semantic
    assert not external_binding & syntax_or_binding
    assert {"/amp/init_scale", "/amp/growth_factor", "/amp/backoff_factor", "/amp/growth_interval"} <= numeric
    assert not {"/amp/init_scale", "/amp/growth_factor", "/amp/backoff_factor", "/amp/growth_interval"} & semantic

    amp_relation = next(
        rule for rule in contract._semantic_relation_rules()
        if rule["rule_id"] == "REL_AMP_EXECUTABLE_PARAMETERS"
    )
    assert amp_relation["frozen"] == {
        "/amp/enabled": True,
        "/amp/scaler_type": "GradScaler",
        "/amp/init_scale": 65536.0,
        "/amp/growth_factor": 2.0,
        "/amp/backoff_factor": 0.5,
        "/amp/growth_interval": 2000,
        "/amp/nonfinite_loss_allowed": 0,
        "/amp/nonfinite_gradient_allowed": 0,
        "/amp/optimizer_skipped_steps_allowed": 0,
        "/amp/overflow_events_allowed": 0,
        "/acceptance/amp_skip_or_overflow_allowed": False,
    }


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        (("init_scale",), 65536),
        (("init_scale",), True),
        (("init_scale",), "65536.0"),
        (("init_scale",), None),
        (("growth_factor",), 2),
        (("growth_factor",), False),
        (("growth_factor",), "2.0"),
        (("growth_factor",), None),
        (("backoff_factor",), 0),
        (("backoff_factor",), True),
        (("backoff_factor",), "0.5"),
        (("backoff_factor",), None),
        (("growth_interval",), 2000.0),
        (("growth_interval",), False),
        (("growth_interval",), "2000"),
        (("growth_interval",), None),
    ],
)
def test_amp_executable_parameters_reject_wrong_builtin_types(field, bad):
    candidate = _training()
    _set_path(candidate, ("amp",) + field, bad)
    with pytest.raises(contract.TrainingContractError, match=r"closed schema scalar type mismatch"):
        contract._validate_closed_schema(candidate)


@pytest.mark.parametrize(
    ("field", "bad", "category"),
    [
        (("init_scale",), float("nan"), "nonfinite"),
        (("init_scale",), float("inf"), "nonfinite"),
        (("init_scale",), float("-inf"), "nonfinite"),
        (("init_scale",), 0.0, "out_of_range"),
        (("init_scale",), -1.0, "out_of_range"),
        (("init_scale",), 32768.0, "frozen_literal_mismatch"),
        (("growth_factor",), float("nan"), "nonfinite"),
        (("growth_factor",), float("inf"), "nonfinite"),
        (("growth_factor",), float("-inf"), "nonfinite"),
        (("growth_factor",), 1.0, "out_of_range"),
        (("growth_factor",), 0.5, "out_of_range"),
        (("growth_factor",), 1.5, "frozen_literal_mismatch"),
        (("backoff_factor",), float("nan"), "nonfinite"),
        (("backoff_factor",), float("inf"), "nonfinite"),
        (("backoff_factor",), float("-inf"), "nonfinite"),
        (("backoff_factor",), 0.0, "out_of_range"),
        (("backoff_factor",), 1.0, "out_of_range"),
        (("backoff_factor",), 0.25, "frozen_literal_mismatch"),
        (("growth_interval",), 0, "out_of_range"),
        (("growth_interval",), -1, "out_of_range"),
        (("growth_interval",), 1000, "frozen_literal_mismatch"),
    ],
)
def test_amp_executable_parameters_reject_nonfinite_range_and_literal_drift(field, bad, category):
    candidate = _training()
    pointer = "/amp/" + field[0]
    _set_path(candidate, ("amp",) + field, bad)
    contract._validate_closed_schema(candidate)
    with pytest.raises(contract.TrainingContractError, match=rf"numeric {category} at {re.escape(pointer)}"):
        contract._validate_numeric_constraints(candidate)
    with pytest.raises(contract.TrainingContractError) as public:
        contract.validate_training_contract(candidate, _baseline())
    assert "frozen content drift" not in str(public.value)


def test_amp_relation_mutations_and_detached_validation_are_fail_closed():
    valid = _training()
    before = copy.deepcopy(valid)
    detached = contract.validate_training_contract(valid, _baseline())
    assert valid == before
    detached["amp"]["init_scale"] = 1.0
    assert valid == before

    relation = next(
        rule for rule in contract._semantic_relation_rules()
        if rule["rule_id"] == "REL_AMP_EXECUTABLE_PARAMETERS"
    )
    for pointer, expected in relation["frozen"].items():
        candidate = _training()
        _set_json_pointer(candidate, pointer, _semantic_value_drift(expected))
        with pytest.raises(contract.TrainingContractError, match=r"rule_id=REL_AMP_EXECUTABLE_PARAMETERS"):
            contract._validate_semantic_relation_rule(candidate, relation)
        with pytest.raises(contract.TrainingContractError) as public:
            contract.validate_training_contract(candidate, _baseline())
        assert "frozen content drift" not in str(public.value)


@pytest.mark.parametrize("bad", [-1, 1, 10**100, True, 0.0])
def test_numeric_seed_zero_exploit_replay(bad):
    candidate = _training()
    candidate["initialization"]["seed"] = bad
    if type(bad) is int:
        expected = "out_of_range" if bad < 0 else "frozen_literal_mismatch"
        with pytest.raises(contract.TrainingContractError, match=rf"numeric {expected} at /initialization/seed"):
            contract._validate_numeric_constraints(candidate)
        with pytest.raises(contract.TrainingContractError, match=rf"numeric {expected} at /initialization/seed"):
            contract.validate_training_contract(candidate, _baseline())
    else:
        with pytest.raises(contract.TrainingContractError, match=r"closed schema scalar type mismatch at /initialization/seed"):
            contract.validate_training_contract(candidate, _baseline())


def _in_range_alternative(constraint):
    exact = constraint["exact"]
    if constraint["expected_type"] is int:
        for value in (exact + 1, exact - 1, 0, 1, 2):
            if value == exact:
                continue
            lower = constraint["lower"]
            upper = constraint["upper"]
            if lower is not None and (value < lower or (value == lower and not constraint["lower_inclusive"])):
                continue
            if upper is not None and (value > upper or (value == upper and not constraint["upper_inclusive"])):
                continue
            return value
    else:
        for value in (exact / 2.0, exact + 0.01, 0.5, 0.0):
            if value == exact:
                continue
            lower = constraint["lower"]
            upper = constraint["upper"]
            if lower is not None and (value < lower or (value == lower and not constraint["lower_inclusive"])):
                continue
            if upper is not None and (value > upper or (value == upper and not constraint["upper_inclusive"])):
                continue
            return float(value)
    raise AssertionError("no in-range alternative for constraint")


def test_all_numeric_constraints_reject_range_nonfinite_and_literal_drift():
    registry = contract._TRAINING_CONTRACT_NUMERIC_CONSTRAINTS
    executed = 0
    for pointer, constraint in registry.items():
        cases = []
        if constraint["expected_type"] is float:
            cases.extend([(float("nan"), "nonfinite"), (float("inf"), "nonfinite"), (float("-inf"), "nonfinite")])
        lower = constraint["lower"]
        if lower is not None:
            bad = lower if not constraint["lower_inclusive"] else lower - 1
            bad = int(bad) if constraint["expected_type"] is int else float(bad)
            cases.append((bad, "out_of_range"))
        upper = constraint["upper"]
        if upper is not None:
            bad = upper if not constraint["upper_inclusive"] else upper + 1
            bad = int(bad) if constraint["expected_type"] is int else float(bad)
            cases.append((bad, "out_of_range"))
        cases.append((_in_range_alternative(constraint), "frozen_literal_mismatch"))
        for bad, category in cases:
            candidate = _training()
            _set_json_pointer(candidate, pointer, bad)
            with pytest.raises(contract.TrainingContractError, match=rf"numeric {category} at {re.escape(pointer)}"):
                contract._validate_numeric_constraints(candidate)
            with pytest.raises(contract.TrainingContractError, match=rf"numeric {category} at {re.escape(pointer)}"):
                contract.validate_training_contract(candidate, _baseline())
            executed += 1
    assert executed >= 100


def _set_json_pointer(value, pointer, replacement):
    tokens = pointer.lstrip("/").split("/")
    current = value
    for token in tokens[:-1]:
        current = current[int(token)] if type(current) is list else current[token]
    final = tokens[-1]
    if type(current) is list:
        current[int(final)] = replacement
    else:
        current[final] = replacement


def _pointer_path(pointer):
    return tuple(int(token) if token.isdigit() else token for token in pointer.lstrip("/").split("/"))


def _semantic_string_alternatives(expected):
    candidates = [expected + "_drift", "", " ", expected.swapcase(), expected + "X"]
    return tuple(dict.fromkeys(candidate for candidate in candidates if candidate != expected))


def _semantic_wrong_type(expected):
    if type(expected) is str:
        return None
    if type(expected) is bool:
        return 0
    if expected is None:
        return False
    raise AssertionError(type(expected))


def _semantic_same_type_alternative(expected):
    if type(expected) is str:
        return expected + "_drift"
    if type(expected) is bool:
        return not expected
    if expected is None:
        return None
    raise AssertionError(type(expected))


def _semantic_value_drift(expected):
    if type(expected) is str:
        return expected + "_drift"
    if type(expected) is bool:
        return not expected
    if type(expected) is int:
        return expected + 1
    if type(expected) is float:
        return expected + 0.125
    if expected is None:
        return "not-null"
    raise AssertionError(type(expected))


def test_semantic_registry_exactly_covers_all_in_scope_non_numeric_leaves():
    inventory = contract.semantic_contract_inventory(_training())
    assert inventory["leaf_count"] == 129
    assert inventory["registered_leaf_count"] == 129
    assert inventory["missing"] == ()
    assert inventory["extra"] == ()
    assert inventory["duplicate"] == ()
    assert inventory["sequence_rule_count"] == 7
    assert inventory["relation_rule_count"] == 12
    rule_ids = [rule["rule_id"] for rule in contract._TRAINING_CONTRACT_SEMANTIC_RULES]
    assert len(rule_ids) == len(set(rule_ids)) == 148
    contract._validate_semantic_rules(_training())


@pytest.mark.parametrize("bad", ["raw", "EMA", "", " ", None, False])
def test_ema_development_evaluation_weight_exploit_rejected_before_digest(bad):
    candidate = _training()
    candidate["ema"]["development_evaluation_weights"] = bad
    with pytest.raises(contract.TrainingContractError, match=r"pointer=/ema/development_evaluation_weights") as direct:
        contract._validate_semantic_rules(candidate)
    assert "rule_id=SEM_EMA_DEVELOPMENT_WEIGHTS" in str(direct.value)
    with pytest.raises(contract.TrainingContractError, match=r"pointer=/ema/development_evaluation_weights"):
        contract._validate_cross_fields(candidate)
    with pytest.raises(contract.TrainingContractError, match=r"(semantic|closed schema)") as public:
        contract.validate_training_contract(candidate, _baseline())
    assert "frozen content drift" not in str(public.value)


def test_every_semantic_leaf_rejects_same_type_and_wrong_type_before_digest():
    for rule in contract._semantic_leaf_rules():
        pointer = rule["pointer"]
        expected = rule["expected"]
        if expected is not None:
            candidate = _training()
            _set_json_pointer(candidate, pointer, _semantic_same_type_alternative(expected))
            with pytest.raises(contract.TrainingContractError, match=r"semantic"):
                contract._validate_semantic_rules(candidate)
            with pytest.raises(contract.TrainingContractError):
                contract.validate_training_contract(candidate, _baseline())

        candidate = _training()
        _set_json_pointer(candidate, pointer, _semantic_wrong_type(expected))
        with pytest.raises(contract.TrainingContractError, match=r"semantic"):
            contract._validate_semantic_rules(candidate)
        with pytest.raises(contract.TrainingContractError):
            contract.validate_training_contract(candidate, _baseline())


def test_every_string_semantic_leaf_rejects_empty_whitespace_and_case_drift():
    executed = 0
    for rule in contract._semantic_leaf_rules():
        expected = rule["expected"]
        if type(expected) is not str:
            continue
        for bad in _semantic_string_alternatives(expected):
            candidate = _training()
            _set_json_pointer(candidate, rule["pointer"], bad)
            with pytest.raises(contract.TrainingContractError, match=r"semantic"):
                contract._validate_semantic_rules(candidate)
            executed += 1
    assert executed >= 200


def test_semantic_ordered_list_rules_reject_reverse_drop_duplicate_and_append():
    for rule in contract._semantic_sequence_rules():
        pointer = rule["pointer"]
        original = copy.deepcopy(_value_at_pointer(_training(), pointer))
        mutations = [
            list(reversed(original)),
            original[:-1],
            original[:-1] + [copy.deepcopy(original[-2])],
            original + [copy.deepcopy(original[0])],
        ]
        for bad in mutations:
            candidate = _training()
            _set_path(candidate, _pointer_path(pointer), bad)
            with pytest.raises(contract.TrainingContractError, match=rf"rule_id={re.escape(rule['rule_id'])}"):
                contract._validate_semantic_sequence_rules(candidate)
            with pytest.raises(contract.TrainingContractError):
                contract.validate_training_contract(candidate, _baseline())


def test_semantic_relation_matrix_rejects_left_right_and_synchronized_drift():
    cases = 0
    for rule in contract._semantic_relation_rules():
        pointers = tuple(rule["frozen"])
        for selected in (pointers[:1], pointers[-1:], pointers):
            candidate = _training()
            for pointer in selected:
                _set_path(candidate, _pointer_path(pointer), _semantic_value_drift(rule["frozen"][pointer]))
            with pytest.raises(contract.TrainingContractError, match=rf"rule_id={re.escape(rule['rule_id'])}"):
                contract._validate_semantic_relation_rule(candidate, rule)
            with pytest.raises(contract.TrainingContractError):
                contract.validate_training_contract(candidate, _baseline())
            cases += 1
    assert cases == 36


def test_semantic_layering_distinguishes_closed_numeric_path_semantic_relation_and_digest():
    wrong_type = _training()
    wrong_type["initialization"]["seed"] = 0.0
    with pytest.raises(contract.TrainingContractError, match="closed schema scalar type mismatch"):
        contract.validate_training_contract(wrong_type, _baseline())

    numeric = _training()
    numeric["initialization"]["seed"] = -1
    with pytest.raises(contract.TrainingContractError, match="numeric out_of_range"):
        contract.validate_training_contract(numeric, _baseline())

    path = _training()
    path["source_bindings"]["vendor_recipe"]["relative_path"] = "/host/recipe.yml"
    with pytest.raises(contract.TrainingContractError, match="POSIX path"):
        contract.validate_training_contract(path, _baseline())

    semantic = _training()
    semantic["ema"]["development_evaluation_weights"] = "raw"
    with pytest.raises(contract.TrainingContractError, match="semantic"):
        contract.validate_training_contract(semantic, _baseline())

    relation = _training()
    relation["evaluation_and_selection"]["primary_evaluator_required_before_training"] = False
    relation["evaluation_and_selection"]["training_launch_blocked"] = False
    with pytest.raises(contract.TrainingContractError, match="semantic"):
        contract.validate_training_contract(relation, _baseline())

    digest = _training()
    digest["source_bindings"]["vendor_recipe"]["sha256"] = "a" * 64
    contract._validate_closed_schema(digest)
    contract._validate_numeric_constraints(digest)
    contract._validate_path_and_sha_fields(digest)
    contract._validate_semantic_rules(digest)
    with pytest.raises(contract.TrainingContractError, match="training contract frozen content drift"):
        contract.validate_training_contract(digest, _baseline())


def test_numeric_semantic_layers_are_distinct():
    wrong_type = _training()
    wrong_type["initialization"]["seed"] = 0.0
    with pytest.raises(contract.TrainingContractError, match="closed schema scalar type mismatch"):
        contract.validate_training_contract(wrong_type, _baseline())

    nonfinite = _training()
    nonfinite["optimizer"]["default_lr"] = float("nan")
    contract._validate_closed_schema(nonfinite)
    with pytest.raises(contract.TrainingContractError, match="numeric nonfinite"):
        contract.validate_training_contract(nonfinite, _baseline())

    out_of_range = _training()
    out_of_range["initialization"]["seed"] = -1
    with pytest.raises(contract.TrainingContractError, match="numeric out_of_range"):
        contract.validate_training_contract(out_of_range, _baseline())

    literal = _training()
    literal["initialization"]["seed"] = 1
    with pytest.raises(contract.TrainingContractError, match="numeric frozen_literal_mismatch"):
        contract.validate_training_contract(literal, _baseline())

    cross = _training()
    cross["evaluation_and_selection"]["training_launch_blocked"] = True
    contract._validate_numeric_constraints(cross)
    with pytest.raises(contract.TrainingContractError, match=r"semantic literal mismatch"):
        contract.validate_training_contract(cross, _baseline())

    digest = _training()
    digest["owner_decision"] = "T1_RANDOM_INITIALIZATION_RENAMED"
    contract._validate_numeric_constraints(digest)
    contract._validate_cross_fields(digest)
    with pytest.raises(contract.TrainingContractError, match=r"semantic literal mismatch"):
        contract.validate_training_contract(digest, _baseline())


def test_closed_schema_rejects_nested_container_kind_swaps():
    for path, bad, message in [
        (("source_bindings",), [], "object type mismatch"),
        (("augmentation", "transforms"), {}, "list type mismatch"),
        (("augmentation", "transforms", 0), [], "object type mismatch"),
    ]:
        candidate = _training()
        _set_path(candidate, path, bad)
        with pytest.raises(contract.TrainingContractError, match=message):
            contract._validate_closed_schema(candidate)


def test_validation_layers_have_distinct_rejections():
    exact = _training()
    exact["source_bindings"]["unexpected_nested_key"] = "value"
    with pytest.raises(contract.TrainingContractError, match="closed schema extra keys"):
        contract.validate_training_contract(exact, _baseline())

    cross = _training()
    cross["evaluation_and_selection"]["training_launch_blocked"] = True
    contract._validate_closed_schema(cross)
    contract._validate_numeric_constraints(cross)
    with pytest.raises(contract.TrainingContractError, match=r"semantic literal mismatch"):
        contract.validate_training_contract(cross, _baseline())

    digest = _training()
    digest["owner_decision"] = "T1_RANDOM_INITIALIZATION_RENAMED"
    contract._validate_closed_schema(digest)
    contract._validate_numeric_constraints(digest)
    contract._validate_path_and_sha_fields(digest)
    contract._validate_cross_fields(digest)
    with pytest.raises(contract.TrainingContractError, match=r"semantic literal mismatch"):
        contract.validate_training_contract(digest, _baseline())


def test_baseline_raw_canonical_model_category_and_role_binding():
    baseline_raw = (ROOT / contract.BASELINE_CONFIG_RELATIVE_PATH).read_bytes()
    assert len(baseline_raw) == 4316
    assert hashlib.sha256(baseline_raw).hexdigest() == contract.BASELINE_CONFIG_RAW_SHA256
    baseline = _baseline()
    assert hashlib.sha256(contract.canonical_training_contract_bytes(baseline)).hexdigest() == contract.BASELINE_CONFIG_CANONICAL_SHA256
    assert baseline["model"]["num_classes"] == 10
    assert baseline["model"]["PResNet"] == {"depth": 18, "pretrained": False}
    assert baseline["roles"]["train_core"]["status"] == "allowed"
    assert baseline["roles"]["development"]["status"] == "allowed"
    assert baseline["roles"]["confirmatory"]["status"] == "sealed_and_forbidden"
    assert baseline["roles"]["test"]["status"] == "forbidden"


def test_every_frozen_leaf_mutation_is_rejected():
    valid = _training()
    leaves = list(_leaves(valid))
    assert len(leaves) >= 180
    for path, value in leaves:
        candidate = copy.deepcopy(valid)
        _set_path(candidate, path, _different(value))
        with pytest.raises(contract.TrainingContractError):
            contract.validate_training_contract(candidate, _baseline())


@pytest.mark.parametrize("field", ["schema_version", "initialization.seed", "initialization.formal_run_count", "topology.world_size", "topology.train_micro_batch", "schedule.epochs"])
@pytest.mark.parametrize("bad", [True, 1.0, "1", np.int64(1)])
def test_strict_integer_matrix(field, bad):
    candidate = _training()
    path = tuple(field.split("."))
    _set_path(candidate, path, bad)
    with pytest.raises(contract.TrainingContractError):
        contract.validate_training_contract(candidate, _baseline())


def test_int_subclass_is_rejected():
    class IntSubclass(int):
        pass

    candidate = _training()
    candidate["schema_version"] = IntSubclass(1)
    with pytest.raises(contract.TrainingContractError):
        contract.validate_training_contract(candidate, _baseline())


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), True])
def test_nonfinite_and_bool_float_fields_are_rejected(bad):
    candidate = _training()
    candidate["optimizer"]["default_lr"] = bad
    with pytest.raises(contract.TrainingContractError):
        contract.validate_training_contract(candidate, _baseline())


@pytest.mark.parametrize("path", ["/absolute/config.json", "../escape.json", "a//b.json", "a/./b.json", "a\\b.json", "a\x00b.json", ""])
def test_nonportable_config_path_is_rejected(path):
    with pytest.raises(contract.TrainingContractError):
        contract.load_training_contract(ROOT, path)


@pytest.mark.parametrize("raw_transform", [
    lambda raw: b"\xef\xbb\xbf" + raw,
    lambda raw: raw.replace(b"\n", b"\r\n"),
    lambda raw: raw.rstrip(b"\n"),
    lambda raw: raw + b"\n",
    lambda raw: raw.replace(b'"schema_version": 1,', b'"schema_version": 1,\n  "schema_version": 1,', 1),
])
def test_bom_crlf_trailing_lf_and_duplicate_key_are_rejected(tmp_path, raw_transform):
    raw = (ROOT / contract.TRAINING_CONFIG_RELATIVE_PATH).read_bytes()
    portable = _portable_root(tmp_path, raw_transform(raw))
    with pytest.raises(contract.TrainingContractError):
        contract.load_training_contract(portable)


def test_semantically_equivalent_json_repack_is_rejected(tmp_path):
    repacked = json.dumps(_training(), sort_keys=True, indent=4).encode("utf-8") + b"\n"
    portable = _portable_root(tmp_path, repacked)
    with pytest.raises(contract.TrainingContractError, match="raw identity drift"):
        contract.load_training_contract(portable)


def test_invalid_sha_and_source_path_are_rejected():
    for path, bad in [
        (("source_bindings", "vendor_recipe", "sha256"), "A" * 64),
        (("source_bindings", "vendor_recipe", "relative_path"), "/host/recipe.yml"),
        (("source_bindings", "conversion_r3", "artifact_root"), "../data"),
    ]:
        candidate = _training()
        _set_path(candidate, path, bad)
        with pytest.raises(contract.TrainingContractError):
            contract.validate_training_contract(candidate, _baseline())


@pytest.mark.parametrize(("path", "bad"), [
    (("initialization", "pretrained"), True),
    (("initialization", "checkpoint"), "checkpoint.pth"),
    (("initialization", "network_download_forbidden"), False),
    (("initialization", "external_checkpoint_forbidden"), False),
])
def test_pretrained_checkpoint_download_and_cache_routes_are_rejected(path, bad):
    candidate = _training()
    _set_path(candidate, path, bad)
    with pytest.raises(contract.TrainingContractError):
        contract.validate_training_contract(candidate, _baseline())
    candidate = _training()
    candidate["initialization"]["pretrained_url"] = "https://example.invalid/model.pth"
    with pytest.raises(contract.TrainingContractError):
        contract.validate_training_contract(candidate, _baseline())


def test_batch_scheduler_warmup_and_decay_cross_field_drift_is_rejected():
    mutations = [
        (("topology", "effective_train_batch"), 32),
        (("learning_rate", "scheduler_step_unit"), "iteration"),
        (("learning_rate", "warmup_optimizer_steps"), 120),
        (("learning_rate", "milestones"), [100]),
        (("learning_rate", "expected_decay_events_within_120_epochs"), 1),
    ]
    for path, bad in mutations:
        candidate = _training()
        _set_path(candidate, path, bad)
        with pytest.raises(contract.TrainingContractError):
            contract.validate_training_contract(candidate, _baseline())


def test_transform_order_membership_parameters_and_multiscale_are_frozen():
    candidates = []
    reordered = _training()
    reordered["augmentation"]["transforms"][0:2] = reversed(reordered["augmentation"]["transforms"][0:2])
    candidates.append(reordered)
    removed = _training()
    removed["augmentation"]["transforms"].pop()
    candidates.append(removed)
    changed = _training()
    changed["augmentation"]["transforms"][2]["p"] = 0.7
    candidates.append(changed)
    multiscale = _training()
    multiscale["augmentation"]["multiscale_enabled"] = True
    candidates.append(multiscale)
    for candidate in candidates:
        with pytest.raises(contract.TrainingContractError):
            contract.validate_training_contract(candidate, _baseline())


def test_optimizer_tiebreaker_checkpoint_and_acceptance_mutations_are_rejected():
    mutations = [
        (("optimizer", "parameter_groups_must_be_mutually_exclusive"), False),
        (("evaluation_and_selection", "tie_breakers"), ["AR500", "AP50", "earlier_epoch"]),
        (("checkpoint_policy", "atomic_write_required"), False),
        (("checkpoint_policy", "required_state"), ["raw_model"]),
        (("acceptance", "amp_skip_or_overflow_allowed"), True),
        (("amp", "init_scale"), 32768.0),
    ]
    for path, bad in mutations:
        candidate = _training()
        _set_path(candidate, path, bad)
        with pytest.raises(contract.TrainingContractError):
            contract.validate_training_contract(candidate, _baseline())


def test_forbidden_roles_and_secondary_selection_cannot_be_enabled():
    mutations = [
        (("data_roles", "confirmatory"), "allowed"),
        (("data_roles", "test"), "allowed"),
        (("evaluation_and_selection", "secondary_diagnostic_only"), False),
        (("evaluation_and_selection", "secondary_cannot_certify_or_select"), False),
        (("evaluation_and_selection", "development_only"), False),
        (("acceptance", "confirmatory_metrics_accessed"), True),
        (("acceptance", "test_accessed"), True),
    ]
    for path, bad in mutations:
        candidate = _training()
        _set_path(candidate, path, bad)
        with pytest.raises(contract.TrainingContractError):
            contract.validate_training_contract(candidate, _baseline())


def test_host_user_data_root_and_environment_fallback_fields_are_rejected():
    for key, value in [
        ("data_root", "/data/visdrone"),
        ("host_path", "/host"),
        ("username", "user"),
        ("autodl_path", "/root/autodl-tmp"),
        ("environment_fallback", "DATA_ROOT"),
    ]:
        candidate = _training()
        candidate[key] = value
        with pytest.raises(contract.TrainingContractError):
            contract.validate_training_contract(candidate, _baseline())


def test_portable_tree_needs_no_runtime_artifact(tmp_path):
    portable = _portable_root(tmp_path, (ROOT / contract.TRAINING_CONFIG_RELATIVE_PATH).read_bytes())
    loaded = contract.load_training_contract(portable)
    assert loaded["training_contract_id"] == "rtdetrv2_r18_visdrone_baseline_training_v1"
    assert not (portable / "artifacts").exists()
    with pytest.raises(contract.TrainingContractError):
        contract.training_contract_binding(portable)


def test_import_has_no_torch_cuda_model_data_or_filesystem_side_effect(tmp_path):
    script = (
        "import importlib.util,json,sys; from pathlib import Path; "
        "root=Path(sys.argv[1]); before=sorted(str(p.relative_to(root)) for p in root.rglob('*')); "
        "s=importlib.util.spec_from_file_location('isolated_contract',sys.argv[2]); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        "after=sorted(str(p.relative_to(root)) for p in root.rglob('*')); "
        "print(json.dumps({'torch': 'torch' in sys.modules, 'unchanged': before==after})); "
        "raise SystemExit(0 if 'torch' not in sys.modules and before==after else 3)"
    )
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": "", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path), str(SOURCE)], check=False, text=True, capture_output=True, env=env)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"torch": False, "unchanged": True}


def test_baseline_mutations_are_rejected():
    mutations = [
        (("baseline_contract", "baseline_id"), "other"),
        (("model", "num_classes"), 80),
        (("model", "PResNet", "pretrained"), True),
        (("roles", "confirmatory", "status"), "allowed"),
        (("roles", "test", "status"), "allowed"),
    ]
    for path, bad in mutations:
        baseline = _baseline()
        _set_path(baseline, path, bad)
        with pytest.raises(contract.TrainingContractError):
            contract.validate_training_contract(_training(), baseline)


def test_runtime_root_identity_rejects_root_and_intermediate_symlinks(tmp_path):
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    real_root = real_parent / "repo"
    real_root.mkdir()
    _portable_root(real_root, (ROOT / contract.TRAINING_CONFIG_RELATIVE_PATH).read_bytes())

    root_alias = tmp_path / "root-alias"
    root_alias.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(contract.TrainingContractError):
        contract.load_training_contract(root_alias)

    parent_alias = tmp_path / "parent-alias"
    parent_alias.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(contract.TrainingContractError):
        contract.load_training_contract(parent_alias / "repo")


def test_runtime_root_identity_rejects_relative_noncanonical_file_and_missing_roots(tmp_path):
    real_root = tmp_path / "repo"
    real_root.mkdir()
    _portable_root(real_root, (ROOT / contract.TRAINING_CONFIG_RELATIVE_PATH).read_bytes())
    relative = os.path.relpath(real_root, start=Path.cwd())
    for candidate in (
        relative,
        str(real_root) + "/.",
        str(real_root) + "/../" + real_root.name,
        str(real_root) + "/",
        str(real_root).replace("/", "\\"),
    ):
        with pytest.raises(contract.TrainingContractError):
            contract.load_training_contract(candidate)

    file_root = tmp_path / "root-file"
    file_root.write_bytes(b"not a directory")
    with pytest.raises(contract.TrainingContractError):
        contract.load_training_contract(file_root)
    with pytest.raises(contract.TrainingContractError):
        contract.load_training_contract(tmp_path / "missing-root")


@pytest.mark.parametrize("relative", [contract.TRAINING_CONFIG_RELATIVE_PATH, contract.BASELINE_CONFIG_RELATIVE_PATH])
@pytest.mark.parametrize("kind", ["symlink", "hardlink", "directory", "fifo", "missing", "bytes"])
def test_runtime_config_and_baseline_object_matrix(tmp_path, relative, kind):
    portable = _portable_fixture(tmp_path)
    path = portable / relative
    if kind == "symlink":
        _replace_with_symlink(path, ROOT / relative)
    elif kind == "hardlink":
        _replace_with_hardlink(path, tmp_path / "hardlink-source")
    elif kind == "directory":
        path.unlink()
        path.mkdir()
    elif kind == "fifo":
        _replace_with_fifo(path)
    elif kind == "missing":
        path.unlink()
    elif kind == "bytes":
        path.write_bytes(path.read_bytes() + b"mutation")
    else:
        raise AssertionError(kind)
    with pytest.raises(contract.TrainingContractError):
        contract.load_training_contract(portable)


def test_runtime_config_intermediate_symlink_is_rejected(tmp_path):
    portable = _portable_fixture(tmp_path)
    original = portable / "configs" / "baseline"
    external = tmp_path / "external-baseline"
    original.rename(external)
    original.symlink_to(external, target_is_directory=True)
    with pytest.raises(contract.TrainingContractError):
        contract.load_training_contract(portable)


def test_runtime_replacement_is_rejected_before_content_read(tmp_path):
    portable = _portable_fixture(tmp_path)
    training_path = portable / contract.TRAINING_CONFIG_RELATIVE_PATH
    _replace_with_hardlink(training_path, tmp_path / "training-replacement")
    with pytest.raises(contract.TrainingContractError):
        contract.load_training_contract(portable)


def test_vendor_root_and_intermediate_symlinks_are_rejected(tmp_path):
    portable = _portable_fixture(tmp_path)
    vendor = portable / "vendor"
    external_vendor = tmp_path / "external-vendor"
    vendor.rename(external_vendor)
    vendor.symlink_to(external_vendor, target_is_directory=True)
    with pytest.raises(contract.TrainingContractError):
        contract.training_contract_binding(portable)

    portable = _portable_fixture(tmp_path / "second")
    original = portable / "vendor" / "rtdetrv2_pytorch" / "configs" / "rtdetrv2"
    external = tmp_path / "external-rtdetrv2-config"
    original.rename(external)
    original.symlink_to(external, target_is_directory=True)
    with pytest.raises(contract.TrainingContractError):
        contract.training_contract_binding(portable)


@pytest.mark.parametrize("relative", _vendor_source_paths(_training()))
@pytest.mark.parametrize("kind", ["symlink", "hardlink", "directory", "fifo", "missing", "bytes"])
def test_vendor_source_object_matrix(full_portable_fixture, relative, kind, tmp_path):
    portable = full_portable_fixture
    path = portable / relative
    original = path.read_bytes()
    original_mode = stat.S_IMODE(path.stat().st_mode)
    hardlink_source = tmp_path / "vendor-hardlink-source"
    try:
        if kind == "symlink":
            _replace_with_symlink(path, ROOT / relative)
        elif kind == "hardlink":
            _replace_with_hardlink(path, hardlink_source)
        elif kind == "directory":
            path.unlink()
            path.mkdir()
        elif kind == "fifo":
            _replace_with_fifo(path)
        elif kind == "missing":
            path.unlink()
        elif kind == "bytes":
            path.write_bytes(original + b"mutation")
        else:
            raise AssertionError(kind)
        with pytest.raises(contract.TrainingContractError):
            contract.training_contract_binding(portable)
    finally:
        if path.is_symlink() or path.is_file() or path.is_fifo():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
        path.write_bytes(original)
        path.chmod(original_mode)
        if hardlink_source.exists():
            hardlink_source.unlink()


def test_outside_fifo_symlink_is_rejected_without_reading_the_target(tmp_path):
    portable = _portable_fixture(tmp_path)
    outside_fifo = tmp_path / "outside.fifo"
    os.mkfifo(outside_fifo)
    _replace_with_symlink(portable / contract.TRAINING_CONFIG_RELATIVE_PATH, outside_fifo)
    script = (
        "import importlib.util,sys\n"
        "spec=importlib.util.spec_from_file_location('isolated_contract',sys.argv[2])\n"
        "module=importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "try:\n"
        "    module.load_training_contract(sys.argv[1])\n"
        "except module.TrainingContractError:\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(3)\n"
    )
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(ROOT / "src"),
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(portable), str(SOURCE)],
        check=False,
        text=True,
        capture_output=True,
        env=env,
        timeout=2,
    )
    assert result.returncode == 0, result.stderr
    assert stat.S_ISFIFO(os.lstat(outside_fifo).st_mode)


def test_runtime_boundary_does_not_leak_file_descriptors(tmp_path):
    portable = _portable_fixture(tmp_path)
    before = len(os.listdir("/proc/self/fd"))
    for _ in range(3):
        contract.load_training_contract(portable)
    root_alias = tmp_path / "root-alias"
    root_alias.symlink_to(portable, target_is_directory=True)
    for _ in range(3):
        with pytest.raises(contract.TrainingContractError):
            contract.load_training_contract(root_alias)
    after = len(os.listdir("/proc/self/fd"))
    assert after == before


def test_runtime_boundary_preserves_portable_tree_bytes(tmp_path):
    portable = _portable_fixture(tmp_path)
    before = {
        path.relative_to(portable): path.read_bytes()
        for path in portable.rglob("*")
        if path.is_file()
    }
    contract.load_training_contract(portable)
    after = {
        path.relative_to(portable): path.read_bytes()
        for path in portable.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_runtime_path_consumers_have_no_unsafe_path_methods():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    forbidden = {"resolve", "read_bytes", "is_file", "is_symlink", "realpath"}
    calls = [
        (node.lineno, node.func.attr)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in forbidden
    ]
    assert calls == []
    assert "os.path.realpath" not in SOURCE.read_text(encoding="utf-8")


def _vendor_manifest_paths() -> tuple[str, ...]:
    manifest = json.loads((ROOT / "manifests/rtdetrv2_upstream.json").read_text(encoding="utf-8"))
    return tuple(row["relative_path"] for row in manifest["files"])


def _write_manifest(root: Path, value: dict) -> None:
    path = root / "manifests/rtdetrv2_upstream.json"
    path.write_bytes(json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n")


def test_vendor_runtime_binding_returns_full_observed_identity_on_portable_fixture(full_portable_fixture):
    portable = full_portable_fixture
    binding = contract.training_contract_binding(portable)
    assert binding["vendor_runtime_binding"] == {
        "relative_path": "vendor/rtdetrv2_pytorch",
        "file_count": 124,
        "directory_count_excluding_root": 25,
        "total_size_bytes": 373735,
        "compact_inventory_sha256": "0fc6803665bc4b5720e983345b2cacb0147eceead9f882588880b6f8a0e68051",
        "manifest_relative_path": "manifests/rtdetrv2_upstream.json",
        "manifest_size_bytes": 33264,
        "manifest_raw_sha256": "f65a2d475365346a5dd5ce4f46b022a135b187e21421e922cc41eee4d28d20ae",
        "manifest_inventory_sha256": "2312c80d5b0fba88d43ffc6807c3fc150ae74b77740f2ab65072f044e033d6d7",
    }


@pytest.mark.parametrize("relative", _vendor_manifest_paths())
def test_every_tracked_vendor_file_byte_mutation_is_bound(full_portable_fixture, relative):
    portable = full_portable_fixture
    path = portable / relative
    original = path.read_bytes()
    original_mode = stat.S_IMODE(path.stat().st_mode)
    try:
        path.write_bytes(original + b"independent-vendor-mutation")
        with pytest.raises(contract.TrainingContractError):
            contract.training_contract_binding(portable)
    finally:
        path.write_bytes(original)
        path.chmod(original_mode)


@pytest.mark.parametrize("kind", ["missing", "extra", "rename", "empty", "directory_symlink", "file_hardlink", "fifo", "socket"])
def test_vendor_runtime_structure_mutations_are_rejected(full_portable_fixture, kind, monkeypatch, tmp_path):
    portable = full_portable_fixture
    vendor = portable / "vendor/rtdetrv2_pytorch"
    target = vendor / "LICENSE"
    original = target.read_bytes()
    original_mode = stat.S_IMODE(target.stat().st_mode)
    renamed = vendor / "LICENSE.renamed"
    external = tmp_path / "external-references"
    outside = tmp_path / "hardlink-source"
    try:
        if kind == "missing":
            target.unlink()
        elif kind == "extra":
            (vendor / "unbound-extra.txt").write_bytes(b"extra")
        elif kind == "rename":
            target.rename(renamed)
        elif kind == "empty":
            target.write_bytes(b"")
        elif kind == "directory_symlink":
            directory = vendor / "references"
            directory.rename(external)
            directory.symlink_to(external, target_is_directory=True)
        elif kind == "file_hardlink":
            outside.write_bytes(original)
            target.unlink()
            os.link(outside, target)
        elif kind == "fifo":
            target.unlink()
            os.mkfifo(target)
        elif kind == "socket":
            original_lstat = contract._lstat_at

            def fake_lstat(name, parent_fd):
                if name == "LICENSE":
                    return os.stat_result((stat.S_IFSOCK | 0o600, 0, 0, 1, os.getuid(), os.getgid(), 0, 0, 0, 0))
                return original_lstat(name, parent_fd)

            monkeypatch.setattr(contract, "_lstat_at", fake_lstat)
        else:
            raise AssertionError(kind)
        with pytest.raises(contract.TrainingContractError):
            contract.training_contract_binding(portable)
    finally:
        if renamed.exists():
            renamed.rename(target)
        if (vendor / "references").is_symlink():
            (vendor / "references").unlink()
        if external.exists():
            external.rename(vendor / "references")
        if (vendor / "unbound-extra.txt").exists():
            (vendor / "unbound-extra.txt").unlink()
        if target.exists() or target.is_fifo():
            target.unlink()
        target.write_bytes(original)
        target.chmod(original_mode)
        if outside.exists():
            outside.unlink()


def test_vendor_executable_mode_flip_is_rejected(full_portable_fixture):
    portable = full_portable_fixture
    path = portable / "vendor/rtdetrv2_pytorch/tools/onnx2trt.sh"
    original_mode = stat.S_IMODE(path.stat().st_mode)
    try:
        path.chmod(path.stat().st_mode ^ stat.S_IXUSR)
        with pytest.raises(contract.TrainingContractError):
            contract.training_contract_binding(portable)
    finally:
        path.chmod(original_mode)


@pytest.mark.parametrize("kind", ["missing", "append", "bom", "crlf", "duplicate", "repack"])
def test_vendor_manifest_raw_mutations_are_rejected(full_portable_fixture, kind):
    portable = full_portable_fixture
    path = portable / "manifests/rtdetrv2_upstream.json"
    raw = path.read_bytes()
    original_mode = stat.S_IMODE(path.stat().st_mode)
    try:
        if kind == "missing":
            path.unlink()
        elif kind == "append":
            path.write_bytes(raw + b"\n")
        elif kind == "bom":
            path.write_bytes(b"\xef\xbb\xbf" + raw)
        elif kind == "crlf":
            path.write_bytes(raw.replace(b"\n", b"\r\n"))
        elif kind == "duplicate":
            mutated = raw.replace(b'"schema_version": 1,', b'"schema_version": 1,\n  "schema_version": 1,', 1)
            assert mutated != raw
            path.write_bytes(mutated)
        elif kind == "repack":
            value = json.loads(raw.decode("utf-8"))
            path.write_bytes(json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n")
        else:
            raise AssertionError(kind)
        with pytest.raises(contract.TrainingContractError):
            contract.training_contract_binding(portable)
    finally:
        path.write_bytes(raw)
        path.chmod(original_mode)


def test_vendor_manifest_coordinated_repack_is_still_rejected(full_portable_fixture):
    portable = full_portable_fixture
    license_path = portable / "vendor/rtdetrv2_pytorch/LICENSE"
    original_license = license_path.read_bytes()
    manifest_path = portable / "manifests/rtdetrv2_upstream.json"
    original_manifest = manifest_path.read_bytes()
    try:
        license_path.write_bytes(original_license + b"coordinated mutation")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mutated_sha = hashlib.sha256(license_path.read_bytes()).hexdigest()
        for row in manifest["files"]:
            if row["relative_path"] == "vendor/rtdetrv2_pytorch/LICENSE":
                row["size_bytes"] = license_path.stat().st_size
                row["sha256"] = mutated_sha
        manifest["canonical_inventory_sha256"] = hashlib.sha256(
            json.dumps(manifest["files"], ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        _write_manifest(portable, manifest)
        with pytest.raises(contract.TrainingContractError):
            contract.training_contract_binding(portable)
    finally:
        license_path.write_bytes(original_license)
        manifest_path.write_bytes(original_manifest)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("source_bindings", "vendor_runtime", "relative_path"), "../vendor"),
        (("source_bindings", "vendor_runtime", "file_count"), 123),
        (("source_bindings", "vendor_runtime", "directory_count_excluding_root"), 24),
        (("source_bindings", "vendor_runtime", "total_size_bytes"), 1),
        (("source_bindings", "vendor_runtime", "compact_inventory_sha256"), "0" * 64),
        (("source_bindings", "vendor_manifest", "relative_path"), "manifest.json"),
        (("source_bindings", "vendor_manifest", "size_bytes"), 1),
        (("source_bindings", "vendor_manifest", "raw_sha256"), "0" * 64),
        (("source_bindings", "vendor_manifest", "canonical_inventory_sha256"), "0" * 64),
    ],
)
def test_vendor_contract_declarations_fail_before_digest(path, replacement):
    candidate = _training()
    _set_path(candidate, path, replacement)
    with pytest.raises(contract.TrainingContractError):
        contract.validate_training_contract(candidate, _baseline())


def test_vendor_inventory_failure_routes_do_not_leak_descriptors(full_portable_fixture):
    portable = full_portable_fixture
    before = len(os.listdir("/proc/self/fd"))
    path = portable / "vendor/rtdetrv2_pytorch/LICENSE"
    original = path.read_bytes()
    try:
        path.write_bytes(b"bad")
        for _ in range(3):
            with pytest.raises(contract.TrainingContractError):
                contract.training_contract_binding(portable)
        after = len(os.listdir("/proc/self/fd"))
        assert after == before
    finally:
        path.write_bytes(original)


def _conversion_file_names() -> tuple[str, ...]:
    return tuple(sorted(contract._CONVERSION_FLAT_FILE_NAMES))


def _flip_one_byte(path: Path) -> int | None:
    with path.open("r+b") as handle:
        original = handle.read(1)
        if original:
            handle.seek(0)
            handle.write(bytes([original[0] ^ 1]))
            return original[0]
    path.write_bytes(b"x")
    return None


def _restore_one_byte(path: Path, original: int | None) -> None:
    if original is None:
        path.write_bytes(b"")
        return
    with path.open("r+b") as handle:
        handle.seek(0)
        handle.write(bytes([original]))


def _stat_variant(value: os.stat_result, *, mode=None, nlink=None, device=None, metadata=False):
    fields = list(value)
    if mode is not None:
        fields[0] = mode
    if device is not None:
        fields[2] = device
    if nlink is not None:
        fields[3] = nlink
    if metadata:
        fields[8] += 1
        fields[9] += 1
    return os.stat_result(tuple(fields))


def test_conversion_r3_completion_missing_is_rejected(full_portable_fixture):
    portable = full_portable_fixture
    path = portable / contract.CONVERSION_R3_RELATIVE_PATH / "completion.json"
    original = path.read_bytes()
    original_mode = stat.S_IMODE(path.stat().st_mode)
    path.unlink()
    try:
        with pytest.raises(contract.TrainingContractError):
            contract.training_contract_binding(portable)
    finally:
        path.write_bytes(original)
        path.chmod(original_mode)


@pytest.mark.parametrize("relative", _conversion_file_names())
def test_every_conversion_r3_file_byte_mutation_is_bound(full_portable_fixture, relative):
    portable = full_portable_fixture
    path = portable / contract.CONVERSION_R3_RELATIVE_PATH / relative
    original = _flip_one_byte(path)
    try:
        with pytest.raises(contract.TrainingContractError):
            contract.training_contract_binding(portable)
    finally:
        _restore_one_byte(path, original)


@pytest.mark.parametrize("kind", ["missing", "extra", "rename", "empty", "directory", "symlink"])
def test_conversion_r3_real_structure_mutations_are_rejected(full_portable_fixture, kind, tmp_path):
    portable = full_portable_fixture
    root = portable / contract.CONVERSION_R3_RELATIVE_PATH
    target = root / "invocation.json"
    original = target.read_bytes()
    original_mode = stat.S_IMODE(target.stat().st_mode)
    renamed = root / "invocation.renamed.json"
    outside = tmp_path / "outside.json"
    extra = root / "unbound-extra.json"
    try:
        if kind == "missing":
            target.unlink()
        elif kind == "extra":
            extra.write_bytes(b"extra")
        elif kind == "rename":
            target.rename(renamed)
        elif kind == "empty":
            target.write_bytes(b"")
        elif kind == "directory":
            target.unlink()
            target.mkdir()
        elif kind == "symlink":
            outside.write_bytes(original)
            target.unlink()
            target.symlink_to(outside)
        else:
            raise AssertionError(kind)
        with pytest.raises(contract.TrainingContractError):
            contract.training_contract_binding(portable)
    finally:
        if renamed.exists():
            renamed.rename(target)
        if target.is_symlink():
            target.unlink()
        elif target.is_dir():
            target.rmdir()
        if extra.exists():
            extra.unlink()
        target.write_bytes(original)
        target.chmod(original_mode)
        if outside.exists():
            outside.unlink()


@pytest.mark.parametrize("kind", ["hardlink", "fifo", "socket", "device", "mode", "device_drift", "metadata"])
def test_conversion_r3_sandbox_safe_object_mutations_are_rejected(full_portable_fixture, kind, monkeypatch):
    portable = full_portable_fixture
    target_name = "conversion_audit.json"
    original_lstat = contract._lstat_at

    def fake_lstat(name, parent_fd):
        observed = original_lstat(name, parent_fd)
        if name != target_name:
            return observed
        if kind == "hardlink":
            return _stat_variant(observed, nlink=2)
        if kind == "fifo":
            return _stat_variant(observed, mode=stat.S_IFIFO | 0o600)
        if kind == "socket":
            return _stat_variant(observed, mode=stat.S_IFSOCK | 0o600)
        if kind == "device":
            return _stat_variant(observed, mode=stat.S_IFCHR | 0o600)
        if kind == "mode":
            return _stat_variant(observed, mode=stat.S_IFREG | 0o644)
        if kind == "device_drift":
            return _stat_variant(observed, device=observed.st_dev + 1)
        if kind == "metadata":
            return _stat_variant(observed, metadata=True)
        raise AssertionError(kind)

    monkeypatch.setattr(contract, "_lstat_at", fake_lstat)
    with pytest.raises(contract.TrainingContractError):
        contract.training_contract_binding(portable)


@pytest.mark.parametrize(
    "kind",
    [
        "inventory_extra_key",
        "inventory_missing_key",
        "inventory_reorder",
        "inventory_duplicate",
        "inventory_row_path",
        "inventory_row_size",
        "inventory_row_sha",
        "inventory_repack",
        "completion_status",
        "completion_type",
        "completion_reference",
        "config_type",
        "config_split",
        "category_type",
        "source_missing",
        "source_value",
        "coordinated_repack",
    ],
)
def test_conversion_r3_authority_mutations_are_rejected(full_portable_fixture, kind):
    portable = full_portable_fixture
    root = portable / contract.CONVERSION_R3_RELATIVE_PATH
    paths = {
        "inventory": root / "artifact_inventory.json",
        "completion": root / "completion.json",
        "config": root / "config.json",
        "category": root / "category_contract.json",
        "source": root / "source_identity.json",
        "lineage": root / "train_core_lineage.jsonl",
    }
    originals = {name: path.read_bytes() for name, path in paths.items()}
    try:
        if kind.startswith("inventory_"):
            value = json.loads(originals["inventory"].decode("utf-8"))
            if kind == "inventory_extra_key":
                value["extra"] = True
            elif kind == "inventory_missing_key":
                value.pop("artifacts")
            elif kind == "inventory_reorder":
                value["artifacts"] = list(reversed(value["artifacts"]))
            elif kind == "inventory_duplicate":
                value["artifacts"][1] = copy.deepcopy(value["artifacts"][0])
            elif kind == "inventory_row_path":
                value["artifacts"][0]["relative_path"] = "../escape"
            elif kind == "inventory_row_size":
                value["artifacts"][0]["size_bytes"] += 1
            elif kind == "inventory_row_sha":
                value["artifacts"][0]["sha256"] = "0" * 64
            elif kind == "inventory_repack":
                paths["inventory"].write_bytes(json.dumps(value, indent=2).encode("utf-8") + b"\n")
            if kind != "inventory_repack":
                paths["inventory"].write_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        elif kind.startswith("completion"):
            value = json.loads(originals["completion"].decode("utf-8"))
            if kind == "completion_status":
                value["status"] = "FAILED"
            elif kind == "completion_type":
                value["schema_version"] = True
            else:
                value["artifact_inventory_sha256"] = "0" * 64
            paths["completion"].write_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        elif kind.startswith("config"):
            value = json.loads(originals["config"].decode("utf-8"))
            if kind == "config_type":
                value["schema_version"] = True
            else:
                value["split"]["seed"] = 0
            paths["config"].write_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        elif kind == "category_type":
            value = json.loads(originals["category"].decode("utf-8"))
            value["num_classes"] = True
            paths["category"].write_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        elif kind.startswith("source"):
            value = json.loads(originals["source"].decode("utf-8"))
            if kind == "source_missing":
                value.pop("protocol_config_sha256")
            else:
                value["protocol_id"] = "OTHER"
            paths["source"].write_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        else:
            raw = bytearray(originals["lineage"])
            raw[0] ^= 1
            paths["lineage"].write_bytes(bytes(raw))
        with pytest.raises(contract.TrainingContractError):
            contract.training_contract_binding(portable)
    finally:
        for name, raw in originals.items():
            paths[name].write_bytes(raw)


def test_conversion_r3_enumeration_drift_is_rejected(full_portable_fixture, monkeypatch):
    portable = full_portable_fixture
    original_lstat = contract._lstat_at
    seen = 0

    def drifting_lstat(name, parent_fd):
        nonlocal seen
        observed = original_lstat(name, parent_fd)
        if name == "conversion_audit.json":
            seen += 1
            if seen >= 2:
                return _stat_variant(observed, metadata=True)
        return observed

    monkeypatch.setattr(contract, "_lstat_at", drifting_lstat)
    with pytest.raises(contract.TrainingContractError):
        contract.training_contract_binding(portable)


def test_conversion_r3_eintr_is_recovered(full_portable_fixture, monkeypatch):
    original_read = os.read
    raised = False

    def interrupted_once(fd, size):
        nonlocal raised
        if not raised:
            raised = True
            raise InterruptedError
        return original_read(fd, size)

    monkeypatch.setattr(os, "read", interrupted_once)
    binding = contract.training_contract_binding(full_portable_fixture)
    assert raised
    assert binding["conversion_r3_runtime_binding"]["file_count"] == 25


def test_conversion_r3_fd_leak_success_and_failure(full_portable_fixture):
    portable = full_portable_fixture
    completion = portable / contract.CONVERSION_R3_RELATIVE_PATH / "completion.json"
    original = completion.read_bytes()
    original_mode = stat.S_IMODE(completion.stat().st_mode)
    before = len(os.listdir("/proc/self/fd"))
    contract.training_contract_binding(portable)
    after_success = len(os.listdir("/proc/self/fd"))
    completion.unlink()
    try:
        with pytest.raises(contract.TrainingContractError):
            contract.training_contract_binding(portable)
        after_failure = len(os.listdir("/proc/self/fd"))
        assert after_success == before == after_failure
    finally:
        completion.write_bytes(original)
        completion.chmod(original_mode)
