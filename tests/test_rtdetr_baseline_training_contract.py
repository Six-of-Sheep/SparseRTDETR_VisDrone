from __future__ import annotations

import copy
from collections import OrderedDict, defaultdict
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
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


def _portable_root(tmp_path: Path, training_bytes: bytes) -> Path:
    for relative in (contract.TRAINING_CONFIG_RELATIVE_PATH, contract.BASELINE_CONFIG_RELATIVE_PATH):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        source = ROOT / relative
        target.write_bytes(training_bytes if relative == contract.TRAINING_CONFIG_RELATIVE_PATH else source.read_bytes())
    training = _training()
    sources = [training["source_bindings"]["vendor_recipe"], *training["source_bindings"]["vendor_includes"]]
    for item in sources:
        target = tmp_path / item["relative_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / item["relative_path"]).read_bytes())
    return tmp_path


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
        "raw_size_bytes": 8058,
        "raw_sha256": "8ce30e636caad84ee6e3e0ec10d384730a00861ccf179c1e765c095465c6cd5d",
        "canonical_size_bytes": 6866,
        "canonical_sha256": contract.TRAINING_CONTRACT_CANONICAL_SHA256,
    }


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
    assert len(roles) == 31
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
    assert cases == 93


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


def test_closed_schema_strict_scalar_and_amp_placeholder_types():
    mutations = [
        (("schema_version",), True, "scalar type mismatch"),
        (("initialization", "seed"), False, "scalar type mismatch"),
        (("topology", "world_size"), True, "scalar type mismatch"),
        (("model", "nms"), 0, "scalar type mismatch"),
        (("source_bindings", "vendor_recipe", "relative_path"), np.str_("vendor/x"), "scalar type mismatch"),
        (("source_bindings", "vendor_recipe", "sha256"), np.str_("a" * 64), "scalar type mismatch"),
        (("optimizer", "default_lr"), float("nan"), "non-finite float"),
        (("ema", "decay"), float("inf"), "non-finite float"),
        (("amp", "init_scale"), None, "literal mismatch"),
        (("amp", "growth_factor"), True, "literal mismatch"),
        (("amp", "backoff_factor"), 1, "literal mismatch"),
        (("amp", "growth_interval"), 1.0, "literal mismatch"),
        (("amp", "init_scale"), "other", "literal mismatch"),
    ]
    for path, bad, message in mutations:
        candidate = _training()
        _set_path(candidate, path, bad)
        with pytest.raises(contract.TrainingContractError, match=message):
            contract._validate_closed_schema(candidate)


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
    cross["topology"]["effective_train_batch"] = 32
    contract._validate_closed_schema(cross)
    with pytest.raises(contract.TrainingContractError, match="effective train batch arithmetic drift"):
        contract.validate_training_contract(cross, _baseline())

    digest = _training()
    digest["owner_decision"] = "T1_RANDOM_INITIALIZATION_RENAMED"
    contract._validate_closed_schema(digest)
    contract._validate_path_and_sha_fields(digest)
    contract._validate_cross_fields(digest)
    with pytest.raises(contract.TrainingContractError, match="training contract frozen content drift"):
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
        (("amp", "init_scale"), 65536.0),
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
    binding = contract.training_contract_binding(portable)
    assert binding["training_contract_id"] == "rtdetrv2_r18_visdrone_baseline_training_v1"
    assert not (portable / "artifacts").exists()


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
