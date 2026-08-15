from __future__ import annotations

import ast
import copy
from collections import OrderedDict, defaultdict
import hashlib
import importlib.util
import json
import os
import re
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


def _numeric_leaves(value, pointer=""):
    if type(value) is dict:
        for key, child in value.items():
            child_pointer = (pointer + "/" + key) if pointer else ("/" + key)
            yield from _numeric_leaves(child, child_pointer)
    elif type(value) is list:
        for index, child in enumerate(value):
            yield from _numeric_leaves(child, pointer + "/" + str(index))
    elif type(value) in {int, float}:
        yield pointer, value


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


def _portable_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "portable-repo"
    root.mkdir(parents=True)
    return _portable_root(root, (ROOT / contract.TRAINING_CONFIG_RELATIVE_PATH).read_bytes())


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


def test_numeric_registry_exactly_covers_all_builtin_numeric_leaves():
    numeric = dict(_numeric_leaves(_training()))
    registry = contract._TRAINING_CONTRACT_NUMERIC_CONSTRAINTS
    assert len(numeric) == 50
    assert set(numeric) == set(registry)
    for pointer, value in numeric.items():
        constraint = registry[pointer]
        assert constraint["expected_type"] is type(value)
        assert constraint["finite"] is True
        assert constraint["exact"] == value
        assert type(constraint["role"]) is str and constraint["role"]
    contract._validate_numeric_constraints(_training())


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
    assert inventory["leaf_count"] == 127
    assert inventory["registered_leaf_count"] == 127
    assert inventory["missing"] == ()
    assert inventory["extra"] == ()
    assert inventory["duplicate"] == ()
    assert inventory["sequence_rule_count"] == 6
    assert inventory["relation_rule_count"] == 11
    rule_ids = [rule["rule_id"] for rule in contract._TRAINING_CONTRACT_SEMANTIC_RULES]
    assert len(rule_ids) == len(set(rule_ids)) == 144
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
    assert cases == 33


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
    relation["evaluation_and_selection"]["primary_evaluator_independently_certified"] = True
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
    cross["evaluation_and_selection"]["training_launch_blocked"] = False
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
    cross["evaluation_and_selection"]["training_launch_blocked"] = False
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
def test_vendor_source_object_matrix(tmp_path, relative, kind):
    portable = _portable_fixture(tmp_path)
    path = portable / relative
    if kind == "symlink":
        _replace_with_symlink(path, ROOT / relative)
    elif kind == "hardlink":
        _replace_with_hardlink(path, tmp_path / "vendor-hardlink-source")
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
        contract.training_contract_binding(portable)


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
        contract.training_contract_binding(portable)
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
    contract.training_contract_binding(portable)
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
