import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bridge = _load("bridge_v2b_checkpoint_revision", "bridge_v2b_checkpoint_revision.py")
_load("verify_v2b_checkpoint_revision", "verify_v2b_checkpoint_revision.py")


def test_bridge_binding_digest_and_failure_are_explicit():
    parent = {"binding_sha256": "a" * 64}
    current = {
        "schema_version": 1, "run_id": "new", "code": {},
        "config": {"seed": 2}, "initial_weights": {"x": 1},
        "data": {"x": 1}, "binding_sha256": "b" * 64,
    }
    contracts = {
        "parent_checkpoint_sha256": "c" * 64,
        "scientific_source_sha256": "d" * 64,
        "training_contract_sha256": "e" * 64,
        "execution_source_sha256": "f" * 64,
        "execution_contract_sha256": "0" * 64,
        "lineage_sha256": "1" * 64,
    }
    result = bridge._bridge_binding(current, parent, contracts)
    assert result["binding_sha256"] == bridge.canonical_sha(
        {k: v for k, v in result.items() if k != "binding_sha256"}
    )
    assert "revision_bridge" not in result["config"]
    assert result["provenance"]["revision_bridge"]["kind"] == "revision_bridge"
    assert result["provenance"]["revision_bridge"]["preserved_failure"]["replayed"] is False


def test_payload_descriptor_detects_tensor_mutation():
    import torch

    left = {"x": torch.tensor([1.0, 2.0])}
    right = {"x": torch.tensor([1.0, 3.0])}
    assert bridge.state_digest(left) != bridge.state_digest(right)


def test_manifest_and_auth_are_canonical(tmp_path):
    value = {"b": 2, "a": 1}
    path = tmp_path / "x.json"
    digest = bridge.write_json(path, value)
    assert path.read_bytes() == b'{"a":1,"b":2}'
    assert digest == bridge.sha_bytes(path.read_bytes())
    with pytest.raises(ValueError):
        bridge.write_json(path, value)
