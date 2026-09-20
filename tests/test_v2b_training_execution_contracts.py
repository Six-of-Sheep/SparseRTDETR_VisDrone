import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("v2b_contracts", ROOT / "tools" / "v2b_training_execution_contracts.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)

def test_runtime_locators_are_not_identity():
    a = {"x": 1, "runtime_locator": {"path": "/one"}, "nested": [{"runtime_locator": {"path": "/two"}, "y": 2}]}
    b = {"x": 1, "runtime_locator": {"path": "/different"}, "nested": [{"runtime_locator": {"path": "/else"}, "y": 2}]}
    assert MOD.canonical_sha(MOD.without_runtime(a)) == MOD.canonical_sha(MOD.without_runtime(b))

def test_canonical_json_stable():
    value = {"z": [2, 1], "a": {"b": True, "a": "x"}}
    assert MOD.canonical_bytes(value) == MOD.canonical_bytes(json.loads(MOD.canonical_bytes(value)))

def test_contract_files_are_canonical():
    directory = ROOT / "contracts" / "v2b" / "rev001"
    for name in ("training_contract.json", "execution_source.json", "lineage.json", "execution_contract.json", "campaign_authorization.json", "smoke_authorization.json"):
        raw = (directory / name).read_bytes()
        assert MOD.canonical_bytes(json.loads(raw)) == raw

def test_campaign_authorization_is_unconsumed():
    auth = json.loads((ROOT / "contracts" / "v2b" / "rev001" / "campaign_authorization.json").read_bytes())
    assert auth["consumed"] is False and auth["formal_launch_permitted"] is False

def test_lineage_boundary_and_failure():
    lineage = json.loads((ROOT / "contracts" / "v2b" / "rev001" / "lineage.json").read_bytes())
    assert lineage["kind"] == "revision_aware_continuation"
    assert lineage["resume_boundary"]["start_epoch"] == 54 and lineage["resume_boundary"]["start_window"] == 1
    assert lineage["preserved_failure"]["exception"] == "TrainCoreDataError: augmented batch changed after yield"
    assert lineage["preserved_failure"]["replayed"] is False

def test_execution_source_binds_71c16cf():
    doc = json.loads((ROOT / "contracts" / "v2b" / "rev001" / "execution_source.json").read_bytes())
    assert doc["source_commit"] == "71c16cf881eaacd58f678ee64f1a5df2ccf3ad9d"
    assert any(row["relative_path"] == "tools/repository_contract_check.py" for row in doc["files"])

def test_training_semantics():
    doc = json.loads((ROOT / "contracts" / "v2b" / "rev001" / "training_contract.json").read_bytes())
    data = doc["data"]
    assert (data["resolution"], data["seed"], data["physical_batch_size"], data["accumulation_steps"], data["effective_batch_size"]) == (896, 2, 8, 2, 16)
    assert doc["source_checkpoint"]["epoch"] == 53

def test_execution_contract_forbids_formal_launch():
    doc = json.loads((ROOT / "contracts" / "v2b" / "rev001" / "execution_contract.json").read_bytes())
    assert doc["permissions"]["formal_continuation_launch"] is False
    assert (doc["campaign"]["resume_epoch"], doc["campaign"]["first_window"]) == (54, 1)
