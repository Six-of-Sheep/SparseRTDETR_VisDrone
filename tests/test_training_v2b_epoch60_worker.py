from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "tools/run_training_v2b_epoch60_continuation_worker.py"
CAMPAIGN = ROOT / "tools/run_training_v2b_epoch60_continuation_campaign.py"


def load_worker():
    spec = importlib.util.spec_from_file_location("epoch60_worker_test", WORKER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_campaign():
    spec = importlib.util.spec_from_file_location("epoch60_campaign_test", CAMPAIGN)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_bootstrap_has_no_project_or_model_import_before_authenticated_source_selection():
    tree = ast.parse(WORKER.read_text(encoding="utf-8"))
    top_imports = []
    source_imports = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            top_imports.append(node)
        if isinstance(node, ast.FunctionDef) and node.name == "_source_imports":
            source_imports.extend(n for n in ast.walk(node) if isinstance(n, ast.ImportFrom))
    rendered = {(node.module or "") for node in top_imports if isinstance(node, ast.ImportFrom)}
    assert not any(name.startswith("sparse_rtdetr") or name.startswith("torch") for name in rendered)
    assert any((node.module or "").startswith("sparse_rtdetr") for node in source_imports)


def test_contract_is_hashed_before_project_import(tmp_path):
    worker = load_worker()
    contract = tmp_path / "contract.json"
    raw = b'{"repo_root":"/tmp"}\n'
    contract.write_bytes(raw)
    value, reference = worker._load_contract(str(contract.resolve()), hashlib.sha256(raw).hexdigest())
    assert value == {"repo_root": "/tmp"}
    assert reference["sha256_scope"] == "complete_file_bytes"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        worker._load_contract(str(contract.resolve()), "0" * 64)


def test_contract_parser_rejects_duplicate_keys_and_nonfinite_numbers(tmp_path):
    worker = load_worker()
    for name, raw, message in (
        ("duplicate", b'{"a":1,"a":2}\n', "duplicate"),
        ("nan", b'{"a":NaN}\n', "nonfinite"),
    ):
        path = tmp_path / f"{name}.json"
        path.write_bytes(raw)
        with pytest.raises(ValueError, match=message):
            worker._load_contract(str(path.resolve()), hashlib.sha256(raw).hexdigest())


@pytest.mark.parametrize("stage,receipts,replays,evaluations,checkpoints,expected", [
    ("smoke", 4, 0, 0, 2,
     {"epoch": 31, "epoch_active": True, "optimizer_updates": 9124, "microsteps": 18248}),
    ("smoke_replay", 2, 2, 0, 1,
     {"epoch": 31, "epoch_active": True, "optimizer_updates": 9124, "microsteps": 18248}),
    ("formal60", 9120, 0, 2, 30,
     {"epoch": 60, "epoch_active": False, "optimizer_updates": 18240, "microsteps": 36480}),
])
def test_worker_final_ledger_is_exact(stage, receipts, replays, evaluations, checkpoints, expected):
    worker = load_worker()
    engine = types.SimpleNamespace(state_dict=lambda: dict(expected))
    components = types.SimpleNamespace(engine=engine)
    actual = worker._final_clocks(
        stage, components, [None] * receipts, [None] * replays,
        [None] * evaluations, {str(i): None for i in range(checkpoints)})
    assert actual == expected


def test_worker_ledger_rejects_one_missing_formal_receipt():
    worker = load_worker()
    clocks = {"epoch": 60, "epoch_active": False, "optimizer_updates": 18240, "microsteps": 36480}
    components = types.SimpleNamespace(engine=types.SimpleNamespace(state_dict=lambda: clocks))
    with pytest.raises(RuntimeError, match="coverage drift"):
        worker._final_clocks("formal60", components, [None] * 9119, [], [None, None],
                             {str(i): None for i in range(30)})


def test_campaign_declares_the_exact_twelve_stage_plan_and_no_retry():
    tree = ast.parse(CAMPAIGN.read_text(encoding="utf-8"))
    text = CAMPAIGN.read_text(encoding="utf-8")
    assert 'len(report["gpu_worker_invocations"]) != 12' in text
    assert '"formal_retry_count": 0' in text
    assert "while " not in text
    # One subprocess site owned by the controller, never shell=True.
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "run"]
    assert len(calls) == 1
    assert not any(keyword.arg == "shell" and isinstance(keyword.value, ast.Constant)
                   and keyword.value.value is True for keyword in calls[0].keywords)


def test_campaign_creates_execution_parent_before_any_worker_invocation(tmp_path, monkeypatch):
    campaign = load_campaign()
    output = tmp_path / "new-campaign"
    contracts_dir = campaign._create_campaign_directories(output)

    assert contracts_dir == output / "contracts"
    assert contracts_dir.is_dir()
    assert (output / "execution").is_dir()
    assert output.stat().st_mode & 0o777 == 0o700
    assert contracts_dir.stat().st_mode & 0o777 == 0o700
    assert (output / "execution").stat().st_mode & 0o777 == 0o700

    observed = []

    def fake_invoke(*args, **kwargs):
        observed.append((output / "execution").is_dir())
        return {"returncode": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr(campaign, "_invoke", fake_invoke)
    campaign._invoke(None, None, cwd=tmp_path)
    assert observed == [True]
