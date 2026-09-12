from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from sparse_rtdetr.baseline import training_v2a_pilot as pilot


ROOT = Path(__file__).resolve().parents[1]


def _losses() -> dict[str, float]:
    losses = {"loss_vfl": 1.0, "loss_bbox": 2.0, "loss_giou": 3.0}
    for family in ("aux", "dn", "enc"):
        for index in range(2):
            for name in ("loss_vfl", "loss_bbox", "loss_giou"):
                losses[f"{name}_{family}_{index}"] = float(len(losses) + 1)
    return losses


def _gradient_evidence() -> dict:
    return {
        "parameter_count": 12,
        "rows": {
            name: {"exists": True, "finite": True, "nonzero": True}
            for name in pilot.ENCODER_PARAMETER_NAMES
        },
    }


def _optimizer_state_evidence() -> dict:
    return {
        "parameter_count": 12,
        "after_first_optimizer_update": True,
        "all_state_entries_present": True,
    }


class _Optimizer:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.steps = 0

    def zero_grad(self) -> None:
        self.events.append("zero_grad")

    def step(self) -> None:
        self.events.append("step")
        self.steps += 1

    def state_dict(self) -> dict[str, int]:
        return {"steps": self.steps}


class _EMA:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.updates = 0

    def update(self, model: object) -> None:
        del model
        self.events.append("ema")
        self.updates += 1

    def state_dict(self) -> dict[str, int]:
        return {"updates": self.updates}


class _Evaluator:
    evaluator_id = "visdrone_official_primary_evaluator_v1"

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def evaluate(self, ema: object, epoch: int) -> dict[str, float]:
        del ema
        self.events.append("evaluate")
        return {"AP": 20.0 + epoch, "AP50": 30.0 + epoch, "AR500": 40.0 + epoch}


def _ports(events: list[str]) -> dict:
    optimizer = _Optimizer(events)
    ema = _EMA(events)

    def strict_load() -> dict[str, object]:
        events.append("strict_load")
        return {"model": object()}

    def optimizer_factory(model: object) -> _Optimizer:
        del model
        events.append("optimizer_factory")
        return optimizer

    def ema_factory(model: object) -> _EMA:
        del model
        events.append("ema_factory")
        return ema

    return {
        "strict_load": strict_load,
        "optimizer_factory": optimizer_factory,
        "ema_factory": ema_factory,
        "evaluator": _Evaluator(events),
        "batches": [{"batch_size": 16}],
        "compute_loss": lambda batch: _losses(),
        "backward": lambda loss: events.append("backward"),
        "raw_model_state": lambda: {"weight": [1]},
        "ema_state": lambda: {"weight": [2]},
        "encoder_gradient_evidence": _gradient_evidence(),
        "encoder_optimizer_state_evidence": _optimizer_state_evidence(),
    }


@pytest.fixture
def policy(tmp_path: Path) -> dict:
    return pilot.build_t7h_pilot_policy(ROOT, target_root=tmp_path / "pilot-target")


def test_contract_is_closed_and_parent_overlay_is_exact() -> None:
    contract = pilot.load_t7h_pilot_contract(ROOT)
    assert contract["contract_id"] == pilot.PILOT_ID
    assert contract["overlay"]["checkpoint_epochs"] == list(range(1, 11))
    assert contract["overlay"]["development_evaluation_epochs"] == list(range(1, 11))
    mutated = copy.deepcopy(contract)
    mutated["overlay"]["executed_epochs"] = True
    with pytest.raises(pilot.T7HPilotError):
        pilot.validate_t7h_pilot_contract(mutated)
    mutated = copy.deepcopy(contract)
    mutated["requirements"]["unexpected"] = False
    with pytest.raises(pilot.T7HPilotError):
        pilot.validate_t7h_pilot_contract(mutated)


def test_policy_validates_parent_before_detached_overlay(policy: dict) -> None:
    parent = policy["parent_contract"]
    effective = policy["effective_parent_contract"]
    assert effective["schedule"]["epochs"] == 10
    assert effective["acceptance"]["required_epochs_complete"] == 10
    assert effective["checkpoint_policy"]["final_checkpoint"] == "epoch 10"
    changed = pilot._changed_pointers(parent, effective)
    assert changed == {
        "/schedule/epochs",
        "/acceptance/required_epochs_complete",
        "/checkpoint_policy/final_checkpoint",
    }
    assert policy["production"]["training_ready"] is False
    mutated = copy.deepcopy(policy)
    mutated["effective_parent_contract"]["optimizer"]["default_lr"] = 0.2
    with pytest.raises(pilot.T7HPilotError):
        pilot.validate_t7h_pilot_policy(mutated)


def test_import_isolation_has_no_torch_filesystem_or_cuda_side_effect(tmp_path: Path) -> None:
    probe = (
        "import sys; before=set(sys.modules); "
        "import sparse_rtdetr.baseline.training_v2a_pilot; "
        "print({'torch': 'torch' in sys.modules, 'numpy': 'numpy' in sys.modules, "
        "'new': sorted(set(sys.modules)-before)})"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", probe],
        cwd=ROOT,
        env={**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": "src"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "'torch': False" in result.stdout
    assert "'numpy': False" in result.stdout
    assert not list(tmp_path.iterdir())


def test_descriptor_is_unconsumed_and_rejects_mutation(policy: dict) -> None:
    descriptor = pilot.build_t7h_pilot_descriptor(policy)
    assert descriptor["authorization_required"] is True
    assert descriptor["authorization_created"] is False
    assert descriptor["data_accessed"] is False
    mutated = copy.deepcopy(descriptor)
    mutated["run_id"] = "other"
    with pytest.raises(pilot.T7HPilotError):
        pilot.validate_t7h_pilot_descriptor(mutated)


def test_cpu_fake_records_ten_epochs_loss_families_and_encoder_state(policy: dict) -> None:
    events: list[str] = []
    result = pilot.run_t7h_cpu_fake(policy, _ports(events))
    assert result["status"] == pilot.SUCCESS_STATUS
    assert result["epochs"] == 10
    assert result["checkpoint_epochs"] == list(range(1, 11))
    assert result["evaluation_epochs"] == list(range(1, 11))
    assert len(result["loss_records"]) == 10
    for row in result["loss_records"]:
        assert row["loss_total"] == sum(row["loss_family_subtotals"].values())
        assert set(row["loss_family_subtotals"]) == {"base", "aux", "dn", "encoder"}
    assert result["encoder_gradient_evidence"]["parameter_count"] == 12
    assert result["encoder_optimizer_state_evidence"]["after_first_optimizer_update"] is True
    assert result["production"]["formal_training_executed"] is False
    assert result["forbidden_operations"] == {
        "owner_authorization_created": 0,
        "owner_authorization_consumed": 0,
        "data_access": 0,
        "gpu_cuda": 0,
        "tmux": 0,
        "training": 0,
        "independent_audit": 0,
    }
    assert events.count("step") == 10
    assert events.count("ema") == 10
    assert events.count("evaluate") == 10


def test_cpu_fake_delegates_corrected_weighted_loss_and_rejects_injection(policy: dict) -> None:
    events: list[str] = []
    ports = _ports(events)
    calls = {"count": 0}

    def loss(batch):
        del batch
        calls["count"] += 1
        return _losses()

    ports["compute_loss"] = loss
    result = pilot.run_t7h_cpu_fake(policy, ports)
    assert result["status"] == pilot.SUCCESS_STATUS
    assert calls["count"] == 10
    with pytest.raises(pilot.T7HPilotError, match="production capabilities"):
        pilot.run_t7h_cpu_fake(policy, {**ports, "production_capability": object()})


def test_cpu_fake_rejects_replay_and_missing_encoder_evidence(tmp_path: Path) -> None:
    target = tmp_path / "pilot-target"
    policy = pilot.build_t7h_pilot_policy(ROOT, target_root=target)
    pilot.run_t7h_cpu_fake(policy, _ports([]))
    with pytest.raises(pilot.T7HPilotError, match="target"):
        pilot.run_t7h_cpu_fake(policy, _ports([]))

    other = pilot.build_t7h_pilot_policy(ROOT, target_root=tmp_path / "missing-evidence")
    bad_ports = _ports([])
    del bad_ports["encoder_gradient_evidence"]
    with pytest.raises(pilot.T7HPilotError, match="required"):
        pilot.run_t7h_cpu_fake(other, bad_ports)


def test_fake_result_is_canonical_and_round_trips(policy: dict) -> None:
    result = pilot.run_t7h_cpu_fake(policy, _ports([]))
    checked = pilot.validate_t7h_pilot_result(result, policy)
    assert checked == result
    assert result["result_sha256"] == pilot._digest({key: value for key, value in result.items() if key != "result_sha256"})
    progress = Path(result["progress_inventory"]["path"])
    assert progress.is_file()
    rows = [json.loads(line) for line in progress.read_text(encoding="utf-8").splitlines()]
    assert [row["epoch"] for row in rows] == list(range(1, 11))
