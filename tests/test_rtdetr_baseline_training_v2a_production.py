from __future__ import annotations

import copy
import os
from pathlib import Path
import subprocess
import sys

import pytest

from sparse_rtdetr.baseline import training_v2a_contract as contract
from sparse_rtdetr.baseline import training_v2a_production as production
from sparse_rtdetr.baseline import training_v2a_runtime as runtime


ROOT = Path(__file__).resolve().parents[1]


def _data_roots(tmp_path: Path) -> dict[str, str]:
    train = tmp_path / "train-core"
    development = tmp_path / "development"
    train.mkdir()
    development.mkdir()
    (train / "train_core_coco.json").write_bytes(b"{}")
    (development / "development_coco.json").write_bytes(b"{}")
    return {"train_core": str(train), "development": str(development)}


def _environment() -> dict[str, object]:
    return {
        "gpu_name": "NVIDIA GeForce RTX 4090 D",
        "cuda_version": "12.4",
        "graphics_clock_mhz": 1500,
        "exclusive_host": True,
        "other_training_load": 0,
    }


@pytest.fixture
def frozen_binding() -> dict:
    return contract.training_v2a_contract_binding(ROOT)


@pytest.fixture
def policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frozen_binding: dict) -> dict:
    monkeypatch.setattr(runtime._contract, "training_v2a_contract_binding", lambda _root: copy.deepcopy(frozen_binding))
    roots = _data_roots(tmp_path)
    return production.build_v2a_production_policy(
        ROOT,
        data_roots=roots,
        environment_identity=_environment(),
        target_root=tmp_path / "t7d-target",
    )


class _Optimizer:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.zero_grad_calls = 0
        self.step_calls = 0

    def zero_grad(self) -> None:
        self.events.append("zero_grad")
        self.zero_grad_calls += 1

    def step(self) -> None:
        self.events.append("step")
        self.step_calls += 1

    def state_dict(self) -> dict[str, int]:
        return {"steps": self.step_calls}


class _EMA:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.update_calls = 0

    def update(self, model: object) -> None:
        del model
        self.events.append("ema")
        self.update_calls += 1

    def state_dict(self) -> dict[str, int]:
        return {"updates": self.update_calls}


class _PrimaryEvaluator:
    evaluator_id = "visdrone_official_primary_evaluator_v1"

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.calls: list[int] = []

    def evaluate(self, ema: object, epoch: int) -> dict[str, float]:
        del ema
        self.events.append("evaluate")
        self.calls.append(epoch)
        return {"AP": 50.0 + epoch, "AP50": 60.0, "AR500": 70.0}


def _fake_ports(policy: dict, events: list[str]) -> dict:
    optimizer = _Optimizer(events)
    ema = _EMA(events)
    evaluator = _PrimaryEvaluator(events)

    def strict_load() -> object:
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
        "evaluator": evaluator,
        "batches": [{"batch_size": policy["topology"]["train_micro_batch"]}],
        "compute_loss": lambda batch: 1.0,
        "backward": lambda loss: events.append("backward"),
        "raw_model_state": lambda: {"weight": [1]},
        "ema_state": lambda: {"weight": [2]},
        "rng_state": {"seed": 0},
        "optimizer_object": optimizer,
        "ema_object": ema,
    }


def _authorization_context(policy: dict, tmp_path: Path) -> dict:
    artifact = tmp_path / "detached-authorization.json"
    raw = b"detached-owner-authorization"
    artifact.write_bytes(raw)
    roots = policy["data_roots"]["roles"]
    flat_roots = {
        role: {
            "role": role,
            "root": item["root"],
            "annotation": item["annotation"],
        }
        for role, item in roots.items()
    }
    value = {
        "schema_version": 1,
        "kind": production.AUTHORIZATION_CONTEXT_KIND,
        "authorization_id": "owner-auth-t7d-test",
        "run_id": "run-t7d-test",
        "nonce": "nonce-t7d-test",
        "contract_id": policy["contract_id"],
        "baseline_id": policy["baseline_id"],
        "contract_sha256": policy["contract_sha256"],
        "authorization_path": str(artifact),
        "authorization_size_bytes": len(raw),
        "authorization_sha256": production._sha_bytes(raw),
        "consumed": False,
        "consumption_receipt_path": None,
        "owner": {"owner_id": "owner-test"},
        "data_roots": flat_roots,
        "environment_identity": {
            "gpu_name": "NVIDIA GeForce RTX 4090 D",
            "cuda_version": "12.4",
            "graphics_clock_mhz": 1500,
            "exclusive_host": True,
            "other_training_load": 0,
        },
        "descriptor_binding_sha256": "0" * 64,
    }
    value["context_sha256"] = production._context_digest(value)
    return value


def test_import_isolation_has_no_torch_filesystem_or_cuda_side_effect(tmp_path: Path) -> None:
    probe = (
        "import sys; before=set(sys.modules); "
        "import sparse_rtdetr.baseline.training_v2a_production; "
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


def test_policy_binds_t7c_and_uses_independent_target_names(policy: dict) -> None:
    assert policy["runtime_policy"]["runtime_id"].endswith("t7c")
    assert policy["production_id"].endswith("t7d_r1")
    assert policy["target"]["names"]["lock"].startswith("t7d_")
    assert policy["target"]["names"]["checkpoint_prefix"].startswith("t7d_")
    assert policy["production"] == {
        "owner_authorization_consumed": False,
        "gpu_probe_executed": False,
        "formal_training_executed": False,
        "training_ready": False,
        "independent_audit_pass": False,
    }


def test_cpu_fake_call_order_step_ema_primary_and_terminal(policy: dict) -> None:
    events: list[str] = []
    ports = _fake_ports(policy, events)
    result = production.run_v2a_cpu_fake(policy, ports, epochs=2)
    assert result["status"] == production.SUCCESS_STATUS
    assert result["mode"] == production.CPU_FAKE_MODE
    assert result["optimizer_updates"] == 2
    assert events.count("zero_grad") == 2
    assert events.count("step") == 2
    assert events.count("ema") == 2
    assert events.count("evaluate") == 2
    assert result["call_order"][:2] == ["strict_load", "create_optimizer"]
    assert result["call_order"][-1] == "epoch_loop"
    assert result["terminal_stdout"]["terminal_count"] == 1
    assert all("t7d_" in row["path"] or "t7d_" in Path(row["path"]).name for row in result["checkpoint_inventory"])
    assert result["production"]["formal_training_executed"] is False


def test_cpu_fake_duplicate_and_production_port_injection_are_rejected(policy: dict) -> None:
    ports = _fake_ports(policy, [])
    production.run_v2a_cpu_fake(policy, ports, epochs=1)
    with pytest.raises(production.V2AProductionError, match="already exists"):
        production.run_v2a_cpu_fake(policy, ports, epochs=1)

    separate = copy.deepcopy(policy)
    with pytest.raises(production.V2AProductionError, match="production capabilities"):
        production.run_v2a_cpu_fake(
            separate,
            {**_fake_ports(separate, []), "production_capability": object()},
        )


def test_detached_descriptor_context_digest_and_cross_field_mutations(policy: dict, tmp_path: Path) -> None:
    context = _authorization_context(policy, tmp_path)
    descriptor = production.build_v2a_detached_launch_descriptor(
        ROOT,
        authorization_context=context,
        run_id=context["run_id"],
        nonce=context["nonce"],
        target_root=tmp_path / "descriptor-target",
    )
    assert production.validate_v2a_detached_launch_descriptor(descriptor)["aggregate_sha256"] == descriptor["aggregate_sha256"]
    assert production.validate_v2a_authorization_context(context, descriptor=descriptor)["consumed"] is False

    mutated = copy.deepcopy(descriptor)
    mutated["run_id"] = "different-run"
    with pytest.raises(production.V2AProductionError):
        production.validate_v2a_detached_launch_descriptor(mutated)

    mutated_context = copy.deepcopy(context)
    mutated_context["nonce"] = "different-nonce"
    with pytest.raises(production.V2AProductionError):
        production.validate_v2a_authorization_context(mutated_context, descriptor=descriptor)

    forbidden_role = copy.deepcopy(context)
    forbidden_role["data_roots"]["test"] = {"role": "test", "root": "/tmp/test", "annotation": "test.json"}
    with pytest.raises(production.V2AProductionError):
        production.validate_v2a_authorization_context(forbidden_role)


def test_production_entry_rejects_unconsumed_artifact_drift_and_existing_target(policy: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context = _authorization_context(policy, tmp_path)
    descriptor = production.build_v2a_detached_launch_descriptor(
        ROOT,
        authorization_context=context,
        run_id=context["run_id"],
        nonce=context["nonce"],
        target_root=tmp_path / "occupied-target",
    )
    Path(descriptor["target_root"]).mkdir()
    with pytest.raises(production.V2AProductionError, match="target"):
        production.run_v2a_production_entry(descriptor, context)

    Path(descriptor["target_root"]).rmdir()
    Path(context["authorization_path"]).write_bytes(b"changed-owner-authorization")
    with pytest.raises(production.V2AProductionError, match="artifact identity"):
        production.run_v2a_production_entry(descriptor, context)


def test_terminal_classifier_covers_success_failure_exception_and_signal() -> None:
    terminal = {"kind": production.TERMINAL_KIND, "status": production.SUCCESS_STATUS}
    stdout = b"log before\n" + production._canonical(terminal) + b"\n"
    assert production.classify_v2a_terminal(0, stdout)["classification"] == "SUCCESS"
    assert production.classify_v2a_terminal(3, b"failure\n")["classification"] == "FAILURE"
    assert production.classify_v2a_terminal(1, b"", exception=RuntimeError("x"))["classification"] == "EXCEPTION"
    assert production.classify_v2a_terminal(-9, b"", signal_number=9)["classification"] == "SIGNAL"
    parsed = production.parse_v2a_production_stdout(stdout)
    assert parsed["stdout_bytes"] == stdout
    assert parsed["terminal_count"] == 1


def test_bf16_context_has_no_scaler_construction_or_source_surface(policy: dict) -> None:
    source = (ROOT / production.PRODUCTION_MODULE_RELATIVE_PATH).read_text(encoding="utf-8")
    assert "GradScaler" not in source

    class Context:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class FakeTorch:
        bfloat16 = object()

        @staticmethod
        def autocast(**kwargs):
            assert kwargs == {"device_type": "cuda", "dtype": FakeTorch.bfloat16, "enabled": True}
            return Context()

    with production._production_autocast(FakeTorch, policy["runtime_policy"]["contract_binding"]):
        pass
    assert not hasattr(FakeTorch, "GradScaler")


def test_production_loss_uses_vendor_weight_dict_exactly_once() -> None:
    class Criterion:
        weight_dict = {"classification": 2.0, "boxes": 5.0}

    assert production._weighted_loss(
        Criterion(),
        {"classification": 3.0, "boxes": 4.0, "diagnostic": 1000.0},
    ) == 26.0
    with pytest.raises(production.V2AProductionError, match="weight dictionary"):
        production._weighted_loss(object(), {"classification": 3.0})


def test_production_device_transfer_is_recursive() -> None:
    calls: list[str] = []

    class TensorLike:
        def __init__(self, name: str) -> None:
            self.name = name

        def to(self, device: str):
            calls.append(f"{self.name}:{device}")
            return self

    value = {"a": TensorLike("a"), "b": [TensorLike("b")], "c": (TensorLike("c"),)}
    assert production._move_to_device(value, "cuda:0") == value
    assert calls == ["a:cuda:0", "b:cuda:0", "c:cuda:0"]


def test_real_runtime_source_has_bound_data_evaluator_and_binary_checkpoints() -> None:
    source = (ROOT / production.PRODUCTION_MODULE_RELATIVE_PATH).read_text(encoding="utf-8")
    assert "primary development evaluator adapter was not supplied" not in source
    assert "_build_runtime_ports" in source
    assert "_evaluate_primary_metrics" in source
    assert "_weighted_loss" in source
    assert "clip_grad_norm_" in source
    assert "torch_save_v1" in source
    assert 'set_sharing_strategy("file_system")' in source
