from __future__ import annotations

import copy
import importlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from sparse_rtdetr.baseline import training_v2a_contract as contract
from sparse_rtdetr.baseline import training_v2a_runtime as runtime


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def frozen_binding() -> dict:
    return contract.training_v2a_contract_binding(ROOT)


def _data_roots(tmp_path: Path) -> dict[str, str]:
    train = tmp_path / "train-core"
    development = tmp_path / "development"
    train.mkdir()
    development.mkdir()
    (train / "train_core_coco.json").write_bytes(b"{}")
    (development / "development_coco.json").write_bytes(b"{}")
    return {"train_core": str(train), "development": str(development)}


def _environment() -> dict:
    return {
        "gpu_name": "NVIDIA GeForce RTX 4090 D",
        "cuda_version": "12.4",
        "graphics_clock_mhz": 1500,
        "exclusive_host": True,
        "other_training_load": 0,
        "observation_id": "fake-observation-t7c",
    }


def _policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, binding: dict) -> dict:
    monkeypatch.setattr(runtime._contract, "training_v2a_contract_binding", lambda _root: copy.deepcopy(binding))
    return runtime.build_v2a_runtime_policy(
        ROOT,
        data_roots=_data_roots(tmp_path),
        environment_identity=_environment(),
        target_root=tmp_path / "t7c-runtime-target",
    )


class _FakeParameter:
    def __init__(self, requires_grad: bool = True) -> None:
        self.requires_grad = requires_grad


class _FakeModel:
    def __init__(self) -> None:
        self.parameters_by_name = [
            ("backbone.conv.weight", _FakeParameter()),
            ("backbone.bn.weight", _FakeParameter()),
            ("encoder.norm.weight", _FakeParameter()),
            ("decoder.class_embed.weight", _FakeParameter()),
            ("frozen.weight", _FakeParameter(False)),
        ]

    def named_parameters(self):
        return iter(self.parameters_by_name)


class _FakeOptimizer:
    def __init__(self) -> None:
        self.zero_grad_calls = 0
        self.step_calls = 0

    def zero_grad(self) -> None:
        self.zero_grad_calls += 1

    def step(self):
        self.step_calls += 1
        return None


class _FakeEMA:
    def __init__(self) -> None:
        self.update_calls = 0

    def update(self, model) -> None:
        self.update_calls += 1


class _PrimaryFakeEvaluator:
    evaluator_id = "visdrone_official_primary_evaluator_v1"

    def __init__(self) -> None:
        self.calls = []

    def evaluate(self, ema_model, epoch):
        self.calls.append((ema_model, epoch))
        return {"AP": 0.5 + epoch * 0.001, "AP50": 0.6, "AR500": 0.7}


def _fake_ports() -> dict:
    model = _FakeModel()
    optimizer = _FakeOptimizer()
    ema = _FakeEMA()
    evaluator = _PrimaryFakeEvaluator()
    return {
        "model": model,
        "optimizer": optimizer,
        "ema": ema,
        "evaluator": evaluator,
        "batches": [{"batch_size": 16}],
        "compute_loss": lambda batch: 1.0,
        "backward": lambda loss: None,
        "raw_model_state": lambda: {"weight": [1, 2]},
        "ema_state": lambda: {"weight": [2, 3]},
    }


def test_import_isolation_has_no_torch_cuda_or_target_side_effect(tmp_path: Path):
    probe = (
        "import sys; before=set(sys.modules); "
        "import sparse_rtdetr.baseline.training_v2a_runtime; "
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


def test_policy_reads_binding_and_binds_all_input_closures(tmp_path: Path, monkeypatch, frozen_binding):
    policy = _policy(tmp_path, monkeypatch, frozen_binding)
    assert policy["contract_binding"] == frozen_binding
    assert policy["topology"]["train_micro_batch"] == 16
    assert policy["topology"]["effective_train_batch"] == 16
    assert policy["topology"]["train_workers"] == 4
    assert policy["topology"]["development_batch"] == 32
    assert policy["topology"]["development_workers"] == 4
    assert policy["topology"]["drop_last_train"] is True
    assert policy["topology"]["drop_last_development"] is False
    assert policy["optimizer"]["parameter_groups"][0]["weight_decay"] == 0.0
    assert policy["optimizer"]["parameter_groups"][1]["learning_rate"] == 1e-4
    assert policy["amp"]["grad_scaler_enabled"] is False
    assert policy["data_roots"]["confirmatory_read"] is False
    assert policy["data_roots"]["test_read"] is False
    for role in ("train_core", "development"):
        role_binding = policy["data_roots"]["roles"][role]
        assert Path(role_binding["annotation_path"]).parent == ROOT / "artifacts/data/visdrone_protocol_v2_conversion_r3"
        assert Path(role_binding["annotation_path"]).name == role_binding["annotation"]
        assert Path(role_binding["annotation_path"]).parent != Path(role_binding["root"])
    assert policy["environment_identity"]["hardware_probe_executed"] is False
    assert policy["production"]["formal_training_executed"] is False


def test_parent_data_and_environment_mutations_fail_closed(tmp_path: Path, frozen_binding):
    roots = _data_roots(tmp_path)
    with pytest.raises(runtime.V2ARuntimeError):
        runtime.bind_authorized_data_roots(
            frozen_binding,
            {"train_core": roots["train_core"], "development": roots["train_core"]},
            repo_root=ROOT,
        )
    with pytest.raises(runtime.V2ARuntimeError):
        runtime.bind_environment_identity(frozen_binding, {**_environment(), "graphics_clock_mhz": 1501})
    with pytest.raises(runtime.V2ARuntimeError):
        runtime.bind_environment_identity(frozen_binding, {**_environment(), "other_training_load": 1})
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    with pytest.raises(runtime.V2ARuntimeError):
        runtime.bind_training_parent_identity(ROOT, occupied)


def test_optimizer_groups_are_complete_mutually_exclusive_and_same_lr(frozen_binding):
    audit = runtime.audit_v2a_optimizer_parameters(_FakeModel(), frozen_binding)
    assert audit["group_counts"] == {"norm_or_bn": 2, "default": 2}
    assert audit["parameter_count"] == 4
    assert audit["covered_all_trainable_parameters"] is True
    assert audit["backbone_non_norm_lr_override"] is False

    mutated = copy.deepcopy(frozen_binding)
    mutated["contract"]["optimizer"]["parameter_groups"][0]["include_substrings"] = ["Norm"]
    with pytest.raises(runtime.V2ARuntimeError):
        runtime.audit_v2a_optimizer_parameters(_FakeModel(), mutated)

    mutated = copy.deepcopy(frozen_binding)
    mutated["contract"]["optimizer"]["parameter_groups"][1]["learning_rate"] = 0.1
    with pytest.raises(runtime.V2ARuntimeError):
        runtime.audit_v2a_optimizer_parameters(_FakeModel(), mutated)


def test_pretrained_strict_load_requires_local_state_and_rejects_key_drift(frozen_binding):
    class Tensor:
        def __init__(self, count):
            self.count = count

        def numel(self):
            return self.count

    counts = [1] * 114 + [11209824 - 114]
    state = {f"key_{index}": Tensor(count) for index, count in enumerate(counts)}

    class FakeCuda:
        @staticmethod
        def is_initialized():
            return False

    class FakeTorch:
        cuda = FakeCuda()

        @staticmethod
        def load(path, *, map_location, weights_only):
            assert map_location == "cpu"
            assert weights_only is True
            return state

    class FakeModel:
        def __init__(self, **kwargs):
            assert kwargs["pretrained"] is False
            self.kwargs = kwargs

        def load_state_dict(self, value, strict):
            assert strict is True
            return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    loaded = runtime.load_v2a_pretrained_backbone(
        ROOT,
        binding=frozen_binding,
        torch_module=FakeTorch,
        presnet_class=FakeModel,
    )
    assert loaded["strict_load_pass"] is True
    assert loaded["state_key_count"] == 115
    assert loaded["tensor_numel"] == 11209824

    class DriftModel(FakeModel):
        def load_state_dict(self, value, strict):
            return SimpleNamespace(missing_keys=["missing"], unexpected_keys=[])

    with pytest.raises(runtime.V2ARuntimeError):
        runtime.load_v2a_pretrained_backbone(ROOT, binding=frozen_binding, torch_module=FakeTorch, presnet_class=DriftModel)


def test_bf16_context_has_no_scaler_api(frozen_binding):
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

    with runtime.v2a_bf16_autocast_context(frozen_binding, torch_module=FakeTorch):
        pass
    assert not hasattr(FakeTorch, "GradScaler")


def test_selection_preserves_unrounded_float_and_tie_breakers(frozen_binding):
    selected = runtime.select_v2a_development_candidate(
        [
            {"epoch": 1, "AP": 0.5000000000001, "AP50": 0.5, "AR500": 0.5, "weights": "ema"},
            {"epoch": 2, "AP": 0.5, "AP50": 0.99, "AR500": 0.99, "weights": "ema"},
            {"epoch": 3, "AP": 0.5, "AP50": 0.99, "AR500": 0.99, "weights": "ema"},
        ],
        frozen_binding,
    )
    assert selected["epoch"] == 1
    selected = runtime.select_v2a_development_candidate(
        [
            {"epoch": 3, "AP": 0.5, "AP50": 0.7, "AR500": 0.6, "weights": "ema"},
            {"epoch": 2, "AP": 0.5, "AP50": 0.7, "AR500": 0.6, "weights": "ema"},
        ],
        frozen_binding,
    )
    assert selected["epoch"] == 2
    with pytest.raises(runtime.V2ARuntimeError):
        runtime.select_v2a_development_candidate(
            [{"epoch": 1, "AP": 0.5, "AP50": 0.5, "AR500": 0.5, "weights": "raw"}], frozen_binding
        )


def test_checkpoint_records_contract_authority_groups_and_no_scaler(tmp_path: Path, monkeypatch, frozen_binding):
    policy = _policy(tmp_path, monkeypatch, frozen_binding)
    checkpoint = runtime.build_v2a_checkpoint(
        policy,
        epoch=1,
        optimizer_updates=1,
        raw_model_state={"weight": [1]},
        ema_state={"weight": [2]},
        optimizer_audit={"parameter_rows": [{"parameter_name": "x", "group": "default"}], "group_counts": {"norm_or_bn": 0, "default": 1}},
    )
    checked = runtime.validate_v2a_checkpoint(checkpoint, policy)
    assert checked["authority_identity"]["weight"]["sha256"] == contract.AUTHORITY_WEIGHT_SHA256
    assert checked["amp_state"] == {
        "enabled": True,
        "autocast_dtype": "bfloat16",
        "grad_scaler_enabled": False,
        "scaler_policy": "disabled_absent",
    }
    assert "scaler_state" not in checked
    mutated = copy.deepcopy(checkpoint)
    mutated["amp_state"]["scaler_state"] = {}
    with pytest.raises(runtime.V2ARuntimeError):
        runtime.validate_v2a_checkpoint(mutated, policy)


def test_fake_runtime_closes_process_parent_data_environment_and_progress(tmp_path: Path, monkeypatch, frozen_binding):
    policy = _policy(tmp_path, monkeypatch, frozen_binding)
    ports = _fake_ports()
    result = runtime.run_v2a_fake_runtime(policy, ports, epochs=2)
    assert result["status"] == "FAKE_TERMINAL_COMPLETE"
    assert result["closure"] == {
        "process_terminal_json": True,
        "training_parent_identity": True,
        "authorization_data_roots": True,
        "environment_identity": True,
        "epoch_progress_evidence": True,
    }
    assert result["optimizer_updates"] == 2
    assert ports["optimizer"].zero_grad_calls == 2
    assert ports["optimizer"].step_calls == 2
    assert ports["ema"].update_calls == 2
    assert len(ports["evaluator"].calls) == 2
    parsed = runtime.parse_v2a_terminal_stdout(result["terminal_stdout"]["stdout_bytes"] if "stdout_bytes" in result["terminal_stdout"] else b"")
    assert parsed["value"]["run_id"] == runtime.RUNTIME_ID


def test_fake_runtime_rejects_secondary_evaluator_and_batch_adaptation(tmp_path: Path, monkeypatch, frozen_binding):
    policy = _policy(tmp_path, monkeypatch, frozen_binding)
    ports = _fake_ports()
    ports["evaluator"].evaluator_id = "secondary"
    with pytest.raises(runtime.V2ARuntimeError):
        runtime.run_v2a_fake_runtime(policy, ports)

    second = tmp_path / "second"
    second.mkdir()
    policy = _policy(second, monkeypatch, frozen_binding)
    ports = _fake_ports()
    ports["batches"] = [{"batch_size": 8}]
    with pytest.raises(runtime.V2ARuntimeError):
        runtime.run_v2a_fake_runtime(policy, ports)


def test_terminal_parser_selects_last_canonical_terminal_line_and_rejects_mutation():
    first = {"kind": runtime.PROCESS_TERMINAL_KIND, "status": "old"}
    second = {"kind": runtime.PROCESS_TERMINAL_KIND, "status": "new"}
    stdout = b"vendor line\n" + runtime._canonical(first) + b"\n" + runtime._canonical(second) + b"\n"
    parsed = runtime.parse_v2a_terminal_stdout(stdout)
    assert parsed["value"] == second
    assert parsed["stdout_sha256"] == runtime._sha_bytes(stdout)
    with pytest.raises(runtime.V2ARuntimeError):
        runtime.parse_v2a_terminal_stdout(stdout.replace(b"T7C_V2A_PROCESS_TERMINAL_RESULT", b"T7C_V2A_PROCESS_TERMINAL_MUTATED"))


def test_runtime_public_api_and_v1_surface_are_separate():
    module = importlib.import_module("sparse_rtdetr.baseline.training_v2a_runtime")
    assert all(hasattr(module, name) for name in module.__all__)
    assert "V2ARuntimeError" in module.__all__
    assert "training_t6_engine" not in module.__all__
    assert runtime.TARGET_RELATIVE_PATH != "artifacts/training/rtdetrv2_r18_visdrone_training_t6b_v1"
