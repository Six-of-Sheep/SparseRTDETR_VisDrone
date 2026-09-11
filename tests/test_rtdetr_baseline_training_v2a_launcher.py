from __future__ import annotations

import copy
import os
from pathlib import Path
import subprocess
import sys

import pytest

from sparse_rtdetr.baseline import training_v2a_contract as contract
from sparse_rtdetr.baseline import training_v2a_launcher as launcher
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


def _git_identity() -> dict[str, object]:
    return {
        "branch": "codex/t7e-fake",
        "head": "1" * 40,
        "parent": "2" * 40,
        "tree": "3" * 40,
        "upstream": "origin/codex/t7e-fake",
        "upstream_sha": "4" * 40,
        "worktree_status": "",
        "cache_counts": {"pycache": 0, "pyc": 0, "pytest_cache": 0},
    }


def _host_policy() -> dict[str, object]:
    return {
        "policy_id": "T7E_HOST_GPU_POLICY_R1",
        "tmux_path": "/usr/bin/tmux",
        "cuda_visible_devices": "",
        "gpu_index": 0,
        "gpu_probe_executed": False,
        "cuda_initialized": False,
        "exclusive_host_required": True,
        "training_allowed": True,
        "formal_training_executed": False,
        "network_download": False,
    }


def _target_identity(tmp_path: Path, t7d_target: str) -> dict[str, object]:
    control = tmp_path / "t7e-control"
    control.mkdir()
    return {
        "control_root": str(control),
        "plan_path": str(control / "t7e_launch_plan.json"),
        "descriptor_path": str(control / "t7e_descriptor.json"),
        "authorization_context_path": str(control / "t7e_authorization_context.json"),
        "receipt_path": str(control / "t7e_consumption_receipt.json"),
        "result_path": str(control / "t7e_immediate_result.json"),
        "stdout_path": str(control / "t7e_stdout.bin"),
        "stderr_path": str(control / "t7e_stderr.bin"),
        "snapshot_path": str(control / "t7e_immediate_snapshot.json"),
        "lock_path": str(control / "t7e_lock"),
        "t7d_target_root": t7d_target,
        "t7d_process_root": str(tmp_path / "t7d-process"),
        "t7d_outer_root": str(tmp_path / "t7d-outer"),
        "protected_paths": [str(tmp_path / "historical-v1-target"), str(tmp_path / "historical-t6b-target")],
        "tmux_session_name": "t7e_fake_session",
        "session_absent": True,
    }


def _authorization_context(policy: dict, tmp_path: Path) -> dict:
    artifact = tmp_path / "t7a-authority.json"
    raw = b"detached-t7a-authority-fixture"
    artifact.write_bytes(raw)
    roots = policy["data_roots"]["roles"]
    flat_roots = {
        role: {"role": item["role"], "root": item["root"], "annotation": item["annotation"]}
        for role, item in roots.items()
    }
    value = {
        "schema_version": 1,
        "kind": production.AUTHORIZATION_CONTEXT_KIND,
        "authorization_id": "t7d-context-fixture",
        "run_id": "t7e-run-fixture",
        "nonce": "t7e-nonce-fixture",
        "contract_id": policy["contract_id"],
        "baseline_id": policy["baseline_id"],
        "contract_sha256": policy["contract_sha256"],
        "authorization_path": str(artifact),
        "authorization_size_bytes": len(raw),
        "authorization_sha256": production._sha_bytes(raw),
        "consumed": False,
        "consumption_receipt_path": None,
        "owner": {"owner_id": "owner:t7e-test"},
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


@pytest.fixture
def policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    frozen = contract.training_v2a_contract_binding(ROOT)
    monkeypatch.setattr(runtime._contract, "training_v2a_contract_binding", lambda _root: copy.deepcopy(frozen))
    roots = _data_roots(tmp_path)
    return production.build_v2a_production_policy(
        ROOT,
        data_roots=roots,
        environment_identity=_environment(),
        target_root=tmp_path / "t7d-target",
    )


@pytest.fixture
def auth_plan(policy: dict, tmp_path: Path) -> tuple[dict, dict]:
    context = _authorization_context(policy, tmp_path)
    descriptor = production.build_v2a_detached_launch_descriptor(
        ROOT,
        authorization_context=context,
        run_id=context["run_id"],
        nonce=context["nonce"],
        target_root=policy["target"]["path"],
    )
    target = _target_identity(tmp_path, descriptor["target_root"])
    authorization = launcher.build_v2a_owner_authorization(
        tmp_path / "t7e-owner-authorization.json",
        t7d_policy=policy,
        descriptor=descriptor,
        authorization_context=context,
        git_identity=_git_identity(),
        host_gpu_policy=_host_policy(),
        target_identity=target,
        owner="owner:t7e-test",
        authorization_id="t7e-owner-authorization-fixture",
    )
    plan = launcher.build_v2a_launch_plan(authorization)
    return authorization, plan


def test_import_isolation_has_no_torch_numpy_filesystem_or_cuda_side_effect(tmp_path: Path) -> None:
    probe = (
        "import sys; before=set(sys.modules); "
        "import sparse_rtdetr.baseline.training_v2a_launcher; "
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


def test_closed_layers_bind_t7d_authority_and_fixed_argv(auth_plan: tuple[dict, dict]) -> None:
    authorization, plan = auth_plan
    assert plan["production_argv"][1:] == [
        "-m",
        launcher.PRODUCTION_MODULE_NAME,
        "--descriptor",
        plan["descriptor_path"],
        "--authorization-context",
        plan["authorization_context_path"],
    ]
    assert plan["parameter_layers"]["descriptor"]["sha256"] == plan["descriptor_sha256"]
    assert plan["parameter_layers"]["authorization_context"]["sha256"] == plan["authorization_context_sha256"]
    assert authorization["t7a_authority_sha256"] == launcher._digest(authorization["t7a_authority"])
    launcher.validate_v2a_launch_plan(plan, authorization)


def test_cpu_fake_exactly_once_call_graph_and_terminal_evidence(auth_plan: tuple[dict, dict]) -> None:
    authorization, plan = auth_plan
    calls: list[str] = []

    def new_session(argv, cwd, policy, value):
        calls.append("launcher")
        assert argv == value["production_argv"]
        assert cwd == value["repo_root"]
        return {"return_code": 0, "stdout": b"accepted", "stderr": b"", "pid": 0}

    def has_session(session_name, value):
        calls.append("snapshot")
        assert session_name == value["tmux_session_name"]
        return {"return_code": 0, "present": True, "stdout": b"present", "stderr": b""}

    result = launcher.run_v2a_cpu_fake_launch(
        authorization,
        plan,
        {"new_session": new_session, "has_session": has_session},
    )
    assert result["status"] == launcher.TMUX_ACCEPTED
    assert result["training_certified"] is False
    assert result["production"]["formal_training_executed"] is False
    assert result["call_counts"] == {"preflight": 1, "persist": 1, "consume": 1, "launcher": 1, "immediate_snapshot": 1}
    assert calls == ["launcher", "snapshot"]
    assert Path(authorization["target_identity"]["receipt_path"]).is_file()
    assert Path(authorization["target_identity"]["t7d_target_root"]).exists() is False


def test_duplicate_consumption_is_rejected_without_second_fake_call(auth_plan: tuple[dict, dict]) -> None:
    authorization, plan = auth_plan
    calls: list[str] = []
    ports = {
        "new_session": lambda *args: (calls.append("launch") or {"return_code": 0, "stdout": b"", "stderr": b"", "pid": 0}),
        "has_session": lambda *args: (calls.append("snapshot") or {"return_code": 0, "present": True, "stdout": b"", "stderr": b""}),
    }
    launcher.run_v2a_cpu_fake_launch(authorization, plan, ports)
    with pytest.raises(launcher.V2ALauncherError, match="must be absent|already exists"):
        launcher.run_v2a_cpu_fake_launch(authorization, plan, ports)
    assert calls == ["launch", "snapshot"]


def test_preflight_failure_has_zero_writes(auth_plan: tuple[dict, dict], tmp_path: Path) -> None:
    authorization, plan = auth_plan
    occupied = tmp_path / "occupied-before-preflight"
    occupied.write_bytes(b"occupied")
    mutated = copy.deepcopy(authorization)
    mutated["target_identity"]["protected_paths"][0] = str(occupied)
    mutated["authorization_sha256"] = launcher._authorization_digest(mutated)
    mutated_plan = launcher.build_v2a_launch_plan(mutated)
    with pytest.raises(launcher.V2ALauncherError, match="protected target"):
        launcher.run_v2a_cpu_fake_launch(mutated, mutated_plan, {})
    assert not Path(mutated["target_identity"]["plan_path"]).exists()
    assert not Path(mutated["target_identity"]["receipt_path"]).exists()


def test_post_consumption_launcher_failure_publishes_permanent_result(auth_plan: tuple[dict, dict]) -> None:
    authorization, plan = auth_plan
    result = launcher.run_v2a_cpu_fake_launch(
        authorization,
        plan,
        {
            "new_session": lambda *args: {"return_code": 17, "stdout": b"reject", "stderr": b"failed", "pid": 0},
            "has_session": lambda *args: {"return_code": 1, "present": False, "stdout": b"", "stderr": b"missing"},
        },
    )
    assert result["status"] == launcher.PERMANENT_FAIL
    assert result["production"]["training_ready"] is False
    assert Path(authorization["target_identity"]["receipt_path"]).is_file()
    assert Path(authorization["target_identity"]["result_path"]).is_file()


@pytest.mark.parametrize("field", ["nonce", "t7a_authority", "git_identity", "host_gpu_policy"])
def test_authorization_mutations_fail_closed(auth_plan: tuple[dict, dict], field: str) -> None:
    authorization, _ = auth_plan
    mutated = copy.deepcopy(authorization)
    if field == "nonce":
        mutated[field] = "mutated"
    elif field == "t7a_authority":
        mutated[field]["weight"]["sha256"] = "f" * 64
    elif field == "git_identity":
        mutated[field]["head"] = "f" * 40
    else:
        mutated[field]["cuda_visible_devices"] = "0"
    with pytest.raises(launcher.V2ALauncherError):
        launcher.validate_v2a_owner_authorization(mutated)


@pytest.mark.parametrize("field", ["production_argv", "descriptor_sha256", "parameter_layers"])
def test_plan_mutations_fail_closed(auth_plan: tuple[dict, dict], field: str) -> None:
    authorization, plan = auth_plan
    mutated = copy.deepcopy(plan)
    if field == "production_argv":
        mutated[field][-1] = "/tmp/forged-context.json"
    elif field == "descriptor_sha256":
        mutated[field] = "f" * 64
    else:
        mutated[field]["descriptor"]["path"] = "/tmp/forged.json"
    with pytest.raises(launcher.V2ALauncherError):
        launcher.validate_v2a_launch_plan(mutated, authorization)


def test_public_fake_ports_cannot_inject_production_capabilities(auth_plan: tuple[dict, dict]) -> None:
    authorization, plan = auth_plan
    with pytest.raises(launcher.V2ALauncherError, match="production capabilities"):
        launcher.run_v2a_cpu_fake_launch(authorization, plan, {"subprocess": object()})


def test_authorization_file_identity_rejects_symlink_and_hardlink(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_bytes(b"x")
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(source)
    with pytest.raises(launcher.V2ALauncherError):
        launcher._authorization_file_identity(symlink)
    hardlink = tmp_path / "hardlink.json"
    os.link(source, hardlink)
    with pytest.raises(launcher.V2ALauncherError):
        launcher._authorization_file_identity(source)


def test_production_authorization_identity_matches_receipt_schema(auth_plan: tuple[dict, dict]) -> None:
    authorization, plan = auth_plan
    path = Path(authorization["authorization_path"])
    path.write_bytes(launcher.canonical_v2a_launcher_bytes(authorization))
    path.chmod(0o600)
    identity = launcher._authorization_file_identity(path)
    assert set(identity) == {"path", "size_bytes", "sha256", "mode", "nlink"}
    receipt = launcher.consume_v2a_authorization(
        authorization,
        plan,
        mode="production",
        owner_file_identity=identity,
    )
    assert receipt["authorization_file_identity"] == identity
    assert receipt["consumed"] is True


def test_ast_and_real_call_boundary_are_closed() -> None:
    source = (ROOT / launcher.MODULE_RELATIVE_PATH).read_text(encoding="utf-8")
    assert "shell=False" in source
    assert "O_EXCL" in source
    assert "os.fsync" in source
    assert "tmux" in source
    assert "--descriptor" in source
    assert "--authorization-context" in source
