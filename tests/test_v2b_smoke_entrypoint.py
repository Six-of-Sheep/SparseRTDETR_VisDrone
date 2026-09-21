from __future__ import annotations

import json
import importlib.util
import os
import subprocess
import sys
from pathlib import Path


ENTRYPOINT = Path(__file__).resolve().parents[1] / "tools" / "run_v2b_rev1_smoke_entrypoint.py"
GPU_SMOKE = Path(__file__).resolve().parents[1] / "tools" / "run_v2b_rev1_gpu_smoke.py"


def _load_gpu_smoke_module():
    spec = importlib.util.spec_from_file_location("v2b_gpu_smoke_test_module", GPU_SMOKE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _invocation(
    tmp_path: Path, *, include_pythonpath: bool = True, pythonpath: str | None = None
) -> tuple[Path, Path]:
    (tmp_path / "src").mkdir()
    observed = tmp_path / "observed.json"
    controller = tmp_path / "controller.py"
    observed_literal = repr(str(observed))
    controller.write_text(
        "import json, os, sys\n"
        "payload = {'argv': sys.argv[1:], 'env': {k: os.environ.get(k) for k in "
        "['PYTHONPATH','PYTHONHASHSEED','CUBLAS_WORKSPACE_CONFIG','PATH','HOME']}}\n"
        f"json.dump(payload, open({observed_literal}, 'w', encoding='utf-8'), sort_keys=True)\n",
        encoding="utf-8",
    )
    env = {
        "PYTHONHASHSEED": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PATH": os.environ.get("PATH", os.defpath),
        "HOME": os.environ.get("HOME", str(tmp_path)),
    }
    if include_pythonpath:
        env["PYTHONPATH"] = pythonpath or str((tmp_path / "src").resolve())
    value = {
        "schema_version": 2,
        "repository_root": str(tmp_path.resolve()),
        "working_directory": str(tmp_path.resolve()),
        "controller_python": sys.executable,
        "controller_launcher": str(controller.resolve()),
        "controller_argv": [sys.executable, str(controller.resolve()), "ok"],
        "controller_env": env,
    }
    invocation = tmp_path / "invocation.json"
    invocation.write_bytes(
        (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    )
    return invocation, observed


def _run(
    invocation: Path, *, parent_pythonpath: str | None = None
) -> subprocess.CompletedProcess[str]:
    env = {
        "PATH": os.environ.get("PATH", os.defpath),
        "HOME": os.environ.get("HOME", str(invocation.parent)),
    }
    if parent_pythonpath is not None:
        env["PYTHONPATH"] = parent_pythonpath
    return subprocess.run(
        [sys.executable, str(ENTRYPOINT), "--invocation", str(invocation)],
        cwd=invocation.parent,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_clean_parent_uses_sealed_controller_environment(tmp_path):
    invocation, observed = _invocation(tmp_path)
    result = _run(invocation)
    assert result.returncode == 0, result.stderr
    payload = json.loads(observed.read_text(encoding="utf-8"))
    assert payload["env"]["PYTHONPATH"] == str((tmp_path / "src").resolve())
    assert payload["env"]["PYTHONHASHSEED"] == "0"
    assert payload["env"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_missing_controller_pythonpath_fails_closed(tmp_path):
    invocation, _ = _invocation(tmp_path, include_pythonpath=False)
    result = _run(invocation, parent_pythonpath=str((tmp_path / "src").resolve()))
    assert result.returncode != 0
    assert "CONTROLLER_ENVIRONMENT_MISSING:PYTHONPATH" in result.stderr


def test_wrong_controller_pythonpath_fails_closed(tmp_path):
    invocation, _ = _invocation(tmp_path, pythonpath=str(tmp_path / "wrong"))
    result = _run(invocation)
    assert result.returncode != 0
    assert "CONTROLLER_PYTHONPATH_MISMATCH" in result.stderr


def test_parent_environment_cannot_substitute_for_missing_bound_value(tmp_path):
    invocation, _ = _invocation(tmp_path, include_pythonpath=False)
    result = _run(invocation, parent_pythonpath=str((tmp_path / "src").resolve()))
    assert result.returncode != 0
    assert "CONTROLLER_ENVIRONMENT_MISSING:PYTHONPATH" in result.stderr


def test_gpu_admission_evidence_does_not_serialize_live_capability(tmp_path):
    module = _load_gpu_smoke_module()

    class Monitor:
        def admission(self):
            return object()

    evidence = module._gpu_admission_evidence(
        tmp_path, Monitor(), {"policy_sha256": "policy", "gpu_uuid": "gpu"}
    )
    json.dumps(evidence, sort_keys=True)
    assert evidence == {
        "capability_type": "object",
        "gpu_uuid": "gpu",
        "monitor_root": str(tmp_path / "quiescence-native-hardware"),
        "policy_sha256": "policy",
        "status": "PASS",
    }
