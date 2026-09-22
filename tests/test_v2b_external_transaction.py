from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SEALER = ROOT / "tools" / "seal_v2b_rev1_transaction.py"


def test_external_sealer_refuses_dirty_or_unpushed_git(tmp_path: Path) -> None:
    result = subprocess.run(
        ["python", str(SEALER), "--repo", str(ROOT), "--root", str(tmp_path / "SparseRTDETR_VisDrone_v2b_rev1_gpu_smoke_bridge999_20260922"), "--plan-id", "x", "--authorization-id", "x", "--session", "x", "--python", "python", "--controller", ROOT / "tools/run_v2b_rev1_smoke_controller.py", "--entrypoint", ROOT / "tools/run_v2b_rev1_smoke_entrypoint.py", "--runner", ROOT / "tools/run_v2b_rev1_gpu_smoke_lifecycle.py", "--source", ROOT / "contracts/v2b/rev001/execution_source_r32.json", "--source-sha", "0" * 64, "--contract", ROOT / "contracts/v2b/rev001/execution_contract_r32.json", "--contract-sha", "0" * 64, "--contract-id", "v2b-execution-contract-032", "--revision-verifier", ROOT / "tools/verify_v2b_external_transaction_r32.py", "--checkpoint", ROOT / "missing.pt", "--checkpoint-sha", "0" * 64, "--binding-sha", "0" * 64, "--manifest", ROOT / "missing.json", "--manifest-sha", "0" * 64, "--policy-authority", ROOT / "contracts/v2b/rev001/runtime_policy_authority_r1.json", "--policy-id", "hardware-policy:t8b:paired-smoke-r2", "--policy-sha", "0" * 64, "--training-contract-sha", "0" * 64, "--gpu-uuid", "GPU-00000000-0000-0000-0000-000000000000"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Git must be clean" in result.stderr or "Git must be clean" in result.stdout


def test_transaction_artifact_identity_is_not_raw_sha_placeholder() -> None:
    source = json.loads((ROOT / "contracts/v2b/rev001/execution_source_r32.json").read_bytes())
    body = {key: value for key, value in source.items() if key != "execution_source_sha256"}
    canonical = (json.dumps(body, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()
    assert source["execution_source_sha256"] == hashlib.sha256(canonical).hexdigest()


@pytest.mark.parametrize("field", ["authorization_mode", "runtime_locator"])
def test_external_contract_declares_authority(field: str) -> None:
    contract = json.loads((ROOT / "contracts/v2b/rev001/execution_contract_r32.json").read_bytes())
    assert field in contract
    assert contract["authorization_mode"] == "external_transaction"
    assert contract["runtime_locator"]["external_transaction_parent"]
