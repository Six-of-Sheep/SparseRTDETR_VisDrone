"""CPU tests of the GPU-check entry: arm allocation, runtime constraints, exception accounting.

Neither torch nor CUDA is imported here, and the module-level imports of ``pdra_gpu_check``
are numpy/core only, so these tests run in the numpy-only check environment as well as in
the frozen torch interpreter.

Run from the repository root:  python -m pytest tools/pdra/test_pdra_gpu_check.py -q
"""

from __future__ import annotations

import json
import os
import sys
import time

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _path in (_HERE, _REPO):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import pdra_gpu_check as G


def test_arm_allocation_sums_to_frozen_updates():
    assert G.CHECK_UPDATES == 50
    assert sum(G.ARM_ALLOCATION.values()) == G.CHECK_UPDATES
    assert G.CAPS["comparator_updates"] == G.CHECK_UPDATES
    assert G.CAPS["forward_calls"] == 20
    assert set(G.ARM_ALLOCATION) == {"pdra_difference_strips", "c_whole_box_pairwise", "b_single_box_quality"}
    assert all(value > 0 for value in G.ARM_ALLOCATION.values())


def test_acceptance_scope_forbids_formal_learning():
    scope = G.ACCEPTANCE_SCOPE
    assert scope["kind"] == "engineering_wiring_check_only"
    assert scope["authorizes_formal_learning"] is False
    assert scope["authorizes_stage1_training"] is False
    assert scope["authorizes_ap_ar_scoring"] is False
    assert scope["authorizes_548_image_pass"] is False
    assert scope["settles_capacity_control"] is False


def test_wall_constraint_breach_raises():
    with pytest.raises(G.RuntimeConstraintError) as excinfo:
        G.enforce_runtime(started=time.monotonic() - 10.0, label="test",
                          caps={"wall_seconds": 1, "gpu_peak_bytes": 10 ** 18, "output_bytes": 10 ** 18})
    assert excinfo.value.observation["label"] == "test"
    assert excinfo.value.observation["elapsed_seconds"] >= 10.0


def test_runtime_within_caps_returns_observation():
    observation = G.enforce_runtime(started=time.monotonic(), label="test",
                                    caps={"wall_seconds": 3600, "gpu_peak_bytes": 10 ** 18, "output_bytes": 10 ** 18})
    assert observation["label"] == "test"
    assert observation["elapsed_seconds"] >= 0.0
    assert observation["cuda_peak_reserved_bytes"] is None      # CPU path: no torch module passed
    assert observation["output_bytes"] is None                  # no output directory passed


def test_output_bytes_constraint(tmp_path):
    (tmp_path / "file.json").write_bytes(b"x" * 64)
    caps = {"wall_seconds": 3600, "gpu_peak_bytes": 10 ** 18, "output_bytes": 8}
    with pytest.raises(G.RuntimeConstraintError) as excinfo:
        G.enforce_runtime(started=time.monotonic(), label="test", caps=caps, out=tmp_path)
    assert excinfo.value.observation["output_bytes"] == 64
    caps["output_bytes"] = 1024
    observation = G.enforce_runtime(started=time.monotonic(), label="test", caps=caps, out=tmp_path)
    assert observation["output_bytes"] == 64


def test_accounting_counts_and_detached_snapshot():
    accounting = G.Accounting(task="t", mode="run")
    accounting.phase("forward")
    accounting.complete("restore")
    accounting.forwards(5)
    accounting.update("pdra_difference_strips")
    accounting.observe({"label": "x", "cuda_peak_reserved_bytes": 7, "cuda_peak_allocated_bytes": 3})
    accounting.observe({"label": "y", "cuda_peak_reserved_bytes": 5, "cuda_peak_allocated_bytes": 4})
    snapshot = accounting.snapshot()
    assert snapshot["phase"] == "forward"
    assert snapshot["phases_completed"] == ["restore"]
    assert snapshot["executed_forward_calls"] == 5
    assert snapshot["executed_updates"] == {"pdra_difference_strips": 1, "c_whole_box_pairwise": 0,
                                            "b_single_box_quality": 0}
    assert snapshot["budget_observations"] == 2
    assert snapshot["cuda_peak_reserved_bytes"] == 7 and snapshot["cuda_peak_allocated_bytes"] == 4
    snapshot["executed_forward_calls"] = 999
    assert accounting.snapshot()["executed_forward_calls"] == 5


def test_exception_accounting_writes_stop(tmp_path):
    parent = tmp_path / "bogus-parent.json"
    parent.write_bytes(b"{}")
    out = tmp_path / G.TASK / "dryrun-test"
    returncode = G.main(["dry-run", "--parent", str(parent), "--out", str(out), "--source-commit", "TEST"])
    assert returncode == 2
    assert (out / "START.json").is_file()
    assert not (out / "RESULT.json").exists() and not (out / "CLOSED.json").exists()
    stop = json.loads((out / "STOP.json").read_bytes())
    assert stop["status"] == "FAIL"
    assert stop["exception"]["type"] == "ValueError"
    assert "parent contract SHA differs" in stop["exception"]["message"]
    assert stop["automatic_retry"] is False
    assert stop["accounting"]["phase"] == "inputs"
    assert stop["accounting"]["executed_forward_calls"] == 0
    assert all(value == 0 for value in stop["accounting"]["executed_updates"].values())
    assert stop["accounting"]["budget_observations"] == 1
    assert "Traceback" in stop["traceback"]
