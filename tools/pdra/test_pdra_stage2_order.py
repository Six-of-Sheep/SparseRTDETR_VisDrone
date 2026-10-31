"""CPU regression of the rev-8 stage-2 entry order: audited restore before any CUDA touch.

The rev-7 stage-2 entry loaded the three arm weights onto CUDA *before* running the
audited engine restore; ``deterministic_device_setup`` refuses to run once CUDA is
initialized ("CUDA was initialized before deterministic setup"), so the first real
stage-2 execution stopped in the restore step.  This test executes the production
``run_stage2`` control flow end to end with the external boundaries faked (torch, the
engine restore, the forward pass, the arms, the project imports) and asserts the actual
call order:

  1. the four stage-1 SHA/commit pins are verified first, on the CPU;
  2. the restore stub -- which mirrors the deterministic set-up precondition -- runs
     before any ``arm.to("cuda:0")``;
  3. every arm is loaded and moved to CUDA only after the restore completed, and the
     detector baseline capture follows the arm construction (the stage-1 invariant that
     keeps the CPU-RNG consumption inside the captured baseline) and precedes inference.

A negative control proves the harness would have caught the rev-7 order, and a pin
mismatch and a learning-spec drift both stop before the restore.  Neither torch nor CUDA
is required (both faked), so the test runs in the numpy-only CI environment as well as
in the frozen interpreter.

Run from the repository root:  python -m pytest tools/pdra/test_pdra_stage2_order.py -q
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import types

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _path in (_HERE, _REPO):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import pdra_stage1 as S1

ARM_KEYS = ("pdra_difference_strips", "c_whole_box_pairwise", "b_single_box_quality")


def _sha_file(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Events:
    def __init__(self):
        self.events = []

    def record(self, kind, detail=""):
        self.events.append((kind, detail))


def _module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _build_fixture(tmp_path, *, break_pin=False):
    """Real on-disk stage-1 fixture: RESULT / CLOSED / LEARNING_FROZEN and three arm files."""
    attempt = tmp_path / "r18-pdra-stage1-test" / "attempt-test"
    arms_dir = attempt / "arms"
    arms_dir.mkdir(parents=True)
    arms_record = {}
    for key in ARM_KEYS:
        arm_file = arms_dir / f"{key}.pt"
        arm_file.write_bytes(f"{key}-weights".encode())
        arms_record[key] = {"path": str(arm_file), "bytes": arm_file.stat().st_size,
                            "sha256": _sha_file(arm_file)}
    result = {"status": "PASS", "source_commit": "b" * 40,
              "criteria_sha256": S1.SCORE.criteria_sha256(), "arms": arms_record}
    result_path = attempt / "RESULT.json"
    result_path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    result_sha = _sha_file(result_path)
    closed = {"result": {"path": str(result_path), "bytes": result_path.stat().st_size,
                         "sha256": result_sha},
              "returncode": 0, "automatic_retry": False}
    closed_path = attempt / "CLOSED.json"
    closed_path.write_text(json.dumps(closed, sort_keys=True), encoding="utf-8")
    learning_path = attempt / "LEARNING_FROZEN.json"
    learning_path.write_text(json.dumps({"learning_spec_sha256": "test-spec-sha"}),
                             encoding="utf-8")
    pins = {"stage1_result_path": result_path,
            "stage1_result_sha256": ("0" * 64 if break_pin else result_sha),
            "stage1_closed_sha256": _sha_file(closed_path),
            "stage1_learning_frozen_sha256": _sha_file(learning_path),
            "stage1_source_commit": "b" * 40}
    return pins


def _restore_stub(events):
    def restore(**kwargs):
        if any(kind == "arm_to" for kind, _ in events.events):
            raise ValueError("CUDA was initialized before deterministic setup")
        events.record("restore")
        return {"engine": object(), "binding": {}, "restored": {}, "model": object(),
                "runtime": {}}
    return restore


def _install_fakes(monkeypatch, events):
    class FakeArm:
        def __init__(self, key):
            self.key = key
            self.loaded = None

        def load_state_dict(self, state):
            self.loaded = state
            events.record("arm_load", self.key)

        def to(self, device):
            events.record("arm_to", f"{self.key}:{device}")
            return self

        def eval(self):
            events.record("arm_eval", self.key)
            return self

    class TapContext:
        def __init__(self, model):
            self.model = model

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def capture_stub(engine, binding):
        events.record("capture")
        return {"state": "before"}

    cuda = types.SimpleNamespace(is_initialized=lambda: False,
                                 max_memory_reserved=lambda: 0,
                                 max_memory_allocated=lambda: 0,
                                 reset_peak_memory_stats=lambda: None)
    checkpoint_mod = _module("sparse_rtdetr.protocol_r18.checkpoint",
                             capture=capture_stub,
                             exact_equal=lambda left, right: True)

    class FakeDevelopment548:
        def __init__(self, spec):
            self.records = []

    protocol = _module("sparse_rtdetr.protocol_r18", checkpoint=checkpoint_mod)
    evaluation_data = _module("sparse_rtdetr.r0.evaluation_data",
                              Development548=FakeDevelopment548)
    r0 = _module("sparse_rtdetr.r0", evaluation_data=evaluation_data)
    rtdetr = _module("sparse_rtdetr", protocol_r18=protocol, r0=r0)
    fakes = {
        "torch": _module("torch", load=lambda *a, **k: {"fake_state": True}, cuda=cuda),
        "pdra_features": _module("pdra_features"),
        "pdra_learn": _module(
            "pdra_learn",
            build_arms=lambda *a, **k: {"arms": {key: FakeArm(key) for key in ARM_KEYS}},
            learning_spec_sha=lambda *a, **k: "test-spec-sha"),
        "pdra_tap": _module(
            "pdra_tap", restore_frozen_seed0_engine=_restore_stub(events),
            EncoderTap=TapContext,
            five_view_forward=lambda data, rows, model, tap=None, device="cuda:0":
                {"forward_calls": 5}),
        "slice_core": _module("slice_core", merge_candidates=lambda pool: {}),
        "sparse_rtdetr": rtdetr,
        "sparse_rtdetr.protocol_r18": protocol,
        "sparse_rtdetr.protocol_r18.checkpoint": checkpoint_mod,
        "sparse_rtdetr.r0": r0,
        "sparse_rtdetr.r0.evaluation_data": evaluation_data,
    }
    for name, module in fakes.items():
        monkeypatch.setitem(sys.modules, name, module)
    real_read = S1.read

    def read_stub(path):
        if str(path).endswith("STAGE0_SUMMARY.json"):
            return {"scale": {"all": {"pairs": 0}}}
        return real_read(path)

    monkeypatch.setattr(S1, "read", read_stub)


def _call_run_stage2(tmp_path, pins, *, attempt_name):
    out = tmp_path / "r18-pdra-stage2-test" / attempt_name
    out.mkdir(parents=True)
    accounting = S1.G.Accounting("r18-pdra-stage2-test", "run-stage2")
    result = S1.run_stage2(
        {"training_root": str(tmp_path), "training": {},
         "binding_path": str(tmp_path / "binding.json"),
         "checkpoint": {"path": str(tmp_path / "checkpoint.pt"), "sha256": "0" * 64},
         "gpu_uuid": "GPU-test", "data_spec": {}},
        out, {"status": "PASS", "feature_binding": {"verified": True}},
        pins["stage1_result_path"].parent / "arms", "c" * 40,
        budget_started=time.monotonic(), accounting=accounting, **pins)
    return result, accounting


def test_stage2_restore_precedes_every_arm_cuda_move(tmp_path, monkeypatch):
    events = _Events()
    pins = _build_fixture(tmp_path)
    _install_fakes(monkeypatch, events)
    result, accounting = _call_run_stage2(tmp_path, pins, attempt_name="attempt-run")
    kinds = [kind for kind, _ in events.events]
    assert kinds.count("restore") == 1
    assert kinds.index("restore") < kinds.index("arm_load")
    assert kinds.index("restore") < kinds.index("arm_to")
    assert kinds.index("arm_to") < kinds.index("capture")
    assert kinds.count("capture") == 2          # baseline + final detector check
    moved = [detail for kind, detail in events.events if kind == "arm_to"]
    assert moved == [f"{key}:cuda:0" for key in ARM_KEYS]
    assert result["status"] == "PASS"
    assert result["forward_calls"] == 137 * 5
    assert result["detector_updates"] == 0
    assert result["stage1_binding"]["verified"] is True
    for key in ARM_KEYS:
        assert result["arms_provenance"][key]["stage1_record_sha256_match"] is True
    snapshot = accounting.snapshot()
    assert snapshot["phases_completed"] == ["stage1_binding", "restore", "arms",
                                            "inference", "publish"]
    assert snapshot["executed_forward_calls"] == 137 * 5
    assert all(count == 0 for count in snapshot["executed_updates"].values())


def test_stage2_pin_mismatch_stops_before_restore(tmp_path, monkeypatch):
    events = _Events()
    pins = _build_fixture(tmp_path, break_pin=True)
    _install_fakes(monkeypatch, events)
    with pytest.raises(ValueError, match="stage-1 RESULT differs from the pinned SHA"):
        _call_run_stage2(tmp_path, pins, attempt_name="attempt-pin")
    assert events.events == []


def test_stage2_learning_spec_drift_is_rejected(tmp_path, monkeypatch):
    pins = _build_fixture(tmp_path)
    monkeypatch.setitem(sys.modules, "pdra_learn",
                        _module("pdra_learn",
                                learning_spec_sha=lambda *a, **k: "other-spec-sha"))
    with pytest.raises(ValueError, match="incompatible with the current learning code"):
        S1.verify_stage1_binding(**pins)


def test_rev7_order_would_have_failed_the_restore_precondition():
    events = _Events()
    events.record("arm_to", "pdra_difference_strips:cuda:0")
    stub = _restore_stub(events)
    with pytest.raises(ValueError, match="CUDA was initialized before deterministic setup"):
        stub()
