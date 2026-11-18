"""R29 window delivery (CPU, pure): tools/r28/r29_input.py.

The modes and the frozen rule; the host delivery hands on the audited object itself; the device delivery hands on an
equal copy on the stream's device and leaves the audited tensor and its seal untouched; the stream of a delivery is the
audited stream of r28_pipeline; the formal mode is only accepted for the arm the acceptance accepted; the loader worker
override is a dev-round measurement; the steady segments of the measurement.  The prepared mode (pinned, non-blocking)
and the bitwise acceptance against the stored receipts need CUDA: they are the accept-R29 stage.
"""
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "r28"))
import r28_common as K  # noqa: E402
import r28_pipeline as PL  # noqa: E402
import r29_control as RC  # noqa: E402
import r29_input as RI  # noqa: E402

PER_EPOCH = 4


def windows(data, completed, *, limit=None):
    first = completed % PER_EPOCH
    for k in range(PER_EPOCH - first if limit is None else limit):
        u = completed + k + 1
        x = torch.full((4, 3, 2, 2), float(u) / 100)
        t = [{"boxes": torch.full((1, 4), 0.5), "labels": torch.tensor([u]), "image_id": f"img{u}"}]
        yield x, t, {"update": u, "epoch": completed // PER_EPOCH + 1, "window": first + k}


def items(mode="overlap", start=2, stop=7):
    with PL.Stream(mode, windows, None, start, stop, PER_EPOCH, lambda inp: None) as st:
        return list(st)


def contract(rule=None, **fu):
    return {"contract_sha256": "new", "authorization_basis": K.R29_BASIS, "input_pipeline": {"r29_window_delivery": RI.RULE if rule is None else rule},
            "followup": {"authorization_basis": K.R29_BASIS, "arm": None, "arms": ["QP", "QC"], **fu}}


def test_modes_and_frozen_rule():
    assert RI.MODES == ("host", "device", "prepared") and RI.FORMAL in RI.MODES and RI.VERSION == "r29-window-delivery-v1"
    assert RI.RULE["mode"] == RI.FORMAL and RI.RULE["reference_mode"] == "host"
    assert (RI.RULE["workers"], RI.RULE["ready_limit"], RI.RULE["input_pipeline_version"]) == (PL.WORKERS, PL.READY_LIMIT, PL.VERSION) == (2, 1, "r28-input-pipeline-v1")
    assert RC.ACCEPT["workers"] == RI.RULE["workers"] and RC.ACCEPT["ready_limit"] == RI.RULE["ready_limit"]
    assert RI.SCOPE != "formal" and RI.SCOPE != RC.SCOPE
    with pytest.raises(Exception, match="window delivery mode"):
        RI.Delivery("gpu", "cpu")
    with pytest.raises(Exception, match="CUDA"):
        RI.Delivery("prepared", "cpu")


def test_host_delivery_hands_on_the_audited_object():
    d = RI.Delivery("host", "cpu")
    for i in items():
        out, m = d.take(i)
        assert out is i.x and set(m) == {"deliver_s"} and m["deliver_s"] >= 0
    assert d.account() == {**d.binding(), "taken": d.taken, "deliver_seconds": d.deliver_s} and d.binding() == {"version": RI.VERSION, "mode": "host", "device": "cpu"}


def test_device_delivery_is_an_equal_copy_and_keeps_the_seal():
    d = RI.Delivery("device", "cpu")
    for i in items():
        before = i.x.clone()
        out, m = d.take(i)
        assert torch.equal(out, before) and out.dtype == i.x.dtype and tuple(out.shape) == tuple(i.x.shape) and "prepare_s" not in m
        i.verify()                                   # the audited tensor and its seal are untouched by the delivery
        assert torch.equal(i.x, before)
    assert d.taken == 5 and d.account()["mode"] == "device"


def test_the_stream_of_a_delivery_is_the_audited_stream(monkeypatch):
    fd = types.ModuleType("formal_data")
    fd.windows, fd.WINDOWS_PER_EPOCH = windows, PER_EPOCH
    fd.input_record = lambda *a: None
    monkeypatch.setitem(sys.modules, "formal_data", fd)
    seen = []
    for mode in ("host", "device"):
        d = RI.Delivery(mode, "cpu")
        with d.stream("overlap", None, 2, 9, lambda inp: None) as st:
            assert isinstance(st, PL.Stream) and st.mode == "overlap"
            rows = [(i.inp["update"], i.inp["epoch"], i.inp["window"], d.take(i)[0].clone()) for i in st]
            acc = st.account()
        assert [r[:3] for r in rows] == [(3, 1, 2), (4, 1, 3), (5, 2, 0), (6, 2, 1), (7, 2, 2), (8, 2, 3), (9, 3, 0)]
        assert acc["max_prepared_ahead"] == 1 and acc["taken"] == 7 == d.taken and acc["workers"] == 2
        seen.append(rows)
        assert not hasattr(fd.input_record, "__wrapped__")   # the clock of the stream was removed
    assert all(torch.equal(a[3], b[3]) for a, b in zip(*seen))
    refused = []
    d = RI.Delivery("host", "cpu")
    with pytest.raises(RuntimeError, match="reference"):
        with d.stream("overlap", None, 2, 9, lambda inp: (_ for _ in ()).throw(RuntimeError("reference mismatch")) if inp["update"] == 4 else None) as st:
            for i in st:
                refused.append(i.inp["update"])
    assert refused == [3]                             # the failing window is never delivered


def test_formal_mode_needs_the_acceptance_of_that_arm(monkeypatch):
    c = contract()
    closed = {"window_delivery": {"QP": {"mode": RI.FORMAL, "equal_to_the_host_receipts": True}}}
    monkeypatch.setattr(K, "stage_closed", lambda c_, name, arm=None: closed)
    monkeypatch.delenv("R29_DELIVERY", raising=False)
    assert RI.formal_mode(c, "QP") == RI.FORMAL
    if RI.FORMAL != "host":
        with pytest.raises(Exception, match="did not accept"):
            RI.formal_mode(c, "QC")
        monkeypatch.setattr(K, "stage_closed", lambda c_, name, arm=None: {"window_delivery": {"QP": {"mode": RI.FORMAL, "equal_to_the_host_receipts": False}}})
        with pytest.raises(Exception, match="did not accept"):
            RI.formal_mode(c, "QP")
    with pytest.raises(Exception, match="arms QP and QC"):
        RI.formal_mode(c, "FG")
    with pytest.raises(Exception, match="rule of the contract"):
        RI.formal_mode(contract(rule={**RI.RULE, "workers": 4}), "QP")
    with pytest.raises(Exception, match="arms QP and QC"):
        RI.formal_mode({**c, "authorization_basis": K.GEO_BASIS}, "QP")
    monkeypatch.setenv("R29_DELIVERY", "host")
    assert RI.formal_mode(c, "QC") == "host"          # the reference delivery needs no acceptance of another mode
    for bad in ("device", "prepared", ""):
        monkeypatch.setenv("R29_DELIVERY", bad)
        with pytest.raises(Exception, match="R29_DELIVERY"):
            RI.formal_mode(c, "QP")


def test_loader_worker_override_is_a_dev_measurement(monkeypatch):
    fd = types.ModuleType("formal_data")
    calls = []
    fd.training_loader = lambda dataset, *, epoch, completed_windows=0, workers=2, window_limit=None: calls.append(workers) or workers
    monkeypatch.setitem(sys.modules, "formal_data", fd)
    original = fd.training_loader
    with RI.loader_workers(2):
        assert fd.training_loader is original and fd.training_loader(None, epoch=61, workers=2) == 2
    monkeypatch.setattr(K, "DEV", False)
    with pytest.raises(Exception, match="dev-round measurement"):
        with RI.loader_workers(4):
            pass
    assert fd.training_loader is original
    monkeypatch.setattr(K, "DEV", True)
    with RI.loader_workers(4):
        assert fd.training_loader is not original and fd.training_loader(None, epoch=61, completed_windows=3, workers=2, window_limit=5) == 4
    assert fd.training_loader is original and calls == [2, 4]
    with pytest.raises(Exception, match="loader workers"):
        with RI.loader_workers(3):
            pass


def test_plan_and_steady_segments():
    assert RI.parse_plan("QP:host:2, QC:device:4 QP:prepared:8") == [("QP", "host", 2), ("QC", "device", 4), ("QP", "prepared", 8)]
    for bad in ("FG:host:2", "QP:gpu:2", "QP:host:3", ""):
        with pytest.raises(Exception):
            RI.parse_plan(bad)
    rows = [{"update": 100 + k, "cycle_s": 2.0 + (5.0 if k in (0, 9) else 0.0), "wait_s": 0.1 if k == 0 else 0.0, "train_call_s": 1.5, "train_s": 1.0,
             "deliver_s": 0.1, "prepared": 1, "input": {"next_s": 0.7, "audit_s": 0.6, "fetch_s": 0.1, "reference_check_s": 0.0},
             **({"save_s": 3.0} if k == 9 else {})} for k in range(10)]
    s = RI.segments(rows, 4)
    assert (s["steady_updates"], s["steady_first_update"], s["steady_last_update"]) == (5, 104, 108) and s["prepared_windows_seen"] == [1]
    st = s["steady"]
    assert st["cycle_s"]["median"] == 2.0 and st["gpu_update_s"]["median"] == 1.0 and st["deliver_s"]["median"] == 0.1
    assert abs(st["engine_checks_before_its_timer_s"]["median"] - 0.4) < 1e-12 and abs(st["consumer_other_s"]["median"] - 0.5) < 1e-12
    assert st["producer_prepare_s"] is None and st["producer_audit_s"]["n"] == 5 and s["last_update"]["save_s"] == 3.0
    assert RI.stats([3.0, None, 1.0, 2.0]) == {"n": 3, "median": 2.0, "mean": 2.0, "p10": 1.0, "p90": 3.0, "min": 1.0, "max": 3.0}


def test_release_delta_against_the_host_release():
    assert set(RC.HOST_ADDED) <= set(RC.ALLOWED_ADDED) and set(RC.HOST_CHANGED) <= set(RC.ALLOWED_CHANGED) | set(RC.ALLOWED_ADDED)
    assert not any(p.startswith(("src/", "vendor/", "configs/")) for p in RC.HOST_CHANGED + RC.HOST_ADDED)
    assert "tools/r28/r28_pipeline.py" in RC.CORE_IDENTICAL and "src/sparse_rtdetr/protocol_r18/engine.py" in RC.CORE_IDENTICAL
    assert "tests/test_r29_input.py" in RC.TESTS and RC.HOST_ROUND.name == "r29-query-unzoom-20261116-001"
    assert RC.PIN_HOST["contract"] == "17a2ca0cf3eb87870006b7879b49b3f188b23324e41b11faacdba8241e7550f8"
    assert RC.PIN_HOST["commit"] == "efc113dcd8324b32c7a862e3db4ff28985c813a6" and sorted(RC.PIN_HOST["steps"]) == ["QC-full", "QC-resumed", "QP-full", "QP-resumed"]
    assert {"recipe", "parent", "data", "identity", "identity_sha256", "r29_read_definition", "r29_rules", "evaluation_policy", "workload"}.isdisjoint(RC.HOST_CONTRACT_DIFFERENT)
