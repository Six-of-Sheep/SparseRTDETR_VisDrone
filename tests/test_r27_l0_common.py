"""R27-002 L0 runner common layer: shared-field comparison and fixed constants (pure CPU, no model, no torch)."""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "r27_l0"))
import l0_common as K  # noqa: E402


def row(update=18241):
    return {"input": {"update": update, "epoch": 61, "window": 0, "image_ids_sha256": "a" * 64, "gt_ids_sha256": "b" * 64,
                      "augmentation_seeds_sha256": "c" * 64, "augmentation_trace_sha256": "d" * 64,
                      "augmented_samples_sha256": "e" * 64, "input_tensor_sha256": "f" * 64, "target_sha256": "0" * 64,
                      "augmentation_strong": True},
            "gt_count": 916, "loss": 10.27, "lr": 5.499938797812981e-05, "gradient_norm_before_clip": 127.1,
            "microbatch_losses": [{"loss_vfl": 0.44}], "update_seconds": 2.3, "ema_updates": update}


def test_shared_fields_are_the_declared_list():
    assert K.SHARED_FIELDS == tuple(f"input.{k}" for k in K.INPUT_FIELDS) + ("lr", "gt_count", "ema_updates")
    assert len(K.INPUT_FIELDS) == 11 and len(K.SHARED_FIELDS) == 14
    s = K.shared(row())
    assert tuple(s) == K.SHARED_FIELDS
    assert not any(k in s for k in ("loss", "gradient_norm_before_clip", "update_seconds", "microbatch_losses", "monitor"))


def test_compare_shared_accepts_identical_rows_and_unshared_changes():
    a, b = row(), row()
    assert K.compare_shared(a, b) == K.shared(a)
    b.update(loss=9.0, gradient_norm_before_clip=1.0, update_seconds=99.0, microbatch_losses=[], monitor={"x": 1})
    assert K.compare_shared(a, b) == K.shared(a)


@pytest.mark.parametrize("field", K.SHARED_FIELDS)
def test_compare_shared_rejects_each_changed_shared_field(field):
    a, b = row(), row()
    target, key = (b["input"], field[len("input."):]) if field.startswith("input.") else (b, field)
    old = target[key]
    target[key] = (not old) if isinstance(old, bool) else (old + 1 if isinstance(old, int) else
                                                           (old * 1.0000001 if isinstance(old, float) else "9" * 64))
    with pytest.raises(ValueError, match=f"shared field differs: {field.replace('.', '[.]')} "):
        K.compare_shared(a, b)


def test_compare_shared_names_the_first_differing_field_and_checks_types():
    a, b = row(), row()
    b["input"]["target_sha256"] = "1" * 64
    b["lr"] = 0.0
    with pytest.raises(ValueError, match="input.target_sha256"):
        K.compare_shared(a, b)
    c = row()
    c["input"]["augmentation_strong"] = 1  # equal to True by value, different type
    with pytest.raises(ValueError, match="input.augmentation_strong"):
        K.compare_shared(a, c)


def test_shared_rejects_missing_or_extra_input_keys():
    a = row()
    del a["input"]["target_sha256"]
    with pytest.raises(ValueError):
        K.shared(a)
    b = row()
    b["input"]["arm"] = "L0"
    with pytest.raises(ValueError):
        K.shared(b)
    c = copy.deepcopy(row())
    del c["ema_updates"]
    with pytest.raises(KeyError):
        K.shared(c)


def test_gate_rotations_and_schedule_constants():
    assert K.GATE_D == (("600", "APS", 0.50), ("600", "AP75_small", 0.30), ("600", "AP", -0.10), ("300", "AP", -0.20),
                        ("300", "APS", -0.20), ("600", "APM", -0.50), ("600", "APL", -0.50))
    assert K.ROTATIONS == (("A0", "L0", "P", "S15"), ("P", "S15", "A0", "L0"), ("L0", "A0", "S15", "P"))
    assert all(sorted(r) == ["A0", "L0", "P", "S15"] for r in K.ROTATIONS)
    assert K.SCHEMA == "r27-l0-state-v1" and K.ARM == "L0" and K.ENDPOINT == 36480
    assert K.SAVE_EPOCHS == (60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120) and len(K.SAVE_EPOCHS) == 13
    assert K.FORMAL_SAVE_EPOCHS == (60, 75, 90, 105, 120) and set(K.FORMAL_SAVE_EPOCHS) <= set(K.SAVE_EPOCHS)
    assert (K.EVAL_BATCH, K.EVAL_WARMUP, K.EVAL_BATCHES, K.BUDGETS) == (4, 4, 137, (600, 300))
    assert (K.COST_BATCH, K.COST_WARMUP, K.COST_PASSES, K.COST_BATCHES_PER_PASS) == (4, 2, 3, 16)
    assert K.PARENT_SHA == "a2b6412e3c85de7592829f24aff09e355fdd11cb53798bc498bd04b882e1bf8a"
    assert K.endpoint_path().name == "checkpoint-epoch-120.pth" and K.endpoint_path().parent.name == "formal-L0"


def test_workload_totals():
    w = K.WORKLOAD
    assert (w["cpu_fixture"]["updates"], w["cpu_fixture"]["forwards"]) == (3, 6)
    assert (w["cuda_zero_step"]["updates"], w["cuda_zero_step"]["forwards"]) == (0, 4)
    assert (w["cuda_accept"]["updates"], w["cuda_accept"]["forwards"]) == (3, 6)
    assert (w["formal"]["updates"], w["formal"]["forwards"]) == (18240, 36480)
    assert w["evaluation"]["forwards"] == 141 and w["evaluation"]["score_budget_calls"] == 2
    assert w["evaluation"]["per_class_APL_COCOeval_calls"] == 1
    assert w["cost4"]["forwards_per_endpoint"] == 50 and w["cost4"]["forwards"] == 200
    assert w["normal_gpu_total"] == {"updates": 18243, "forwards": 36831}


def test_cells_reproduce_a_difference_in_pp():
    p = {"600": {"APS": 0.2332513847416923}}
    a0 = {"600": {"APS": 0.2217618575337924}}
    out = K.cells(p, a0, (("600", "APS", 0.50),))
    assert out["top600_APS"]["difference_pp"] == 1.148952720789989 and out["top600_APS"]["passed"]
