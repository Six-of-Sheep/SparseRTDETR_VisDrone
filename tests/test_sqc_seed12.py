"""SQC-002 adapter unit tests (CPU; no data, no checkpoint, no CUDA). Science modules are byte-identical to SQC-001."""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "scale_baseline"))

import s15_cross_scale as X
import sqc_run as B
import sqc_seed12_run as A
from sparse_rtdetr.research import scale_baseline as S
from sparse_rtdetr.research import shared_query_cls as Q

FILE = {"path": "/x/checkpoint-epoch-120.pth", "sha256": "a" * 64, "bytes": 1}


def _c(seed=1):
    pin = A.SEEDS[seed]
    return {"detector_seed": seed, "S15_activation_checkpoint_flag": True,
            "endpoint": {"file": FILE, "sidecar": {"sha256": pin["sidecar_sha256"]}, "contract_sha256": A.S12_CONTRACT_SHA,
                         "formal_identity": pin["formal_identity"], "schema": A.SCHEMA, "parent_sha256": pin["parent_sha256"],
                         "ema_weights_sha256": pin["ema_weights_sha256"], "train_canvas": "H"}}


def _side(seed=1):
    pin = A.SEEDS[seed]
    return {"status": "COMPLETE", "schema": A.SCHEMA, "scope": "formal", "arm": "S15", "seed": seed,
            "formal_identity": pin["formal_identity"], "contract_sha256": A.S12_CONTRACT_SHA, "global_update": A.END,
            "parent_sha256": pin["parent_sha256"],
            "cursor": {"seed": seed, "end": True, "completed_updates": A.END, "completed_epochs": 120}, "file": FILE}


def _payload(seed=1):
    return {"formal_binding_sha256": A.binding_sha(_side(seed)), "version": S.VERSION, "arm": "S15",
            "parent_sha256": A.SEEDS[seed]["parent_sha256"], "config": S.production_config(seed).binding(),
            "activation_checkpoint": True, "engine": {"completed_updates": A.END}, "ema": {"updates": A.END}}


def test_seed_table_is_complete_and_seed_specific():
    assert sorted(A.SEEDS) == [1, 2] and A.SEED_ORDER == (1, 2)
    assert A.SEEDS[1].keys() == A.SEEDS[2].keys()
    for key in A.SEEDS[1]:
        assert len(A.SEEDS[1][key]) == 64 and A.SEEDS[1][key] != A.SEEDS[2][key]


def test_head_initialisation_stays_seed0():
    assert Q.INIT_SEED == 0 == A.HEAD_INIT_SEED
    assert A.verify_head_init() == A.HEAD_INIT_SHA
    state = {k: v.clone() for k, v in Q.initial_state().items()}
    state["fc1.weight"][0, 0] += 1e-3
    with pytest.raises(ValueError):
        A.verify_head_init(state)


@pytest.mark.parametrize("seed", [1, 2])
def test_good_side_and_payload_accepted(seed):
    A.check_side(_side(seed), _c(seed))
    A.check_payload(_payload(seed), _c(seed), A.binding_sha(_side(seed)))


@pytest.mark.parametrize("field,value", [
    ("seed", 2), ("schema", "s15-complete-state-v1"), ("scope", "fixture"), ("arm", "A0"), ("status", "PARTIAL"),
    ("global_update", 31920), ("parent_sha256", A.SEEDS[2]["parent_sha256"]), ("formal_identity", A.SEEDS[2]["formal_identity"]),
    ("contract_sha256", "0" * 64), ("file", {**FILE, "sha256": "b" * 64})])
def test_side_refusals(field, value):
    with pytest.raises(ValueError):
        A.check_side({**_side(1), field: value}, _c(1))


@pytest.mark.parametrize("field,value", [("seed", 2), ("end", False), ("completed_updates", 31920), ("completed_epochs", 105)])
def test_cursor_refusals(field, value):
    side = _side(1)
    with pytest.raises(ValueError):
        A.check_side({**side, "cursor": {**side["cursor"], field: value}}, _c(1))


@pytest.mark.parametrize("field,value", [
    ("config", S.production_config(2).binding()), ("config", S.production_config(0).binding()),
    ("config", S.Config(seed=1).binding()), ("parent_sha256", A.SEEDS[2]["parent_sha256"]), ("activation_checkpoint", False),
    ("arm", "A0"), ("version", "other"), ("formal_binding_sha256", "0" * 64), ("engine", {"completed_updates": 31920}),
    ("ema", {"updates": 36479})])
def test_payload_refusals(field, value):
    with pytest.raises(ValueError):
        A.check_payload({**_payload(1), field: value}, _c(1), A.binding_sha(_side(1)))


def _diff(apl=0.5, ap=-0.1, aps=-0.2, a75s=-0.2, apm=-0.5, ap300=-0.2, aps300=-0.2):
    return {"600": {"APL": apl, "AP": ap, "APS": aps, "AP75_small": a75s, "APM": apm},
            "300": {"AP": ap300, "APS": aps300}}


def test_gate_r_boundaries():
    assert A.gate_r(_diff(), True)["passed"]
    assert not A.gate_r(_diff(), False)["passed"] and A.gate_r(_diff(), False)["failed"] == ["I_not_passed"]
    for kw, cell in (("apl", "top600_APL"), ("ap", "top600_AP"), ("aps", "top600_APS"), ("a75s", "top600_AP75_small"),
                     ("apm", "top600_APM"), ("ap300", "top300_AP"), ("aps300", "top300_APS")):
        low = {"apl": 0.4999, "ap": -0.1001, "aps": -0.2001, "a75s": -0.2001, "apm": -0.5001, "ap300": -0.2001, "aps300": -0.2001}
        r = A.gate_r(_diff(**{kw: low[kw]}), True)
        assert not r["passed"] and r["failed"] == [cell]


def _rows():
    gt = {1: 23, 2: 5, 3: 3, 4: 779, 5: 115, 6: 87, 7: 12, 8: 5, 9: 32, 10: 7}
    return [{"category_id": k, "GT_large": n, "B2-B0_pp": float(k), "B2-B1_pp": 1.0, "B2-A0_pp": 0.0, "B1-B0_pp": -1.0}
            for k, n in gt.items()]


def test_contributions_keep_ten_class_macro_and_fixed_group():
    out = A.contributions(_rows())
    c = out["comparisons"]["B2-B0_pp"]
    assert c["macro_total_pp"] == pytest.approx(5.5)
    assert c["GT_le_12_classes_pp"] == pytest.approx((2 + 3 + 7 + 8 + 10) / 10)
    assert c["GT_le_12_classes_pp"] + c["other_classes_pp"] == pytest.approx(c["macro_total_pp"])
    assert set(c["named_class_difference_pp"]) == {"car", "van", "truck", "tricycle"}
    rows = _rows()
    rows[6]["GT_large"] = 13
    with pytest.raises(ValueError):
        A.contributions(rows)


def test_contract_field_guards():
    good = {"head_init": {"cpu_generator_seed": 0, "sha256": A.HEAD_INIT_SHA}}
    A.verify_head_contract(good)
    with pytest.raises(ValueError):
        A.verify_head_contract({"head_init": {"cpu_generator_seed": 1, "sha256": A.HEAD_INIT_SHA}})


def test_old_seed0_guard_untouched_and_binding_explicit(monkeypatch):
    assert X.SEED0_PARENT == "a2b6412e3c85de7592829f24aff09e355fdd11cb53798bc498bd04b882e1bf8a"
    assert X.endpoint_payload.__module__ == "s15_cross_scale" and X.validate_payload.__module__ == "s15_cross_scale"
    original = B.load_models
    assert original.__module__ == "sqc_run"
    monkeypatch.setattr(B, "load_models", original)
    A.bind()
    assert B.load_models is A.load_models
    assert X.endpoint_payload.__module__ == "s15_cross_scale"
    assert torch.is_tensor(Q.initial_state()["fc1.weight"])


def test_workload_counts_are_the_sqc001_ones():
    w = A.workload(1)
    for key in ("H_full_batches", "L_partial_batches", "head", "checks", "evaluation", "canvases"):
        assert w[key] == B.WORKLOAD[key]
    assert "timing" in w and "not_run" in w["timing"]
    assert w["detector"].startswith("seed1 ") and A.workload(2)["detector"].startswith("seed2 ")
