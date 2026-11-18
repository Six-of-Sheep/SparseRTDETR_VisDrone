"""R30-002 control checks (CPU, pure): TASK_R30_SEEDS_002.

Unit names and order; the seed argument of the two science modules (default 0, certified parents only); the planned
workload; the release difference rule; the reading of task section 4 (single-seed rule, three-seed label, mean and
sample standard deviation, APM / APL signs, the scale trade-off registration).  No model forward.
"""
import copy
import inspect
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "r28"))
import r28_common as K  # noqa: E402
import r28_eval as E  # noqa: E402
import r30_control as R30  # noqa: E402
import r30_loop as RL  # noqa: E402
import r30_run as RUN  # noqa: E402
from sparse_rtdetr.research import r27_l0_control as L0  # noqa: E402
from sparse_rtdetr.research import r28_fovea_geo as FG  # noqa: E402
from sparse_rtdetr.research import scale_baseline as S  # noqa: E402

L0_600 = dict(zip(R30.METRICS, (0.3005, 0.2238, 0.2023, 0.3956, 0.4951, 0.5056, 0.3013)))
L0_300 = dict(zip(R30.METRICS, (0.2963, 0.2198, 0.1998, 0.3908, 0.4925, 0.4957, 0.2981)))
BASE = {"600": L0_600, "300": L0_300}


def shifted(d600=None, d300=None):
    out = copy.deepcopy(BASE)
    for b, d in (("600", d600 or {}), ("300", d300 or {})):
        for k, v in d.items():
            out[b][k] += v / 100.
    return out


def three(*fg):
    return {str(i): {"FG": v, "L0": BASE} for i, v in enumerate(fg)}


GAIN = {"APS": 2.0, "AP75_small": 2.3, "AP": 1.2, "APL": -0.5, "APM": 0.4}


def test_units_and_order():
    assert R30.ORDER == ("L0-s1", "FG-s1", "L0-s2", "FG-s2") and R30.SEEDS == (1, 2) and R30.ARMS == ("L0", "FG")
    assert [R30.parse(u) for u in R30.ORDER] == [("L0", 1), ("FG", 1), ("L0", 2), ("FG", 2)]
    assert R30.parse("FG-s0") == ("FG", 0) and R30.unit_name("FG", 2) == "FG-s2"
    for bad in ("W-s1", "FG-s3", "FG", "L0-s1 ", "QP-s1", None):
        with pytest.raises(Exception):
            R30.parse(bad)
    assert K.formal_re("L0-s1").fullmatch("formal-L0-s1") and K.formal_re("L0-s1").fullmatch("formal-L0-s1-r2")
    assert not K.formal_re("L0-s1").fullmatch("formal-L0-s2") and not K.formal_re("L0-s1").fullmatch("formal-L0-s1-r1")


def test_seed_argument_defaults_to_the_frozen_path():
    for mod in (L0, FG):
        for fn in ("create", "build_eval_model"):
            p = inspect.signature(getattr(mod, fn)).parameters["seed"]
            assert p.default == 0
    assert L0.IDENTITY_SHA256 == "136401a290c57de2f3ed69115e73159910f3a30ec17581f4b6a4019e90b5a0f8"
    assert FG.IDENTITY_SHA256["FG"] == "ba60f19aa2a06ab2f5704b0f7d076d1f9e495b2794820f8da5accc5a4a7633a5"
    assert FG.WIDTH_HEX == "0x1.3080000000000p+4" and FG.WIDTH == 19.03125
    assert sorted(S.PARENT_SHAS) == [0, 1, 2] and S.PARENT_SHAS[0] == L0.PARENT_SHA == K.PARENT_SHA
    assert S.PARENT_SHAS[1].startswith("6f8fd316") and S.PARENT_SHAS[2].startswith("c36e3c1f")


def test_seed_without_certified_parent_and_foreign_parent_are_refused(tmp_path):
    f = tmp_path / "not-a-parent.pth"
    f.write_bytes(b"x")
    for seed in (3, -1, True, "1", 1.0):
        with pytest.raises(Exception, match="certified parent"):
            L0.create(str(f), "cpu", seed=seed)
    for seed in (0, 1, 2):
        with pytest.raises(Exception, match="parent identity"):
            L0.create(str(f), "cpu", seed=seed)


def test_planned_workload():
    w = R30.WORKLOAD
    assert w["wire-s0"]["updates"] == 8 and w["wire-s0"]["forwards"] == 16
    assert w["accept"]["updates_per_seed"] == 20 and w["accept"]["forwards_per_seed"] == 40 and w["accept"]["updates"] == 40
    assert w["formal"]["updates_per_unit"] == 18240 and w["formal"]["forwards"] == 4 * 36480
    assert w["evaluation"]["forwards_per_unit"] == 141 and w["evaluation"]["forwards"] == 4 * 141
    assert w["evaluation"]["score_budget_calls"] == 8 and w["evaluation"]["per_class_COCOeval_calls"] == 4
    assert w["formal"]["save_epochs"] == list(range(60, 121, 5)) and len(w["formal"]["save_epochs"]) == 13
    assert R30.METRICS == E.METRICS and len(R30.READING["cells"]) == 14


def test_release_difference_rule():
    old = {"src/a.py": "1", "tools/r28/r28_loop.py": "2", **{p: "0" for p in R30.SCIENCE_CHANGED}}
    new = {**old, **{p: "9" for p in R30.SCIENCE_CHANGED}, **{p: "n" for p in R30.ALLOWED_ADDED}}
    d = R30.source_delta(old, new)
    assert d["changed"] == sorted(R30.SCIENCE_CHANGED) and d["added"] == sorted(R30.ALLOWED_ADDED) and not d["removed"]
    for bad in ({**new, "tools/r28/r28_loop.py": "x"}, {**new, "src/new.py": "x"}, {k: v for k, v in new.items() if k != "src/a.py"},
                {**new, R30.SCIENCE_CHANGED[0]: "0"}):
        with pytest.raises(Exception):
            R30.source_delta(old, bad)


def test_single_seed_rule_needs_all_three_cells():
    ok = R30.single_seed(R30.difference_pp(shifted(GAIN), BASE))
    assert ok["small_gain_repeated"] is True and sorted(ok["cells"]) == ["top600_AP", "top600_AP75_small", "top600_APS"]
    for cell, value in (("APS", 0.9), ("AP75_small", 0.9), ("AP", -0.2)):
        d = R30.difference_pp(shifted({**GAIN, cell: value}), BASE)
        assert R30.single_seed(d)["small_gain_repeated"] is False
    # top300 cells are reported but are not part of the rule
    assert R30.single_seed(R30.difference_pp(shifted(GAIN, {"APS": -5.0, "AP": -5.0}), BASE))["small_gain_repeated"] is True
    exact = {"600": {m: 0.0 for m in R30.METRICS}, "300": {m: 0.0 for m in R30.METRICS}}
    exact["600"].update(APS=1.0, AP75_small=1.0, AP=0.0)
    assert R30.single_seed(exact)["small_gain_repeated"] is True       # the bounds are inclusive


def test_labels():
    good, weak = shifted(GAIN), shifted({**GAIN, "APS": 0.5})
    assert R30.summary(three(good, good, good))["label"] == "small_gain_repeated_3of3"
    assert R30.summary(three(good, good, weak))["label"] == "small_gain_repeated_2of3"
    assert R30.summary(three(good, weak, good))["label"] == "small_gain_repeated_2of3"
    assert R30.summary(three(good, weak, weak))["label"] == "small_gain_not_repeated"
    assert R30.summary(three(weak, weak, weak))["label"] == "small_gain_not_repeated"
    s = R30.summary(three(good, weak, good))
    assert s["seeds_with_small_gain_repeated"] == ["0", "2"] and s["new_seeds_only_rule"] == {"1": False, "2": True}
    with pytest.raises(Exception):
        R30.label([True, True])
    with pytest.raises(Exception):
        R30.summary({"0": {"FG": good, "L0": BASE}, "1": {"FG": good, "L0": BASE}})


def test_all_fourteen_cells_mean_and_sample_sd():
    a, b, c = shifted({"APS": 2.0, "APL": -0.5}), shifted({"APS": 1.0, "APL": -1.5}), shifted({"APS": 3.0, "APL": 0.5}, {"AP50": -0.6})
    s = R30.summary(three(a, b, c))
    d = s["F_geo_minus_L0_pp"]
    assert sorted(d) == ["0", "1", "2"] and all(sorted(d[k]) == ["300", "600"] and sorted(d[k]["600"]) == sorted(R30.METRICS) for k in d)
    agg = s["aggregate_over_seeds"]
    assert sum(len(agg[bk]) for bk in agg) == 14
    assert agg["600"]["APS"]["mean_pp"] == pytest.approx(2.0, abs=1e-9) and agg["600"]["APS"]["sample_sd_pp"] == pytest.approx(1.0, abs=1e-9)
    assert agg["600"]["APL"]["mean_pp"] == pytest.approx(-0.5, abs=1e-9) and agg["600"]["APL"]["sample_sd_pp"] == pytest.approx(1.0, abs=1e-9)
    assert agg["300"]["AP50"]["mean_pp"] == pytest.approx(-0.2, abs=1e-9)
    assert agg["300"]["AP50"]["sample_sd_pp"] == pytest.approx(math.sqrt(0.12), abs=1e-9) and agg["300"]["AP50"]["n"] == 3
    assert agg["600"]["AP75"]["sample_sd_pp"] == 0.0


def test_size_cells_and_scale_tradeoff_are_registered_next_to_the_label():
    neg = shifted(GAIN)
    s = R30.summary(three(neg, neg, neg))
    assert s["label"] == "small_gain_repeated_3of3" and s["scale_tradeoff"] is True and s["scale_tradeoff_label"] == "存在尺度取舍"
    assert all(s["APM600_APL600_by_seed"][k]["APL"]["sign"] == "-" and s["APM600_APL600_by_seed"][k]["APM"]["sign"] == "+" for k in "012")
    pos = shifted({**GAIN, "APL": 0.2})
    s = R30.summary(three(neg, neg, pos))
    assert s["scale_tradeoff"] is False and s["scale_tradeoff_label"] is None and s["label"] == "small_gain_repeated_3of3"
    assert s["APM600_APL600_by_seed"]["2"]["APL"]["sign"] == "+"
    weak = shifted({"APS": 0.1, "APL": -1.0})
    s = R30.summary(three(weak, weak, weak))
    assert s["label"] == "small_gain_not_repeated" and s["scale_tradeoff"] is True      # neither cancels the other


def test_runner_seed_binding(monkeypatch):
    monkeypatch.delenv("R30_UNIT", raising=False)
    monkeypatch.delenv("R30_SEED", raising=False)
    assert RUN.process_seed("r30-check") == 0 and RUN.process_seed("summarize") == 0
    for action in ("r30-wire", "r30-accept", "formal", "evaluate"):
        with pytest.raises(Exception):
            RUN.process_seed(action)
    monkeypatch.setenv("R30_SEED", "0")
    assert RUN.process_seed("r30-wire") == 0
    with pytest.raises(Exception):
        RUN.process_seed("r30-accept")
    with pytest.raises(Exception):
        RUN.process_seed("r30-check")
    monkeypatch.setenv("R30_SEED", "2")
    assert RUN.process_seed("r30-accept") == 2
    with pytest.raises(Exception):
        RUN.process_seed("r30-wire")
    monkeypatch.delenv("R30_SEED")
    for unit, seed in (("L0-s1", 1), ("FG-s1", 1), ("L0-s2", 2), ("FG-s2", 2)):
        monkeypatch.setenv("R30_UNIT", unit)
        assert RUN.process_seed("formal") == seed and RUN.process_seed("evaluate") == seed
        with pytest.raises(Exception):
            RUN.process_seed("r30-check")
    monkeypatch.setenv("R30_UNIT", "FG-s0")
    with pytest.raises(Exception):
        RUN.process_seed("formal")


def test_delivery_override_only_selects_host(monkeypatch):
    monkeypatch.delenv("R30_DELIVERY", raising=False)
    assert R30.delivery_candidate() == "device"
    monkeypatch.setenv("R30_DELIVERY", "host")
    assert R30.delivery_candidate() == "host"
    monkeypatch.setenv("R30_DELIVERY", "prepared")
    with pytest.raises(Exception):
        R30.delivery_candidate()


def test_ledger_rows_and_class_table(tmp_path):
    p = tmp_path / "UPDATES.jsonl"
    p.write_text('{"a":1}\n{"a":2}\n{"a":3}\n')
    assert RL.ledger_rows(p) == [{"a": 1}, {"a": 2}, {"a": 3}] and RL.ledger_rows(p, 2) == [{"a": 1}, {"a": 2}]
    tabs = {"L0": [{"category_id": 1, "GT": 5, "APS": 0.1}, {"category_id": 2, "GT": 7, "APS": None}],
            "FG": [{"category_id": 1, "GT": 5, "APS": 0.3}, {"category_id": 2, "GT": 7, "APS": 0.2}]}
    assert RL.class_table(tabs, "APS") == [{"category_id": 1, "GT": 5, "L0": 0.1, "FG": 0.3}, {"category_id": 2, "GT": 7, "L0": None, "FG": 0.2}]
    assert RL.L0_ROW_KEYS == ("input", "gt_count", "loss", "lr", "gradient_norm_before_clip", "microbatch_losses", "update_seconds", "ema_updates")
