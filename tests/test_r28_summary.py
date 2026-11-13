"""R28 summary arithmetic and the registered decision rule (pure CPU; every number below is synthetic).

TASK_R28_001 section 7 with AMENDMENT_002 section 5: M (seven cells of W - L0), APL6, C, the K trigger, and A (W - K).
"""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "r28"))
import r28_common as K  # noqa: E402
import r28_eval as E  # noqa: E402

N = E.METRICS


def v(six, three):
    return {"600": dict(zip(N, six)), "300": dict(zip(N, three))}


L0 = v((0.3000, 0.2200, 0.2000, 0.3950, 0.4950, 0.5050, 0.3010), (0.2960, 0.2190, 0.1990, 0.3900, 0.4920, 0.4950, 0.2980))
GTS = (23, 5, 3, 779, 115, 87, 12, 5, 32, 7)
APL_L0 = [{"category_id": i + 1, "GT": g, "APL": 0.30 + 0.05 * i} for i, g in enumerate(GTS)]


def contract():
    ref = {"values": {a: copy.deepcopy(L0) for a in ("P", "C", "A0", "L0")}, "per_class_APL_top600":
           {a: copy.deepcopy(APL_L0) for a in ("A0", "S15", "L0")}}
    s15 = copy.deepcopy(L0)
    s15["600"]["APS"] += 0.0276
    ref["values"]["S15"] = s15
    return {"reference": ref, "labels": K.LABELS,
            "gates": {"M_W_minus_L0": [list(x) for x in K.GATE_M], "A_W_minus_K": list(K.GATE_A)}}


def arm(delta600=None, delta300=None, apl=None):
    values = copy.deepcopy(L0)
    for k, d in (delta600 or {}).items():
        values["600"][k] += d / 100.
    for k, d in (delta300 or {}).items():
        values["300"][k] += d / 100.
    table = copy.deepcopy(APL_L0)
    for cat, d in (apl or {}).items():
        table[cat - 1]["APL"] += d / 100.
    return {"values": values, "per_class_APL_top600": table}


def cost(ratio):
    return {"ratios": {"model_gpu": {"W/L0": ratio}, "end_to_end": {"W/L0": 1.01}}}


GOOD = {"APS": 0.80, "AP75_small": 0.40, "AP": 0.20}


def test_fixed_rule_constants():
    assert [tuple(x) for x in K.GATE_M] == [("600", "APS", 0.50), ("600", "AP75_small", 0.30), ("600", "AP", -0.10), ("300", "AP", -0.20),
                                           ("300", "APS", -0.20), ("600", "APM", -0.50), ("600", "APL", -0.50)]
    assert K.APL6_CLASSES == (1, 4, 5, 6, 7, 9) and K.APL6_MINIMUM == -0.50 and K.COST_LIMIT == 1.15
    assert K.GATE_A == ("600", "APS", 0.30) and K.G0_MINIMUM == 1.208
    assert sum(GTS[c - 1] for c in K.APL6_CLASSES) == 1048 and sum(GTS) == 1068


def test_all_cells_and_cost_pass_triggers_k():
    r = E.summary_w(contract(), arm(GOOD), cost(1.10))
    assert r["M_W_minus_L0"]["passed"] and r["C"]["passed"] and r["decision"] == {
        "label_key": "M_passed", "K_trigger": True, "exploratory": False, "label": "M_passed"}
    assert abs(r["M_W_minus_L0"]["cells"]["top600_APS"]["difference_pp"] - 0.80) < 1e-9
    assert abs(r["S15_small_gain_retained"]["share"] - 0.80 / 2.76) < 1e-9
    assert abs(r["differences_pp"]["W-S15"]["600"]["APS"] - (0.80 - 2.76)) < 1e-9


def test_apl6_and_cost_exactly_at_their_limits_pass():
    cells = {"a": {"passed": True}}
    assert E.decide_m(cells, -0.5, 1.15) == {"label_key": "M_passed", "K_trigger": True, "exploratory": False}
    assert E.decide_m(cells, 0.0, 1.1500001)["label_key"] == "closed"


def test_only_apl_fails_with_apl6_inside_is_exploratory_k():
    # APL600 falls by 0.9 pp through two sparse classes (c3, c10) only: APL6 is untouched
    w = arm({**GOOD, "APL": -0.90}, apl={3: -4.5, 10: -4.5})
    r = E.summary_w(contract(), w, cost(1.12))
    assert r["M_W_minus_L0"]["failed_cells"] == ["top600_APL"] and abs(r["APL6"]["difference_pp"]) < 1e-9 and r["APL6"]["passed"]
    assert r["decision"]["label_key"] == "M_failed_APL" and r["decision"]["K_trigger"] and r["decision"]["exploratory"]
    assert r["decision"]["label"] == "M_failed_APL; continue_K_exploratory" and not r["M_W_minus_L0"]["passed"]


def test_only_apl_fails_with_apl6_below_closes():
    w = arm({**GOOD, "APL": -0.90}, apl={4: -1.5, 5: -1.5, 6: -1.5})
    r = E.summary_w(contract(), w, cost(1.12))
    assert abs(r["APL6"]["difference_pp"] - (-4.5 / 6)) < 1e-9 and not r["APL6"]["passed"]
    assert r["decision"]["label_key"] == "closed" and not r["decision"]["K_trigger"]


@pytest.mark.parametrize("delta600, delta300, ratio", [
    ({"APS": 0.49, "AP75_small": 0.40}, None, 1.10),                 # small gain not reached
    ({**GOOD, "AP75_small": 0.29}, None, 1.10),
    ({**GOOD, "APM": -0.51}, None, 1.10),                            # another protection cell
    (GOOD, {"AP": -0.21}, 1.10),
    ({**GOOD, "APL": -0.90, "APM": -0.60}, None, 1.10),              # APL is not the only failed cell
    (GOOD, None, 1.16),                                              # cost
    ({**GOOD, "APL": -0.90}, None, 1.16),                            # APL-only but cost fails
])
def test_other_failures_close_without_k(delta600, delta300, ratio):
    r = E.summary_w(contract(), arm(delta600, delta300), cost(ratio))
    assert r["decision"] == {"label_key": "closed", "K_trigger": False, "exploratory": False, "label": "closed_fixed_config"}


def test_apl6_uses_exactly_the_six_registered_classes():
    a = E.apl6(arm(apl={1: 1.0, 4: 2.0, 5: 3.0, 6: 4.0, 7: 5.0, 9: 6.0, 2: -50, 3: -50, 8: -50, 10: -50})["per_class_APL_top600"], APL_L0)
    assert [r["category_id"] for r in a["classes"]] == [1, 4, 5, 6, 7, 9] and abs(a["difference_pp"] - 3.5) < 1e-9
    assert a["large_GT_covered"] == 1048 and a["large_GT_total"] == 1068
    bad = copy.deepcopy(APL_L0)
    bad[3]["GT"] += 1
    with pytest.raises(ValueError):
        E.apl6(bad, APL_L0)


def test_a_rule_and_k_summary():
    c = contract()
    w = arm(GOOD)
    w_result = E.summary_w(c, w, cost(1.10))
    cost_k = {"ratios": {"model_gpu": {"K/L0": 1.02}, "end_to_end": {"K/L0": 1.0}}}
    k_low = arm({"APS": 0.40})                      # W - K APS600 = +0.40
    r = E.summary_k(c, w_result, w, k_low, cost_k)
    assert r["A_W_minus_K"]["passed"] and r["A_W_minus_K"]["label"] == "adaptive_increment_established"
    assert abs(r["A_W_minus_K"]["difference_pp"] - 0.40) < 1e-9 and abs(r["differences_pp"]["K-L0"]["600"]["APS"] - 0.40) < 1e-9
    assert r["A_W_minus_K"]["minimum_pp"] == 0.30 and r["A_W_minus_K"]["cell"] == "top600_APS"
    k_high = arm({"APS": 0.60})                     # +0.20: not established (never called "equivalent")
    r = E.summary_k(c, w_result, w, k_high, cost_k)
    assert not r["A_W_minus_K"]["passed"] and r["A_W_minus_K"]["label"] == "adaptive_increment_not_established"
    assert r["M_W_minus_L0_decision_unchanged"] == w_result["decision"]
