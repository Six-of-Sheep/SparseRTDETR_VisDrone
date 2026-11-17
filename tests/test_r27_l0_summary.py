"""R27-002 L0 summary arithmetic (pure CPU, synthetic L0 inputs; no model, no torch, no scorer).

P/C/S15/A0 numbers are the R27-001 certified readings copied from return-001/summary/RESULT.json; every "L0" value
below is synthetic (a copy or a perturbation of a reference reading), never an evaluation of an L0 model.
"""
import copy
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "r27_l0"))
import l0_common as K  # noqa: E402
import l0_summary as M  # noqa: E402

N = M.METRICS


def v(six, three):
    return {"600": dict(zip(N, six)), "300": dict(zip(N, three))}


REF = {
    "P": v((0.30615496615028065, 0.2332513847416923, 0.2104479879430917, 0.39764542040881123, 0.4804911938781984,
            0.5130936288725753, 0.30733259474459373),
           (0.30189509328965214, 0.2290501293944892, 0.20774394072461114, 0.39278398655140123, 0.47821972055966716,
            0.5029690495706708, 0.3038988489653735)),
    "C": v((0.30183381962746547, 0.22768556739823073, 0.2048475741318266, 0.395437847407559, 0.48736607532503623,
            0.5074642543574635, 0.3014300300324815),
           (0.29786256286641966, 0.22407877809429816, 0.20267133030217682, 0.3908031214103899, 0.4850213294479697,
            0.49818087983509435, 0.2989917847235283)),
    "S15": v((0.3200866660864559, 0.2513747771125344, 0.23720953090636165, 0.40495283253729625, 0.4446219718630393,
              0.5294398951854375, 0.326617727108982),
             (0.3158812870411125, 0.24734999661182375, 0.234330056498393, 0.4013240711944936, 0.44069524292638923,
              0.5203235106972568, 0.3233450788664634)),
    "A0": v((0.2978928000787488, 0.2217618575337924, 0.19968104528445318, 0.39344481665007297, 0.4725232045664882,
             0.5033924745103661, 0.2989492964371395),
            (0.29347668234713176, 0.2177929184225555, 0.19712874220974094, 0.3883243218218308, 0.4686511018043208,
             0.4929628974488825, 0.2952200358559255)),
}
G = {"top600_APS": 1.148952720789989, "top600_AP75_small": 1.0766942658638508, "top600_AP": 0.8262166071531851,
     "top300_AP": 0.8418410942520382, "top300_APS": 1.1257210971933718, "top600_APM": 0.4200603758738264,
     "top600_APL": 0.7967989311710166}  # R27-001 RESULT.json G_P_minus_A0 cells, difference_pp
GT = (23, 5, 3, 779, 115, 87, 12, 5, 32, 7)
APL = {
    "P": (0.6399537142019007, 0.11817563272314839, 0.30336633663366336, 0.8532420140584542, 0.5876954078734409,
          0.5254560256071599, 0.4136951571096959, 0.26799704815202013, 0.8041834991056822, 0.2911471033168172),
    "C": (0.6644867922169699, 0.12292509610548633, 0.40429042904290424, 0.8571335992680974, 0.5772404382828289,
          0.5298757479289273, 0.3943903648585036, 0.25454974068835456, 0.7978486811182612, 0.2709198637400287),
    "S15": (0.6662202796654307, 0.20214315083676637, 0.008415841584158416, 0.8564880248356544, 0.542596588368475,
            0.49842116265381364, 0.32992449923725403, 0.282733004483244, 0.7918748444352076, 0.26740232253038965),
    "A0": (0.661278752037809, 0.14247489378597117, 0.2378830740216879, 0.8489140881045164, 0.6015885269497703,
           0.5254929910656874, 0.38129431439224676, 0.2776520509193776, 0.7611541718070195, 0.28749918258079676),
}
CLASSES = {a: [{"category_id": i + 1, "GT": GT[i], "APL": x} for i, x in enumerate(xs)] for a, xs in APL.items()}
COUNTS = {"cpu_fixture": {"actual_updates": 3, "forwards": 6}, "cuda_zero_step": {"actual_updates": 0, "forwards": 2, "reference_forwards": 2},
          "cuda_accept": {"actual_updates": 3, "forwards": 6}, "formal": {"actual_updates": 18240, "forwards": 36480},
          "evaluation": {"actual_updates": 0, "forwards": 141, "score_budget_calls": 2, "per_class_COCOeval_calls": 1},
          "cost4": {"actual_updates": 0, "forwards": 200}}
COST = {"L0_weights_sha256": "w" * 64, "medians": {e: {"model_gpu_ms_median_pass": 1.} for e in K.COST_ENDPOINTS},
        "ratios": {"P_over_L0": {"model_gpu": 1.}}}
R27 = {"file": {"path": "x", "bytes": 1, "sha256": K.R27_RESULT_SHA}, "label": "E_failed: fixed R27 configuration closed",
       "E_passed": False, "I_passed": False, "G_passed": True}


def contract():
    return {"contract_sha256": "c" * 64, "labels": K.LABELS, "workload": K.WORKLOAD,
            "gates": {"D_P_minus_L0": [list(x) for x in K.GATE_D]},
            "reference": {"values": copy.deepcopy(REF), "per_class_APL_top600": copy.deepcopy(CLASSES),
                          "P_minus_A0_self_check": {"difference_pp": dict(G)}}}


def synthetic_eval(values=None, classes=None):
    return {"values": copy.deepcopy(values or REF["A0"]), "per_class_APL_top600": copy.deepcopy(classes or CLASSES["A0"]),
            "endpoint": {"path": "synthetic", "bytes": 0, "sha256": "0" * 64}, "weights_sha256": "w" * 64,
            "counts": dict(COUNTS["evaluation"])}


def at_gate(b, m, minimum, step=0):
    """Synthetic (P, L0): reference P and L0 = A0, except one cell whose difference is exactly its minimum (step=0)
    or the next representable difference below it (step=1).  The cell uses 0 and |minimum|/100 so the product is exact."""
    p, l0 = copy.deepcopy(REF["P"]), copy.deepcopy(REF["A0"])
    hi, lo = (minimum / 100., 0.) if minimum > 0 else (0., -minimum / 100.)
    assert 100. * (hi - lo) == minimum
    for _ in range(step):
        while 100. * (hi - lo) >= minimum:  # move the non-zero side: a subnormal step from 0 would be absorbed
            hi, lo = (math.nextafter(hi, -math.inf), lo) if minimum > 0 else (hi, math.nextafter(lo, math.inf))
    p[b][m], l0[b][m] = hi, lo
    return p, l0


def test_l0_equal_to_a0_reproduces_the_r27_g_cells():
    d = M.decision_d(REF["P"], REF["A0"], K.GATE_D, K.LABELS)
    assert {k: c["difference_pp"] for k, c in d["cells"].items()} == G and list(d["cells"]) == list(G)
    assert d["passed"] and d["label"] == K.LABELS["D_passed"] and d["failed_cells"] == []
    diff = M.differences(REF, REF["A0"])
    assert diff["P-L0"]["difference_pp"] == diff["P-A0"]["difference_pp"]
    assert all(x == 0. for b in M.BUDGETS for x in diff["L0-A0"]["difference_pp"][b].values())
    assert [k for k, x in diff.items() if x["new"]] == ["P-L0", "C-L0", "L0-A0", "L0-S15"]
    assert all(x["source"] == M.SOURCE_OLD for k, x in diff.items() if k in ("P-A0", "P-S15", "C-A0", "P-C"))
    assert all(len(x["difference_pp"][b]) == 7 for x in diff.values() for b in M.BUDGETS)
    assert M.self_check(REF, K.GATE_D, G)["equal_to_contract_self_check"]
    with pytest.raises(ValueError):
        M.self_check(REF, K.GATE_D, {**G, "top600_APL": G["top600_APL"] + 1e-12})


@pytest.mark.parametrize("b,m,minimum", K.GATE_D)
def test_gate_boundary_and_each_cell_alone_decides(b, m, minimum):
    on = M.decision_d(*at_gate(b, m, minimum), K.GATE_D, K.LABELS)
    assert on["cells"][f"top{b}_{m}"]["difference_pp"] == minimum and on["cells"][f"top{b}_{m}"]["passed"] and on["passed"]
    below = M.decision_d(*at_gate(b, m, minimum, step=1), K.GATE_D, K.LABELS)
    assert below["cells"][f"top{b}_{m}"]["difference_pp"] < minimum
    assert not below["passed"] and below["failed_cells"] == [f"top{b}_{m}"] and below["label"] == K.LABELS["D_failed"]
    assert sum(c["passed"] for c in below["cells"].values()) == 6


def test_per_class_contributions_sum_to_the_apl600_difference():
    values = {**REF, "L0": REF["A0"]}
    r = M.per_class(CLASSES, CLASSES["A0"], values)
    assert r["columns"] == ["A0", "S15", "C", "P", "L0"] and [t["large_GT"] for t in r["table"]] == list(GT)
    for name, x, y in M.CLASS_PAIRS:
        p = r["differences"][name]
        assert len(p["classes"]) == 10 and abs(p["residual_pp"]) <= 1e-9
        assert p["APL600_difference_pp"] == 100. * (values[x]["600"]["APL"] - values[y]["600"]["APL"])
        assert all(c["macro_contribution_pp"] == c["difference_pp"] / 10 for c in p["classes"])
    assert r["differences"]["P-L0"]["APL600_difference_pp"] == G["top600_APL"]
    moved = copy.deepcopy(CLASSES["A0"])
    moved[2]["APL"] += 1e-6  # 1e-5 pp in the macro: the sum no longer matches the aggregate
    with pytest.raises(ValueError, match="per-class contributions"):
        M.per_class(CLASSES, moved, values)
    other_gt = copy.deepcopy(CLASSES["A0"])
    other_gt[3]["GT"] += 1
    with pytest.raises(ValueError, match="large GT counts"):
        M.per_class(CLASSES, other_gt, values)


def test_workload_reconciliation_reports_differences():
    w = M.workload(K.WORKLOAD, COUNTS)
    assert w["all_equal"] and w["unequal_stages"] == []
    assert w["normal_gpu_total"]["actual"] == {"updates": 18243, "forwards": 36831, "score_budget_calls": 2,
                                               "per_class_COCOeval_calls": 1}
    off = copy.deepcopy(COUNTS)
    off["formal"]["forwards"] -= 2
    off["evaluation"]["score_budget_calls"] = 3
    w = M.workload(K.WORKLOAD, off)
    assert not w["all_equal"] and w["unequal_stages"] == ["formal", "evaluation"]
    assert w["stages"]["formal"]["differs"] == ["forwards"] and w["stages"]["evaluation"]["differs"] == ["score_budget_calls"]
    assert w["normal_gpu_total"]["differs"] == ["forwards", "score_budget_calls"] and not w["normal_gpu_total"]["equal"]
    assert w["stages"]["cost4"]["equal"] and w["stages"]["formal"]["actual"]["forwards"] == 36478
    assert w["stages"]["cuda_zero_step"]["actual"]["forwards"] == 4 and w["stages"]["cuda_zero_step"]["equal"]
    resumed = copy.deepcopy(COUNTS)  # a resumed formal process closes with its own counts only: reported, not hidden
    resumed["formal"] = {"actual_updates": 9120, "forwards": 18240}
    w = M.workload(K.WORKLOAD, resumed, {"resumed": True, "start_update": 27360})
    assert not w["all_equal"] and w["unequal_stages"] == ["formal"] and w["formal_chain"]["resumed"]


def test_cost_is_copied_and_bound_to_the_evaluated_weights():
    r = M.cost(COST, "w" * 64, K.COST_ENDPOINTS)
    assert r["medians"] == COST["medians"] and r["ratios"] == COST["ratios"] and r["L0_weights_equal_to_evaluation"]
    with pytest.raises(ValueError, match="another L0 model"):
        M.cost(COST, "v" * 64, K.COST_ENDPOINTS)


def test_build_result_structure_and_fixed_statements():
    res = M.build(contract(), synthetic_eval(), COST, COUNTS, R27)
    assert list(res) == ["status", "contract_sha256", "arm", "endpoint", "values", "value_sources", "D_P_minus_L0", "label",
                         "differences_pp", "P_minus_A0_self_check", "per_class_APL_top600", "cost4", "workload", "R27_001",
                         "statements"]
    assert {k: c["difference_pp"] for k, c in res["D_P_minus_L0"]["cells"].items()} == G
    assert set(res["values"]) == {"P", "C", "S15", "A0", "L0"} and res["value_sources"]["L0"] == M.SOURCE_L0
    assert res["R27_001"]["label"] == R27["label"] and res["R27_001"]["result_file"]["sha256"] == K.R27_RESULT_SHA
    assert set(res["statements"]) == {"old_E_I_G", "D_origin", "single_seed", "development_reuse", "D_failed_reading",
                                      "D_passed_reading", "L0_minus_A0", "P_minus_L0", "cost"}
    bad = synthetic_eval()
    bad["counts"]["score_budget_calls"] = 3
    with pytest.raises(ValueError, match="scorer call counts"):
        M.build(contract(), bad, COST, COUNTS, R27)
    low = copy.deepcopy(REF["A0"])
    low["600"]["APS"] = REF["P"]["600"]["APS"] - 0.004  # synthetic: P - L0 = 0.4 pp on APS600, below the +0.50 minimum
    failed = M.build(contract(), synthetic_eval(low), COST, COUNTS, R27)
    assert failed["label"] == K.LABELS["D_failed"] and failed["D_P_minus_L0"]["failed_cells"] == ["top600_APS"]
