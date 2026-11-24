"""R32-001 reading arithmetic (CPU, pure, no torch): TASK_R32_W_SEEDS_001 section 5.

Seed 0 uses the stored values of R28-001 (W) and R30-002 (L0, F_geo) and of the S15 top600 metric files; the two new
seeds are synthetic.  Nothing here is a result of the round.
"""
import copy
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "r28"))
import r32_reading as RD  # noqa: E402

M = RD.METRICS


def table(t600, t300):
    return {"600": dict(zip(M, t600)), "300": dict(zip(M, t300))}


W0 = table((0.3086567643886324, 0.24148359880902054, 0.2278016542232594, 0.38873186070187543, 0.5397136493395969, 0.5168595853586602, 0.31375492884771905),
           (0.3048160515533133, 0.23796950340380596, 0.22581171897111124, 0.3846426035802965, 0.5349491086614001, 0.5081275977815047, 0.31059072987224606))
L0 = {"0": table((0.30054005272689926, 0.2238294714480465, 0.20231585685439496, 0.39561519581083376, 0.4951035581025013, 0.5056373378269972, 0.3013441188978117),
                 (0.296328453392772, 0.21978520314185324, 0.19977749969484612, 0.39084105276155096, 0.49249422115010244, 0.4957413517155677, 0.298144706233112)),
      "1": table((0.30290162806340576, 0.2254072810959945, 0.20096192156510068, 0.40102132976280724, 0.4488479214760187, 0.5117730782383165, 0.3021050336978677),
                 (0.29915302846553293, 0.2214872383014935, 0.1984273043087975, 0.39754100430751294, 0.4471313233414998, 0.5033517132844336, 0.29943692607154815)),
      "2": table((0.30150564214328784, 0.22422677959481147, 0.19754164165515165, 0.3956624409754658, 0.4788377695020645, 0.5073386223449639, 0.30020660994953113),
                 (0.29728285860917436, 0.22057918094470946, 0.19507027145705713, 0.39110390790351923, 0.47338524031732887, 0.4978725507427179, 0.2967229653433951))}
FG = {"0": table((0.3132075511405779, 0.24417841447853297, 0.22563462872819284, 0.39927912432216967, 0.48974805105438635, 0.5204665645521643, 0.31774994843857113),
                 (0.30886883114511904, 0.24018994577228878, 0.223532589519686, 0.3947358187439386, 0.4854469030134367, 0.5107338698420886, 0.31447622126515595)),
      "1": table((0.3125475953792031, 0.24292413189730772, 0.21997304578970933, 0.3975386616786504, 0.49392415189057903, 0.5236201015237968, 0.3128094180277362),
                 (0.3084796782334975, 0.23944254627554676, 0.2173201668412622, 0.39287859734005454, 0.4909695145090021, 0.5144770074955133, 0.3093202682967768)),
      "2": table((0.3131752190657607, 0.24051421340733264, 0.2181047609638724, 0.40049352470324595, 0.4831228144863633, 0.522589440798317, 0.31467696111668625),
                 (0.3089131072189109, 0.23635495411376953, 0.21546939593493017, 0.39644887546023355, 0.47840342628266563, 0.5129485690294762, 0.31119772252606426))}
S15 = {"0": dict(zip(M, (0.3200866660864559, 0.2513747771125344, 0.23720953090636165, 0.40495283253729625, 0.4446219718630393, 0.5294398951854375, 0.326617727108982))),
       "1": dict(zip(M, (0.32275410693313805, 0.2518638980699198, 0.2336804215936788, 0.41066023104790234, 0.42352199717772643, 0.5337588046562199, 0.3271312429300467))),
       "2": dict(zip(M, (0.3226902485312341, 0.2531481646852, 0.24237263525556532, 0.4098810662004296, 0.42790301920128576, 0.5336622362570765, 0.33194718653206917)))}
GT = (23, 5, 3, 779, 115, 87, 12, 5, 32, 7)
APL_W0 = (0.7280770118527669, 0.253386586978172, 0.45841584158415843, 0.8582844211755881, 0.5801038820840037, 0.5507299724103222, 0.45726867268770216,
          0.2124983026030689, 0.7893603934266024, 0.5090114085935846)
APL_L0 = {"0": (0.6690878911991505, 0.11027294185114715, 0.3681753889674682, 0.8555318949623206, 0.592309945410607, 0.5360973508950543, 0.4162493040511631,
                0.2537276804603537, 0.8002209168124275, 0.34936226641532075),
          "1": (0.7173858355173615, 0.2880680023375209, 0.044936810754246145, 0.8602604261139484, 0.5809538215125665, 0.5067168172801948, 0.4005954810097106,
                0.053613193441814345, 0.7657468220722274, 0.2702020047205959),
          "2": (0.6788310456239112, 0.18896433180465397, 0.09690947666195188, 0.8561658517633276, 0.567002269706104, 0.5373953885640657, 0.4084958680947286,
                0.31696480802969607, 0.8162737578365513, 0.3213748969356555)}
APL_FG = {"0": (0.6814848442868939, 0.2953445600970353, 0.11297029702970296, 0.8575771683797689, 0.5912601797772618, 0.5352190045808289, 0.4403849227023179,
                0.12530799663817313, 0.820151634411328, 0.4377799026405526),
          "1": (0.672009493616874, 0.20753942970421171, 0.28155940594059403, 0.8605763059492006, 0.5802965069829205, 0.5206927833825684, 0.43714156782942604,
                0.10738389056296935, 0.7925873267514996, 0.47945480818552555),
          "2": (0.6543501414650584, 0.2522501414021001, 0.08672324375294672, 0.8605927482533938, 0.5754638640797822, 0.5078950501399905, 0.39956518601830193,
                0.295530233210684, 0.7902867815020705, 0.40857075503930484)}


def rows(apl):
    return [{"category_id": i + 1, "GT": g, "APL": v} for i, (g, v) in enumerate(zip(GT, apl))]


def shifted(base, d600=None, d300=None):
    out = copy.deepcopy(base)
    for b, d in (("600", d600 or {}), ("300", d300 or {})):
        for k, v in d.items():
            out[b][k] += v / 100.
    return out


def synthetic_w(seed, d600=None, d300=None, on="L0"):
    """A W table for a new seed as the L0 (or F_geo) table of that seed plus the given differences in pp, with per-class
    large-AP rows whose macro mean is the APL600 cell."""
    base = {"L0": L0, "FG": FG}[on][seed]
    w = shifted(base, d600, d300)
    apl = list({"L0": APL_L0, "FG": APL_FG}[on][seed])
    shift = w["600"]["APL"] - math.fsum(apl) / 10
    return w, rows([v + shift for v in apl])


def world(w1, w2):
    values = {"0": {"W": W0, "L0": L0["0"], "FG": FG["0"]}}
    per_class = {"0": {"W": rows(APL_W0), "L0": rows(APL_L0["0"]), "FG": rows(APL_FG["0"])}}
    for s, (w, apl) in (("1", w1), ("2", w2)):
        values[s] = {"W": w, "L0": L0[s], "FG": FG[s]}
        per_class[s] = {"W": apl, "L0": rows(APL_L0[s]), "FG": rows(APL_FG[s])}
    return values, S15, per_class


GAIN = {"APS": 1.8, "AP75_small": 2.5, "AP": 0.8, "APM": -0.7, "APL": 4.0}


def test_constants_of_the_task():
    assert RD.BOUND == (("600", "APS", 1.0), ("600", "AP75_small", 1.0), ("600", "AP", 0.0))
    assert RD.SEVEN == (("600", "APS", 0.50), ("600", "AP75_small", 0.30), ("600", "AP", -0.10), ("300", "AP", -0.20), ("300", "APS", -0.20),
                        ("600", "APM", -0.50), ("600", "APL", -0.50))
    assert RD.MEDIUM_COST == -0.50 and RD.RARE_LARGE_GT == 12 and len(RD.READING["cells"]) == 14
    assert RD.LABELS_A == ("W_small_gain_repeated_3of3", "W_small_gain_repeated_2of3", "W_small_gain_not_repeated")
    assert RD.LABELS_C == ("W_medium_cost_repeated_3of3", "W_medium_cost_repeated_2of3", "W_medium_cost_not_repeated")
    assert RD.LABEL_B_NONE == "no_practical_lead_in_any_seed"
    assert sum(g <= 12 for g in GT) == 5


def test_seed0_values_of_the_task_text():
    """Task section 1 and 5: W - L0 = APS +1.765, AP75-small +2.549, AP +0.812, APM -0.688, APL +4.461; W - F_geo =
    APS -0.269, AP75-small +0.217, AP -0.455; S15 - W = AP +1.143, APS +0.989, AP75-small +0.941, APM +1.622, APL -9.509."""
    d = RD.difference_pp(W0, L0["0"])
    assert d["600"]["APS"] == 1.7654127360974048 and d["600"]["AP75_small"] == 2.5485797368864445 and d["600"]["AP"] == 0.8116711661733111
    assert d["600"]["APM"] == -0.6883335108958333 and d["600"]["APL"] == 4.461009123709559 and d["300"]["APS"] == 1.8184300261952724
    f = RD.difference_pp(W0, FG["0"])
    assert f["600"]["APS"] == -0.2694815669512435 and f["600"]["AP75_small"] == 0.21670254950665602 and f["600"]["AP"] == -0.4550786751945535
    assert RD.bound_cells(d)["met"] is True and RD.lead(f)["reading"] == "neither"
    assert RD.seven_cells(d) == {"cells": RD.seven_cells(d)["cells"], "failed_cells": ["top600_APM"], "passed": False}
    s = RD.difference600_pp(S15["0"], W0["600"])
    assert [round(s[m], 3) for m in ("AP", "APS", "AP75_small", "APM", "APL")] == [1.143, 0.989, 0.941, 1.622, -9.509]


def test_bound_is_inclusive_and_all_three_cells_are_needed():
    exact = RD.difference_pp(shifted(L0["1"], {"APS": 0.0}), L0["1"])
    exact["600"].update(APS=1.0, AP75_small=1.0, AP=0.0)
    assert RD.bound_cells(exact)["met"] is True
    for cell, value in (("APS", 0.9999999999), ("AP75_small", 0.9999999999), ("AP", -1e-12)):
        d = copy.deepcopy(exact)
        d["600"][cell] = value
        got = RD.bound_cells(d)
        assert got["met"] is False and [k for k, v in got["cells"].items() if not v["passed"]] == [f"top600_{cell}"]
    d = copy.deepcopy(exact)
    d["300"].update(APS=-9.0, AP=-9.0)                   # the top300 cells are not part of reading A or B
    assert RD.bound_cells(d)["met"] is True


def test_lead_uses_the_same_bound_in_both_directions():
    base = RD.difference_pp(L0["1"], L0["1"])
    assert RD.lead(base)["reading"] == "neither"         # AP >= 0 holds both ways at zero, the small cells do not
    w = copy.deepcopy(base)
    w["600"].update(APS=1.0, AP75_small=1.2, AP=0.0)
    assert RD.lead(w)["reading"] == "W_leads" and RD.lead(RD.negated(w))["reading"] == "F_geo_leads"
    w["600"]["AP"] = -0.01
    assert RD.lead(w)["reading"] == "neither" and RD.lead(RD.negated(w))["reading"] == "neither"
    w["600"].update(APS=-1.0, AP75_small=-1.0, AP=0.0)
    assert RD.lead(w)["reading"] == "F_geo_leads"
    w["600"]["APS"] = -0.99
    assert RD.lead(w)["reading"] == "neither"


def test_labels():
    assert RD.label_a([True, True, True]) == "W_small_gain_repeated_3of3"
    assert RD.label_a([True, True, False]) == RD.label_a([True, False, True]) == "W_small_gain_repeated_2of3"
    assert RD.label_a([True, False, False]) == "W_small_gain_not_repeated"
    assert RD.label_c([True, True, True]) == "W_medium_cost_repeated_3of3"
    assert RD.label_c([True, False, True]) == "W_medium_cost_repeated_2of3"
    assert RD.label_c([True, False, False]) == "W_medium_cost_not_repeated"
    assert RD.label_b(["neither"] * 3) == "no_practical_lead_in_any_seed"
    assert RD.label_b(["neither", "W_leads", "neither"]) == "W_leads_1of3_F_geo_leads_0of3"
    assert RD.label_b(["neither", "F_geo_leads", "F_geo_leads"]) == "W_leads_0of3_F_geo_leads_2of3"
    assert RD.label_b(["neither", "W_leads", "F_geo_leads"]) == "W_leads_1of3_F_geo_leads_1of3"
    for bad in ([False, True, True], [True, True], [True, 1, True]):
        with pytest.raises(ValueError):
            RD.label_a(bad)
        with pytest.raises(ValueError):
            RD.label_c(bad)
    for bad in (["W_leads", "neither", "neither"], ["neither", "neither"], ["neither", "tie", "neither"]):
        with pytest.raises(ValueError):
            RD.label_b(bad)


def test_reading_three_of_three_and_no_lead():
    out = RD.reading(*world(synthetic_w("1", GAIN, {"APS": 1.7, "AP": 0.8}), synthetic_w("2", GAIN, {"APS": 1.7, "AP": 0.8})))
    assert out["reading_A"]["label"] == "W_small_gain_repeated_3of3" and out["reading_A"]["seeds_satisfied"] == ["0", "1", "2"]
    assert out["reading_B"]["label"] == "no_practical_lead_in_any_seed" and out["reading_B"]["readings"] == {"0": "neither", "1": "neither", "2": "neither"}
    assert out["reading_C"]["label"] == "W_medium_cost_repeated_3of3"
    assert out["W_unique_advantage"] == {"status": "not_established", "changed_by_this_task": False, "new_seeds_reading_W_leads": [],
                                         "rule": out["W_unique_advantage"]["rule"], "inconsistent_with_seed0": False}
    assert out["seven_cells_W_minus_L0"]["0"]["failed_cells"] == ["top600_APM"] and out["seven_cells_W_minus_L0"]["1"]["failed_cells"] == ["top600_APM"]
    assert abs(out["W_minus_L0_pp"]["1"]["600"]["APS"] - 1.8) < 1e-9 and abs(out["W_minus_L0_pp"]["2"]["300"]["APS"] - 1.7) < 1e-9
    assert sorted(out["S15_minus_W_top600_pp"]["1"]) == sorted(M)
    agg = out["aggregate_over_seeds"]["W_minus_L0"]["600"]["APS"]
    v = agg["values_pp"]
    assert agg["n"] == 3 and abs(agg["mean_pp"] - sum(v) / 3) < 1e-12
    sd = math.sqrt(sum((x - sum(v) / 3) ** 2 for x in v) / 2)
    assert abs(agg["sample_sd_pp"] - sd) < 1e-12 and abs(agg["t95_halfwidth_of_mean_pp"] - 4.302652729911275 * sd / math.sqrt(3)) < 1e-12
    assert out["APL"]["0"]["sign"] == "+" and abs(out["APL"]["0"]["W_minus_L0_pp"] - 4.461009123709559) < 1e-12


def test_reading_two_of_three_one_seed_fails_and_medium_cost_not_repeated():
    weak = dict(GAIN, APS=0.95, APM=-0.49)
    out = RD.reading(*world(synthetic_w("1", weak), synthetic_w("2", dict(GAIN, APM=0.2))))
    assert out["reading_A"]["label"] == "W_small_gain_repeated_2of3" and out["reading_A"]["seeds_satisfied"] == ["0", "2"]
    assert out["reading_C"]["label"] == "W_medium_cost_not_repeated" and out["reading_C"]["seeds_at_or_below_the_bound"] == ["0"]
    assert out["reading_C"]["per_seed"]["2"]["sign"] == "+" and out["reading_C"]["per_seed"]["1"]["sign"] == "-"
    out = RD.reading(*world(synthetic_w("1", dict(GAIN, AP=-0.05)), synthetic_w("2", dict(GAIN, AP75_small=0.5))))
    assert out["reading_A"]["label"] == "W_small_gain_not_repeated" and out["reading_C"]["label"] == "W_medium_cost_repeated_3of3"


def test_medium_cost_bound_is_inclusive():
    w, apl = synthetic_w("1", GAIN)
    w["600"]["APM"] = L0["1"]["600"]["APM"] - 0.005     # exactly -0.50 pp up to rounding of the subtraction
    out = RD.reading(*world((w, apl), synthetic_w("2", dict(GAIN, APM=-0.4999))))
    d = out["reading_C"]["per_seed"]["1"]["W_minus_L0_pp"]
    assert (d <= -0.50) == out["reading_C"]["per_seed"]["1"]["at_or_below_the_bound"]
    assert out["reading_C"]["per_seed"]["2"]["at_or_below_the_bound"] is False


def test_a_new_seed_reading_w_leads_is_reported_as_inconsistent_with_seed0():
    lead = {"APS": 1.3, "AP75_small": 1.1, "AP": 0.2, "APM": -0.3, "APL": 1.0}
    out = RD.reading(*world(synthetic_w("1", lead, on="FG"), synthetic_w("2", {k: -v for k, v in lead.items()}, on="FG")))
    assert out["reading_B"]["readings"] == {"0": "neither", "1": "W_leads", "2": "F_geo_leads"}
    assert out["reading_B"]["label"] == "W_leads_1of3_F_geo_leads_1of3"
    u = out["W_unique_advantage"]
    assert u["status"] == "not_established" and u["changed_by_this_task"] is False and u["new_seeds_reading_W_leads"] == ["1"] and u["inconsistent_with_seed0"] is True


def test_apl_split_seed0_matches_the_stored_cells():
    for other, cell in ((APL_L0["0"], 4.461009123709559), (APL_FG["0"], 4.996559828521052)):
        s = RD.apl_split(rows(APL_W0), rows(other))
        assert s["rare"]["classes"] == [2, 3, 7, 8, 10] and s["other"]["classes"] == [1, 4, 5, 6, 9]
        assert s["rare"]["large_GT"] == 32 and s["other"]["large_GT"] == 1036 and s["classes_in_the_macro_mean"] == 10
        assert abs(s["macro_difference_pp"] - cell) < 1e-9
        assert abs(s["rare"]["contribution_pp"] - sum(100 * (APL_W0[i] - other[i]) for i in (1, 2, 6, 7, 9)) / 10) < 1e-12
    bad = rows(APL_L0["0"])
    bad[0]["GT"] = 24
    with pytest.raises(ValueError):
        RD.apl_split(rows(APL_W0), bad)
    with pytest.raises(ValueError):
        RD.apl_split(rows(APL_W0)[:9], rows(APL_L0["0"])[:9])
    none = rows(APL_L0["0"])
    none[2]["APL"] = None
    with pytest.raises(ValueError):
        RD.apl_split(rows(APL_W0), none)


def test_reading_refuses_per_class_rows_that_do_not_add_up():
    w, apl = synthetic_w("1", GAIN)
    apl[3]["APL"] += 0.01
    with pytest.raises(ValueError):
        RD.reading(*world((w, apl), synthetic_w("2", GAIN)))


def test_geometry_is_the_instance_weighted_geometric_mean():
    def r(n, g, lt, lo=0.8, hi=1.4, ident=0):
        return {"n_small": n, "g_small": g, "small_g_lt_1": lt, "slope_min": lo, "slope_max": hi, "identity_images": ident}
    rows_ = [(61, r(10, 1.2, 0.1)), (61, r(30, 1.3, 0.2, 0.7, 1.5)), (117, r(0, None, None)), (118, r(20, 1.1, 0.5, ident=2)), (120, r(40, 1.25, 0.0))]
    g = RD.geometry(rows_)
    want = math.exp((10 * math.log(1.2) + 30 * math.log(1.3) + 20 * math.log(1.1) + 40 * math.log(1.25)) / 100)
    assert g["all"]["updates"] == 5 and g["all"]["updates_with_small_gt"] == 4 and g["all"]["small_gt_instances"] == 100
    assert abs(g["all"]["g_small_geometric_mean"] - want) < 1e-12
    assert abs(g["all"]["small_share_with_g_below_1"] - (1 + 6 + 10 + 0) / 100) < 1e-12
    assert g["all"]["slope_minimum"] == 0.7 and g["all"]["slope_maximum"] == 1.5 and g["all"]["identity_images"] == 2
    assert g["strong_augmentation_epochs"]["updates"] == 3 and g["strong_augmentation_epochs"]["small_gt_instances"] == 40
    assert g["last_three_epochs"]["updates"] == 2 and g["last_three_epochs"]["small_gt_instances"] == 60
    assert abs(g["last_three_epochs"]["g_small_mean_of_update_values"] - (1.1 + 1.25) / 2) < 1e-12
    assert RD.geometry([])["all"]["g_small_geometric_mean"] is None
    with pytest.raises(ValueError):
        RD.geometry([(61, r(3, None, None))])


def test_endpoint_summary_is_the_mean_and_sample_sd_of_three_seeds():
    values, _, _ = world(synthetic_w("1", GAIN), synthetic_w("2", GAIN))
    t = RD.endpoint_summary(values)
    assert sorted(t) == ["FG", "L0", "W"]
    cell = t["L0"]["600"]["APS"]
    v = [100 * L0[s]["600"]["APS"] for s in ("0", "1", "2")]
    assert cell["values"] == v and cell["n"] == 3 and abs(cell["mean"] - sum(v) / 3) < 1e-12
    assert abs(cell["sample_sd"] - math.sqrt(sum((x - sum(v) / 3) ** 2 for x in v) / 2)) < 1e-12
    assert t["W"]["600"]["AP"]["values"][0] == 100 * W0["600"]["AP"]
    with pytest.raises(ValueError):
        RD.endpoint_summary({"0": values["0"], "1": values["1"]})
