"""R28-K follow-up (TASK_R28_K_FOLLOWUP_003): pure CPU checks of the adapter; every number below is synthetic.

Covered: the follow-up block needs the registered authorization basis; stages of the closed round are read from pinned
receipts bound to the prior contract and never from the new round; the release difference is limited to the adapter
files; the per-class APM decomposition sums to the aggregate; the coverage table counts gained / lost GT; the summary
arithmetic keeps the closed W decision.
"""
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "r28"))
import r28_common as K  # noqa: E402
import r28_eval as E  # noqa: E402
import r28_followup as FU  # noqa: E402

BASIS = K.FOLLOWUP_BASIS


def test_followup_block_requires_the_registered_basis():
    assert K.followup({"version": "R28-001-contract-v1"}) is None
    good = {"authorization_basis": BASIS, "followup": {"authorization_basis": BASIS}}
    assert K.followup(good) is good["followup"]
    for bad in ({"followup": {"authorization_basis": BASIS}},
                {"authorization_basis": "other", "followup": {"authorization_basis": BASIS}},
                {"authorization_basis": BASIS, "followup": {"authorization_basis": "other"}}):
        with pytest.raises(ValueError):
            K.followup(bad)
    with pytest.raises(ValueError):
        K.prior_contract({"version": "R28-001-contract-v1"})


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return K.ref(path)


def test_inherited_stage_is_read_from_the_prior_round_only(tmp_path, monkeypatch):
    prior, new = tmp_path / "prior", tmp_path / "new"
    new.mkdir()
    monkeypatch.setattr(K, "ROUND", new)
    ref = _write(prior / "g0" / "CLOSED.json", {"status": "PASS", "contract_sha256": "old", "arm": "W", "G0_passed": True})
    c = {"contract_sha256": "new", "authorization_basis": BASIS,
         "followup": {"authorization_basis": BASIS, "prior_contract_sha256": "old", "inherited": {"g0": ref}}}
    assert K.stage_closed(c, "g0")["G0_passed"] is True
    assert K.g0_passed(c)["G0_passed"] is True
    with pytest.raises(ValueError):                      # an inherited receipt of another arm
        K.stage_closed(c, "g0", "K")
    with pytest.raises(ValueError):                      # a new-round stage that is not closed
        K.stage_closed(c, "accept-K")
    _write(new / "accept-K" / "CLOSED.json", {"status": "PASS", "contract_sha256": "new", "arm": "K"})
    assert K.stage_closed(c, "accept-K", "K")["arm"] == "K"
    _write(new / "accept-K2" / "CLOSED.json", {"status": "PASS", "contract_sha256": "old", "arm": "K"})
    with pytest.raises(ValueError):                      # a new-round receipt bound to the prior contract
        K.stage_closed(c, "accept-K2")
    (new / "g0").mkdir()
    with pytest.raises(ValueError):                      # an inherited name must not exist in the new round
        K.stage_closed(c, "g0")
    (new / "g0").rmdir()
    (prior / "g0" / "CLOSED.json").write_text(json.dumps({"status": "PASS", "contract_sha256": "old", "arm": "W", "G0_passed": False}))
    with pytest.raises(ValueError):                      # the pinned receipt changed after prepare
        K.stage_closed(c, "g0")
    with pytest.raises(ValueError):                      # no W formal directory in a follow-up round
        K.formal_dir(c, "W")


def test_source_delta_is_limited_to_the_adapter():
    old = {"src/a.py": "1", "tools/r28/r28_common.py": "2", "tools/r28/r28_cost.py": "3", "tests/t.py": "4"}
    new = {**old, "tools/r28/r28_common.py": "2b", "tools/r28/r28_followup.py": "5"}
    d = FU.source_delta(old, new)
    assert d["changed"] == ["tools/r28/r28_common.py"] and d["added"] == ["tools/r28/r28_followup.py"] and d["removed"] == []
    assert d["unchanged_files"] == 3 and d["science_tree_identical"] is True
    for bad in ({**new, "src/a.py": "x"},                                  # science file changed
                {**new, "src/b.py": "x"},                                  # science file added
                {**new, "tools/r28/r28_cost.py": "x"},                     # a tool outside the adapter list changed
                {k: v for k, v in new.items() if k != "tests/t.py"}):     # a file removed
        with pytest.raises(ValueError):
            FU.source_delta(old, bad)


def _apm(values):
    return [{"category_id": i + 1, "GT": 100 + i, "APM": v} for i, v in enumerate(values)]


def test_apm_decomposition_sums_to_the_aggregate():
    l0 = [0.30 + 0.01 * i for i in range(10)]
    w = [v - 0.02 if i < 3 else v + 0.005 for i, v in enumerate(l0)]
    k = [v + 0.001 for v in l0]
    tables = {"L0": _apm(l0), "W": _apm(w), "K": _apm(k)}
    agg = {a: sum(r["APM"] for r in tables[a]) / 10 for a in tables}
    d = FU.apm_decomposition(tables, agg)
    assert d["medium_GT_total"] == sum(100 + i for i in range(10)) and len(d["classes"]) == 10
    t = d["totals"]["W-L0"]
    assert t["APM600_difference_pp"] == pytest.approx(100 * (3 * -0.02 + 7 * 0.005) / 10)
    assert t["sum_of_contributions_pp"] == pytest.approx(t["APM600_difference_pp"], abs=1e-12)
    assert t["negative_contribution_classes"] == [1, 2, 3]
    assert d["classes"][0]["W-L0"]["difference_pp"] == pytest.approx(-2.0)
    assert d["classes"][0]["W-L0"]["contribution_pp"] == pytest.approx(-0.2)
    assert d["totals"]["K-L0"]["negative_contribution_classes"] == []
    assert d["totals"]["W-K"]["APM600_difference_pp"] == pytest.approx(t["APM600_difference_pp"] - 0.1)
    with pytest.raises(ValueError):                      # an aggregate that is not the macro mean
        FU.apm_decomposition(tables, {**agg, "W": agg["W"] + 1e-6})
    broken = copy.deepcopy(tables)
    broken["K"][4]["APM"] = None
    with pytest.raises(ValueError):                      # a class without a medium AP
        FU.apm_decomposition(broken, agg)
    other = copy.deepcopy(tables)
    other["W"][0]["GT"] += 1
    with pytest.raises(ValueError):                      # lists scored against different GT
        FU.apm_decomposition(other, agg)


def test_size_rows_use_inclusive_coco_ranges():
    gt = [{"area": a} for a in (10.0, 1024.0, 1024.5, 9216.0, 9216.5)]
    assert [r["area"] for r in FU.size_rows(gt, "small")] == [10.0, 1024.0]
    assert [r["area"] for r in FU.size_rows(gt, "medium")] == [1024.0, 1024.5, 9216.0]


def test_coverage_table_counts_gained_and_lost():
    import numpy as np
    box = lambda k: [100.0 * k, 0.0, 100.0 * k + 50.0, 50.0]              # 50 x 50 = medium
    loose = lambda k: [100.0 * k, 0.0, 100.0 * k + 50.0, 32.0]            # IoU 0.64: covered at 0.5 only
    gs = (0.9, 0.95, 1.1, 1.3, 1.5)
    gt = [{"image": "a", "category_id": 1, "area": 2500.0, "predicted": g, "prior": 1.1, "bbox_xyxy": box(k)}
          for k, g in enumerate(gs)]
    idx = {"L0": {"a": {1: np.array([box(0), box(1), box(2)])}},
           "W": {"a": {1: np.array([box(0), loose(2), box(3), box(4)]), 2: np.array([box(1)])}},   # wrong class never covers
           "K": {"a": {1: np.array([box(0), box(1), box(2), box(3)])}}}
    rows = FU.size_rows(gt, "medium")
    iou = FU.best_ious(rows, idx)
    t = FU.coverage_table(rows, iou, "predicted")
    assert [r["bin"] for r in t] == ["g<1.0", "1.0<=g<1.2", "1.2<=g<1.4", "g>=1.4", "all"] and [r["n"] for r in t] == [2, 1, 1, 1, 5]
    low = t[0]["coverage@0.5"]
    assert low["covered"] == {"L0": 2, "W": 1, "K": 2} and low["coverage"]["W"] == 0.5
    assert low["W-L0"] == {"difference_pp": -50.0, "gained": 0, "lost": 1} and low["K-L0"] == {"difference_pp": 0.0, "gained": 0, "lost": 0}
    mid5, mid75 = t[1]["coverage@0.5"], t[1]["coverage@0.75"]
    assert mid5["covered"]["W"] == 1 and mid75["covered"]["W"] == 0 and mid75["W-L0"]["lost"] == 1
    total = t[4]["coverage@0.5"]
    assert total["covered"] == {"L0": 3, "W": 4, "K": 4}
    assert total["W-L0"] == {"difference_pp": pytest.approx(20.0), "gained": 2, "lost": 1}
    assert total["W-K"] == {"difference_pp": 0.0, "gained": 1, "lost": 1}
    by_prior = FU.coverage_table(rows, iou, "prior")                       # K's own g: one bin holds every GT
    assert [r["n"] for r in by_prior] == [0, 5, 0, 0, 5] and by_prior[0]["coverage@0.5"]["coverage"]["W"] is None
    assert by_prior[1]["coverage@0.5"] == total


def test_registered_identities_and_adapter_lists():
    assert FU.PIN["commit"] == "59e367de86cb55dbe1fe6606962d4f0bf65458e3" and FU.PIN["parent"] == K.PARENT_SHA
    assert FU.W_DECISION == {"label_key": "closed", "K_trigger": False, "exploratory": False, "label": "closed_fixed_config"}
    assert not any(p.startswith("src/") for p in FU.ALLOWED_CHANGED + FU.ALLOWED_ADDED)
    assert FU.WORKLOAD["planned_complete_detector_forwards"] == 36777 and FU.WORKLOAD["planned_updates"] == 18243
    assert set(FU.INHERITED).isdisjoint({"followup-check", "accept-K", "formal-K", "eval-K", "cost-K", "medium", "summary-K"})


def test_original_rule_still_closes_this_w_shape():
    """The decision rule is untouched: an APM-only failure does not trigger K (the follow-up never calls it a trigger)."""
    cells = {k: {"passed": k != "top600_APM"} for k in ("top600_APS", "top600_AP75_small", "top600_AP", "top300_AP",
                                                         "top300_APS", "top600_APM", "top600_APL")}
    assert E.decide_m(cells, 1.57, 1.10) == {"label_key": "closed", "K_trigger": False, "exploratory": False}
