"""R30-001 diagnostic (TASK_R30_DIAG_001): pure CPU checks of the table arithmetic; every number below is synthetic."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "tools" / "r28"), str(Path(__file__).resolve().parents[1] / "src")]
import r30_diag as D  # noqa: E402


def test_bins_match_the_registered_edges():
    assert [D.bin_index(g) for g in (0.2, 0.999, 1.0, 1.19, 1.2, 1.39, 1.4, 3.0)] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert D.BIN_NAMES == ("g<1.0", "1.0<=g<1.2", "1.2<=g<1.4", "g>=1.4") and D.THRESHOLDS == (0.5, 0.75)


def test_area_groups_partition_the_small_rows():
    m = D.area_masks([1.0, 255.9, 256.0, 1024.0])
    assert m["area<16^2"].tolist() == [True, True, False, False] and m["16^2<=area<=32^2"].tolist() == [False, False, True, True]
    assert m["small"].all()
    with pytest.raises(ValueError):
        D.area_masks([1025.0])


def test_bin_table_counts_and_recall():
    g = [0.9, 0.9, 1.1, 1.5]
    area = [100.0, 400.0, 100.0, 400.0]
    iou = {"A": np.array([0.6, 0.4, 0.8, 0.76]), "B": np.array([0.4, 0.4, 0.5, 0.2])}
    rows = D.bin_table(g, area, iou, pairs=(("A", "B"),))
    assert len(rows) == 5 * 3
    by = {(r["bin"], r["area_group"]): r for r in rows}
    r = by[("g<1.0", "small")]
    assert r["n"] == 2 and r["recall@0.5"] == {"A": 0.5, "B": 0.0} and r["difference_pp@0.5"]["A-B"] == 50.0
    assert by[("g<1.0", "area<16^2")]["n"] == 1 and by[("g<1.0", "16^2<=area<=32^2")]["n"] == 1
    assert by[("1.2<=g<1.4", "small")]["n"] == 0 and by[("1.2<=g<1.4", "small")]["recall@0.5"]["A"] is None
    a = by[("all", "small")]
    assert a["n"] == 4 and a["recall@0.75"] == {"A": 0.5, "B": 0.0} and a["recall@0.5"]["B"] == 0.25
    for group in D.AREA_GROUPS:
        assert sum(by[(b, group)]["n"] for b in D.BIN_NAMES) == by[("all", group)]["n"]


def test_miss_table_shares():
    g = [0.9, 0.9, 1.1, 1.5]
    area = [100.0, 400.0, 100.0, 400.0]
    other = {0.5: [True, True, True, False], 0.75: [False] * 4}
    fg = {0.5: [False, True, False, True], 0.75: [False] * 4}
    t = D.miss_table(g, area, other, fg)
    rows = {(r["bin"], r["area_group"]): r for r in t["IoU>=0.5"]}
    assert rows[("all", "small")]["other_hit_FG_miss"] == 2 and rows[("all", "small")]["FG_hit_other_miss"] == 1
    assert rows[("g<1.0", "small")]["other_hit_FG_miss"] == 1 and rows[("g<1.0", "small")]["share_of_all_such_GT"] == 0.5
    assert rows[("g<1.0", "small")]["share_of_bin"] == 0.5 and rows[("g<1.0", "small")]["bin_share_of_GT"] == 0.5
    assert rows[("g>=1.4", "small")]["FG_hit_other_miss"] == 1
    assert all(r["share_of_all_such_GT"] is None for r in t["IoU>=0.75"])
    assert sum(rows[(b, "small")]["other_hit_FG_miss"] for b in D.BIN_NAMES) == rows[("all", "small")]["other_hit_FG_miss"]


def test_geometry_reading_and_cross_bins():
    g = np.array([0.5, 2.0, 1.0, 1.3])
    area = np.array([100.0, 100.0, 5000.0, 20000.0])
    r = D.geometry_reading(g, area)
    assert r["small"]["n"] == 2 and abs(r["small"]["g_geomean"] - 1.0) < 1e-12 and r["small"]["frac_g_lt_1"] == 0.5
    assert r["small"]["bin_counts"] == [1, 0, 0, 1] and r["medium"]["n"] == 1 and r["large"]["n"] == 1
    assert r["small_16^2<=area<=32^2"]["n"] == 0 and "g_geomean" not in r["small_16^2<=area<=32^2"]
    x = D.cross_bins([0.9, 0.9, 1.5], [1.5, 0.9, 1.5])
    assert x[0][0] == 1 and x[0][3] == 1 and x[3][3] == 1 and sum(map(sum, x)) == 3


def test_records_equal_reports_the_first_difference():
    a = [{"image_id": "x", "score": 0.5, "bbox_xyxy": [0.0, 1.0, 2.0, 3.0]}, {"image_id": "y", "score": 0.25, "bbox_xyxy": [0.0] * 4}]
    assert D.records_equal(a, [dict(r) for r in a]) is None
    b = [dict(a[0]), {**a[1], "score": 0.250001}]
    assert D.records_equal(a, b)["index"] == 1
    assert D.records_equal(a, a[:1])["lengths"] == [2, 1]


def test_batch_order_is_the_evaluation_call_order():
    ids = [f"i{k}" for k in range(16)]
    order = D.batch_order(ids, 2, 4)
    assert order[:2] == [ids[0:4], ids[4:8]] and order[2:] == [ids[0:4], ids[4:8], ids[8:12], ids[12:16]]
    with pytest.raises(ValueError):
        D.batch_order(ids[:6], 0, 2)


def test_oracle_context_serves_the_planned_batches_only():
    from sparse_rtdetr.research import r28_fovea_geo as FG
    from sparse_rtdetr.research import r28_resample as RS
    maps = {f"i{k}": torch.full((12, 21), float(k), dtype=torch.float64) for k in range(8)}
    order = D.batch_order(list(maps), 1, 2)
    ctx = D.make_oracle_context(RS.Context, maps, order, FG.WIDTH)
    assert isinstance(ctx, RS.Context) and vars(ctx)["r28_fwhm"].hex() == FG.WIDTH_HEX and ctx.net is None
    x = torch.zeros(4, 3, 8, 8)
    d = [ctx.demand(x) for _ in range(3)]
    assert [float(t[0, 0, 0]) for t in d] == [0.0, 0.0, 4.0] and d[2].dtype == torch.float64 and tuple(d[2].shape) == (4, 12, 21)
    assert [float(d[2][j, 0, 0]) for j in range(4)] == [4.0, 5.0, 6.0, 7.0] and ctx.calls == 3
    with pytest.raises(ValueError):
        ctx.demand(x)
    ctx2 = D.make_oracle_context(RS.Context, maps, order, FG.WIDTH)
    with pytest.raises(ValueError):
        ctx2.demand(torch.zeros(3, 3, 8, 8))


def test_label_demand_plan_differs_from_uniform_only_where_there_is_demand():
    from sparse_rtdetr.research import r28_fovea_geo as FG
    d = torch.zeros(2, 12, 21, dtype=torch.float64)
    d[1, 2, 3] = 5.0
    plan = FG.plan_from_demand(d)
    assert plan.identity.tolist() == [True, False]


def test_difference_pp_and_pins():
    x = {b: {m: 0.25 for m in D.METRICS} for b in ("600", "300")}
    y = {b: {m: 0.24 for m in D.METRICS} for b in ("600", "300")}
    assert abs(D.difference_pp(x, y)["600"]["APS"] - 1.0) < 1e-9
    assert D.GT_COUNTS == {"GT_before": 38759, "GT_after": 38599} and D.SMALL_GT == 26441
    assert D.REPORT_W["g<1.0"] == (3039, 79.53, 72.36) and D.REPORT_W["all"] == (26441, 82.73, 83.23)
    assert set(D.ALLOWED_ADDED) == {"tools/r28/r30_diag.py", "tests/test_r30_diag.py"}
