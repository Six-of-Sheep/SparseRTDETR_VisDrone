"""R27-002 four-endpoint timing: the pure aggregation and the fixed counting constants (CPU, synthetic numbers, no model)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "r27_l0"))
import l0_common as K  # noqa: E402
import l0_cost as T  # noqa: E402

# per-pass totals (model ms, end-to-end s) per endpoint, chosen so that the median is not the mean and not the first pass
MODEL = {"A0": [100., 400., 200.], "L0": [210., 190., 800.], "P": [900., 310., 290.], "S15": [410., 390., 1000.]}
END = {"A0": [2., 1., 9.], "L0": [3., 30., 2.5], "P": [4., 3.5, 40.], "S15": [50., 4.5, 5.]}


def passes(model=MODEL, end=END):
    out = []
    for p, a in T.ORDER:
        out.append({"pass": p, "endpoint": a, "model_gpu_ms_total": model[a][p], "end_to_end_s_total": end[a][p],
                    "load_decode_resize_s_total": 1. + p, "H2D_ms_total": 10. * (p + 1), "rank_s_total": .1 * (p + 1),
                    "peak_allocated": 1000 + p})
    return out


def test_fixed_constants():
    assert K.ROTATIONS == (("A0", "L0", "P", "S15"), ("P", "S15", "A0", "L0"), ("L0", "A0", "S15", "P"))
    assert (K.COST_BATCH, K.COST_WARMUP, K.COST_PASSES, K.COST_BATCHES_PER_PASS) == (4, 2, 3, 16)
    assert K.COST_PER_ENDPOINT == 2 + 3 * 16 == 50 and len(K.COST_ENDPOINTS) * K.COST_PER_ENDPOINT == 200
    assert T.IMAGES == 64 and set(T.CANVAS) == set(K.COST_ENDPOINTS) == {"A0", "L0", "P", "S15"}
    assert T.CANVAS == {"A0": "L", "L0": "H", "P": "H", "S15": "H"}
    assert all(sorted(r) == sorted(K.COST_ENDPOINTS) for r in K.ROTATIONS)
    assert T.ORDER == ((0, "A0"), (0, "L0"), (0, "P"), (0, "S15"), (1, "P"), (1, "S15"), (1, "A0"), (1, "L0"),
                       (2, "L0"), (2, "A0"), (2, "S15"), (2, "P"))
    assert K.WORKLOAD["cost4"]["forwards"] == 200 and K.WORKLOAD["cost4"]["rotations"] == [list(r) for r in K.ROTATIONS]


def test_six_ratios_are_the_fixed_pairs():
    assert T.PAIRS == (("L0", "A0"), ("P", "A0"), ("S15", "A0"), ("P", "L0"), ("P", "S15"), ("L0", "S15"))
    r = T.ratios_of({"A0": 1., "L0": 2., "P": 4., "S15": 8.}, {"A0": 10., "L0": 10., "P": 5., "S15": 20.})
    assert r["model_gpu"] == {"L0/A0": 2., "P/A0": 4., "S15/A0": 8., "P/L0": 2., "P/S15": .5, "L0/S15": .25}
    assert r["end_to_end"] == {"L0/A0": 1., "P/A0": .5, "S15/A0": 2., "P/L0": .5, "P/S15": .25, "L0/S15": .5}


def test_median_of_pass_totals_then_ratio():
    med, ratios, per_pass = T.aggregate(passes())
    assert {a: med[a]["model_gpu_ms_median_pass"] for a in med} == {"A0": 200., "L0": 210., "P": 310., "S15": 410.}
    assert {a: med[a]["end_to_end_s_median_pass"] for a in med} == {"A0": 2., "L0": 3., "P": 4., "S15": 5.}
    assert med["P"]["model_gpu_ms_per_pass"] == MODEL["P"] and med["S15"]["end_to_end_s_per_pass"] == END["S15"]
    assert med["A0"]["per_image_model_gpu_ms"] == 200. / 64 and med["A0"]["peak_allocated"] == 1002
    assert med["L0"]["load_decode_resize_s_median_pass"] == 2. and med["L0"]["H2D_ms_median_pass"] == 20.
    assert med["L0"]["rank_s_median_pass"] == pytest.approx(.2)
    # the ratio is taken between medians (not the median of per-pass ratios, not the ratio of means)
    assert ratios["model_gpu"]["P/S15"] == 310. / 410. and ratios["model_gpu"]["L0/A0"] == 210. / 200.
    assert ratios["end_to_end"]["P/A0"] == 2. and ratios["end_to_end"]["L0/S15"] == 3. / 5.
    assert set(ratios["model_gpu"]) == set(ratios["end_to_end"]) == {f"{a}/{b}" for a, b in T.PAIRS}
    assert len(per_pass) == 3 and per_pass[0]["model_gpu"]["P/S15"] == 900. / 410.
    assert per_pass[1]["end_to_end"]["L0/A0"] == 30. and per_pass[2]["model_gpu"]["S15/A0"] == 5.
    assert ratios["model_gpu"]["P/S15"] != sorted(p["model_gpu"]["P/S15"] for p in per_pass)[1]


def test_pass_totals_sum_the_batch_records():
    recs = [{"load_decode_resize_cpu_s": 1., "H2D_ms": 2., "model_gpu_ms": 3., "D2H_after_model_s": 9., "rank_top600_cpu_s": 4.,
             "end_to_end_s": 5.}] * 16
    assert T.pass_totals(recs) == {"model_gpu_ms_total": 48., "end_to_end_s_total": 80., "load_decode_resize_s_total": 16.,
                                   "H2D_ms_total": 32., "rank_s_total": 64.}


def test_wrong_rotation_or_missing_pass_is_refused():
    p = passes()
    p[4], p[5] = p[5], p[4]
    with pytest.raises(ValueError, match="rotations"):
        T.aggregate(p)
    with pytest.raises(ValueError, match="rotations"):
        T.aggregate(passes()[:-1])


def test_module_import_is_light():
    assert "torch" not in T.__dict__ and set(T.LOADERS) == set(K.COST_ENDPOINTS)
    assert "不覆盖原 R27 的终态成本与 E 判据" in T.NOTE and "不从 P 的内部耗时推算 A0" in T.NOTE
