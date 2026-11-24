"""TH2 C0 (TASK_TH2_C0_001): the pure pieces and the stage functions on synthetic arrays.  No data file, no model,
no scorer.  The numbers here are synthetic and say nothing about VisDrone."""
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "r28"))
import c0_ref as R  # noqa: E402
import th2_c0 as C  # noqa: E402

B = C.B

M = 548


def world(seed=0, empty=0):
    """Per-GT arrays of a synthetic development set; the last `empty` images of the count order have no small GT."""
    rng = np.random.default_rng(seed)
    cnt = rng.poisson(rng.gamma(1.0, 60.0, M))
    cnt[:3] = 0
    image = np.repeat(np.arange(M), cnt)
    n = len(image)
    kind = rng.choice(3, size=n, p=(0.7, 0.25, 0.05))
    area = np.where(kind == 0, rng.uniform(40.0, 1024.0, n), np.where(kind == 1, rng.uniform(1025.0, 9215.0, n), rng.uniform(9216.0, 40000.0, n)))
    area[rng.random(n) < 0.01] = 1024.0                       # the closed edge of small
    if empty:
        poor = np.argsort(R.per_image(area <= 1024.0, image, M), kind="stable")[:empty]
        area[np.isin(image, poor) & (area <= 1024.0)] = 2000.0
    base = rng.random(n)
    iou = {}
    for arm, shift in (("L0", 0.0), ("FG", 0.02), ("S15", 0.05)):
        for s in C.SEEDS:
            iou[f"{arm}-s{s}"] = np.clip(base + shift + 0.08 * rng.standard_normal(n), 0.0, 1.0)
    small = R.per_image(area <= 1024.0, image, M)
    demand = rng.gamma(0.4, 1.0, (M, 12, 21)) * (1.0 + small / 40.0)[:, None, None]
    d_star = rng.gamma(0.4, 1.0, (M, 12, 21)) * (0.2 + small / 20.0)[:, None, None]
    return {"image": image, "area": area, "iou": iou, "demand": demand, "d_star": d_star}


def brute(inp, route, cheap, expensive, mask, thr):
    """Routed hit rate (pp) GT by GT."""
    hit = np.where(route[inp["image"]], inp["iou"][expensive] >= thr, inp["iou"][cheap] >= thr)
    return 100.0 * hit[mask].sum() / mask.sum()


@pytest.fixture(scope="module")
def tables():
    inp = world(1)
    return inp, C.hit_tables(inp)


def test_reference_is_the_controllers_file_and_agrees_with_the_task_text(capsys):
    assert B.file_sha(ROOT / C.REF[0]) == C.REF[1]
    R.selftest()
    assert "selftest PASS" in capsys.readouterr().out
    text = C.text_against_reference()
    assert text["subset_sizes"] == [156] * 3 + [157] * 4
    assert C.SCORINGS == 39 and len(C.SCORED) == 11
    assert np.array_equal(R.boot_weights(M), C.T.boot_weights(M))


def test_splice_takes_whole_images_in_development_order():
    ids = [f"img{i}" for i in (3, 0, 2, 1)]

    def records(tag, order):
        return [{"image_id": i, "query_id": q, "category_id": 1 + q % 10, "score": 1.0 - q / 600, "bbox_xyxy": [q, q, q + 5, q + 5], "arm": tag}
                for i in order for q in range(600)]

    cheap, expensive = records("L", ids), records("H", [ids[2], ids[0], ids[3], ids[1]])
    lo, hi = C.by_image(cheap, ids), C.by_image(expensive, ids)
    assert C.splice(ids, np.zeros(4, bool), lo, hi) == cheap
    full = C.splice(ids, np.ones(4, bool), lo, hi)
    assert full != expensive and sorted(map(json.dumps, full)) == sorted(map(json.dumps, expensive))
    assert [r["image_id"] for r in full[::600]] == ids
    mixed = C.splice(ids, np.array([True, False, False, True]), lo, hi)
    assert [(r["image_id"], r["arm"]) for r in mixed[::600]] == [(ids[0], "H"), (ids[1], "L"), (ids[2], "L"), (ids[3], "H")]
    assert all(mixed[600 * j:600 * j + 600] == (hi if t == "H" else lo)[ids[j]] for j, t in enumerate("HLLH"))
    assert [r["query_id"] for r in mixed[:600]] == list(range(600))
    with pytest.raises(ValueError):
        C.by_image(cheap[:-1], ids)
    with pytest.raises(ValueError):
        C.by_image(cheap[1:] + cheap[:1], ids)
    with pytest.raises(ValueError):
        C.by_image(cheap, ids[:3] + ["other"])
    with pytest.raises(ValueError):
        C.splice(ids, np.zeros(3, bool), lo, hi)


def test_assignments():
    inp = world(2)
    res = C.hit_tables(inp)
    a = res["_assignments"]
    assert set(a) == {f"P-k{k}" for k in R.K_CURVE} | {"N-k158"} | {f"RAND-r{r}" for r in range(7)}
    assert all(int(a[f"P-k{k}"].sum()) == k for k in R.K_CURVE) and int(a["N-k158"].sum()) == 158
    ks = list(R.K_CURVE)
    assert all(not (a[f"P-k{x}"] & ~a[f"P-k{y}"]).any() for x, y in zip(ks, ks[1:]))          # nested
    assert (sum(a[f"RAND-r{r}"].astype(int) for r in range(7)) == 2).all()
    t = inp["demand"].sum(axis=(1, 2))
    assert t[a["P-k158"]].min() >= t[~a["P-k158"]].max()
    n = R.per_image(inp["area"] <= 1024.0, inp["image"], M)
    assert n[a["N-k158"]].min() >= n[~a["N-k158"]].max()
    assert np.array_equal(res["_columns"]["T"], t) and np.array_equal(res["_columns"]["N"], n)
    assert np.array_equal(res["_columns"]["P_k219"], a["P-k219"].astype(float))


def test_hit_tables_against_brute_force(tables):
    inp, res = tables
    masks = C.group_masks(inp["area"])
    a = res["_assignments"]
    for s in C.SEEDS:
        for k in R.K_CURVE:
            row = res["T1"]["P"][f"k{k}"][f"s{s}"]
            want = brute(inp, a[f"P-k{k}"], f"L0-s{s}", f"S15-s{s}", masks["small"], 0.75)
            assert abs(row["routed_pp"] - want) < 1e-9
            lo = brute(inp, np.zeros(M, bool), f"L0-s{s}", f"S15-s{s}", masks["small"], 0.75)
            hi = brute(inp, np.ones(M, bool), f"L0-s{s}", f"S15-s{s}", masks["small"], 0.75)
            ell = np.log(1 + 1.25 * k / M) / np.log(2.25)
            assert abs(row["cheap_pp"] - lo) < 1e-9 and abs(row["expensive_pp"] - hi) < 1e-9
            assert abs(row["Q_pp"] - (want - (lo + ell * (hi - lo)))) < 1e-9
            assert abs(row["capture"] - (want - lo) / (hi - lo)) < 1e-9
            assert abs(row["gt_share"] - masks["small"][a[f"P-k{k}"][inp["image"]]].sum() / masks["small"].sum()) < 1e-12
            six = res["T6"]["P"][f"k{k}"][f"s{s}"]
            assert abs(six["routed_pp"] - brute(inp, a[f"P-k{k}"], f"FG-s{s}", f"S15-s{s}", masks["small"], 0.75)) < 1e-9
        row = res["T1"]["N"]["k158"][f"s{s}"]
        assert abs(row["routed_pp"] - brute(inp, a["N-k158"], f"L0-s{s}", f"S15-s{s}", masks["small"], 0.75)) < 1e-9
        for label, group, thr in C.T4_ROWS:
            for k in (158, 219, 341):
                row = res["T4"]["P"][f"k{k}"][label][f"s{s}"]
                assert abs(row["routed_pp"] - brute(inp, a[f"P-k{k}"], f"L0-s{s}", f"S15-s{s}", masks[group], thr)) < 1e-9
            row = res["T4"]["N"]["k158"][label][f"s{s}"]
            assert abs(row["routed_pp"] - brute(inp, a["N-k158"], f"L0-s{s}", f"S15-s{s}", masks[group], thr)) < 1e-9
    mean = res["T1"]["P"]["k158"]["mean"]
    per = [res["T1"]["P"]["k158"][f"s{s}"] for s in C.SEEDS]
    assert mean["k"] == 158 and abs(mean["Q_pp"] - np.mean([p["Q_pp"] for p in per])) < 1e-12
    assert res["gate"]["q_hit"] == [p["Q_pp"] for p in per] and res["gate"]["lead_hit"] == [p["lead_pp"] for p in per]
    assert res["gate"]["q_hit_count"] == [res["T1"]["N"]["k158"][f"s{s}"]["Q_pp"] for s in C.SEEDS]
    assert max(abs(v) for v in res["hit_level_mixing_pp"].values()) < 1e-9 and len(res["hit_level_mixing_pp"]) == 12
    ci = res["T1"]["P_k158_Q_ci95"]
    assert all(ci[k][0] < res["T1"]["P"]["k158"][k]["Q_pp"] < ci[k][1] for k in ("s0", "s1", "s2", "mean"))
    assert ci["mean"][1] - ci["mean"][0] <= max(ci[f"s{s}"][1] - ci[f"s{s}"][0] for s in C.SEEDS) + 1e-12


def test_oracle_cross_seed_and_pairings(tables):
    inp, res = tables
    small = C.group_masks(inp["area"])["small"]
    d = {s: R.per_image(((inp["iou"][f"S15-s{s}"] >= 0.75).astype(float) - (inp["iou"][f"L0-s{s}"] >= 0.75)) * small, inp["image"], M)
         for s in C.SEEDS}
    for s in C.SEEDS:
        for k in R.K_CURVE:
            o = res["T2"]["O"][f"k{k}"][f"s{s}"]["capture"]
            assert abs(o - np.sort(d[s])[::-1][:k].sum() / d[s].sum()) < 1e-9
            assert o >= res["T1"]["P"][f"k{k}"][f"s{s}"]["capture"] - 1e-12 and o >= res["T1"]["N"][f"k{k}"][f"s{s}"]["capture"] - 1e-12
            for t in C.SEEDS:
                if t != s:
                    x = res["T2"]["X"][f"s{t}->s{s}"][f"k{k}"]["capture"]
                    assert abs(x - d[s][R.order_desc(d[t])[:k]].sum() / d[s].sum()) < 1e-9 and x <= o + 1e-12
    assert len([k for k in res["T2"]["X"] if k != "mean"]) == 6
    assert abs(res["T2"]["X"]["mean"]["k158"]["capture"]
               - np.mean([v["k158"]["capture"] for k, v in res["T2"]["X"].items() if k != "mean"])) < 1e-12
    pair = res["T2"]["P_k158_Q_pairings"]
    assert len(pair) == 9
    for s in C.SEEDS:
        assert pair[f"L0-s{s}|S15-s{s}"]["Q_pp"] == res["T1"]["P"]["k158"][f"s{s}"]["Q_pp"]
    a = res["_assignments"]["P-k158"]
    want = brute(inp, a, "L0-s0", "S15-s2", small, 0.75)
    assert abs(pair["L0-s0|S15-s2"]["routed_pp"] - want) < 1e-9


def test_rank_correlation_is_spearman(tables):
    stats = pytest.importorskip("scipy.stats")
    inp, res = tables
    rng = np.random.default_rng(5)
    x, y = rng.integers(0, 6, 300).astype(float), rng.integers(0, 4, 300).astype(float)        # heavy ties
    assert abs(C.rank_correlation(x, y) - stats.spearmanr(x, y)[0]) < 1e-12
    assert np.array_equal(C.avg_rank([3.0, 1.0, 3.0, 2.0]), [3.5, 1.0, 3.5, 2.0])
    assert C.rank_correlation(np.ones(5), np.arange(5.0)) is None
    t, n = inp["demand"].sum(axis=(1, 2)), R.per_image(inp["area"] <= 1024.0, inp["image"], M)
    assert abs(res["T2"]["rank_correlation"]["T_N"] - stats.spearmanr(t, n)[0]) < 1e-12
    assert set(res["T2"]["rank_correlation"]) == {"d_s_d_t", "T_N", "T_Dstar", "T_d", "N_d"}
    assert set(res["T2"]["rank_correlation"]["d_s_d_t"]) == {"s0-s1", "s0-s2", "s1-s2"}


def test_group_table(tables):
    inp, res = tables
    small = C.group_masks(inp["area"])["small"]
    n = R.per_image(small, inp["image"], M)
    for name, stat in (("N", n), ("P", inp["demand"].sum(axis=(1, 2)))):
        groups = np.array_split(R.order_desc(stat), 5)
        for s in C.SEEDS:
            rows = res["T3"][name][f"s{s}"]
            assert [r["images"] for r in rows] == [110, 110, 110, 109, 109]
            assert abs(sum(r["share_of_total_lead"] for r in rows) - 1) < 1e-9 and abs(sum(r["small_gt_share"] for r in rows) - 1) < 1e-9
            g = groups[1]
            sel = small & np.isin(inp["image"], g)
            assert abs(rows[1]["L0_hit75_pp"] - 100 * (inp["iou"][f"L0-s{s}"][sel] >= 0.75).mean()) < 1e-9
            assert abs(rows[1]["lead_per_gt_pp"] - 100 * ((inp["iou"][f"S15-s{s}"][sel] >= 0.75).mean() - (inp["iou"][f"L0-s{s}"][sel] >= 0.75).mean())) < 1e-9
        assert abs(res["T3"][name]["mean"][0]["share_of_total_lead"] - np.mean([res["T3"][name][f"s{s}"][0]["share_of_total_lead"] for s in C.SEEDS])) < 1e-12


def test_images_without_small_gt_give_null_rates_and_valid_json():
    res = C.hit_tables(world(3, empty=150))
    last = res["T3"]["N"]["s0"][4]
    assert last["small_gt"] == 0 and last["L0_hit75_pp"] is None and last["lead_per_gt_pp"] is None and last["share_of_total_lead"] == 0
    assert res["T3"]["N"]["mean"][4]["L0_hit75_pp"] is None
    body = {k: v for k, v in res.items() if not k.startswith("_")}
    json.dumps(body, allow_nan=False)
    assert C.clean({"a": float("nan"), "b": [np.float64(1.5), np.int64(2), np.bool_(True), float("inf")]}) == {"a": None, "b": [1.5, 2, True, None]}
    assert res["statistics"]["N"]["exact_ties_between_neighbours"] > 0


def values_world(shift=0.0, frac=None):
    """Scorer values: every routed row sits at cheap + frac(k) * lead + shift."""
    k_of = {"P-k158": 158, "P-k219": 219, "P-k341": 341, "N-k158": 158, **{f"RAND-r{r}": len(s) for r, s in enumerate(R.balanced_subsets(M))}}
    values = {}
    for s in C.SEEDS:
        lo = {m: 0.30 + 0.01 * j + 0.001 * s for j, m in enumerate(C.METRICS)}
        hi = {m: lo[m] + 0.02 + 0.002 * j for j, m in enumerate(C.METRICS)}
        v = {"L0": lo, "S15": hi}
        for name, k in k_of.items():
            f = (2 / 7 if name.startswith("RAND") else float(R.loglinear(k / M))) if frac is None else frac
            v[name] = {m: lo[m] + f * (hi[m] - lo[m]) + (0.0 if name.startswith("RAND") else shift) for m in C.METRICS}
        values[f"s{s}"] = v
    return values, k_of


def test_ap_tables_and_verdict():
    values, k_of = values_world()
    t5 = C.ap_tables(values, k_of)
    json.dumps(t5, allow_nan=False)
    for s in C.SEEDS:
        row = t5[f"s{s}"]
        assert all(abs(row["routed"]["P-k158"]["Q_pp"][m]) < 1e-12 for m in C.METRICS)
        assert all(abs(row["random"]["mixing_pp"][m]) < 1e-12 and row["random"]["std_pp"][m] < 1e-12 for m in C.METRICS)
        assert abs(row["lead_pp"]["AP"] - 2.0) < 1e-12 and abs(row["lead_pp"]["APL"] - 2.8) < 1e-12
        assert abs(row["routed"]["P-k341"]["minus_chord_pp"]["AP"] - 100 * (float(R.loglinear(341 / M)) - 341 / M) * 0.02) < 1e-12
        assert row["routed"]["RAND-r0"]["k"] in (156, 157)
    assert abs(t5["mean"]["lead_pp"]["APS"] - 2.2) < 1e-12 and t5["mean"]["routed"]["N-k158"]["k"] == 158
    gate = {"q_hit": [1.2, 1.1, 1.3], "lead_hit": [5.0, 5.2, 5.4], "q_hit_count": [1.5, 1.4, 1.6]}
    v = C.verdict_of(gate, t5)
    assert v["label"] == "UNDECIDED" and v["GO"] == {"G1_small_hit75": True, "G2_APS": False, "G3_AP": True}      # Q_APS = 0
    assert abs(v["thresholds_pp"]["G1"] - 1.02) < 1e-12 and abs(v["thresholds_pp"]["G2"] - 0.42) < 1e-12
    up, _ = values_world(shift=0.006)
    v = C.verdict_of(gate, C.ap_tables(up, k_of))
    assert v["label"] == "GO-CANDIDATE" and abs(v["readings_pp"]["means"]["Q_APS"] - 0.6) < 1e-9
    assert v["registered_with_the_label"]["order_N_clears_G1"] is True
    assert abs(v["registered_with_the_label"]["mixing_pp"]["AP"]["mean"]) < 1e-12
    v = C.verdict_of({**gate, "q_hit": [-0.2, 0.1, -0.3]}, C.ap_tables(up, k_of))
    assert v["label"] == "STOP" and v["STOP"]["S1_small_hit75"] is True
    down, _ = values_world(shift=-0.004)
    v = C.verdict_of(gate, C.ap_tables(down, k_of))
    assert v["label"] == "UNDECIDED" and v["GO"]["G3_AP"] is False
    json.dumps(v, allow_nan=False)
    low, _ = values_world(frac=0.0)                                   # random rows at the cheap arm: mixing = -(2/7) lead
    assert abs(C.ap_tables(low, k_of)["s0"]["random"]["mixing_pp"]["AP"] + 100 * 0.02 * 2 / 7) < 1e-12


def close(out, c, result, **counts):
    B.write_new(out / "CLOSED.json", {"status": "PASS", "contract_sha256": c["contract_sha256"], "counts": dict(counts),
                                      "started_at_unix": time.time(), "elapsed_seconds": 1.0, "process_cpu_seconds": 1.0,
                                      "closed_at": "now", **result})


class FakeScorer:
    offset = 0.0

    def __init__(self, c, seed, counts):
        self.c, self.seed, self.counts = c, seed, counts
        self.values, _ = values_world(shift=0.006)

    def score(self, out, name, route):
        assert route.shape == (M,) and route.dtype == np.bool_
        self.counts.update(score_budget_calls=1)
        path = out / f"{name}.top600-metrics.json"
        B.write_new(path, {"row": name, "routed_images": int(route.sum())})
        if name in ("L0", "S15"):
            v = {m: self.c["registered"]["top600"][f"{name}-s{self.seed}"][m] + self.offset for m in C.METRICS}
        else:
            v = self.values[f"s{self.seed}"][name]
        return v, B.ref(path), {"L0": name == "L0", "S15": name == "S15"}


def contract(tmp_path, dev=False):
    return {"outputs": str(tmp_path), "home": str(tmp_path), "dev": dev, "contract_sha256": "0" * 64, "source_commit": "c", "source_branch": "b",
            "task": {}, "reference": {}, "development_ids": [f"{i:064x}" for i in range(M)], "boundaries": C.BOUNDARIES,
            "definitions": C.DEFINITIONS, "verdict_rule": C.VERDICT_RULE,
            "registered": {"top600": {k: dict(zip(C.METRICS, v)) for k, v in C.REG_AP.items()}}}


def run(c, name, fn, *args):
    out = Path(c["outputs"]) / name
    out.mkdir()
    counts = Counter(forwards=0)
    result = fn(c, out, counts, *args)
    close(out, c, result, **counts)
    return result


def test_stage_functions_on_synthetic_arrays(tmp_path, monkeypatch):
    c = contract(tmp_path)
    monkeypatch.setattr(C, "load_inputs", lambda c: world(4))
    monkeypatch.setattr(C, "Scorer", FakeScorer)
    monkeypatch.setattr(C, "GROUP_COUNTS", {g: int(v.sum()) for g, v in C.group_masks(world(4)["area"]).items()})
    (tmp_path / "wire").mkdir()
    close(tmp_path / "wire", c, {"passed": True, "checks": {"x": 1}})
    with pytest.raises(ValueError):
        run(c, "hits", C.hits)                                    # the pure endpoints have not been scored
    (tmp_path / "hits").rmdir()
    for s in C.SEEDS:
        e = run(c, f"ends-s{s}", C.ends, f"s{s}")
        assert e["passed"] and max(e["max_abs_difference_from_registered"].values()) == 0 and e["spliced_list_equals_stored_file"] == {"L0": True, "S15": True}
    h = run(c, "hits", C.hits)
    assert h["routed_images"]["P-k158"] == 158 and len(h["gate"]["q_hit"]) == 3
    lines = (tmp_path / "hits" / "per_image.csv").read_text().split("\n")
    assert len(lines) == M + 2 and lines[-1] == "" and lines[0].startswith("index,image_id,T,N,Dstar,hit75_small_L0_s0")
    assert lines[0].split(",")[-7:] == [f"P_k{k}" for k in R.K_CURVE] and len(lines[1].split(",")) == len(lines[0].split(","))
    z = np.load(tmp_path / "hits" / "assignments.npz")
    assert int(z["P-k341"].sum()) == 341 and z["RAND-r6"].dtype == np.bool_
    col = {k: i for i, k in enumerate(lines[0].split(","))}
    assert sum(int(line.split(",")[col["P_k158"]]) for line in lines[1:-1]) == 158
    for s in C.SEEDS:
        r = run(c, f"routed-s{s}", C.routed, f"s{s}")
        assert list(r["values_600"]) == list(C.SCORED)
    t = run(c, "tables", C.tables)
    res = json.loads((tmp_path / "tables" / "RESULT.json").read_text())
    assert t["label"] == res["verdict"]["label"] and res["verdict"]["label"] in ("STOP", "GO-CANDIDATE", "UNDECIDED")
    assert len(res["metric_files"]) == 39 and res["workload"]["score_budget_calls"] == 39 and res["workload"]["model_forwards"] == 0
    assert set(res) >= {"T1", "T2", "T3", "T4", "T5", "T6", "verdict", "wire", "ends", "boundaries"}
    assert res["T5"]["s1"]["endpoints"]["L0"]["AP"] == C.REG_AP["L0-s1"][0]
    assert res["verdict"]["readings_pp"]["Q_hit"] == h["gate"]["q_hit"]


def test_ends_stops_on_a_value_that_differs_and_dev_rounds_stop_after_wiring(tmp_path, monkeypatch):
    c = contract(tmp_path)
    monkeypatch.setattr(C, "Scorer", FakeScorer)
    monkeypatch.setattr(FakeScorer, "offset", 2e-9)
    (tmp_path / "wire").mkdir()
    close(tmp_path / "wire", c, {"passed": True})
    with pytest.raises(ValueError, match="4.7 item 4"):
        run(c, "ends-s0", C.ends, "s0")
    d = contract(tmp_path / "dev-round-1", dev=True)
    for name, fn, args in (("hits", C.hits, ()), ("routed-s0", C.routed, ("s0",)), ("tables", C.tables, ())):
        with pytest.raises(ValueError, match="dev round"):
            fn(d, tmp_path, Counter(forwards=0), *args)


def test_limits_guard_and_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "WS", tmp_path)
    home = tmp_path / "experiments" / "thesis-study2" / "th2-c0-x"
    home.mkdir(parents=True)
    with pytest.raises(ValueError, match="STORAGE_ESTIMATE"):
        C.limits_guard(home)
    C.storage(home)
    assert [p.name for p in home.iterdir()] == ["STORAGE_ESTIMATE.json"]
    est = json.loads((home / "STORAGE_ESTIMATE.json").read_text())
    assert est["estimate_bytes"] == sum(C.STORAGE_PLAN.values()) <= est["cap_bytes"] == 200_000_000
    with pytest.raises(ValueError, match="empty"):
        C.storage(home)
    assert C.limits_guard(home)["bytes_used"] == (home / "STORAGE_ESTIMATE.json").stat().st_size
    now = time.time()
    (home / "chain-runs.log").write_text(f"dev-round-1 start {now - 500}\ndev-round-1 end {now - 400}\nformal start {now - 50}\n")
    assert 149 < C.chain_seconds(home) < 153
    (home / "chain-runs.log").write_text(f"formal start {now - 4 * 3600 - 5}\n")
    with pytest.raises(ValueError, match="time cap"):
        C.limits_guard(home)
    (home / "chain-runs.log").unlink()
    with (home / "big.bin").open("wb") as f:
        f.truncate(C.CAP_BYTES + 1)
    with pytest.raises(ValueError, match="storage cap"):
        C.limits_guard(home)
    other = tmp_path / "elsewhere"
    other.mkdir()
    with pytest.raises(ValueError, match="thesis-study2"):
        C.storage(other)
