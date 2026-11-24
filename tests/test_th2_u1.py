"""TH2 U1 (TASK_TH2_U1_001): the arm registry of the scale baseline (refusal cases of task 4.3) and the pure reading
functions of tools/r28/th2_u1.py on synthetic arrays.  No file of the experiments is read; no model is built."""
import json
import os
import subprocess
import sys
import types
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "r28"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

import th2_u1 as X  # noqa: E402
import c0_ref as R  # noqa: E402
import u1_ref as U  # noqa: E402
from sparse_rtdetr.protocol_r18.model import Config  # noqa: E402
from sparse_rtdetr.research import scale_baseline as S  # noqa: E402

M = X.M


# ---------------------------------------------------------------- arm registry
def test_registry_and_S15_defaults():
    assert S.ARM == "S15" and S.ARM_HW == {"S15": (1152, 2016), "U896": (896, 1568)}
    assert S.ARM_SEEDS == {"S15": (0, 1, 2), "U896": (0,)}
    for seed in (0, 1, 2):
        assert S.production_config(seed) == Config(seed=seed, input_hw=(1152, 2016))
        S.validate_production(Config(seed=seed, input_hw=(1152, 2016)))
    assert S.production_config(0, "U896") == Config(seed=0, input_hw=(896, 1568))
    assert S.binding(S.production_config(0))["arm"] == "S15"
    assert S.binding(S.production_config(0, "U896"), arm="U896")["arm"] == "U896"
    only_canvas = [k for k, v in S.production_config(0, "U896").binding().items() if v != S.production_config(0).binding()[k]]
    assert only_canvas == ["input_hw"]
    for bad in ("U1024", "s15", None, 896):
        with pytest.raises(ValueError, match="unknown scale-baseline arm"):
            S.arm_hw(bad)


def test_U896_refuses_seed1_seed2_and_foreign_parents(tmp_path):
    dummy = tmp_path / "not-a-parent.pth"
    dummy.write_bytes(b"x" * 64)
    for seed in (1, 2, 3):
        with pytest.raises(ValueError, match="no certified parent"):
            S.create(dummy, "cpu", seed=seed, arm="U896")
    for seed in (1, 2):                                   # the S15 arm still admits its three seeds (it fails later, on the file)
        with pytest.raises(ValueError, match="parent identity"):
            S.create(dummy, "cpu", seed=seed)
    with pytest.raises(ValueError, match="parent identity"):
        S.create(dummy, "cpu", seed=0, arm="U896")        # a file that is not the certified seed0 parent
    with pytest.raises(ValueError, match="config seed"):
        S.create(dummy, "cpu", Config(seed=1, input_hw=(896, 1568)), seed=0, arm="U896")


def test_production_canvas_must_be_the_arms_own(tmp_path):
    dummy = tmp_path / "not-a-parent.pth"
    dummy.write_bytes(b"x" * 64)
    for hw in ((768, 1344), (1024, 1792), (1152, 2016), (896, 1600), (1568, 896)):
        with pytest.raises(ValueError, match="U896 production recipe differs"):
            S.create(dummy, "cpu", Config(seed=0, input_hw=hw), seed=0, arm="U896")
    for hw in ((768, 1344), (896, 1568), (1024, 1792)):
        with pytest.raises(ValueError, match="S15 production recipe differs"):
            S.create(dummy, "cpu", Config(seed=0, input_hw=hw), seed=0)
    with pytest.raises(ValueError, match="U896 production recipe differs"):     # anything but the canvas may not differ either
        S.create(dummy, "cpu", Config(seed=0, input_hw=(896, 1568), physical_batch_size=4), seed=0, arm="U896")
    fixture = Config(seed=0, input_hw=(1152, 2016), physical_batch_size=1, amp_dtype="float32")
    with pytest.raises(ValueError, match="fixture must be U896 canvas"):
        S.create(dummy, "cpu", fixture, seed=0, fixture=True, arm="U896")
    with pytest.raises(ValueError, match="unknown scale-baseline arm"):
        S.create(dummy, "cpu", seed=0, arm="U1024")


def _state_file(path, arm, config):
    torch.save({"version": S.VERSION, "arm": arm, "parent_sha256": S.PARENT_SHAS[0], "config": config.binding()}, path)
    return X.B.file_sha(path)


def _engine(arm, config):
    return types.SimpleNamespace(r25_arm=arm, r25_parent=S.PARENT_SHAS[0], config=config)


def test_complete_state_of_one_arm_is_refused_by_the_other(tmp_path):
    s15, u896 = S.production_config(0), S.production_config(0, "U896")
    files = {"S15": tmp_path / "s15.pth", "U896": tmp_path / "u896.pth", "renamed": tmp_path / "renamed.pth"}
    sha = {"S15": _state_file(files["S15"], "S15", s15), "U896": _state_file(files["U896"], "U896", u896),
           "renamed": _state_file(files["renamed"], "S15", u896)}          # the U896 canvas under the S15 name
    refusal = "cross-version/arm/canvas/recipe"
    with pytest.raises(ValueError, match=refusal):
        S.restore(_engine("U896", u896), files["S15"], sha["S15"])        # S15 state -> U896 engine
    with pytest.raises(ValueError, match=refusal):
        S.restore(_engine("S15", s15), files["U896"], sha["U896"])        # U896 state -> S15 engine
    with pytest.raises(ValueError, match=refusal):
        S.restore(_engine("U896", u896), files["renamed"], sha["renamed"])  # the arm name alone is enough
    with pytest.raises(ValueError, match="saved SHA"):
        S.restore(_engine("U896", u896), files["U896"], sha["S15"])
    for arm, cfg in (("S15", s15), ("U896", u896)):                        # the own state passes the identity line
        with pytest.raises(KeyError, match="optimizer_names"):
            S.restore(_engine(arm, cfg), files[arm], sha[arm])


def test_bind_arm_is_once_per_process():
    code = """
import sys
sys.path.insert(0, %r)
import th2_u1
import s15_formal_train as T
import s15_formal_eval as E
from sparse_rtdetr.research import scale_baseline as S
assert T.ARM == E.ARM == "S15" and T.fixture_config(0).input_hw == (1152, 2016) and T._cfg({"seed": 2}, False) == S.production_config(2)
for mod in (T, E):
    try:
        mod.bind_arm("U1024")
    except ValueError:
        pass
    else:
        raise SystemExit("unknown arm accepted")
    mod.bind_arm("U896")
    mod.bind_arm("U896")
    try:
        mod.bind_arm("S15")
    except (ValueError, RuntimeError):
        pass
    else:
        raise SystemExit("second arm accepted")
    assert mod.ARM == "U896"
assert T.fixture_config(0).input_hw == (896, 1568) and T._cfg({"seed": 0}, False) == S.production_config(0, "U896")
assert S.ARM == "S15"
print("OK")
""" % str(ROOT / "tools" / "r28")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    done = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, env=env, cwd=ROOT)
    assert done.returncode == 0 and done.stdout.strip().endswith("OK"), done.stdout + done.stderr


# ---------------------------------------------------------------- tool: text, limits, scoring ledger
def test_task_numbers_agree_with_the_reference_files():
    t = X.text_against_reference()
    assert round(t["CHORD"], 6) == 0.288889 and round(t["LOGLINEAR"], 6) == 0.380182 and t["tokens"] == {"L": 21168, "U": 28812, "H": 47628}
    assert [round(v, 4) for v in t["scale_l"]] == [0.7055, 0.9960, 1.4111]
    assert set(X.STAGES) == set(X.CPU_STAGES) | set(X.CUDA_STAGES) and len(X.SLOTS) == 5


def test_limits_guard(tmp_path):
    with pytest.raises(ValueError, match="STORAGE_ESTIMATE"):
        X.limits_guard(tmp_path)
    est = tmp_path / "STORAGE_ESTIMATE.json"
    est.write_text(json.dumps({"cap_bytes": X.CAP_BYTES, "estimate_bytes": X.CAP_BYTES + 1}))
    with pytest.raises(ValueError, match="above the cap"):
        X.limits_guard(tmp_path)
    est.write_text(json.dumps({"cap_bytes": X.CAP_BYTES + 5, "estimate_bytes": 10}))
    with pytest.raises(ValueError, match="above the cap"):
        X.limits_guard(tmp_path)
    est.write_text(json.dumps({"cap_bytes": X.CAP_BYTES, "estimate_bytes": 3_000_000_000}))
    got = X.limits_guard(tmp_path)
    assert got["bytes_used"] == est.stat().st_size and got["estimate_bytes"] == 3_000_000_000


def test_scoring_ledger_spends_each_slot_once(tmp_path):
    from collections import Counter
    c = {"home": str(tmp_path), "dev": False, "contract_sha256": "0" * 64}
    counts = Counter()
    with pytest.raises(ValueError):
        X.spend({**c, "dev": True}, X.SLOTS[0], counts)
    with pytest.raises(ValueError):
        X.spend(c, "score/top100", counts)
    for slot in X.SLOTS:
        X.spend(c, slot, counts)
        with pytest.raises(ValueError):
            X.spend(c, slot, counts)
    assert counts["score_budget_calls"] == 5 and [r["slot"] for r in X.scorings(tmp_path)] == list(X.SLOTS)


def test_native_groups_refuse_an_unknown_size():
    g = X.native_group([1920, 1360, 960, 1360], [1080, 765, 540, 765])
    assert g.tolist() == [0, 1, 2, 1]
    with pytest.raises(ValueError):
        X.native_group([1920, 1400], [1080, 788])


# ---------------------------------------------------------------- synthetic world
def world(seed=3, shift_u=0.35):
    rng = np.random.default_rng(seed)
    group = np.array([0] * 19 + [1] * 408 + [2] * 121)
    rng.shuffle(group)
    per = rng.integers(1, 40, M)
    image = np.repeat(np.arange(M), per)
    n = len(image)
    s = np.exp(rng.uniform(np.log(5.0), np.log(150.0), n))
    area = s ** 2
    cat = rng.integers(1, 11, n)
    p = 1.0 / (1.0 + np.exp(-(np.log(s) - 3.2) * 1.6))
    draw = rng.random(n)
    iou = {"a": np.where(draw < p, 0.8, 0.55), "b": np.where(draw < p + 0.10, 0.8, 0.55),
           "u": np.where(draw < p + 0.10 * shift_u, 0.8, 0.55), "l": np.where(draw < p + 0.01, 0.8, 0.3),
           "fg": np.where(draw < p + 0.05, 0.8, 0.3)}
    routed = np.zeros(M, dtype=bool)
    routed[np.argsort(-per, kind="mergesort")[:U.K_GATE]] = True
    return {"image": image, "area": area, "s": s, "cat": cat, "occlusion": rng.integers(0, 4, n), "crowd": rng.integers(0, 30, n),
            "g0": rng.uniform(0.8, 1.6, n), "iou": iou, "routed": routed, "group": group}


def test_GTView_keeps_the_scored_rows():
    w = world()
    n = len(w["image"])
    extra = 7
    z = {"image": np.concatenate([w["image"], np.full(extra, M - 1)]), "category": np.concatenate([w["cat"], np.ones(extra, dtype=np.int64)]),
         "xyxy": np.zeros((n + extra, 4)), "area": np.concatenate([w["area"], np.ones(extra)]), "s": np.concatenate([w["s"], np.ones(extra)]),
         "occlusion": np.concatenate([w["occlusion"], np.zeros(extra, dtype=np.int64)]), "crowd": np.concatenate([w["crowd"], np.zeros(extra, dtype=np.int64)]),
         "g0": np.concatenate([w["g0"], np.ones(extra)]), "scored": np.concatenate([np.ones(n, dtype=bool), np.zeros(extra, dtype=bool)])}
    gt = X.GTView(z)
    assert gt.n == n and np.array_equal(gt.image, w["image"]) and gt.bounds[0] == 0 and gt.bounds[-1] == n
    assert all(np.array_equal(gt.image[gt.sl(i)], np.full(gt.sl(i).stop - gt.sl(i).start, i)) for i in (0, 17, M - 1))
    m = gt.masks()
    assert int(m["small"].sum() + m["medium"].sum() + m["large"].sum()) == n and np.array_equal(m["small"], w["area"] <= 32.0 ** 2)


def test_prediction_tables_follow_the_reference_calls():
    w = world()
    small = w["area"] <= 32.0 ** 2
    t = X.prediction_tables(w["image"], small, w["s"], w["cat"], w["iou"]["a"], w["iou"]["b"], w["group"])
    ha, hb = (w["iou"]["a"] >= 0.75) & small, (w["iou"]["b"] >= 0.75) & small
    ra, rb = 100.0 * ha.sum() / small.sum(), 100.0 * hb.sum() / small.sum()
    assert abs(t["endpoints"]["A0_small_hit75_pp"] - ra) < 1e-9 and abs(t["endpoints"]["S15_small_hit75_pp"] - rb) < 1e-9
    assert t["predictions"]["1_linear"]["f"] == U.CHORD and t["predictions"]["2_loglinear"]["f"] == U.LOGLINEAR
    gt_group = w["group"][w["image"]]
    lead, count = [], []
    for k in range(3):
        m = small & (gt_group == k)
        lead.append(100.0 * (hb[m].sum() - ha[m].sum()) / m.sum())
        count.append(float(m.sum()))
        row = t["prediction_3"]["groups"][k]
        assert row["small_gt"] == int(m.sum()) and row["images"] == int((w["group"] == k).sum()) and abs(row["lead_pp"] - lead[-1]) < 1e-9
    ref = U.group_concave(lead, count, X.native_scale())
    assert abs(t["predictions"]["3_group_concave"]["f"] - ref["f"]) < 1e-9 and abs(t["prediction_3"]["group_concave"]["b2"] - ref["b2"]) < 1e-7
    hit = (w["iou"]["a"] >= 0.75).astype(np.float64)
    pred = U.shift_prediction(w["s"], w["cat"], hit, small)
    summary = U.shift_summary(pred, ra, rb)
    assert abs(t["predictions"]["4_size_shift"]["f"] - summary["f_law"]) < 1e-12
    assert abs(t["prediction_4"]["summary"]["kappa"] - summary["kappa"]) < 1e-9
    g1 = U.shift_prediction(w["s"], w["cat"], hit, small & (gt_group == 1))
    assert abs(t["prediction_4"]["groups"][1]["predicted_rate_pp"]["M_U"] - g1[float(U.M_U)]) < 1e-9
    for v in t["predictions"].values():
        assert abs(v["implied_U896_small_hit75_pp"] - (ra + v["f"] * (rb - ra))) < 1e-9
    json.dumps(t, allow_nan=False)


def _ap(base):
    return {m: base + 0.01 * i for i, m in enumerate(X.METRICS)}


def _aps():
    return {"U896": {"600": _ap(0.31), "300": _ap(0.305)}, "A0": {"600": _ap(0.30), "300": _ap(0.295)},
            "S15": {"600": _ap(0.32), "300": _ap(0.315)}, "L0": {"600": _ap(0.301)}, "FG": {"600": _ap(0.313)},
            "routed_A0_S15": {"600": _ap(0.312)}, "routed_L0_S15": {"600": _ap(0.311)}, "routed_L0_S15_small_hit75_pp": 46.0}


def test_reading_tables_against_brute_force():
    w = world()
    small = w["area"] <= 32.0 ** 2
    pred = X.prediction_tables(w["image"], small, w["s"], w["cat"], w["iou"]["a"], w["iou"]["b"], w["group"])
    t, columns = X.reading_tables(w, _aps(), pred)
    hit = {k: (v >= 0.75) & small for k, v in w["iou"].items()}
    n = small.sum()
    f = (hit["u"].sum() - hit["a"].sum()) / (hit["b"].sum() - hit["a"].sum())
    u2 = t["U2"]["small_hit75"]
    assert abs(u2["f"] - f) < 1e-12 and u2["ci95"][0] <= f <= u2["ci95"][1]
    assert abs(u2["U896_minus_A0_pp"] - 100.0 * (hit["u"].sum() - hit["a"].sum()) / n) < 1e-9
    for k, v in u2["predictions"].items():
        assert abs(v["measured_minus_predicted"] - (f - v["f"])) < 1e-12 and v["inside_ci95"] == (u2["ci95"][0] <= v["f"] <= u2["ci95"][1])
    gt_routed = w["routed"][w["image"]]
    routed_hits = np.where(gt_routed, hit["b"], hit["a"]).sum()
    u3 = t["U3"]["small_hit75"]
    assert abs(u3["routed_minus_U896_pp"] - 100.0 * (routed_hits - hit["u"].sum()) / n) < 1e-9
    assert u3["stage1_reading"] == U.stage1_reading(u3["routed_minus_U896_pp"]) and u3["routed_images"] == 158
    assert abs(t["U3"]["AP_level_top600"]["routed_minus_U896_pp"]["APS"] - 0.2) < 1e-9
    u4 = t["U4"]
    assert abs(u4["FG_minus_U896"]["small_hit75_pp"] - 100.0 * (hit["fg"].sum() - hit["u"].sum()) / n) < 1e-9
    eq = u4["equivalent_budget"]["small_hit75"]
    assert eq == {"gain_pp": eq["gain_pp"], "gain_u_pp": eq["gain_u_pp"], "gain_h_pp": eq["gain_h_pp"],
                  **U.equivalent_budget(eq["gain_pp"], eq["gain_u_pp"], eq["gain_h_pp"])}
    assert abs(u4["equivalent_budget"]["APS"]["gain_pp"] - 1.2) < 1e-9 and u4["equivalent_budget"]["APS"]["status"] == "between_U_and_H"
    u5 = t["U5"]
    assert sum(r["n"] for r in u5["size_bin_all_GT"]) == len(small)
    for key in ("class_small_GT", "occlusion_small_GT", "crowd_small_GT", "F_geo_magnification_small_GT", "native_size_small_GT"):
        assert sum(r["n"] for r in u5[key]) == int(n), key
    cls3 = small & (w["cat"] == 3)
    row = u5["class_small_GT"][2]
    assert row["group"] == "3" and abs(row["A0_hit75_pp"] - 100.0 * (w["iou"]["a"][cls3] >= 0.75).mean()) < 1e-9
    assert abs(row["FG_minus_L0_pp"] - 100.0 * ((w["iou"]["fg"][cls3] >= 0.75).mean() - (w["iou"]["l"][cls3] >= 0.75).mean())) < 1e-9
    nat = u5["native_size_against_predictions"]
    assert [r["group"] for r in nat] == ["1920x1080", "1360x765", "960x540"]
    assert abs(nat[1]["prediction_3_lead_U_pp"] - pred["prediction_3"]["group_concave"]["predicted_lead_U"][1]) < 1e-12
    ap2 = t["U2"]["AP_level"]["top600"]
    assert abs(ap2["f"]["APS"] - 0.5) < 1e-9 and "APL" not in ap2["f"] and set(ap2["APL_values"]) == {"A0", "U896", "S15"}
    assert abs(t["U2"]["medium_hit75"]["A0_pp"] - t["U1"]["hit_level_pp"]["A0-s0"]["medium_hit75"]) < 1e-9
    assert set(columns) == {"n_small", "hit75_small_A0", "hit75_small_U896", "hit75_small_S15", "hit75_small_L0", "hit75_small_FG", "P_k158"}
    assert columns["n_small"].sum() == n and columns["hit75_small_U896"].sum() == hit["u"].sum() and columns["P_k158"].sum() == 158
    json.dumps(t, allow_nan=False)


def test_reading_tables_at_the_two_ends():
    w = world()
    small = w["area"] <= 32.0 ** 2
    pred = X.prediction_tables(w["image"], small, w["s"], w["cat"], w["iou"]["a"], w["iou"]["b"], w["group"])
    for k, f in (("a", 0.0), ("b", 1.0)):
        ww = {**w, "iou": {**w["iou"], "u": w["iou"][k]}}
        t, _ = X.reading_tables(ww, _aps(), pred)
        assert t["U2"]["small_hit75"]["f"] == f and t["U2"]["small_hit75"]["ci95"] == [f, f]
    bad = {**w, "routed": np.ones(M, dtype=bool)}
    with pytest.raises(ValueError):
        X.reading_tables(bad, _aps(), pred)


# ---------------------------------------------------------------- input alignment
def _ledger(n, **change):
    rows = []
    for i in range(1, n + 1):
        u = X.PARENT_UPDATE + i
        inp = {"update": u, "epoch": 61, "window": i - 1, "image_ids_sha256": f"i{u}", "augmentation_seeds_sha256": f"s{u}",
               "augmentation_strong": True, "gt_ids_sha256": f"g{u}", "augmentation_trace_sha256": f"t{u}",
               "augmented_samples_sha256": f"a{u}", "input_tensor_sha256": f"x{u}", "target_sha256": f"y{u}"}
        rows.append({"input": inp, "gt_count": 10 + i, "loss": 1.0, "lr": 1e-4, "ema_updates": u})
    for (i, key), value in change.items():
        rows[i]["input"][key] = value
    return rows


def _lines(rows, drop=()):
    return [json.dumps({k: v for k, v in r.items() if k not in drop}) for r in rows]


def test_align_rows():
    native = _lines(_ledger(6), drop=("gt_count",))
    mine = _ledger(6)
    for r in mine[:4]:
        r["input"]["input_tensor_sha256"] += "-896"
        r["input"]["target_sha256"] += "-896"
    mine[2]["input"]["gt_ids_sha256"] += "-896"
    got = X.align_rows(_lines(mine), native)
    assert got["updates"] == 6 and got["rows_with_canvas_dependent_difference"] == {
        "gt_ids_sha256": 1, "augmentation_trace_sha256": 0, "augmented_samples_sha256": 0, "input_tensor_sha256": 4, "target_sha256": 4}
    assert got["U896_positive_GT_total_after_sanitize"] == sum(10 + i for i in range(1, 7)) and got["rows_with_equal_clock"] == {"lr": 6, "ema_updates": 6}
    for key, value in (("image_ids_sha256", "other"), ("augmentation_seeds_sha256", "other"), ("augmentation_strong", False),
                       ("window", 99), ("epoch", 62)):
        broken = _ledger(6)
        broken[3]["input"][key] = value
        with pytest.raises(ValueError, match="order / draws differ"):
            X.align_rows(_lines(broken), native)
    with pytest.raises(ValueError, match="longer"):
        X.align_rows(_lines(_ledger(7)), native)
    with pytest.raises(ValueError, match="longer"):
        X.align_rows(_lines(_ledger(5)), native)
