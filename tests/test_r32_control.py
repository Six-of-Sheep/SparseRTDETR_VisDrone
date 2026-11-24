"""R32-001 control checks (CPU, no model forward): TASK_R32_W_SEEDS_001.

Unit names and order; the seed argument of arm W (default 0, certified parents only); the planned workload and the
storage plan; the release difference rule; runner seed binding; the comparison helpers of the endpoint reproduction;
the formal guard; the formal branch of the summary on a synthetic round (ledger comparison with the three reference
ledgers of a seed, the reading of task section 5, per-class tables, training geometry).
"""
import copy
import inspect
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "r28"))
import r28_common as K  # noqa: E402
import r32_control as R32  # noqa: E402
import r32_loop as RL  # noqa: E402
import r32_reading as RD  # noqa: E402
import r32_run as RUN  # noqa: E402

L0_600 = dict(zip(RD.METRICS, (0.3005, 0.2238, 0.2023, 0.3956, 0.4951, 0.5056, 0.3013)))
L0_300 = dict(zip(RD.METRICS, (0.2963, 0.2198, 0.1998, 0.3908, 0.4925, 0.4957, 0.2981)))
BASE = {"600": L0_600, "300": L0_300}
GT_LARGE = (23, 5, 3, 779, 115, 87, 12, 5, 32, 7)
W_GAIN = {"APS": 1.8, "AP75_small": 2.5, "AP": 0.8, "APM": -0.7, "APL": 4.0}
FG_GAIN = {"APS": 2.0, "AP75_small": 2.3, "AP": 1.2, "APM": 0.4, "APL": -0.5}


def shifted(d600=None, d300=None):
    out = copy.deepcopy(BASE)
    for b, d in (("600", d600 or {}), ("300", d300 or {})):
        for k, v in d.items():
            out[b][k] += v / 100.
    return out


def test_units_and_order():
    assert R32.ORDER == ("W-s1", "W-s2") and R32.SEEDS == (1, 2) and R32.ARM == "W"
    assert [R32.parse(u) for u in R32.ORDER] == [("W", 1), ("W", 2)]
    assert R32.parse("W-s0") == ("W", 0) and R32.unit_name(2) == "W-s2"
    for bad in ("FG-s1", "L0-s1", "W-s3", "W", "W-s1 ", "K-s1", None):
        with pytest.raises(Exception):
            R32.parse(bad)
    assert K.formal_re("W-s1").fullmatch("formal-W-s1") and K.formal_re("W-s1").fullmatch("formal-W-s1-r2")
    assert not K.formal_re("W-s1").fullmatch("formal-W-s2") and not K.formal_re("W-s1").fullmatch("formal-W-s1-r1")
    assert R32.TASK == ("TASK_R32_W_SEEDS_001.md", "fb2bb39802a44c30d219ead4be72d562a910aed93bf4aabb3000ac5745a3b157")
    assert R32.reference_label(0) == "R27 formal-P" and R32.reference_label(2) == "R30-002 L0-s2"


def test_seed_argument_defaults_to_the_frozen_path():
    R = pytest.importorskip("sparse_rtdetr.research.r28_resample")
    S = pytest.importorskip("sparse_rtdetr.research.scale_baseline")
    for fn in ("create", "build_eval_model"):
        p = inspect.signature(getattr(R, fn)).parameters["seed"]
        assert p.default == 0 and p.kind is inspect.Parameter.KEYWORD_ONLY
    assert R.IDENTITY_SHA256["W"] == R32.W_IDENTITY == "3f5d40f55de39a59326e7a2563f7a7de2e55686a7eb83ad4fead060b4f4ff541"
    assert R.IDENTITY["W"]["parent_sha256"] == S.PARENT_SHAS[0] == K.PARENT_SHA
    assert sorted(S.PARENT_SHAS) == [0, 1, 2] and S.PARENT_SHAS[1].startswith("6f8fd316") and S.PARENT_SHAS[2].startswith("c36e3c1f")
    assert R.FORMAL_KIND["W"] == "network"


def test_seed_without_certified_parent_and_foreign_parent_are_refused(tmp_path):
    R = pytest.importorskip("sparse_rtdetr.research.r28_resample")
    f = tmp_path / "not-a-parent.pth"
    f.write_bytes(b"x")
    source = {"kind": "constant"}
    for seed in (3, -1, True, "1", 1.0):
        with pytest.raises(Exception, match="certified parent"):
            R.create(str(f), "cpu", "W", source, check=True, seed=seed)
    for seed in (0, 1, 2):
        with pytest.raises(Exception, match="parent identity"):
            R.create(str(f), "cpu", "W", source, check=True, seed=seed)
    with pytest.raises(Exception):
        R.create(str(f), "cpu", "W", source, seed=1)           # a check-only demand source outside a check


def test_planned_workload():
    w = R32.WORKLOAD
    assert w["wire-s0"]["updates"] == 4 and w["wire-s0"]["forwards"] == 8 and w["wire-s0"]["updates"] <= w["wire-s0"]["task_limit_updates"] == 8
    assert w["accept"]["updates_per_seed"] == 10 and w["accept"]["forwards_per_seed"] == 20 and w["accept"]["updates"] == 20
    assert w["accept"]["updates_per_seed"] <= w["accept"]["task_limit_updates_per_seed"] == 20
    assert w["endpoint-s0"]["forwards"] == 141 and w["endpoint-s0"]["warmup"] == 4
    assert w["endpoint-s0"]["score_budget_calls"] == 2 and w["endpoint-s0"]["per_class_COCOeval_calls"] == 1
    assert w["formal"]["updates_per_unit"] == 18240 and w["formal"]["updates"] == 2 * 18240 and w["formal"]["forwards"] == 2 * 36480
    assert w["evaluation"]["forwards_per_unit"] == 141 and w["evaluation"]["forwards"] == 2 * 141
    assert w["evaluation"]["score_budget_calls"] == 4 and w["evaluation"]["per_class_COCOeval_calls"] == 2
    assert w["formal"]["save_epochs"] == list(range(60, 121, 5)) and len(w["formal"]["save_epochs"]) == 13
    assert len(RD.READING["cells"]) == 14 and "timing" in " ".join(R32.NOT_AUTHORISED)


def test_storage_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(K, "ROUND", tmp_path)
    w0 = {"predictions_600": {"bytes": 87894197}, "predictions_300": {"bytes": 43893023}}
    p = R32.storage_plan(323123009, w0, 5 * 1024 ** 3)
    assert p["formal_states"] == 26 and p["formal_states_bytes"] == 26 * 323123009 and p["acceptance_states_bytes"] == 2 * 323123009
    assert p["new_predictions_bytes_estimate"] == 2 * (87894197 + 43893023) and p["seed0_reproduction_predictions_bytes_estimate"] == 87894197 + 43893023
    parts = ("formal_states_bytes", "acceptance_states_bytes", "new_predictions_bytes_estimate", "seed0_reproduction_predictions_bytes_estimate",
             "ledgers_observations_timing_bytes_estimate", "metrics_tensors_logs_release_bytes_estimate")
    assert p["formal_round_bytes_estimate"] == sum(p[k] for k in parts)
    assert p["total_bytes_estimate"] == p["formal_round_bytes_estimate"] + 5 * 1024 ** 3 < p["stop_line_bytes"] == 30 * 1024 ** 3
    assert 9.0 < p["formal_round_bytes_estimate"] / 1e9 < 10.5 and p["disk_free_bytes_at_prepare"] > 0


def test_release_difference_rule():
    old = {"src/a.py": "1", "tools/r28/r28_loop.py": "2", R32.SCIENCE_FILE: "0", R32.WARP_FILE: "w"}
    new = {**old, R32.SCIENCE_FILE: "9", **{p: "n" for p in R32.ALLOWED_ADDED}}
    d = R32.source_delta(old, new)
    assert d["changed"] == [R32.SCIENCE_FILE] and d["added"] == sorted(R32.ALLOWED_ADDED) and not d["removed"]
    for bad in ({**new, "tools/r28/r28_loop.py": "x"}, {**new, "src/new.py": "x"}, {k: v for k, v in new.items() if k != "src/a.py"},
                {**new, R32.SCIENCE_FILE: "0"}, {**new, R32.WARP_FILE: "x"}):
        with pytest.raises(Exception):
            R32.source_delta(old, bad)
    w_release = {"src/a.py": "0", "tools/r28/r28_loop.py": "2", R32.SCIENCE_FILE: "0", R32.WARP_FILE: "w"}
    d = R32.source_delta(old, new, w_release)
    assert d["arm_W_files_against_R28_001"]["files_changed_between_R28_001_and_R30_002"] == ["src/a.py"]
    for bad in ({**w_release, R32.WARP_FILE: "other"}, {**w_release, R32.SCIENCE_FILE: "other"}):
        with pytest.raises(Exception):
            R32.source_delta(old, new, bad)


def test_runner_seed_binding(monkeypatch):
    monkeypatch.delenv("R32_UNIT", raising=False)
    monkeypatch.delenv("R32_SEED", raising=False)
    assert RUN.process_seed("r32-check") == 0 and RUN.process_seed("summarize") == 0
    for action in ("r32-wire", "r32-endpoint", "r32-accept", "formal", "evaluate"):
        with pytest.raises(Exception):
            RUN.process_seed(action)
    monkeypatch.setenv("R32_SEED", "0")
    assert RUN.process_seed("r32-wire") == 0 and RUN.process_seed("r32-endpoint") == 0
    with pytest.raises(Exception):
        RUN.process_seed("r32-accept")
    with pytest.raises(Exception):
        RUN.process_seed("r32-check")
    monkeypatch.setenv("R32_SEED", "2")
    assert RUN.process_seed("r32-accept") == 2
    for action in ("r32-wire", "r32-endpoint"):
        with pytest.raises(Exception):
            RUN.process_seed(action)
    monkeypatch.delenv("R32_SEED")
    for unit, seed in (("W-s1", 1), ("W-s2", 2)):
        monkeypatch.setenv("R32_UNIT", unit)
        assert RUN.process_seed("formal") == seed and RUN.process_seed("evaluate") == seed
        with pytest.raises(Exception):
            RUN.process_seed("r32-check")
    for unit in ("W-s0", "FG-s1", "L0-s2"):
        monkeypatch.setenv("R32_UNIT", unit)
        with pytest.raises(Exception):
            RUN.process_seed("formal")
    assert sorted(RUN.DISPATCH) == ["evaluate", "formal", "r32-accept", "r32-check", "r32-endpoint", "r32-wire", "summarize"]


def test_delivery_override_only_selects_the_serial_host_path(monkeypatch):
    monkeypatch.delenv("R32_DELIVERY", raising=False)
    assert R32.delivery_candidate() == ("overlap", "device")
    monkeypatch.setenv("R32_DELIVERY", "host")
    assert R32.delivery_candidate() == ("serial", "host") == tuple(R32.ACCEPT["reference_path"])
    monkeypatch.setenv("R32_DELIVERY", "prepared")
    with pytest.raises(Exception):
        R32.delivery_candidate()


def test_prediction_and_cell_differences():
    def rec(i, score=0.5, box=(1.0, 2.0, 3.0, 4.0), cat=1):
        return {"image_id": i, "query_id": i, "category_id": cat, "score": score, "bbox_xyxy": list(box)}
    a = [rec(0), rec(1), rec(2)]
    same = RL.prediction_differences(a, copy.deepcopy(a))
    assert same["records_different"] == 0 and same["first_different_record"] is None and same["score_max_abs_difference"] == 0.0
    b = copy.deepcopy(a)
    b[1]["score"] = 0.5 + 2 ** -20
    b[2]["bbox_xyxy"][3] = 4.5
    d = RL.prediction_differences(a, b)
    assert d["records_different"] == 2 and d["first_different_record"] == 1 and d["identity_and_order_equal"] is True
    assert d["score_max_abs_difference"] == 2 ** -20 and d["bbox_max_abs_difference_px"] == 0.5
    assert d["records_with_another_score"] == 1 and d["records_with_another_box"] == 1
    b[0]["category_id"] = 2
    assert RL.prediction_differences(a, b)["identity_and_order_equal"] is False
    with pytest.raises(Exception):
        RL.prediction_differences(a, b[:2])
    t = shifted(W_GAIN)
    assert RL.cell_differences(t, copy.deepcopy(t)) == {"bitwise_equal": True, "cells_different": [], "max_abs_difference": 0.0,
                                                        "max_abs_difference_cell": RL.cell_differences(t, t)["max_abs_difference_cell"]}
    u = copy.deepcopy(t)
    u["300"]["APM"] += 2 ** -30
    got = RL.cell_differences(u, t)
    assert got["bitwise_equal"] is False and got["cells_different"] == ["top300_APM"] and got["max_abs_difference_cell"] == "top300_APM"
    rows = [{"category_id": k, "GT": 3, "APL": 0.5 + k / 100.} for k in (1, 2)] + [{"category_id": 3, "GT": 0, "APL": None}]
    assert RL.class_differences(rows, copy.deepcopy(rows), "APL") == {"bitwise_equal": True, "max_abs_difference": 0.0, "within_tolerance": True}
    other = copy.deepcopy(rows)
    other[0]["APL"] += 1e-12
    assert RL.class_differences(rows, other, "APL")["within_tolerance"] is False
    other[0]["GT"] = 4
    with pytest.raises(Exception):
        RL.class_differences(rows, other, "APL")


def _closed(path, body):
    path.mkdir()
    (path / "CLOSED.json").write_text(json.dumps({"status": "PASS", "contract_sha256": "c", **body}))


CHECKS = {"r32-check": {"counts": {"actual_updates": 0, "forwards": 0}},
          "wire-s0": {"counts": {"actual_updates": 4, "forwards": 8}, "W_equal_to_the_R28_001_receipts": {"a": True, "b": True},
                      "formal_path_equal_to_the_reference_path": True, "W_not_comparable": ["x"], "state_after_update_1_compared": True, "equivalence": {}},
          "endpoint-s0": {"counts": {"actual_updates": 0, "forwards": 141, "score_budget_calls": 2, "per_class_COCOeval_calls": 1},
                          "predictions_identical": True, "prediction_records_compared": {"600": 328800, "300": 164400},
                          "prediction_files_sha256_equal": {"600": True, "300": True}, "scoring_reproduced": True, "per_class_APL_bitwise_equal": True,
                          "per_class_APL_max_abs_difference": 0.0, "not_comparable": ["y"], "dev_prefix_only": False, "reproduction": {}}}


def _accept(seed):
    return {"counts": {"actual_updates": 10, "forwards": 20}, "unit": f"W-s{seed}", "restore_replay_equal": True,
            "window_delivery": {"mode": "device", "pipeline": "overlap", "equal_to_the_serial_host_reference": True},
            "same_windows_as_the_reference_ledger": True, "formal_training_allowed": True, "equivalence": {}}


def test_formal_guard_needs_every_check_and_the_order(tmp_path, monkeypatch):
    monkeypatch.setattr(K, "ROUND", tmp_path)
    monkeypatch.setattr(K, "DEV", False)
    for k in [k for k in __import__("os").environ if k.startswith("R28_DEV_") or k == "R28_PIPELINE"]:
        monkeypatch.delenv(k)
    c = {"contract_sha256": "c", "units": {u: {} for u in R32.ORDER}}
    with pytest.raises(Exception, match="stage not closed"):
        RL.guard(c, tmp_path / "formal-W-s1", None, "W-s1")
    _closed(tmp_path / "r32-check", CHECKS["r32-check"])
    _closed(tmp_path / "wire-s0", CHECKS["wire-s0"])
    _closed(tmp_path / "endpoint-s0", {**CHECKS["endpoint-s0"], "scoring_reproduced": False})
    _closed(tmp_path / "accept-s1", _accept(1))
    _closed(tmp_path / "accept-s2", _accept(2))
    with pytest.raises(Exception, match="do not open"):
        RL.guard(c, tmp_path / "formal-W-s1", None, "W-s1")
    (tmp_path / "endpoint-s0" / "CLOSED.json").write_text(json.dumps({"status": "PASS", "contract_sha256": "c", **CHECKS["endpoint-s0"]}))
    assert RL.guard(c, tmp_path / "formal-W-s1", None, "W-s1") is None
    with pytest.raises(Exception):
        RL.guard(c, tmp_path / "formal-W-s1-r2", None, "W-s1")          # a fresh run writes formal-W-s1
    with pytest.raises(Exception, match="exactly one closed PASS formal-W-s1"):
        RL.guard(c, tmp_path / "formal-W-s2", None, "W-s2")             # W-s1 first
    with pytest.raises(Exception):
        RL.guard(c, tmp_path / "formal-W-s0", None, "W-s0")
    _closed(tmp_path / "formal-W-s1", {"unit": "W-s1"})
    assert RL.guard(c, tmp_path / "formal-W-s2", None, "W-s2") is None
    with pytest.raises(Exception, match="another formal run is refused"):
        RL.guard(c, tmp_path / "formal-W-s1", None, "W-s1")


def _rows(seed, n):
    out = []
    for i in range(n):
        u = K.PARENT_UPDATE + 1 + i
        inp = {k: f"{k}-{seed}-{i}" for k in K.INPUT_FIELDS}
        inp.update(update=u, epoch=61 if i < n - 1 else 118)
        out.append({"input": inp, "lr": 1e-4, "gt_count": 10 + i, "ema_updates": u, "loss": 9.0})
    return out


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return K.ref(path)


def _w_rows(seed, n, g):
    return [{**r, "loss": 8.0, "microbatch_losses": [1.0, 2.0], "update_seconds": 1.0, "gradient_norm_before_clip": 3.0,
             "r28": {"g_small": g + i / 100., "g_medium": 1.1, "g_large": 1.0, "n_small": 100 + i, "small_g_lt_1": 0.05, "slope_min": 0.7,
                     "slope_max": 1.5, "identity_images": 0}} for i, r in enumerate(_rows(seed, n))]


def _apl(cell):
    return [{"category_id": k, "GT": GT_LARGE[k - 1], "APL": cell + (k - 5.5) / 100.} for k in range(1, 11)]


def _world(tmp_path, monkeypatch, w1, w2, n=4):
    monkeypatch.setattr(K, "ROUND", tmp_path)
    monkeypatch.setattr(K, "DEV", False)
    monkeypatch.setattr(K, "ENDPOINT", K.PARENT_UPDATE + n)
    monkeypatch.setattr(R32, "_FORMAL", n)
    refs = tmp_path / "refs"
    fg, w0 = shifted(FG_GAIN), shifted(W_GAIN)
    quoted = {s: {"L0": BASE, "FG": fg} for s in RD.SEED_KEYS}
    s15 = {s: {m: BASE["600"][m] + 0.02 for m in RD.METRICS} for s in RD.SEED_KEYS}
    cls = lambda v: [{"category_id": k, "GT": 30 + k, "L0": v + k / 100., "FG": v + 0.01 + k / 100.} for k in range(1, 11)]
    apl_q = [{"category_id": r["category_id"], "GT": r["GT"], "L0": r["APL"], "FG": f["APL"]}
             for r, f in zip(_apl(BASE["600"]["APL"]), _apl(fg["600"]["APL"]))]
    units = {}
    for u in R32.ORDER:
        seed = R32.parse(u)[1]
        units[u] = {"reference_ledger": _write(refs / f"L0-s{seed}.jsonl", _rows(seed, n)),
                    "equal_ledgers": {"R30-002 FG": _write(refs / f"FG-s{seed}.jsonl", [{**r, "loss": 8.5, "r28": {}} for r in _rows(seed, n)]),
                                      "S15 replication": _write(refs / f"S15-s{seed}.jsonl", [{**r, "loss": 7.5} for r in _rows(seed, n)])}}
    c = {"contract_sha256": "c", "units": units,
         "seed0": {"values": w0, "per_class_APL_top600": _apl(w0["600"]["APL"]), "per_class_note": "note",
                   "ledger": _write(refs / "W-s0.jsonl", _w_rows(0, n, 1.30))},
         "R30": {"values": quoted, "per_class_APL_top600": {s: apl_q for s in RD.SEED_KEYS},
                 "per_class_APS_APM_top600": {s: {"APS": cls(0.2), "APM": cls(0.4)} for s in ("1", "2")}},
         "S15": {"values_top600": s15},
         "reference": {"GT_counts": {"600": 1, "300": 1}, "seed0_W_minus_L0_pp": RD.difference_pp(w0, BASE), "seed0_W_minus_F_geo_pp": RD.difference_pp(w0, fg)},
         "statements": {}, "reading": RD.READING, "demand": {}, "workload": R32.WORKLOAD, "resource_plan": {}}
    for name, body in {**CHECKS, "accept-s1": _accept(1), "accept-s2": _accept(2)}.items():
        _closed(tmp_path / name, body)
    for u, w in zip(R32.ORDER, (w1, w2)):
        seed = R32.parse(u)[1]
        d = tmp_path / f"formal-{u}"
        _closed(d, {"unit": u, "updates": n, "ledger_rows": n, "ledger_rows_copied_on_resume": 0, "resumed": False, "hook_forwards": 2 * n,
                    "plan": {"input_pipeline": "overlap", "window_delivery": {"mode": "device"}, "parent": {"sha256": "p"}, "demand": {}},
                    "gpu_at_start": {}, "gpu_at_end": {}, "loop_wall_seconds": 1.0, "peak_cuda_bytes": 1, "saves": [{}] * 13})
        _write(d / "UPDATES.jsonl", _w_rows(seed, n, 1.28))
        state = d / "checkpoint-update-x.pth"
        endpoint = {"path": str(state), "bytes": 1, "sha256": "s"}
        K.side_of(state).write_text(json.dumps({"file": endpoint}))
        _closed(tmp_path / f"eval-{u}", {"unit": u, "formal_stage": d.name, "endpoint": endpoint, "values": w, "global_update": K.ENDPOINT,
                                           "GT_counts": {"600": 1, "300": 1}, "weights_sha256": "w", "predictions": {}, "metrics": {},
                                           "cocoeval_top600_tensors": {}, "counts": {"forwards": 141, "score_budget_calls": 2, "per_class_COCOeval_calls": 1},
                                           "per_class_APL_top600": _apl(w["600"]["APL"]),
                                           "per_class_APS_top600": [{"category_id": k, "GT": 30 + k, "APS": 0.25 + k / 100.} for k in range(1, 11)],
                                           "per_class_APM_top600": [{"category_id": k, "GT": 30 + k, "APM": 0.39 + k / 100.} for k in range(1, 11)]})
    return c


def test_summarize_formal_branch(tmp_path, monkeypatch):
    """summary-R32 on a synthetic round: two short ledgers, two evaluation receipts, quoted L0 / F_geo / S15 values."""
    c = _world(tmp_path, monkeypatch, shifted(W_GAIN), shifted({**W_GAIN, "APS": 0.4, "APM": 0.3}))
    out = tmp_path / "summary-R32"
    out.mkdir()
    got = RL.summarize(c, out, {})
    result = json.loads((out / "RESULT.json").read_text())
    assert got["reading_A"] == "W_small_gain_repeated_2of3" == result["reading_A"]["label"] and got["seeds_satisfying_A"] == ["0", "1"]
    assert got["reading_B"] == "no_practical_lead_in_any_seed" and got["readings_B"] == {"0": "neither", "1": "neither", "2": "neither"}
    assert got["reading_C"] == "W_medium_cost_repeated_2of3" and got["W_unique_advantage"] == "not_established"
    assert abs(got["seed2_W_minus_L0_APS600_pp"] - 0.4) < 1e-9 and abs(got["seed1_W_minus_FG_APS600_pp"] + 0.2) < 1e-9
    assert sorted(result["W_minus_L0_pp"]) == ["0", "1", "2"] and sorted(result["endpoints"]) == ["FG", "L0", "W"]
    assert sorted(result["S15_minus_W_top600_pp"]["2"]) == sorted(RD.METRICS)
    led = result["ledgers"]["units"]
    assert all(sorted(led[u]["against"]) == ["R30-002 FG", "R30-002 L0", "S15 replication"] and led[u]["rows"] == 4
               and all(v["rows_different"] == 0 and v["rows_compared"] == 4 for v in led[u]["against"].values()) for u in R32.ORDER)
    t = result["per_class_top600"]
    assert sorted(t["1"]) == ["APL", "APM", "APS"] and sorted(t["1"]["APS"][0]) == ["FG", "GT", "L0", "W", "category_id"]
    assert t["0"]["APS"] is None and t["0"]["APM"] is None and len(t["0"]["APL"]) == 10 and t["0"]["APL"][3]["GT"] == 779
    split = result["APL"]["1"]["split"]["W_minus_L0"]
    assert split["rare"]["classes"] == [2, 3, 7, 8, 10] and abs(split["macro_difference_pp"] - 4.0) < 1e-9
    geo = result["training_geometry"]
    assert sorted(geo["units"]) == ["W-s0", "W-s1", "W-s2"] and geo["units"]["W-s1"]["all"]["updates"] == 4
    assert geo["units"]["W-s1"]["last_three_epochs"]["updates"] == 1 and geo["units"]["W-s0"]["all"]["small_gt_instances"] == 406
    assert abs(geo["difference_from_W_s0"]["W-s2"]["all"]["g_small_geometric_mean"] + 0.02) < 1e-3
    assert result["actual"] == {"training_updates": 8, "training_forwards": 16, "evaluation_forwards": 282, "score_budget_calls": 4,
                                "per_class_COCOeval_calls": 2, "check_updates": 24, "check_forwards": 189, "check_score_budget_calls": 2,
                                "check_per_class_COCOeval_calls": 1}
    assert result["training"]["W-s1"]["saves"] == 13 and result["training"]["W-s2"]["other_formal_directories"] == []
    # a ledger row that differs from the reference ledgers of its seed stops the ledger check
    rows = (tmp_path / "formal-W-s2" / "UPDATES.jsonl").read_text().splitlines()
    bad = json.loads(rows[2])
    bad["gt_count"] += 1
    (tmp_path / "formal-W-s2" / "UPDATES.jsonl").write_text("\n".join(rows[:2] + [json.dumps(bad)] + rows[3:]) + "\n")
    with pytest.raises(Exception):
        RL.ledger_check(c, {u: tmp_path / f"formal-{u}" for u in R32.ORDER})


def test_summarize_reports_a_new_seed_lead_as_inconsistent_with_seed0(tmp_path, monkeypatch):
    fg = shifted(FG_GAIN)
    lead = copy.deepcopy(fg)
    for k, v in {"APS": 1.2, "AP75_small": 1.0, "AP": 0.1}.items():
        lead["600"][k] += v / 100.
    c = _world(tmp_path, monkeypatch, lead, shifted(W_GAIN))
    out = tmp_path / "summary-R32"
    out.mkdir()
    got = RL.summarize(c, out, {})
    result = json.loads((out / "RESULT.json").read_text())
    assert got["readings_B"]["1"] == "W_leads" and got["reading_B"] == "W_leads_1of3_F_geo_leads_0of3"
    assert result["W_unique_advantage"]["status"] == "not_established" and result["W_unique_advantage"]["inconsistent_with_seed0"] is True
    assert got["reading_A"] == "W_small_gain_repeated_3of3"
