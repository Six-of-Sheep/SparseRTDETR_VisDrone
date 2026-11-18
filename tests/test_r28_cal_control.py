"""R28-F* calibrated mapping control (TASK_R28_CALIBRATED_F_CONTROL_006): pure CPU checks of the adapter; every number
below is synthetic.

Covered: the registered basis, arm and five-arm timing order; the registered identities and workload; the release
difference (the science tree may only gain the arm FS module); the locked calibration and the input-pipeline switch;
the state binding of the kernel width; the calibration geometry against per-image references; the training-time
description of the match and its folding; the RECORD_ONLY summary arithmetic; the state digests, the difference
listing and the fixed decision rule of the input-pipeline acceptance.
"""
import copy
import json
import math
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "r28"))
import r28_common as K  # noqa: E402
import r28_cost as CO  # noqa: E402
import r28_eval as E  # noqa: E402
import r28_followup as FU  # noqa: E402
import r28_fovea_control as FC  # noqa: E402
import r28_cal_control as CC  # noqa: E402
import r28_loop as LP  # noqa: E402
import r28_pipeline as PL  # noqa: E402
from sparse_rtdetr.research import r28_fovea as FV  # noqa: E402
from sparse_rtdetr.research import r28_fovea_cal as FS  # noqa: E402
from sparse_rtdetr.research import r28_resample as R  # noqa: E402
from sparse_rtdetr.research import r28_warp as W  # noqa: E402

N = E.METRICS
F64 = torch.float64
ROOT = Path(__file__).resolve().parents[1]


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return K.ref(path)


def cal_contract(inherited=None):
    return {"contract_sha256": "new", "authorization_basis": K.CAL_BASIS,
            "followup": {"authorization_basis": K.CAL_BASIS, "arm": "FS", "prior_contract_sha256": "cw",
                         "inherited": inherited or {}, "inherited_contracts": {}}}


# ---------------------------------------------------------------- registration
def test_basis_arm_module_and_five_arm_timing_order():
    assert K.CAL_BASIS == "post_F_user_forwarded_calibrated_mapping_control"
    assert {k: v for k, v in K.FOLLOWUP_ARMS.items() if v not in ("FG", "R29")} == {"post_W_user_approved_K_followup": "K",
                                                                       "post_K_user_forwarded_mapping_control": "F", K.CAL_BASIS: "FS"}
    assert K.ARMS[:4] == ("W", "K", "F", "FS") and K.arm_module("FS") is FS and K.arm_module("F") is FV and K.arm_module("W") is R
    assert K.followup_arm(cal_contract()) == "FS"
    for bad in ({"authorization_basis": K.CAL_BASIS, "followup": {"authorization_basis": K.CAL_BASIS, "arm": "F"}},
                {"authorization_basis": K.FOVEA_BASIS, "followup": {"authorization_basis": K.FOVEA_BASIS, "arm": "FS"}},
                {"authorization_basis": K.CAL_BASIS, "followup": {"authorization_basis": K.FOVEA_BASIS, "arm": "FS"}}):
        with pytest.raises(ValueError):
            K.followup(bad)
    order = K.COST_ENDPOINTS["cost-FS"]
    rotations = K.COST_ROTATIONS["cost-FS"]
    assert order == ("L0", "K", "F", "FS", "W") == CC.ARMS5                    # task section 4: the fixed order,
    assert rotations == (order, order[::-1], order[2:] + order[:2])             # its reverse, rotated left by two
    assert K.WORKLOAD["cost-FS"]["forwards"] == 250 and CO.CANVAS["FS"] == "H"
    assert K.COST_ENDPOINTS["cost-F"] == ("L0", "W", "F") and K.COST_ENDPOINTS["cost-K"] == ("L0", "W", "K")     # earlier rounds unchanged
    assert CC.ARM == "FS" and CC.LABEL == "calibrated_mapping_control_completed" and CC.CONTRIBUTION == "not_established"
    assert CC.PRIMARY == (("600", "APS"), ("600", "AP75_small")) and CC.PAIRS[:3] == (("W", "FS"), ("FS", "F"), ("FS", "L0"))


def test_registered_identities_and_workload():
    assert CC.TASK == ("TASK_R28_CALIBRATED_F_CONTROL_006.md", "a627a7e2119f9946ea8e4c31d470fdb544f955e4a7b01de6d6453a154465acf2")
    assert CC.REVIEW[1].startswith("54864e4e") and CC.ENGINEERING[1].startswith("23b93f18")
    p = CC.PIN_F                                                                # the table of task section 1
    assert p["commit"] == "e78e8cbe310084da5523466833086796c91ea554"
    assert p["manifest"] == "ca0fc9cba3a7d1666a98318119b356365bd499d6d6caad7a2d2ac3c2a5d3e6c8"
    assert p["parent"] == "a2b6412e3c85de7592829f24aff09e355fdd11cb53798bc498bd04b882e1bf8a" == K.PARENT_SHA == FU.PIN["parent"]
    assert p["W_demand"] == "c9ffc7505e47adbcc45d70949c3d87371c6f3a7795f1aaaab886d98b15ee9e5a" == FU.PIN["W_demand"]
    assert p["endpoint"] == "a0dc7d6dec0bd3ed95490c4a20ceb9b8d2926336c2036575febe98a0f8386798"
    assert p["official_commit"] == "c8707bd2e4f10678f6636d24c2915a43b44a59ed" == FV.OFFICIAL["commit"]
    w = CC.WORKLOAD
    assert w["planned_updates"] == 18240 + 80 == 18320 and w["planned_complete_detector_forwards"] == 36480 + 160 + 141 + 250 == 37031
    assert w["pipeline"] == {"updates": 80, "forwards": 160, "paths": {"serial": 32, "overlap": 32, "resumed": 16}}
    assert w["calibrate"]["demand_forward_images"] == 5376 and w["calibrate"]["detector_forwards"] == 0 and w["calibrate"]["updates"] == 0
    assert (w["cal-fixture"]["updates"], w["cal-fixture"]["forwards"], w["wire"]["forwards"], w["cal-check"]["forwards"]) == (3, 6, 11, 0)
    assert CC.PIPE == {"steps": 32, "save_at": 16, "steady_from": 5, "ready_limit": 1, "workers": 2}
    assert (PL.READY_LIMIT, PL.WORKERS, PL.VERSION) == (CC.PIPE["ready_limit"], CC.PIPE["workers"], CC.PIPELINE_RULE["version"])
    assert CC.E118_START == 117 * 304 == 35568 and K.PARENT_UPDATE == 18240
    assert CC.OWN_STAGES == ("cal-check", "calibrate", "cal-fixture", "wire-FS", "pipeline", "formal-FS", "eval-FS", "cost-FS", "summary-FS")
    assert set(CC.OWN_STAGES).isdisjoint(FU.INHERITED) and set(CC.OWN_STAGES).isdisjoint(FC.INHERITED_K) and set(CC.OWN_STAGES).isdisjoint(CC.INHERITED_F)
    assert CC.INHERITED_F == FC.OWN_STAGES and CC.OFFICIAL_WIDTHS == (4.0, 13.0, 30.0)
    assert all((ROOT / t).is_file() for t in CC.TESTS) and len(set(CC.TESTS)) == len(CC.TESTS)
    assert all((ROOT / f).is_file() for f in CC.ALLOWED_ADDED + CC.ALLOWED_CHANGED)
    assert CC.SCIENCE_ADDED == ("src/sparse_rtdetr/research/r28_fovea_cal.py",)
    assert set(CC.COST_PAIRS) <= {f"{a}/{b}" for a in CC.ARMS5 for b in CC.ARMS5 if a != b}
    assert CC.EXCLUDED and CC.COMPARED and not any("state" in e for e in CC.EXCLUDED)      # no state subtree is excluded
    assert (CC.DEV_CAL, CC.DEV_PIPE, CC.DEV_TARGET) == ("R28_DEV_CAL_WINDOWS", "R28_DEV_PIPE", "R28_DEV_CAL_TARGET_WIDTH")
    chain = (ROOT / "tools/r28/r28_cal_chain.sh").read_text()
    assert "R28_DEV_* variables are refused outside a dev round" in chain and "needs --until" in chain


def test_source_delta_lets_the_science_tree_only_gain_the_arm_fs_module():
    old = {"src/sparse_rtdetr/research/r28_warp.py": "1", "src/sparse_rtdetr/research/r28_fovea.py": "2",
           "tools/r28/r28_common.py": "3", "tools/r28/r28_fovea_control.py": "4", "vendor/x.py": "5", "tests/test_r28_fovea_control.py": "6"}
    new = {**old, "tools/r28/r28_common.py": "3b", "tests/test_r28_fovea_control.py": "6b", "src/sparse_rtdetr/research/r28_fovea_cal.py": "7",
           "tools/r28/r28_cal_control.py": "8", "tools/r28/r28_pipeline.py": "9", "tools/r28/r28_cal_chain.sh": "10"}
    d = CC.source_delta(old, new, label="t")
    assert d["changed"] == ["tests/test_r28_fovea_control.py", "tools/r28/r28_common.py"] and d["removed"] == []
    assert d["science_files_added"] == ["src/sparse_rtdetr/research/r28_fovea_cal.py"] and d["existing_science_files_identical"] is True
    for bad in ({**new, "src/sparse_rtdetr/research/r28_fovea.py": "x"},        # arm F's module changed
                {**new, "src/sparse_rtdetr/research/r28_warp.py": "x"},         # arm W's operator changed
                {**new, "src/sparse_rtdetr/research/other.py": "x"},            # another science file added
                {**new, "vendor/x.py": "x"},                                    # detector changed
                {**new, "tools/r28/r28_fovea_control.py": "x"},                 # the F round's control tool changed
                {**new, "tools/r28/unlisted.py": "x"},                          # an unlisted tool added
                {k: v for k, v in new.items() if k != "vendor/x.py"}):          # a file removed
        with pytest.raises(ValueError):
            CC.source_delta(old, bad, label="t")
    assert not any(p.startswith(("src/", "vendor/", "configs/")) for p in CC.ALLOWED_CHANGED)


# ---------------------------------------------------------------- lock, pipeline switch, state binding
def test_locked_calibration_demand_source_and_pipeline_mode(tmp_path, monkeypatch):
    new = tmp_path / "new"
    new.mkdir()
    monkeypatch.setattr(K, "ROUND", new)
    monkeypatch.delenv("R28_PIPELINE", raising=False)
    demand = _write(tmp_path / "w" / "demand" / "CLOSED.json", {"status": "PASS", "contract_sha256": "cw", "arm": "W", "weights": {"path": "x"}})
    c = cal_contract({"demand": demand})
    for fn in (lambda: K.calibration(c), lambda: K.demand_source(c, "FS"), lambda: K.pipeline_mode(c, "FS")):
        with pytest.raises(ValueError):                                         # nothing is closed yet
            fn()
    h = 13.59375 + 2.0 ** -30
    body = {"status": "matched", "h_star_hex": h.hex(), "contract_sha256": "new"}
    closed = {"status": "PASS", "contract_sha256": "new", "arm": "FS", "calibration_status": "matched", "h_star_hex": h.hex()}
    cal = _write(new / "calibrate" / "CALIBRATION.json", body)
    _write(new / "calibrate" / "CLOSED.json", {**closed, "calibration": cal})
    lock = K.calibration(c)
    assert lock == {"fwhm": h, "calibration": cal} and type(lock["fwhm"]) is float and lock["fwhm"].hex() == h.hex()
    source = K.demand_source(c, "FS")
    assert source == {"kind": "network", "file": {"path": "x"}, "fwhm": h, "calibration": cal}
    assert K.demand_source(c, "F") == K.demand_source(c, "W") == {"kind": "network", "file": {"path": "x"}}
    assert FS.check_source("FS", source, False) == {"fwhm": h, "fwhm_hex": h.hex(), "calibration": cal}
    for status in ("calibration_unmatched", "calibration_invalid", "confirmation_invalid"):
        _write(new / "calibrate" / "CLOSED.json", {**closed, "calibration_status": status, "calibration": cal})
        with pytest.raises(ValueError):
            K.demand_source(c, "FS")                                            # arm FS cannot be built without a match
    _write(new / "calibrate" / "CLOSED.json", {**closed, "h_star_hex": (13.0).hex(), "calibration": cal})
    with pytest.raises(ValueError):
        K.calibration(c)                                                        # the stage and the file name different widths
    other = _write(new / "calibrate" / "CALIBRATION.json", {**body, "contract_sha256": "other"})
    _write(new / "calibrate" / "CLOSED.json", {**closed, "calibration": other})
    with pytest.raises(ValueError):
        K.calibration(c)                                                        # a calibration of another contract
    with pytest.raises(ValueError):
        K.calibration({"contract_sha256": "f", "authorization_basis": K.FOVEA_BASIS,
                       "followup": {"authorization_basis": K.FOVEA_BASIS, "arm": "F", "inherited": {}}})
    # the input pipeline of the formal run
    assert K.pipeline_mode(c, "F") == K.pipeline_mode(c, "W") == "serial"       # other arms never read the acceptance
    _write(new / "pipeline" / "CLOSED.json", {"status": "PASS", "contract_sha256": "new", "arm": "FS", "overlap_enabled": True})
    assert K.pipeline_mode(c, "FS") == "overlap"
    monkeypatch.setenv("R28_PIPELINE", "serial")
    assert K.pipeline_mode(c, "FS") == "serial"
    monkeypatch.setenv("R28_PIPELINE", "overlap")
    with pytest.raises(ValueError):
        K.pipeline_mode(c, "FS")                                                # the variable cannot force the new path
    monkeypatch.delenv("R28_PIPELINE")
    for value in (False, 1, "true", None):
        _write(new / "pipeline" / "CLOSED.json", {"status": "PASS", "contract_sha256": "new", "arm": "FS", "overlap_enabled": value})
        assert K.pipeline_mode(c, "FS") == "serial"


def test_state_binding_carries_the_width_for_arm_fs_only():
    c = {"contract_sha256": "x", "identity_sha256": {"FS": "i", "F": "j"}, "parent": {"sha256": "p"}}
    mapping = {"fwhm": 13.5, "fwhm_hex": (13.5).hex(), "calibration": {"path": "c", "bytes": 1, "sha256": "s"}}

    def engine(arm, **extra):
        return types.SimpleNamespace(r28_arm=arm, completed_updates=K.PARENT_UPDATE + 16, s15_activation_checkpoint=True,
                                     config=types.SimpleNamespace(binding=lambda: {"seed": 0}),
                                     r28_ctx=types.SimpleNamespace(binding=lambda: {"kind": "network"}), **extra)

    b = K.binding(c, engine("FS", r28_mapping=mapping), "pipeline-acceptance")
    assert b["mapping"] == mapping and b["arm"] == "FS" and b["scope"] == "pipeline-acceptance" and b["identity_sha256"] == "i"
    assert b["global_update"] == K.PARENT_UPDATE + 16 and b["cursor"]["next_window"] == 16
    assert "mapping" not in K.binding(c, engine("F"), "formal")
    with pytest.raises(ValueError):
        K.binding(c, engine("FS"), "formal")                                    # an arm FS state always names its width
    with pytest.raises(ValueError):
        K.binding(c, engine("F", r28_mapping=mapping), "formal")                # and no other arm does


# ---------------------------------------------------------------- calibration geometry
AREAS = (100.0, 1024.0, 2000.0, 9216.0, 20000.0)


def make_cache(seed=0, windows=3, images=4):
    g = torch.Generator().manual_seed(seed)
    out = {"demand": [], "boxes": [], "raw_area": [], "image_ids": [], "windows": []}
    for w in range(windows):
        for i in range(images):
            n = 0 if (w, i) == (1, 2) else int(torch.randint(1, 7, (1,), generator=g))
            wh = torch.rand(n, 2, generator=g) * 0.08 + 0.004
            centre = wh / 2 + torch.rand(n, 2, generator=g) * (1 - wh)
            out["boxes"].append(torch.cat([centre, wh], 1))
            out["raw_area"].append(torch.tensor([AREAS[(w + i + k) % 5] for k in range(n)], dtype=torch.float32))
            out["image_ids"].append(f"w{w}i{i}")
            out["demand"].append(W.blur3(torch.rand(*W.GRID, generator=g, dtype=F64) * 3))
        out["windows"].append({"input": {"update": K.PARENT_UPDATE + 1 + w}, "images": images})
    out["demand"] = torch.stack(out["demand"])
    return out


def per_image_logs(cache, plan_of, mag, cls):
    """Reference written apart from the block code: one image at a time, log sqrt(ox oy) of the boxes of a class."""
    logs = []
    for d, boxes, area in zip(cache["demand"], cache["boxes"], cache["raw_area"]):
        if not len(boxes):
            continue
        ox, oy = mag(plan_of(d[None]), W.cxcywh_to_xyxy(boxes.to(F64))[None])
        a = area.double()
        sel = {"small": a <= 1024.0, "medium": (a >= 1024.0) & (a <= 9216.0), "large": a >= 9216.0}[cls]
        logs += (0.5 * torch.log(ox * oy))[0][sel].tolist()
    return logs


def test_window_blocks_and_pooled_statistic_match_per_image_references():
    cache = make_cache()
    blocks = CC.window_blocks(cache)
    assert [b["update"] for b in blocks] == [K.PARENT_UPDATE + 1, K.PARENT_UPDATE + 2, K.PARENT_UPDATE + 3]
    assert sum(int(b["real"].sum()) for b in blocks) == sum(len(b) for b in cache["boxes"]) and all(b["non_positive"] == 0 for b in blocks)
    assert not bool(blocks[1]["real"][2].any())                                 # the empty image contributes nothing
    areas = torch.cat(cache["raw_area"])
    assert sum(int(b["small"].sum()) for b in blocks) == int((areas <= 1024).sum())
    assert sum(int((b["small"] & b["medium"]).sum()) for b in blocks) == int((areas == 1024).sum()) > 0     # inclusive ends, as COCOeval
    w_logs, w_aniso = CC.geometry_w(blocks)
    valid, f_logs, f_aniso, detail = CC.geometry_h(blocks, 4.0)
    assert valid and detail["invalid_images"] == 0 and detail["node_slope_min"] > 0 and detail["endpoint_error_max"] <= 1e-12
    for cls in ("small", "medium", "large"):
        for logs, plan_of, mag in ((w_logs, W.plan_from_demand, W.magnification), (f_logs, FV.plan_from_demand, FV.magnification)):
            pool, ref = CC._pool(logs, blocks, cls), per_image_logs(cache, plan_of, mag, cls)
            assert pool["n"] == len(ref) > 0 and abs(pool["sum_log_g"] - math.fsum(ref)) <= 1e-10
            assert abs(pool["L"] - math.fsum(ref) / len(ref)) <= 1e-12 and abs(pool["pooled_geometric_mean"] - math.exp(pool["L"])) <= 1e-15
            assert len(pool["per_window"]) == 3 and sum(pool["per_window_n"]) == pool["n"]
    r = CC.reading(blocks, f_logs, f_aniso)
    assert set(r) == {"small", "medium", "large"} and all(sum(v for k, v in r[c]["bins"].items() if k not in ("min", "max")) == r[c]["n"] for c in r)
    wide_valid, wide_logs, _, _ = CC.geometry_h(blocks, 13.0)
    if wide_valid:                                                              # another width is another geometry on the same cache
        assert CC._pool(wide_logs, blocks, "small")["L"] != CC._pool(f_logs, blocks, "small")["L"]
    bad = copy.deepcopy(cache)
    bad["boxes"][0][0, 2] = 0.0                                                 # a zero-width box is counted, not dropped
    assert CC.window_blocks(bad)[0]["non_positive"] == 1
    short = copy.deepcopy(cache)
    short["windows"][0]["images"] = 3
    with pytest.raises(ValueError):
        CC.window_blocks(short)


def test_invalid_maps_make_a_width_invalid_and_are_not_repaired(monkeypatch):
    blocks = CC.window_blocks(make_cache())
    real = FS.attraction_nodes

    def plateau(s, h):
        raw, g = real(s, h)
        g = g.clone()
        g[1, 9] = g[1, 8]                                                       # one image of every window
        return raw, g

    monkeypatch.setattr(FS, "attraction_nodes", plateau)
    valid, logs, aniso, detail = CC.geometry_h(blocks, 13.0)
    assert valid is False and logs is None and aniso is None and detail["invalid_images"] == 3
    assert detail["invalid_flags"] == {"not_strictly_increasing": 3} and "node_slope_min" not in detail
    out = FS.select(lambda h: (lambda v, lg, an, d: (v, None, d))(*CC.geometry_h(blocks, h)), 0.2)
    assert out["status"] == "calibration_invalid" and out["h_star"] is None and len(out["readings"]) == 66


def test_selection_on_cached_geometry_and_the_historical_caliber(tmp_path):
    blocks = CC.window_blocks(make_cache(seed=3))
    w_logs, _ = CC.geometry_w(blocks)
    w_pool = CC._pool(w_logs, blocks, "small")
    _, f_logs, _, _ = CC.geometry_h(blocks, 4.0)
    f_pool = CC._pool(f_logs, blocks, "small")

    def evaluate(h):
        valid, logs, _, detail = CC.geometry_h(blocks, h)
        return (True, CC._pool(logs, blocks, "small")["L"], detail) if valid else (False, None, detail)

    out = FS.select(evaluate, f_pool["L"])                                     # a target that width 4 reproduces exactly
    assert out["status"] == "matched" and out["h_star"] == 4.0 and out["relative_error"] == 0.0 and out["L_h_star"] == f_pool["L"]
    out = FS.select(evaluate, w_pool["L"])                                     # arm W's target: whatever the rule returns is recorded
    assert out["status"] in FS.STATUSES and out["L_W"] == w_pool["L"] and len(out["readings"]) >= 66
    if out["status"] == "matched":
        assert abs(out["relative_error"]) <= FS.TOLERANCE and evaluate(out["h_star"])[1] == out["L_h_star"]
    # the per-update geometric means a formal ledger stores
    ledger = tmp_path / "UPDATES.jsonl"
    with ledger.open("w") as f:
        for b, g, n in zip(blocks, w_pool["per_window"], w_pool["per_window_n"]):
            f.write(json.dumps({"input": {"update": b["update"]}, "r28": {"g_small": g, "n_small": n}}) + "\n")
        f.write(json.dumps({"input": {"update": blocks[-1]["update"] + 1}, "r28": {"g_small": 9.0, "n_small": 1}}) + "\n")
    stored = CC.ledger_g_small(ledger, [b["update"] for b in blocks])
    hist = CC.against_ledger(blocks, w_pool, stored)
    assert hist["small_counts_equal"] is True and hist["max_abs_difference_of_per_window_geometric_mean"] == 0.0
    assert hist["historical_caliber_recomputed"] == pytest.approx(hist["historical_caliber_stored"], abs=1e-15)
    assert abs(hist["historical_caliber_recomputed"] - w_pool["pooled_geometric_mean"]) > 0      # the two calibers are not one number
    other = CC.against_ledger(blocks, f_pool, stored)
    assert other["max_abs_difference_of_per_window_geometric_mean"] > 1e-6      # arm F's geometry is not arm W's ledger
    moved = {u: (g, n + 1) for u, (g, n) in stored.items()}
    assert CC.against_ledger(blocks, w_pool, moved)["small_counts_equal"] is False
    with pytest.raises(ValueError):
        CC.ledger_g_small(ledger, [blocks[0]["update"], blocks[-1]["update"] + 5])


# ---------------------------------------------------------------- training-time description and its folding
def forward_records(cache, first, h):
    """The two forward records of one update of four images, as r28_fovea_cal.observe stores them."""
    recs, area = [], []
    for at in (first, first + 2):
        d = cache["demand"][at:at + 2]
        t = [{"boxes": cache["boxes"][i]} for i in (at, at + 1)]
        plan = FS.plan_from_demand(d, h)
        pad, n = R._padded_xyxy(t)
        ox, oy = FV.magnification(plan, pad)
        recs.append({"n": n, "demand": d, "xyxy": pad, "ox": ox, "oy": oy})
        area += [cache["raw_area"][at], cache["raw_area"][at + 1]]
    return recs, torch.cat(area)


def test_cal_row_is_the_calibration_statistic_of_one_update():
    cache = make_cache(seed=1, windows=2, images=4)
    blocks = CC.window_blocks(cache)
    recs, area = forward_records(cache, 0, 4.0)
    row = LP.cal_row(recs, area)
    w_logs, w_aniso = CC.geometry_w(blocks)
    valid, f_logs, f_aniso, _ = CC.geometry_h(blocks, 4.0)
    assert valid and set(row) == {"small", "medium", "large"}
    for cls in row:
        cell = row[cls]
        for arm, logs, aniso in (("W", w_logs, w_aniso), ("FS", f_logs, f_aniso)):
            want = logs[0][blocks[0][cls]]
            ratio = aniso[0][blocks[0][cls]]
            assert cell["n"] == int(blocks[0][cls].sum()) == sum(cell[arm]["bins"])
            assert abs(cell[arm]["sum_log_g"] - math.fsum(want.tolist())) <= 1e-10
            assert abs(cell[arm]["sum_log_ox_over_oy"] - math.fsum(ratio.tolist())) <= 1e-10
            assert abs(cell[arm]["sum_abs_log_ox_over_oy"] - math.fsum(ratio.abs().tolist())) <= 1e-10
    json.dumps(row, allow_nan=False)
    with pytest.raises(ValueError):
        LP.cal_row(recs, area[:-1])


def test_formal_calibration_folds_the_ledger(tmp_path):
    cache = make_cache(seed=2, windows=2, images=4)
    rows = []
    for w, epoch in ((0, 61), (1, 62)):
        recs, area = forward_records(cache, 4 * w, 4.0)
        rows.append({"input": {"update": K.PARENT_UPDATE + 1 + w, "epoch": epoch}, "r28": {"cal": LP.cal_row(recs, area)}})
    with (tmp_path / "UPDATES.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    out = CC.formal_calibration(tmp_path)
    assert out["updates"] == 2 and list(out["per_epoch"]) == [61, 62]
    for cls in ("small", "medium", "large"):
        n = sum(r["r28"]["cal"][cls]["n"] for r in rows)
        cell = out["overall"][cls]
        assert cell["n"] == n > 0
        for arm in ("FS", "W"):
            total = math.fsum(r["r28"]["cal"][cls][arm]["sum_log_g"] for r in rows)
            assert abs(cell[arm]["L"] - total / n) <= 1e-15 and sum(cell[arm]["bins"]) == n and abs(sum(cell[arm]["bin_shares"]) - 1) <= 1e-12
        assert abs(cell["FS_over_W_minus_1"] - math.expm1(cell["FS"]["L"] - cell["W"]["L"])) <= 1e-15
    first = rows[0]["r28"]["cal"]["small"]
    assert abs(out["per_epoch"][61]["small"]["FS"]["L"] - first["FS"]["sum_log_g"] / first["n"]) <= 1e-15
    lo, hi = out["small_FS_over_W_minus_1_range"]
    assert lo <= out["overall"]["small"]["FS_over_W_minus_1"] <= hi and set(out["small_FS_over_W_minus_1_by_epoch"]) == {61, 62}
    json.dumps(out, allow_nan=False)


# ---------------------------------------------------------------- summary arithmetic
def v(six, three):
    return {"600": dict(zip(N, six)), "300": dict(zip(N, three))}


L0V = v((0.3000, 0.2200, 0.2000, 0.3950, 0.4950, 0.5050, 0.3010), (0.2960, 0.2190, 0.1990, 0.3900, 0.4920, 0.4950, 0.2980))
GTS = (23, 5, 3, 779, 115, 87, 12, 5, 32, 7)


def shifted(base, d600):
    out = copy.deepcopy(base)
    for k, d in d600.items():
        out["600"][k] += d / 100.
    return out


def rows_of(key, values, gts=GTS):
    return [{"category_id": i + 1, "GT": g, key: x} for i, (g, x) in enumerate(zip(gts, values))]


def test_summary_fs_is_record_only_arithmetic():
    values = {"L0": L0V, "W": shifted(L0V, {"APS": 1.8, "AP75_small": 2.5, "APM": -0.7}),
              "K": shifted(L0V, {"APS": 1.0, "AP75_small": 1.0, "APM": -0.7}), "F": shifted(L0V, {"APS": 0.4, "AP75_small": 0.4, "APM": -0.5}),
              "FS": shifted(L0V, {"APS": 1.5, "AP75_small": 1.2, "APM": -0.2})}
    apl = {a: rows_of("APL", [0.30 + 0.05 * i + s for i in range(10)]) for a, s in (("L0", 0), ("W", 0.01), ("K", 0), ("F", 0), ("FS", 0.02))}
    apm = {a: rows_of("APM", [0.20 + 0.03 * i for i in range(10)]) for a in CC.ARMS5}
    every = {f"{a}/{b}": 1.0 + 0.01 * (CC.ARMS5.index(a) - CC.ARMS5.index(b)) for a in CC.ARMS5 for b in CC.ARMS5 if a != b}
    cost = {"ratios": {"model_gpu": dict(every), "end_to_end": dict(every)}, "medians": {"FS": {}}, "stages": {"table": {}}}
    decision = {"label_key": "closed", "K_trigger": False, "exploratory": False, "label": "closed_fixed_config"}
    w_result = {"values": {"L0": L0V, "W": values["W"], "A0": L0V}, "decision": decision}
    res = CC.summary_fs({}, w_result, values, apl, apm, cost, {"F_round_L0_W_F": {}, "K_round_L0_W_K": {}})
    assert res["label"] == "calibrated_mapping_control_completed" and res["contribution"] == "not_established" and res["record_only"] is True
    assert not any("passed" in k or "gate" in k for k in res)                           # no pass / fail cell anywhere
    p = res["primary_W_minus_FS_pp"]
    assert list(p) == ["W-FS top600 APS", "W-FS top600 AP75_small"]
    assert p["W-FS top600 APS"] == pytest.approx(0.3) and p["W-FS top600 AP75_small"] == pytest.approx(1.3)
    assert list(res["differences_pp"]) == [f"{x}-{y}" for x, y in CC.PAIRS]
    assert res["differences_pp"]["FS-F"]["600"]["APS"] == pytest.approx(1.1) and res["differences_pp"]["FS-L0"]["600"]["APM"] == pytest.approx(-0.2)
    assert res["differences_pp"]["W-FS"]["300"]["AP"] == 0.0 and res["differences_pp"]["W-F"]["600"]["APS"] == pytest.approx(1.4)
    assert set(res["APL6_pp"]) == {"W-FS", "FS-F", "FS-L0"} and res["APL6_pp"]["W-FS"]["difference_pp"] == pytest.approx(-1.0)
    assert res["cost_FS"]["model_gpu"] == {k: every[k] for k in CC.COST_PAIRS} and res["cost_FS"]["all_ratios"]["model_gpu"] == every
    assert res["values"]["FS"] == values["FS"] and res["values"]["F"] == values["F"] and res["values"]["A0"] == L0V
    assert set(res["value_sources"]) == set(CC.ARMS5) and res["W_decision_unchanged"] == decision
    assert res["per_class_APL_top600"][3] == {"category_id": 4, "GT": 779, "L0": pytest.approx(0.45), "K": pytest.approx(0.45),
                                              "F": pytest.approx(0.45), "FS": pytest.approx(0.47), "W": pytest.approx(0.46)}
    with pytest.raises(ValueError):                                                     # an arm missing
        CC.summary_fs({}, w_result, {a: values[a] for a in ("L0", "W", "F", "FS")}, apl, apm, cost, {})
    with pytest.raises(ValueError):                                                     # lists scored against different GT
        CC.summary_fs({}, w_result, values, apl, {**apm, "FS": rows_of("APM", [0.2] * 10, (1,) * 10)}, cost, {})


# ---------------------------------------------------------------- input-pipeline acceptance helpers
def fake_state(seed=0):
    g = torch.Generator().manual_seed(seed)
    t = lambda *s: torch.rand(*s, generator=g)
    return {"version": "v", "arm": "FS", "parent_sha256": "p", "config": {"seed": 0, "lr": 1e-4}, "activation_checkpoint": True,
            "model": {"conv.weight": t(2, 3), "bn.running_mean": t(3), "bn.running_var": t(3), "bn.num_batches_tracked": torch.tensor(7)},
            "model_modes": {"": True, "bn": True}, "optimizer": {"state": {0: {"exp_avg": t(2, 3), "step": torch.tensor(7.0)}},
                                                              "param_groups": [{"lr": 1e-4, "betas": (0.9, 0.999), "param_names": ["conv.weight"]}]},
            "optimizer_names": [["conv.weight"]], "ema": {"module": {"conv.weight": t(2, 3)}, "updates": 7},
            "engine": {"completed_updates": 7, "optimizer_steps": {"conv.weight": 7}},
            "rng": {"python": (3, (1, 2, 3), None), "numpy": ("MT19937", [1, 2], 5, 0, 0.0), "torch_cpu": t(8), "cuda": [t(4)]}}


def test_state_digests_see_every_part_of_a_state():
    a = fake_state()
    base = CC.state_digests(a)
    assert set(base) == set(CC.STATE_PARTS) | {"rng_cpu", "rng_cuda", "bn_buffers"} and base == CC.state_digests(copy.deepcopy(a))
    reordered = {k: a[k] for k in reversed(list(a))}
    reordered["model"] = {k: a["model"][k] for k in reversed(list(a["model"]))}
    assert CC.state_digests(reordered) == base                                  # key order is not content

    def changed(edit):
        b = copy.deepcopy(a)
        edit(b)
        d = CC.state_digests(b)
        return sorted(k for k in d if d[k] != base[k])

    assert changed(lambda s: s["model"]["conv.weight"].__setitem__((0, 0), 9.0)) == ["model"]
    assert changed(lambda s: s["model"]["bn.running_var"].add_(1e-7)) == ["bn_buffers", "model"]
    assert changed(lambda s: s["model"].__setitem__("bn.num_batches_tracked", torch.tensor(8))) == ["bn_buffers", "model"]
    assert changed(lambda s: s["optimizer"]["state"][0]["exp_avg"].mul_(2)) == ["optimizer"]
    assert changed(lambda s: s["optimizer"]["param_groups"][0].__setitem__("lr", 2e-4)) == ["optimizer"]
    assert changed(lambda s: s["ema"].__setitem__("updates", 8)) == ["ema"]
    assert changed(lambda s: s["engine"].__setitem__("completed_updates", 8)) == ["engine"]
    assert changed(lambda s: s["model_modes"].__setitem__("bn", False)) == ["model_modes"]
    assert changed(lambda s: s["rng"]["torch_cpu"].add_(1)) == ["rng", "rng_cpu"]
    assert changed(lambda s: s["rng"]["cuda"][0].add_(1)) == ["rng", "rng_cuda"]
    assert changed(lambda s: s["rng"].__setitem__("python", (3, (1, 2, 4), None))) == ["rng", "rng_cpu"]
    assert changed(lambda s: s["engine"].__setitem__("completed_updates", 7.0)) == ["engine"]       # a type change is a change
    assert changed(lambda s: s["model"].__setitem__("conv.weight", s["model"]["conv.weight"].double())) == ["model"]
    with pytest.raises(ValueError):
        CC.state_digests({k: v for k, v in a.items() if k != "ema"})            # a missing part is not skipped
    with pytest.raises(TypeError):
        CC.deep_digest({"x": {1, 2}})                                           # an unknown value type is not skipped either
    d = CC.Digester()
    d.submit(("serial", 1), a)
    d.submit(("overlap", 1), copy.deepcopy(a))
    out = d.finish()
    assert out == {("serial", 1): base, ("overlap", 1): base}
    bad = CC.Digester()
    bad.submit(("serial", 1), {"version": "v"})
    with pytest.raises(ValueError):
        bad.finish()


def test_differences_lists_every_differing_leaf():
    a = {"input": {"update": 5, "sha": "aa"}, "result": {"loss": 1.5, "digests": ["x", "y"], "finite": True}, "state": {"model": "m"}}
    assert CC.differences(a, copy.deepcopy(a)) == [] and CC.differences([a, a], [a, a]) == []
    b = copy.deepcopy(a)
    b["result"]["digests"][1] = "z"
    b["state"]["model"] = "n"
    assert [d["path"] for d in CC.differences(a, b)] == ["/result/digests/1", "/state/model"]
    assert CC.differences({"u": 1}, {"u": 1.0})[0]["kind"] == "value"           # 1 and 1.0 are not the same record
    assert CC.differences({"u": True}, {"u": 1})[0]["kind"] == "value"
    assert CC.differences({"u": 1}, {"v": 1}) == [{"path": "/u", "kind": "missing"}, {"path": "/v", "kind": "missing"}]
    assert CC.differences([1, 2], [1, 2, 3])[0]["kind"] == "length"
    assert len(CC.differences(list(range(100)), list(range(1, 101)))) == 40     # the listing is bounded, the verdict is not


def timing_rows(n, *, cycle, wait, train_call, capture, next_s, reference, save_at=None):
    rows = []
    for i in range(1, n + 1):
        extra = {"saved_with_prepared_windows": 1, "wait_for_prepared_s": 0.3, "save_s": 2.0} if i == save_at else {}
        first = 20.0 if i == 1 else 0.0                                         # worker start-up in the first cycle
        rows.append({"science": {"input": {"update": K.PARENT_UPDATE + i}},
                     "engineering": {"cycle_s": cycle + first + sum(v for k, v in extra.items() if k.endswith("_s")),
                                     "wait_s": wait + first, "train_call_s": train_call, "train_s": train_call - 0.01,
                                     "acceptance_capture_s": capture,
                                     "input": {"next_s": next_s, "fetch_s": next_s - 0.6, "audit_s": 0.6, "per_image_hash_s": 0.3,
                                               "window_pixel_digest_s": 0.25, "window_target_digest_s": 0.05, "reference_check_s": reference},
                                     **extra}})
    return rows


def test_path_timing_and_the_fixed_decision_rule():
    serial = CC.path_timing(timing_rows(32, cycle=2.2, wait=1.0, train_call=1.15, capture=0.05, next_s=0.99, reference=0.01), 5)
    overlap = CC.path_timing(timing_rows(32, cycle=1.3, wait=0.02, train_call=1.15, capture=0.05, next_s=1.0, reference=0.01, save_at=16), 5)
    assert serial["updates"] == 32 and serial["steady_updates"] == 28 and serial["first_update"]["cycle_s"] == pytest.approx(22.2)
    assert serial["steady_cycle_s"] == pytest.approx(2.15) and serial["steady_cycle_measured_s"] == pytest.approx(2.2)
    assert overlap["steady_cycle_s"] == pytest.approx(1.25)                     # the save and its wait are acceptance-only work
    assert overlap["steady_cycle_measured_s"] == pytest.approx(1.3 + 2.3 / 28)
    assert overlap["steady_consumer_work_s"] == pytest.approx(1.23) and overlap["steady_producer_work_s"] == pytest.approx(1.01)
    assert overlap["steady_bound_s"] == pytest.approx(1.23) and overlap["steady_segments_mean_s"]["input_audit_s"] == pytest.approx(0.6)
    d = CC.decide(serial, overlap, True, True)
    assert d["overlap_enabled"] is True and d["formal_input_pipeline"] == "overlap" and d["measured_speedup"] == pytest.approx(2.15 / 1.25)
    assert CC.decide(serial, overlap, False, True)["formal_input_pipeline"] == "serial"         # not equivalent
    assert CC.decide(serial, overlap, True, False)["formal_input_pipeline"] == "serial"         # restore not exact
    slow = CC.path_timing(timing_rows(32, cycle=2.3, wait=0.02, train_call=2.2, capture=0.05, next_s=1.0, reference=0.01), 5)
    d = CC.decide(serial, slow, True, True)
    assert d["overlap_enabled"] is False and d["measured_steady_cycle_shorter"] is False and d["formal_input_pipeline"] == "serial"
    starved = CC.path_timing(timing_rows(32, cycle=2.0, wait=0.8, train_call=1.1, capture=0.05, next_s=2.4, reference=0.01), 5)
    d = CC.decide(serial, starved, True, True)                                  # measured shorter, but the producer bound is not
    assert d["measured_steady_cycle_shorter"] is True and d["conservative_bound_shorter"] is False and d["overlap_enabled"] is False
    assert CC.steady(timing_rows(8, cycle=1, wait=0, train_call=1, capture=0, next_s=1, reference=0), "cycle_s", 5) == [1, 1, 1, 1]
