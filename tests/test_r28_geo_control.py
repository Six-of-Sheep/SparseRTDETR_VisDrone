"""R28-F_geo strong mapping control (TASK_R28_STRONG_F_CONTROL_009): pure CPU checks of the adapter; every number
below is synthetic.

Covered: the registered basis, arm and five-arm timing order; the registered identities and workload; the release
difference (the science tree may only gain the arm FG module, the core files stay byte-identical) and the
function-level edit list; the selection block and the input-pipeline switch; the state binding of the fixed width;
the training-time geometry description and its folding; the RECORD_ONLY summary arithmetic; the ledger means.
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
import r28_geo_control as GC  # noqa: E402
import r28_loop as LP  # noqa: E402
import r28_pipeline as PL  # noqa: E402
from sparse_rtdetr.research import r28_fovea as FV  # noqa: E402
from sparse_rtdetr.research import r28_fovea_cal as FS  # noqa: E402
from sparse_rtdetr.research import r28_fovea_geo as FG  # noqa: E402
from sparse_rtdetr.research import r28_resample as R  # noqa: E402
from sparse_rtdetr.research import r28_warp as W  # noqa: E402

N = E.METRICS
F64 = torch.float64
ROOT = Path(__file__).resolve().parents[1]
CAL_SHA = "8dcfcd5cf2cbd7ef96a54b1832d9eabd6923a5b7da860a9e634784ed0fa05c59"


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return K.ref(path)


def geo_contract(sel=None, inherited=None):
    return {"contract_sha256": "new", "authorization_basis": K.GEO_BASIS,
            "followup": {"authorization_basis": K.GEO_BASIS, "arm": "FG", "prior_contract_sha256": "cw",
                         "inherited": inherited or {}, "inherited_contracts": {}, "width_selection": sel}}


# ---------------------------------------------------------------- registration
def test_basis_arm_module_and_five_arm_timing_order():
    assert K.GEO_BASIS == "post_006_user_forwarded_fixed_geometry_selected_width_control"
    assert K.FOLLOWUP_ARMS == {"post_W_user_approved_K_followup": "K", "post_K_user_forwarded_mapping_control": "F",
                               K.CAL_BASIS: "FS", K.GEO_BASIS: "FG"}
    assert K.ARMS == ("W", "K", "F", "FS", "FG") and K.arm_module("FG") is FG and K.arm_module("FS") is FS and K.arm_module("F") is FV
    assert K.arm_module("W") is R and K.followup_arm(geo_contract()) == "FG"
    for bad in ({"authorization_basis": K.GEO_BASIS, "followup": {"authorization_basis": K.GEO_BASIS, "arm": "FS"}},
                {"authorization_basis": K.GEO_BASIS, "followup": {"authorization_basis": K.GEO_BASIS, "arm": "F"}},
                {"authorization_basis": K.CAL_BASIS, "followup": {"authorization_basis": K.CAL_BASIS, "arm": "FG"}},
                {"authorization_basis": K.GEO_BASIS, "followup": {"authorization_basis": K.CAL_BASIS, "arm": "FG"}}):
        with pytest.raises(ValueError):
            K.followup(bad)
    order = K.COST_ENDPOINTS["cost-FG"]
    rotations = K.COST_ROTATIONS["cost-FG"]
    assert order == ("L0", "K", "F", "FG", "W") == GC.ARMS5                    # task section 4: the fixed order,
    assert rotations == (order, order[::-1], order[2:] + order[:2])             # its reverse, rotated left by two
    assert K.WORKLOAD["cost-FG"]["forwards"] == 250 and CO.CANVAS["FG"] == "H"
    assert K.COST_ENDPOINTS["cost-FS"] == ("L0", "K", "F", "FS", "W") and K.COST_ENDPOINTS["cost-F"] == ("L0", "W", "F")   # earlier rounds unchanged
    assert GC.ARM == "FG" and GC.LABEL == "strong_fovea_control_completed" and GC.CONTRIBUTION == "not_established"
    assert GC.PRIMARY == (("600", "APS"), ("600", "AP75_small")) and GC.PAIRS[:3] == (("W", "FG"), ("FG", "F"), ("FG", "L0"))


def test_registered_identities_and_workload():
    assert GC.TASK == ("TASK_R28_STRONG_F_CONTROL_009.md", "2e4f29f9cace208f339d2779a1179361ba7e188aa4420927c4006bcf22ff1de2")
    assert GC.REVIEW[0] == "REVIEW_R28_CALIBRATION_009.md" and GC.REVIEW[1].startswith("5e4d6af8")
    p = GC.PIN_CAL                                                              # the table of task section 1
    assert p["commit"] == "f376dc973d8333252d4cf50c9ed37d4fd8b9d244" == FG.SELECTION["source"]["commit"]
    assert p["manifest"] == "07247d2d2a19fca7daa14273617f178051f0c9e4a94ccd1effcb2000d8540cd3" == FG.SELECTION["source"]["release_manifest_sha256"]
    assert p["calibration"] == CAL_SHA == FG.SELECTION["source"]["calibration_sha256"]
    assert p["dev_equivalence"] == "a5a411b4f195062a7f64fdde862b4575f51c7ad3a4997ec1754579e8349e54f0"
    assert p["dev_timing"] == "5ffed79bd8fa914f9a3de6b7d705624b6ec41bc58787346fe61626ac240eea29"
    assert set(GC.CAL_FILES) == set(p) - {"commit", "contract"}
    f = CC.PIN_F
    assert f["commit"] == "e78e8cbe310084da5523466833086796c91ea554" and f["parent"] == "a2b6412e3c85de7592829f24aff09e355fdd11cb53798bc498bd04b882e1bf8a"
    assert f["W_demand"] == "c9ffc7505e47adbcc45d70949c3d87371c6f3a7795f1aaaab886d98b15ee9e5a"
    assert f["endpoint"] == "a0dc7d6dec0bd3ed95490c4a20ceb9b8d2926336c2036575febe98a0f8386798"
    w = GC.WORKLOAD
    assert w["planned_updates"] == 18240 + 10 == 18250 and w["planned_complete_detector_forwards"] == 36480 + 20 + 141 + 250 == 36891
    assert w["accept"] == {"updates": 10, "forwards": 20, "paths": {"serial": 4, "overlap": 4, "resumed": 2}}
    assert w["geo-confirm"]["demand_forward_images"] == 512 and w["geo-confirm"]["detector_forwards"] == 0 and w["geo-confirm"]["updates"] == 0
    assert (w["wire"]["forwards"], w["geo-check"]["forwards"]) == (11, 0)
    assert GC.ACCEPT == {"steps": 4, "save_at": 2, "ready_limit": 1, "workers": 2} and GC.CONFIRM == {"epoch": 118, "windows": 32, "occurrences": 512}
    assert (PL.READY_LIMIT, PL.WORKERS, PL.VERSION) == (GC.ACCEPT["ready_limit"], GC.ACCEPT["workers"], GC.PIPELINE_RULE["version"])
    assert "no timing enters the decision" in GC.PIPELINE_RULE["decision"]     # equality and exact restore are the criterion
    assert GC.OWN_STAGES == ("geo-check", "geo-confirm", "wire-FG", "accept-FG", "formal-FG", "eval-FG", "cost-FG", "summary-FG", "ledger-means")
    for other in (FU.INHERITED, FC.INHERITED_K, CC.INHERITED_F, CC.OWN_STAGES):
        assert set(GC.OWN_STAGES).isdisjoint(other)
    assert all((ROOT / t).is_file() for t in GC.TESTS) and len(set(GC.TESTS)) == len(GC.TESTS)
    assert all((ROOT / f).is_file() for f in GC.ALLOWED_ADDED + GC.ALLOWED_CHANGED + GC.CORE_IDENTICAL)
    assert GC.SCIENCE_ADDED == ("src/sparse_rtdetr/research/r28_fovea_geo.py",)
    assert set(GC.CORE_IDENTICAL).isdisjoint(GC.ALLOWED_CHANGED) and set(GC.EDITS) == {f for f in GC.ALLOWED_CHANGED if f.startswith("tools/")}
    assert set(GC.COST_PAIRS) <= {f"{a}/{b}" for a in GC.ARMS5 for b in GC.ARMS5 if a != b}
    assert GC.SCOPE == "fg-acceptance" and GC.DEV_CONFIRM == "R28_DEV_CONFIRM_WINDOWS"
    chain = (ROOT / "tools/r28/r28_geo_chain.sh").read_text()
    assert "R28_DEV_* variables are refused outside a dev round" in chain and "needs --until" in chain and K.GEO_BASIS in chain
    assert "calibration_unmatched None" in chain and FG.WIDTH_HEX in chain and "\r" not in chain


def test_source_delta_lets_the_science_tree_only_gain_the_arm_fg_module():
    old = {p: "c" for p in GC.CORE_IDENTICAL}
    old.update({"tools/r28/r28_common.py": "3", "tools/r28/r28_loop.py": "4", "vendor/x.py": "5", "tests/test_r28_cal_control.py": "6"})
    new = {**old, "tools/r28/r28_common.py": "3b", "tests/test_r28_cal_control.py": "6b", "src/sparse_rtdetr/research/r28_fovea_geo.py": "7",
           "tools/r28/r28_geo_control.py": "8", "tools/r28/r28_geo_chain.sh": "9"}
    d = GC.source_delta(old, new, label="t")
    assert d["changed"] == ["tests/test_r28_cal_control.py", "tools/r28/r28_common.py"] and d["removed"] == []
    assert d["science_files_added"] == ["src/sparse_rtdetr/research/r28_fovea_geo.py"] and d["existing_science_files_identical"] is True
    assert set(d["core_files_identical"]) == set(GC.CORE_IDENTICAL)
    for bad in ({**new, "src/sparse_rtdetr/research/r28_fovea.py": "x"},        # arm F's module changed
                {**new, "src/sparse_rtdetr/research/r28_fovea_cal.py": "x"},    # arm FS's module changed
                {**new, "src/sparse_rtdetr/research/other.py": "x"},            # another science file added
                {**new, "vendor/x.py": "x"},                                    # detector changed
                {**new, "tools/r28/r28_pipeline.py": "x"},                      # the input pipeline changed
                {**new, "tools/r28/r28_cal_control.py": "x"},                   # the comparison code changed
                {**new, "tools/r27_l0/l0_loop.py": "x"},                        # the loop base changed
                {**new, "src/sparse_rtdetr/protocol_r18/engine.py": "x"},       # the engine changed
                {**new, "tools/r28/unlisted.py": "x"},                          # an unlisted tool added
                {k: v for k, v in new.items() if k != "vendor/x.py"}):          # a file removed
        with pytest.raises(ValueError):
            GC.source_delta(old, bad, label="t")
    assert not any(p.startswith(("src/", "vendor/", "configs/")) for p in GC.ALLOWED_CHANGED)


def test_function_delta_names_exactly_the_edited_functions(tmp_path):
    old = "import os\n\ndef keep(a):\n    return a\n\n\ndef edit(a):\n    return a + 1\n\n\nclass C:\n    def m(self):\n        return 1\n\n    def n(self):\n        return 2\n"
    new = old.replace("return a + 1", "return a + 2").replace("return 2\n", "return 3\n") + "\n\ndef fresh():\n    return 0\n"
    d = GC.function_delta(old, new)
    assert d == {"changed": ["C.n", "edit"], "added": ["fresh"], "removed": [], "unchanged": ["C.m", "keep"]}
    assert GC.function_delta(old, old)["changed"] == [] and GC.function_delta(new, old)["removed"] == ["fresh"]
    # the registered edit list against a synthetic "old release": only the listed functions may differ
    rel = "tools/r28/r28_common.py"
    text = (ROOT / rel).read_text()
    for name in ("a", "b"):
        for r in GC.EDITS:
            (tmp_path / name / r).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / name / r).write_text((ROOT / r).read_text())
    with pytest.raises(ValueError):                                             # identical trees: the registered edits are missing
        GC.edits(tmp_path / "a", tmp_path / "b")
    assert "def save(" in text and "def selected_width(" in text
    assert set(GC.KEPT[rel]) <= set(GC.functions(text)) and set(GC.EDITS[rel]["changed"]) | set(GC.EDITS[rel]["added"]) <= set(GC.functions(text))
    for r, want in GC.EDITS.items():
        names = set(GC.functions((ROOT / r).read_text()))
        assert set(want["changed"]) | set(want["added"]) | set(GC.KEPT.get(r, ())) <= names, r


# ---------------------------------------------------------------- selection, pipeline switch, state binding
def test_selected_width_demand_source_and_pipeline_mode(tmp_path, monkeypatch):
    new = tmp_path / "new"
    new.mkdir()
    monkeypatch.setattr(K, "ROUND", new)
    monkeypatch.delenv("R28_PIPELINE", raising=False)
    monkeypatch.setitem(FG.SELECTION["source"], "calibration_sha256", None)     # the file below is synthetic: pin its SHA for this test
    demand = _write(tmp_path / "w" / "demand" / "CLOSED.json", {"status": "PASS", "contract_sha256": "cw", "arm": "W", "weights": {"path": "x"}})
    body = {"status": "calibration_unmatched", "h_star": None, "h_star_hex": None,
            "closest_if_unmatched": {"h": FG.WIDTH, "h_hex": FG.WIDTH_HEX, "L": 0.18, "relative_error": -0.0498}}
    cal = _write(tmp_path / "cal" / "CALIBRATION.json", body)
    monkeypatch.setitem(FG.SELECTION["source"], "calibration_sha256", cal["sha256"])
    sel = {"kind": FG.SELECTION_KIND, "fwhm": FG.WIDTH, "fwhm_hex": FG.WIDTH_HEX, "calibration": cal,
           "calibration_status": "calibration_unmatched", "h_star": None}
    c = geo_contract(sel, {"demand": demand})
    lock = K.selected_width(c)
    assert lock == {"fwhm": 19.03125, "selection": sel} and type(lock["fwhm"]) is float
    source = K.demand_source(c, "FG")
    assert source == {"kind": "network", "file": {"path": "x"}, "fwhm": FG.WIDTH, "selection": sel}
    assert K.demand_source(c, "F") == K.demand_source(c, "W") == {"kind": "network", "file": {"path": "x"}}
    assert FG.check_source("FG", source, False) == {"fwhm": FG.WIDTH, "fwhm_hex": FG.WIDTH_HEX, "selection": sel}
    for fn in (lambda: K.calibration(c), lambda: K.demand_source(c, "FS")):     # no matched lock exists; arm FS cannot be built
        with pytest.raises(ValueError):
            fn()
    for change in ({"fwhm_hex": (13.0).hex()}, {"h_star": FG.WIDTH}, {"calibration_status": "matched"}, {"kind": "calibrated"}, {"fwhm": 19.0}):
        with pytest.raises(ValueError):
            K.selected_width(geo_contract({**sel, **change}, {"demand": demand}))
    with pytest.raises(ValueError):
        K.selected_width(geo_contract(None, {"demand": demand}))
    with pytest.raises(ValueError):                                             # only the F_geo round has a selected width
        K.selected_width({"contract_sha256": "f", "authorization_basis": K.CAL_BASIS,
                          "followup": {"authorization_basis": K.CAL_BASIS, "arm": "FS", "inherited": {}, "width_selection": sel}})
    for bad in ({**body, "status": "matched", "h_star": FG.WIDTH, "h_star_hex": FG.WIDTH_HEX},                 # a matched file is another experiment
                {**body, "closest_if_unmatched": {**body["closest_if_unmatched"], "h_hex": (13.0).hex()}}):   # another closest width
        other = _write(tmp_path / "cal2" / "CALIBRATION.json", bad)
        monkeypatch.setitem(FG.SELECTION["source"], "calibration_sha256", other["sha256"])
        with pytest.raises(ValueError):
            K.selected_width(geo_contract({**sel, "calibration": other}, {"demand": demand}))
    monkeypatch.setitem(FG.SELECTION["source"], "calibration_sha256", cal["sha256"])
    # the input pipeline of the formal run
    assert K.pipeline_mode(c, "F") == K.pipeline_mode(c, "W") == "serial"       # other arms never read the acceptance
    with pytest.raises(ValueError):
        K.pipeline_mode(c, "FG")                                                # the acceptance is not closed yet
    _write(new / "accept-FG" / "CLOSED.json", {"status": "PASS", "contract_sha256": "new", "arm": "FG", "overlap_enabled": True})
    assert K.pipeline_mode(c, "FG") == "overlap"
    monkeypatch.setenv("R28_PIPELINE", "serial")
    assert K.pipeline_mode(c, "FG") == "serial"
    monkeypatch.setenv("R28_PIPELINE", "overlap")
    with pytest.raises(ValueError):
        K.pipeline_mode(c, "FG")                                                # the variable cannot force the new path
    monkeypatch.delenv("R28_PIPELINE")
    for value in (False, 1, "true", None):
        _write(new / "accept-FG" / "CLOSED.json", {"status": "PASS", "contract_sha256": "new", "arm": "FG", "overlap_enabled": value})
        assert K.pipeline_mode(c, "FG") == "serial"
    _write(new / "pipeline" / "CLOSED.json", {"status": "PASS", "contract_sha256": "new", "arm": "FS", "overlap_enabled": True})
    assert K.pipeline_mode(c, "FG") == "serial"                                 # an arm FS acceptance does not enable arm FG


def test_state_binding_carries_the_width_for_arm_fg_and_separates_f_fs_fg():
    c = {"contract_sha256": "x", "identity_sha256": {"FS": "i", "F": "j", "FG": "g"}, "parent": {"sha256": "p"}}
    mapping = {"fwhm": FG.WIDTH, "fwhm_hex": FG.WIDTH_HEX, "selection": {"kind": FG.SELECTION_KIND}}

    def engine(arm, **extra):
        return types.SimpleNamespace(r28_arm=arm, completed_updates=K.PARENT_UPDATE + 2, s15_activation_checkpoint=True,
                                     config=types.SimpleNamespace(binding=lambda: {"seed": 0}),
                                     r28_ctx=types.SimpleNamespace(binding=lambda: {"kind": "network"}), **extra)

    b = K.binding(c, engine("FG", r28_mapping=mapping), GC.SCOPE)
    assert b["mapping"] == mapping and b["arm"] == "FG" and b["scope"] == "fg-acceptance" and b["identity_sha256"] == "g"
    assert b["global_update"] == K.PARENT_UPDATE + 2 and b["cursor"]["next_window"] == 2
    fs = K.binding(c, engine("FS", r28_mapping={"fwhm": FG.WIDTH, "fwhm_hex": FG.WIDTH_HEX, "calibration": {}}), GC.SCOPE)
    f = K.binding(c, engine("F"), GC.SCOPE)
    assert "mapping" not in f and len({K.sha_bytes(K.canonical(v)) for v in (b, fs, f)}) == 3
    assert (b["arm"], b["identity_sha256"]) != (fs["arm"], fs["identity_sha256"]) and b["mapping"] != fs["mapping"]
    with pytest.raises(ValueError):
        K.binding(c, engine("FG"), "formal")                                    # an arm FG state always names its width
    with pytest.raises(ValueError):
        K.binding(c, engine("F", r28_mapping=mapping), "formal")                # and arm F never does


# ---------------------------------------------------------------- training-time geometry description
AREAS = (100.0, 1024.0, 2000.0, 9216.0, 20000.0)


def make_cache(seed=0, windows=2, images=4):
    g = torch.Generator().manual_seed(seed)
    out = {"demand": [], "boxes": [], "raw_area": []}
    for w in range(windows):
        for i in range(images):
            n = int(torch.randint(1, 7, (1,), generator=g))
            wh = torch.rand(n, 2, generator=g) * 0.08 + 0.004
            centre = wh / 2 + torch.rand(n, 2, generator=g) * (1 - wh)
            out["boxes"].append(torch.cat([centre, wh], 1))
            out["raw_area"].append(torch.tensor([AREAS[(w + i + k) % 5] for k in range(n)], dtype=torch.float32))
            out["demand"].append(W.blur3(torch.rand(*W.GRID, generator=g, dtype=F64) * 3))
    out["demand"] = torch.stack(out["demand"])
    return out


def forward_records(cache, first):
    """The two forward records of one update of four images, as the shared observe stores them for arm FG."""
    recs, area = [], []
    for at in (first, first + 2):
        d = cache["demand"][at:at + 2]
        t = [{"boxes": cache["boxes"][i]} for i in (at, at + 1)]
        plan = FG.plan_from_demand(d)
        pad, n = R._padded_xyxy(t)
        ox, oy = FV.magnification(plan, pad)
        recs.append({"n": n, "demand": d, "xyxy": pad, "ox": ox, "oy": oy})
        area += [cache["raw_area"][at], cache["raw_area"][at + 1]]
    return recs, torch.cat(area)


def test_cal_row_names_the_arm_and_keeps_arm_fs_rows_unchanged():
    cache = make_cache(seed=1)
    recs, area = forward_records(cache, 0)
    fg, fs = LP.cal_row(recs, area, "FG"), LP.cal_row(recs, area)
    assert set(fg) == {"small", "medium", "large"}
    for cls in fg:
        assert set(fg[cls]) == {"n", "FG", "W"} and set(fs[cls]) == {"n", "FS", "W"}
        assert fg[cls]["FG"] == fs[cls]["FS"] and fg[cls]["W"] == fs[cls]["W"] and fg[cls]["n"] == fs[cls]["n"] == sum(fg[cls]["FG"]["bins"])
    # the arm's own cell is the fixed-width geometry of the same GT
    ox = torch.cat([r["ox"][i, :k] for r in recs for i, k in enumerate(r["n"])])
    oy = torch.cat([r["oy"][i, :k] for r in recs for i, k in enumerate(r["n"])])
    small = area.double() <= 1024.0
    assert abs(fg["small"]["FG"]["sum_log_g"] - math.fsum((0.5 * torch.log(ox * oy))[small].tolist())) <= 1e-10
    json.dumps(fg, allow_nan=False)
    for arm in ("F", "W", "K"):
        with pytest.raises(ValueError):
            LP.cal_row(recs, area, arm)


def test_schedule_geometry_folds_the_ledger(tmp_path):
    cache = make_cache(seed=2)
    rows = []
    for w, epoch in ((0, 61), (1, 62)):
        recs, area = forward_records(cache, 4 * w)
        rows.append({"input": {"update": K.PARENT_UPDATE + 1 + w, "epoch": epoch}, "r28": {"cal": LP.cal_row(recs, area, "FG")}})
    with (tmp_path / "UPDATES.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    out = GC.schedule_geometry(tmp_path)
    assert out["updates"] == 2 and list(out["per_epoch"]) == [61, 62]
    for cls in ("small", "medium", "large"):
        n = sum(r["r28"]["cal"][cls]["n"] for r in rows)
        cell = out["overall"][cls]
        assert cell["n"] == n > 0
        for arm in ("FG", "W"):
            total = math.fsum(r["r28"]["cal"][cls][arm]["sum_log_g"] for r in rows)
            per = [math.exp(r["r28"]["cal"][cls][arm]["sum_log_g"] / r["r28"]["cal"][cls]["n"]) for r in rows if r["r28"]["cal"][cls]["n"]]
            assert abs(cell[arm]["L"] - total / n) <= 1e-15 and abs(cell[arm]["sum_log_g"] - total) <= 1e-12
            assert sum(cell[arm]["bins"]) == n and abs(sum(cell[arm]["bin_shares"]) - 1) <= 1e-12
            assert abs(cell[arm]["per_update_geometric_mean_mean"] - sum(per) / len(per)) <= 1e-12
        assert abs(cell["FG_over_W_minus_1"] - math.expm1(cell["FG"]["L"] - cell["W"]["L"])) <= 1e-15
    lo, hi = out["small_FG_over_W_minus_1_range"]
    assert lo <= out["overall"]["small"]["FG_over_W_minus_1"] <= hi and set(out["small_FG_over_W_minus_1_by_epoch"]) == {61, 62}
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


def test_summary_fg_is_record_only_arithmetic():
    values = {"L0": L0V, "W": shifted(L0V, {"APS": 1.8, "AP75_small": 2.5, "APM": -0.7}),
              "K": shifted(L0V, {"APS": 1.0, "AP75_small": 1.0, "APM": -0.7}), "F": shifted(L0V, {"APS": 0.4, "AP75_small": 0.4, "APM": -0.5}),
              "FG": shifted(L0V, {"APS": 1.5, "AP75_small": 1.2, "APM": -0.2})}
    apl = {a: rows_of("APL", [0.30 + 0.05 * i + s for i in range(10)]) for a, s in (("L0", 0), ("W", 0.01), ("K", 0), ("F", 0), ("FG", 0.02))}
    apm = {a: rows_of("APM", [0.20 + 0.03 * i for i in range(10)]) for a in GC.ARMS5}
    every = {f"{a}/{b}": 1.0 + 0.01 * (GC.ARMS5.index(a) - GC.ARMS5.index(b)) for a in GC.ARMS5 for b in GC.ARMS5 if a != b}
    cost = {"ratios": {"model_gpu": dict(every), "end_to_end": dict(every)}, "medians": {"FG": {}}, "stages": {"table": {}}}
    decision = {"label_key": "closed", "K_trigger": False, "exploratory": False, "label": "closed_fixed_config"}
    w_result = {"values": {"L0": L0V, "W": values["W"], "A0": L0V}, "decision": decision}
    res = GC.summary_fg({}, w_result, values, apl, apm, cost, {"F_round_L0_W_F": {}, "K_round_L0_W_K": {}})
    assert res["label"] == "strong_fovea_control_completed" and res["contribution"] == "not_established" and res["record_only"] is True
    assert not any("passed" in k or "gate" in k for k in res)                           # no pass / fail cell anywhere
    p = res["primary_W_minus_FG_pp"]
    assert list(p) == ["W-FG top600 APS", "W-FG top600 AP75_small"]
    assert p["W-FG top600 APS"] == pytest.approx(0.3) and p["W-FG top600 AP75_small"] == pytest.approx(1.3)
    assert list(res["differences_pp"]) == [f"{x}-{y}" for x, y in GC.PAIRS]
    assert res["differences_pp"]["FG-F"]["600"]["APS"] == pytest.approx(1.1) and res["differences_pp"]["FG-L0"]["600"]["APM"] == pytest.approx(-0.2)
    assert res["differences_pp"]["W-FG"]["300"]["AP"] == 0.0 and res["differences_pp"]["W-F"]["600"]["APS"] == pytest.approx(1.4)
    assert set(res["APL6_pp"]) == {"W-FG", "FG-F", "FG-L0"} and res["APL6_pp"]["W-FG"]["difference_pp"] == pytest.approx(-1.0)
    assert res["cost_FG"]["model_gpu"] == {k: every[k] for k in GC.COST_PAIRS} and res["cost_FG"]["all_ratios"]["model_gpu"] == every
    assert res["values"]["FG"] == values["FG"] and res["values"]["F"] == values["F"] and res["values"]["A0"] == L0V and "FS" not in res["values"]
    assert set(res["value_sources"]) == set(GC.ARMS5) and res["W_decision_unchanged"] == decision
    assert res["per_class_APL_top600"][3] == {"category_id": 4, "GT": 779, "L0": pytest.approx(0.45), "K": pytest.approx(0.45),
                                              "F": pytest.approx(0.45), "FG": pytest.approx(0.47), "W": pytest.approx(0.46)}
    with pytest.raises(ValueError):                                                     # an arm missing
        GC.summary_fg({}, w_result, {a: values[a] for a in ("L0", "W", "F", "FG")}, apl, apm, cost, {})
    with pytest.raises(ValueError):                                                     # lists scored against different GT
        GC.summary_fg({}, w_result, values, apl, {**apm, "FG": rows_of("APM", [0.2] * 10, (1,) * 10)}, cost, {})
    assert all(k in GC.STATEMENTS for k in ("registration", "W_decision", "not_F_star", "not_isolated", "record_only", "contribution", "no_trigger"))


# ---------------------------------------------------------------- stored ledger means
def test_ledger_fold_is_the_mean_of_per_update_geometric_means(tmp_path):
    rows = [{"input": {"update": 18241, "epoch": 61}, "r28": {"g_small": 1.2, "g_medium": 1.1, "g_large": None, "n_small": 5}},
            {"input": {"update": 18242, "epoch": 61}, "r28": {"g_small": 1.4, "g_medium": None, "g_large": None, "n_small": 7}},
            {"input": {"update": 18243, "epoch": 62}, "r28": {"g_small": None, "g_medium": 1.3, "g_large": 1.05, "n_small": 0}}]
    with (tmp_path / "UPDATES.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    out = GC.ledger_fold(tmp_path / "UPDATES.jsonl")
    assert (out["rows"], out["first_update"], out["last_update"], out["epochs"], out["epoch_count"], out["small_GT_observations"]) == (3, 18241, 18243, [61, 62], 2, 12)
    s, m, big = (out["fields"][k] for k in GC.LEDGER_FIELDS)
    assert s == {"field": "r28.g_small", "mean": pytest.approx(1.3), "valid_updates": 2, "missing_updates": 1, "min": 1.2, "max": 1.4}
    assert (m["mean"], m["valid_updates"], m["missing_updates"]) == (pytest.approx(1.2), 2, 1)
    assert (big["mean"], big["valid_updates"], big["missing_updates"]) == (1.05, 1, 2)
