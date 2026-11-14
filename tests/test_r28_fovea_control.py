"""R28-F mapping control (TASK_R28_FOVEA_MAPPING_CONTROL_004): pure CPU checks of the adapter; every number below is
synthetic.

Covered: the registered authorization bases and the arm each trains; the release difference (the science tree may
only gain the arm F module); the stand-in mechanism that runs an unmodified official-style file; the dense tent
reference; the RECORD_ONLY summary arithmetic; reading per-class values back from kept COCOeval tensors; folding the
training ledger; the stage table of the timing; the registered workload.
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "r28"))
import r28_common as K  # noqa: E402
import r28_cost as CO  # noqa: E402
import r28_eval as E  # noqa: E402
import r28_followup as FU  # noqa: E402
import r28_fovea_control as FC  # noqa: E402
from sparse_rtdetr.research import r28_fovea as FV  # noqa: E402
from sparse_rtdetr.research import r28_resample as R  # noqa: E402

N = E.METRICS


def test_bases_arms_and_modules():
    assert K.FOLLOWUP_ARMS == {"post_W_user_approved_K_followup": "K", "post_K_user_forwarded_mapping_control": "F"}
    assert K.ARMS == ("W", "K", "F") and K.arm_module("W") is R and K.arm_module("K") is R and K.arm_module("F") is FV
    with pytest.raises(ValueError):
        K.arm_module("L0")
    f = {"authorization_basis": K.FOVEA_BASIS, "followup": {"authorization_basis": K.FOVEA_BASIS, "arm": "F"}}
    k = {"authorization_basis": K.FOLLOWUP_BASIS, "followup": {"authorization_basis": K.FOLLOWUP_BASIS, "arm": "K"}}
    assert K.followup_arm(f) == "F" and K.followup_arm(k) == "K" and K.followup_arm({"version": "x"}) is None
    for bad in ({"authorization_basis": K.FOVEA_BASIS, "followup": {"authorization_basis": K.FOVEA_BASIS, "arm": "K"}},
                {"authorization_basis": K.FOLLOWUP_BASIS, "followup": {"authorization_basis": K.FOLLOWUP_BASIS, "arm": "F"}},
                {"authorization_basis": "other", "followup": {"authorization_basis": "other", "arm": "F"}},
                {"authorization_basis": K.FOVEA_BASIS, "followup": {"authorization_basis": K.FOLLOWUP_BASIS, "arm": "F"}}):
        with pytest.raises(ValueError):
            K.followup(bad)
    assert K.COST_ENDPOINTS["cost-F"] == ("L0", "W", "F") and all(sorted(r) == ["F", "L0", "W"] for r in K.COST_ROTATIONS["cost-F"])
    assert [r[0] for r in K.COST_ROTATIONS["cost-F"]] == ["L0", "F", "W"]          # every endpoint leads one pass


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return K.ref(path)


def test_stages_inherited_from_two_rounds_and_formal_dir(tmp_path, monkeypatch):
    w, k, new = tmp_path / "w", tmp_path / "k", tmp_path / "new"
    new.mkdir()
    monkeypatch.setattr(K, "ROUND", new)
    demand = _write(w / "demand" / "CLOSED.json", {"status": "PASS", "contract_sha256": "cw", "arm": "W", "weights": {"path": "x"}})
    eval_k = _write(k / "eval-K" / "CLOSED.json", {"status": "PASS", "contract_sha256": "ck", "arm": "K"})
    c = {"contract_sha256": "new", "authorization_basis": K.FOVEA_BASIS,
         "followup": {"authorization_basis": K.FOVEA_BASIS, "arm": "F", "prior_contract_sha256": "cw",
                      "inherited": {"demand": demand, "eval-K": eval_k}, "inherited_contracts": {"eval-K": "ck"}}}
    assert K.stage_closed(c, "demand")["arm"] == "W" and K.stage_closed(c, "eval-K", "K")["arm"] == "K"
    assert K.demand_source(c, "F") == K.demand_source(c, "W") == {"kind": "network", "file": {"path": "x"}}
    wrong = copy.deepcopy(c)
    del wrong["followup"]["inherited_contracts"]
    with pytest.raises(ValueError):                      # a K-round receipt is not accepted under the W contract digest
        K.stage_closed(wrong, "eval-K")
    for arm in ("W", "K"):
        with pytest.raises(ValueError):                  # this round has a formal directory of arm F only
            K.formal_dir(c, arm)
    with pytest.raises(ValueError):
        K.formal_dir(c, "F")                             # none closed yet
    _write(new / "formal-F" / "CLOSED.json", {"status": "PASS", "contract_sha256": "new", "arm": "F"})
    assert K.formal_dir(c, "F").name == "formal-F"


def test_source_delta_lets_the_science_tree_only_gain_the_arm_f_module():
    old = {"src/sparse_rtdetr/research/r28_warp.py": "1", "src/sparse_rtdetr/research/r28_resample.py": "2",
           "tools/r28/r28_common.py": "3", "tools/r28/r28_demand.py": "4", "vendor/x.py": "5"}
    new = {**old, "tools/r28/r28_common.py": "3b", "src/sparse_rtdetr/research/r28_fovea.py": "6", "tools/r28/r28_fovea_control.py": "7"}
    d = FC.source_delta(old, new, label="t")
    assert d["changed"] == ["tools/r28/r28_common.py"] and d["science_files_added"] == ["src/sparse_rtdetr/research/r28_fovea.py"]
    assert d["existing_science_files_identical"] is True and d["removed"] == [] and d["unchanged_files"] == 4
    for bad in ({**new, "src/sparse_rtdetr/research/r28_warp.py": "x"},          # arm W operator changed
                {**new, "src/sparse_rtdetr/research/r28_resample.py": "x"},      # arm W wrapper changed
                {**new, "src/sparse_rtdetr/research/other.py": "x"},             # another science file added
                {**new, "vendor/x.py": "x"},                                     # detector changed
                {**new, "tools/r28/r28_demand.py": "x"},                         # the demand tool changed
                {k: v for k, v in new.items() if k != "vendor/x.py"}):           # a file removed
        with pytest.raises(ValueError):
            FC.source_delta(old, bad, label="t")
    assert not any(p.startswith(("src/", "vendor/", "configs/")) for p in FC.ALLOWED_CHANGED)


OFFICIAL_STYLE = '''
import numpy as np
import torch
from mmcv.utils import Registry
from mmdet.models.builder import build
from ..utils.vis import vis_batched_imgs

GRID_GENERATORS = Registry('grid_generator')


def window(size):
    return np.arange(0, size, 1, dtype=np.float)


@GRID_GENERATORS.register_module()
class Thing(object):
    def __init__(self):
        self.filter = torch.FloatTensor(window(5)).cuda()
'''


def test_official_module_runs_an_unmodified_file_and_restores_the_process(tmp_path):
    path = tmp_path / "grid_generator.py"
    path.write_text(OFFICIAL_STYLE)
    before = {n: sys.modules.get(n) for n in ("mmcv", "mmcv.utils", "mmdet", "mmdet.models.builder")}
    cuda, had = torch.Tensor.cuda, hasattr(np, "float")
    with FC.official_module(path) as m:
        t = m.Thing()
        assert t.filter.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0] and t.filter.dtype == torch.float32 and t.filter.device.type == "cpu"
    assert torch.Tensor.cuda is cuda and hasattr(np, "float") == had
    assert {n: sys.modules.get(n) for n in before} == before and "r28_official_fovea" not in sys.modules


def test_reference_nodes_and_synthetic_demands():
    d = FC.synthetic_demands()
    assert list(d) == ["constant", "centre", "edge", "bimodal"] and all(tuple(v.shape) == (12, 21) and float(v.min()) >= 0 for v in d.values())
    assert FV.plan_from_demand(torch.stack(list(d.values()))).identity.tolist() == [True, False, False, False]
    s = FV.node_saliency((d["bimodal"] + 0.2).sum(0)[None], 51)
    raw, _ = FV.attraction_nodes(s)
    ref = FC.reference_nodes(s[0].tolist(), FV.FWHM, FV.PAD)
    assert max(abs(a - b) for a, b in zip(ref, raw[0].tolist())) <= 1e-10


def test_reference_axis_is_the_production_tent():
    plan = FV.plan_from_demand(FC.synthetic_demands()["edge"][None])
    x = torch.rand(1, 3, 72, 126, generator=torch.Generator().manual_seed(0))
    out, info = FV.resample(x, plan, out_hw=(48, 84))
    ax, sx, lo, hi = FC.reference_axis(plan.gx[0].numpy(), 84, 126)
    ay, sy, _, _ = FC.reference_axis(plan.gy[0].numpy(), 48, 72)
    ref = np.einsum("ih,chw,jw->cij", ay, x[0].double().numpy(), ax)
    assert np.abs(out[0].double().numpy() - ref).max() <= 1e-5 and info["taps_x"] >= sx and info["taps_y"] >= sy
    assert np.abs(ax.sum(1) - 1).max() <= 1e-12 and 0 < lo < 1.5 < hi
    v = np.linspace(-0.2, 1.2, 501)
    g = plan.gx[0].numpy()
    assert np.abs(FC._np_warp(g, FC._np_unwarp(g, v)) - v).max() <= 1e-12
    assert np.abs(FC._np_unwarp(g, v) - FV.unwarp(plan.gx, torch.from_numpy(v)[None])[0].numpy()).max() <= 1e-12


def v(six, three):
    return {"600": dict(zip(N, six)), "300": dict(zip(N, three))}


L0 = v((0.3000, 0.2200, 0.2000, 0.3950, 0.4950, 0.5050, 0.3010), (0.2960, 0.2190, 0.1990, 0.3900, 0.4920, 0.4950, 0.2980))
GTS = (23, 5, 3, 779, 115, 87, 12, 5, 32, 7)


def shifted(base, d600):
    out = copy.deepcopy(base)
    for k, d in d600.items():
        out["600"][k] += d / 100.
    return out


def rows(key, values, gts=GTS):
    return [{"category_id": i + 1, "GT": g, key: x} for i, (g, x) in enumerate(zip(gts, values))]


def test_summary_f_is_record_only_arithmetic():
    values = {"L0": L0, "W": shifted(L0, {"APS": 1.8, "AP75_small": 2.5, "APM": -0.7}),
              "K": shifted(L0, {"APS": 1.0, "AP75_small": 1.0, "APM": -0.7}), "F": shifted(L0, {"APS": 1.5, "AP75_small": 1.2, "APM": -0.2})}
    apl = {a: rows("APL", [0.30 + 0.05 * i + s for i in range(10)]) for a, s in (("L0", 0), ("W", 0.01), ("K", 0), ("F", 0.02))}
    apm = {a: rows("APM", [0.20 + 0.03 * i for i in range(10)]) for a in FC.ARMS4}
    pairs = ("F/L0", "W/L0", "W/F", "F/W")
    cost = {"ratios": {"model_gpu": {**dict(zip(pairs, (1.12, 1.10, 0.98, 1.02))), "L0/W": 0.9}, "end_to_end": dict.fromkeys(pairs, 1.0)},
            "medians": {"F": {}}, "stages": {"table": {}}}
    decision = {"label_key": "closed", "K_trigger": False, "exploratory": False, "label": "closed_fixed_config"}
    res = FC.summary_f({}, {"values": {"L0": L0, "W": values["W"], "A0": L0}, "decision": decision}, values, apl, apm, cost, {"model_gpu": {}})
    assert res["label"] == "mapping_control_completed" and res["contribution"] == "not_established" and res["record_only"] is True
    assert not any("passed" in k or "gate" in k for k in res)                           # no pass / fail cell anywhere
    p = res["primary_W_minus_F_pp"]
    assert list(p) == ["W-F top600 APS", "W-F top600 AP75_small"]
    assert p["W-F top600 APS"] == pytest.approx(0.3) and p["W-F top600 AP75_small"] == pytest.approx(1.3)
    assert list(res["differences_pp"]) == ["W-F", "F-L0", "F-K", "W-L0", "K-L0", "W-K"]
    assert res["differences_pp"]["F-L0"]["600"]["APM"] == pytest.approx(-0.2) and res["differences_pp"]["F-K"]["600"]["APS"] == pytest.approx(0.5)
    assert res["differences_pp"]["W-F"]["300"]["AP"] == 0.0
    assert res["APL6_pp"]["W-F"]["difference_pp"] == pytest.approx(-1.0) and res["APL6_pp"]["F-L0"]["difference_pp"] == pytest.approx(2.0)
    assert res["cost_F"]["model_gpu"] == dict(zip(pairs, (1.12, 1.10, 0.98, 1.02))) and res["W_decision_unchanged"] == decision
    assert res["values"]["F"] == values["F"] and res["values"]["A0"] == L0 and set(res["value_sources"]) == set(FC.ARMS4)
    assert res["per_class_APL_top600"][3] == {"category_id": 4, "GT": 779, "L0": pytest.approx(0.45), "W": pytest.approx(0.46),
                                              "K": pytest.approx(0.45), "F": pytest.approx(0.47)}
    with pytest.raises(ValueError):                                                     # lists scored against different GT
        FC.summary_f({}, {"values": {}, "decision": decision}, values, apl, {**apm, "F": rows("APM", [0.2] * 10, (1,) * 10)}, cost, {})
    with pytest.raises(ValueError):                                                     # an arm missing
        FC.summary_f({}, {"values": {}, "decision": decision}, {a: values[a] for a in ("L0", "W", "F")}, apl, apm, cost, {})


class FakeParams:
    iouThrs = np.linspace(0.5, 0.95, 10)
    recThrs = np.linspace(0, 1, 101)
    catIds = [1, 2, 3]
    areaRng = [[0, 1e10], [0, 1024], [1024, 9216], [9216, 1e10]]
    areaRngLbl = ["all", "small", "medium", "large"]
    maxDets = [1, 10, 100]


class FakeEval:
    params = FakeParams()

    def __init__(self):
        g = np.random.default_rng(0)
        p = g.random((10, 101, 3, 4, 3))
        p[:, :, 2, 3, :] = -1                               # class 3 has no large GT
        self.eval = {"precision": p, "recall": g.random((10, 3, 4, 3)), "scores": g.random((10, 101, 3, 4, 3))}
        self.stats = np.arange(12, dtype=np.float64)


def test_kept_tensors_give_the_per_class_values_back(tmp_path):
    e = FakeEval()
    dataset = {"annotations": [{"category_id": 1, "area": 100.0}, {"category_id": 1, "area": 2000.0}, {"category_id": 2, "area": 1024.0},
                               {"category_id": 3, "area": 50.0, "iscrowd": 1}, {"category_id": 3, "area": 60.0}]}
    path = FU.keep_tensors(e, dataset, tmp_path / "t.npz")
    with pytest.raises(ValueError):
        FU.keep_tensors(e, dataset, path)                   # never overwritten
    t = FC.tensor_tables(path)
    p = e.eval["precision"]
    assert [r["category_id"] for r in t["classes"]] == [1, 2, 3] and t["maxDets"] == 100 and t["stats"] == list(range(12))
    assert t["classes"][0]["GT"] == {"all": 2, "small": 1, "medium": 1, "large": 0}
    assert t["classes"][1]["GT"] == {"all": 1, "small": 1, "medium": 1, "large": 0}     # area 1024 lies in both ranges
    assert t["classes"][2]["GT"] == {"all": 1, "small": 1, "medium": 0, "large": 0}     # the crowd row is not counted
    assert t["classes"][0]["AP"] == pytest.approx(p[:, :, 0, 0, 2].mean()) and t["classes"][1]["APM"] == pytest.approx(p[:, :, 1, 2, 2].mean())
    assert t["classes"][0]["AP50"] == pytest.approx(p[0, :, 0, 0, 2].mean()) and t["classes"][0]["AP75"] == pytest.approx(p[5, :, 0, 0, 2].mean())
    assert t["classes"][1]["AP75_small"] == pytest.approx(p[5, :, 1, 1, 2].mean()) and t["classes"][2]["APL"] is None
    assert t["macro"]["APL"] == pytest.approx(np.mean([p[:, :, k, 3, 2].mean() for k in (0, 1)]))
    assert t["macro"]["APS"] == pytest.approx(np.mean([p[:, :, k, 1, 2].mean() for k in range(3)]))


def test_kept_tensors_of_a_real_cocoeval_reproduce_its_stats(tmp_path):
    """pycocotools on a small synthetic data set (not a development image): the macro values read back from the kept
    tensors are the scorer's own summary numbers."""
    import contextlib
    import io
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    g = np.random.default_rng(1)
    images = [{"id": i + 1, "width": 640, "height": 480} for i in range(6)]
    gt, det = [], []
    for im in images:
        for k in range(12):
            side = float(g.choice([14.0, 50.0, 130.0]))
            x, y = float(g.uniform(0, 640 - side)), float(g.uniform(0, 480 - side))
            cat = int(g.integers(1, 4))
            gt.append({"id": len(gt) + 1, "image_id": im["id"], "category_id": cat, "bbox": [x, y, side, side], "area": side * side, "iscrowd": 0})
            if g.random() < 0.8:
                j = float(g.uniform(-0.12, 0.12)) * side
                det.append({"image_id": im["id"], "category_id": cat, "bbox": [x + j, y - j, side, side], "score": float(g.random())})
            if g.random() < 0.3:
                det.append({"image_id": im["id"], "category_id": int(g.integers(1, 4)), "bbox": [x + side, y, side, side], "score": float(g.random())})
    dataset = {"info": {}, "images": images, "categories": [{"id": i, "name": str(i)} for i in (1, 2, 3)], "annotations": gt}
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(); coco.dataset = dataset; coco.createIndex()
        e = COCOeval(coco, coco.loadRes(det), "bbox")
        e.params.maxDets = [1, 10, 100]
        e.evaluate(); e.accumulate(); e.summarize()
    t = FC.tensor_tables(FU.keep_tensors(e, dataset, tmp_path / "real.npz"))
    for key, stat in (("AP", 0), ("AP50", 1), ("AP75", 2), ("APS", 3), ("APM", 4), ("APL", 5)):
        assert t["macro"][key] == pytest.approx(float(e.stats[stat]), abs=1e-12), key
    assert 0 < t["macro"]["AP"] < 1 and t["stats"] == [float(v) for v in e.stats] and t["maxDets"] == 100
    assert sum(r["GT"]["all"] for r in t["classes"]) == len(gt) and all(r["GT"]["small"] + r["GT"]["medium"] + r["GT"]["large"] == r["GT"]["all"]
                                                                        for r in t["classes"])


def test_formal_observations_fold_the_ledger(tmp_path):
    def row(lo, hi, taps, foot, g_small, lt1, clamped=0):
        return {"update_seconds": 2.0, "r28": {"g_small": g_small, "g_medium": 1.0, "g_large": None, "small_g_lt_1": lt1, "identity_images": 0,
                                               "fovea": {"node_slope_min": lo, "node_slope_max": hi, "endpoint_error_max": 1e-16,
                                                         "clamped_nodes": clamped, "clamp_max": 1e-17, "clamped_interior_nodes": 0,
                                                         "taps_max": taps, "footprint_max_px": foot}}}
    with (tmp_path / "UPDATES.jsonl").open("w") as f:
        for r in (row(0.8, 1.25, 3, 1.9, 1.10, 0.2), row(0.5, 2.0, 5, 3.0, 1.20, 0.4, clamped=2), row(0.9, 1.1, 3, 1.6, None, None)):
            f.write(json.dumps(r) + "\n")
    o = FC.formal_observations(tmp_path)
    assert o["updates"] == 3 and o["node_slope_min"] == 0.5 and o["node_slope_max"] == 2.0 and o["taps_max"] == 5 and o["footprint_max_px"] == 3.0
    assert o["magnification_min"] == 0.5 and o["magnification_max"] == 2.0 and o["clamped_nodes"] == 2 and o["taps_histogram"] == {"3": 2, "5": 1}
    assert o["clamped_interior_nodes"] == 0
    assert o["seconds_per_update_mean"] == 2.0 and o["per_update_means"]["g_small"] == pytest.approx(1.15)
    assert o["per_update_means"]["g_large"] is None and o["per_update_means"]["small_g_lt_1"] == pytest.approx(0.3)
    json.dumps(o, allow_nan=False)


def test_stage_table_medians_and_shares():
    def one(endpoint, total, stages):
        return {"endpoint": endpoint, "model_gpu_ms_total": total, "stage_ms_total": stages}
    z = dict.fromkeys(CO.STAGES, 0.0)
    passes = [one("F", 100.0, {**z, "demand": 10.0, "detector": 80.0, "other": 10.0}), one("F", 110.0, {**z, "demand": 12.0, "detector": 90.0, "other": 8.0}),
              one("F", 90.0, {**z, "demand": 8.0, "detector": 75.0, "other": 7.0}), one("A0", 50.0, None)]
    t = CO.stage_table(passes, ("A0", "F"), 64)
    assert list(t) == ["F"] and t["F"]["stage_ms_median_pass"]["demand"] == 10.0 and t["F"]["stage_ms_median_pass"]["detector"] == 80.0
    assert t["F"]["share_of_model_gpu"]["detector"] == pytest.approx(0.8) and t["F"]["per_image_ms"]["demand"] == pytest.approx(10 / 64)
    assert CO.CANVAS["F"] == "H" and CO.STAGES == ("demand", "map", "render", "detector", "unwarp", "other")


def test_registered_identities_and_workload():
    assert FC.TASK[1] == "0cd799d9f149c0d5108f5f1c5d8a0b881f976aaf0d28cb9b452c5750c26a7d96"
    assert FC.PIN_K["commit"] == "1b4f2a24f7d5a6fbbc953d6ad7e742e3fd2f63ca" and FC.PIN_K["endpoint"].startswith("5abb8d02")
    assert FU.PIN["W_demand"].startswith("c9ffc750") and FU.PIN["W_endpoint"].startswith("a88f2a15") and FU.PIN["parent"] == K.PARENT_SHA
    assert FC.WORKLOAD["planned_complete_detector_forwards"] == 36777 and FC.WORKLOAD["planned_updates"] == 18243
    assert FC.WORKLOAD["wire"]["forwards"] == 7 and FC.WORKLOAD["fovea-check"]["forwards"] == 6
    assert set(FC.OWN_STAGES).isdisjoint(FU.INHERITED) and set(FC.OWN_STAGES).isdisjoint(FC.INHERITED_K) and set(FU.INHERITED).isdisjoint(FC.INHERITED_K)
    assert FC.PRIMARY == (("600", "APS"), ("600", "AP75_small")) and FC.LABEL == "mapping_control_completed"
    assert set(FV.OFFICIAL["files"]) == {v[1] for v in FC.OFFICIAL_FILES.values()}
    assert FC.TOL["fp64_same_formula"] == 1e-10 and FC.TOL["official_fp32_nodes"] == 1e-5 and FC.TOL["round_trip"] == 1e-9
