"""R29 control checks (CPU, pure): TASK_R29_001 sections 4-7.

Registration, timing order and planned workload; the fixed rules of section 6 and their branches; the audit windows,
the audit ledger entry and the base-ledger comparison; the release difference rule; the folding of the read readings;
the summary arithmetic.  No model forward of a formal ledger; one synthetic read of a few queries feeds the folding.
"""
import copy
import functools
import json
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "r28"))
import r28_common as K  # noqa: E402
import r28_cost as CO  # noqa: E402
import r28_eval as E  # noqa: E402
import r28_geo_control as GC  # noqa: E402
import r28_run as RUN  # noqa: E402
import r29_control as RC  # noqa: E402
from sparse_rtdetr.baseline import training_v2b_deterministic_sampling as DS  # noqa: E402
from sparse_rtdetr.research import r28_fovea as FV  # noqa: E402
from sparse_rtdetr.research import r28_fovea_geo as FG  # noqa: E402
from sparse_rtdetr.research import r29_unzoom as UZ  # noqa: E402

GTS = (23, 5, 3, 779, 115, 87, 12, 5, 32, 7)
BASE = {"600": dict(zip(E.METRICS, (0.3132, 0.2442, 0.2256, 0.3993, 0.4898, 0.5205, 0.3178))),
        "300": dict(zip(E.METRICS, (0.3090, 0.2430, 0.2245, 0.3950, 0.4870, 0.5110, 0.3150)))}


def r29_contract(**fu):
    return {"contract_sha256": "new", "authorization_basis": K.R29_BASIS,
            "followup": {"authorization_basis": K.R29_BASIS, "arm": None, "arms": ["QP", "QC"], **fu}}


def shifted(base, d600=None, d300=None):
    out = copy.deepcopy(base)
    for b, d in (("600", d600 or {}), ("300", d300 or {})):
        for k, v in d.items():
            out[b][k] += v / 100.
    return out


def ratios(p_n=1.04, c_n=1.08, p_c=None):
    p_c = p_n / c_n if p_c is None else p_c
    return {"QP/FG": p_n, "QC/FG": c_n, "QP/QC": p_c, "FG/QP": 1 / p_n, "FG/QC": 1 / c_n, "QC/QP": 1 / p_c}


def rows_of(key, values, gts=GTS):
    return [{"category_id": i + 1, "GT": g, key: x} for i, (g, x) in enumerate(zip(gts, values))]


# ---------------------------------------------------------------- registration
def test_basis_arms_order_and_workload():
    assert K.R29_BASIS == "post_009_user_forwarded_query_unzoom_read" and K.FOLLOWUP_ARMS[K.R29_BASIS] == "R29" and "R29" not in K.ARMS
    assert K.ARMS[5:] == K.R29_ARMS == RC.ARMS == UZ.ARMS == ("QP", "QC") and K.arm_module("QP") is UZ and K.arm_module("QC") is UZ
    assert K.arm_module("FG") is FG and RC.SHORT == {"FG": "N", "QC": "C", "QP": "P"} and RC.ORDER == ("QP", "QC")      # P is trained first
    c = r29_contract()
    assert K.r29(c) and K.followup_arm(c) == "R29"
    assert not K.r29({"authorization_basis": K.GEO_BASIS, "followup": {"authorization_basis": K.GEO_BASIS, "arm": "FG"}})
    assert not K.r29({"authorization_basis": "x"})
    for bad in (r29_contract(arm="FG"), r29_contract(arm="QP")):                     # the R29 block names no single trained arm
        with pytest.raises(ValueError):
            K.r29(bad)
    mixed = {**r29_contract(), "authorization_basis": K.GEO_BASIS}                  # the R29 block under the F_geo basis
    assert not K.r29(mixed)
    with pytest.raises(ValueError):
        K.followup(mixed)
    # task section 5: N, C, P in one process; the pass orders N-C-P, P-C-N, C-P-N; 2 warm-up + 3 x 16 batches per endpoint
    assert K.COST_ENDPOINTS["cost-R29"] == ("FG", "QC", "QP")
    assert K.COST_ROTATIONS["cost-R29"] == (("FG", "QC", "QP"), ("QP", "QC", "FG"), ("QC", "QP", "FG"))
    assert K.COST_PER_ENDPOINT == 2 + 3 * 16 and K.WORKLOAD["cost-R29"]["forwards"] == 150
    assert CO.READ_STAGES == ("demand", "map", "render", "detector", "read", "unwarp", "other") and set(CO.STAGES) < set(CO.READ_STAGES)
    assert all(CO.CANVAS[a] == "H" for a in ("FG", "QP", "QC"))
    # earlier rounds unchanged
    assert K.COST_ENDPOINTS["cost-FG"] == ("L0", "K", "F", "FG", "W") == GC.ARMS5 and K.FOLLOWUP_ARMS[K.GEO_BASIS] == "FG"
    # task section 7: 36,494 updates and 73,422 complete detector forwards; four scorer calls, one per-class evaluation per arm
    w = RC.WORKLOAD
    assert w["accept"] == {"updates": 14, "forwards": 28, "paths": {"N": 2, "QP": {"full": 4, "resumed": 2}, "QC": {"full": 4, "resumed": 2}}}
    assert w["planned_updates"] == 36494 == 14 + 2 * 18240 and w["planned_complete_detector_forwards"] == 73422 == 72960 + 28 + 282 + 2 + 150
    assert w["score_budget_calls"] == 4 and w["per_class_COCOeval_calls"] == 2 and w["ncheck"]["forwards"] == 2 and RC.NCHECK["images"] == 8
    assert RC.ACCEPT == {"N_updates": 2, "steps": 4, "save_at": 2, "ready_limit": 1, "workers": 2}
    assert K.ENDPOINT - K.PARENT_UPDATE == 18240 and K.EVAL_WARMUP + K.EVAL_BATCHES == 141
    assert set(RC.OWN_STAGES).isdisjoint(RC.INHERITED_FG) and "eval-FG" in RC.INHERITED_FG and "accept-FG" in RC.INHERITED_FG
    assert RC.TASK[0] == "TASK_R29_001.md" and RC.DECISION[0] == "DECISION_R29_001.md" and len(RC.TASK[1]) == len(RC.DECISION[1]) == 64
    assert RC.PIN_FG["commit"] == "e1c904f21ad08607871db401be90e2eff47c0465"
    assert RC.PIN_FG["manifest"] == "327f596171f50f77e3dad6ca9d50f3be19a6bb0306c1937733434ca7d25310ca"
    assert RC.PIN_FG["parent"] == K.PARENT_SHA == "a2b6412e3c85de7592829f24aff09e355fdd11cb53798bc498bd04b882e1bf8a"
    assert RC.PIN_FG["demand"] == "c9ffc7505e47adbcc45d70949c3d87371c6f3a7795f1aaaab886d98b15ee9e5a"
    assert RC.PIN_FG["endpoint"] == "f0632fc8ed71e593e5eaab1bf233aafed1e9ec57dcfe44c893721286cdffeb57"
    assert RC.PIN_FG["weights"] == "cd62f314f3713d83fd742ff511600902fd990cf48165d9b0a90c7fec4359e219"


def test_runner_dispatch_and_stage_ownership():
    for action in ("r29-check", "r29-wire", "r29-ncheck", "r29-accept"):
        assert RUN.DISPATCH[action] == "r29_control" and callable(getattr(RC, action.replace("-", "_")))
    assert "r29-check" in RUN.CPU_ACTIONS and not {"r29-wire", "r29-ncheck", "r29-accept"} & set(RUN.CPU_ACTIONS)
    assert callable(RC.prepare) and callable(RC.summarize)
    for name in ("r29-check", "accept-R29"):
        with pytest.raises(ValueError):
            RC._own(r29_contract(), K.ROUND / "elsewhere" / name, name)             # a stage writes its own directory of the round only


# ---------------------------------------------------------------- rules of section 6
def test_rules_are_the_registered_numbers():
    r = RC.RULES
    assert r["N_gate"]["pair"] == ["QP", "FG"] and r["N_gate"]["primary"] == [["600", "APS", 0.30], ["600", "AP75_small", 0.0]]
    assert r["N_gate"]["model_gpu_ratio"] == ["QP/FG", 1.10]
    assert r["M_gate"]["pair"] == ["QP", "QC"] and r["M_gate"]["primary"] == [["600", "APS", 0.20], ["600", "AP75_small", 0.0]]
    assert r["M_gate"]["model_gpu_ratio"] == ["QP/QC", 1.05]
    protection = [["600", "AP", -0.10], ["300", "AP", -0.20], ["300", "APS", -0.20], ["600", "APM", -0.50], ["600", "APL", -0.50]]
    assert r["N_gate"]["protection"] == r["M_gate"]["protection"] == protection
    t = r["tradeoff_reading"]
    assert t["model_gpu_ratio"] == ["QP/QC", 0.90] and t["label"] == "pointwise_accuracy_cost_tradeoff"
    assert t["cells"] == [["600", "AP", -0.10], ["600", "APS", -0.10], ["600", "AP75_small", -0.10]]
    assert r["branches"]["both"] == "single_seed_method_signal" and set(r["branches"]) == {"both", "N_only", "N_failed"}


def test_decision_branches():
    n = BASE
    # both gates reached
    values = {"FG": n, "QC": shifted(n, {"APS": 0.10}), "QP": shifted(n, {"APS": 0.45, "AP75_small": 0.20})}
    d = RC.decide(values, ratios())
    assert d["branch"] == "both" and d["label"] == "single_seed_method_signal" and d["N_gate"]["passed"] and d["M_gate"]["passed"]
    assert d["N_gate"]["pair"] == "P - N" and d["M_gate"]["pair"] == "P - C" and not d["beats_C_but_not_N"] and d["notes"] == []
    assert abs(d["N_gate"]["primary"]["top600_APS"]["difference_pp"] - 0.45) <= 1e-9
    assert abs(d["M_gate"]["primary"]["top600_APS"]["difference_pp"] - 0.35) <= 1e-9 and len(d["N_gate"]["protection"]) == 5
    # N reached, M not: C is as good as P
    values = {"FG": n, "QC": shifted(n, {"APS": 0.35, "AP75_small": 0.20}), "QP": shifted(n, {"APS": 0.45, "AP75_small": 0.20})}
    d = RC.decide(values, ratios())
    assert d["branch"] == "N_only" and d["M_gate"]["failed"] == ["top600_APS"] and "not established" in d["reading"]
    # N not reached although P is above C: never a detection gain
    values = {"FG": n, "QC": shifted(n, {"APS": -0.30, "AP75_small": -0.10}), "QP": shifted(n, {"APS": 0.25, "AP75_small": 0.20})}
    d = RC.decide(values, ratios())
    assert d["branch"] == "N_failed" and d["label"] == "net_increment_target_not_reached" and d["M_gate"]["passed"] and d["beats_C_but_not_N"]
    assert d["N_gate"]["failed"] == ["top600_APS"] and not d["N_gate"]["only_protection_or_cost_failed"] and len(d["notes"]) == 1
    # a small-object gain with AP75-small below N
    values = {"FG": n, "QC": n, "QP": shifted(n, {"APS": 0.60, "AP75_small": -0.05})}
    assert RC.decide(values, ratios())["N_gate"]["failed"] == ["top600_AP75_small"]
    # only a protection cell fails: registered as not reached, the trade-off is shown, no threshold moves
    values = {"FG": n, "QC": n, "QP": shifted(n, {"APS": 0.60, "AP75_small": 0.30, "APM": -0.60})}
    d = RC.decide(values, ratios())
    assert d["branch"] == "N_failed" and d["N_gate"]["failed"] == ["top600_APM"] and d["N_gate"]["only_protection_or_cost_failed"]
    assert any("protection" in s for s in d["notes"])
    values = {"FG": n, "QC": n, "QP": shifted(n, {"APS": 0.60, "AP75_small": 0.30}, {"AP": -0.25})}
    assert RC.decide(values, ratios())["N_gate"]["failed"] == ["top300_AP"]
    # only the cost cell fails
    values = {"FG": n, "QC": n, "QP": shifted(n, {"APS": 0.60, "AP75_small": 0.30})}
    d = RC.decide(values, ratios(p_n=1.11, c_n=1.11))
    assert d["branch"] == "N_failed" and d["N_gate"]["failed"] == ["model_gpu_ratio"] and d["N_gate"]["only_protection_or_cost_failed"]
    d = RC.decide(values, ratios(p_n=1.09, c_n=1.03))
    assert d["N_gate"]["passed"] and d["M_gate"]["failed"] == ["model_gpu_ratio"] and d["branch"] == "N_only"
    # the cost reading: an independent label, never a replacement of N_gate
    values = {"FG": n, "QC": shifted(n, {"APS": 0.05}), "QP": n}
    d = RC.decide(values, ratios(p_n=1.00, c_n=1.12))
    assert d["branch"] == "N_failed" and d["tradeoff_reading"]["label"] == "pointwise_accuracy_cost_tradeoff"
    assert RC.decide(values, ratios(p_n=1.00, c_n=1.10))["tradeoff_reading"]["label"] is None                # ratio 0.909 > 0.90
    values = {"FG": n, "QC": shifted(n, {"APS": 0.15}), "QP": n}
    assert RC.decide(values, ratios(p_n=1.00, c_n=1.12))["tradeoff_reading"]["label"] is None                # APS 0.15 pp below C
    json.dumps(d, allow_nan=False)


def test_summary_arithmetic():
    values = {"FG": BASE, "QC": shifted(BASE, {"APS": 0.10, "APM": -0.20}), "QP": shifted(BASE, {"APS": 0.45, "AP75_small": 0.20, "APL": 0.30})}
    apl = {a: rows_of("APL", [0.30 + 0.05 * i + s for i in range(10)]) for a, s in (("FG", 0.0), ("QC", 0.01), ("QP", 0.02))}
    apm = {a: rows_of("APM", [0.20 + 0.03 * i for i in range(10)]) for a in values}
    cost = {"ratios": {"model_gpu": ratios(), "end_to_end": ratios(1.03, 1.06)}, "medians": {"FG": {}}, "stages": {"table": {}},
            "exclusive_gpu": {"before": {"others": []}, "after": {"others": []}}}
    res = RC.summary_r29(r29_contract(), values, apl, apm, cost)
    assert res["arms"] == {"N": "FG", "C": "QC", "P": "QP"} and set(res["values"]) == {"N", "C", "P"} and res["values"]["P"] == values["QP"]
    assert list(res["differences_pp"]) == ["P-N", "C-N", "P-C"]
    d = res["differences_pp"]
    assert abs(d["P-N"]["600"]["APS"] - 0.45) <= 1e-9 and abs(d["C-N"]["600"]["APM"] + 0.20) <= 1e-9 and abs(d["P-C"]["600"]["APS"] - 0.35) <= 1e-9
    assert abs(d["P-N"]["300"]["AP"]) <= 1e-12 and res["decision"]["branch"] == "both"
    assert set(res["APL6_pp"]) == {"P-N", "C-N", "P-C"} and abs(res["APL6_pp"]["P-N"]["difference_pp"] - 2.0) <= 1e-9
    assert res["cost"]["model_gpu"] == {"P/N": 1.04, "C/N": 1.08, "P/C": 1.04 / 1.08} and res["cost"]["end_to_end"]["C/N"] == 1.06
    assert "N" in res["value_sources"] and "not rescored" in res["value_sources"]["N"]
    json.dumps(res, allow_nan=False)
    with pytest.raises(ValueError):
        RC.summary_r29(r29_contract(), {k: v for k, v in values.items() if k != "QC"}, apl, apm, cost)


# ---------------------------------------------------------------- audit windows, ledger entry, base ledger
def test_audit_windows():
    full = RC.audit_updates(K.PARENT_UPDATE, K.ENDPOINT)
    assert len(full) == 61 and K.ENDPOINT in full and min(full) == K.PARENT_UPDATE + 1
    first = sorted(full - {K.ENDPOINT})
    assert len(first) == 60 and all((u - 1) % K.WINDOWS_PER_EPOCH == 0 for u in first) and first[1] - first[0] == K.WINDOWS_PER_EPOCH == 304
    # a resumed span audits the windows of its own range only, and its last update
    resumed = RC.audit_updates(K.PARENT_UPDATE + 5 * 304, K.ENDPOINT)
    assert resumed == frozenset(u for u in full if u > K.PARENT_UPDATE + 5 * 304) and len(resumed) == 56
    assert RC.audit_updates(K.PARENT_UPDATE, K.PARENT_UPDATE + 4) == frozenset({K.PARENT_UPDATE + 1, K.PARENT_UPDATE + 4})


def test_audit_row_and_base_differences():
    assert RC.audit_row(None) is None
    good = {"identity_batch": False, "images": 8, "layers": [{"location_gradient": {"finite": True, "abs_max": 1.0, "abs_mean": 0.1}}] * 3}
    ident = {"identity_batch": True, "images": 8, "layers": []}
    assert RC.audit_row([good, good]) == {"forwards": [good, good]} and RC.audit_row([good, ident])["forwards"][1] is ident
    no_grad = {"identity_batch": False, "images": 8, "layers": [{}] * 3}
    bad_grad = {"identity_batch": False, "images": 8, "layers": [{"location_gradient": {"finite": False}}] * 3}
    two_layers = {"identity_batch": False, "images": 8, "layers": good["layers"][:2]}
    for sink in ([good], [good, good, good], [good, no_grad], [bad_grad, good], [good, two_layers]):
        with pytest.raises(ValueError):
            RC.audit_row(sink)
    row = {"input": {"update": 18241, "ids": [1, 2]}, "gt_count": 40, "lr": 1e-4, "r28": {"g_small": 1.2, "cal": {"FG": 1}}, "loss": 3.0}
    base = {k: RC._digest(row[k]) for k in RC.BASE_FIELDS}
    assert RC.BASE_FIELDS == ("input", "gt_count", "lr", "r28") and RC.base_differences(row, base) == []
    assert RC.base_differences({**row, "loss": 9.0, "r29": {"x": 1}}, base) == []                            # detector readings are not compared
    assert RC.base_differences({**row, "lr": 2e-4}, base) == ["lr"]
    assert RC.base_differences({**row, "r28": {"g_small": 1.2000001, "cal": {"FG": 1}}, "gt_count": 41}, base) == ["gt_count", "r28"]


# ---------------------------------------------------------------- release difference
def test_source_delta_rule():
    old = {p: "a" for p in RC.CORE_IDENTICAL + RC.ALLOWED_CHANGED + ("src/sparse_rtdetr/research/r28_fovea.py", "configs/x.json")}
    new = {**old, **{p: "b" for p in RC.ALLOWED_CHANGED}, **{p: "n" for p in RC.ALLOWED_ADDED}}
    d = RC.source_delta(old, new)
    assert d["changed"] == sorted(RC.ALLOWED_CHANGED) and d["added"] == sorted(RC.ALLOWED_ADDED) and d["removed"] == []
    assert d["science_files_added"] == ["src/sparse_rtdetr/research/r29_unzoom.py"] and d["existing_science_files_identical"] is True
    assert set(RC.ALLOWED_CHANGED).isdisjoint(RC.CORE_IDENTICAL) and not any(p.startswith(("src/", "vendor/", "configs/")) for p in RC.ALLOWED_CHANGED)
    for core in ("src/sparse_rtdetr/research/r28_fovea_geo.py", "src/sparse_rtdetr/baseline/training_v2b_deterministic_sampling.py",
                 "vendor/rtdetrv2_pytorch/src/zoo/rtdetr/rtdetrv2_decoder.py", "tools/r28/r28_pipeline.py", "tools/r28/r28_geo_control.py"):
        assert core in RC.CORE_IDENTICAL
    for bad in ({**new, "src/sparse_rtdetr/research/r28_fovea.py": "b"},              # an existing science file changed
                {**new, "src/sparse_rtdetr/research/r28_fovea_geo.py": "b"},          # arm FG changed
                {**new, "tools/r28/r28_pipeline.py": "b"},                            # the input pipeline changed
                {**new, "src/sparse_rtdetr/research/other.py": "n"},                  # a second new science file
                {**new, "tools/r28/extra.py": "n"},                                   # a tool outside the registered list
                {k: v for k, v in new.items() if k != "configs/x.json"}):             # a removed file
        with pytest.raises(ValueError):
            RC.source_delta(old, bad)
    assert set(RC.EDITS) == {p for p in RC.ALLOWED_CHANGED if p.startswith("tools/")}
    assert all(set(RC.KEPT[p]).isdisjoint(RC.EDITS[p]["changed"]) for p in RC.KEPT)


# ---------------------------------------------------------------- folding of the read readings
def _audit(mode):
    points = [4, 4, 4]
    stub = types.SimpleNamespace(ms_deformable_attn_core=functools.partial(DS.deformable_attention_core_func_v2, method="default"),
                                 num_points_list=points, num_points_scale=torch.tensor([0.25] * 12), offset_scale=0.5)
    g = torch.Generator().manual_seed(5)

    def nodes(n, seed):
        step = 0.4 + 1.6 * torch.rand(n, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
        c = torch.cat([torch.zeros(1, dtype=torch.float64), step.cumsum(0)])
        return c / c[-1]

    plan = FV.Plan(torch.stack([nodes(50, 1), nodes(50, 2)]), torch.stack([nodes(30, 3), nodes(30, 4)]), torch.zeros(2, dtype=torch.bool), {})
    shapes = [(6, 10), (3, 5), (2, 3)]
    value = torch.randn(2, 81, 2, 3, generator=g)
    ref = torch.cat([0.1 + 0.8 * torch.rand(2, 7, 2, generator=g), 0.05 + 0.4 * torch.rand(2, 7, 2, generator=g)], -1)[:, :, None, :]
    off = (torch.rand(2, 7, 2, 12, 2, generator=g) * 2 - 1).requires_grad_(True)
    aw = torch.softmax(torch.randn(2, 7, 2, 12, generator=g), -1)
    layers = []
    for index in range(3):
        sink = []
        reader = UZ.Reader(mode, plan, 5, sink)                                     # 5 ordinary queries: the first 2 are denoising queries
        out = UZ.read(stub, reader, index, torch.float32, value, shapes, ref, off, aw)
        out.sum().backward()
        layers.append(sink[0])
    return {"identity_batch": False, "images": 2, "layers": layers}


@pytest.mark.parametrize("mode", ["direct", "dense"])
def test_read_readings_fold(mode):
    fwd = _audit(mode)
    row = RC.audit_row([fwd, fwd])
    assert all(layer["dn_queries"] == 2 and layer["dn"]["queries"] == 4 and layer["normal"]["queries"] == 10 for layer in fwd["layers"])
    assert RC._displacement(row) > 0 and RC._displacement(row, "mean") > 0
    s = RC._read_summary([row, row])
    assert set(s) == {"normal", "dn"} and set(s["normal"]) == {"0", "1", "2"}
    one = s["dn"]["1"]
    assert one["forwards"] == 4 and one["queries"] == 16 and one["displacement_feature_px_max"] >= one["displacement_feature_px_mean"] > 0
    assert set(one["border"]) == {"N", "P", "C"} and abs(sum(one["border"]["P"].values()) - 1) <= 1e-9
    assert one["read_difference_P_vs_C"]["samples_relative_l2"] > 0 and "interior" in one["read_difference_P_vs_C"]
    ident = {"identity_batch": True, "images": 2, "layers": []}
    assert RC._displacement({"forwards": [ident, ident]}) == 0.0 and RC._read_summary([{"forwards": [ident, ident]}]) == {"normal": {}, "dn": {}}
    json.dumps(s, allow_nan=False)
