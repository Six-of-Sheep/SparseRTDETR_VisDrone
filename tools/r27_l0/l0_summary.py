"""R27-002 L0 control: summary arithmetic as pure functions (no file system, no torch, no model forward, no scoring).

Inputs are plain dicts (contract, eval-L0 CLOSED, cost4 CLOSED, per-stage counts); l0_eval.summarize does the file reads.
P/C/S15/A0 readings come only from the contract's `reference` block (R27-001 certified readings, reused, never rescored).
"""
import math

import l0_common as K

METRICS = ("AP", "APS", "AP75_small", "APM", "APL", "AP50", "AP75")
BUDGETS = ("600", "300")
NEW = (("P-L0", "P", "L0"), ("C-L0", "C", "L0"), ("L0-A0", "L0", "A0"), ("L0-S15", "L0", "S15"))
OLD = (("P-A0", "P", "A0"), ("P-S15", "P", "S15"), ("C-A0", "C", "A0"), ("P-C", "P", "C"))
SOURCE_OLD = "R27-001 certified readings (reused)"
SOURCE_L0 = "R27-002 L0 e120 EMA view A evaluation (new)"
SOURCE_NEW = "new comparison: L0 from the R27-002 evaluation (new), the other arm from R27-001 certified readings (reused)"
CLASS_ARMS = ("A0", "S15", "C", "P", "L0")
CLASS_PAIRS = (("P-L0", "P", "L0"), ("L0-A0", "L0", "A0"), ("C-L0", "C", "L0"))
TOLERANCE_PP = 1e-9
STAGES = ("cpu_fixture", "cuda_zero_step", "cuda_accept", "formal", "evaluation", "cost4")
GPU_STAGES = STAGES[1:]
COUNT_KEYS = (("updates", "updates", "actual_updates"), ("forwards", "forwards", "forwards"),
              ("score_budget_calls", "score_budget_calls", "score_budget_calls"),
              ("per_class_COCOeval_calls", "per_class_APL_COCOeval_calls", "per_class_COCOeval_calls"))  # name, plan key, CLOSED key
STATEMENTS = {
    "old_E_I_G": "R27-001 的 E/I/G 判定永久保持不变；本汇总不重算、不改写旧判定，也不给旧臂重评分。",
    "D_origin": "D 是看到 R27-001 结果之后选定的后续对照，不属于 R27-001 的原始预登记。",
    "single_seed": "单 seed（seed0）结果，不提供跨 seed 稳定性证明。",
    "development_reuse": "同一 548 图开发集被重复使用，不提供独立的泛化或稳定性证明。",
    "D_failed_reading": "D 未过时保留实际差值，不声称零效应；不因旧 G 全过而继续扩 seed、改路由或延程。",
    "D_passed_reading": "D 全过时不自动启动跨 seed 复验；参数量、推理与训练成本代价保留。",
    "L0_minus_A0": "L0−A0 描述共同 H→L 输入与训练流程的变化，不单独分离渲染、目标过滤和训练适应三个原因。",
    "P_minus_L0": "P−L0 检验完整支路训练处理（含新增容量与共适应），不独立证明分数路由、局部信息来源或原创性；原 I 未过不被 D 改写。",
    "cost": "四端点测速是新环境下的补测，不覆盖 R27-001 的终态成本与 E 判据；未知的等成本原生精度不从本次测速插值得到。",
}


def check_values(name, v):
    K.require(set(v) == set(BUDGETS) and all(set(v[b]) == set(METRICS) for b in BUDGETS), f"{name}: budgets/metrics differ")
    K.require(all(type(v[b][m]) is float and math.isfinite(v[b][m]) and 0. <= v[b][m] <= 1. for b in BUDGETS for m in METRICS),
              f"{name}: a metric is not a finite 0..1 float")


def difference_pp(x, y):
    """Full-precision differences in percentage points, 7 metrics x 2 budgets (same arithmetic as K.cells)."""
    return {b: {m: 100. * (x[b][m] - y[b][m]) for m in METRICS} for b in BUDGETS}


def differences(reference, l0):
    K.require(set(reference) == {"P", "C", "S15", "A0"}, "reference arms")
    arms = {**reference, "L0": l0}
    for a, v in arms.items():
        check_values(a, v)
    origin = {**{a: SOURCE_OLD for a in reference}, "L0": SOURCE_L0}
    out = {}
    for group, new in ((NEW, True), (OLD, False)):
        for name, x, y in group:
            out[name] = {"new": new, "source": SOURCE_NEW if new else SOURCE_OLD, "minuend": {"arm": x, "source": origin[x]},
                         "subtrahend": {"arm": y, "source": origin[y]}, "difference_pp": difference_pp(arms[x], arms[y])}
    return out


def decision_d(p, l0, rule, labels):
    """D = every cell of P - L0 reaches its minimum (a difference exactly equal to the minimum passes)."""
    cells = K.cells(p, l0, rule)
    K.require(len(cells) == len(rule) == 7, "D rule must have seven distinct cells")
    passed = all(v["passed"] for v in cells.values())
    return {"comparison": "P - L0", "cells": cells, "failed_cells": [k for k, v in cells.items() if not v["passed"]],
            "passed": passed, "label_key": "D_passed" if passed else "D_failed",
            "label": labels["D_passed" if passed else "D_failed"]}


def self_check(reference, rule, recorded):
    """P - A0 recomputed from the contract reference must equal the differences recorded at prepare, cell by cell."""
    got = {k: v["difference_pp"] for k, v in K.cells(reference["P"], reference["A0"], rule).items()}
    K.require(got == recorded, ("P-A0 self-check differs from the contract", got, recorded))
    return {"comparison": "P - A0 (reused readings)", "difference_pp": got, "equal_to_contract_self_check": True}


def per_class(reference_apl, l0_apl, values):
    """Per-class APL (top600) table, per-class differences and macro contributions (difference / classes)."""
    arms = {**{a: reference_apl[a] for a in ("A0", "S15", "C", "P")}, "L0": l0_apl}
    cats = [r["category_id"] for r in arms["A0"]]
    gts = [r["GT"] for r in arms["A0"]]
    K.require(cats == list(range(1, 11)), "per-class categories")
    for a in CLASS_ARMS:
        K.require([r["category_id"] for r in arms[a]] == cats, f"per-class categories differ: {a}")
        K.require([r["GT"] for r in arms[a]] == gts, f"per-class large GT counts differ from the reference arms: {a}")
        K.require(all(type(r["APL"]) is float and 0. <= r["APL"] <= 1. for r in arms[a]), f"per-class APL unavailable: {a}")
    table = [{"category_id": cat, "large_GT": gts[i], **{a: arms[a][i]["APL"] for a in CLASS_ARMS}} for i, cat in enumerate(cats)]
    pairs = {}
    for name, x, y in CLASS_PAIRS:
        rows = []
        for i, cat in enumerate(cats):
            d = 100. * (arms[x][i]["APL"] - arms[y][i]["APL"])
            rows.append({"category_id": cat, "large_GT": gts[i], "difference_pp": d, "macro_contribution_pp": d / len(cats)})
        total = math.fsum(r["macro_contribution_pp"] for r in rows)
        target = 100. * (values[x]["600"]["APL"] - values[y]["600"]["APL"])
        K.require(abs(total - target) <= TOLERANCE_PP,
                  f"per-class contributions of {name} sum to {total!r} pp, APL600 difference is {target!r} pp")
        pairs[name] = {"classes": rows, "contribution_sum_pp": total, "APL600_difference_pp": target,
                       "residual_pp": total - target, "tolerance_pp": TOLERANCE_PP, "sum_equals_APL600_difference": True}
    return {"columns": list(CLASS_ARMS), "table": table, "differences": pairs,
            "sources": {**{a: SOURCE_OLD for a in ("A0", "S15", "C", "P")}, "L0": SOURCE_L0},
            "large_GT_equal_across_arms": True}


def workload(plan, actual, formal_chain=None):
    """Planned (contract workload) against actual (CLOSED counts per stage); differences are reported, never hidden.
    A resumed formal run closes with the counts of its own process only: it is then reported as unequal, with the chain."""
    K.require(set(actual) == set(STAGES), ("workload stages", sorted(actual)))
    stages = {}
    for s in STAGES:
        planned = {name: plan[s].get(pk, 0) for name, pk, _ in COUNT_KEYS}
        got = {name: actual[s].get(ck, 0) for name, _, ck in COUNT_KEYS}
        got["forwards"] += actual[s].get("reference_forwards", 0)  # zero stage: native reference forwards are counted apart
        stages[s] = {"planned": planned, "actual": got, "equal": planned == got,
                     "differs": [k for k in planned if planned[k] != got[k]], "CLOSED_counts": dict(actual[s])}
    names = [name for name, _, _ in COUNT_KEYS]
    total = {side: {k: sum(stages[s][side][k] for s in GPU_STAGES) for k in names} for side in ("planned", "actual")}
    K.require(all(total["planned"][k] == plan["normal_gpu_total"][k] for k in ("updates", "forwards")),
              "planned stage rows do not add up to the contract's normal_gpu_total")
    total.update(equal=total["planned"] == total["actual"], stages=list(GPU_STAGES),
                 differs=[k for k in names if total["planned"][k] != total["actual"][k]])
    return {"stages": stages, "normal_gpu_total": total, "formal_chain": formal_chain, "all_equal": all(v["equal"] for v in stages.values()) and total["equal"],
            "unequal_stages": [s for s in STAGES if not stages[s]["equal"]],
            "note": "counts are complete model forwards / optimizer updates / scorer calls as recorded in each stage's "
                    "CLOSED.json; activation-checkpoint recomputation and failed or dev runs are not part of these rows"}


def cost_fields(cost_closed):
    """The single place that names what is read from cost4/CLOSED.json (to be aligned with the timing module)."""
    return {"L0_weights_sha256": cost_closed["L0_weights_sha256"], "medians": cost_closed["medians"],
            "ratios": cost_closed["ratios"]}


def cost(cost_closed, eval_weights, endpoints):
    f = cost_fields(cost_closed)
    K.require(f["L0_weights_sha256"] == eval_weights, "cost4 timed another L0 model than the evaluated e120 EMA")
    K.require(set(f["medians"]) == set(endpoints), ("cost4 endpoints", sorted(f["medians"])))
    return {"source": "cost4/CLOSED.json of this round (one process, four endpoints), copied unchanged",
            "medians": f["medians"], "ratios": f["ratios"], "L0_weights_sha256": f["L0_weights_sha256"],
            "L0_weights_equal_to_evaluation": True}


def build(c, ev, cost_closed, stage_counts, r27, formal_chain=None):
    """RESULT.json body.  r27 = {"file": ref, "label": ..., "E_passed": ..., "I_passed": ..., "G_passed": ...} (quoted only)."""
    reference, l0 = c["reference"]["values"], ev["values"]
    K.require(ev["counts"].get("score_budget_calls") == 2 and ev["counts"].get("per_class_COCOeval_calls") == 1,
              "eval-L0 scorer call counts")
    d = decision_d(reference["P"], l0, [tuple(x) for x in c["gates"]["D_P_minus_L0"]], c["labels"])
    return {"status": "PASS", "contract_sha256": c["contract_sha256"], "arm": K.ARM,
            "endpoint": {"file": ev["endpoint"], "weights_sha256": ev["weights_sha256"], "kind": "EMA", "epoch": 120,
                         "view": "A", "images": 548, "budgets": [600, 300]},
            "values": {**reference, "L0": l0},
            "value_sources": {**{a: SOURCE_OLD for a in reference}, "L0": SOURCE_L0},
            "D_P_minus_L0": d, "label": d["label"],
            "differences_pp": differences(reference, l0),
            "P_minus_A0_self_check": self_check(reference, [tuple(x) for x in c["gates"]["D_P_minus_L0"]],
                                                c["reference"]["P_minus_A0_self_check"]["difference_pp"]),
            "per_class_APL_top600": per_class(c["reference"]["per_class_APL_top600"], ev["per_class_APL_top600"],
                                              {**reference, "L0": l0}),
            "cost4": cost(cost_closed, ev["weights_sha256"], K.COST_ENDPOINTS),
            "workload": workload(c["workload"], stage_counts, formal_chain),
            "R27_001": {"result_file": r27["file"], "label": r27["label"], "E_passed": r27["E_passed"],
                        "I_passed": r27["I_passed"], "G_passed": r27["G_passed"], "status": "unchanged (quoted, not recomputed)"},
            "statements": STATEMENTS}
