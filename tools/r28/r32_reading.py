"""R32-001: the reading of TASK_R32_W_SEEDS_001 section 5 as pure arithmetic (no model, no file, no torch).

All differences are of the same seed, in percentage points, from e120 EMA values of view A on the 548 development
images.  Every rule is decided on the unrounded value and a value on the bound satisfies the rule.
  reading A   W against L0: APS600 >= +1.0, AP75-small600 >= +1.0 and AP600 >= 0 on one seed; three-seed label.
  reading B   W against F_geo, the same bound in both directions: W_leads / F_geo_leads / neither per seed.
  reading C   W - L0 APM600 <= -0.50 per seed; three-seed label.
  APL         sign and value per seed; split into the five classes with at most 12 large GT and the other five.
  seven cells the R28-001 table (W - L0) listed for every seed, as a record only.
  aggregate   mean, sample standard deviation (n - 1) and the t half-width of the three-seed mean: a description.
  geometry    training-time magnification of the small GT from the ledger rows (key "r28").
"""
import math

METRICS = ("AP", "APS", "AP75_small", "APM", "APL", "AP50", "AP75")
BUDGET_KEYS = ("600", "300")
SEED_KEYS = ("0", "1", "2")
BOUND = (("600", "APS", 1.0), ("600", "AP75_small", 1.0), ("600", "AP", 0.0))      # pp, all three at once
SEVEN = (("600", "APS", 0.50), ("600", "AP75_small", 0.30), ("600", "AP", -0.10), ("300", "AP", -0.20),
         ("300", "APS", -0.20), ("600", "APM", -0.50), ("600", "APL", -0.50))      # R28-001, W - L0, pp
MEDIUM_COST = -0.50                    # pp; the APM600 bound of the R28-001 seven cells
RARE_LARGE_GT = 12                     # classes with at most this many large GT form the first APL group
T975_DF2 = 4.302652729911275           # 97.5 % point of Student t with 2 degrees of freedom
LABELS_A = ("W_small_gain_repeated_3of3", "W_small_gain_repeated_2of3", "W_small_gain_not_repeated")
LABEL_B_NONE = "no_practical_lead_in_any_seed"
LABELS_C = ("W_medium_cost_repeated_3of3", "W_medium_cost_repeated_2of3", "W_medium_cost_not_repeated")
STRONG_LAST_EPOCH = 117                # the loader's strong augmentation ends with epoch 117

READING = {
    "cells": [f"top{b}_{m}" for b in BUDGET_KEYS for m in METRICS],
    "difference": "same-seed difference of two arms, percentage points, e120 EMA, view A, 548 development images; decided on "
                  "the unrounded value, a value on the bound satisfies the rule",
    "sources": {"W seed 0": "R28-001 (eval-W), quoted", "L0 and F_geo, seeds 0-2": "R30-002 (summary-R30), quoted",
                "S15, seeds 0-2": "the top600 metric files of its three seeds, quoted", "W seeds 1-2": "this round"},
    "reported": "W - L0 and W - F_geo, 14 cells each, for every new seed; mean and sample standard deviation over the three seeds; "
                "S15 - W on the seven top600 cells",
    "A": {"rule": [list(r) for r in BOUND], "arms": "W - L0", "labels": {"3": LABELS_A[0], "2": LABELS_A[1], "1": LABELS_A[2]}},
    "B": {"rule": [list(r) for r in BOUND], "arms": "W - F_geo and F_geo - W", "per_seed": ["W_leads", "F_geo_leads", "neither"],
          "labels": {"all neither": LABEL_B_NONE, "otherwise": "W_leads_<n>of3_F_geo_leads_<m>of3"}},
    "W_unique_advantage": "changed to established only if all three seeds read W_leads; seed 0 reads neither, so no outcome of "
                          "this task changes it",
    "C": {"cell": "top600_APM", "arms": "W - L0", "maximum_pp": MEDIUM_COST,
          "labels": {"3": LABELS_C[0], "2": LABELS_C[1], "1": LABELS_C[2]}, "W_minus_F_geo": "reported per seed, no label"},
    "APL": {"cell": "top600_APL", "split": f"classes with at most {RARE_LARGE_GT} large GT (five) and the other five; each group's "
                                           "share of the 10-class macro difference", "label": None},
    "seven_cells": {"rule": [list(r) for r in SEVEN], "note": "the R28-001 table listed for each seed as a record; the seed-0 "
                                                              "registration is unchanged and no K run follows from it"},
    "geometry": "small-GT magnification and the share of small GT with g < 1 during training, against the W seed-0 ledger",
    "bound_note": "+1.0 is the practical lower bound already used in the paper, not a significance or power test; the n = 3 sample "
                  "standard deviation and t half-width are descriptions and do not establish equivalence of two arms",
    "expectation_of_the_controller": "reading A most likely 3of3, reading B most likely no_practical_lead_in_any_seed; not a criterion",
}


def require(cond, text):
    if not cond:
        raise ValueError(text)


def difference_pp(x, y):
    return {b: {m: 100. * (x[b][m] - y[b][m]) for m in METRICS} for b in BUDGET_KEYS}


def difference600_pp(x, y):
    """Seven cells from two flat top600 tables."""
    return {m: 100. * (x[m] - y[m]) for m in METRICS}


def negated(d):
    return {b: {m: -v for m, v in cells.items()} for b, cells in d.items()}


def bound_cells(diff):
    cells = {f"top{b}_{m}": {"difference_pp": diff[b][m], "minimum_pp": lo, "passed": diff[b][m] >= lo} for b, m, lo in BOUND}
    return {"cells": cells, "met": all(v["passed"] for v in cells.values())}


def lead(diff):
    """W - F_geo of one seed -> W_leads, F_geo_leads or neither (the same bound in both directions)."""
    w, f = bound_cells(diff), bound_cells(negated(diff))
    require(not (w["met"] and f["met"]), "both arms cannot lead on one seed")
    return {"reading": "W_leads" if w["met"] else "F_geo_leads" if f["met"] else "neither", "W_minus_F_geo": w, "F_geo_minus_W": f}


def seven_cells(diff):
    cells = {f"top{b}_{m}": {"difference_pp": diff[b][m], "minimum_pp": lo, "passed": diff[b][m] >= lo} for b, m, lo in SEVEN}
    return {"cells": cells, "failed_cells": [k for k, v in cells.items() if not v["passed"]], "passed": all(v["passed"] for v in cells.values())}


def sign(v):
    return "+" if v > 0 else "-" if v < 0 else "0"


def label_a(met):
    require(len(met) == 3 and all(type(m) is bool for m in met) and met[0] is True, "reading A needs three seeds and seed 0 satisfied")
    return LABELS_A[3 - sum(met)]


def label_b(readings):
    require(len(readings) == 3 and all(r in ("W_leads", "F_geo_leads", "neither") for r in readings) and readings[0] == "neither",
            "reading B needs three seeds and seed 0 neither")
    w, f = readings.count("W_leads"), readings.count("F_geo_leads")
    return LABEL_B_NONE if w == f == 0 else f"W_leads_{w}of3_F_geo_leads_{f}of3"


def label_c(cost):
    require(len(cost) == 3 and all(type(m) is bool for m in cost) and cost[0] is True, "reading C needs three seeds and seed 0 at the bound")
    return LABELS_C[3 - sum(cost)]


def aggregate(diffs):
    """diffs: {seed: 14-cell table}.  Mean, sample standard deviation (n - 1) and t half-width of the mean, per cell."""
    seeds = sorted(diffs)
    require(seeds == list(SEED_KEYS), "the aggregate is over seeds 0, 1, 2")
    out = {}
    for b in BUDGET_KEYS:
        out[b] = {}
        for m in METRICS:
            v = [diffs[s][b][m] for s in seeds]
            mean = math.fsum(v) / 3
            sd = math.sqrt(math.fsum((x - mean) ** 2 for x in v) / 2)
            out[b][m] = {"mean_pp": mean, "sample_sd_pp": sd, "t95_halfwidth_of_mean_pp": T975_DF2 * sd / math.sqrt(3), "values_pp": v, "n": 3}
    return out


def endpoint_summary(values):
    """values[seed][arm][budget][metric] (fractions) -> per arm and cell the three seed values in AP points (x100),
    their mean and sample standard deviation (n - 1).  A description of the three endpoints, not a comparison."""
    require(sorted(values) == list(SEED_KEYS), "the endpoint table is over seeds 0, 1, 2")
    out = {}
    for arm in sorted(values["0"]):
        out[arm] = {}
        for b in BUDGET_KEYS:
            out[arm][b] = {}
            for m in METRICS:
                v = [100. * values[s][arm][b][m] for s in SEED_KEYS]
                mean = math.fsum(v) / 3
                out[arm][b][m] = {"values": v, "mean": mean, "sample_sd": math.sqrt(math.fsum((x - mean) ** 2 for x in v) / 2), "n": 3}
    return out


def apl_split(mine, theirs):
    """Two per-class large-AP tables (rows category_id, GT, APL) -> the macro difference split into the classes with at
    most RARE_LARGE_GT large GT and the other classes.  A class without a large AP in either table is left out of the
    macro mean, as COCOeval does; the two tables must then agree on which classes those are."""
    a = {r["category_id"]: r for r in mine}
    b = {r["category_id"]: r for r in theirs}
    require(sorted(a) == sorted(b) == list(range(1, 11)) and all(a[k]["GT"] == b[k]["GT"] for k in a), "per-class APL tables")
    require(all((a[k]["APL"] is None) == (b[k]["APL"] is None) for k in a), "the two tables lack a large AP for different classes")
    used = [k for k in sorted(a) if a[k]["APL"] is not None]
    rows = [{"category_id": k, "large_GT": a[k]["GT"], "minuend": a[k]["APL"], "subtrahend": b[k]["APL"],
             "difference_pp": None if k not in used else 100. * (a[k]["APL"] - b[k]["APL"]),
             "group": "rare" if a[k]["GT"] <= RARE_LARGE_GT else "other"} for k in sorted(a)]
    groups = {}
    for g in ("rare", "other"):
        part = [r for r in rows if r["group"] == g]
        groups[g] = {"classes": [r["category_id"] for r in part], "large_GT": sum(r["large_GT"] for r in part),
                     "contribution_pp": math.fsum(r["difference_pp"] for r in part if r["difference_pp"] is not None) / len(used)}
    return {"classes": rows, "classes_in_the_macro_mean": len(used), "rare": groups["rare"], "other": groups["other"],
            "macro_difference_pp": groups["rare"]["contribution_pp"] + groups["other"]["contribution_pp"],
            "rare_rule": f"large GT <= {RARE_LARGE_GT}"}


def geometry(rows):
    """rows: iterable of (epoch, r28 ledger entry).  The entry holds, for the small GT of one update window, the
    geometric-mean magnification g_small, their number n_small and the share with g < 1.  Weighted by n_small the
    log-means give the geometric mean over every small GT instance of the span."""
    def zero():
        return {"updates": 0, "updates_with_small_gt": 0, "small_gt": 0, "log": [], "lt1": [], "g": [], "slope_min": [], "slope_max": [],
                "identity_images": 0}

    def fold(acc, r):
        acc["updates"] += 1
        acc["slope_min"].append(r["slope_min"])
        acc["slope_max"].append(r["slope_max"])
        acc["identity_images"] += r["identity_images"]
        n = r["n_small"]
        if n:
            require(r["g_small"] is not None and r["small_g_lt_1"] is not None and r["g_small"] > 0, "ledger row with small GT but no reading")
            acc["updates_with_small_gt"] += 1
            acc["small_gt"] += n
            acc["log"].append(n * math.log(r["g_small"]))
            acc["lt1"].append(n * r["small_g_lt_1"])
            acc["g"].append(r["g_small"])

    def close(acc):
        n = acc["small_gt"]
        return {"updates": acc["updates"], "updates_with_small_gt": acc["updates_with_small_gt"], "small_gt_instances": n,
                "g_small_geometric_mean": math.exp(math.fsum(acc["log"]) / n) if n else None,
                "g_small_mean_of_update_values": math.fsum(acc["g"]) / len(acc["g"]) if acc["g"] else None,
                "g_small_update_minimum": min(acc["g"]) if acc["g"] else None, "g_small_update_maximum": max(acc["g"]) if acc["g"] else None,
                "small_share_with_g_below_1": math.fsum(acc["lt1"]) / n if n else None,
                "slope_minimum": min(acc["slope_min"]) if acc["slope_min"] else None,
                "slope_maximum": max(acc["slope_max"]) if acc["slope_max"] else None, "identity_images": acc["identity_images"]}

    spans = {"all": zero(), "strong_augmentation_epochs": zero(), "last_three_epochs": zero()}
    for epoch, r in rows:
        fold(spans["all"], r)
        fold(spans["strong_augmentation_epochs" if epoch <= STRONG_LAST_EPOCH else "last_three_epochs"], r)
    return {k: close(v) for k, v in spans.items()}


def reading(values, s15, per_class):
    """values[seed][arm][budget][metric] for arms W, L0, FG and seeds "0", "1", "2"; s15[seed][metric] (top600);
    per_class[seed][arm] = per-class large-AP rows.  Everything of task section 5 except the training geometry."""
    require(sorted(values) == list(SEED_KEYS) and all(sorted(v) == ["FG", "L0", "W"] for v in values.values()), "three seeds, arms W / L0 / FG")
    require(sorted(s15) == list(SEED_KEYS) and sorted(per_class) == list(SEED_KEYS), "S15 values and per-class tables of three seeds")
    w_l0 = {s: difference_pp(values[s]["W"], values[s]["L0"]) for s in SEED_KEYS}
    w_fg = {s: difference_pp(values[s]["W"], values[s]["FG"]) for s in SEED_KEYS}
    s15_w = {s: difference600_pp(s15[s], values[s]["W"]["600"]) for s in SEED_KEYS}
    a = {s: bound_cells(w_l0[s]) for s in SEED_KEYS}
    b = {s: lead(w_fg[s]) for s in SEED_KEYS}
    apm = {s: {"W_minus_L0_pp": w_l0[s]["600"]["APM"], "sign": sign(w_l0[s]["600"]["APM"]), "at_or_below_the_bound": w_l0[s]["600"]["APM"] <= MEDIUM_COST,
               "W_minus_F_geo_pp": w_fg[s]["600"]["APM"], "W_minus_F_geo_sign": sign(w_fg[s]["600"]["APM"])} for s in SEED_KEYS}
    apl = {}
    for s in SEED_KEYS:
        split = {"W_minus_L0": apl_split(per_class[s]["W"], per_class[s]["L0"]), "W_minus_F_geo": apl_split(per_class[s]["W"], per_class[s]["FG"])}
        for key, d in (("W_minus_L0", w_l0), ("W_minus_F_geo", w_fg)):
            require(abs(split[key]["macro_difference_pp"] - d[s]["600"]["APL"]) < 1e-9, f"seed {s}: the per-class APL rows do not add up to the APL600 cell ({key})")
        apl[s] = {"W_minus_L0_pp": w_l0[s]["600"]["APL"], "sign": sign(w_l0[s]["600"]["APL"]), "W_minus_F_geo_pp": w_fg[s]["600"]["APL"],
                  "W_minus_F_geo_sign": sign(w_fg[s]["600"]["APL"]), "split": split}
    readings_b = [b[s]["reading"] for s in SEED_KEYS]
    w_leads_new = [s for s in ("1", "2") if b[s]["reading"] == "W_leads"]
    return {
        "W_minus_L0_pp": w_l0, "W_minus_F_geo_pp": w_fg, "S15_minus_W_top600_pp": s15_w,
        "aggregate_over_seeds": {"W_minus_L0": aggregate(w_l0), "W_minus_F_geo": aggregate(w_fg)},
        "reading_A": {"per_seed": a, "seeds_satisfied": [s for s in SEED_KEYS if a[s]["met"]], "label": label_a([a[s]["met"] for s in SEED_KEYS])},
        "reading_B": {"per_seed": b, "readings": dict(zip(SEED_KEYS, readings_b)), "label": label_b(readings_b)},
        "W_unique_advantage": {"status": "not_established", "changed_by_this_task": False,
                               "rule": "established only if all three seeds read W_leads; seed 0 reads neither",
                               "new_seeds_reading_W_leads": w_leads_new, "inconsistent_with_seed0": bool(w_leads_new)},
        "reading_C": {"per_seed": apm, "seeds_at_or_below_the_bound": [s for s in SEED_KEYS if apm[s]["at_or_below_the_bound"]],
                      "label": label_c([apm[s]["at_or_below_the_bound"] for s in SEED_KEYS])},
        "APL": apl,
        "seven_cells_W_minus_L0": {s: seven_cells(w_l0[s]) for s in SEED_KEYS},
    }
