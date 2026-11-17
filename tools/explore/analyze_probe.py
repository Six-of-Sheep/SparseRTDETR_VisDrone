"""Local analysis of diag-001 RESULT.json (pure stdlib, no numpy).

Applies the pre-registered decision rules (R-A1/A2/B1/B2/C, fixed before the
probe ran) and prints per-band cuts from the per-GT rows.
"""
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
imgs = data["images"]
rows = [(im["image_id"], g) for im in imgs for g in im["gts"]]
agg = data["aggregate"]
scale = {im["image_id"]: (1344.0 * 768.0) / (im["width"] * im["height"]) for im in imgs}


def q(vals, p):
    vals = sorted(vals)
    if not vals:
        return None
    k = (len(vals) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    return round(vals[f] + (vals[c] - vals[f]) * (k - f), 4)


def band_rows(band):
    return [g for _, g in rows if g["band"] == band]


def frac(vals):
    return round(sum(1 for v in vals if v) / len(vals), 4) if vals else None


print("== identity ==")
print(json.dumps(data["frozen"], indent=1))
print(json.dumps(data["model"], indent=1))
print(json.dumps(data["anchors"], indent=1))
print("elapsed", data["elapsed_seconds"], "cuda", data["cuda_initialized"])
print("images:", [im["image_id"][:12] for im in imgs])

print("== gts ==")
for band in ("small", "medium", "large"):
    b = band_rows(band)
    areas = [g["area"] * scale[iid] for iid, g in rows if g["band"] == band]
    print(band, "n=", len(b), "canvas_area med=", q(areas, 50), "q90=", q(areas, 90))

print("== coverage per band ==")
for band in ("small", "medium", "large"):
    b = band_rows(band)
    pool50 = [g for g in b if g["best_iou"] >= 0.5]
    pool75 = [g for g in b if g["best_iou"] >= 0.75]
    sel50 = [g for g in b if g["sel_best_iou"] >= 0.5]
    sel75 = [g for g in b if g["sel_best_iou"] >= 0.75]
    lost50 = [g for g in pool50 if g["sel_best_iou"] < 0.5]
    lost75 = [g for g in pool75 if g["sel_best_iou"] < 0.75]
    print("%s pool50=%.4f sel50=%.4f gap50=%.4f | pool75=%.4f sel75=%.4f gap75=%.4f | lost50=%d lost75=%d"
          % (band, len(pool50) / len(b), len(sel50) / len(b), len(pool50) / len(b) - len(sel50) / len(b),
             len(pool75) / len(b), len(sel75) / len(b), len(pool75) / len(b) - len(sel75) / len(b),
             len(lost50), len(lost75)))
    if lost50:
        rp = [g["best_rank_pct"] for g in lost50]
        print("  lost50 rank_pct: q10=%s med=%s q90=%s frac<=1.417%%=%s frac<=3%%=%s"
              % (q(rp, 10), q(rp, 50), q(rp, 90),
                 frac([v <= 0.01417 for v in rp]), frac([v <= 0.03 for v in rp])))
    rb = [g["best_rank_pct"] for g in b]
    print("  all best rank_pct med=%s q90=%s best_selected=%s"
          % (q(rb, 50), q(rb, 90), frac([g["best_selected"] for g in b])))

print("== small best-token error cuts ==")
small = band_rows("small")
for lo, hi in ((0.0, 0.3), (0.3, 0.5), (0.5, 0.75), (0.75, 1.01)):
    c = [g for g in small if lo <= g["best_iou"] < hi]
    ce = [g["center_err_over_gt"] for g in c if g["center_err_over_gt"] is not None]
    sr = [g["pred_w_over_gt_w"] for g in c if g["pred_w_over_gt_w"] is not None]
    print("[%s,%s) n=%d center med=%s q90=%s | pred_w/gt_w med=%s q10=%s q90=%s"
          % (lo, hi, len(c), q(ce, 50), q(ce, 90), q(sr, 50), q(sr, 10), q(sr, 90)))
lv = [g["best_level"] for _, g in rows]
print("best_level distribution all:", {k: lv.count(k) for k in sorted(set(lv))})

print("== aggregate sections ==")
for key in ("pool_cov", "sel_cov", "lost_at_0.5", "dup_selected_per_gt_ge50",
            "pred_w_over_gt_w", "small_best03_075", "small_best_iou_hist",
            "selected_level_counts", "selected_matched_by_band", "selected_unmatched",
            "score_size_spearman", "matcher_pairs"):
    print(key, "=", json.dumps(agg[key]))

print("== pre-registered rules ==")
s = band_rows("small")
pool50f = frac([g["best_iou"] >= 0.5 for g in s])
sel50f = frac([g["sel_best_iou"] >= 0.5 for g in s])
lost = [g for g in s if g["best_iou"] >= 0.5 and g["sel_best_iou"] < 0.5]
a1 = (pool50f - sel50f) >= 0.03
a2 = (frac([g["best_rank_pct"] <= 0.03 for g in lost]) or 0) >= 0.40 if lost else False
b1 = frac([g["best_iou"] >= 0.75 for g in s]) <= 0.55
b2 = ((agg["small_best03_075"]["abs_log_size_ratio_median"] or 0) >= 0.35 or
      (agg["small_best03_075"]["center_err_over_gt_median"] or 0) >= 0.4)
mp = agg["matcher_pairs"]
c1 = mp["small"]["iou_median"] >= mp["medium"]["iou_median"] - 0.05
c2 = mp["small"]["loc_share_mean"] >= 0.3
print("R-A1 gap>=3pp:", a1, " (gap=%.4f)" % (pool50f - sel50f))
print("R-A2 lost frac(rank_pct<=3%%)>=40%%:", a2, " n_lost=", len(lost))
print("R-B1 pool75small<=0.55:", b1, " (pool75=%.4f)" % frac([g["best_iou"] >= 0.75 for g in s]))
print("R-B2 coarseness signature:", b2, " (center=%s logsize=%s)"
      % (agg["small_best03_075"]["center_err_over_gt_median"],
         agg["small_best03_075"]["abs_log_size_ratio_median"]))
print("R-C kill-Hg:", (c1 and c2), " (small_iou=%s medium_iou=%s loc_share=%s)"
      % (mp["small"]["iou_median"], mp["medium"]["iou_median"], mp["small"]["loc_share_mean"]))
print("matcher iou diff small-medium:", round(mp["small"]["iou_median"] - mp["medium"]["iou_median"], 4))
