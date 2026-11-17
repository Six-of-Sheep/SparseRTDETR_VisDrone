"""R28 descriptive CPU summary (action `dose`): recall of the certified small development GT by realised magnification.

AMENDMENT_002 section 6 keeps this table as a description only: it adds no forward and no scoring, and neither a
monotone nor a non-monotone pattern is read as a causal statement.  Inputs are files that already exist: the per-GT
magnification written by g0 (predicted demand of the frozen network), this arm's top600 predictions and the stored
L0 top600 predictions.  A GT counts as recalled at threshold t when some prediction of its image with the same class
has IoU >= t (no one-to-one assignment, no score threshold; the 600 candidates of the image are all used).
"""
from pathlib import Path

import r28_common as K
import r28_loop as LP

THRESHOLDS = (0.5, 0.75)


def _by_image(path):
    import json
    import numpy as np
    with Path(path).open() as f:
        records = json.load(f)
    out = {}
    for p in records:
        out.setdefault(p["image_id"], {}).setdefault(p["category_id"], []).append(p["bbox_xyxy"])
    return {i: {c: np.array(b, dtype=np.float64) for c, b in cats.items()} for i, cats in out.items()}, len(records)


def _best_iou(box, boxes):
    import numpy as np
    if boxes is None or not len(boxes):
        return 0.0
    lt = np.maximum(boxes[:, :2], box[:2])
    rb = np.minimum(boxes[:, 2:], box[2:])
    inter = np.clip(rb - lt, 0, None).prod(1)
    union = (boxes[:, 2:] - boxes[:, :2]).prod(1) + (box[2:] - box[:2]).prod() - inter
    return float((inter / union).max())


def table(per_gt, mine, theirs):
    """Pure: per_gt rows (image, category_id, area, predicted, bbox_xyxy) and two prediction indexes -> the table."""
    import numpy as np
    rows = [r for r in per_gt if r["area"] <= K.SMALL_AREA]
    bins = np.array([LP._bin(r["predicted"]) for r in rows])
    iou = {"arm": np.empty(len(rows)), "L0": np.empty(len(rows))}
    for n, r in enumerate(rows):
        box = np.array(r["bbox_xyxy"], dtype=np.float64)
        for name, index in (("arm", mine), ("L0", theirs)):
            iou[name][n] = _best_iou(box, index.get(r["image"], {}).get(r["category_id"]))
    out = {"small_GT": len(rows), "bins": []}
    for b, label in enumerate(LP.BIN_NAMES + ("all",)):
        sel = np.ones(len(rows), dtype=bool) if label == "all" else bins == b
        row = {"bin": label, "n": int(sel.sum())}
        for t in THRESHOLDS:
            key = f"recall@{t}"
            a = float((iou["arm"][sel] >= t).mean()) if sel.any() else None
            l = float((iou["L0"][sel] >= t).mean()) if sel.any() else None
            row[key] = {"arm": a, "L0": l, "difference_pp": None if a is None else 100. * (a - l)}
        out["bins"].append(row)
    top, bottom = out["bins"][3], out["bins"][0]
    out["high_minus_low_difference_pp"] = {
        f"recall@{t}": (None if top["n"] == 0 or bottom["n"] == 0 else
                        top[f"recall@{t}"]["difference_pp"] - bottom[f"recall@{t}"]["difference_pp"]) for t in THRESHOLDS}
    return out


def dose(c, out, counts):
    arm = K.ARM
    out = Path(out)
    K.require(out.resolve() == (K.ROUND / f"dose-{arm}").resolve(), f"this action writes <round>/dose-{arm}")
    g0, ev = K.stage_closed(c, "g0"), K.stage_closed(c, f"eval-{arm}", arm)
    per_gt = K.read_json(K.pinned(g0["files"]["per_gt"]))
    mine, n_mine = _by_image(K.pinned(ev["predictions"]["600"]))
    theirs, n_theirs = _by_image(K.pinned(c["L0_predictions"]["600"]))
    K.require(n_mine == n_theirs == 548 * 600, "top600 prediction record counts")
    res = table(per_gt, mine, theirs)
    K.require(counts["forwards"] == 0 and not counts.get("score_budget_calls"), "dose must not run a model or a scorer")
    return {"stage": f"dose-{arm}", "arm": arm, **res,
            "magnification": "per-GT g of the frozen demand network's plan at deployment (g0/per_gt.json); for arm K the "
                             "grouping still uses this network g, so the K table is a like-for-like grouping, not K's own g",
            "recall_rule": "same class, best IoU over the image's 600 candidates >= threshold; no one-to-one assignment",
            "inputs": {"per_gt": g0["files"]["per_gt"], "arm_predictions": ev["predictions"]["600"],
                       "L0_predictions": c["L0_predictions"]["600"]},
            "reading": "description only (AMENDMENT_002 section 6): no gate, no causal claim from the pattern"}
