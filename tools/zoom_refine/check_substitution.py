"""r20 integration check: implemented planner + substitution on the certified endpoint.

Runs on CPU against the frozen parent raw cache and the fixed 548-image development split.
Window detections are taken from the development annotations, so the substitution path is
driven end to end through the certified scorer and must land in the neighbourhood of the
probe's O2 (repair every band detection matched at IoU>=0.3) reading.  This checks the
implementation, not the method: it cannot estimate the window pass's own box quality.

usage: python check_substitution.py <release_source> <worktree_root> <parent_job> <contract> <out_json>
"""
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch

SRC, WORKTREE, PARENT, CONTRACT, OUT = (Path(a) for a in sys.argv[1:6])
sys.path.insert(0, str(SRC / "src"))
sys.path.insert(0, str(WORKTREE))

from sparse_rtdetr.r0 import evaluation as ev                                  # noqa: E402
from sparse_rtdetr.r0.evaluation_data import Development548                    # noqa: E402
from tools.zoom_refine.zoom_refine import (                                    # noqa: E402
    WINDOWS_PER_IMAGE, Window, invariants, plan_windows, substitute,
)

ROOT = SRC
KEYS = ("AP", "AP50", "AP75", "APS", "APM", "APL", "AP75_small")


def main():
    contract = json.loads(CONTRACT.read_text())
    data = Development548(contract["data"]["development"])
    rows, parsed = data.records, [data.raw(r) for r in data.records]
    cache = torch.load(PARENT / "inference/raw-queries-native-R18-e120.pth", map_location="cpu", weights_only=True)
    logits, boxes = cache["pred_logits"], cache["pred_boxes"]
    out = {"check": "zoom_refine_implementation_v1", "windows_per_image": WINDOWS_PER_IMAGE}

    for budget in (600, 300):
        preds = [p for i, row in enumerate(rows) for p in ev.rank_candidates(logits[i], boxes[i], row, budget)]
        base = ev.score_budget(rows, parsed, preds, ROOT)["COCO_bridge"]["metrics"]
        by_image = {}
        for p in preds:
            by_image.setdefault(p["image_id"], []).append(p)

        repaired, stats = [], {"windows": 0, "band_points": 0, "covered": 0, "coverage": [], "images_with_windows": 0}
        for row in rows:
            dets = by_image[row["stable_image_id"]]
            windows, plan = plan_windows(int(row["width"]), int(row["height"]), dets)
            stats["windows"] += len(windows)
            stats["band_points"] += plan["band_points"]
            stats["covered"] += plan["covered"]
            if windows:
                stats["images_with_windows"] += 1
            stats["coverage"].append(plan["coverage"])
            # stand-in window detections: the development annotations, attributed to the window that contains them
            window_dets = []
            for position, obj in enumerate(parsed[rows.index(row)]["rows"]):
                if obj["category"] not in range(1, 11) or obj["score"] == 0:
                    continue
                b = [float(v) for v in obj["box"]]
                cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
                owner = 0
                for w in windows:
                    if w.x0 <= cx <= w.x1 and w.y0 <= cy <= w.y1:
                        owner = w.view_id
                        break
                if owner == 0:
                    continue
                window_dets.append({"category_id": int(obj["category"]), "query_id": position,
                                    "score": 0.99, "window_id": owner, "bbox_xyxy": b})
            fixed, _ = substitute(dets, window_dets)
            repaired.extend(fixed)

        report = invariants(preds, repaired)
        after = ev.score_budget(rows, parsed, repaired, ROOT)["COCO_bridge"]["metrics"]
        cov = [c for c in stats["coverage"] if c is not None]
        stats["coverage"] = None
        stats["coverage_med"] = st.median(cov) if cov else 0.0
        stats["coverage_mean"] = float(np.mean(cov)) if cov else 0.0
        out[str(budget)] = {"base": {k: round(base[k], 6) for k in KEYS},
                            "after_gt_stand_in": {k: round(after[k], 6) for k in KEYS},
                            "delta_pp": {k: round((after[k] - base[k]) * 100, 4) for k in KEYS},
                            "invariants": report, "plan_stats": stats}
        print(f"[budget {budget}] invariants={report} plan={stats}")
        print(f"   base   {json.dumps({k: round(base[k], 6) for k in KEYS})}")
        print(f"   after  {json.dumps({k: round(after[k], 6) for k in KEYS})}")
        print(f"   delta  {json.dumps({k: round((after[k] - base[k]) * 100, 4) for k in KEYS})}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
