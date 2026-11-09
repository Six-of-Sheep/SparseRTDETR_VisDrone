"""TASK_R26_001 summary: all readings of the three diagnostic arms, N vs historical S15 window, and measured cost.

Reads CLOSED/UPDATES/OBS/MONITORED files of the round; writes SUMMARY.json.  No model, no GPU.
"""
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROUND = Path(sys.argv[1])
S15_UPDATES = Path(sys.argv[2])


def rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def pairs_key(p):
    return (p["image"], p["gt"])


def main():
    out = {"arms": {}}
    hist = {r["input"]["update"]: r for r in rows(S15_UPDATES) if 18241 <= r["input"]["update"] <= 18368}
    obs = {}
    for arm in ("N", "U", "P"):
        d = ROUND / f"diag-{arm}"
        closed = json.loads((d / "CLOSED.json").read_text())
        upd = rows(d / "UPDATES.jsonl")
        assert [r["input"]["update"] for r in upd] == list(range(18241, 18369))
        o0 = json.loads((d / "OBS-u0.json").read_text())
        o1 = json.loads((d / "OBS-u128.json").read_text())
        obs[arm] = (o0, o1)
        losses = [r["loss"] for r in upd]
        a = {"updates": len(upd), "loss_first": losses[0], "loss_last": losses[-1],
             "loss_mean_1_16": statistics.mean(losses[:16]), "loss_mean_113_128": statistics.mean(losses[-16:]),
             "loss_mean_all": statistics.mean(losses),
             "grad_norm_pre_clip_mean": statistics.mean(r["gradient_norm_before_clip"] for r in upd),
             "grad_norm_pre_clip_max": max(r["gradient_norm_before_clip"] for r in upd),
             "update_seconds_median": statistics.median(r["update_seconds"] for r in upd),
             "all_finite": all(r["loss"] == r["loss"] for r in upd),
             "train_wall_seconds": closed["train_wall_seconds"],
             "peak_allocated_bytes_training": closed["peak_allocated_bytes_training"],
             "monitor64_overlap": {k: v for k, v in closed["monitor64_training_overlap"].items() if k != "ids"},
             "obs_u0": {k: v for k, v in o0["summary"].items() if k != "block_stats"},
             "obs_u128": {k: v for k, v in o1["summary"].items() if k != "block_stats"},
             "obs_block_stats_u0": o0["summary"]["block_stats"][:1],
             "obs_block_stats_u128": o1["summary"]["block_stats"][:1],
             "monitored": json.loads((d / "MONITORED.json").read_text())}
        a["obs_delta_u128_minus_u0"] = {k: a["obs_u128"][k] - a["obs_u0"][k] for k in
                                       ("final_eval_loss_mean", "loss_vfl_mean", "loss_bbox_mean", "loss_giou_mean")}
        # input alignment and N vs historical S15
        same_input = all(r["input"] == hist[r["input"]["update"]]["input"] for r in upd)
        a["inputs_equal_S15_formal_002"] = same_input
        a["lr_equal_S15"] = all(r["lr"] == hist[r["input"]["update"]]["lr"] for r in upd)
        dl = [r["loss"] - hist[r["input"]["update"]]["loss"] for r in upd]
        a["loss_minus_S15_same_update"] = {"mean": statistics.mean(dl), "max_abs": max(map(abs, dl)),
                                           "exact_equal_count": sum(x == 0 for x in dl)}
        dg = [r["gradient_norm_before_clip"] - hist[r["input"]["update"]]["gradient_norm_before_clip"] for r in upd]
        a["grad_norm_minus_S15"] = {"mean": statistics.mean(dg), "max_abs": max(map(abs, dg)),
                                    "exact_equal_count": sum(x == 0 for x in dg)}
        out["arms"][arm] = a
    # matched-pair change on common pairs across arms at u128
    common = None
    for arm in obs:
        k = {pairs_key(p): p for p in obs[arm][1]["pairs"]}
        common = set(k) if common is None else common & set(k)
    out["matching"] = {"pairs_u128": {a: len(obs[a][1]["pairs"]) for a in obs}, "common_pairs_u128": len(common)}
    for arm in obs:
        k0 = {pairs_key(p): p for p in obs[arm][0]["pairs"]}
        k1 = {pairs_key(p): p for p in obs[arm][1]["pairs"]}
        changed = sum(1 for key in set(k0) | set(k1) if key not in k0 or key not in k1 or k0[key]["query"] != k1[key]["query"])
        both = set(k0) & set(k1)
        out["matching"][arm] = {"pairs_u0": len(k0), "pairs_u128": len(k1), "assignment_changed_u0_u128": changed,
                                "common_u0_u128": len(both)}
        by = {}
        for key in both:
            b = ["small", "medium", "large"][k1[key]["bucket"]]
            s = by.setdefault(b, {"n": 0, "d_iou": 0., "d_l1": 0.})
            s["n"] += 1; s["d_iou"] += k1[key]["iou"] - k0[key]["iou"]; s["d_l1"] += k1[key]["l1"] - k0[key]["l1"]
        out["matching"][arm]["common_u0_u128_by_size"] = {b: {"n": s["n"], "iou_delta_mean": s["d_iou"] / s["n"],
                                                             "l1_delta_mean": s["d_l1"] / s["n"]} for b, s in by.items()}
        bucket = {}
        for key in common:
            p = {p["image"] + "|" + str(p["gt"]): p for p in []}
        cm = {}
        for key in common:
            b = ["small", "medium", "large"][k1[key]["bucket"]]
            s = cm.setdefault(b, {"n": 0, "iou": 0., "l1": 0.})
            s["n"] += 1; s["iou"] += k1[key]["iou"]; s["l1"] += k1[key]["l1"]
        out["matching"][arm]["common_across_arms_u128_by_size"] = {b: {"n": s["n"], "iou_mean": s["iou"] / s["n"],
                                                                      "l1_mean": s["l1"] / s["n"]} for b, s in cm.items()}
    # cost
    cpath = ROUND / "cost" / "CLOSED.json"
    if cpath.exists():
        c = json.loads(cpath.read_text())
        cost = {}
        for arm in ("N", "U", "P"):
            passes = [p for p in c["passes"] if p["arm"] == arm]
            per_pass = [sum(b["end_to_end_s"] for b in p["batches"]) for p in passes]
            gpu = [sum(b["stage_ms"]["model"] for b in p["batches"]) for p in passes]
            stages = {}
            for k in passes[0]["batches"][0]["stage_ms"]:
                stages[k] = statistics.median(sum(b["stage_ms"][k] for b in p["batches"]) / len(p["batches"]) for p in passes)
            cost[arm] = {"end_to_end_s_per_pass": per_pass, "end_to_end_median_s": statistics.median(per_pass),
                         "model_gpu_ms_per_pass": gpu, "model_gpu_ms_median_per_pass": statistics.median(gpu),
                         "model_gpu_ms_per_image_median": statistics.median(gpu) / 64,
                         "stage_ms_per_batch4_median": stages,
                         "load_resize_cpu_s_median_per_pass": statistics.median(sum(b["load_resize_cpu_s"] for b in p["batches"]) for p in passes),
                         "H2D_ms_median_per_pass": statistics.median(sum(b["H2D_ms"] for b in p["batches"]) for p in passes),
                         "peak_allocated": [p["peak_allocated"] for p in passes],
                         "parameters": c["parameters"][arm], "uninstrumented_equal": c["uninstrumented_check"][arm]}
        for arm in ("U", "P"):
            cost[arm]["ratio_model_gpu_vs_N"] = cost[arm]["model_gpu_ms_median_per_pass"] / cost["N"]["model_gpu_ms_median_per_pass"]
            cost[arm]["ratio_block_vs_N"] = cost[arm]["stage_ms_per_batch4_median"]["target_block"] / cost["N"]["stage_ms_per_batch4_median"]["target_block"]
            cost[arm]["ratio_end_to_end_vs_N"] = cost[arm]["end_to_end_median_s"] / cost["N"]["end_to_end_median_s"]
        cost["N"]["block_share_of_model_gpu"] = cost["N"]["stage_ms_per_batch4_median"]["target_block"] / cost["N"]["stage_ms_per_batch4_median"]["model"]
        out["cost"] = cost
    (ROUND / "SUMMARY.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps({a: {k: out["arms"][a][k] for k in ("loss_mean_1_16", "loss_mean_113_128", "obs_delta_u128_minus_u0",
                                                          "inputs_equal_S15_formal_002", "loss_minus_S15_same_update")}
                      for a in out["arms"]}, indent=1))


if __name__ == "__main__":
    main()
