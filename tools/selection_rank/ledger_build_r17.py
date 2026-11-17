"""Assemble the accurate r17 execution ledger (three ranking arms, one acceptance phase).

Read-only over published stage artifacts:
  python3 ledger_build_r17.py <job_root> <runtime_root> <out_ledger.json>

A batch forward is one model forward on one micro-batch (8 images) unless stated; endpoint
inference batches are 4 images.  All stage wall seconds count toward the round budget, whose
limit comes from the plan (12 GPU-hours) with the 20 GiB storage declaration checked by the
closeout.
"""
import hashlib
import json
import sys
from pathlib import Path

ARMS = ("A0", "A1", "A2")
PLANNED_FORWARDS = {"training": 2 * 3648 * 3, "acceptance": 6, "inference": 155 * 3}
GPU_HOURS_LIMIT = 12.0


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    job = Path(sys.argv[1])
    runtime = Path(sys.argv[2])
    out = Path(sys.argv[3])
    consumed = load(job / "CONSUMED.json") if (job / "CONSUMED.json").is_file() else None
    acc = load(job / "accept-ranking" / "RESULT.json")
    endp = {arm: load(job / ("formal-%s" % arm) / "RESULT.json") for arm in ARMS}
    inf = {arm: load(job / ("infer-%s-A-emaC" % arm) / "RESULT.json") for arm in ARMS}
    arms = {}
    for arm in ARMS:
        arms[arm] = {
            "ranking_mode": endp[arm].get("install_mode"),
            "training": {"updates": endp[arm]["updates"], "passes": endp[arm]["epoch"],
                         "batch_forwards": 2 * endp[arm]["updates"],
                         "image_forwards": 16 * endp[arm]["updates"],
                         "wall_seconds": endp[arm]["training_wall_seconds"],
                         "epoch_wall_seconds": endp[arm].get("epoch_wall_seconds"),
                         "peak_reserved_bytes": endp[arm].get("peak_reserved_bytes", 0),
                         "dual_ema_c": endp[arm].get("dual_ema")},
            "endpoint_inference": {"detector_updates": 0, "ema_updates": inf[arm]["ema_updates"],
                                   "model_forwards": inf[arm]["model_forwards"],
                                   "images": inf[arm]["images"], "batch_size": inf[arm]["batch_size"],
                                   "elapsed_seconds": inf[arm]["elapsed_seconds"],
                                   "peak_reserved_bytes": inf[arm].get("peak_reserved_bytes", 0)}}
    evaluation = load(job / "evaluation" / "RESULT.json")
    chain_exit = load(job / "CHAIN_EXIT.json") if (job / "CHAIN_EXIT.json").is_file() else None
    training_forwards = sum(row["training"]["batch_forwards"] for row in arms.values())
    acceptance_forwards = 6
    inference_forwards = sum(row["endpoint_inference"]["model_forwards"] for row in arms.values())
    gpu_total = (acc["elapsed_seconds"] + sum(row["training"]["wall_seconds"] for row in arms.values())
                 + sum(row["endpoint_inference"]["elapsed_seconds"] for row in arms.values()))
    ledger = {
        "schema": "r18-selection-rank-ledger-v1",
        "task_id": consumed["task_id"] if consumed else None,
        "acceptance": {"updates": acc["actual_updates"], "roles": acc.get("roles"),
                       "batch_forwards": acceptance_forwards,
                       "elapsed_seconds": acc["elapsed_seconds"],
                       "peak_reserved_bytes": acc.get("peak_reserved_bytes", 0),
                       "cases": {name: {"loss_key_counts": row.get("loss_key_counts"),
                                        "restore": row.get("restore")} for name, row in acc["cases"].items()}},
        "arms": arms,
        "evaluation": {"score_budget_calls": evaluation.get("score_budget_calls"),
                       "gates": evaluation.get("gates"), "verdict": evaluation.get("quality"),
                       "cpu_only": True},
        "totals": {
            "formal_updates": sum(row["training"]["updates"] for row in arms.values()),
            "acceptance_updates": acc["actual_updates"],
            "grand_total_updates": sum(row["training"]["updates"] for row in arms.values()) + acc["actual_updates"],
            "training_batch_forwards": training_forwards,
            "training_image_forwards": sum(row["training"]["image_forwards"] for row in arms.values()),
            "acceptance_batch_forwards": acceptance_forwards,
            "inference_batch_forwards": inference_forwards,
            "batch_forwards_total": training_forwards + acceptance_forwards + inference_forwards,
            "batch_forwards_planned": sum(PLANNED_FORWARDS.values()),
            "gpu_stage_wall_seconds": gpu_total,
            "gpu_hours": gpu_total / 3600.0,
            "gpu_hours_limit": GPU_HOURS_LIMIT,
            "within_budget": gpu_total <= GPU_HOURS_LIMIT * 3600,
        },
        "chain_exit": chain_exit,
        "note": ("batch forward = one model forward on one micro-batch (8 images) unless stated; endpoint inference "
                 "batches are 4 images; the planned ledger is 3 x 3648 training updates (21,888 forwards) + 3 "
                 "acceptance updates (6 forwards) + 3 x 155 inference forwards = 22,359; the second EMA adds no "
                 "model forward and no RNG draw; all stage wall seconds count toward the 12 GPU-hour budget"),
    }
    body = json.dumps(ledger, indent=1, ensure_ascii=False, sort_keys=True)
    out.write_bytes(body.encode("utf-8"))
    print("LEDGER", out, len(body), hashlib.sha256(body.encode()).hexdigest())


if __name__ == "__main__":
    main()
