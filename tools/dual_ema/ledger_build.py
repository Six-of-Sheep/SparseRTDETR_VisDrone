"""Assemble the accurate execution ledger for the r18-dual-ema chain.

Read-only over published stage artifacts:
  python3 ledger_build.py <job_root> <runtime_root> <out_ledger.json>

Itemizes batch forwards and updates per activity (training, acceptance,
restore+replay, endpoint inference for both EMA records), all stage wall
seconds and peak memory, and the whole-round budget check (2 GPU-hours).
A batch forward is one model forward on one micro-batch (8 images) unless
stated; endpoint inference batches are 4 images.
"""
import hashlib
import json
import sys
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    job = Path(sys.argv[1])
    runtime = Path(sys.argv[2])
    out = Path(sys.argv[3])
    arm = "D0"
    emas = ("R", "C")
    consumed = load(job / "CONSUMED.json") if (job / "CONSUMED.json").is_file() else None
    acc = load(job / ("accept-" + arm) / "RESULT.json")
    endp = load(job / ("formal-" + arm) / "RESULT.json")
    inf = {name: load(job / ("infer-%s-A-ema%s" % (arm, name)) / "RESULT.json") for name in emas}
    epochs = endp.get("epoch_wall_seconds") or []
    row = {
        "install_mode": acc.get("install_mode"),
        "dual_ema": {"acceptance": acc.get("dual_ema"), "endpoint": endp.get("dual_ema"),
                     "checkpoint_passes": endp.get("checkpoint_passes")},
        "acceptance": {
            "updates": acc["actual_updates"],
            "sequence": acc["sequence"],
            "direct_updates": 2,
            "direct_batch_forwards": 4,
            "restore_replay_updates": 1,
            "restore_replay_batch_forwards": 2,
            "probe_model_forwards": 0,
            "probe_backwards": 0,
            "ema_r_updates": acc["dual_ema"]["ema_r_updates"],
            "ema_c_updates": acc["dual_ema"]["ema_c_updates"],
            "rng_neutral_alt_ema_updates": acc["dual_ema"]["rng_neutral_alt_ema_updates"],
            "peak_reserved_bytes": acc.get("peak_reserved_bytes", 0),
            "elapsed_seconds": acc["elapsed_seconds"],
        },
        "training": {
            "updates": endp["updates"],
            "passes": endp["epoch"],
            "batch_forwards": 2 * endp["updates"],
            "image_forwards": 16 * endp["updates"],
            "wall_seconds": endp["training_wall_seconds"],
            "epoch_wall_seconds": epochs,
            "peak_reserved_bytes": endp.get("peak_reserved_bytes", 0),
        },
        "endpoint_inference": {name: {
            "detector_updates": 0,
            "ema_updates": inf[name]["ema_updates"],
            "model_forwards": inf[name]["model_forwards"],
            "batch_size": inf[name]["batch_size"],
            "warmup_batches": inf[name].get("warmup_batches"),
            "images": inf[name]["images"],
            "elapsed_seconds": inf[name]["elapsed_seconds"],
            "peak_reserved_bytes": inf[name].get("peak_reserved_bytes", 0),
        } for name in emas},
    }
    evaluation = load(job / "evaluation" / "RESULT.json")
    chain_exit = load(job / "CHAIN_EXIT.json")
    gpu_total = (row["acceptance"]["elapsed_seconds"] + row["training"]["wall_seconds"]
                 + sum(row["endpoint_inference"][n]["elapsed_seconds"] for n in emas))
    training_forwards = row["training"]["batch_forwards"]
    acceptance_forwards = row["acceptance"]["direct_batch_forwards"] + row["acceptance"]["restore_replay_batch_forwards"]
    inference_forwards = sum(row["endpoint_inference"][n]["model_forwards"] for n in emas)
    ledger = {
        "schema": "r18-dual-ema-ledger-v1",
        "task_id": consumed["task_id"] if consumed else None,
        "arms": {arm: row},
        "evaluation": {"score_budget_calls": evaluation.get("score_budget_calls"),
                       "quality": evaluation.get("quality"), "cpu_only": True},
        "totals": {
            "formal_updates": row["training"]["updates"],
            "acceptance_updates": row["acceptance"]["updates"],
            "training_batch_forwards": training_forwards,
            "training_image_forwards": row["training"]["image_forwards"],
            "acceptance_batch_forwards": acceptance_forwards,
            "inference_batch_forwards": inference_forwards,
            "batch_forwards_total": training_forwards + acceptance_forwards + inference_forwards,
            "batch_forwards_planned": 7612,
            "inference_model_forwards": inference_forwards,
            "gpu_stage_wall_seconds": gpu_total,
            "gpu_hours": gpu_total / 3600.0,
            "gpu_hours_limit": 2.0,
            "within_budget": gpu_total <= 2 * 3600,
        },
        "chain_exit": chain_exit,
        "note": ("batch forward = one model forward on one micro-batch unless stated; the planned ledger is "
                 "3648 training updates (7296 batch forwards) + 3 acceptance updates (6 forwards) + 2 x 155 "
                 "inference forwards = 7612; the second EMA adds no model forward and no RNG draw; all stage "
                 "wall seconds count toward the round budget"),
    }
    body = json.dumps(ledger, indent=1, ensure_ascii=False, sort_keys=True)
    out.write_bytes(body.encode("utf-8"))
    print("LEDGER", out, len(body), hashlib.sha256(body.encode()).hexdigest())


if __name__ == "__main__":
    main()
