"""Assemble the accurate execution ledger for the r18-detail-arms chain.

Read-only over published stage artifacts:
  python3 ledger_build.py <job_root> <runtime_root> <out_ledger.json>

Itemizes batch forwards and updates per activity (training, acceptance,
restore+replay, endpoint inference), all stage wall seconds and peak memory,
and the whole-round budget check (12 GPU-hours).  A batch forward is one model
forward on one micro-batch (8 images) unless stated; endpoint inference
batches are 4 images.
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
    arms = ("G0", "G1", "G2")
    consumed = load(job / "CONSUMED.json") if (job / "CONSUMED.json").is_file() else None
    rows = {}
    for arm in arms:
        acc = load(job / ("accept-" + arm) / "RESULT.json")
        endp = load(job / ("formal-" + arm) / "RESULT.json")
        inf = load(job / ("infer-" + arm + "-A") / "RESULT.json")
        probe = acc.get("arm_probe")
        epochs = endp.get("epoch_wall_seconds") or []
        rows[arm] = {
            "install_mode": acc.get("install_mode"),
            "acceptance": {
                "updates": acc["actual_updates"],
                "sequence": acc["sequence"],
                "direct_updates": 2,
                "direct_batch_forwards": 4,
                "restore_replay_updates": 1,
                "restore_replay_batch_forwards": 2,
                "probe_model_forwards": 4 if probe else 0,
                "probe_backwards": 2 if probe else 0,
                "probe_evidence": {k: probe[k] for k in (
                    "zero_gate_bitwise_outputs", "zero_gate_bitwise_loss",
                    "shared_parameter_gradients_bitwise_identical",
                    "branch_source_stop_gradient_enforced",
                    "gate_grad_nonzero_at_zero_gate", "conv_grad_zero_at_zero_gate",
                    "gate_moved_off_zero", "ema_gate_nonzero")} if probe else None,
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
            "endpoint_inference": {
                "detector_updates": 0,
                "model_forwards": inf["model_forwards"],
                "batch_size": inf["batch_size"],
                "warmup_batches": inf.get("warmup_batches"),
                "images": inf["images"],
                "elapsed_seconds": inf["elapsed_seconds"],
                "peak_reserved_bytes": inf.get("peak_reserved_bytes", 0),
            },
        }
    evaluation = load(job / "evaluation" / "RESULT.json")
    chain_exit = load(job / "CHAIN_EXIT.json")
    gpu_total = sum(r["acceptance"]["elapsed_seconds"] + r["training"]["wall_seconds"]
                    + r["endpoint_inference"]["elapsed_seconds"] for r in rows.values())
    ledger = {
        "schema": "r18-detail-arms-ledger-v1",
        "task_id": consumed["task_id"] if consumed else None,
        "arms": rows,
        "evaluation": {"score_budget_calls": evaluation.get("score_budget_calls"),
                       "quality": evaluation.get("quality"), "cpu_only": True},
        "totals": {
            "formal_updates": sum(r["training"]["updates"] for r in rows.values()),
            "acceptance_updates": sum(r["acceptance"]["updates"] for r in rows.values()),
            "training_batch_forwards": sum(r["training"]["batch_forwards"] for r in rows.values()),
            "training_image_forwards": sum(r["training"]["image_forwards"] for r in rows.values()),
            "inference_model_forwards": sum(r["endpoint_inference"]["model_forwards"] for r in rows.values()),
            "gpu_stage_wall_seconds": gpu_total,
            "gpu_hours": gpu_total / 3600.0,
            "gpu_hours_limit": 12.0,
            "within_budget": gpu_total <= 12 * 3600,
        },
        "chain_exit": chain_exit,
        "note": ("batch forward = one model forward on one micro-batch unless stated; "
                 "installed arms carry an extra acceptance probe (4 forwards / 2 backwards); "
                 "all stage wall seconds count toward the round budget"),
    }
    body = json.dumps(ledger, indent=1, ensure_ascii=False, sort_keys=True)
    out.write_bytes(body.encode("utf-8"))
    print("LEDGER", out, len(body), hashlib.sha256(body.encode()).hexdigest())


if __name__ == "__main__":
    main()
