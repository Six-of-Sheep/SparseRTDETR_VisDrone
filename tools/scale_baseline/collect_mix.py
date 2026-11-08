"""TASK_MIX_SCALE_001 read-only collector (no torch, no CUDA, no scoring, no model files opened).

Writes into OUT: UPDATE_STATS.json (formal ledger statistics by canvas/epoch), INVENTORY.json (every round file
with bytes/SHA256, release source counted separately), SERVER_ONLY_SHA256.txt (checkpoints, acceptance/fixture
states, ledgers, predictions, refusal tables) and RETURN_LIST.txt (small-file return whitelist).

usage: python3 collect_mix.py ROUND OUT
"""
import hashlib, json, os, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path

ROUND, OUT = Path(sys.argv[1]), Path(sys.argv[2])
A = ROUND / "formal-001"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stats(v):
    v = sorted(v)
    return {"n": len(v), "sum": sum(v), "mean": sum(v) / len(v), "median": statistics.median(v),
            "p95": v[min(len(v) - 1, int(.95 * len(v)))], "max": v[-1], "min": v[0]}


rows = [json.loads(l) for l in open(A / "train-MIX/UPDATES.jsonl")]
by = defaultdict(list)
for r in rows:
    by[r["canvas"]].append(r)
per_epoch = defaultdict(Counter)
loss_epoch = defaultdict(list)
for r in rows:
    per_epoch[r["input"]["epoch"]][r["canvas"]] += 1
    loss_epoch[r["input"]["epoch"]].append(r["loss"])
upd = {"rows": len(rows), "first_update": rows[0]["input"]["update"], "last_update": rows[-1]["input"]["update"],
       "canvas": {k: {"updates": len(v), "train_window_seconds": stats([r["update_seconds"] for r in v]),
                      "peak_allocated_bytes_max": max(r["peak_allocated_bytes"] for r in v),
                      "gt_count_sum": sum(r["gt_count"] for r in v), "loss": stats([r["loss"] for r in v]),
                      "gradient_norm_before_clip": stats([r["gradient_norm_before_clip"] for r in v])}
                  for k, v in sorted(by.items())},
       "per_epoch_counts": {str(k): dict(v) for k, v in sorted(per_epoch.items())},
       "per_epoch_mean_loss": {str(k): sum(v) / len(v) for k, v in sorted(loss_epoch.items())},
       "last_row": {k: rows[-1][k] for k in ("canvas", "loss", "lr", "gradient_norm_before_clip", "ema_updates")}}
OUT.mkdir(exist_ok=True)
with open(OUT / "UPDATE_STATS.json", "x") as f:
    json.dump(upd, f, indent=2)

inventory, server_only, returned = [], [], []
LARGE = (".pth", "UPDATES.jsonl", "predictions.json", ".tsv")
for p in sorted(ROUND.rglob("*")):
    if not p.is_file() or "release-001/source" in str(p) or p.parent == OUT:
        continue
    rel = p.relative_to(ROUND).as_posix()
    row = {"path": rel, "bytes": p.stat().st_size, "sha256": sha(p)}
    inventory.append(row)
    if p.name.endswith(LARGE) and not rel.startswith("MIX_LH_V1_TABLE"):
        server_only.append(row)
    elif (not rel.startswith("cpu-train-001/bad-") and not rel.endswith(".partial") and not rel.endswith(".lock")
          and rel not in ("TASK_MIX_SCALE_001.md", "REVIEW_AND_DIRECTION_001.md")):  # same SHA already local
        returned.append(row)
release_bytes = sum(q.stat().st_size for q in (ROUND / "release-001/source").rglob("*") if q.is_file())
with open(OUT / "INVENTORY.json", "x") as f:
    json.dump({"files": inventory, "total_bytes_excluding_release_source": sum(r["bytes"] for r in inventory),
               "release_source_bytes": release_bytes}, f, indent=2)
with open(OUT / "SERVER_ONLY_SHA256.txt", "x") as f:
    for r in server_only:
        f.write(f"{r['sha256']}  {r['path']}  {r['bytes']}\n")
with open(OUT / "RETURN_LIST.txt", "x") as f:
    for r in returned:
        f.write(f"{r['sha256']}  {r['path']}\n")
print(json.dumps({"rows": len(rows), "inventory": len(inventory), "server_only": len(server_only),
                  "server_only_bytes": sum(r["bytes"] for r in server_only), "return": len(returned),
                  "return_bytes": sum(r["bytes"] for r in returned)}))
