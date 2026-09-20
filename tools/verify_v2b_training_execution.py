#!/usr/bin/env python3
"""Verify REV1 training/execution contracts without launching anything."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from v2b_training_execution_contracts import canonical_sha, load_json, sha_bytes, without_runtime

def verify(contract_dir: Path, repo_root: Path) -> dict:
    names = ["training_contract.json", "execution_source.json", "lineage.json", "execution_contract.json", "campaign_authorization.json", "smoke_authorization.json"]
    docs = {name: load_json(contract_dir / name) for name in names}
    training, execution, lineage, ec, auth, smoke = (docs[name] for name in names)
    for name, key in (("training_contract.json", "training_contract_sha256"), ("execution_source.json", "execution_source_sha256"), ("lineage.json", "lineage_sha256"), ("execution_contract.json", "execution_contract_sha256"), ("campaign_authorization.json", "authorization_sha256"), ("smoke_authorization.json", "smoke_authorization_sha256")):
        doc = docs[name]; body = {k: v for k, v in doc.items() if k != key}
        if doc.get(key) != canonical_sha(without_runtime(body)): raise ValueError(f"{name} digest mismatch")
    if execution["source_commit"] != "71c16cf881eaacd58f678ee64f1a5df2ccf3ad9d": raise ValueError("execution source historical identity missing")
    if ec["training_contract_sha256"] != training["training_contract_sha256"] or ec["execution_source_sha256"] != execution["execution_source_sha256"] or ec["lineage_sha256"] != lineage["lineage_sha256"]: raise ValueError("execution bindings are not exact")
    if auth["execution_contract_sha256"] != ec["execution_contract_sha256"] or auth["consumed"] is not False or auth["formal_launch_permitted"] is not False: raise ValueError("campaign authorization is not unconsumed fail-closed")
    boundary = lineage["resume_boundary"]
    if lineage["kind"] != "revision_aware_continuation" or boundary["start_epoch"] != 54 or boundary["start_window"] != 1 or boundary["mid_window_resume"] is not False: raise ValueError("lineage boundary invalid")
    failure = lineage["preserved_failure"]
    if failure["preserved"] is not True or failure["replayed"] is not False or failure["reinterpreted"] is not False: raise ValueError("historical failure was not preserved")
    data = training["data"]
    if data["effective_batch_size"] != 16 or data["resolution"] != 896 or data["seed"] != 2 or data["physical_batch_size"] != 8 or data["accumulation_steps"] != 2: raise ValueError("training semantics mismatch")
    cp = training["source_checkpoint"]; cp_path = Path(cp["runtime_locator"]["checkpoint"]); state_path = Path(cp["runtime_locator"]["state"])
    if sha_bytes(cp_path.read_bytes()) != cp["sha256"] or sha_bytes(state_path.read_bytes()) != cp["state_file_sha256"]: raise ValueError("checkpoint drift")
    nodes = ["external_authorities", "scientific_source", "training_contract", "execution_source", "execution_contract"]
    edges = [["external_authorities", "scientific_source"], ["scientific_source", "training_contract"], ["training_contract", "execution_source"], ["execution_source", "execution_contract"]]
    indegree = {n: 0 for n in nodes}
    for a, b in edges: indegree[b] += 1
    queue = [n for n in nodes if indegree[n] == 0]; visited = 0
    while queue:
        node = queue.pop(0); visited += 1
        for a, b in edges:
            if a == node:
                indegree[b] -= 1
                if indegree[b] == 0: queue.append(b)
    if visited != len(nodes): raise ValueError("DAG cycle")
    return {"status": "PASS", "graph": {"nodes": nodes, "edges": edges}, "training_contract_sha256": training["training_contract_sha256"], "execution_source_sha256": execution["execution_source_sha256"], "lineage_sha256": lineage["lineage_sha256"], "execution_contract_sha256": ec["execution_contract_sha256"], "campaign_authorization_sha256": auth["authorization_sha256"], "campaign_id": auth["campaign_id"], "smoke_status": smoke["status"], "checkpoint_epoch": cp["epoch"], "checkpoint_sha256": cp["sha256"]}

def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--repo-root", type=Path, required=True); ap.add_argument("--contract-dir", type=Path, required=True); args = ap.parse_args()
    try: print(json.dumps(verify(args.contract_dir.resolve(strict=True), args.repo_root.resolve(strict=True)), sort_keys=True, indent=2)); return 0
    except Exception as exc: print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True)); return 2

if __name__ == "__main__": raise SystemExit(main())
