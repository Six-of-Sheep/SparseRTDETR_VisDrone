#!/usr/bin/env python3
"""Verify the minimal P3 v2b REV1 authority/scientific/evaluation DAG."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from seal_v2b_contract_revision import (SealError, canonical_bytes, canonical_sha,
                                        file_row, identity_without_runtime,
                                        inventory, sha_bytes)


def _load(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if canonical_bytes(value) != raw:
        raise SealError(f"non-canonical JSON: {path}")
    if type(value) is not dict:
        raise SealError(f"JSON object required: {path}")
    return value


def _verify_authorities(doc: dict[str, Any]) -> dict[str, Any]:
    authorities = doc.get("authorities")
    if type(authorities) is not list or not authorities:
        raise SealError("authority list missing")
    body = {k: v for k, v in doc.items() if k != "manifest_sha256"}
    if doc.get("manifest_sha256") != canonical_sha(identity_without_runtime(body)):
        raise SealError("authority manifest digest mismatch")
    result = {}
    for authority in authorities:
        aid = authority.get("authority_id")
        locator = authority.get("runtime_locator", {}).get("path")
        if type(locator) is not str or not Path(locator).is_absolute():
            raise SealError(f"authority locator missing: {aid}")
        root = Path(locator).resolve(strict=True)
        identity = {k: v for k, v in authority.items()
                    if k not in {"identity_sha256", "runtime_locator", "runtime_file_count"}}
        if authority.get("identity_sha256") != canonical_sha(identity):
            raise SealError(f"authority identity digest mismatch: {aid}")
        rows = identity.get("inventory", {}).get("rows")
        if type(rows) is not list or not rows:
            raise SealError(f"authority rows missing: {aid}")
        actual = inventory(root, [row["relative_path"] for row in rows])
        if actual != identity["inventory"]:
            raise SealError(f"authority content drift: {aid}")
        result[aid] = True
    return {"count": len(result), "pass": True}


def _verify_source(doc: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    body = {k: v for k, v in doc.items() if k != "scientific_source_sha256"}
    if doc.get("scientific_source_sha256") != canonical_sha(body):
        raise SealError("scientific source digest mismatch")
    rows = doc.get("files")
    if type(rows) is not list or not rows:
        raise SealError("scientific source file list missing")
    for row in rows:
        if file_row(repo_root, row["relative_path"]) != row:
            raise SealError(f"scientific source drift: {row.get('relative_path')}")
    return {"file_count": len(rows), "pass": True}


def _verify_eval(doc: dict[str, Any], repo_root: Path, authorities: dict[str, Any]) -> dict[str, Any]:
    body = {k: v for k, v in doc.items() if k != "evaluation_contract_sha256"}
    if doc.get("evaluation_contract_sha256") != canonical_sha(identity_without_runtime(body)):
        raise SealError("evaluation contract digest mismatch")
    if doc.get("training_authorized") is not False or doc.get("continuation_authorized") is not False:
        raise SealError("evaluation contract unexpectedly authorizes training")
    if doc.get("development_authority_id") not in authorities:
        raise SealError("evaluation authority reference is unknown")
    for row in doc.get("evaluator_source", {}).get("files", []):
        if file_row(repo_root, row["relative_path"]) != row:
            raise SealError(f"evaluator source drift: {row.get('relative_path')}")
    entries = doc.get("checkpoints")
    if type(entries) is not list or len(entries) == 0 or len(entries) % 2:
        raise SealError("evaluation checkpoint matrix is invalid")
    seen = {}
    for entry in entries:
        selector = entry.get("weight_selector")
        if selector not in {"raw", "ema"}:
            raise SealError("unknown weight selector")
        locator = entry.get("runtime_locator", {}).get("path")
        if type(locator) is not str or not Path(locator).is_absolute():
            raise SealError("checkpoint runtime locator missing")
        path = Path(locator).resolve(strict=True)
        if sha_bytes(path.read_bytes()) != entry["checkpoint"]["sha256"]:
            raise SealError(f"checkpoint drift: {path}")
        key = entry["checkpoint"]["checkpoint_id"]
        seen.setdefault(key, set()).add(selector)
    if any(value != {"raw", "ema"} for value in seen.values()):
        raise SealError("raw/EMA pairing is incomplete")
    return {"checkpoint_count": len(seen), "entry_count": len(entries), "pass": True}


def verify(contract_dir: Path, repo_root: Path) -> dict[str, Any]:
    auth = _load(contract_dir / "external_authorities.json")
    sci = _load(contract_dir / "scientific_source.json")
    evaluation = _load(contract_dir / "evaluation_contract.json")
    revision = _load(contract_dir / "revision.json")
    body = {k: v for k, v in revision.items() if k != "revision_sha256"}
    if revision.get("revision_sha256") != canonical_sha(body):
        raise SealError("revision digest mismatch")
    graph = revision.get("graph", {})
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    indegree = {node: 0 for node in nodes}
    for edge in edges:
        if type(edge) is not list or len(edge) != 2 or edge[0] not in indegree or edge[1] not in indegree:
            raise SealError("invalid identity graph edge")
        indegree[edge[1]] += 1
    queue = [node for node in nodes if indegree[node] == 0]
    visited = 0
    while queue:
        node = queue.pop(0)
        visited += 1
        for edge in edges:
            if edge[0] == node:
                indegree[edge[1]] -= 1
                if indegree[edge[1]] == 0:
                    queue.append(edge[1])
    if visited != len(nodes):
        raise SealError("identity graph contains a cycle")
    authority_result = _verify_authorities(auth)
    authority_map = {item["authority_id"]: item for item in auth["authorities"]}
    source_result = _verify_source(sci, repo_root)
    eval_result = _verify_eval(evaluation, repo_root, authority_map)
    expected = {"external_authorities.json": authority_result,
                "scientific_source.json": source_result,
                "evaluation_contract.json": eval_result}
    for name, ref in revision.get("stages", {}).items():
        path = contract_dir / ref["path"]
        if sha_bytes(path.read_bytes()) != ref["sha256"]:
            raise SealError(f"stage digest drift: {name}")
    return {"status": "PASS", "revision_id": revision["revision_id"],
            "dag": "PASS", "external_authorities": authority_result,
            "scientific_source": source_result, "evaluation": eval_result}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract-dir", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.contract_dir.resolve(strict=True), args.repo_root.resolve(strict=True))
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
