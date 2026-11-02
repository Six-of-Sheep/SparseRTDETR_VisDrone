"""r19 contract metadata caller.

Every prior release stays immutable. --execution-plan
allows only the user-authorized resource-semantics amendment. Legacy numeric
budget keys remain for parser compatibility; effective policy is explicit.
"""
import argparse
import hashlib
import json
import secrets
from pathlib import Path

W = Path("/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace")
DUAL = W / "experiments/localization-distribution/r18-dual-ema-20261031-001"
R = W / "experiments/localization-distribution/r18-tiny-positives-20261102-001"
REL = R / "execution-release-001"
MANIFEST_NAME = "execution-source-001.json"
FIXTURE_SUFFIX = "001"
PRODUCTION_SUFFIX = "001"
RELEASE_SUFFIX = "001"
BASE_CONTRACT = DUAL / "production-contract-003.json"
PY = "/home/lyy/.cache/r0-seed0-e120-20261010-001/environment/bin/python"
PLAN_REL = "configs/research/r18_tiny_positives_v1.json"
ACCEPTANCE_DIR = "accept-positives"
PHASES = ("accept-positives", "formal-A0", "formal-A1", "formal-A2", "infer-A0-A-emaC",
          "infer-A1-A-emaC", "infer-A2-A-emaC", "evaluation")
CARRY = ("environment", "resource_limits", "storage", "dual_ema", "evaluation_policy", "materials",
         "parent_binding", "parent_checkpoint", "initial_model_sha256", "parent_metrics",
         "reference_metrics", "data", "data_sha256", "python", "gpu_uuid", "boot_id",
         "holdout_authorized", "local_polling", "no_retry", "source_commit")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def reference(path):
    p = Path(path)
    return {"path": str(p), "sha256": sha(p), "bytes": p.stat().st_size}


def digest(value):
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode()
    return hashlib.sha256(body).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def emit(value):
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


def release_identity():
    manifest = REL / MANIFEST_NAME
    build = load(REL / "BUILD.json")
    source_sha = sha(manifest)
    assert build["manifest_sha256"] == source_sha, "release manifest does not match its build receipt"
    return manifest, source_sha, build


def build_gate(fixture_job):
    job = Path(fixture_job)
    assert (job / "CHAIN_EXIT.json").is_file(), "the CPU fixture chain is not terminal"
    chain_exit = load(job / "CHAIN_EXIT.json")
    assert chain_exit["returncode"] == 0, ("CPU fixture chain failed", chain_exit)
    manifest, source_sha, build = release_identity()
    fixture_contract = R / ("cpu-fixture-" + FIXTURE_SUFFIX + "-contract.json")
    contract = load(fixture_contract)
    assert contract["contract_sha256"] == digest({k: v for k, v in contract.items()
                                                  if k != "contract_sha256"}), "fixture contract digest"
    phases = {}
    for phase in PHASES:
        closed = job / phase / "CLOSED.json"
        result = job / phase / "RESULT.json"
        assert closed.is_file() and result.is_file(), ("CPU fixture phase incomplete", phase)
        c = load(closed)
        assert c["status"] == "PASS" and c["phase"] == phase, ("CPU fixture phase not closed", phase)
        assert c["source_sha256"] == source_sha, ("CPU fixture phase source differs", phase)
        phases[phase] = {"status": "PASS", "closed": reference(closed), "result": reference(result)}
    acceptance = load(job / ACCEPTANCE_DIR / "RESULT.json")
    assert acceptance["actual_updates"] == 4 and acceptance["sequence"] == [1, 1, 1, 1], "CPU acceptance sequence"
    assert acceptance["pos_state_restore_exact"] is True, "CPU acceptance positives restore not exact"
    assert acceptance["state_usable_for_formal"] is False, "CPU acceptance state must not be reusable"
    assert acceptance["roles"] == ["A0", "A1", "A2", "empty"], "CPU acceptance roles differ"
    for case, row in acceptance["cases"].items():
        if case == "empty":
            assert row["pos"]["pairs_total"] == 0, "CPU acceptance empty case produced positives work"
    evaluation = load(job / "evaluation" / "RESULT.json")
    gate = {
        "schema": "r18-tiny-positives-cpu-gate-v1",
        "status": "PASS",
        "scope": "CPU_FIXTURE",
        "task_id": contract["task_id"],
        "contract_sha256": contract["contract_sha256"],
        "contract": reference(fixture_contract),
        "source_sha256": source_sha,
        "source_manifest": reference(manifest),
        "plan_sha256": contract["plan_sha256"],
        "release_derived_head": build["derived_head"],
        "phases": phases,
        "acceptance": {"actual_updates": acceptance["actual_updates"],
                       "sequence": acceptance["sequence"], "roles": acceptance["roles"],
                       "full_state_RNG_exact": acceptance["full_state_RNG_exact"],
                       "pos_state_restore_exact": acceptance["pos_state_restore_exact"],
                       "cases": sorted(acceptance["cases"])},
        "evaluation": {"quality": evaluation["quality"], "cost_passed": evaluation["cost_passed"]},
        "chain_exit": chain_exit,
        "cuda_hidden": True,
        "gpu_used": False,
        "note": ("CPU fixture entry gate over the synthetic fixture: four independent single-update "
                 "sub-runs (A0, A1, A2, empty supervision) with no GPU visible. The graded entry uses its "
                 "own contract and its own output root, and acceptance state is not reusable."),
    }
    gate["gate_content_sha256"] = digest(gate)
    path = R / ("cpu-gate-" + RELEASE_SUFFIX + ".json")
    assert not path.exists(), "the CPU gate already exists: " + str(path)
    path.write_bytes(json.dumps(gate, indent=1, ensure_ascii=False).encode("utf-8"))
    return {"status": "GATE-PUBLISHED", "gate": str(path), "gate_sha256": sha(path),
            "gate_content_sha256": gate["gate_content_sha256"], "source_sha256": source_sha,
            "phases": {k: v["status"] for k, v in phases.items()},
            "acceptance": gate["acceptance"], "gate_bytes": path.stat().st_size}


def build_contract(scope, gate_path, execution_plan=None, prior_costs=None):
    fixture = scope == "fixture"
    base = load(BASE_CONTRACT)
    manifest, source_sha, build = release_identity()
    contract = {key: base[key] for key in CARRY}
    contract.update({
        "scope": "CPU_FIXTURE" if fixture else "PRODUCTION",
        "GPU_authorized": not fixture,
        "formal_authorized": not fixture,
        "epochs": 2 if fixture else 12,
        "updates_per_epoch": 1 if fixture else 304,
        "total_updates": 2 if fixture else 3648,
        "task_id": "r18-tiny-positives-20261102-001-" + (("cpu-fixture-" + FIXTURE_SUFFIX) if fixture else ("production-" + PRODUCTION_SUFFIX)),
        "source_root": str((REL / "source").resolve()),
        "source_manifest": str(manifest),
        "source_sha256": source_sha,
        "plan": str(REL / "source" / PLAN_REL),
        "plan_sha256": sha(REL / "source" / PLAN_REL),
        "job_root": str(R / (("cpu-fixture-" + FIXTURE_SUFFIX) if fixture else ("production-" + PRODUCTION_SUFFIX))),
        "runtime_root": str(R / (("cpu-fixture-" + FIXTURE_SUFFIX + "-runtime") if fixture else ("production-" + PRODUCTION_SUFFIX + "-runtime"))),
        "launch_nonce": secrets.token_hex(16),
        "budget_limits": {"gpu_hours": 8.0, "wall_hours": 24, "server_new_bytes": 25769803776,
                          "return_bytes": 1048576, "batch_forwards_planned": 22361,
                          "no_vram_quota": True, "record_peak_memory": True},
        "positives": {"arms": ["A0", "A1", "A2"],
                      "arm_install": {"A0": None, "A1": None, "A2": None},
                      "pos_install": {"A0": None, "A1": "A1", "A2": "A2"},
                      "acceptance": {"updates": 4, "cases": ["A0", "A1", "A2", "empty"]},
                      "primary_endpoint": "FT012_EMA_C", "view": "A", "budgets": [600, 300],
                      "no_selection_no_tuning": True},
    })
    if execution_plan is not None:
        plan_path = Path(execution_plan).resolve()
        amended = load(plan_path)
        original_plan = load(REL / "source" / PLAN_REL)
        metadata = {"revision", "revision_log", "budget", "execution_resource_policy"}
        assert {k: v for k, v in amended.items() if k not in metadata} == {k: v for k, v in original_plan.items() if k not in metadata}, "scientific plan changed"
        policy = amended["execution_resource_policy"]
        assert policy["version"] == "r19-resource-reference-only-v1"
        assert all(v is None for v in policy["hard_resource_caps"].values()), "artificial resource cap restored"
        assert policy["automatic_retries"] == 0 and policy["cost_overrun_is_scientific_failure"] is False
        contract["plan"] = str(plan_path)
        contract["plan_sha256"] = sha(plan_path)
        contract["execution_resource_policy"] = policy
        contract["budget_limits"]["semantics"] = "reference values and legacy parser compatibility; not resource stop authority"
        contract["CPU_gate_reuse"] = {"reason": "execution declaration only; release-007 runtime source and scientific plan projection unchanged", "original_plan_sha256": sha(REL / "source" / PLAN_REL)}
    if fixture:
        contract["development_ids"] = base["development_ids"][:8]
        contract["fixed64"] = base["development_ids"][:8]
        assert len(contract["development_ids"]) == len(contract["fixed64"]) == 8
    else:
        contract["development_ids"] = base["development_ids"]
        contract["fixed64"] = base["fixed64"]
        assert len(set(contract["development_ids"])) == 548 and len(set(contract["fixed64"])) == 64
        assert set(contract["fixed64"]) <= set(contract["development_ids"])
        contract["CPU_gate"] = reference(Path(gate_path).resolve())
        gate = load(gate_path)
        assert gate["status"] == "PASS" and gate["source_sha256"] == source_sha, "the CPU gate is not valid for this release"
    if prior_costs is not None:
        history_path = Path(prior_costs).resolve()
        history = load(history_path)
        assert history["formal_updates"] == 0 and history["score_calls"] == 0, "formal interruption needs user decision"
        contract["prior_attempt_costs"] = reference(history_path)
        contract["materials"] = list(contract["materials"]) + [reference(history_path)]
    path = R / (("cpu-fixture-" + FIXTURE_SUFFIX + "-contract.json") if fixture else ("production-contract-" + PRODUCTION_SUFFIX + ".json"))
    token = R / (("cpu-fixture-" + FIXTURE_SUFFIX + "-token.json") if fixture else ("production-token-" + PRODUCTION_SUFFIX + ".json"))
    assert not path.exists() and not token.exists(), "contract or token already exists"
    assert not Path(contract["job_root"]).exists() and not Path(contract["runtime_root"]).exists(), "attempt output already exists"
    contract["contract_path"] = str(path)
    contract["contract_sha256"] = digest({k: v for k, v in contract.items() if k != "contract_sha256"})
    policy = load(contract["plan"]).get("execution_resource_policy", {})
    if policy:
        contract["execution_resource_policy"] = policy
        contract["budget_limits"]["semantics"] = "reference estimates only; no artificial resource caps"
        contract["engineering_retry_authorization"] = policy.get("engineering_repairs", {})
        contract["no_retry_semantics"] = "no blind process retry; authorized engineering fixes use new frozen attempts; formal interruption needs user decision"
        contract["contract_sha256"] = digest({k: v for k, v in contract.items() if k != "contract_sha256"})
    with path.open("xb") as stream:
        stream.write(json.dumps(contract, indent=1, ensure_ascii=False).encode("utf-8"))
    token = R / (("cpu-fixture-" + FIXTURE_SUFFIX + "-token.json") if fixture else ("production-token-" + PRODUCTION_SUFFIX + ".json"))
    token_value = {"task_id": contract["task_id"],
                                  "contract_sha256": contract["contract_sha256"],
                                  "nonce": contract["launch_nonce"]}
    with token.open("xb") as stream:
        stream.write(json.dumps(token_value, indent=1).encode("utf-8"))
    return {"status": "CONTRACT-PUBLISHED", "scope": contract["scope"], "contract": str(path),
            "contract_sha256": contract["contract_sha256"], "contract_bytes": path.stat().st_size,
            "token": str(token), "plan_sha256": contract["plan_sha256"], "source_sha256": source_sha,
            "job_root": contract["job_root"], "runtime_root": contract["runtime_root"],
            "epochs": contract["epochs"], "total_updates": contract["total_updates"],
            "development_ids": len(contract["development_ids"]), "fixed64": len(contract["fixed64"]),
            "gpu_uuid": contract["gpu_uuid"], "boot_id": contract["boot_id"], "python": PY,
            "CPU_gate": contract.get("CPU_gate")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", required=True, choices=("fixture", "gate", "production"))
    parser.add_argument("--fixture-job", default=None)
    parser.add_argument("--gate", default=None)
    parser.add_argument("--prior-costs", default=None, help="retained failed acceptance costs; formal interruptions cannot be restarted here")
    parser.add_argument("--execution-plan", default=None, help="resource-semantics amendment; scientific projection must equal the release plan")
    parser.add_argument("--release-suffix", default="001")
    parser.add_argument("--suffix", default="001")
    parser.add_argument("--production-suffix", default="001")
    args = parser.parse_args()
    global REL, MANIFEST_NAME, FIXTURE_SUFFIX, PRODUCTION_SUFFIX, RELEASE_SUFFIX
    PRODUCTION_SUFFIX = args.production_suffix
    RELEASE_SUFFIX = args.release_suffix
    REL = R / ("execution-release-" + args.release_suffix)
    MANIFEST_NAME = "execution-source-" + args.release_suffix + ".json"
    FIXTURE_SUFFIX = args.suffix
    if args.scope == "gate":
        assert args.fixture_job, "--fixture-job is required for the gate scope"
        emit(build_gate(args.fixture_job))
    elif args.scope == "production":
        assert args.gate, "--gate is required for the production scope"
        emit(build_contract("production", args.gate, args.execution_plan, args.prior_costs))
    else:
        emit(build_contract("fixture", None))


if __name__ == "__main__":
    main()
