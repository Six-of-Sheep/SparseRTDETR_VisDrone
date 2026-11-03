"""r24 contract metadata caller: CPU fixture, CPU gate and the production contract.

The frozen releases and the r19 contracts stay immutable.  This round writes its own identity:
a new task id, the r24 release, the r24 arm map, the retired r19 install map, the unified
evaluation definition and the measured-cost policy.  Scientific anchors (data, parent bindings,
certified reference metrics, evaluation policy, materials) are carried from the r19 production
contract, which is the certified baseline of this lineage.

usage:
  prepare_contracts_r24.py --scope fixture
  prepare_contracts_r24.py --scope gate --fixture-job <job dir>
  prepare_contracts_r24.py --scope production --gate <cpu-gate json>
"""
import argparse
import hashlib
import json
import secrets
import sys
from pathlib import Path

W = Path("/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace")
R19 = W / "experiments/localization-distribution/r18-tiny-positives-20261102-001"
R = W / "experiments/localization-distribution/r24-roi-ld-20261103-001"
REL = R / "execution-release-r24-001"
MANIFEST_NAME = "execution-source-r24-001.json"
BASE_CONTRACT = R19 / "production-contract-002.json"
TEACHER_INDEX = (W / "experiments/localization-distribution/r18-multiview-distill-20261020-001/"
                 "attempt-gpu-002/teacher/index.json")
REV1_PLAN = R / "plan-001/r24_roi_ld_v1.json"
REV2_PLAN = R / "plan-002/r24_roi_ld_v1.rev2.json"
PLAN_AMENDMENT = R / "plan-002/PLAN_AMENDMENT_r24.json"
PREP_DIR = R / "prep-cpu-002"
PY = "/home/lyy/.cache/r0-seed0-e120-20261010-001/environment/bin/python"
PLAN_REL = "configs/research/r24_roi_ld_v1.rev2.json"
ACCEPTANCE_DIR = "accept-roi-ld"
PHASES = ("accept-roi-ld", "formal-A0", "formal-A1", "formal-A2", "infer-A0-A-emaC",
          "infer-A1-A-emaC", "infer-A2-A-emaC", "evaluation")
FIXTURE_SUFFIX = "r24-001"
PRODUCTION_SUFFIX = "r24-001"
RELEASE_SUFFIX = "r24-001"
CARRY = ("environment", "resource_limits", "storage", "dual_ema", "evaluation_policy", "materials",
         "parent_binding", "parent_checkpoint", "initial_model_sha256", "parent_metrics",
         "reference_metrics", "data", "data_sha256", "python", "gpu_uuid", "boot_id",
         "holdout_authorized", "local_polling", "no_retry")


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


def round_blocks(source_root):
    """The r24 blocks the engine enforces on every contract, read from the release itself."""
    sys.path.insert(0, str(source_root))
    sys.path.insert(0, str(source_root / "src"))
    from sparse_rtdetr.protocol_r18 import roi_ld, roi_ld_arm, roi_ld_data, slice_ft_execution
    cache = reference(TEACHER_INDEX)
    assert cache["sha256"] == roi_ld_data.FROZEN_INDEX["sha256"], "teacher cache index changed"
    assert cache["bytes"] == roi_ld_data.FROZEN_INDEX["bytes"], "teacher cache index size changed"
    assert cache["path"] == roi_ld_data.FROZEN_INDEX["path"], "teacher cache index path changed"
    return {
        "roi_ld": {"objective": roi_ld.OBJECTIVE, "modes": list(roi_ld_arm.MODES),
                   "roi_install": {a: slice_ft_execution.ROI_INSTALL[a] for a in ("A0", "A1", "A2")},
                   "target_iou": float(roi_ld.TARGET_IOU),
                   "targets_per_gt": int(roi_ld.TARGETS_PER_GT),
                   "band_area_px": float(roi_ld.BAND_AREA_PX),
                   "canvas_hw": list(roi_ld.CANVAS_HW),
                   "prefilter_iou_original_coordinates": float(roi_ld_data.PREFILTER_IOU),
                   "constants_sha256": roi_ld.constants_sha256(),
                   "teacher_cache": cache,
                   "common_support": ("a band ground-truth row is supervised only when the declared "
                                      "group's and the other group's best same-class box both reach "
                                      "the bar, so both armed arms supervise the same rows and the "
                                      "same number of pairs")},
        "positives": {"arms": ["A0", "A1", "A2"], "arm_install": {"A0": None, "A1": None, "A2": None},
                      "pos_install": {a: slice_ft_execution.POS_INSTALL_RETIRED[a] for a in ("A0", "A1", "A2")},
                      "retired": True,
                      "retired_note": ("the r19 tiny-band extra-positive pair is retired this round: no "
                                       "arm installs it, its modules stay present and inert, and every "
                                       "arm runs the unchanged native regression on top"),
                      "acceptance": {"updates": 4, "cases": ["A0", "A1", "A2", "empty"]},
                      "primary_endpoint": "FT012_EMA_C", "view": "A", "budgets": [600, 300],
                      "no_selection_no_tuning": True},
        "evaluation_definition": dict(slice_ft_execution.EVALUATION_DEFINITION),
    }


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
    assert acceptance["roles"] == ["A0", "A1", "A2", "empty"], "CPU acceptance roles differ"
    assert acceptance["roi_state_restore_exact"] is True, "CPU acceptance roi restore not exact"
    assert acceptance["full_state_RNG_exact"] is True, "CPU acceptance full state not exact"
    assert acceptance["state_usable_for_formal"] is False, "CPU acceptance state must not be reusable"
    assert acceptance["roi_install"] == {"A0": None, "A1": "R1", "A2": "R2"}, "CPU acceptance arm map differs"
    assert acceptance["pos_install"] == {"A0": None, "A1": None, "A2": None}, "CPU acceptance retirement differs"
    assert acceptance["a1_a2_common_support"] is True, "CPU acceptance common support not demonstrated"
    for case, row in acceptance["cases"].items():
        if case == "empty":
            assert row["roi"]["roi_selected"] == 0 and row["roi"]["roi_band_gt"] == 0, "CPU acceptance empty case produced roi work"
    evaluation = load(job / "evaluation" / "RESULT.json")
    assert evaluation["score_budget_calls"] == 6 and evaluation["scored_endpoint"] == "FT012_EMA_C", "CPU evaluation scope differs"
    assert evaluation["ema_r_scored"] is False, "CPU evaluation scored EMA-R"
    gate = {
        "schema": "r24-roi-ld-cpu-gate-v1",
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
                       "roi_state_restore_exact": acceptance["roi_state_restore_exact"],
                       "a1_a2_common_support": acceptance["a1_a2_common_support"],
                       "a1_a2_selected": acceptance["a1_a2_selected"],
                       "a1_a2_control_below_bar_rows": acceptance["a1_a2_control_below_bar_rows"],
                       "a1_a2_method_one_sided_rows": acceptance["a1_a2_method_one_sided_rows"],
                       "cases": sorted(acceptance["cases"])},
        "evaluation": {"quality": evaluation["quality"], "cost_passed": evaluation["cost_passed"],
                       "score_budget_calls": evaluation["score_budget_calls"],
                       "scored_endpoint": evaluation["scored_endpoint"]},
        "chain_exit": chain_exit,
        "cuda_hidden": True,
        "gpu_used": False,
        "note": ("CPU fixture entry gate over the synthetic fixture: four independent single-update "
                 "sub-runs (A0 native, A1 whole-image-view control, A2 raw-pixel-tile method, empty "
                 "supervision) with no GPU visible, then the six FT012 EMA-C budget scorings over the "
                 "fixture development ids.  The graded entry uses its own contract and its own output "
                 "root, and acceptance state is not reusable."),
    }
    gate["gate_content_sha256"] = digest(gate)
    path = R / ("cpu-gate-" + RELEASE_SUFFIX + ".json")
    assert not path.exists(), "the CPU gate already exists: " + str(path)
    path.write_bytes(json.dumps(gate, indent=1, ensure_ascii=False).encode("utf-8"))
    return {"status": "GATE-PUBLISHED", "gate": str(path), "gate_sha256": sha(path),
            "gate_content_sha256": gate["gate_content_sha256"], "source_sha256": source_sha,
            "phases": {k: v["status"] for k, v in phases.items()},
            "acceptance": gate["acceptance"], "gate_bytes": path.stat().st_size}


def build_contract(scope):
    fixture = scope == "fixture"
    base = load(BASE_CONTRACT)
    manifest, source_sha, build = release_identity()
    source_root = (REL / "source").resolve()
    blocks = round_blocks(source_root)
    plan = source_root / PLAN_REL
    contract = {key: base[key] for key in CARRY}
    contract.update({
        "scope": "CPU_FIXTURE" if fixture else "PRODUCTION",
        "GPU_authorized": not fixture,
        "formal_authorized": not fixture,
        "epochs": 2 if fixture else 12,
        "updates_per_epoch": 1 if fixture else 304,
        "total_updates": 2 if fixture else 3648,
        "task_id": "r24-roi-ld-20261103-001-" + (("cpu-fixture-" + FIXTURE_SUFFIX) if fixture
                                                 else ("production-" + PRODUCTION_SUFFIX)),
        "source_root": str(source_root),
        "source_manifest": str(manifest),
        "source_sha256": source_sha,
        "source_commit": build["commit"],
        "plan": str(plan),
        "plan_sha256": sha(plan),
        "job_root": str(R / (("cpu-fixture-" + FIXTURE_SUFFIX) if fixture else ("production-" + PRODUCTION_SUFFIX))),
        "runtime_root": str(R / (("cpu-fixture-" + FIXTURE_SUFFIX + "-runtime") if fixture
                                 else ("production-" + PRODUCTION_SUFFIX + "-runtime"))),
        "launch_nonce": secrets.token_hex(16),
        "budget_limits": {"gpu_hours": 8.0, "wall_hours": 24, "server_new_bytes": 25769803776,
                          "return_bytes": 1048576, "batch_forwards_planned": 22361,
                          "no_vram_quota": True, "record_peak_memory": True,
                          "semantics": ("reference estimates and legacy parser compatibility; the round "
                                        "stops on real resource safety limits, not on these estimates")},
        "materials": list(contract["materials"]) + [reference(TEACHER_INDEX), reference(REV1_PLAN),
                                                    reference(REV2_PLAN), reference(PLAN_AMENDMENT)]
                     + [reference(PREP_DIR / name) for name in
                        ("ENGINE_CHECKS.json", "COMPARE.json", "A0.json", "A1.json", "A2.json")
                        if (PREP_DIR / name).is_file()],
    })
    contract.update(blocks)
    amended = load(plan)
    contract["cost_policy"] = dict(amended["cost_policy"])
    contract["roi_ld"]["roi_install"] = dict(amended["roi_install"])
    if fixture:
        contract["development_ids"] = base["development_ids"][:8]
        contract["fixed64"] = base["development_ids"][:8]
        assert len(contract["development_ids"]) == len(contract["fixed64"]) == 8
    else:
        contract["development_ids"] = base["development_ids"]
        contract["fixed64"] = base["fixed64"]
        assert len(set(contract["development_ids"])) == 548 and len(set(contract["fixed64"])) == 64
        assert set(contract["fixed64"]) <= set(contract["development_ids"])
        contract["CPU_gate"] = reference(Path(GATE).resolve())
        gate = load(GATE)
        assert gate["status"] == "PASS" and gate["source_sha256"] == source_sha, "the CPU gate is not valid for this release"
    path = R / (("cpu-fixture-" + FIXTURE_SUFFIX + "-contract.json") if fixture
                else ("production-contract-" + PRODUCTION_SUFFIX + ".json"))
    token_path = R / (("cpu-fixture-" + FIXTURE_SUFFIX + "-token.json") if fixture
                      else ("production-token-" + PRODUCTION_SUFFIX + ".json"))
    assert not path.exists() and not token_path.exists(), "contract or token already exists"
    assert not Path(contract["job_root"]).exists() and not Path(contract["runtime_root"]).exists(), "attempt output already exists"
    # The engine's load() re-reads and PARSES every material, so a plain-text receipt would make the
    # frozen contract unusable at launch.  Refuse it here instead of discovering it on the GPU node.
    for row in contract["materials"]:
        json.loads(Path(row["path"]).read_text(encoding="utf-8"))
    contract["contract_path"] = str(path)
    contract["contract_sha256"] = digest({k: v for k, v in contract.items() if k != "contract_sha256"})
    with path.open("xb") as stream:
        stream.write(json.dumps(contract, indent=1, ensure_ascii=False).encode("utf-8"))
    token_value = {"task_id": contract["task_id"], "contract_sha256": contract["contract_sha256"],
                   "nonce": contract["launch_nonce"]}
    with token_path.open("xb") as stream:
        stream.write(json.dumps(token_value, indent=1).encode("utf-8"))
    return {"status": "CONTRACT-PUBLISHED", "scope": contract["scope"], "contract": str(path),
            "contract_sha256": contract["contract_sha256"], "contract_bytes": path.stat().st_size,
            "token": str(token_path), "plan_sha256": contract["plan_sha256"], "source_sha256": source_sha,
            "job_root": contract["job_root"], "runtime_root": contract["runtime_root"],
            "epochs": contract["epochs"], "total_updates": contract["total_updates"],
            "development_ids": len(contract["development_ids"]), "fixed64": len(contract["fixed64"]),
            "gpu_uuid": contract["gpu_uuid"], "boot_id": contract["boot_id"], "python": PY,
            "CPU_gate": contract.get("CPU_gate"), "materials": len(contract["materials"]),
            "roi_install": contract["roi_ld"]["roi_install"], "pos_install": contract["positives"]["pos_install"],
            "evaluation_definition": contract["evaluation_definition"]}


GATE = None


def main():
    global REL, MANIFEST_NAME, FIXTURE_SUFFIX, PRODUCTION_SUFFIX, RELEASE_SUFFIX, GATE
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", required=True, choices=("fixture", "gate", "production"))
    parser.add_argument("--fixture-job", default=None)
    parser.add_argument("--gate", default=None)
    parser.add_argument("--suffix", default=FIXTURE_SUFFIX)
    parser.add_argument("--production-suffix", default=PRODUCTION_SUFFIX)
    parser.add_argument("--release-suffix", default=RELEASE_SUFFIX)
    args = parser.parse_args()
    FIXTURE_SUFFIX = args.suffix
    PRODUCTION_SUFFIX = args.production_suffix
    RELEASE_SUFFIX = args.release_suffix
    REL = R / ("execution-release-" + args.release_suffix)
    MANIFEST_NAME = "execution-source-" + args.release_suffix + ".json"
    GATE = args.gate
    if args.scope == "gate":
        assert args.fixture_job, "--fixture-job is required for the gate scope"
        emit(build_gate(args.fixture_job))
    elif args.scope == "production":
        assert args.gate, "--gate is required for the production scope"
        emit(build_contract("production"))
    else:
        emit(build_contract("fixture"))


if __name__ == "__main__":
    main()
