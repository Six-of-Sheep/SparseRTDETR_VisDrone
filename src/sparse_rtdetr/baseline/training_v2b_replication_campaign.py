"""Frozen paired-seed replication controller; no retry or implicit extension."""
from __future__ import annotations
import copy
from dataclasses import asdict
import os
from pathlib import Path
import subprocess
import sys
import time

from .training_v2b_campaign import (
    CampaignError, GuardianSupervisor, _canonical, _data_path, _safe_id,
    _worker_environment, _write_json, file_reference, read_reference,
    source_snapshot, validate_frozen_source,
)
from .training_v2b_evidence import strict_json_loads, canonical_sha256

KIND = "v2b_seed_resolution_replication_freeze"
AUTHORIZATION_SHA = "0ea553eef36857f9686fa8bad3ad2b708af9f22a83f8f194b4405157a6a3b055"
CELLS = {"s1_r640": (1, 640), "s1_r896": (1, 896),
         "s2_r640": (2, 640), "s2_r896": (2, 896)}
STAGES = ("capacity", "smoke", "smoke_replay", "control30")
FREEZE_KEYS = {
    "schema_version", "kind", "campaign_id", "repo_root", "output_root", "source",
    "clean_archive_reference", "qualification_reference", "authorization_reference",
    "foundation_reference", "common", "cells", "formal_order",
}
COMMON_KEYS = {"torch_version", "code_files", "train_core", "pretrained", "policy_bundle",
               "expected_gpu_uuid", "num_workers", "prefetch_factor"}
CELL_KEYS = {"config", "development_binding", "epoch_order_sha256", "capacity_plan", "invocations"}


def _require(value, message):
    if not value:
        raise CampaignError(message)


def _json(reference):
    return strict_json_loads(read_reference(reference))


def _same(left, right, label):
    _require(_canonical(left) == _canonical(right), label + " differs")


def epoch_orders(seed):
    from .training_v2b import logical_batch_indices
    _require(type(seed) is int and seed in (1, 2), "only predeclared seeds one and two")
    return {str(epoch): canonical_sha256([list(batch) for batch in logical_batch_indices(
        4869, seed=seed, epoch=epoch, logical_batch_size=16)]) for epoch in range(1, 31)}


def fixed_config(seed, input_size):
    from .training_v2b import V2BConfig
    _require((seed, input_size) in CELLS.values(), "unapproved replication cell")
    return asdict(V2BConfig(seed=seed, input_size=input_size, physical_batch_size=8,
                           accumulation_steps=2, sampling_backend="deterministic_gather"))


def invocation_plan(campaign_id, output_root):
    campaign_id = _safe_id(campaign_id)
    root = Path(output_root)
    result = {}
    for cell in CELLS:
        result[cell] = {}
        for stage in STAGES:
            identifier = _safe_id(campaign_id + "-" + cell + "-" + stage)
            run_stage = "smoke" if stage == "smoke_replay" else stage
            result[cell][stage] = {
                "run_id": _safe_id(campaign_id + "-" + cell + "-" + run_stage),
                "invocation_id": identifier,
                "output_dir": str(root / "execution" / (cell + "-" + stage)),
            }
    return result


def _external_policy(policy, authorization):
    from .training_v2b_admission import _validate_policy
    _require(type(authorization) is dict and authorization.get("sha256") == AUTHORIZATION_SHA,
             "the explicit paired-seed authority is required")
    read_reference(authorization)
    _same(policy.get("authorization_reference"), authorization, "policy authorization")
    _require(policy.get("setter_mode") == "external_admin_acknowledged",
             "replication cannot set or reset GPU clocks")
    _same([policy.get("clock_min_mhz"), policy.get("clock_max_mhz")], [1500, 1500], "clock ceiling")
    _same(policy.get("settings_after_exit"), "keep_1500_1500", "clock exit behavior")
    read_reference(policy["external_clock_receipt"])
    for scope, seconds in (("paired_smoke", 600), ("train_core_30epoch", 43200)):
        _validate_policy(policy, scope, seconds)


def _development_identity(cells):
    left, right = cells["s1_r640"]["development_binding"], cells["s1_r896"]["development_binding"]
    for size in (640, 896):
        _same(cells["s1_r"+str(size)]["development_binding"],
              cells["s2_r"+str(size)]["development_binding"], "cross-seed development")
    invariant = lambda value: {key: item for key, item in value.items()
                                if key not in {"policy", "binding_sha256"}}
    _same(invariant(left), invariant(right), "development bytes and membership")
    policy = copy.deepcopy(left["policy"])
    policy["input_size"] = [896, 896]
    resize = [entry for entry in policy["transform"] if entry.get("type") == "Resize"]
    _require(len(resize) == 1 and resize[0]["size"] == [640, 640], "unique development resize")
    resize[0]["size"] = [896, 896]
    _same(policy, right["policy"], "development resolution intervention")


def validate_replication_freeze(freeze, *, verify_files=True):
    _require(type(freeze) is dict and set(freeze) == FREEZE_KEYS, "replication freeze keys differ")
    _same([freeze["schema_version"], freeze["kind"]], [1, KIND], "replication freeze kind")
    _safe_id(freeze["campaign_id"])
    root = Path(freeze["repo_root"]).resolve(strict=True)
    output = Path(freeze["output_root"])
    _require(output.is_absolute() and output.resolve() == output and
             output.is_relative_to(root / "artifacts" / "training") and
             output != root / "artifacts" / "training", "replication output path escaped")
    _same(freeze["source"]["repo_root"], str(root), "source checkout")
    _require(type(freeze["common"]) is dict and set(freeze["common"]) == COMMON_KEYS,
             "replication common fields differ")
    common = freeze["common"]
    _same(common["code_files"], freeze["source"]["code_files"], "complete source inventory")
    _same([common["num_workers"], common["prefetch_factor"]], [2, 2], "worker and prefetch counts")
    _same(freeze["formal_order"], list(CELLS), "predeclared formal order")
    _require(type(freeze["cells"]) is dict and set(freeze["cells"]) == set(CELLS),
             "exactly four predeclared replication cells are required")
    identities = invocation_plan(freeze["campaign_id"], output)
    for cell, (seed, size) in CELLS.items():
        value = freeze["cells"][cell]
        _require(type(value) is dict and set(value) == CELL_KEYS, "cell fields differ")
        _same(value["config"], fixed_config(seed, size), "fixed scientific configuration")
        _same(value["epoch_order_sha256"], epoch_orders(seed), "full thirty-epoch sampler plan")
        _same(value["invocations"], identities[cell], "unique invocation identities")
    _development_identity(freeze["cells"])
    _external_policy(common["policy_bundle"], freeze["authorization_reference"])
    _same(common["expected_gpu_uuid"], common["policy_bundle"]["gpu_uuid"], "GPU identity")
    if verify_files:
        from .training_v2b_replication_gate import validate_replication_qualification
        from .training_v2b_replication_capacity import validate_replication_capacity_plan
        from .training_v2b_development import validate_development_binding
        qualification = validate_replication_qualification(freeze["qualification_reference"])
        _require(qualification.get("status") == "PASS" and
                 qualification.get("eligible_for_fixed_paired_seed1_and_seed2") is True,
                 "authenticated resolution evidence does not qualify for seed replication")
        _same(qualification["conditional_authorization_reference"], freeze["authorization_reference"],
              "qualified execution authorization")
        archive = _json(freeze["clean_archive_reference"])
        _require(archive.get("status") == "PASS" and
                 archive.get("candidate_tree") == freeze["source"]["tree"],
                 "clean archive does not verify this exact source tree")
        foundation = _json(freeze["foundation_reference"])
        _same(foundation.get("predeclared_additional_seeds"), [1, 2], "original seed declaration")
        for cell, (_, size) in CELLS.items():
            value = freeze["cells"][cell]
            validate_replication_capacity_plan(_json(value["capacity_plan"]),
                seed=value["config"]["seed"], input_size=size, train_core=common["train_core"], verify_files=True)
            validate_development_binding(value["development_binding"], verify_files=True)
        for field in ("annotation", "manifest"):
            _data_path(common["train_core"][field]["path"],
                       expected_name="train_core_"+("coco.json" if field == "annotation" else "manifest.json"))
            read_reference(common["train_core"][field])
        _data_path(common["train_core"]["image_root"])
        read_reference(common["pretrained"])
        validate_frozen_source(freeze["source"])
    return copy.deepcopy(freeze)


def make_replication_freeze(*, repo_root, campaign_id, output_root, input_spec,
                            capacity_metadata, qualification_reference, clean_archive_reference,
                            policy_bundle, authorization_reference, foundation_reference):
    """Prepare four seed cells on CPU after evidence qualification, before any GPU work."""
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "preparation requires CUDA hidden")
    from .training_v2b_development import build_development_binding
    from .training_v2b_replication_gate import validate_replication_qualification
    from .training_v2b_replication_capacity import make_replication_capacity_plan
    import torch
    root = Path(repo_root).resolve(strict=True)
    campaign_id = _safe_id(campaign_id)
    output = Path(output_root).absolute()
    _require(output.resolve() == output and output.is_relative_to(root/"artifacts"/"training")
             and output != root/"artifacts"/"training" and not output.exists(),
             "replication requires a unique new training output")
    _require(type(input_spec) is dict and set(input_spec) == {"train_core", "development", "pretrained"},
             "exact authorized input roles required")
    _require(type(capacity_metadata) is dict and set(capacity_metadata) == {"1", "2"},
             "both predeclared seed metadata plans required")
    for role in ("train_core", "development"):
        value = input_spec[role]
        _require(type(value) is dict and set(value) == {"annotation", "manifest", "image_root"},
                 "exact input specification fields required")
        for field in ("annotation", "manifest"):
            _data_path(value[field]["path"], expected_name=role+"_"+(
                "coco.json" if field == "annotation" else "manifest.json"))
            read_reference(value[field])
        _data_path(value["image_root"])
    _data_path(input_spec["pretrained"]["path"], expected_name="ResNet18_vd_pretrained_from_paddle.pth")
    read_reference(input_spec["pretrained"])
    qualification = validate_replication_qualification(qualification_reference)
    _require(qualification.get("status") == "PASS" and
             qualification.get("eligible_for_fixed_paired_seed1_and_seed2") is True,
             "seed replication qualification has not passed")
    _same(qualification["conditional_authorization_reference"], authorization_reference, "qualification authority")
    _external_policy(policy_bundle, authorization_reference)
    source = source_snapshot(root)
    archive = _json(clean_archive_reference)
    _require(archive.get("status") == "PASS" and archive.get("candidate_tree") == source["tree"],
             "this source tree lacks passed clean-archive evidence")
    dev_spec = input_spec["development"]
    developments = {size: build_development_binding(
        repo_root=root, annotation_file=dev_spec["annotation"]["path"],
        annotation_sha256=dev_spec["annotation"]["sha256"],
        manifest_file=dev_spec["manifest"]["path"], manifest_sha256=dev_spec["manifest"]["sha256"],
        image_root=dev_spec["image_root"], input_size=size) for size in (640, 896)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700, exist_ok=False)
    plans = output / "frozen-plans"; plans.mkdir(mode=0o700)
    identities = invocation_plan(campaign_id, output)
    cells = {}
    for cell, (seed, size) in CELLS.items():
        metadata = capacity_metadata[str(seed)]
        _require(set(metadata) == {"report", "plan"}, "capacity metadata reference fields differ")
        capacity = make_replication_capacity_plan(
            seed=seed, input_size=size, metadata_report_reference=metadata["report"],
            metadata_plan_reference=metadata["plan"])
        capacity_ref = _write_json(plans/(cell+"-capacity.json"), capacity)
        cells[cell] = {"config": fixed_config(seed, size), "development_binding": developments[size],
                       "epoch_order_sha256": epoch_orders(seed), "capacity_plan": capacity_ref,
                       "invocations": identities[cell]}
    freeze = {
        "schema_version": 1, "kind": KIND, "campaign_id": campaign_id, "repo_root": str(root),
        "output_root": str(output), "source": source,
        "clean_archive_reference": clean_archive_reference, "qualification_reference": qualification_reference,
        "authorization_reference": authorization_reference, "foundation_reference": foundation_reference,
        "common": {"torch_version": str(torch.__version__), "code_files": source["code_files"],
                   "train_core": {**input_spec["train_core"], "expected_samples": 4869,
                                  "expected_batches_per_epoch": 304},
                   "pretrained": input_spec["pretrained"], "policy_bundle": policy_bundle,
                   "expected_gpu_uuid": policy_bundle["gpu_uuid"], "num_workers": 2, "prefetch_factor": 2},
        "cells": cells, "formal_order": list(CELLS),
    }
    validate_replication_freeze(freeze)
    _require(not torch.cuda.is_initialized(), "CPU preparation initialized CUDA")
    reference = _write_json(output/"freeze.json", freeze)
    return {"freeze": freeze, "freeze_reference": reference}


def replication_worker_contract(freeze, freeze_reference, cell, stage, *,
                                stage_gate_reference=None, matched_reference=None, replay_source=None):
    _require(cell in CELLS and stage in STAGES, "unknown replication invocation")
    value = freeze["cells"][cell]
    return copy.deepcopy({
        "schema_version": 1, "kind": "v2b_seed_resolution_replication_worker",
        "campaign_id": freeze["campaign_id"], "cell_id": cell, "stage": stage,
        **value["invocations"][stage], "repo_root": freeze["repo_root"],
        "freeze_reference": freeze_reference, "qualification_reference": freeze["qualification_reference"],
        "stage_gate_reference": stage_gate_reference, **freeze["common"],
        "config": value["config"], "development_binding": value["development_binding"],
        "epoch_order_sha256": value["epoch_order_sha256"], "capacity_plan": value["capacity_plan"],
        "matched_reference": matched_reference, "replay_source": replay_source,
    })


def prerequisite_keys(cell, stage, completed):
    """Exact dependency closure; no later formal cell can bypass an earlier cell."""
    _require(cell in CELLS and stage in STAGES, "unknown prerequisite cell or stage")
    if stage == "capacity":
        return []
    keys = {(cell, "capacity")}
    if stage == "smoke_replay":
        keys.add((cell, "smoke"))
    if cell.endswith("r896") and stage in ("smoke", "smoke_replay"):
        keys.add((cell.replace("r896", "r640"), "smoke"))
    if stage == "control30":
        keys = {(name, prior) for name in CELLS for prior in ("capacity", "smoke", "smoke_replay")}
        index = list(CELLS).index(cell)
        if index:
            keys.add((list(CELLS)[index-1], "control30"))
    _require(keys.issubset(completed), "required earlier capacity, replay or formal stage is missing")
    return sorted(keys)


def _completed_evidence(freeze, freeze_reference, entry, *, native_recheck=True):
    from .training_v2b_admission import wait_for_monitored_finish, _validate_policy
    completion = _json(entry["completion_reference"])
    result = _json(entry["worker_result_reference"])
    cell, stage = entry["cell_id"], entry["stage"]
    expected = freeze["cells"][cell]["invocations"][stage]
    _require(completion.get("kind") == "v2b_seed_resolution_replication_completion" and
             completion.get("status") == result.get("status") == "PASS", "prerequisite did not pass")
    for obj in (completion, result):
        for key, value in (("campaign_id", freeze["campaign_id"]), ("cell_id", cell),
                           ("stage", stage), ("freeze_reference", freeze_reference)):
            _same(obj.get(key), value, "prerequisite "+key)
    _same(completion["result_reference"], entry["worker_result_reference"], "completed worker reference")
    contract = _json(completion["contract_reference"])
    _same(result.get("contract_reference"), completion["contract_reference"], "completed worker contract")
    for key, value in expected.items():
        _same(contract.get(key), value, "completed invocation "+key)
    _same(result.get("run_id"), expected["run_id"], "completed run identity")
    _same(result.get("invocation_id"), expected["invocation_id"], "completed invocation identity")
    launch = _json(completion["launch_reference"])
    _same(launch["contract"], completion["contract_reference"], "launched contract")
    _require(type(result.get("worker_pid")) is int and result["worker_pid"] == launch["pid"],
             "worker PID differs from actual launch")
    if native_recheck:
        observed = wait_for_monitored_finish(result["monitor_reference"], timeout_seconds=1.)
        _same(observed, completion["monitor"], "native completion evidence")
    monitor = completion["monitor"]
    _same(monitor["owner"], launch["owner"], "native and launched owner")
    _require(monitor.get("worker_exited") is True and monitor.get("status") == "PASS",
             "native completion has no successful owner exit")
    scope, limit = ("train_core_30epoch", 43200) if stage == "control30" else ("paired_smoke", 600)
    policy = _validate_policy(freeze["common"]["policy_bundle"], scope, limit)
    for key, value in (("run_id", expected["run_id"]),
                       ("gpu_uuid", freeze["common"]["expected_gpu_uuid"]),
                       ("policy_sha256", policy["policy_sha256"])):
        _same(monitor.get(key), value, "native "+key)
    if stage == "capacity":
        reference = completion.get("capacity_memory_audit_reference")
        _require(type(reference) is dict, "capacity native memory audit is absent")
        audit = _json(reference)
        _require(audit.get("status") == "PASS", "capacity native memory audit failed")
    return result, completion


def make_stage_gate(freeze, freeze_reference, cell, stage, completed, smoke_comparisons):
    _require(stage != "capacity", "capacity has qualification, not a later-stage gate")
    keys = prerequisite_keys(cell, stage, completed)
    prerequisites = []
    for key in keys:
        entry = completed[key]
        _completed_evidence(freeze, freeze_reference, entry)
        prerequisites.append(copy.deepcopy(entry))
    comparisons = {}
    if stage == "control30":
        _require(set(smoke_comparisons) == set(CELLS), "all four same-cell smoke replays must pass")
        for name, ref in smoke_comparisons.items():
            observed = _json(ref)
            _require(observed.get("status") == "PASS", "smoke comparison failed")
            comparisons[name] = ref
    gate = {"schema_version": 1, "kind": "v2b_seed_resolution_replication_stage_gate",
            "status": "PASS", "campaign_id": freeze["campaign_id"], "freeze_reference": freeze_reference,
            "cell_id": cell, "stage": stage, "prerequisites": prerequisites,
            "smoke_comparisons": comparisons}
    path = Path(freeze["output_root"])/"stage-gates"/(cell+"-"+stage+".json")
    return _write_json(path, gate)


def _verify_stage_completion(freeze, freeze_reference, contract, contract_reference,
                             result_reference, result, completion, candidate_reference, owner):
    """Use the worker's complete verifier before publishing a stage completion.

    The candidate is a unique validation input, never a returned prerequisite.
    The canonical completion is published only after this check and exit evidence.
    """
    from .training_v2b_replication_worker import (
        _successful_cell_source, _verify_completion,
    )
    verified_result, source_contract = _successful_cell_source(
        result_reference, cell_id=contract["cell_id"], stage=contract["stage"],
        contract=contract, freeze=freeze)
    _same(verified_result, result, "fresh completed worker result")
    _same(source_contract, contract, "actual launched worker contract")
    _same(result["contract_reference"], contract_reference, "launched contract reference")
    _same(result["freeze_reference"], freeze_reference, "completed freeze reference")
    _same(completion["monitor"]["owner"], owner, "live supervisor and final guardian owner")
    verified = _verify_completion(
        candidate_reference, result_reference, verified_result, source_contract, contract, freeze)
    _same(verified, completion, "verified completion candidate")
    return verified


def _run_stage(freeze, freeze_reference, contract, contract_reference):
    """Own one worker, retain the first failure, and publish only verified completion."""
    from .training_v2b_admission import _validate_policy, wait_for_monitored_finish
    root, execution = Path(freeze["repo_root"]), Path(freeze["output_root"])/"execution"
    cell, stage = contract["cell_id"], contract["stage"]
    label = cell+"-"+stage
    scope, limit = ("train_core_30epoch", 43200) if stage == "control30" else ("paired_smoke", 600)
    policy = _validate_policy(freeze["common"]["policy_bundle"], scope, limit)
    argv = [sys.executable, "-B", str(root/"tools/run_training_v2b_replication_worker.py"),
            "--contract", contract_reference["path"], "--contract-sha256", contract_reference["sha256"]]
    start = time.monotonic()
    with (execution/(label+".stdout.log")).open("xb", buffering=0) as out, \
            (execution/(label+".stderr.log")).open("xb", buffering=0) as err:
        process = subprocess.Popen(argv, cwd=root,
            env=_worker_environment({"expected_gpu_uuid": freeze["common"]["expected_gpu_uuid"]}),
            stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
        supervisor = None
        completion = None
        result_reference = None
        first_error = first_traceback = None
        cleanup_records, evidence_errors = [], []
        guardian_cleanup = None

        def describe(error):
            return {"type": type(error).__name__, "message": str(error)}

        def remember(error):
            nonlocal first_error, first_traceback
            if first_error is None:
                first_error, first_traceback = error, error.__traceback__

        def fallback_evidence(payload):
            # This controller stderr is already owned by the outer campaign.
            # A failed disk write must never replace the original runtime error.
            try:
                sys.stderr.write(_canonical(payload).decode("utf-8")+"\n")
                sys.stderr.flush()
            except BaseException:
                pass

        def evidence_error(operation, error):
            remember(error)
            row = {"operation": operation, "error": describe(error)}
            evidence_errors.append(row)
            fallback_evidence({
                "status": "STOP_NO_RETRY", "cell_id": cell, "stage": stage,
                "contract_reference": contract_reference,
                "first_failure": describe(first_error), "evidence_error": row,
            })

        def cleanup_step(operation, action):
            """Every completed cleanup attempt gets its own immutable receipt."""
            row = {"schema_version": 1, "kind": "v2b_replication_controller_cleanup_step",
                   "cell_id": cell, "stage": stage, "contract_reference": contract_reference,
                   "operation": operation}
            result = None
            try:
                result = action()
                row.update(status="PASS", result=result)
            except BaseException as error:
                remember(error)
                row.update(status="ERROR", error=describe(error))
            row["first_failure"] = None if first_error is None else describe(first_error)
            record = {"operation": operation, "status": row["status"], "reference": None}
            if "error" in row:
                record["error"] = row["error"]
            cleanup_records.append(record)
            path = execution/(label+".cleanup-"+str(len(cleanup_records)).zfill(2)+"-"+operation+".json")
            try:
                record["reference"] = _write_json(path, row)
            except BaseException as error:
                evidence_error("publish_cleanup_"+operation, error)
            return result

        def publish_failure():
            # Preserve the public failure schema; additional per-step details
            # live in separate cleanup records and the private summary.
            failure = {
                "status": "STOP_NO_RETRY", "cell_id": cell, "stage": stage,
                "contract_reference": contract_reference,
                "only_owned_campaign_processes_signalled": True,
                "guardian_cleanup": guardian_cleanup, "failure": describe(first_error),
            }
            try:
                _write_json(execution/(label+".controller-failure.json"), failure)
            except BaseException as error:
                evidence_error("publish_controller_failure", error)
            summary = {
                "schema_version": 1, "kind": "v2b_replication_controller_cleanup_summary",
                "status": "STOP_NO_RETRY", "cell_id": cell, "stage": stage,
                "contract_reference": contract_reference, "first_failure": describe(first_error),
                "only_owned_campaign_processes_signalled": True,
                "cleanup_records": cleanup_records, "evidence_errors": evidence_errors,
            }
            try:
                _write_json(execution/(label+".cleanup-summary.json"), summary)
            except BaseException as error:
                evidence_error("publish_cleanup_summary", error)

        try:
            supervisor = GuardianSupervisor(process.pid, Path(contract["output_dir"])/"native-hardware",
                run_id=contract["run_id"], gpu_uuid=freeze["common"]["expected_gpu_uuid"],
                policy_sha256=policy["policy_sha256"])
            launch_reference = _write_json(execution/(label+".launch.json"),
                {"argv": argv, "pid": process.pid, "owner": supervisor.owner,
                 "contract": contract_reference, "cell_id": cell, "stage": stage,
                 "monotonic_started": start})
            while process.poll() is None:
                _require(time.monotonic() <= start+limit+300, "worker total deadline exceeded")
                supervisor.check()
                time.sleep(.05)
            result_path = Path(contract["output_dir"])/"worker-result.json"
            _require(process.returncode == 0 and result_path.is_file(),
                     label+" worker failed; no automatic retry")
            result_reference = file_reference(result_path)
            result = _json(result_reference)
            _require(result.get("status") == "PASS" and result.get("worker_pid") == process.pid,
                     "worker did not pass with the owned PID")
            for key, value in (("contract_reference", contract_reference),
                               ("freeze_reference", freeze_reference), ("run_id", contract["run_id"]),
                               ("invocation_id", contract["invocation_id"]), ("cell_id", cell), ("stage", stage)):
                _same(result.get(key), value, "worker result "+key)
            monitor = wait_for_monitored_finish(result["monitor_reference"])
            _require(monitor.get("status") == "PASS", "guardian did not close successfully")
            validate_frozen_source(freeze["source"])
            completion = {
                "schema_version": 1, "kind": "v2b_seed_resolution_replication_completion", "status": "PASS",
                "campaign_id": freeze["campaign_id"], "cell_id": cell, "stage": stage,
                "freeze_reference": freeze_reference, "contract_reference": contract_reference,
                "result_reference": result_reference, "launch_reference": launch_reference,
                "monitor": monitor, "elapsed_seconds": time.monotonic()-start,
                "capacity_memory_audit_reference": None,
            }
            if stage == "capacity":
                from .training_v2b_replication_capacity import audit_replication_capacity_native_memory
                audit = audit_replication_capacity_native_memory(completion)
                _require(audit.get("status") == "PASS", "capacity resource margin did not pass")
                completion["capacity_memory_audit_reference"] = _write_json(
                    execution/(label+".memory-audit.json"), audit)
            candidate_reference = _write_json(execution/(label+".completion-candidate.json"), completion)
            _verify_stage_completion(
                freeze, freeze_reference, contract, contract_reference, result_reference, result,
                completion, candidate_reference, supervisor.owner)
        except BaseException as error:
            remember(error)

        if first_error is not None:
            if supervisor is not None:
                cleanup_step("stop_owned_worker", supervisor.stop_owned_worker)
            else:
                # An unreaped direct child cannot have had its PID reused.
                cleanup_step("stop_unreaped_child",
                             lambda: process.kill() if process.poll() is None else None)
            cleanup_step("wait_owned_worker", lambda: process.wait(timeout=5.))
            if supervisor is not None:
                # The existing pidfd helper refuses to signal a guardian until
                # owner exit is proven, including after a failed wait.
                guardian_cleanup = cleanup_step("close_failed_guardian", supervisor.close_failed_guardian)

        if supervisor is not None:
            cleanup_step("close_supervisor_handles", supervisor.close)
        cleanup_step("publish_exit_receipt", lambda: _write_json(execution/(label+".exit.json"),
            {"returncode": process.poll(), "pid": process.pid, "contract": contract_reference,
             "stdout": file_reference(execution/(label+".stdout.log")),
             "stderr": file_reference(execution/(label+".stderr.log"))}))

        if first_error is not None:
            publish_failure()
            raise first_error.with_traceback(first_traceback)

        try:
            completion_reference = _write_json(execution/(label+".completion.json"), completion)
        except BaseException as error:
            remember(error)
            publish_failure()
            raise first_error.with_traceback(first_traceback)
        return {"cell_id": cell, "stage": stage, "worker_result_reference": result_reference,
                "completion_reference": completion_reference}

def run_replication_campaign(freeze, *, freeze_reference):
    _same(_json(freeze_reference), freeze, "invoked freeze")
    validate_replication_freeze(freeze)
    root = Path(freeze["output_root"])
    execution = root/"execution"; execution.mkdir(mode=0o700, exist_ok=False)
    contracts = execution/"contracts"; contracts.mkdir(mode=0o700)
    (root/"stage-gates").mkdir(mode=0o700)
    completed, comparisons = {}, {}
    report = {"schema_version": 1, "kind": "v2b_seed_resolution_replication_campaign_result",
              "status": "RUNNING", "campaign_id": freeze["campaign_id"],
              "freeze_reference": freeze_reference, "completed": [], "smoke_comparisons": comparisons,
              "formal_started": False, "automatic_retry": False, "primary_endpoint_epoch": 30,
              "health_only_epochs": [10, 20], "training_extension_authorized": False}
    started = time.monotonic()

    def execute(cell, stage):
        validate_frozen_source(freeze["source"])
        gate = None if stage == "capacity" else make_stage_gate(
            freeze, freeze_reference, cell, stage, completed, comparisons)
        matched = replay = None
        if cell.endswith("r896") and stage != "capacity":
            peer_stage = "control30" if stage == "control30" else "smoke"
            matched = completed[(cell.replace("r896", "r640"), peer_stage)]["worker_result_reference"]
        if stage == "smoke_replay":
            replay = completed[(cell, "smoke")]["worker_result_reference"]
        contract = replication_worker_contract(freeze, freeze_reference, cell, stage,
            stage_gate_reference=gate, matched_reference=matched, replay_source=replay)
        from .training_v2b_replication_worker import validate_replication_worker_contract
        validate_replication_worker_contract(contract, verify_files=True)
        reference = _write_json(contracts/(cell+"-"+stage+".json"), contract)
        entry = _run_stage(freeze, freeze_reference, contract, reference)
        completed[(cell, stage)] = entry
        report["completed"].append(entry)

    try:
        for cell in CELLS:
            execute(cell, "capacity")
        from .training_v2b_replication_worker import compare_replication_smoke
        for cell in CELLS:
            execute(cell, "smoke")
            execute(cell, "smoke_replay")
            comparison = compare_replication_smoke(
                completed[(cell, "smoke")]["worker_result_reference"],
                completed[(cell, "smoke_replay")]["worker_result_reference"],
                freeze_reference=freeze_reference)
            _require(comparison.get("status") == "PASS", "same-cell replay did not pass")
            comparisons[cell] = _write_json(execution/(cell+".smoke-comparison.json"), comparison)
        formal_reference = _write_json(root/"formal-start.json",
            {"status": "SMOKE_AND_CAPACITY_CLOSED", "freeze_reference": freeze_reference,
             "smoke_comparisons": comparisons, "completed": list(report["completed"]),
             "formal_order": list(CELLS), "code_change_after_this_point_requires_stop": True})
        report.update(formal_started=True, formal_start_reference=formal_reference)
        for cell in freeze["formal_order"]:
            execute(cell, "control30")
        validate_frozen_source(freeze["source"])
        report["status"] = "PASS"
        report["elapsed_seconds"] = time.monotonic()-started
        _write_json(root/"campaign-result.json", report)
        return report
    except BaseException as exc:
        report.update(status="STOP_NO_RETRY", elapsed_seconds=time.monotonic()-started,
                      failure={"type": type(exc).__name__, "message": str(exc)})
        _write_json(root/"campaign-result.json", report)
        raise
