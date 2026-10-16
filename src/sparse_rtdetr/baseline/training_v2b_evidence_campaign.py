"""Frozen development-only 2x2 resolution evidence campaign.

Preparation is CPU-only. Each of the four independent inference processes obtains
fresh native admission and is watched by the same pidfd guardian supervisor as the
training controls. This module never sets clocks, trains, retries or selects a
checkpoint. Historical checkpoints and evidence are read-only inputs.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from .training_v2b_campaign import (
    CampaignError, GuardianSupervisor, _canonical, _data_path, _json_reference,
    _safe_id, _worker_environment, _write_json, file_reference, read_reference,
    source_snapshot, validate_frozen_source,
)

KIND = "v2b_development_resolution_evidence_campaign"
AUTHORIZATION_SHA256 = "0ea553eef36857f9686fa8bad3ad2b708af9f22a83f8f194b4405157a6a3b055"
CELL_SIZES = {
    "t640_e640": (640, 640),
    "t640_e896": (640, 896),
    "t896_e640": (896, 640),
    "t896_e896": (896, 896),
}
CHECKPOINT_AUTHORITIES = {
    "640": {"sha256": "6ab0ae0a05817ee88f1ea083ca7338763f06f14756d8aade45c763e1cea63ccb",
            "size_bytes": 323271617},
    "896": {"sha256": "104ed411619e0906be81b22ae949d6926ac3aafe0d712f4160f5bf22690c126f",
            "size_bytes": 323548737},
}
FORBIDDEN = [
    "training_in_this_campaign", "checkpoint_selection", "confirmatory", "test",
    "960", "120_epoch_run", "trainable_sparse", "clock_setter", "clock_reset",
    "historical_overwrite", "automatic_retry",
]
_KEYS = {
    "schema_version", "kind", "campaign_id", "output_root", "source",
    "authorization_reference", "foundation_reference", "checkpoints",
    "development_bindings", "official_gt_bindings", "policy_bundle",
    "expected_gpu_uuid", "sequence", "cache_policy", "evaluator_evidence_reference",
    "diagnostic_policy", "replication_seeds", "training_started",
    "on_any_failure", "forbidden",
}


def _same(observed: Any, expected: Any, label: str) -> None:
    if _canonical(observed) != _canonical(expected):
        raise CampaignError(label + " differs from the frozen evidence campaign")


def _output_path(repo_root: str | Path, output_root: str | Path) -> Path:
    root = Path(repo_root).resolve(strict=True)
    output = Path(output_root).absolute()
    allowed = root / "artifacts" / "evidence"
    if (output == allowed or not output.is_relative_to(allowed)
            or output.resolve() != output or output.is_symlink()):
        raise CampaignError("evidence output must be a new canonical artifacts/evidence directory")
    return output


def _external_policy(policy: dict, authorization: dict) -> None:
    if (type(policy) is not dict or policy.get("setter_mode") != "external_admin_acknowledged"
            or type(authorization) is not dict
            or authorization.get("sha256") != AUTHORIZATION_SHA256):
        raise CampaignError("evidence requires its explicit authorization and external clock receipt")
    _same(policy.get("authorization_reference"), authorization, "authorization binding")
    _same([policy.get("clock_min_mhz"), policy.get("clock_max_mhz")], [1500, 1500], "clock ceiling")
    _same(policy.get("settings_after_exit"), "keep_1500_1500", "clock exit policy")
    read_reference(authorization)
    read_reference(policy.get("external_clock_receipt"))
    from .training_v2b_admission import _validate_policy
    _validate_policy(policy, "development_cross_eval", 1800)


def cross_worker_contract(campaign: dict, cell_key: str) -> dict:
    if cell_key not in CELL_SIZES:
        raise CampaignError("only the declared four resolution cells are permitted")
    train_size, eval_size = CELL_SIZES[cell_key]
    output = Path(campaign["output_root"]) / "execution" / cell_key
    return copy.deepcopy({
        "schema_version": 1, "kind": "v2b_development_cross_eval_worker",
        "run_id": campaign["campaign_id"] + "-" + cell_key,
        "campaign_id": campaign["campaign_id"], "cell_key": cell_key,
        "repo_root": campaign["source"]["repo_root"], "output_dir": str(output),
        "source_training_size": train_size, "evaluation_size": eval_size,
        "checkpoint": campaign["checkpoints"][str(train_size)],
        "development_binding": campaign["development_bindings"][str(eval_size)],
        "official_gt_binding": campaign["official_gt_bindings"][str(eval_size)],
        "code_files": campaign["source"]["code_files"],
        "policy_bundle": campaign["policy_bundle"],
        "expected_gpu_uuid": campaign["expected_gpu_uuid"],
        "authorization_reference": campaign["authorization_reference"],
        "cache_policy": campaign["cache_policy"],
        "evaluator_evidence_reference": campaign["evaluator_evidence_reference"],
    })



def validate_uniform_development(campaign: dict) -> None:
    """Both sizes must represent the same bytes, images, order and primary GT."""
    left, right = (campaign["development_bindings"][str(size)] for size in (640, 896))
    invariant = lambda value: {key: item for key, item in value.items()
                               if key not in {"policy", "binding_sha256"}}
    _same(invariant(left), invariant(right), "cross-cell development identity")
    policy = copy.deepcopy(left["policy"])
    policy["input_size"] = [896, 896]
    resize = [item for item in policy["transform"] if item.get("type") == "Resize"]
    if len(resize) != 1 or resize[0].get("size") != [640, 640]:
        raise CampaignError("development has no unique frozen source resize")
    resize[0]["size"] = [896, 896]
    _same(policy, right["policy"], "development policy changes only evaluation size")
    _same(campaign["official_gt_bindings"]["640"], campaign["official_gt_bindings"]["896"],
          "cross-cell complete primary GT")


def validate_evidence_campaign(campaign: dict, *, verify_files: bool = True) -> None:
    if type(campaign) is not dict or set(campaign) != _KEYS:
        raise CampaignError("evidence campaign schema keys differ")
    _same([campaign["schema_version"], campaign["kind"]], [1, KIND], "campaign schema")
    _safe_id(campaign["campaign_id"])
    root = Path(campaign["source"]["repo_root"]).resolve(strict=True)
    output = _output_path(root, campaign["output_root"])
    _same(str(output), campaign["output_root"], "output path")
    _same(campaign["sequence"], list(CELL_SIZES), "four-cell order")
    _same(campaign["replication_seeds"], [1, 2], "predeclared paired seeds")
    _same(campaign["training_started"], False, "evaluation-only scope")
    _same(campaign["on_any_failure"], "STOP_NO_RETRY", "failure policy")
    _same(campaign["forbidden"], FORBIDDEN, "forbidden operations")
    if campaign["cache_policy"] != "ema_geometry_replay_9120_v1":
        raise CampaignError("unknown common inference geometry policy")
    _external_policy(campaign["policy_bundle"], campaign["authorization_reference"])
    _same(campaign["expected_gpu_uuid"], campaign["policy_bundle"]["gpu_uuid"], "GPU identity")
    for field in ("checkpoints", "development_bindings", "official_gt_bindings"):
        if type(campaign[field]) is not dict or set(campaign[field]) != {"640", "896"}:
            raise CampaignError(field + " must contain exactly 640 and 896")
    for size, expected in CHECKPOINT_AUTHORITIES.items():
        reference = campaign["checkpoints"][size]
        if (type(reference) is not dict or reference.get("sha256") != expected["sha256"]
                or reference.get("size_bytes") != expected["size_bytes"]):
            raise CampaignError("epoch-30 checkpoint differs from audited seed-zero authority")
        _data_path(reference["path"], expected_name="checkpoint-epoch-030.pt")
    from .training_v2b_resolution_diagnostics import frozen_method
    _same(campaign["diagnostic_policy"], frozen_method(), "bootstrap/error/router definitions")
    validate_uniform_development(campaign)
    if verify_files:
        for ref in campaign["checkpoints"].values():
            read_reference(ref)
        foundation = _json_reference(campaign["foundation_reference"])
        # Seed declarations are repeated in the campaign to make selection visible.
        if foundation.get("predeclared_additional_seeds") != [1, 2]:
            raise CampaignError("original foundation does not predeclare seeds one and two")
        read_reference(campaign["evaluator_evidence_reference"])
        from .training_v2b_cross_eval import validate_worker_contract
        for cell in CELL_SIZES:
            validate_worker_contract(cross_worker_contract(campaign, cell), verify_files=True)


def make_evidence_campaign(*, repo_root: str | Path, campaign_id: str,
                           output_root: str | Path, checkpoints: dict,
                           development: dict, lineage_reference: dict,
                           policy_bundle: dict, authorization_reference: dict,
                           foundation_reference: dict, evaluator_evidence_reference: dict,
                           diagnostic_policy: dict, cache_policy: str,
                           raw_annotation_root: str | None = None) -> dict:
    """Prepare the immutable four-cell plan; never initialize CUDA or read train_core."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise CampaignError("evidence preparation requires empty CUDA_VISIBLE_DEVICES")
    from .training_v2b_development import build_development_binding
    from .training_v2b_official_gt import build_official_gt_binding

    root = Path(repo_root).resolve(strict=True)
    campaign_id = _safe_id(campaign_id)
    output = _output_path(root, output_root)
    if output.exists():
        raise CampaignError("evidence campaign output already exists")
    if type(development) is not dict or set(development) != {"annotation", "manifest", "image_root"}:
        raise CampaignError("only explicit development annotation, manifest and image root are accepted")
    for key, filename in (("annotation", "development_coco.json"), ("manifest", "development_manifest.json")):
        _data_path(development[key]["path"], expected_name=filename)
        read_reference(development[key])
    _data_path(development["image_root"])
    _data_path(lineage_reference["path"], expected_name="development_lineage.jsonl")
    read_reference(lineage_reference)
    _external_policy(policy_bundle, authorization_reference)
    dev_bindings, gt_bindings = {}, {}
    for size in (640, 896):
        binding = build_development_binding(
            repo_root=root, annotation_file=development["annotation"]["path"],
            annotation_sha256=development["annotation"]["sha256"],
            manifest_file=development["manifest"]["path"],
            manifest_sha256=development["manifest"]["sha256"],
            image_root=development["image_root"], input_size=size,
        )
        if binding["image_count"] != 548 or binding["annotation_count"] != 38759:
            raise CampaignError("development cardinality differs from the audited split")
        dev_bindings[str(size)] = binding
        gt_bindings[str(size)] = build_official_gt_binding(
            development_binding=binding, lineage_reference=lineage_reference,
            raw_annotation_root=raw_annotation_root,
        )
    source = source_snapshot(root)
    campaign = {
        "schema_version": 1, "kind": KIND, "campaign_id": campaign_id,
        "output_root": str(output), "source": source,
        "authorization_reference": copy.deepcopy(authorization_reference),
        "foundation_reference": copy.deepcopy(foundation_reference),
        "checkpoints": copy.deepcopy(checkpoints),
        "development_bindings": dev_bindings, "official_gt_bindings": gt_bindings,
        "policy_bundle": copy.deepcopy(policy_bundle),
        "expected_gpu_uuid": policy_bundle["gpu_uuid"], "sequence": list(CELL_SIZES),
        "cache_policy": cache_policy,
        "evaluator_evidence_reference": copy.deepcopy(evaluator_evidence_reference),
        "diagnostic_policy": copy.deepcopy(diagnostic_policy),
        "replication_seeds": [1, 2], "training_started": False,
        "on_any_failure": "STOP_NO_RETRY", "forbidden": list(FORBIDDEN),
    }
    validate_evidence_campaign(campaign)
    validate_frozen_source(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700, exist_ok=False)
    ref = _write_json(output / "campaign.json", campaign)
    return {"campaign": campaign, "campaign_reference": ref}


def _run_evaluation(campaign: dict, contract: dict, contract_ref: dict,
                    execution: Path) -> dict:
    """Watch a single new inference worker and authenticate its post-exit monitor."""
    from .training_v2b_admission import _validate_policy, wait_for_monitored_finish
    label = contract["cell_key"]
    root = Path(campaign["source"]["repo_root"])
    argv = [sys.executable, str(root / "tools/run_training_v2b_cross_eval.py"),
            "--contract", contract_ref["path"], "--contract-sha256", contract_ref["sha256"]]
    policy = _validate_policy(contract["policy_bundle"], "development_cross_eval", 1800)
    started = time.monotonic()
    with open(execution / (label + ".stdout.log"), "xb", buffering=0) as stdout, \
            open(execution / (label + ".stderr.log"), "xb", buffering=0) as stderr:
        proc = subprocess.Popen(argv, cwd=root, env=_worker_environment(campaign),
                                stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                start_new_session=True)
        supervisor = None
        try:
            supervisor = GuardianSupervisor(
                proc.pid, Path(contract["output_dir"]) / "native-hardware",
                run_id=contract["run_id"], gpu_uuid=contract["expected_gpu_uuid"],
                policy_sha256=policy["policy_sha256"],
            )
            launch_ref = _write_json(execution / (label + ".launch.json"), {
                "argv": argv, "pid": proc.pid, "owner": supervisor.owner,
                "contract": contract_ref, "monotonic_started": started, "cell_key": label,
            })
            while proc.poll() is None:
                if time.monotonic() > started + 2100:
                    raise CampaignError("evaluation worker exceeded its fixed total wall limit")
                supervisor.check()
                time.sleep(.05)
            result_path = Path(contract["output_dir"]) / "worker-result.json"
            if proc.returncode != 0 or not result_path.is_file():
                raise CampaignError(f"evaluation {label} failed (exit {proc.returncode}); no retry")
            result_ref = file_reference(result_path)
            result = _json_reference(result_ref)
            if (result.get("status") != "PASS" or result.get("contract_reference") != contract_ref
                    or result.get("run_id") != contract["run_id"]):
                raise CampaignError("evaluation result is not bound to this exact invocation")
            if result.get("worker_pid") != proc.pid:
                raise CampaignError("evaluation result PID differs from the owned launched worker")
            monitor = wait_for_monitored_finish(result["monitor_reference"])
            if monitor.get("status") != "PASS":
                raise CampaignError("evaluation monitor did not close after its worker exited")
            validate_frozen_source(campaign["source"])
            return {"result_reference": result_ref, "monitor": monitor,
                    "launch_reference": launch_ref, "elapsed_seconds": time.monotonic() - started}
        except BaseException as exc:
            if supervisor is not None:
                supervisor.stop_owned_worker()
            elif proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5.)
            cleanup = supervisor.close_failed_guardian() if supervisor is not None else None
            _write_json(execution / (label + ".supervisor-stop.json"), {
                "error_type": type(exc).__name__, "error": str(exc),
                "owner": supervisor.owner if supervisor is not None else {"pid": proc.pid},
                "only_owned_campaign_processes_signalled": True,
                "pidfd_supervision_established": supervisor is not None,
                "guardian_cleanup": cleanup, "no_retry": True, "monotonic_ns": time.monotonic_ns(),
            })
            raise
        finally:
            if supervisor is not None:
                supervisor.close()
            _write_json(execution / (label + ".exit.json"), {
                "returncode": proc.poll(), "pid": proc.pid, "contract": contract_ref,
                "stdout": file_reference(execution / (label + ".stdout.log")),
                "stderr": file_reference(execution / (label + ".stderr.log")),
            })


def run_evidence_campaign(campaign: dict, *, campaign_reference: dict) -> dict:
    """Run exactly the four frozen cells sequentially; no training transition."""
    _same(_json_reference(campaign_reference), campaign, "campaign bytes")
    validate_evidence_campaign(campaign)
    validate_frozen_source(campaign["source"])
    output = _output_path(campaign["source"]["repo_root"], campaign["output_root"])
    execution = output / "execution"
    execution.mkdir(mode=0o700, exist_ok=False)
    contracts = execution / "contracts"
    contracts.mkdir(mode=0o700, exist_ok=False)
    completed = {}
    result = {
        "schema_version": 1, "kind": KIND, "campaign_reference": campaign_reference,
        "campaign_id": campaign["campaign_id"], "status": "RUNNING", "completed": completed,
        "training_started": False, "checkpoint_selection": False, "automatic_retry": False,
    }
    _write_json(execution / "campaign-start.json", result)
    try:
        for cell in CELL_SIZES:
            _same(_json_reference(campaign_reference), campaign, "campaign bytes")
            validate_frozen_source(campaign["source"])
            contract = cross_worker_contract(campaign, cell)
            contract_ref = _write_json(contracts / (cell + ".json"), contract)
            completion = _run_evaluation(campaign, contract, contract_ref, execution)
            completed[cell] = {
                "contract_reference": contract_ref,
                "completion_reference": _write_json(execution / (cell + ".complete.json"), completion),
                "result_reference": completion["result_reference"],
            }
        _same(list(completed), list(CELL_SIZES), "complete four-cell ledger")
        validate_frozen_source(campaign["source"])
        result["status"] = "PASS"
        _write_json(execution / "campaign-result.json", result)
        return result
    except BaseException as exc:
        result.update(status="STOP_NO_RETRY",
                      failure={"type": type(exc).__name__, "message": str(exc)})
        _write_json(execution / "campaign-result.json", result)
        raise
