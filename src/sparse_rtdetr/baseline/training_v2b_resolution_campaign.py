"""One immutable 896 x 8 x 2 campaign against the authenticated completed 640 B.

Only the workers execute CUDA. Preparation reads approved metadata on CPU; the
controller reuses the existing native admission, worker environment and guardian.
There is no retry, alternate batch, clock setter, or checkpoint continuation path.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import os
from pathlib import Path
import time
from typing import Any

from .training_v2b_campaign import (
    CampaignError, _canonical, _data_path, _digest, _json_reference,
    _run_worker, _safe_id, _write_json, read_reference, source_snapshot,
    validate_frozen_source,
)


KIND = "v2b_896_matched_resolution_campaign"
_SEQUENCE = ["smoke-896", "replay-896", "capacity-896", "control-896"]
_SMOKE = {
    "logical_windows": 4, "checkpoint_after_window": 2,
    "replay_windows": 2, "development_preview_images": 4,
}
_CONTROL = {
    "epochs": 30, "fixed_evaluation_epochs": [10, 20, 30], "primary_endpoint": 30,
    "nominal_recipe_epochs": 120, "augmentation_stop_internal_epoch": 117,
    "checkpoint_each_completed_epoch": True,
}
_FORBIDDEN = [
    "other_resolutions", "confirmatory", "test", "sparse", "data_protocol_changes",
    "model_selection_certification", "batch_fallback", "clock_reset",
]
_KEYS = {
    "schema_version", "kind", "campaign_id", "output_root", "authorization_reference",
    "source", "torch_version", "common_config", "arms", "num_workers", "prefetch_factor",
    "train_core", "pretrained", "development_binding", "epoch_order_sha256",
    "policy_bundle", "expected_gpu_uuid", "matched_control_reference",
    "matched_control_result_reference", "capacity_plan", "smoke", "control",
    "sequence", "formal_freeze", "on_any_failure", "historical_control_read_only",
    "no_model_selection_certification", "forbidden",
}


def _resolution():
    from . import training_v2b_resolution
    return training_v2b_resolution


def _same(observed: Any, expected: Any, label: str) -> None:
    if _canonical(observed) != _canonical(expected):
        raise CampaignError(label + " differs from the fixed 896 campaign")


def _pass(value: Any, label: str) -> None:
    if type(value) is not dict or value.get("status") != "PASS":
        raise CampaignError(label + " did not pass; no later stage is permitted")


def _output_path(repo_root: str | Path, output_root: str | Path) -> Path:
    root = Path(repo_root).resolve(strict=True)
    output = Path(output_root).absolute()
    allowed = root / "artifacts" / "training"
    if (output == allowed or not output.is_relative_to(allowed)
            or output.resolve() != output or output.is_symlink()):
        raise CampaignError("896 output must be a new canonical directory under this checkout's artifacts/training")
    return output


def _external_policy(policy: dict, authorization_reference: dict) -> None:
    # Reject a setter mode before loading any evidence or starting a worker.
    if type(policy) is not dict or policy.get("setter_mode") != "external_admin_acknowledged":
        raise CampaignError("896 permits only external_admin_acknowledged; no clock setter is available")
    _same(authorization_reference.get("sha256"), _resolution().RESOLUTION_AUTHORIZATION_SHA256,
          "896 authorization SHA")
    _same(policy.get("authorization_reference"), authorization_reference, "authorization reference")
    _same([policy.get("clock_min_mhz"), policy.get("clock_max_mhz")], [1500, 1500], "clock setting")
    _same(policy.get("settings_after_exit"), "keep_1500_1500", "clock exit policy")
    read_reference(authorization_reference)
    read_reference(policy.get("external_clock_receipt"))
    from .training_v2b_admission import _validate_policy
    _validate_policy(policy, "paired_smoke", 600)
    _validate_policy(policy, "train_core_30epoch", 43200)


def resolution_worker_contract(campaign: dict, *, stage: str, output_dir: Path,
                               replay_source: dict | None = None) -> dict:
    if stage not in {"smoke", "smoke_replay", "capacity", "control30"}:
        raise CampaignError("unknown fixed 896 worker stage")
    if (stage == "smoke_replay") != (replay_source is not None):
        raise CampaignError("only 896 smoke replay requires its completed 896 smoke reference")
    suffix = {"smoke": "smoke", "smoke_replay": "smoke",
              "capacity": "capacity", "control30": "control"}[stage]
    return copy.deepcopy({
        "schema_version": 1, "kind": "v2b_896_control_worker",
        "campaign_id": campaign["campaign_id"], "stage": stage, "arm": "B",
        "run_id": campaign["campaign_id"] + "-" + suffix + "-896-b",
        "repo_root": campaign["source"]["repo_root"], "output_dir": str(output_dir),
        "config": campaign["common_config"], "num_workers": campaign["num_workers"],
        "prefetch_factor": campaign["prefetch_factor"], "torch_version": campaign["torch_version"],
        "code_files": campaign["source"]["code_files"], "train_core": campaign["train_core"],
        "pretrained": campaign["pretrained"], "development_binding": campaign["development_binding"],
        "epoch_order_sha256": campaign["epoch_order_sha256"], "policy_bundle": campaign["policy_bundle"],
        "expected_gpu_uuid": campaign["expected_gpu_uuid"], "paired_reference": None,
        "replay_source": replay_source, "matched_control_reference": campaign["matched_control_reference"],
        "capacity_plan": campaign["capacity_plan"],
    })


def _validate_campaign(campaign: dict) -> dict:
    from .training_v2b import V2BConfig
    import torch

    if type(campaign) is not dict or set(campaign) != _KEYS:
        raise CampaignError("896 campaign schema keys differ")
    _same([campaign["schema_version"], campaign["kind"]], [1, KIND], "campaign schema")
    _safe_id(campaign["campaign_id"])
    root = Path(campaign["source"]["repo_root"]).resolve(strict=True)
    output = _output_path(root, campaign["output_root"])
    _same(str(output), campaign["output_root"], "output path")
    _same(campaign["common_config"], asdict(V2BConfig(
        input_size=896, physical_batch_size=8, accumulation_steps=2,
        sampling_backend="deterministic_gather",
    )), "scientific configuration")
    _same(campaign["arms"], {"B": {"physical_batch_size": 8, "accumulation_steps": 2}}, "arm")
    _same([campaign["num_workers"], campaign["prefetch_factor"]], [2, 2], "loader topology")
    _same(campaign["torch_version"], str(torch.__version__), "PyTorch version")
    _same(campaign["smoke"], _SMOKE, "smoke plan")
    _same(campaign["control"], _CONTROL, "fixed thirty-epoch endpoint")
    _same(campaign["sequence"], _SEQUENCE, "stage order")
    _same(campaign["formal_freeze"], "after capacity admission, before control-896", "formal freeze")
    _same(campaign["on_any_failure"], "STOP_NO_RETRY", "failure policy")
    _same(campaign["historical_control_read_only"], True, "historical evidence policy")
    _same(campaign["no_model_selection_certification"], True, "scientific claim scope")
    _same(campaign["forbidden"], _FORBIDDEN, "forbidden operations")
    _external_policy(campaign["policy_bundle"], campaign["authorization_reference"])
    resolution = _resolution()
    matched = resolution.load_matched_control(campaign["matched_control_reference"])
    _same(campaign["matched_control_result_reference"], matched["result_reference"], "matched result reference")
    if root == Path(matched["contract"]["repo_root"]).resolve():
        raise CampaignError("896 requires an isolated checkout; the completed 640 checkout is read-only")
    resolution.validate_capacity_plan(campaign["capacity_plan"])
    template = resolution_worker_contract(campaign, stage="control30", output_dir=output / "execution/control-896")
    resolution.validate_matched_definition(template, matched)
    return matched


def make_resolution_campaign(*, repo_root: str | Path, campaign_id: str,
                             output_root: str | Path, matched_control_reference: dict,
                             policy_bundle: dict, authorization_reference: dict) -> dict:
    """Prepare once on CPU, deriving data and all non-resolution settings from B."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise CampaignError("896 CPU preparation requires empty CUDA_VISIBLE_DEVICES")
    from .training_v2b import V2BConfig
    from .training_v2b_data import TrainCoreDataConfig, build_train_core_loader
    from .training_v2b_development import build_development_binding

    root = Path(repo_root).resolve(strict=True)
    campaign_id = _safe_id(campaign_id)
    output = _output_path(root, output_root)
    if output.exists():
        raise CampaignError("896 campaign output already exists; history cannot be overwritten")
    _external_policy(policy_bundle, authorization_reference)
    resolution = _resolution()
    matched = resolution.load_matched_control(matched_control_reference)
    old = matched["contract"]
    if root == Path(old["repo_root"]).resolve():
        raise CampaignError("896 preparation cannot write in the completed 640 checkout")
    config = V2BConfig(**{**old["config"], "input_size": 896})
    train, dev = old["train_core"], old["development_binding"]
    for role, data in (("train_core", train), ("development", dev)):
        for key in ("annotation", "manifest"):
            suffix = "_coco.json" if key == "annotation" else "_manifest.json"
            _data_path(data[key]["path"], expected_name=role + suffix)
        _data_path(data["image_root"])
    _data_path(old["pretrained"]["path"], expected_name="ResNet18_vd_pretrained_from_paddle.pth")
    loader = build_train_core_loader(
        TrainCoreDataConfig(seed=config.seed, input_size=896, logical_batch_size=16,
                            num_workers=2, prefetch_factor=2),
        annotation_file=train["annotation"]["path"], annotation_sha256=train["annotation"]["sha256"],
        manifest_file=train["manifest"]["path"], manifest_sha256=train["manifest"]["sha256"],
        image_root=train["image_root"], repo_root=root,
    )
    if len(loader.dataset) != 4869 or loader.batches_per_epoch != 304:
        raise CampaignError("896 metadata does not describe the frozen full train_core epoch")
    capacity_plan = resolution.make_capacity_plan(loader)
    development = build_development_binding(
        repo_root=root, annotation_file=dev["annotation"]["path"],
        annotation_sha256=dev["annotation"]["sha256"], manifest_file=dev["manifest"]["path"],
        manifest_sha256=dev["manifest"]["sha256"], image_root=dev["image_root"], input_size=896,
    )
    if development["image_count"] != 548:
        raise CampaignError("896 development cardinality differs from the matched control")
    source = source_snapshot(root)
    value = {
        "schema_version": 1, "kind": KIND, "campaign_id": campaign_id, "output_root": str(output),
        "authorization_reference": copy.deepcopy(authorization_reference), "source": source,
        "torch_version": old["torch_version"], "common_config": asdict(config),
        "arms": {"B": {"physical_batch_size": 8, "accumulation_steps": 2}},
        "num_workers": 2, "prefetch_factor": 2, "train_core": copy.deepcopy(train),
        "pretrained": copy.deepcopy(old["pretrained"]), "development_binding": development,
        "epoch_order_sha256": copy.deepcopy(old["epoch_order_sha256"]),
        "policy_bundle": copy.deepcopy(policy_bundle), "expected_gpu_uuid": old["expected_gpu_uuid"],
        "matched_control_reference": copy.deepcopy(matched_control_reference),
        "matched_control_result_reference": copy.deepcopy(matched["result_reference"]),
        "capacity_plan": capacity_plan, "smoke": copy.deepcopy(_SMOKE), "control": copy.deepcopy(_CONTROL),
        "sequence": list(_SEQUENCE), "formal_freeze": "after capacity admission, before control-896",
        "on_any_failure": "STOP_NO_RETRY", "historical_control_read_only": True,
        "no_model_selection_certification": True, "forbidden": list(_FORBIDDEN),
    }
    _validate_campaign(value)
    validate_frozen_source(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700, exist_ok=False)
    reference = _write_json(output / "campaign.json", value)
    return {"campaign_reference": reference, "campaign": value}


def _ensure_frozen(campaign: dict, campaign_reference: dict) -> None:
    _same(_json_reference(campaign_reference), campaign, "campaign bytes or in-memory settings")
    validate_frozen_source(campaign["source"])


def _capacity_identity(campaign: dict, completion: dict, result: dict, execution: Path) -> dict:
    """Bind the capacity result and native monitor to this exact owned invocation."""
    expected = resolution_worker_contract(campaign, stage="capacity", output_dir=execution / "capacity-896")
    _same(result.get("contract_reference", {}).get("path"),
          str(execution / "contracts/capacity-896.json"), "capacity contract path")
    contract = _json_reference(result["contract_reference"])
    _same(contract, expected, "capacity worker contract")
    for name in ("run_id", "campaign_id", "stage", "arm"):
        _same(result.get(name), expected[name], "capacity result " + name)
    if type(result.get("worker_pid")) is not int or result["worker_pid"] <= 0:
        raise CampaignError("capacity result lacks its actual worker PID")
    _same(completion.get("launch_reference", {}).get("path"),
          str(execution / "capacity-B.launch.json"), "capacity launch path")
    launch = _json_reference(completion["launch_reference"])
    _same(launch.get("contract"), result["contract_reference"], "capacity launch contract")
    _same([launch.get("stage"), launch.get("arm"), launch.get("pid")],
          ["capacity", "B", result["worker_pid"]], "capacity launch identity")
    monitor, reference = completion["monitor"], result.get("monitor_reference")
    identity_fields = ("run_id", "run_binding_sha256", "policy_sha256", "monitor_pid",
                       "owner", "guardian_identity", "gpu_uuid")
    if (type(reference) is not dict or
            any(name not in monitor or name not in reference for name in identity_fields)):
        raise CampaignError("capacity result or final monitor lacks native identity fields")
    for name in identity_fields:
        _same(monitor[name], reference[name], "capacity monitor " + name)
    _same(monitor["run_id"], expected["run_id"], "capacity monitor run")
    _same(monitor["gpu_uuid"], campaign["expected_gpu_uuid"], "capacity monitor GPU")
    _same(launch.get("owner"), monitor["owner"], "capacity launch owner")
    _same(monitor["owner"].get("pid"), result["worker_pid"], "capacity monitor worker PID")
    final_path = str(execution / "capacity-896/native-hardware/monitor-final.json")
    _same(reference.get("monitor_final_report"), final_path, "capacity requested final monitor path")
    _same(monitor.get("final_report_reference", {}).get("path"), final_path, "capacity final monitor path")
    _same(result.get("binding_reference", {}).get("path"),
          str(execution / "capacity-896/run-binding.json"), "capacity run-binding path")
    binding = _json_reference(result["binding_reference"])
    _same(binding.get("run_id"), expected["run_id"], "capacity bound run")
    _same(binding.get("binding_sha256"), monitor["run_binding_sha256"], "capacity monitor run binding")
    return {
        "contract_reference": result["contract_reference"], "launch_reference": completion["launch_reference"],
        "binding_reference": result["binding_reference"], "run_id": expected["run_id"],
        "worker_pid": result["worker_pid"], "monitor_reference": reference,
        "result_launch_binding_and_final_monitor_match": True,
    }


def _capacity_admission(campaign: dict, completion: dict, execution: Path) -> dict:
    assessment = {
        "schema_version": 1, "scope": "896_capacity_admission",
        "capacity_completion": copy.deepcopy(completion),
        "capacity_plan_sha256": _digest(campaign["capacity_plan"]),
        "minimum_free_mib": 3072, "no_batch_or_resolution_fallback": True,
    }
    try:
        _pass(completion.get("monitor"), "capacity final hardware monitor")
        _same(completion.get("result_reference", {}).get("path"),
              str(execution / "capacity-896/worker-result.json"), "capacity result path")
        result = _json_reference(completion["result_reference"])
        _pass(result, "capacity worker")
        assessment["invocation_identity"] = _capacity_identity(campaign, completion, result, execution)
        assessment["worker_acceptance"] = result.get("capacity_acceptance")
        _pass(assessment["worker_acceptance"], "capacity worker acceptance")
        assessment["native_memory_audit"] = _resolution().audit_capacity_native_memory(completion)
        _pass(assessment["native_memory_audit"], "independent capacity native memory audit")
        assessment["status"] = "PASS"
    except BaseException as exc:
        assessment.update(status="STOP_NO_RETRY", error_type=type(exc).__name__, error=str(exc))
        _write_json(execution / "capacity-admission.json", assessment)
        raise
    return _write_json(execution / "capacity-admission.json", assessment)


def run_resolution_campaign(campaign_reference: dict) -> dict:
    """Run each fixed stage once; failed evidence never permits a later stage."""
    campaign = _json_reference(campaign_reference)
    _validate_campaign(campaign)
    _ensure_frozen(campaign, campaign_reference)
    execution = Path(campaign["output_root"]) / "execution"
    execution.mkdir(mode=0o700, exist_ok=False)
    (execution / "contracts").mkdir()
    finished = {}
    formal_started = False
    began = time.monotonic()
    try:
        for stage, key in (("smoke", "smoke-896"), ("smoke_replay", "replay-896")):
            _ensure_frozen(campaign, campaign_reference)
            replay = finished["smoke-896"]["result_reference"] if stage == "smoke_replay" else None
            contract = resolution_worker_contract(campaign, stage=stage, output_dir=execution / key,
                                                  replay_source=replay)
            contract_ref = _write_json(execution / "contracts" / (key + ".json"), contract)
            finished[key] = _run_worker(campaign, contract, contract_ref, execution)
            _write_json(execution / (key + ".complete.json"), finished[key])
            _ensure_frozen(campaign, campaign_reference)
        comparison = _resolution().compare_resolution_smoke(execution / "smoke-896", execution / "replay-896")
        comparison_ref = _write_json(execution / "smoke-replay-comparison.json", comparison)
        _pass(comparison, "896 smoke and fresh-process recovery comparison")
        _ensure_frozen(campaign, campaign_reference)
        capacity = resolution_worker_contract(campaign, stage="capacity", output_dir=execution / "capacity-896")
        capacity_ref = _write_json(execution / "contracts/capacity-896.json", capacity)
        finished["capacity-896"] = _run_worker(campaign, capacity, capacity_ref, execution)
        _write_json(execution / "capacity-896.complete.json", finished["capacity-896"])
        _ensure_frozen(campaign, campaign_reference)
        admission_ref = _capacity_admission(campaign, finished["capacity-896"], execution)
        _ensure_frozen(campaign, campaign_reference)
        formal = resolution_worker_contract(campaign, stage="control30", output_dir=execution / "control-896")
        freeze_ref = _write_json(execution / "formal-freeze.json", {
            "schema_version": 1, "campaign_reference": campaign_reference,
            "source": campaign["source"], "matched_control_reference": campaign["matched_control_reference"],
            "matched_control_result_reference": campaign["matched_control_result_reference"],
            "smoke_comparison": comparison_ref, "capacity_admission": admission_ref,
            "completed_pilot_workers": copy.deepcopy(finished), "formal_template": formal,
            "fixed_epochs": [10, 20, 30], "primary_endpoint": 30,
            "fresh_initialization_no_pilot_checkpoint_reuse": True,
            "source_or_scientific_configuration_change_requires_stop": True,
        })
        _ensure_frozen(campaign, campaign_reference)
        _same(_json_reference(freeze_ref)["formal_template"], formal, "persisted formal template")
        formal_ref = _write_json(execution / "contracts/control-896.json", formal)
        formal_started = True
        finished["control-896"] = _run_worker(campaign, formal, formal_ref, execution)
        _write_json(execution / "control-896.complete.json", finished["control-896"])
        _ensure_frozen(campaign, campaign_reference)
        result = {
            "schema_version": 1, "status": "PASS", "campaign_reference": campaign_reference,
            "formal_freeze": freeze_ref, "capacity_admission": admission_ref, "workers": finished,
            "matched_control_reference": campaign["matched_control_reference"],
            "matched_control_result_reference": campaign["matched_control_result_reference"],
            "elapsed_seconds": time.monotonic() - began, "formal_started": True,
            "primary_endpoint": 30, "fixed_evaluation_epochs": [10, 20, 30],
            "model_selection_certified": False, "other_resolutions_authorized": False,
        }
        _write_json(execution / "campaign-result.json", result)
        return result
    except BaseException as exc:
        result = {
            "schema_version": 1, "status": "STOP_NO_RETRY", "campaign_reference": campaign_reference,
            "workers": finished, "formal_started": formal_started,
            "elapsed_seconds": time.monotonic() - began,
            "error_type": type(exc).__name__, "error": str(exc),
            "continue_or_splice_comparison_allowed": False,
            "automatic_retry_or_batch_fallback": False,
        }
        _write_json(execution / "campaign-result.json", result)
        raise
