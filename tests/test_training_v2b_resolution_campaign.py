"""Synthetic CPU tests of 896 controller transitions; no native GPU or real data."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sparse_rtdetr.baseline import training_v2b_admission as admission
from sparse_rtdetr.baseline import training_v2b_data as data
from sparse_rtdetr.baseline import training_v2b_development as development
from sparse_rtdetr.baseline import training_v2b_resolution as resolution
from sparse_rtdetr.baseline import training_v2b_resolution_campaign as campaign
from sparse_rtdetr.baseline.training_v2b import V2BConfig
from sparse_rtdetr.baseline.training_v2b_campaign import (
    CampaignError, _digest, _json_reference, _write_json, file_reference, read_reference,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    def forbidden(*args, **kwargs):
        raise AssertionError("controller unit test attempted CUDA or tensor archive loading")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)


def _capacity_plan():
    return {
        "schema_version": 1, "kind": "896_fixed_order_capacity_plan",
        "input_size": 896, "physical_batch_size": 8, "accumulation_steps": 2,
        "logical_batch_size": 16, "seed": 0,
        "warmup_positions": [[1, index] for index in range(4)],
        "stress_positions": [[14, 82], [25, 21], [27, 128]],
        "raw_bounds": {
            "image_count": 4869, "annotation_count": 261886,
            "counts_in_loader_order_sha256": "1" * 64,
            "max_per_image": 100, "max_per_physical_microbatch": 200,
            "max_per_logical_batch": 320,
            "raw_microbatch_max_position": [14, 82], "raw_logical_max_position": [25, 21],
            "proof": "no_mosaic_or_copy; converter_crop_sanitize_only_remove_boxes",
        },
        "synthetic_envelope": {
            "target_counts": resolution._envelope_counts(100, 200),
            "repeat_windows": 2, "cpu_fixture_seed": 896, "scientific_samples": False,
            "purpose": "bound_DN_padding_and_GT_cost_in_both_physical_microbatch_positions",
        },
        "development": {
            "role": "development", "image_count": 548, "batch_size": 4,
            "autocast": False, "weights": "ema", "scientific_eligible": False,
        },
        "resource_gate": {
            "minimum_conservative_free_mib": 3072, "native_minimum_free_mib": 3072,
            "no_empty_cache_or_model_offload": True,
        },
    }


@pytest.fixture
def scene(tmp_path_factory, monkeypatch):
    # Deliberately synthetic directories avoid any production artifact path.
    base = tmp_path_factory.mktemp("resolution-controller")
    root, old_root, metadata = base / "checkout", base / "sealed_source", base / "metadata"
    for path in (root, old_root, metadata):
        path.mkdir()
    source_names = [
        "src/sparse_rtdetr/baseline/training_v2b_engine.py",
        "src/sparse_rtdetr/baseline/training_v2b_checkpoint.py",
        "src/sparse_rtdetr/baseline/training_v2b_runtime.py",
        "src/sparse_rtdetr/baseline/training_v2b_device.py",
        "src/sparse_rtdetr/baseline/training_v2b_deterministic_sampling.py",
        "src/sparse_rtdetr/baseline/primary_evaluator.py",
        "src/sparse_rtdetr/baseline/postprocessor.py",
        "src/sparse_rtdetr/data_protocol/evaluation.py",
        "src/sparse_rtdetr/data_protocol/protocol.py",
        "vendor/synthetic_source.py",
    ]
    code, old_code = {}, {}
    for name in source_names:
        for checkout, inventory in ((root, code), (old_root, old_code)):
            path = checkout / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# synthetic source identity: " + name + "\n")
            inventory[name] = file_reference(path)
    source = {
        "repo_root": str(root), "commit": "a" * 40, "tree": "b" * 40,
        "branch": "cpu-fixture", "code_files": code, "inventory_sha256": _digest(code),
    }
    for name in ("train_core_coco.json", "train_core_manifest.json",
                 "development_coco.json", "development_manifest.json"):
        (metadata / name).write_text('{"synthetic":true}\n')
    train_images, dev_images = base / "synthetic_train_images", base / "synthetic_development_images"
    train_images.mkdir()
    dev_images.mkdir()
    pretrained = base / "ResNet18_vd_pretrained_from_paddle.pth"
    pretrained.write_bytes(b"synthetic reference bytes, never a tensor archive")
    train = {
        "annotation": file_reference(metadata / "train_core_coco.json"),
        "manifest": file_reference(metadata / "train_core_manifest.json"),
        "image_root": str(train_images), "expected_samples": 4869, "expected_batches_per_epoch": 304,
    }
    dev = {
        "schema_version": 1, "role": "development", "protocol_id": "P3-VISDRONE-DATA-PROTOCOL-V2",
        "repo_root": str(old_root),
        "annotation": {k: v for k, v in file_reference(metadata / "development_coco.json").items()
                       if k != "sha256_scope"},
        "manifest": {k: v for k, v in file_reference(metadata / "development_manifest.json").items()
                     if k != "sha256_scope"},
        "image_root": str(dev_images), "image_count": 548, "annotation_count": 10,
        "image_order_sha256": "2" * 64, "image_manifest_sha256": "3" * 64,
        "policy": development._policy(), "source": {"synthetic": True},
    }
    dev["binding_sha256"] = _digest(dev)
    old = {
        "kind": "v2b_640_control_worker", "stage": "control30", "arm": "B",
        "repo_root": str(old_root),
        "config": asdict(V2BConfig(physical_batch_size=8, accumulation_steps=2,
                                  sampling_backend="deterministic_gather")),
        "num_workers": 2, "prefetch_factor": 2, "torch_version": str(torch.__version__),
        "code_files": old_code, "train_core": train, "pretrained": file_reference(pretrained),
        "development_binding": dev,
        "epoch_order_sha256": {str(epoch): _digest(["synthetic", epoch]) for epoch in range(1, 31)},
        "expected_gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000",
    }
    contract_ref = _write_json(base / "historical-contract.json", old)
    result_ref = _write_json(base / "historical-result.json", {
        "status": "PASS", "stage": "control30", "arm": "B", "receipt_count": 9120,
        "run_id": "synthetic-old-b", "worker_pid": 4242, "contract_reference": contract_ref,
    })
    completion_ref = _write_json(base / "control-B.complete.json", {
        "result_reference": result_ref,
        "monitor": {"status": "PASS", "run_id": "synthetic-old-b", "owner": {"pid": 4242}},
    })
    authorization, external = base / "authorization.txt", base / "external-clock.json"
    authorization.write_text("synthetic controller-test authorization; no real execution\n")
    external.write_text('{"synthetic_external_admin_receipt":true}\n')
    auth_ref = file_reference(authorization)
    policy = {
        "setter_mode": "external_admin_acknowledged", "authorization_reference": auth_ref,
        "external_clock_receipt": file_reference(external), "clock_min_mhz": 1500,
        "clock_max_mhz": 1500, "settings_after_exit": "keep_1500_1500",
    }
    state = SimpleNamespace(
        base=base, root=root, old_root=old_root, source=source, old=old,
        completion_ref=completion_ref, result_ref=result_ref, auth_ref=auth_ref, policy=policy,
        calls=[], events=[], metadata_calls=[], policy_calls=[], worker_hook=None, completion_hook=None,
        fail_stage=None, compare_status="PASS", native_status="PASS", native_error=False,
        worker_acceptance="PASS", capacity_monitor="PASS",
    )

    def matched(reference):
        assert reference == completion_ref
        complete = _json_reference(reference)
        result = _json_reference(complete["result_reference"])
        return {
            "completion": complete, "result": result,
            "contract": _json_reference(result["contract_reference"]),
            "result_reference": complete["result_reference"],
        }

    def source_snapshot(requested):
        assert Path(requested) == root
        return deepcopy(source)

    def frozen(expected):
        assert expected == source
        for name, reference in source["code_files"].items():
            if file_reference(root / name) != reference:
                raise CampaignError("synthetic source changed")

    def loader(config, **kwargs):
        state.metadata_calls.append(("train", config, kwargs))
        assert (config.seed, config.input_size, config.logical_batch_size,
                config.num_workers, config.prefetch_factor) == (0, 896, 16, 2, 2)
        assert kwargs["image_root"] == train["image_root"]
        return SimpleNamespace(dataset=range(4869), batches_per_epoch=304, config=config)

    def capacity_plan(bound_loader):
        assert bound_loader.config.input_size == 896
        return _capacity_plan()

    def development_binding(**kwargs):
        state.metadata_calls.append(("development", kwargs))
        assert kwargs["input_size"] == 896
        value = deepcopy(dev)
        value["repo_root"] = str(root)
        value["policy"] = development._policy(896)
        value.pop("binding_sha256")
        return {**value, "binding_sha256": _digest(value)}

    def policy_check(bundle, scope, deadline):
        assert bundle == policy
        state.policy_calls.append((scope, deadline))
        return {**deepcopy(bundle), "authorized_scope": scope, "workload_deadline_seconds": deadline}

    def worker(value, contract, contract_reference, execution):
        stage = contract["stage"]
        state.calls.append(deepcopy(contract))
        state.events.append(("worker", stage))
        if stage == state.fail_stage:
            raise CampaignError("synthetic worker failure at " + stage)
        output = Path(contract["output_dir"])
        output.mkdir()
        hardware = output / "native-hardware"
        hardware.mkdir()
        binding_sha = _digest(["synthetic-binding", contract["run_id"]])
        owner = {"pid": 100, "start_ticks": 7, "readable": True}
        identity = {
            "run_id": contract["run_id"], "run_binding_sha256": binding_sha,
            "policy_sha256": "7" * 64, "monitor_pid": 101,
            "owner": owner, "guardian_identity": {"pid": 101, "start_ticks": 8, "readable": True},
            "gpu_uuid": contract["expected_gpu_uuid"],
        }
        final = {**deepcopy(identity), "worker_exited": True, "sampled_clock_compliance": True,
                 "status": state.capacity_monitor if stage == "capacity" else "PASS"}
        final_reference = _write_json(hardware / "monitor-final.json", final)
        binding_reference = _write_json(output / "run-binding.json", {
            "run_id": contract["run_id"], "binding_sha256": binding_sha,
        })
        result = {
            "status": "PASS", "stage": stage, "arm": "B", "run_id": contract["run_id"],
            "campaign_id": contract["campaign_id"], "worker_pid": owner["pid"],
            "contract_reference": contract_reference, "binding_reference": binding_reference,
            "monitor_reference": {**deepcopy(identity), "monitor_final_report": final_reference["path"],
                                  "worker_must_exit": True},
        }
        if stage == "capacity":
            result["capacity_acceptance"] = {"status": state.worker_acceptance}
        ref = _write_json(output / "worker-result.json", result)
        launch = _write_json(execution / (stage + "-B.launch.json"), {
            "contract": contract_reference, "pid": owner["pid"], "owner": owner, "stage": stage, "arm": "B",
        })
        if state.worker_hook:
            state.worker_hook(value, contract)
        completion = {
            "result_reference": ref, "launch_reference": launch, "elapsed_seconds": 0.01,
            "monitor": {**final, "final_report_reference": final_reference},
        }
        if state.completion_hook:
            state.completion_hook(contract, completion)
        return completion

    def compare(smoke, replay):
        assert Path(smoke).name == "smoke-896" and Path(replay).name == "replay-896"
        state.events.append(("compare", "same-resolution"))
        return {"status": state.compare_status, "synthetic_cpu_transition_fixture": True}

    def native_audit(completion):
        assert _json_reference(completion["result_reference"])["stage"] == "capacity"
        state.events.append(("audit", "native-memory"))
        if state.native_error:
            raise CampaignError("synthetic native memory sampling gap")
        return {"status": state.native_status, "synthetic_cpu_transition_fixture": True,
                "minimum_native_free_mib": 4096 if state.native_status == "PASS" else 2048}

    monkeypatch.setattr(resolution, "RESOLUTION_AUTHORIZATION_SHA256", auth_ref["sha256"])
    monkeypatch.setattr(resolution, "load_matched_control", matched)
    monkeypatch.setattr(resolution, "make_capacity_plan", capacity_plan)
    monkeypatch.setattr(resolution, "compare_resolution_smoke", compare)
    monkeypatch.setattr(resolution, "audit_capacity_native_memory", native_audit)
    monkeypatch.setattr(campaign, "source_snapshot", source_snapshot)
    monkeypatch.setattr(campaign, "validate_frozen_source", frozen)
    monkeypatch.setattr(campaign, "_run_worker", worker)
    monkeypatch.setattr(data, "build_train_core_loader", loader)
    monkeypatch.setattr(development, "build_development_binding", development_binding)
    monkeypatch.setattr(admission, "_validate_policy", policy_check)

    def prepare():
        return campaign.make_resolution_campaign(
            repo_root=root, campaign_id="synthetic-896",
            output_root=root / "artifacts/training/synthetic-896",
            matched_control_reference=completion_ref, policy_bundle=policy,
            authorization_reference=auth_ref,
        )

    state.prepare = prepare
    return state


def _read_result(prepared):
    output = Path(prepared["campaign"]["output_root"]) / "execution"
    return output, json.loads((output / "campaign-result.json").read_bytes())


def test_prepare_changes_only_resolution_and_uses_metadata_only(scene):
    prepared = scene.prepare()
    value = prepared["campaign"]
    assert not scene.calls and not scene.events
    assert [call[0] for call in scene.metadata_calls] == ["train", "development"]
    differences = {key for key in value["common_config"] if value["common_config"][key] != scene.old["config"][key]}
    assert differences == {"input_size"}
    assert value["matched_control_reference"] == scene.completion_ref
    assert value["matched_control_result_reference"] == scene.result_ref
    assert value["epoch_order_sha256"] == scene.old["epoch_order_sha256"]
    assert value["capacity_plan"] == _capacity_plan()
    assert not (Path(value["output_root"]) / "execution").exists()
    assert read_reference(prepared["campaign_reference"])


def test_exact_sequence_uses_same_source_plan_and_fresh_formal_identity(scene):
    prepared = scene.prepare()
    result = campaign.run_resolution_campaign(prepared["campaign_reference"])
    assert result["status"] == "PASS" and result["primary_endpoint"] == 30
    assert result["fixed_evaluation_epochs"] == [10, 20, 30]
    assert scene.events == [
        ("worker", "smoke"), ("worker", "smoke_replay"), ("compare", "same-resolution"),
        ("worker", "capacity"), ("audit", "native-memory"), ("worker", "control30"),
    ]
    smoke, replay, capacity, formal = scene.calls
    assert smoke["run_id"] == replay["run_id"]
    assert len({smoke["run_id"], capacity["run_id"], formal["run_id"]}) == 3
    for contract in scene.calls:
        assert contract["kind"] == "v2b_896_control_worker" and contract["arm"] == "B"
        assert contract["config"] == prepared["campaign"]["common_config"]
        assert contract["code_files"] == scene.source["code_files"]
        assert contract["capacity_plan"] == _capacity_plan()
        assert contract["matched_control_reference"] == scene.completion_ref
        assert contract["paired_reference"] is None
    assert replay["replay_source"] == result["workers"]["smoke-896"]["result_reference"]
    assert all(contract["replay_source"] is None for contract in (smoke, capacity, formal))
    freeze = _json_reference(result["formal_freeze"])
    assessment = _json_reference(freeze["capacity_admission"])
    assert assessment["status"] == "PASS"
    assert assessment["worker_acceptance"]["status"] == assessment["native_memory_audit"]["status"] == "PASS"
    assert freeze["formal_template"] == formal
    assert freeze["fresh_initialization_no_pilot_checkpoint_reuse"] is True
    assert not result["model_selection_certified"]


@pytest.mark.parametrize("stage", ["smoke", "smoke_replay", "capacity", "control30"])
def test_each_worker_failure_prevents_all_later_stages(scene, stage):
    prepared = scene.prepare()
    scene.fail_stage = stage
    with pytest.raises(CampaignError, match="synthetic worker failure"):
        campaign.run_resolution_campaign(prepared["campaign_reference"])
    stages = ["smoke", "smoke_replay", "capacity", "control30"]
    assert [call["stage"] for call in scene.calls] == stages[:stages.index(stage) + 1]
    _, result = _read_result(prepared)
    assert result["status"] == "STOP_NO_RETRY"
    assert result["formal_started"] is (stage == "control30")
    assert result["continue_or_splice_comparison_allowed"] is False


def test_smoke_comparison_failure_prohibits_capacity_and_formal(scene):
    prepared = scene.prepare()
    scene.compare_status = "STOP_NO_RETRY"
    with pytest.raises(CampaignError, match="recovery comparison"):
        campaign.run_resolution_campaign(prepared["campaign_reference"])
    output, result = _read_result(prepared)
    assert [call["stage"] for call in scene.calls] == ["smoke", "smoke_replay"]
    assert result["formal_started"] is False
    assert not (output / "formal-freeze.json").exists()


@pytest.mark.parametrize("failure", ["worker_acceptance", "capacity_monitor", "native_status", "native_error"])
def test_capacity_requires_worker_final_monitor_and_independent_native_audit(scene, failure):
    prepared = scene.prepare()
    setattr(scene, failure, True if failure == "native_error" else "STOP_NO_RETRY")
    with pytest.raises(CampaignError):
        campaign.run_resolution_campaign(prepared["campaign_reference"])
    output, result = _read_result(prepared)
    assert [call["stage"] for call in scene.calls] == ["smoke", "smoke_replay", "capacity"]
    assessment = json.loads((output / "capacity-admission.json").read_bytes())
    assert assessment["status"] == "STOP_NO_RETRY" and assessment["minimum_free_mib"] == 3072
    assert result["formal_started"] is False
    assert not (output / "formal-freeze.json").exists()


@pytest.mark.parametrize("stage", ["smoke", "capacity", "control30"])
def test_source_change_after_any_stage_stops_frozen_campaign(scene, stage):
    prepared = scene.prepare()

    def mutate(_, contract):
        if contract["stage"] == stage:
            name = next(iter(scene.source["code_files"]))
            (scene.root / name).write_text("changed synthetic source\n")

    scene.worker_hook = mutate
    with pytest.raises(CampaignError, match="source changed"):
        campaign.run_resolution_campaign(prepared["campaign_reference"])
    _, result = _read_result(prepared)
    assert result["status"] == "STOP_NO_RETRY"
    assert scene.calls[-1]["stage"] == stage
    assert result["formal_started"] is (stage == "control30")


def test_in_memory_settings_cannot_change_after_capacity(scene):
    prepared = scene.prepare()

    def mutate(value, contract):
        if contract["stage"] == "capacity":
            value["common_config"]["learning_rate"] *= 2

    scene.worker_hook = mutate
    with pytest.raises(CampaignError, match="in-memory settings"):
        campaign.run_resolution_campaign(prepared["campaign_reference"])
    output, result = _read_result(prepared)
    assert result["formal_started"] is False
    assert not (output / "formal-freeze.json").exists()


@pytest.mark.parametrize("field,key,value", [
    ("common_config", "input_size", 960),
    ("common_config", "physical_batch_size", 4),
    ("common_config", "accumulation_steps", 4),
    ("common_config", "sampling_backend", "native"),
    ("control", "epochs", 10),
    ("control", "fixed_evaluation_epochs", [10]),
    ("policy_bundle", "setter_mode", "direct"),
    ("policy_bundle", "clock_max_mhz", 1600),
])
def test_repacked_campaign_cannot_expand_scope_before_any_worker(scene, field, key, value):
    prepared = scene.prepare()
    changed = deepcopy(prepared["campaign"])
    changed[field][key] = value
    reference = _write_json(Path(changed["output_root"]) / "mutated-campaign.json", changed)
    with pytest.raises(CampaignError):
        campaign.run_resolution_campaign(reference)
    assert not scene.calls
    assert not (Path(changed["output_root"]) / "execution").exists()


def test_repeated_invocation_cannot_overwrite_or_resume(scene):
    prepared = scene.prepare()
    campaign.run_resolution_campaign(prepared["campaign_reference"])
    output, _ = _read_result(prepared)
    before = file_reference(output / "campaign-result.json")
    calls = len(scene.calls)
    with pytest.raises(FileExistsError):
        campaign.run_resolution_campaign(prepared["campaign_reference"])
    assert len(scene.calls) == calls
    assert file_reference(output / "campaign-result.json") == before


def test_bound_historical_completion_mutation_is_detected_before_launch(scene):
    prepared = scene.prepare()
    Path(scene.completion_ref["path"]).write_text('{"changed":true}\n')
    with pytest.raises(CampaignError, match="bound file changed"):
        campaign.run_resolution_campaign(prepared["campaign_reference"])
    assert not scene.calls


def test_prepare_refuses_old_checkout_without_writing(scene):
    before = sorted(str(path.relative_to(scene.old_root)) for path in scene.old_root.rglob("*"))
    with pytest.raises(CampaignError, match="completed 640 checkout"):
        campaign.make_resolution_campaign(
            repo_root=scene.old_root, campaign_id="synthetic-896",
            output_root=scene.old_root / "artifacts/training/synthetic-896",
            matched_control_reference=scene.completion_ref, policy_bundle=scene.policy,
            authorization_reference=scene.auth_ref,
        )
    assert sorted(str(path.relative_to(scene.old_root)) for path in scene.old_root.rglob("*")) == before
    assert not scene.metadata_calls and not scene.calls


def _cli(monkeypatch, scene):
    specification = importlib.util.spec_from_file_location(
        "resolution_cli_cpu_fixture", ROOT / "tools/run_training_v2b_resolution_campaign.py",
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", scene.root)
    return module


def _prepare_args(scene):
    return [
        "prepare", "--campaign-id", "synthetic-896",
        "--output-root", str(scene.root / "artifacts/training/synthetic-896"),
        "--matched-completion", scene.completion_ref["path"],
        "--matched-completion-sha256", scene.completion_ref["sha256"],
        "--policy-evidence-dir", str(scene.base / "synthetic_policy"),
        "--authorization", scene.auth_ref["path"], "--authorization-sha256", scene.auth_ref["sha256"],
        "--external-clock-receipt", scene.policy["external_clock_receipt"]["path"],
        "--external-clock-receipt-sha256", scene.policy["external_clock_receipt"]["sha256"],
    ]


@pytest.mark.parametrize("mode", ["direct", "sudo_n"])
def test_cli_has_no_clock_setter_mode(scene, monkeypatch, mode):
    cli = _cli(monkeypatch, scene)
    with pytest.raises(SystemExit) as caught:
        cli.main([*_prepare_args(scene), "--setter-mode", mode])
    assert caught.value.code == 2
    assert not scene.metadata_calls and not scene.calls


@pytest.mark.parametrize("name", ["authorization", "matched-completion", "external-clock-receipt"])
def test_cli_checks_every_requested_complete_sha_before_policy_or_prepare(scene, monkeypatch, name):
    cli = _cli(monkeypatch, scene)
    arguments = _prepare_args(scene)
    arguments[arguments.index("--" + name + "-sha256") + 1] = "0" * 64
    monkeypatch.setattr(admission, "build_policy_bundle", lambda *a, **k: pytest.fail("policy built before SHA gate"))
    with pytest.raises(CampaignError, match="complete-file SHA256"):
        cli.main(arguments)
    assert not scene.metadata_calls and not scene.calls


def test_cli_cpu_prepare_derives_inputs_from_completed_b(scene, monkeypatch, capsys):
    cli = _cli(monkeypatch, scene)

    def policy(_, **kwargs):
        assert kwargs["setter_mode"] == "external_admin_acknowledged"
        assert kwargs["authorization_reference"] == scene.auth_ref
        assert kwargs["external_clock_receipt"] == scene.policy["external_clock_receipt"]
        return deepcopy(scene.policy)

    monkeypatch.setattr(admission, "build_policy_bundle", policy)
    assert cli.main(_prepare_args(scene)) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "PREPARED_NO_GPU_EXECUTED"
    assert not scene.calls


def test_cli_nonempty_cuda_visibility_cannot_prepare(scene, monkeypatch):
    cli = _cli(monkeypatch, scene)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(CampaignError, match="empty CUDA_VISIBLE_DEVICES"):
        cli.main(_prepare_args(scene))
    assert not scene.metadata_calls and not scene.calls


def test_cli_run_requires_requested_campaign_sha(scene, monkeypatch):
    prepared = scene.prepare()
    cli = _cli(monkeypatch, scene)
    with pytest.raises(CampaignError, match="campaign bytes differ"):
        cli.main(["run", "--campaign", prepared["campaign_reference"]["path"],
                  "--campaign-sha256", "0" * 64])
    assert not scene.calls


def test_real_896_authorization_constant_is_specific():
    assert resolution.RESOLUTION_AUTHORIZATION_SHA256 == "ece816c4bfc93b62c403a00986a738f853b86e91a05953531ab93c0998538819"


def test_old_authorization_is_refused_before_policy_validation(scene, monkeypatch):
    old_authorization = scene.base / "old-640-authorization.txt"
    old_authorization.write_text("synthetic obsolete 640-only authorization\n")
    obsolete = file_reference(old_authorization)
    policy = deepcopy(scene.policy)
    policy["authorization_reference"] = obsolete
    monkeypatch.setattr(admission, "_validate_policy", lambda *a: pytest.fail("old authorization entered policy"))
    with pytest.raises(CampaignError, match="896 authorization SHA"):
        campaign._external_policy(policy, obsolete)


def test_other_run_monitor_with_same_gpu_cannot_admit_current_capacity(scene):
    prepared = scene.prepare()

    def substitute(contract, completion):
        if contract["stage"] == "capacity":
            foreign = deepcopy(completion["monitor"])
            foreign.pop("final_report_reference")
            foreign["run_id"] = "another-896-run"
            foreign["owner"] = {"pid": 200, "start_ticks": 9, "readable": True}
            reference = _write_json(scene.base / "another-run-monitor.json", foreign)
            completion["monitor"] = {**foreign, "final_report_reference": reference}

    scene.completion_hook = substitute
    with pytest.raises(CampaignError, match="capacity monitor"):
        campaign.run_resolution_campaign(prepared["campaign_reference"])
    output, result = _read_result(prepared)
    assert result["formal_started"] is False
    assert ("audit", "native-memory") not in scene.events
    assert json.loads((output / "capacity-admission.json").read_bytes())["status"] == "STOP_NO_RETRY"


def test_capacity_admission_binds_the_callers_exact_campaign_template(scene):
    prepared = scene.prepare()
    value = prepared["campaign"]
    execution = Path(value["output_root"]) / "execution"
    execution.mkdir()
    (execution / "contracts").mkdir()
    foreign = campaign.resolution_worker_contract(value, stage="capacity",
                                                  output_dir=execution / "capacity-896")
    foreign["campaign_id"] = "another-campaign"
    foreign["run_id"] = "another-campaign-capacity-896-b"
    reference = _write_json(execution / "contracts/capacity-896.json", foreign)
    completion = campaign._run_worker(value, foreign, reference, execution)
    with pytest.raises(CampaignError, match="capacity worker contract"):
        campaign._capacity_admission(value, completion, execution)
    assert ("audit", "native-memory") not in scene.events
    assert json.loads((execution / "capacity-admission.json").read_bytes())["status"] == "STOP_NO_RETRY"
