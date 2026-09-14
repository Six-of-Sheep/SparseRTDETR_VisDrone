"""Synthetic contract tests for matched resolution evidence and capacity admission."""
from copy import deepcopy
from dataclasses import asdict
import gzip
import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from sparse_rtdetr.baseline import training_v2b_control as control
from sparse_rtdetr.baseline import training_v2b_resolution as resolution
from sparse_rtdetr.baseline.training_v2b import V2BConfig
from sparse_rtdetr.baseline.training_v2b_campaign import file_reference
from sparse_rtdetr.baseline.training_v2b_evidence import canonical_sha256


def _receipt(binding="1" * 64, augmented="2" * 64):
    samples = [{
        "index": i, "epoch": 1, "order_position": i, "image_id": i + 1,
        "stable_image_id": f"{i:064x}", "relative_path": f"train/images/{i}.jpg",
        "source_path": f"/authorized/train_core/images/{i}.jpg",
        "image_sha256": "3" * 64, "augmentation_seed": 2**62 + i,
        "binding_sha256": binding,
    } for i in range(16)]
    value = {"epoch": 1, "logical_batch_index": 0, "binding_sha256": binding,
             "samples": samples, "augmented_sha256": augmented}
    return {**value, "receipt_sha256": canonical_sha256(value)}


def _rehash(receipt):
    receipt["receipt_sha256"] = canonical_sha256(
        {k: v for k, v in receipt.items() if k != "receipt_sha256"})
    return receipt


def test_matched_source_preserves_large_integer_seed_but_allows_resize_bytes():
    old, new = _receipt(), _receipt("4" * 64, "5" * 64)
    resolution.verify_matched_input(old, new)
    assert new["samples"][0]["augmentation_seed"] > 2**53


@pytest.mark.parametrize("field,value", [
    ("augmentation_seed", 2**62 + 1), ("augmentation_seed", float(2**62)),
    ("image_sha256", "7" * 64), ("source_path", "/different/image.jpg"),
    ("stable_image_id", "8" * 64), ("order_position", 5),
])
def test_matched_source_rejects_any_scientific_identity_change(field, value):
    new = _receipt("4" * 64, "5" * 64)
    new["samples"][0][field] = value
    with pytest.raises(control.ControlWorkerError):
        resolution.verify_matched_input(_receipt(), _rehash(new))


@pytest.fixture
def capacity_plan():
    counts = [53] * 4869
    counts[0] = 462
    for i in range(1, 1 + 261886 - sum(counts)):
        counts[i] += 1
    dataset = SimpleNamespace(
        images=[{"id": i} for i in range(len(counts))],
        annotations={i: (None,) * count for i, count in enumerate(counts)},
    )
    class Dataset:
        images = dataset.images
        annotations = dataset.annotations
        def __len__(self):
            return len(self.images)
    loader = SimpleNamespace(config=SimpleNamespace(input_size=896, logical_batch_size=16, seed=0),
                             dataset=Dataset())
    return resolution.make_capacity_plan(loader)


def test_capacity_envelope_covers_both_microbatch_positions(capacity_plan):
    plan = capacity_plan
    resolution.validate_capacity_plan(plan)
    bounds = plan["raw_bounds"]
    counts = plan["synthetic_envelope"]["target_counts"]
    assert len(counts) == 16
    for micro in (counts[:8], counts[8:]):
        assert sum(micro) == bounds["max_per_physical_microbatch"]
        assert max(micro) == bounds["max_per_image"] == 462
    assert set(resolution._FIXED_STRESS).issubset(map(tuple, plan["stress_positions"]))


@pytest.mark.parametrize("field,value", [
    ("input_size", 960), ("physical_batch_size", 4), ("accumulation_steps", 4), ("seed", 1),
])
def test_capacity_plan_refuses_design_drift(capacity_plan, field, value):
    capacity_plan[field] = value
    with pytest.raises(control.ControlWorkerError):
        resolution.validate_capacity_plan(capacity_plan)


def test_capacity_margin_cannot_be_relaxed(capacity_plan):
    capacity_plan["resource_gate"]["minimum_conservative_free_mib"] = 1024
    with pytest.raises(control.ControlWorkerError):
        resolution.validate_capacity_plan(capacity_plan)


def _reference(path, sha="a" * 64, size=1):
    return {"path": str(path), "sha256": sha, "size_bytes": size,
            "sha256_scope": "complete_file_bytes"}


def _contract(tmp_path, plan):
    root = tmp_path / "checkout"
    root.mkdir(exist_ok=True)
    def ref(name, sha="a" * 64, size=1):
        return _reference(root / name, sha, size)
    return {
        "schema_version": 1, "kind": "v2b_896_control_worker", "campaign_id": "synthetic-896",
        "stage": "smoke", "arm": "B", "run_id": "synthetic-896-smoke-b",
        "repo_root": str(root), "output_dir": str(root / "artifacts/training/smoke"),
        "config": asdict(V2BConfig(input_size=896, physical_batch_size=8, accumulation_steps=2,
                                   sampling_backend="deterministic_gather")),
        "num_workers": 2, "prefetch_factor": 2, "torch_version": str(control.torch.__version__),
        "code_files": {"source.py": ref("source.py")},
        "train_core": {
            "annotation": ref("train_core_coco.json", control.TRAIN_ANNOTATION_SHA256, 51148042),
            "manifest": ref("train_core_manifest.json", control.TRAIN_MANIFEST_SHA256, 2955809),
            "image_root": str(root / "train_core/images"),
            "expected_samples": 4869, "expected_batches_per_epoch": 304,
        },
        "pretrained": ref("ResNet18_vd_pretrained_from_paddle.pth", control.PRETRAINED_SHA256, 44878642),
        "development_binding": {
            "annotation": ref("development_coco.json"), "manifest": ref("development_manifest.json"),
            "image_root": str(root / "development/images"), "policy": {"input_size": [896, 896]},
        },
        "epoch_order_sha256": control.frozen_epoch_orders(), "policy_bundle": {"synthetic": True, "setter_mode": "external_admin_acknowledged",
                          "authorization_reference": {"sha256": resolution.RESOLUTION_AUTHORIZATION_SHA256}},
        "expected_gpu_uuid": "GPU-1faee6f0-1da7-4ede-2475-67a5a00274a8",
        "paired_reference": None, "replay_source": None,
        "matched_control_reference": ref("control-B.complete.json", resolution.MATCHED_COMPLETION_SHA256),
        "capacity_plan": plan,
    }


def test_896_worker_contract_requires_explicit_kind_and_historical_control(tmp_path, capacity_plan):
    contract = _contract(tmp_path, capacity_plan)
    assert control.validate_worker_contract(contract, verify_files=False) == contract
    old = deepcopy(contract)
    old["kind"] = "v2b_640_control_worker"
    with pytest.raises(control.ControlWorkerError):
        control.validate_worker_contract(old, verify_files=False)
    old = deepcopy(contract)
    old["paired_reference"] = _reference(tmp_path / "worker-result.json")
    with pytest.raises(control.ControlWorkerError):
        control.validate_worker_contract(old, verify_files=False)


@pytest.mark.parametrize("field,value", [("input_size", 640), ("input_size", 960),
                                        ("physical_batch_size", 16), ("accumulation_steps", 1)])
def test_896_worker_rejects_silent_fallback(tmp_path, capacity_plan, field, value):
    contract = _contract(tmp_path, capacity_plan)
    contract["config"][field] = value
    with pytest.raises(control.ControlWorkerError):
        control.validate_worker_contract(contract, verify_files=False)


def _initialization():
    return {"seed": 0, "seed_initialized_before_model": True,
            "parameters": {"inventory": [{"name": "parameter", "sha256": "1" * 64}]},
            "pretrained": {"sha256": "2" * 64}, "sampling": {"backend": "deterministic_gather"},
            "model_state": {"inventory": [
                {"name": "bn.running_mean", "sha256": "3" * 64},
                {"name": "decoder.anchors", "shape": [1, 8400, 4], "sha256": "4" * 64},
                {"name": "decoder.valid_mask", "shape": [1, 8400, 1], "sha256": "5" * 64},
            ]}}


def test_only_resolution_geometry_is_excluded_from_cross_resolution_initial_state():
    old = _initialization()
    new = deepcopy(old)
    new["model_state"]["inventory"][1]["shape"] = [1, 16464, 4]
    new["model_state"]["inventory"][1]["sha256"] = "6" * 64
    assert resolution.compare_matched_initialization({"initialization": old}, new)["status"] == "PASS"
    new["model_state"]["inventory"][0]["sha256"] = "7" * 64
    with pytest.raises(control.ControlWorkerError):
        resolution.compare_matched_initialization({"initialization": old}, new)


def _native_completion(tmp_path, plan, free=4096., tamper=False):
    entries, previous = [], None
    for i in range(3):
        record = {"kind": "health", "started_ns": (i + 1) * 10**9,
                  "gpu": {"uuid": "synthetic-gpu", "memory_free_mib": free, "memory_used_mib": 16000.}}
        row = {"sequence": i, "previous_sha256": previous, "record": record}
        row["sha256"] = canonical_sha256(row)
        previous = row["sha256"]
        entries.append(row)
    if tamper:
        entries[1]["record"]["gpu"]["memory_free_mib"] = 8000.
    raw = b"".join((json.dumps(row) + "\n").encode() for row in entries)
    sample = tmp_path / "monitor-samples.jsonl.gz"
    sample.write_bytes(gzip.compress(raw, mtime=0))
    monitor = {"status": "PASS", "worker_exited": True, "sampled_clock_compliance": True,
               "gpu_uuid": "synthetic-gpu", "samples": file_reference(sample),
               "sample_hash_chain": {"records": 3, "last_sha256": previous}}
    output = tmp_path / "capacity"
    native = output / "native-hardware"
    native.mkdir(parents=True)
    monitor.update(run_id="capacity-example", run_binding_sha256="a" * 64, policy_sha256="b" * 64,
                   owner={"pid": 100, "start_ticks": 20}, monitor_pid=101,
                   guardian_identity={"pid": 101, "start_ticks": 21})
    final = native / "monitor-final.json"
    final.write_text(json.dumps(monitor))
    def write(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return file_reference(path)
    config = {"input_size": 896, "physical_batch_size": 8, "accumulation_steps": 2}
    contract = {"kind": "v2b_896_control_worker", "stage": "capacity", "arm": "B",
                "campaign_id": "example", "run_id": "capacity-example", "capacity_plan": plan,
                "config": config, "expected_gpu_uuid": "synthetic-gpu", "output_dir": str(output)}
    contract_ref = write("contract.json", contract)
    binding_ref = write("binding.json", {"run_id": "capacity-example", "binding_sha256": "a" * 64,
                                         "config": config})
    continuation = {k: monitor[k] for k in ("run_id", "run_binding_sha256", "policy_sha256",
                    "owner", "gpu_uuid", "monitor_pid", "guardian_identity")}
    continuation["monitor_final_report"] = str(final)
    result = {"status": "PASS", "stage": "capacity", "arm": "B", "run_id": "capacity-example",
              "campaign_id": "example", "worker_pid": 100, "capacity_acceptance": {"status": "PASS"},
              "contract_reference": contract_ref, "binding_reference": binding_ref,
              "monitor_reference": continuation}
    result_ref = write("worker-result.json", result)
    launch_ref = write("launch.json", {"pid": 100, "owner": monitor["owner"], "contract": contract_ref})
    return {"monitor": {**monitor, "final_report_reference": file_reference(final)},
            "result_reference": result_ref, "launch_reference": launch_ref}


def test_capacity_native_margin_requires_authenticated_complete_stream(tmp_path, capacity_plan):
    result = resolution.audit_capacity_native_memory(_native_completion(tmp_path, capacity_plan))
    assert result["status"] == "PASS" and result["sampled_minimum_free_mib"] == 4096.


@pytest.mark.parametrize("free,tamper", [(3071., False), (4096., True)])
def test_capacity_native_margin_or_chain_failure_stops(tmp_path, capacity_plan, free, tamper):
    with pytest.raises(control.ControlWorkerError):
        resolution.audit_capacity_native_memory(_native_completion(tmp_path, capacity_plan, free, tamper))


def test_capacity_allocator_margin_is_checked_without_fallback(capacity_plan):
    memory = {"conservative_free_at_reserved_peak_bytes": 3072 * 1024**2,
              "device_free_bytes": 3072 * 1024**2}
    resolution._check_capacity_memory(memory, capacity_plan)
    memory["conservative_free_at_reserved_peak_bytes"] -= 1
    with pytest.raises(control.ControlWorkerError):
        resolution._check_capacity_memory(memory, capacity_plan)


@pytest.mark.parametrize("setter", ["direct", "sudo_n"])
def test_896_worker_never_accepts_clock_setters(tmp_path, capacity_plan, setter):
    contract = _contract(tmp_path, capacity_plan)
    contract["policy_bundle"]["setter_mode"] = setter
    with pytest.raises(control.ControlWorkerError):
        control.validate_worker_contract(contract, verify_files=False)


def test_capacity_audit_rejects_self_consistent_same_gpu_monitor_from_other_worker(tmp_path, capacity_plan):
    completion = _native_completion(tmp_path, capacity_plan)
    final = Path(completion["monitor"]["final_report_reference"]["path"])
    different = json.loads(final.read_text())
    different["owner"] = {"pid": 200, "start_ticks": 40}
    different["run_id"] = "other-run"
    final.write_text(json.dumps(different))
    completion["monitor"] = {**different, "final_report_reference": file_reference(final)}
    with pytest.raises(control.ControlWorkerError):
        resolution.audit_capacity_native_memory(completion)


def test_896_worker_rejects_old_or_unbound_authorization(tmp_path, capacity_plan):
    contract = _contract(tmp_path, capacity_plan)
    contract["policy_bundle"]["authorization_reference"]["sha256"] = "0" * 64
    with pytest.raises(control.ControlWorkerError):
        control.validate_worker_contract(contract, verify_files=False)
