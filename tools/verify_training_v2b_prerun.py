"""Bounded CPU train_core preflight; optionally collect passive native hardware.

There is no training/evaluation/GPU-compute entry point in this tool. Each worker
setting reads only the explicitly bounded common train_core logical batches.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import time

# Set the process-start contract before any NumPy / torch import. A library
# cannot repair an already initialized incompatible MKL/OpenMP process.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["PYTHONNOUSERSITE"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from sparse_rtdetr.baseline.training_v2b import (
    V2BConfig, build_v2b_components, logical_batch_indices,
)
from sparse_rtdetr.baseline.training_v2b_data import (
    TrainCoreDataConfig, build_train_core_loader, validate_worker_environment,
)
from sparse_rtdetr.baseline.training_v2b_evidence import (
    canonical_sha256, collect_native_cpu_observation, file_reference,
    strict_json_loads, validate_run_binding, write_exclusive_json,
)
from sparse_rtdetr.baseline.training_v2b_hardware import (
    HardwarePolicy, collect_native_hardware_probe,
)
from sparse_rtdetr.baseline.training_v2b_runtime import (
    V2BTrainingSession, build_train_core_run_binding,
)


MODEL_CONFIG = ROOT / "configs/baseline/rtdetrv2_r18_visdrone_baseline_v2b_engineering.json"
RUNTIME_CONFIG = ROOT / "configs/baseline/rtdetrv2_r18_visdrone_baseline_v2b_runtime.json"


def _no_cuda(*args, **kwargs):
    raise RuntimeError("CUDA initialization is forbidden in the CPU pre-run verifier")


def _validate_policy(model_policy, runtime_policy):
    """Validate the exact comparison and coverage before reading input bytes."""
    if type(model_policy) is not dict or type(runtime_policy) is not dict:
        raise ValueError("pre-run policies must be objects")
    if model_policy.get("scope") != "cpu_synthetic_engineering":
        raise ValueError("wrong model construction policy scope")
    model_readiness = model_policy.get("readiness")
    expected_model_readiness = {
        "cpu_verification_evidence_required": True, "gpu_training_authorized": False,
        "real_dataset_access_authorized": False, "confirmatory_test_access_authorized": False,
        "high_resolution_experiment_authorized": False, "scientific_certified": False,
    }
    if canonical_sha256(model_readiness) != canonical_sha256(expected_model_readiness):
        raise ValueError("model policy readiness differs from CPU-only construction scope")
    configuration = model_policy.get("config")
    if type(configuration) is not dict or set(configuration) != set(V2BConfig.__dataclass_fields__):
        raise ValueError("model construction policy must explicitly declare all configuration fields")
    base = V2BConfig(**configuration)
    if base != V2BConfig():
        raise ValueError("pre-run requires the declared common 640/seed0/logical16 model")
    if runtime_policy.get("current_scope") != "CPU_tests_and_bounded_train_core_loader_only":
        raise ValueError("wrong pre-run scope")
    readiness = runtime_policy.get("readiness")
    ready_keys = {"formal_training_authorized", "GPU_smoke_authorized",
                  "high_resolution_896_960_authorized", "confirmatory_test_access_authorized",
                  "scientific_certified"}
    if (type(readiness) is not dict or set(readiness) != ready_keys
            or any(value is not False for value in readiness.values())):
        raise ValueError("pre-run policy cannot authorize training or certify science")
    data = runtime_policy.get("data")
    expected_data = {"role": "train_core", "logical_batch_size": 16,
                     "workers_to_verify": [0, 2, 4], "multiprocessing_context": "spawn",
                     "persistent_workers": False, "pin_memory": False, "MKL_THREADING_LAYER": "GNU"}
    if type(data) is not dict or any(
        canonical_sha256(data.get(key)) != canonical_sha256(value)
        for key, value in expected_data.items()
    ):
        raise ValueError("pre-run requires the exact train_core worker0/2/4 startup and data policy")
    smoke = runtime_policy.get("next_gpu_stage")
    expected_arms = [
        {"id": "640x16x1", "input_size": 640, "physical_batch_size": 16, "accumulation_steps": 1},
        {"id": "640x8x2", "input_size": 640, "physical_batch_size": 8, "accumulation_steps": 2},
    ]
    if type(smoke) is not dict or smoke.get("authorized") is not False:
        raise ValueError("GPU smoke is not authorized in this tool")
    if canonical_sha256(smoke.get("arms")) != canonical_sha256(expected_arms):
        raise ValueError("pre-run requires exactly the named 640x16x1 and 640x8x2 pair")
    if (type(smoke.get("logical_windows_per_arm")) is not int or smoke["logical_windows_per_arm"] != 4
            or type(smoke.get("checkpoint_after_logical_window")) is not int
            or smoke["checkpoint_after_logical_window"] != 2
            or smoke.get("replay_remaining_windows_in_fresh_runtime") is not True):
        raise ValueError("short smoke must plan four windows, checkpoint two and fresh replay")
    control = runtime_policy.get("scientific_control_after_smoke")
    if (type(control) is not dict or control.get("authorized") is not False
            or control.get("early_checkpoint_epoch") != 10
            or control.get("primary_comparison_endpoint_epoch") != 30
            or control.get("epoch_10_alone_certifies_batch_equivalence") is not False):
        raise ValueError("scientific control must remain unlaunched with epoch10 early and epoch30 primary")
    return base


def _pretrained_reference(path, authority):
    """Reject protected paths and wrong authority names before the first read."""
    candidate = Path(path).absolute()
    expected_name = Path(authority["weight_relative_path"]).name
    forbidden = {"confirmatory", "test", "development", "val", "visdrone2019-det-test-dev",
                 "visdrone2019-det-test-challenge", "visdrone2019-det-val"}
    if (candidate.name != expected_name or any(
            part.casefold() in forbidden or part.casefold().startswith("confirmatory_")
            for part in candidate.parts)):
        raise ValueError("pretrained path is not the allowed authority")
    resolved = candidate.resolve()
    if (resolved.name != expected_name or any(
            part.casefold() in forbidden or part.casefold().startswith("confirmatory_")
            for part in resolved.parts)):
        raise ValueError("resolved pretrained path crosses a forbidden role or authority name")
    reference = file_reference(resolved)
    if (reference["sha256"] != authority["weight_sha256"]
            or reference["size_bytes"] != authority["weight_size_bytes"]):
        raise ValueError("pretrained authority file differs")
    return reference


def _git_identity():
    result = {}
    for key, arguments in (("head", ["rev-parse", "HEAD"]),
                           ("head_tree", ["rev-parse", "HEAD^{tree}"]),
                           ("status_porcelain", ["status", "--porcelain=v1"])):
        captured = subprocess.run(["git", *arguments], cwd=ROOT, check=False,
                                  capture_output=True, text=True, timeout=15)
        result[key] = captured.stdout.strip() if captured.returncode == 0 else None
    result["source_binding_is_authoritative"] = True
    return result


def verify(args):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", args.run_id):
        raise ValueError("run_id must be a short safe lowercase identifier")
    if not 1 <= args.batches <= 2:
        raise ValueError("loader-only preflight is capped at two common logical batches")
    if torch.cuda.is_initialized():
        raise RuntimeError("pre-run verification must start without a CUDA context")
    torch.cuda._lazy_init = _no_cuda
    torch.set_num_threads(2)
    started = time.monotonic()
    model_policy = strict_json_loads(MODEL_CONFIG.read_bytes())
    runtime_policy = strict_json_loads(RUNTIME_CONFIG.read_bytes())
    base = _validate_policy(model_policy, runtime_policy)
    artifact_root = ROOT / "artifacts"
    if artifact_root.is_symlink():
        raise ValueError("artifact root cannot be a symlink")
    artifact_root.mkdir(exist_ok=True)
    output = artifact_root / ("v2b_prerun_" + args.run_id)
    output.mkdir(exist_ok=False)
    authority = model_policy["pretrained_authority"]
    pretrained = _pretrained_reference(args.pretrained_path, authority)
    source_paths = {str(path.relative_to(ROOT)): path for path in
                    (Path(__file__).resolve(), MODEL_CONFIG, RUNTIME_CONFIG)}
    data_paths = {
        "annotation_file": args.annotation_file, "annotation_sha256": args.annotation_sha256,
        "manifest_file": args.manifest_file, "manifest_sha256": args.manifest_sha256,
        "image_root": args.image_root, "repo_root": ROOT,
    }
    arms = []
    components_by_arm = []
    common_loader = None
    for definition in runtime_policy["next_gpu_stage"]["arms"]:
        configuration = {**model_policy["config"], **{key: definition[key] for key in
                         ("input_size", "physical_batch_size", "accumulation_steps")}}
        components = build_v2b_components(
            V2BConfig(**configuration), repo_root=ROOT,
            pretrained_path=pretrained["path"], pretrained_sha256=authority["weight_sha256"],
        )
        loader = build_train_core_loader(TrainCoreDataConfig(
            seed=base.seed, input_size=base.input_size,
            logical_batch_size=base.logical_batch_size, num_workers=0,
        ), **data_paths)
        binding = build_train_core_run_binding(
            components, loader, run_id=args.run_id + "-" + definition["id"],
            repo_root=ROOT, additional_code_paths=source_paths,
        )
        V2BTrainingSession(components, loader, binding)
        reference = write_exclusive_json(output / (definition["id"] + "-binding.json"), binding)
        initialization = write_exclusive_json(output / (definition["id"] + "-initialization.json"),
                                               components.initialization)
        observation = collect_native_cpu_observation(binding)
        environment = write_exclusive_json(output / (definition["id"] + "-cpu.json"), observation.as_dict())
        if common_loader is not None and loader.binding != common_loader.binding:
            raise AssertionError("paired physical batch arms changed common loader semantics")
        common_loader = loader
        arms.append({"arm": definition, "binding": reference, "initialization": initialization,
                     "native_cpu_observation": environment,
                     "engine": components.engine.state_dict(), "loader": loader.state_dict(),
                     "runtime": components.runtime.identity,
                     "model_parameters": sum(p.numel() for p in components.model.parameters()),
                     "batchnorm_modules": len(components.batchnorm_inventory)})
        components_by_arm.append(components)
    first, second = components_by_arm
    for key in ("parameters", "model_state", "pretrained"):
        if first.initialization[key] != second.initialization[key]:
            raise AssertionError("paired actual initialization differs: " + key)
    references, expected_batches = [], None
    for workers in runtime_policy["data"]["workers_to_verify"]:
        loader = build_train_core_loader(TrainCoreDataConfig(
            seed=base.seed, input_size=base.input_size,
            logical_batch_size=base.logical_batch_size, num_workers=workers,
        ), **data_paths)
        before = loader.state_dict()
        batches = list(loader.preview_batches(args.batches, epoch=1))
        if len(batches) != args.batches or loader.state_dict() != before or loader.has_active_iterator:
            raise AssertionError("bounded loader preview advanced state or left a worker iterator open")
        if loader.binding != common_loader.binding:
            raise AssertionError("worker count changed semantic data identity")
        if expected_batches is None:
            expected_batches = batches
        for observed, expected in zip(batches, expected_batches):
            if observed.evidence() != expected.evidence() or not torch.equal(observed.images, expected.images):
                raise AssertionError("worker setting changed actual augmented inputs")
            for actual_target, expected_target in zip(observed.targets, expected.targets):
                if actual_target.keys() != expected_target.keys():
                    raise AssertionError("worker setting changed target keys")
                for key in actual_target:
                    if not torch.equal(actual_target[key], expected_target[key]):
                        raise AssertionError("worker setting changed target tensor: " + key)
        payload = {"workers": workers, "environment": validate_worker_environment(loader.config),
                   "loader_binding_sha256": loader.binding_sha256, "cursor_before": before,
                   "cursor_after": loader.state_dict(), "batches": [batch.evidence() for batch in batches],
                   "targets_per_image": [[len(target["labels"]) for target in batch.targets] for batch in batches]}
        references.append(write_exclusive_json(output / f"loader-workers{workers}.json", payload))
    plan = {
        "seed": base.seed, "logical_batch_size": base.logical_batch_size,
        "dataset_size": len(common_loader.dataset), "full_batches_per_epoch": common_loader.batches_per_epoch,
        "dropped_samples_per_epoch": len(common_loader.dataset) % base.logical_batch_size,
        "epoch_order_sha256": {str(epoch): canonical_sha256([list(row) for row in logical_batch_indices(
            len(common_loader.dataset), seed=base.seed, epoch=epoch, logical_batch_size=base.logical_batch_size,
        )]) for epoch in range(1, 31)},
        "early_checkpoint_epoch": 10, "primary_comparison_endpoint_epoch": 30,
        "experiment_started": False,
    }
    plan_ref = write_exclusive_json(output / "common-order-plan.json", plan)
    hardware_ref, hardware_binding_ref, hardware_gate = None, None, None
    if args.passive_gpu_uuid is not None:
        requested = build_train_core_run_binding(
            first, common_loader, run_id=args.run_id + "-passive-hardware-request",
            repo_root=ROOT, additional_code_paths=source_paths,
            requested_device="cuda:0", cuda_gpu_uuid=args.passive_gpu_uuid,
        )
        hardware_binding_ref = write_exclusive_json(output / "passive-hardware-binding.json", requested)
        probe = collect_native_hardware_probe(requested, policy=HardwarePolicy(expected_gpu_uuid=args.passive_gpu_uuid))
        hardware_ref = write_exclusive_json(output / "passive-hardware.json", probe.as_dict())
        hardware_gate = probe.as_dict()["gate"]
    for arm in arms:
        validate_run_binding(strict_json_loads(Path(arm["binding"]["path"]).read_bytes()))
    if torch.cuda.is_initialized() or any(c.engine.optimizer_updates for c in components_by_arm):
        raise AssertionError("pre-run verifier executed an unauthorized device or optimizer update")
    report = {
        "schema_version": 1, "status": "CPU_PRERUN_PASS", "run_id": args.run_id,
        "source_git_observation": _git_identity(), "runtime_policy": file_reference(RUNTIME_CONFIG),
        "pretrained": pretrained, "arms": arms, "loader_verifications": references,
        "common_initial_state_equal": True, "actual_augmented_inputs_equal_across_workers": True,
        "common_sample_order_plan": plan_ref,
        "real_train_core_images_read": args.batches * base.logical_batch_size,
        "data_passes": len(runtime_policy["data"]["workers_to_verify"]),
        "model_forward_executed": False, "optimizer_updates": 0,
        "cuda_initialized": False, "gpu_compute_executed": False, "formal_training_executed": False,
        "development_confirmatory_test_data_accessed": False,
        "native_passive_hardware": hardware_ref, "hardware_requested_binding": hardware_binding_ref,
        "hardware_gate": hardware_gate, "GPU_admission_reusable": False,
        "gpu_ready": False, "scientific_certified": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    report["report_sha256"] = canonical_sha256(report)
    reference = write_exclusive_json(output / "verification.json", report)
    reread = strict_json_loads(Path(reference["path"]).read_bytes())
    if reread != report:
        raise AssertionError("pre-run report readback differs")
    print(reference["path"])
    print("CPU_PRERUN_PASS; GPU_READY=false; OPTIMIZER_UPDATES=0; CUDA_INITIALIZED=false")
    if hardware_gate is not None:
        print("NATIVE_HARDWARE_GATE=" + hardware_gate["status"])
    return reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-id", "pretrained-path", "annotation-file", "annotation-sha256",
                 "manifest-file", "manifest-sha256", "image-root"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--passive-gpu-uuid", default=None,
                        help="Explicit native NVSMI/sysfs/journal observation only; never CUDA initialization")
    verify(parser.parse_args())


if __name__ == "__main__":
    main()
