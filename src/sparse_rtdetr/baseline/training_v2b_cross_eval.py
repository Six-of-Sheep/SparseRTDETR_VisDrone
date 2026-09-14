"""One independent, supervised epoch-30 EMA development cross-evaluation process.

No optimizer, training engine, training loader, checkpoint RNG restore or training
configuration spoofing is used. Learned weights and every non-geometry buffer
remain byte-identical. All four cells use the same reproduced EMA cache history.
"""

from __future__ import annotations

from dataclasses import asdict
import copy
import hashlib
import io
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from . import primary_evaluator
from . import training_v2b_development as development
from . import training_v2b_evidence as evidence
from .training_v2b import V2BConfig, _build_vendor_objects, _configure_model_sampling
from .training_v2b_checkpoint import inspect_checkpoint
from .training_v2b_device import prepare_runtime, place_module, validate_module_placement
from .training_v2b_geometry import snapshot_model_geometry
from .training_v2b_official_gt import (
    build_official_gt_binding, load_official_ground_truth, read_regular_reference, with_predictions,
)
from .training_v2b_primary_cache import prepare_primary_cache
from .postprocessor import VisDronePostProcessor


class CrossEvaluationError(ValueError):
    """A changed frozen evaluation contract, state, prediction budget or input."""


CACHE_POLICY = "ema_geometry_replay_9120_v1"
CELL_KEYS = ("t640_e640", "t640_e896", "t896_e640", "t896_e896")
_EPOCH30_IDENTITIES = {
    640: ("6ab0ae0a05817ee88f1ea083ca7338763f06f14756d8aade45c763e1cea63ccb", 323271617),
    896: ("104ed411619e0906be81b22ae949d6926ac3aafe0d712f4160f5bf22690c126f", 323548737),
}
_GEOMETRY_BUFFERS = frozenset(("decoder.anchors", "decoder.valid_mask"))
_CONTRACT_KEYS = {
    "schema_version", "kind", "run_id", "campaign_id", "cell_key", "repo_root", "output_dir",
    "source_training_size", "evaluation_size", "checkpoint", "development_binding",
    "official_gt_binding", "code_files", "policy_bundle", "expected_gpu_uuid",
    "authorization_reference", "cache_policy", "evaluator_evidence_reference",
}


def _require(condition, message):
    if not condition:
        raise CrossEvaluationError(message)


def _size(value):
    _require(type(value) is int and value in (640, 896), "only exact integer 640/896 is allowed")
    return value


def _reference_json(reference):
    return evidence.strict_json_loads(read_regular_reference(reference))


def policy():
    return {
        "scope": "development_cross_eval", "checkpoint_epoch": 30,
        "weight_kind": "ema", "ema_updates": 9120,
        "evaluation_batch_size": 4, "forward_dtype": "float32", "autocast": False,
        "raw_queries_per_image": 300, "raw_classes": 10,
        "flattened_predictions_per_image": 300, "nms": False, "score_threshold": None,
        "prediction_clipping": False, "cache_policy": CACHE_POLICY,
        "source_physical_batch": 8, "source_accumulation": 2,
        "training_performed": False, "checkpoint_selection_performed": False,
        "primary_entry": "sparse_rtdetr.baseline.primary_evaluator.evaluate_primary_v1",
        "primary_ground_truth": "complete_development_raw_lineage",
        "coco_ground_truth": "unchanged_converted_development_formal_gt",
        "hardware_scope_deadline_seconds": 1800,
    }


def validate_worker_contract(contract: dict, *, verify_files: bool = True) -> dict:
    _require(type(contract) is dict and set(contract) == _CONTRACT_KEYS,
             "cross-evaluation worker contract schema differs")
    checked = copy.deepcopy(contract)
    _require(checked["schema_version"] == 1 and type(checked["schema_version"]) is int
             and checked["kind"] == "v2b_development_cross_eval_worker", "worker kind differs")
    train_size, eval_size = _size(checked["source_training_size"]), _size(checked["evaluation_size"])
    _require(checked["cell_key"] == f"t{train_size}_e{eval_size}"
             and checked["cell_key"] in CELL_KEYS, "cell does not match its train/eval geometry")
    _require(checked["cache_policy"] == CACHE_POLICY, "unapproved geometry cache policy")
    checkpoint = checked["checkpoint"]
    _require(type(checkpoint) is dict and (checkpoint.get("sha256"), checkpoint.get("size_bytes"))
             == _EPOCH30_IDENTITIES[train_size], "checkpoint does not name the frozen epoch-30 source")
    checkpoint_path = Path(checkpoint.get("path", ""))
    _require(checkpoint_path.is_absolute() and ".." not in checkpoint_path.parts
             and checkpoint_path.name == "checkpoint-epoch-030.pt"
             and not any(part.casefold() in {"confirmatory", "test"}
                         or part.casefold().startswith("confirmatory_") for part in checkpoint_path.parts),
             "checkpoint path crosses an unapproved artifact role")
    for key in ("run_id", "campaign_id"):
        _require(type(checked[key]) is str and checked[key] and "/" not in checked[key],
                 "worker run/campaign identity is invalid")
    root, output = Path(checked["repo_root"]), Path(checked["output_dir"])
    _require(root.is_absolute() and output.is_absolute() and ".." not in root.parts
             and ".." not in output.parts and output.is_relative_to(root / "artifacts"),
             "cross-evaluation paths must remain in the explicit engineering artifacts root")
    _require(type(checked["code_files"]) is dict and checked["code_files"],
             "a frozen source inventory is required")
    bundle = checked["policy_bundle"]
    _require(type(bundle) is dict and bundle.get("authorization_reference") == checked["authorization_reference"]
             and bundle.get("gpu_uuid") == checked["expected_gpu_uuid"]
             and bundle.get("setter_mode") == "external_admin_acknowledged"
             and bundle.get("authorized_scope_limits_seconds", {}).get("development_cross_eval") == 1800,
             "cross-evaluation requires its exact external-clock evaluation authorization")
    development.validate_development_binding(checked["development_binding"], verify_files=False)
    _require(development._policy_input_size(checked["development_binding"]["policy"]) == eval_size
             and checked["development_binding"]["repo_root"] == str(root),
             "actual development resize or repository differs from evaluation size")
    gt = checked["official_gt_binding"]
    _require(type(gt) is dict and gt.get("role") == "development"
             and gt.get("annotation") == checked["development_binding"]["annotation"]
             and gt.get("manifest") == checked["development_binding"]["manifest"],
             "official GT and development inputs are not the same membership")
    if verify_files:
        _require(Path(__file__).resolve().is_relative_to(root.resolve(strict=True)),
                 "cross-evaluation module is imported from another checkout")
        for name, reference in checked["code_files"].items():
            _require(type(name) is str and Path(reference["path"]) == root / name
                     and ".." not in Path(name).parts and not Path(name).is_absolute(),
                     "frozen code path differs")
            read_regular_reference(reference)
        for name in (
            "src/sparse_rtdetr/baseline/training_v2b_cross_eval.py",
            "src/sparse_rtdetr/baseline/training_v2b_official_gt.py",
            "src/sparse_rtdetr/baseline/training_v2b_primary_cache.py",
        ):
            _require(name in checked["code_files"], "cross-evaluation source missing from frozen inventory")
        development.validate_development_binding(checked["development_binding"])
        load_official_ground_truth(gt, checked["development_binding"])
        read_regular_reference(checked["authorization_reference"])
        _reference_json(checked["evaluator_evidence_reference"])
    return checked


def _state_identity(model, *, exclude_geometry=False):
    return evidence.initial_parameter_reference({
        key: tensor for key, tensor in model.state_dict().items()
        if not exclude_geometry or key not in _GEOMETRY_BUFFERS
    })


def replay_ema_anchors(raw: torch.Tensor, *, updates: int, decay: float, warmups: int):
    """Replay the frozen two separate in-place EMA operations on constant FP32 anchors."""
    _require(raw.device.type == "cpu" and raw.dtype == torch.float32,
             "EMA geometry reconstruction is CPU FP32 only")
    _require(type(updates) is int and updates >= 0
             and type(warmups) is int and warmups > 0
             and type(decay) in (int, float) and 0 <= decay < 1,
             "invalid frozen EMA history")
    result = raw.detach().clone()
    with torch.no_grad():
        for update in range(1, updates + 1):
            d = decay * (1 - math.exp(-update / warmups))
            result.mul_(d)
            result.add_((1 - d) * raw)
    return result


def adapt_evaluation_geometry(model, *, source_training_size: int, evaluation_size: int,
                              updates: int, decay: float, warmups: int,
                              expected_raw_anchors: torch.Tensor,
                              expected_raw_mask: torch.Tensor):
    """Change only declared geometry after proving source EMA-cache reproduction."""
    source_training_size, evaluation_size = _size(source_training_size), _size(evaluation_size)
    _require(all(value.device.type == "cpu" for value in model.state_dict().values()),
             "geometry adaptation must precede all CUDA placement")
    before = snapshot_model_geometry(model, expected_input_size=source_training_size)
    preserved = _state_identity(model, exclude_geometry=True)
    encoder, decoder = model.encoder, model.decoder
    raw, mask = decoder._generate_anchors()
    _require(torch.equal(raw, expected_raw_anchors) and torch.equal(mask, expected_raw_mask),
             "reconstructed source geometry differs from authenticated raw model buffers")
    replayed = replay_ema_anchors(raw, updates=updates, decay=decay, warmups=warmups)
    _require(torch.equal(replayed, decoder.anchors) and torch.equal(mask, decoder.valid_mask),
             "source EMA geometry replay is not byte-exact; no silent canonical fallback")
    source_replay = {
        "status": "PASS", "updates": updates, "decay": decay, "warmups": warmups,
        "source_raw_anchors_byte_exact": True, "source_ema_anchors_byte_exact": True,
        "source_ema_anchor_sha256": before["decoder"]["anchors"]["sha256"],
    }
    encoder.eval_spatial_size = [evaluation_size, evaluation_size]
    decoder.eval_spatial_size = [evaluation_size, evaluation_size]
    for index in encoder.use_encoder_idx:
        name = f"pos_embed{index}"
        _require(name not in encoder._buffers and name not in encoder._parameters,
                 "vendor position cache unexpectedly registered")
        stride = encoder.feat_strides[index]
        setattr(encoder, name, encoder.build_2d_sincos_position_embedding(
            evaluation_size // stride, evaluation_size // stride,
            encoder.hidden_dim, encoder.pe_temperature,
        ))
    target_raw, target_mask = decoder._generate_anchors()
    decoder.anchors = replay_ema_anchors(target_raw, updates=updates, decay=decay, warmups=warmups)
    decoder.valid_mask = target_mask
    after = snapshot_model_geometry(model, expected_input_size=evaluation_size)
    _require(_state_identity(model, exclude_geometry=True) == preserved,
             "geometry adaptation changed a learned parameter or non-geometry buffer")
    if source_training_size == evaluation_size:
        _require(after == before, "same-size cell no longer reproduces all stored EMA geometry bytes")
    return {
        "cache_policy": CACHE_POLICY, "source_replay": source_replay,
        "source_geometry": before, "evaluation_geometry": after,
        "mutation_allowlist": [
            "encoder.eval_spatial_size", "decoder.eval_spatial_size",
            *("encoder.pos_embed" + str(i) for i in encoder.use_encoder_idx),
            "decoder.anchors", "decoder.valid_mask",
        ],
        "learned_and_non_geometry_state_before": preserved,
        "learned_and_non_geometry_state_byte_identical": True,
        "batchnorm_state_byte_identical": True,
        "same_size_full_geometry_byte_identical": source_training_size == evaluation_size,
    }


def prepare_cross_eval_model(*, checkpoint: dict, source_training_size: int,
                             evaluation_size: int, repo_root: str | Path):
    """Authenticate epoch-30 CPU payload and strictly load EMA; never restore training RNG."""
    source_training_size, evaluation_size = _size(source_training_size), _size(evaluation_size)
    raw = read_regular_reference(checkpoint)
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    del raw
    _require(type(payload) is dict and type(payload.get("binding")) is dict, "checkpoint payload is invalid")
    inspected = inspect_checkpoint(checkpoint["path"], expected_sha256=checkpoint["sha256"],
                                   expected_binding=payload["binding"])
    engine, bound = payload["engine"], payload["binding"]["config"]
    _require(engine["epoch"] == 30 and engine["epoch_active"] is False
             and engine["optimizer_updates"] == 9120 and engine["microsteps"] == 18240
             and engine["epoch_start_optimizer_updates"] == 8816 and engine["phase"] == 0
             and engine["failed"] is False and payload["ema"]["updates"] == 9120,
             "checkpoint is not the complete paired seed-0 epoch-30 endpoint")
    expected = {"input_size": source_training_size, "physical_batch_size": 8,
                "accumulation_steps": 2, "seed": 0, "sampling_backend": "deterministic_gather",
                "ema_decay": 0.9999, "ema_warmups": 2000}
    _require(all(bound.get(name) == value and type(bound.get(name)) is type(value)
                 for name, value in expected.items()), "checkpoint source training contract differs")
    _require(engine["config"]["expected_input_size"] == [source_training_size] * 2
             and payload["ema_layout"]["config"] == {"decay": 0.9999, "warmups": 2000},
             "checkpoint EMA/history geometry differs")
    config = V2BConfig(seed=0, input_size=source_training_size, physical_batch_size=8,
                       accumulation_steps=2, pretrained_required=False,
                       sampling_backend="deterministic_gather")
    model, criterion, postprocessor, resolved, _, _ = _build_vendor_objects(Path(repo_root), config)
    del criterion
    sampling = _configure_model_sampling(model, config)
    loaded = model.load_state_dict(payload["ema"]["module"], strict=True)
    _require(not loaded.missing_keys and not loaded.unexpected_keys, "EMA strict loading failed")
    for key, value in model.state_dict().items():
        _require(torch.equal(value, payload["ema"]["module"][key]),
                 "EMA load changed checkpoint tensor bytes: " + key)
    _require(all(value.dtype == torch.float32 for value in model.parameters()),
             "EMA learned weights must be FP32")
    loaded_state = _state_identity(model)
    model.requires_grad_(False)
    geometry = adapt_evaluation_geometry(
        model, source_training_size=source_training_size, evaluation_size=evaluation_size,
        updates=9120, decay=0.9999, warmups=2000,
        expected_raw_anchors=payload["model"]["decoder.anchors"],
        expected_raw_mask=payload["model"]["decoder.valid_mask"],
    )
    source = {
        "checkpoint": copy.deepcopy(checkpoint), "checkpoint_binding_sha256": inspected["binding_sha256"],
        "checkpoint_engine": engine, "source_training_config": expected,
        "source_runtime": payload["runtime"], "source_pretrained": bound["pretrained"],
        "weights": "ema", "strict_loaded_state": loaded_state, "sampling": sampling,
        "geometry": geometry, "checkpoint_rng_restored": False,
        "optimizer_or_training_loader_constructed": False,
    }
    del payload
    model.eval()
    postprocessor.eval()
    return model, postprocessor, source, config.cuda_backend_policy


def prediction_origin_indices(outputs, original_sizes, processed):
    """Record the exact query/class origin without changing the official prediction rows."""
    import torchvision
    probabilities = outputs["pred_logits"].sigmoid()
    scores, indices = torch.topk(probabilities.flatten(1), 300, dim=-1)
    boxes = torchvision.ops.box_convert(outputs["pred_boxes"], in_fmt="cxcywh", out_fmt="xyxy")
    boxes *= original_sizes.repeat(1, 2).unsqueeze(1)
    selected = boxes.gather(1, (indices // 10).unsqueeze(-1).repeat(1, 1, 4))
    _require(len(processed) == len(indices), "postprocessed image coverage differs")
    for index, row in enumerate(processed):
        _require(torch.equal(row["scores"], scores[index])
                 and torch.equal(row["labels"], indices[index] % 10 + 1)
                 and torch.equal(row["boxes"], selected[index]),
                 "raw query-to-flattened prediction provenance differs from frozen postprocessor")
    return indices, probabilities


def _write_npz(path: Path, **arrays):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    return evidence.file_reference(path)


def _cpu_rng():
    numpy_state = np.random.get_state()
    return random.getstate(), torch.get_rng_state().clone(), (
        numpy_state[0], numpy_state[1].copy(), *numpy_state[2:])


def _rng_equal(left, right):
    return (left[0] == right[0] and torch.equal(left[1], right[1])
            and left[2][0] == right[2][0] and np.array_equal(left[2][1], right[2][1])
            and left[2][2:] == right[2][2:])


def _memory():
    free, total = torch.cuda.mem_get_info(0)
    return {
        "cuda_total_bytes": int(total), "cuda_free_bytes_at_end": int(free),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        "allocated_bytes_at_end": int(torch.cuda.memory_allocated(0)),
        "reserved_bytes_at_end": int(torch.cuda.memory_reserved(0)),
    }


def run_worker(contract: dict, *, contract_reference: dict) -> dict:
    """Run one exact cell. Guardian clearance is only final after actual process exit."""
    from .training_v2b_admission import MonitoredHardwareSession
    from faster_coco_eval import COCO

    checked = validate_worker_contract(contract, verify_files=True)
    _require(_reference_json(contract_reference) == checked, "worker contract file bytes differ")
    output = Path(checked["output_dir"])
    output.mkdir(parents=False, exist_ok=False)
    started = time.monotonic_ns()
    monitor = None
    report = {
        "schema_version": 1, "status": "RUNNING", "stage": "development_cross_eval",
        "run_id": checked["run_id"], "campaign_id": checked["campaign_id"],
        "cell_key": checked["cell_key"], "contract_reference": copy.deepcopy(contract_reference),
        "worker_pid": os.getpid(), "guardian_clearance_required_after_worker_exit": True,
        "source_training_size": checked["source_training_size"],
        "evaluation_size": checked["evaluation_size"], "policy": policy(),
        "automatic_retry_or_resolution_fallback": False,
        "evaluator_evidence_reference": checked["evaluator_evidence_reference"],
        "certification_claimed": False,
    }
    try:
        required_env = {
            "PYTHONNOUSERSITE": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
            "MKL_THREADING_LAYER": "GNU", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "CUDA_VISIBLE_DEVICES": checked["expected_gpu_uuid"],
        }
        _require(all(os.environ.get(key) == value for key, value in required_env.items()),
                 "worker startup environment differs from the frozen evaluation policy")
        model, postprocessor, source, backend = prepare_cross_eval_model(
            checkpoint=checked["checkpoint"], source_training_size=checked["source_training_size"],
            evaluation_size=checked["evaluation_size"], repo_root=checked["repo_root"],
        )
        official, image_map, attributes = load_official_ground_truth(
            checked["official_gt_binding"], checked["development_binding"])
        dataset = development._DevelopmentDataset(checked["development_binding"])
        _require(len(dataset.images) == 548 and tuple(image_map) == tuple(image["id"] for image in dataset.images),
                 "full development coverage or order differs")
        report["cpu_preparation_reference"] = evidence.write_exclusive_json(
            output / "cpu-model-preparation.json", source)
        report["official_gt_binding_reference"] = evidence.write_exclusive_json(
            output / "official-gt-binding.json", checked["official_gt_binding"])
        report["diagnostic_attributes_reference"] = evidence.write_exclusive_json(
            output / "development-raw-attributes.json", attributes)
        report["estimated_primary_integral_bytes"] = sum((image.width+1)*(image.height+1)*8
                                                        for image in official.images)
        config = {
            **policy(), "input_size": checked["evaluation_size"], "seed": 0,
            "source_training_size": checked["source_training_size"],
            "device": "cuda:0", "cuda_gpu_uuid": checked["expected_gpu_uuid"],
            "authorization_reference": checked["authorization_reference"],
            "sampling_backend": "deterministic_gather", "cuda_backend_policy": backend,
            "checkpoint": checked["checkpoint"], "official_gt_binding_sha256":
                checked["official_gt_binding"]["binding_sha256"],
        }
        binding = evidence.build_run_binding(
            run_id=checked["run_id"],
            code_paths={name: reference["path"] for name, reference in checked["code_files"].items()},
            config=config, initial_parameters=evidence.initial_parameter_reference(dict(model.named_parameters())),
            initial_state=_state_identity(model),
            input_manifests={"development": checked["development_binding"]["manifest"]["path"]},
        )
        report["binding_reference"] = evidence.write_exclusive_json(output / "run-binding.json", binding)
        evidence.write_exclusive_json(output / "worker-start.json", report)
        monitor = MonitoredHardwareSession(binding, checked["policy_bundle"],
                                            output / "native-hardware", "development_cross_eval", 1800)
        monitor.start()
        runtime = prepare_runtime(device="cuda:0", seed=0, binding=binding,
                                  gpu_probe=monitor.admission(), expected_gpu_uuid=checked["expected_gpu_uuid"])
        cpu_state, cpu_geometry = _state_identity(model), snapshot_model_geometry(
            model, expected_input_size=checked["evaluation_size"])
        place_module(model, runtime)
        place_module(postprocessor, runtime)
        _require(_state_identity(model) == cpu_state and snapshot_model_geometry(
            model, expected_input_size=checked["evaluation_size"]) == cpu_geometry,
            "CUDA placement changed state or ordinary position caches")
        wrapper = VisDronePostProcessor(vendor_postprocessor=postprocessor).eval()
        evaluator = dataset.evaluator_type(COCO(copy.deepcopy(dataset.coco)), ["bbox"])
        evaluator.world_size = 1
        development._axes(evaluator.coco_eval["bbox"])
        before_rng, before_cuda_rng = _cpu_rng(), torch.cuda.get_rng_state(0).clone()
        torch.cuda.reset_peak_memory_stats(0)
        predictions, receipts, logits, boxes, sizes, indices, probabilities, batch_timings = [], [], [], [], [], [], [], []
        inference_started = time.monotonic_ns()
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            for batch_index, start in enumerate(range(0, len(dataset.images), 4)):
                monitor.check(stage="cross_eval_before_batch")
                image_rows = dataset.images[start:start+4]
                items = [dataset.item(image) for image in image_rows]
                samples = torch.stack([item[0] for item in items]).to(runtime.device)
                original_sizes = torch.stack([item[1]["orig_size"] for item in items]).to(runtime.device)
                torch.cuda.synchronize(0)
                batch_started = time.monotonic_ns()
                outputs = model(samples)
                for name, shape in (("pred_logits", [len(items), 300, 10]),
                                    ("pred_boxes", [len(items), 300, 4])):
                    _require(name in outputs and list(outputs[name].shape) == shape
                             and outputs[name].dtype == torch.float32
                             and bool(torch.isfinite(outputs[name]).all()),
                             "raw query logits/boxes changed shape, dtype or numerical health")
                _require(bool((outputs["pred_boxes"][..., 2:] > 0).all()),
                         "raw query boxes have nonpositive extent")
                processed = wrapper(outputs, original_sizes)
                origin, sigmoid = prediction_origin_indices(outputs, original_sizes, processed)
                indices.append(origin.detach().cpu().numpy().copy())
                probabilities.append(sigmoid.detach().cpu().numpy().copy())
                cpu_predictions = {}
                for image, row in zip(image_rows, processed):
                    _require(set(row) == {"labels", "boxes", "scores"} and len(row["scores"]) == 300
                             and bool(torch.isfinite(row["boxes"]).all())
                             and bool((row["boxes"][:, 2:] > row["boxes"][:, :2]).all()),
                             "the unchanged 300 flattened predictions are incomplete")
                    cpu_predictions[image["id"]] = {key: value.detach().cpu() for key, value in row.items()}
                logits.append(outputs["pred_logits"].detach().cpu().numpy().copy())
                boxes.append(outputs["pred_boxes"].detach().cpu().numpy().copy())
                sizes.append(original_sizes.detach().cpu().numpy().copy())
                predictions.extend(evaluator.prepare_for_coco_detection(cpu_predictions))
                evaluator.update(cpu_predictions)
                receipts.extend(item[2] for item in items)
                torch.cuda.synchronize(0)
                batch_timings.append({"batch": batch_index, "image_ids": [image["id"] for image in image_rows],
                                      "forward_postprocess_update_seconds":
                                          (time.monotonic_ns()-batch_started)/1e9})
                monitor.check(stage="cross_eval_after_batch")
        inference_seconds = (time.monotonic_ns() - inference_started)/1e9
        _require(len(predictions) == 548 * 300 and len(receipts) == 548
                 and development._digest(receipts) == checked["development_binding"]["image_manifest_sha256"],
                 "prediction budget or authenticated development image coverage differs")
        report["prediction_artifact"] = evidence.write_exclusive_json(output / "predictions.json", predictions)
        report["raw_query_artifact"] = _write_npz(
            output / "raw-queries.npz", image_ids=np.asarray(list(image_map), dtype=np.int64),
            pred_logits=np.concatenate(logits), pred_boxes=np.concatenate(boxes),
            original_sizes=np.concatenate(sizes), topk_indices=np.concatenate(indices),
            pred_sigmoid_scores=np.concatenate(probabilities),
        )
        report["raw_query_schema"] = {
            "pred_logits": {"shape": [548, 300, 10], "dtype": "float32", "meaning": "pre_sigmoid_class_logits"},
            "pred_boxes": {"shape": [548, 300, 4], "dtype": "float32", "meaning": "normalized_cxcywh"},
            "pred_sigmoid_scores": {"shape": [548, 300, 10], "dtype": "float32",
                                    "meaning": "runtime_fp32_sigmoid_used_to_authenticate_vendor_topk"},
            "original_sizes": {"shape": [548, 2], "meaning": "width_height_original_pixels"},
            "topk_indices": {"shape": [548, 300], "dtype": "int64",
                             "meaning": "query_index_times_10_plus_zero_based_class"},
            "raw_query_order": "unchanged_model_query_index",
            "flattened_prediction_order": "unchanged_vendor_top300_query_class_score_order",
        }
        report["image_receipts_reference"] = evidence.write_exclusive_json(output / "image-receipts.json", receipts)
        report["batch_timings_reference"] = evidence.write_exclusive_json(output / "batch-timings.json", batch_timings)
        report["memory"] = _memory()
        report["inference_elapsed_seconds"] = inference_seconds
        monitor.check(stage="cross_eval_before_metrics")
        metrics_started = time.monotonic_ns()
        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        coco, axes, precision, recall = development._coco_metrics(evaluator.coco_eval["bbox"])
        report["coco_secondary"] = {
            "evaluator_id": "coco_secondary_vendor_v1", "ground_truth": "unchanged_converted_formal",
            "metrics": coco, "axes": axes,
            "tensor_artifact": development._publish_tensors(output / "coco_tensors.npz", precision, recall),
        }
        official_input = with_predictions(official, image_map, predictions)
        primary_binding = checked["development_binding"]["source"]["primary_contract"]
        primary = primary_evaluator.evaluate_primary_v1(official_input, primary_binding)
        primary_evaluator.validate_primary_evaluator_result(primary, official_input, primary_binding)
        primary_dict = evidence.strict_json_loads(development._canonical(asdict(primary)))
        report["primary_official_gt"] = {
            "evaluator_id": primary_evaluator.EVALUATOR_ID if hasattr(primary_evaluator, "EVALUATOR_ID")
                else "visdrone_official_primary_evaluator_v1",
            "entry": policy()["primary_entry"], "ground_truth_binding_sha256":
                checked["official_gt_binding"]["binding_sha256"],
            "metrics": {name: primary_dict[name] for name in ("AP", "AP50", "AP75", "AR1", "AR10", "AR100", "AR500")},
            "units": "percent", "result_artifact": evidence.write_exclusive_json(
                output / "primary-official-gt-result.json", primary_dict),
        }
        monitor.check(stage="cross_eval_after_official_primary")
        cache = prepare_primary_cache(official_input, primary_binding, image_map)
        cached = cache.score(tuple(image_map))
        for name, value in cached.items():
            _require(math.isclose(value, primary_dict[name], rel_tol=0.0, abs_tol=1e-10),
                     "bootstrap cache point estimate differs from full primary entry: " + name)
        report["primary_cache_point_check"] = {
            "status": "PASS", "metrics": cached, "absolute_tolerance_percent": 1e-10,
            "description": cache.describe(), "full_primary_entry_used_for_reported_point_estimate": True,
        }
        legacy = development._primary_result(dataset, dataset.images, predictions)
        report["primary_legacy_formal_gt"] = {
            "metrics": {name: legacy[name] for name in ("AP", "AP50", "AP75", "AR1", "AR10", "AR100", "AR500")},
            "ignore_regions_loaded": False, "units": "percent",
            "result_artifact": evidence.write_exclusive_json(output / "primary-legacy-formal-gt-result.json", legacy),
        }
        report["metrics_elapsed_seconds"] = (time.monotonic_ns() - metrics_started)/1e9
        _require(_state_identity(model) == cpu_state and snapshot_model_geometry(
            model, expected_input_size=checked["evaluation_size"]) == cpu_geometry,
            "evaluation changed a learned weight, BN buffer, anchor or position cache")
        _require(_rng_equal(before_rng, _cpu_rng()) and torch.equal(before_cuda_rng, torch.cuda.get_rng_state(0)),
                 "deterministic evaluation consumed CPU or CUDA RNG")
        validate_module_placement(model, runtime.device)
        validate_worker_contract(checked, verify_files=True)
        report["state_preservation"] = {
            "learned_and_all_buffers_byte_identical": True, "geometry_byte_identical": True,
            "cpu_rng_unchanged": True, "cuda_rng_unchanged": True, "checkpoint_rng_restored": False,
        }
        report["runtime"] = runtime.identity
        report["full_development_coverage"] = {"images": 548, "predictions": 548*300,
                                              "image_order_sha256": checked["development_binding"]["image_order_sha256"]}
        monitor.check(stage="cross_eval_before_finish")
        report["monitor_reference"] = monitor.finish()
        report["execution_elapsed_seconds"] = (time.monotonic_ns()-started)/1e9
        report["status"] = "PASS"
        evidence.write_exclusive_json(output / "worker-result.json", report)
        return report
    except BaseException as exc:
        report["status"] = "STOP_NO_RETRY"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        try:
            evidence.write_exclusive_json(output / "worker-result.json", report)
        finally:
            if monitor is not None:
                monitor.abort("cross-evaluation failure: " + type(exc).__name__ + ": " + str(exc))
        raise


run_cross_eval_worker = run_worker
