"""CPU evidence closure for four completed development resolution cells.

Only explicitly SHA-bound metadata and saved outputs are opened. The reference
point metrics come from the full existing primary and faster COCO evaluators.
The caches must reproduce those points before paired cluster resampling.
"""
from __future__ import annotations

import copy
import hashlib
import io
import math
import os
from pathlib import Path
import time

import numpy as np

from .training_v2b_campaign import (
    CampaignError, _canonical, _json_reference, _write_json, file_reference,
    read_reference, validate_frozen_source,
)
from .training_v2b_evidence_campaign import (
    CELL_SIZES, cross_worker_contract, validate_evidence_campaign,
)
from .training_v2b_official_gt import (
    load_official_ground_truth, read_regular_reference, with_predictions,
)
from .training_v2b_primary_cache import prepare_primary_cache
from . import training_v2b_resolution_diagnostics as diagnostics


def _same(observed, expected, name):
    if _canonical(observed) != _canonical(expected):
        raise CampaignError(name + " changed during resolution analysis")


def compare_point_metrics(cached, reference, names, *, tolerance=1e-10):
    """A cache discrepancy cannot be excused by statistical confidence intervals."""
    checked = {}
    for name in names:
        a, b = cached.get(name), reference.get(name)
        if (type(a) not in (int, float) or type(b) not in (int, float)
                or not math.isfinite(a) or not math.isfinite(b)
                or not math.isclose(a, b, rel_tol=0., abs_tol=tolerance)):
            raise CampaignError("full evaluator/cache point discrepancy: " + name)
        checked[name] = {"cache_percent": a, "reference_percent": b, "delta_pp": a-b}
    return {"status": "PASS", "absolute_tolerance_percent": tolerance, "metrics": checked}


def load_raw_queries(reference, image_ids):
    raw = read_regular_reference(reference)
    required = {"image_ids", "pred_logits", "pred_boxes", "original_sizes",
                "topk_indices", "pred_sigmoid_scores"}
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise CampaignError("raw query sidecar schema differs from the frozen worker")
        arrays = {name: archive[name].copy() for name in archive.files}
    ids = arrays.pop("image_ids")
    if ids.dtype != np.int64 or ids.tolist() != list(image_ids):
        raise CampaignError("raw query image order or identity differs")
    expected = {
        "pred_logits": ((len(ids), 300, 10), np.dtype("float32")),
        "pred_boxes": ((len(ids), 300, 4), np.dtype("float32")),
        "pred_sigmoid_scores": ((len(ids), 300, 10), np.dtype("float32")),
        "topk_indices": ((len(ids), 300), np.dtype("int64")),
    }
    for name, (shape, dtype) in expected.items():
        if arrays[name].shape != shape or arrays[name].dtype != dtype or not np.isfinite(arrays[name]).all():
            raise CampaignError("raw sidecar tensor shape, dtype or numerical health differs: " + name)
    sizes = arrays["original_sizes"]
    if sizes.shape != (len(ids), 2) or not np.isfinite(sizes).all() or (sizes <= 0).any():
        raise CampaignError("raw sidecar original image dimensions are invalid")
    return {int(image_id): {name: array[i] for name, array in arrays.items()}
            for i, image_id in enumerate(ids)}



def _validate_cell_artifacts(worker, completion, expected, contract_reference):
    """Re-read owned leaf artifacts and native evidence before statistical admission."""
    from .training_v2b_admission import _validate_policy, wait_for_monitored_finish

    output = Path(expected["output_dir"])
    hardware = output / "native-hardware"
    monitor_reference = worker.get("monitor_reference")
    if (type(monitor_reference) is not dict
            or monitor_reference.get("monitor_final_report") != str(hardware / "monitor-final.json")):
        raise CampaignError("native monitor reference escaped its owned cell")
    observed_monitor = wait_for_monitored_finish(monitor_reference, timeout_seconds=1.)
    _same(observed_monitor, completion["monitor"], "persisted native completion")

    launch = _json_reference(completion["launch_reference"])
    if (launch.get("contract") != contract_reference or launch.get("cell_key") != expected["cell_key"]
            or type(worker.get("worker_pid")) is not int
            or launch.get("pid") != worker["worker_pid"]
            or observed_monitor.get("owner") != launch.get("owner")
            or observed_monitor.get("owner", {}).get("pid") != worker["worker_pid"]):
        raise CampaignError("native monitor, launched PID and worker contract are not the same invocation")
    policy = _validate_policy(expected["policy_bundle"], "development_cross_eval", 1800)
    for key, value in (("run_id", expected["run_id"]), ("gpu_uuid", expected["expected_gpu_uuid"]),
                       ("policy_sha256", policy["policy_sha256"])):
        _same(observed_monitor.get(key), value, "native " + key)

    leaves = {
        key: worker.get(key) for key in (
            "cpu_preparation_reference", "official_gt_binding_reference",
            "diagnostic_attributes_reference", "binding_reference",
            "image_receipts_reference", "batch_timings_reference",
            "prediction_artifact", "raw_query_artifact",
        )
    }
    leaves.update(
        coco_tensors=worker["coco_secondary"]["tensor_artifact"],
        full_primary=worker["primary_official_gt"]["result_artifact"],
        legacy_primary=worker["primary_legacy_formal_gt"]["result_artifact"],
    )
    for name, ref in leaves.items():
        if type(ref) is not dict:
            raise CampaignError("required cell evidence reference is absent: " + name)
        path = Path(ref.get("path", ""))
        if (not path.is_absolute() or ".." in path.parts or not path.is_relative_to(output)
                or path.resolve() != path):
            raise CampaignError("cell evidence reference escaped its owned output: " + name)
        read_regular_reference(ref)
    binding = _json_reference(worker["binding_reference"])
    _same(binding.get("run_id"), expected["run_id"], "native run binding identity")
    _same(binding.get("binding_sha256"), observed_monitor.get("run_binding_sha256"),
          "native run binding SHA")
    _same(_json_reference(worker["official_gt_binding_reference"]), expected["official_gt_binding"],
          "actual official GT binding")
    preparation = _json_reference(worker["cpu_preparation_reference"])
    _same(preparation.get("checkpoint"), expected["checkpoint"], "actual loaded checkpoint")
    from .training_v2b_development import _digest
    receipts = _json_reference(worker["image_receipts_reference"])
    if (type(receipts) is not list or len(receipts) != 548
            or _digest(receipts) != expected["development_binding"]["image_manifest_sha256"]):
        raise CampaignError("actual development image receipts changed")
    return {"status": "PASS", "native_monitor_reverified": True,
            "owned_leaf_artifact_count": len(leaves), "actual_input_receipts_reverified": True}


def _load_completed_cells(campaign, campaign_reference, result_reference):
    result = _json_reference(result_reference)
    if (result.get("status") != "PASS" or result.get("campaign_reference") != campaign_reference
            or result.get("campaign_id") != campaign["campaign_id"]
            or set(result.get("completed", {})) != set(CELL_SIZES)
            or result.get("training_started") is not False):
        raise CampaignError("the complete evaluation campaign has not passed")
    cells, completions = {}, {}
    for cell in CELL_SIZES:
        item = result["completed"][cell]
        completion = _json_reference(item["completion_reference"])
        worker = _json_reference(item["result_reference"])
        expected = cross_worker_contract(campaign, cell)
        if (completion.get("result_reference") != item["result_reference"]
                or completion.get("monitor", {}).get("status") != "PASS"
                or worker.get("status") != "PASS"
                or worker.get("contract_reference") != item["contract_reference"]
                or worker.get("run_id") != expected["run_id"]):
            raise CampaignError("cell or native monitor did not finish successfully: " + cell)
        _same(_json_reference(item["contract_reference"]), expected, cell + " worker contract")
        if worker.get("state_preservation") != {
                "learned_and_all_buffers_byte_identical": True, "geometry_byte_identical": True,
                "cpu_rng_unchanged": True, "cuda_rng_unchanged": True, "checkpoint_rng_restored": False}:
            raise CampaignError("cell did not preserve the complete inference state: " + cell)
        _validate_cell_artifacts(worker, completion, expected, item["contract_reference"])
        cells[cell], completions[cell] = worker, completion
    return cells, completions


def run_resolution_analysis(*, campaign_reference: dict, campaign_result_reference: dict,
                            output_dir: str | Path) -> dict:
    """Run fixed diagnostics and resamples once. Never launch a seed replication."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise CampaignError("offline analysis requires empty CUDA_VISIBLE_DEVICES")
    campaign = _json_reference(campaign_reference)
    if Path(__file__).resolve() != Path(campaign["source"]["repo_root"]) / "src/sparse_rtdetr/baseline/training_v2b_resolution_analysis.py":
        raise CampaignError("analysis module imported from a different checkout")
    validate_evidence_campaign(campaign)
    validate_frozen_source(campaign["source"])
    _same(campaign["diagnostic_policy"], diagnostics.frozen_method(), "predeclared diagnostic method")
    cells, completions = _load_completed_cells(campaign, campaign_reference, campaign_result_reference)
    output = Path(output_dir).absolute()
    parent = Path(campaign["output_root"])
    if (not output.is_relative_to(parent) or output == parent or output.resolve() != output
            or output.is_symlink()):
        raise CampaignError("analysis output must be a new directory inside its evidence campaign")
    output.mkdir(parents=False, exist_ok=False)
    started = time.monotonic()
    report = {
        "schema_version": 1, "status": "RUNNING", "campaign_reference": campaign_reference,
        "campaign_result_reference": campaign_result_reference,
        "method": diagnostics.frozen_method(), "primary_points_from_full_evaluator": True,
        "faster_coco_points_replayed": False, "gpu_work_launched": False,
        "training_launched": False, "confirmatory_or_test_accessed": False,
        "cells": {},
    }
    progress_path = output / "progress.jsonl"
    progress = progress_path.open("xb", buffering=0)

    def event(stage, **fields):
        row = {"stage": stage, "elapsed_seconds": time.monotonic()-started, **fields}
        progress.write(_canonical(row)+b"\n")
        os.fsync(progress.fileno())

    try:
        dev = campaign["development_bindings"]["640"]
        coco = __import__("json").loads(read_regular_reference(dev["annotation"]))
        manifest = __import__("json").loads(read_regular_reference(dev["manifest"]))
        lineage_ref = campaign["official_gt_bindings"]["640"]["lineage"]
        lineage = [__import__("json").loads(line)
                   for line in read_regular_reference(lineage_ref).splitlines()]
        image_ids = tuple(sorted(image["id"] for image in coco["images"]))
        if len(image_ids) != 548:
            raise CampaignError("analysis development membership differs")
        event("authorized_metadata_loaded", images=len(image_ids), lineage_rows=len(lineage))
        caches, primary_caches, error_reports, predictions_by_cell, raw_by_cell = {}, {}, {}, {}, {}
        for cell, (_, evaluation_size) in CELL_SIZES.items():
            worker = cells[cell]
            predictions = __import__("json").loads(read_regular_reference(worker["prediction_artifact"]))
            cache = diagnostics.prepare_coco_cache(coco, predictions, max_dets=100)
            observed = cache.score(image_ids)
            reference = {key: value["value"] for key, value in worker["coco_secondary"]["metrics"].items()}
            coco_check = compare_point_metrics(
                observed, reference,
                ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
                 "AR100", "AR_small", "AR_medium", "AR_large"),
            )
            binding = campaign["development_bindings"][str(evaluation_size)]
            official, image_map, attributes = load_official_ground_truth(
                campaign["official_gt_bindings"][str(evaluation_size)], binding)
            primary_input = with_predictions(official, image_map, predictions)
            primary_cache = prepare_primary_cache(primary_input, binding["source"]["primary_contract"], image_map)
            primary_check = compare_point_metrics(
                primary_cache.score(image_ids), worker["primary_official_gt"]["metrics"],
                ("AP", "AP50", "AP75", "AR500"),
            )
            full_primary = _json_reference(worker["primary_official_gt"]["result_artifact"])
            _same(full_primary.get("canonical_input_sha256"), primary_cache.input_sha256,
                  cell + " full primary input identity")
            _same({k: full_primary[k] for k in worker["primary_official_gt"]["metrics"]},
                  worker["primary_official_gt"]["metrics"], cell + " full primary artifact")
            event("reference_points_reproduced", cell=cell, coco_AP=observed["AP"],
                  coco_AP_small=observed["AP_small"],
                  primary_AP=worker["primary_official_gt"]["metrics"]["AP"])
            error_cache = diagnostics.prepare_coco_cache(coco, predictions, max_dets=300)
            errors = diagnostics.decompose_errors(
                coco, predictions, lineage_rows=lineage, cache=error_cache,
                source_references={"annotation": dev["annotation"], "lineage": lineage_ref,
                                   "predictions": worker["prediction_artifact"]},
            )
            raw_by_image = load_raw_queries(worker["raw_query_artifact"], image_ids)
            raw_report = diagnostics.raw_query_coverage(coco, predictions, raw_by_image)
            error_reference = _write_json(output / (cell + ".errors.json"), errors)
            raw_reference = _write_json(output / (cell + ".raw-query-coverage.json"), raw_report)
            report["cells"][cell] = {
                "worker_result": _json_reference(campaign_result_reference)["completed"][cell]["result_reference"],
                "prediction_artifact": worker["prediction_artifact"],
                "raw_query_artifact": worker["raw_query_artifact"],
                "primary_official_gt": copy.deepcopy(worker["primary_official_gt"]),
                "primary_legacy_formal_gt": copy.deepcopy(worker["primary_legacy_formal_gt"]),
                "coco_secondary": copy.deepcopy(worker["coco_secondary"]),
                "coco_cache_check": coco_check, "primary_cache_check": primary_check,
                "errors": error_reference, "raw_query_coverage": raw_reference,
                "inference_elapsed_seconds": worker["inference_elapsed_seconds"],
                "execution_elapsed_seconds": worker["execution_elapsed_seconds"],
                "memory": worker["memory"],
            }
            caches[cell], primary_caches[cell], error_reports[cell] = cache, primary_cache, errors
            predictions_by_cell[cell], raw_by_cell[cell] = predictions, raw_by_image
            del error_cache
            event("cell_diagnostics_completed", cell=cell)
        report["faster_coco_points_replayed"] = True
        # Native-resolution diagonals characterize the currently observed total
        # benefit; the same-weight t640 pair isolates inference scaling for routing.
        for label, coarse, fine in (
                ("diagonal", "t640_e640", "t896_e896"),
                ("same_training640", "t640_e640", "t640_e896")):
            oracle = diagnostics.routing_oracle_diagnostic(
                coco, predictions_by_cell[coarse], predictions_by_cell[fine],
                raw_coarse_by_image=raw_by_cell[coarse], coarse_cache=caches[coarse], fine_cache=caches[fine],
            )
            report[label + "_routing_oracle"] = _write_json(output / (label + ".routing-oracle.json"), oracle)
        event("routing_completed")

        def bootstrap_progress(kind, completed, total):
            if completed == 1 or completed % 25 == 0 or completed == total:
                event("bootstrap", kind=kind, completed=completed, total=total)

        bootstrap = diagnostics.paired_bootstrap_2x2(
            caches, manifest["records"], primary_scorers=primary_caches,
            error_reports=error_reports, progress=bootstrap_progress,
        )
        report["paired_bootstrap"] = _write_json(output / "paired-bootstrap.json", bootstrap)
        gate = copy.deepcopy(bootstrap["seed_gate"])
        gate["engineering_and_hardware_campaign_passed"] = all(
            completion["monitor"]["status"] == "PASS" for completion in completions.values())
        gate["conditional_execution_authority"] = campaign["authorization_reference"]
        gate["predeclared_additional_seeds"] = [1, 2]
        gate["eligible_for_fixed_paired_seed1_and_seed2"] = (
            gate["eligible_for_fixed_paired_seed1_and_seed2"]
            and gate["engineering_and_hardware_campaign_passed"])
        report["seed_gate"] = gate
        _same(_json_reference(campaign_reference), campaign, "campaign input")
        validate_frozen_source(campaign["source"])
        report["elapsed_seconds"] = time.monotonic()-started
        report["status"] = "PASS"
        event("finished", seed_gate=gate["eligible_for_fixed_paired_seed1_and_seed2"])
        progress.close()
        report["progress_reference"] = file_reference(progress_path)
        _write_json(output / "analysis-result.json", report)
        return report
    except BaseException as exc:
        report.update(status="FAIL", elapsed_seconds=time.monotonic()-started,
                      failure={"type": type(exc).__name__, "message": str(exc)})
        event("failed", **report["failure"])
        progress.close()
        report["progress_reference"] = file_reference(progress_path)
        _write_json(output / "analysis-result.json", report)
        raise
    finally:
        if not progress.closed:
            progress.close()
