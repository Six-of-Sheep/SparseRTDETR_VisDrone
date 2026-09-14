"""CPU integration boundaries that guard nonlinear bootstrap evidence."""
import io
import hashlib
import numpy as np
import pytest
from sparse_rtdetr.baseline.training_v2b_campaign import CampaignError
from sparse_rtdetr.baseline.training_v2b_resolution_analysis import compare_point_metrics, load_raw_queries


def test_cache_reference_mismatch_is_not_hidden_by_report_tolerance():
    assert compare_point_metrics({"AP": 20.}, {"AP": 20.}, ["AP"])["status"] == "PASS"
    with pytest.raises(CampaignError, match="point discrepancy"):
        compare_point_metrics({"AP": 20.0001}, {"AP": 20.}, ["AP"])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), None, True])
def test_nonfinite_missing_or_boolean_primary_point_rejected(value):
    with pytest.raises(CampaignError, match="point discrepancy"):
        compare_point_metrics({"AP": value}, {"AP": 20.}, ["AP"])


def test_raw_sidecar_image_identity_and_complete_sha_are_required(tmp_path):
    path = tmp_path / "raw-queries.npz"
    np.savez_compressed(path, image_ids=np.array([1], dtype=np.int64),
                        pred_logits=np.zeros((1, 300, 10), dtype=np.float32),
                        pred_boxes=np.ones((1, 300, 4), dtype=np.float32),
                        pred_sigmoid_scores=np.full((1, 300, 10), .5, dtype=np.float32),
                        topk_indices=np.arange(300, dtype=np.int64).reshape(1, 300),
                        original_sizes=np.array([[640, 640]], dtype=np.int64))
    raw = path.read_bytes()
    ref = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
           "size_bytes": len(raw), "sha256_scope": "complete_file_bytes"}
    result = load_raw_queries(ref, [1])
    assert result[1]["pred_logits"].shape == (300, 10)
    with pytest.raises(CampaignError, match="image order"):
        load_raw_queries(ref, [2])
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError):
        load_raw_queries(ref, [1])


@pytest.fixture
def completed_cell(tmp_path, monkeypatch):
    from sparse_rtdetr.baseline import training_v2b_admission as admission
    from sparse_rtdetr.baseline.training_v2b_campaign import _write_json
    from sparse_rtdetr.baseline.training_v2b_development import _digest

    output = tmp_path / "cell"
    output.mkdir()
    hardware = output / "native-hardware"
    hardware.mkdir()
    owner = {"pid": 12345}
    policy_sha = "c" * 64
    monitor_ref = {
        "monitor_final_report": str(hardware / "monitor-final.json"),
        "run_id": "fixture-t640_e640", "run_binding_sha256": "b" * 64,
        "policy_sha256": policy_sha, "monitor_pid": 23456,
        "owner": owner, "guardian_identity": {"pid": 23456},
        "gpu_uuid": "fixture-gpu",
    }
    monitor = {k: v for k, v in monitor_ref.items() if k != "monitor_final_report"}
    monitor.update(status="PASS", worker_exited=True)
    for name in ("samples", "setter_receipt", "startup_evidence", "guardian_heartbeat"):
        monitor[name] = _write_json(hardware / (name + ".json"), {"fixture": True})
    final_ref = _write_json(hardware / "monitor-final.json", monitor)
    monitor["final_report_reference"] = final_ref
    receipt_rows = [{"image_id": i} for i in range(548)]
    expected = {"output_dir": str(output), "run_id": monitor_ref["run_id"],
                "cell_key": "t640_e640", "expected_gpu_uuid": "fixture-gpu",
                "policy_bundle": {}, "official_gt_binding": {"fixture_gt": True},
                "checkpoint": {"fixture_checkpoint": True},
                "development_binding": {"image_manifest_sha256": _digest(receipt_rows)}}
    contract_reference = _write_json(tmp_path / "contract.json", expected)
    worker = {"worker_pid": owner["pid"], "monitor_reference": monitor_ref}
    documents = {
        "cpu_preparation_reference": {"checkpoint": expected["checkpoint"]},
        "official_gt_binding_reference": expected["official_gt_binding"],
        "diagnostic_attributes_reference": [],
        "binding_reference": {"run_id": expected["run_id"], "binding_sha256": "b" * 64},
        "image_receipts_reference": receipt_rows,
        "batch_timings_reference": [],
        "prediction_artifact": [],
        "raw_query_artifact": {"raw_fixture_only": True},
    }
    for key, value in documents.items():
        worker[key] = _write_json(output / (key + ".json"), value)
    for key, subkey in (("coco_secondary", "tensor_artifact"),
                        ("primary_official_gt", "result_artifact"),
                        ("primary_legacy_formal_gt", "result_artifact")):
        worker[key] = {subkey: _write_json(output / (key + ".json"), {"fixture": True})}
    launch = {"pid": owner["pid"], "owner": owner, "contract": contract_reference,
              "cell_key": expected["cell_key"]}
    completion = {"monitor": monitor, "launch_reference": _write_json(tmp_path / "launch.json", launch)}
    monkeypatch.setattr(admission, "_validate_policy", lambda *a: {"policy_sha256": policy_sha})
    return worker, completion, expected, contract_reference


def test_offline_cell_revalidates_actual_native_report_and_all_owned_leaves(completed_cell):
    from sparse_rtdetr.baseline.training_v2b_resolution_analysis import _validate_cell_artifacts
    result = _validate_cell_artifacts(*completed_cell)
    assert result["native_monitor_reverified"] is True
    assert result["owned_leaf_artifact_count"] == 11


def test_offline_cell_rejects_changed_native_samples_even_when_summary_says_pass(completed_cell):
    from pathlib import Path
    from sparse_rtdetr.baseline.training_v2b_resolution_analysis import _validate_cell_artifacts
    from sparse_rtdetr.baseline.training_v2b_admission import MonitoredHardwareError
    path = Path(completed_cell[1]["monitor"]["samples"]["path"])
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(MonitoredHardwareError, match="evidence changed"):
        _validate_cell_artifacts(*completed_cell)


def test_offline_cell_rejects_missing_coco_tensor_and_wrong_launched_pid(completed_cell):
    from pathlib import Path
    from sparse_rtdetr.baseline.training_v2b_resolution_analysis import _validate_cell_artifacts
    worker, completion, expected, reference = completed_cell
    worker["worker_pid"] += 1
    with pytest.raises(CampaignError, match="same invocation"):
        _validate_cell_artifacts(worker, completion, expected, reference)
    worker["worker_pid"] -= 1
    Path(worker["coco_secondary"]["tensor_artifact"]["path"]).unlink()
    with pytest.raises((OSError, ValueError)):
        _validate_cell_artifacts(worker, completion, expected, reference)
