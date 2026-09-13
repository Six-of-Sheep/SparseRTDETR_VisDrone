"""CPU-only evidence tests. GPU subprocesses are mocked explicitly."""
import copy
import json
import os
from pathlib import Path
import subprocess

import pytest
import torch

from sparse_rtdetr.baseline.training_v2b_checkpoint import CheckpointError, save_checkpoint
from sparse_rtdetr.baseline.training_v2b_engine import AccumulationEngine

from sparse_rtdetr.baseline import training_v2b_evidence as evidence


@pytest.fixture(autouse=True)
def no_unmocked_gpu_process(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("GPU/native subprocesses require an explicit test mock")
    monkeypatch.setattr(evidence.subprocess, "run", forbidden)


class EvidenceDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.register_buffer("tracked", torch.tensor(0))

    def forward(self, images, targets=None):
        self.tracked.add_(1)
        return {"prediction": images.mean(dim=(1, 2, 3)) * self.weight.sum()}


class EvidenceCriterion:
    def __init__(self, zero=False):
        self.zero = zero

    def __call__(self, outputs, targets):
        return {"proxy": outputs["prediction"].square().mean() * (0.0 if self.zero else 1.0)}


@pytest.fixture
def cpu_state(request):
    model = EvidenceDetector()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False)
    engine = AccumulationEngine(
        model, EvidenceCriterion(zero=getattr(request, "param", False)), optimizer,
        physical_batch_size=8, accumulation_steps=2, amp_dtype="float32",
        clip_max_norm=0.1, expected_input_size=2,
    )
    images = torch.arange(192, dtype=torch.float32).reshape(16, 3, 2, 2) / 192
    targets = [{"labels": torch.tensor([index % 2]),
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])} for index in range(16)]
    return {"model": model, "optimizer": optimizer, "engine": engine,
            "images": images, "targets": targets}


@pytest.fixture
def bound(tmp_path, cpu_state):
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    inputs = {"images": cpu_state["images"]}
    for index, target in enumerate(cpu_state["targets"]):
        inputs.update({f"targets.{index}.{name}": tensor for name, tensor in target.items()})
    return evidence.build_run_binding(
        run_id="cpu-evidence-fixture", code_paths={"source": source},
        config={"device": "cpu", "scope": "cpu_synthetic_engineering",
                **cpu_state["engine"].state_dict()["config"], "input_size": 2,
                "logical_batch_size": 16},
        initial_parameters=evidence.initial_parameter_reference(dict(cpu_state["model"].named_parameters())),
        initial_state=evidence.initial_parameter_reference(cpu_state["model"].state_dict()),
        synthetic_input={"fixture": "16 deterministic CPU images and all normalized targets",
                         "tensor_sha256": evidence.initial_parameter_reference(inputs)["sha256"]})


@pytest.fixture
def artifacts(tmp_path, bound, cpu_state):
    engine = cpu_state["engine"]
    engine.begin_epoch(1)
    rows = [engine.train_window(cpu_state["images"], cpu_state["targets"]) for _ in range(2)]
    engine.finish_epoch(expected_optimizer_steps=2)
    checkpoint = save_checkpoint(
        tmp_path / "checkpoint.pt", model=cpu_state["model"], optimizer=cpu_state["optimizer"],
        engine=engine, ema=None, binding=bound,
    )
    counters = {"epoch": 1, "optimizer_updates": 2, "microsteps": 4, "samples_seen": 32}
    window = evidence.write_exclusive_json(
        tmp_path / "windows.json",
        {"run_binding_sha256": bound["binding_sha256"], "counters": counters, "windows": rows})
    return checkpoint, counters, [window]


def metric(bound, checkpoint, counters, *, max_dets_index=1):
    parent = Path(checkpoint["path"]).parent
    prediction = evidence.write_exclusive_json(
        parent / "predictions.json",
        [{"image_id": 0, "category_id": 1, "bbox": [0.0, 0.0, 2.0, 2.0], "score": 0.9}])
    parameters = {
        "iouType": "bbox", "useCats": 1, "iouThrs": [0.5, 0.75],
        "recThrs": [0.0, 0.5, 1.0], "catIds": [1],
        "areaRng": [[0.0, 1e10], [0.0, 1024.0]], "areaRngLbl": ["all", "small"],
        "maxDets": [100, 500], "area_coordinate_space": "original_image_pixels",
    }
    precision = torch.full((2, 3, 1, 2, 2), 0.1502, dtype=torch.float64)
    precision[..., 0] = 0.1
    return evidence.build_coco_ap_record(
        precision, parameters=parameters, prediction_reference=prediction,
        source_path=parent / "precision.json", binding=bound,
        checkpoint_reference=checkpoint, counters=counters, weights="raw",
        evaluator_id="supplied_synthetic_coco_fixture", max_dets_index=max_dets_index,
        area_index=1, iou_indices=[0, 1])


def rewrite_windows(tmp_path, windows, mutate, name="mutated_windows.json"):
    document = evidence.strict_json_loads(Path(windows[0]["path"]).read_bytes())
    mutate(document)
    return [evidence.write_exclusive_json(tmp_path / name, document)]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), {1: "key"}, (1, 2)])
def test_canonical_rejects_non_json_or_nonfinite(value):
    with pytest.raises(evidence.EvidenceError):
        evidence.canonical_json_bytes({"nested": [value]})


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"x":NaN}', b'{"x":Infinity}'])
def test_strict_reader_rejects_ambiguous_json(raw):
    with pytest.raises(evidence.EvidenceError):
        evidence.strict_json_loads(raw)


def test_file_and_payload_digest_scopes_are_not_interchangeable(tmp_path):
    payload = {"message": "工程", "value": 2}
    reference = evidence.write_exclusive_json(tmp_path / "evidence.json", payload)
    raw = Path(reference["path"]).read_bytes()
    assert reference["sha256"] == evidence.sha256_bytes(raw)
    assert reference["sha256_scope"] == "complete_file_bytes"
    assert raw.endswith(b"\n")
    assert reference["sha256"] != evidence.canonical_sha256(payload)
    assert evidence.strict_json_loads(raw) == payload
    with pytest.raises(evidence.EvidenceError, match="exclusive"):
        evidence.write_exclusive_json(reference["path"], {"overwrite": True})


def test_invalid_json_is_rejected_before_file_creation(tmp_path):
    path = tmp_path / "must-not-exist.json"
    with pytest.raises(evidence.EvidenceError):
        evidence.write_exclusive_json(path, {"bad": float("nan")})
    assert not path.exists()


def test_initial_hash_uses_actual_cpu_values_and_scalar_buffer():
    torch = pytest.importorskip("torch")
    one = evidence.initial_parameter_reference({"w": torch.tensor([1., 2.]), "n": torch.tensor(0)})
    same = evidence.initial_parameter_reference({"n": torch.tensor(0), "w": torch.tensor([1., 2.])})
    changed = evidence.initial_parameter_reference({"w": torch.tensor([1., 3.]), "n": torch.tensor(0)})
    assert one == same
    assert one["sha256"] != changed["sha256"]
    bf16 = evidence.initial_parameter_reference({"w": torch.tensor([1., 2.], dtype=torch.bfloat16)})
    assert bf16["inventory"][0]["size_bytes"] == 4
    anchors = evidence.initial_parameter_reference({"anchors": torch.tensor([0.5, float("inf")])})
    altered = evidence.initial_parameter_reference({"anchors": torch.tensor([0.5, -float("inf")])})
    assert anchors["sha256"] != altered["sha256"]  # identity, not a health certificate


def test_binding_fixes_config_source_and_synthetic_identity(bound):
    body = {key: value for key, value in bound.items() if key != "binding_sha256"}
    assert bound["binding_sha256"] == evidence.canonical_sha256(body)
    assert bound["data"]["kind"] == "synthetic"
    assert bound["data"]["real_dataset_accessed"] is False
    changed = copy.deepcopy(bound)
    changed["config"]["physical_batch_size"] = 16
    with pytest.raises(evidence.EvidenceError, match="digest"):
        evidence.validate_run_binding(changed)
    source = Path(bound["code"]["source"]["path"])
    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(evidence.EvidenceError, match="size/hash"):
        evidence.validate_run_binding(bound)


def test_manifest_binding_is_actual_file_identity_and_rejects_sealed_roles(tmp_path, bound):
    path = tmp_path / "train_manifest.json"
    path.write_text('{"fixture":true}\n')
    kwargs = dict(run_id="manifest-fixture",
                  code_paths={key: value["path"] for key, value in bound["code"].items()},
                  config=bound["config"], initial_parameters=bound["initial_weights"])
    actual = evidence.build_run_binding(**kwargs, input_manifests={"train_core": path})
    assert actual["data"]["dataset_executed"] is False
    assert actual["data"]["manifests"]["train_core"]["sha256"] == evidence.sha256_bytes(path.read_bytes())
    for role in ("confirmatory", "test"):
        with pytest.raises(evidence.EvidenceError, match="roles"):
            evidence.build_run_binding(**kwargs, input_manifests={role: tmp_path / "never-read"})


def test_real_cpu_observation_never_queries_cuda(monkeypatch, bound):
    torch = pytest.importorskip("torch")
    def forbidden(*args, **kwargs):
        raise AssertionError("CUDA query is forbidden in CPU observation")
    for name in ("is_available", "is_initialized", "device_count"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    observation = evidence.collect_native_cpu_observation(bound)
    payload = evidence.validate_native_observation(observation, bound)
    assert payload["pid"] == os.getpid()
    assert payload["boot_id"] == Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    assert payload["device_type"] == "cpu"
    assert payload["hardware_probe_executed"] is False
    assert payload["gpu_observed"] is False
    assert payload["torch"]["cuda_runtime_or_device_queried"] is False
    assert payload["monotonic_finished_ns"] >= payload["monotonic_started_ns"]
    assert payload["utc_started"].endswith("Z")


def test_supplied_observation_cannot_be_relabelled_native(bound):
    observation = evidence.collect_native_cpu_observation(bound)
    supplied = observation.as_dict()
    supplied["hardware_probe_executed"] = True
    with pytest.raises(evidence.EvidenceError, match="supplied"):
        evidence.validate_native_observation(supplied, bound)
    with pytest.raises(evidence.EvidenceError, match="collector"):
        evidence.NativeObservation(object(), supplied)


def test_native_observation_rejects_stale_and_identity_drift(monkeypatch, bound):
    observation = evidence.collect_native_cpu_observation(bound)
    with pytest.raises(evidence.EvidenceError, match="boot"):
        evidence.validate_native_observation(observation, bound, expected_boot_id="another-boot")
    with pytest.raises(evidence.EvidenceError, match="UUID"):
        evidence.validate_native_observation(observation, bound, expected_gpu_uuid="GPU-other")
    other = copy.deepcopy(bound)
    other["run_id"] = "other-run"
    other["binding_sha256"] = evidence.canonical_sha256(
        {key: value for key, value in other.items() if key != "binding_sha256"})
    with pytest.raises(evidence.EvidenceError, match="run identity"):
        evidence.validate_native_observation(observation, other)
    finished = observation.as_dict()["monotonic_finished_ns"]
    monkeypatch.setattr(evidence.time, "monotonic_ns", lambda: finished + 61_000_000_000)
    with pytest.raises(evidence.EvidenceError, match="stale"):
        evidence.validate_native_observation(observation, bound)


def test_gpu_collector_uses_mocked_process_and_does_not_invent_cap(monkeypatch, bound):
    xml = b"""<nvidia_smi_log><driver_version>580.178.04</driver_version>
    <cuda_version>12.4</cuda_version><gpu><uuid>GPU-fixture</uuid>
    <product_name>Mock 4090 D</product_name>
    <clocks><graphics_clock>1500 MHz</graphics_clock></clocks>
    <max_clocks><graphics_clock>3105 MHz</graphics_clock></max_clocks>
    <power_readings><power_limit>425 W</power_limit></power_readings>
    <fb_memory_usage><total>24576 MiB</total><used>15 MiB</used></fb_memory_usage>
    <temperature><gpu_temp>52 C</gpu_temp></temperature>
    <processes><process_info><pid>1234</pid><process_name>fixture</process_name>
    <type>C</type><used_memory>1 MiB</used_memory></process_info></processes>
    </gpu></nvidia_smi_log>"""
    calls = []
    def mocked_process(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout=xml, stderr=b"")
    monkeypatch.setattr(evidence.subprocess, "run", mocked_process)
    monkeypatch.setattr(evidence, "_kernel_driver_version", lambda: "580.178.04-mock")
    monkeypatch.setattr(evidence, "_rapl_observation", lambda: {
        "source": "mock_sysfs", "available": True, "packages": [
            {"enabled": "1", "constraints": [
                {"name": "long_term", "power_limit_uw": 125000000},
                {"name": "short_term", "power_limit_uw": 150000000}]}]})
    observation = evidence.collect_native_gpu_observation(bound)
    value = evidence.validate_native_observation(observation, bound, expected_gpu_uuid="GPU-fixture")
    assert calls[0][0] == ["nvidia-smi", "-q", "-x"]
    assert value["hardware_probe_executed"] is True  # mock-native interface coverage only
    assert value["current_graphics_clock"] == "1500 MHz"
    assert value["device_max_graphics_clock"] == "3105 MHz"
    assert value["graphics_clock_cap_proof"]["verified"] is False
    assert value["hardware_admission_pass"] is False
    assert value["gpu_ready"] is False
    assert value["raw_process_capture"]["sha256"] == evidence.sha256_bytes(xml)
    with pytest.raises(evidence.EvidenceError, match="UUID"):
        evidence.validate_native_observation(observation, bound, expected_gpu_uuid="GPU-wrong")
    with pytest.raises(evidence.EvidenceError, match="not closed"):
        evidence.validate_native_observation(observation, bound, require_gpu_admission=True)


def test_completed_report_requires_real_refs_and_stays_cpu_synthetic(bound, artifacts):
    checkpoint, counters, windows = artifacts
    observation = evidence.collect_native_cpu_observation(bound)
    report = evidence.build_completed_report(bound, observation, checkpoint_reference=checkpoint,
                                             counters=counters, window_references=windows)
    assert report["status"] == "completed"
    assert report["scope"] == "cpu_synthetic"
    assert report["scientific_certified"] is False
    assert report["gpu_ready"] is False
    assert report["metrics"] == []
    changed = copy.deepcopy(report)
    changed["gpu_ready"] = True
    changed["report_sha256"] = evidence.canonical_sha256(
        {k: v for k, v in changed.items() if k != "report_sha256"})
    with pytest.raises(evidence.EvidenceError, match="certification"):
        evidence.validate_completed_report(changed, binding=bound, observation=observation)
    with pytest.raises(evidence.EvidenceError, match="window"):
        evidence.build_completed_report(bound, observation, checkpoint_reference=checkpoint, counters=counters)
    Path(checkpoint["path"]).write_text("changed checkpoint")
    with pytest.raises(evidence.EvidenceError, match="size/hash"):
        evidence.validate_completed_report(report, binding=bound, observation=observation)


def test_completed_report_rejects_wrong_checkpoint_run(bound, artifacts):
    checkpoint, counters, windows = artifacts
    checkpoint = {**checkpoint, "binding_sha256": "a" * 64}
    observation = evidence.collect_native_cpu_observation(bound)
    with pytest.raises(evidence.EvidenceError, match="checkpoint run"):
        evidence.build_completed_report(bound, observation, checkpoint_reference=checkpoint,
                                        counters=counters, window_references=windows)


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), 100.01])
def test_required_metric_rejects_cap500_stats0_sentinel_and_nonfinite(bound, artifacts, bad):
    checkpoint, counters, _ = artifacts
    record = metric(bound, checkpoint, counters)
    record["value"] = bad
    with pytest.raises(evidence.EvidenceError):
        evidence.validate_metric_record(record, binding=bound, checkpoint_reference=checkpoint, counters=counters)


def test_metrics_bind_evaluator_cap_area_units_weights_and_predictions(bound, artifacts):
    checkpoint, counters, _ = artifacts
    record = metric(bound, checkpoint, counters)
    assert evidence.validate_metric_record(record, binding=bound, checkpoint_reference=checkpoint,
                                            counters=counters)["value"] == pytest.approx(15.02)
    for field, value in (("checkpoint_sha256", "f" * 64), ("input_identity_sha256", "f" * 64),
                         ("weights", "unknown"), ("raw_prediction_sha256", None), ("units", "fraction"),
                         ("area", {})):
        changed = {**record, field: value}
        with pytest.raises(evidence.EvidenceError):
            evidence.validate_metric_record(changed, binding=bound, checkpoint_reference=checkpoint, counters=counters)
    wrong_role = {**record, "evaluator": {"id": "COCO", "family": "coco", "role": "primary"}}
    with pytest.raises(evidence.EvidenceError, match="evaluator"):
        evidence.validate_metric_record(wrong_role, binding=bound, checkpoint_reference=checkpoint, counters=counters)


def test_report_counts_must_match_persisted_window_evidence(bound, artifacts):
    checkpoint, counters, windows = artifacts
    changed_counts = {**counters, "samples_seen": 17}
    observation = evidence.collect_native_cpu_observation(bound)
    with pytest.raises(evidence.EvidenceError, match="counters"):
        evidence.build_completed_report(bound, observation, checkpoint_reference=checkpoint,
                                        counters=changed_counts, window_references=windows)


def test_native_capability_is_immutable_and_copied_payload_is_not_authority(bound):
    observation = evidence.collect_native_cpu_observation(bound)
    copied = observation.as_dict()
    copied["pid"] = -1
    assert observation.as_dict()["pid"] == os.getpid()
    with pytest.raises(evidence.EvidenceError, match="immutable"):
        observation._raw = b"supplied"


def test_initial_full_state_includes_buffers_and_checks_parameter_subset(tmp_path, cpu_state, bound):
    initial = bound["initial_weights"]
    assert initial["identity_scope"] == "model_state_dict"
    assert {row["name"] for row in initial["inventory"]} == {"weight", "tracked"}
    assert {row["name"] for row in initial["parameter_reference"]["inventory"]} == {"weight"}
    changed_state = cpu_state["model"].state_dict()
    changed_state["weight"] = changed_state["weight"] + 1
    with pytest.raises(evidence.EvidenceError, match="parameter identity"):
        evidence.build_run_binding(
            run_id="wrong-full-state", code_paths={k: v["path"] for k, v in bound["code"].items()},
            config=bound["config"], initial_parameters=initial["parameter_reference"],
            initial_state=evidence.initial_parameter_reference(changed_state),
            synthetic_input=bound["data"]["identity"])


def test_completed_report_rejects_json_fake_checkpoint_despite_consistent_reference(tmp_path, bound, artifacts):
    _, counters, windows = artifacts
    fake = evidence.write_exclusive_json(
        tmp_path / "fake_checkpoint.json", {"binding_sha256": bound["binding_sha256"], "counters": counters})
    fake["binding_sha256"] = bound["binding_sha256"]
    observation = evidence.collect_native_cpu_observation(bound)
    with pytest.raises(evidence.EvidenceError, match="checkpoint payload"):
        evidence.build_completed_report(bound, observation, checkpoint_reference=fake,
                                        counters=counters, window_references=windows)


def test_checkpoint_actual_counters_cannot_be_replaced_with_self_reported_metadata(bound, artifacts):
    checkpoint, counters, windows = artifacts
    fake = {**checkpoint, "microsteps": 6}
    observation = evidence.collect_native_cpu_observation(bound)
    with pytest.raises(evidence.EvidenceError, match="metadata mismatch"):
        evidence.build_completed_report(bound, observation, checkpoint_reference=fake,
                                        counters=counters, window_references=windows)


@pytest.mark.parametrize("field,value", [
    ("optimizer_updates", 2), ("microsteps", 4), ("epoch", 2), ("phase", 1),
    ("logical_batch_size", 32), ("physical_batch_size", 4), ("accumulation_steps", 4),
    ("samples_seen", 17), ("target_total", 15), ("denominator", 15),
    ("microbatch_target_counts", [8]), ("coefficients", [0.25, 0.75]),
    ("microbatch_losses", [{"proxy": 1.0}]), ("loss", 999.0),
    ("gradient_norm_before_clip", -1.0), ("dn_num_groups", [None]),
])
def test_each_window_is_checked_against_checkpoint_and_own_loss_accounting(
        tmp_path, bound, artifacts, field, value):
    checkpoint, counters, windows = artifacts
    replaced = rewrite_windows(tmp_path, windows,
                               lambda document: document["windows"][0].update({field: value}))
    observation = evidence.collect_native_cpu_observation(bound)
    with pytest.raises(evidence.EvidenceError, match="window|batch|phase|gradient"):
        evidence.build_completed_report(bound, observation, checkpoint_reference=checkpoint,
                                        counters=counters, window_references=replaced)


def test_nonfinite_intermediate_window_is_rejected_even_with_matching_file_hash(tmp_path, bound, artifacts):
    checkpoint, counters, windows = artifacts
    document = evidence.strict_json_loads(Path(windows[0]["path"]).read_bytes())
    document["windows"][0]["loss"] = float("nan")
    invalid = tmp_path / "nonfinite_window.json"
    invalid.write_text(json.dumps(document), encoding="utf-8")
    observation = evidence.collect_native_cpu_observation(bound)
    with pytest.raises(evidence.EvidenceError, match="strict JSON"):
        evidence.build_completed_report(bound, observation, checkpoint_reference=checkpoint,
                                        counters=counters, window_references=[evidence.file_reference(invalid)])


def test_window_ledger_cannot_omit_earlier_updates(tmp_path, bound, artifacts):
    checkpoint, counters, windows = artifacts
    replaced = rewrite_windows(tmp_path, windows, lambda document: document["windows"].pop(0))
    observation = evidence.collect_native_cpu_observation(bound)
    with pytest.raises(evidence.EvidenceError, match="contiguous"):
        evidence.build_completed_report(bound, observation, checkpoint_reference=checkpoint,
                                        counters=counters, window_references=replaced)


def test_split_window_ledger_has_verified_chunk_and_final_counters(tmp_path, bound, artifacts):
    checkpoint, counters, windows = artifacts
    document = evidence.strict_json_loads(Path(windows[0]["path"]).read_bytes())
    first = {"run_binding_sha256": bound["binding_sha256"], "windows": document["windows"][:1],
             "counters": {"epoch": 1, "optimizer_updates": 1, "microsteps": 2, "samples_seen": 16}}
    second = {**document, "windows": document["windows"][1:]}
    chunks = [evidence.write_exclusive_json(tmp_path / "chunk1.json", first),
              evidence.write_exclusive_json(tmp_path / "chunk2.json", second)]
    observation = evidence.collect_native_cpu_observation(bound)
    report = evidence.build_completed_report(bound, observation, checkpoint_reference=checkpoint,
                                             counters=counters, window_references=chunks)
    assert report["checkpoint_progress"]["epoch_active"] is False
    assert report["checkpoint_progress"]["config"]["physical_batch_size"] == 8
    changed = copy.deepcopy(report)
    changed["checkpoint_progress"]["epoch_active"] = True
    changed["report_sha256"] = evidence.canonical_sha256({k: v for k, v in changed.items() if k != "report_sha256"})
    with pytest.raises(evidence.EvidenceError, match="progress"):
        evidence.validate_completed_report(changed, binding=bound, observation=observation)


@pytest.mark.parametrize("cpu_state", [True], indirect=True)
def test_true_zero_gradient_training_is_legal_completed_evidence(bound, artifacts):
    checkpoint, counters, windows = artifacts
    rows = evidence.strict_json_loads(Path(windows[0]["path"]).read_bytes())["windows"]
    assert all(row["gradient_norm_before_clip"] == 0 and row["loss"] == 0 for row in rows)
    observation = evidence.collect_native_cpu_observation(bound)
    report = evidence.build_completed_report(bound, observation, checkpoint_reference=checkpoint,
                                             counters=counters, window_references=windows)
    assert report["status"] == "completed"


def test_maxdets_100_precision_cannot_be_relabelled_500(bound, artifacts):
    checkpoint, counters, _ = artifacts
    record = metric(bound, checkpoint, counters, max_dets_index=0)
    assert record["max_dets"] == 100
    assert record["value"] == pytest.approx(10.0)
    changed = {**record, "max_dets": 500}
    with pytest.raises(evidence.EvidenceError, match="max_dets"):
        evidence.validate_metric_record(changed, binding=bound,
                                        checkpoint_reference=checkpoint, counters=counters)
    changed["precision_selection"] = {**record["precision_selection"], "max_dets_index": 1}
    with pytest.raises(evidence.EvidenceError, match="value differs"):
        evidence.validate_metric_record(changed, binding=bound,
                                        checkpoint_reference=checkpoint, counters=counters)


@pytest.mark.parametrize("field,value", [
    ("iou_definition", {"thresholds": [0.5]}),
    ("area", {"label": "small", "range": [0, 10000], "coordinate_space": "original_image_pixels"}),
    ("evaluator_parameters_sha256", "a" * 64),
    ("raw_prediction_sha256_scope", "canonical_prediction_payload"),
    ("stats_index", 0),
])
def test_metric_labels_must_match_bound_axis_parameters(bound, artifacts, field, value):
    checkpoint, counters, _ = artifacts
    record = {**metric(bound, checkpoint, counters), field: value}
    with pytest.raises(evidence.EvidenceError):
        evidence.validate_metric_record(record, binding=bound,
                                        checkpoint_reference=checkpoint, counters=counters)


@pytest.mark.parametrize("which", ["prediction_reference", "source_reference"])
def test_metric_provenance_rechecks_actual_prediction_and_precision_file_bytes(bound, artifacts, which):
    checkpoint, counters, _ = artifacts
    record = metric(bound, checkpoint, counters)
    Path(record[which]["path"]).write_text('{"altered":true}\n')
    with pytest.raises(evidence.EvidenceError, match="size/hash"):
        evidence.validate_metric_record(record, binding=bound,
                                        checkpoint_reference=checkpoint, counters=counters)


def test_source_axis_shape_and_unavailable_sentinel_are_rejected_before_publication(tmp_path, bound, artifacts):
    checkpoint, counters, _ = artifacts
    record = metric(bound, checkpoint, counters)
    source = evidence.strict_json_loads(Path(record["source_reference"]["path"]).read_bytes())
    kwargs = dict(
        parameters=source["parameters"], prediction_reference=record["prediction_reference"],
        binding=bound, checkpoint_reference=checkpoint, counters=counters, weights="raw",
        evaluator_id="synthetic", max_dets_index=1, area_index=1, iou_indices=[0, 1])
    for number, precision in enumerate((
        torch.zeros(2, 3, 1, 2, 1), torch.full((2, 3, 1, 2, 2), -1.0),
        torch.full((2, 3, 1, 2, 2), float("inf")),
    )):
        destination = tmp_path / f"must-not-publish-{number}.json"
        with pytest.raises(evidence.EvidenceError):
            evidence.build_coco_ap_record(precision, source_path=destination, **kwargs)
        assert not destination.exists()


def test_completed_report_carries_a_verified_supplied_cpu_metric(bound, artifacts):
    checkpoint, counters, windows = artifacts
    record = metric(bound, checkpoint, counters)
    observation = evidence.collect_native_cpu_observation(bound)
    report = evidence.build_completed_report(bound, observation, checkpoint_reference=checkpoint,
                                             counters=counters, window_references=windows, metrics=[record])
    assert report["metrics"][0]["source_kind"] == "supplied_cpu_coco_precision"
    assert report["scientific_certified"] is False


@pytest.mark.parametrize("field,value", [
    ("physical_batch_size", 4), ("accumulation_steps", 1),
    ("amp_dtype", "bfloat16"), ("bn_statistics", "frozen"), ("clip_max_norm", 1.0),
    ("logical_batch_size", 8), ("input_size", 4), ("expected_input_size", [4, 4]),
    ("device", "cuda"), ("scope", "production"), ("physical_batch_size", None),
    ("bn_backward_layout", "cpu_native_cuda_native"), ("bn_backward_layout", None),
])
def test_report_rejects_self_consistent_binding_that_mislabels_actual_engine(
        tmp_path, bound, artifacts, cpu_state, field, value):
    _, counters, windows = artifacts
    false_binding = copy.deepcopy(bound)
    if value is None:
        del false_binding["config"][field]
    else:
        false_binding["config"][field] = value
    false_binding["binding_sha256"] = evidence.canonical_sha256(
        {key: item for key, item in false_binding.items() if key != "binding_sha256"})
    if field == "device":
        # Schema2 now rejects device mislabeling before writing any checkpoint,
        # earlier than the completed-report comparison tested below.
        with pytest.raises(CheckpointError, match="runtime device"):
            save_checkpoint(tmp_path / "mislabelled_engine.pt", model=cpu_state["model"],
                            optimizer=cpu_state["optimizer"], engine=cpu_state["engine"],
                            ema=None, binding=false_binding)
        assert not (tmp_path / "mislabelled_engine.pt").exists()
        return
    # This is still an actual CPU checkpoint with a valid matching binding hash.
    # Only the semantic comparison can detect that the declaration is false.
    checkpoint = save_checkpoint(
        tmp_path / "mislabelled_engine.pt", model=cpu_state["model"],
        optimizer=cpu_state["optimizer"], engine=cpu_state["engine"], ema=None,
        binding=false_binding)
    renamed = rewrite_windows(
        tmp_path, windows,
        lambda document: document.update(run_binding_sha256=false_binding["binding_sha256"]))
    observation = evidence.collect_native_cpu_observation(false_binding)
    with pytest.raises(evidence.EvidenceError, match="bound engine"):
        evidence.build_completed_report(false_binding, observation, checkpoint_reference=checkpoint,
                                        counters=counters, window_references=renamed)
