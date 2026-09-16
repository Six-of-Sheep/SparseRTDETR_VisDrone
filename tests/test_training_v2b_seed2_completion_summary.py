"""Synthetic CPU tests; no real endpoint scoring, images, model or GPU work."""
from __future__ import annotations

from collections import Counter
import copy
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
from types import ModuleType

import numpy as np
import pytest
import torch

from sparse_rtdetr.baseline import training_v2b_checkpoint as checkpoint
from sparse_rtdetr.baseline import training_v2b_replication_summary as original
from sparse_rtdetr.baseline import training_v2b_seed2_completion_summary as summary
from sparse_rtdetr.baseline.training_v2b_control import _tensor_identity
from sparse_rtdetr.baseline.training_v2b_device import cpu_runtime_identity, cuda_backend_policy_for_sampling
from sparse_rtdetr.baseline.training_v2b_engine import BN_BACKWARD_LAYOUT
from sparse_rtdetr.baseline.training_v2b_evidence import (
    canonical_sha256, file_reference, initial_parameter_reference, write_exclusive_json,
)


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    def blocked(*args, **kwargs):
        pytest.fail("CPU summary attempted a GPU observation or initialization")
    for name in ("_lazy_init", "init", "is_available", "device_count"):
        monkeypatch.setattr(torch.cuda, name, blocked)


def _identity(value):
    from sparse_rtdetr.baseline.training_v2b_control import _state_identity
    return _state_identity(value)


def _component(kind, state):
    return {"type": kind, "state": state, "structure": checkpoint._structure(state),
            "config": {key: value for key, value in state.items() if key in checkpoint._COMPONENT_CONFIG}}


def fixture_payload(*, stored_cuda=False):
    engine = {
        "schema_version": 1, "epoch": 30, "epoch_active": False, "epoch_start_optimizer_updates": 8816,
        "optimizer_updates": 9120, "microsteps": 18240, "phase": 0, "failed": False,
        "config": {"physical_batch_size": 8, "accumulation_steps": 2, "amp_dtype": "bfloat16",
                   "clip_max_norm": .1, "bn_statistics": "train", "expected_input_size": [896, 896],
                   "bn_backward_layout": BN_BACKWARD_LAYOUT},
    }
    loader = {
        "schema_version": 1, "binding_sha256": "b" * 64, "order_sha256": "c" * 64,
        "epoch": 30, "epoch_active": False, "epoch_start_optimizer_updates": 8816,
        "optimizer_updates": 9120, "completed_batches": 304, "logical_batch_size": 16,
        "dataset_size": 4869, "batches_per_epoch": 304, "dropped_samples": 5,
    }
    binding = {"initial_weights": {"fixture": True}, "data": {"loader_binding_sha256": "b" * 64},
               "code": {"fixture": True}, "config": {"device": "cpu", "input_size": 896, "seed": 2}}
    binding["binding_sha256"] = canonical_sha256(binding)
    state = {
        "head.weight": torch.tensor([.5]),
        "bn.running_mean": torch.tensor([.25]), "bn.running_var": torch.tensor([1.]),
        "bn.num_batches_tracked": torch.tensor(18240),
        "decoder.anchors": torch.tensor([[.5, float("inf")]]),
        "decoder.valid_mask": torch.tensor([[True, False]]),
    }
    layout = {
        "modules": {"": "CPUFixtureOnly"},
        "state": {name: checkpoint._tensor_spec(value) for name, value in state.items()},
        "parameters": [{"name": "head.weight", "shape": [1], "dtype": "torch.float32", "requires_grad": True}],
        "caches": {
            "decoder:persistent_cache:anchors": checkpoint._tensor_spec(state["decoder.anchors"], digest=True),
            "decoder:persistent_cache:valid_mask": checkpoint._tensor_spec(state["decoder.valid_mask"], digest=True),
        },
    }
    options = {"weight_decay": .0001, "amsgrad": False}
    generator = torch.Generator(device="cpu").manual_seed(2)
    numpy = np.random.RandomState(2).get_state()
    scheduler = _component("torch.optim.lr_scheduler.MultiStepLR",
                           {"milestones": {1000: 1}, "gamma": .1, "last_epoch": 24, "_step_count": 25})
    warmup = _component("src.optim.warmup.LinearWarmup", {"last_step": 9120, "warmup_duration": 2000})
    payload = {
        "format": checkpoint.FORMAT, "schema_version": 2, "torch_version": str(torch.__version__),
        "binding": binding, "binding_sha256": binding["binding_sha256"],
        "model": copy.deepcopy(state), "model_layout": copy.deepcopy(layout), "model_training": {"": True},
        "optimizer": {"state": {0: {"step": torch.tensor(9120.), "exp_avg": torch.tensor([0.]),
                                   "exp_avg_sq": torch.tensor([1.])}},
                      "param_groups": [{"params": [0], "lr": .0001, **options}]},
        "optimizer_layout": {"type": "torch.optim.adamw.AdamW", "defaults": {},
                             "groups": [{"names": ["head.weight"], "ids": [0], "options": options}]},
        "ema": {"module": state, "updates": 9120},
        "ema_layout": {"type": "CPUFixtureEMA", "model": layout, "config": {"decay": .9999, "warmups": 2000}},
        "engine": engine, "engine_type": "CPUFixtureEngine",
        "scheduler": scheduler, "warmup": warmup,
        "runtime": cpu_runtime_identity(), "sampler": {"type": "CPUFixtureLoader", "state": loader},
        "rng": {"python": random.Random(2).getstate(),
                "numpy": {"algorithm": "MT19937", "keys": torch.tensor(numpy[1].astype(np.int64)),
                          "position": int(numpy[2]), "has_gauss": int(numpy[3]),
                          "cached_gaussian": float(numpy[4])},
                "torch_cpu": generator.get_state(), "cuda": {}, "generators": {}},
    }
    if stored_cuda:
        uuid = "GPU-12345678-1234-1234-1234-123456789abc"
        policy = cuda_backend_policy_for_sampling("deterministic_gather")
        payload["runtime"].update(
            device="cuda:0", cuda_build_version="12.1", cuda_visible_devices=uuid,
            cuda_devices=[{"index": 0, "uuid": uuid, "name": "SYNTHETIC_GPU_METADATA",
                           "total_memory": 24 * 1024**3, "capability": [8, 6], "rng_state_bytes": 16}],
            backend_policy=policy)
        payload["binding"]["config"].update(device="cuda:0", cuda_gpu_uuid=uuid,
                                             sampling_backend="deterministic_gather", cuda_backend_policy=policy)
        body = {key: value for key, value in payload["binding"].items() if key != "binding_sha256"}
        payload["binding"]["binding_sha256"] = canonical_sha256(body)
        payload["binding_sha256"] = payload["binding"]["binding_sha256"]
        payload["rng"]["cuda"] = {"0": torch.arange(16, dtype=torch.uint8)}
    return payload


def bridge_fixture(tmp_path, *, mutation=None, stored_cuda=False):
    payload = fixture_payload(stored_cuda=stored_cuda)
    expected = copy.deepcopy(payload)
    digest = initial_parameter_reference(payload["ema"]["module"])
    if mutation is not None:
        mutation(payload)
    path = tmp_path / "checkpoint-epoch-030.pt"
    torch.save(payload, path)
    checkpoint_ref = file_reference(path)
    evaluation = {
        "role": "development", "logged_epoch": 30, "full_development_evaluation": True,
        "policy": {"input_size": [896, 896]},
        "weights": {"kind": "ema", "ema_updates": 9120, "state_sha256": digest["sha256"]},
        "training_state_unchanged": {"ema_updates": 9120, "ema_sha256": digest["sha256"]},
    }
    evaluation["result_artifact"] = write_exclusive_json(tmp_path / "development.json", evaluation)
    final = {
        "clocks": {"engine": expected["engine"], "loader": expected["sampler"]["state"],
                   "ema_updates": 9120, "warmup": _identity(expected["warmup"]["state"]),
                   "scheduler": _identity({**expected["scheduler"]["state"], "milestones": Counter({1000: 1})}),
                   "learning_rates": [.0001], "adam_parameter_count_with_state": 1, "adam_step_values": [9120],
                   "adam_steps_by_parameter": {"head.weight": 9120}},
        "rng": {
            "python": canonical_sha256(_identity(expected["rng"]["python"])),
            "numpy": canonical_sha256(_identity(checkpoint._numpy_state(expected["rng"]["numpy"]))),
            "torch_cpu": _tensor_identity(expected["rng"]["torch_cpu"]),
            "cuda": {key: _tensor_identity(value) for key, value in expected["rng"]["cuda"].items()},
        },
        "raw_parameters": initial_parameter_reference({"head.weight": expected["model"]["head.weight"]}),
    }
    source = {
        "seed": 2, "input_size": 896, "checkpoint_reference": checkpoint_ref,
        "binding": expected["binding"], "evaluation": evaluation, "final": final,
        "result": {"runtime": expected["runtime"], "checkpoints": {"epoch_30": {"checkpoint": checkpoint_ref}}},
    }
    return source, payload


def test_real_cpu_checkpoint_semantics_and_all_ema_buffers_are_bridged(tmp_path, monkeypatch):
    source, payload = bridge_fixture(tmp_path)
    observed = []
    native = checkpoint.inspect_checkpoint
    def inspection(*args, **kwargs):
        observed.append("unchanged_native_payload_inspection")
        return native(*args, **kwargs)
    monkeypatch.setattr(checkpoint, "inspect_checkpoint", inspection)
    result = summary.audit_checkpoint_ema_evaluation(summary._Reader(), source)
    row = result["cells"][0]
    assert observed == ["unchanged_native_payload_inspection"]
    assert result["status"] == "PASS_CHECKPOINT_EMA_EVALUATION_BINDING"
    assert row["module_state_reference"] == initial_parameter_reference(payload["ema"]["module"])
    assert row["tensor_count"] == 6 and row["bn_buffer_counts"] == {
        "running_mean": 1, "running_var": 1, "num_batches_tracked": 1}
    assert row["anchors"] and row["valid_masks"]
    assert row["all_tensors_cpu"] and row["checkpoint_bytes_verified_before_and_after_load"]
    assert result["ema_wrapper_digest_compared"] is False
    assert result["runtime"]["forbidden_calls"] == []
    assert result["runtime"]["cuda_initialized_before"] is False
    assert result["runtime"]["cuda_initialized_after"] is False


@pytest.mark.parametrize("tensor", ["head.weight", "bn.running_mean", "bn.num_batches_tracked"])
def test_changed_ema_parameter_or_buffer_cannot_hide_behind_checkpoint_sha(tmp_path, tensor):
    def change(payload):
        payload["ema"]["module"][tensor].add_(1)
    source, payload = bridge_fixture(tmp_path, mutation=change)
    with pytest.raises(original.ReplicationSummaryError, match="EMA module tensors"):
        summary.audit_checkpoint_ema_evaluation(summary._Reader(), source)


@pytest.mark.parametrize("mutation", ["optimizer_future", "rng_shape", "schema", "ema_updates", "binding", "epoch"])
def test_original_payload_failures_remain_failures(tmp_path, mutation):
    def change(payload):
        if mutation == "optimizer_future":
            payload["optimizer"]["state"][0]["step"] = torch.tensor(9121.)
        elif mutation == "rng_shape":
            payload["rng"]["torch_cpu"] = torch.zeros(1, dtype=torch.uint8)
        elif mutation == "schema":
            payload["schema_version"] = 99
        elif mutation == "ema_updates":
            payload["ema"]["updates"] = 9119
        elif mutation == "binding":
            payload["binding"]["config"]["seed"] = 1
        else:
            payload["engine"]["epoch"] = 29
    source, payload = bridge_fixture(tmp_path, mutation=change)
    with pytest.raises((checkpoint.CheckpointError, original.ReplicationSummaryError)):
        summary.audit_checkpoint_ema_evaluation(summary._Reader(), source)


@pytest.mark.parametrize("value", [float("nan"), float("-inf")])
def test_illegal_anchor_values_are_rejected_by_native_semantics(tmp_path, value):
    def change(payload):
        payload["ema"]["module"]["decoder.anchors"][0, 1] = value
    source, payload = bridge_fixture(tmp_path, mutation=change)
    with pytest.raises(checkpoint.CheckpointError, match="anchor"):
        summary.audit_checkpoint_ema_evaluation(summary._Reader(), source)


def test_checkpoint_bytes_are_checked_after_both_cpu_loads(tmp_path):
    source, payload = bridge_fixture(tmp_path)
    class ChangingReader(summary._Reader):
        def __init__(self):
            super().__init__()
            self.checkpoint_reads = 0
        def read(self, reference, **kwargs):
            if reference["path"] == source["checkpoint_reference"]["path"]:
                self.checkpoint_reads += 1
                if self.checkpoint_reads == 2:
                    with Path(reference["path"]).open("ab") as stream:
                        stream.write(b"changed-after-load")
            return super().read(reference, **kwargs)
    with pytest.raises(original.ReplicationSummaryError, match="bound regular|SHA|size"):
        summary.audit_checkpoint_ema_evaluation(ChangingReader(), source)


@pytest.mark.parametrize("action", ["cuda", "model", "file", "socket", "process"])
def test_scoped_bridge_guards_forbid_unrelated_work_and_restore_their_patches(tmp_path, action):
    before = torch.nn.Parameter.__dict__["__new__"]
    with pytest.raises(original.ReplicationSummaryError, match="forbids"):
        with summary._checkpoint_cpu_guard(tmp_path / "checkpoint.pt"):
            if action == "cuda":
                torch.cuda.device_count()
            elif action == "model":
                torch.nn.Linear(1, 1)
            elif action == "file":
                (tmp_path / "synthetic-image.jpg").read_bytes()
            elif action == "socket":
                socket.socket()
            else:
                subprocess.run([sys.executable, "-V"], check=True)
    assert torch.nn.Parameter.__dict__["__new__"] is before
    assert summary._AUDIT_SCOPES == []


def test_even_a_caught_guard_violation_cannot_produce_pass(tmp_path):
    with pytest.raises(original.ReplicationSummaryError, match="guard was violated"):
        with summary._checkpoint_cpu_guard(tmp_path / "checkpoint.pt"):
            try:
                torch.cuda.device_count()
            except original.ReplicationSummaryError:
                pass


@pytest.mark.parametrize("case", ["bad_sha", "forbidden", "epoch_override", "visible_cuda", "old_input"])
def test_cli_rejects_ungranted_scope_before_torch_or_research_imports(tmp_path, case):
    cli = Path(summary.__file__).resolve().parents[3] / "tools/summarize_training_v2b_seed2_completion.py"
    path = tmp_path / "new-completion.json"
    path.write_text("{}")
    args = ["--completion", str(path), "--completion-sha256", file_reference(path)["sha256"],
            "--output-dir", str(tmp_path / "summary")]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    if case == "bad_sha":
        args[3] = "bad"
    elif case == "forbidden":
        args[1] = str(tmp_path / "confirmatory" / "never-open.json")
    elif case == "epoch_override":
        args += ["--epoch", "60"]
    elif case == "visible_cuda":
        env["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        args[0] = "--campaign-result"
    wrapper = (
        "import runpy,sys\n"
        "class Block:\n"
        " def find_spec(self,name,path=None,target=None):\n"
        "  if name=='torch' or name.startswith('sparse_rtdetr'):\n"
        "   raise AssertionError('UNAUTHORIZED_IMPORT')\n"
        "sys.meta_path.insert(0,Block())\n"
        "sys.argv=sys.argv[1:]\n"
        "runpy.run_path(sys.argv[0],run_name='__main__')\n"
    )
    run = subprocess.run([sys.executable, "-B", "-c", wrapper, str(cli), *args],
                         env=env, capture_output=True, text=True, timeout=20)
    assert run.returncode == 2
    assert "UNAUTHORIZED_IMPORT" not in run.stdout + run.stderr


def _ref(path):
    return {"path": str(path), "sha256": "a" * 64, "size_bytes": 1, "sha256_scope": "complete_file_bytes"}


def _source(seed, size, root, campaign="historical"):
    return {"seed": seed, "input_size": size, "result": {"campaign_id": campaign},
            "result_reference": _ref(root / f"s{seed}_r{size}" / "worker-result.json"),
            "contract": {"output_dir": str(root / f"s{seed}_r{size}")},
            "checkpoint_reference": _ref(root / f"s{seed}_r{size}" / "checkpoint-epoch-030.pt"),
            "evaluation": {"fixture": True}}


def context_fixture(monkeypatch, tmp_path):
    from sparse_rtdetr.baseline import training_v2b_campaign as campaign
    root = tmp_path / "new-campaign"
    root.mkdir()
    sources = {(seed, size): _source(seed, size, root) for seed in summary.SEEDS for size in summary.SIZES}
    qualification = {"status": "PASS", "eligible_for_fixed_paired_seed1_and_seed2": True,
                     "gpu_admission_granted": False,
                     "conditional_authorization_reference": _ref(tmp_path / "authorization.txt")}
    qual_ref = write_exclusive_json(tmp_path / "qualification.json", qualification)
    old = {"authorization_reference": qualification["conditional_authorization_reference"],
           "cells": {cell: {"config": {"fixture": cell}} for cell in original.CELLS}}
    old_ref = write_exclusive_json(tmp_path / "old-freeze.json", old)
    rows = [{"cell_id": cell, "stage": stage,
             "worker_result_reference": sources[int(cell[1]), int(cell.split("_r")[1])]["result_reference"],
             "completion_reference": _ref(tmp_path / f"{cell}-{stage}.completion.json")}
            for cell in original.CELLS for stage in ("capacity", "smoke", "smoke_replay", "control30")
            if (cell, stage) != ("s2_r896", "control30")]
    failure = {"status": "STOP_NO_RETRY", "completed": rows}
    failure_ref = write_exclusive_json(tmp_path / "old-failure.json", failure)
    partial = {"three_pairs_complete": False}
    partial_ref = write_exclusive_json(tmp_path / "partial.json", partial)
    freeze = {
        "repo_root": str(Path(summary.__file__).resolve().parents[3]), "output_root": str(root),
        "campaign_id": "new-completion", "source": {"fixture": True}, "qualification_reference": qual_ref,
        "cells": {"s2_r896": {"config": {"seed": 2, "input_size": 896}}},
        "historical": {
            "freeze_reference": old_ref, "control_result_reference": sources[2, 640]["result_reference"],
            "failed_campaign_reference": failure_ref, "partial_summary_reference": partial_ref,
            "historical_capacity_reference": _ref(tmp_path / "original-capacity.json"),
        },
    }
    freeze_ref = write_exclusive_json(root / "freeze.json", freeze)
    stage_gate_ref = write_exclusive_json(root / "stage-gate.json", {"prerequisites": [], "smoke_comparisons": {}})
    contract = {"fixture": True, "stage_gate_reference": stage_gate_ref}
    contract_ref = write_exclusive_json(root / "contract.json", contract)
    result = {"kind": summary.NEW_RESULT_KIND, "stage": "control30", "status": "PASS",
              "campaign_id": freeze["campaign_id"],
              "historical_control_reference": sources[2, 640]["result_reference"]}
    result_ref = write_exclusive_json(root / "worker-result.json", result)
    completion = {"kind": summary.NEW_COMPLETION_KIND, "status": "PASS"}
    completion_ref = write_exclusive_json(root / "completion.json", completion)
    checked = {
        "freeze": freeze, "freeze_reference": freeze_ref, "result": result, "result_reference": result_ref,
        "contract": contract, "contract_reference": contract_ref,
        "completion": completion, "completion_reference": completion_ref,
        "historical": {"freeze": old, "qualification": qualification, "failed_campaign": failure,
                       "partial_summary": partial, "verified_leaf_references": [partial_ref]},
    }
    calls = []
    module = ModuleType("sparse_rtdetr.baseline.training_v2b_seed2_completion_worker")
    def verify(reference, *, expected_stage, verify_files):
        calls.append(("new_completion_gate", reference, expected_stage, verify_files))
        return checked
    module.validate_completed_seed2_896 = verify
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(campaign, "validate_frozen_source", lambda source: calls.append(("frozen_source", source)))
    def historical(reader, entry, old_freeze):
        calls.append(("historical_source", entry["cell_id"]))
        return sources[int(entry["cell_id"][1]), int(entry["cell_id"].split("_r")[1])]
    monkeypatch.setattr(summary, "_historical_source", historical)
    monkeypatch.setattr(summary, "_record_completion", lambda *a: None)
    monkeypatch.setattr(summary, "_training_source", lambda *a, **kw: {
        **sources[2, 896], "result": result, "result_reference": result_ref})
    monkeypatch.setattr(summary, "_seed0_sources", lambda *a: (
        {(0, size): sources[0, size] for size in summary.SIZES}, {"fixture": True}, {"fixture": True}))
    monkeypatch.setattr(summary, "_historical_ema_binding", lambda *a: {"fixture": "old-CPU-audit"})
    monkeypatch.setattr(original, "_load_context", lambda *a: pytest.fail("old full-PASS loader invoked"))
    monkeypatch.setattr(original, "_completed_invocations", lambda *a: pytest.fail("old sixteen-stage gate invoked"))
    return checked, calls, module


def test_new_gate_builds_six_sources_without_promoting_old_campaign(monkeypatch, tmp_path):
    checked, calls, module = context_fixture(monkeypatch, tmp_path)
    reader = summary._Reader()
    context = summary._load_context(reader, checked["completion_reference"])
    assert calls[0] == ("new_completion_gate", checked["completion_reference"], "control30", True)
    assert set(context["sources"]) == {(seed, size) for seed in summary.SEEDS for size in summary.SIZES}
    assert context["sources"][2, 640]["result"]["campaign_id"] == "historical"
    assert context["sources"][2, 896]["result"]["campaign_id"] == "new-completion"
    assert context["historical"]["failed_campaign"]["status"] == "STOP_NO_RETRY"
    assert len(context["historical_ema_bindings"]) == 5
    assert checked["freeze"]["historical"]["partial_summary_reference"] in reader.inventory()


def test_missing_new_completion_fails_before_historical_or_scoring(monkeypatch, tmp_path):
    checked, calls, module = context_fixture(monkeypatch, tmp_path)
    def reject(*args, **kwargs):
        raise original.ReplicationSummaryError("new completion absent")
    module.validate_completed_seed2_896 = reject
    with pytest.raises(original.ReplicationSummaryError, match="new completion absent"):
        summary._load_context(summary._Reader(), checked["completion_reference"])
    assert calls == []


@pytest.mark.parametrize("mutation", ["promote_failure", "partial_claim", "missing", "duplicate", "wrong_new_kind"])
def test_historical_partial_or_incomplete_set_cannot_be_aggregated(monkeypatch, tmp_path, mutation):
    checked, calls, module = context_fixture(monkeypatch, tmp_path)
    if mutation == "promote_failure":
        checked["historical"]["failed_campaign"]["status"] = "PASS"
    elif mutation == "partial_claim":
        checked["historical"]["partial_summary"]["three_pairs_complete"] = True
    elif mutation == "missing":
        checked["historical"]["failed_campaign"]["completed"].pop()
    elif mutation == "duplicate":
        rows = checked["historical"]["failed_campaign"]["completed"]
        rows[-1] = rows[0]
    else:
        checked["result"]["kind"] = "v2b_seed_resolution_replication_worker_result"
        checked["result_reference"] = write_exclusive_json(tmp_path / "wrong-kind.json", checked["result"])
    with pytest.raises(original.ReplicationSummaryError):
        summary._load_context(summary._Reader(), checked["completion_reference"])


def test_existing_historical_ema_report_is_authenticated_without_unpickling(tmp_path, monkeypatch):
    source, payload = bridge_fixture(tmp_path)
    report = summary.audit_checkpoint_ema_evaluation(summary._Reader(), source)
    report["runtime"]["forbidden_cuda_calls"] = report["runtime"].pop("forbidden_calls")
    manifest = write_exclusive_json(tmp_path / "manifest.json", {"synthetic_manifest": True})
    script = write_exclusive_json(tmp_path / "script-fixture.json", {"synthetic_verifier": True})
    report.update(manifest_reference=manifest, script_reference=script,
                  input_references=[source["checkpoint_reference"], source["evaluation"]["result_artifact"]])
    report_ref = write_exclusive_json(tmp_path / "old-ema-report.json", report)
    partial = {"ema_binding_audits": {"s2_r896": {
        "checkpoint_reference": source["checkpoint_reference"], "report_reference": report_ref,
        "manifest_reference": manifest, "script_reference": script,
        "module_sha256": report["cells"][0]["module_state_reference"]["sha256"]}}}
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("historical checkpoint deserialized"))
    reader = summary._Reader()
    result = summary._historical_ema_binding(reader, source, partial)
    assert result["source_report_and_full_tensor_inventory_authenticated"] is True
    assert result["new_checkpoint_deserialization"] is False
    assert set(reader.references) == {ref["path"] for ref in (
        report_ref, manifest, script, source["checkpoint_reference"], source["evaluation"]["result_artifact"])}
    partial["ema_binding_audits"]["s2_r896"]["module_sha256"] = "f" * 64
    with pytest.raises(original.ReplicationSummaryError, match="full EMA module"):
        summary._historical_ema_binding(summary._Reader(), source, partial)


def _endpoint(seed, size):
    delta = (1., -2., 3.)[seed] if size == 896 else 0.
    return {
        "seed": seed, "input_size": size, "epoch": 30,
        "uniform_official_primary": {"metrics": {name: 30. + delta for name in original.PRIMARY_METRICS}},
        "legacy_primary_reported_separately": {"metrics": {name: 25. - delta for name in original.PRIMARY_METRICS}},
        "converted_COCO_reported_separately": {"metrics": {
            name: (10. + 2 * delta if name == "AP_small" else 20. + delta) for name in original.COCO_METRICS}},
    }


def builder_fixture(monkeypatch, tmp_path, *, bridge_failure=False):
    checked, calls, module = context_fixture(monkeypatch, tmp_path)
    context = summary._load_context(summary._Reader(), checked["completion_reference"])
    monkeypatch.setattr(summary, "_load_context", lambda *a: context)
    def bridge(*args):
        if bridge_failure:
            raise original.ReplicationSummaryError("EMA mismatch")
        return {"status": "PASS_CHECKPOINT_EMA_EVALUATION_BINDING",
                "cells": [{"module_state_reference": {"sha256": "e" * 64}}]}
    monkeypatch.setattr(summary, "audit_checkpoint_ema_evaluation", bridge)
    monkeypatch.setattr(summary, "audit_training_pair", lambda *a, **k: {
        "status": "PASS", "resources": {"640": {"fixture": True}, "896": {"fixture": True}}})
    monkeypatch.setattr(summary, "_official_context", lambda *a: {
        "binding": {"fixture": True}, "adapter_current": {"fixture": True}, "development_rebinding": {"fixture": True}})
    monkeypatch.setattr(summary, "_score_endpoint", lambda reader, source, *a: _endpoint(source["seed"], source["input_size"]))
    return context


def test_builder_preserves_old_origin_and_uses_descriptive_three_pair_statistics(monkeypatch, tmp_path):
    context = builder_fixture(monkeypatch, tmp_path)
    output = Path(context["freeze"]["output_root"]) / "summary"
    result = summary.build_seed2_completion_summary(completion_reference=context["completion_reference"],
                                                    output_dir=str(output))
    report = result["summary"]
    assert json.loads((output / "summary.json").read_bytes()) == report
    assert report["status"] == "PASS" and report["three_pairs_complete"] is True
    assert report["historical_campaign_status"] == "STOP_NO_RETRY"
    assert report["old_16_invocation_PASS_gate_invoked"] is False
    assert report["endpoints"]["s2_r640"]["origin_campaign_id"] == "historical"
    assert report["endpoints"]["s2_r640"]["is_new_campaign_training_product"] is False
    assert report["endpoints"]["s2_r896"]["is_new_campaign_training_product"] is True
    stats = report["statistics"]
    assert stats["statistical_significance_claimed"] is False
    assert stats["seed0_was_used_for_conditional_replication_qualification"] is True
    primary = stats["mean_sample_sd_and_range"]["uniform_official_primary"]["AP"]
    assert primary["paired_difference_896_minus_640_pp"]["values"] == [1., -2., 3.]
    assert primary["paired_difference_896_minus_640_pp"]["sample_sd"] == pytest.approx((19 / 3) ** .5)
    assert stats["paired_seed_results"][2]["differences"]["converted_COCO"]["AP_small"] == 6.
    assert "AP_small" not in stats["mean_sample_sd_and_range"]["uniform_official_primary"]
    assert result["summary_reference"] == file_reference(output / "summary.json")
    with pytest.raises(FileExistsError):
        summary.build_seed2_completion_summary(completion_reference=context["completion_reference"], output_dir=str(output))


def test_failed_ema_bridge_leaves_one_immutable_failure_and_no_scores(monkeypatch, tmp_path):
    context = builder_fixture(monkeypatch, tmp_path, bridge_failure=True)
    output = Path(context["freeze"]["output_root"]) / "failure-summary"
    monkeypatch.setattr(summary, "_score_endpoint", lambda *a: pytest.fail("scored before EMA bridge"))
    with pytest.raises(original.ReplicationSummaryError, match="EMA mismatch"):
        summary.build_seed2_completion_summary(completion_reference=context["completion_reference"], output_dir=str(output))
    report = json.loads((output / "summary.json").read_bytes())
    assert report["status"] == "STOP_NO_RETRY" and report["three_pairs_complete"] is False
    assert report["endpoints"] == {} and report["historical_campaign_status"] == "STOP_NO_RETRY"


def test_scoped_file_descriptor_allowance_is_bound_to_only_checkpoint_inode(tmp_path):
    other = tmp_path / "other.json"
    other.write_text("{}")
    descriptor = os.open(other, os.O_RDONLY)
    try:
        with pytest.raises(original.ReplicationSummaryError, match="forbids open"):
            with summary._checkpoint_cpu_guard(tmp_path / "checkpoint.pt"):
                os.fdopen(descriptor, "rb", closefd=False)
    finally:
        os.close(descriptor)


@pytest.mark.parametrize("field", ["scheduler", "learning_rate", "adam_lag", "valid_rng_change", "raw_parameter"])
def test_valid_payload_still_must_match_final_clocks_rng_and_parameters(tmp_path, field):
    def change(payload):
        if field == "scheduler":
            state = {**payload["scheduler"]["state"], "last_epoch": 23, "_step_count": 24}
            payload["scheduler"] = _component("torch.optim.lr_scheduler.MultiStepLR", state)
        elif field == "learning_rate":
            payload["optimizer"]["param_groups"][0]["lr"] = .0002
        elif field == "adam_lag":
            payload["optimizer"]["state"][0]["step"] = torch.tensor(9119.)
        elif field == "valid_rng_change":
            payload["rng"]["torch_cpu"] = torch.Generator(device="cpu").manual_seed(99).get_state()
        else:
            payload["model"]["head.weight"].add_(1)
    source, payload = bridge_fixture(tmp_path, mutation=change)
    with pytest.raises(original.ReplicationSummaryError, match="payload/final"):
        summary.audit_checkpoint_ema_evaluation(summary._Reader(), source)


def test_stored_cuda_runtime_and_rng_are_checked_as_cpu_metadata_without_device_calls(tmp_path):
    source, payload = bridge_fixture(tmp_path, stored_cuda=True)
    assert payload["runtime"]["device"] == "cuda:0"
    result = summary.audit_checkpoint_ema_evaluation(summary._Reader(), source)
    assert result["status"] == "PASS_CHECKPOINT_EMA_EVALUATION_BINDING"
    assert result["runtime"]["forbidden_calls"] == []
    assert result["cells"][0]["all_tensors_cpu"] is True
    assert result["native_payload_semantics"]["all_saved_RNG_bytes_match_final"] is True
    assert result["native_payload_semantics"]["CUDA_RNG_restored"] is False


def test_new_completion_inventory_authenticates_settled_stdout_and_stderr(tmp_path):
    def artifact(name, body=None):
        return write_exclusive_json(tmp_path / name, {"fixture": True} if body is None else body)
    logs = {stream: artifact(stream + ".json") for stream in ("stdout", "stderr")}
    capture = artifact("log-capture.json", {"references": logs})
    result = {"binding_reference": artifact("binding.json")}
    completion = {
        "launch_reference": artifact("launch.json"),
        "monitor": {name: artifact(name + ".json") for name in original.NATIVE_LEAVES},
        "capacity_memory_audit_reference": None,
        "log_capture_reference": capture, "exit_reference": artifact("exit.json"),
    }
    completion["monitor"]["final_report_reference"] = artifact("monitor-final.json")
    completion_ref = artifact("completion.json", completion)
    reader = summary._Reader()
    summary._record_completion(reader, result, completion, completion_ref)
    assert logs["stdout"] in reader.inventory() and logs["stderr"] in reader.inventory()
    Path(logs["stderr"]["path"]).write_text("mutated closed pipe log")
    with pytest.raises(original.ReplicationSummaryError):
        summary._record_completion(summary._Reader(), result, completion, completion_ref)


def test_new_smoke_replay_byte_inventory_does_not_deserialize_its_checkpoints(tmp_path, monkeypatch):
    def artifact(name, body=None):
        return write_exclusive_json(tmp_path / name, {"fixture": True} if body is None else body)
    contract_ref = artifact("smoke-contract.json", {"stage_gate_reference": artifact("smoke-gate.json")})
    checkpoint_ref = artifact("smoke-checkpoint-byte-fixture.json")
    result = {
        "contract_reference": contract_ref, "initial_state_reference": artifact("initial.json"),
        "final_state_reference": artifact("final.json"), "receipts": [artifact("window.json")],
        "checkpoints": {"window_4": {"checkpoint": checkpoint_ref, "state_reference": artifact("boundary.json")}},
    }
    result_ref = artifact("smoke-result.json", result)
    completion_ref = artifact("smoke-completion.json")
    comparison_ref = artifact("comparison.json")
    gate_ref = artifact("formal-gate.json", {
        "prerequisites": [{"worker_result_reference": result_ref, "completion_reference": completion_ref}],
        "smoke_comparisons": {"s2_r896": comparison_ref},
    })
    monkeypatch.setattr(summary, "_record_completion", lambda *a: None)
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("smoke checkpoint deserialized"))
    reader = summary._Reader()
    summary._record_new_chain(reader, {"contract": {"stage_gate_reference": gate_ref}})
    assert checkpoint_ref in reader.inventory() and comparison_ref in reader.inventory()
    assert result["receipts"][0] in reader.inventory()
