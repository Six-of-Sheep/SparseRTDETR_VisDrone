"""CPU-only invariants for the isolated paired seed/resolution capacity diagnostic.

Fixtures are synthetic; the immutable historical metadata is separately verified
by the CPU evidence report. No fixture grants hardware admission.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import gzip
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sparse_rtdetr.baseline import training_v2b_replication_capacity as capacity
from sparse_rtdetr.baseline.training_v2b import V2BConfig, build_v2b_components, logical_batch_indices
from sparse_rtdetr.baseline.training_v2b_data import _sample_seed
from sparse_rtdetr.baseline.training_v2b_evidence import (
    canonical_json_bytes, canonical_sha256, file_reference, strict_json_loads, write_exclusive_json,
)

ROOT = Path(__file__).resolve().parents[1]
GPU = "GPU-1faee6f0-1da7-4ede-2475-67a5a00274a8"


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    torch.set_num_threads(2)
    def forbidden(*args, **kwargs):
        raise AssertionError("capacity CPU tests cannot initialize CUDA")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    yield
    assert not torch.cuda.is_initialized()


def ref(path, sha="a" * 64, size=1):
    return {"path": str(path), "sha256": sha, "size_bytes": size,
            "sha256_scope": "complete_file_bytes"}


def rehash(plan):
    plan["plan_sha256"] = canonical_sha256({k: v for k, v in plan.items() if k != "plan_sha256"})
    return plan


def fixture_plan(tmp_path, *, seed=1, size=896):
    windows = []
    for epoch, index in [(1, n) for n in range(4)] + list(capacity.STRESS_POSITIONS[seed]):
        indices = list(logical_batch_indices(4869, seed=seed, epoch=epoch, logical_batch_size=16))[index]
        samples = []
        for offset, dataset_index in enumerate(indices):
            stable = f"{dataset_index:064x}"
            position = index * 16 + offset
            samples.append({
                "augmentation_seed": _sample_seed(seed, epoch, position, stable),
                "dataset_index": dataset_index, "image_id": dataset_index + 1,
                "image_sha256": "b" * 64, "order_position": position, "raw_gt_count": 1,
                "relative_path": f"train_core/images/{dataset_index}.jpg",
                "stable_image_id": stable,
            })
        windows.append({
            "epoch_index": epoch - 1, "epoch_logged": epoch, "logical_batch_index": index,
            "logical_sum_gt": 16, "physical": [capacity.physical_bounds([1]*8, n) for n in (0, 1)],
            "position_start": index * 16, "samples": samples, "seed": seed,
            "selection_reason": ["synthetic fixture"],
            "update_ordinal": (epoch - 1) * 304 + index + 1,
        })
    metadata = {
        "fixed_windows": windows, "annotation": ref(tmp_path / "train_core_coco.json"),
        "manifest": ref(tmp_path / "train_core_manifest.json"), "sources": {},
        "counts_in_loader_order_sha256": canonical_sha256([1] * 4869),
        "epoch_orders": [
            {"epoch_logged": epoch, "logical_order_sha256": canonical_sha256([
                list(batch) for batch in logical_batch_indices(
                    4869, seed=seed, epoch=epoch, logical_batch_size=16)])}
            for epoch in range(1, 31)],
    }
    return capacity._derive_plan(
        seed, size, ref(tmp_path / "original-report.json", capacity.METADATA_REPORT_SHA256),
        ref(tmp_path / "original-plan.json", capacity.METADATA_PLAN_SHA256[seed]),
        {"seed_summaries": {str(seed): copy.deepcopy(capacity.SEED_BOUNDS[seed])}}, metadata)


@pytest.fixture
def plan(tmp_path):
    return fixture_plan(tmp_path)


@pytest.mark.parametrize("seed,size", [(1, 640), (1, 896), (2, 640), (2, 896)])
def test_plan_exact_one_based_orders_and_integer_seeds(tmp_path, seed, size):
    value = fixture_plan(tmp_path, seed=seed, size=size)
    assert capacity.validate_replication_capacity_plan(
        value, seed=seed, input_size=size, verify_files=False) == value
    assert len(value["fixed_windows"]) == 7
    assert len(value["epoch_order_sha256"]) == 30
    seeds = [s["augmentation_seed"] for row in value["fixed_windows"] for s in row["samples"]]
    assert any(n > 2**53 for n in seeds)
    roundtrip = strict_json_loads(canonical_json_bytes(value))
    assert [s["augmentation_seed"] for row in roundtrip["fixed_windows"] for s in row["samples"]] == seeds
    assert value["execution_policy"]["source_epoch_is_one_based"] is True


@pytest.mark.parametrize("field,value", [
    ("seed", 0), ("seed", True), ("input_size", 960),
    ("physical_batch_size", 4), ("accumulation_steps", 4),
    ("logical_batch_size", 8), ("epochs", 10),
])
def test_plan_rejects_design_drift_even_after_rehash(plan, field, value):
    plan[field] = value
    with pytest.raises(capacity.ReplicationCapacityError):
        capacity.validate_replication_capacity_plan(rehash(plan), verify_files=False)


@pytest.mark.parametrize("mutation", [
    "rounded_seed", "float_seed", "wrong_epoch", "wrong_index", "wrong_order_hash",
    "lower_margin", "lower_native_margin", "fewer_windows", "synthetic_less_dense",
    "changed_bounds", "changed_dn_bound", "development_subset", "wrong_workers", "allow_checkpoint",
])
def test_plan_rejects_rehashed_scientific_or_safety_drift(plan, mutation):
    row = plan["fixed_windows"][0]
    if mutation == "rounded_seed":
        sample = next(s for r in plan["fixed_windows"] for s in r["samples"]
                      if s["augmentation_seed"] != int(float(s["augmentation_seed"])))
        sample["augmentation_seed"] = int(float(sample["augmentation_seed"]))
    elif mutation == "float_seed":
        row["samples"][0]["augmentation_seed"] = float(row["samples"][0]["augmentation_seed"])
    elif mutation == "wrong_epoch":
        row["epoch_index"] = row["epoch_logged"]
    elif mutation == "wrong_index":
        row["samples"][0]["dataset_index"] += 1
    elif mutation == "wrong_order_hash":
        plan["epoch_order_sha256"]["30"] = "f" * 64
    elif mutation == "lower_margin":
        plan["resource_gate"]["minimum_conservative_free_mib"] = 3071
    elif mutation == "lower_native_margin":
        plan["resource_gate"]["minimum_sampled_native_free_mib"] = 3071
    elif mutation == "fewer_windows":
        plan["fixed_windows"].pop()
    elif mutation == "synthetic_less_dense":
        plan["synthetic_envelope"]["target_counts"][8] = 461
    elif mutation == "changed_bounds":
        plan["bounds"]["max_per_logical"] -= 1
    elif mutation == "changed_dn_bound":
        row["physical"][0]["dn_padded_queries_after_removal_upper"] = 100
    elif mutation == "development_subset":
        plan["execution_policy"]["full_development_images"] = 16
    elif mutation == "wrong_workers":
        plan["execution_policy"]["num_workers"] = 0
    else:
        plan["execution_policy"]["checkpoints_created"] = True
    with pytest.raises(capacity.ReplicationCapacityError):
        capacity.validate_replication_capacity_plan(rehash(plan), verify_files=False)


def test_capacity_dn_padding_is_not_monotone_in_removed_gt():
    before = capacity.physical_bounds([60] * 8, 0)
    after = capacity.physical_bounds([50] * 8, 0)
    assert before["dn_padded_queries_raw"] == 120
    assert after["dn_padded_queries_raw"] == 200
    assert before["dn_padded_queries_after_removal_upper"] == 200
    assert after["dn_active_positive_indices_raw"] <= before["dn_active_positive_indices_after_removal_upper"]


def test_joint_envelope_bounds_both_slots_without_componentwise_claim():
    first = capacity.physical_bounds(list(capacity.JOINT_COUNTS[:8]), 0)
    second = capacity.physical_bounds(list(capacity.JOINT_COUNTS[8:]), 1)
    assert first["sum_gt"] == second["sum_gt"] == 1147
    assert first["max_gt"] == second["max_gt"] == 462
    assert first["dn_padded_queries_after_removal_upper"] == 924
    assert first["decoder_queries_after_removal_upper"] == 1224
    assert first["raw_counts"][1] == 98 < 431


class TinyConditionalDN(nn.Module):
    def __init__(self):
        super().__init__()
        self.required = nn.Linear(2, 2)
        self.decoder = nn.Module()
        self.decoder.denoising_class_embed = nn.Embedding(3, 2)

    def objective(self, *, use_dn, zero_dn=False):
        loss = self.required(torch.ones(2, 2)).square().sum()
        if use_dn:
            loss = loss + self.decoder.denoising_class_embed(torch.tensor([0, 1])).sum() * (0. if zero_dn else 1.)
        return loss


def tiny():
    model = TinyConditionalDN()
    return model, torch.optim.AdamW(model.parameters(), lr=.01)


def step(model, optimizer, updates, *, use_dn, zero_dn=False):
    before = capacity.capture_adam_state(model, optimizer, updates)
    optimizer.zero_grad(set_to_none=True)
    model.objective(use_dn=use_dn, zero_dn=zero_dn).backward()
    optimizer.step()
    after = capacity.capture_adam_state(model, optimizer, updates + 1)
    groups = [1, None] if use_dn else [None, None]
    proof = capacity.verify_adam_transition(before, after, dn_groups=groups)
    return before, after, proof


def test_named_adam_lazy_dn_initialization_and_absent_later_gradient():
    model, optimizer = tiny()
    _, first, proof = step(model, optimizer, 0, use_dn=False)
    assert proof["conditional_DN_absence"] == [capacity.DN_PARAMETER]
    assert first["parameters"][capacity.DN_PARAMETER] == {
        "state_present": False, "step": None, "grad_present": False}
    _, second, _ = step(model, optimizer, 1, use_dn=True)
    assert second["parameters"][capacity.DN_PARAMETER]["step"] == 1
    _, third, proof = step(model, optimizer, 2, use_dn=False)
    assert third["parameters"][capacity.DN_PARAMETER] == {
        "state_present": True, "step": 1, "grad_present": False}
    assert third["parameters"]["required.weight"]["step"] == 3


def test_zero_gradient_tensor_still_initializes_and_advances_adam():
    model, optimizer = tiny()
    _, after, proof = step(model, optimizer, 0, use_dn=True, zero_dn=True)
    assert torch.count_nonzero(model.decoder.denoising_class_embed.weight.grad).item() == 0
    assert after["parameters"][capacity.DN_PARAMETER]["grad_present"] is True
    assert after["parameters"][capacity.DN_PARAMETER]["step"] == 1
    assert proof["conditional_DN_absence"] == []


def test_detached_mandatory_parameter_is_not_excused_by_empty_dn():
    model, optimizer = tiny()
    before = capacity.capture_adam_state(model, optimizer, 0)
    optimizer.zero_grad(set_to_none=True)
    model.required.bias.sum().backward()
    optimizer.step()
    after = capacity.capture_adam_state(model, optimizer, 1)
    with pytest.raises(capacity.ReplicationCapacityError, match="unexplained unused trainable"):
        capacity.verify_adam_transition(before, after, dn_groups=[None, None])


def test_dn_disabled_requires_grad_none_instead_of_stale_zero_tensor():
    model, optimizer = tiny()
    before, after, _ = step(model, optimizer, 0, use_dn=True, zero_dn=True)
    with pytest.raises(capacity.ReplicationCapacityError, match="DN embedding gradient"):
        capacity.verify_adam_transition(before, after, dn_groups=[None, None])


@pytest.mark.parametrize("bad_step", [.5, float("nan"), float("inf"), -1., 2.])
def test_adam_clock_requires_finite_integral_actual_update(bad_step):
    model, optimizer = tiny()
    step(model, optimizer, 0, use_dn=True)
    optimizer.state[model.required.weight]["step"].fill_(bad_step)
    with pytest.raises(capacity.ReplicationCapacityError, match="clock"):
        capacity.capture_adam_state(model, optimizer, 1)


@pytest.mark.parametrize("mutation", ["missing_moment", "wrong_shape", "wrong_dtype", "unknown_parameter",
                                      "omitted_trainable", "duplicate_parameter", "fresh_state"])
def test_adam_live_layout_and_moments_are_required(mutation):
    model, optimizer = tiny()
    if mutation == "fresh_state":
        optimizer.state[model.required.weight] = {}
        with pytest.raises(capacity.ReplicationCapacityError, match="fresh capacity"):
            capacity.capture_adam_state(model, optimizer, 0)
        return
    step(model, optimizer, 0, use_dn=True)
    values = optimizer.state[model.required.weight]
    if mutation == "missing_moment":
        del values["exp_avg_sq"]
    elif mutation == "wrong_shape":
        values["exp_avg"] = torch.zeros(1)
    elif mutation == "wrong_dtype":
        values["exp_avg"] = values["exp_avg"].double()
    elif mutation == "unknown_parameter":
        optimizer.state[nn.Parameter(torch.ones(1))] = {}
    elif mutation == "omitted_trainable":
        optimizer.param_groups[0]["params"].pop()
    else:
        optimizer.param_groups[0]["params"].append(model.required.weight)
    with pytest.raises(capacity.ReplicationCapacityError):
        capacity.capture_adam_state(model, optimizer, 1)


@pytest.mark.parametrize("mutation", ["missing_row", "digest", "step_bool", "presence_bool", "counter",
                                      "skipped_step", "spurious_step", "empty_dn_groups"])
def test_saved_named_adam_evidence_cannot_be_repacked(mutation):
    model, optimizer = tiny()
    before, after, _ = step(model, optimizer, 0, use_dn=True)
    groups = [1, 1]
    if mutation == "missing_row":
        del after["parameters"]["required.weight"]
    elif mutation == "digest":
        after["layout_sha256"] = "f" * 64
    elif mutation == "step_bool":
        after["parameters"]["required.weight"]["step"] = True
    elif mutation == "presence_bool":
        after["parameters"]["required.weight"]["state_present"] = 1
    elif mutation == "counter":
        after["state_count"] -= 1
    elif mutation == "skipped_step":
        after["parameters"]["required.weight"]["step"] = 0
    elif mutation == "spurious_step":
        after["parameters"]["required.weight"]["step"] = 2
    else:
        groups = []
    with pytest.raises(capacity.ReplicationCapacityError):
        capacity.verify_adam_transition(before, after, dn_groups=groups)


def memory(*, total=24576, reserved=16384, allocated=15000, peak_reserved=17408,
           peak_allocated=16000, free=6144):
    overhead = max(0, total - free - reserved)
    return {key: value * 1024**2 for key, value in {
        "device_total_bytes": total, "device_free_bytes": free,
        "reserved_bytes": reserved, "allocated_bytes": allocated,
        "peak_reserved_bytes": peak_reserved, "peak_allocated_bytes": peak_allocated,
        "observed_nonallocator_bytes": overhead,
        "conservative_free_at_reserved_peak_bytes": total - peak_reserved - overhead,
    }.items()}


def test_allocator_peak_and_contemporaneous_native_overhead_joint_margin():
    capacity._memory_gate(memory())
    assert memory()["conservative_free_at_reserved_peak_bytes"] == 5120 * 1024**2
    with pytest.raises(capacity.ReplicationCapacityError, match="3072 MiB"):
        capacity._memory_gate(memory(peak_reserved=19457))
    changed = memory()
    changed["conservative_free_at_reserved_peak_bytes"] += 1
    with pytest.raises(capacity.ReplicationCapacityError, match="arithmetic"):
        capacity._memory_gate(changed)


def native_records(*, clock_free=4096):
    rows = []
    for index in range(3):
        stamp = 1_000_000_000 + index * 500_000_000
        rows.append({"kind": "health", "started_ns": stamp,
                     "gpu": {"uuid": GPU, "memory_free_mib": 5000., "memory_used_mib": 19576.}})
        rows.append({"kind": "clock", "started_ns": stamp + 100_000_000,
                     "graphics_clock_mhz": 1500., "clock_mhz": 1500., "sm_clock_mhz": 1500.,
                     "memory_free_mib": float(clock_free), "memory_used_mib": 24576. - clock_free})
    return rows


def write_native(tmp_path, rows, *, chain_mutation=None):
    entries, previous = [], None
    for index, row in enumerate(rows):
        entry = {"sequence": index, "previous_sha256": previous, "record": row}
        entry["sha256"] = canonical_sha256(entry)
        entries.append(entry)
        previous = entry["sha256"]
    summary = {"records": len(entries), "last_sha256": previous}
    if chain_mutation == "payload":
        entries[2]["record"]["gpu"]["memory_free_mib"] = 9000.
    elif chain_mutation == "sequence":
        entries[2]["sequence"] += 1
    elif chain_mutation == "previous":
        entries[2]["previous_sha256"] = "f" * 64
    elif chain_mutation == "truncated":
        entries.pop()
    raw = b"\n".join(canonical_json_bytes(entry) for entry in entries) + b"\n"
    path = tmp_path / "native-samples.jsonl.gz"
    with path.open("xb") as handle:
        handle.write(gzip.compress(raw, mtime=0))
    return {"samples": file_reference(path), "gpu_uuid": GPU, "sample_hash_chain": summary}


def test_native_memory_audit_uses_faster_clock_stream_and_complete_hash_chain(tmp_path):
    monitor = write_native(tmp_path, native_records(clock_free=3072))
    result = capacity._audit_native_memory_samples(monitor)
    assert result["minimum_native_free_mib"] == 3072.
    assert result["maximum_sampled_clock_mhz"] == 1500.
    assert result["health_samples"] == result["clock_samples"] == 3
    assert result["records"] == 6


@pytest.mark.parametrize("mutation", [
    "payload", "sequence", "previous", "truncated", "gpu", "health_gap", "clock_gap",
    "health_low_memory", "clock_low_memory", "sm_clock", "graphics_clock", "few_samples",
])
def test_native_memory_rejects_gaps_clocks_submargin_and_identity(tmp_path, mutation):
    rows = native_records()
    chain = mutation if mutation in {"payload", "sequence", "previous", "truncated"} else None
    if mutation == "gpu":
        rows[2]["gpu"]["uuid"] = "GPU-other"
    elif mutation == "health_gap":
        rows[4]["started_ns"] += 2_000_000_000
    elif mutation == "clock_gap":
        rows[5]["started_ns"] += 1_000_000_000
    elif mutation == "health_low_memory":
        rows[2]["gpu"]["memory_free_mib"] = 3071.999
    elif mutation == "clock_low_memory":
        rows[3]["memory_free_mib"] = 3071.999
    elif mutation == "sm_clock":
        rows[3]["sm_clock_mhz"] = 1501.
    elif mutation == "graphics_clock":
        rows[3]["graphics_clock_mhz"] = rows[3]["clock_mhz"] = 1501.
    elif mutation == "few_samples":
        rows = rows[:-2]
    monitor = write_native(tmp_path, rows, chain_mutation=chain)
    with pytest.raises(capacity.ReplicationCapacityError):
        capacity._audit_native_memory_samples(monitor)


@pytest.mark.parametrize("seed", [1, 2])
def test_actual_rtdetr_initial_layout_and_resolution_geometry_without_cuda(seed):
    records = []
    for size in (640, 896):
        config = V2BConfig(seed=seed, input_size=size, physical_batch_size=8, accumulation_steps=2,
                           pretrained_required=False, sampling_backend="deterministic_gather")
        components = build_v2b_components(config, repo_root=ROOT)
        loader = SimpleNamespace(has_active_iterator=False, state_dict=lambda: {
            "epoch": 0, "optimizer_updates": 0, "completed_batches": 0, "epoch_active": False})
        state = capacity._state(components, loader, full=True)
        assert state["adam"]["parameter_count"] == 326
        assert state["adam"]["state_count"] == 0
        assert capacity.DN_PARAMETER in state["adam"]["parameters"]
        assert state["ema_updates"] == state["engine"]["optimizer_updates"] == 0
        assert len(state["raw_bn"]) == 69
        assert components.initialization["seed"] == seed
        assert components.geometry["anchors"][1] == sum((size // stride) ** 2 for stride in (8, 16, 32))
        records.append(components.initialization["parameters"])
        del components
    assert records[0] == records[1]


def test_actual_rtdetr_conditional_dn_adam_participation_on_cpu_synthetic():
    config = V2BConfig(seed=1, input_size=128, physical_batch_size=1, accumulation_steps=2,
                       pretrained_required=False, sampling_backend="deterministic_gather")
    components = build_v2b_components(config, repo_root=ROOT)
    generator = torch.Generator().manual_seed(79)
    images = torch.rand(2, 3, 128, 128, generator=generator)
    components.engine.begin_epoch(1)
    for index, counts in enumerate(((0, 0), (1, 1), (0, 0))):
        before = capacity.capture_adam_state(components.model, components.optimizer, index)
        targets = [{"labels": torch.zeros(count, dtype=torch.int64),
                    "boxes": torch.tensor([[.5, .5, .2, .2]], dtype=torch.float32).repeat(count, 1)}
                   for count in counts]
        window = components.engine.train_window(images, targets)
        after = capacity.capture_adam_state(components.model, components.optimizer, index + 1)
        proof = capacity.verify_adam_transition(before, after, dn_groups=window["dn_num_groups"])
        assert proof["named_parameter_count"] == 326
        assert after["parameters"][capacity.DN_PARAMETER]["step"] == (None if index == 0 else 1)
        assert components.ema.updates == components.warmup.last_step == index + 1
        assert window["microsteps"] == 2 * (index + 1)


@pytest.mark.parametrize("seed,size", [(1, 640), (1, 896), (2, 640), (2, 896)])
def test_capacity_entry_forwards_frozen_seed_to_loader_cpu_and_admitted_runtime(
        tmp_path, monkeypatch, seed, size):
    from sparse_rtdetr.baseline import training_v2b_replication_worker as worker
    from sparse_rtdetr.baseline import training_v2b_admission as admission
    value = fixture_plan(tmp_path, seed=seed, size=size)
    plan_reference = write_exclusive_json(tmp_path / "capacity-plan.json", value)
    config = V2BConfig(seed=seed, input_size=size, physical_batch_size=8, accumulation_steps=2,
                       sampling_backend="deterministic_gather")
    output = tmp_path / "owned-output"
    contract = {
        "stage": "capacity", "stage_gate_reference": None, "matched_reference": None, "replay_source": None,
        "config": asdict(config), "num_workers": 2, "prefetch_factor": 2,
        "capacity_plan": plan_reference, "epoch_order_sha256": value["epoch_order_sha256"],
        "output_dir": str(output), "cell_id": f"seed{seed}-r{size}", "campaign_id": "synthetic-capacity",
        "run_id": "synthetic-capacity-run", "invocation_id": "synthetic-capacity-invocation",
        "freeze_reference": ref(tmp_path / "freeze.json"), "qualification_reference": ref(tmp_path / "qualification.json"),
        "train_core": {"annotation": value["annotation"], "manifest": value["manifest"],
                       "image_root": str(tmp_path / "train_core/images")},
        "repo_root": str(ROOT), "pretrained": ref(tmp_path / "pretrained.pth"),
        "expected_gpu_uuid": GPU, "code_files": {"source.py": ref(tmp_path / "source.py")},
        "policy_bundle": {"synthetic_only": True},
    }
    contract_reference = write_exclusive_json(tmp_path / "contract.json", contract)
    calls = []
    def validate(value, *, verify_files):
        calls.append(("validate", verify_files))
        assert verify_files is True
        return value
    monkeypatch.setattr(worker, "validate_replication_worker_contract", validate)
    monkeypatch.setattr(capacity, "validate_replication_capacity_plan",
                        lambda actual, **kwargs: value)
    monkeypatch.setattr(capacity.control, "_environment", lambda _: {"synthetic_only": True})
    class Dataset:
        images = [{"id": n} for n in range(4869)]
        annotations = {n: [None] for n in range(4869)}
        def __len__(self):
            return 4869
    loader = SimpleNamespace(dataset=Dataset(), batches_per_epoch=304,
        _plan=lambda epoch: logical_batch_indices(4869, seed=seed, epoch=epoch, logical_batch_size=16))
    def build_loader(config, **kwargs):
        calls.append(("loader", config.seed, config.input_size, config.num_workers, config.prefetch_factor))
        return loader
    monkeypatch.setattr(capacity, "build_train_core_loader", build_loader)
    token = object()
    def build(config, **kwargs):
        if "runtime" in kwargs:
            calls.append(("cuda_factory", config.seed, config.input_size, kwargs["runtime"] is token))
            raise RuntimeError("synthetic wiring stop before any CUDA model")
        calls.append(("cpu_factory", config.seed, config.input_size))
        return SimpleNamespace(initialization={"synthetic_only": True})
    monkeypatch.setattr(capacity, "build_v2b_components", build)
    monkeypatch.setattr(capacity, "build_train_core_run_binding",
                        lambda *args, **kwargs: {"code": {"source.py": {}}, "synthetic_only": True})
    class Monitor:
        def __init__(self, *args):
            calls.append(("monitor_construct",))
        def start(self):
            calls.append(("monitor_start",))
        def admission(self):
            calls.append(("monitor_admission",))
            return "synthetic-token"
        def abort(self, reason):
            calls.append(("monitor_abort",))
    monkeypatch.setattr(admission, "MonitoredHardwareSession", Monitor)
    def runtime(**kwargs):
        calls.append(("runtime", kwargs["seed"], kwargs["device"], kwargs["gpu_probe"]))
        return token
    monkeypatch.setattr(capacity, "prepare_runtime", runtime)
    with pytest.raises(RuntimeError, match="synthetic wiring stop"):
        capacity.run_replication_capacity_worker(contract, contract_reference=contract_reference)
    assert calls == [
        ("validate", True), ("loader", seed, size, 2, 2), ("cpu_factory", seed, size),
        ("monitor_construct",), ("monitor_start",), ("monitor_admission",),
        ("runtime", seed, "cuda:0", "synthetic-token"), ("cuda_factory", seed, size, True),
        ("monitor_abort",),
    ]
    assert strict_json_loads((output / "worker-result.json").read_bytes())["status"] == "STOP_NO_RETRY"


def synthetic_completion(tmp_path, monkeypatch, *, mutation=None):
    """Persist a fake stopped worker's full evidence chain, never a live admission."""
    from sparse_rtdetr.baseline.training_v2b_admission import wait_for_monitored_finish
    from sparse_rtdetr.baseline.training_v2b_engine import AccumulationEngine
    output = tmp_path / "owned-capacity"
    output.mkdir()
    native_root = output / "native-hardware"
    native_root.mkdir()
    plan = fixture_plan(tmp_path)
    plan_reference = write_exclusive_json(tmp_path / "capacity-plan.json", plan)
    config = asdict(V2BConfig(seed=1, input_size=896, physical_batch_size=8,
                             accumulation_steps=2, sampling_backend="deterministic_gather"))
    contract = {
        "kind": "v2b_seed_resolution_replication_worker", "stage": "capacity",
        "cell_id": "seed1-r896", "campaign_id": "synthetic-campaign", "run_id": "synthetic-run",
        "invocation_id": "synthetic-invocation", "output_dir": str(output), "config": config,
        "capacity_plan": plan_reference, "freeze_reference": ref(tmp_path / "freeze.json"),
        "qualification_reference": ref(tmp_path / "qualification.json"), "expected_gpu_uuid": GPU,
        "train_core": {"annotation": plan["annotation"], "manifest": plan["manifest"]},
    }
    if mutation == "configuration":
        contract["config"]["seed"] = 2
    contract_reference = write_exclusive_json(tmp_path / "contract.json", contract)
    binding = {"run_id": contract["run_id"], "binding_sha256": "b" * 64,
               "config": asdict(V2BConfig(seed=1, input_size=896, physical_batch_size=8,
                                         accumulation_steps=2))}
    binding_reference = write_exclusive_json(output / "run-binding.json", binding)
    owner = {"pid": 123456, "synthetic_only": True}
    final = {
        **write_native(native_root, native_records()),
        "run_id": contract["run_id"], "run_binding_sha256": binding["binding_sha256"],
        "policy_sha256": "c" * 64, "monitor_pid": 123457, "owner": owner,
        "guardian_identity": {"synthetic_only": True}, "status": "PASS",
        "worker_exited": True, "sampled_clock_compliance": True,
    }
    for field in ("startup_evidence", "setter_receipt", "guardian_heartbeat"):
        final[field] = write_exclusive_json(native_root / (field + ".json"), {"synthetic_only": True})
    final_reference = write_exclusive_json(native_root / "monitor-final.json", final)
    monitor_reference = {key: final[key] for key in (
        "run_id", "run_binding_sha256", "policy_sha256", "monitor_pid", "owner", "guardian_identity", "gpu_uuid")}
    monitor_reference["monitor_final_report"] = final_reference["path"]
    monitor = wait_for_monitored_finish(monitor_reference, timeout_seconds=1.)
    launch_reference = write_exclusive_json(tmp_path / "launch.json", {
        "contract": contract_reference, "owner": owner, "pid": owner["pid"]})
    model, optimizer = tiny()
    engine = AccumulationEngine(model, nn.Identity(), optimizer, physical_batch_size=8,
                                accumulation_steps=2, expected_input_size=896)
    engine.begin_epoch(1)
    cases = []
    names = [row["case_id"] for row in plan["fixed_windows"]] + [
        "synthetic-envelope-01", "synthetic-envelope-02"]
    for index, case_id in enumerate(names):
        engine.optimizer_updates, engine.microsteps = index, 2 * index
        before = {"adam": capacity.capture_adam_state(model, optimizer, index),
                  "engine": engine.state_dict()}
        _, adam_after, proof = step(model, optimizer, index, use_dn=True)
        engine.optimizer_updates, engine.microsteps = index + 1, 2 * (index + 1)
        after = {"adam": adam_after, "engine": engine.state_dict()}
        case = {"case_id": case_id, "state_before": before, "state_after": after,
                "adam_transition": proof, "window": {"dn_num_groups": [1, None]}, "memory": memory()}
        if mutation == "state_gap" and index == 0:
            case["state_after"]["unexplained_change"] = True
        cases.append(write_exclusive_json(output / (case_id + ".json"), case))
    before_development = {**after, "full_state_hashes": {"synthetic_only": True}}
    before_reference = write_exclusive_json(output / "before-development-state.json", before_development)
    final_state = copy.deepcopy(before_development)
    if mutation == "development_mutation":
        final_state["full_state_hashes"]["synthetic_only"] = False
    state_reference = write_exclusive_json(output / "capacity-final-state.json", final_state)
    memory_value = memory()
    memory_reference = write_exclusive_json(output / "development-memory.json", memory_value)
    result = {
        "kind": capacity.RESULT_KIND, "stage": "capacity", "status": "PASS",
        **{key: contract[key] for key in ("cell_id", "campaign_id", "run_id", "invocation_id",
                                          "freeze_reference", "qualification_reference", "capacity_plan")},
        "seed": 1, "input_size": 896, "contract_reference": contract_reference,
        "worker_pid": owner["pid"] if mutation != "identity" else owner["pid"] + 1,
        "binding_reference": binding_reference, "monitor_reference": monitor_reference,
        "capacity_acceptance": {"status": "PASS"}, "formal_training_samples": 0, "checkpoints_created": False,
        "cases": cases, "diagnostic_optimizer_updates": 10 if mutation == "extra_update" else 9,
        "development_memory_reference": memory_reference, "development_memory": memory_value,
        "before_development_state_reference": before_reference, "final_state_reference": state_reference,
    }
    if mutation == "development_memory":
        result["development_memory"] = memory(free=7000)
    result_reference = write_exclusive_json(output / "worker-result.json", result)
    validator = capacity.validate_replication_capacity_plan
    def fake_authority(value, **kwargs):
        assert kwargs.pop("verify_files", True) is True
        return validator(value, **kwargs, verify_files=False)
    monkeypatch.setattr(capacity, "validate_replication_capacity_plan", fake_authority)
    return {"result_reference": result_reference, "launch_reference": launch_reference,
            "monitor": monitor, "capacity_memory_audit_reference": None}


def test_public_capacity_audit_binds_result_guardian_and_both_memory_minima(tmp_path, monkeypatch):
    completion = synthetic_completion(tmp_path, monkeypatch)
    result = capacity.audit_replication_capacity_native_memory(completion)
    assert result["kind"] == "v2b_seed_resolution_replication_capacity_memory_audit"
    assert result["status"] == "PASS"
    assert result["result_reference"] == completion["result_reference"]
    assert result["monitor_final_report_reference"] == completion["monitor"]["final_report_reference"]
    assert result["required_margin_mib"] == 3072
    assert result["minimum_native_free_mib"] == 4096
    assert result["minimum_conservative_free_mib"] == 5120
    assert result["all_nine_cases_and_evaluation_memory_authenticated"] is True


@pytest.mark.parametrize("mutation", [
    "configuration", "state_gap", "development_mutation", "development_memory", "identity", "extra_update"])
def test_public_capacity_audit_rejects_cross_evidence_inconsistency(tmp_path, monkeypatch, mutation):
    completion = synthetic_completion(tmp_path, monkeypatch, mutation=mutation)
    with pytest.raises(capacity.ReplicationCapacityError):
        capacity.audit_replication_capacity_native_memory(completion)
