from __future__ import annotations

import copy
import functools
import hashlib
import importlib.util
import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from sparse_rtdetr.baseline.training_v2b import (
    V2BConfig, V2BConfigurationError, build_v2b_components, input_hw,
    logical_batch_indices, seed_cpu_sources, seed_worker,
    validate_model_geometry, validate_model_sampling,
)
from sparse_rtdetr.baseline.training_v2b_checkpoint import (
    save_checkpoint, restore_checkpoint,
)
from sparse_rtdetr.baseline.training_v2b_evidence import (
    build_run_binding, build_completed_report, collect_native_cpu_observation,
    validate_completed_report, write_exclusive_json,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    torch.set_num_threads(2)
    def forbidden(*args, **kwargs):
        raise AssertionError("CUDA initialization is forbidden in CPU engineering tests")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)


def config(**kwargs):
    fields = dict(input_size=128, physical_batch_size=1, accumulation_steps=2,
                  pretrained_required=False, warmup_steps=2, ema_warmups=2)
    fields.update(kwargs)
    return V2BConfig(**fields)


def build(**kwargs):
    return build_v2b_components(config(**kwargs), repo_root=ROOT)


def batch(counts=(0, 3), *, size=128, seed=73):
    generator = torch.Generator().manual_seed(seed)
    images = torch.rand((len(counts), 3, size, size), generator=generator)
    targets = []
    for count in counts:
        centers = .2 + .6 * torch.rand((count, 2), generator=generator)
        wh = .05 + .15 * torch.rand((count, 2), generator=generator)
        targets.append({"labels": torch.arange(count, dtype=torch.int64) % 10,
                        "boxes": torch.cat((centers, wh), dim=1)})
    return images, targets


def bind(components, images, targets, run_id="real-r18-cpu"):
    digest = hashlib.sha256(images.numpy().tobytes())
    for target in targets:
        for key in ("labels", "boxes"):
            digest.update(target[key].numpy().tobytes())
    code_paths = {
        row["relative_path"]: ROOT / row["relative_path"]
        for key in ("vendor_sources", "package_sources")
        for row in components.initialization[key]
    }
    return build_run_binding(
        run_id=run_id, code_paths=code_paths,
        config={**components.config.binding_config(), "sampling": components.initialization["sampling"]},
        initial_parameters=components.initialization["parameters"],
        initial_state=components.initialization["model_state"],
        synthetic_input={"fixture": "generated normalized boxes and RGB tensors",
                         "tensor_sha256": digest.hexdigest()},
    )


@pytest.mark.parametrize("kwargs", [
    {"seed": True}, {"seed": -1}, {"seed": 2**32},
    {"input_size": 96}, {"input_size": 129}, {"input_size": 960}, {"input_size": 1056}, {"input_size": 1152},
    {"physical_batch_size": 0}, {"accumulation_steps": True},
    {"amp_dtype": "float16"}, {"bn_statistics": "backbone_only"},
    {"learning_rate": float("nan")}, {"weight_decay": -1},
    {"pretrained_required": 1}, {"num_denoising": -1},
    {"sampling_backend": "discrete"}, {"sampling_backend": True},
])
def test_invalid_foundation_config(kwargs):
    with pytest.raises(V2BConfigurationError):
        config(**kwargs)


def test_engineering_checker_never_reads_runtime_artifact_contents(monkeypatch):
    spec = importlib.util.spec_from_file_location("v2b_source_checker", ROOT / "tools/repository_contract_check.py")
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    original_bytes, original_text = Path.read_bytes, Path.read_text
    def guard(original):
        def read(path, *args, **kwargs):
            assert not path.resolve().is_relative_to(ROOT / "artifacts"), str(path)
            return original(path, *args, **kwargs)
        return read
    monkeypatch.setattr(Path, "read_bytes", guard(original_bytes))
    monkeypatch.setattr(Path, "read_text", guard(original_text))
    assert checker.check_repository(ROOT, source_only=True)


def test_engineering_target_configuration_explicitly_selects_candidate():
    path = ROOT / "configs/baseline/rtdetrv2_r18_visdrone_baseline_v2b_engineering.json"
    policy = json.loads(path.read_text())
    assert V2BConfig().sampling_backend == "native"
    assert policy["config"] == asdict(V2BConfig(sampling_backend="deterministic_gather"))
    assert policy["scope"] == "cpu_synthetic_engineering"
    assert all(value is False for key, value in policy["readiness"].items()
               if key != "cpu_verification_evidence_required")


def test_actual_seed_initialization_and_registry_isolation():
    seed_cpu_sources(777)
    a = build(physical_batch_size=2, accumulation_steps=1, seed=8)
    draws_a = (random.random(), np.random.rand(), torch.rand(3))
    seed_cpu_sources(999)
    _ = torch.rand(100)
    b = build(seed=8)
    draws_b = (random.random(), np.random.rand(), torch.rand(3))
    assert a.initialization["parameters"] == b.initialization["parameters"]
    assert a.initialization["model_state"] == b.initialization["model_state"]
    assert len(a.initialization["model_state"]["inventory"]) > len(a.initialization["parameters"]["inventory"])
    assert a.initialization["vendor_sources"]
    assert a.initialization["package_sources"]
    assert draws_a[:2] == draws_b[:2]
    assert torch.equal(draws_a[2], draws_b[2])
    assert sum(p.numel() for p in a.model.parameters()) == 20_094_584
    assert len(a.batchnorm_inventory) == 69
    assert all(x["affine"] and x["track_running_stats"] for x in a.batchnorm_inventory)
    assert not torch.cuda.is_initialized()


def test_sampling_selection_is_per_model_preserves_initial_state_and_binds_ema():
    native = build(seed=23)
    native_identity = validate_model_sampling(native.model, sampling_backend="native")
    native_draws = (random.random(), np.random.rand(), torch.rand(3))
    candidate = build(seed=23, sampling_backend="deterministic_gather")
    candidate_draws = (random.random(), np.random.rand(), torch.rand(3))
    assert native.config.sampling_backend == V2BConfig().sampling_backend == "native"
    assert candidate.initialization["parameters"] == native.initialization["parameters"]
    assert candidate.initialization["model_state"] == native.initialization["model_state"]
    assert native_draws[:2] == candidate_draws[:2]
    assert torch.equal(native_draws[2], candidate_draws[2])
    assert validate_model_sampling(native.model, sampling_backend="native") == native_identity
    identity = validate_model_sampling(candidate.model, sampling_backend="deterministic_gather")
    assert candidate.initialization["sampling"] == identity
    assert validate_model_sampling(candidate.ema.module, sampling_backend="deterministic_gather") == identity
    assert len(identity["modules"]) == 3
    assert identity["core"]["source"]["relative_path"] == (
        "src/sparse_rtdetr/baseline/training_v2b_deterministic_sampling.py"
    )
    assert identity["core"]["source"]["sha256"] == hashlib.sha256(
        (ROOT / identity["core"]["source"]["relative_path"]).read_bytes()
    ).hexdigest()
    assert native.config.binding_config()["cuda_backend_policy"]["deterministic_algorithms"] is False
    policy = candidate.config.binding_config()["cuda_backend_policy"]
    assert policy["deterministic_algorithms"] is True
    assert policy["deterministic_warn_only"] is False
    assert policy["cublas_workspace_config"] == ":4096:8"


@pytest.mark.parametrize("change", ["native_callable", "partial_args", "partial_keywords", "method", "missing_module"])
def test_candidate_sampling_rejects_actual_callable_or_inventory_drift(change):
    candidate = build(sampling_backend="deterministic_gather")
    model = candidate.model
    attention = model.decoder.decoder.layers[0].cross_attn
    core = attention.ms_deformable_attn_core
    if change == "native_callable":
        from src.zoo.rtdetr.rtdetrv2_decoder import deformable_attention_core_func_v2
        attention.ms_deformable_attn_core = functools.partial(deformable_attention_core_func_v2, method="default")
    elif change == "partial_args":
        attention.ms_deformable_attn_core = functools.partial(core.func, None, method="default")
    elif change == "partial_keywords":
        attention.ms_deformable_attn_core = functools.partial(core.func, method="discrete")
    elif change == "method":
        attention.method = "discrete"
    else:
        model.decoder.decoder.layers[0].cross_attn = torch.nn.Identity()
    with pytest.raises(V2BConfigurationError, match="sampling"):
        validate_model_sampling(model, sampling_backend="deterministic_gather")


def test_rejects_vendor_or_package_cached_from_another_checkout(monkeypatch, tmp_path):
    build()
    for module_name in ("src.core.workspace", "sparse_rtdetr.baseline.config"):
        with monkeypatch.context() as changed:
            changed.setattr(sys.modules[module_name], "__file__", str(tmp_path / "wrong.py"))
            with pytest.raises(V2BConfigurationError, match="different checkout"):
                build()


def test_pretraining_preserves_common_non_backbone_initialization(tmp_path):
    a = build(seed=42)
    pretrained = {k: v.clone() for k, v in a.model.backbone.state_dict().items()}
    first_weight = next(k for k, v in pretrained.items() if v.is_floating_point())
    pretrained[first_weight].fill_(.125)
    path = tmp_path / "synthetic_backbone.pth"
    torch.save(pretrained, path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    b = build_v2b_components(config(seed=42, pretrained_required=True),
                             repo_root=ROOT, pretrained_path=path,
                             pretrained_sha256=sha)
    for name, parameter in a.model.named_parameters():
        if not name.startswith("backbone."):
            assert torch.equal(parameter, dict(b.model.named_parameters())[name])
    assert b.initialization["pretrained"]["strict_load"] is True
    assert b.initialization["pretrained"]["missing_keys"] == []
    assert torch.equal(b.model.backbone.state_dict()[first_weight], pretrained[first_weight])
    with pytest.raises(V2BConfigurationError, match="SHA-256"):
        build_v2b_components(config(), repo_root=ROOT, pretrained_path=path,
                              pretrained_sha256="0" * 64)
    with pytest.raises(V2BConfigurationError, match="authority"):
        build_v2b_components(config(pretrained_required=True), repo_root=ROOT)


@pytest.mark.skipif(not os.environ.get("P3_V2B_PRETRAINED"),
                    reason="explicit local pretrained authority not supplied")
def test_existing_pretrained_authority_on_real_cpu_model():
    path = Path(os.environ["P3_V2B_PRETRAINED"])
    sha = os.environ["P3_V2B_PRETRAINED_SHA256"]
    c = build_v2b_components(config(pretrained_required=True), repo_root=ROOT,
                             pretrained_path=path, pretrained_sha256=sha)
    assert c.initialization["pretrained"]["sha256"] == sha
    assert c.initialization["pretrained"]["missing_keys"] == []
    assert c.initialization["pretrained"]["unexpected_keys"] == []
    assert len(c.batchnorm_inventory) == 69


def test_logical_order_and_tail_are_independent_of_model_rng():
    expected = list(logical_batch_indices(35, seed=4, epoch=1))
    assert len(expected) == 2 and all(len(x) == 16 for x in expected)
    assert len({x for group in expected for x in group}) == 32
    _ = torch.rand(1000)
    assert list(logical_batch_indices(35, seed=4, epoch=1)) == expected
    assert list(logical_batch_indices(35, seed=4, epoch=1, completed_batches=1)) == expected[1:]
    assert list(logical_batch_indices(35, seed=4, epoch=2)) != expected
    with pytest.raises(V2BConfigurationError):
        list(logical_batch_indices(35, seed=4, epoch=1, completed_batches=3))
    seed_cpu_sources(52)
    seed_worker(0)
    first = (random.random(), np.random.rand())
    seed_worker(0)
    assert first == (random.random(), np.random.rand())


def test_resolved_geometry_and_configuration_640_then_128():
    c = build(input_size=640)
    assert c.geometry["anchors"] == [1, 8400, 4]
    assert c.geometry["position_caches"] == {"pos_embed2": [1, 400, 256]}
    for name in ("train_dataloader", "val_dataloader"):
        ops = c.resolved_config[name]["dataset"]["transforms"]["ops"]
        assert next(x["size"] for x in ops if x["type"] == "Resize") == [640, 640]
    assert c.resolved_config["train_dataloader"]["collate_fn"]["scales"] is None
    assert c.resolved_config["val_dataloader"]["total_batch_size"] == 8
    c.engine.begin_epoch(1)
    images, targets = batch()
    with pytest.raises((ValueError, RuntimeError), match="size|shape"):
        c.engine.train_window(images, targets)
    small = build()
    assert small.geometry["anchors"] == [1, 336, 4]
    assert small.model.decoder.num_queries == 300
    assert small.criterion.num_classes == 10
    with pytest.raises(V2BConfigurationError, match="spatial"):
        validate_model_geometry(small.model, config(input_size=640))


@pytest.mark.parametrize("sampling_backend", ["native", "deterministic_gather"])
def test_real_loss_families_empty_targets_and_variable_dn(sampling_backend):
    c = build(amp_dtype="float32", sampling_backend=sampling_backend)
    c.model.train()
    for counts, expected_dn in [((2,), 200), ((0,), 0), ((101,), 202)]:
        images, targets = batch(counts)
        outputs = c.model(images, targets)
        losses = c.criterion(outputs, targets)
        base = [k for k in losses if not any(tag in k for tag in ("_aux_", "_dn_", "_enc_"))]
        assert len(base) == 3
        assert sum("_aux_" in k for k in losses) == 6
        assert sum("_enc_" in k for k in losses) == 3
        assert sum("_dn_" in k for k in losses) == (9 if expected_dn else 0)
        assert all(torch.isfinite(value).all() for value in losses.values())
        if expected_dn:
            assert outputs["dn_aux_outputs"][0]["pred_logits"].shape[1] == expected_dn
        else:
            assert "dn_aux_outputs" not in outputs
            assert losses["loss_vfl"] > 0
            assert losses["loss_bbox"] == 0


@pytest.mark.parametrize("sampling_backend", ["native", "deterministic_gather"])
def test_real_bf16_update_covers_encoder_heads_and_bn_ema_clock(sampling_backend):
    c = build(sampling_backend=sampling_backend)
    images, targets = batch()
    norms = dict((n, m) for n, m in c.model.named_modules()
                 if isinstance(m, torch.nn.modules.batchnorm._BatchNorm))
    first_name, first_bn = next(iter(norms.items()))
    old_mean = first_bn.running_mean.clone()
    c.engine.begin_epoch(1)
    observed = []
    hook = c.model.register_forward_hook(lambda module, args, out: observed.append(out["pred_logits"].dtype))
    report = c.engine.train_window(images, targets)
    hook.remove()
    assert observed == [torch.bfloat16, torch.bfloat16]
    assert report["optimizer_updates"] == 1
    assert c.ema.updates == c.warmup.last_step == 1
    assert all(m.num_batches_tracked.item() == 2 for m in norms.values())
    ema_bn = dict(c.ema.module.named_modules())[first_name]
    assert ema_bn.num_batches_tracked.item() == 0  # vendor integer-buffer policy
    d = c.ema.decay_fn(1)
    torch.testing.assert_close(ema_bn.running_mean, old_mean*d + first_bn.running_mean*(1-d))
    encoder_heads = [(n, p) for n, p in c.model.named_parameters() if n.startswith("decoder.enc_")]
    assert len(encoder_heads) == 12
    for name, p in encoder_heads:
        # Initial zero last-layer weights legitimately make some gradients zero.
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p in c.optimizer.state, name
    assert not torch.cuda.is_initialized()


def test_full_vs_split_real_model_when_bn_statistics_and_dn_are_controlled():
    a = build(physical_batch_size=2, accumulation_steps=1,
              amp_dtype="float32", bn_statistics="frozen", num_denoising=0)
    b = build(amp_dtype="float32", bn_statistics="frozen", num_denoising=0)
    assert a.initialization["parameters"] == b.initialization["parameters"]
    images, targets = batch()
    a.engine.begin_epoch(1)
    b.engine.begin_epoch(1)
    full = a.engine.train_window(images, targets)
    split = b.engine.train_window(images, targets)
    assert split["loss"] == pytest.approx(full["loss"], rel=2e-5, abs=1e-6)
    for (name, pa), (_, pb) in zip(a.model.named_parameters(), b.model.named_parameters()):
        if pa.grad is None or pb.grad is None:
            assert pa.grad is None and pb.grad is None, name
        else:
            torch.testing.assert_close(pa.grad, pb.grad, rtol=5e-4, atol=2e-6, msg=name)
    for model in (a.model, b.model):
        assert all(m.num_batches_tracked == 0 for m in model.modules()
                   if isinstance(m, torch.nn.modules.batchnorm._BatchNorm))


@pytest.mark.parametrize("sampling_backend", ["native", "deterministic_gather"])
def test_actual_r18_checkpoint_replay_and_native_cpu_evidence(tmp_path, sampling_backend):
    c = build(seed=19, sampling_backend=sampling_backend)
    images, targets = batch((1, 3), seed=11)
    binding = bind(c, images, targets)
    c.engine.begin_epoch(1)
    window = c.engine.train_window(images, targets)
    generator = torch.Generator().manual_seed(900)
    reference = save_checkpoint(
        tmp_path / "step1.pth", model=c.model, optimizer=c.optimizer,
        ema=c.ema, engine=c.engine, scheduler=c.scheduler, warmup=c.warmup,
        binding=binding, extra_rng_generators={"inputs": generator},
    )
    counters = dict(epoch=1, optimizer_updates=1, microsteps=2, samples_seen=2)
    ledger = write_exclusive_json(
        tmp_path / "windows.json",
        {"run_binding_sha256": binding["binding_sha256"],
         "windows": [window], "counters": counters},
    )
    observation = collect_native_cpu_observation(binding)
    report = build_completed_report(binding, observation, checkpoint_reference=reference,
                                     counters=counters, window_references=[ledger])
    validate_completed_report(report, binding=binding, observation=observation)
    assert report["scientific_certified"] is False
    assert report["gpu_ready"] is False
    write_exclusive_json(tmp_path / "cpu_report.json", report)

    next_images = torch.rand(images.shape, generator=generator)
    uninterrupted = c.engine.train_window(next_images, targets)
    c.engine.finish_epoch(expected_optimizer_steps=2)
    expected = {k: v.clone() for k, v in c.model.state_dict().items()}
    expected_ema = copy.deepcopy(c.ema.state_dict())
    expected_optimizer = copy.deepcopy(c.optimizer.state_dict())
    expected_engine = c.engine.state_dict()
    expected_scheduler = c.scheduler.state_dict()
    expected_warmup = c.warmup.state_dict()

    restored = build(seed=19, sampling_backend=sampling_backend)
    new_generator = torch.Generator().manual_seed(999999)
    restore_checkpoint(
        reference["path"], model=restored.model, optimizer=restored.optimizer,
        ema=restored.ema, engine=restored.engine, scheduler=restored.scheduler,
        warmup=restored.warmup, expected_binding=binding,
        expected_sha256=reference["sha256"],
        extra_rng_generators={"inputs": new_generator},
    )
    resumed_images = torch.rand(images.shape, generator=new_generator)
    assert torch.equal(next_images, resumed_images)
    resumed = restored.engine.train_window(resumed_images, targets)
    restored.engine.finish_epoch(expected_optimizer_steps=2)
    assert resumed["loss"] == uninterrupted["loss"]
    for name, value in restored.model.state_dict().items():
        assert torch.equal(value, expected[name]), name
    for name, value in restored.ema.module.state_dict().items():
        assert torch.equal(value, expected_ema["module"][name]), name
    assert restored.ema.updates == expected_ema["updates"]
    def assert_tree_equal(actual, wanted):
        if isinstance(wanted, torch.Tensor):
            assert torch.equal(actual, wanted)
        elif isinstance(wanted, dict):
            assert actual.keys() == wanted.keys()
            for key in wanted:
                assert_tree_equal(actual[key], wanted[key])
        elif isinstance(wanted, (list, tuple)):
            assert len(actual) == len(wanted)
            for value, expected_value in zip(actual, wanted):
                assert_tree_equal(value, expected_value)
        else:
            assert actual == wanted
    assert_tree_equal(restored.optimizer.state_dict(), expected_optimizer)
    assert restored.engine.state_dict() == expected_engine
    assert restored.scheduler.state_dict() == expected_scheduler
    assert restored.warmup.state_dict() == expected_warmup
    assert not torch.cuda.is_initialized()


def test_config_admits_square_1024_only_as_an_explicit_size():
    assert V2BConfig(input_size=1024).input_size == 1024
    for size in (992, 1056, 1344):
        with pytest.raises(V2BConfigurationError):
            V2BConfig(input_size=size)


def test_config_admits_the_a3b_canvas_only_as_an_explicit_height_width_list():
    canvas = V2BConfig(input_size=(768, 1344))
    assert canvas.input_size == [768, 1344]
    assert canvas.binding_config()["input_size"] == [768, 1344]
    assert input_hw(canvas.input_size) == [768, 1344] and input_hw(896) == [896, 896]
    assert V2BConfig(input_size=896).binding_config()["input_size"] == 896
    for size in ([1344, 768], [768, 768], [768.0, 1344], [True, 1344], [768, 1344, 3], [736, 1344], "768x1344"):
        with pytest.raises(V2BConfigurationError):
            V2BConfig(input_size=size)


def _vendor_position_embedding(model, w, h):
    return type(model.encoder).build_2d_sincos_position_embedding(
        w, h, model.encoder.hidden_dim, model.encoder.pe_temperature)


def test_a3b_canvas_geometry_is_height_then_width_with_row_major_positions():
    c = build(input_size=[768, 1344])
    assert c.geometry["input_size"] == [768, 1344]
    assert c.geometry["anchors"] == [1, 96 * 168 + 48 * 84 + 24 * 42, 4]
    assert c.geometry["position_caches"] == {"pos_embed2": [1, 24 * 42, 256]}
    for component in (c.model.encoder, c.model.decoder, c.ema.module.encoder, c.ema.module.decoder):
        assert list(component.eval_spatial_size) == [768, 1344]
    for name in ("train_dataloader", "val_dataloader"):
        ops = c.resolved_config[name]["dataset"]["transforms"]["ops"]
        assert next(x["size"] for x in ops if x["type"] == "Resize") == [768, 1344]
    assert c.engine.state_dict()["config"]["expected_input_size"] == [768, 1344]
    # Cached eval anchors are laid out as the [H, W] feature maps generate them in training.
    anchors, valid = c.model.decoder._generate_anchors([[96, 168], [48, 84], [24, 42]])
    assert torch.equal(c.model.decoder.anchors, anchors) and torch.equal(c.model.decoder.valid_mask, valid)
    # The vendor grid wraps every h tokens on a non-square canvas; the override follows token order.
    rows = c.model.encoder.pos_embed2.view(24, 42, 256)
    assert torch.equal(rows[0, 24, :128], rows[0, 0, :128])  # same row: first (row) half equal
    assert not torch.equal(rows[0, 24, 128:], rows[0, 0, 128:])
    assert torch.equal(rows[1, 0, 128:], rows[0, 0, 128:])  # same column: second (column) half equal
    assert not torch.equal(c.model.encoder.pos_embed2, _vendor_position_embedding(c.model, 42, 24))
    assert (c.ema.module.encoder.build_2d_sincos_position_embedding
            is c.model.encoder.build_2d_sincos_position_embedding)
    torch.testing.assert_close(c.ema.module.encoder.pos_embed2, c.model.encoder.pos_embed2, rtol=0, atol=0)
    c.engine.begin_epoch(1)
    with pytest.raises((ValueError, RuntimeError), match="size|shape"):
        c.engine.train_window(torch.rand((2, 3, 1344, 768)), batch()[1])


def test_row_major_position_override_is_the_vendor_embedding_on_squares():
    from sparse_rtdetr.baseline.training_v2b import _row_major_position_embedding
    square = build(input_size=640)
    assert "build_2d_sincos_position_embedding" not in square.model.encoder.__dict__
    for side in (20, 28, 32, 24):
        assert torch.equal(_row_major_position_embedding(side, side), _vendor_position_embedding(square.model, side, side))
    assert not torch.equal(_row_major_position_embedding(42, 24), _vendor_position_embedding(square.model, 42, 24))
    canvas = build(input_size=[768, 1344])
    del canvas.model.encoder.build_2d_sincos_position_embedding
    with pytest.raises(V2BConfigurationError, match="row-major"):
        validate_model_geometry(canvas.model, canvas.config)
