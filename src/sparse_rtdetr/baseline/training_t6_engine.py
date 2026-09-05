"""Production training engine boundary for the detached T6B launch path.

The module is import-safe.  Runtime libraries, models, data sources and the
primary evaluator are resolved only after the T6B entry has accepted a
detached authorization.  CPU tests use the same loop through narrow injected
ports and never need a vendor import.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import random
from typing import Any, Callable, Iterable, Mapping


ENGINE_SCHEMA_VERSION = 1
ENGINE_ID = "rtdetrv2_r18_visdrone_training_t6_engine_v1"
ENGINE_MODULE_RELATIVE_PATH = "src/sparse_rtdetr/baseline/training_t6_engine.py"


class _ProductionCapability:
    """Private identity passed only by the authenticated entry boundary."""


_PRODUCTION_CAPABILITY = _ProductionCapability


__all__ = (
    "TrainingEngineError",
    "validate_training_policy",
    "resolve_production_ports",
    "run_training_engine",
)


class TrainingEngineError(RuntimeError):
    """Raised when the frozen production training boundary is violated."""


def _fail(message: str) -> None:
    raise TrainingEngineError(message)


def _assert_builtin(value: Any, field: str = "value") -> None:
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                _fail(f"{field} has a non-string key")
            _assert_builtin(child, f"{field}.{key}")
        return
    if type(value) is list:
        for index, child in enumerate(value):
            _assert_builtin(child, f"{field}[{index}]")
        return
    if type(value) is float and not math.isfinite(value):
        _fail(f"{field} is non-finite")
    if type(value) not in {str, int, float, bool, type(None)}:
        _fail(f"{field} is not a builtin JSON value")


def _exact(value: Any, keys: set[str], field: str) -> dict[str, Any]:
    if type(value) is not dict:
        _fail(f"{field} must be a builtin dict")
    actual = set(value)
    if actual != keys:
        _fail(f"{field} keys drift: missing={sorted(keys - actual)} extra={sorted(actual - keys)}")
    return value


def _strict_equal(actual: Any, expected: Any, field: str) -> None:
    if type(actual) is not type(expected):
        _fail(f"{field} type drift")
    if type(actual) is dict:
        if set(actual) != set(expected):
            _fail(f"{field} keys drift")
        for key in expected:
            _strict_equal(actual[key], expected[key], f"{field}.{key}")
        return
    if type(actual) is list:
        if len(actual) != len(expected):
            _fail(f"{field} length drift")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _strict_equal(left, right, f"{field}[{index}]")
        return
    if actual != expected:
        _fail(f"{field} value drift")


def _finite_number(value: Any, field: str) -> float:
    if type(value) not in {int, float} or isinstance(value, bool):
        _fail(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{field} must be finite")
    return result


def _finite_observation(value: Any, field: str) -> None:
    if type(value) in {int, float} and not isinstance(value, bool):
        _finite_number(value, field)
    elif type(value) is dict:
        for key, child in value.items():
            _finite_observation(child, f"{field}.{key}")
    elif type(value) is list:
        for index, child in enumerate(value):
            _finite_observation(child, f"{field}[{index}]")


def _canonical(value: Any) -> bytes:
    _assert_builtin(value, "canonical value")
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise TrainingEngineError("value is not canonical JSON") from exc


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: Any) -> str:
    return _sha_bytes(_canonical(value))


def _copy(value: Any) -> Any:
    return copy.deepcopy(value)


def validate_training_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the frozen T1/T2 policy consumed by the real engine."""

    if type(policy) is not dict:
        _fail("training policy must be a builtin dict")
    required = {
        "initialization",
        "topology",
        "schedule",
        "optimizer",
        "learning_rate",
        "amp",
        "ema",
        "evaluator",
        "checkpoint",
        "execution",
    }
    value = _exact(policy, required, "training policy")
    _assert_builtin(value, "training policy")
    initialization = _exact(
        value["initialization"],
        {"random_initialization", "pretrained", "checkpoint", "seed"},
        "initialization",
    )
    _strict_equal(
        initialization,
        {"random_initialization": True, "pretrained": False, "checkpoint": None, "seed": 0},
        "initialization",
    )
    topology = _exact(
        value["topology"],
        {"world_size", "device", "train_batch_size", "development_batch_size"},
        "topology",
    )
    _strict_equal(
        topology,
        {"world_size": 1, "device": "cuda:0", "train_batch_size": 16, "development_batch_size": 32},
        "topology",
    )
    schedule = _exact(
        value["schedule"],
        {"epochs", "development_evaluation_every_epoch", "checkpoint_every_epoch"},
        "schedule",
    )
    _strict_equal(
        schedule,
        {"epochs": 120, "development_evaluation_every_epoch": True, "checkpoint_every_epoch": True},
        "schedule",
    )
    optimizer = _exact(
        value["optimizer"],
        {
            "type",
            "default_lr",
            "backbone_non_norm_lr",
            "betas",
            "default_weight_decay",
            "norm_bn_weight_decay",
            "clip_max_norm",
        },
        "optimizer",
    )
    _strict_equal(
        optimizer,
        {
            "type": "AdamW",
            "default_lr": 0.0001,
            "backbone_non_norm_lr": 0.00001,
            "betas": [0.9, 0.999],
            "default_weight_decay": 0.0001,
            "norm_bn_weight_decay": 0.0,
            "clip_max_norm": 0.1,
        },
        "optimizer",
    )
    learning_rate = _exact(
        value["learning_rate"],
        {
            "warmup_type",
            "warmup_optimizer_steps",
            "scheduler_type",
            "scheduler_step_unit",
            "milestones",
            "gamma",
            "expected_decay_events_within_120_epochs",
        },
        "learning_rate",
    )
    _strict_equal(
        learning_rate,
        {
            "warmup_type": "LinearWarmup",
            "warmup_optimizer_steps": 2000,
            "scheduler_type": "MultiStepLR",
            "scheduler_step_unit": "epoch",
            "milestones": [1000],
            "gamma": 0.1,
            "expected_decay_events_within_120_epochs": 0,
        },
        "learning_rate",
    )
    amp = _exact(
        value["amp"],
        {
            "enabled",
            "scaler_type",
            "init_scale",
            "growth_factor",
            "backoff_factor",
            "growth_interval",
            "nonfinite_events_allowed",
            "skipped_optimizer_steps_allowed",
            "overflow_events_allowed",
        },
        "amp",
    )
    _strict_equal(
        amp,
        {
            "enabled": True,
            "scaler_type": "GradScaler",
            "init_scale": 65536.0,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": 2000,
            "nonfinite_events_allowed": 0,
            "skipped_optimizer_steps_allowed": 0,
            "overflow_events_allowed": 0,
        },
        "amp",
    )
    ema = _exact(
        value["ema"],
        {"enabled", "decay", "warmup_optimizer_updates", "development_weights", "selection_weights"},
        "ema",
    )
    _strict_equal(
        ema,
        {
            "enabled": True,
            "decay": 0.9999,
            "warmup_optimizer_updates": 2000,
            "development_weights": "ema",
            "selection_weights": "ema",
        },
        "ema",
    )
    evaluator = _exact(
        value["evaluator"],
        {"primary_id", "protocol_id", "role", "secondary_can_certify"},
        "evaluator",
    )
    _strict_equal(
        evaluator,
        {
            "primary_id": "visdrone_official_primary_evaluator_v1",
            "protocol_id": "visdrone_official_style_v1",
            "role": "development_only",
            "secondary_can_certify": False,
        },
        "evaluator",
    )
    checkpoint = _exact(
        value["checkpoint"],
        {"roles", "required_atomicity", "required_loadability", "append_only"},
        "checkpoint",
    )
    _strict_equal(
        checkpoint,
        {"roles": ["last", "best", "periodic", "final"], "required_atomicity": True, "required_loadability": True, "append_only": True},
        "checkpoint",
    )
    execution = _exact(value["execution"], {"overwrite", "resume", "retry", "fallback"}, "execution")
    _strict_equal(execution, {"overwrite": False, "resume": False, "retry": False, "fallback": False}, "execution")
    return copy.deepcopy(value)


def _resolve_factory(ports: Mapping[str, Any], name: str) -> Any:
    value = ports.get(name)
    if callable(value):
        return value()
    if value is None:
        _fail(f"ports.{name} is required")
    return value


def _iter_batches(value: Any, field: str) -> Iterable[Any]:
    if callable(value):
        value = value()
    try:
        return iter(value)
    except TypeError as exc:
        raise TrainingEngineError(f"ports.{field} must provide an iterable") from exc


def resolve_production_ports(
    repo_root: str | None = None,
    *,
    data_roles: Mapping[str, Any] | None = None,
    output_root: str | None = None,
) -> dict[str, Any]:
    """Resolve the certified runtime graph only after entry authorization.

    The repository vendors the upstream project as the ``src`` namespace;
    there is no importable ``vendor.rtdetrv2_pytorch`` package.  Keep all of
    these imports in this post-authorization resolver and use the project's
    own isolation helper so registrations do not leak into the caller.
    """

    try:
        torch = importlib.import_module("torch")
        baseline_config = importlib.import_module("sparse_rtdetr.baseline.config")
        baseline_dataset = importlib.import_module("sparse_rtdetr.baseline.dataset")
        baseline_postprocessor = importlib.import_module("sparse_rtdetr.baseline.postprocessor")
        primary_evaluator = importlib.import_module("sparse_rtdetr.baseline.primary_evaluator")
        root = repo_root if repo_root is not None else str(baseline_config._default_repo_root())
        vendor_root = baseline_config._vendor_root(root)
        vendor_config_path = baseline_config._vendor_config_path(root)
        with baseline_config._vendor_path(vendor_root):
            yaml_config_module = importlib.import_module("src.core.yaml_config")
            solver_module = importlib.import_module("src.solver")
            optim_module = importlib.import_module("src.optim")
            importlib.import_module("src.data")
            importlib.import_module("src.nn")
            importlib.import_module("src.zoo.rtdetr")
    except TrainingEngineError:
        raise
    except Exception as exc:
        raise TrainingEngineError("certified production runtime is unavailable") from exc

    YAMLConfig = getattr(yaml_config_module, "YAMLConfig", None)
    tasks = getattr(solver_module, "TASKS", None)
    grad_scaler = getattr(optim_module, "GradScaler", None)
    model_ema = getattr(optim_module, "ModelEMA", None)
    if not callable(YAMLConfig) or type(tasks) is not dict or not callable(grad_scaler) or not callable(model_ema):
        _fail("certified vendor API surface is incomplete")
    if "detection" not in tasks:
        _fail("certified vendor TASKS does not expose detection")
    visdrone_dataset = getattr(baseline_dataset, "VisDroneCocoDetection", None)
    visdrone_postprocessor = getattr(baseline_postprocessor, "VisDronePostProcessor", None)
    evaluate_primary = getattr(primary_evaluator, "evaluate_primary_v1", None)
    if not callable(visdrone_dataset) or not callable(visdrone_postprocessor) or not callable(evaluate_primary):
        _fail("certified baseline runtime API surface is incomplete")

    state: dict[str, Any] = {}

    def runtime_config() -> Any:
        config = state.get("config")
        if config is None:
            with baseline_config._vendor_path(vendor_root):
                config = YAMLConfig(
                    str(vendor_config_path),
                    PResNet={"pretrained": False},
                    num_classes=10,
                    remap_mscoco_category=False,
                    device="cuda:0",
                    output_dir=output_root,
                )
            state["config"] = config
        return config

    def model_factory() -> Any:
        model = runtime_config().model
        to = getattr(model, "to", None)
        if not callable(to):
            _fail("certified model does not expose to")
        return to(device="cuda:0")

    def criterion_factory() -> Any:
        return runtime_config().criterion

    def optimizer_factory() -> Any:
        return runtime_config().optimizer

    def scheduler_factory() -> Any:
        return runtime_config().lr_scheduler

    def scaler_factory() -> Any:
        return grad_scaler(
            init_scale=65536.0,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000,
        )

    def ema_factory() -> Any:
        return model_ema(runtime_config().model, decay=0.9999, warmups=2000)

    def dataset_factory(role: str) -> Any:
        if type(data_roles) is not dict or role not in {"train_core", "development"}:
            _fail("production dataset factory requires detached train/development bindings")
        binding = data_roles[role]
        if type(binding) is not dict or type(binding.get("root")) is not str:
            _fail(f"production {role} binding is invalid")
        runtime = baseline_config.build_runtime_config(root, binding["root"], role)
        return visdrone_dataset(
            img_folder=runtime["dataloader"]["img_folder"],
            ann_file=runtime["dataloader"]["annotation_file"],
            transforms=None,
            return_masks=False,
            remap_mscoco_category=False,
            vendor_root=vendor_root,
            role=role,
        )

    def train_batches() -> Any:
        return runtime_config().train_dataloader

    def development_batches() -> Any:
        return runtime_config().val_dataloader

    def postprocessor_factory() -> Any:
        return visdrone_postprocessor(vendor_root=vendor_root)

    class PrimaryEvaluator:
        role = "development_only"

        def evaluate(self, weights: Any, batches: Any, epoch: int) -> dict[str, Any]:
            del weights
            del epoch
            values = list(batches)
            if len(values) != 1:
                _fail("primary evaluator requires one bound development input")
            contract = state.get("primary_contract")
            if contract is None:
                contract = primary_evaluator.load_primary_evaluator_contract(root)
                state["primary_contract"] = contract
            input_value = values[0]
            result = evaluate_primary(input_value, contract)
            primary_evaluator.validate_primary_evaluator_result(result, input_value, contract)
            return {
                "primary_score": result.AP,
                "evaluator_id": primary_evaluator.PRIMARY_EVALUATOR_ID,
                "protocol_id": primary_evaluator.PRIMARY_PROTOCOL_ID,
                "role": "development_only",
            }

    class CheckpointWriter:
        def __init__(self) -> None:
            if type(output_root) is not str or not output_root:
                _fail("production checkpoint writer requires an evidence root")
            self._root = importlib.import_module("pathlib").Path(output_root)

        def save_atomic(self, role: str, epoch: int, state_value: dict[str, Any]) -> dict[str, Any]:
            if role not in {"last", "best", "periodic", "final"}:
                _fail("unknown checkpoint role")
            os_module = importlib.import_module("os")
            tempfile_module = importlib.import_module("tempfile")
            self._root.mkdir(parents=True, exist_ok=True)
            target = self._root / f"checkpoint-{role}-{epoch:04d}.pth"
            fd, temporary = tempfile_module.mkstemp(prefix=f".{target.name}.", dir=str(self._root))
            try:
                with os_module.fdopen(fd, "wb") as stream:
                    torch.save(state_value, stream)
                    stream.flush()
                    os_module.fsync(stream.fileno())
                os_module.replace(temporary, target)
                directory_fd = os_module.open(self._root, os_module.O_RDONLY | getattr(os_module, "O_DIRECTORY", 0))
                try:
                    os_module.fsync(directory_fd)
                finally:
                    os_module.close(directory_fd)
            except Exception:
                try:
                    os_module.unlink(temporary)
                except OSError:
                    pass
                raise
            return {"role": role, "epoch": epoch, "path": str(target), "state_sha256": _digest(state_value)}

        def is_loadable(self, reference: dict[str, Any]) -> bool:
            path = importlib.import_module("pathlib").Path(reference.get("path", ""))
            if not path.is_file() or path.is_symlink():
                return False
            try:
                torch.load(path, map_location="cpu", weights_only=False)
            except Exception:
                return False
            return True

    resolved: dict[str, Any] = {
        "torch": torch,
        "YAMLConfig": YAMLConfig,
        "TASKS": tasks,
        "GradScaler": grad_scaler,
        "ModelEMA": model_ema,
        "baseline_config": baseline_config,
        "VisDroneCocoDetection": visdrone_dataset,
        "VisDronePostProcessor": visdrone_postprocessor,
        "primary_evaluator": primary_evaluator,
        "model_factory": model_factory,
        "criterion_factory": criterion_factory,
        "optimizer_factory": optimizer_factory,
        "scheduler_factory": scheduler_factory,
        "scaler_factory": scaler_factory,
        "ema_factory": ema_factory,
        "train_batches": train_batches,
        "development_batches": development_batches,
        "checkpoint_writer": CheckpointWriter,
        "evaluator": PrimaryEvaluator,
        "dataset_factory": dataset_factory,
        "postprocessor_factory": postprocessor_factory,
        "seed": lambda seed: torch.manual_seed(seed),
        "production_authorized": True,
    }
    return resolved


def _loss_value(loss: Any) -> float:
    if type(loss) in {int, float} and not isinstance(loss, bool):
        return _finite_number(loss, "loss")
    item = getattr(loss, "item", None)
    if callable(item):
        return _finite_number(item(), "loss.item()")
    _fail("criterion must return a numeric or scalar-like loss")


def _backward(ports: Mapping[str, Any], scaled_loss: Any) -> None:
    backward = ports.get("backward")
    if callable(backward):
        backward(scaled_loss)
        return
    method = getattr(scaled_loss, "backward", None)
    if callable(method):
        method()
        return
    _fail("no backward port is available")


def _snapshot_value(value: Any) -> Any:
    if type(value) in {str, int, float, bool, type(None)}:
        return value
    if type(value) is dict:
        return {str(key): _snapshot_value(child) for key, child in value.items()}
    if type(value) is list:
        return [_snapshot_value(child) for child in value]
    return {"type": type(value).__name__}


def _checkpoint(writer: Any, role: str, epoch: int, state: Mapping[str, Any]) -> dict[str, Any]:
    save = getattr(writer, "save_atomic", None)
    if not callable(save):
        _fail("checkpoint writer must expose save_atomic")
    reference = save(role, epoch, copy.deepcopy(dict(state)))
    if type(reference) is not dict:
        _fail("checkpoint writer returned a non-dict reference")
    _assert_builtin(reference, "checkpoint reference")
    loadable = getattr(writer, "is_loadable", None)
    if not callable(loadable) or loadable(reference) is not True:
        _fail(f"checkpoint {role} is not loadable")
    return copy.deepcopy(reference)


def _validate_engine_result(value: Any) -> dict[str, Any]:
    """Validate the engine-owned terminal result before another layer binds it."""

    keys = {
        "schema_version",
        "engine_id",
        "mode",
        "seed",
        "epochs",
        "optimizer_type",
        "scheduler_type",
        "warmup_optimizer_steps",
        "optimizer_steps",
        "nonfinite_events",
        "skipped_optimizer_steps",
        "overflow_events",
        "evaluator_role",
        "test_accessed",
        "confirmatory_accessed",
        "speed_measured",
        "epoch_reports",
        "checkpoint_references",
        "terminal",
        "scientific_training_certified",
        "engine_result_sha256",
    }
    result = _exact(value, keys, "engine result")
    _assert_builtin(result, "engine result")
    if result["schema_version"] != ENGINE_SCHEMA_VERSION or result["engine_id"] != ENGINE_ID or result["mode"] not in {"production", "cpu_fake"}:
        _fail("engine result identity drift")
    if result["seed"] != 0 or result["epochs"] != 120 or result["optimizer_type"] != "AdamW" or result["scheduler_type"] != "MultiStepLR" or result["warmup_optimizer_steps"] != 2000:
        _fail("engine result policy drift")
    for field in ("optimizer_steps", "nonfinite_events", "skipped_optimizer_steps", "overflow_events"):
        if type(result[field]) is not int or result[field] < 0:
            _fail(f"engine result {field} drift")
    for field in ("test_accessed", "confirmatory_accessed", "speed_measured", "scientific_training_certified"):
        if result[field] is not False:
            _fail(f"engine result {field} must remain false")
    if result["evaluator_role"] != "development_only" or result["terminal"] != "TERMINAL_COMPLETE" or type(result["epoch_reports"]) is not list or type(result["checkpoint_references"]) is not list:
        _fail("engine result terminal evidence drift")
    _sha = result["engine_result_sha256"]
    if type(_sha) is not str or len(_sha) != 64 or any(char not in "0123456789abcdef" for char in _sha):
        _fail("engine result digest drift")
    body = _copy(result)
    del body["engine_result_sha256"]
    if _digest(body) != _sha:
        _fail("engine result aggregate drift")
    return _copy(result)


def run_training_engine(
    policy: Mapping[str, Any],
    ports: Mapping[str, Any] | None = None,
    *,
    mode: str = "production",
    _authorization_capability: object | None = None,
) -> dict[str, Any]:
    """Execute the frozen 120-epoch loop through explicit runtime ports.

    ``cpu_fake`` is only a test mode.  It does not grant authorization and is
    never selected by the production entry descriptor.
    """

    checked_policy = validate_training_policy(policy)
    if mode not in {"production", "cpu_fake"}:
        _fail("unknown engine mode")
    if ports is None:
        ports = resolve_production_ports()
    if type(ports) is not dict:
        _fail("ports must be a builtin dict")
    _assert_builtin({key: _snapshot_value(value) for key, value in ports.items()}, "ports")
    authorizer = ports.get("authorize_production")
    if callable(authorizer) and authorizer() is not True:
        _fail("production port authorization was not granted by the entry boundary")
    if mode == "production":
        if ports.get("production_authorized") is not True or _authorization_capability is not _PRODUCTION_CAPABILITY:
            _fail("production engine requires entry-authorized ports")

    seed = ports.get("seed")
    if seed is None:
        random.seed(0)
    elif callable(seed):
        seed(0)
    else:
        _fail("ports.seed must be callable")

    model = _resolve_factory(ports, "model_factory")
    criterion = _resolve_factory(ports, "criterion_factory")
    optimizer = _resolve_factory(ports, "optimizer_factory")
    scheduler = _resolve_factory(ports, "scheduler_factory")
    scaler = _resolve_factory(ports, "scaler_factory")
    ema = _resolve_factory(ports, "ema_factory")
    writer = _resolve_factory(ports, "checkpoint_writer")
    evaluator = _resolve_factory(ports, "evaluator")
    train_source = ports.get("train_batches")
    development_source = ports.get("development_batches")
    if train_source is None or development_source is None:
        _fail("train_core and development ports are required")
    if any(name in ports for name in ("test_batches", "confirmatory_batches", "speed_measurement")):
        _fail("test, confirmatory and speed ports are forbidden")
    if not callable(getattr(evaluator, "evaluate", None)):
        _fail("primary evaluator must expose evaluate")
    if getattr(evaluator, "role", "development_only") != "development_only":
        _fail("evaluator is not development-only")

    epochs = checked_policy["schedule"]["epochs"]
    optimizer_steps = 0
    nonfinite_events = 0
    skipped_steps = 0
    overflow_events = 0
    epoch_reports: list[dict[str, Any]] = []
    checkpoint_references: list[dict[str, Any]] = []
    best_score: float | None = None

    for epoch in range(1, epochs + 1):
        batches = _iter_batches(train_source, "train_batches")
        batch_count = 0
        epoch_loss = 0.0
        for batch in batches:
            batch_count += 1
            zero_grad = getattr(optimizer, "zero_grad", None)
            if callable(zero_grad):
                zero_grad()
            forward = model if callable(model) else getattr(model, "__call__", None)
            if not callable(forward):
                _fail("model must be callable")
            output = forward(batch)
            loss_fn = criterion if callable(criterion) else getattr(criterion, "__call__", None)
            if not callable(loss_fn):
                _fail("criterion must be callable")
            loss = loss_fn(output, batch)
            value = _loss_value(loss)
            if not math.isfinite(value):
                nonfinite_events += 1
                _fail("non-finite training loss")
            epoch_loss += value
            scale = getattr(scaler, "scale", None)
            scaled = scale(loss) if callable(scale) else loss
            _backward(ports, scaled)
            step = getattr(scaler, "step", None)
            if callable(step):
                step_result = step(optimizer)
                if step_result is False:
                    skipped_steps += 1
            else:
                optimizer_step = getattr(optimizer, "step", None)
                if not callable(optimizer_step):
                    _fail("optimizer must expose step")
                optimizer_step()
            update = getattr(scaler, "update", None)
            if callable(update):
                update()
            optimizer_steps += 1
            ema_update = getattr(ema, "update", None)
            if not callable(ema_update):
                _fail("EMA port must expose update")
            ema_update(model)
        if batch_count == 0:
            _fail("train_core yielded no batches")

        scheduler_step = getattr(scheduler, "step", None)
        if not callable(scheduler_step):
            _fail("scheduler must expose step")
        scheduler_step()

        development_batches = _iter_batches(development_source, "development_batches")
        evaluation = evaluator.evaluate(ema, development_batches, epoch)
        _assert_builtin(evaluation, "evaluation")
        _finite_observation(evaluation, "evaluation")
        score = evaluation.get("primary_score") if type(evaluation) is dict else None
        if score is not None:
            score = _finite_number(score, "evaluation.primary_score")
        state = {"epoch": epoch, "loss": epoch_loss / batch_count, "evaluation": evaluation}
        last = _checkpoint(writer, "last", epoch, state)
        checkpoint_references.append({"role": "last", "epoch": epoch, "reference": last})
        if epoch % 10 == 0:
            periodic = _checkpoint(writer, "periodic", epoch, state)
            checkpoint_references.append({"role": "periodic", "epoch": epoch, "reference": periodic})
        if best_score is None or (score is not None and score > best_score):
            best_score = score if score is not None else best_score
            best = _checkpoint(writer, "best", epoch, state)
            checkpoint_references.append({"role": "best", "epoch": epoch, "reference": best})
        epoch_reports.append(
            {"epoch": epoch, "batch_count": batch_count, "mean_loss": epoch_loss / batch_count, "evaluation": evaluation}
        )

    final = _checkpoint(writer, "final", epochs, {"epoch": epochs, "epochs": epochs, "best_score": best_score})
    checkpoint_references.append({"role": "final", "epoch": epochs, "reference": final})

    counters = ports.get("amp_counters")
    if callable(counters):
        observed = counters()
        if type(observed) is not dict:
            _fail("amp_counters must return a builtin dict")
        if set(observed) != {"overflow_events", "skipped_optimizer_steps"}:
            _fail("amp_counters key set drift")
        if type(observed["overflow_events"]) is not int or type(observed["skipped_optimizer_steps"]) is not int:
            _fail("amp counters must be builtin ints")
        if observed["overflow_events"] < 0 or observed["skipped_optimizer_steps"] < 0:
            _fail("amp counters cannot be negative")
        overflow_events = observed["overflow_events"]
        skipped_steps = observed["skipped_optimizer_steps"]
    if nonfinite_events != 0 or skipped_steps != 0 or overflow_events != 0:
        _fail("AMP/non-finite policy was violated")

    body = {
        "schema_version": ENGINE_SCHEMA_VERSION,
        "engine_id": ENGINE_ID,
        "mode": mode,
        "seed": 0,
        "epochs": epochs,
        "optimizer_type": checked_policy["optimizer"]["type"],
        "scheduler_type": checked_policy["learning_rate"]["scheduler_type"],
        "warmup_optimizer_steps": checked_policy["learning_rate"]["warmup_optimizer_steps"],
        "optimizer_steps": optimizer_steps,
        "nonfinite_events": nonfinite_events,
        "skipped_optimizer_steps": skipped_steps,
        "overflow_events": overflow_events,
        "evaluator_role": "development_only",
        "test_accessed": False,
        "confirmatory_accessed": False,
        "speed_measured": False,
        "epoch_reports": epoch_reports,
        "checkpoint_references": checkpoint_references,
        "terminal": "TERMINAL_COMPLETE",
        "scientific_training_certified": False,
    }
    _assert_builtin(body, "engine result")
    result = {**body, "engine_result_sha256": _digest(body)}
    return _validate_engine_result(result)
