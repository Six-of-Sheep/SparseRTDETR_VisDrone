"""MIX: fixed two-canvas schedule (L=768x1344 / H=1152x2016) continued from the seed0 native e060 state.

TASK_MIX_SCALE_001.  Zero new modules, parameters, losses or teachers.  The MIX engine is the S15 bridge
of the certified seed0 e060 complete state (model/AdamW/EMA/engine/RNG unchanged; only the canvas caches
are the native 1152x2016 regeneration, exactly as S15).  The only scientific change is the canvas used by
each optimizer update, read from a pre-declared table:

* epochs 61-117: per epoch, windows w=0..303 are sorted by (SHA256("MIX_LH_V1|seed=0|epoch=e|window=w"), w);
  the first 152 are L, the other 152 are H.  Samples are NOT reordered: training follows window order.
* epochs 118-120: every window is H.
* both microbatches of an update use the same canvas (one window = one canvas).

The canvas reaches the real loader through the batch-sampler key ``(epoch, position, index, canvas)``.
Prefetching workers render exactly the canvas carried by the key with the native ``RawAugmentation``
of that canvas; they never hold a default canvas.  During training RT-DETRv2 regenerates anchors and
position embeddings from the actual feature maps (``self.training`` branch), so no size cache is
carried from one canvas to the next.  The persistent caches only matter for evaluation, where they are
the native regeneration at the declared evaluation canvas.

Test schedules (constant L, constant H, acceptance L->H->L) have their own identities and can never be
formal: formal states bind the frozen table SHA.
"""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib

import torch
from torch.utils.data import DataLoader

from ..baseline.training_v2b import logical_batch_indices, seed_worker
from ..baseline.training_v2b_data import _sample_seed
from ..localization.data import RawAugmentation, WindowSampler, collate_window
from ..localization.material_paths import MANIFESTS
from ..localization.runio import digest
from ..protocol_r18 import checkpoint as native
from ..protocol_r18.data import BoundDataset, input_record
from ..protocol_r18.engine import tensor_digest
from ..protocol_r18.geometry import geometry_identity, validate_geometry
from ..protocol_r18.model import Config, build
from . import scale_baseline as S

VERSION = "MIX-LH-scale-v1"
ARM = "MIX"
SEED = 0
L_HW, H_HW = (768, 1344), (1152, 2016)
CANVAS = {"L": L_HW, "H": H_HW}
WINDOWS_PER_EPOCH = 304
WINDOW_SIZE = 16
MEMBERS = 4869
FIRST_EPOCH, LAST_MIX_EPOCH, LAST_EPOCH = 61, 117, 120
PARENT_UPDATE = (FIRST_EPOCH - 1) * WINDOWS_PER_EPOCH
END_UPDATE = LAST_EPOCH * WINDOWS_PER_EPOCH
L_PER_MIX_EPOCH = 152
RULE = "MIX_LH_V1|seed={seed}|epoch={epoch}|window={window}"
TABLE_SCHEMA = "mix-lh-v1-table"
TABLE_HEADER = "update\tepoch\twindow\tcanvas\tsha256"
TOTALS = {"L": 8664, "H": 9576}
FORMAL = "formal"
TEST_MODES = ("test-constant-L", "test-constant-H", "acceptance-LHL")
ACCEPTANCE_CANVASES = {PARENT_UPDATE + 1: "L", PARENT_UPDATE + 2: "H", PARENT_UPDATE + 3: "L"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def update_of(epoch, window):
    return (epoch - 1) * WINDOWS_PER_EPOCH + window + 1


# ---------------------------------------------------------------- the pre-declared scale table
def rule_digest(epoch, window, seed=SEED):
    return hashlib.sha256(RULE.format(seed=seed, epoch=epoch, window=window).encode("utf-8")).hexdigest()


def generate_rows(seed=SEED):
    """(update, epoch, window, canvas, digest) for e061-e120.  Uses no RNG; depends on epoch/window only."""
    rows = []
    for epoch in range(FIRST_EPOCH, LAST_EPOCH + 1):
        digests = [rule_digest(epoch, w, seed) for w in range(WINDOWS_PER_EPOCH)]
        low = (set(sorted(range(WINDOWS_PER_EPOCH), key=lambda w: (digests[w], w))[:L_PER_MIX_EPOCH])
               if epoch <= LAST_MIX_EPOCH else set())
        for w in range(WINDOWS_PER_EPOCH):
            rows.append((update_of(epoch, w), epoch, w, "L" if w in low else "H", digests[w]))
    return rows


def table_bytes(rows):
    lines = [TABLE_HEADER] + ["\t".join(map(str, row)) for row in rows]
    return ("\n".join(lines) + "\n").encode("ascii")


def parse_table(raw):
    lines = raw.decode("ascii").split("\n")
    require(lines[0] == TABLE_HEADER and lines[-1] == "", "MIX scale table header/termination")
    rows = []
    for line in lines[1:-1]:
        u, e, w, c, d = line.split("\t")
        rows.append((int(u), int(e), int(w), c, d))
    return rows


def table_counts(rows):
    """Row coverage, per-epoch quota and totals; raises on any deviation."""
    require(len(rows) == END_UPDATE - PARENT_UPDATE
            and [r[0] for r in rows] == list(range(PARENT_UPDATE + 1, END_UPDATE + 1)), "MIX table update coverage")
    per_epoch = {}
    for u, e, w, c, _ in rows:
        require(u == update_of(e, w) and c in CANVAS, "MIX table row identity")
        per_epoch.setdefault(e, {"L": 0, "H": 0})[c] += 1
    for e in range(FIRST_EPOCH, LAST_EPOCH + 1):
        want = {"L": L_PER_MIX_EPOCH, "H": WINDOWS_PER_EPOCH - L_PER_MIX_EPOCH} if e <= LAST_MIX_EPOCH \
            else {"L": 0, "H": WINDOWS_PER_EPOCH}
        require(per_epoch.get(e) == want, f"MIX table quota differs at epoch {e}: {per_epoch.get(e)}")
    totals = {k: sum(v[k] for v in per_epoch.values()) for k in CANVAS}
    require(totals == TOTALS, f"MIX table totals {totals}")
    return {"rows": len(rows), "totals": totals, "per_epoch_mixed": {"L": L_PER_MIX_EPOCH, "H": 152},
            "per_epoch_tail_118_120": {"L": 0, "H": WINDOWS_PER_EPOCH}}


class Schedule:
    """Canvas per global update.  ``formal`` comes only from the frozen, rule-verified table."""

    def __init__(self, mode, canvases, identity):
        require(mode == FORMAL or mode in TEST_MODES, f"unknown MIX schedule mode {mode}")
        self.mode, self._canvases, self.identity = mode, dict(canvases), dict(identity)

    def canvas(self, update):
        require(update in self._canvases, f"MIX schedule {self.mode} has no canvas for update {update}")
        return self._canvases[update]

    def __reduce__(self):  # never shipped to loader workers; the sampler stays in the main process
        raise TypeError("MIX schedule must not be pickled into a worker")


def formal_schedule(path, sha256):
    with open(path, "rb") as stream:
        raw = stream.read()
    require(hashlib.sha256(raw).hexdigest() == sha256, "MIX scale table SHA differs from the contract")
    require(raw == table_bytes(generate_rows()), "MIX scale table bytes are not the declared rule")
    rows = parse_table(raw)
    counts = table_counts(rows)
    return Schedule(FORMAL, {u: c for u, _, _, c, _ in rows},
                    {"mode": FORMAL, "schema": TABLE_SCHEMA, "rule": RULE, "seed": SEED, "table_sha256": sha256,
                     "table_bytes": len(raw), **counts})


def test_schedule(mode):
    require(mode in TEST_MODES, f"not a MIX test schedule: {mode}")
    if mode == "acceptance-LHL":
        canvases = dict(ACCEPTANCE_CANVASES)
    else:
        canvases = {u: mode[-1] for u in range(PARENT_UPDATE + 1, END_UPDATE + 1)}
    return Schedule(mode, canvases, {"mode": mode, "table_sha256": None, "formal_usable": False,
                                     "canvases": {str(u): c for u, c in canvases.items()} if mode == "acceptance-LHL"
                                     else f"constant {mode[-1]} for every update"})


# ---------------------------------------------------------------- canvas-keyed real loader
class MixDataset(BoundDataset):
    """seed0 train_core.  The sampler key carries the canvas; there is no default canvas."""

    def __init__(self, train_core_binding, seed=SEED):
        require(seed == SEED, "MIX is seed0 only")
        r = train_core_binding["roots"]
        super().__init__(role="train_core", manifest_path=train_core_binding["manifest"], images_path=r["images"],
                         annotations_path=r["annotations"], seed=seed, root_binding=r, input_hw=L_HW)
        self.transforms = {k: RawAugmentation(hw) for k, hw in CANVAS.items()}
        self.transform = None

    def __getitem__(self, key):
        epoch, position, index, canvas = key
        require(canvas in CANVAS and self.transform is None, "MIX worker key canvas")
        self.transform = self.transforms[canvas]
        try:
            pixels, target, trace = super().__getitem__((epoch, position, index))
        finally:
            self.transform = None
        require(tuple(pixels.shape) == (3, *CANVAS[canvas]), "MIX worker rendered another canvas")
        trace["spawn_observer"]["canvas"] = canvas  # observer only; excluded from the scientific trace digest
        return pixels, target, trace


class MixWindowSampler(WindowSampler):
    """Native window order; each key additionally carries the scheduled canvas of its window."""

    def __init__(self, dataset_size, *, seed, epoch, completed_windows, schedule, window_limit=None):
        super().__init__(dataset_size, seed=seed, epoch=epoch, completed_windows=completed_windows,
                         window_limit=window_limit)
        self.schedule = schedule

    def __iter__(self):
        for batch in super().__iter__():
            epoch, position, _ = batch[0]
            canvas = self.schedule.canvas(update_of(epoch, position // WINDOW_SIZE))
            yield [(e, p, i, canvas) for e, p, i in batch]


def training_loader(dataset, *, epoch, completed_windows, schedule, workers=2, window_limit=None):
    """Same DataLoader arguments as the native localization.data.training_loader; only the sampler key differs."""
    require(isinstance(dataset, MixDataset) and dataset.role == "train_core" and len(dataset) == MEMBERS
            and dataset.seed == SEED and 1 <= epoch <= 120 and 0 <= completed_windows <= WINDOWS_PER_EPOCH,
            "invalid MIX loader cursor")
    generator = torch.Generator().manual_seed(dataset.seed)
    return DataLoader(dataset, batch_sampler=MixWindowSampler(len(dataset), seed=dataset.seed, epoch=epoch,
                                                              completed_windows=completed_windows, schedule=schedule,
                                                              window_limit=window_limit),
                      num_workers=workers, collate_fn=collate_window, worker_init_fn=seed_worker,
                      multiprocessing_context="spawn" if workers else None,
                      prefetch_factor=2 if workers else None, persistent_workers=False, generator=generator,
                      timeout=300 if workers else 0)


NATIVE_RECEIPT_FIELDS = ("update", "epoch", "window", "image_ids_sha256", "gt_ids_sha256", "augmentation_seeds_sha256",
                         "augmentation_trace_sha256", "augmented_samples_sha256", "input_tensor_sha256",
                         "target_sha256", "augmentation_strong")


def windows(data, completed_updates, schedule, *, limit=None):
    """Rest of one epoch (or a bounded prefix), as tools/query_local/formal_data.windows, plus the canvas.

    The receipt holds the native fields (same schema/values as the native same-canvas loader) and
    canvas, expected_hw, actual_hw (from the collated tensor) and the worker-observed canvas.
    """
    require(type(completed_updates) is int and PARENT_UPDATE <= completed_updates <= END_UPDATE,
            "MIX update cursor outside e060-e120")
    require(isinstance(data, MixDataset) and (data.role, data.seed, len(data), data.manifest_sha256) ==
            ("train_core", SEED, MEMBERS, MANIFESTS["train_core"][0]), "MIX loader requires bound seed0 train_core")
    if completed_updates == END_UPDATE:
        require(limit is None, "no MIX window after e120")
        return
    completed_epochs, first = divmod(completed_updates, WINDOWS_PER_EPOCH)
    epoch = completed_epochs + 1
    loader = training_loader(data, epoch=epoch, completed_windows=first, schedule=schedule, window_limit=limit)
    expected = logical_batch_indices(MEMBERS, seed=SEED, epoch=epoch, completed_batches=first)
    seen = 0
    for images, targets, traces in loader:
        window = first + seen
        update = completed_updates + seen + 1
        canvas = schedule.canvas(update)
        indices = next(expected)
        ids = [data.records[index]["stable_image_id"] for index in indices]
        positions = list(range(window * WINDOW_SIZE, (window + 1) * WINDOW_SIZE))
        seeds = [_sample_seed(SEED, epoch, pos, image_id) for pos, image_id in zip(positions, ids)]
        if ([trace["image_id"] for trace in traces] != ids
                or [target["image_id"] for target in targets] != ids
                or [trace["position"] for trace in traces] != positions
                or [trace["sample_seed"] for trace in traces] != seeds
                or any(trace["epoch"] != epoch or trace["augmentation_strong"] != (epoch < 118) for trace in traces)):
            raise ValueError("MIX actual window order, identity or augmentation differs from the native schedule")
        workers = sorted({trace["spawn_observer"]["canvas"] for trace in traces})
        actual = list(images.shape[-2:])
        require(workers == [canvas] and actual == list(CANVAS[canvas]) and images.shape[0] == WINDOW_SIZE,
                f"MIX window {update}: scheduled {canvas}, worker {workers}, tensor {actual}")
        audit = input_record(images, targets, traces)
        seen += 1
        yield images, targets, {
            "update": update, "epoch": epoch, "window": window,
            "image_ids_sha256": digest(audit["ids"]), "gt_ids_sha256": digest(audit["GT_ids"]),
            "augmentation_seeds_sha256": digest(seeds), "augmentation_trace_sha256": audit["metadata_sha256"],
            "augmented_samples_sha256": digest(audit["augmented_sha256"]),
            "input_tensor_sha256": audit["input_tensor_sha256"], "target_sha256": audit["target_sha256"],
            "augmentation_strong": epoch < 118,
            "canvas": canvas, "expected_hw": list(CANVAS[canvas]), "actual_hw": actual, "worker_canvas": workers[0],
            "schedule_mode": schedule.mode,
        }
    require(seen == len(loader), "MIX loader ended before its declared window count")


def native_fields(receipt):
    return {k: receipt[k] for k in NATIVE_RECEIPT_FIELDS}


# ---------------------------------------------------------------- engine
def base_config(fixture=False):
    """The engine's own (state) recipe: the S15 1152x2016 production recipe, or its CPU FP32 bs1 fixture."""
    return Config(seed=SEED, input_hw=H_HW, physical_batch_size=1, amp_dtype="float32") if fixture \
        else S.production_config(SEED)


def canvas_config(base, canvas):
    """Recipe of one update: the base recipe with only input_hw replaced (L update = native 768 recipe)."""
    return dataclasses.replace(base, input_hw=CANVAS[canvas])


def create(parent, device, schedule, *, fixture=False, activation_checkpoint=True):
    require(isinstance(schedule, Schedule), "MIX engine requires a schedule")
    e, checks = S.create(parent, device, base_config(True) if fixture else None, seed=SEED, fixture=fixture,
                         activation_checkpoint=activation_checkpoint)
    e.r25_arm = ARM
    e.mix_schedule = schedule
    e.mix_base_config = e.config
    return e, {**checks, "schedule": schedule.identity, "engine_base_config": e.config.binding(),
               "L_update_config": canvas_config(e.config, "L").binding(),
               "H_update_config": canvas_config(e.config, "H").binding()}


@contextlib.contextmanager
def _update_config(e, canvas):
    base = e.mix_base_config
    require(e.config is base, "MIX engine recipe was not restored after the previous window")
    e.config = canvas_config(base, canvas)
    try:
        yield e.config
    finally:
        e.config = base


def train_window(e, images, targets, receipt, *, capture=False):
    """One native update on the scheduled canvas; any canvas/receipt/cursor mismatch is refused before compute."""
    update = e.completed_updates + 1
    canvas = e.mix_schedule.canvas(update)
    hw = list(CANVAS[canvas])
    require(receipt["update"] == update and receipt["schedule_mode"] == e.mix_schedule.mode
            and receipt["canvas"] == receipt["worker_canvas"] == canvas
            and receipt["expected_hw"] == receipt["actual_hw"] == hw and list(images.shape[-2:]) == hw,
            f"MIX update {update}: window canvas/cursor differs from the {e.mix_schedule.mode} schedule ({canvas})")
    with _update_config(e, canvas):
        result = e.train_window(images, targets, capture=capture)
    return {**result, "canvas": canvas, "input_hw": hw}


def capture(e):
    require(not e.phase and not e.failed and all(p.grad is None for p in e.model.parameters())
            and e.config is e.mix_base_config, "not a clean MIX update boundary")
    return S._cpu({"version": VERSION, "arm": e.r25_arm, "parent_sha256": e.r25_parent,
                   "config": e.config.binding(), "schedule": e.mix_schedule.identity,
                   "activation_checkpoint": e.s15_activation_checkpoint,
                   "model": e.model.state_dict(), "model_modes": {n: m.training for n, m in e.model.named_modules()},
                   "optimizer": e.optimizer.state_dict(),
                   "optimizer_names": [g["param_names"] for g in e.optimizer.param_groups],
                   "ema": e.ema.state_dict(), "engine": e.state_dict(), "rng": native.capture_rng(e.device)})


def restore(e, state):
    """Load a verified MIX complete state into an engine of the same schedule/recipe/memory mode."""
    require((state["version"], state["arm"], state["parent_sha256"], state["config"], state["schedule"],
             state["activation_checkpoint"]) ==
            (VERSION, ARM, e.r25_parent, e.mix_base_config.binding(), e.mix_schedule.identity,
             e.s15_activation_checkpoint), "MIX cross-version/schedule/recipe/memory-mode state")
    require(state["optimizer_names"] == [g["param_names"] for g in e.optimizer.param_groups], "MIX optimizer names")
    e.model.load_state_dict(state["model"], strict=True)
    e.optimizer.load_state_dict(state["optimizer"])
    e.ema.load_state_dict(state["ema"])
    modes = dict(e.model.named_modules())
    require(set(modes) == set(state["model_modes"]), "MIX training mode keys differ")
    for name, mode in state["model_modes"].items():
        modes[name].training = mode
    e.load_state_dict(state["engine"])
    native.restore_rng(state["rng"], e.device)
    validate_geometry(e.model)
    validate_geometry(e.ema.module)
    require(native.exact_equal(capture(e), state), "MIX complete restored state not exact")
    return {"full_state_exact": True, "update": e.completed_updates}


# ---------------------------------------------------------------- evaluation model
def evaluation_model(ema_state, canvas):
    """Native model at the evaluation canvas loaded with the e120 EMA tensors.

    The state's caches are first proven to be the native regeneration of the engine canvas (H); for an
    L readout only decoder anchors/valid_mask are replaced by the native L regeneration.  Every learned,
    EMA and BN tensor is byte-identical; at H the complete state loads unchanged.
    """
    require(canvas in CANVAS, "MIX evaluation canvas")
    model, _, _ = build(Config(seed=SEED, input_hw=CANVAS[canvas]), synthetic=True)
    validate_geometry(model)
    own = model.state_dict()
    keys = sorted(k for k in ema_state if S._is_size_cache(k))
    require(keys == ["decoder.anchors", "decoder.valid_mask"], f"unexpected size caches {keys}")
    anchors, mask = model.decoder._generate_anchors(spatial_shapes=S._parent_level_shapes(model, H_HW))
    require(native.exact_equal(ema_state["decoder.anchors"], anchors)
            and native.exact_equal(ema_state["decoder.valid_mask"], mask),
            "MIX endpoint size cache is not the native 1152x2016 regeneration")
    loaded = dict(ema_state)
    for k in keys:
        loaded[k] = own[k].detach().clone()
    require(set(loaded) == set(own), "MIX evaluation state keys differ")
    model.load_state_dict(loaded, strict=True)
    validate_geometry(model)
    model.eval().requires_grad_(False)
    after = model.state_dict()
    non_size = [(k, v) for k, v in ema_state.items() if k not in keys]
    require(all(torch.equal(after[k], v) for k, v in non_size), "MIX learned/EMA/BN tensors not byte-identical")
    source_sha, loaded_sha = tensor_digest(ema_state.items()), tensor_digest(after.items())
    if canvas == "H":
        require(loaded_sha == source_sha, "MIX H readout must load the endpoint bytes unchanged")
    require(not any(getattr(m, "_s15_checkpointed", False) for m in model.modules()), "no activation checkpoint at eval")
    return model, {"evaluation_canvas": canvas, "input_hw": list(CANVAS[canvas]), "source_EMA_sha256": source_sha,
                   "loaded_model_sha256": loaded_sha, "non_size_tensor_sha256": tensor_digest(non_size),
                   "non_size_tensors_byte_identical": True,
                   "substituted_state_keys": keys if canvas != "H" else [],
                   "source_size_cache_native_1152x2016": True,
                   "evaluation_anchor_tokens": int(after["decoder.anchors"].shape[1]),
                   "geometry": geometry_identity(model)}
