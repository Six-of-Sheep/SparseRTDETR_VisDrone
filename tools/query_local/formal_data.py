"""Seed0 full-data loader and committed update cursor for R25 formal training."""
from __future__ import annotations

import hashlib
import random

import numpy as np
import torch

from sparse_rtdetr.baseline.training_v2b import logical_batch_indices
from sparse_rtdetr.baseline.training_v2b_data import _sample_seed
from sparse_rtdetr.localization.material_paths import MANIFESTS
from sparse_rtdetr.localization.runio import digest
from sparse_rtdetr.protocol_r18.data import dataset, input_record, training_loader

SEED = 0
MEMBERS = 4869
WINDOW_SIZE = 16
WINDOWS_PER_EPOCH = 304
FIRST_EPOCH = 61
LAST_EPOCH = 120
PARENT_UPDATE = (FIRST_EPOCH - 1) * WINDOWS_PER_EPOCH
END_UPDATE = LAST_EPOCH * WINDOWS_PER_EPOCH


def train_dataset(train_core_binding):
    """Construct the same seed0 role-bound dataset used by the parent protocol."""
    return dataset({
        "data": {"train_core": train_core_binding},
        "seed": SEED,
        "config": {"seed": SEED},
    })


def cursor(completed_updates):
    """Position of the next window; prefetch never commits progress."""
    if type(completed_updates) is not int or not PARENT_UPDATE <= completed_updates <= END_UPDATE:
        raise ValueError("R25 formal update cursor is outside e060–e120")
    completed_epochs, next_window = divmod(completed_updates, WINDOWS_PER_EPOCH)
    return {
        "schema": "r25-full-train-core-cursor-v1",
        "completed_updates": completed_updates,
        "completed_epochs": completed_epochs,
        "next_epoch": completed_epochs + 1,
        "next_window": next_window,
        "end": completed_updates == END_UPDATE,
        "seed": SEED,
        "members": MEMBERS,
        "train_core_manifest_sha256": MANIFESTS["train_core"][0],
        "windows_per_epoch": WINDOWS_PER_EPOCH,
        "logical_batch_size": WINDOW_SIZE,
        "workers": 2,
        "multiprocessing_context": "spawn",
        "prefetch_factor": 2,
        "persistent_workers": False,
        "worker_generator_seed": SEED,
        "worker_rng": "seed_worker: torch.initial_seed() mod 2**32 for Python and NumPy",
        "sampler": "v2b-order:seed:epoch; acknowledge complete logical16 only",
        "augmentation": "v2b-augment:seed:epoch:position:stable_id",
        "first_epoch_without_strong_augmentation": 118,
    }


def windows(data, completed_updates, *, limit=None):
    """Yield the rest of one epoch (or a bounded acceptance prefix).

    Call again with the engine's advanced update clock after an epoch ends.
    The yielded receipt is independent of worker PID and process timing.
    """
    state = cursor(completed_updates)
    if (data.role, data.seed, len(data), data.manifest_sha256) != (
        "train_core", SEED, MEMBERS, state["train_core_manifest_sha256"]
    ):
        raise ValueError("R25 formal loader requires bound seed0 train_core")
    if state["end"]:
        if limit is not None:
            raise ValueError("no R25 acceptance window after e120")
        return
    epoch = state["next_epoch"]
    first = state["next_window"]
    loader = training_loader(data, epoch=epoch, completed_windows=first, workers=2, window_limit=limit)
    expected = logical_batch_indices(MEMBERS, seed=SEED, epoch=epoch, completed_batches=first)
    seen = 0
    for images, targets, traces in loader:
        window = first + seen
        indices = next(expected)
        ids = [data.records[index]["stable_image_id"] for index in indices]
        positions = list(range(window * WINDOW_SIZE, (window + 1) * WINDOW_SIZE))
        seeds = [_sample_seed(SEED, epoch, pos, image_id) for pos, image_id in zip(positions, ids)]
        if ([trace["image_id"] for trace in traces] != ids
            or [target["image_id"] for target in targets] != ids
            or [trace["position"] for trace in traces] != positions
            or [trace["sample_seed"] for trace in traces] != seeds
            or any(trace["epoch"] != epoch or trace["augmentation_strong"] != (epoch < 118)
                   for trace in traces)):
            raise ValueError("R25 actual window order, identity or augmentation differs from the native schedule")
        audit = input_record(images, targets, traces)
        seen += 1
        yield images, targets, {
            "update": completed_updates + seen,
            "epoch": epoch,
            "window": window,
            "image_ids_sha256": digest(audit["ids"]),
            "gt_ids_sha256": digest(audit["GT_ids"]),
            "augmentation_seeds_sha256": digest(seeds),
            "augmentation_trace_sha256": audit["metadata_sha256"],
            "augmented_samples_sha256": digest(audit["augmented_sha256"]),
            "input_tensor_sha256": audit["input_tensor_sha256"],
            "target_sha256": audit["target_sha256"],
            "augmentation_strong": epoch < 118,
        }
    if seen != len(loader):
        raise ValueError("R25 loader ended before its declared window count")


def check_boundaries(data):
    """Exercise real workers at fixed boundaries without model or GPU calls."""
    if torch.cuda.is_initialized():
        raise ValueError("run R25 data boundary checks before any CUDA initialization")
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()

    def read(completed_updates, limit=1):
        return [audit for _, _, audit in windows(data, completed_updates, limit=limit)]

    epoch61 = read(PARENT_UPDATE, 2)
    resumed = read(PARENT_UPDATE + 1)
    if resumed != epoch61[1:]:
        raise ValueError("native e061 resumed worker input differs from its uninterrupted window")
    boundaries = {
        "end_e061": read(61 * WINDOWS_PER_EPOCH - 1)[0],
        "start_e062": read(61 * WINDOWS_PER_EPOCH)[0],
        "end_e117": read(117 * WINDOWS_PER_EPOCH - 1)[0],
        "start_e118": read(117 * WINDOWS_PER_EPOCH)[0],
        "end_e120": read(END_UPDATE - 1)[0],
    }
    actual = [(audit["update"], audit["epoch"], audit["window"], audit["augmentation_strong"])
              for audit in [*epoch61, *boundaries.values()]]
    expected = [(18241, 61, 0, True), (18242, 61, 1, True),
                (18544, 61, 303, True), (18545, 62, 0, True),
                (35568, 117, 303, True), (35569, 118, 0, False),
                (36480, 120, 303, False)]
    if actual != expected or list(windows(data, END_UPDATE)):
        raise ValueError("formal epoch, augmentation or stopping boundary differs")

    # Independently expand each audited epoch's native hash-seeded permutation.
    plans = {}
    for epoch in sorted({row[1] for row in expected}):
        seed = int.from_bytes(hashlib.sha256(f"v2b-order:0:{epoch}".encode()).digest()[:8], "big")
        generator = torch.Generator().manual_seed(seed & ((1 << 63) - 1))
        order = torch.randperm(MEMBERS, generator=generator).tolist()
        native = [index for batch in logical_batch_indices(MEMBERS, seed=SEED, epoch=epoch) for index in batch]
        if native != order[:WINDOW_SIZE * WINDOWS_PER_EPOCH] or len(order) - len(native) != 5:
            raise ValueError("native order, cardinality or dropped tail differs")
        plans[epoch] = {
            "input_order_sha256": digest([data.records[i]["stable_image_id"] for i in native]),
            "dropped_ids": [data.records[i]["stable_image_id"] for i in order[len(native):]],
        }
    final_numpy_rng = np.random.get_state()
    if (random.getstate() != python_rng or not torch.equal(torch_rng, torch.get_rng_state())
        or numpy_rng[0] != final_numpy_rng[0] or not np.array_equal(numpy_rng[1], final_numpy_rng[1])
        or numpy_rng[2:] != final_numpy_rng[2:] or torch.cuda.is_initialized()):
        raise ValueError("native DataLoader or boundary check mutated parent RNG or initialized CUDA")
    return {
        "status": "PASS",
        "start_cursor": cursor(PARENT_UPDATE),
        "terminal_cursor": cursor(END_UPDATE),
        "epoch61": epoch61,
        "restart_input_exact": True,
        "boundaries": boundaries,
        "independent_order_checks": plans,
        "parent_rng_unchanged": True,
        "cuda_initialized": False,
    }
