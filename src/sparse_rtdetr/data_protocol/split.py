"""Deterministic sequence/content atomic grouping and split planning."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from .schema import ProtocolContractError, canonical_json_bytes


@dataclass(frozen=True)
class ImageIdentity:
    relative_path: str
    sequence_key: str
    image_sha256: str
    valid_target_count: int = 0
    class_counts: tuple[int, ...] = (0,) * 10
    small_target_count: int = 0
    identity_closed: bool = True
    abnormal: bool = False

    def __post_init__(self) -> None:
        if not self.relative_path or not self.sequence_key:
            raise ProtocolContractError("image identity requires path and sequence")
        if len(self.image_sha256) != 64:
            raise ProtocolContractError("image identity requires a SHA-256")
        if len(self.class_counts) != 10:
            raise ProtocolContractError("class_counts must contain ten classes")
        if min(self.class_counts) < 0 or self.valid_target_count < 0 or self.small_target_count < 0:
            raise ProtocolContractError("image statistics must be non-negative")


@dataclass(frozen=True)
class AtomicGroup:
    group_id: str
    sequence_keys: tuple[str, ...]
    image_paths: tuple[str, ...]
    image_sha256s: tuple[str, ...]
    image_count: int
    valid_target_count: int
    class_counts: tuple[int, ...]
    small_target_count: int
    identity_closed: bool
    abnormal: bool


@dataclass(frozen=True)
class SplitPlan:
    train_core_group_ids: tuple[str, ...]
    confirmatory_group_ids: tuple[str, ...]
    ordered_candidate_group_ids: tuple[str, ...]
    target_image_count: int
    selected_image_count: int
    selected_group_list_sha256: str
    seed: int
    salt: str
    selection_allowed: bool
    metrics_access_allowed: bool
    single_final_access_only: bool
    checks: tuple[tuple[str, bool], ...]


def _union_find(size: int):
    parents = list(range(size))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    return parents, find


def build_atomic_groups(
    images: tuple[ImageIdentity, ...] | list[ImageIdentity],
    sequence_metadata: dict[str, str] | None = None,
) -> tuple[AtomicGroup, ...]:
    """Merge sequence groups and duplicate-content groups as components."""

    ordered = tuple(sorted(images, key=lambda item: item.relative_path))
    if len({item.relative_path for item in ordered}) != len(ordered):
        raise ProtocolContractError("image paths must be unique")
    if sequence_metadata is not None:
        if set(sequence_metadata) != {item.relative_path for item in ordered}:
            raise ProtocolContractError("sequence metadata does not cover images exactly")
        for item in ordered:
            if sequence_metadata[item.relative_path] != item.sequence_key:
                raise ProtocolContractError("sequence metadata disagrees with filename identity")
    parents, find = _union_find(len(ordered))

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parents[root_right] = root_left

    by_sequence: dict[str, int] = {}
    by_hash: dict[str, int] = {}
    for index, item in enumerate(ordered):
        if item.sequence_key in by_sequence:
            union(index, by_sequence[item.sequence_key])
        else:
            by_sequence[item.sequence_key] = index
        if item.image_sha256 in by_hash:
            union(index, by_hash[item.image_sha256])
        else:
            by_hash[item.image_sha256] = index

    components: dict[int, list[ImageIdentity]] = {}
    for index, item in enumerate(ordered):
        components.setdefault(find(index), []).append(item)
    groups: list[AtomicGroup] = []
    for members in components.values():
        paths = tuple(sorted(item.relative_path for item in members))
        sequences = tuple(sorted({item.sequence_key for item in members}))
        hashes = tuple(sorted({item.image_sha256 for item in members}))
        identity = {"image_paths": paths, "image_sha256s": hashes, "sequence_keys": sequences}
        group_id = "group-" + hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
        groups.append(AtomicGroup(
            group_id=group_id,
            sequence_keys=sequences,
            image_paths=paths,
            image_sha256s=hashes,
            image_count=len(members),
            valid_target_count=sum(item.valid_target_count for item in members),
            class_counts=tuple(sum(item.class_counts[i] for item in members) for i in range(10)),
            small_target_count=sum(item.small_target_count for item in members),
            identity_closed=all(item.identity_closed for item in members),
            abnormal=any(item.abnormal for item in members),
        ))
    return tuple(sorted(groups, key=lambda group: group.group_id))


def _group_list_sha(group_ids: tuple[str, ...]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(group_ids))).hexdigest()


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _stats(groups: tuple[AtomicGroup, ...]) -> tuple[int, tuple[int, ...], int]:
    return (
        sum(group.image_count for group in groups),
        tuple(sum(group.class_counts[i] for group in groups) for i in range(10)),
        sum(group.small_target_count for group in groups),
    )


def _validate_distributions(
    train_core: tuple[AtomicGroup, ...],
    confirmatory: tuple[AtomicGroup, ...],
    tolerance: float,
) -> tuple[tuple[str, bool], ...]:
    _, train_classes, train_small = _stats(train_core)
    _, confirm_classes, confirm_small = _stats(confirmatory)
    train_total = sum(train_classes)
    confirm_total = sum(confirm_classes)
    if train_total <= 0 or confirm_total <= 0:
        raise ProtocolContractError("distribution checks require non-empty targets")
    checks: list[tuple[str, bool]] = []
    for index in range(10):
        train_ratio = train_classes[index] / train_total
        confirm_ratio = confirm_classes[index] / confirm_total
        passed = abs(train_ratio - confirm_ratio) <= tolerance
        checks.append((f"class_{index}_ratio_within_tolerance", passed))
        if not passed:
            raise ProtocolContractError(f"class {index} distribution exceeds tolerance")
    passed = abs(train_small / train_total - confirm_small / confirm_total) <= tolerance
    checks.append(("coco_small_ratio_within_tolerance", passed))
    if not passed:
        raise ProtocolContractError("COCO-small distribution exceeds tolerance")
    return tuple(checks)


def plan_confirmatory_split(
    groups: tuple[AtomicGroup, ...] | list[AtomicGroup],
    *,
    train_image_count: int = 6471,
    target_ratio: float = 0.10,
    development_sequence_keys: frozenset[str] | set[str] = frozenset(),
    seed: int = 20260808,
    salt: str = "P3-confirmatory-v1",
    tolerance: float = 0.05,
) -> SplitPlan:
    """Select one deterministic prefix and fail closed on contract checks."""

    all_groups = tuple(sorted(groups, key=lambda group: group.group_id))
    if not all_groups or train_image_count <= 0:
        raise ProtocolContractError("split planning requires groups and a positive train size")
    candidate = tuple(
        group for group in all_groups
        if group.identity_closed
        and not group.abnormal
        and not (set(group.sequence_keys) & set(development_sequence_keys))
    )
    if not candidate:
        raise ProtocolContractError("no eligible confirmatory groups")
    ordered = tuple(sorted(
        candidate,
        key=lambda group: hashlib.sha256(f"{salt}\0{seed}\0{group.group_id}".encode("utf-8")).hexdigest(),
    ))
    target = _round_half_up(train_image_count * target_ratio)
    choices: list[tuple[tuple[object, ...], int, tuple[str, ...]]] = []
    for length in range(len(ordered) + 1):
        prefix = tuple(group.group_id for group in ordered[:length])
        count = sum(group.image_count for group in ordered[:length])
        key = (abs(count - target), int(count > target), count, length, _group_list_sha(prefix))
        choices.append((key, count, prefix))
    _, selected_count, selected_ids = min(choices, key=lambda item: item[0])
    selected = tuple(group for group in all_groups if group.group_id in selected_ids)
    train_core = tuple(group for group in all_groups if group.group_id not in selected_ids)
    selected_hashes = {image_sha for group in selected for image_sha in group.image_sha256s}
    core_hashes = {image_sha for group in train_core for image_sha in group.image_sha256s}
    if selected_hashes & core_hashes:
        raise ProtocolContractError("duplicate image content crosses split boundary")
    if any(set(group.sequence_keys) & set(development_sequence_keys) for group in selected):
        raise ProtocolContractError("confirmatory sequence overlaps development")
    checks: list[tuple[str, bool]] = [
        ("group_disjoint", not ({group.group_id for group in selected} & {group.group_id for group in train_core})),
        ("image_content_disjoint", not (selected_hashes & core_hashes)),
        ("confirmatory_not_development_sequence", not any(
            set(group.sequence_keys) & set(development_sequence_keys) for group in selected
        )),
    ]
    checks.extend(_validate_distributions(train_core, selected, tolerance))
    return SplitPlan(
        train_core_group_ids=tuple(sorted(group.group_id for group in train_core)),
        confirmatory_group_ids=selected_ids,
        ordered_candidate_group_ids=tuple(group.group_id for group in ordered),
        target_image_count=target,
        selected_image_count=selected_count,
        selected_group_list_sha256=_group_list_sha(selected_ids),
        seed=seed,
        salt=salt,
        selection_allowed=False,
        metrics_access_allowed=False,
        single_final_access_only=True,
        checks=tuple(checks),
    )
