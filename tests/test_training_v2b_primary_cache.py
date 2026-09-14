"""Synthetic CPU checks for primary bootstrap cache and literal copy semantics."""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sparse_rtdetr.baseline import primary_evaluator as primary
from sparse_rtdetr.baseline.training_v2b_primary_cache import (
    PrimaryCacheError, _vector_voc_ap, literal_resample_primary, prepare_primary_cache,
)
from sparse_rtdetr.data_protocol.evaluation import (
    Detection, PrimaryEvaluatorInputV2, PrimaryGroundTruth, PrimaryImageV2,
)


@pytest.fixture(scope="module")
def binding():
    return primary.primary_evaluator_contract_binding(Path(__file__).resolve().parents[1])


def gt(name, image, category, box, *, ignored=False, region=False):
    box = tuple(map(float, box))
    return PrimaryGroundTruth(name, image, category, box,
                              (box[2] - box[0]) * (box[3] - box[1]),
                              region, ignored)


def det(image, category, box, score):
    return Detection(image, category, tuple(map(float, box)), float(score))


def fixture_input():
    images = tuple(PrimaryImageV2(x, 64, 64) for x in ("a", "b", "c", "d"))
    ground_truth = (
        gt("a1", "a", 1, (2, 2, 12, 12)),
        gt("a2", "a", 2, (32, 32, 42, 42)),
        gt("b1", "b", 1, (2, 2, 12, 12)),
        gt("bignored", "b", 1, (25, 25, 45, 45), ignored=True),
        gt("c0", "c", 0, (1, 1, 25, 25), region=True),
        gt("c1removed", "c", 1, (4, 4, 12, 12)),
        gt("c2", "c", 2, (40, 40, 52, 52)),
        gt("c11", "c", 11, (28, 28, 38, 38), ignored=True),
    )
    detections = (
        # Equal-score FP/TP interleaving catches incorrect weighted tie compression.
        det("a", 1, (48, 2, 58, 12), .8),
        det("a", 1, (2, 2, 12, 12), .8),
        det("a", 2, (32, 32, 42, 42), .8),
        det("a", 1, (2, 2, 12, 12), .8),
        det("b", 1, (2, 2, 12, 12), .8),
        det("b", 1, (28, 28, 32, 32), .8),
        det("b", 1, (30, 30, 35, 35), .8),
        det("c", 1, (4, 4, 12, 12), .9),
        det("c", 2, (40, 40, 51, 51), .8),
        det("d", 1, (2, 2, 12, 12), .8),
    )
    return PrimaryEvaluatorInputV2(images, detections, ground_truth)


@pytest.mark.parametrize("ids", [
    (1, 2, 3, 4), (1, 1, 2), (2, 1, 3, 1, 3), (2, 2, 4), (4,), (3, 3),
])
def test_cache_matches_literal_dataset_copy_including_stable_ties(binding, ids):
    value = fixture_input()
    original = copy.deepcopy(value)
    mapping = {1: "a", 2: "b", 3: "c", 4: "d"}
    cache = prepare_primary_cache(value, binding, mapping)
    copied = literal_resample_primary(value, mapping, ids)
    reference = primary.evaluate_primary_v1(copied, binding)
    observed = cache.score(ids)
    for key in ("AP", "AP50", "AP75", "AR500"):
        assert observed[key] == pytest.approx(getattr(reference, key), abs=1e-12, rel=0)
    assert value == original
    assert len({x.image_id for x in copied.images}) == len(ids)
    assert len({x.annotation_id for x in copied.ground_truth}) == len(copied.ground_truth)


def test_cache_resample_order_is_canonical_and_independent_of_draw_order(binding):
    value = fixture_input()
    cache = prepare_primary_cache(value, binding, {1: "a", 2: "b", 3: "c", 4: "d"})
    assert cache.score((2, 1, 3, 1)) == cache.score((1, 1, 2, 3))


def test_global_maxdets_precedes_category_filter(binding):
    value = PrimaryEvaluatorInputV2(
        (PrimaryImageV2("a", 64, 64),),
        tuple([det("a", 2, (2, 2, 12, 12), .9)] * 500
              + [det("a", 1, (32, 32, 42, 42), .8)]),
        (gt("target", "a", 1, (32, 32, 42, 42)),))
    cache = prepare_primary_cache(value, binding, {1: "a"})
    assert cache.score((1,))["AP"] == 0.0
    assert primary.evaluate_primary_v1(value, binding).AP == 0.0


def test_spatial_ignore_removal_precedes_global_maxdets(binding):
    value = PrimaryEvaluatorInputV2(
        (PrimaryImageV2("a", 64, 64),),
        tuple([det("a", 2, (4, 4, 12, 12), .9)] * 500
              + [det("a", 1, (40, 40, 50, 50), .8)]),
        (gt("region", "a", 0, (1, 1, 25, 25), region=True),
         gt("target", "a", 1, (40, 40, 50, 50)),))
    cache = prepare_primary_cache(value, binding, {1: "a"})
    assert cache.filtered_counts == {"ground_truth": 1, "detections": 1}
    assert cache.score((1,))["AP"] == 100.0
    assert primary.evaluate_primary_v1(value, binding).AP == 100.0


def test_ignored_gt_remains_in_denominator_and_absorbs_multiple_detections(binding):
    value = PrimaryEvaluatorInputV2(
        (PrimaryImageV2("a", 64, 64),),
        (det("a", 1, (2, 2, 12, 12), .9),
         det("a", 1, (28, 28, 32, 32), .8),
         det("a", 1, (30, 30, 34, 34), .7)),
        (gt("standard", "a", 1, (2, 2, 12, 12)),
         gt("ignored", "a", 1, (25, 25, 45, 45), ignored=True)))
    cache = prepare_primary_cache(value, binding, {1: "a"})
    observed = cache.score((1, 1))
    assert observed["AP"] == 50.0
    assert observed["AR500"] == 50.0
    assert primary.evaluate_primary_v1(literal_resample_primary(value, {1: "a"}, (1, 1)),
                                       binding).AP == observed["AP"]


def test_vectorized_integral_matches_frozen_loop():
    rng = np.random.default_rng(51)
    for n in (0, 1, 17, 99):
        tp = np.cumsum(rng.integers(0, 2, size=n)).astype(float)
        fp = np.arange(1, n + 1) - tp
        rc = tp / max(1, n)
        pr = tp / np.maximum(1, tp + fp)
        assert _vector_voc_ap(rc, pr) == primary._voc_ap(rc, pr)


@pytest.mark.parametrize("ids", [(), (True,), (0,), (99,), (1.0,)])
def test_unknown_or_ambiguous_resampling_ids_are_rejected(binding, ids):
    cache = prepare_primary_cache(fixture_input(), binding, {1: "a", 2: "b", 3: "c", 4: "d"})
    with pytest.raises(PrimaryCacheError):
        cache.score(ids)


def test_nonbijective_identity_and_mutated_binding_fail(binding):
    with pytest.raises(PrimaryCacheError):
        prepare_primary_cache(fixture_input(), binding, {1: "a", 2: "a"})
    broken = copy.deepcopy(binding)
    broken["config_raw_sha256"] = "0" * 64
    with pytest.raises(primary.PrimaryEvaluatorContractError):
        prepare_primary_cache(fixture_input(), broken, {1: "a", 2: "b", 3: "c", 4: "d"})


def test_evidence_counts_exclude_others_and_spatially_removed_gt(binding):
    value = fixture_input()
    cache = prepare_primary_cache(value, binding, {1: "a", 2: "b", 3: "c", 4: "d"})
    reference = primary.evaluate_primary_v1(value, binding)
    assert cache.filtered_counts["ground_truth"] == reference.evaluated_ground_truth_count
    assert cache.filtered_counts["detections"] == reference.evaluated_detection_count
    assert cache.filtered_counts["ground_truth"] == 5
