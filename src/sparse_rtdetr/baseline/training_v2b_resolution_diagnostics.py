"""Frozen CPU-only paired resolution diagnostics on saved development outputs.

No file discovery, image reads, model forward, CUDA, training, or protocol writes.
Primary scoring is injected by the uniform GT adapter; COCO small-object metrics
retain the converted COCO contract. AP is re-accumulated, never image-averaged.
"""
from __future__ import annotations

import builtins
import contextlib
import copy
import hashlib
import importlib.machinery
import importlib.metadata
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import threading
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Callable, Mapping, Sequence

import numpy as np


CELLS = ("t640_e640", "t640_e896", "t896_e640", "t896_e896")
CONTRASTS = {
    "diagonal_total": (-1., 0., 0., 1.),
    "inference_at_train640": (-1., 1., 0., 0.),
    "inference_at_train896": (0., 0., -1., 1.),
    "training_at_eval640": (-1., 0., 1., 0.),
    "training_at_eval896": (0., -1., 0., 1.),
    "training_mean": (-.5, -.5, .5, .5),
    "inference_mean": (-.5, .5, -.5, .5),
    "interaction": (1., -1., -1., 1.),
}
IOU_THRESHOLDS = np.linspace(.5, .95, 10)
RECALL_THRESHOLDS = np.linspace(0., 1., 101)
AREAS = ("all", "small", "medium", "large")
GT_REASONS = ("true_positive", "maxdets_limited", "matching_competition",
              "classification", "localization", "classification_and_localization",
              "no_saved_candidate")
PRED_REASONS = ("true_positive", "evaluation_ignored", "maxdets_excluded",
                "duplicate", "classification", "localization",
                "classification_and_localization", "background")


class DiagnosticError(ValueError):
    """A frozen diagnostic input, identity, or method is inconsistent."""


@dataclass(frozen=True)
class DiagnosticPolicy:
    cluster_replicates: int = 2000
    image_replicates: int = 200
    rng_seed: int = 20260915
    expected_images: int = 548
    expected_clusters: int = 75
    confidence: float = .95
    coco_max_dets: int = 100
    error_iou_thresholds: tuple = (.5, .75)
    background_iou: float = .1
    routing_grid: int = 4
    routing_tiles: int = 4
    coarse_input: int = 640
    fine_input: int = 896
    halo_pixels: int = 16
    routing_min_probability: float = .05
    routing_small_input_area: float = 1024.
    fixture_only: bool = False


FROZEN_POLICY = DiagnosticPolicy()


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
                      separators=(",", ":"), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def frozen_method():
    return {
        "schema_version": 1, "policy": asdict(FROZEN_POLICY),
        "cells": list(CELLS), "contrasts": {k: list(v) for k, v in CONTRASTS.items()},
        "cluster": "sequence_key_connected_by_identical_image_sha256",
        "resampling": "clusters_with_replacement_keep_all_member_images",
        "tie_order": "original_integer_image_id,copy_index,original_prediction_row",
        "ci": "paired_percentile_quantile_0.025_0.975_numpy_linear",
        "primary_score": "injected_uniform_development_GT_adapter",
        "seed_gate": "primary_diagonal_AP>0 and coco_diagonal_AP_small>0 and "
                     "cluster_AP_small_CI_lower>0 and small_unmatched_at_0.5_decreases",
        "seed_gate_not_training_seed_inference": True,
        "image_sensitivity_is_not_a_gate": True,
        "oracle_is_not_deployable": True,
        "raw_query_observation": {
            "iou_thresholds": [.1, .5, .75],
            "queries": 300, "classes": 10, "saved_flat_pairs": 300,
            "required_arrays": ["pred_logits", "pred_boxes", "pred_sigmoid_scores", "topk_indices"],
            "probabilities": "saved_runtime_float32_sigmoid",
            "geometry": "eager_float32_cxcywh_to_xyxy_scale_original_then_xywh",
            "best_query": "maximum_IoU_then_lowest_query_id_without_class_score",
            "score_rank": "full_3000_pairs_canonical_flat_index_ties_and_full_tie_interval",
            "retention": "actual_runtime_topk_indices_authoritative",
            "coverage": "many_to_one_candidate_upper_bound_not_AP_or_causal_repair",
            "absence": "saved_geometry_exists_or_raw_geometry_absent_or_geometric_queries_lost_to_flat_topk",
        },
        "spatial_ignore_coverage_not_implied_by_empty_sidecar": True,
        "preregistration_scope": "before_new_resampling_and_cross_eval_analysis;"
                                "historical_diagonal_scores_already_observed",
    }


def _policy(policy):
    if type(policy) is not DiagnosticPolicy:
        raise DiagnosticError("policy must be an immutable DiagnosticPolicy")
    if not policy.fixture_only and policy != FROZEN_POLICY:
        raise DiagnosticError("production diagnostic policy drift")
    if (policy.cluster_replicates < 2 or policy.image_replicates < 2
            or policy.expected_images < 1 or policy.expected_clusters < 1
            or policy.confidence != .95 or policy.routing_grid < 1
            or not 0 < policy.routing_tiles <= policy.routing_grid ** 2):
        raise DiagnosticError("invalid diagnostic policy")
    return policy


def _validate_coco(coco, predictions):
    if not isinstance(coco, dict) or any(k not in coco for k in ("images", "annotations", "categories")):
        raise DiagnosticError("COCO images, annotations and categories are required")
    images = {}
    for row in coco["images"]:
        key = row.get("id")
        if (type(key) is not int or key < 0 or key in images
                or type(row.get("width")) is not int or row["width"] < 1
                or type(row.get("height")) is not int or row["height"] < 1
                or row.get("split", "development") not in ("val", "development")):
            raise DiagnosticError("invalid or non-development image identity")
        images[key] = row
    if not images:
        raise DiagnosticError("at least one development image is required")
    categories = {x["id"] for x in coco["categories"]}
    if len(categories) != len(coco["categories"]) or not categories:
        raise DiagnosticError("category identity is empty or duplicated")
    if any(type(x) is not int or x not in range(1, 11) for x in categories):
        raise DiagnosticError("COCO categories must be in 1..10")
    ids = set()
    for row in coco["annotations"]:
        if type(row.get("id")) is not int or row["id"] <= 0 or row["id"] in ids:
            raise DiagnosticError("GT IDs must be unique positive builtin integers")
        ids.add(row["id"])
        _box_row(row, images, categories)
        if (not math.isfinite(float(row["area"])) or row["area"] <= 0
                or row.get("iscrowd", 0) not in (0, 1)):
            raise DiagnosticError("invalid GT area/crowd flag")
    for row in predictions:
        _box_row(row, images, categories)
        if not math.isfinite(float(row["score"])) or not 0 <= row["score"] <= 1:
            raise DiagnosticError("prediction score must be finite in [0,1]")
    return tuple(sorted(images))


def _box_row(row, images, categories):
    if (type(row.get("image_id")) is not int or type(row.get("category_id")) is not int
            or row["image_id"] not in images or row["category_id"] not in categories):
        raise DiagnosticError("row image/category is not declared")
    box = row.get("bbox")
    if (not isinstance(box, (list, tuple)) or len(box) != 4
            or not all(math.isfinite(float(x)) for x in box)
            or box[2] <= 0 or box[3] <= 0):
        raise DiagnosticError("box must be finite original-pixel xywh with positive extents")


def _resample_ids(image_ids, allowed):
    ids = tuple(image_ids)
    if not ids or any(type(x) is not int or x not in allowed for x in ids):
        raise DiagnosticError("resampling requires known nonempty builtin integer image IDs")
    return tuple(sorted(ids))


def literal_resample_coco(coco, predictions, image_ids):
    """Clone IDs before literal evaluation; repeated imgIds alone are deduplicated."""
    allowed = _validate_coco(coco, predictions)
    ids = _resample_ids(image_ids, allowed)
    images = {x["id"]: x for x in coco["images"]}
    gt_by_image, pred_by_image = defaultdict(list), defaultdict(list)
    for row in coco["annotations"]:
        gt_by_image[row["image_id"]].append(row)
    for row in predictions:
        pred_by_image[row["image_id"]].append(row)
    result = {"images": [], "annotations": [],
              "categories": copy.deepcopy(coco["categories"])}
    result_predictions, ann_id = [], 0
    for new_id, old_id in enumerate(ids, 1):
        image = copy.deepcopy(images[old_id])
        image["id"] = new_id
        if "stable_image_id" in image:
            image["stable_image_id"] = f"copy:{new_id}:{image['stable_image_id']}"
        result["images"].append(image)
        for row in gt_by_image[old_id]:
            row = copy.deepcopy(row)
            ann_id += 1
            row.update(id=ann_id, image_id=new_id)
            if "stable_annotation_id" in row:
                row["stable_annotation_id"] = f"copy:{new_id}:{row['stable_annotation_id']}"
            result["annotations"].append(row)
        for row in pred_by_image[old_id]:
            row = copy.deepcopy(row)
            row.pop("id", None)
            row["image_id"] = new_id
            result_predictions.append(row)
    return result, result_predictions



_REFERENCE_COCO_BACKEND = None
_REFERENCE_COCO_IMPORT_LOCK = threading.Lock()


def _reference_coco_backend():
    """Load the installed reference engine without changing vendor aliases.

    The vendor calls faster_coco_eval.init_as_pycocotools(), replacing the public
    package names. Faster's evaluate() does not expose Python evalImgs records.
    Load the original distribution under a private package name instead. Its
    mask.py uses an absolute pycocotools._mask import, so only these private
    modules receive an import hook; process-wide imports remain untouched.
    """
    global _REFERENCE_COCO_BACKEND
    with _REFERENCE_COCO_IMPORT_LOCK:
        if _REFERENCE_COCO_BACKEND is not None:
            return _REFERENCE_COCO_BACKEND
        try:
            distribution = importlib.metadata.distribution("pycocotools")
        except importlib.metadata.PackageNotFoundError as error:
            raise DiagnosticError("reference pycocotools distribution is required") from error
        root = Path(distribution.locate_file("pycocotools")).resolve()
        namespace = "_sparse_rtdetr_reference_pycocotools"
        if any(name == namespace or name.startswith(namespace + ".") for name in tuple(sys.modules)):
            raise DiagnosticError("reference COCO private import namespace is occupied")
        installed = []

        def private_import(name, globals=None, locals=None, fromlist=(), level=0):
            if level == 0 and (name == "pycocotools" or name.startswith("pycocotools.")):
                name = namespace + name[len("pycocotools"):]
            return builtins.__import__(name, globals, locals, fromlist, level)

        def load_private(spec):
            if spec is None or spec.loader is None:
                raise DiagnosticError("reference COCO distribution module is unavailable")
            module = importlib.util.module_from_spec(spec)
            if isinstance(spec.loader, importlib.machinery.SourceFileLoader):
                module.__dict__["__builtins__"] = dict(vars(builtins), __import__=private_import)
            sys.modules[spec.name] = module
            installed.append(spec.name)
            spec.loader.exec_module(module)
            return module

        try:
            package = load_private(importlib.util.spec_from_file_location(
                namespace, root / "__init__.py", submodule_search_locations=[str(root)]))
            native_spec = importlib.machinery.PathFinder.find_spec(
                namespace + "._mask", [str(root)])
            if native_spec is None or not native_spec.origin:
                raise DiagnosticError("reference COCO native mask extension is unavailable")
            # CPython can reuse a previously loaded extension object for the
            # same file. Reinitializing its spec would mutate that public
            # object's metadata, so reuse an authenticated existing native
            # module without calling module_from_spec on it again.
            native = sys.modules.get("pycocotools._mask")
            if (native is not None and getattr(native, "__file__", None)
                    and Path(native.__file__).resolve() == Path(native_spec.origin).resolve()):
                sys.modules[native_spec.name] = native
                installed.append(native_spec.name)
            else:
                # Cython also registers its compiled canonical module name on
                # first initialization, even for a private import spec. Remove
                # only that incidental registration, or restore its prior
                # binding, so the public package map has the same final state.
                public_native_present = "pycocotools._mask" in sys.modules
                public_native = sys.modules.get("pycocotools._mask")
                try:
                    native = load_private(native_spec)
                finally:
                    if public_native_present:
                        sys.modules["pycocotools._mask"] = public_native
                    else:
                        sys.modules.pop("pycocotools._mask", None)
            package._mask = native
            modules = {"__init__.py": package, Path(native.__file__).name: native}
            for name in ("mask", "coco", "cocoeval"):
                module = load_private(importlib.util.spec_from_file_location(
                    namespace + "." + name, root / (name + ".py")))
                setattr(package, name, module)
                modules[name + ".py"] = module
            identities = {}
            for name, module in modules.items():
                path = Path(module.__file__).resolve()
                if path.parent != root:
                    raise DiagnosticError("reference COCO module escaped its distribution")
                identities[name] = {
                    "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            identity = {
                "distribution": "pycocotools", "version": distribution.version,
                "loader": "isolated_distribution_package_without_public_alias_changes",
                "namespace": namespace, "files": identities,
            }
            _REFERENCE_COCO_BACKEND = (package.coco.COCO, package.cocoeval.COCOeval, identity)
        except BaseException:
            for name in reversed(installed):
                sys.modules.pop(name, None)
            raise
        return _REFERENCE_COCO_BACKEND


class CocoMetricCache:
    """Reference COCO per-image matches and exact nonlinear AP re-accumulation."""

    def __init__(self, coco, predictions, max_dets=100):
        self.image_ids = _validate_coco(coco, predictions)
        if type(max_dets) is not int or not 1 <= max_dets <= 300:
            raise DiagnosticError("cached COCO maxDets must be a builtin integer in 1..300")
        self.max_dets = max_dets
        self.coco = copy.deepcopy(coco)
        self.predictions = copy.deepcopy(list(predictions))
        self.gt_sha256 = _digest(coco)
        self.prediction_sha256 = _digest(predictions)
        self.categories = tuple(sorted(x["id"] for x in coco["categories"]))
        # Use the reference distribution even after the vendor installs its
        # faster-coco-eval aliases. Never change the vendor's global imports.
        COCO, COCOeval, backend_identity = _reference_coco_backend()
        self.backend_identity = copy.deepcopy(backend_identity)
        with contextlib.redirect_stdout(io.StringIO()):
            gt = COCO()
            gt.dataset = copy.deepcopy(coco)
            gt.createIndex()
            dt = COCO()
            rows = []
            for index, original in enumerate(predictions, 1):
                row = copy.deepcopy(original)
                row.update(id=index, area=float(row["bbox"][2] * row["bbox"][3]), iscrowd=0)
                rows.append(row)
            dt.dataset = {"images": copy.deepcopy(coco["images"]),
                          "categories": copy.deepcopy(coco["categories"]),
                          "annotations": rows}
            dt.createIndex()
            evaluator = COCOeval(gt, dt, "bbox")
            evaluator.params.maxDets = sorted(set((1, min(10, max_dets), max_dets)))
            evaluator.evaluate()
        expected_records = len(self.categories) * len(AREAS) * len(self.image_ids)
        if (not isinstance(evaluator.evalImgs, list)
                or len(evaluator.evalImgs) != expected_records
                or any(record is not None and not isinstance(record, dict)
                       for record in evaluator.evalImgs)):
            raise DiagnosticError("reference COCO matcher did not expose per-image records")
        self._records = {}
        for category_index, category in enumerate(self.categories):
            for area_index, area in enumerate(AREAS):
                for image_index, image_id in enumerate(self.image_ids):
                    index = (category_index * len(AREAS) + area_index) * len(self.image_ids) + image_index
                    record = evaluator.evalImgs[index]
                    self._records[image_id, category, area] = record
        self.axes = {"iouThrs": list(map(float, IOU_THRESHOLDS)),
                     "recThrs": list(map(float, RECALL_THRESHOLDS)),
                     "areaRng": copy.deepcopy(evaluator.params.areaRng),
                     "areaRngLbl": list(AREAS), "maxDets": max_dets,
                     "coordinate_space": "original_image_pixels"}
        self._image_table = {x["id"]: x for x in coco["images"]}
        self._gt_table = {x["id"]: x for x in coco["annotations"]}

    def score(self, image_ids: Sequence[int], *, max_dets=None):
        ids = _resample_ids(image_ids, self.image_ids)
        maximum = self.max_dets if max_dets is None else max_dets
        if type(maximum) is not int or not 1 <= maximum <= self.max_dets:
            raise DiagnosticError("requested maxDets exceeds the cached prefix")
        area_values = {}
        for area in AREAS:
            precision = np.full((10, 101, len(self.categories)), -1., dtype=np.float64)
            recall = np.full((10, len(self.categories)), -1., dtype=np.float64)
            for category_index, category in enumerate(self.categories):
                records = [self._records[i, category, area] for i in ids]
                records = [x for x in records if x is not None]
                if not records:
                    continue
                count_gt = sum(int(np.count_nonzero(~np.asarray(x["gtIgnore"], dtype=bool)))
                               for x in records)
                if count_gt == 0:
                    continue
                scores = np.concatenate([np.asarray(x["dtScores"][:maximum]) for x in records])
                order = np.argsort(-scores, kind="stable")
                matched = np.concatenate([x["dtMatches"][:, :maximum] for x in records], axis=1)[:, order]
                ignored = np.concatenate([x["dtIgnore"][:, :maximum] for x in records], axis=1)[:, order]
                tp = np.cumsum((matched > 0) & ~ignored, axis=1, dtype=np.float64)
                fp = np.cumsum((matched == 0) & ~ignored, axis=1, dtype=np.float64)
                for threshold in range(10):
                    rc = tp[threshold] / count_gt
                    pr = tp[threshold] / (tp[threshold] + fp[threshold] + np.spacing(1))
                    recall[threshold, category_index] = rc[-1] if rc.size else 0.
                    q = np.zeros(101, dtype=np.float64)
                    if pr.size:
                        pr = np.maximum.accumulate(pr[::-1])[::-1]
                        positions = np.searchsorted(rc, RECALL_THRESHOLDS, side="left")
                        valid = positions < len(pr)
                        q[valid] = pr[positions[valid]]
                    precision[threshold, :, category_index] = q
            area_values[area] = precision, recall

        def mean(array):
            valid = array[array > -1.]
            return float(np.mean(valid) * 100.) if valid.size else None

        all_precision, all_recall = area_values["all"]
        result = {"AP": mean(all_precision), "AP50": mean(all_precision[0]),
                  "AP75": mean(all_precision[5]), f"AR{maximum}": mean(all_recall)}
        for area in AREAS[1:]:
            result[f"AP_{area}"] = mean(area_values[area][0])
            result[f"AR_{area}"] = mean(area_values[area][1])
        return result

    def matched_ids(self, image_id, *, iou=.5, area="small", max_dets=100):
        threshold = int(np.argmin(abs(IOU_THRESHOLDS - iou)))
        if abs(float(IOU_THRESHOLDS[threshold]) - iou) > 1e-10 or max_dets > self.max_dets:
            raise DiagnosticError("unsupported match threshold/maxDets")
        matched = set()
        for category in self.categories:
            record = self._records[image_id, category, area]
            if record is not None:
                values = record["dtMatches"][threshold, :max_dets]
                ignored = record["dtIgnore"][threshold, :max_dets]
                matched.update(int(x) for x in values[~ignored] if x > 0)
        return matched

    def describe(self):
        return {"gt_canonical_sha256": self.gt_sha256,
                "prediction_canonical_sha256": self.prediction_sha256,
                "image_ids": list(self.image_ids), "axes": copy.deepcopy(self.axes),
                "matching_backend": "pycocotools.COCOeval",
                "matching_backend_identity": copy.deepcopy(self.backend_identity),
                "accumulation": "literal_instance_order_exact_dataset_COCO_AP",
                "faster_backend_replay_required": True}


def prepare_coco_cache(coco, predictions, max_dets=100):
    return CocoMetricCache(coco, predictions, max_dets=max_dets)


def build_development_clusters(manifest_records, image_ids):
    allowed = set(image_ids)
    rows = list(manifest_records)
    if (len(rows) != len(allowed) or {x["coco_image_id"] for x in rows} != allowed
            or any(x.get("split") not in ("val", "development") for x in rows)):
        raise DiagnosticError("development manifest must cover exactly the scored images")
    parents = {i: i for i in allowed}

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        parents[max(a, b)] = min(a, b)

    sequences, hashes = {}, {}
    for row in rows:
        i, sequence, sha = row["coco_image_id"], row["sequence_key"], row["image_sha256"]
        if (type(i) is not int or not isinstance(sequence, str) or not sequence
                or not isinstance(sha, str) or len(sha) != 64
                or any(c not in "0123456789abcdef" for c in sha)):
            raise DiagnosticError("invalid sequence or image-content identity")
        for key, table in ((sequence, sequences), (sha, hashes)):
            if key in table:
                union(i, table[key])
            else:
                table[key] = i
    groups = defaultdict(list)
    for i in sorted(allowed):
        groups[find(i)].append(i)
    return tuple(tuple(v) for _, v in sorted(groups.items()))


def _combined_score(cache, primary_scorer, ids):
    values = cache.score(ids)
    metrics = {f"coco_{k}": values[k] for k in ("AP", "AP50", "AP75", "AP_small", "AR_small")}
    if primary_scorer is not None:
        fn = getattr(primary_scorer, "score", primary_scorer)
        result = fn(ids)
        if not isinstance(result, Mapping) or "AP" not in result:
            raise DiagnosticError("injected primary scorer must return at least AP")
        metrics["primary_AP"] = result["AP"]
    if any(v is None or not math.isfinite(float(v)) for v in metrics.values()):
        raise DiagnosticError("a predeclared bootstrap endpoint became unavailable/nonfinite")
    return {k: float(v) for k, v in metrics.items()}


def paired_bootstrap_2x2(caches, manifest_records, *, primary_scorers=None,
                        policy=FROZEN_POLICY, error_reports=None, progress=None):
    policy = _policy(policy)
    if set(caches) != set(CELLS) or (primary_scorers is not None and set(primary_scorers) != set(CELLS)):
        raise DiagnosticError("exactly all four predeclared cells are required")
    first = caches[CELLS[0]]
    ids = first.image_ids
    if any(x.image_ids != ids or x.gt_sha256 != first.gt_sha256
           or x.max_dets != policy.coco_max_dets for x in caches.values()):
        raise DiagnosticError("four-cell COCO GT, image identity or maxDets mismatch")
    clusters = build_development_clusters(manifest_records, ids)
    if len(ids) != policy.expected_images or len(clusters) != policy.expected_clusters:
        raise DiagnosticError("frozen image/connected-cluster count mismatch")
    scorers = primary_scorers or {cell: None for cell in CELLS}
    point_cells = {cell: _combined_score(caches[cell], scorers[cell], ids) for cell in CELLS}
    metric_names = tuple(point_cells[CELLS[0]])
    if any(tuple(x) != metric_names for x in point_cells.values()):
        raise DiagnosticError("cell metric schema mismatch")
    point = np.asarray([[point_cells[c][m] for m in metric_names] for c in CELLS])
    coefficients = np.asarray(list(CONTRASTS.values()))
    point_contrasts = coefficients @ point
    output = {}

    for kind, units, count, seed in (
        ("cluster", clusters, policy.cluster_replicates, policy.rng_seed),
        ("image_sensitivity", tuple((i,) for i in ids), policy.image_replicates, policy.rng_seed + 1),
    ):
        rng = np.random.Generator(np.random.PCG64(seed))
        draws = rng.integers(0, len(units), size=(count, len(units)), dtype=np.int64)
        distribution = np.empty((count, len(CONTRASTS), len(metric_names)), dtype=np.float64)
        for replicate, draw in enumerate(draws):
            repeated = tuple(sorted(i for j in draw for i in units[int(j)]))
            values = np.asarray([list(_combined_score(caches[c], scorers[c], repeated).values())
                                 for c in CELLS])
            distribution[replicate] = coefficients @ values
            if progress is not None:
                progress(kind, replicate + 1, count)
        ci = np.quantile(distribution, [.025, .975], axis=0, method="linear")
        output[kind] = {
            "replicates": count, "unit_count": len(units), "seed": seed,
            "draws_sha256": hashlib.sha256(draws.astype("<i8").tobytes()).hexdigest(),
            "distribution_sha256": hashlib.sha256(distribution.astype("<f8").tobytes()).hexdigest(),
            "distribution_axes": ["replicate", "contrast", "metric"],
            "distribution": distribution.tolist(),
            "contrasts": {
                name: {metric: {"point_pp": float(point_contrasts[j, k]),
                                "ci95_pp": [float(ci[0, j, k]), float(ci[1, j, k])]}
                       for k, metric in enumerate(metric_names)}
                for j, name in enumerate(CONTRASTS)},
        }
    diagonal = output["cluster"]["contrasts"]["diagonal_total"]
    gate = {"available": False, "eligible_for_fixed_paired_seed1_and_seed2": False,
            "not_model_certification": True, "not_training_seed_uncertainty": True}
    if primary_scorers is not None and error_reports is not None:
        if set(error_reports) != set(CELLS):
            raise DiagnosticError("seed gate requires four bound error reports")
        small_counts = set()
        for cell in CELLS:
            error = error_reports[cell]
            if (error.get("gt_sha256") != caches[cell].gt_sha256
                    or error.get("prediction_sha256") != caches[cell].prediction_sha256
                    or error.get("max_dets_per_image_per_category") != policy.coco_max_dets
                    or "0.5" not in error.get("by_iou", {})):
                raise DiagnosticError("seed gate error report identity/policy mismatch")
            small_counts.add(error["by_iou"]["0.5"]["small_gt_count"])
        if len(small_counts) != 1:
            raise DiagnosticError("seed gate small-GT denominator mismatch")
        before = error_reports["t640_e640"]["by_iou"]["0.5"]["small_unmatched"]
        after = error_reports["t896_e896"]["by_iou"]["0.5"]["small_unmatched"]
        checks = {
            "uniform_primary_diagonal_positive": diagonal["primary_AP"]["point_pp"] > 0.,
            "coco_small_diagonal_positive": diagonal["coco_AP_small"]["point_pp"] > 0.,
            "cluster_small_ci_lower_positive": diagonal["coco_AP_small"]["ci95_pp"][0] > 0.,
            "small_unmatched_at_iou50_decreases": after < before,
        }
        gate.update(available=True, checks=checks,
                    eligible_for_fixed_paired_seed1_and_seed2=all(checks.values()) and not policy.fixture_only)
    return {
        "schema_version": 1, "method": frozen_method(), "effective_policy": asdict(policy),
        "method_sha256": _digest(frozen_method()), "fixture_only": policy.fixture_only,
        "cells": list(CELLS), "metrics": list(metric_names), "contrasts": list(CONTRASTS),
        "point_cells": point_cells, "clusters": [list(x) for x in clusters],
        "manifest_canonical_sha256": _digest(manifest_records),
        "cache_identities": {c: caches[c].describe() for c in CELLS},
        "resampling": output, "seed_gate": gate,
        "interpretation": [
            "CI is conditional on these fixed trained weights and development clusters.",
            "It does not estimate training-seed variation or certify held-out generalization.",
            "Only the predeclared diagonal gate is used; other contrast CIs are descriptive.",
            "Image bootstrap is a sensitivity analysis, not a replacement for cluster CI.",
            "Uniform-primary and COCO-small use explicitly different GT/evaluator contracts.",
        ],
    }


def _iou(boxes_a, boxes_b):
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, :2] + a[:, None, 2:], b[None, :, :2] + b[None, :, 2:])
    intersection = np.maximum(0., hi - lo).prod(axis=2)
    union = a[:, 2:].prod(axis=1)[:, None] + b[:, 2:].prod(axis=1)[None, :] - intersection
    return intersection / np.maximum(union, np.finfo(np.float64).tiny)


def _lineage_labels(coco, lineage_rows, source_references):
    if lineage_rows is None:
        return {}, {"available": False, "reason": "development lineage not supplied"}
    images = {x.get("stable_image_id"): x for x in coco["images"]}
    if None in images or len(images) != len(coco["images"]):
        raise DiagnosticError("lineage join requires unique stable image identities")
    gt = {x.get("stable_annotation_id"): x for x in coco["annotations"]}
    if None in gt or len(gt) != len(coco["annotations"]):
        raise DiagnosticError("lineage join requires unique stable annotation identities")
    labels, seen = {}, set()
    for row in lineage_rows:
        key = row["annotation_id"]
        if key in seen or row["image_id"] not in images or row["split"] not in ("val", "development"):
            raise DiagnosticError("lineage duplicates or non-development image identity")
        seen.add(key)
        raw = row["raw_fields"]
        if len(raw) != 8 or raw[6] not in ("0", "1") or raw[7] not in ("0", "1", "2"):
            raise DiagnosticError("unexpected raw truncation/occlusion label domain")
        if key in gt:
            target = gt[key]
            box = row["bbox_xyxy"]
            xywh = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
            if (row["status"] != "keep" or xywh != list(target["bbox"])
                    or row["area"] != target["area"]
                    or row["coco_category_id"] != target["category_id"]
                    or images[row["image_id"]]["id"] != target["image_id"]):
                raise DiagnosticError("lineage/formal COCO annotation join mismatch")
            labels[target["id"]] = {"truncation": int(raw[6]), "occlusion": int(raw[7])}
    if set(labels) != {x["id"] for x in coco["annotations"]}:
        raise DiagnosticError("development lineage does not cover all formal GT")
    return labels, {
        "available": True, "formal_gt_join_count": len(labels),
        "lineage_row_count": len(lineage_rows),
        "lineage_canonical_sha256": _digest(lineage_rows),
        "source_references": copy.deepcopy(source_references or {}),
        "spatial_ignore_coverage_verified": False,
        "empty_ignore_sidecar_does_not_prove_official_coverage": True,
    }


def decompose_errors(coco, predictions, lineage_rows=None, *, source_references=None,
                     max_dets=100, iou_thresholds=(.5, .75), background_iou=.1,
                     include_rows=True, cache=None):
    """Mutually exclusive observed symptoms; orthogonal metadata are not causes."""
    cache = cache or prepare_coco_cache(coco, predictions, max_dets=300)
    if (cache.gt_sha256 != _digest(coco) or cache.prediction_sha256 != _digest(predictions)
            or cache.max_dets < 300 or not 0 < background_iou < min(iou_thresholds)):
        raise DiagnosticError("error cache/input or threshold mismatch")
    labels, lineage_binding = _lineage_labels(coco, lineage_rows, source_references)
    targets, candidates = defaultdict(list), defaultdict(list)
    for row in coco["annotations"]:
        targets[row["image_id"]].append(row)
    for row_id, row in enumerate(predictions, 1):
        candidates[row["image_id"]].append((row_id, row))
    if (type(max_dets) is not int or not 1 <= max_dets <= 300
            or not iou_thresholds or any(x not in (.5, .75) for x in iou_thresholds)):
        raise DiagnosticError("error ledger threshold or maxDets drift")
    per_threshold = {}
    for threshold in iou_thresholds:
        gt_counts, pred_counts = Counter(), Counter()
        gt_rows, pred_rows, per_image = [], [], []
        for image_id in cache.image_ids:
            gt = targets[image_id]
            dt = candidates[image_id]
            overlap = _iou([x[1]["bbox"] for x in dt], [x["bbox"] for x in gt])
            matched = cache.matched_ids(image_id, iou=threshold, max_dets=max_dets)
            all_matched = cache.matched_ids(image_id, iou=threshold, max_dets=300)
            image_gt_counts = Counter()
            for j, target in enumerate(gt):
                if target["area"] > 1024. or target.get("iscrowd", 0):
                    continue
                values = overlap[:, j]
                same = np.asarray([x[1]["category_id"] == target["category_id"] for x in dt], dtype=bool)
                if target["id"] in matched:
                    reason = "true_positive"
                elif target["id"] in all_matched:
                    reason = "maxdets_limited"
                elif np.any((values >= threshold) & same):
                    reason = "matching_competition"
                elif np.any((values >= threshold) & ~same):
                    reason = "classification"
                elif np.any((values >= background_iou) & same):
                    reason = "localization"
                elif np.any(values >= background_iou):
                    reason = "classification_and_localization"
                else:
                    reason = "no_saved_candidate"
                gt_counts[reason] += 1
                image_gt_counts[reason] += 1
                x, y, w, h = target["bbox"]
                image = cache._image_table[image_id]
                neighbor_iou = _iou([target["bbox"]], [z["bbox"] for z in gt])[0]
                tags = {**labels.get(target["id"], {"truncation": None, "occlusion": None}),
                        "neighbors_iou_ge_0_1": int(np.count_nonzero(neighbor_iou >= .1) - 1),
                        "border_touch_1px": bool(x <= 1 or y <= 1 or
                                                x + w >= image["width"] - 1 or y + h >= image["height"] - 1)}
                if include_rows:
                    gt_rows.append({"image_id": image_id, "annotation_id": target["id"],
                                    "stable_annotation_id": target.get("stable_annotation_id"),
                                    "reason": reason, "tags": tags})
            dt_state = {}
            ti = int(np.argmin(abs(IOU_THRESHOLDS - threshold)))
            for category in cache.categories:
                record = cache._records[image_id, category, "small"]
                if record is None:
                    continue
                for index, det_id in enumerate(record["dtIds"]):
                    if index >= max_dets:
                        dt_state[det_id] = "maxdets_excluded"
                    elif record["dtIgnore"][ti, index]:
                        dt_state[det_id] = "evaluation_ignored"
                    elif record["dtMatches"][ti, index] > 0:
                        dt_state[det_id] = "true_positive"
            for index, (row_id, prediction) in enumerate(dt):
                reason = dt_state.get(row_id)
                if reason is None:
                    values = overlap[index]
                    same = np.asarray([x["category_id"] == prediction["category_id"] for x in gt], dtype=bool)
                    if np.any((values >= threshold) & same):
                        reason = "duplicate"
                    elif np.any((values >= threshold) & ~same):
                        reason = "classification"
                    elif np.any((values >= background_iou) & same):
                        reason = "localization"
                    elif np.any(values >= background_iou):
                        reason = "classification_and_localization"
                    else:
                        reason = "background"
                pred_counts[reason] += 1
                if include_rows:
                    pred_rows.append({"image_id": image_id, "prediction_row_id": row_id, "reason": reason})
            per_image.append({"image_id": image_id, "gt_counts": dict(image_gt_counts)})
        # Dataset/category score-order symptoms, separate from the exclusive GT ledger.
        ranking = {}
        ti = int(np.argmin(abs(IOU_THRESHOLDS - threshold)))
        for category in cache.categories:
            parts = [cache._records[i, category, "small"] for i in cache.image_ids]
            parts = [r for r in parts if r is not None]
            if not parts:
                continue
            scores = np.concatenate([np.asarray(r["dtScores"][:max_dets]) for r in parts])
            states = np.concatenate([r["dtMatches"][ti, :max_dets] > 0 for r in parts])
            ignored = np.concatenate([r["dtIgnore"][ti, :max_dets] for r in parts])
            order = np.argsort(-scores, kind="stable")
            states = states[order][~ignored[order]]
            tp, fp = int(states.sum()), int((~states).sum())
            inversions = int(np.cumsum(~states, dtype=np.int64)[states].sum())
            ranking[str(category)] = {
                "true_positives": tp, "false_positives": fp,
                "fp_ranked_before_tp_pairs": inversions,
                "fp_before_tp_pair_fraction": inversions / (tp * fp) if tp and fp else None,
                "tie_policy": "COCO_stable_score_order",
            }
        small_total = sum(gt_counts.values())
        if sum(pred_counts.values()) != len(predictions):
            raise DiagnosticError("prediction partition does not conserve rows")
        per_threshold[str(float(threshold))] = {
            "gt_counts": {k: gt_counts[k] for k in GT_REASONS},
            "prediction_counts": {k: pred_counts[k] for k in PRED_REASONS},
            "small_gt_count": small_total,
            "small_unmatched": small_total - gt_counts["true_positive"],
            "small_recall": gt_counts["true_positive"] / small_total if small_total else None,
            "per_image": per_image, "gt_rows": gt_rows, "prediction_rows": pred_rows,
            "ranking_by_category": ranking,
        }
    return {"schema_version": 1, "gt_sha256": cache.gt_sha256,
            "prediction_sha256": cache.prediction_sha256, "max_dets_per_image_per_category": max_dets,
            "area": "COCO_small_original_pixels_inclusive_0_1024",
            "metric_basis": "converted_COCO_secondary_not_uniform_primary",
            "by_iou": per_threshold, "lineage_binding": lineage_binding,
            "interpretation": [
                "Reasons are exclusive by the declared precedence, not identified causal mechanisms.",
                "Matching competition combines ranking/assignment/crowding; AP ranking is not GT recall.",
                "No saved candidate means absent from retained predictions, not absent from all model queries.",
                "Occlusion, truncation, neighborhood density and border contact are orthogonal descriptors.",
                "Border contact is not a replacement for raw truncation labels.",
                "Ranking pair inversions are an orthogonal score-order diagnostic, not an added error count.",
                "COCO maxDets is per image per category; global model top300 is a different truncation.",
            ]}


def _tile(box, width, height, grid):
    cx = (float(box[0]) + float(box[2]) / 2.) / width
    cy = (float(box[1]) + float(box[3]) / 2.) / height
    x = min(grid - 1, max(0, math.floor(cx * grid)))
    y = min(grid - 1, max(0, math.floor(cy * grid)))
    return y * grid + x


def routing_oracle_diagnostic(coco, coarse_predictions, fine_predictions, *,
                             raw_coarse_by_image, policy=FROZEN_POLICY,
                             coarse_cache=None, fine_cache=None):
    """Equal-budget tile-capture diagnostic; never a stitched/deployable AP claim."""
    policy = _policy(policy)
    coarse = coarse_cache or prepare_coco_cache(coco, coarse_predictions)
    fine = fine_cache or prepare_coco_cache(coco, fine_predictions)
    if (coarse.gt_sha256 != fine.gt_sha256 or coarse.gt_sha256 != _digest(coco)
            or coarse.prediction_sha256 != _digest(coarse_predictions)
            or fine.prediction_sha256 != _digest(fine_predictions)
            or set(raw_coarse_by_image) != set(coarse.image_ids)):
        raise DiagnosticError("routing inputs must share exact GT/images and raw coarse coverage")
    grid, budget = policy.routing_grid, policy.routing_tiles
    targets = defaultdict(list)
    for row in coco["annotations"]:
        targets[row["image_id"]].append(row)
    totals = Counter()
    rows = []
    for image_id in coarse.image_ids:
        image = coarse._image_table[image_id]
        coarse_tp = coarse.matched_ids(image_id, iou=.5, max_dets=policy.coco_max_dets)
        fine_tp = fine.matched_ids(image_id, iou=.5, max_dets=policy.coco_max_dets)
        recoverable = fine_tp - coarse_tp
        gt_tiles = Counter(_tile(x["bbox"], image["width"], image["height"], grid)
                           for x in targets[image_id] if x["id"] in recoverable)
        raw = raw_coarse_by_image[image_id]
        logits = np.asarray(raw["pred_logits"], dtype=np.float64)
        boxes = np.asarray(raw["pred_boxes"], dtype=np.float64)
        if (logits.ndim != 2 or logits.shape[1] != 10
                or boxes.shape != (len(logits), 4) or not len(logits)
                or not np.isfinite(logits).all() or not np.isfinite(boxes).all()
                or np.any(boxes[:, 2:] <= 0) or np.any(boxes < 0) or np.any(boxes > 1)
                or (not policy.fixture_only and len(logits) != 300)):
            raise DiagnosticError("raw coarse query geometry must be finite [300,10]/[300,4]")
        probability = (1. / (1. + np.exp(-np.clip(logits, -80, 80)))).max(axis=1)
        tile_scores = np.zeros(grid * grid, dtype=np.float64)
        for box, p in zip(boxes, probability):
            area = box[2] * box[3] * policy.coarse_input ** 2
            if p >= policy.routing_min_probability and area <= policy.routing_small_input_area:
                xywh = ((box[0] - box[2] / 2.) * image["width"],
                        (box[1] - box[3] / 2.) * image["height"],
                        box[2] * image["width"], box[3] * image["height"])
                tile_scores[_tile(xywh, image["width"], image["height"], grid)] += 4. * p * (1. - p)
        # Route selection uses coarse predictions only; GT is read only by oracle and measurement.
        predicted = sorted(range(grid * grid), key=lambda k: (-tile_scores[k], k))[:budget]
        random_tiles = sorted(range(grid * grid), key=lambda k: hashlib.sha256(
            f"{policy.rng_seed}:{image_id}:{k}".encode()).digest())[:budget]
        oracle = sorted(range(grid * grid), key=lambda k: (-gt_tiles[k], k))[:budget]
        selected = {"random": random_tiles, "coarse_query_uncertainty_small": predicted,
                    "gt_recoverable_oracle_non_deployable": oracle}
        captured = {key: sum(gt_tiles[k] for k in value) for key, value in selected.items()}
        if any(len(set(v)) != budget for v in selected.values()):
            raise DiagnosticError("routing budget/uniqueness mismatch")
        totals["recoverable_small_gt"] += len(recoverable)
        for key, value in captured.items():
            totals[key] += value
        rows.append({"image_id": image_id, "recoverable_small_gt": len(recoverable),
                     "selected_tiles": selected, "captured": captured,
                     "coarse_tile_scores": tile_scores.tolist()})
    core = policy.fine_input // grid
    crop = core + 2 * policy.halo_pixels
    return {
        "schema_version": 1, "policy": asdict(policy), "method_sha256": _digest(frozen_method()),
        "coarse_identity": coarse.describe(), "fine_identity": fine.describe(),
        "recoverable_definition": "fine_TP_minus_coarse_TP_at_COCO_small_IoU0.5_maxDets100",
        "totals": dict(totals),
        "capture_fractions": {key: totals[key] / totals["recoverable_small_gt"]
                              if totals["recoverable_small_gt"] else None
                              for key in ("random", "coarse_query_uncertainty_small",
                                          "gt_recoverable_oracle_non_deployable")},
        "per_image": rows,
        "cost": {"grid": [grid, grid], "tiles_per_image": budget,
                 "core_pixels_in_fine_canvas": [core, core], "halo_pixels": policy.halo_pixels,
                 "billed_crop_pixels": [crop, crop], "edge_padding_refund": False,
                 "pixel_proxy_per_image": policy.coarse_input ** 2 + budget * crop ** 2,
                 "dense_fine_pixel_reference": policy.fine_input ** 2,
                 "actual_sparse_flops": None, "actual_sparse_latency": None,
                 "diagnostic_gpu_forward_count": 0},
        "limitations": [
            "Oracle sees GT only for an explicitly non-deployable equal-budget upper bound.",
            "Tile recovery is based on full-frame fine predictions with global context.",
            "It does not show that an actual crop/sparse operator can reproduce these detections.",
            "No boxes, labels, scores or detections are corrected or stitched to claim AP.",
            "Pixel proxy is not measured FLOPs, latency, memory saving or end-to-end speedup.",
            "This unsupervised fixed rule is not fitted or tuned on development labels.",
        ],
    }


def _raw_boxes_original_xywh(boxes, width, height):
    """Reproduce eager FP32 vendor cxcywh->xyxy->original xywh arithmetic."""
    boxes = np.asarray(boxes, dtype=np.float32)
    half = boxes[:, 2:] * np.float32(.5)
    xyxy = np.concatenate((boxes[:, :2] - half, boxes[:, :2] + half), axis=1)
    xyxy *= np.asarray([width, height, width, height], dtype=np.float32)
    xywh = xyxy.copy()
    xywh[:, 2:] = xyxy[:, 2:] - xyxy[:, :2]
    return xywh


def raw_query_coverage(coco, predictions, raw_by_image, *, source_references=None,
                       policy=FROZEN_POLICY, include_rows=True):
    """Raw-query geometry/ranking upper bounds; no oracle box/label/AP repair.

    raw_by_image rows require pred_logits, pred_boxes, pred_sigmoid_scores and
    actual topk_indices. The last two are saved from runtime, so saturated score
    ties are not reconstructed from CPU sigmoid or guessed from saved boxes.
    """
    policy = _policy(policy)
    ids = _validate_coco(coco, predictions)
    if set(raw_by_image) != set(ids):
        raise DiagnosticError("raw query coverage must cover exactly the COCO image IDs")
    images = {x["id"]: x for x in coco["images"]}
    gt_by_id, pred_by_id = defaultdict(list), defaultdict(list)
    for row in coco["annotations"]:
        if row["area"] <= 1024. and not row.get("iscrowd", 0):
            gt_by_id[row["image_id"]].append(row)
    for row in predictions:
        pred_by_id[row["image_id"]].append(row)
    thresholds = (.1, .5, .75)
    counts = {str(t): Counter() for t in thresholds}
    reasons, gt_rows, image_checks = Counter(), [], []
    raw_hasher = hashlib.sha256()
    for image_id in ids:
        raw = raw_by_image[image_id]
        required = ("pred_logits", "pred_boxes", "pred_sigmoid_scores", "topk_indices")
        if any(k not in raw for k in required):
            raise DiagnosticError("runtime sigmoid scores and actual flat topk indices are required")
        logits = np.asarray(raw["pred_logits"])
        boxes = np.asarray(raw["pred_boxes"])
        probability = np.asarray(raw["pred_sigmoid_scores"])
        top = np.asarray(raw["topk_indices"])
        q = len(logits)
        retained = pred_by_id[image_id]
        if (logits.shape != (q, 10) or not q or boxes.shape != (q, 4)
                or probability.shape != logits.shape or top.shape != (len(retained),)
                or top.dtype.kind not in ("i", "u")
                or not np.isfinite(logits).all() or not np.isfinite(boxes).all()
                or not np.isfinite(probability).all()
                or np.any(boxes < 0) or np.any(boxes > 1) or np.any(boxes[:, 2:] <= 0)
                or np.any(probability < 0) or np.any(probability > 1)
                or np.any(top < 0) or np.any(top >= q * 10)
                or len(np.unique(top)) != len(top)
                or (not policy.fixture_only and (q != 300 or len(top) != 300
                    or logits.dtype != np.float32 or boxes.dtype != np.float32
                    or probability.dtype != np.float32))):
            raise DiagnosticError("raw query shape/dtype/finiteness/topk contract mismatch")
        analytical = 1. / (1. + np.exp(-np.clip(logits.astype(np.float64), -80, 80)))
        if not np.allclose(analytical, probability, rtol=1e-6, atol=2e-7):
            raise DiagnosticError("runtime probabilities are inconsistent with raw logits")
        for name, values in zip(required, (logits, boxes, probability, top)):
            raw_hasher.update(_canonical({"image_id": image_id, "array": name,
                                          "shape": list(values.shape), "dtype": str(values.dtype)}))
            raw_hasher.update(np.ascontiguousarray(values).tobytes())
        flat = probability.reshape(-1)
        selected_mask = np.zeros(q * 10, dtype=bool)
        selected_mask[top] = True
        if len(top) and (np.any(np.diff(flat[top].astype(np.float64)) > 0)
                         or (np.any(~selected_mask)
                             and float(flat[top].min()) < float(flat[~selected_mask].max()))):
            raise DiagnosticError("saved flat topk indices are not the runtime score top-k")
        dimensions = images[image_id]
        geometry = _raw_boxes_original_xywh(boxes, dimensions["width"], dimensions["height"])
        maximum_score_error = 0.
        for row, pair in zip(retained, top):
            query, category = int(pair) // 10, int(pair) % 10 + 1
            score_error = abs(float(row["score"]) - float(flat[pair]))
            maximum_score_error = max(maximum_score_error, score_error)
            if (row["category_id"] != category or score_error > 2e-7
                    or not np.array_equal(np.asarray(row["bbox"], dtype=np.float32), geometry[query])):
                raise DiagnosticError("saved prediction does not reproduce actual query/class/score/box")
        order = np.argsort(-flat.astype(np.float64), kind="stable")
        rank = np.empty(q * 10, dtype=np.int64)
        rank[order] = np.arange(1, q * 10 + 1)
        sorted_negative = -flat[order].astype(np.float64)
        rank_low = np.searchsorted(sorted_negative, -flat.astype(np.float64), side="left") + 1
        rank_high = np.searchsorted(sorted_negative, -flat.astype(np.float64), side="right")
        targets = gt_by_id[image_id]
        raw_iou = _iou(geometry, [x["bbox"] for x in targets])
        saved_iou = _iou([x["bbox"] for x in retained], [x["bbox"] for x in targets])
        top1 = probability.argmax(axis=1) + 1
        for j, target in enumerate(targets):
            best = int(np.argmax(raw_iou[:, j]))
            maximum = float(raw_iou[best, j])
            category = target["category_id"] - 1
            pair = best * 10 + category
            saved_maximum = float(saved_iou[:, j].max()) if len(retained) else 0.
            saved_same = np.asarray([x["category_id"] == target["category_id"] for x in retained],
                                    dtype=bool)
            coverage = {}
            for threshold in thresholds:
                raw_geometry = raw_iou[:, j] >= threshold
                raw_pairs = np.flatnonzero(raw_geometry) * 10 + category
                saved_any = bool(np.any(saved_iou[:, j] >= threshold))
                saved_correct = bool(np.any((saved_iou[:, j] >= threshold) & saved_same))
                retained_correct = bool(np.any(selected_mask[raw_pairs]))
                entry = {"raw_geometry_covered": bool(np.any(raw_geometry)),
                         "saved_any_class_covered": saved_any,
                         "saved_correct_class_covered": saved_correct,
                         "geometric_query_correct_pair_retained": retained_correct}
                coverage[str(threshold)] = entry
                counts[str(threshold)]["small_gt_count"] += 1
                for key, value in entry.items():
                    counts[str(threshold)][key] += value
            if saved_maximum >= .1:
                reason = "saved_geometric_candidate_exists"
            elif maximum < .1:
                reason = "raw_geometry_absent"
            else:
                reason = "geometric_queries_lost_to_flat_topk"
            reasons[reason] += 1
            if include_rows:
                gt_rows.append({
                    "image_id": image_id, "annotation_id": target["id"],
                    "stable_annotation_id": target.get("stable_annotation_id"),
                    "best_iou_query_id": best, "highest_raw_iou": maximum,
                    "highest_saved_iou_any_class": saved_maximum,
                    "best_query_correct_class_probability": float(flat[pair]),
                    "correct_pair_score_rank_canonical": int(rank[pair]),
                    "correct_pair_score_rank_tie_interval": [int(rank_low[pair]), int(rank_high[pair])],
                    "correct_pair_actual_flat_topk_retained": bool(selected_mask[pair]),
                    "best_query_top1_category_canonical": int(top1[best]),
                    "best_query_top1_correct": bool(top1[best] == target["category_id"]),
                    "absence_reason_at_iou_0_1": reason, "coverage": coverage,
                })
        image_checks.append({"image_id": image_id, "queries": q, "class_pairs": q * 10,
                             "saved_pairs": len(top), "max_score_abs_difference": maximum_score_error,
                             "saved_query_geometry_reproduced_exactly_in_float32": True})
    return {
        "schema_version": 1, "method": "raw_query_geometry_and_runtime_score_coverage_v1",
        "gt_canonical_sha256": _digest(coco), "prediction_canonical_sha256": _digest(predictions),
        "raw_array_semantic_sha256": raw_hasher.hexdigest(),
        "source_references": copy.deepcopy(source_references or {}),
        "iou_thresholds": list(thresholds), "area": "COCO_small_original_pixels_inclusive_0_1024",
        "best_geometry_tie_policy": "lowest_query_id",
        "score_rank_tie_policy": "report_full_tie_interval_and_canonical_flat_index_order",
        "actual_flat_topk_membership_authoritative": True,
        "by_iou": {k: dict(v) for k, v in counts.items()},
        "saved_candidate_absence_partition_at_0_1": dict(reasons),
        "per_gt": gt_rows, "image_checks": image_checks,
        "interpretation": [
            "Geometry coverage is many-to-one and an optimistic candidate upper bound.",
            "Query/class retention is observed from runtime indices, not reconstructed from box equality.",
            "Ranking ties can straddle top300; canonical rank does not replace actual membership.",
            "Best-geometry query is selected without looking at its class score.",
            "These statistics do not identify a causal AP gain or repair boxes, classes, scores, or labels.",
            "Model query capacity, flat top300 retention and evaluator maxDets are separate limits.",
        ],
    }
