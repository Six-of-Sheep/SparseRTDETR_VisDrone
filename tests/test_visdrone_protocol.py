"""Synthetic CPU oracles for the frozen VisDrone protocol."""

from __future__ import annotations

import os
import hashlib
import inspect
import unittest

from sparse_rtdetr.data_protocol import (
    COCODiagnosticInput,
    ImageIdentity,
    PrimaryEvaluatorInput,
    annotate_records,
    assert_primary_input,
    assert_secondary_input,
    build_atomic_groups,
    canonical_json_bytes,
    category_mapping,
    coco_to_training_id,
    ensure_allowed_dataset_path,
    parse_annotation_bytes,
    plan_confirmatory_split,
    plan_confirmatory_split_v2,
    protocol_schema,
    protocol_v2_schema,
    raw_to_training_id,
    stable_annotation_id,
    stable_image_id,
    training_to_coco_id,
)
from sparse_rtdetr.data_protocol.evaluation import Detection
from sparse_rtdetr.data_protocol.parser import AnnotationParseError
from sparse_rtdetr.data_protocol.schema import ProtocolContractError
from sparse_rtdetr.data_protocol.split import AtomicGroup, ProtocolV2ContractError


class VisDroneProtocolTests(unittest.TestCase):
    def test_parser_standard_and_single_trailing_comma(self):
        standard = parse_annotation_bytes(b"1,2,3,4,1,1,0,0\n")[0]
        trailing = parse_annotation_bytes(b"1,2,3,4,1,1,0,0,\n")[0]
        self.assertEqual(standard.fields.bbox_xyxy, (1.0, 2.0, 4.0, 6.0))
        self.assertEqual(standard.fields.area, 12.0)
        self.assertFalse(standard.had_single_trailing_empty_field)
        self.assertTrue(trailing.had_single_trailing_empty_field)
        self.assertNotEqual(standard.raw_line_sha256, trailing.raw_line_sha256)

    def test_parser_rejects_field_count_nan_inf_and_non_integer_category(self):
        for payload in (
            b"1,2,3,4,1,1,0\n",
            b"1,2,3,4,1,1,0,0,extra\n",
            b"1,2,3,4,nan,1,0,0\n",
            b"1,2,3,4,1,inf,0,0\n",
            b"1,2,3,4,1,1.5,0,0\n",
        ):
            with self.assertRaises(AnnotationParseError):
                parse_annotation_bytes(payload)

    def test_category_boundaries_and_round_trip(self):
        self.assertEqual(raw_to_training_id(1), 0)
        self.assertEqual(training_to_coco_id(9), 10)
        self.assertEqual(coco_to_training_id(1), 0)
        self.assertEqual(category_mapping(0).reason_code, "SPATIAL_IGNORE_REGION")
        self.assertEqual(category_mapping(11).reason_code, "UNSCORED_CATEGORY_11")
        with self.assertRaises(ProtocolContractError):
            raw_to_training_id(0)
        with self.assertRaises(ProtocolContractError):
            coco_to_training_id(11)

    def test_lineage_duplicate_nonpositive_and_ignore_semantics(self):
        parsed = parse_annotation_bytes(
            b"1,2,3,4,1,1,0,0\n"
            b"1,2,3,4,1,1,0,0\n"
            b"2,2,0,4,1,1,0,0\n"
            b"3,3,4,4,0,1,0,0\n"
            b"0,0,20,20,1,0,0,0\n"
            b"4,4,4,4,1,11,0,0\n"
        )
        lineage = annotate_records("train", "train/000001.txt", parsed)
        self.assertEqual(lineage[0].status.value, "keep")
        self.assertEqual(lineage[1].reason_code, "FILTERED_DUPLICATE_EXACT_GT")
        self.assertEqual(lineage[2].reason_code, "NON_POSITIVE_FORMAL_BBOX")
        self.assertEqual(lineage[3].reason_code, "SCORE_ZERO_IGNORED")
        self.assertTrue(lineage[4].ignore_region)
        self.assertEqual(lineage[5].reason_code, "UNSCORED_CATEGORY_11")
        self.assertEqual(lineage[0].physical_line_number, 1)
        self.assertEqual(len(lineage[0].raw_line_sha256), 64)
        self.assertNotEqual(lineage[0].annotation_id, lineage[1].annotation_id)

    def test_ids_and_canonical_bytes_are_stable(self):
        path = "train/000001.jpg"
        image_id = stable_image_id("train", path)
        first = stable_annotation_id(image_id, 4, "a" * 64)
        second = stable_annotation_id(image_id, 4, "a" * 64)
        self.assertEqual(first, second)
        self.assertEqual(canonical_json_bytes({"b": 1, "a": "x"}), b'{"a":"x","b":1}')
        self.assertEqual(image_id, stable_image_id("train", path))

    def test_protocol_schema_and_bytes_are_repeatable(self):
        first = canonical_json_bytes(protocol_schema())
        second = canonical_json_bytes(protocol_schema())
        self.assertEqual(first, second)
        self.assertIn(b'"test_access_allowed":false', first)
        self.assertIn(b'"target_confirmatory_images":647', first)

    def test_v1_planner_source_and_v2_identity_are_separate(self):
        from sparse_rtdetr.data_protocol.split import plan_confirmatory_split

        self.assertEqual(
            hashlib.sha256(inspect.getsource(plan_confirmatory_split).encode("utf-8")).hexdigest(),
            "ab6e54ef4ab83add4810b1f08e4345a18e6a1a5b97f8365205ae58f9fb927aad",
        )
        self.assertEqual(protocol_schema()["protocol_id"], "P3-VISDRONE-DATA-PROTOCOL-V1")
        self.assertEqual(protocol_v2_schema()["protocol_id"], "P3-VISDRONE-DATA-PROTOCOL-V2")
        self.assertNotIn("selection_policy", protocol_schema()["split"])
        self.assertEqual(
            protocol_v2_schema()["split"]["selection_policy"],
            "feasibility_first_nearest_hash_prefix_v2",
        )

    def test_test_split_path_is_rejected_without_filesystem_access(self):
        for split, path in (("test", "test/000001.jpg"), ("train", "train/test/000001.jpg")):
            with self.assertRaises(ProtocolContractError):
                ensure_allowed_dataset_path(path, split)
        self.assertEqual(ensure_allowed_dataset_path("val/000001.jpg", "val"), "val/000001.jpg")

    @staticmethod
    def _image(path, sequence, sha, counts=(1,) + (0,) * 9, small=0):
        return ImageIdentity(path, sequence, sha, sum(counts), counts, small)

    def test_connected_atomic_groups_merge_sequence_and_duplicate_content(self):
        images = [
            self._image("a/0001.jpg", "seq-a", "a" * 64),
            self._image("a/0002.jpg", "seq-a", "b" * 64),
            self._image("b/0001.jpg", "seq-b", "b" * 64),
            self._image("c/0001.jpg", "seq-c", "c" * 64),
        ]
        groups = build_atomic_groups(images)
        self.assertEqual(len(groups), 2)
        merged = next(group for group in groups if "seq-a" in group.sequence_keys)
        self.assertEqual(merged.sequence_keys, ("seq-a", "seq-b"))
        self.assertEqual(merged.image_count, 3)

    def test_sequence_metadata_and_shared_val_exclusion(self):
        images = [
            self._image("a/1.jpg", "shared", "a" * 64),
            self._image("b/1.jpg", "private", "b" * 64),
        ]
        metadata = {item.relative_path: item.sequence_key for item in images}
        groups = build_atomic_groups(images, metadata)
        plan = plan_confirmatory_split(
            groups,
            train_image_count=10,
            target_ratio=0.5,
            development_sequence_keys={"shared"},
            tolerance=1.0,
        )
        self.assertEqual(len(plan.confirmatory_group_ids), 1)
        shared_group = next(group.group_id for group in groups if "shared" in group.sequence_keys)
        self.assertNotEqual(plan.confirmatory_group_ids[0], shared_group)
        self.assertFalse(plan.selection_allowed)

    def test_hash_prefix_and_tie_break_are_deterministic(self):
        images = [
            self._image(f"{name}/1.jpg", name, sha, counts=(1,) + (0,) * 9)
            for name, sha in (("a", "a" * 64), ("b", "b" * 64), ("c", "c" * 64))
        ]
        groups = build_atomic_groups(images)
        first = plan_confirmatory_split(groups, train_image_count=4, target_ratio=0.5, tolerance=1.0)
        second = plan_confirmatory_split(groups, train_image_count=4, target_ratio=0.5, tolerance=1.0)
        self.assertEqual(first, second)
        self.assertEqual(first.target_image_count, 2)

    def test_distribution_constraint_fails_closed(self):
        images = [
            self._image("a/1.jpg", "a", "a" * 64, counts=(10,) + (0,) * 9),
            self._image("b/1.jpg", "b", "b" * 64, counts=(0, 10) + (0,) * 8),
        ]
        with self.assertRaises(ProtocolContractError):
            plan_confirmatory_split(build_atomic_groups(images), train_image_count=2, target_ratio=0.5)

    def test_v2_feasibility_first_selects_only_passing_prefixes(self):
        images = [
            self._image(f"{name}/1.jpg", name, sha, counts=(1,) * 10)
            for name, sha in (("a", "a" * 64), ("b", "b" * 64), ("c", "c" * 64))
        ]
        plan = plan_confirmatory_split_v2(
            build_atomic_groups(images), train_image_count=4, target_ratio=0.5, tolerance=0.05
        )
        self.assertEqual(plan.selection_policy, "feasibility_first_nearest_hash_prefix_v2")
        self.assertEqual(plan.evaluated_prefix_count, 3)
        self.assertEqual(plan.feasible_prefix_count, 2)
        self.assertEqual(plan.selected_prefix_length, 2)
        self.assertEqual(plan.selected_image_count, 2)
        self.assertTrue(all(passed for _, passed in plan.checks))
        self.assertFalse(plan.selection_allowed)
        self.assertFalse(plan.metrics_access_allowed)

    def test_v2_no_feasible_prefix_fails_with_v2_error(self):
        images = [
            self._image("a/1.jpg", "a", "a" * 64, counts=(10,) + (0,) * 9),
            self._image("b/1.jpg", "b", "b" * 64, counts=(0, 10) + (0,) * 8),
        ]
        with self.assertRaises(ProtocolV2ContractError):
            plan_confirmatory_split_v2(build_atomic_groups(images), train_image_count=2, target_ratio=0.5)

    def test_v2_not_over_target_tie_break(self):
        groups = tuple(
            AtomicGroup(
                group_id=f"group-{name}", sequence_keys=(name,), image_paths=(f"{name}.jpg",),
                image_sha256s=(name * 64,), image_count=2, valid_target_count=20,
                class_counts=(2,) * 10, small_target_count=2, identity_closed=True, abnormal=False,
            )
            for name in ("a", "b", "c")
        )
        plan = plan_confirmatory_split_v2(groups, train_image_count=5, target_ratio=0.5, tolerance=0.05)
        self.assertEqual(plan.target_image_count, 3)
        self.assertEqual(plan.selected_image_count, 2)

    def test_primary_and_secondary_schemas_are_separate(self):
        detection = Detection("image", 1, (0, 0, 2, 2), 0.5)
        primary = PrimaryEvaluatorInput(("image",), (detection,), tuple())
        secondary = COCODiagnosticInput(("image",), (detection,), tuple())
        self.assertIs(assert_primary_input(primary), primary)
        self.assertIs(assert_secondary_input(secondary), secondary)
        with self.assertRaises(ProtocolContractError):
            assert_primary_input(secondary)
        with self.assertRaises(ProtocolContractError):
            assert_secondary_input(primary)

    def test_protocol_cpu_boundary(self):
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES", ""), "")
        self.assertNotIn("torch", globals())
        self.assertNotIn("torchvision", globals())


if __name__ == "__main__":
    unittest.main()
