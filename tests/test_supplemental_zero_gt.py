"""Small, data-free regressions through the actual production interfaces.

The two reviewed line bytes and their recorded identities are fixed fixtures.
Padding is synthetic category-11 data, NOT a reconstruction of either original
annotation file. Recorded file SHAs are identity inputs to the production binder;
these tests do not certify a sealed file, read an image, or compute dataset AP.
"""
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
import confirmation_prepare as prepare
from sparse_rtdetr.localization.data import parse_raw
from sparse_rtdetr.data_protocol.supplemental_zero_gt import SupplementalZeroGT, specification
from sparse_rtdetr.baseline import primary_evaluator as primary
from sparse_rtdetr.r0.evaluation import primary_inputs, bridge_inputs

# Independent recorded expectations, rather than deriving the fixture from ROWS.
FIXTURES = (
    ('9999985_00000_d_0000020.txt',
     'ed0c75ca472b4571f42ee5d0ac61e8681a81dcdf6d9daf3307d334cfb01c6546',
     '999d84a9fcc6632645ffb9b2eca416f4e2d08a966d811d24277960e593f8c814',
     12, b'611,158,4,0,1,4,0,0\r\n',
     '620332c35ca9ce25489b90f61cd656bbcc26181d7f07aec513f63a8b12932a1e'),
    ('9999999_00590_d_0000267.txt',
     '4b1f0ecb0197dbc9575118fc6947383202dde95f62dde6ef750c9ff59ed8a62c',
     'ccdc1ba4f8b821ef06539696767d1270876d0781c3db07d68eea7235e5c95155',
     89, b'545,414,10,0,0,0,0,0\r\n',
     'ceb68c4451e61767b24758e1f2bc9a4c5f6167bec0729d492de83b8a7246efe0'),
)


def fixture():
    rows, parsed = [], []
    for filename, image, file_sha, line, raw, line_sha in FIXTURES:
        if hashlib.sha256(raw).hexdigest() != line_sha:
            raise AssertionError('reviewed line bytes drifted')
        rows.append(dict(stable_image_id=image, width=640, height=480,
                         annotation_relative_path='fixture/' + filename,
                         annotation_sha256=file_sha, raw_annotation_rows=line))
        parsed.append(parse_raw(b'1,1,1,1,0,11,0,0\n' * (line - 1) + raw,
                                image, expected_rows=line))
    return rows, parsed


class ProductionZeroGTTests(unittest.TestCase):
    def setUp(self):
        self.rows, self.parsed = fixture()
        self.spec = json.loads((ROOT / 'configs/supplemental_zero_gt_extension_v1.json').read_text())
        self.policy = SupplementalZeroGT.bind(self.rows, self.parsed, self.spec)

    def test_exact_lines_and_production_protocol(self):
        self.assertEqual(self.spec, specification())
        prepare.validate_parsed(self.rows, self.parsed, gt_extension=self.policy)
        self.assertEqual(len(self.policy.admitted), 2)
        for raw, expected in zip(self.parsed, FIXTURES):
            self.assertEqual(raw['rows'][-1]['physical_line'], expected[3])
            self.assertEqual(raw['rows'][-1]['raw_line_sha256'], expected[5])
        inputs = primary_inputs(self.rows, self.parsed, [])
        primary._validate_input(inputs, gt_extension=self.policy)
        for gt in inputs.ground_truth:
            if gt.area == 0:
                self.policy.validate_gt(gt)

    def test_production_reader_extension_reference(self):
        from supplemental_data import extension
        from sparse_rtdetr.localization.runio import reference
        self.assertIsNone(extension({}, self.rows, self.parsed))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'policy.json'
            path.write_text(json.dumps(self.spec), encoding='utf-8')
            contract = {'evaluation_input_extension': reference(path)}
            self.assertEqual(extension(contract, self.rows, self.parsed), self.policy)
            # Actual reader reference check must run before parsing a replacement.
            path.write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, 'immutable reference changed'):
                extension(contract, self.rows, self.parsed)

    def test_default_interfaces_remain_strict(self):
        with self.assertRaisesRegex(RuntimeError, 'GT geometry'):
            prepare.evaluator_inputs(self.rows, self.parsed, [], 600)
        with self.assertRaises(ValueError):
            primary._validate_input(primary_inputs(self.rows, self.parsed, []))
        with self.assertRaises(ValueError):
            bridge_inputs(self.rows, self.parsed, [])

    def test_empty_predictions_and_retained_denominator(self):
        for budget in (600, 300):
            proof = prepare.evaluator_inputs(self.rows, self.parsed, [], budget, gt_extension=self.policy)
            self.assertEqual(proof['counts'], dict(GT_before=1, GT_after=1,
                                                   detections_before=0, detections_after=0))
            self.assertFalse(proof['AP_scored'])
        dataset, _, _ = bridge_inputs(self.rows, self.parsed, [], gt_extension=self.policy)
        self.assertEqual(len(dataset['annotations']), 1)
        gt = dataset['annotations'][0]
        self.assertEqual((gt['category_id'], gt['area'], gt['iscrowd']), (4, 0., 0))
        self.assertEqual(gt['bbox'], [611., 158., 4., 0.])

    def test_ignore_raster_and_filter_are_unchanged(self):
        inputs = primary_inputs(self.rows, self.parsed, [])
        image = inputs.images[1]
        regions = [g for g in inputs.ground_truth if g.category_id == 0]
        integral = primary._build_integral(image, regions, gt_extension=self.policy)
        self.assertEqual(int(integral[-1, -1]), 22)
        boxes = ([545., 414., 555., 415.], [545., 413., 555., 417.], [10., 10., 12., 12.])
        fractions = [primary._ignored_fraction(integral, image, tuple(b)) for b in boxes]
        self.assertEqual(fractions, [1., .5, 0.])
        pred = [dict(image_id=image.image_id, category_id=4, score=.9 - i*.1,
                     bbox_xyxy=b) for i, b in enumerate(boxes)]
        proof = prepare.evaluator_inputs(self.rows, self.parsed, pred, 300, gt_extension=self.policy)
        self.assertEqual(proof['counts']['detections_before'], 3)
        self.assertEqual(proof['counts']['detections_after'], 1)

    def test_protocol_binding_refusals(self):
        for key, value in (('protocol', 'wrong'), ('manifest_sha256', '0'*64),
                           ('task_id', 'wrong'), ('official_commit', '0'*40),
                           ('prediction_positive_geometry', False)):
            with self.subTest(key=key):
                bad = copy.deepcopy(self.spec); bad[key] = value
                with self.assertRaises(ValueError):
                    SupplementalZeroGT.bind(self.rows, self.parsed, bad)
        bad = copy.deepcopy(self.spec); bad['allowed_rows'][0]['physical_line'] += 1
        with self.assertRaises(ValueError):
            SupplementalZeroGT.bind(self.rows, self.parsed, bad)

    def test_recorded_file_and_member_refusals(self):
        for key, value in (('annotation_relative_path', 'fixture/wrong.txt'),
                           ('annotation_sha256', '0'*64), ('stable_image_id', '0'*64)):
            with self.subTest(key=key):
                rows = copy.deepcopy(self.rows); rows[0][key] = value
                with self.assertRaises(ValueError):
                    SupplementalZeroGT.bind(rows, self.parsed, self.spec)
        with self.assertRaises(ValueError):
            SupplementalZeroGT.bind([self.rows[0]]*2, self.parsed, self.spec)
        with self.assertRaisesRegex(RuntimeError, 'extension member identity'):
            prepare.validate_parsed(self.rows[::-1], self.parsed[::-1], gt_extension=self.policy)

    def test_raw_line_identity_and_flags_refusals(self):
        for key, value in (('physical_line', 13), ('raw_line_sha256', '0'*64),
                           ('id', '0'*64), ('category', 3), ('score', 0.), ('role', 'valid_gt')):
            with self.subTest(key=key):
                parsed = copy.deepcopy(self.parsed); parsed[0]['rows'][-1][key] = value
                with self.assertRaises(ValueError):
                    SupplementalZeroGT.bind(self.rows, parsed, self.spec)

    def test_negative_nonfinite_fractional_and_area_refusals(self):
        for box, area in (([611.,158.,610.,158.],0.),
                          ([611.,158.,615.,float('nan')],0.),
                          ([611.,158.,float('inf'),158.],0.),
                          ([611.5,158.,615.5,158.],0.),
                          ([611.,158.,615.,158.],1.),
                          ([611.,158.,615.,158.],float('nan'))):
            with self.subTest(box=box, area=area):
                parsed = copy.deepcopy(self.parsed)
                parsed[0]['rows'][-1].update(box=box, raw_area=area)
                with self.assertRaises(ValueError):
                    SupplementalZeroGT.bind(self.rows, parsed, self.spec)

    def test_bound_primary_identity_area_and_flags_refusals(self):
        inputs = primary_inputs(self.rows, self.parsed, [])
        index = next(i for i, g in enumerate(inputs.ground_truth) if g.category_id == 4)
        original = inputs.ground_truth[index]
        for delta in (dict(annotation_id='wrong'), dict(image_id=inputs.images[1].image_id),
                      dict(area=1.), dict(category_id=3), dict(ignored=True), dict(ignore_region=True)):
            with self.subTest(delta=delta):
                gt = list(inputs.ground_truth); gt[index] = replace(original, **delta)
                with self.assertRaises(ValueError):
                    primary._validate_input(replace(inputs, ground_truth=tuple(gt)), gt_extension=self.policy)

    def test_positive_gt_area_conflict_refused(self):
        rows = [dict(stable_image_id='positive', width=20, height=20, raw_annotation_rows=1)]
        parsed = [parse_raw(b'2,2,3,3,1,1,0,0\n', 'positive', expected_rows=1)]
        parsed[0]['rows'][0]['raw_area'] = 10.
        with self.assertRaisesRegex(ValueError, 'area conflict'):
            SupplementalZeroGT.bind(rows, parsed, self.spec)

    def test_primary_negative_nonfinite_and_area_conflicts(self):
        rows = [dict(stable_image_id='positive', width=20, height=20, raw_annotation_rows=1)]
        parsed = [parse_raw(b'2,2,3,3,1,1,0,0\n', 'positive', expected_rows=1)]
        policy = SupplementalZeroGT.bind(rows, parsed, self.spec)
        inputs = primary_inputs(rows, parsed, [])
        for delta in (dict(bbox_xyxy=(2.,2.,1.,5.)),
                      dict(bbox_xyxy=(2.,2.,float('nan'),5.)),
                      dict(bbox_xyxy=(2.,2.,float('inf'),5.)), dict(area=10.)):
            with self.subTest(delta=delta):
                bad = replace(inputs, ground_truth=(replace(inputs.ground_truth[0], **delta),))
                for mode in ({}, {'gt_extension': policy}):
                    with self.assertRaises(ValueError):
                        primary._validate_input(bad, **mode)

    def test_policy_exact_type_required(self):
        inputs = primary_inputs(self.rows, self.parsed, [])
        with self.assertRaisesRegex(ValueError, 'exact production policy'):
            primary._validate_input(inputs, gt_extension=object())

    def test_positive_data_matches_default_interfaces(self):
        rows = [dict(stable_image_id='positive', width=20, height=20, raw_annotation_rows=1)]
        parsed = [parse_raw(b'2,2,3,3,1,1,0,0\n', 'positive', expected_rows=1)]
        policy = SupplementalZeroGT.bind(rows, parsed, self.spec)
        self.assertEqual(prepare.evaluator_inputs(rows, parsed, [], 300),
                         prepare.evaluator_inputs(rows, parsed, [], 300, gt_extension=policy))

    def test_prediction_geometry_protection_in_both_interfaces(self):
        for box in ([611.,158.,615.,158.], [5.,5.,4.,6.],
                    [1.,1.,float('nan'),3.], [1.,1.,float('inf'),3.]):
            with self.subTest(box=box):
                pred = [dict(image_id=self.rows[0]['stable_image_id'], category_id=4,
                             score=.9, bbox_xyxy=box)]
                with self.assertRaisesRegex(RuntimeError, 'prediction geometry'):
                    prepare.evaluator_inputs(self.rows, self.parsed, pred, 300, gt_extension=self.policy)
                inputs = primary_inputs(self.rows, self.parsed, pred)
                with self.assertRaises(ValueError):
                    primary._validate_input(inputs, gt_extension=self.policy)
                with self.assertRaises(ValueError):
                    bridge_inputs(self.rows, self.parsed, pred, gt_extension=self.policy)


if __name__ == '__main__':
    unittest.main()
