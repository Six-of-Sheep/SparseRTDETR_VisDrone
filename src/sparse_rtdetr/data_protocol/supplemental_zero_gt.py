"""Explicit, finite GT exception for two authenticated supplemental raw rows.

This is an evaluation INPUT protocol extension, not a general degenerate-box
policy. No detector geometry function or process-global module is replaced.
"""
from dataclasses import dataclass
import math
from pathlib import PurePosixPath

from .schema import stable_annotation_id

VERSION = 'supplemental-zero-gt-input-extension-v1'
MANIFEST = '1af7eb4248be3097122d2168e18519c8db3ac11e17faff46aca8e3d78e5cfef4'
TASK = 'sahi-original-c-supplemental-20261018-001'
ROWS = (
    ('9999985_00000_d_0000020.txt', 'ed0c75ca472b4571f42ee5d0ac61e8681a81dcdf6d9daf3307d334cfb01c6546',
     '999d84a9fcc6632645ffb9b2eca416f4e2d08a966d811d24277960e593f8c814', 12,
     '620332c35ca9ce25489b90f61cd656bbcc26181d7f07aec513f63a8b12932a1e', (611., 158., 615., 158.), 4, 1.),
    ('9999999_00590_d_0000267.txt', '4b1f0ecb0197dbc9575118fc6947383202dde95f62dde6ef750c9ff59ed8a62c',
     'ccdc1ba4f8b821ef06539696767d1270876d0781c3db07d68eea7235e5c95155', 89,
     'ceb68c4451e61767b24758e1f2bc9a4c5f6167bec0729d492de83b8a7246efe0', (545., 414., 555., 414.), 0, 0.),
)


def need(ok, message):
    if not ok:
        raise ValueError('supplemental GT extension: ' + message)


def specification():
    return dict(protocol=VERSION, task_id=TASK, manifest_sha256=MANIFEST,
                official_commit='005445782213e20cb91bc50a597db3dd949e749a',
                raw_rows_unchanged=True, prediction_positive_geometry=True,
                independent_qualification='BLOCKED_PENDING_ACCESS_HISTORY',
                allowed_rows=[dict(file=r[0], image_id=r[1], file_sha256=r[2], physical_line=r[3],
                                   raw_line_sha256=r[4], box=list(r[5]), category=r[6], score=r[7]) for r in ROWS])


@dataclass(frozen=True)
class SupplementalZeroGT:
    members: tuple
    admitted: frozenset

    @classmethod
    def bind(cls, rows, parsed, spec):
        need(spec == specification(), 'version/manifest/exception specification')
        need(len(rows) == len(parsed) and len({r['stable_image_id'] for r in rows}) == len(rows), 'member binding')
        admitted = set()
        for row, raw in zip(rows, parsed):
            for obj in raw['rows']:
                b = obj['box']
                need(len(b) == 4 and all(type(v) in (int, float) and math.isfinite(v) for v in b), 'nonfinite GT')
                need(b[2] >= b[0] and b[3] >= b[1], 'negative GT extent')
                need(type(obj['raw_area']) in (int, float) and math.isfinite(obj['raw_area']) and
                     obj['raw_area'] == (b[2]-b[0])*(b[3]-b[1]), 'GT original area conflict')
                if b[2] > b[0] and b[3] > b[1]:
                    continue
                candidates = [r for r in ROWS if r[1] == row['stable_image_id'] and r[3] == obj['physical_line']]
                need(len(candidates) == 1, 'unreviewed zero-area GT identity')
                r = candidates[0]
                need(PurePosixPath(row['annotation_relative_path']).name == r[0] and row['annotation_sha256'] == r[2], 'original file identity')
                need(obj['raw_line_sha256'] == r[4] and tuple(b) == r[5] and
                     obj['category'] == r[6] and obj['score'] == r[7] and obj['raw_area'] == 0 and
                     obj['id'] == stable_annotation_id(r[1], r[3], r[4]) and obj['role'] == 'nonpositive_area',
                     'reviewed raw row changed')
                need(all(float(x).is_integer() for x in b), 'fractional zero-area GT forbidden')
                admitted.add(obj['id'])
        return cls(tuple(r['stable_image_id'] for r in rows), frozenset(admitted))

    def validate_gt(self, gt):
        expected = [r for r in ROWS if stable_annotation_id(r[1], r[3], r[4]) == gt.annotation_id]
        need(gt.annotation_id in self.admitted and len(expected) == 1, 'GT not admitted from authenticated raw rows')
        r = expected[0]
        need(gt.image_id == r[1] and gt.bbox_xyxy == r[5] and gt.area == 0 and
             gt.category_id == r[6] and gt.ignore_region is (r[6] == 0) and gt.ignored is (r[7] == 0),
             'GT identity/geometry/flags conflict')

    def zero_xywh(self, box):
        need(any(tuple(box) == r[5] and stable_annotation_id(r[1], r[3], r[4]) in self.admitted for r in ROWS),
             'unreviewed zero geometry')
        return box[0], box[1], box[2]-box[0], box[3]-box[1]


def checked(policy):
    need(type(policy) is SupplementalZeroGT, 'explicit exact production policy required')
    return policy
