"""pytest mirror of the PDRA CPU self-checks (constructed inputs only).

Run from the repository root:  python3 -m pytest tools/pdra/test_pdra_core.py -q
"""

from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (_HERE, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pdra_selfcheck as S


def test_coordinate_mapping():
    assert S.check_coordinate_mapping()["status"] == "PASS"


def test_input_swap():
    assert S.check_input_swap()["status"] == "PASS"


def test_empty_difference_regions():
    assert S.check_empty_difference_regions()["status"] == "PASS"


def test_abstention():
    assert S.check_abstention()["status"] == "PASS"


def test_cyclic_preference():
    assert S.check_cyclic_preference()["status"] == "PASS"


def test_near_neighbor_distinct_targets():
    assert S.check_near_neighbor_distinct_targets()["status"] == "PASS"


def test_original_c_fallback():
    assert S.check_original_c_fallback()["status"] == "PASS"


def test_control_fairness():
    assert S.check_control_fairness()["status"] == "PASS"


def test_label_ambiguity():
    assert S.check_label_ambiguity()["status"] == "PASS"


def test_frozen_core_consistency():
    assert S.check_frozen_core_consistency()["status"] == "PASS"


@pytest.mark.parametrize("name", ["feature_stride_is_placeholder_bound_at_stage1"])
def test_frozen_constants_are_recorded(name):
    from pdra_core import CANVAS_HW, DELTA_PREF, DELTA_ABSTAIN, THETA_REGION
    assert CANVAS_HW == (768, 1344)
    assert 0.0 < DELTA_PREF < 0.5 and 0.0 < DELTA_ABSTAIN <= 1.0 and 0.0 < THETA_REGION <= 1.0
