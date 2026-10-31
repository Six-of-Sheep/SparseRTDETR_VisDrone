"""pytest mirror of the torch wiring self-checks.

Skips cleanly when torch/torchvision are absent (the CI environment installs numpy
only), so it is safe to collect in the repository check matrix.

Run from the repository root:  python -m pytest tools/pdra/test_pdra_torch.py -q
"""

from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _path in (_HERE, _REPO):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def test_all_torch_checks():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    import pdra_torch_selfcheck as T
    results = [check() for check in T.CHECKS]
    failed = [r for r in results if r["status"] != "PASS"]
    assert not failed, failed
