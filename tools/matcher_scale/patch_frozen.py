"""Apply the two minimal frozen-file patches of the r12 matcher-scale round.

Patches (exact anchors asserted, never applied twice):

1. ``src/sparse_rtdetr/protocol_r18/criterion.py``
   the criterion recipe gains ``'matching': matching_declaration(self)`` so every
   saved behaviour identity carries the declared matching mode; the native
   declaration is the default, so an uninstalled criterion's recipe is unchanged.

2. ``src/sparse_rtdetr/protocol_r18/checkpoint.py``
   ``behavior_identity`` gains ``"matching": declared_matching(engine)`` in its
   expected recipe, derived from the engine's declared install mode, and refuses a
   criterion whose declaration differs from the engine's mode.  With no declared
   mode (the frozen native chain) the expected dict is byte-identical to before.

usage: patch_frozen.py <source_root>  [--revert]
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

PATCHES = {
    "src/sparse_rtdetr/protocol_r18/criterion.py": [
        ("from ..localization import consumer_geometry as geo\n",
         "from ..localization import consumer_geometry as geo\nfrom .matcher_scale import matching_declaration\n"),
        ("'loss_weight_application':'native_criterion_once; engine_sum_all'}",
         "'loss_weight_application':'native_criterion_once; engine_sum_all','matching':matching_declaration(self)}"),
    ],
    "src/sparse_rtdetr/protocol_r18/checkpoint.py": [
        ("from .engine import assert_healthy\n",
         "from .engine import assert_healthy\nfrom .matcher_scale import declaration as matching_declaration, NATIVE_DECLARATION\n"),
        ("def behavior_identity(engine):",
         "def declared_matching(engine):\n"
         "    \"\"\"The matching declaration this engine's arm declares (native when uninstalled).\n\n"
         "    Non-vacuous: the criterion's own declaration must agree with the declared arm mode,\n"
         "    so a state saved under one matching mode cannot be restored under another.\n"
         "    \"\"\"\n"
         "    expected=matching_declaration(getattr(engine,'install_mode',None))\n"
         "    actual=dict(getattr(engine.criterion,'matching_declaration',NATIVE_DECLARATION))\n"
         "    if actual!=expected:\n"
         "        raise ValueError('criterion matching declaration differs from the engine install mode')\n"
         "    return expected\n\n\n"
         "def behavior_identity(engine):"),
        ('"loss_weight_application":"native_criterion_once; engine_sum_all"}',
         '"loss_weight_application":"native_criterion_once; engine_sum_all",\n        "matching":declared_matching(engine)}'),
    ],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path(sys.argv[1]).resolve()
    revert = "--revert" in sys.argv
    report = {}
    for relative, pairs in PATCHES.items():
        path = root / relative
        text = path.read_text(encoding="utf-8")
        before = sha(path)
        for old, new in pairs:
            if revert:
                old, new = new, old
            count = text.count(old)
            if count != 1:
                raise SystemExit("anchor count %d != 1 in %s: %r" % (count, relative, old[:60]))
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8")
        report[relative] = {"sha256_before": before, "sha256_after": sha(path),
                            "direction": "revert" if revert else "apply"}
    import json
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
