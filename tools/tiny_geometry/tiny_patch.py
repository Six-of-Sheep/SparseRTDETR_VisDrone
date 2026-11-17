"""Apply the frozen-file patches of the r18 round (tiny-band geometry on the r18 chain).

Two frozen chain files carry entries that cannot live in an overlay because the overlay
rule is byte-equality with a committed blob and these files must keep their frozen base
bytes everywhere else (exact anchors asserted, never applied twice, ``--revert``
supported):

1. ``criterion.py`` (5): the geometry arm declaration joins the criterion recipe; the
   wrapper carries a per-forward observation sink whose rows merge into the frozen
   diagnostics; ``loss_boxes`` observes every non-denoising family call in the native arms
   and delegates to the installed term otherwise, while the denoising families keep their
   native arithmetic untouched.
2. ``checkpoint.py`` (2): ``behavior_identity`` expects the declaration derived from the
   engine's geometry mode, so a state saved under one arm cannot be restored under another.

Everything else of this round ships as a whole-file overlay (slice_finetune, slice_ft_job,
slice_ft_execution, slice_ft_progress, the two geometry modules, the tools and the plan).

usage: tiny_patch.py <source_root>  [--revert]
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

FORWARD_OLD = "                with geo.matching(self.matcher):return super().forward(outputs,targets,**kwargs)\n"
FORWARD_NEW = (
    "                with geo.matching(self.matcher):losses=super().forward(outputs,targets,**kwargs)\n"
    "                for row in self.last_diagnostics:\n"
    "                    extra=self._geo_stats.pop(row['head'],None)\n"
    "                    if extra is not None:row.update(extra)\n"
    "                return losses\n")

BOXES_OLD = "            return super().loss_boxes(outputs,targets,indices,num_boxes,boxes_weight)\n"
BOXES_NEW = (
    "            term=getattr(self,'tiny_geometry',None)\n"
    "            dn=id(outputs) in getattr(self,'_dn_ids',set())\n"
    "            if term is None or dn:\n"
    "                losses=super().loss_boxes(outputs,targets,indices,num_boxes,boxes_weight)\n"
    "                if not dn:\n"
    "                    self._geo_stats[self._families.get(id(outputs),'final')]=tiny_geometry_arm.observe(outputs['pred_boxes'][idx],gt,module)\n"
    "                return losses\n"
    "            return term.loss_boxes(super().loss_boxes,module,outputs,targets,indices,num_boxes,boxes_weight,\n"
    "                                   family=self._families.get(id(outputs),'final'),sink=self._geo_stats)\n")

PATCHES = {
    "src/sparse_rtdetr/protocol_r18/criterion.py": [
        ("from ..localization import consumer_geometry as geo\n",
         "from ..localization import consumer_geometry as geo\n"
         "from .tiny_geometry_arm import NATIVE_DECLARATION as GEOMETRY_NATIVE\n"
         "from . import tiny_geometry_arm\n"),
        ("'loss_weight_application':'native_criterion_once; engine_sum_all'}",
         "'loss_weight_application':'native_criterion_once; engine_sum_all',"
         "'tiny_geometry':getattr(self,'tiny_geometry_declaration',GEOMETRY_NATIVE)}"),
        ("            self.last_diagnostics=[]\n",
         "            self.last_diagnostics=[]\n            self._geo_stats={}\n"),
        (FORWARD_OLD, FORWARD_NEW),
        (BOXES_OLD, BOXES_NEW),
    ],
    "src/sparse_rtdetr/protocol_r18/checkpoint.py": [
        ("from .engine import assert_healthy\n",
         "from .engine import assert_healthy\nfrom . import tiny_geometry_arm\n"),
        ('        "loss_weight_application":"native_criterion_once; engine_sum_all"}',
         '        "loss_weight_application":"native_criterion_once; engine_sum_all",\n'
         '        "tiny_geometry":tiny_geometry_arm.declared(engine)}'),
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
                raise SystemExit("anchor count %d != 1 in %s: %r" % (count, relative, old[:70]))
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8")
        report[relative] = {"sha256_before": before, "sha256_after": sha(path),
                            "direction": "revert" if revert else "apply", "patches": len(pairs)}
    import json
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
