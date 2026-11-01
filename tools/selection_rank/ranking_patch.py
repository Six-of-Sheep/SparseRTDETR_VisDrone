"""Apply the frozen-file patches of the r17 round (three ranking arms on the r18 chain).

Three frozen chain files carry entries that cannot live in an overlay because the
overlay rule is byte-equality with a committed blob and these files must keep their
frozen base bytes everywhere else (exact anchors asserted, never applied twice,
``--revert`` supported):

1. ``criterion.py`` (4): the ranking arm declaration joins the criterion recipe, the
   final family's matched (query, ground-truth) index pairs are recorded, and ``forward``
   adds the term's ``loss_rank`` after the native losses (once per micro-batch, only when
   a term is installed).
2. ``worker.py`` (1): the native loss-family count check becomes mode-aware
   (13/22 with a term, 12/21 otherwise).
3. ``checkpoint.py`` (2): ``behavior_identity`` expects the declaration derived from the
   engine's ranking mode, so a state saved under one arm cannot be restored under another.

Everything else of this round ships as a whole-file overlay (slice_finetune, slice_ft_job,
slice_ft_execution, slice_ft_progress, the two ranking modules, the tools and the plan).

usage: ranking_patch.py <source_root>  [--revert]
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

FORWARD_OLD = "                with geo.matching(self.matcher):return super().forward(outputs,targets,**kwargs)\n"
FORWARD_NEW = (
    "                with geo.matching(self.matcher):losses=super().forward(outputs,targets,**kwargs)\n"
    "                term=getattr(self,'selection_rank',None)\n"
    "                if term is None:return losses\n"
    "                losses=dict(losses)\n"
    "                losses['loss_rank']=term.criterion_loss(targets,self.last_diagnostics)\n"
    "                return losses\n")

WORKER_OLD = ("        require(len(losses) in (12,21) and len(diag)==len(losses)//3 "
              "and all(d['denominator']>=1 for d in diag),'native loss family/denominator differs')\n")
WORKER_NEW = ("        counts=(13,22) if getattr(getattr(e,'criterion',None),'selection_rank',None) is not None else (12,21)\n"
              "        require(len(losses) in counts and len(diag)==len(losses)//3 "
              "and all(d['denominator']>=1 for d in diag),'native loss family/denominator differs')\n")

PATCHES = {
    "src/sparse_rtdetr/protocol_r18/criterion.py": [
        ("from ..localization import consumer_geometry as geo\n",
         "from ..localization import consumer_geometry as geo\n"
         "from .selection_rank_arm import NATIVE_DECLARATION as RANKING_NATIVE\n"),
        ("'loss_weight_application':'native_criterion_once; engine_sum_all'}",
         "'loss_weight_application':'native_criterion_once; engine_sum_all',"
         "'selection_rank':getattr(self,'selection_rank_declaration',RANKING_NATIVE)}"),
        ("                    'denominator':num_boxes,'matched_queries':[x.tolist() for x,y in indices],\n",
         "                    'denominator':num_boxes,'matched_queries':[x.tolist() for x,y in indices],\n"
         "                    'matched_indices':[(x.tolist(),y.tolist()) for x,y in indices],\n"),
        (FORWARD_OLD, FORWARD_NEW),
    ],
    "src/sparse_rtdetr/protocol_r18/worker.py": [(WORKER_OLD, WORKER_NEW)],
    "src/sparse_rtdetr/protocol_r18/checkpoint.py": [
        ("from .engine import assert_healthy\n",
         "from .engine import assert_healthy\nfrom . import selection_rank_arm\n"),
        ('        "loss_weight_application":"native_criterion_once; engine_sum_all"}',
         '        "loss_weight_application":"native_criterion_once; engine_sum_all",\n'
         '        "selection_rank":selection_rank_arm.declared(engine)}'),
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
