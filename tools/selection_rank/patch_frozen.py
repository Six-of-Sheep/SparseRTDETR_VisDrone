"""Apply the four minimal frozen-file patches of the r15 selection-rank round.

Patches (exact anchors asserted, never applied twice, ``--revert`` supported):

1. ``src/sparse_rtdetr/protocol_r18/criterion.py`` -- ``get_loss`` records the final
   family's matched (query, ground-truth) index pairs next to the existing vfl
   diagnostics (``matched_indices``); the ranking term maps a matched query to its
   encoder pool token through the cached top-k indices of the same forward.
2. ``src/sparse_rtdetr/protocol_r18/criterion.py`` -- ``forward`` adds the ranking
   term to the returned loss dict as ``loss_rank`` after the native losses, once per
   micro-batch, and only when a term is installed (``criterion.selection_rank``);
   without a term the returned object is exactly the native result.
3. ``src/sparse_rtdetr/protocol_r18/worker.py`` -- the native loss-family count check
   becomes mode-aware: 13/22 with an installed term, 12/21 otherwise.
4. ``src/sparse_rtdetr/protocol_r18/slice_ft_job.py`` -- the same correction in the
   fine-tune update check.

The CPU fixture ``slice_finetune_cpu.py`` keeps its native 21/12 assertion; it belongs
to the native chain and is not reused by a term-installed arm (declared, not patched).

usage: patch_frozen.py <source_root>  [--revert]
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

JOB_OLD = "    require(all(len(z) in (12,21) for z in r['microbatch_losses']),'FT native loss count')\n"
JOB_NEW = ("    counts=(13,22) if getattr(e.criterion,'selection_rank',None) is not None else (12,21)\n"
           "    require(all(len(z) in counts for z in r['microbatch_losses']),'FT native loss count')\n")

PATCHES = {
    "src/sparse_rtdetr/protocol_r18/criterion.py": [
        ("                    'denominator':num_boxes,'matched_queries':[x.tolist() for x,y in indices],\n",
         "                    'denominator':num_boxes,'matched_queries':[x.tolist() for x,y in indices],\n"
         "                    'matched_indices':[(x.tolist(),y.tolist()) for x,y in indices],\n"),
        (FORWARD_OLD, FORWARD_NEW),
    ],
    "src/sparse_rtdetr/protocol_r18/worker.py": [(WORKER_OLD, WORKER_NEW)],
    "src/sparse_rtdetr/protocol_r18/slice_ft_job.py": [(JOB_OLD, JOB_NEW)],
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
