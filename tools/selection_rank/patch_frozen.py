"""Apply the frozen-file patches of the r16 round (chain entry for the ranking term).

Patches (exact anchors asserted, never applied twice, ``--revert`` supported).  Files are
relative to the staged source root; the whole set is applied to the frozen
release-003 base in one run.

1. ``criterion.py`` (4): the ranking arm declaration joins the criterion recipe, the final
   family's matched (query, ground-truth) index pairs are recorded, and ``forward`` adds
   the term's ``loss_rank`` after the native losses (once per micro-batch, only when a
   term is installed).
2. ``worker.py`` (1) and ``slice_ft_job.py`` (1): the native loss-family count check
   becomes mode-aware (13/22 with a term, 12/21 otherwise).
3. ``slice_ft_job.py`` (3): ``construct`` installs the declared ranking arm before the
   model is moved to CUDA (the same placement as the frozen detail-path module) and
   stamps ``engine.ranking_mode`` plus the execution binding.
4. ``slice_finetune.py`` (4): ``RANKING_INSTALL`` declares the arms, ``capture`` carries
   the term's own state, and ``restore`` loads it (refusing a mismatch in either
   direction).
5. ``checkpoint.py`` (2): ``behavior_identity`` expects the declaration derived from the
   engine's ranking mode, so a state saved under one arm cannot be restored under another.

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
    "src/sparse_rtdetr/protocol_r18/slice_ft_job.py": [
        (JOB_OLD, JOB_NEW),
        ("    model,criterion,_=build(config,synthetic=True)\n    mode=ft.ARM_INSTALL[arm]\n",
         "    model,criterion,_=build(config,synthetic=True)\n    mode=ft.ARM_INSTALL[arm]\n"
         "    ranking=ft.RANKING_INSTALL.get(arm)\n"
         "    if ranking is not None:\n"
         "        from . import selection_rank_arm\n"
         "        selection_rank_arm.install(model,criterion,ranking)\n"),
        ("    e=ft.FineTuneEngine(model,criterion,opt,ema,config,arm=arm,plan_sha256=c['plan_sha256'],"
         "parent_sha256=parent_sha,alt_ema=alt)\n",
         "    e=ft.FineTuneEngine(model,criterion,opt,ema,config,arm=arm,plan_sha256=c['plan_sha256'],"
         "parent_sha256=parent_sha,alt_ema=alt)\n"
         "    e.ranking_mode=ranking\n"
         "    if ranking is not None:\n"
         "        from . import selection_rank_arm\n"
         "        selection_rank_arm.assert_installed(e.model,e.criterion,ranking)\n"),
        ("        'scope':scope,'arm':arm,'install_mode':mode,'parent_sha256':parent_sha,'runtime':runtime,"
         "'CPU_fixture':fixture,\n",
         "        'scope':scope,'arm':arm,'install_mode':mode,'ranking_mode':ranking,"
         "'parent_sha256':parent_sha,'runtime':runtime,'CPU_fixture':fixture,\n"),
    ],
    "src/sparse_rtdetr/protocol_r18/slice_finetune.py": [
        ("from .model import CommonEMA\n",
         "from .model import CommonEMA\nfrom . import selection_rank_arm\n"),
        ("ARM_INSTALL = {'D0': None}\n",
         "ARM_INSTALL = {'D0': None}\nRANKING_INSTALL = {'D0': None}\n"),
        ("    value={'format':VERSION,'engine':engine.state_dict(),'behavior':behavior,\n",
         "    value={'format':VERSION,'engine':engine.state_dict(),'behavior':behavior,\n"
         "           'selection_rank':selection_rank_arm.state(engine),\n"),
        ("        engine.load_state_dict(p['engine']);native_state.restore_rng(p['rng'],engine.device)\n",
         "        engine.load_state_dict(p['engine']);native_state.restore_rng(p['rng'],engine.device)\n"
         "        selection_rank_arm.load(engine,p.get('selection_rank'))\n"),
    ],
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
