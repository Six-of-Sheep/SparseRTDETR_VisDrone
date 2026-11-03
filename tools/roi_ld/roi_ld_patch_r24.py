"""Install the r24 roi-ld round into the frozen release-008 protocol_r18 sources.

Anchored text substitution: every anchor must occur exactly once or the script aborts, and
``--revert`` restores the original bytes.  Three files are patched, the same three the closed r19
round patched, so the two mechanisms are installed by one convention.

1. ``criterion.py`` (5): the roi-ld arm declaration joins the criterion recipe; the wrapper
   carries a second per-forward observation sink for the roi statistics; ``loss_boxes`` hands
   every non-denoising family to the installed roi term and, on the native arm, runs the
   unchanged native arithmetic and records the roi band statistics with ``roi_selected = 0``.
   The denoising families keep their native arithmetic untouched.
2. ``checkpoint.py`` (2): ``behavior_identity`` expects the roi declaration derived from the
   engine's own criterion, so a state saved under one armed mode cannot be restored under the
   other (the declaration is part of the behaviour identity).
3. ``slice_finetune.py`` (8): the round version; the retired tiny-positives install map plus the
   roi-ld arm map; the roi imports; the r24 science-plan reader; the two dataset hooks that
   attach the frozen-cache roi boxes; and the checkpoint capture/restore of the roi term state.

usage: roi_ld_patch_r24.py <source_root> [--revert]
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

FORWARD_OLD = ("                for row in self.last_diagnostics:\n"
               "                    extra=self._pos_stats.pop(row['head'],None)\n"
               "                    if extra is not None:row.update(extra)\n")
FORWARD_NEW = (FORWARD_OLD
               + "                    extra=self._roi_stats.pop(row['head'],None)\n"
               + "                    if extra is not None:row.update(extra)\n")

BOXES_OLD = (
    "            term=getattr(self,'tiny_positives',None)\n"
    "            dn=id(outputs) in getattr(self,'_dn_ids',set())\n"
    "            if term is None or dn:\n"
    "                losses=super().loss_boxes(outputs,targets,indices,num_boxes,boxes_weight)\n"
    "                if not dn:\n"
    "                    self._pos_stats[self._families.get(id(outputs),'final')]=tiny_positives_arm.observe_call(\n"
    "                        outputs['pred_boxes'][idx],gt,module,band_gt_count=tiny_positives_arm.band_gt(targets))\n"
    "                return losses\n"
    "            return term.loss_boxes(super().loss_boxes,module,outputs,targets,indices,num_boxes,boxes_weight,\n"
    "                                   family=self._families.get(id(outputs),'final'),sink=self._pos_stats)\n")
BOXES_NEW = (
    "            term=getattr(self,'tiny_positives',None)\n"
    "            roi=getattr(self,'roi_ld',None)\n"
    "            family=self._families.get(id(outputs),'final')\n"
    "            dn=id(outputs) in getattr(self,'_dn_ids',set())\n"
    "            if roi is not None and not dn:\n"
    "                return roi.loss_boxes(super().loss_boxes,module,outputs,targets,indices,num_boxes,boxes_weight,\n"
    "                                      family=family,sink=self._roi_stats)\n"
    "            if term is None or dn:\n"
    "                losses=super().loss_boxes(outputs,targets,indices,num_boxes,boxes_weight)\n"
    "                if not dn:\n"
    "                    self._pos_stats[family]=tiny_positives_arm.observe_call(\n"
    "                        outputs['pred_boxes'][idx],gt,module,band_gt_count=tiny_positives_arm.band_gt(targets))\n"
    "                    # r24 native-arm observation: the band statistics are recorded with\n"
    "                    # roi_selected = 0; no term is installed and no pair is appended.\n"
    "                    self._roi_stats[family]=roi_ld.observe_call(\n"
    "                        outputs['pred_boxes'][idx],gt,module,band_gt_count=roi_ld.band_gt(targets))\n"
    "                return losses\n"
    "            return term.loss_boxes(super().loss_boxes,module,outputs,targets,indices,num_boxes,boxes_weight,\n"
    "                                   family=family,sink=self._pos_stats)\n")

PLAN_OLD = """def read_plan(path):
    p = json.loads(Path(path).read_text(encoding='utf-8'))
    if p['version'] != VERSION or p['arms'] != list(ARMS) or p['updates_per_arm'] != UPDATES:
        raise ValueError('different fixed fine-tune plan')
    if p['GPU_authorized'] is not False or p['parent']['checkpoint_sha256'] != PARENT_SHA:
        raise ValueError('preparation authority or parent differs')
    if p.get('arm_install') != {a: ARM_INSTALL[a] for a in ARMS}:
        raise ValueError('arm install map differs')
    if p.get('pos_install') != {a: POS_INSTALL[a] for a in ARMS}:
        raise ValueError('positives install map differs')
    declared = p.get('tiny_positives') or {}
    if (declared.get('objective') != tiny_positives_arm.OBJECTIVE
            or float(declared.get('band_area_px') or -1.0) != float(tiny_positives_arm.BAND_AREA_PX)
            or float(declared.get('iou_tau') or -1.0) != float(tiny_positives_arm.IOU_TAU)
            or int(declared.get('extras_per_gt') or -1) != int(tiny_positives_arm.EXTRAS_PER_GT)
            or declared.get('constants_sha256') != tiny_positives_arm.constants_sha256()):
        raise ValueError('declared tiny-positives constants differ')
    gates = p.get('gates') or {}
    if not isinstance(gates, dict) or not all(k in gates for k in
            ('A2_minus_A0', 'A2_minus_A1', 'combination')):
        raise ValueError('declared gates differ')
    if ((gates['A2_minus_A0'].get('lhs'), gates['A2_minus_A0'].get('rhs')) != ('A2', 'A0')
            or (gates['A2_minus_A1'].get('lhs'), gates['A2_minus_A1'].get('rhs')) != ('A2', 'A1')):
        raise ValueError('gate identity differs')
    if ((p.get('budget') or {}).get('gpu_hours') != 8.0 or (p.get('budget') or {}).get('wall_hours') != 24
            or (p.get('budget') or {}).get('server_new_bytes') != 25769803776):
        raise ValueError('declared round budget differs')
    if list(p.get('checkpoint_passes', ())) != list(CHECKPOINT_PASSES):
        raise ValueError('declared checkpoint cadence differs')
    if (p['ema_r'].get('updates_start') != 0 or p['ema_r'].get('decay') != .9999
            or p['ema_c'].get('updates_start') != ALT_OFFSET or p['ema_c'].get('decay') != .9999
            or p['ema_c'].get('warmups') != 2000):
        raise ValueError('dual-EMA clock/recipe declaration differs')
    return p
"""

PLAN_NEW = """def read_plan(path):
    p = json.loads(Path(path).read_text(encoding='utf-8'))
    if p['version'] != VERSION or p['arms'] != list(ARMS) or p['schedule']['updates_per_arm'] != UPDATES:
        raise ValueError('different fixed fine-tune plan')
    if p['parent']['checkpoint_sha256'] != PARENT_SHA or p['parent'].get('ema_only') is not True:
        raise ValueError('frozen parent declaration differs')
    if p.get('arm_install') != {a: ARM_INSTALL[a] for a in ARMS}:
        raise ValueError('arm install map differs')
    if p.get('pos_install') != {a: POS_INSTALL[a] for a in ARMS}:
        raise ValueError('the retired tiny-positives install map differs')
    if p.get('roi_install') != {a: ROI_INSTALL[a] for a in ARMS}:
        raise ValueError('roi-ld install map differs')
    if int(p['schedule'].get('passes_per_arm') or -1) != PASSES:
        raise ValueError('declared pass count differs')
    constants = (p.get('mechanism') or {}).get('constants') or {}
    if ([int(v) for v in (constants.get('canvas_hw') or [])] != list(roi_ld.CANVAS_HW)
            or float(constants.get('band_area_px') or -1.0) != float(roi_ld.BAND_AREA_PX)
            or float(constants.get('target_iou_tau') or -1.0) != float(roi_ld.TARGET_IOU)
            or int(constants.get('targets_per_gt') or -1) != int(roi_ld.TARGETS_PER_GT)
            or float(constants.get('prefilter_iou_original_coordinates') or -1.0) != float(roi_ld_data.PREFILTER_IOU)):
        raise ValueError('declared roi-ld constants differ')
    cache = (p.get('mechanism') or {}).get('teacher_cache') or {}
    if (cache.get('path') != roi_ld_data.FROZEN_INDEX['path']
            or cache.get('sha256') != roi_ld_data.FROZEN_INDEX['sha256']
            or int(cache.get('bytes') or -1) != int(roi_ld_data.FROZEN_INDEX['bytes'])):
        raise ValueError('declared teacher cache differs')
    gates = p.get('gates') or {}
    if not isinstance(gates, dict) or not all(k in gates for k in
            ('A2_minus_A0', 'A2_minus_A1', 'cost', 'combination')):
        raise ValueError('declared gates differ')
    if ((gates['A2_minus_A0'].get('lhs'), gates['A2_minus_A0'].get('rhs')) != ('A2', 'A0')
            or (gates['A2_minus_A1'].get('lhs'), gates['A2_minus_A1'].get('rhs')) != ('A2', 'A1')):
        raise ValueError('gate identity differs')
    if (gates['cost'].get('metric') != 'inference_stage_seconds'
            or (gates['cost'].get('lhs'), gates['cost'].get('rhs')) != ('A2', 'A0')
            or abs(float(gates['cost'].get('threshold')) - 1.05) > 1e-12):
        raise ValueError('cost gate/semantics differ')
    if ((p.get('budget') or {}).get('gpu_hours') != 8.0 or (p.get('budget') or {}).get('wall_hours') != 24
            or (p.get('budget') or {}).get('server_new_bytes') != 25769803776):
        raise ValueError('declared round budget differs')
    if list(p.get('checkpoint_passes', ())) != list(CHECKPOINT_PASSES):
        raise ValueError('declared checkpoint cadence differs')
    if (p['schedule']['ema_r'].get('updates_start') != 0 or p['schedule']['ema_r'].get('decay') != .9999
            or p['schedule']['ema_c'].get('updates_start') != ALT_OFFSET or p['schedule']['ema_c'].get('decay') != .9999
            or p['schedule']['ema_c'].get('warmups') != 2000
            or p['schedule'].get('primary_endpoint') != 'FT012_EMA_C'):
        raise ValueError('dual-EMA clock/recipe declaration differs')
    definition = p.get('evaluation_definition') or {}
    if (int(definition.get('scorings') or -1) != 6 or definition.get('ema') != 'C'
            or definition.get('ema_r') != 'state_only'
            or definition.get('trained_A0_required_equal_to_parent') is not False):
        raise ValueError('evaluation definition differs')
    policy = p.get('cost_policy') or {}
    if policy.get('predeclared_ratio') is not None or policy.get('ratio_source') != 'measured_from_the_inference_stage':
        raise ValueError('cost policy predeclares an inference ratio')
    return p
"""

PATCHES = {
    "src/sparse_rtdetr/protocol_r18/criterion.py": [
        ("from .tiny_positives_arm import NATIVE_DECLARATION as POSITIVES_NATIVE\n"
         "from . import tiny_positives_arm\n",
         "from .tiny_positives_arm import NATIVE_DECLARATION as POSITIVES_NATIVE\n"
         "from . import tiny_positives_arm\n"
         "from .roi_ld_arm import NATIVE_DECLARATION as ROI_NATIVE\n"
         "from . import roi_ld_arm\n"
         "from . import roi_ld\n"),
        ("'tiny_positives':getattr(self,'tiny_positives_declaration',POSITIVES_NATIVE)}",
         "'tiny_positives':getattr(self,'tiny_positives_declaration',POSITIVES_NATIVE),"
         "'roi_ld':getattr(self,'roi_ld_declaration',ROI_NATIVE)}"),
        ("            self.last_diagnostics=[]\n            self._pos_stats={}\n",
         "            self.last_diagnostics=[]\n            self._pos_stats={}\n            self._roi_stats={}\n"),
        (FORWARD_OLD, FORWARD_NEW),
        (BOXES_OLD, BOXES_NEW),
    ],
    "src/sparse_rtdetr/protocol_r18/checkpoint.py": [
        ("from .engine import assert_healthy\n",
         "from .engine import assert_healthy\nfrom . import roi_ld_arm\n"),
        ('        "tiny_positives":tiny_positives_arm.declared(engine)}',
         '        "tiny_positives":tiny_positives_arm.declared(engine),\n'
         '        "roi_ld":roi_ld_arm.declared(engine)}'),
    ],
    "src/sparse_rtdetr/protocol_r18/slice_finetune.py": [
        ("VERSION = 'r18-tiny-positives-v2'", "VERSION = 'r24-roi-ld-v1'"),
        ("from . import tiny_positives_arm\n",
         "from . import tiny_positives_arm\nfrom . import roi_ld_arm\nfrom . import roi_ld_data\n"),
        ("ARM_INSTALL = {'A0': None, 'A1': None, 'A2': None}\n"
         "POS_INSTALL = {'A0': None, 'A1': 'A1', 'A2': 'A2'}\n",
         "ARM_INSTALL = {'A0': None, 'A1': None, 'A2': None}\n"
         "# r24: the r19 tiny-positives item is RETIRED.  Its modules stay present and importable,\n"
         "# but no arm installs it, so every arm runs the unchanged native regression on top.\n"
         "POS_INSTALL = {'A0': None, 'A1': None, 'A2': None}\n"
         "# Band ROI-resolution box distillation: A1 feeds the whole-image view (no new information),\n"
         "# A2 feeds the four raw-pixel tiles (the method).  A0 installs nothing.\n"
         "ROI_INSTALL = {'A0': None, 'A1': 'R1', 'A2': 'R2'}\n"),
        (PLAN_OLD, PLAN_NEW),
        ("class Dataset(RawRoleDataset):\n"
         "    def __init__(self, arm, **kwargs):\n"
         "        super().__init__(role='train_core', seed=0, **kwargs)\n"
         "        self.ft_transform = Transform(arm)\n",
         "class Dataset(RawRoleDataset):\n"
         "    def __init__(self, arm, **kwargs):\n"
         "        super().__init__(role='train_core', seed=0, **kwargs)\n"
         "        self.ft_transform = Transform(arm)\n"
         "        self.roi_mode = ROI_INSTALL[arm]\n"
         "        self.roi_cache = None if self.roi_mode is None else roi_ld_data.frozen_cache()\n"),
        ("                     spawn_observer={'pid':os.getpid(),'CUDA_initialized':torch.cuda.is_initialized()})\n"
         "        return x,t,trace\n",
         "                     spawn_observer={'pid':os.getpid(),'CUDA_initialized':torch.cuda.is_initialized()})\n"
         "        if self.roi_mode is not None:\n"
         "            t=roi_ld_data.attach(t,trace,raw,row,self.roi_mode,self.roi_cache)\n"
         "            trace['roi_mode']=self.roi_mode\n"
         "        return x,t,trace\n"),
        ("           'tiny_positives':tiny_positives_arm.state(engine),\n",
         "           'tiny_positives':tiny_positives_arm.state(engine),\n"
         "           'roi_ld':roi_ld_arm.state(engine),\n"),
        ("        tiny_positives_arm.load(engine,p.get('tiny_positives'))\n",
         "        tiny_positives_arm.load(engine,p.get('tiny_positives'))\n"
         "        roi_ld_arm.load(engine,p.get('roi_ld'))\n"),
    ],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def apply(root, revert=False):
    """Patch the three release-source files in place; returns a per-file receipt."""
    root = Path(root).resolve()
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
                            "direction": "revert" if revert else "apply", "anchors": len(pairs)}
    return report


def main():
    root = Path(sys.argv[1]).resolve()
    report = apply(root, revert="--revert" in sys.argv)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
