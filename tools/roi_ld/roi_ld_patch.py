"""Install the ROI-ld term into the frozen protocol_r18 sources by anchored text substitution.

Every replacement is anchored on a string that must occur exactly once, so a changed source
raises instead of silently patching the wrong place.  ``--revert`` restores the original bytes.
The three targets are the same three the closed tiny-positives round patched, so the two
mechanisms are installed by the same convention.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

FORWARD_OLD = (
    "                for row in self.last_diagnostics:\n"
    "                    extra=self._pos_stats.pop(row['head'],None)\n"
    "                    if extra is not None:row.update(extra)\n")
FORWARD_NEW = (
    "                for row in self.last_diagnostics:\n"
    "                    extra=self._pos_stats.pop(row['head'],None)\n"
    "                    if extra is not None:row.update(extra)\n"
    "                    extra=self._roi_stats.pop(row['head'],None)\n"
    "                    if extra is not None:row.update(extra)\n")

BOXES_OLD = "            term=getattr(self,'tiny_positives',None)\n"
BOXES_NEW = (
    "            term=getattr(self,'tiny_positives',None)\n"
    "            roi=getattr(self,'roi_ld',None)\n"
    "            if roi is not None and not (id(outputs) in getattr(self,'_dn_ids',set())):\n"
    "                return roi.loss_boxes(super().loss_boxes,module,outputs,targets,indices,num_boxes,boxes_weight,\n"
    "                                      family=self._families.get(id(outputs),'final'),sink=self._roi_stats)\n")

DATASET_INIT_OLD = (
    "class Dataset(RawRoleDataset):\n"
    "    def __init__(self, arm, **kwargs):\n"
    "        super().__init__(role='train_core', seed=0, **kwargs)\n"
    "        self.ft_transform = Transform(arm)\n")
DATASET_INIT_NEW = (
    "class Dataset(RawRoleDataset):\n"
    "    def __init__(self, arm, **kwargs):\n"
    "        from . import roi_ld_data\n"
    "        super().__init__(role='train_core', seed=0, **kwargs)\n"
    "        self.ft_transform = Transform(arm)\n"
    "        self.roi_mode = ROI_INSTALL[arm]\n"
    "        self.roi_cache = None if self.roi_mode is None else roi_ld_data.frozen_cache()\n")

DATASET_GET_OLD = (
    "        trace.update(image_sha256=row['image_sha256'],annotation_sha256=row['annotation_sha256'],\n"
    "                     spawn_observer={'pid':os.getpid(),'CUDA_initialized':torch.cuda.is_initialized()})\n"
    "        return x,t,trace\n")
DATASET_GET_NEW = (
    "        trace.update(image_sha256=row['image_sha256'],annotation_sha256=row['annotation_sha256'],\n"
    "                     spawn_observer={'pid':os.getpid(),'CUDA_initialized':torch.cuda.is_initialized()})\n"
    "        if self.roi_mode is not None:\n"
    "            from . import roi_ld_data\n"
    "            t=roi_ld_data.attach(t,trace,raw,row,self.roi_mode,self.roi_cache)\n"
    "            trace['roi_mode']=self.roi_mode\n"
    "        return x,t,trace\n")

POS_INSTALL_OLD = "POS_INSTALL = {'A0': None, 'A1': 'A1', 'A2': 'A2'}\n"
POS_INSTALL_NEW = (
    "POS_INSTALL = {'A0': None, 'A1': 'A1', 'A2': 'A2'}\n"
    "# ROI-resolution box distillation: A1 feeds the whole-image view (no new information),\n"
    "# A2 feeds the four raw-pixel tiles (the method).  A0 installs nothing.\n"
    "ROI_INSTALL = {'A0': None, 'A1': 'R1', 'A2': 'R2'}\n")

PATCHES = {
    "src/sparse_rtdetr/protocol_r18/criterion.py": [
        ("from . import tiny_positives_arm\n",
         "from . import tiny_positives_arm\n"
         "from .roi_ld_arm import NATIVE_DECLARATION as ROI_NATIVE\n"
         "from . import roi_ld_arm\n"),
        ("'tiny_positives':getattr(self,'tiny_positives_declaration',POSITIVES_NATIVE)}",
         "'tiny_positives':getattr(self,'tiny_positives_declaration',POSITIVES_NATIVE),"
         "'roi_ld':getattr(self,'roi_ld_declaration',ROI_NATIVE)}"),
        ("            self.last_diagnostics=[]\n            self._pos_stats={}\n",
         "            self.last_diagnostics=[]\n            self._pos_stats={}\n            self._roi_stats={}\n"),
        (FORWARD_OLD, FORWARD_NEW),
        (BOXES_OLD, BOXES_NEW),
    ],
    "src/sparse_rtdetr/protocol_r18/checkpoint.py": [
        ("from . import tiny_positives_arm\n",
         "from . import tiny_positives_arm\nfrom . import roi_ld_arm\n"),
        ('        "tiny_positives":tiny_positives_arm.declared(engine)}',
         '        "tiny_positives":tiny_positives_arm.declared(engine),\n'
         '        "roi_ld":roi_ld_arm.declared(engine)}'),
    ],
    "src/sparse_rtdetr/protocol_r18/slice_finetune.py": [
        (POS_INSTALL_OLD, POS_INSTALL_NEW),
        (DATASET_INIT_OLD, DATASET_INIT_NEW),
        (DATASET_GET_OLD, DATASET_GET_NEW),
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
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
