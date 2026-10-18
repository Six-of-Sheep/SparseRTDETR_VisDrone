"""Targeted actual dual-input interface check; synthetic predictions, no AP."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
from zero_area_gt_review import ROOT, exact, ref, require, fingerprint, isolated_adapter


def main():
    p=argparse.ArgumentParser();p.add_argument('--review-gate',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU visibility')
    import torch
    torch.set_num_threads(1);require(not torch.cuda.is_initialized(),'CUDA initialized')
    def forbidden(*args,**kwargs):raise RuntimeError('interface check forbids model/CUDA/AP')
    torch.nn.Module.__init__=forbidden;torch.cuda._lazy_init=forbidden;torch.autograd.backward=forbidden
    from sparse_rtdetr.r0 import evaluation
    from sparse_rtdetr.baseline import primary_evaluator as primary
    from sparse_rtdetr.localization.data import parse_raw,read_bound
    from pycocotools import _mask
    from confirmation_prepare import evaluator_inputs
    evaluation.score_budget=forbidden;evaluation.bridge_metrics=forbidden
    gate=json.loads(Path(a.review_gate).read_bytes());require(gate['status']=='REVIEW_CPU_PASS_PRODUCTION_REMAINS_STOP','prior review gate')
    mr=gate['development']['manifest'];rows=exact(mr)['records']
    # Resolve the roots from the same authenticated metadata parent as review.
    parent=Path('/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace/experiments/localization-distribution/sahi-sixth-view-20261018-001/execution-003/execution-contract.json')
    require(ref(parent)['sha256']=='3e494a8e5171eda5b6d2d8dba3f3893cb5dbcd8e81816e2d5e2118a70a38e793','parent SHA')
    spec=json.loads(parent.read_bytes())['parents'][0]['construction']['data']['development']
    parsed=[];preds=[];start=time.monotonic()
    for r in rows:
        ap=Path(spec['roots']['annotations'])/Path(r['annotation_relative_path']).name
        parsed.append(parse_raw(read_bound(ap,r['annotation_sha256'],r['annotation_size_bytes']),r['stable_image_id'],expected_rows=r['raw_annotation_rows']))
        preds.append(dict(image_id=r['stable_image_id'],category_id=4,score=.9,bbox_xyxy=[1.,1.,2.,2.]))
    proofs={str(b):evaluator_inputs(rows,parsed,preds,b) for b in (600,300)}
    require(proofs['600']['canonical_input_sha256']==proofs['300']['canonical_input_sha256']==gate['development']['primary_input_sha256'],'actual primary input SHA differs')
    require(all(v['counts']==gate['development']['filter_counts'] for v in proofs.values()),'actual bridge filter counts differ')
    adapter,bridge=isolated_adapter(primary,evaluation)
    for budget in (600,300):
        require(evaluation.primary_inputs(rows,parsed,preds)==adapter._validate_input(evaluation.primary_inputs(rows,parsed,preds)),'primary interface differs')
        require(evaluation.bridge_inputs(rows,parsed,preds)==bridge(rows,parsed,preds),'bridge interface differs')
    require(not torch.cuda.is_initialized(),'CUDA initialized')
    result=dict(status='PASS',entry='confirmation_prepare.evaluator_inputs -> original primary_inputs/bridge_inputs',members=548,budgets=[600,300],proofs=proofs,prior_review_gate=ref(a.review_gate),compiled_COCO_mask=ref(_mask.__file__),original_dataset_AP_recomputed=False,model_predictions_read=False,production_changed=False,GPU_forward_calls=0,scientific_groups=0,elapsed_seconds=time.monotonic()-start)
    out=Path(a.output);require(not out.exists(),'new output required')
    out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(dict(status='PASS',members=548,budgets=[600,300],elapsed_seconds=result['elapsed_seconds'])))


if __name__=='__main__':main()
