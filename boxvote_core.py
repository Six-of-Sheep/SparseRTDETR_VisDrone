"""Fixed NMS followed by one-shot score-weighted coordinate voting. GT-free."""
from dataclasses import replace
import math
import numpy as np
from slice_core import _validated_candidates, _box64, classwise_nms

VERSION = 'hard-nms-original-neighborhood-score-vote-fp64-v1'
METRICS = ('AP','APS','AP75_small','APM','APL','AP50','AP75')
LIMITS = {
 'V-five-C': {'600': {'AP':.20,'APS':.30,'AP75_small':0.,'APM':-.20,'APL':-.20},
              '300': {'AP':-.20,'APS':-.20}},
 'V-five-A': {'600': {'AP':.30,'APS':.50,'AP75_small':-.20,'APM':-.50,'APL':-.50},
              '300': {'AP':-.20,'APS':-.20}}}
PAIRS = {'V-five-C':('V-five','C'), 'V-five-A':('V-five','A'),
         'V-whole-B':('V-whole','B'),'V-five-V-whole':('V-five','V-whole')}


def coordinate_vote(candidates):
    items = sorted(_validated_candidates(candidates),key=lambda c:c.identity)
    kept = classwise_nms(items)
    classes = {}
    for p in items: classes.setdefault(p.category_id,[]).append(p)
    arrays = {cat:np.asarray([p.box_xyxy for p in values],dtype=np.float64)
              for cat,values in classes.items()}
    result=[];zero=0;neighborhoods=0;moved=0;sum_l1=0.;max_linf=0.
    for k in kept:
        values=classes[k.category_id];boxes=arrays[k.category_id]
        b=np.asarray(k.box_xyxy,dtype=np.float64)
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            ext=np.maximum(0.,np.minimum(b[2:],boxes[:,2:])-np.maximum(b[:2],boxes[:,:2]))
            inter=ext[:,0]*ext[:,1]
            union=(b[2]-b[0])*(b[3]-b[1])+(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])-inter
            if not np.isfinite(union).all() or np.any(union<=0):raise ValueError('vote invalid union')
            iou=inter/union
        if not np.isfinite(iou).all() or np.any(iou<0) or np.any(iou>1):raise ValueError('vote invalid IoU')
        # Sequential binary64 additions, original identity order (no tree sum/FMA).
        weight=0.;coords=[0.,0.,0.,0.]
        for index in np.flatnonzero(iou>.5):
            p=values[int(index)];weight+=p.score;neighborhoods+=1
            for d in range(4):coords[d]+=p.score*p.box_xyxy[d]
        if not math.isfinite(weight) or not all(math.isfinite(v) for v in coords):
            raise ValueError('vote nonfinite accumulator')
        if weight==0.:box=k.box_xyxy;zero+=1
        else:box=_box64(tuple(v/weight for v in coords))
        delta=[abs(a-b) for a,b in zip(box,k.box_xyxy)]
        moved+=int(any(delta));sum_l1+=sum(delta);max_linf=max(max_linf,max(delta))
        result.append(replace(k,box_xyxy=box))
    if [(p.identity,p.score) for p in result]!=[(p.identity,p.score) for p in kept]:
        raise ValueError('vote altered identity/score/order')
    return {'all':result,'top600':result[:600],'top300':result[:300],
        'stats':{'version':VERSION,'candidate_count':len(items),'kept_count':len(kept),
                 'zero_weight_fallbacks':zero,'neighborhood_memberships':neighborhoods,
                 'moved_boxes':moved,'coordinate_L1_sum_px':sum_l1,'max_Linf_px':max_linf}}


def compare_scores(scores):
    for arm in ('A','B','C','V-whole','V-five'):
        for budget in ('600','300'):
            for m in METRICS:
                x=scores[arm][budget][m]
                if isinstance(x,bool) or not math.isfinite(x) or not 0<=x<=1:raise ValueError('metric units/finite')
    diff={b:{p:{m:(scores[a][b][m]-scores[c][b][m])*100. for m in METRICS}
             for p,(a,c) in PAIRS.items()} for b in ('600','300')}
    gates={p:{b:{m:{'difference_pp':diff[b][p][m],'minimum_pp':v,'passed':diff[b][p][m]>=v}
                 for m,v in ms.items()} for b,ms in bs.items()} for p,bs in LIMITS.items()}
    return {'differences_pp':diff,'gates':gates,
            'quality_passed':all(v['passed'] for bs in gates.values() for ms in bs.values() for v in ms.values()),
            'end_to_end_cost':'NOT_MEASURED','units':'percentage points; converted once'}
