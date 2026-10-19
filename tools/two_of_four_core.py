"""Fixed GT-free subset merge and pre-registered screening (not old C gates)."""
import math

PAIRS = ((1,2),(1,3),(1,4),(2,3),(2,4),(3,4))
PROTECTION = {'AP75_small':-.2, 'APM':-.5, 'APL':-.5}


def merge_pair(pool, pair):
    from slice_core import classwise_nms
    if type(pair) not in (tuple,list) or tuple(pair) not in PAIRS:
        raise ValueError('not one of six frozen pairs')
    # Original Candidate identities are retained, never renumbered or matched across views.
    return classwise_nms([p for p in pool if p.view_id in (0,*pair)])[:600]


def screen(metrics, baseline):
    for record in (metrics,baseline['A'],baseline['B'],baseline['C']):
        if any(not math.isfinite(record[m]) or not 0<=record[m]<=1
               for m in ('AP','APS',*PROTECTION)):
            raise ValueError('invalid screening metric units/value')
    gates={}
    for m in ('AP','APS'):
        gain=baseline['C'][m]-baseline['A'][m]
        if gain<=0: raise ValueError('nonpositive reference gain')
        actual=metrics[m]-baseline['A'][m];required=.8*gain
        gates[m+'_retention']={'actual_gain_pp':100*actual,'minimum_gain_pp':100*required,
                              'retained_fraction':actual/gain,'passed':actual>=required}
    for ref in ('C','A'):
        for m,minimum in PROTECTION.items():
            actual=100*(metrics[m]-baseline[ref][m])
            gates[m+'-'+ref]={'difference_pp':actual,'minimum_pp':minimum,'passed':actual>=minimum}
    return {'passed':all(g['passed'] for g in gates.values()),'gates':gates,
            'differences_pp':{a:{m:100*(metrics[m]-v) for m,v in b.items() if m in metrics}
                              for a,b in baseline.items()}}


def select(results):
    if list(results)!=['P'+''.join(map(str,p)) for p in PAIRS]:
        raise ValueError('all six results in frozen order required')
    return next((name for name,r in results.items() if r['passed']),None)


def coverage(boxes, categories, gt_boxes, gt_categories):
    """Independent correct-class geometric best IoU, not evaluation matching."""
    import numpy as np
    a=np.asarray(boxes,dtype=float).reshape(-1,4)
    b=np.asarray(gt_boxes,dtype=float).reshape(-1,4)
    for v in (a,b):
        if not np.isfinite(v).all() or (len(v) and np.any(v[:,2:]<=v[:,:2])):
            raise ValueError('invalid diagnostic geometry')
    if len(a)!=len(categories) or len(b)!=len(gt_categories):raise ValueError('category length')
    if not len(a) or not len(b):return np.zeros(len(b))
    area=lambda v:(v[:,2]-v[:,0])*(v[:,3]-v[:,1])
    inter=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])).prod(2)
    iou=inter/(area(a)[:,None]+area(b)[None,:]-inter)
    return np.where(np.asarray(categories)[:,None]==np.asarray(gt_categories)[None,:],iou,0).max(0)
