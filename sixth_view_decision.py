"""Pre-registered decision from supplied endpoint numbers; never scores data."""
import math
METRICS=('AP','APS','AP75_small','APM','APL','AP50','AP75')
LIMITS={'center-C':{'600':{'AP':.2,'APS':.3,'AP75_small':0.,'APM':-.2,'APL':-.2},'300':{'AP':-.2,'APS':-.2}},
        'center-A':{'600':{'AP':.3,'APS':.5,'AP75_small':-.2,'APM':-.5,'APL':-.5},'300':{'AP':-.2,'APS':-.2}}}
PAIRS={'center-C':('C-center','C'),'flip-C':('C-flip','C'),'center-flip':('C-center','C-flip'),'center-A':('C-center','A')}
def assess_quality(scores):
    for a in ('A','C','C-center','C-flip'):
        for b in ('600','300'):
            for m in METRICS:
                x=scores[a][b][m]
                if isinstance(x,bool) or not math.isfinite(x) or not 0<=x<=1:raise ValueError('expected raw 0..1 score units')
    diff={b:{p:{m:(scores[a][b][m]-scores[c][b][m])*100 for m in METRICS} for p,(a,c) in PAIRS.items()} for b in ('600','300')}
    gates={p:{b:{m:{'pp':diff[b][p][m],'minimum_pp':v,'passed':diff[b][p][m]>=v} for m,v in ms.items()} for b,ms in bs.items()} for p,bs in LIMITS.items()}
    quality=all(v['passed'] for bs in gates.values() for ms in bs.values() for v in ms.values())
    control=all(diff['600']['center-flip'][m]>0 for m in ('AP','APS'))
    return {'differences_pp':diff,'gates':gates,'main_14_quality_passed':quality,
            'center_position_extra_value_supported':quality and control,'same_view_control_strict_AP_and_APS':control,
            'quality_permits_next_seeds':quality and control,'cost_validation_still_required':True,
            'flip_must_not_be_promoted':True,'no_AP_recalculation':True}
