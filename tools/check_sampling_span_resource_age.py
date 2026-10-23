"""Small no-model regression of the real resource collection/screen interface."""
import argparse, copy, json, os, sys
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    from sparse_rtdetr.protocol_r18 import span_runtime as rt
    seed=rt.fixture_sample()
    c=dict(task_id=rt.TASK,protocol=rt.FORMAL_PROTOCOL,profile='CPU_FIXTURE_ONLY',
        attempt='attempt-age-unit',job_root=str(out/'attempt-age-unit'),source={'sha256':'1'*64},
        contract_sha256='2'*64,resources=rt.execution()['resources'],boot_id=seed['boot_id'],
        storage_identity={k:seed['storage'][k] for k in ('canonical_mount','mount','st_dev')},gpu_uuid='CPU_FIXTURE_NO_GPU')
    checks=[];cases=[]
    def check(name,ok):rt.require(ok,name);checks.append(name)
    def case(name,age,*,stage='score',duration=.01,source_stage='score',expected=False):
        sample=copy.deepcopy(seed);sample['sampled_epoch']=1000.;sample['monotonic']=50.
        sample['sampling_timing']=dict(started=dict(epoch=1000.,monotonic=50.),
            completed=dict(epoch=1000.+duration,monotonic=50.+duration),source='controlled sampler timeline, not model data')
        sample['sampling_phase_identity']=rt.resource_context(c,source_stage)
        before=set(out.rglob('resource-age-rejected-*.json'));error=None
        with patch.object(rt.time,'time',return_value=1000.+age),patch.object(rt.time,'monotonic',return_value=50.+age):
            try:rt.screen(sample,c,entry=False,gpu=False,stage=stage)
            except RuntimeError as exc:error=str(exc)
        after=set(out.rglob('resource-age-rejected-*.json'));rejected=sorted(after-before)
        check(name+' decision', (error is None)==expected)
        if not expected:
            check(name+' original error',error=='stale resource sample')
            check(name+' one persistent rejected sample',len(rejected)==1)
            record=rt.read(rejected[0]);diag=record['age']
            check(name+' exact age/threshold',diag['actual_age_seconds']==age and diag['threshold_seconds']==25)
            check(name+' sampling/checked times',diag['sampling_started']==sample['sampling_timing']['started'] and
                diag['sampling_completed']==sample['sampling_timing']['completed'] and diag['checked']['epoch']==1000.+age)
            check(name+' duration/identity',abs(diag['sampling_elapsed_seconds']-duration)<1e-9 and diag['phase_identity']['stage']==stage)
            check(name+' preserves original sample time',record['sample']['sampled_epoch']==1000.)
        cases.append(dict(case=name,admitted=error is None,error=error,rejections=[rt.reference(x) for x in rejected]))
    case('normal',.25,expected=True)
    case('exact-threshold',25.)
    case('expired',26.)
    case('negative',-.5)
    case('delayed-sampler',26.,duration=26.)
    case('stage-switch-old-sample',26.,source_stage='infer-P')
    case('stage-switch-new-sample',.25,stage='score',expected=True)
    # Wall reversal remains refused even when monotonic elapsed is positive.
    sample=copy.deepcopy(seed);sample.update(sampled_epoch=1000.,monotonic=50.)
    with patch.object(rt.time,'time',return_value=999.5),patch.object(rt.time,'monotonic',return_value=50.25):
        try:rt.screen(sample,c,entry=False,gpu=False,stage='score')
        except RuntimeError:checks.append('wall reversal not bypassed using monotonic')
        else:raise RuntimeError('negative wall age bypassed')
    # Production snapshot wrapper must preserve collector time, including delay.
    def delayed():
        value=copy.deepcopy(seed);value.update(sampled_epoch=1000.,monotonic=50.);return value
    with patch.object(rt,'fixture_sample',side_effect=delayed),patch.object(rt.time,'time',side_effect=[1000.,1026.,1026.]),patch.object(rt.time,'monotonic',side_effect=[50.,76.,76.]):
        try:rt.snapshot(c,entry=False,gpu=False,stage='score')
        except RuntimeError as exc:check('snapshot delay is not restamped',str(exc)=='stale resource sample')
        else:raise RuntimeError('snapshot rewrote sample time')
    # Fresh stage transitions collect again; no reused age_check grants permission.
    first=rt.snapshot(c,entry=True,gpu=False,stage='infer-P')
    second=rt.snapshot(c,entry=True,gpu=False,stage='score')
    check('real snapshot stage transition identity',first['sampling_phase_identity']['stage']=='infer-P' and second['sampling_phase_identity']['stage']=='score')
    check('real snapshot newly collected',second['sampled_epoch']>=first['sampled_epoch'] and second is not first)
    check('pre-run rejection did not consume job root',not Path(c['job_root']).exists())
    check('frozen threshold unchanged',c['resources']['max_resource_age_seconds']==25)
    check('no CUDA initialized','torch' not in sys.modules or not sys.modules['torch'].cuda.is_initialized())
    rt.publish(out/'CPU_GATE.json',dict(status='RESOURCE_AGE_CPU_PASS_NO_MODEL',checks=checks,checks_run=len(checks),skips=0,
        cases=cases,model_constructions=0,model_forward_calls=0,CUDA_initialized=False,
        original_STOP_cause='UNKNOWN: rejected sample was not persisted; bounded live reproduction did not reproduce'))
    print(json.dumps(dict(status='RESOURCE_AGE_CPU_PASS_NO_MODEL',checks_run=len(checks),skips=0)))

if __name__=='__main__':main()
