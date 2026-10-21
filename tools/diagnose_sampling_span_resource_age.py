"""Bounded no-model resource-age reproduction; preserves every rejected sample.

Initial diagnosis uses the unmodified screen. Injected cases test the frozen
25-second rule; the observation spy returns the real clock value unchanged.
"""
import argparse, copy, json, os, sys, time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    from sparse_rtdetr.protocol_r18 import span_runtime as rt
    real=time.time;mono=time.monotonic
    seed=rt.fixture_sample()
    c=dict(task_id=rt.TASK,protocol=rt.FORMAL_PROTOCOL,profile='CPU_FIXTURE_ONLY',
        attempt='attempt-age-diagnosis',job_root=str(a.output.resolve()/'attempt-age-diagnosis'),
        resources=rt.execution()['resources'],boot_id=seed['boot_id'],
        storage_identity={k:seed['storage'][k] for k in ('canonical_mount','mount','st_dev')},
        gpu_uuid='CPU_FIXTURE_NO_GPU')
    threshold=c['resources']['max_resource_age_seconds'];records=[];anomalies=[]
    def probe(name,sample,clock=None):
        calls=[];start=dict(epoch=real(),monotonic=mono())
        def observed():
            value=real() if clock is None else clock
            calls.append(dict(epoch=value,monotonic=mono()))
            return value
        error=None
        try:
            with patch.object(rt.time,'time',observed):rt.screen(sample,c,entry=False,gpu=False)
        except RuntimeError as exc:error=str(exc)
        end=dict(epoch=real(),monotonic=mono())
        checked=calls[-1] if calls else None
        return dict(case=name,sample=sample,sampling_clock='Python time.time / CLOCK_REALTIME; time.monotonic / CLOCK_MONOTONIC',
            check_start=start,check_completed=end,clock_calls=calls,
            actual_age_seconds=None if checked is None else checked['epoch']-sample['sampled_epoch'],
            threshold_seconds=threshold,error=error,admitted=error is None)
    # One cold and bounded 20000 hot calls: no model, forward, or GPU access.
    for i in range(20001):
        begin=dict(epoch=real(),monotonic=mono())
        sample=rt.fixture_sample()
        complete=dict(epoch=real(),monotonic=mono())
        row=probe('cold' if i==0 else 'hot-'+str(i),sample)
        row.update(sampling_started=begin,sampling_completed=complete,
            sampling_elapsed_seconds=complete['monotonic']-begin['monotonic'])
        if i==0:records.append(row)
        if row['error']:
            anomalies.append(row)
            # Preserve the first unexpected failure; no repeated retry.
            break
    for name,offset in [('normal',.25),('expired',26.),('negative',-.5),('delayed-sampling',26.)]:
        sample=rt.fixture_sample();row=probe(name,sample,sample['sampled_epoch']+offset)
        row['injected_clock_for_boundary_test']=True
        records.append(row)
    prior=rt.fixture_sample();one=probe('stage-A',prior,prior['sampled_epoch']+.1)
    two=probe('stage-B-old-sample',prior,prior['sampled_epoch']+26.)
    new=rt.fixture_sample();three=probe('stage-B-new-sample',new,new['sampled_epoch']+.1)
    records.extend([one,two,three])
    result=dict(scope='NO_MODEL_CONSTRUCTION_NO_FORWARD_NO_CUDA',
        hot_calls_completed=i,threshold_unchanged=threshold,records=records,unexpected_rejections=anomalies,
        torch_imported='torch' in sys.modules,
        CUDA_initialized=False if 'torch' not in sys.modules else sys.modules['torch'].cuda.is_initialized())
    (a.output/'DIAGNOSIS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(a.output),hot_calls=i,unexpected_rejections=len(anomalies),
        first_anomaly_age=None if not anomalies else anomalies[0]['actual_age_seconds'])))

if __name__=='__main__':main()
