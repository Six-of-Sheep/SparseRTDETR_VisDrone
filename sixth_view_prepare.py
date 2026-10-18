"""Preparation-only CLI. No model, GPU, production inference or scoring dispatch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))

def encoded(x):return (json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def publish(p,x):
    with p.open('xb') as f:f.write(encoded(x))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def worker(out,fail):
    import torch
    torch.set_num_threads(1)
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_initialized()
    if fail:raise RuntimeError('EXPECTED_PREPARATION_CPU_WORKER_FAILURE')
    from PIL import Image
    import numpy as np
    from types import SimpleNamespace
    from sparse_rtdetr.localization.data import RawAugmentation,parse_raw
    from sparse_rtdetr.r0 import evaluation as ev
    from sparse_rtdetr.baseline import primary_evaluator as primary
    from infer_seed import make_view_pixels,pixels
    from sixth_view_adapter import make_six_pixels,decode_extra
    from sixth_view_core import center_window,merge_six,SixthCandidate
    from slice_core import Candidate
    def forbidden(*a,**kw):raise RuntimeError('CPU preparation forbids models/CUDA/backward/AP')
    ev.score_budget=ev.bridge_metrics=primary.evaluate_primary_v1=forbidden
    transform=RawAugmentation((768,1344))
    torch.nn.Module.__init__=torch.cuda._lazy_init=torch.autograd.backward=forbidden
    before=torch.get_rng_state().clone()
    images=out/'images';images.mkdir();rows=[]
    for i,(h,w) in enumerate([(31,47),(32,48),(33,48),(32,49)]):
        a=np.arange(h*w*3,dtype=np.uint8).reshape(h,w,3)
        p=images/f'cpu-{i}.png';Image.fromarray(a).save(p)
        rows.append({'stable_image_id':f'cpu-{i}','relative_path':p.name,'height':h,'width':w,
                     'image_sha256':digest(p),'image_size_bytes':p.stat().st_size,'split':'val'})
    data=SimpleNamespace(spec={'roots':{'images':str(images)}},transform=transform)
    old,old_geometry=make_view_pixels(data,rows,True);checks=[]
    for variant in ('C-center','C-flip'):
        views,geometry=make_six_pixels(data,rows,variant)
        assert len(views)==6 and all(len(v)==4 for v in views)
        assert all(torch.equal(views[v][k],old[v][k]) for v in range(5) for k in range(4))
        assert [g['original_tiles'] for g in geometry]==old_geometry
        for k,row in enumerate(rows):
            with Image.open(images/row['relative_path']) as im:
                expected=im.crop(center_window(row['height'],row['width'])) if variant=='C-center' else im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                try:assert torch.equal(views[5][k],pixels(data,expected))
                finally:expected.close()
            logits=torch.zeros(300,10,dtype=torch.float32);boxes=torch.tensor([.5,.5,.25,.5]).repeat(300,1)
            candidates=decode_extra(logits,boxes,row,variant)
            assert len(candidates)==600 and all(p.view_id==5 for p in candidates)
            x,y,x2,y2=center_window(row['height'],row['width']);w=x2-x;h=y2-y
            expected=(x+w*.375,y+h*.25,x+w*.625,y+h*.75) if variant=='C-center' else (row['width']*.375,row['height']*.25,row['width']*.625,row['height']*.75)
            assert candidates[0].box_xyxy==expected
            boxes[0]=torch.tensor([-.5,.5,.25,.5]);outside=decode_extra(logits,boxes,row,variant)
            assert any(p.box_xyxy[0]<0 or p.box_xyxy[2]>row['width'] for p in outside)
        checks.append({'variant':variant,'original_five_pixels_exact':True,'extra_pixels_exact':True,'FP32_decode_view5_and_outside_exact':True})
        del views
    # Original NMS -> budgets -> original ignore -> no refill, using a complete
    # real-shaped candidate pool, not a manually reordered prediction list.
    base=[Candidate((q*20.,cat*20.,q*20.+10.,cat*20.+10.),.9,v,q,cat,(cat-1)*300+q+1)
          for v in range(5) for cat in (1,2) for q in range(300)]
    extra=[SixthCandidate((10000.+q*20.,cat*20.,10010.+q*20.,cat*20.+10.),.4,5,q,cat,(cat-1)*300+q+1)
           for cat in (1,2) for q in range(300)]
    result=merge_six(base,extra,variant='C-center')
    row={'stable_image_id':'synthetic-ignore','height':100,'width':20000,'split':'val'}
    raw=parse_raw(b'0,0,6000,100,0,0,0,0\n',row['stable_image_id'])
    filters=[]
    for budget in (600,300):
        predictions=[{'image_id':row['stable_image_id'],'category_id':p.category_id,'score':p.score,'bbox_xyxy':list(p.box_xyxy)} for p in result[f'top{budget}']]
        _,filtered,counts=ev.bridge_inputs([row],[raw],predictions)
        assert counts['detections_before']==budget and counts['detections_after']==0 and not filtered
        filters.append({'budget':budget,**counts})
    assert result['kept_count']==1200 and result['top300']==result['top600'][:300]
    # The old literal disabled path must retain the entire serialized result.
    from slice_core import merge_candidates,serialize_candidates
    disabled=merge_six(base);original=merge_candidates(base)
    assert disabled==original and serialize_candidates(disabled['top600'])==serialize_candidates(original['top600'])
    assert not torch.cuda.is_initialized() and torch.equal(before,torch.get_rng_state())
    publish(out/'CPU_RESULT.json',{'status':'PASS','synthetic_only':True,'image_cases':checks,'filter_order':filters,
             'disabled_C_exact':True,'original_NMS_reused_byte_unchanged':True,'new_scientific_scores':0,
             'model_constructions':0,'CUDA_initialized':False,'forward_calls':0,'detector_updates':0})
    publish(out/'CLOSED.json',{'status':'PASS','result_sha256':digest(out/'CPU_RESULT.json')})

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['cpu-check','cpu-worker']);p.add_argument('--output',required=True,type=Path);p.add_argument('--inject-failure',action='store_true');a=p.parse_args()
    out=a.output.resolve()
    if a.mode=='cpu-check':
        out.mkdir(parents=True,exist_ok=False);publish(out/'START.json',{'mode':a.mode,'time':time.time(),'pid':os.getpid(),'science_authorized':False})
        start=time.monotonic();records=[]
        for name,failure in [('success',False),('worker-exception',True)]:
            dest=out/name;dest.mkdir()
            cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'cpu-worker','--output',str(dest)]+(['--inject-failure'] if failure else [])
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_THREADING_LAYER='GNU')
            with (dest/'worker.log').open('xb') as log:
                child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env);code=child.wait()
            if code!=(2 if failure else 0):raise RuntimeError('CPU worker unexpected exit: '+(dest/'worker.log').read_text())
            assert (dest/('STOP.json' if failure else 'CLOSED.json')).is_file()
            records.append({'case':name,'worker_pid':child.pid,'returncode':code,'exited':True})
        before={str(q):digest(q) for q in out.rglob('*') if q.is_file()}
        duplicate=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'cpu-check','--output',str(out)],capture_output=True)
        duplicate_worker=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'cpu-worker','--output',str(out/'success')],capture_output=True)
        assert duplicate.returncode!=0 and duplicate_worker.returncode!=0 and before=={str(q):digest(q) for q in out.rglob('*') if q.is_file()}
        publish(out/'CPU_GATE.json',{'status':'PASS','cases':records,'duplicate_start_rejected_without_changes':True,
                'elapsed_seconds':time.monotonic()-start,'scope':'preparation CPU only; no production dispatch','GPU_calls':0,'science_scores':0,
                'source_refs':{name:digest(ROOT/name) for name in ['sixth_view_core.py','sixth_view_adapter.py','sixth_view_decision.py','sixth_view_prepare.py','slice_core.py','infer_seed.py','configs/sahi_sixth_view.json']}})
        return 0
    if (out/'CLOSED.json').exists() or (out/'STOP.json').exists() or (out/'CONSUMED.json').exists():raise ValueError('CPU worker already consumed')
    if not (out.parent/'START.json').is_file() or out.name not in ('success','worker-exception'):raise ValueError('isolated CPU parent binding missing')
    publish(out/'CONSUMED.json',{'scope':'synthetic-only','pid':os.getpid(),'source_sha256':digest(Path(__file__))})
    try:worker(out,a.inject_failure);return 0
    except BaseException as e:
        publish(out/'STOP.json',{'error':repr(e),'traceback':traceback.format_exc(),'automatic_retry':False});return 2

if __name__=='__main__':raise SystemExit(main())
