"""Reject corrupt/wrong-scope native full states without any detector forward."""
import sys,os,json,resource,signal,time,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
for k,v in {'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_THREADING_LAYER':'GNU'}.items():os.environ[k]=v
resource.setrlimit(resource.RLIMIT_AS,(4<<30,4<<30));signal.alarm(300)
import torch
from sparse_rtdetr.research.sabl_control import make_engine,VERSION
from sparse_rtdetr.research import sabl_state as st
from sparse_rtdetr.protocol_r18.checkpoint import file_sha

def main(out):
    out=Path(out);out.mkdir();start=time.monotonic();torch.set_num_threads(2);torch.set_num_interop_threads(1)
    binding={'scope':'CPU_SYNTHETIC_SABL','arm':'S','version':VERSION,'contract':'SYNTHETIC'}
    engine=make_engine('S',True);receipt=st.save(engine,binding,out/'original.pt');del engine;gc.collect()
    checks=[]
    for name in ('SHA','arm','schema','nan','EMA','optimizer'):
        engine=make_engine('S',True);p=out/'original.pt';expected=receipt['sha256'];identity=dict(binding)
        if name=='SHA':expected='0'*64
        elif name=='arm':identity['arm']='A'
        else:
            raw=torch.load(p,map_location='cpu',weights_only=True)
            if name=='schema':del raw['criterion']
            if name=='nan':raw['model']['boundary.cls.weight'][0,0]=float('nan')
            if name=='EMA':raw['ema']['updates']=1
            if name=='optimizer':raw['optimizer']['param_groups'][0]['lr']=.1
            p=out/(name+'.pt');torch.save(raw,p);expected=file_sha(p);del raw
        try:st.restore(engine,identity,p,expected)
        except (ValueError,RuntimeError):
            if not engine.failed:raise AssertionError('restore did not seal failed engine')
            checks.append(name+'_refused_failed_engine_sealed')
        else:raise AssertionError('accepted '+name)
        del engine;gc.collect()
    result={'status':'PASS','checks':checks,'seconds':time.monotonic()-start,'CPU_detector_forward':0,'CPU_optimizer_updates':0,'GPU_cost':0,'scope':'SYNTHETIC_NOT_SCIENTIFIC'}
    (out/'RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main(sys.argv[1])
