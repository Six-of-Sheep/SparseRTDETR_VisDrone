"""Actual CPU CLI replay closure. Synthetic data/state never enter GPU definition."""
import os,sys,json,subprocess,time,gc,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
os.environ['CUDA_VISIBLE_DEVICES']=''
from sabl_oom_replay import write,read,sha,digest,expected,replay,witnesses,CAPS,validate,source

def run_fixture(a):
    import torch
    import sabl_formal as formal
    from sparse_rtdetr.research import sabl_formal_state as state,sabl_formal_data as data
    out=Path(a.out);out.mkdir();start=time.monotonic();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    c={'mode':'CPU_SYNTHETIC_SABL_FORMAL','metadata':None,'manifest_sha256':'f'*64};base=out/'CPU_ONLY_INITIAL.pt';engine,binding,_,_=formal.fresh(c,'S','cpu');cp=state.save(engine,binding,base);ds=data.Fixture();hooks,_=witnesses(engine);rows=[]
    for i in [1,2]:
        images,targets,identity=ds.window(i);r=engine.train_window(images,targets,capture=False);rows.append(expected({'update':i,'input':identity,'result':r}))
    for h in hooks:h.remove()
    del engine;gc.collect();engine,binding,_,_=formal.fresh(c,'S','cpu');audit=state.restore(engine,binding,str(base),cp['sha256']);assert audit['full_state_exact']
    if a.fault=='input':rows[0]['input_SHA']='0'*64
    if a.fault=='result':rows[0]['result_SHA']='0'*64
    result=replay(out,engine,binding,(ds.window(i) for i in [1,2,3]),rows,False,start,end=3)
    write(out/'CPU_FIXTURE_SCOPE.json',{'CPU_synthetic':True,'baseline_forward':4,'baseline_updates':2,'replay_counts':result['counts'],'real_images_or_checkpoint':False,'GPU_forward':0,'CUDA_initialized':torch.cuda.is_initialized()})
    assert not torch.cuda.is_initialized()
    return 0 if a.fault=='none' and result['status']=='PASS_CPU_REPLAY' else 1

def main(output):
    out=Path(output);out.mkdir();start=time.monotonic();calls=[];checks=[];CPU_forward=0;CPU_updates=0
    def check(n,v):
        checks.append({'name':n,'pass':bool(v)});(out/'CHECKS.json').write_text(json.dumps(checks,indent=2));assert v,n
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');env.pop('PYTHONPATH',None)
    for fault in ['none','input','result']:
        target=out/fault;argv=[sys.executable,'-B',str(ROOT/'tools/sabl_ipc.py'),'exec','--task',str(out),'--',sys.executable,'-B',str(ROOT/'tools/sabl_oom_replay.py'),'fixture','--out',str(target),'--fault',fault]
        p=subprocess.run(argv,cwd=ROOT,env=env,capture_output=True,timeout=150);(out/(fault+'.stdout')).write_bytes(p.stdout);(out/(fault+'.stderr')).write_bytes(p.stderr);calls.append({'fault':fault,'returncode':p.returncode});check('actual CLI '+fault,(p.returncode==0)==(fault=='none'))
        r=read(target/'RESULT.json');scope=read(target/'CPU_FIXTURE_SCOPE.json');CPU_forward+=4+r['counts']['forward'];CPU_updates+=2+r['counts']['actual_updates']
        if fault=='none':
            check('two original rows exact',r['matched_prior_updates']==2)
            check('complete synthetic u3',r['counts']=={'forward':6,'backward':6,'actual_updates':3,'reserved_updates':3} and r['last_complete_update']==3)
            check('state checkpoint/closure',Path(r['saved_complete_boundary']['path']).is_file() and (target/'CLOSED.json').exists())
        else:
            check('first mismatch STOP '+fault,r['status']=='STOP_TRAJECTORY_MISMATCH' and (target/'STOP.json').exists() and not (target/'CLOSED.json').exists())
            check('no progression past first mismatch '+fault,r['counts']['forward']==(0 if fault=='input' else 2))
        check('CUDA never initialized '+fault,not scope['CUDA_initialized'] and scope['GPU_forward']==0)
        states=[]
        for cp in target.glob('*.pt'):
            assert cp.resolve().is_relative_to(out.resolve());states.append({'name':cp.name,'bytes':cp.stat().st_size,'sha256':sha(cp)});cp.unlink()
        write(target/'RETIRED_CPU_TENSORS.json',states)
    p=subprocess.run(argv,cwd=ROOT,env=env,capture_output=True,timeout=30);(out/'duplicate.stderr').write_bytes(p.stderr);check('duplicate refuses',p.returncode!=0 and b'FileExistsError' in p.stderr)
    # Fixed-scope validation is meaningful without real paths or model construction.
    good={'mode':'GPU_DIAGNOSTIC_ONLY','schema':'SABL_OOM_U52_REPLAY_V1','caps':CAPS,'target_update':52,'end_update':60,'allocator_bytes':21<<30,'first_attempt_forward_cap':120,'first_attempt_update_cap':60,'first_attempt_GPU_seconds':600,'comparison':'exact input and RESULT_KEYS canonical SHA; elapsed excluded; first difference STOP; no tolerance','result_keys':list(__import__('sabl_diagnostic').RESULT_KEYS),'diagnostic_state_for_formal':False}
    validate(good)
    for key,value in [('allocator_bytes',22<<30),('end_update',61),('first_attempt_forward_cap',121),('comparison','loose'),('diagnostic_state_for_formal',True)]:
        bad=json.loads(json.dumps(good));bad[key]=value
        try:validate(bad)
        except ValueError:check('reject '+key,True)
        else:check('reject '+key,False)
    write(out/'RESULT.json',{'status':'PASS_OOM_REPLAY_CPU_CLI','checks':len(checks),'calls':calls,'CPU_detector_forward':CPU_forward,'CPU_optimizer_updates':CPU_updates,'GPU_forward':0,'real_checkpoint_or_images_loaded':False,'seconds':time.monotonic()-start,'not_CUDA_memory_validation':True,'production_attempt_created':False})
    print(json.dumps(read(out/'RESULT.json')))
if __name__=='__main__':main(sys.argv[1])
