"""Actual spawn/tensor IPC checks, plus read-only first-two real train windows."""
import os,sys,json,time,subprocess,tempfile,traceback,hashlib,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from sabl_ipc import root_for,verify,env_for

def dump(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)

def observe_addresses(out):
    import multiprocessing.connection as con
    if getattr(con.arbitrary_address,'sabl_observed',False):return
    original=con.arbitrary_address
    def address(family):
        result=original(family)
        if family=='AF_UNIX':
            row={'pid':os.getpid(),'address':result,'bytes':len(os.fsencode(result)),'TMPDIR':os.environ.get('TMPDIR'),'tempfile':tempfile.gettempdir()}
            with (Path(out)/('address-%d.jsonl'%os.getpid())).open('a') as f:f.write(json.dumps(row)+'\n')
        return result
    address.sabl_observed=True;con.arbitrary_address=address

class ObservedDataset:
    # This wrapper observes only spawn_observer (excluded by production input_record).
    role='train_core';seed=0
    def __init__(self,out,spec=None):
        self.out=str(out);self.spec=spec;self.base=None
        if spec is not None:
            from sparse_rtdetr.research.sabl_formal_data import Dataset
            self.base=Dataset(spec)
    def __len__(self):return 4869
    def __getitem__(self,key):
        import torch,multiprocessing
        observe_addresses(self.out)
        info={'pid':os.getpid(),'TMPDIR':os.environ.get('TMPDIR'),'TEMP':os.environ.get('TEMP'),'TMP':os.environ.get('TMP'),'tempfile':tempfile.gettempdir(),'mp_tempdir':multiprocessing.util.get_temp_dir(),'CUDA_initialized':torch.cuda.is_initialized()}
        if self.base is None:return torch.full((3,8,8),float(key[2])),{'index':key[2]},info
        im,gt,trace=self.base[key];trace['spawn_observer']['IPC']=info
        return im,gt,trace

def receive(out,spec=None,expected=None):
    import torch,multiprocessing
    from sparse_rtdetr.localization.data import training_loader
    from sparse_rtdetr.protocol_r18.data import input_record
    from sparse_rtdetr.protocol_r18.engine import output_digest
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    ds=ObservedDataset(out,spec);loader=training_loader(ds,epoch=1,completed_windows=0,window_limit=2)
    assert loader.num_workers==2 and loader.timeout==300 and loader.prefetch_factor==2 and not loader.persistent_workers
    assert torch.multiprocessing.get_sharing_strategy()=='file_descriptor'
    iterator=iter(loader);workers=list(iterator._workers);rows=[];infos=[]
    for images,targets,traces in iterator:
        if spec is None:
            assert images.shape==(16,3,8,8);identity={'indexes':[x['index'] for x in targets]};infos+=traces
        else:
            identity=input_record(images,targets,traces);infos+=[x['IPC'] for x in identity.pop('spawn_workers')]
        rows.append({'input':identity,'window_input_SHA':output_digest([images,targets])})
    assert len(rows)==2 and len({x['pid'] for x in infos})==2
    assert all(p.exitcode==0 and not p.is_alive() for p in workers)
    root=Path(os.environ['TMPDIR'])
    assert all(x['TMPDIR']==x['TEMP']==x['TMP']==x['tempfile']==str(root) and Path(x['mp_tempdir']).is_relative_to(root) and not x['CUDA_initialized'] for x in infos)
    addresses=[json.loads(line) for p in out.glob('address-*.jsonl') for line in p.read_text().splitlines()]
    assert len(addresses)>=2 and all(x['bytes']<108 and Path(x['address']).is_relative_to(root) for x in addresses)
    if expected:
        assert [x['window_input_SHA'] for x in rows]==expected['window_input_SHA'],'authenticated short input/GT differs'
        assert [x['input'] for x in rows]==expected['N24_first_two_inputs'],'certified N24 first two input summaries differ'
    assert not torch.cuda.is_initialized()
    result={'status':'PASS_REAL_MULTIPROCESS_DATALOADER_CPU' if spec else 'PASS_CONSTRUCTED_MULTIPROCESS_CPU','detector_loaded':False,'model_forward':0,'GPU_forward':0,'training_update':0,'windows':rows,'worker_observations':list({x['pid']:x for x in infos}.values()),'addresses':addresses,'worker_exitcodes':[p.exitcode for p in workers],'workers':2,'timeout':300,'prefetch_factor':2,'sharing_strategy':'file_descriptor','CUDA_initialized':False,'fresh_formal_start_required':True,'cursor_or_RNG_exported_for_training':False}
    dump(out/'RESULT.json',result);return result

def real_check(a):
    import sabl_corrected_formal as formal
    out=Path(a.out);out.mkdir();start=time.monotonic()
    try:
        assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
        verify(out);m=formal.computation(a.manifest);c=formal.validate(formal.read(a.contract))
        assert c['commit']==m['commit'] and c['manifest_sha256']==formal.sha(a.manifest)
        f=formal.read(formal.FIXTURE);v=formal.exact(f['short_input_reference']);expected={'window_input_SHA':v['window_input_SHA']}
        expected['N24_first_two_inputs']=[]
        with Path(f['N24_first_inputs']['path']).open() as events:
            for _ in range(2):expected['N24_first_two_inputs'].append(json.loads(next(events))['input'])
        # Full log SHA is checked here, not reconstructed from query numbers.
        formal.ref_exact(f['N24_first_inputs']);formal.ref_exact(f['short_input_reference'])
        result=receive(out,c['metadata']['data']['train_core'],expected)
        dump(out/'EXIT.json',{'returncode':0,'seconds':time.monotonic()-start,'only_CPU_data':True})
        dump(out/'CLOSED.json',{'result_SHA':formal.sha(out/'RESULT.json'),'no_training_state':True})
        print(json.dumps({'status':result['status'],'windows':2,'matched_short_and_N24_input_hashes':True,'seconds':time.monotonic()-start}))
    except BaseException as e:
        (out/'stderr.txt').write_text(traceback.format_exc());dump(out/'STOP.json',{'error':repr(e),'development_only':True});raise

def probe(out):
    from multiprocessing.connection import Listener
    observe_addresses(out);error=None;listener=None
    try:listener=Listener()
    except OSError as e:error=str(e)
    finally:
        if listener:listener.close()
    rows=[json.loads(s) for p in out.glob('address-*.jsonl') for s in p.read_text().splitlines()]
    dump(out/'PROBE.json',{'error':error,'addresses':rows,'TMPDIR':os.environ.get('TMPDIR'),'not_original_failed_address':True})

def suite(output):
    out=Path(output);out.mkdir();start=time.monotonic();calls=[]
    def call(name,argv,env,ok=True):
        p=subprocess.run(argv,cwd=ROOT,env=env,capture_output=True,timeout=120)
        (out/(name+'.stdout')).write_bytes(p.stdout);(out/(name+'.stderr')).write_bytes(p.stderr)
        calls.append({'name':name,'returncode':p.returncode});dump(out/(name+'-EXIT.json'),calls[-1]);assert (p.returncode==0)==ok,p.stderr.decode()[-2000:]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');env.pop('PYTHONPATH',None)
    long=out/('isolated-old-layout-'+'x'*100)/'compiler-cache/tmp';long.mkdir(parents=True)
    failure=out/'old-long';failure.mkdir();oldenv=dict(env,TMPDIR=str(long),TEMP=str(long),TMP=str(long))
    call('old-long',[sys.executable,'-B',__file__,'probe','--out',str(failure)],oldenv)
    bad=json.loads((failure/'PROBE.json').read_text());assert bad['error']=='AF_UNIX path too long' and all(x['bytes']>=108 for x in bad['addresses'])
    task=out/'short-tensor-transfer';task.mkdir()
    helper=ROOT/'tools/sabl_ipc.py'
    call('actual-bootstrap',[sys.executable,'-B',str(helper),'exec','--task',str(task),'--',sys.executable,'-B',__file__,'receive','--out',str(task)],env)
    result=json.loads((task/'RESULT.json').read_text());assert len(result['addresses'])==2
    call('wrong-env-refused',[sys.executable,'-B',__file__,'receive','--out',str(task)],oldenv,False)
    # Ownership and stale Python caches must reject, without creating a model.
    conflict=out/'conflict';env_for(conflict,create=True);marker=root_for(conflict)/'OWNER.json';before=marker.read_bytes();marker.write_text('{}')
    try:
        try:env_for(conflict)
        except ValueError:pass
        else:raise AssertionError('owner conflict accepted')
    finally:marker.write_bytes(before)
    import tempfile as tf
    oldcache=tf.tempdir;oldenvvars={k:os.environ.get(k) for k in ['TMPDIR','TEMP','TMP']}
    try:
        os.environ.update({k:str(root_for(conflict)) for k in ['TMPDIR','TEMP','TMP']});tf.tempdir=str(long)
        try:verify(conflict)
        except ValueError:pass
        else:raise AssertionError('stale tempfile cache accepted')
    finally:
        tf.tempdir=oldcache
        for k,v in oldenvvars.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v
    dump(out/'RESULT.json',{'status':'PASS_IPC_STARTUP_AND_TENSOR_TRANSPORT','calls':calls,'ownership_conflict_refused':True,'stale_tempfile_cache_refused':True,'old':bad,'new_addresses':result['addresses'],'new_worker_exitcodes':result['worker_exitcodes'],'production_worker_count_unchanged':2,'timeout_unchanged':300,'sharing_strategy_unchanged':'file_descriptor','GPU_forward':0,'detector_loaded':False,'seconds':time.monotonic()-start})
    print(json.dumps({'status':'PASS_IPC_STARTUP_AND_TENSOR_TRANSPORT','seconds':time.monotonic()-start}))

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['suite','probe','receive']);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out)
    if a.mode=='suite':suite(out)
    elif a.mode=='probe':probe(out)
    else:verify(out);receive(out)
if __name__=='__main__':main()
