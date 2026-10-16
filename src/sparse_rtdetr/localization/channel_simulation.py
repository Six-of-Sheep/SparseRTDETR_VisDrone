"""Fixed engineering-only channel simulation, zero Torch/CUDA/model work."""
import json,os,subprocess,sys,time
from pathlib import Path
from .runio import read,write,publish,digest,require,execution_binding,reference
from . import channel,guard

CASES=('interleaved-mirror-unreadable','persistent-loss','conflict','no-progress')

def child(out,case):
    b=channel.binding();root=Path(b['runtime_root']);w=channel.Writer(b,'progress')
    def emit(u):return w.append({'event':'update_complete','update':u,'completed_epochs':3,'monotonic':time.monotonic(),'pid':os.getpid()})
    first=emit(913)
    if case=='persistent-loss':
        (root/'progress.jsonl').unlink();time.sleep(30);return
    if case=='conflict':
        bad={**first,'payload':{**first['payload'],'update':914}};bad.pop('sha256');bad['sha256']=digest(bad)
        with (root/'progress.jsonl').open('ab') as f:f.write(channel.canonical(bad)+b'\n')
        time.sleep(30);return
    if case=='no-progress':
        for _ in range(30):
            # Both same-sequence retries and new-sequence same-operation
            # records must not refresh semantic completion time.
            w.append({**first['payload'],'monotonic':time.monotonic()})
            with (root/'progress.jsonl').open('ab') as f:f.write(channel.canonical(first)+b'\n')
            time.sleep(1)
        return
    # NAS progress does not exist and cannot be read. This simulated mirror
    # error is recorded apart from the authoritative stream, then recovered.
    failures=0
    for _ in range(2):
        try:read(Path(out)/'progress.json')
        except FileNotFoundError:failures+=1
    write(root/'simulated-mirror-errors.json',{'count':failures,'control_effect':False})
    second={**first,'seq':2,'previous':first['sha256'],'payload':{'event':'update_complete','update':914,'completed_epochs':3,'monotonic':time.monotonic(),'pid':os.getpid()}}
    second.pop('sha256');second['sha256']=digest(second);raw=channel.canonical(second)+b'\n';cut=len(raw)//2
    with (root/'progress.jsonl').open('ab') as f:f.write(channel.canonical(first)+b'\n'+raw[:cut]);f.flush();os.fsync(f.fileno())
    # Two actual Linux observer processes contend for the persisted cursor.
    code="from sparse_rtdetr.localization.channel import latest; import sys,time; "
    code+="\nfor _ in range(10): latest(sys.argv[1],'progress');time.sleep(.4)\n"
    observers=[subprocess.Popen([sys.executable,'-B','-c',code,str(root)],env=dict(os.environ,CUDA_VISIBLE_DEVICES='')) for _ in range(2)]
    time.sleep(3)
    with (root/'progress.jsonl').open('ab') as f:f.write(raw[cut:]);f.flush();os.fsync(f.fileno())
    sample=channel.resource();require(0<=time.monotonic()-sample['monotonic']<25,'simulation resource stale')
    for observer in observers:require(observer.wait(timeout=10)==0,'concurrent outer observer failed')
    for name in ('RESULT.json','execution-binding.json'):write(Path(out)/name,{'simulation':True,'case':case,'cuda_initialized':False})
    write(root/'worker-exit.json',{'simulation':True,'pid':os.getpid()})

def run(root,runtime,source_sha256):
    root=Path(root);root.mkdir(exist_ok=False,parents=True);started=time.time();results=[]
    original=guard.Progress
    class SimulationProgress(original):
        def observe(self,now,**kwargs):return super().observe(now,**kwargs,idle=5,operation=7)
    guard.Progress=SimulationProgress
    try:
        for case in CASES:
            out=root/case;ssd=Path(runtime)/case
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(Path(__file__).resolve().parents[2]))
            command=[sys.executable,'-B','-m','sparse_rtdetr.localization.channel_simulation','child',str(out),case]
            guard.supervise(command,out,env,gpu_uuid=None,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),gpu=False,
                            runtime=ssd,run_identity={'simulation':True,'case':case,'source_sha256':source_sha256,'supervisor_pid':os.getpid(),'supervisor_start':guard.identity(os.getpid())['start']})
            ex=read(ssd/'supervisor-exit.json');expected='PASS' if case==CASES[0] else 'STOP'
            require(ex['status']==expected and not ex['remaining_live'] and not ex['unknown_pids'] and not ex['cleanup_errors'],'simulation failure/exit mismatch: '+case)
            if case=='persistent-loss':require(ex['guard_error']['operation']=='progress_read' and ex['guard_error']['errno']==2,'channel absence was not classified')
            if case=='conflict':require('conflicting duplicate' in ex['guard_error']['error'],'conflict not detected')
            if case=='no-progress':require(ex['reason'] in ('NO_PROGRESS_ALL_OBSERVABLES','SINGLE_OPERATION_NO_COMPLETION'),'duplicates reset timeout')
            if case==CASES[0]:
                b=read(ssd/'identity.json');r=channel.Reader(ssd/'progress.jsonl',b,'progress');last,_=r.poll()
                require(last['seq']==2 and last['payload']['update']==914 and read(ssd/'simulated-mirror-errors.json')['count']==2,'interleaved/mirror scenario changed')
                # Exercise actual Linux outer observer locks/incremental cursor.
                require(channel.latest(ssd,'progress')==last['payload'] and channel.latest(ssd,'progress')==last['payload'],'outer channel source mismatch')
            results.append({'case':case,'expected':expected,'exit':reference(ssd/'supervisor-exit.json'),'result':ex})
    finally:guard.Progress=original
    result={'status':'PASS','source_sha256':source_sha256,'cases':results,'gpu_updates':0,'model_forwards':0,
            'simulation_timeout_seconds':{'idle':5,'operation':7},'production_timeout_seconds':{'idle':300,'operation':1800},
            'started_epoch':started,'finished_epoch':time.time(),'elapsed_seconds':time.time()-started}
    write(root/'RESULT.json',result);return result

if __name__=='__main__':
    if sys.argv[1]=='child':child(sys.argv[2],sys.argv[3])
    else:print(json.dumps(run(*sys.argv[2:]),allow_nan=False))
