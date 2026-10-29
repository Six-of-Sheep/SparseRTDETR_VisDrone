"""Bounded S u52 OOM replay. Original computation/state/trajectory remain authoritative."""
import os,sys,json,time,math,signal,subprocess,hashlib,traceback,decimal,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
for k,v in {'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_THREADING_LAYER':'GNU','PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1'}.items():os.environ[k]=v
from sabl_ipc import verify as verify_ipc,root_for,owned_family_bytes
from sabl_diagnostic import RESULT_KEYS
from sabl_corrected_short import ref_exact
from sabl_formal import write,atomic,read,sha,digest,reference,need
BASE='6ab588f8bf84065ac524f083c36f8febbf24d463';BASE_MANIFEST='7fe84a5747c4cd5c34e37717ec954433a30b2d8f1f3bd6808c326d6b658f3d97'
SCOPE='SABL_OOM_U52_REPLAY_V1';CAPS={'attempts':2,'forward':240,'actual_updates':120,'GPU_seconds':1200,'wall_seconds':2400,'allocated_bytes':21<<30,'return_bytes':5<<20,'self_storage_bytes':2<<30}
WS=Path('/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace');EX=WS/'experiments/localization-distribution';OLD=EX/'r18-sabl-corrected-formal-20261029-001'
def exact(r):return read(ref_exact(r)['path'])
def reduced(result):return {k:result[k] for k in RESULT_KEYS}
def expected(row):return {'update':row['update'],'input_SHA':digest(row['input']),'result_SHA':digest(reduced(row['result'])),'loss':row['result']['loss'],'gradient_norm_before_clip':row['result']['gradient_norm_before_clip']}
def source(manifest):
    from r18_deploy import verify_release
    m=read(manifest);verify_release(ROOT,sha(manifest));need(Path(manifest).resolve()==ROOT/'RELEASE_MANIFEST.json','execution manifest path')
    need(m['computation_base']==BASE and m['computation_base_manifest']==BASE_MANIFEST,'diagnostic computation base')
    base=read(ROOT/'tests/fixtures/sabl_oom_base.json')
    for r in base['computation_files']:need(sha(ROOT/r['path'])==r['sha256'],'immutable computation dependency '+r['path'])
    return m
def size(p):return sum(x.stat().st_size for x in Path(p).rglob('*') if x.is_file())
def prepare(a):
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only preparation');m=source(a.manifest);out=Path(a.out);out.mkdir()
    f=read(ROOT/'tests/fixtures/sabl_oom_base.json');ev=exact(f['source_evidence'])
    for name in ['checkpoint','sidecar','original_contract','old_manifest','old_STOP','old_cost','old_log']:ref_exact(ev[name])
    need(exact(ev['old_manifest'])['commit']==BASE and ev['old_manifest']['sha256']==BASE_MANIFEST,'original science source')
    cp=exact(ev['sidecar']);need(cp['updates']==0 and cp['binding']['arm']=='S' and cp['sha256']==ev['checkpoint']['sha256'],'e000 full state only')
    rows=[]
    with Path(ev['old_log']['path']).open() as stream:
        for line in stream:
            row=json.loads(line);need(row['update']==len(rows)+1,'reference update order');rows.append(expected(row))
    need(len(rows)==51,'only 51 completed updates exist');write(out/'EXPECTED.json',rows)
    cost=exact(ev['old_cost']);need(cost['cumulative']['forward']==32133 and cost['remaining']['forward']==29738,'prior cost identity')
    c={'schema':SCOPE,'mode':'GPU_DIAGNOSTIC_ONLY','caps':CAPS,'commit':m['commit'],'manifest_SHA':sha(a.manifest),'task':str(out.parent.resolve()),'original':ev,'expected':reference(out/'EXPECTED.json'),'comparison':'exact input and RESULT_KEYS canonical SHA; elapsed excluded; first difference STOP; no tolerance','result_keys':list(RESULT_KEYS),'target_update':52,'end_update':60,'allocator_bytes':21<<30,'allocator_policy':'unchanged inherited environment; native backend required','allocator_env_now':{k:os.environ.get(k) for k in ['PYTORCH_CUDA_ALLOC_CONF','PYTORCH_ALLOC_CONF']},'formal_remaining_reserved':cost['remaining'],'formal_cumulative_before':cost['cumulative'],'first_attempt_forward_cap':120,'first_attempt_update_cap':60,'first_attempt_GPU_seconds':600,'old_failure_preserved':True,'diagnostic_state_for_formal':False}
    write(out/'DEFINITION.json',c);budget={'scope':SCOPE,'approved':CAPS,'formal_balance_not_spendable_by_debug':cost['remaining'],'historical_costs_reset':False,'formal_training_authorized':False};budget_path=out.parent/'DEBUG_BUDGET.json'
    if budget_path.exists():need(read(budget_path)==budget,'existing independent budget conflict; never overwrite')
    else:write(budget_path,budget)
    print(json.dumps({'status':'PREPARED_INDEPENDENT_DIAGNOSTIC','definition_SHA':sha(out/'DEFINITION.json'),'expected_updates':51,'first_attempt_updates':60,'GPU_forward':0}))
def validate(c):
    need(c['mode']=='GPU_DIAGNOSTIC_ONLY' and c['schema']==SCOPE and c['caps']==CAPS and c['target_update']==52 and c['end_update']==60 and c['allocator_bytes']==21<<30,'diagnostic scope/caps')
    need(c['first_attempt_forward_cap']==120 and c['first_attempt_update_cap']==60 and c['first_attempt_GPU_seconds']==600,'fixed first attempt')
    need(c['comparison']=='exact input and RESULT_KEYS canonical SHA; elapsed excluded; first difference STOP; no tolerance' and c['result_keys']==list(RESULT_KEYS),'fixed comparison')
    need(c['diagnostic_state_for_formal'] is False,'debug not formal')
    return c
def resources(c):
    from sabl_corrected_short import no_jobs
    import torch,torchvision
    no_jobs()
    for line in subprocess.check_output(['ps','-eo','pid,ppid,args'],text=True).splitlines()[1:]:
        fields=line.strip().split(None,2)
        if len(fields)==3 and int(fields[0]) not in [os.getpid(),os.getppid()] and any(x in fields[2] for x in ['sabl_oom_replay.py run','sabl_oom_replay.py launch']):raise ValueError('other OOM replay process')
    need(Path(sys.executable).resolve()==Path(exact(c['original']['original_contract'])['metadata']['python']).resolve(),'certified interpreter')
    old=exact(c['original']['original_contract']);env=old['metadata']['environment']
    need(str(torch.__version__)==env['torch'] and str(torchvision.__version__)==env['torchvision'] and torch.version.cuda==env['cuda_build'] and torch.backends.cudnn.version()==env['cudnn'],'identical runtime')
    need({k:os.environ.get(k) for k in c['allocator_env_now']}==c['allocator_env_now'] and torch.cuda.get_allocator_backend()=='native','no allocator policy change')
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines();need(len(lines)==1,'one GPU')
    uuid,name,total,free=[s.strip() for s in lines[0].split(',')];need(uuid==old['metadata']['gpu_uuid'] and name=='NVIDIA GeForce RTX 4090 D','GPU identity');need(int(free)>=22*1024 and int(total)>=23*1024,'21GiB plus spare resource gate')
    family=sum(size(p) for p in EX.glob('r18-sabl-*') if p.is_dir());private,_=owned_family_bytes(EX);external=sum(size(p) for p in [Path('/tmp/torchinductor_lyy'),Path('/home/lyy/.triton/cache'),Path('/home/lyy/.nv/ComputeCache'),Path('/home/lyy/.cache/torch/inductor'),Path('/home/lyy/.cache/triton')] if p.exists())
    import shutil
    need(family+private+external+(2<<30)<=30<<30 and shutil.disk_usage(c['task']).free>=4<<30,'diagnostic storage/spare gate')
    need(not torch.cuda.is_initialized(),'parent CUDA hidden')
    return {'GPU_uuid':uuid,'total_MiB':int(total),'free_MiB':int(free),'minimum_free_MiB':22528,'allocator_bytes':c['allocator_bytes'],'spare_policy':'no other compute jobs; free>=22GiB before21GiB limit','family_bytes':family,'external_bytes_including_IPC':private+external,'allocated_backend':'native','allocator_env':c['allocator_env_now'],'CUDA_initialized':False}

class Meter:
    def __init__(self,out,cuda,task,start):
        self.out,self.cuda,self.task,self.start=out,cuda,task,start;self.forward=0;self.updates=0;self.backward=0;self.reserved_updates=0;self.peak=0;self.max_reserved=0;self.samples=[];self.update=0;self.micro=0
    def guard(self):
        need(time.monotonic()-self.start<600,'attempt600s limit');need(size(self.task)+size(root_for(self.task))<2<<30,'2GiB diagnostic storage')
        if self.cuda:
            need(time.time()-read(self.task/'ATTEMPT_001_ONCE.json')['wall_start_epoch']<2400,'debug total wall2400s')
            import torch
            self.peak=max(self.peak,torch.cuda.max_memory_allocated());self.max_reserved=max(self.max_reserved,torch.cuda.max_memory_reserved());need(self.peak<=21<<30,'allocated21GiB limit')
    def publish(self):atomic(self.out/'ledger.json',{'forward':self.forward,'backward':self.backward,'actual_updates':self.updates,'reserved_updates':self.reserved_updates,'seconds':time.monotonic()-self.start,'peak_allocated_bytes':self.peak,'peak_reserved_bytes':self.max_reserved,'diagnostic_only':True})
    def memory(self,tag):
        if not self.cuda:return {'tag':tag,'CPU_only':True}
        import torch
        try:
            free,total=torch.cuda.mem_get_info();stats=torch.cuda.memory_stats();v={'tag':tag,'update':self.update,'micro':self.micro,'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),'free':free,'total':total,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'inactive_split':stats.get('inactive_split_bytes.all.current'),'num_alloc_retries':stats.get('num_alloc_retries'),'num_ooms':stats.get('num_ooms')}
            self.samples.append(v);self.peak=max(self.peak,v['peak_allocated']);self.max_reserved=max(self.max_reserved,v['peak_reserved']);atomic(self.out/'MEMORY.json',self.samples);return v
        except BaseException as e:
            v={'tag':tag,'sample_failed':repr(e),'values':'UNKNOWN'};self.samples.append(v);atomic(self.out/'MEMORY.json',self.samples);return v
    def before_forward(self,module,args):
        self.guard();need(self.forward<120,'attempt forward cap');self.forward+=1;self.micro=(self.forward-1)%2;self.publish()
        if self.update==52:self.memory('before_forward')
    def after_forward(self,module,args,out):
        import torch
        need(str(out['pred_logits'].dtype)==('torch.bfloat16' if self.cuda else 'torch.float32'),'native precision');need(module.last_boundary_dtype=='torch.float32','boundary precision')
        if self.update==52:self.memory('after_forward')
def witnesses(engine):
    # Same diagnostic hooks as the original engineering-S training stage; no returned gradients.
    seen={'head':False,'F8':False,'backbone':False};handles=[]
    def observe(name):
        def f(g):
            if not seen[name]:seen[name]=bool(g.isfinite().all() and (g!=0).any())
        return f
    for p in engine.model.backbone.parameters():
        if p.requires_grad:handles.append(p.register_hook(observe('backbone')))
    for p in engine.model.boundary.parameters():handles.append(p.register_hook(observe('head')))
    def feature(module,args,value):value[0].register_hook(observe('F8'))
    handles.append(engine.model.encoder.register_forward_hook(feature));return handles,seen
def history(meter,out,tag):
    if not meter.cuda:return
    import torch
    try:
        snap=torch.cuda.memory._snapshot();p=out/(tag+'-allocator-snapshot.json');write(p,snap)
        traces=snap.get('device_traces',[[]]);events=traces[0] if traces else []
        large=[x for x in events if x.get('size',0)>=1<<30 or x.get('action')=='oom']
        write(out/(tag+'-LARGE_ALLOCATIONS.json'),{'events':large,'all_trace_events':len(events),'original_request_rounded_GiB':1.83,'not_assumed_same_exact_size':True,'snapshot':reference(p)})
    except BaseException as e:write(out/(tag+'-MEMORY_UNKNOWN.json'),{'error':repr(e),'values':'UNKNOWN'})
def replay(out,engine,binding,data_iter,expected_rows,cuda,start,end=60):
    import torch
    from sparse_rtdetr.research import sabl_formal_state as state
    meter=Meter(out,cuda,out.parent,start);handles,seen=witnesses(engine);handles+=[engine.model.register_forward_pre_hook(meter.before_forward),engine.model.register_forward_hook(meter.after_forward)]
    original_backward=torch.Tensor.backward;original_step=engine.optimizer.step;status='STOP';error=None;matched=0;history_on=False;checkpoint=None
    def backward(t,*args,**kw):
        meter.guard();need(meter.backward<120,'backward/update hard bound');meter.backward+=1;meter.publish()
        if meter.update==52:meter.memory('before_backward')
        try:return original_backward(t,*args,**kw)
        finally:
            if meter.update==52:meter.memory('after_backward_or_exception')
    def optimizer_step(*args,**kw):
        meter.guard();need(meter.updates<60,'optimizer update cap');meter.updates+=1;meter.publish();return original_step(*args,**kw)
    engine.optimizer.step=optimizer_step;torch.Tensor.backward=backward
    try:
        with (out/'REPLAY.jsonl').open('x') as log:
            for images,targets,identity in data_iter:
                u=engine.completed_updates+1;meter.update=u
                if u>end:break
                row=expected_rows[u-1] if u<=len(expected_rows) else None
                if row:need(digest(identity)==row['input_SHA'],'STOP_TRAJECTORY_INPUT_MISMATCH u'+str(u))
                if cuda and u==52:

                    try:torch.cuda.memory._record_memory_history(enabled='all',context='state',stacks='python',max_entries=8192);history_on=True
                    except BaseException as e:write(out/'HISTORY_START_UNKNOWN.json',{'error':repr(e),'observation':'UNKNOWN'})
                    meter.memory('window52_before');write(out/'TARGET_INPUT.json',{'identity_SHA':digest(identity),'identity':identity,'original_u52_identity_saved':False,'chosen_by_original_seed_order':True})
                need(meter.updates<60,'attempt actual update cap');meter.reserved_updates+=1;meter.publish()
                result=engine.train_window(images,targets,capture=False);meter.guard();meter.publish()
                if row:need(digest(reduced(result))==row['result_SHA'],'STOP_TRAJECTORY_RESULT_MISMATCH u'+str(u));matched+=1
                log.write(json.dumps({'update':u,'input_match':True if row else None,'result_match':True if row else None,'input_SHA':digest(identity),'result_SHA':digest(reduced(result)),'loss':result['loss'],'gradient_norm':result['gradient_norm_before_clip'],'memory':meter.memory('window_complete')},allow_nan=False)+'\n');log.flush()
                if u==52 and cuda:history(meter,out,'u52');torch.cuda.memory._record_memory_history(enabled=None);history_on=False
                if u==end:
                    checkpoint=state.save(engine,binding,out/('checkpoint-diagnostic-u%03d.pt'%u));status='PASS_U52_THROUGH_U60' if cuda else 'PASS_CPU_REPLAY';break
            else:raise RuntimeError('missing required window')
        need(matched==len(expected_rows),'missing exact replay records');need(engine.completed_updates==end and engine.ema.updates==end and all(p.grad is None for p in engine.model.parameters()),'complete diagnostic boundary')
    except BaseException as e:
        error=repr(e);status='STOP_TRAJECTORY_MISMATCH' if 'STOP_TRAJECTORY_' in str(e) else ('STOP_CUDA_OOM' if isinstance(e,torch.OutOfMemoryError) else 'STOP_DIAGNOSTIC_EXCEPTION');(out/'stderr.txt').write_text(traceback.format_exc())
    finally:
        meter.memory('final');meter.publish()
        if history_on:history(meter,out,'failure');torch.cuda.memory._record_memory_history(enabled=None)
        torch.Tensor.backward=original_backward;engine.optimizer.step=original_step
        for h in handles:h.remove()
    r={'status':status,'error':error,'matched_prior_updates':matched,'available_prior_updates':len(expected_rows),'comparison':'exact input and RESULT_KEYS SHA; elapsed excluded','full_prior_gradient_tensors_or_intermediate_states_compared':False,'original_u52_result_available':False,'counts':{'forward':meter.forward,'backward':meter.backward,'actual_updates':meter.updates,'reserved_updates':meter.reserved_updates},'seconds':time.monotonic()-start,'peak_allocated_bytes':meter.peak,'peak_reserved_bytes':meter.max_reserved,'gradient_witness':seen,'last_complete_update':engine.completed_updates,'saved_complete_boundary':checkpoint,'diagnostic_state_for_formal':False,'no_automatic_formal_training':True,'scientific_quality':'UNKNOWN'}
    write(out/'RESULT.json',r)
    if status.startswith('PASS'):write(out/'CLOSED.json',{'result':reference(out/'RESULT.json'),'diagnostic_only':True})
    else:write(out/'STOP.json',{'result':reference(out/'RESULT.json'),'no_blind_retry':True})
    return r
def run(a):
    out=Path(a.out);out.mkdir();start=time.monotonic();c=validate(read(a.definition));m=source(a.manifest);need(c['commit']==m['commit'] and c['manifest_SHA']==sha(a.manifest),'definition/source binding');need(sha(a.definition)==a.definition_sha,'child definition binding');need(read(a.authorization)=={'scope':SCOPE,'definition_SHA':sha(a.definition),'execution_manifest_SHA':sha(a.manifest),'attempt':1,'authorize_once':True},'debug authority')
    task=Path(c['task']);verify_ipc(task);original=exact(c['original']['original_contract']);os.environ['CUDA_VISIBLE_DEVICES']=original['metadata']['gpu_uuid']
    cache=out/'compiler-cache'
    for k,s in [('TORCHINDUCTOR_CACHE_DIR','inductor'),('TRITON_CACHE_DIR','triton'),('CUDA_CACHE_PATH','cuda'),('TORCH_EXTENSIONS_DIR','extensions'),('XDG_CACHE_HOME','xdg')]:
        p=cache/s;p.mkdir(parents=True,exist_ok=True);os.environ[k]=str(p)
    import torch
    import sabl_formal as formal
    from sparse_rtdetr.research import sabl_formal_state as state,sabl_formal_data as data
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    need(torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported(),'same CUDA/BF16');torch.cuda.set_per_process_memory_fraction((21<<30)/torch.cuda.get_device_properties(0).total_memory,0);torch.cuda.reset_peak_memory_stats()
    write(out/'RUNTIME.json',{'only_intervention':'allocator fraction 18GiB to21GiB','allocator_backend':torch.cuda.get_allocator_backend(),'allocator_environment':{k:os.environ.get(k) for k in ['PYTORCH_CUDA_ALLOC_CONF','PYTORCH_ALLOC_CONF']},'allocator_fraction':(21<<30)/torch.cuda.get_device_properties(0).total_memory,'device_total_bytes':torch.cuda.get_device_properties(0).total_memory,'precision':'unchanged nativeBF16/boundaryFP32','deterministic':torch.are_deterministic_algorithms_enabled(),'CUDA_before_restoring':True,'debug_state_may_not_be_used_for_formal':True})
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('attempt600s')));signal.alarm(600)
    engine,binding,shared,head=formal.fresh(original,'S','cuda');cp=c['original']['checkpoint'];audit=state.restore(engine,binding,cp['path'],cp['sha256']);need(audit['full_state_exact'] and engine.completed_updates==0 and binding==c['original']['binding'],'original e000 exact full restore');write(out/'RESTORE.json',audit)
    ds=data.Dataset(original['metadata']['data']['train_core']);expected_rows=exact(c['expected'])
    try:r=replay(out,engine,binding,data.windows(ds,1),expected_rows,True,start)
    finally:signal.alarm(0)
    print(json.dumps({'status':r['status'],'counts':r['counts'],'matched_prior_updates':r['matched_prior_updates'],'seconds':r['seconds']}));return 0 if r['status'].startswith('PASS') else 1
def launch(a):
    c=validate(read(a.definition));m=source(a.manifest);need(c['commit']==m['commit'] and c['manifest_SHA']==sha(a.manifest),'definition/source binding');need(sha(a.definition)==a.approve_definition_SHA and sha(a.manifest)==a.approve_manifest_SHA and m['commit']==a.approve_commit,'explicit frozen identity')
    need(a.authorize_debug,'separate debug authority');need(not m['candidate_not_for_real_execution'],'candidate cannot execute GPU');task=Path(c['task']);verify_ipc(task);need(not (task/'ATTEMPT_001_ONCE.json').exists() and not (task/'attempt-001').exists(),'duplicate debug attempt')
    res=resources(c);write(task/'FRESH_RESOURCE_001.json',res)
    for name in ['checkpoint','sidecar','original_contract','old_manifest','old_STOP','old_cost','old_log']:ref_exact(c['original'][name])
    need(read(task/'DEBUG_BUDGET.json')['approved']==CAPS,'debug budget binding')
    auth=task/'authorization-attempt-001.json';write(auth,{'scope':SCOPE,'definition_SHA':sha(a.definition),'execution_manifest_SHA':sha(a.manifest),'attempt':1,'authorize_once':True})
    argv=[sys.executable,'-B',str(ROOT/'tools/sabl_oom_replay.py'),'run','--definition',a.definition,'--definition-sha',sha(a.definition),'--manifest',a.manifest,'--authorization',str(auth),'--out',str(task/'attempt-001')]
    write(task/'ATTEMPT_001_ONCE.json',{'argv':argv,'wall_start_epoch':time.time(),'no_blind_retry':True,'max_attempts':2});start=time.monotonic();p=subprocess.Popen(argv,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    try:stdout,stderr=p.communicate(timeout=605)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid,signal.SIGTERM)
        try:stdout,stderr=p.communicate(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);stdout,stderr=p.communicate()
    elapsed=time.monotonic()-start;(task/'attempt-001.stdout').write_bytes(stdout);(task/'attempt-001.stderr').write_bytes(stderr)
    ledger=read(task/'attempt-001/ledger.json') if (task/'attempt-001/ledger.json').exists() else {'forward':0,'actual_updates':0,'backward':0};need(elapsed<=1200,'total diagnostic GPU phase');need(ledger['forward']<=240 and ledger['actual_updates']<=120,'independent global debug cap')
    prior=c['formal_cumulative_before'];cost={'scope':'INDEPENDENT_ENGINEERING_DIAGNOSTIC_NOT_FORMAL','approved_debug_caps':CAPS,'attempts_used':1,'new':{'forward':ledger['forward'],'actual_updates':ledger['actual_updates'],'GPU_stage_seconds_conservative_process_wall':elapsed},'debug_remaining':{'attempts':1,'forward':240-ledger['forward'],'actual_updates':120-ledger['actual_updates'],'GPU_seconds':1200-elapsed},'formal_remaining_reserved_unchanged':c['formal_remaining_reserved'],'previous_all_research':prior,'cumulative_all_research':{'forward':prior['forward']+ledger['forward'],'GPU_stage_seconds':float(decimal.Decimal(str(prior['GPU_stage_seconds']))+decimal.Decimal(str(elapsed)))},'old_cost_overwritten':False,'failed_cost_refunded':False,'debug_unused_budget_for_formal':False,'second_attempt_requires_specific_evidence_and_new_frozen_entry':True}
    write(task/'COST_ADDITION.json',cost);write(task/'PROCESS_EXIT_001.json',{'returncode':p.returncode,'process_wall_seconds':elapsed,'ledger':ledger,'no_automatic_second_attempt':True});print(json.dumps({'returncode':p.returncode,'cost':cost,'result':read(task/'attempt-001/RESULT.json') if (task/'attempt-001/RESULT.json').exists() else 'UNKNOWN_NO_RESULT'}));return 0 if p.returncode==0 else 1
def fixture(a):
    from check_sabl_oom_replay import run_fixture
    return run_fixture(a)

def main():
    import argparse
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('fixture');q.add_argument('--out',required=True);q.add_argument('--fault',choices=['none','input','result'],default='none')
    q=s.add_parser('prepare');q.add_argument('--manifest',required=True);q.add_argument('--out',required=True)
    q=s.add_parser('launch')
    for n in ['definition','manifest','approve-definition-SHA','approve-manifest-SHA','approve-commit']:q.add_argument('--'+n,required=True)
    q.add_argument('--authorize-debug',action='store_true')
    q=s.add_parser('run')
    for n in ['definition','definition-sha','manifest','authorization','out']:q.add_argument('--'+n,required=True)
    a=p.parse_args()
    try:return globals()[a.command](a)
    except BaseException as e:
        if a.command=='run':
            out=Path(a.out)
            if out.is_dir() and not (out/'STOP.json').exists():
                sample={'values':'UNKNOWN'}
                try:
                    import torch
                    if torch.cuda.is_initialized():
                        free,total=torch.cuda.mem_get_info();sample={'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),'peak_allocated':torch.cuda.max_memory_allocated(),'free':free,'total':total}
                except BaseException:pass
                write(out/'STOP.json',{'error':repr(e),'setup_or_teardown_exception':True,'resource_sample':sample,'no_blind_retry':True});(out/'stderr.txt').write_text(traceback.format_exc())
        raise
if __name__=='__main__':raise SystemExit(main())
