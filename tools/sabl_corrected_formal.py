"""Corrected S/A formal wiring, reusing existing training, evaluation and cost methods."""
import os,sys,time,json,math,copy,gc,subprocess,signal,shutil,traceback,types,decimal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
for k,v in {'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_THREADING_LAYER':'GNU','PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1'}.items():os.environ[k]=v
import sabl_formal as old
from sparse_rtdetr.research import sabl_formal_contract as oldct
from sabl_corrected_short import ref_exact
from sabl_ipc import root_for,verify as verify_ipc,owned_family_bytes
need,read,sha,digest,reference=oldct.need,oldct.read,oldct.sha,oldct.digest,oldct.reference
write,atomic,closed,stage_close=old.write,old.atomic,old.closed,old.stage_close
BASE='2c85c22d5738f8dc84c7527c3408602561baae09'
BASE_MANIFEST='5f9e5a2b977ad3e5b2881324eadf670f4e2b2ef82bb1e0ef79aa2e74047fd69d'
SCHEMA='SABL_CORRECTED_SA_FORMAL24_V3_IPC21'
CPU=oldct.CPU;GPU=oldct.GPU;D=decimal.Decimal
POLICY={**oldct.POLICY,'training_arms':['S','A'],'cost_arms':['N','S','A'],'N24_train_reused':True,'new_forwards':29842,'train_forwards':29184,'endpoint_forwards':274,'cost_forwards':384,'actual_updates':14592}
CAPS={'new_forward':29842,'new_GPU_seconds':'57131.83263277169','new_wall_seconds':86400,'allocated_bytes':21<<30,'new_storage_bytes':12<<30,'family_storage_bytes':30<<30,'return_bytes':4<<20}
EXISTING={'forward':61871,'GPU_seconds':'94472.36945352436','wall_seconds':172800,'family_storage_bytes':30<<30}
FAULTS=['none','worker_exception','forward_exhausted','nonfinite','bad_endpoint','missing_close','bad_input']
FIXTURE=ROOT/'tests/fixtures/sabl_corrected_formal_base.json'
WS=Path('/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace');EX=WS/'experiments/localization-distribution'

def exact(r):return read(ref_exact(r)['path'])

def computation(manifest):
    from r18_deploy import verify_release
    m=read(manifest);verify_release(ROOT,sha(manifest));f=read(FIXTURE)
    need(Path(manifest).resolve()==ROOT/'RELEASE_MANIFEST.json','actual release manifest')
    need(m['computation_base']==BASE and m['computation_base_manifest']==BASE_MANIFEST,'compute base')
    actual=[r for r in m['files'] if r['path'].startswith(('src/','configs/','vendor/'))]
    need(actual==f['scientific_files'],'scientific dependency drift')
    for r in f['reused_execution_files']:need(sha(ROOT/r['path'])==r['sha256'],'reused implementation drift '+r['path'])
    return m

def profile(c):
    return {'epochs':3,'windows':1,'engineering_epochs':2,'images':6,'fixed':4} if c['mode']==CPU else oldct.profile(c)

def forwards(c):
    p=profile(c);return 2*p['epochs']*p['windows']*2+2*((p['images']+3)//4)+3*8*(p['fixed']//4)

def validate(c):
    need(set(c)=={'schema','mode','policy','caps','existing_limits','prior','requested_addition','requested_cumulative','planned_new_forwards','arms','N_checkpoint_SHA','manifest_sha256','commit','attempt_root','metadata','reuse_N24','short_proofs','wall','fixture','authorization_required','ipc','runtime','independent_debug'},'contract fields')
    need(c['schema']==SCHEMA and c['mode'] in (CPU,GPU),'corrected formal scope')
    need(digest(c['policy'])==digest(POLICY) and digest(c['caps'])==digest(CAPS) and digest(c['existing_limits'])==digest(EXISTING),'policy/caps changed')
    need(c['runtime']==RUNTIME,'runtime allocator configuration')
    need(c['arms']==['S','A'] and c['N_checkpoint_SHA']==oldct.N_SHA and c['authorization_required'] is True,'fresh S/A only')
    need(c['planned_new_forwards']==forwards(c),'forward budget')
    need(c['fixture']['fault'] in FAULTS and (c['mode']==CPU or c['fixture']['fault']=='none'),'production fixture forbidden')
    prior=c['prior'];need(type(prior['forward']) is int and prior['forward']>=0 and math.isfinite(prior['GPU_seconds']) and prior['GPU_seconds']>=0,'prior accounting')
    cumulative={'forward':prior['forward']+CAPS['new_forward'],'GPU_seconds':str(D(str(prior['GPU_seconds']))+D(CAPS['new_GPU_seconds']))}
    request={'forward':max(0,cumulative['forward']-EXISTING['forward']),'GPU_seconds':str(max(D(0),D(cumulative['GPU_seconds'])-D(EXISTING['GPU_seconds'])))}
    need(c['requested_addition']==request and c['requested_cumulative']==cumulative,'exact expansion calculation')
    need(Path(c['attempt_root']).is_absolute(),'absolute attempt')
    need(c['ipc']=={'root':str(root_for(Path(c['attempt_root']).parent)),'before_new_Python':True,'owner':'task_exclusive'},'private IPC contract')
    if c['mode']==GPU:
        f=read(FIXTURE);actual=exact(f['latest_cost'])['cumulative']
        need(prior=={'forward':actual['forward'],'GPU_seconds':actual['GPU_stage_seconds']},'authenticated prior accounting')
        need(c['metadata'] is not None and c['reuse_N24'] is not None and c['wall']==wall_account(),'real bindings/wall')
        need(c['short_proofs']==f['short_proofs'] and c['reuse_N24']['scores']==f['N24_scores'],'accepted proof/reuse bindings')
        need(prior['forward']==32133 and c['requested_addition']=={'forward':104,'GPU_seconds':'0'},'latest formal family ledger identity')
        need(c['independent_debug']==debug_binding(),'independent diagnostic identity/budget isolation')
    else:need(c['independent_debug'] is None,'CPU not real diagnostic')
    return c

def wall_account():
    f=read(FIXTURE);w=copy.deepcopy(f['wall_reconciliation'])
    for item in f['wall_terminal_refs']:
        v=exact(item['reference']);key=item['seconds_key']
        if key:need(0<=v[key]<item['full_wall_allowance'],'historical wall envelope '+item['name'])
        else:
            need(v['returncode']==0 and v['cost']['new']['conservative_GPU_stage_seconds']<3600,'old evaluation exit envelope')
    formal=w['rows'][1];ref_exact(formal['authorization']);ref_exact(formal['exit'])
    dt=Path(formal['exit']['path']).stat().st_mtime-Path(formal['authorization']['path']).stat().st_mtime
    need(0<dt<=formal['upper_seconds'] and formal['recorded_entry_seconds']<=dt,'old formal wall interval changed')
    need(sum(w['forward_counts_reconciled'])==32253 and w['formal_cumulative_forward']==32133 and w['independent_debug_forward']==120 and w['historical_wall_upper_bound_seconds']+CAPS['new_wall_seconds']<=172800,'cumulative wall insufficiency')
    proof=exact(f['latest_OOM_whole_wall_proof']);need(proof['old_latest_OOM_whole_call_seconds']+60<600 and w['remaining_wall_lower_bound_seconds']==123357,'latest complete wall envelope')
    return w

def debug_binding():
    f=read(FIXTURE);v={k:exact(r) for k,r in f['independent_debug'].items()};cost=v['COST_ADDITION.json'];r=v['attempt-001/RESULT.json'];done=v['DEBUG_CLOSED.json']
    need(cost['scope']=='INDEPENDENT_ENGINEERING_DIAGNOSTIC_NOT_FORMAL' and cost['debug_unused_budget_for_formal'] is False,'debug distinct scope')
    need(r['status']=='PASS_U52_THROUGH_U60' and r['matched_prior_updates']==51 and r['counts']=={'forward':120,'backward':120,'actual_updates':60,'reserved_updates':60} and r['diagnostic_state_for_formal'] is False,'bounded replay evidence')
    need(done['status']=='CLOSED_ENGINEERING_DIAGNOSTIC_NO_FOLLOWUP' and not done['second_attempt_executed'] and done['unused_debug_allowance_not_for_formal'],'debug closed')
    need(cost['previous_all_research']['forward']==32133 and cost['new']['forward']==120 and cost['new']['actual_updates']==60,'debug lineage')
    need(cost['formal_remaining_reserved_unchanged']=={'forward':29738,'GPU_stage_seconds':57131.83263277169},'debug did not spend formal reserve')
    return {'references':f['independent_debug'],'spent':cost['new'],'unused_allowance_for_formal':False,'state_for_formal':False}

def short_evidence():
    f=read(FIXTURE)
    actual_costs=[reference(p) for p in sorted(EX.glob('r18-sabl-*/COST_ADDITION.json'))]
    need(actual_costs==f['known_family_cost_files'],'new/changed family ledger; stale prior forbidden')
    proof={k:exact(v) for k,v in f['short_proofs'].items()}
    r=proof['attempt-short-001/slim/RESULT.json'];a=proof['TERMINAL_AUDIT.json'];c=proof['COST_ADDITION.json']
    need(a['status']=='VERIFIED_PASS_SHORT_ENGINEERING_ONLY' and a['commit']==BASE and a['manifest_sha256']==BASE_MANIFEST,'short source')
    need(r['CPU_synthetic'] is False and r['ledger']=={'forward':12,'actual_updates':6,'reserved_updates':6},'short exact budget')
    need(proof['attempt-short-001/CLOSED.json']['result_sha256']==f['short_proofs']['attempt-short-001/slim/RESULT.json']['sha256'],'short closure')
    need(c['cumulative']['forward']==32029 and c['remaining']['forward']==12560,'short cumulative')
    for arm in ['S','A']:need(r['arms'][arm]['replay_full_state_exact'] and r['arms'][arm]['loss_output_gradient_DN_exact'],'accepted compute/replay')
    ipc=exact(f['previous_IPC_latest_cost']);stop=exact(f['previous_IPC_latest_STOP']);exit=exact(f['previous_IPC_latest_exit'])
    need(ipc['previous']['forward']==c['cumulative']['forward'] and ipc['previous']['GPU_stage_seconds']==c['cumulative']['GPU_stage_seconds'],'prior IPC lineage')
    need(ipc['new']['forward']==ipc['new']['actual_updates']==0 and stop['stage']=='engineering-S' and exit['returncode']==1,'IPC STOP preserved')
    latest=exact(f['latest_cost']);stop=exact(f['latest_STOP']);exit=exact(f['latest_exit'])
    need(latest['previous']==ipc['cumulative'],'OOM lineage')
    need(latest['new']['forward']==104 and latest['new']['actual_updates']==51 and stop['error_type']=='OutOfMemoryError' and exit['returncode']==1,'original OOM STOP preserved')
    need(latest['cumulative']['forward']==32133 and latest['remaining']['forward']==29738,'latest formal remaining forward')
    need(D(str(latest['cumulative']['GPU_stage_seconds']))+D(CAPS['new_GPU_seconds'])<=D(EXISTING['GPU_seconds']),'latest precise GPU balance')
    debug_binding()
    return proof

RUNTIME={'allocator_backend':'native','allocator_configuration':'default_unset','preinit_free_min_bytes':22<<30,'postinit_free_min_bytes':(21<<30)+(512<<20),'allocator_fraction_bytes':21<<30,'paper_cost_gates_unchanged':True}

def allocator_policy(torch):
    env={k:os.environ.get(k) for k in ['PYTORCH_CUDA_ALLOC_CONF','PYTORCH_ALLOC_CONF']}
    need(all(v in [None,''] for v in env.values()),'nondefault allocator environment')
    backend=torch.cuda.memory.get_allocator_backend();need(backend=='native','default native allocator required')
    return {'backend':backend,'environment':env}

def GPU_memory_gate(total,used,free,postinit=False):
    need(all(type(x) is int and x>=0 for x in [total,used,free]) and 0<total and used+free<=total+(16<<20),'GPU memory fields')
    need(free>=RUNTIME['postinit_free_min_bytes' if postinit else 'preinit_free_min_bytes'],'GPU runtime headroom')
    return {'total_bytes':total,'used_bytes':used,'free_bytes':free,'postinit':postinit,'minimum_free_bytes':RUNTIME['postinit_free_min_bytes' if postinit else 'preinit_free_min_bytes']}

def compute_processes(allow_self=False):
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True).strip()
    rows=[x.strip() for x in text.splitlines() if x.strip()]
    for row in rows:
        need(allow_self and int(row.split(',')[0].strip())==os.getpid(),'other GPU compute process '+row)
    return rows

def initialized_resources(torch):
    policy=allocator_policy(torch);rows=compute_processes(True);free,total=map(int,torch.cuda.mem_get_info())
    memory=GPU_memory_gate(total,total-free,free,True)
    return {'allocator':policy,'memory':memory,'compute_processes':rows,'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved(),'before_model_construction':True,'extra_model_forward':0,'used_includes_CUDA_context_and_system_usage':True}

def size(p):return sum(x.stat().st_size for x in Path(p).rglob('*') if x.is_file())

def resources(c,out):
    import torch,torchvision
    from sabl_corrected_short import no_jobs
    no_jobs()
    ps=subprocess.check_output(['ps','-eo','pid,ppid,args'],text=True)
    for line in ps.splitlines()[1:]:
        s=line.strip().split(None,2)
        if len(s)==3 and int(s[0]) not in [os.getpid(),os.getppid()] and any(x in s[2] for x in ['sabl_corrected_formal.py run','sabl_corrected_formal.py launch']):raise ValueError('related formal job')
    need(Path(sys.executable).resolve()==Path(c['metadata']['python']).resolve(),'runtime interpreter')
    env=c['metadata']['environment'];need(str(torch.__version__)==env['torch'] and str(torchvision.__version__)==env['torchvision'] and torch.version.cuda==env['cuda_build'] and torch.backends.cudnn.version()==env['cudnn'],'runtime versions')
    policy=allocator_policy(torch);processes=compute_processes()
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,pci.bus_id,memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    need(len(lines)==1,'one physical GPU');uuid,name,pci,total,used,free=[x.strip() for x in lines[0].split(',')]
    need(uuid==c['metadata']['gpu_uuid'] and name=='NVIDIA GeForce RTX 4090 D' and pci=='00000000:01:00.0'  ,'GPU identity')
    memory=GPU_memory_gate(int(total)*(1<<20),int(used)*(1<<20),int(free)*(1<<20))
    family=sum(size(p) for p in EX.glob('r18-sabl-*') if p.is_dir())
    external=sum(size(Path(p)) for p in ['/tmp/torchinductor_lyy','/home/lyy/.triton/cache','/home/lyy/.nv/ComputeCache','/home/lyy/.cache/torch/inductor','/home/lyy/.cache/triton'] if Path(p).exists())
    private,private_rows=owned_family_bytes(EX);external+=private
    need(family+external+(12<<30)<=30<<30,'family storage balance')
    need(shutil.disk_usage(out).free>=30<<30,'original fresh free disk floor30GiB')
    return {'family_bytes':family,'known_external_cache_all_bytes':external,'private_IPC_roots':private_rows,'family_available_lower_bound':(30<<30)-family-external,'new_storage_requested':12<<30,'disk_free_bytes':shutil.disk_usage(out).free,'wall':wall_account(),'allocator':policy,'GPU_memory':memory,'GPU_compute_processes':processes,'GPU_used_scope':'system/non-PyTorch baseline; CUDA not initialized by this process','GPU_uuid':uuid,'CUDA_initialized':torch.cuda.is_initialized()}

def material(c):
    from sparse_rtdetr.localization.material_paths import MANIFESTS,metadata_path,SELECTION_SHA
    from sparse_rtdetr.localization.data import read_bound,parse_raw,validate_role_roots
    from PIL import Image
    import io
    m=c['metadata'];f=read(FIXTURE)
    exact(f['original_definition']);original=exact(f['original_contract'])
    need(all(m[k]==original['metadata'][k] for k in ['parent_record','checkpoint','data','selection','python','environment','historical_N','fixed_ids']),'original material/recipe identity')
    ref_exact(m['checkpoint']);need(m['checkpoint']['sha256']==oldct.N_SHA,'parent N')
    history=m['historical_N']
    for key in ['endpoint','authentication','source','original_STOP','score_result','infer_result']:exact(history[key])
    need(not Path(history['original_CLOSED_path']).exists(),'historical N closure limitation changed')
    seen={}
    for role,spec in m['data'].items():
        need(role in ['train_core','development'] and spec['count']==MANIFESTS[role][1] and spec['manifest']['sha256']==MANIFESTS[role][0],'frozen data role/count')
        metadata_path(role,spec['manifest']['path'],spec['manifest']['sha256']);rows=exact(spec['manifest'])['records']
        need(len(rows)==spec['count'] and len({r['stable_image_id'] for r in rows})==len(rows),'member identity')
        roots=spec['roots'];validate_role_roots(role,roots['images'],roots['annotations'],roots)
        for row in rows:
            b=read_bound(Path(roots['images'])/Path(row['relative_path']).name,row['image_sha256'],row['image_size_bytes'])
            a=read_bound(Path(roots['annotations'])/Path(row['annotation_relative_path']).name,row['annotation_sha256'],row['annotation_size_bytes'])
            raw=parse_raw(a,row['stable_image_id'],expected_rows=row['raw_annotation_rows'])
            if role=='development':need(not raw['counts'].get('nonpositive_area',0),'unsupported nonpositive development GT')
            with Image.open(io.BytesIO(b)) as im:need(im.size==(row['width'],row['height']),'image dimensions');im.verify()
        seen[role]=rows
    need(set(seen)=={'train_core','development'},'only frozen data roles')
    selected=exact(m['selection']);metadata_path('selection',m['selection']['path'],SELECTION_SHA)
    ordered=sorted(seen['development'],key=lambda r:(__import__('hashlib').sha256(('loc-trace-v1/'+r['stable_image_id']).encode()).hexdigest(),r['stable_image_id']))[:64]
    need(selected['selected']==[dict(r,selection_key=__import__('hashlib').sha256(('loc-trace-v1/'+r['stable_image_id']).encode()).hexdigest()) for r in ordered] and m['fixed_ids']==[r['stable_image_id'] for r in ordered],'fixed64 identity')
    for refs in [f['N24_scores'],f['original_N_scores']]:
        for b,r in refs.items():
            v=exact(r);need(v['official']['evaluated_image_count']==548 and v['COCO_bridge']['maxDets']==[1,10,100],'reused score protocol')
    import sabl_ns_eval as evaluator
    evaluator.verify_baseline(c)
    return {'all_member_bytes_verified':{r:len(v) for r,v in seen.items()},'fixed64_verified':True,'original_N_N24_scores_reused':True,'no_model_forward':True}

def N_config_matches(saved,current):
    # Registered N-only read-only bridge: identity tag differs, all computation fields exact.
    return current.get('arm')=='r18-sabl-control-lrtb-logsigmoid-v3' and saved==dict(current,arm='r18-sabl-control-v1')

def N_model(c,out,device):
    import torch
    from sparse_rtdetr.research import sabl_control as sc
    from sparse_rtdetr.protocol_r18.model import Config
    from sparse_rtdetr.protocol_r18.engine import tensor_digest
    from sparse_rtdetr.baseline import training_v2b_checkpoint as layout
    if c['mode']==CPU:
        old.seed(device);model=sc.build('N',Config(input_hw=(128,128),physical_batch_size=1,amp_dtype='float32'))[0]
        return model.eval().requires_grad_(False).to(device),{'synthetic_untrained_N_for_cost_only':True}
    f=read(FIXTURE);ep=exact(f['N24_endpoint']);need(ep==c['reuse_N24']['endpoint'],'N24 endpoint identity')
    r=ref_exact(ep['checkpoint']);state=torch.load(r['path'],map_location='cpu',weights_only=True)
    need(state['format']=='sparse_rtdetr.sabl_formal24_fullstate_v1' and state['binding']==ep['binding'],'N24 original state/binding')
    b=ep['binding'];need(b['version']=='r18-sabl-control-v1' and b['arm']=='N' and b['release_manifest']=='1af3d80bc0b28aa384da0511715ea81083e709849bc076abb226f2e2ff82c1ab' and b['N_SHA']==oldct.N_SHA,'explicit original N v1 identity')
    need(r['sha256']=='fbf798c33649253a410c2feeac01bb62757089221709726a02493117469de921','certified N24 SHA')
    need(ep['epochs']==24 and ep['updates']==7296 and ep['EMA'] and not ep['acceptance_state_used'] and not ep['synthetic'],'N24 endpoint')
    need(state['runtime']==str(torch.__version__) and state['device']=='cuda:0','N24 runtime')
    need(N_config_matches(state['engine']['config'],sc.TrainConfig().binding()) and state['engine']['completed_updates']==7296 and state['engine']['phase']==0 and not state['engine']['failed'],'N24 engine/config')
    need(state['criterion']['arm']=='N' and state['criterion']['version']=='r18-sabl-control-v1' and state['ema']['updates']==7296 and state['ema']['decay']==.9999 and state['ema']['warmups']==2000,'N24 EMA/supervision')
    model=sc.build('N',Config())[0];current=layout._model_layout(model)
    need(state['model_layout']==current,'N-only old/new full layout')
    layout._check_model_state(state['model'],current,'N24 raw');layout._check_model_state(state['ema']['module'],current,'N24 EMA')
    for name,v in model.state_dict().items():
        if name.rsplit('.',1)[-1] in ('anchors','valid_mask','num_points_scale'):need(torch.equal(v,state['ema']['module'][name]),'N24 immutable cache')
    expected=tensor_digest(sorted(state['ema']['module'].items()));need(expected==f['N24_expected_EMA_module_SHA'],'authenticated old EMA module SHA')
    model.load_state_dict(state['ema']['module'],strict=True);need(tensor_digest(sorted(model.state_dict().items()))==expected,'N24 EMA copy exact')
    report={'N24_read_only_EMA_bridge':True,'old_version':b['version'],'current_wrapper_version':sc.VERSION,'checkpoint':r,'EMA_module_SHA':expected,'optimizer_engine_RNG_restored':False,'N_calculation_changed':False,'training_config_identity_mapping':{'old_arm':'r18-sabl-control-v1','current_arm':sc.VERSION,'all_other_fields_exact':True}}
    model.eval().requires_grad_(False);del state;gc.collect()
    if out is not None and not (out/'N24_EMA_BRIDGE.json').exists():write(out/'N24_EMA_BRIDGE.json',report)
    return model.to(device),report

def plan(a):
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);m=computation(a.manifest);synthetic=a.synthetic;f=read(FIXTURE)
    prior={'forward':17,'GPU_seconds':2.0};metadata=None;reuse=None;short=None;wall=None
    if not synthetic:
        proof=short_evidence();cost=exact(f['latest_cost']);prior={'forward':cost['cumulative']['forward'],'GPU_seconds':cost['cumulative']['GPU_stage_seconds']}
        original=exact(f['original_contract']);definition=exact(f['original_definition']);metadata=copy.deepcopy(original['metadata']);metadata['gpu_uuid']=definition['gpu_uuid']
        need(exact(f['N24_closed'])['result']['sha256']==f['N24_result']['sha256'],'N24 original closure')
        nresult=exact(f['N24_result']);reuse={'endpoint':exact(f['N24_endpoint']),'scores':f['N24_scores'],'ordered_training_input_SHA':nresult['training_input_SHA'],'N24_stage_closed':True,'original_N_history_status':'AUTHENTICATED_OUTPUT_NOT_ORIGINAL_STAGE_CLOSURE'}
        short=f['short_proofs'];wall=wall_account()
    cumulative={'forward':prior['forward']+29842,'GPU_seconds':str(D(str(prior['GPU_seconds']))+D(CAPS['new_GPU_seconds']))}
    request={'forward':max(0,cumulative['forward']-EXISTING['forward']),'GPU_seconds':str(max(D(0),D(cumulative['GPU_seconds'])-D(EXISTING['GPU_seconds'])))}
    c={'schema':SCHEMA,'mode':CPU if synthetic else GPU,'policy':POLICY,'caps':CAPS,'existing_limits':EXISTING,'prior':prior,'requested_addition':request,'requested_cumulative':cumulative,'planned_new_forwards':0,'arms':['S','A'],'N_checkpoint_SHA':oldct.N_SHA,'manifest_sha256':sha(a.manifest),'commit':m['commit'],'attempt_root':str(Path(a.attempt_root).resolve()),'metadata':metadata,'reuse_N24':reuse,'short_proofs':short,'wall':wall,'fixture':{'fault':a.fault},'authorization_required':True,'ipc':{'root':str(root_for(Path(a.attempt_root).resolve().parent)),'before_new_Python':True,'owner':'task_exclusive'},'runtime':RUNTIME,'independent_debug':None if synthetic else debug_binding()}
    c['planned_new_forwards']=forwards(c);validate(c)
    if not synthetic:
        need(a.fault=='none','no production fixture')
        report=material(c);resource=resources(c,out.parent)
        import torch;need(not torch.cuda.is_initialized(),'CPU review must not initialize CUDA')
        bridge=exact(f['N24_bridge_proof']);need(bridge['checkpoint']['sha256']==reuse['endpoint']['checkpoint']['sha256'] and bridge['EMA_module_SHA']==f['N24_expected_EMA_module_SHA'] and bridge['model_forward']==0 and not bridge['CUDA_initialized_after_loader'],'authenticated N24 bridge reused without detector loading')
        write(out.parent/'REVIEW.json',{'status':'PASS_CPU_FORMAL_CONTRACT_REVIEW','materials':report,'resources':resource,'N24_bridge':bridge,'execution_authorized':False,'extra_required':request,'no_GPU_forward':True})
    write(out,c);return 0

def wire(c):
    # Adapter is local to this execution module. No global contract/code mutation.
    adapter=types.SimpleNamespace(**{k:getattr(oldct,k) for k in dir(oldct) if not k.startswith('__')})
    adapter.ARMS=('S','A');adapter.profile=profile;adapter.exact=exact;old.ct=adapter
    endpoint_original=old.endpoint
    def endpoint(contract,arm,out,device):
        if arm=='N':
            model,bridge=N_model(contract,out,device)
            return model,read(FIXTURE)['N24_endpoint'] if contract['mode']==GPU else {'SYNTHETIC_UNTRAINED_N':True}
        return endpoint_original(contract,arm,out,device)
    old.endpoint=endpoint
    return adapter

class Meter:
    def __init__(self,c,out,start,cuda):
        self.c=c;self.out=out;self.start=start;self.cuda=cuda;self.telemetry=old.Telemetry(cuda);self.stage='admission';self.stage_count=0
        self.ledger={'inherited':c['prior'],'new_forward':0,'actual_updates':0,'reserved_updates':0,'total_forward':c['prior']['forward'],'GPU_stage_seconds':c['prior']['GPU_seconds'],'stage_forwards':{},'CPU_fixture':c['mode']==CPU}
        self.limit=1 if c['fixture']['fault']=='forward_exhausted' else forwards(c);self.publish()
    def elapsed(self):return time.monotonic()-self.start
    def publish(self):
        self.ledger['GPU_stage_seconds']=self.c['prior']['GPU_seconds']+(self.elapsed() if self.c['mode']==GPU else 0)
        self.ledger['CPU_fixture_seconds']=self.elapsed() if self.c['mode']==CPU else None;atomic(self.out/'ledger.json',self.ledger)
    def guard(self):
        need(self.elapsed()<(600 if self.c['mode']==CPU else float(self.c['caps']['new_GPU_seconds'])),'new stage deadline')
        need(self.elapsed()<86400,'new wall deadline')
        root=Path(self.c['attempt_root']).parent;private=size(root_for(root));need(size(root)+private<12<<30,'new formal storage including IPC')
        if self.c['mode']==GPU:
            need(self.c['wall']['historical_wall_upper_bound_seconds']+self.elapsed()<=172800,'cumulative wall')
            baseline=read(root/'fresh-launch-admission/RESOURCES.json')
            need(baseline['family_bytes']+baseline['known_external_cache_all_bytes']+size(root)+private<=30<<30,'family storage')
        if self.cuda:need(self.telemetry.observe_peak()<=self.c['caps']['allocated_bytes'],'allocated cap')
    def begin(self,stage):self.guard();self.stage=stage;self.stage_count=self.ledger['new_forward'];self.telemetry.begin(stage)
    def finish(self):self.guard();self.telemetry.finish();self.ledger['stage_forwards'][self.stage]=self.ledger['new_forward']-self.stage_count;self.publish()
    def forward(self,*_):
        self.guard();need(self.ledger['new_forward']<self.limit,'new forward cap');self.ledger['new_forward']+=1;self.ledger['total_forward']=self.c['prior']['forward']+self.ledger['new_forward'];self.publish()
    def update(self,done=False):
        key='actual_updates' if done else 'reserved_updates';need(self.ledger[key]<2*profile(self.c)['epochs']*profile(self.c)['windows'],'update cap');self.ledger[key]+=1;self.publish()

def score(c,out,meter):
    from collections import Counter
    from sparse_rtdetr.research.sabl_formal_data import development
    from sparse_rtdetr.r0.evaluation import score_budget
    folder=out/'score';folder.mkdir();meter.begin('score');rows,reader=development(c);truth=[reader.raw(row) for row in rows];refs={}
    for arm in ['S','A']:
        src=closed(out/('infer-'+arm),c);need(src['image_ids']==[r['stable_image_id'] for r in rows],'score IDs')
        need(exact(src['endpoint'])==exact(closed(out/('train-'+arm),c)['endpoint']),'score endpoint binding');refs[arm+'24']={}
        for budget in ['600','300']:
            pred=exact(src['predictions'][budget]);allowed=set(src['image_ids'])
            need(all(p['image_id'] in allowed for p in pred) and all(n<=int(budget) for n in Counter(p['image_id'] for p in pred).values()),'score membership/budget')
            r=score_budget(rows,truth,pred,str(ROOT));r['PR26_binding']={'contract':digest(c),'endpoint':src['endpoint'],'arm':arm,'budget':int(budget),'image_ids_SHA':digest(src['image_ids']),'predictions':src['predictions'][budget],'invalid_ordinary_queries':src['invalid_ordinary_queries']}
            p=folder/(arm+'-'+budget+'.json');write(p,r);refs[arm+'24'][budget]=reference(p);meter.guard()
    if c['mode']==CPU:refs['N24']=dict(refs['S24']);refs['N_original']=dict(refs['S24'])
    else:
        refs['N24']=c['reuse_N24']['scores'];refs['N_original']=c['metadata']['historical_N']['scores']
        for entries in [refs['N24'],refs['N_original']]:
            for ref in entries.values():exact(ref)
    stage_close(folder,{'status':'PASS','scores':refs,'new_score_groups':4,'evaluator_calls':8,'N24_recomputed':False,'original_N_recomputed':False,'CPU_historical_scores_are_explicit_alias_fixture':c['mode']==CPU},c);meter.finish()

def run(a):
    out=Path(a.out);out.mkdir();start=time.monotonic();meter=None
    try:
        need(sha(a.contract)==a.contract_sha,'child contract bytes binding');c=validate(read(a.contract));m=computation(a.manifest);need(c['manifest_sha256']==sha(a.manifest) and c['commit']==m['commit'] and str(out.resolve())==c['attempt_root'],'run binding')
        need(read(a.authorization)==authority(c),'separate execution/budget authority absent');synthetic=c['mode']==CPU
        need(synthetic or not m['candidate_not_for_real_execution'],'candidate no GPU')
        if synthetic:os.environ['CUDA_VISIBLE_DEVICES']=''
        else:
            short_evidence();need((out.parent/'fresh-launch-admission/RESOURCES.json').exists(),'fresh admission receipt');os.environ['CUDA_VISIBLE_DEVICES']=c['metadata']['gpu_uuid']
        ipc=verify_ipc(out.parent);write(out/'IPC_ENVIRONMENT.json',ipc)
        cache=out.parent/'compiler-cache'
        for k,s in [('TORCHINDUCTOR_CACHE_DIR','inductor'),('TRITON_CACHE_DIR','triton'),('CUDA_CACHE_PATH','cuda'),('TORCH_EXTENSIONS_DIR','extensions'),('XDG_CACHE_HOME','xdg')]:
            p=cache/s;p.mkdir(parents=True,exist_ok=True);os.environ[k]=str(p)
        import torch,torchvision
        torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
        need(torch.__version__.split('+')[0]=='2.4.1' and torchvision.__version__.split('+')[0]=='0.19.1','runtime');device='cpu' if synthetic else 'cuda'
        if not synthetic:
            need(torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported(),'CUDA/BF16');allocator_policy(torch);torch.cuda.set_per_process_memory_fraction(c['caps']['allocated_bytes']/torch.cuda.get_device_properties(0).total_memory,0)
            write(out/'RUNTIME_RESOURCES.json',initialized_resources(torch))
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('formal deadline')))
        for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:(_ for _ in ()).throw(RuntimeError('external termination; STOP')))
        signal.alarm(600 if synthetic else math.ceil(float(c['caps']['new_GPU_seconds'])));adapter=wire(c);meter=Meter(c,out,start,None if synthetic else torch.cuda)
        write(out/'CONSUMED.json',{'scope':SCHEMA,'contract':digest(c),'CPU_synthetic':synthetic,'no_retry':True})
        if c['fixture']['fault']=='worker_exception':raise RuntimeError('SYNTHETIC_WORKER_EXCEPTION')
        for arm in ['S','A']:old.train(c,arm,True,out,meter,device)
        initial=[read(out/('train-'+arm)/'INITIAL.json') for arm in ['S','A']];need(initial[0]==initial[1],'independent S/A shared native/head initialization')
        if not synthetic:
            accepted=exact(read(FIXTURE)['short_proofs']['attempt-short-001/slim/RESULT.json'])['arms']['S']['initial'];need(initial[0]['native_SHA']==accepted['native'] and initial[0]['head_SHA']==accepted['head'],'accepted fresh initial bytes')
            values={arm:[read(out/('train-'+arm)/('epoch-%03d.json'%e)) for e in [1,2]] for arm in ['S','A']}
            projected=1.25*sum(max(x['stage_seconds']/x['updates'] for x in rows)*22*304 for rows in values.values())+1800
            write(out/'BUDGET_PROJECTION.json',{'remaining_projected_seconds':projected,'remaining_cap_seconds':float(c['caps']['new_GPU_seconds'])-meter.elapsed(),'not_quality_prediction':True});need(projected<=float(c['caps']['new_GPU_seconds'])-meter.elapsed(),'engineering budget feasibility')
        write(out/'ENGINEERING_ALL_PASS.json',{'same_initial':True,'same_heads':True,'epochs_per_arm':2,'no_AP':True})
        for arm in ['S','A']:old.train(c,arm,False,out,meter,device)
        if c['fixture']['fault']=='bad_endpoint':
            p=out/'train-S/endpoint.json';v=read(p);v['updates']=0;atomic(p,v)
            resultpath=out/'train-S/RESULT.json';v=read(resultpath);v['endpoint']=reference(p);atomic(resultpath,v)
            closepath=out/'train-S/CLOSED.json';v=read(closepath);v['result']=reference(resultpath);atomic(closepath,v)
        if c['fixture']['fault']=='missing_close':(out/'train-S/CLOSED.json').rename(out/'train-S/CLOSED_REMOVED_FOR_FIXTURE.json')
        for arm in ['S','A']:old.infer(c,arm,out,meter,device)
        if c['fixture']['fault']=='bad_input':
            p=out/'infer-S/predictions-600.json';v=read(p);v[0]['image_id']='WRONG';atomic(p,v)
            resultpath=out/'infer-S/RESULT.json';v=read(resultpath);v['predictions']['600']=reference(p);atomic(resultpath,v)
            closepath=out/'infer-S/CLOSED.json';v=read(closepath);v['result']=reference(resultpath);atomic(closepath,v)
        score(c,out,meter);adapter.ARMS=('N','S','A')
        try:old.cost(c,out,meter,device)
        finally:adapter.ARMS=('S','A')
        hashes=[closed(out/('train-'+arm),c)['training_input_SHA'] for arm in ['S','A']];need(hashes[0]==hashes[1],'S/A ordered inputs')
        if not synthetic:need(hashes[0]==c['reuse_N24']['ordered_training_input_SHA'],'N24/SA ordered training input identity')
        old.compare(c,out,meter);need(meter.ledger['new_forward']==forwards(c) and meter.ledger['actual_updates']==meter.ledger['reserved_updates']==2*profile(c)['epochs']*profile(c)['windows'],'terminal counts')
        slim=out/'slim';slim.mkdir();wanted=['ENGINEERING_ALL_PASS.json','score/RESULT.json','cost/RESULT.json','compare/COMPARISON.json','compare/TABLE.md']+['score/'+arm+'-'+b+'.json' for arm in ['S','A'] for b in ['600','300']]
        for arm in ['S','A']:wanted+=['train-'+arm+'/'+x for x in ['INITIAL.json','ENGINEERING_PASS.json','endpoint.json','RESULT.json','CLOSED.json']]+['infer-'+arm+'/'+x for x in ['RESULT.json','CLOSED.json']]
        if not synthetic:wanted+=['N24_EMA_BRIDGE.json','BUDGET_PROJECTION.json','RUNTIME_RESOURCES.json']
        for n in wanted:
            p=slim/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((out/n).read_bytes())
        result={'status':'PASS_ENGINEERING_CORRECTED_FORMAL','scientific_status':read(out/'compare/COMPARISON.json')['scientific_status'],'CPU_synthetic':synthetic,'ledger':meter.ledger,'telemetry':meter.telemetry.arms,'peak_allocated_bytes':None if synthetic else meter.telemetry.observe_peak(),'short_GPU_evidence_reused':True,'short_state_for_formal':False,'N24_retrained':False,'must_exit':True,'no_followup':True}
        write(slim/'RESULT.json',result);write(slim/'CONTRACT.json',c);write(slim/'SHA_INDEX.json',{p.relative_to(slim).as_posix():reference(p) for p in slim.rglob('*') if p.is_file()});need(size(slim)<4<<20,'slim cap');meter.guard();meter.publish()
        write(out/'EXIT.json',{'returncode':0,'elapsed_seconds':meter.elapsed(),'ledger':meter.ledger});write(out/'CLOSED.json',{'status':result['status'],'contract':digest(c),'result':reference(slim/'RESULT.json'),'no_followup':True});return 0
    except BaseException as e:
        (out/'stderr.txt').write_text(traceback.format_exc());peak=None
        if meter:
            try:meter.publish();peak=meter.telemetry.observe_peak()
            except BaseException:pass
        write(out/'STOP.json',{'error':repr(e),'error_message':str(e),'error_type':type(e).__name__,'stage':None if meter is None else meter.stage,'seconds':time.monotonic()-start,'ledger':None if meter is None else meter.ledger,'final_peak_allocated_bytes':peak,'no_retry':True});write(out/'EXIT.json',{'returncode':1});traceback.print_exc();return 1
    finally:signal.alarm(0)

def authority(c):return {'scope':SCHEMA,'contract':digest(c),'authorize_once':True,'CPU_synthetic':c['mode']==CPU,'budget_grant':c['requested_addition'] if c['mode']==GPU else {'forward':0,'GPU_seconds':'0'}}

def cost_record(c,ledger,elapsed):
    prior=c['prior'];total=D(str(prior['GPU_seconds']))+D(str(elapsed))
    return {'previous':{'forward':prior['forward'],'GPU_stage_seconds':prior['GPU_seconds']},
      'new':{'forward':ledger['new_forward'],'actual_updates':ledger['actual_updates'],'GPU_stage_seconds_conservative_process_wall':elapsed},
      'cumulative':{'forward':prior['forward']+ledger['new_forward'],'GPU_stage_seconds':float(total)},
      'remaining':{'forward':c['requested_cumulative']['forward']-prior['forward']-ledger['new_forward'],'GPU_stage_seconds':float(D(c['requested_cumulative']['GPU_seconds'])-total)},
      'approved_expansion':c['requested_addition'],'approved_cumulative_limits':c['requested_cumulative'],'old_cost_overwritten':False,'failed_cost_refunded':False,
      'scope':'FORMAL_ONLY_INDEPENDENT_DEBUG_SEPARATE',
      'independent_debug':c['independent_debug'],
      'cumulative_all_research':{'forward':prior['forward']+ledger['new_forward']+(0 if c['independent_debug'] is None else c['independent_debug']['spent']['forward']),'GPU_stage_seconds':float(total+(D(0) if c['independent_debug'] is None else D(str(c['independent_debug']['spent']['GPU_stage_seconds_conservative_process_wall']))))},
      'debug_unused_budget_for_formal':False}

def data_check(a):
    from check_sabl_ipc import real_check
    real_check(a);return 0

def launch(a):
    need(sha(a.contract)==a.approve_contract_sha,'approved contract bytes binding');c=validate(read(a.contract));m=computation(a.manifest);task=Path(c['attempt_root']).parent
    verify_ipc(task)
    need(a.approve_commit==m['commit']==c['commit'] and a.approve_manifest==sha(a.manifest)==c['manifest_sha256'],'explicit version approval')
    need(not Path(c['attempt_root']).exists() and not (task/'FORMAL_ONCE.json').exists() and not (task/'authorization.json').exists(),'duplicate formal attempt');synthetic=c['mode']==CPU
    if synthetic:need(a.authorize_cpu and not a.authorize_gpu,'CPU-only authority')
    else:
        need(a.authorize_gpu and not a.authorize_cpu and not m['candidate_not_for_real_execution'],'separate GPU authority')
        need(a.grant_forward==c['requested_addition']['forward'] and D(a.grant_GPU_seconds)==D(c['requested_addition']['GPU_seconds']),'explicit exact budget grant')
        fresh=task/'fresh-launch-admission';fresh.mkdir();write(fresh/'MATERIALS.json',material(c));write(fresh/'RESOURCES.json',resources(c,task));short_evidence()
    write(task/'authorization.json',authority(c))
    argv=[sys.executable,'-B',str(ROOT/'tools/sabl_corrected_formal.py'),'run','--contract',str(Path(a.contract).resolve()),'--contract-sha',sha(a.contract),'--manifest',str(Path(a.manifest).resolve()),'--authorization',str(task/'authorization.json'),'--out',c['attempt_root']]
    write(task/'FORMAL_ONCE.json',{'contract':digest(c),'argv':argv,'no_retry':True,'epoch':time.time()});begin=time.monotonic();proc=subprocess.Popen(argv,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    try:stdout,stderr=proc.communicate(timeout=600 if synthetic else float(c['caps']['new_GPU_seconds']))
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGTERM)
        try:stdout,stderr=proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);stdout,stderr=proc.communicate()
    (task/'worker.stdout').write_bytes(stdout);(task/'worker.stderr').write_bytes(stderr);elapsed=time.monotonic()-begin
    ledger=read(Path(c['attempt_root'])/'ledger.json') if (Path(c['attempt_root'])/'ledger.json').exists() else {'new_forward':0,'actual_updates':0}
    write(task/'PROCESS_EXIT.json',{'returncode':proc.returncode,'process_wall_seconds':elapsed,'ledger':ledger,'no_retry':True})
    if not synthetic:
        write(task/'COST_ADDITION.json',cost_record(c,ledger,elapsed))
    return 0 if proc.returncode==0 else 1

def main():
    import argparse
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('plan');q.add_argument('--manifest',required=True);q.add_argument('--attempt-root',required=True);q.add_argument('--out',required=True);q.add_argument('--synthetic',action='store_true');q.add_argument('--fault',choices=FAULTS,default='none')
    q=sub.add_parser('data-check')
    for n in ['contract','manifest','out']:q.add_argument('--'+n,required=True)
    q=sub.add_parser('launch')
    for n in ['contract','manifest','approve-commit','approve-manifest','approve-contract-sha']:q.add_argument('--'+n,required=True)
    q.add_argument('--authorize-cpu',action='store_true');q.add_argument('--authorize-gpu',action='store_true');q.add_argument('--grant-forward',type=int,default=0);q.add_argument('--grant-GPU-seconds',default='0',dest='grant_GPU_seconds')
    q=sub.add_parser('run')
    for n in ['contract','contract-sha','manifest','authorization','out']:q.add_argument('--'+n,required=True)
    a=p.parse_args();return globals()[a.command.replace('-','_')](a)

if __name__=='__main__':raise SystemExit(main())
