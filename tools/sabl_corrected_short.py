"""Two-arm v3 short training only. Uses existing model, Engine and full-state IO."""
import os,sys,json,hashlib,time,math,subprocess,signal,shutil,contextlib,gc,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
for k,v in {'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_THREADING_LAYER':'GNU','PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1'}.items():os.environ[k]=v
from sparse_rtdetr.research.sabl_formal_contract import need,sha,read,digest,reference
BASE='f25b74488f0082cf835b844b89fc8a1f3205b365'
BASE_MANIFEST='54529a9faccecde599753b03b42095f6d7b8d83f7259cf5e3aadae0109cf7d9b'
VERSION='r18-sabl-control-lrtb-logsigmoid-v3'
SCOPE='SABL_CORRECTED_SHORT_V3'
CPU='CPU_SYNTHETIC_CORRECTED_SHORT'
GPU='GPU_CORRECTED_SHORT_ONLY'
N_SHA='be5e9e1fb7a514eceaf2a1343dc29d2f512115bdd85b7fcc26dc11d3a66c9a83'
WS=Path('/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace')
EX=WS/'experiments/localization-distribution'
OLD=EX/'r18-sabl-ns-e24-evaluation-20261028-001'
PY='/home/lyy/.cache/r0-seed0-e120-20261010-001/environment/bin/python'
CAPS={'forward':12,'actual_updates':6,'GPU_seconds':900,'wall_seconds':1800,'allocated_bytes':18<<30,'new_storage_bytes':2<<30,'family_storage_bytes':30<<30,'return_bytes':1<<20}
POLICY={'arms':['S','A'],'sequence':[1,2,2],'seed':0,'scientific_base':BASE,'version':VERSION,'warmup_forward':0,'formal_training':False,'scoring':False,'speed':False,'acceptance_state_for_formal':False}
FAULTS=['none','forward_exhausted','update_exhausted','nonfinite','supervision','bad_input','missing_export','exception']

def write(p,x):
    p=Path(p)
    with p.open('x',encoding='utf8') as f:json.dump(x,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())

def size(root):
    return sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file())

def ref_exact(r):
    need(type(r) is dict and 'path' in r and 'sha256' in r,'reference identity')
    sizes=[r[k] for k in ('bytes','size_bytes') if k in r]
    need(sizes and all(type(v) is int and v>0 for v in sizes) and len(set(sizes))==1,'reference size fields/conflict')
    p=Path(r['path']);need(p.is_absolute() and p.is_file() and not p.is_symlink(),'reference path')
    need(p.stat().st_size==sizes[0] and sha(p)==r['sha256'],'reference size/SHA')
    return {'path':str(p),'bytes':sizes[0],'sha256':r['sha256']}

def source(manifest):
    from r18_deploy import verify_release
    r=verify_release(ROOT,sha(manifest));m=read(manifest)
    need(Path(manifest).resolve()==ROOT/'RELEASE_MANIFEST.json','actual release manifest required')
    f=read(ROOT/'tests/fixtures/sabl_short_scientific_base.json')
    need(f['commit']==BASE and f['manifest_sha256']==BASE_MANIFEST,'scientific base')
    actual=[x for x in m['files'] if x['path'].startswith(('src/','configs/','vendor/'))]
    need(actual==f['files'],'scientific files differ from f25b744')
    need(m['scientific_base_commit']==BASE and m['scientific_base_manifest']==BASE_MANIFEST,'manifest base')
    return m

def family_size():
    return sum(size(p) for p in EX.glob('r18-sabl-*') if p.is_dir())

def history():
    expected=read(ROOT/'tests/fixtures/sabl_short_scientific_base.json')['real_metadata']['cost']
    costref=ref_exact(expected);cost=read(costref['path'])
    counts=[18,31061,661,1,2,274]
    need(sum(counts)==cost['cumulative']['forward']==32017,'historical forward accounting')
    refs=[
      EX/'r18-sabl-control-acceptance-20261027-001/attempt-acceptance-001/EXIT.json',
      EX/'r18-sabl-formal24-20261027-001/attempt-formal-001/STOP.json',
      EX/'r18-sabl-diagnostic-20261028-001/attempt-diagnostic-001/RESULT.json',
      EX/'r18-sabl-forward-validation-20261028-001/PROCESS_EXIT.json',
      EX/'r18-sabl-forward-validation-20261028-002/GPU_PROCESS_EXIT_001.json',
      OLD/'attempt-evaluate-001/EXIT.json']
    vals=[read(p) for p in refs]
    need(vals[0]['returncode']==0 and vals[1]['no_retry'] and vals[1]['ledger']['new_forward']==counts[1],'old initial/STOP')
    need(vals[2]['counts']['forward']==661 and vals[3]['returncode']==1 and vals[4]['returncode']==0 and vals[5]['returncode']==0,'old closures')
    anchor=EX/'r18-sabl-control-acceptance-20261027-001/metadata-001/contract.json'
    need(read(anchor)['forward_cap']==18 and read(anchor)['wall_seconds']==1800,'first historical contract')
    # Charge the entire calendar envelope, INCLUDING idle/development gaps.
    # Its UTC-day floor is earlier than the first certified GPU execution.
    first=math.floor(anchor.stat().st_mtime/86400)*86400
    now=time.time();need(now>=max(p.stat().st_mtime for p in refs),'server clock predates receipts')
    bound=math.ceil(now-first)
    need(bound+CAPS['wall_seconds']<=172800,'historical wall lower balance insufficient')
    need(cost['remaining']['forward']>=12 and cost['remaining']['GPU_stage_seconds']>=900,'historical cost balance')
    family=family_size();need(family+CAPS['new_storage_bytes']<=CAPS['family_storage_bytes'],'family storage balance')
    return {'cost':costref,'cumulative':cost['cumulative'],'remaining':cost['remaining'],
      'accounted_GPU_runs':[{'forward':n,'terminal':reference(p)} for n,p in zip(counts,refs)],
      'wall_anchor':reference(anchor),'wall_anchor_observed_mtime':anchor.stat().st_mtime,'wall_conservative_day_floor_epoch':first,
      'wall_snapshot_epoch':now,'historical_active_wall_exact':'UNKNOWN','historical_wall_upper_bound_seconds':bound,
      'wall_remaining_lower_bound_seconds':172800-bound,'wall_method':'entire chronological envelope including all idle/development gaps, not exact active time',
      'family_storage_bytes':family,'family_remaining_bytes':(30<<30)-family}

def windows(c):
    import torch
    from sparse_rtdetr.protocol_r18.engine import output_digest
    if c['mode']==CPU:
        from sparse_rtdetr.research.sabl_formal_data import Fixture
        d=Fixture();w=[d.window(k)[:2] for k in (1,2)]
        members=[{'stable_image_id':r['stable_image_id']} for r in d.records[:2]]
    else:
        from sparse_rtdetr.localization.data import RawRoleDataset,WindowSampler
        m=c['materials'];spec=m['train'];ds=RawRoleDataset(role='train_core',manifest_path=spec['manifest']['path'],images_path=spec['roots']['images'],annotations_path=spec['roots']['annotations'],root_binding=spec['roots'])
        keys=[k for window in WindowSampler(len(ds),seed=0,epoch=1,completed_windows=0,window_limit=2) for k in window]
        need(len(keys)==32 and len({k[2] for k in keys})==32,'fixed first32')
        members=[{k:ds.records[i][k] for k in ('stable_image_id','relative_path','annotation_relative_path','image_sha256','annotation_sha256','image_size_bytes','annotation_size_bytes')} for _,_,i in keys]
        rows=[ds[k] for k in keys]
        w=[(torch.stack([r[0] for r in rows[k:k+16]]),[r[1] for r in rows[k:k+16]]) for k in (0,16)]
    hashes=[output_digest([im,t]) for im,t in w]
    return w,{'members':members,'window_input_SHA':hashes,'selection':'original seed0 epoch1 first two logical16 windows; no GT selection'}

def no_jobs():
    need(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU compute task exists')
    ps=subprocess.check_output(['ps','-eo','pid,ppid,args'],text=True)
    own={os.getpid(),os.getppid()}
    for l in ps.splitlines()[1:]:
        s=l.strip().split(None,2)
        if len(s)==3 and int(s[0]) not in own and any(x in s[2] for x in ['sabl_formal.py run','sabl_diagnostic.py run','sabl_forward_check.py run','sabl_acceptance.py run','sabl_ns_eval.py run','sabl_corrected_short.py run','sabl_corrected_short.py launch']):
            raise ValueError('related job exists')

def validate(c):
    need(set(c)=={'schema','mode','policy','caps','attempt','family_task','manifest_sha256','commit','materials','history','inputs','fixture'},'contract fields')
    need(c['schema']==SCOPE and c['mode'] in (CPU,GPU) and digest(c['policy'])==digest(POLICY) and digest(c['caps'])==digest(CAPS),'frozen scope/policy/caps')
    need(c['fixture'] in FAULTS and (c['mode']==CPU or c['fixture']=='none'),'production fixture forbidden')
    need(Path(c['attempt']).is_absolute() and Path(c['attempt']).parent==Path(c['family_task']),'attempt task binding')
    if c['mode']==GPU:
        need(c['materials'] is not None and c['history'] is not None and c['materials']['parent']['sha256']==N_SHA,'real material binding')
    else:need(c['materials'] is None and c['history'] is None,'synthetic cannot bind real material')
    return c

def prepare(a):
    out=Path(a.out);out.mkdir()
    try:
        m=source(a.manifest)
        synthetic=a.synthetic;need(synthetic or not m['candidate_not_for_real_execution'],'candidate cannot execute GPU')
        c={'schema':SCOPE,'mode':CPU if synthetic else GPU,'policy':POLICY,'caps':CAPS,'attempt':str(out.parent/'attempt-short-001'),'family_task':str(out.parent),'manifest_sha256':sha(a.manifest),'commit':m['commit'],'materials':None,'history':None,'inputs':None,'fixture':a.fault}
        if not synthetic:
            need(a.fault=='none' and Path(sys.executable).resolve()==Path(PY).resolve(),'production interpreter/fixture')
            registered=read(ROOT/'tests/fixtures/sabl_short_scientific_base.json')['real_metadata']
            definition=Path(ref_exact(registered['definition'])['path']);v=read(definition)
            need(v['old_scientific_commit']=='79ba59371c21ecfd4fa76885c413941403797214' and v['old_source_manifest']=='1af3d80bc0b28aa384da0511715ea81083e709849bc076abb226f2e2ff82c1ab','original metadata identity')
            n=ref_exact(v['refs']['original_parent_N']);need(n['sha256']==N_SHA,'common parent')
            need(v['refs']['original_contract']==registered['original_contract'],'original contract binding')
            original=read(ref_exact(registered['original_contract'])['path'])
            train=original['metadata']['data']['train_core']
            train={'manifest':ref_exact(train['manifest']),'roots':train['roots']}
            need(train['manifest']['sha256']=='df87e6a377a5546d3d3fe0cd0e860d15c2fe5e8003ea87d887153579999a4e9d','train identity')
            oldshort=EX/'r18-sabl-control-acceptance-20261027-001/attempt-acceptance-001/slim/RESULT.json'
            need(reference(oldshort)==registered['old_short'],'old input reference SHA')
            oldr=read(oldshort);need(oldr['status']=='PASS_ENGINEERING','old input reference')
            c['materials']={'parent':n,'train':train,'original_definition':reference(definition),'old_short':reference(oldshort),'old_input_SHA':oldr['arms']['S']['input_SHA'],'old_native_initial_SHA':oldr['arms']['S']['shared_initial_SHA'],'old_head_initial_SHA':oldr['arms']['S']['head_initial_SHA'],'python':PY,'gpu_uuid':v['gpu_uuid'],'environment':v['environment']}
            c['history']=history();no_jobs()
            q=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True)
            rows={x.split(',')[0].strip():int(x.split(',')[1]) for x in q.strip().splitlines()}
            need(rows[c['materials']['gpu_uuid']]>=18432,'free GPU memory')
            need(shutil.disk_usage(out).free>=2<<30,'free disk for short')
        import torch,torchvision
        need(torch.__version__.split('+')[0]=='2.4.1' and torchvision.__version__.split('+')[0]=='0.19.1','runtime')
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        need(not torch.cuda.is_initialized(),'prepare must remain CUDA-hidden')
        _,c['inputs']=windows(c)
        if not synthetic:need(c['inputs']['window_input_SHA']==c['materials']['old_input_SHA'],'original short input hashes')
        validate(c)
        write(out/'DEFINITION.json',c);write(out/'READY.json',{'status':'PASS_CPU_PREPARE','definition_sha256':sha(out/'DEFINITION.json'),'GPU_initialized':False,'GPU_forward':0})
        return 0
    except BaseException as e:
        write(out/'STOP.json',{'error':repr(e),'stage':'prepare','GPU_forward':0,'no_retry':True});traceback.print_exc();return 1

class Observation:
    """Read-only observations; no extra detector forward or backward."""
    def __init__(self,engine,synthetic):
        import torch
        from sparse_rtdetr.research import sabl_control as sc
        self.engine=engine;self.torch=torch;self.rows=[];self.current=[];self.original=sc.encode;self.groups={};self.handles=[];self.synthetic=synthetic
        def encode(r,g):
            value=self.original(r,g)
            cx,cy,w,h=g.detach().float().unbind(-1)
            edge=torch.stack((cx-w/2,cx+w/2,cy-h/2,cy+h/2),-1)
            x,scale=sc.buckets(r);want=(x-edge[...,None])/scale[...,None]
            need(torch.equal(value[0],want),'actual supervision L/R/T/B')
            need(not value[0].requires_grad and not r.requires_grad,'GT/reference detach')
            self.current.append({'matches':len(g),'target_LRTB_exact':True,'reference_detached':True})
            return value
        self.wrapper=encode;sc.encode=encode
        for name,p in engine.model.named_parameters():
            group='native' if name.startswith('native.') else name.split('.')[1]
            def hook(g,key=group):
                self.groups.setdefault(key,[]).append(bool(torch.isfinite(g).all()) and bool((g!=0).any()))
                return None
            self.handles.append(p.register_hook(hook))
    def finish(self):
        self.rows.extend(self.current)
        need(len(self.current)==2 and all(x['matches']>0 for x in self.current),'two matched ordinary microbatches required')
        for g in ['native','pre','att_x','att_y','post_x','post_y','up_x','up_y','cls','offset']:
            need(g in self.groups and any(self.groups[g]),'nonzero finite gradient group '+g)
        out={'encode_calls':self.current,'nonzero_finite_gradient_groups':{k:any(v) for k,v in self.groups.items()}}
        self.current=[];self.groups={};return out
    def close(self):
        from sparse_rtdetr.research import sabl_control as sc
        sc.encode=self.original
        for h in self.handles:h.remove()

def run(a):
    out=Path(a.out);out.mkdir();begin=time.monotonic();ledger={'forward':0,'actual_updates':0,'reserved_updates':0};engine=None;obs=None;meter=None;peak=None
    def persist():
        p=out/'ledger.tmp';p.write_text(json.dumps(ledger));os.replace(p,out/'ledger.json')
    try:
        c=validate(read(a.definition));synthetic=c['mode']==CPU;m=source(a.manifest)
        need(sha(a.definition)==a.definition_sha and c['manifest_sha256']==sha(a.manifest) and c['commit']==m['commit'],'definition/source binding')
        need(out.resolve()==Path(c['attempt']).resolve(),'attempt identity')
        auth=read(a.authorization);need(auth=={'scope':SCOPE,'contract':digest(c),'authorize_once':True,'CPU_synthetic':synthetic},'authorization')
        if synthetic:os.environ['CUDA_VISIBLE_DEVICES']=''
        else:
            need(not m['candidate_not_for_real_execution'],'synthetic release cannot run GPU')
            os.environ['CUDA_VISIBLE_DEVICES']=c['materials']['gpu_uuid']
            ref_exact(c['materials']['parent']);ref_exact(c['history']['cost']);no_jobs()
        # All compiler/driver/temp writes are bound beneath the new task.
        task=Path(c['family_task']);cache=task/'compiler-cache'
        for k,sub in [('TORCHINDUCTOR_CACHE_DIR','inductor'),('TRITON_CACHE_DIR','triton'),('CUDA_CACHE_PATH','cuda'),('TORCH_EXTENSIONS_DIR','extensions'),('XDG_CACHE_HOME','xdg'),('TMPDIR','tmp')]:
            p=cache/sub;p.mkdir(parents=True,exist_ok=True);os.environ[k]=str(p)
        import tempfile;tempfile.tempdir=os.environ['TMPDIR']
        import torch,numpy as np,random,torchvision
        torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
        need(torch.__version__.split('+')[0]=='2.4.1' and torchvision.__version__.split('+')[0]=='0.19.1','runtime')
        if not synthetic:
            need(torch.cuda.is_available() and torch.cuda.device_count()==1,'single CUDA')
            need(torch.version.cuda==c['materials']['environment']['cuda_build'],'CUDA build')
            need(torch.backends.cudnn.version()==c['materials']['environment']['cudnn'],'cudnn')
            total=torch.cuda.get_device_properties(0).total_memory;need(total>=18<<30,'GPU capacity')
            torch.cuda.set_per_process_memory_fraction((18<<30)/total,0)
        from sparse_rtdetr.research.sabl_control import make_engine,VERSION as actual_version
        from sparse_rtdetr.research import sabl_state as state
        from sparse_rtdetr.protocol_r18.engine import output_digest,tensor_digest
        from sparse_rtdetr.baseline.training_v2b_checkpoint import _equal
        from sabl_acceptance import Telemetry
        need(actual_version==VERSION,'model version')
        meter=Telemetry(None if synthetic else torch.cuda)
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('short stage deadline')))
        signal.alarm(300 if synthetic else CAPS['GPU_seconds'])
        write(out/'CONSUMED.json',{'contract':digest(c),'no_retry':True,'formal_state_use':False})
        def guard():
            need(time.monotonic()-begin<(300 if synthetic else 900),'short stage time')
            need(size(task)<2<<30,'short new storage')
            if not synthetic:
                need(c['history']['family_storage_bytes']+size(task)<30<<30,'cumulative family storage')
                need(meter.observe_peak()<=18<<30,'allocated cap')
        def charge(module,args):
            guard();limit=0 if c['fixture']=='forward_exhausted' else 12
            need(ledger['forward']<limit,'short forward cap')
            ledger['forward']+=1;persist()
        def seed():
            random.seed(0);np.random.seed(0);torch.manual_seed(0)
            if not synthetic:torch.cuda.manual_seed_all(0)
        w,inputs=windows(c);need(inputs==c['inputs'],'bound next-window identity')
        write(out/'INPUT.json',inputs)
        initial=None;results={}
        for arm in ['S','A']:
            meter.begin(arm);seed()
            engine=make_engine(arm,synthetic,None if synthetic else c['materials']['parent']['path'],'cpu' if synthetic else 'cuda')
            init={'native':tensor_digest(engine.model.native.state_dict().items()),'head':tensor_digest(engine.model.boundary.state_dict().items())}
            if initial is None:initial=init
            need(init==initial,'independent same native/BN/head initialization')
            if not synthetic:need(init=={'native':c['materials']['old_native_initial_SHA'],'head':c['materials']['old_head_initial_SHA']},'original parent/head initial bytes')
            need(not engine.optimizer.state and engine.ema.updates==engine.completed_updates==0,'fresh optimizer/EMA/engine')
            binding={'scope':'CPU_SYNTHETIC_SABL' if synthetic else 'GPU_SABL_ACCEPTANCE_ONLY','arm':arm,'version':VERSION,'contract':digest(c),'release_manifest':c['manifest_sha256'],'N_SHA':N_SHA}
            folder=out/arm;folder.mkdir();engine.model.register_forward_pre_hook(charge);obs=Observation(engine,synthetic)
            seed();summaries=[]
            def update(which,role):
                limit=0 if c['fixture']=='update_exhausted' else 6
                need(ledger['reserved_updates']<limit,'short update cap')
                ledger['reserved_updates']+=1;persist()
                if c['fixture']=='nonfinite':w[which][0][0,0,0,0]=float('nan')
                if c['fixture']=='bad_input':raise ValueError('bound next-window identity fixture')
                before={n:p.detach().cpu().clone() for n,p in engine.model.named_parameters()}
                r=engine.train_window(*w[which],capture=True);ledger['actual_updates']+=1;persist();meter.update(r,role)
                supervision=obs.finish()
                if c['fixture']=='supervision':raise ValueError('actual supervision L/R/T/B fixture')
                changed={}
                for n,p in engine.model.named_parameters():
                    if not torch.equal(before[n],p.detach().cpu()):changed[n]=True
                need(any(n.startswith('native.') for n in changed),'shared parameters updated')
                for g in ['pre','att_x','att_y','post_x','post_y','up_x','up_y','cls','offset']:
                    need(any(n.startswith('boundary.'+g+'.') for n in changed),'boundary parameter updated '+g)
                need(all(p.grad is None for p in engine.model.parameters()),'complete update cleared gradients')
                need(engine.model.last_dtype==('torch.float32' if synthetic else 'torch.bfloat16') and engine.model.last_boundary_dtype=='torch.float32','actual native/boundary precision')
                need(r['finite'] and all(math.isfinite(v) for row in r['microbatch_losses'] for v in row.values()),'finite objective/gradients')
                row={'role':role,'update':r['update'],'input_SHA':inputs['window_input_SHA'][which],'logits_dtype':engine.model.last_dtype,'boundary_dtype':engine.model.last_boundary_dtype,'supervision':supervision,'changed_parameter_count':len(changed),'changed_parameter_names':list(changed),'losses':r['microbatch_losses'],'finite':r['finite'],'gradient_norm_before_clip':r['gradient_norm_before_clip'],'output_sha256':r['output_sha256'],'gradient_sha256_before_clip':r['gradient_sha256_before_clip'],'dn_noise_sha256':r['dn_noise_sha256']}
                write(folder/(role+'.json'),row);summaries.append(row)
                return r
            first=update(0,'u1');cp=state.save(engine,binding,folder/'C1.pt');guard()
            direct=update(1,'u2_direct');expected=state.capture(engine,binding)
            restored=state.restore(engine,binding,folder/'C1.pt',cp['sha256'])
            need(engine.state_dict()['next_window']==1 and engine.completed_updates==1,'restored next-window position')
            replay=update(1,'u2_replay');actual=state.capture(engine,binding)
            need(_equal(expected,actual),'post-replay full model optimizer EMA RNG state')
            need(all(direct[k]==replay[k] for k in ['microbatch_losses','gradient_sha256_before_clip','output_sha256','dn_noise_sha256']),'direct/replay loss gradient output DN')
            need(inputs==windows(c)[1],'next-window source identity after replay')
            final=state.save(engine,binding,folder/'C2.pt');guard();meter.finish()
            results[arm]={'initial':init,'updates_executed':3,'unique_update_boundary':2,'restore':restored,'replay_full_state_exact':True,'next_window_replay_exact':True,'loss_output_gradient_DN_exact':True,'telemetry':meter.arms[arm],'updates':summaries,'checkpoint':final}
            obs.close();obs=None;del engine,expected,actual;engine=None;gc.collect()
            if not synthetic:torch.cuda.empty_cache()
        need(ledger=={'forward':12,'actual_updates':6,'reserved_updates':6},'terminal exact budget');guard()
        need(c['fixture']!='exception','terminal fixture')
        result={'status':'PASS_ENGINEERING_CORRECTED_SHORT','scientific_base':BASE,'version':VERSION,'contract':digest(c),'ledger':ledger,'arms':results,'GPU_forward':0 if synthetic else 12,'CPU_synthetic':synthetic,'seconds':time.monotonic()-begin,'peak_allocated_bytes':None if synthetic else meter.observe_peak(),'precision':{'native':'FP32' if synthetic else 'BF16','boundary':'FP32'},'compiler_cache_bytes':size(cache),'new_task_bytes':size(task),'scientific_quality':'UNKNOWN','formal_training_started':False,'acceptance_state_for_formal':False,'no_followup':True}
        slim=out/'slim';slim.mkdir()
        write(slim/'RESULT.json',result);write(slim/'CONTRACT.json',c)
        need(c['fixture']!='missing_export','closed export fixture')
        write(slim/'SHA_INDEX.json',{p.name:sha(p) for p in slim.iterdir()})
        need(size(slim)<1<<20,'slim return bytes')
        write(out/'CLOSED.json',{'status':result['status'],'result_sha256':sha(slim/'RESULT.json'),'contract':digest(c),'no_followup':True})
        write(out/'EXIT.json',{'returncode':0,'wall_seconds':time.monotonic()-begin,'ledger':ledger})
        return 0
    except BaseException as e:
        if obs is not None:obs.close()
        if meter is not None:
            try:meter.finish(complete=False);peak=meter.observe_peak()
            except BaseException:peak=None
        write(out/'STOP.json',{'error':repr(e),'ledger':ledger,'seconds':time.monotonic()-begin,'final_peak_allocated_bytes':peak,'peak_status':'UNKNOWN' if peak is None else 'MEASURED_ON_EXCEPTION','no_retry':True})
        write(out/'EXIT.json',{'returncode':1,'wall_seconds':time.monotonic()-begin,'ledger':ledger})
        traceback.print_exc();return 1
    finally:signal.alarm(0)

def launch(a):
    c=validate(read(a.definition));m=source(a.manifest);synthetic=c['mode']==CPU;task=Path(c['family_task'])
    need(sha(a.manifest)==c['manifest_sha256'] and m['commit']==c['commit'],'launch version')
    need(a.approve_commit==m['commit'] and a.approve_manifest==sha(a.manifest),'explicit approved identity')
    need((synthetic and a.authorize_cpu and not a.authorize_gpu) or (not synthetic and a.authorize_gpu and not a.authorize_cpu),'execution authority')
    need(not Path(c['attempt']).exists() and not (task/'SHORT_ONCE.json').exists() and not (task/'authorization.json').exists(),'duplicate launch')
    if not synthetic:
        need(m['GPU_execution_frozen'] is True,'GPU execution package not frozen')
        need(time.time()-Path(a.definition).stat().st_mtime<600,'admission freshness')
        fresh=history();need(fresh['cost']==c['history']['cost'],'latest history ledger changed')
        no_jobs();need(Path(sys.executable).resolve()==Path(PY).resolve(),'launch interpreter')
    auth={'scope':SCOPE,'contract':digest(c),'authorize_once':True,'CPU_synthetic':synthetic}
    write(task/'authorization.json',auth)
    argv=[sys.executable,'-B',str(ROOT/'tools/sabl_corrected_short.py'),'run','--definition',str(Path(a.definition).resolve()),'--definition-sha',sha(a.definition),'--manifest',str(Path(a.manifest).resolve()),'--authorization',str(task/'authorization.json'),'--out',c['attempt']]
    write(task/'SHORT_ONCE.json',{'contract':digest(c),'argv':argv,'no_retry':True,'timestamp_epoch':time.time()})
    begin=time.monotonic();proc=subprocess.Popen(argv,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    # A watchdog waits for this one child only; it never starts another run.
    try:stdout,stderr=proc.communicate(timeout=300 if synthetic else 900)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGTERM)
        try:stdout,stderr=proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);stdout,stderr=proc.communicate()
    (task/'worker.stdout').write_bytes(stdout);(task/'worker.stderr').write_bytes(stderr)
    elapsed=time.monotonic()-begin
    ledger=read(Path(c['attempt'])/'ledger.json') if (Path(c['attempt'])/'ledger.json').exists() else {'forward':0,'actual_updates':0,'reserved_updates':0}
    ret={'returncode':proc.returncode,'process_wall_seconds':elapsed,'ledger':ledger,'stderr_bytes':len(stderr),'stdout_bytes':len(stdout),'no_retry':True}
    write(task/'PROCESS_EXIT.json',ret)
    if not synthetic:
        prior=c['history'];newf=ledger['forward'];newsec=elapsed
        cost={'previous_cost':prior['cost'],'previous':prior['cumulative'],'new':{'forward':newf,'actual_updates':ledger['actual_updates'],'GPU_stage_seconds_conservative_process_wall':newsec},'cumulative':{'forward':prior['cumulative']['forward']+newf,'GPU_stage_seconds':prior['cumulative']['GPU_stage_seconds']+newsec},'remaining':{'forward':prior['remaining']['forward']-newf,'GPU_stage_seconds':prior['remaining']['GPU_stage_seconds']-newsec},'wall_before_upper_bound_seconds':prior['historical_wall_upper_bound_seconds'],'wall_remaining_lower_bound_seconds':prior['wall_remaining_lower_bound_seconds']-elapsed,'old_failed_cost_refunded':False}
        write(task/'COST_ADDITION.json',cost)
    need(elapsed<(300 if synthetic else 900) and elapsed<1800,'launch stage/wall budget')
    if proc.returncode:return 1
    closed=read(Path(c['attempt'])/'CLOSED.json');result=Path(c['attempt'])/'slim/RESULT.json'
    need(closed['result_sha256']==sha(result) and read(result)['ledger']=={'forward':12,'actual_updates':6,'reserved_updates':6},'terminal closure')
    return 0

def main():
    import argparse
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True)
    q=sub.add_parser('prepare');q.add_argument('--manifest',required=True);q.add_argument('--out',required=True);q.add_argument('--synthetic',action='store_true');q.add_argument('--fault',choices=FAULTS,default='none')
    q=sub.add_parser('launch');q.add_argument('--manifest',required=True);q.add_argument('--definition',required=True);q.add_argument('--approve-commit',required=True);q.add_argument('--approve-manifest',required=True);q.add_argument('--authorize-cpu',action='store_true');q.add_argument('--authorize-gpu',action='store_true')
    q=sub.add_parser('run')
    for n in ['definition','definition-sha','manifest','authorization','out']:q.add_argument('--'+n,required=True)
    a=p.parse_args();return globals()[a.cmd](a)

if __name__=='__main__':raise SystemExit(main())
