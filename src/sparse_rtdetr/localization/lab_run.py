"""Independent new-host acceptance + fixed nonformal geometry diagnostic only."""
import argparse,contextlib,gc,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
from .runio import read,write,publish,sha,digest,require,event,reference,execution_binding,original_process_exited
from . import guard,channel
MATERIAL_ROOT=Path('/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace/experiments/localization-distribution/f0-e026-labdiag-20261009-001')
ROOT=MATERIAL_ROOT/'layout-recovery-001'
SOURCE=Path(__file__).resolve().parents[3]
PARENT_SHA='22345901f817b9c24c8b7167adfa0c2991d424437a2991339f90301efe356f62'
MODES=('acceptance','diagnostic');GIB=1024**3
VOLATILE={'wall_epoch','monotonic','pid','elapsed_seconds','complete_cycle_seconds','output_sha256','gradient_sha256_before_clip'}
COMPARE_POLICY={'float_fields':['loss','microbatch_losses','gradient_norm_before_clip'],'atol':1e-7,'rtol':1e-5,'other_recorded_fields':'exact','unrecorded_outputs_gradients_states':'not claimed comparable'}

def compare(old,new):
    differences=[];bad=[]
    def walk(a,b,path):
        if isinstance(a,dict):
            require(isinstance(b,dict) and set(a)==set(b),'cross-machine schema keys differ: '+path)
            for k in a:walk(a[k],b[k],path+'/'+str(k))
        elif isinstance(a,list):
            require(isinstance(b,list) and len(a)==len(b),'cross-machine list shape differs: '+path)
            for i,(x,y) in enumerate(zip(a,b)):walk(x,y,path+'/'+str(i))
        elif path=='/loss' or path=='/gradient_norm_before_clip' or path.startswith('/microbatch_losses/'):
            import math
            require(type(a) in (int,float) and type(b) in (int,float) and math.isfinite(a) and math.isfinite(b),'nonfinite/missing compared metric')
            delta=abs(a-b);limit=1e-7+1e-5*max(abs(a),abs(b));r={'field':path,'reference':a,'actual':b,'abs_difference':delta,'limit':limit,'bitwise_numeric_equal':a==b,'pass':delta<=limit}
            differences.append(r)
            if not r['pass']:bad.append(r)
        elif type(a)!=type(b) or a!=b:bad.append({'field':path,'reference':a,'actual':b,'pass':False})
    walk({k:v for k,v in old.items() if k not in VOLATILE},{k:v for k,v in new.items() if k not in VOLATILE},'')
    return {'status':'PASS' if not bad else 'STOP_CROSS_MACHINE_MISMATCH','exact_all_fields':not bad and all(x['bitwise_numeric_equal'] for x in differences),'floating_fields':differences,'mismatches':bad}

def snapshot(path):
    import shutil
    gpu=guard.probe.bounded_command(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.free,temperature.gpu,utilization.gpu,clocks.current.graphics,power.limit','--format=csv,noheader,nounits'],timeout=12)
    require(gpu['status']=='OBSERVED','GPU telemetry missing');f=[x.strip() for x in gpu['stdout'].strip().split(',')];require(len(f)==8,'GPU inventory not single')
    device={k:{'status':'OBSERVED','value':v} for k,v in zip(['uuid','name','memory.total','memory.free','temperature.gpu','utilization.gpu','clocks.current.graphics','power.limit'],[f[0],f[1],*[float(x) for x in f[2:]]])}
    apps=guard.probe.bounded_command(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name,used_memory','--format=csv,noheader,nounits'],timeout=12);require(apps['status']=='OBSERVED','compute inventory missing')
    processes=[]
    for line in apps['stdout'].splitlines():
        bits=[x.strip() for x in line.split(',')];require(len(bits)==4,'compute parse failed');processes.append({'pid':int(bits[0]),'uuid':bits[1]})
    mem={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())}
    cpu=[]
    for h in Path('/sys/class/hwmon').glob('hwmon*'):
        if (h/'name').read_text().strip()=='coretemp':
            for t in h.glob('temp*_input'):cpu.append(int(t.read_text().strip())/1000.)
    require(cpu,'CPU temperature visibility missing')
    mount_rows=guard.probe.read_text('/proc/self/mountinfo').splitlines();matches=[]
    for line in mount_rows:
        left,right=line.split(' - ',1);a,b=left.split(),right.split()
        if Path(path).is_relative_to(Path(a[4])):matches.append((len(a[4]),{'mountpoint':a[4],'type':b[0],'source':b[1],'rw':'rw' in a[5].split(',')}))
    mount=max(matches,key=lambda x:x[0])[1];usage=shutil.disk_usage(path);home=shutil.disk_usage('/home/lyy')
    storage={'canonical_mount':mount['mountpoint'],'platform_entry':mount['mountpoint'],'mount':mount,'st_dev':Path(path).stat().st_dev,
             'planning_cap_bytes':usage.total,'admitted_available_bytes':usage.free,'ssd_free_bytes':home.free}
    vm=dict(line.split() for line in Path('/proc/vmstat').read_text().splitlines())
    return {'sampled_epoch':time.time(),'monotonic':time.monotonic(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'snapshot':{'gpu':{'devices':[device]},'resources':{'cpu_count':len(os.sched_getaffinity(0)),'ram_available':mem['MemAvailable']},
                        'cpu_temperature_max':max(cpu),'shm_free':shutil.disk_usage('/dev/shm').free},
            'compute_apps':processes,'compute_apps_status':'OBSERVED','memory_events':{'vmstat':{'oom_kill':int(vm['oom_kill'])}},'storage':storage}

def screen(s,*,gpu_uuid=None,entry=False,gpu=True,disk_bytes=0,baseline_events=None):
    require(s['snapshot']['resources']['cpu_count']>=4 and s['snapshot']['resources']['ram_available']>=8*GIB,'lab CPU/RAM reserve')
    require(s['snapshot']['shm_free']>=4*GIB and s['storage']['admitted_available_bytes']>=10*GIB and s['storage']['ssd_free_bytes']>=4*GIB,'lab storage/shm reserve')
    require(s['snapshot']['cpu_temperature_max']<90,'CPU temperature threshold')
    if baseline_events is not None:require(s['memory_events']==baseline_events,'OOM event changed')
    d=s['snapshot']['gpu']['devices'][0];uid=d['uuid']['value'];require(gpu_uuid in (None,uid),'GPU identity changed')
    require(d['temperature.gpu']['value']<80 and d['memory.free']['value']>=(20000 if entry else 3072),'GPU temp/memory protection')
    require(not s['compute_apps'] if entry else len(s['compute_apps'])<=1,'other GPU experiment present')
    return uid

def verify_source(c):
    require(c['source_root']==str(SOURCE) and sha(c['source_manifest'])==c['source_sha256'],'wrong lab executing source')
    for r in read(c['source_manifest'])['files']:
        p=SOURCE/r['path'];require(not p.is_symlink() and p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],'lab source changed')

def verify_materials(c):
    require(c['comparison_policy']==COMPARE_POLICY,'comparison policy changed')
    required={c[k] for k in ('parent_binding','parent_sidecar','parent_contract','reference','needed_data','data_verification','import_result')}
    required|={c['pretrained']['path'],c['data']['train_core']['manifest']}
    require(required<={r['path'] for r in c['materials']} and len(c['materials'])==len({r['path'] for r in c['materials']}),'required material SHA bindings missing/duplicate')
    for r in c['materials']:
        p=Path(r['path']);require(p.is_absolute() and p.is_relative_to(MATERIAL_ROOT) and not p.is_symlink(),'material outside dedicated lab scope')
        require(reference(p)==r,'frozen material bytes changed: '+str(p))
    require(sha(c['reference'])==c['reference_sha256'],'cross-machine expected reference changed')
    origin=read(c['parent_binding']);side=read(c['parent_sidecar']);oldc=read(c['parent_contract'])
    require(oldc['contract_sha256']=='3f8e2e3a5562598fdf97e044f4c833f2ca4dc68aed7e8600158471d427863158' and digest({k:v for k,v in oldc.items() if k!='contract_sha256'})==oldc['contract_sha256'],'old contract identity changed')
    from .recovery import digest_checkpoint_binding
    require(side['sha256']==PARENT_SHA and side['completed_updates']==7904 and side['binding_sha256']==digest_checkpoint_binding(origin),'parent sidecar/binding changed')

def expected_reference(c):
    require(sha(c['reference'])==c['reference_sha256'],'frozen comparison reference bytes changed')
    return read(c['reference'])

def load(path):
    c=read(path);require(c['contract_sha256']==digest({k:v for k,v in c.items() if k!='contract_sha256'}),'lab contract digest differs')
    require(c['job_root']==str(ROOT) and c['workload']=={'acceptance_updates':[7905,7906,7906],'diagnostic_prefix':73,'diagnostic_window':73,'forbidden_optimizer_update':7978,'formal_training':False},'workload changed')
    require(c['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'lab boot changed');verify_source(c)
    require(sha(c['parent_checkpoint'])==PARENT_SHA and sha(c['parent_binding'])==c['parent_binding_sha256'],'parent bytes/binding changed')
    verify_materials(c);return c

def prepared(request):
    from .control import environment
    c=read(request);require(not (ROOT/'contract.json').exists(),'lab contract already frozen')
    s=snapshot(ROOT);gpu=screen(s,entry=True);actual=environment()
    expected={'python_version':'3.10.16','torch':'2.4.1','torchvision':'0.19.1','numpy':'1.26.4','scipy':'1.13.1','pillow':'12.1.0','cuda_build':'12.4','cudnn':90100}
    require(all(actual[k]==v for k,v in expected.items()),'isolated runtime not expected versions')
    from .storage import identity
    c.update(schema='f0-lab-migration-v1',job_root=str(ROOT),gpu_uuid=gpu,boot_id=s['boot_id'],environment={k:actual[k] for k in ('torch','torchvision','cuda_build','cudnn')},environment_observation=actual,
             storage={'identity':identity(s['storage'])},resource_observation=s,freeze_epoch=time.time())
    c['contract_sha256']=digest(c);verify_source(c);verify_materials(c)
    require(sha(c['parent_checkpoint'])==PARENT_SHA,'parent state changed');publish(ROOT/'contract.json',c);return c

def complete(c,mode):
    exit=read(Path(c['runtime_root'])/mode/'supervisor-exit.json');require(exit['status']=='PASS','prior lab phase not PASS')
    folder=ROOT/mode;dispatch=read(ROOT/(mode+'-dispatch.json'));require(original_process_exited(dispatch,guard.identity(dispatch['pid'])),'prior phase still alive')
    require(dispatch['contract_sha256']==c['contract_sha256'] and dispatch['mode']==mode,'prior dispatch identity changed')
    channel.verify_archive(Path(c['runtime_root'])/mode,folder,{**execution_binding(c,mode),'supervisor_pid':dispatch['pid'],'supervisor_start':dispatch['process_identity']['start']})
    require(read(folder/'execution-binding.json')==execution_binding(c,mode),'prior phase source/identity differs')
    for name in ('RESULT.json','execution-binding.json','worker-exit.json','binding.json'):
        p=(Path(c['runtime_root'])/mode if name=='worker-exit.json' else folder)/name
        require(exit['completion_receipts'][name]==reference(p),'prior phase completion changed')
    result=read(folder/'RESULT.json')
    if mode=='acceptance':require(result['status']=='PASS' and result['actual_updates']==3 and result['updates']==[7905,7906,7906] and result['original_restore']['checkpoint_sha256']==PARENT_SHA and result['original_restore']['completed_updates']==7904,'acceptance workload/parent changed')
    return result

def prerequisite(c,mode):
    require(mode in MODES,'mode out of scope')
    if mode=='diagnostic':
        r=complete(c,'acceptance');require(r['same_next_window_exact'] and r['same_next_state_exact'],'newhost acceptance not exact')

def supervisor_token(c,mode,path):
    token=read(ROOT/(mode+'-launch-consumed.json'))
    command=[c['python'],'-B',str(SOURCE/'tools/localization/deploy.py'),'lab-supervise','--contract',str(path),'--mode',mode]
    require(token['mode']==mode and token['contract_sha256']==c['contract_sha256'] and token['command']==command and token['no_retry'] is True,'supervision token changed/missing')

def launch(path,mode):
    c=load(path);require(mode in MODES and not (ROOT/mode).exists(),'phase already launched/unknown')
    prerequisite(c,mode)
    s=snapshot(ROOT);screen(s,gpu_uuid=c['gpu_uuid'],entry=True)
    command=[c['python'],'-B',str(SOURCE/'tools/localization/deploy.py'),'lab-supervise','--contract',str(path),'--mode',mode]
    publish(ROOT/(mode+'-launch-consumed.json'),{'mode':mode,'contract_sha256':c['contract_sha256'],'command':command,'epoch':time.time(),'no_retry':True})
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MKL_THREADING_LAYER='GNU',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (ROOT/(mode+'-dispatcher.log')).open('x') as log:p=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    r={'pid':p.pid,'process_identity':guard.identity(p.pid),'contract_sha256':c['contract_sha256'],'mode':mode,'started_epoch':time.time()};publish(ROOT/(mode+'-dispatch.json'),r);return r

def supervise(path,mode):
    c=load(path);prerequisite(c,mode);supervisor_token(c,mode,path);guard.snapshot=snapshot;guard.screen=screen
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=c['gpu_uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    command=[c['python'],'-B',str(SOURCE/'tools/localization/deploy.py'),'lab-worker','--contract',str(path),'--mode',mode]
    return guard.supervise(command,ROOT/mode,env,gpu_uuid=c['gpu_uuid'],boot_id=c['boot_id'],checkpoint_bytes=512*1024**2,
       runtime=Path(c['runtime_root'])/mode,run_identity={**execution_binding(c,mode),'supervisor_pid':os.getpid(),'supervisor_start':guard.identity(os.getpid())['start']})

def restored(c,runtime,holder):
    from . import worker,checkpoint
    engine,b=worker.make_engine(c,runtime,'gpu_short_window');holder['engine']=engine
    origin=read(c['parent_binding']);from .cache_adapter import strict_restore
    r=strict_restore(engine,origin,c['parent_checkpoint'],PARENT_SHA,Path(c['runtime_root'])/holder['mode']/'parent-cache-import.json')
    require(r['full_state_exact'] and engine.completed_updates==7904 and engine.ema.updates==7904,'originale026 restore not exact')
    b['lineage']={'lab_attempt':'f0-e026-labdiag-20261009-001','parent_checkpoint_sha256':PARENT_SHA,'parent_binding_sha256':c['parent_binding_sha256'],'state_for_formal':False}
    return engine,b,r

def check_window(c,result,metadata,folder):
    ref=expected_reference(c);old=ref['updates'][result['update']-7905]
    actual={**result,'event':'update_complete','completed_epochs':result['update']//304,'input_sha256':digest(metadata)}
    comparison=compare(old,actual);publish(folder/f"cross-{result['update']}.json",comparison)
    require(comparison['status']=='PASS','cross-machine recorded trajectory mismatch at '+str(result['update']))
    return actual

def receiver_input_check(c,images,targets,traces,win,metadata):
    """Rehash received CPU bytes, independent of worker trace receipt."""
    require(len(images)==len(targets)==len(traces)==16,'not complete16 received samples')
    for image,target,trace in zip(images,targets,traces):
        h=hashlib.sha256();h.update(image.contiguous().numpy().tobytes())
        for name in ('boxes','labels','raw_area','ignore_spatial_boxes','ignore_union','ignore_class_boxes','ignore_class_labels'):
            h.update(name.encode());h.update(target[name].contiguous().numpy().tobytes())
        h.update(json.dumps([target['gt_ids'],target['ignore_ids']],separators=(',',':')).encode())
        require(h.hexdigest()==trace['augmented_tensor_sha256'],'received augmented image/target bytes differ from worker receipt')
    ref=expected_reference(c)
    if win<73:require(digest(metadata)==ref['updates'][win]['input_sha256'],'cross-machine input bytes/identity differ before update')
    else:require(metadata==ref['failed_input'],'target input bytes/identity differ before forward')

def acceptance(c,folder,holder):
    import torch
    from . import worker,checkpoint
    from .data import training_loader
    runtime=worker.prepare_cuda(c,folder);data=worker.dataset(c,'train_core');plan=worker.plan_epoch(data,27)
    engine,b,r=restored(c,runtime,holder);write(folder/'binding.json',b);event(folder,'resume_parent_exact',completed_updates=7904,completed_epochs=26)
    direct=[];saved=None;checkpoints=[]
    for win,(images,targets,traces) in enumerate(training_loader(data,epoch=27,window_limit=2)):
        metadata=worker.check_inputs(images,targets,traces,plan,27,win);holder['input']=metadata;receiver_input_check(c,images,targets,traces,win,metadata);worker.live_admission(folder,c)
        result=engine.train_window(images,targets,capture=True);direct.append({**result,'input':metadata});check_window(c,result,metadata,folder);event(folder,'update_complete',**result,completed_epochs=26)
        if win==0:saved=worker.save_state(engine,b,folder,'C7905.pth');checkpoints.append(saved)
    require(len(direct)==2 and engine.completed_updates==7906,'acceptance direct workload differs')
    cp=worker.save_state(engine,b,folder,'C7906-direct.pth');checkpoints.append(cp);expected=checkpoint.read(cp['path'],cp['sha256'],b)
    del holder['engine'];del engine;gc.collect();torch.cuda.empty_cache()
    clone,other=worker.make_engine(c,runtime,'gpu_short_window');other['lineage']=b['lineage'];require(other==b,'acceptance new binding differs');holder['engine']=clone
    from .cache_adapter import strict_restore
    replay_restore=strict_restore(clone,b,saved['path'],saved['sha256'],Path(c['runtime_root'])/'acceptance'/'replay-cache-import.json');replay=[]
    for images,targets,traces in training_loader(data,epoch=27,completed_windows=1,window_limit=1):
        metadata=worker.check_inputs(images,targets,traces,plan,27,1);holder['input']=metadata;receiver_input_check(c,images,targets,traces,1,metadata);worker.live_admission(folder,c)
        replay.append({**clone.train_window(images,targets,capture=True),'input':metadata})
    require(len(replay)==1,'acceptance replay expanded')
    require({k:v for k,v in direct[-1].items() if k!='elapsed_seconds'}=={k:v for k,v in replay[0].items() if k!='elapsed_seconds'},'same restored next window differs')
    require(checkpoint.exact_equal(expected,checkpoint.capture(clone,b)),'same restored next complete state/RNG differs')
    checkpoints.append(worker.save_state(clone,b,folder,'C7906-replay.pth'))
    publish(folder/'RESULT.json',{'status':'PASS','actual_updates':3,'updates':[7905,7906,7906],'original_restore':r,'replay_restore':replay_restore,
        'same_next_window_exact':True,'same_next_state_exact':True,'checkpoints':checkpoints,'state_for_diagnostic_or_formal':False,'gpu_peak_allocated':torch.cuda.max_memory_allocated(),'gpu_peak_reserved':torch.cuda.max_memory_reserved()})

def _target_forward_impl(engine,images,targets,c,folder,metadata):
    import copy,torch
    from .engine import _move,_batchnorm_backward_layout,assert_healthy
    from .distribution import require_finite
    from .operators import NumericalIntegrityError
    require(not engine.failed and engine.completed_updates==7977 and engine.ema.updates==7977 and engine.phase==0,'target7978 not at correct boundary')
    require(tuple(images.shape)==(16,3,768,1344) and len(targets)==16,'not full original target input shape')
    require_finite(images,'training images');require(not bool(((images<0)|(images>1)).any()),'target image range')
    for t in targets:
        require_finite(t['boxes'],'GT boxes');require(not bool((t['boxes'][:,2:]<=0).any()),'target GT positive dimensions')
        require(t['boxes'].shape==(len(t['labels']),4) and len(t['gt_ids'])==len(t['boxes']),'target GT/identity shape mismatch')
    engine.model.train();engine.criterion.train();require(not engine.model.decoder.trace_enabled,'diagnostic must preserve training path')
    from .engine import learning_rate
    for group in engine.optimizer.param_groups:group['lr']=learning_rate(7978)
    engine.optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize(engine.device)
    def forbidden(*args,**kwargs):raise RuntimeError('forbidden optimizer/EMA update7978')
    engine.optimizer.step=forbidden;engine.ema.update=forbidden
    with _batchnorm_backward_layout(engine.model,engine.device):
        for micro in range(2):
            subset=_move(targets[micro*8:(micro+1)*8],engine.device);batch=images[micro*8:(micro+1)*8].to(engine.device)
            try:
                with torch.autocast(engine.device.type,dtype=torch.bfloat16):output=engine.model(batch,targets=subset)
                with torch.autocast(engine.device.type,enabled=False):losses=engine.criterion(output,subset);objective=sum(losses.values())*.5
                require_finite(objective,'equal microbatch objective');objective.backward();engine.phase+=1;assert_healthy(engine.model,gradients=True)
                # Original loss conversion/diagnostic copy is retained before
                # the next physical forward, but never optimizer/EMA/cursor.
                values={k:float(v.detach()) for k,v in losses.items()};copy.deepcopy(engine.criterion.last_diagnostics)
                del output,losses,objective,batch,subset
            except NumericalIntegrityError as error:
                evidence=getattr(error,'evidence',{});require(evidence.get('kind')=='FINITE_DECODED_BOX_BELOW_FROZEN_GEOMETRY_BOUND','different/earlier numerical exception at target')
                evidence.update(microbatch=micro,completed_updates=7977,failed_update_candidate=7978,zero_window=73,input=metadata)
                for r in evidence['records']:
                    r['global_sample']=micro*8+r['sample'];r['augmented_sha256']=metadata['augmented_sha256'][r['global_sample']]
                small=channel.receipt_root(folder)/'BAD_BOX_EVIDENCE.json'
                publish(small,{'status':'TARGET_GEOMETRY_REPRODUCED','error':str(error),'evidence':evidence,'optimizer_update7978_submitted':False})
                if small!=folder/'BAD_BOX_EVIDENCE.json':publish(folder/'BAD_BOX_EVIDENCE.json',read(small))
                return {'status':'TARGET_GEOMETRY_REPRODUCED','microbatch':micro,'bad_queries':evidence['bad_queries'],'evidence':reference(folder/'BAD_BOX_EVIDENCE.json'),'optimizer_update7978_submitted':False}
    return {'status':'STOP_NOT_REPRODUCED','optimizer_update7978_submitted':False,'forward_microbatches':2}

def target_forward(engine,*args,**kwargs):
    try:return _target_forward_impl(engine,*args,**kwargs)
    finally:engine.failed=True

def diagnostic(c,folder,holder):
    from . import worker
    from .data import training_loader
    runtime=worker.prepare_cuda(c,folder);data=worker.dataset(c,'train_core');plan=worker.plan_epoch(data,27)
    engine,b,r=restored(c,runtime,holder);write(folder/'binding.json',b);event(folder,'resume_parent_exact',completed_updates=7904,completed_epochs=26)
    count=0
    for win,(images,targets,traces) in enumerate(training_loader(data,epoch=27,window_limit=73)):
        metadata=worker.check_inputs(images,targets,traces,plan,27,win);holder['input']=metadata;receiver_input_check(c,images,targets,traces,win,metadata);worker.live_admission(folder,c)
        result=engine.train_window(images,targets);actual=check_window(c,result,metadata,folder);event(folder,'update_complete',**result,completed_epochs=26);count+=1
    require(count==73 and engine.completed_updates==7977,'diagnostic prefix expanded/incomplete')
    targets_seen=0;result=None
    for images,targets,traces in training_loader(data,epoch=27,completed_windows=73,window_limit=1):
        metadata=worker.check_inputs(images,targets,traces,plan,27,73);require(metadata==expected_reference(c)['failed_input'],'target input identities/bytes differ')
        holder['input']=metadata;receiver_input_check(c,images,targets,traces,73,metadata);worker.live_admission(folder,c);result=target_forward(engine,images,targets,c,folder,metadata);targets_seen+=1
    require(targets_seen==1 and engine.completed_updates==engine.ema.updates==7977,'target load/update scope changed')
    publish(folder/'RESULT.json',{**result,'prefix_updates':73,'original_restore':r,'source_sha256':c['source_sha256'],'contract_sha256':c['contract_sha256'],'state_for_formal':False})

def worker_entry(path,mode):
    c=load(path);prerequisite(c,mode);supervisor_token(c,mode,path);guard.snapshot=snapshot;guard.screen=screen;folder=ROOT/mode;holder={'mode':mode};telemetry=Path(c['runtime_root'])/mode;started=time.time()
    require(read(telemetry/'supervisor-start.json')['pid']==os.getppid(),'lab worker supervisor ownership mismatch')
    dispatch=read(ROOT/(mode+'-dispatch.json'));ident=guard.identity(os.getppid())
    require(dispatch['contract_sha256']==c['contract_sha256'] and dispatch['mode']==mode and dispatch['pid']==os.getppid() and ident is not None and ident['start']==dispatch['process_identity']['start'],'worker dispatch PID/start changed')
    publish(folder/'execution-binding.json',execution_binding(c,mode))
    try:
        (acceptance if mode=='acceptance' else diagnostic)(c,folder,holder)
    except BaseException as error:
        evidence=getattr(error,'evidence',{});engine=holder.get('engine')
        if engine is not None:evidence={**evidence,'engine_phase':engine.phase,'completed_updates':engine.completed_updates}
        publish(telemetry/'failure.json',{'status':'STOP_NO_RETRY','error':repr(error),'traceback':traceback.format_exc(),'evidence':evidence,'input':holder.get('input')})
        raise
    finally:write(telemetry/'worker-exit.json',{'pid':os.getpid(),'mode':mode,'started_epoch':started,'finished_epoch':time.time(),'body_returned':not (telemetry/'failure.json').exists()})
    return 0

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('action',choices=['lab-prepare','lab-launch','lab-supervise','lab-worker']);p.add_argument('--request');p.add_argument('--contract');p.add_argument('--mode',choices=MODES);a=p.parse_args(argv)
    if a.action=='lab-prepare':r=prepared(a.request)
    elif a.action=='lab-launch':r=launch(a.contract,a.mode)
    elif a.action=='lab-supervise':return supervise(a.contract,a.mode)
    else:return worker_entry(a.contract,a.mode)
    print(json.dumps(r,allow_nan=False),flush=True);return 0
