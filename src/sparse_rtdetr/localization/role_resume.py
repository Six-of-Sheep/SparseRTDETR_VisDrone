"""One independent e045 continuation with restored lab resource constraints."""
import argparse,gc,gzip,json,os,subprocess,sys,time,traceback
from pathlib import Path
from . import channel,checkpoint,guard,lab_run,lab_consumer,numerics,consumer_geometry,worker
from .runio import require,read,write,publish,sha,digest,reference,event,execution_binding,original_process_exited
SOURCE=Path(__file__).resolve().parents[3];ROOT=SOURCE.parent.parent;BASE=ROOT.parent
PARENT_SHA='3362fcc7be58377e0b70f7cca0bdee997a9f8f17b981cc855147a827046e7c7a'
DIAGNOSTIC_SHA='e194724095885d266359122d77f62b94235335057efad3ab799d004a2b1d3904'
PREVIOUS_SOURCE='6956c5ecb309bebed9256fe32496cfa46b91352e26a18983cc1537690ec26351'
PREVIOUS_CONTRACT='236a97c1896ea2201ff1aa21441ecbb20a58a44f0f42d62ff8de11a40636af58'
MODES=('diagnostic','acceptance','formal','trace-cache','trace-analysis')
WORKLOAD={'diagnostic_parent_update':13693,'diagnostic_epoch':46,'diagnostic_window':13,'diagnostic_actual_updates':0,
          'acceptance_updates':[13694,13695,13695],'acceptance_actual_updates':3,
          'formal_parent_update':13680,'formal_remaining_updates':22800,'formal_end_update':36480,
          'formal_endpoint':'e120_EMA','trace_images':64,'trace_batch_forwards':16}
LIMITS={'CPU_PL1_uW':125000000,'CPU_PL2_uW':150000000,'GPU_locked_min_MHz':1500,'GPU_locked_max_MHz':1500,'CPU_temperature_C':90,'GPU_temperature_C':80}
COMPARISON='C13693 complete original restore; only e046 window13 two micro forward/loss/backward zero update; fixed13694/13695/replay13695 exact result/state/RNG; formal independently original e04513680; no diagnostic or acceptance state for formal'
ACTIVE_RUNTIME=None

def snapshot(path):
    sample=lab_run.snapshot(path);cap=Path('/sys/class/powercap/intel-rapl:0')
    sample['cpu_power_constraints']={name:(cap/name).read_text().strip() for name in ('constraint_0_name','constraint_1_name','constraint_0_power_limit_uw','constraint_1_power_limit_uw')}
    return sample

def cap_check(sample):
    cap=sample['cpu_power_constraints'];require(cap['constraint_0_name']=='long_term' and cap['constraint_1_name']=='short_term','CPU power constraint identities differ')
    require(0<int(cap['constraint_0_power_limit_uw'])<=125000000 and 0<int(cap['constraint_1_power_limit_uw'])<=150000000,'restored CPU power caps changed')
    clock=sample['snapshot']['gpu']['devices'][0]['clocks.current.graphics'];require(clock['status']=='OBSERVED' and 0<float(clock['value'])<=1500,'GPU exceeds restored 1500MHz cap')

def screen(sample,**kwargs):
    try:
        cap_check(sample)
        return lab_consumer.screen(sample,**kwargs)
    except RuntimeError as error:
        # Preserve the rejected real sample BEFORE unwinding the unchanged guard.
        if ACTIVE_RUNTIME is not None and Path(ACTIVE_RUNTIME).is_dir():
            publish(Path(ACTIVE_RUNTIME)/('rejected-resource-'+str(time.time_ns())+'.json'),
                    {'status':'REJECTED_RESOURCE_SAMPLE_NOT_ADMITTED','error':repr(error),'sample':sample,'identity':read(Path(ACTIVE_RUNTIME)/'identity.json')})
        raise

def authorization(c):
    return {'parent_checkpoint_sha256':DIAGNOSTIC_SHA,'parent_binding_sha256':sha(c['diagnostic_binding']),
            'parent_contract_sha256':read(c['diagnostic_contract'])['contract_sha256'],'parent_update':13693,'epoch':46,'window':13,'micro_phase_sequence':[[0,1],[1,2]]}
def identity(c,mode):
    r={**execution_binding(c,mode),'progress_protocol':c['progress_protocol']}
    if mode=='diagnostic':r['diagnostic_authorization']=c['diagnostic_authorization']
    return r

def verify_source(c):
    require(c['source_root']==str(SOURCE) and sha(c['source_manifest'])==c['source_sha256'],'role source manifest changed')
    for r in read(c['source_manifest'])['files']:
        p=SOURCE/r['path'];require(p.is_file() and not p.is_symlink() and p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],'role frozen source changed: '+r['path'])
    require(c['numerical_implementation']==numerics.identity() and c['consumer_geometry_contract']==consumer_geometry.identity(),'arithmetic or consumer contract changed')
    require(c['progress_protocol']==channel.PROTOCOL_NAME,'progress protocol changed')

def verify_materials(c):
    require(len(c['materials'])==len({r['path'] for r in c['materials']}),'duplicate material')
    required={c[k] for k in ('parent_binding','parent_sidecar','parent_contract','diagnostic_binding','diagnostic_sidecar','diagnostic_contract','data_verification','selection_path','fixed_inputs','caps_receipt','caps_script','cpu_protocol_positive','cpu_protocol_negative','cpu_protocol_program','cpu_negative_program','criterion_diff')}
    required|={c['data'][r]['manifest'] for r in ('train_core','development')}|{c['pretrained']['path']}
    require(required<={r['path'] for r in c['materials']},'required role material missing')
    for r in c['materials']:
        p=Path(r['path']);require(p.is_absolute() and p.is_relative_to(BASE) and not p.is_symlink() and reference(p)==r,'role material changed: '+str(p))
    for prefix,want,u,scope in (('parent',PARENT_SHA,13680,'formal_F0_seed0_e120'),('diagnostic',DIAGNOSTIC_SHA,13693,'gpu_short_window')):
        side=read(c[prefix+'_sidecar']);binding=read(c[prefix+'_binding']);contract=read(c[prefix+'_contract']);path=c[prefix+'_checkpoint']
        require(sha(path)==want and side['sha256']==want and side['size_bytes']==Path(path).stat().st_size and side['path']==path,'original '+prefix+' bytes/sidecar mismatch')
        require(side['binding_sha256']==digest(binding) and side['completed_updates']==u and side['scope']==scope,'original '+prefix+' binding/scope/clock mismatch')
        require(contract['contract_sha256']==digest({k:v for k,v in contract.items() if k!='contract_sha256'})==binding['contract_sha256'] and contract['source_sha256']==binding['source_sha256'],'original '+prefix+' source/contract mismatch')
        require(c[prefix+'_lineage']=={'parent_checkpoint_sha256':want,'parent_binding_sha256':sha(c[prefix+'_binding']),'parent_source_sha256':binding['source_sha256'],'parent_contract_sha256':contract['contract_sha256'],'parent_update':u,'formal_forbidden':prefix=='diagnostic'},'role lineage differs')
    require(read(c['parent_binding'])['source_sha256']==PREVIOUS_SOURCE and read(c['parent_contract'])['contract_sha256']==PREVIOUS_CONTRACT,'original e045 source differs')
    dc=read(c['diagnostic_contract'])
    require(dc['source_sha256']=='19f691e64f9566a42dea3fc64dbc7944c81b96f5f51b7a94d18cc956fa56cf46' and dc['contract_sha256']=='4d86fab6c1074937d6250b2fe545691c87182f695c5380a2dc7ea80f5ac5f667' and dc['parent_lineage']['parent_checkpoint_sha256']==PARENT_SHA,'isolated C13693 is not old e045 descendant')
    require(c['diagnostic_authorization']==authorization(c),'diagnostic authorization differs')
    proof=read(c['data_verification']);require(proof['status']=='PASS' and proof['named_pairs']==4933 and proof['train_core']==4869 and proof['development_fixed64']==64 and c['data_sha256']==sha(c['data_verification']),'existing data closure differs')
    fixed=read(c['fixed_inputs']);require(fixed['epoch']==46 and fixed['windows']==[13,14] and fixed['gpu_calls']==0,'role input scope differs')
    cap=read(c['caps_receipt']);require(c['resource_limits']==LIMITS and cap['status']=='PRIOR_CPU_125W_150W_GPU_1500MHZ_REQUEST_APPLIED' and cap['gpu_uuid']==c['expected_gpu_uuid'],'cap receipt differs')
    require(cap['script_sha256']==sha(c['caps_script']) and cap['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'cap script/boot differs')
    op=cap['operations'][-1];require(op['returncode']==0 and op['command']==['nvidia-smi','-i',c['expected_gpu_uuid'],'--lock-gpu-clocks=1500,1500'],'GPU lock not authenticated')
    happy=read(c['cpu_protocol_positive']);neg=read(c['cpu_protocol_negative'])
    require(happy['status']=='REAL_CPU_PROTOCOL_CHAIN_CLEAN_EXIT_AND_ARCHIVE_PASS' and happy['semantic_operations']==2 and set(happy['observed_phases'])=={1,2} and happy['observed_phases']==sorted(happy['observed_phases']),'real role CPU chain incomplete')
    require(happy['gpu_calls']==happy['model_forward_calls']==0 and not happy['torch_imported'] and happy['supervisor_exit']['status']=='PASS' and not any(happy['supervisor_exit'][k] for k in ('remaining_live','unknown_pids','cleanup_errors')),'real role CPU chain scope/exit differs')
    require(neg['status']=='REAL_CPU_PRODUCER_OBSERVER_ARCHIVE_NEGATIVE_CASES_PASS' and len(neg['cases'])==20 and neg['gpu_calls']==neg['model_forward_calls']==0 and not neg['torch_imported'],'role CPU negative evidence absent')
    for n,r in happy['implementation'].items():
        require(n in ('channel.py','runio.py','guard.py') and all(reference(SOURCE/'src/sparse_rtdetr/localization'/n)[k]==r[k] for k in ('bytes','sha256')),'executing channel differs from real CPU proof')
    require(set(happy['implementation'])=={'channel.py','runio.py','guard.py'},'real CPU implementation binding incomplete')
    require(read(c['criterion_diff'])['status']=='REAL_F0CRITERION_38_LOSS_ALL_OUTPUT_AND_MODEL_GRADIENTS_EXACT','real criterion difference proof absent')

def load(path):
    c=read(path);require(c['contract_sha256']==digest({k:v for k,v in c.items() if k!='contract_sha256'}),'role contract digest differs')
    require(c['job_root']==str(ROOT) and ROOT.name=='terminal-dn-role-002' and c['runtime_root']=='/home/lyy/.cache/f0-e026-labdiag-20261009-001/runtime/terminal-dn-role-002' and c['workload']==WORKLOAD and c['comparison_policy']==COMPARISON,'role root/workload/comparison differs')
    require(c['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'role boot changed')
    verify_source(c);verify_materials(c);return c

def prepare(path):
    c=read(path)
    require('contract_sha256' not in c,'freeze request must not inherit an old contract digest')
    require(not (ROOT/'contract.json').exists() and c['job_root']==str(ROOT) and c['workload']==WORKLOAD and c['comparison_policy']==COMPARISON,'role freeze already used or workload differs')
    verify_source(c);verify_materials(c)
    import torch,torchvision,numpy,xml.etree.ElementTree as ET
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(),'role freeze must remain CPU only')
    actual={'python':sys.version.split()[0],'torch':str(torch.__version__),'torchvision':torchvision.__version__,'numpy':numpy.__version__,'cuda_build':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'prefix':sys.prefix}
    require(actual['python']=='3.10.16' and actual['torch']=='2.4.1' and actual['cuda_build']=='12.4' and actual['cudnn']==90100 and str(Path(sys.executable))==c['python'],'role runtime differs')
    require(c['new_cpu_checks'] and all(p['tests']>0 for p in c['new_cpu_checks']),'new role CPU gate absent')
    for p in c['new_cpu_checks']:
        require(reference(p['xml']['path'])==p['xml'] and reference(p['log']['path'])==p['log'],'role CPU evidence changed')
        cases=list(ET.parse(p['xml']['path']).getroot().iter('testcase'));require(len(cases)==p['tests'] and all(not any(t.find(k) is not None for k in ('error','failure','skipped')) for t in cases),'role CPU checks incomplete')
    s=snapshot(ROOT);uid=screen(s,entry=True,disk_bytes=40*1024**3+90*512*1024**2);require(uid==c['expected_gpu_uuid'],'role GPU differs')
    from .storage import identity as storage_identity
    c.update(schema='f0-terminal-dn-role-continuation-v1',gpu_uuid=uid,boot_id=s['boot_id'],environment={k:actual[k] for k in ('torch','torchvision','cuda_build','cudnn')},
             environment_observation=actual,storage={'identity':storage_identity(s['storage'])},resource_observation=s,freeze_epoch=time.time())
    c['contract_sha256']=digest(c);publish(ROOT/'contract.json',c);return c

def under_load_proof(c,runtime):
    runtime=Path(runtime);b=read(runtime/'identity.json');reader=channel.Reader(runtime/'resources.jsonl',b,'resources');reader.poll()
    require(reader.offset==reader.path.stat().st_size,'unclosed load-resource record')
    samples=[]
    for line in (runtime/'resources.jsonl').read_text().splitlines():
        p=json.loads(line)['payload'];d=p['snapshot']['gpu']['devices'][0]
        if p['compute_apps'] and d['uuid']['value']==c['gpu_uuid'] and d['utilization.gpu']['value']>1:
            cap_check(p);samples.append({'sampled_epoch':p['sampled_epoch'],'GPU_graphics_MHz':d['clocks.current.graphics']['value'],'GPU_utilization_percent':d['utilization.gpu']['value'],
                'CPU_temperature_C':p['snapshot']['cpu_temperature_max'],'GPU_temperature_C':d['temperature.gpu']['value'],'CPU_caps':p['cpu_power_constraints']})
    require(samples and any(p['GPU_graphics_MHz']==1500 for p in samples),'no authenticated1500MHz sample during actual acceptance load; no GPU retry')
    return {'status':'ACTUAL_AUTHENTICATED_LOAD_CAPS_PASS','resources':reference(runtime/'resources.jsonl'),'busy_samples':len(samples),'observed_graphics_MHz':sorted(set(p['GPU_graphics_MHz'] for p in samples)),
            'CPU_temperature_max_C':max(p['CPU_temperature_C'] for p in samples),'GPU_temperature_max_C':max(p['GPU_temperature_C'] for p in samples),'samples':samples}

def complete(c,mode):
    runtime=Path(c['runtime_root'])/mode;folder=ROOT/mode;ex=read(runtime/'supervisor-exit.json');d=read(ROOT/(mode+'-dispatch.json'))
    require(ex['status']=='PASS' and ex['returncode']==0 and not ex['remaining_live'] and not ex['unknown_pids'] and not ex['cleanup_errors'] and original_process_exited(d,guard.identity(d['pid'])),'role phase not cleanly complete')
    require(d['mode']==mode and d['contract_sha256']==c['contract_sha256'],'role dispatch differs')
    channel.verify_archive(runtime,folder,{**identity(c,mode),'supervisor_pid':d['pid'],'supervisor_start':d['process_identity']['start']})
    require(read(folder/'execution-binding.json')==execution_binding(c,mode),'role execution belongs to another source')
    for name in ('RESULT.json','execution-binding.json','worker-exit.json','binding.json'):
        p=runtime/name if name=='worker-exit.json' else folder/name;require(ex['completion_receipts'][name]==reference(p),'role receipt changed')
    r=read(folder/'RESULT.json')
    if mode=='diagnostic':
        require(r['status']=='PASS' and r['forward_microbatches']==r['backward_microbatches']==2 and r['actual_updates']==0 and r['completed_updates']==13693 and r['phase_after']==2,'role diagnostic workload incomplete')
        require(r['original_restore']['checkpoint_sha256']==DIAGNOSTIC_SHA and r['original_restore']['full_state_exact'] and r['optimizer_exact_unchanged'] and r['ema_exact_unchanged'] and r['learned_parameters_exact_unchanged'] and r['raw_gradients_finite'] and r['all_required_consumers_observed'],'role diagnostic state/consumers failed')
        receipt=read(folder/'original-restore.json');require(receipt==read(runtime/'original-restore.json') and receipt['result']==r['original_restore'] and receipt['execution']==identity(c,mode) and receipt['phase']==0,'independent full restore receipt differs')
        reader=channel.Reader(runtime/'progress.jsonl',read(runtime/'identity.json'),'progress');reader.poll();require(reader.clock.get('_diagnostic_micro')==1,'both real diagnostic micro phases not authenticated')
    elif mode=='acceptance':
        require(r['status']=='PASS' and r['actual_updates']==3 and r['updates']==[13694,13695,13695] and r['same_next_window_exact'] and r['same_next_state_exact'] and r['original_restore']['checkpoint_sha256']==DIAGNOSTIC_SHA and not r['state_for_formal'],'role exact3 failed')
        under_load_proof(c,runtime)
    elif mode=='formal':require(r['status']=='F0_E120_EMA_LOCKED' and r['updates']==36480 and r['actual_updates']==22800 and r['original_restore']['checkpoint_sha256']==PARENT_SHA and not r['used_acceptance_checkpoint'],'role formal endpoint incomplete')
    else:require(r['status']=='PASS' and r['images']==64,'role fixed64 incomplete')
    return r

def prerequisite(c,mode):
    require(mode in MODES,'unknown role mode')
    for prior in MODES[:MODES.index(mode)]:complete(c,prior)

def command(c,path,mode):return [c['python'],'-B',str(SOURCE/'tools/localization/deploy.py'),'role-supervise','--contract',str(path),'--mode',mode]

def token(c,path,mode):
    r=read(ROOT/(mode+'-launch-consumed.json'));require(r['command']==command(c,path,mode) and r['contract_sha256']==c['contract_sha256'] and r['mode']==mode and r['no_retry'],'role launch token differs')

def launch(path,mode):
    c=load(path);prerequisite(c,mode);require(not (ROOT/mode).exists(),'role phase already used; no GPU retry')
    screen(snapshot(ROOT),entry=True,gpu=mode!='trace-analysis',disk_bytes=40*1024**3+(90*512*1024**2 if mode=='formal' else 0));cmd=command(c,path,mode)
    publish(ROOT/(mode+'-launch-consumed.json'),{'command':cmd,'mode':mode,'contract_sha256':c['contract_sha256'],'no_retry':True,'epoch':time.time()})
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MKL_THREADING_LAYER='GNU',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (ROOT/(mode+'-dispatcher.log')).open('x') as log:p=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    r={'pid':p.pid,'process_identity':guard.identity(p.pid),'contract_sha256':c['contract_sha256'],'mode':mode,'started_epoch':time.time()};publish(ROOT/(mode+'-dispatch.json'),r);return r

def supervise(path,mode):
    global ACTIVE_RUNTIME
    c=load(path);prerequisite(c,mode);token(c,path,mode);guard.snapshot=snapshot;guard.screen=screen;ACTIVE_RUNTIME=Path(c['runtime_root'])/mode
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if mode=='trace-analysis' else c['gpu_uuid'],OMP_NUM_THREADS='16' if mode=='trace-analysis' else '2',MKL_NUM_THREADS='16' if mode=='trace-analysis' else '2')
    cmd=[c['python'],'-B',str(SOURCE/'tools/localization/deploy.py'),'role-worker','--contract',str(path),'--mode',mode]
    return guard.supervise(cmd,ROOT/mode,env,gpu_uuid=c['gpu_uuid'],boot_id=c['boot_id'],gpu=mode!='trace-analysis',checkpoint_bytes=512*1024**2,formal=mode=='formal',
        runtime=ACTIVE_RUNTIME,run_identity={**identity(c,mode),'supervisor_pid':os.getpid(),'supervisor_start':guard.identity(os.getpid())['start']})

def restored_parent(c,runtime,holder,scope):
    engine,binding=worker.make_engine(c,runtime,scope);holder['engine']=engine
    from .cache_adapter import strict_restore
    r=strict_restore(engine,read(c['parent_binding']),c['parent_checkpoint'],PARENT_SHA,Path(c['runtime_root'])/holder['mode']/'parent-cache-import.json')
    require(r['full_state_exact'] and engine.completed_updates==engine.ema.updates==13680 and engine.phase==0,'e045 full restore failed');numerics.validate_binding(binding,c)
    receipt={'status':'ORIGINAL_E045_STRICT_FULL_RESTORE_PASS','result':r,'execution':identity(c,holder['mode']),'parent_binding':reference(c['parent_binding']), 'parent_sidecar':reference(c['parent_sidecar'])}
    publish(Path(c['runtime_root'])/holder['mode']/'original-restore.json',receipt);publish(ROOT/holder['mode']/'original-restore.json',receipt);return engine,binding,r

def input_check(c,images,targets,traces,metadata,window):
    import hashlib
    require(window in (13,14) and len(images)==len(targets)==len(traces)==16,'role fixed input expanded')
    expected=read(c['fixed_inputs'])['inputs'][str(window)];require(metadata==expected['metadata'] and digest(metadata)==expected['metadata_digest'] and [t['gt_ids'] for t in targets]==expected['gt_ids'],'role input/GT differs')
    for image,target,trace in zip(images,targets,traces):
        h=hashlib.sha256();h.update(image.contiguous().numpy().tobytes())
        for k in ('boxes','labels','raw_area','ignore_spatial_boxes','ignore_union','ignore_class_boxes','ignore_class_labels'):h.update(k.encode());h.update(target[k].contiguous().numpy().tobytes())
        h.update(json.dumps([target['gt_ids'],target['ignore_ids']],separators=(',',':')).encode());require(h.hexdigest()==trace['augmented_tensor_sha256'],'role augmented bytes differ')

def restored_diagnostic(c,runtime,holder):
    engine,binding=worker.make_engine(c,runtime,'gpu_short_window');holder['engine']=engine
    from .cache_adapter import strict_restore
    restored=strict_restore(engine,read(c['diagnostic_binding']),c['diagnostic_checkpoint'],DIAGNOSTIC_SHA,Path(c['runtime_root'])/holder['mode']/'parent-cache-import.json')
    require(restored['full_state_exact'] and engine.completed_updates==engine.ema.updates==13693,'C13693 original full restore failed')
    numerics.validate_binding(binding,c)
    receipt={'status':'ORIGINAL_C13693_STRICT_FULL_RESTORE_PASS','result':restored,'parent_binding':reference(c['diagnostic_binding']),'parent_sidecar':reference(c['diagnostic_sidecar']),
             'execution':identity(c,holder['mode']),'completed_updates':engine.completed_updates,'ema_updates':engine.ema.updates,'phase':engine.phase,'state_for_formal':False}
    telemetry=Path(c['runtime_root'])/holder['mode'];publish(telemetry/'original-restore.json',receipt)
    publish(ROOT/holder['mode']/'original-restore.json',receipt)
    return engine,binding,restored


def diagnostic(c,folder,holder):
    import copy,torch
    from .data import training_loader
    from ..baseline.training_v2b_engine import _batchnorm_backward_layout,_move
    from .engine import assert_healthy,tensor_digest,output_digest,learning_rate
    runtime=worker.prepare_cuda(c,folder);data=worker.dataset(c,'train_core');plan=worker.plan_epoch(data,46)
    engine,binding,parent=restored_diagnostic(c,runtime,holder);write(folder/'binding.json',binding)
    before_optimizer=copy.deepcopy(engine.optimizer.state_dict());before_ema=copy.deepcopy(engine.ema.state_dict());before_parameters={n:p.detach().clone() for n,p in engine.model.named_parameters()}
    rng_before=checkpoint.capture_rng(engine.device);audit=consumer_geometry.Audit();holder['audit']=audit;outputs=[];loss_rows=[];forward=backward=windows=0
    engine.model.train();engine.criterion.train();engine.optimizer.zero_grad(set_to_none=True)
    original_step=engine.optimizer.step;original_ema=engine.ema.update
    def forbidden(*args,**kwargs):raise RuntimeError('diagnostic cannot submit optimizer/EMA update')
    engine.optimizer.step=forbidden;engine.ema.update=forbidden
    try:
        for images,targets,traces in training_loader(data,epoch=46,completed_windows=13,window_limit=1):
            metadata=worker.check_inputs(images,targets,traces,plan,46,13);holder['input']=metadata;input_check(c,images,targets,traces,metadata,13);worker.live_admission(folder,c);windows+=1
            with _batchnorm_backward_layout(engine.model,engine.device),consumer_geometry.auditing(audit):
                for micro in range(2):
                    audit.microbatch=micro;subset=_move(targets[micro*8:(micro+1)*8],engine.device);batch=images[micro*8:(micro+1)*8].to(engine.device)
                    with torch.autocast(engine.device.type,dtype=torch.bfloat16):out=engine.model(batch,targets=subset)
                    forward+=1;outputs.append(output_digest(out))
                    with torch.autocast(engine.device.type,enabled=False):losses=engine.criterion(out,subset);objective=sum(losses.values())*.5
                    consumer_geometry.tensor('diagnostic/objective',objective);objective.backward();engine.phase+=1;backward+=1;assert_healthy(engine.model,gradients=True)
                    loss_rows.append({k:float(v.detach()) for k,v in losses.items()});event(folder,'terminal_dn_micro_backward',diagnostic_parent_sha256=DIAGNOSTIC_SHA,microbatch=micro,epoch=46,window=13,completed_updates=13693,phase=engine.phase)
                    del out,losses,objective,batch,subset
            require(windows==1 and forward==backward==2 and engine.phase==2 and engine.completed_updates==engine.ema.updates==13693,'diagnostic scope/cursor changed')
        require(checkpoint.exact_equal(before_optimizer,engine.optimizer.state_dict()) and checkpoint.exact_equal(before_ema,engine.ema.state_dict()) and checkpoint.exact_equal(before_parameters,{n:p.detach() for n,p in engine.model.named_parameters()}),'diagnostic submitted state update')
        lab_consumer.consumers_complete(audit)
        gradients={n:None if p.grad is None else p.grad.detach().cpu().clone() for n,p in engine.model.named_parameters()}
        tensor_count=sum(v is not None for v in gradients.values());gradient_sha=tensor_digest(gradients.items())
        with (folder/'raw-gradients.pth').open('xb') as f:torch.save({'schema':'consumer-diagnostic-raw-unclipped-gradients-v1','parent_sha256':DIAGNOSTIC_SHA,'completed_updates':13693,'phase':2,'gradients':gradients},f);f.flush();os.fsync(f.fileno())
        grad_ref=reference(folder/'raw-gradients.pth')
        publish(folder/'raw-gradient-inventory.json',{n:{'parameter_shape':list(p.shape),'parameter_dtype':str(p.dtype),'gradient_present':gradients[n] is not None,
                 'gradient_shape':list(gradients[n].shape) if gradients[n] is not None else None,'gradient_dtype':str(gradients[n].dtype) if gradients[n] is not None else None,
                 'gradient_sha256':tensor_digest([(n,gradients[n])])} for n,p in engine.model.named_parameters()})
        publish(folder/'consumer-observations.json',audit.summary())
        publish(folder/'RESULT.json',{'status':'PASS','forward_microbatches':2,'backward_microbatches':2,'actual_updates':0,'completed_updates':13693,'phase_after':2,
                'original_restore':parent,'optimizer_exact_unchanged':True,'ema_exact_unchanged':True,'learned_parameters_exact_unchanged':True,
                'raw_gradients_finite':True,'raw_gradient_sha256':gradient_sha,'raw_gradient_tensor_count':tensor_count,'raw_gradients':grad_ref,
                'all_required_consumers_observed':True,'losses':loss_rows,'output_sha256':outputs,'planned_lr13694':learning_rate(13694),'optimizer_lr_write_performed':False,
                'rng_before_after_equal':checkpoint.exact_equal(rng_before,checkpoint.capture_rng(engine.device)),'state_for_formal':False,'geometry_contract':consumer_geometry.identity()})
    finally:
        engine.optimizer.step=original_step;engine.ema.update=original_ema
        engine.failed=True
        # Keep phase2/raw gradients. Never clip/reset phase/save resumable diagnostic state.


def acceptance(c,folder,holder):
    import torch
    from .data import training_loader
    runtime=worker.prepare_cuda(c,folder);data=worker.dataset(c,'train_core');plan=worker.plan_epoch(data,46)
    engine,binding,parent=restored_diagnostic(c,runtime,holder);write(folder/'binding.json',binding);results=[]
    for win,(images,targets,traces) in enumerate(training_loader(data,epoch=46,completed_windows=13,window_limit=2),13):
        metadata=worker.check_inputs(images,targets,traces,plan,46,win);holder['input']=metadata;input_check(c,images,targets,traces,metadata,win);worker.live_admission(folder,c)
        r=engine.train_window(images,targets,capture=True);lab_consumer.clock_check(engine,r,13681+win);r['input']=metadata;results.append(r);event(folder,'update_complete',**r,completed_epochs=45)
        if win==13:saved=worker.save_state(engine,binding,folder,'C13694.pth')
    require(len(results)==2 and engine.completed_updates==13695,'acceptance workload changed')
    direct=worker.save_state(engine,binding,folder,'C13695-direct.pth');expected_state=checkpoint.read(direct['path'],direct['sha256'],binding);holder.pop('engine');del engine;gc.collect();torch.cuda.empty_cache()
    clone,new_binding=worker.make_engine(c,runtime,'gpu_short_window');holder['engine']=clone;require(new_binding==binding,'new same-next binding differs')
    restored=numerics.restore_active(clone,binding,saved['path'],saved['sha256'],Path(c['runtime_root'])/'acceptance/replay-cache-import.json',c)
    count=0
    for images,targets,traces in training_loader(data,epoch=46,completed_windows=14,window_limit=1):
        metadata=worker.check_inputs(images,targets,traces,plan,46,14);holder['input']=metadata;input_check(c,images,targets,traces,metadata,14);worker.live_admission(folder,c)
        r=clone.train_window(images,targets,capture=True);lab_consumer.clock_check(clone,r,13695);r['input']=metadata;count+=1
    require(count==1 and {k:v for k,v in r.items() if k!='elapsed_seconds'}=={k:v for k,v in results[-1].items() if k!='elapsed_seconds'},'same next result/output/loss/gradient/input mismatch')
    require(checkpoint.exact_equal(expected_state,checkpoint.capture(clone,binding)),'same next complete state/RNG mismatch')
    replay=worker.save_state(clone,binding,folder,'C13695-replay.pth');publish(folder/'RESULT.json',{'status':'PASS','actual_updates':3,'updates':[13694,13695,13695],
            'original_restore':parent,'replay_restore':restored,'same_next_window_exact':True,'same_next_state_exact':True,'checkpoints':[saved,direct,replay],
            'state_for_formal':False,'consumer_geometry_contract':consumer_geometry.identity(),'complete_update_seconds':[v['elapsed_seconds'] for v in results]+[r['elapsed_seconds']]})

def formal(c,folder,holder):
    from .data import training_loader
    runtime=worker.prepare_cuda(c,folder);data=worker.dataset(c,'train_core');engine,binding,parent=restored_parent(c,runtime,holder,'formal_F0_seed0_e120');write(folder/'binding.json',binding)
    event(folder,'resume_parent_exact',**parent,scope='formal_F0_seed0_e120',used_acceptance_checkpoint=False);started=time.monotonic();times=[];count=0
    with gzip.open(folder/'input-acknowledgements.jsonl.gz','xt',encoding='utf-8') as ack:
        for epoch in range(46,121):
            verify_source(c);require(engine.completed_updates==(epoch-1)*304,'role formal epoch clock differs')
            plan=worker.plan_epoch(data,epoch);write(folder/f'input-plan-epoch-{epoch:03d}.json',plan);event(folder,'epoch_begin',epoch=epoch,completed_epochs=epoch-1,dropped_ids=plan['ids'][4864:]);t=time.monotonic();windows=0
            for win,(images,targets,traces) in enumerate(training_loader(data,epoch=epoch)):
                meta=worker.check_inputs(images,targets,traces,plan,epoch,win);holder['input']=meta;worker.live_admission(folder,c)
                if epoch==46 and win in (13,14):input_check(c,images,targets,traces,meta,win)
                r=engine.train_window(images,targets);lab_consumer.clock_check(engine,r,(epoch-1)*304+win+1)
                ack.write(json.dumps(meta,allow_nan=False)+'\n');ack.flush();event(folder,'update_complete',**r,completed_epochs=engine.completed_updates//304,input_sha256=digest(meta));windows+=1;count+=1
            require(windows==304 and engine.completed_updates==epoch*304,'role formal epoch incomplete')
            last=worker.save_state(engine,binding,folder,f'checkpoint-epoch-{epoch:03d}.pth');duration=time.monotonic()-t;times.append(duration)
            publish(folder/f'epoch-{epoch:03d}-complete.json',{'epoch':epoch,'updates':engine.completed_updates,'checkpoint':last,'epoch_wall_seconds':duration,'strong_augmentation':epoch<118})
    require(count==22800 and engine.completed_updates==engine.ema.updates==36480,'role formal endpoint count differs')
    require(checkpoint.exact_equal(checkpoint.read(last['path'],last['sha256'],binding),checkpoint.capture(engine,binding)),'role e120 disk/live differs')
    endpoint={'status':'F0_E120_EMA_LOCKED','arm':'F0','seed':0,'epoch':120,'updates':36480,'checkpoint':last,'binding':reference(folder/'binding.json'),'source_sha256':c['source_sha256'],'contract_sha256':c['contract_sha256'],
              'original_restore':parent,'used_acceptance_checkpoint':False,'parent_update':13680,'actual_updates':count,'training_wall_seconds':time.monotonic()-started,'epoch_wall_seconds':times,
              'laboratory_actual_cost':None,'evaluation_performed':False,'additional_arms':[],'holdout_accessed':False,'numerical_implementation':numerics.identity()}
    publish(folder/'endpoint-lock.json',endpoint);publish(folder/'RESULT.json',endpoint)

def worker_entry(path,mode):
    global ACTIVE_RUNTIME
    c=load(path);prerequisite(c,mode);token(c,path,mode);guard.snapshot=snapshot;guard.screen=screen
    folder=ROOT/mode;runtime=Path(c['runtime_root'])/mode;holder={'mode':mode};started=time.time();d=read(ROOT/(mode+'-dispatch.json'));b=read(runtime/'identity.json')
    current=guard.identity(os.getppid());require(current and current['start']==d['process_identity']['start'] and os.getppid()==d['pid'] and read(runtime/'supervisor-start.json')['pid']==os.getppid(),'role supervisor identity differs')
    require(all(b.get(k)==v for k,v in identity(c,mode).items()),'role runtime identity differs');ACTIVE_RUNTIME=runtime;publish(folder/'execution-binding.json',execution_binding(c,mode))
    from . import control
    control.verify_source=verify_source
    try:
        if mode=='diagnostic':diagnostic(c,folder,holder)
        elif mode=='acceptance':acceptance(c,folder,holder)
        elif mode=='formal':formal(c,folder,holder)
        else:
            write(folder/'binding.json',{'execution':execution_binding(c,mode),'numerical_implementation':numerics.identity()})
            (worker.run_trace_cache if mode=='trace-cache' else worker.run_trace_analysis)(c,folder,holder)
    except BaseException as error:
        engine=holder.get('engine');evidence=getattr(error,'evidence',{})
        if engine is not None:evidence={**evidence,'engine_phase':engine.phase,'completed_updates':engine.completed_updates}
        publish(runtime/'failure.json',{'status':'STOP_NO_RETRY','error':repr(error),'traceback':traceback.format_exc(),'evidence':evidence,'input':holder.get('input')});raise
    finally:write(runtime/'worker-exit.json',{'pid':os.getpid(),'mode':mode,'started_epoch':started,'finished_epoch':time.time(),'body_returned':not (runtime/'failure.json').exists()})
    return 0

def start_chain(path):
    c=load(path);complete(c,'acceptance');cmd=[c['python'],'-B',str(SOURCE/'tools/localization/deploy.py'),'role-chain-worker','--contract',str(path)]
    publish(ROOT/'continuation-chain-consumed.json',{'contract_sha256':c['contract_sha256'],'source_sha256':c['source_sha256'],'command':cmd,'modes':list(MODES[2:]),'no_retry':True,'epoch':time.time()})
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MKL_THREADING_LAYER='GNU',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (ROOT/'continuation-chain.log').open('x') as log:p=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    d={'pid':p.pid,'process_identity':guard.identity(p.pid),'contract_sha256':c['contract_sha256'],'started_epoch':time.time()};publish(ROOT/'continuation-chain-dispatch.json',d);return d

def chain_worker(path):
    c=load(path);r=read(ROOT/'continuation-chain-consumed.json');require(r['contract_sha256']==c['contract_sha256'] and r['source_sha256']==c['source_sha256'] and r['modes']==list(MODES[2:]) and r['no_retry'] and r['command']==[c['python'],'-B',str(SOURCE/'tools/localization/deploy.py'),'role-chain-worker','--contract',str(path)],'role chain token differs')
    started=time.time();phase=None
    try:
        complete(c,'acceptance')
        for phase in MODES[2:]:
            d=launch(path,phase)
            while True:
                runtime=Path(c['runtime_root'])/phase
                if (runtime/'supervisor-exit.json').is_file():require(read(runtime/'supervisor-exit.json')['status']=='PASS','role supervisor stopped; no retry: '+phase)
                if original_process_exited(d,guard.identity(d['pid'])):complete(c,phase);break
                time.sleep(5)
        publish(ROOT/'CHAIN_RESULT.json',{'status':'F0_E120_FIXED64_COMPLETE_REMOTE_EVIDENCE_PENDING_WINDOWS_RETURN','source_sha256':c['source_sha256'],'contract_sha256':c['contract_sha256'],'started_epoch':started,'finished_epoch':time.time(),'elapsed_seconds':time.time()-started,'phases':list(MODES),'no_gpu_retry':True,'instance_power_operation':False});return 0
    except BaseException as error:
        publish(ROOT/'CHAIN_STOP.json',{'status':'STOP_PRESERVE_NO_RETRY','phase':phase,'error':repr(error),'traceback':traceback.format_exc(),'started_epoch':started,'finished_epoch':time.time(),'elapsed_seconds':time.time()-started,'gpu_retry':False,'worker_may_still_be_running':'inspect actual supervisor/worker'});raise

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('action',choices=('role-prepare','role-launch','role-supervise','role-worker','role-start-chain','role-chain-worker'));p.add_argument('--request');p.add_argument('--contract');p.add_argument('--mode',choices=MODES);a=p.parse_args(argv)
    if a.action=='role-prepare':r=prepare(a.request)
    elif a.action=='role-launch':r=launch(a.contract,a.mode)
    elif a.action=='role-supervise':return supervise(a.contract,a.mode)
    elif a.action=='role-worker':return worker_entry(a.contract,a.mode)
    elif a.action=='role-chain-worker':return chain_worker(a.contract)
    else:r=start_chain(a.contract)
    print(json.dumps(r,allow_nan=False),flush=True);return 0
