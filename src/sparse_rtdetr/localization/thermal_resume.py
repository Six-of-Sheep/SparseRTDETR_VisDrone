"""One independent e031 continuation with restored lab resource constraints."""
import argparse,gc,gzip,json,os,subprocess,sys,time,traceback
from pathlib import Path
from . import channel,checkpoint,guard,lab_run,lab_consumer,numerics,consumer_geometry,worker
from .runio import require,read,write,publish,sha,digest,reference,event,execution_binding,original_process_exited
SOURCE=Path(__file__).resolve().parents[3];ROOT=SOURCE.parent.parent;BASE=ROOT.parent
PARENT_SHA='59204abf7016661d15ba013de34449f024e732588e9256206468e269b0936b5d'
PREVIOUS_SOURCE='43c43bb892833299a9f6fe62408bf40ccfa77e278d29ec5ee558151b53c002ce'
PREVIOUS_CONTRACT='928aa1460fe6869e4b63fd07f63f3196e7ff12f45e55e24fe0449bdc8f885d60'
MODES=('acceptance','formal','trace-cache','trace-analysis')
WORKLOAD={'parent_update':9424,'parent_epoch':31,'acceptance_updates':[9425,9426,9426],'acceptance_actual_updates':3,
          'formal_remaining_updates':27056,'formal_end_update':36480,'formal_endpoint':'e120_EMA','trace_images':64,'trace_batch_forwards':16}
LIMITS={'CPU_PL1_uW':125000000,'CPU_PL2_uW':150000000,'GPU_locked_min_MHz':1500,'GPU_locked_max_MHz':1500,'CPU_temperature_C':90,'GPU_temperature_C':80}
COMPARISON='original e031 full restore; e032 windows0/1 inputs GT augmentation exact; fixed9425/9426/replay9426 result fullstate RNG exact; formal independently from original e031; no old derived-value assumption'
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

def identity(c,mode):return {**execution_binding(c,mode),'progress_protocol':c['progress_protocol']}

def verify_source(c):
    require(c['source_root']==str(SOURCE) and sha(c['source_manifest'])==c['source_sha256'],'thermal source manifest changed')
    for r in read(c['source_manifest'])['files']:
        p=SOURCE/r['path'];require(p.is_file() and not p.is_symlink() and p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],'thermal frozen source changed: '+r['path'])
    require(c['numerical_implementation']==numerics.identity() and c['consumer_geometry_contract']==consumer_geometry.identity(),'arithmetic or consumer contract changed')
    require(c['progress_protocol']==channel.PROTOCOL_NAME,'progress protocol changed')

def verify_materials(c):
    require(len(c['materials'])==len({r['path'] for r in c['materials']}),'duplicate thermal material')
    required={c[k] for k in ('parent_binding','parent_sidecar','parent_contract','data_verification','selection_path','fixed_inputs','parent_cpu_audit','caps_receipt','caps_script')}
    required|={c['data'][r]['manifest'] for r in ('train_core','development')}|{c['pretrained']['path']}
    require(required<={r['path'] for r in c['materials']},'thermal required material missing')
    for r in c['materials']:
        p=Path(r['path']);require(p.is_absolute() and p.is_relative_to(BASE) and not p.is_symlink() and reference(p)==r,'thermal material changed: '+str(p))
    from .protocol_evidence import verify
    verify(c,SOURCE,BASE)
    b=read(c['parent_binding']);side=read(c['parent_sidecar']);old=read(c['parent_contract'])
    require(sha(c['parent_checkpoint'])==PARENT_SHA and side['sha256']==PARENT_SHA and side['size_bytes']==327151969 and side['path']==c['parent_checkpoint'],'e031 original bytes/sidecar differ')
    import hashlib
    require(side['binding_sha256']==hashlib.sha256(checkpoint.canonical(b)).hexdigest() and side['scope']=='formal_F0_seed0_e120' and side['completed_updates']==9424,'e031 parent binding/clock differs')
    require(b['source_sha256']==PREVIOUS_SOURCE and b['contract_sha256']==PREVIOUS_CONTRACT and old['contract_sha256']==PREVIOUS_CONTRACT and digest({k:v for k,v in old.items() if k!='contract_sha256'})==PREVIOUS_CONTRACT,'e031 original source/contract changed')
    require(old['parent_lineage']['parent_checkpoint_sha256']=='22345901f817b9c24c8b7167adfa0c2991d424437a2991339f90301efe356f62','e031 does not descend from original e026')
    require(c['parent_lineage']=={'attempt':'thermal-resume-002','parent_checkpoint_sha256':PARENT_SHA,'parent_binding_sha256':sha(c['parent_binding']),
            'parent_source_sha256':PREVIOUS_SOURCE,'parent_contract_sha256':PREVIOUS_CONTRACT,'parent_update':9424,'original_e026_sha256':old['parent_lineage']['parent_checkpoint_sha256']},'thermal lineage changed')
    require(read(c['parent_cpu_audit'])['status']=='E031_TRUSTED_CPU_SAVED_PAYLOAD_FINITE_BINDING_PASS_NOT_EXECUTED_RESTORE','saved-payload audit absent')
    proof=read(c['data_verification']);require(proof['status']=='PASS' and proof['named_pairs']==4933 and proof['train_core']==4869 and proof['development_fixed64']==64 and c['data_sha256']==sha(c['data_verification']),'existing data closure differs')
    fixed=read(c['fixed_inputs']);require(fixed['epoch']==32 and fixed['windows']==[0,1] and fixed['gpu_calls']==0,'new fixed input scope differs')
    cap=read(c['caps_receipt']);require(c['resource_limits']==LIMITS and cap['status']=='PRIOR_CPU_125W_150W_GPU_1500MHZ_REQUEST_APPLIED' and cap['gpu_uuid']==c['expected_gpu_uuid'],'cap authorization receipt differs')
    require(cap['script_sha256']==sha(c['caps_script']) and cap['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'cap script/boot differs')
    op=cap['operations'][-1];require(op['returncode']==0 and op['command']==['nvidia-smi','-i',c['expected_gpu_uuid'],'--lock-gpu-clocks=1500,1500'],'GPU lock not authenticated')

def load(path):
    c=read(path);require(c['contract_sha256']==digest({k:v for k,v in c.items() if k!='contract_sha256'}),'thermal contract digest differs')
    require(c['job_root']==str(ROOT) and ROOT.name=='thermal-resume-002' and c['runtime_root']=='/home/lyy/.cache/f0-e026-labdiag-20261009-001/runtime/thermal-resume-002' and c['workload']==WORKLOAD and c['comparison_policy']==COMPARISON,'thermal root/workload/comparison differs')
    require(c['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'thermal boot changed')
    verify_source(c);verify_materials(c);return c

def prepare(path):
    c=read(path);require(not (ROOT/'contract.json').exists() and c['job_root']==str(ROOT) and c['workload']==WORKLOAD and c['comparison_policy']==COMPARISON,'thermal freeze already used or workload differs')
    verify_source(c);verify_materials(c)
    import torch,torchvision,numpy,xml.etree.ElementTree as ET
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(),'thermal freeze must remain CPU only')
    actual={'python':sys.version.split()[0],'torch':str(torch.__version__),'torchvision':torchvision.__version__,'numpy':numpy.__version__,'cuda_build':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'prefix':sys.prefix}
    require(actual['python']=='3.10.16' and actual['torch']=='2.4.1' and actual['cuda_build']=='12.4' and actual['cudnn']==90100 and str(Path(sys.executable))==c['python'],'thermal runtime differs')
    require(c['new_cpu_checks'] and all(p['tests']>0 for p in c['new_cpu_checks']),'new thermal CPU gate absent')
    for p in c['new_cpu_checks']:
        require(reference(p['xml']['path'])==p['xml'] and reference(p['log']['path'])==p['log'],'thermal CPU evidence changed')
        cases=list(ET.parse(p['xml']['path']).getroot().iter('testcase'));require(len(cases)==p['tests'] and all(not any(t.find(k) is not None for k in ('error','failure','skipped')) for t in cases),'thermal CPU checks incomplete')
    s=snapshot(ROOT);uid=screen(s,entry=True,disk_bytes=40*1024**3+90*512*1024**2);require(uid==c['expected_gpu_uuid'],'thermal GPU differs')
    from .storage import identity as storage_identity
    c.update(schema='f0-lab-thermal-e031-continuation-v1',gpu_uuid=uid,boot_id=s['boot_id'],environment={k:actual[k] for k in ('torch','torchvision','cuda_build','cudnn')},
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
    require(ex['status']=='PASS' and ex['returncode']==0 and not ex['remaining_live'] and not ex['unknown_pids'] and not ex['cleanup_errors'] and original_process_exited(d,guard.identity(d['pid'])),'thermal phase not cleanly complete')
    require(d['mode']==mode and d['contract_sha256']==c['contract_sha256'],'thermal dispatch differs')
    channel.verify_archive(runtime,folder,{**identity(c,mode),'supervisor_pid':d['pid'],'supervisor_start':d['process_identity']['start']})
    require(read(folder/'execution-binding.json')==execution_binding(c,mode),'thermal execution belongs to another source')
    for name in ('RESULT.json','execution-binding.json','worker-exit.json','binding.json'):
        p=runtime/name if name=='worker-exit.json' else folder/name;require(ex['completion_receipts'][name]==reference(p),'thermal receipt changed')
    r=read(folder/'RESULT.json')
    if mode=='acceptance':
        require(r['status']=='PASS' and r['actual_updates']==3 and r['updates']==[9425,9426,9426] and r['same_next_window_exact'] and r['same_next_state_exact'] and r['original_restore']['checkpoint_sha256']==PARENT_SHA and not r['state_for_formal'],'thermal exact3 failed')
        under_load_proof(c,runtime)
    elif mode=='formal':require(r['status']=='F0_E120_EMA_LOCKED' and r['updates']==36480 and r['actual_updates']==27056 and r['original_restore']['checkpoint_sha256']==PARENT_SHA and not r['used_acceptance_checkpoint'],'thermal formal endpoint incomplete')
    else:require(r['status']=='PASS' and r['images']==64,'thermal fixed64 incomplete')
    return r

def prerequisite(c,mode):
    require(mode in MODES,'unknown thermal mode')
    for prior in MODES[:MODES.index(mode)]:complete(c,prior)

def command(c,path,mode):return [c['python'],'-B',str(SOURCE/'tools/localization/deploy.py'),'thermal-supervise','--contract',str(path),'--mode',mode]

def token(c,path,mode):
    r=read(ROOT/(mode+'-launch-consumed.json'));require(r['command']==command(c,path,mode) and r['contract_sha256']==c['contract_sha256'] and r['mode']==mode and r['no_retry'],'thermal launch token differs')

def launch(path,mode):
    c=load(path);prerequisite(c,mode);require(not (ROOT/mode).exists(),'thermal phase already used; no GPU retry')
    screen(snapshot(ROOT),entry=True,gpu=mode!='trace-analysis',disk_bytes=40*1024**3+(90*512*1024**2 if mode=='formal' else 0));cmd=command(c,path,mode)
    publish(ROOT/(mode+'-launch-consumed.json'),{'command':cmd,'mode':mode,'contract_sha256':c['contract_sha256'],'no_retry':True,'epoch':time.time()})
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MKL_THREADING_LAYER='GNU',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (ROOT/(mode+'-dispatcher.log')).open('x') as log:p=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    r={'pid':p.pid,'process_identity':guard.identity(p.pid),'contract_sha256':c['contract_sha256'],'mode':mode,'started_epoch':time.time()};publish(ROOT/(mode+'-dispatch.json'),r);return r

def supervise(path,mode):
    global ACTIVE_RUNTIME
    c=load(path);prerequisite(c,mode);token(c,path,mode);guard.snapshot=snapshot;guard.screen=screen;ACTIVE_RUNTIME=Path(c['runtime_root'])/mode
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if mode=='trace-analysis' else c['gpu_uuid'],OMP_NUM_THREADS='16' if mode=='trace-analysis' else '2',MKL_NUM_THREADS='16' if mode=='trace-analysis' else '2')
    cmd=[c['python'],'-B',str(SOURCE/'tools/localization/deploy.py'),'thermal-worker','--contract',str(path),'--mode',mode]
    return guard.supervise(cmd,ROOT/mode,env,gpu_uuid=c['gpu_uuid'],boot_id=c['boot_id'],gpu=mode!='trace-analysis',checkpoint_bytes=512*1024**2,formal=mode=='formal',
        runtime=ACTIVE_RUNTIME,run_identity={**identity(c,mode),'supervisor_pid':os.getpid(),'supervisor_start':guard.identity(os.getpid())['start']})

def restored_parent(c,runtime,holder,scope):
    engine,binding=worker.make_engine(c,runtime,scope);holder['engine']=engine
    from .cache_adapter import strict_restore
    r=strict_restore(engine,read(c['parent_binding']),c['parent_checkpoint'],PARENT_SHA,Path(c['runtime_root'])/holder['mode']/'parent-cache-import.json')
    require(r['full_state_exact'] and engine.completed_updates==engine.ema.updates==9424 and engine.phase==0,'e031 full restore failed');numerics.validate_binding(binding,c)
    receipt={'status':'ORIGINAL_E031_STRICT_FULL_RESTORE_PASS','result':r,'execution':identity(c,holder['mode']),'parent_binding':reference(c['parent_binding']), 'parent_sidecar':reference(c['parent_sidecar'])}
    publish(Path(c['runtime_root'])/holder['mode']/'original-restore.json',receipt);publish(ROOT/holder['mode']/'original-restore.json',receipt);return engine,binding,r

def input_check(c,images,targets,traces,metadata,window):
    import hashlib
    require(window in (0,1) and len(images)==len(targets)==len(traces)==16,'thermal fixed input expanded')
    expected=read(c['fixed_inputs'])['inputs'][str(window)];require(metadata==expected['metadata'] and digest(metadata)==expected['metadata_digest'] and [t['gt_ids'] for t in targets]==expected['gt_ids'],'thermal input/GT differs')
    for image,target,trace in zip(images,targets,traces):
        h=hashlib.sha256();h.update(image.contiguous().numpy().tobytes())
        for k in ('boxes','labels','raw_area','ignore_spatial_boxes','ignore_union','ignore_class_boxes','ignore_class_labels'):h.update(k.encode());h.update(target[k].contiguous().numpy().tobytes())
        h.update(json.dumps([target['gt_ids'],target['ignore_ids']],separators=(',',':')).encode());require(h.hexdigest()==trace['augmented_tensor_sha256'],'thermal augmented bytes differ')

def acceptance(c,folder,holder):
    import torch
    from .data import training_loader
    runtime=worker.prepare_cuda(c,folder);data=worker.dataset(c,'train_core');plan=worker.plan_epoch(data,32)
    engine,binding,parent=restored_parent(c,runtime,holder,'gpu_short_window');write(folder/'binding.json',binding);results=[]
    for win,(images,targets,traces) in enumerate(training_loader(data,epoch=32,window_limit=2)):
        meta=worker.check_inputs(images,targets,traces,plan,32,win);holder['input']=meta;input_check(c,images,targets,traces,meta,win);worker.live_admission(folder,c)
        r=engine.train_window(images,targets,capture=True);lab_consumer.clock_check(engine,r,9425+win);r['input']=meta;results.append(r);event(folder,'update_complete',**r,completed_epochs=31)
        if win==0:saved=worker.save_state(engine,binding,folder,'C9425.pth')
    require(len(results)==2 and engine.completed_updates==9426,'thermal acceptance work expanded')
    direct=worker.save_state(engine,binding,folder,'C9426-direct.pth');expected=checkpoint.read(direct['path'],direct['sha256'],binding);holder.pop('engine');del engine;gc.collect();torch.cuda.empty_cache()
    clone,new=worker.make_engine(c,runtime,'gpu_short_window');holder['engine']=clone;require(new==binding,'thermal replay binding differs')
    restored=numerics.restore_active(clone,binding,saved['path'],saved['sha256'],Path(c['runtime_root'])/'acceptance/replay-cache-import.json',c);count=0
    for images,targets,traces in training_loader(data,epoch=32,completed_windows=1,window_limit=1):
        meta=worker.check_inputs(images,targets,traces,plan,32,1);holder['input']=meta;input_check(c,images,targets,traces,meta,1);worker.live_admission(folder,c)
        r=clone.train_window(images,targets,capture=True);lab_consumer.clock_check(clone,r,9426);r['input']=meta;count+=1
    require(count==1 and {k:v for k,v in r.items() if k!='elapsed_seconds'}=={k:v for k,v in results[-1].items() if k!='elapsed_seconds'},'thermal same next result/output/gradient mismatch')
    require(checkpoint.exact_equal(expected,checkpoint.capture(clone,binding)),'thermal same next full state/RNG mismatch')
    replay=worker.save_state(clone,binding,folder,'C9426-replay.pth');publish(folder/'RESULT.json',{'status':'PASS','actual_updates':3,'updates':[9425,9426,9426],'original_restore':parent,'replay_restore':restored,
        'same_next_window_exact':True,'same_next_state_exact':True,'checkpoints':[saved,direct,replay],'state_for_formal':False,'complete_update_seconds':[r['elapsed_seconds'] for r in results]+[r['elapsed_seconds']], 'resource_limits':c['resource_limits']})

def formal(c,folder,holder):
    from .data import training_loader
    runtime=worker.prepare_cuda(c,folder);data=worker.dataset(c,'train_core');engine,binding,parent=restored_parent(c,runtime,holder,'formal_F0_seed0_e120');write(folder/'binding.json',binding)
    event(folder,'resume_parent_exact',**parent,scope='formal_F0_seed0_e120',used_acceptance_checkpoint=False);started=time.monotonic();times=[];count=0
    with gzip.open(folder/'input-acknowledgements.jsonl.gz','xt',encoding='utf-8') as ack:
        for epoch in range(32,121):
            verify_source(c);require(engine.completed_updates==(epoch-1)*304,'thermal formal epoch clock differs')
            plan=worker.plan_epoch(data,epoch);write(folder/f'input-plan-epoch-{epoch:03d}.json',plan);event(folder,'epoch_begin',epoch=epoch,completed_epochs=epoch-1,dropped_ids=plan['ids'][4864:]);t=time.monotonic();windows=0
            for win,(images,targets,traces) in enumerate(training_loader(data,epoch=epoch)):
                meta=worker.check_inputs(images,targets,traces,plan,epoch,win);holder['input']=meta;worker.live_admission(folder,c)
                if epoch==32 and win in (0,1):input_check(c,images,targets,traces,meta,win)
                r=engine.train_window(images,targets);lab_consumer.clock_check(engine,r,(epoch-1)*304+win+1)
                ack.write(json.dumps(meta,allow_nan=False)+'\n');ack.flush();event(folder,'update_complete',**r,completed_epochs=engine.completed_updates//304,input_sha256=digest(meta));windows+=1;count+=1
            require(windows==304 and engine.completed_updates==epoch*304,'thermal formal epoch incomplete')
            last=worker.save_state(engine,binding,folder,f'checkpoint-epoch-{epoch:03d}.pth');duration=time.monotonic()-t;times.append(duration)
            publish(folder/f'epoch-{epoch:03d}-complete.json',{'epoch':epoch,'updates':engine.completed_updates,'checkpoint':last,'epoch_wall_seconds':duration,'strong_augmentation':epoch<118})
    require(count==27056 and engine.completed_updates==engine.ema.updates==36480,'thermal formal endpoint count differs')
    require(checkpoint.exact_equal(checkpoint.read(last['path'],last['sha256'],binding),checkpoint.capture(engine,binding)),'thermal e120 disk/live differs')
    endpoint={'status':'F0_E120_EMA_LOCKED','arm':'F0','seed':0,'epoch':120,'updates':36480,'checkpoint':last,'binding':reference(folder/'binding.json'),'source_sha256':c['source_sha256'],'contract_sha256':c['contract_sha256'],
              'original_restore':parent,'used_acceptance_checkpoint':False,'parent_update':9424,'actual_updates':count,'training_wall_seconds':time.monotonic()-started,'epoch_wall_seconds':times,
              'laboratory_actual_cost':None,'evaluation_performed':False,'additional_arms':[],'holdout_accessed':False,'numerical_implementation':numerics.identity()}
    publish(folder/'endpoint-lock.json',endpoint);publish(folder/'RESULT.json',endpoint)

def worker_entry(path,mode):
    global ACTIVE_RUNTIME
    c=load(path);prerequisite(c,mode);token(c,path,mode);guard.snapshot=snapshot;guard.screen=screen
    folder=ROOT/mode;runtime=Path(c['runtime_root'])/mode;holder={'mode':mode};started=time.time();d=read(ROOT/(mode+'-dispatch.json'));b=read(runtime/'identity.json')
    current=guard.identity(os.getppid());require(current and current['start']==d['process_identity']['start'] and os.getppid()==d['pid'] and read(runtime/'supervisor-start.json')['pid']==os.getppid(),'thermal supervisor identity differs')
    require(all(b.get(k)==v for k,v in identity(c,mode).items()),'thermal runtime identity differs');ACTIVE_RUNTIME=runtime;publish(folder/'execution-binding.json',execution_binding(c,mode))
    from . import control
    control.verify_source=verify_source
    try:
        if mode=='acceptance':acceptance(c,folder,holder)
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
    c=load(path);complete(c,'acceptance');cmd=[c['python'],'-B',str(SOURCE/'tools/localization/deploy.py'),'thermal-chain-worker','--contract',str(path)]
    publish(ROOT/'continuation-chain-consumed.json',{'contract_sha256':c['contract_sha256'],'source_sha256':c['source_sha256'],'command':cmd,'modes':list(MODES[1:]),'no_retry':True,'epoch':time.time()})
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MKL_THREADING_LAYER='GNU',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (ROOT/'continuation-chain.log').open('x') as log:p=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    d={'pid':p.pid,'process_identity':guard.identity(p.pid),'contract_sha256':c['contract_sha256'],'started_epoch':time.time()};publish(ROOT/'continuation-chain-dispatch.json',d);return d

def chain_worker(path):
    c=load(path);r=read(ROOT/'continuation-chain-consumed.json');require(r['contract_sha256']==c['contract_sha256'] and r['source_sha256']==c['source_sha256'] and r['modes']==list(MODES[1:]) and r['no_retry'] and r['command']==[c['python'],'-B',str(SOURCE/'tools/localization/deploy.py'),'thermal-chain-worker','--contract',str(path)],'thermal chain token differs')
    started=time.time();phase=None
    try:
        complete(c,'acceptance')
        for phase in MODES[1:]:
            d=launch(path,phase)
            while True:
                runtime=Path(c['runtime_root'])/phase
                if (runtime/'supervisor-exit.json').is_file():require(read(runtime/'supervisor-exit.json')['status']=='PASS','thermal supervisor stopped; no retry: '+phase)
                if original_process_exited(d,guard.identity(d['pid'])):complete(c,phase);break
                time.sleep(5)
        publish(ROOT/'CHAIN_RESULT.json',{'status':'F0_E120_FIXED64_COMPLETE_REMOTE_EVIDENCE_PENDING_WINDOWS_RETURN','source_sha256':c['source_sha256'],'contract_sha256':c['contract_sha256'],'started_epoch':started,'finished_epoch':time.time(),'elapsed_seconds':time.time()-started,'phases':list(MODES),'no_gpu_retry':True,'instance_power_operation':False});return 0
    except BaseException as error:
        publish(ROOT/'CHAIN_STOP.json',{'status':'STOP_PRESERVE_NO_RETRY','phase':phase,'error':repr(error),'traceback':traceback.format_exc(),'started_epoch':started,'finished_epoch':time.time(),'elapsed_seconds':time.time()-started,'gpu_retry':False,'worker_may_still_be_running':'inspect actual supervisor/worker'});raise

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('action',choices=('thermal-prepare','thermal-launch','thermal-supervise','thermal-worker','thermal-start-chain','thermal-chain-worker'));p.add_argument('--request');p.add_argument('--contract');p.add_argument('--mode',choices=MODES);a=p.parse_args(argv)
    if a.action=='thermal-prepare':r=prepare(a.request)
    elif a.action=='thermal-launch':r=launch(a.contract,a.mode)
    elif a.action=='thermal-supervise':return supervise(a.contract,a.mode)
    elif a.action=='thermal-worker':return worker_entry(a.contract,a.mode)
    elif a.action=='thermal-chain-worker':return chain_worker(a.contract)
    else:r=start_chain(a.contract)
    print(json.dumps(r,allow_nan=False),flush=True);return 0
