"""Consumer contract diagnosis, exact recovery check, independent e026 formal and fixed64."""
import argparse,gc,gzip,json,os,subprocess,sys,time,traceback
from pathlib import Path
from . import channel,checkpoint,guard,lab_run,numerics,worker,consumer_geometry
from .runio import require,read,write,publish,sha,digest,event,reference,execution_binding,original_process_exited
SOURCE=Path(__file__).resolve().parents[3]
ROOT=SOURCE.parent.parent
BASE=ROOT.parent
PARENT_SHA='22345901f817b9c24c8b7167adfa0c2991d424437a2991339f90301efe356f62'
DIAGNOSTIC_SHA='f40a0f8bee15d9c188fd856f055c5f599bb7e213b2af66bca5d6e18b92b116bc'
MODES=('diagnostic','acceptance','formal','trace-cache','trace-analysis')
WORKLOAD={'diagnostic_window':73,'diagnostic_parent_update':7977,'diagnostic_actual_updates':0,
          'acceptance_updates':[7978,7979,7979],'acceptance_actual_updates':3,
          'formal_parent_update':7904,'formal_remaining_updates':28576,'formal_end_update':36480,
          'formal_endpoint':'e120_EMA','trace_images':64,'trace_batch_forwards':16}

def diagnostic_authorization(c):
    return {'parent_checkpoint_sha256':DIAGNOSTIC_SHA,'parent_binding_sha256':c['diagnostic_lineage']['parent_binding_sha256'],
            'parent_contract_sha256':c['diagnostic_lineage']['parent_contract_sha256'],'parent_update':7977,'micro_phase_sequence':[[0,1],[1,2]]}


def runtime_identity(c,mode):
    result={**execution_binding(c,mode),'progress_protocol':c['progress_protocol']}
    if mode=='diagnostic':result['diagnostic_authorization']=c['diagnostic_authorization']
    return result

def screen(sample,**kwargs):
    require(sample['snapshot']['resources']['cpu_count']>=4 and sample['snapshot']['resources']['ram_available']>=8*1024**3,'lab CPU/RAM reserve')
    require(sample['snapshot']['shm_free']>=4*1024**3 and sample['storage']['ssd_free_bytes']>=4*1024**3,'lab SSD/shm reserve')
    require(sample['snapshot']['cpu_temperature_max']<90,'lab CPU temperature threshold')
    if kwargs.get('baseline_events') is not None:require(sample['memory_events']==kwargs['baseline_events'],'lab OOM event changed')
    require(sample['storage']['admitted_available_bytes']>=max(10*1024**3,kwargs.get('disk_bytes',0)), 'lab complete checkpoint storage reserve')
    if not kwargs.get('gpu',True):return None
    lab_run.screen(sample,**kwargs)
    return sample['snapshot']['gpu']['devices'][0]['uuid']['value']

def verify_source(c):
    require(c['source_root']==str(SOURCE) and sha(c['source_manifest'])==c['source_sha256'],'numerical executing source differs')
    for r in read(c['source_manifest'])['files']:
        p=SOURCE/r['path'];require(p.is_file() and not p.is_symlink() and p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],'frozen numerical source changed: '+r['path'])
    require(c['numerical_implementation']==numerics.identity(),'frozen numerical implementation changed')
    require(c['consumer_geometry_contract']==consumer_geometry.identity(),'frozen consumer rules changed')
    require(c['progress_protocol']==channel.PROTOCOL_NAME and c['diagnostic_authorization']==diagnostic_authorization(c),'frozen diagnostic progress protocol/parent differs')

def verify_materials(c):
    require(len(c['materials'])==len({r['path'] for r in c['materials']}),'duplicate material binding')
    required={c[k] for k in ('parent_binding','parent_contract','parent_sidecar','reference','data_verification','selection_path','diagnostic_binding','diagnostic_sidecar','diagnostic_contract','fixed_inputs','diagnostic_cpu_proof')}
    required|={c['data'][role]['manifest'] for role in ('train_core','development')}|{c['pretrained']['path']}
    require(required<={r['path'] for r in c['materials']},'required material binding absent')
    for r in c['materials']:
        p=Path(r['path']);require(p.is_absolute() and p.is_relative_to(BASE) and not p.is_symlink(),'material outside isolated lab root')
        require(reference(p)==r,'material SHA changed: '+str(p))
    from .protocol_evidence import verify as verify_protocol_evidence
    verify_protocol_evidence(c,SOURCE,BASE)
    require(sha(c['parent_checkpoint'])==PARENT_SHA,'wrong original formal parent')
    origin=read(c['parent_binding']);side=read(c['parent_sidecar']);old=read(c['parent_contract'])
    require(old['contract_sha256']=='3f8e2e3a5562598fdf97e044f4c833f2ca4dc68aed7e8600158471d427863158' and digest({k:v for k,v in old.items() if k!='contract_sha256'})==old['contract_sha256'],'original contract identity changed')
    require(side['sha256']==PARENT_SHA and side['completed_updates']==7904 and side['binding_sha256']==digest(origin),'original sidecar/binding changed')
    proof=read(c['data_verification']);require(proof['status']=='PASS' and proof['named_pairs']==4933 and proof['train_core']==4869 and proof['development_fixed64']==64,'full authorized data closure absent')
    require(c['data_sha256']==sha(c['data_verification']) and len(proof['files'])==4933,'data closure digest/membership differs')
    require(c['parent_lineage']=={'attempt':'consumer-geometry-002','parent_checkpoint_sha256':PARENT_SHA,
            'parent_binding_sha256':sha(c['parent_binding']),'parent_source_sha256':origin['source_sha256'],
            'parent_contract_sha256':old['contract_sha256'],'parent_update':7904},'new lineage is not original e026 binding')

    diag=read(c['diagnostic_binding']);side=read(c['diagnostic_sidecar']);dc=read(c['diagnostic_contract'])
    import hashlib
    require(side['binding_sha256']==hashlib.sha256(checkpoint.canonical(diag)).hexdigest() and side['size_bytes']==Path(c['diagnostic_checkpoint']).stat().st_size and side['path']==c['diagnostic_checkpoint'],'diagnostic sidecar binding/size/path mismatch')
    require(digest({k:v for k,v in dc.items() if k!='contract_sha256'})==dc['contract_sha256'],'original diagnostic contract digest changed')
    require(sha(c['diagnostic_checkpoint'])==DIAGNOSTIC_SHA and side['sha256']==DIAGNOSTIC_SHA and side['completed_updates']==7977 and side['scope']=='gpu_short_window','C7977 diagnostic parent changed')
    require(diag['source_sha256']=='f89fa5ff3c6058a3160579350c574bf57199e412fadc51c48e27c2ce35c7d9db' and diag['contract_sha256']==dc['contract_sha256']=='a56906f085935722b7a83912d6c866fea890d138601a29e4637574e15055320c','diagnostic original source/contract differs')
    require(c['diagnostic_lineage']=={'attempt':'consumer-geometry-002','parent_checkpoint_sha256':DIAGNOSTIC_SHA,'parent_binding_sha256':sha(c['diagnostic_binding']),
            'parent_source_sha256':diag['source_sha256'],'parent_contract_sha256':dc['contract_sha256'],'parent_update':7977,'formal_forbidden':True},'diagnostic lineage differs')
    require(read(c['diagnostic_cpu_proof'])['status']=='C7977_ORIGINAL_CPU_FULL_PAYLOAD_BINDING_LINEAGE_PASS','C7977 CPU proof absent')
    require(read(c['fixed_inputs'])['windows']==[73,74] and read(c['fixed_inputs'])['gpu_calls']==0,'pre-GPU inputs changed')

def load(path):
    c=read(path);require(c['contract_sha256']==digest({k:v for k,v in c.items() if k!='contract_sha256'}),'numerical contract digest differs')
    require(c['job_root']==str(ROOT) and c['workload']==WORKLOAD and c['runtime_root']==str(Path('/home/lyy/.cache/f0-e026-labdiag-20261009-001/runtime/consumer-geometry-002')),'numerical workload/root expanded')
    require(c['comparison_policy']=='fixed pre-GPU windows73/74 input/GT/raw exact; consumer contract and schedule exact; same next results/fullstate/RNG exact; no old derived expectation','comparison policy changed')
    verify_source(c);verify_materials(c)
    require(guard.probe.read_text('/proc/sys/kernel/random/boot_id').strip()==c['boot_id'],'lab boot changed')
    return c

def prepare(path):
    c=read(path);require(not (ROOT/'contract.json').exists(),'numerical contract already frozen')
    require(c['job_root']==str(ROOT) and c['workload']==WORKLOAD,'invalid numerical workload')
    verify_source(c);verify_materials(c)
    import torch,torchvision,numpy
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(),'freeze must remain CPU only')
    actual={'python':sys.version.split()[0],'torch':str(torch.__version__),'torchvision':torchvision.__version__,
            'numpy':numpy.__version__,'cuda_build':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'prefix':sys.prefix}
    require(actual['python']=='3.10.16' and actual['torch']=='2.4.1' and actual['torchvision']=='0.19.1' and actual['cuda_build']=='12.4' and actual['cudnn']==90100,'isolated runtime changed')
    require(str(Path(sys.executable))==c['python'],'isolated executable differs')
    import xml.etree.ElementTree as ET
    require({proof['kind'] for proof in c['new_cpu_checks']}=={'consumer','lifecycle'} and all(proof['tests']>0 for proof in c['new_cpu_checks']),'new CPU gate missing')
    for proof in c['new_cpu_checks']:
        require(reference(proof['xml']['path'])==proof['xml'] and reference(proof['log']['path'])==proof['log'],'CPU evidence changed')
        cases=list(ET.parse(proof['xml']['path']).getroot().iter('testcase'))
        require(len(cases)==proof['tests'] and all(not any(t.find(k) is not None for k in ('error','failure','skipped')) for t in cases),'CPU check incomplete or failed')
    s=lab_run.snapshot(ROOT);uid=screen(s,entry=True,disk_bytes=40*1024**3+96*512*1024**2)
    from .storage import identity
    c.update(schema='f0-lab-numerical-repair-v1',gpu_uuid=uid,boot_id=s['boot_id'],environment={k:actual[k] for k in ('torch','torchvision','cuda_build','cudnn')},
             environment_observation=actual,storage={'identity':identity(s['storage'])},resource_observation=s,freeze_epoch=time.time())
    c['contract_sha256']=digest(c);publish(ROOT/'contract.json',c);return c

def complete(c,mode):
    runtime=Path(c['runtime_root'])/mode;folder=ROOT/mode;ex=read(runtime/'supervisor-exit.json');dispatch=read(ROOT/(mode+'-dispatch.json'))
    require(ex['status']=='PASS' and ex['returncode']==0 and not ex['remaining_live'] and not ex['unknown_pids'] and not ex['cleanup_errors'],'prior numerical phase did not exit cleanly')
    require(original_process_exited(dispatch,guard.identity(dispatch['pid'])),'prior numerical phase still live')
    require(dispatch['mode']==mode and dispatch['contract_sha256']==c['contract_sha256'],'prior phase dispatch differs')
    channel.verify_archive(runtime,folder,{**runtime_identity(c,mode),'supervisor_pid':dispatch['pid'],'supervisor_start':dispatch['process_identity']['start']})
    require(read(folder/'execution-binding.json')==execution_binding(c,mode),'prior numerical phase belongs to another source')
    for name in ('RESULT.json','execution-binding.json','worker-exit.json','binding.json'):
        p=runtime/name if name=='worker-exit.json' else folder/name
        require(ex['completion_receipts'][name]==reference(p),'prior complete receipt changed')
    result=read(folder/'RESULT.json')
    if mode=='diagnostic':
        receipt=read(runtime/'original-restore.json')
        require(sha(runtime/'original-restore.json')==sha(folder/'original-restore.json') and receipt['result']==result['original_restore'],'independent restore receipt changed')
        require(receipt['status']=='ORIGINAL_C7977_STRICT_FULL_RESTORE_PASS' and receipt['completed_updates']==receipt['ema_updates']==7977 and receipt['phase']==0 and receipt['execution']==runtime_identity(c,mode),'independent strict restore receipt incomplete')
        reader=channel.Reader(runtime/'progress.jsonl',read(runtime/'identity.json'),'progress');reader.poll()
        require(reader.clock.get('_diagnostic_micro')==1,'both diagnostic micro phases not authenticated')
        require(result['status']=='PASS' and result['forward_microbatches']==result['backward_microbatches']==2 and result['actual_updates']==0 and result['completed_updates']==7977,'diagnostic scope incomplete')
        require(result['original_restore']['full_state_exact'] and result['original_restore']['checkpoint_sha256']==DIAGNOSTIC_SHA and result['original_restore']['completed_updates']==7977 and result['phase_after']==2,'diagnostic original restoration/phase differs')
        require(result['optimizer_exact_unchanged'] and result['ema_exact_unchanged'] and result['learned_parameters_exact_unchanged'] and result['raw_gradients_finite'] and result['all_required_consumers_observed'],'diagnostic state/consumer check failed')
    elif mode=='acceptance':
        require(result['status']=='PASS' and result['actual_updates']==3 and result['updates']==[7978,7979,7979] and result['same_next_window_exact'] and result['same_next_state_exact'],'consumer exact recovery gate failed')
        require(result['original_restore']['checkpoint_sha256']==DIAGNOSTIC_SHA,'acceptance used wrong parent')
    elif mode=='formal':
        require(result['status']=='F0_E120_EMA_LOCKED' and result['updates']==36480 and result['actual_updates']==28576 and not result['used_acceptance_checkpoint'],'formal endpoint absent')
        require(result['original_restore']['checkpoint_sha256']==PARENT_SHA and result['original_restore']['completed_updates']==7904 and result['numerical_implementation']==numerics.identity(),'formal original parent/numerics differs')
    else:require(result['status']=='PASS','fixed64 phase incomplete')
    return result

def prerequisite(c,mode):
    require(mode in MODES,'unknown numerical phase')
    for prior in MODES[:MODES.index(mode)]:complete(c,prior)

def token(c,mode,path):
    record=read(ROOT/(mode+'-launch-consumed.json'))
    command=[c['python'],'-B',str(SOURCE/'tools/localization/deploy.py'),'consumer-supervise','--contract',str(path),'--mode',mode]
    require(record['mode']==mode and record['contract_sha256']==c['contract_sha256'] and record['command']==command and record['no_retry'],'numerical launch token differs')

def launch(path,mode):
    c=load(path);prerequisite(c,mode);require(not (ROOT/mode).exists(),'phase already exists; no GPU retry')
    screen(lab_run.snapshot(ROOT),entry=True,gpu=mode!='trace-analysis',disk_bytes=40*1024**3+(96*512*1024**2 if mode=='formal' else 0))
    command=[c['python'],'-B',str(SOURCE/'tools/localization/deploy.py'),'consumer-supervise','--contract',str(path),'--mode',mode]
    publish(ROOT/(mode+'-launch-consumed.json'),{'mode':mode,'contract_sha256':c['contract_sha256'],'command':command,'epoch':time.time(),'no_retry':True})
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MKL_THREADING_LAYER='GNU',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (ROOT/(mode+'-dispatcher.log')).open('x') as log:p=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    value={'pid':p.pid,'process_identity':guard.identity(p.pid),'contract_sha256':c['contract_sha256'],'mode':mode,'started_epoch':time.time()};publish(ROOT/(mode+'-dispatch.json'),value);return value

def supervise(path,mode):
    c=load(path);prerequisite(c,mode);token(c,mode,path);guard.snapshot=lab_run.snapshot;guard.screen=screen
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if mode=='trace-analysis' else c['gpu_uuid'],OMP_NUM_THREADS='16' if mode=='trace-analysis' else '2',MKL_NUM_THREADS='16' if mode=='trace-analysis' else '2')
    command=[c['python'],'-B',str(SOURCE/'tools/localization/deploy.py'),'consumer-worker','--contract',str(path),'--mode',mode]
    return guard.supervise(command,ROOT/mode,env,gpu_uuid=c['gpu_uuid'],boot_id=c['boot_id'],gpu=mode!='trace-analysis',checkpoint_bytes=512*1024**2,
        formal=mode=='formal',runtime=Path(c['runtime_root'])/mode,run_identity={**runtime_identity(c,mode),'supervisor_pid':os.getpid(),'supervisor_start':guard.identity(os.getpid())['start']})

def restored_parent(c,runtime,holder,scope):
    engine,binding=worker.make_engine(c,runtime,scope);holder['engine']=engine
    from .cache_adapter import strict_restore
    result=strict_restore(engine,read(c['parent_binding']),c['parent_checkpoint'],PARENT_SHA,Path(c['runtime_root'])/holder['mode']/'parent-cache-import.json')
    require(result['full_state_exact'] and engine.completed_updates==engine.ema.updates==7904,'original e026 full restore failed')
    numerics.validate_binding(binding,c);return engine,binding,result

def input_check(c,images,targets,traces,metadata,window):
    import hashlib
    require(window in (73,74),'diagnostic/acceptance input window expanded')
    require(len(images)==len(targets)==len(traces)==16,'not complete logical16 input')
    expected=read(c['fixed_inputs'])['inputs'][str(window)]
    require(metadata==expected['metadata'] and digest(metadata)==expected['metadata_digest'] and [t['gt_ids'] for t in targets]==expected['gt_ids'],'fixed pre-GPU input/GT identity changed')
    for image,target,trace in zip(images,targets,traces):
        h=hashlib.sha256();h.update(image.contiguous().numpy().tobytes())
        for key in ('boxes','labels','raw_area','ignore_spatial_boxes','ignore_union','ignore_class_boxes','ignore_class_labels'):
            h.update(key.encode());h.update(target[key].contiguous().numpy().tobytes())
        h.update(json.dumps([target['gt_ids'],target['ignore_ids']],separators=(',',':')).encode())
        require(h.hexdigest()==trace['augmented_tensor_sha256'],'received input bytes changed')

def clock_check(engine,result,update):
    from .engine import learning_rate
    require(engine.completed_updates==engine.ema.updates==update and engine.phase==0 and not engine.failed,'complete clock changed')
    expected={'update':update,'epoch':(update-1)//304+1,'window':(update-1)%304,'microsteps':update*2,
              'lr':learning_rate(update),'coefficients':[.5,.5],'optimizer_steps':1,'ema_updates':1,'finite':True}
    require(all(result[k]==v for k,v in expected.items()),'input/update/schedule count invariant failed')

def restored_diagnostic(c,runtime,holder):
    engine,binding=worker.make_engine(c,runtime,'gpu_short_window');holder['engine']=engine
    from .cache_adapter import strict_restore
    restored=strict_restore(engine,read(c['diagnostic_binding']),c['diagnostic_checkpoint'],DIAGNOSTIC_SHA,Path(c['runtime_root'])/holder['mode']/'parent-cache-import.json')
    require(restored['full_state_exact'] and engine.completed_updates==engine.ema.updates==7977,'C7977 original full restore failed')
    numerics.validate_binding(binding,c)
    receipt={'status':'ORIGINAL_C7977_STRICT_FULL_RESTORE_PASS','result':restored,'parent_binding':reference(c['diagnostic_binding']),'parent_sidecar':reference(c['diagnostic_sidecar']),
             'execution':runtime_identity(c,holder['mode']),'completed_updates':engine.completed_updates,'ema_updates':engine.ema.updates,'phase':engine.phase,'state_for_formal':False}
    telemetry=Path(c['runtime_root'])/holder['mode'];publish(telemetry/'original-restore.json',receipt)
    publish(ROOT/holder['mode']/'original-restore.json',receipt)
    return engine,binding,restored


def consumers_complete(audit):
    required=('position/query_pos','attention/offset','attention/locations','attention/bilinear_output','matcher/cost','matcher/giou/output',
              'loss/iou/output','loss/giou/output','classification/per_class','fgl/logp','ddf/kl_terms','decode/box32','ignore/predicted/area','support/atoms')
    require(all(audit.counts.get(k,0)>0 for k in required),'actual consumer observation incomplete')
    require(audit.counts['attention/locations']==6,'not both microbatches all three actual decoder layers')
    require(audit.gradient_counts.get('attention/core_output',0)==6,'actual full attention backward missing')
    return True


def diagnostic(c,folder,holder):
    import copy,torch
    from .data import training_loader
    from ..baseline.training_v2b_engine import _batchnorm_backward_layout,_move
    from .engine import assert_healthy,tensor_digest,output_digest,learning_rate
    runtime=worker.prepare_cuda(c,folder);data=worker.dataset(c,'train_core');plan=worker.plan_epoch(data,27)
    engine,binding,parent=restored_diagnostic(c,runtime,holder);write(folder/'binding.json',binding)
    before_optimizer=copy.deepcopy(engine.optimizer.state_dict());before_ema=copy.deepcopy(engine.ema.state_dict());before_parameters={n:p.detach().clone() for n,p in engine.model.named_parameters()}
    rng_before=checkpoint.capture_rng(engine.device);audit=consumer_geometry.Audit();holder['audit']=audit;outputs=[];loss_rows=[];forward=backward=windows=0
    engine.model.train();engine.criterion.train();engine.optimizer.zero_grad(set_to_none=True)
    original_step=engine.optimizer.step;original_ema=engine.ema.update
    def forbidden(*args,**kwargs):raise RuntimeError('diagnostic cannot submit optimizer/EMA update')
    engine.optimizer.step=forbidden;engine.ema.update=forbidden
    try:
        for images,targets,traces in training_loader(data,epoch=27,completed_windows=73,window_limit=1):
            metadata=worker.check_inputs(images,targets,traces,plan,27,73);holder['input']=metadata;input_check(c,images,targets,traces,metadata,73);worker.live_admission(folder,c);windows+=1
            with _batchnorm_backward_layout(engine.model,engine.device),consumer_geometry.auditing(audit):
                for micro in range(2):
                    audit.microbatch=micro;subset=_move(targets[micro*8:(micro+1)*8],engine.device);batch=images[micro*8:(micro+1)*8].to(engine.device)
                    with torch.autocast(engine.device.type,dtype=torch.bfloat16):out=engine.model(batch,targets=subset)
                    forward+=1;outputs.append(output_digest(out))
                    with torch.autocast(engine.device.type,enabled=False):losses=engine.criterion(out,subset);objective=sum(losses.values())*.5
                    consumer_geometry.tensor('diagnostic/objective',objective);objective.backward();engine.phase+=1;backward+=1;assert_healthy(engine.model,gradients=True)
                    loss_rows.append({k:float(v.detach()) for k,v in losses.items()});event(folder,'diagnostic_micro_backward',diagnostic_parent_sha256=DIAGNOSTIC_SHA,microbatch=micro,completed_updates=7977,phase=engine.phase)
                    del out,losses,objective,batch,subset
            require(windows==1 and forward==backward==2 and engine.phase==2 and engine.completed_updates==engine.ema.updates==7977,'diagnostic scope/cursor changed')
        require(checkpoint.exact_equal(before_optimizer,engine.optimizer.state_dict()) and checkpoint.exact_equal(before_ema,engine.ema.state_dict()) and checkpoint.exact_equal(before_parameters,{n:p.detach() for n,p in engine.model.named_parameters()}),'diagnostic submitted state update')
        consumers_complete(audit)
        gradients={n:None if p.grad is None else p.grad.detach().cpu().clone() for n,p in engine.model.named_parameters()}
        tensor_count=sum(v is not None for v in gradients.values());gradient_sha=tensor_digest(gradients.items())
        with (folder/'raw-gradients.pth').open('xb') as f:torch.save({'schema':'consumer-diagnostic-raw-unclipped-gradients-v1','parent_sha256':DIAGNOSTIC_SHA,'completed_updates':7977,'phase':2,'gradients':gradients},f);f.flush();os.fsync(f.fileno())
        grad_ref=reference(folder/'raw-gradients.pth')
        publish(folder/'raw-gradient-inventory.json',{n:{'parameter_shape':list(p.shape),'parameter_dtype':str(p.dtype),'gradient_present':gradients[n] is not None,
                 'gradient_shape':list(gradients[n].shape) if gradients[n] is not None else None,'gradient_dtype':str(gradients[n].dtype) if gradients[n] is not None else None,
                 'gradient_sha256':tensor_digest([(n,gradients[n])])} for n,p in engine.model.named_parameters()})
        publish(folder/'consumer-observations.json',audit.summary())
        publish(folder/'RESULT.json',{'status':'PASS','forward_microbatches':2,'backward_microbatches':2,'actual_updates':0,'completed_updates':7977,'phase_after':2,
                'original_restore':parent,'optimizer_exact_unchanged':True,'ema_exact_unchanged':True,'learned_parameters_exact_unchanged':True,
                'raw_gradients_finite':True,'raw_gradient_sha256':gradient_sha,'raw_gradient_tensor_count':tensor_count,'raw_gradients':grad_ref,
                'all_required_consumers_observed':True,'losses':loss_rows,'output_sha256':outputs,'planned_lr7978':learning_rate(7978),'optimizer_lr_write_performed':False,
                'rng_before_after_equal':checkpoint.exact_equal(rng_before,checkpoint.capture_rng(engine.device)),'state_for_formal':False,'geometry_contract':consumer_geometry.identity()})
    finally:
        engine.optimizer.step=original_step;engine.ema.update=original_ema
        engine.failed=True
        # Keep phase2/raw gradients. Never clip/reset phase/save resumable diagnostic state.


def acceptance(c,folder,holder):
    import torch
    from .data import training_loader
    runtime=worker.prepare_cuda(c,folder);data=worker.dataset(c,'train_core');plan=worker.plan_epoch(data,27)
    engine,binding,parent=restored_diagnostic(c,runtime,holder);write(folder/'binding.json',binding);results=[]
    for win,(images,targets,traces) in enumerate(training_loader(data,epoch=27,completed_windows=73,window_limit=2),73):
        metadata=worker.check_inputs(images,targets,traces,plan,27,win);holder['input']=metadata;input_check(c,images,targets,traces,metadata,win);worker.live_admission(folder,c)
        r=engine.train_window(images,targets,capture=True);clock_check(engine,r,7905+win);r['input']=metadata;results.append(r);event(folder,'update_complete',**r,completed_epochs=26)
        if win==73:saved=worker.save_state(engine,binding,folder,'C7978.pth')
    require(len(results)==2 and engine.completed_updates==7979,'acceptance workload changed')
    direct=worker.save_state(engine,binding,folder,'C7979-direct.pth');expected_state=checkpoint.read(direct['path'],direct['sha256'],binding);holder.pop('engine');del engine;gc.collect();torch.cuda.empty_cache()
    clone,new_binding=worker.make_engine(c,runtime,'gpu_short_window');holder['engine']=clone;require(new_binding==binding,'new same-next binding differs')
    restored=numerics.restore_active(clone,binding,saved['path'],saved['sha256'],Path(c['runtime_root'])/'acceptance/replay-cache-import.json',c)
    count=0
    for images,targets,traces in training_loader(data,epoch=27,completed_windows=74,window_limit=1):
        metadata=worker.check_inputs(images,targets,traces,plan,27,74);holder['input']=metadata;input_check(c,images,targets,traces,metadata,74);worker.live_admission(folder,c)
        r=clone.train_window(images,targets,capture=True);clock_check(clone,r,7979);r['input']=metadata;count+=1
    require(count==1 and {k:v for k,v in r.items() if k!='elapsed_seconds'}=={k:v for k,v in results[-1].items() if k!='elapsed_seconds'},'same next result/output/loss/gradient/input mismatch')
    require(checkpoint.exact_equal(expected_state,checkpoint.capture(clone,binding)),'same next complete state/RNG mismatch')
    replay=worker.save_state(clone,binding,folder,'C7979-replay.pth');publish(folder/'RESULT.json',{'status':'PASS','actual_updates':3,'updates':[7978,7979,7979],
            'original_restore':parent,'replay_restore':restored,'same_next_window_exact':True,'same_next_state_exact':True,'checkpoints':[saved,direct,replay],
            'state_for_formal':False,'consumer_geometry_contract':consumer_geometry.identity()})

def formal(c,folder,holder):
    from .data import training_loader
    runtime=worker.prepare_cuda(c,folder);data=worker.dataset(c,'train_core')
    engine,binding,parent=restored_parent(c,runtime,holder,'formal_F0_seed0_e120');write(folder/'binding.json',binding)
    event(folder,'resume_parent_exact',**parent,scope='formal_F0_seed0_e120',used_acceptance_checkpoint=False)
    started=time.monotonic();epoch_times=[];count=0
    with gzip.open(folder/'input-acknowledgements.jsonl.gz','xt',encoding='utf-8') as ack:
        for epoch in range(27,121):
            verify_source(c);require(engine.completed_updates==(epoch-1)*304,'formal resume epoch clock changed')
            plan=worker.plan_epoch(data,epoch);write(folder/f'input-plan-epoch-{epoch:03d}.json',plan);event(folder,'epoch_begin',epoch=epoch,completed_epochs=epoch-1,dropped_ids=plan['ids'][4864:]);t=time.monotonic();windows=0
            for win,(images,targets,traces) in enumerate(training_loader(data,epoch=epoch)):
                metadata=worker.check_inputs(images,targets,traces,plan,epoch,win);holder['input']=metadata;worker.live_admission(folder,c)
                if epoch==27 and win<73:lab_run.receiver_input_check(c,images,targets,traces,win,metadata)
                elif epoch==27 and win in (73,74):input_check(c,images,targets,traces,metadata,win)
                result=engine.train_window(images,targets);clock_check(engine,result,(epoch-1)*304+win+1)
                ack.write(json.dumps(metadata,allow_nan=False)+'\n');ack.flush();event(folder,'update_complete',**result,completed_epochs=engine.completed_updates//304,input_sha256=digest(metadata));windows+=1;count+=1
            require(windows==304 and engine.completed_updates==epoch*304,'formal complete epoch missing')
            last=worker.save_state(engine,binding,folder,f'checkpoint-epoch-{epoch:03d}.pth');duration=time.monotonic()-t;epoch_times.append(duration)
            publish(folder/f'epoch-{epoch:03d}-complete.json',{'epoch':epoch,'updates':engine.completed_updates,'checkpoint':last,'epoch_wall_seconds':duration,'strong_augmentation':epoch<118})
    require(count==28576 and engine.completed_updates==engine.ema.updates==36480,'formal fixed endpoint count changed')
    require(checkpoint.exact_equal(checkpoint.read(last['path'],last['sha256'],binding),checkpoint.capture(engine,binding)),'e120 disk/live state differs')
    endpoint={'status':'F0_E120_EMA_LOCKED','arm':'F0','seed':0,'epoch':120,'updates':36480,'checkpoint':last,'binding':reference(folder/'binding.json'),
              'source_sha256':c['source_sha256'],'contract_sha256':c['contract_sha256'],'original_restore':parent,'used_acceptance_checkpoint':False,
              'parent_update':7904,'actual_updates':count,'training_wall_seconds':time.monotonic()-started,'epoch_wall_seconds':epoch_times,
              'laboratory_actual_cost':None,'evaluation_performed':False,'additional_arms':[],'holdout_accessed':False,'numerical_implementation':numerics.identity()}
    publish(folder/'endpoint-lock.json',endpoint);publish(folder/'RESULT.json',endpoint)

def worker_entry(path,mode):
    c=load(path);prerequisite(c,mode);token(c,mode,path);guard.snapshot=lab_run.snapshot;guard.screen=screen
    folder=ROOT/mode;runtime=Path(c['runtime_root'])/mode;holder={'mode':mode};started=time.time();dispatch=read(ROOT/(mode+'-dispatch.json'))
    require(read(runtime/'supervisor-start.json')['pid']==os.getppid(),'worker supervisor differs')
    current=guard.identity(os.getppid());require(current and current['start']==dispatch['process_identity']['start'] and os.getppid()==dispatch['pid'],'worker supervisor PID/start changed')
    actual_identity=read(runtime/'identity.json')
    require(all(actual_identity.get(k)==v for k,v in runtime_identity(c,mode).items()),'worker runtime protocol identity differs from frozen contract')
    publish(folder/'execution-binding.json',execution_binding(c,mode))
    # Reuse only the fixed64 worker, with the new frozen source validator.
    from . import control
    control.verify_source=verify_source
    try:
        if mode=='diagnostic':diagnostic(c,folder,holder)
        elif mode=='acceptance':acceptance(c,folder,holder)
        elif mode=='formal':formal(c,folder,holder)
        else:
            # Guard completion authenticates a phase binding even for CPU analysis.
            write(folder/'binding.json',{'execution':execution_binding(c,mode),'numerical_implementation':numerics.identity()})
            (worker.run_trace_cache if mode=='trace-cache' else worker.run_trace_analysis)(c,folder,holder)
    except BaseException as error:
        evidence=getattr(error,'evidence',{});engine=holder.get('engine')
        if engine is not None:evidence={**evidence,'engine_phase':engine.phase,'completed_updates':engine.completed_updates}
        if holder.get('audit') is not None:publish(runtime/'consumer-partial.json',{'status':'PARTIAL_FAILURE_OBSERVATIONS_NOT_PASS',**holder['audit'].summary()})
        publish(runtime/'failure.json',{'status':'STOP_NO_RETRY','error':repr(error),'traceback':traceback.format_exc(),'evidence':evidence,'input':holder.get('input')});raise
    finally:write(runtime/'worker-exit.json',{'pid':os.getpid(),'mode':mode,'started_epoch':started,'finished_epoch':time.time(),'body_returned':not (runtime/'failure.json').exists()})
    return 0

def start_chain(path):
    """One remote-only continuation chain; Windows observation is not a lease."""
    c=load(path);complete(c,'diagnostic');complete(c,'acceptance')
    command=[c['python'],'-B',str(SOURCE/'tools/localization/deploy.py'),'consumer-chain-worker','--contract',str(path)]
    publish(ROOT/'continuation-chain-consumed.json',{'contract_sha256':c['contract_sha256'],'source_sha256':c['source_sha256'],
            'command':command,'modes':list(MODES[2:]),'no_retry':True,'epoch':time.time()})
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MKL_THREADING_LAYER='GNU',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (ROOT/'continuation-chain.log').open('x') as log:p=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    value={'pid':p.pid,'process_identity':guard.identity(p.pid),'contract_sha256':c['contract_sha256'],'started_epoch':time.time()}
    publish(ROOT/'continuation-chain-dispatch.json',value);return value

def chain_worker(path):
    c=load(path);record=read(ROOT/'continuation-chain-consumed.json')
    require(record['contract_sha256']==c['contract_sha256'] and record['source_sha256']==c['source_sha256'] and record['modes']==list(MODES[2:]) and record['no_retry'],'continuation chain token changed')
    require(record['command']==[c['python'],'-B',str(SOURCE/'tools/localization/deploy.py'),'consumer-chain-worker','--contract',str(path)],'continuation chain command changed')
    started=time.time();phase=None
    try:
        complete(c,'diagnostic');complete(c,'acceptance')
        for phase in MODES[2:]:
            dispatch=launch(path,phase)
            while True:
                runtime=Path(c['runtime_root'])/phase
                if (runtime/'supervisor-exit.json').is_file():
                    ex=read(runtime/'supervisor-exit.json');require(ex['status']=='PASS','phase supervisor stopped; no retry: '+phase)
                current=guard.identity(dispatch['pid'])
                if original_process_exited(dispatch,current):
                    complete(c,phase);break
                time.sleep(5)
            print(json.dumps({'event':'phase_clean_complete','mode':phase,'elapsed_seconds':time.time()-started}),flush=True)
        publish(ROOT/'CHAIN_RESULT.json',{'status':'F0_E120_FIXED64_COMPLETE_REMOTE_EVIDENCE_PENDING_WINDOWS_RETURN',
                'source_sha256':c['source_sha256'],'contract_sha256':c['contract_sha256'],'started_epoch':started,'finished_epoch':time.time(),
                'elapsed_seconds':time.time()-started,'phases':list(MODES),'no_gpu_retry':True,'instance_power_operation':False})
        return 0
    except BaseException as error:
        publish(ROOT/'CHAIN_STOP.json',{'status':'STOP_PRESERVE_NO_RETRY','phase':phase,'error':repr(error),'traceback':traceback.format_exc(),
                'started_epoch':started,'finished_epoch':time.time(),'elapsed_seconds':time.time()-started,'gpu_retry':False,
                'worker_may_still_be_running':'inspect actual supervisor/worker; chain failure is not GPU failure'})
        raise

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('action',choices=('consumer-prepare','consumer-launch','consumer-supervise','consumer-worker','consumer-start-chain','consumer-chain-worker'));p.add_argument('--request');p.add_argument('--contract');p.add_argument('--mode',choices=MODES);a=p.parse_args(argv)
    if a.action=='consumer-prepare':r=prepare(a.request)
    elif a.action=='consumer-launch':r=launch(a.contract,a.mode)
    elif a.action=='consumer-supervise':return supervise(a.contract,a.mode)
    elif a.action=='consumer-start-chain':r=start_chain(a.contract)
    elif a.action=='consumer-chain-worker':return chain_worker(a.contract)
    else:return worker_entry(a.contract,a.mode)
    print(json.dumps(r,allow_nan=False),flush=True);return 0
