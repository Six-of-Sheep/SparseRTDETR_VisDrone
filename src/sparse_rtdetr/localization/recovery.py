"""Dedicated one-attempt e003 recovery; never replays the original formal launch."""
from __future__ import annotations
import argparse,gc,gzip,json,os,subprocess,sys,tarfile,time
from pathlib import Path
from .runio import (TASK_ID,REMOTE_ROOT,DESIGN_SHA,SELECTION_SHA,PRETRAINED_SHA,KEY_EPOCHS,require,read,write,publish,
                    sha,digest,reference,scoped,execution_binding,original_process_exited)

ROOT=REMOTE_ROOT+'/recovery-002'
RUNTIME_ROOT='/root/autodl-tmp/'+TASK_ID+'-runtime/recovery-002'
ACCEPTANCE_ROOT=REMOTE_ROOT+'/recovery-001/acceptance'
PARENT_SHA='8cc7d52b70e8e03d5e941e9fa3a51c055ab68379d92797ffbb6ac50d111a9fc9'
PARENT_SOURCE='adeef18eaa7dfb95e44e4faf026f5b87277dccb131c910324b056af60ce34a0c'
PARENT_CONTRACT='8b421505ec2f8ff27114ec09644b29a01595c8338807e5c89f6917e8b09cacc8'
PARENT_PATH=REMOTE_ROOT+'/formal/checkpoint-epoch-003.pth'
PHASES=('formal','trace-cache','trace-analysis')
WORKLOAD={'seed':0,'arm':'F0','parent_updates':912,'acceptance_direct_updates':[913,914],
          'acceptance_replay_update':914,'acceptance_actual_updates':0,'reused_acceptance_actual_updates':3,'resume_updates':35568,'target_updates':36480,
          'target_epoch':120,'trace_images':64,'trace_batch_forwards':16,'trace_batch_size':4,
          'acceptance_states_for_formal':False,'automatic_gpu_retry':False,'total_wallclock_cap_seconds':None,
          'idle_seconds':300,'operation_seconds':1800,'additional_arms':[],'evaluation':False,'holdout':False}
CORE=('model.py','criterion.py','data.py','distribution.py','engine.py','checkpoint.py','ignore.py',
      'operators.py','moment_prior.py','trace.py','trace_contract.py','material_paths.py')

def validate(c):
    from .control import PYTHON_REMOTE
    require(c.get('schema')=='f0-recovery-execution-v1' and c.get('task_id')==TASK_ID and c.get('recovery') is True,'not a recovery contract')
    require(c.get('job_root')==ROOT and c.get('state')=='FROZEN_READY' and c.get('workload')==WORKLOAD,'recovery root/workload/state changed')
    require(c.get('runtime_root')==RUNTIME_ROOT and c.get('attempt')=='recovery-002','independent SSD runtime/attempt changed')
    require(c['parent_checkpoint']['path']==PARENT_PATH and c['parent_checkpoint']['sha256']==PARENT_SHA,'acceptance/historical state cannot replace originale003')
    require(c['parent_source_sha256']==PARENT_SOURCE and c['parent_contract_sha256']==PARENT_CONTRACT,'parent source/contract changed')
    require(c['python']==PYTHON_REMOTE and c['design_sha256']==DESIGN_SHA and c['selection_sha256']==SELECTION_SHA,'recipe/env path/design changed')
    require(c['checkpoint_bytes']==512*1024**2,'checkpoint capacity bound must remain conservative512MiB')
    require(c['contract_sha256']==digest({k:v for k,v in c.items() if k!='contract_sha256'}),'recovery contract digest mismatch')

def check_parent(c,*,checkpoint_bytes=True):
    from .control import source_files
    root=Path(REMOTE_ROOT);parent=read(root/'contract.json')
    require(parent['contract_sha256']==PARENT_CONTRACT and digest({k:v for k,v in parent.items() if k!='contract_sha256'})==PARENT_CONTRACT,'original contract bytes/body changed')
    require(sha(c['parent_contract_file']['path'])==c['parent_contract_file']['sha256'],'parent contract file SHA changed')
    require(c['parent_contract_file']['path']==str(root/'contract.json'),'parent contract path changed')
    old_manifest=root/'release-002/source-manifest.json';old_root=root/'release-002/source'
    require(sha(old_manifest)==PARENT_SOURCE,'original source manifest changed')
    listing=read(old_manifest)
    require({r['path'] for r in listing['files']}=={p.relative_to(old_root).as_posix() for p in source_files(old_root)},'parent source inventory changed')
    for row in listing['files']:
        p=scoped(old_root,row['path']);require(p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'parent source file changed')
    new_root=Path(c['source_root'])
    protected=['src/sparse_rtdetr/localization/'+n for n in CORE]
    protected += [r['path'] for r in listing['files'] if r['path'].startswith(('vendor/','configs/','src/sparse_rtdetr/baseline/','src/sparse_rtdetr/data_protocol/'))]
    originals={r['path']:r for r in listing['files']}
    for name in protected:require(sha(new_root/name)==originals[name]['sha256'],'training/trace algorithm changed: '+name)
    for key in ('config','data','data_sha256','pretrained','selection_path','selection_sha256','design_sha256','environment','gpu_uuid'):
        require(c[key]==parent[key],'parent recipe/input/runtime field differs: '+key)
    binding_path=root/'formal/binding.json';side_path=Path(PARENT_PATH+'.complete.json')
    require(c['parent_binding_file']['path']==str(binding_path) and sha(binding_path)==c['parent_binding_file']['sha256'],'parent binding bytes/path changed')
    binding=read(binding_path);side=read(side_path)
    require(side['path']==PARENT_PATH and side['sha256']==PARENT_SHA and side['completed_updates']==912 and side['scope']=='formal_F0_seed0_e120','e003 sidecar invalid')
    require(binding['source_sha256']==PARENT_SOURCE and binding['contract_sha256']==PARENT_CONTRACT and binding['scope']=='formal_F0_seed0_e120','wrong parent binding')
    require(side['binding_sha256']==digest_checkpoint_binding(binding),'parent binding digest differs')
    require(read(root/'formal/supervisor-exit.json')['status']=='STOP','old stopped formal unexpectedly changed')
    if checkpoint_bytes:require(sha(PARENT_PATH)==PARENT_SHA and Path(PARENT_PATH).stat().st_size==side['size_bytes'],'originale003 bytes changed')
    return binding

def digest_checkpoint_binding(binding):
    from .checkpoint import canonical
    import hashlib
    return hashlib.sha256(canonical(binding)).hexdigest()

def verify_storage(c,*,required=40*1024**3):
    from .storage import observe,identity,screen,path_on_mount
    sample=observe(c['job_root']);require(identity(sample)==c['storage']['identity'],'recovery storage mount changed')
    for key in ('io_gate','restore_gate'):
        row=c['storage'][key];p=path_on_mount(row['path'])
        require(sha(p)==row['sha256'] and read(p)['status']=='PASS','original storage gate changed')
    require(read(c['storage']['io_gate']['path'])['mount']['source']==sample['mount']['source'],'I/O gate for another mount')
    gate=read(c['storage']['restore_gate']['path'])
    require(gate['source_sha256']==PARENT_SOURCE and gate['full_state_exact'] and gate['same_next_window_exact'],'reused save/restore gate differs from parent')
    screen(sample,required);return sample

def load(path):
    from .control import verify_source,verify_local_checks
    c=read(path);validate(c)
    require(Path('/proc/sys/kernel/random/boot_id').read_text().strip()==c['boot_id'],'boot changed')
    verify_source(c);verify_local_checks(c);check_parent(c);verify_storage(c)
    verify_materials(c)
    from .acceptance_reuse import verify as verify_reuse
    verify_reuse(c)
    return c

def verify_materials(c):
    require(sha(c['raw_closure']['path'])==c['data_sha256'],'original raw closure changed')
    raw=read(c['raw_closure']['path'])
    require(raw['status']=='PASS' and raw['raw_files']==4933 and raw['data_binding']==c['data'] and raw['selection_sha256']==SELECTION_SHA,'raw closure identity/status invalid')
    require(sha(c['selection_path'])==SELECTION_SHA and sha(c['pretrained']['path'])==PRETRAINED_SHA,'materials changed')

def prepare(request_path):
    from .control import verify_source,verify_local_checks
    from .guard import snapshot,screen
    request=read(request_path);root=Path(ROOT);root.mkdir(exist_ok=True)
    require(not (root/'contract.json').exists() and not any((root/p).exists() for p in PHASES),'immutable recovery contract/phase already exists')
    sample=snapshot(root);gpu=screen(sample,entry=True,disk_bytes=40*1024**3+122*request['checkpoint_bytes'])
    require(gpu==request['gpu_uuid'] and sample['boot_id']==request['boot_id'],'fresh resources differ from reviewed request')
    c={**request,'schema':'f0-recovery-execution-v1','state':'FROZEN_READY','task_id':TASK_ID,'recovery':True,'job_root':ROOT,
       'workload':WORKLOAD,'freeze_epoch':time.time(),'resource_observation':sample}
    from .acceptance_reuse import freeze_bytes
    c['reused_acceptance_bytes']=freeze_bytes(c)
    from .control import environment
    actual=environment()
    from .acceptance_reuse import runtime_equal
    runtime_equal(c['environment'],actual);c['environment_observation']=actual
    ssd=Path(RUNTIME_ROOT)
    require(not ssd.exists() and ssd==ssd.resolve() and ssd.parent.is_relative_to('/root/autodl-tmp'),'new SSD runtime already exists/noncanonical')
    require(Path('/root/autodl-tmp').stat().st_dev!=root.stat().st_dev,'runtime SSD is not independent of NAS')
    c['contract_sha256']=digest(c);validate(c);verify_source(c);verify_local_checks(c);check_parent(c);verify_storage(c);verify_materials(c)
    from .acceptance_reuse import verify as verify_reuse
    verify_reuse(c)
    publish(root/'contract.json',c);return c

def completed(c,mode):
    from .guard import identity
    root=Path(c['job_root']);folder=root/mode;telemetry=Path(c['runtime_root'])/mode;exit=read(telemetry/'supervisor-exit.json')
    require(exit['status']=='PASS' and not (telemetry/'failure.json').exists() and not (telemetry/'mirror-error.json').exists(),'recovery phase not PASS: '+mode)
    dispatch=read(root/(mode+'-dispatch.json'))
    require(dispatch['contract_sha256']==c['contract_sha256'] and original_process_exited(dispatch,identity(dispatch['pid'])),'phase belongs elsewhere or supervisor alive')
    from .channel import verify_archive
    verify_archive(telemetry,folder,{**execution_binding(c,mode),'attempt':c['attempt'],'supervisor_pid':dispatch['pid'],'supervisor_start':dispatch['process_identity']['start']})
    for name in ('RESULT.json','execution-binding.json','worker-exit.json'):
        require(exit['completion_receipts'][name]==reference((telemetry if name=='worker-exit.json' else folder)/name),'completed phase receipt changed')
    require(read(folder/'execution-binding.json')==execution_binding(c,mode),'phase source/contract/runtime binding changed')
    if mode in ('acceptance','formal'):
        require(exit['completion_receipts']['binding.json']==reference(folder/'binding.json'),'checkpoint binding changed')
        b=read(folder/'binding.json')
        expected_scope='gpu_short_window' if mode=='acceptance' else 'formal_F0_seed0_e120'
        origin=read(c['parent_binding_file']['path'])
        for key in ('config','data_sha256'):
            require(b[key]==c[key],'phase config/data changed')
        require(b['scope']==expected_scope and b['run_id']==TASK_ID and b['pretrained_sha256']==PRETRAINED_SHA and b['runtime']==origin['runtime'] and b['model']==origin['model'],'phase state scope/recipe/runtime changed')
        require(b['source_sha256']==c['source_sha256'] and b['contract_sha256']==c['contract_sha256'] and
                b['lineage']=={'attempt':'recovery-002','parent_checkpoint_sha256':PARENT_SHA,'parent_source_sha256':PARENT_SOURCE,
                              'parent_contract_sha256':PARENT_CONTRACT,'acceptance_state':mode=='acceptance'},'phase state lineage changed')
    if mode=='formal':
        endpoint=read(folder/'endpoint-lock.json')
        require(endpoint==read(folder/'RESULT.json'),'e120 endpoint lock differs from authenticated result')
        cp=endpoint['checkpoint'];require(cp['path']==str(folder/'checkpoint-epoch-120.pth') and cp['completed_updates']==36480 and cp['scope']=='formal_F0_seed0_e120','wrong recovered e120 endpoint')
        require(endpoint['binding']==reference(folder/'binding.json') and endpoint['parent_sha256']==PARENT_SHA and endpoint['contract_sha256']==c['contract_sha256'],'endpoint binding/origin changed')
    return read(folder/'RESULT.json')

def prerequisites(c,mode):
    from .acceptance_reuse import verify
    verify(c)
    needed={'formal':[], 'trace-cache':['formal'], 'trace-analysis':['formal','trace-cache']}[mode]
    for phase in needed:completed(c,phase)
    if mode=='formal':
        r=read(Path(ACCEPTANCE_ROOT)/'RESULT.json')
        require(r['status']=='PASS' and r['actual_updates']==3 and r['direct_updates']==[913,914] and r['replayed_update']==914 and r['same_next_state_exact'] and r['same_next_window_exact'] and r['parent_sha256']==PARENT_SHA,'recovery acceptance workload/identity invalid')
        require(all(cp['size_bytes']<=c['checkpoint_bytes'] for cp in r['checkpoints']),'measured checkpoint exceeds frozen capacity bound')
        ready=read(Path(ROOT)/'windows-backup-ready.json')
        require(ready['source_sha256']==c['source_sha256'] and ready['contract_sha256']==c['contract_sha256'] and 0<=time.time()-ready['observed_epoch']<120,'recovery backup watcher not ready/fresh')
        require(ready['free_bytes']>=20*c['checkpoint_bytes']+2*1024**3,'Windows reserve insufficient for originals plus preserved transport payloads')

def dispatch(contract_path,mode):
    from .control import REPO
    from .guard import snapshot,screen,identity
    require(mode in PHASES,'new GPU acceptance/other phase prohibited')
    c=load(contract_path);prerequisites(c,mode);root=Path(ROOT)
    require(mode in PHASES and not (root/mode).exists(),'phase exists; no automatic recovery retry')
    sample=snapshot(root);screen(sample,gpu_uuid=c['gpu_uuid'],entry=True,gpu=mode!='trace-analysis',
          disk_bytes=40*1024**3+(122 if mode=='formal' else 4)*c['checkpoint_bytes'])
    require(sample['boot_id']==c['boot_id'],'boot changed at dispatch')
    command=[c['python'],'-B',str(REPO/'tools/localization/deploy.py'),'recovery-supervise','--contract',str(Path(contract_path).resolve()),'--mode',mode]
    token=root/(mode+'-launch-consumed.json')
    publish(token,{'contract_sha256':c['contract_sha256'],'mode':mode,'consumed_epoch':time.time(),'command':command,'automatic_retry':False,'resources':sample})
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',MKL_THREADING_LAYER='GNU',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (root/(mode+'-dispatcher.log')).open('x') as log:
        child=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    result={'mode':mode,'pid':child.pid,'process_identity':identity(child.pid),'contract_sha256':c['contract_sha256'],
            'token':reference(token),'dispatched_epoch':time.time(),'status':'DISPATCHED_NOT_ACCEPTED'}
    publish(root/(mode+'-dispatch.json'),result);return result

def supervise(contract_path,mode):
    from .control import REPO
    from .guard import supervise as guard
    c=load(contract_path);prerequisites(c,mode);token=read(Path(ROOT)/(mode+'-launch-consumed.json'))
    require(token['contract_sha256']==c['contract_sha256'] and token['mode']==mode,'recovery launch token changed')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=c['gpu_uuid'] if mode!='trace-analysis' else '',OMP_NUM_THREADS='2' if mode!='trace-analysis' else '16',MKL_NUM_THREADS='2' if mode!='trace-analysis' else '16')
    command=[c['python'],'-B',str(REPO/'tools/localization/deploy.py'),'recovery-worker','--contract',str(Path(contract_path).resolve()),'--mode',mode]
    from .guard import identity
    return guard(command,Path(ROOT)/mode,env,gpu_uuid=c['gpu_uuid'],boot_id=c['boot_id'],gpu=mode!='trace-analysis',checkpoint_bytes=c['checkpoint_bytes'],formal=mode=='formal',
                 runtime=Path(c['runtime_root'])/mode,run_identity={**execution_binding(c,mode),'attempt':c['attempt'],
                 'supervisor_pid':os.getpid(),'supervisor_start':identity(os.getpid())['start']})

def restored_engine(c,runtime,scope,holder):
    from . import worker,checkpoint
    engine,active=worker.make_engine(c,runtime,scope);holder['engine']=engine
    origin=read(c['parent_binding_file']['path'])
    require(active['model']==origin['model'] and active['runtime']==origin['runtime'] and active['config']==origin['config'],'new construction differs from e003 recipe/runtime')
    restored=checkpoint.restore(engine,origin,PARENT_PATH,PARENT_SHA)
    require(restored['full_state_exact'] is True and restored['completed_updates']==912,'parent full restoration receipt invalid')
    require(engine.completed_updates==engine.ema.updates==912 and engine.phase==0 and not engine.failed,'restore cursor/EMA not912')
    # checkpoint.restore already recaptures the complete old binding/state/RNG
    # exactly. Only after this succeeds are new output bindings introduced.
    return engine,active,restored

def comparable(result):return {k:v for k,v in result.items() if k not in ('elapsed_seconds','complete_cycle_seconds')}

def acceptance(c,out,holder):
    import torch
    from . import worker,checkpoint
    from .data import training_loader
    from .runio import event
    runtime=worker.prepare_cuda(c,out);data=worker.dataset(c,'train_core');plan=worker.plan_epoch(data,4)
    engine,binding,restored=restored_engine(c,runtime,'gpu_short_window',holder)
    write(Path(out)/'binding.json',binding);event(out,'parent_full_restore',**restored)
    direct=[];saved=None;torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    for window,(images,targets,traces) in enumerate(training_loader(data,epoch=4,window_limit=2)):
        require(engine.completed_updates==912+window,'acceptance clock before update')
        metadata=worker.check_inputs(images,targets,traces,plan,4,window);holder['input']=metadata;worker.live_admission(out,c)
        result=engine.train_window(images,targets,capture=True);result['input']=metadata;direct.append(result)
        event(out,'update_complete',**result,completed_epochs=3)
        if window==0:saved=worker.save_state(engine,binding,out,'C913.pth')
    require(len(direct)==2 and engine.completed_updates==914,'acceptance direct windows expanded/incomplete')
    direct_cp=worker.save_state(engine,binding,out,'C914-direct.pth')
    expected=checkpoint.read(direct_cp['path'],direct_cp['sha256'],binding)
    peak={'allocated_bytes':torch.cuda.max_memory_allocated(),'reserved_bytes':torch.cuda.max_memory_reserved()}
    holder.pop('engine');del engine;gc.collect();torch.cuda.empty_cache()
    clone,other=worker.make_engine(c,runtime,'gpu_short_window');holder['engine']=clone
    require(other==binding,'acceptance reconstruction binding changed')
    replay_restore=checkpoint.restore(clone,binding,saved['path'],saved['sha256']);replayed=[]
    for images,targets,traces in training_loader(data,epoch=4,completed_windows=1,window_limit=1):
        require(clone.completed_updates==913,'replay starts at wrong cursor')
        metadata=worker.check_inputs(images,targets,traces,plan,4,1);holder['input']=metadata;worker.live_admission(out,c)
        result=clone.train_window(images,targets,capture=True);result['input']=metadata;replayed.append(result)
    require(len(replayed)==1 and clone.completed_updates==914,'replay workload changed')
    require(comparable(direct[-1])==comparable(replayed[0]),'same restored window output/loss/gradient/input mismatch')
    require(checkpoint.exact_equal(expected,checkpoint.capture(clone,binding)),'same restored next complete state/RNG mismatch')
    replay_cp=worker.save_state(clone,binding,out,'C914-replay.pth')
    publish(Path(out)/'RESULT.json',{'status':'PASS','parent_sha256':PARENT_SHA,'direct_updates':[913,914],'replayed_update':914,
            'actual_updates':3,'same_next_state_exact':True,'same_next_window_exact':True,'original_restore':restored,'replay_restore':replay_restore,
            'checkpoints':[saved,direct_cp,replay_cp],'complete_update_seconds':[r['elapsed_seconds'] for r in direct+replayed],
            'input_evidence':[r['input'] for r in direct+replayed],'peak_memory':peak,'elapsed_seconds':time.monotonic()-start,'formal_state_usable':False})

def resume(c,out,holder):
    from . import worker,checkpoint
    from .data import training_loader
    from .runio import event
    runtime=worker.prepare_cuda(c,out);data=worker.dataset(c,'train_core')
    engine,binding,restored=restored_engine(c,runtime,'formal_F0_seed0_e120',holder)
    write(Path(out)/'binding.json',binding);publish(Path(out)/'parent-restore.json',restored)
    event(out,'resume_parent_exact',completed_updates=912,completed_epochs=3,parent_sha256=PARENT_SHA)
    start=time.monotonic();epoch_times=[];new_updates=0;last=None
    with gzip.open(Path(out)/'input-acknowledgements.jsonl.gz','xt',encoding='utf-8') as acknowledgements:
        for epoch in range(4,121):
            from .control import verify_source
            verify_source(c);require(engine.completed_updates==(epoch-1)*304,'resume epoch clock changed')
            plan=worker.plan_epoch(data,epoch);write(Path(out)/f'input-plan-epoch-{epoch:03d}.json',plan)
            event(out,'epoch_begin',epoch=epoch,completed_epochs=epoch-1,dropped_ids=plan['ids'][4864:]);begin=time.monotonic();count=0
            for window,(images,targets,traces) in enumerate(training_loader(data,epoch=epoch)):
                require(engine.completed_updates==(epoch-1)*304+window,'resume data/update clock mismatch')
                metadata=worker.check_inputs(images,targets,traces,plan,epoch,window);holder['input']=metadata
                worker.live_admission(out,c);result=engine.train_window(images,targets)
                acknowledgements.write(json.dumps(metadata,allow_nan=False)+'\n');acknowledgements.flush()
                event(out,'update_complete',**result,completed_epochs=engine.completed_updates//304,input_sha256=digest(metadata))
                count+=1;new_updates+=1
            require(count==304 and engine.completed_updates==epoch*304,'resume epoch count changed')
            last=worker.save_state(engine,binding,out,f'checkpoint-epoch-{epoch:03d}.pth');elapsed=time.monotonic()-begin;epoch_times.append(elapsed)
            publish(Path(out)/f'epoch-{epoch:03d}-complete.json',{'epoch':epoch,'updates':engine.completed_updates,'checkpoint':last,'epoch_wall_seconds':elapsed,'strong_augmentation':epoch<118})
    require(new_updates==35568 and engine.completed_updates==36480,'resume fixed workload not completed')
    disk=checkpoint.read(last['path'],last['sha256'],binding)
    require(checkpoint.exact_equal(disk,checkpoint.capture(engine,binding)),'e120 disk/live complete state differs')
    elapsed=time.monotonic()-start
    endpoint={'status':'F0_E120_EMA_LOCKED','arm':'F0','seed':0,'epoch':120,'updates':36480,'new_updates':35568,
              'checkpoint':last,'binding':reference(Path(out)/'binding.json'),'source_sha256':c['source_sha256'],'contract_sha256':c['contract_sha256'],
              'parent_sha256':PARENT_SHA,'start_updates':912,'redo_completed_updates':49,'used_short_checkpoint':False,'used_acceptance_checkpoint':False,
              'training_wall_seconds':elapsed,'epoch_wall_seconds':epoch_times,'task_time_cost_estimate_cny':elapsed/3600*1.88,'actual_bill_cny':None,
              'evaluation_performed':False,'additional_arms':[],'holdout_accessed':False}
    publish(Path(out)/'endpoint-lock.json',endpoint);publish(Path(out)/'RESULT.json',endpoint)

def worker_entry(contract_path,mode):
    from . import worker
    c=load(contract_path);out=Path(ROOT)/mode;supervisor=read(Path(c['runtime_root'])/mode/'supervisor-start.json')
    require(supervisor['pid']==os.getppid(),'recovery worker is not owned by the recorded supervisor')
    from .channel import binding
    b=binding();require(b is not None and b['runtime_root']==str(Path(c['runtime_root'])/mode),'worker telemetry root mismatch')
    for key,value in execution_binding(c,mode).items():require(b[key]==value,'worker channel binding mismatch')
    actions={'acceptance':acceptance,'formal':resume,'trace-cache':worker.run_trace_cache,'trace-analysis':worker.run_trace_analysis}
    worker.execute(c,mode,out,actions=actions);return 0

def status():
    from .guard import identity
    root=Path(ROOT);s={'task_id':TASK_ID,'job_root':ROOT,'observed_epoch':time.time(),'phases':{},'key_checkpoints':[],'acceptance_checkpoints':[]}
    c=read(root/'contract.json');validate(c)
    for mode in PHASES:
        row={};folder=root/mode
        if (folder/'RESULT.json').is_file():row['RESULT.json']=read(folder/'RESULT.json')
        telemetry=Path(RUNTIME_ROOT)/mode
        if telemetry.exists():
            from .channel import latest
            b=read(telemetry/'identity.json')
            for key,value in execution_binding(c,mode).items():require(b[key]==value,'outer progress channel run identity mismatch')
            row['progress.json']=latest(telemetry,'progress')
            row['resource_sample']=latest(telemetry,'resources')
            for name in ('supervisor-exit.json','failure.json','mirror-error.json'):
                if (telemetry/name).is_file():row[name]=read(telemetry/name)
        if (root/(mode+'-dispatch.json')).is_file():
            d=read(root/(mode+'-dispatch.json'));row['dispatch']=d;row['supervisor_current_identity']=identity(d['pid'])
        if row:s['phases'][mode]=row
    for epoch in (0,1,3,30,60,90,120):
        base=Path(REMOTE_ROOT)/'formal' if epoch<=3 else root/'formal';p=base/f'checkpoint-epoch-{epoch:03d}.pth.complete.json'
        if p.is_file():s['key_checkpoints'].append({'epoch':epoch,'receipt':read(p),'receipt_file':reference(p)})
    for name in ('C913.pth','C914-direct.pth','C914-replay.pth'):
        p=Path(ACCEPTANCE_ROOT)/(name+'.complete.json')
        if p.is_file():s['acceptance_checkpoints'].append({'name':name,'receipt':read(p),'receipt_file':reference(p)})
    return s

def evidence_export():
    c=load(Path(ROOT)/'contract.json')
    for mode in PHASES:completed(c,mode)
    root=Path(ROOT);out=root/'export'
    if (out/'RESULT.json').is_file():
        r=read(out/'RESULT.json');require(r['contract_sha256']==c['contract_sha256'] and r['archive']==reference(out/'evidence.tar.gz'),'existing recovery export changed');return r
    require(not out.exists(),'incomplete recovery export preserved')
    allowed=[]
    for p in root.rglob('*'):
        require(not p.is_symlink(),'symlink in recovery evidence')
        if p.is_file() and p.suffix!='.pth':allowed.append(p)
    require(not any(p.name.endswith(('.partial','.publishing','.next')) for p in allowed),'incomplete recovery artifact remains')
    # Add immutable old evidence/source manifests/raw membership via the prior
    # frozen receipts, without taking any unselected annotation directory scan.
    allowed += [Path(c['parent_contract_file']['path']),Path(c['parent_binding_file']['path']),Path(REMOTE_ROOT)/'formal/supervisor-exit.json',
                Path(c['source_manifest']),Path(REMOTE_ROOT)/'release-002/source-manifest.json',Path(c['raw_closure']['path'])]
    from .control import raw_export_files
    raw_paths=raw_export_files({**c,'job_root':REMOTE_ROOT})
    frozen_raw={row['annotation']['path']:row['annotation'] for row in read(c['raw_closure']['path'])['files']}
    for p in raw_paths:require(reference(p)==frozen_raw[str(p)],'original raw evidence bytes changed after training')
    allowed += raw_paths
    verify_fixed_trace_evidence(c)
    local=Path(c['local_checks']['path']);allowed.append(local)
    allowed += [scoped(local.parent,r['path']) for r in read(local)['evidence_files']]
    listing=[{'path':p.relative_to(REMOTE_ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(set(allowed))]
    epoch_rows=[]
    for epoch in range(121):
        base=Path(REMOTE_ROOT)/'formal' if epoch<=3 else root/'formal';p=base/f'checkpoint-epoch-{epoch:03d}.pth'
        side=read(str(p)+'.complete.json')
        binding=read(base/'binding.json')
        require(side['path']==str(p) and side['completed_updates']==epoch*304 and side['scope']=='formal_F0_seed0_e120' and
                side['size_bytes']==p.stat().st_size and side['binding_sha256']==digest_checkpoint_binding(binding) and sha(p)==side['sha256'],
                'logical epoch complete state missing/changed')
        epoch_rows.append({'epoch':epoch,'checkpoint':side})
        print(json.dumps({'event':'export_epoch_sha_verified','epoch':epoch,'wall_epoch':time.time()}),flush=True)
    out.mkdir(exist_ok=False)
    write(out/'manifest.json',{'task_id':TASK_ID,'contract_sha256':c['contract_sha256'],'source_sha256':c['source_sha256'],
                              'files':listing,'epoch_lineage':epoch_rows,'original_stop_preserved':True})
    archive=out/'evidence.tar.gz'
    with tarfile.open(archive,'x:gz') as t:
        t.add(out/'manifest.json',arcname='manifest.json',recursive=False)
        for r in listing:t.add(Path(REMOTE_ROOT)/r['path'],arcname=r['path'],recursive=False)
    result={'task_id':TASK_ID,'source_sha256':c['source_sha256'],'contract_sha256':c['contract_sha256'],'archive':reference(archive),'files':len(listing),'finished_epoch':time.time()}
    write(out/'RESULT.json',result);return result

def verify_fixed_trace_evidence(c):
    root=Path(c['job_root']);collection=read(root/'trace-cache/RESULT.json');analysis=read(root/'trace-analysis/RESULT.json')
    selected=[r['stable_image_id'] for r in read(c['selection_path'])['selected']]
    entries=collection['files'];reports=analysis['image_reports']
    require(len(selected)==len(entries)==len(reports)==64 and len(set(selected))==64,'fixed64 evidence cardinality changed')
    require([e['image_id'] for e in entries]==selected,'cache identity order differs from fixed64')
    for index,(item,report) in enumerate(zip(entries,reports)):
        cache=root/'trace-cache'/f'trace-{index:03d}.npz';path=root/'trace-analysis'/f'image-{index:03d}.json'
        require(item['path']==str(cache) and not cache.is_symlink() and reference(cache)=={k:v for k,v in item.items() if k!='image_id'},'fixed cache missing/changed')
        require(report['path']==str(path) and not path.is_symlink() and reference(path)==report,'fixed analysis report missing/changed')
        require(read(path)['image_id']==selected[index],'analysis report identity changed')

def main(argv):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    prep=sub.add_parser('prepare-recovery');prep.add_argument('--request',required=True)
    for name in ('launch-recovery-check','launch-resume'):cmd=sub.add_parser(name);cmd.add_argument('--contract',required=True)
    cmd=sub.add_parser('launch-recovery-trace');cmd.add_argument('--contract',required=True);cmd.add_argument('--mode',choices=['trace-cache','trace-analysis'],required=True)
    for name in ('recovery-supervise','recovery-worker'):
        cmd=sub.add_parser(name);cmd.add_argument('--contract',required=True);cmd.add_argument('--mode',choices=list(PHASES),required=True)
    sub.add_parser('recovery-status');sub.add_parser('recovery-export')
    cmd=sub.add_parser('remote-recovery');cmd.add_argument('--deployment',required=True);cmd.add_argument('--action',dest='remote_action',choices=['prepare','acceptance','resume','trace-cache','trace-analysis','status','export'],required=True);cmd.add_argument('--receipt',required=True);cmd.add_argument('--request')
    for name in ('watch-recovery-backups','sync-recovery'):
        cmd=sub.add_parser(name);cmd.add_argument('--deployment',required=True);cmd.add_argument('--out',required=True)
    a=p.parse_args(argv)
    if a.action=='prepare-recovery':r=prepare(a.request)
    elif a.action=='launch-recovery-check':r=dispatch(a.contract,'acceptance')
    elif a.action=='launch-resume':r=dispatch(a.contract,'formal')
    elif a.action=='launch-recovery-trace':r=dispatch(a.contract,a.mode)
    elif a.action=='recovery-supervise':return supervise(a.contract,a.mode)
    elif a.action=='recovery-worker':return worker_entry(a.contract,a.mode)
    elif a.action=='recovery-status':r=status()
    elif a.action=='recovery-export':r=evidence_export()
    else:
        from . import recovery_client
        if a.action=='remote-recovery':r=recovery_client.remote(a.deployment,a.remote_action,a.receipt,request=a.request)
        elif a.action=='watch-recovery-backups':r=recovery_client.watch(a.deployment,a.out)
        else:r=recovery_client.sync(a.deployment,a.out)
    print(json.dumps(r,ensure_ascii=False,allow_nan=False),flush=True);return 0
