"""Explicit F0 preparation, freeze, single launch, status and evidence export.

Import and --help are inert. All model work lives in a separately supervised
worker; this entry has no FW/MM/MP/evaluation/holdout or automatic resume mode.
"""
from __future__ import annotations
import argparse,hashlib,importlib,io,json,os,shlex,shutil,subprocess,sys,tarfile,time
from pathlib import Path,PurePosixPath
from .runio import (TASK_ID,REMOTE_ROOT,DESIGN_SHA,SELECTION_SHA,PRETRAINED_SHA,KEY_EPOCHS,SHORT_STATES,
                    PHASES,require,read,write,publish,atomic,sha,digest,reference,scoped,
                    execution_binding,original_process_exited)

REPO=Path(__file__).resolve().parents[3]
PYTHON_REMOTE='/root/autodl-tmp/c2-autodl-20261008-p1/environment-work/attempt-20261008T015133-7816d725/env/bin/python'
WORKLOAD={"formal_arms":["F0"],"seed":0,"epochs":120,"updates_per_epoch":304,"formal_updates":36480,
          "short_unique_updates":6,"short_replay_update":6,"short_actual_updates":7,"short_checkpoint_after_update":5,
          "trace_images":64,"trace_batch_size":4,"trace_batch_forwards":16,"key_checkpoint_epochs":list(KEY_EPOCHS),
          "total_wallclock_cap_seconds":None,"idle_seconds":300,"operation_seconds":1800,
          "automatic_gpu_retry":False,"use_short_state_for_formal":False,"holdout":False,"evaluation":False}

def transport():
    folder=REPO/'tools/localization/transport'
    require(folder.is_dir(),'F0 transport source missing')
    sys.path.insert(0,str(folder))
    module=importlib.import_module('local_ops')
    require(Path(module.__file__).resolve()==folder/'local_ops.py','another task transport module is cached')
    return module

def source_files(root):
    paths=[]
    for directory in ('src/sparse_rtdetr','vendor/rtdetrv2_pytorch','configs','tools/localization'):
        folder=root/directory
        if not folder.is_dir():continue
        for path in folder.rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts:continue
            if path.suffix not in ('.py','.yml','.yaml','.json','.toml') and path.name not in ('LICENSE','NOTICE'):continue
            require(not path.is_symlink(),'source symlink not allowed')
            paths.append(path)
    for name in ('LICENSE','pyproject.toml','tools/localization_operator_check.py','tests/test_localization_f0.py','tests/test_localization_lifecycle.py','tests/test_localization_storage.py','tests/test_localization_recovery_prep.py','tests/test_localization_recovery.py','tests/test_localization_channel.py'):
        if (root/name).is_file():paths.append(root/name)
    return sorted(set(paths))

def export_source(out):
    out=Path(out);out.mkdir(exist_ok=False,parents=True);rows=[]
    for path in source_files(REPO):
        relative=path.relative_to(REPO).as_posix();destination=out/'source'/relative
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
        require(sha(destination)==sha(path),'source export byte mismatch')
        rows.append({'path':relative,'bytes':destination.stat().st_size,'sha256':sha(destination)})
    manifest={'schema':'f0-source-export-v1','task_id':TASK_ID,'files':rows}
    write(out/'source-manifest.json',manifest)
    archive=out/'source.tar.gz'
    with tarfile.open(archive,'x:gz') as tar:
        tar.add(out/'source-manifest.json',arcname='source-manifest.json',recursive=False)
        for row in rows:tar.add(out/'source'/row['path'],arcname='source/'+row['path'],recursive=False)
    result={'task_id':TASK_ID,'source_sha256':sha(out/'source-manifest.json'),'source_files':len(rows),
            'archive':reference(archive),'source_root':str(out/'source'),'cloud_contact':False}
    write(out/'EXPORT.json',result);return result

def verify_source(contract):
    root=Path(contract['source_root']).resolve();manifest=Path(contract['source_manifest'])
    require(root==REPO.resolve(),'executing from a different source root')
    require(sha(manifest)==contract['source_sha256'],'source manifest changed')
    listing=read(manifest);require(listing['task_id']==TASK_ID,'source task binding')
    expected=set()
    for row in listing['files']:
        path=scoped(root,row['path']);expected.add(row['path'])
        require(path.is_file() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'source bytes changed: '+row['path'])
    require(expected=={p.relative_to(root).as_posix() for p in source_files(root)},'unexpected source file inventory')

def validate_contract(c):
    require(c.get('schema')=='f0-execution-contract-v1' and c.get('task_id')==TASK_ID,'wrong execution contract')
    require(c.get('workload')==WORKLOAD,'frozen workload or stop policy changed')
    require(c.get('job_root')==REMOTE_ROOT,'wrong F0 job root')
    require(c.get('storage',{}).get('schema')=='f0-file-storage-v1','file-storage contract missing')
    require(c['storage'].get('identity',{}).get('canonical_mount')=='/autodl-fs/data','wrong account file-storage mount')
    require(c.get('design_sha256')==DESIGN_SHA and c.get('selection_sha256')==SELECTION_SHA,'design/selection mismatch')
    require(c.get('pretrained',{}).get('sha256')==PRETRAINED_SHA,'wrong ImageNet initialization')
    require(c.get('state')=='FROZEN_READY','plan is not executable')
    require(isinstance(c.get('gpu_uuid'),str) and c['gpu_uuid'].startswith('GPU-') and len(c['gpu_uuid'])==40,'missing full GPU UUID')
    require(isinstance(c.get('boot_id'),str) and len(c['boot_id'])==36,'missing boot identity')
    require(c.get('python')==PYTHON_REMOTE,'reuse the admitted environment path; no bootstrap execution')
    require(c.get('local_checks',{}).get('status')=='LOCAL_READY' and c['local_checks'].get('source_sha256')==c.get('source_sha256'),'local verification source differs')
    require(isinstance(c['local_checks'].get('sha256'),str) and isinstance(c['local_checks'].get('path'),str),'local readiness file is not bound')
    require(c.get('raw_closure',{}).get('status')=='PASS' and c['raw_closure'].get('raw_files')==4933,'raw closure missing')
    require(c.get('reusable_f0_audit',{}).get('status')=='NO_ELIGIBLE_F0_FOUND','reusable F0 audit not resolved')
    require(c.get('backup_plan',{}).get('epochs')==list(KEY_EPOCHS) and c['backup_plan'].get('all_epoch_checkpoints_retained') is True,'backup plan incomplete')
    body={k:v for k,v in c.items() if k!='contract_sha256'}
    require(c.get('contract_sha256')==digest(body),'contract digest mismatch')

def load_contract(path,*,files=True):
    c=read(path);validate_contract(c)
    require(Path('/proc/sys/kernel/random/boot_id').read_text().strip()==c['boot_id'],'boot changed since freeze')
    from .storage import verify
    verify(c)
    if files:
        verify_source(c)
        verify_local_checks(c)
        require(sha(c['raw_closure']['path'])==c['data_sha256'],'raw closure receipt changed')
        require(sha(c['selection_path'])==SELECTION_SHA,'trace selection changed')
        require(sha(c['pretrained']['path'])==PRETRAINED_SHA,'pretrained source changed')
        for row in c.get('evidence_files',[]):require(sha(row['path'])==row['sha256'],'frozen prerequisite changed')
    return c

def verify_local_checks(c):
    from .local_ready import verify
    row=c['local_checks'];path=Path(row['path'])
    require(path==Path(c['source_root']).parent/'local-checks/LOCAL_READY.json','local check path outside selected release')
    require(sha(path)==row['sha256'],'local readiness file changed')
    verify(path,c['source_sha256'])

def raw_preflight(request,out):
    # Imports parse/data only for explicitly authorized staged role trees.
    from .worker import dataset
    from .data import parse_raw,read_bound
    out=Path(out);out.mkdir(exist_ok=False,parents=True);started=time.monotonic();raw_rows=[];counts={}
    for role in ('train_core','development'):
        data=dataset(request,role);counts[role]=0
        for row in data.records:
            image=data.images_path/Path(row['relative_path']).name
            require(image.is_file() and not image.is_symlink() and image.stat().st_size==row['image_size_bytes'],'bound image missing/size changed')
            path=data.annotations_path/Path(row['annotation_relative_path']).name
            annotation=read_bound(path,row['annotation_sha256'],row['annotation_size_bytes'])
            parsed=parse_raw(annotation,row['stable_image_id'],expected_rows=row['raw_annotation_rows'])
            raw_rows.append({'role':role,'image_id':row['stable_image_id'],'annotation':reference(path),'physical_rows':len(parsed['rows']),'counts':parsed['counts']})
            counts[role]+=1
        require(counts[role]==(4869 if role=='train_core' else 64),'raw role closure cardinality')
    result={'status':'PASS','task_id':TASK_ID,'raw_files':len(raw_rows),'roles':counts,'files':raw_rows,'data_binding':request['data'],'selection_sha256':SELECTION_SHA,
            'image_content_policy':'reuse historical input proof; verify SHA on every actual model read',
            'image_loads':0,'model_calls':0,'elapsed_seconds':time.monotonic()-started}
    write(out/'RESULT.json',result);return reference(out/'RESULT.json')

def stage_raw(request):
    from .worker import dataset
    from .data import read_bound
    result=[]
    for role in ('train_core','development'):
        root=Path(request['data'][role]['roots']['annotations'])
        require(root==Path(REMOTE_ROOT)/'materials'/role/'annotations','raw staging root outside authorized task role')
        root.mkdir(parents=True,exist_ok=True)
        source=Path(request['raw_sources'][role])
        require(source.is_absolute() and source.is_dir() and source.name=='annotations','explicit source annotation directory required')
        parts=[p.lower().replace('-','_') for p in source.resolve().parts]
        require(not any(any(k in p for k in ('holdout','confirmatory','test_dev','testdev')) for p in parts),'forbidden raw source role')
        allowed={'train_core':{'train','train_core','visdrone2019_det_train'},'development':{'val','development','visdrone2019_det_val'}}[role]
        require(parts[-2] in allowed,'source root does not declare the requested original split')
        require(not any(p.is_symlink() for p in (source,*source.parents)),'raw source symlink path')
        data=dataset(request,role)
        for row in data.records:
            name=Path(row['annotation_relative_path']).name;src=source/name;dst=root/name
            raw=read_bound(src,row['annotation_sha256'],row['annotation_size_bytes'])
            if dst.exists():require(sha(dst)==row['annotation_sha256'],'staged raw target differs; preserve it')
            else:
                with dst.open('xb') as out:out.write(raw);out.flush();os.fsync(out.fileno())
            result.append({'role':role,'image_id':row['stable_image_id'],'source':str(src),'target':str(dst),'sha256':sha(dst)})
    require(len(result)==4933,'raw staging membership expanded/incomplete')
    receipt=Path(REMOTE_ROOT)/('raw-staging-'+str(time.time_ns())+'.json')
    write(receipt,{'status':'PASS','task_id':TASK_ID,'files':result,'model_calls':0,'allowed_roles':['train_core','fixed64_development']})
    return reference(receipt)

def environment():
    import platform,torch,torchvision,numpy,scipy,PIL
    from .model import F0Config
    from . import checkpoint,criterion,data,distribution,trace,trace_contract,worker
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(),'environment check must remain CPU only')
    F0Config().validate()
    return {'task_id':TASK_ID,'status':'NEW_IMPORTS_PASS','python':sys.executable,'python_version':platform.python_version(),
            'torch':str(torch.__version__),'torchvision':torchvision.__version__,'numpy':numpy.__version__,'scipy':scipy.__version__,
            'pillow':PIL.__version__,'cuda_build':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
            'model_calls':0,'cuda_initialized':torch.cuda.is_initialized(),'source_root':str(REPO)}

def freeze(request_path,out):
    request=read(request_path);require(os.name=='posix','freeze on actual cloud host after read-only audit')
    require(not Path(out).exists(),'freeze is immutable')
    from .guard import snapshot,screen
    from .model import F0Config
    import torch,torchvision
    require(not torch.cuda.is_initialized(),'freeze must remain CPU only')
    observed=snapshot(REMOTE_ROOT);gpu=screen(observed,entry=True)
    require(gpu==request.get('gpu_uuid') and observed['boot_id']==request.get('boot_id'),'fresh admission differs from reviewed request')
    value={**request,'schema':'f0-execution-contract-v1','task_id':TASK_ID,'state':'FROZEN_READY',
           'job_root':REMOTE_ROOT,'python':PYTHON_REMOTE,'design_sha256':DESIGN_SHA,'selection_sha256':SELECTION_SHA,
           'workload':WORKLOAD,'config':F0Config().binding(),
           'environment':{'torch':str(torch.__version__),'torchvision':torchvision.__version__,'cuda_build':torch.version.cuda,'cudnn':torch.backends.cudnn.version()},
           'freeze_epoch':time.time(),'resource_observation':observed}
    value['contract_sha256']=digest(value);validate_contract(value);verify_source(value);verify_local_checks(value)
    from .storage import verify as verify_storage
    verify_storage(value)
    require(sha(value['raw_closure']['path'])==value['data_sha256'],'raw closure not bound')
    closure=read(value['raw_closure']['path']);require(closure['status']=='PASS' and closure['raw_files']==4933,'raw prerequisite file invalid')
    require(closure['data_binding']==value['data'] and closure['selection_sha256']==SELECTION_SHA,'raw closure is for a different data binding')
    require(sha(value['pretrained']['path'])==PRETRAINED_SHA and sha(value['selection_path'])==SELECTION_SHA,'freeze materials mismatch')
    for row in value.get('evidence_files',[]):require(sha(row['path'])==row['sha256'],'prerequisite receipt mismatch')
    publish(out,value);return value

def phase_complete(c,phase):
    from .guard import identity
    root=Path(c['job_root']);folder=root/phase
    result=read(folder/'supervisor-exit.json')
    require(result['status']=='PASS','prior phase supervisor not PASS: '+phase)
    dispatch=read(root/(phase+'-dispatch.json'))
    require(original_process_exited(dispatch,identity(dispatch['pid'])),'prior supervisor is still running: '+phase)
    require(not (folder/'failure.json').exists(),'prior failure preserved; do not launch dependent work')
    expected=['RESULT.json','execution-binding.json','worker-exit.json']
    if phase in ('short','formal'):expected.append('binding.json')
    for name in expected:
        require(result['completion_receipts'][name]==reference(folder/name),'completed phase receipt changed: '+phase+'/'+name)
    require(read(folder/'execution-binding.json')==execution_binding(c,phase),'prior phase contract/source/data/runtime mismatch: '+phase)
    if phase in ('short','formal'):
        binding=read(folder/'binding.json')
        require(binding['run_id']==TASK_ID and binding['scope']=={'short':'gpu_short_window','formal':'formal_F0_seed0_e120'}[phase],'prior phase state scope mismatch')
        for key in ('contract_sha256','source_sha256','data_sha256','config'):
            require(binding[key]==c[key],'prior phase state binding mismatch: '+key)
        require(binding['pretrained_sha256']==PRETRAINED_SHA and binding['runtime']['gpu_uuid']==c['gpu_uuid'],'prior initialization/device binding mismatch')
        for key,value in c['environment'].items():require(binding['runtime'][key]==value,'prior runtime mismatch: '+key)

def prerequisites(c,mode):
    root=Path(c['job_root'])
    needed={'short':[], 'formal':['short'], 'trace-cache':['short','formal'], 'trace-analysis':['short','formal','trace-cache']}[mode]
    for phase in needed:phase_complete(c,phase)
    if mode=='formal':
        r=read(root/'short/RESULT.json')
        require(r['status']=='PASS' and r['unique_updates']==6 and r['actual_updates']==7 and r['same_next_state_exact'] and r['same_next_window_exact'],'short acceptance invalid')
        backup=read(root/'windows-backup-ready.json')
        require(backup['task_id']==TASK_ID and backup['source_sha256']==c['source_sha256'] and backup['epochs']==list(KEY_EPOCHS) and 0<=time.time()-backup['observed_epoch']<120,'Windows checkpoint backup watcher is not fresh/ready')
        require(backup['free_bytes']>=8*r['checkpoint_bytes']+2*1024**3,'Windows checkpoint backup reserve insufficient')
    if mode=='trace-cache':
        r=read(root/'formal/endpoint-lock.json');require(r['status']=='F0_E120_EMA_LOCKED' and r['updates']==36480 and r['epoch']==120,'wrong F0 endpoint')

def dispatch(contract_path,mode):
    c=load_contract(contract_path);prerequisites(c,mode);root=Path(c['job_root'])
    require(not (root/mode).exists(),'phase path already exists; no automatic GPU retry')
    from .guard import snapshot,screen
    fresh=snapshot(root)
    size=read(root/'short/RESULT.json')['checkpoint_bytes'] if mode!='short' else 512*1024**2
    screen(fresh,gpu_uuid=c['gpu_uuid'],entry=True,gpu=mode!='trace-analysis',disk_bytes=40*1024**3+(122*size if mode=='formal' else 4*size))
    require(fresh['boot_id']==c['boot_id'],'boot changed at dispatch')
    token=root/(mode+'-launch-consumed.json')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',MKL_THREADING_LAYER='GNU',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
             PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    command=[c['python'],'-B',str(REPO/'tools/localization/deploy.py'),'supervise','--contract',str(Path(contract_path).resolve()),'--mode',mode]
    # Exclusive token precedes dispatch; an ambiguous SSH response never authorizes dispatch again.
    publish(token,{'task_id':TASK_ID,'mode':mode,'consumed_epoch':time.time(),'contract_sha256':c['contract_sha256'],
                   'command':command,'automatic_retry':False,'fresh_resources':fresh})
    with (root/(mode+'-dispatcher.log')).open('x') as log:
        process=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    from .guard import identity
    ident=identity(process.pid)
    result={'task_id':TASK_ID,'mode':mode,'pid':process.pid,'process_identity':ident,'dispatched_epoch':time.time(),
            'token':reference(token),'status':'DISPATCHED_NOT_ACCEPTED','command':command}
    publish(root/(mode+'-dispatch.json'),result);return result

def supervise(contract_path,mode):
    c=load_contract(contract_path);prerequisites(c,mode);root=Path(c['job_root']);out=root/mode
    token=read(root/(mode+'-launch-consumed.json'))
    require(token['contract_sha256']==c['contract_sha256'] and token['mode']==mode,'launch token mismatch')
    from .guard import supervise as guard
    size=read(root/'short/RESULT.json')['checkpoint_bytes'] if mode!='short' else 512*1024**2
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=c['gpu_uuid'] if mode!='trace-analysis' else '',
             OMP_NUM_THREADS='2' if mode!='trace-analysis' else '16',MKL_NUM_THREADS='2' if mode!='trace-analysis' else '16')
    command=[c['python'],'-B',str(REPO/'tools/localization/deploy.py'),'worker','--contract',str(Path(contract_path).resolve()),'--mode',mode]
    try:return guard(command,out,env,gpu_uuid=c['gpu_uuid'],boot_id=c['boot_id'],gpu=mode!='trace-analysis',checkpoint_bytes=size,formal=mode=='formal')
    except BaseException as error:
        if out.is_dir() and not (out/'supervisor-exit.json').exists():
            write(out/'supervisor-launch-failure.json',{'status':'STOP_NO_RETRY','error':repr(error),'epoch':time.time()})
        raise

def worker(contract_path,mode):
    c=load_contract(contract_path);root=Path(c['job_root']);out=root/mode
    require((out/'supervisor-start.json').is_file(),'worker requires the independent supervisor')
    supervisor=read(out/'supervisor-start.json');require(supervisor['pid']==os.getppid(),'worker parent is not its recorded supervisor')
    from .worker import execute
    execute(c,mode,out);return 0

def status():
    root=Path(REMOTE_ROOT);result={'task_id':TASK_ID,'observed_epoch':time.time(),'phases':{},'instance_shutdown':False,'billing_state':'UNKNOWN'}
    from .guard import identity
    for mode in PHASES:
        directory=root/mode;row={}
        for name in ('RESULT.json','progress.json','supervisor-exit.json','failure.json'):
            if (directory/name).is_file():row[name]=read(directory/name)
        dispatch_path=root/(mode+'-dispatch.json')
        if dispatch_path.is_file():
            d=read(dispatch_path);row['supervisor_current_identity']=identity(d['pid']);row['dispatch']=d
        if row:result['phases'][mode]=row
    result['key_checkpoints']=[]
    for epoch in KEY_EPOCHS:
        path=root/'formal'/f'checkpoint-epoch-{epoch:03d}.pth.complete.json'
        if path.is_file():result['key_checkpoints'].append({'epoch':epoch,'receipt':read(path),'receipt_file':reference(path)})
    result['short_checkpoints']=[]
    for name in SHORT_STATES:
        path=root/'short'/(name+'.complete.json')
        if path.is_file():result['short_checkpoints'].append({'name':name,'receipt':read(path),'receipt_file':reference(path)})
    return result

def raw_export_files(c):
    from .worker import dataset
    path=Path(c['raw_closure']['path'])
    require(path.resolve().is_relative_to(Path(c['job_root']).resolve()),'raw closure path outside task')
    require(sha(path)==c['data_sha256'],'raw closure receipt changed before export')
    closure=read(path)
    require(closure['status']=='PASS' and closure['raw_files']==4933 and closure['data_binding']==c['data'],'wrong raw closure')
    items={(r['role'],r['image_id']):r for r in closure['files']}
    require(len(items)==len(closure['files'])==4933,'raw closure duplicate/incomplete membership')
    allowed=[];identities=set()
    for role in ('train_core','development'):
        data=dataset(c,role)
        for row in data.records:
            key=(role,row['stable_image_id']);identities.add(key)
            require(key in items,'missing requested raw identity')
            file=data.annotations_path/Path(row['annotation_relative_path']).name
            expected={'path':str(file),'bytes':row['annotation_size_bytes'],'sha256':row['annotation_sha256']}
            require(items[key]['annotation']==expected,'raw export path/digest differs from frozen member')
            # Only after membership/path checks may evidence_export hash/read it.
            allowed.append(file)
    require(identities==set(items),'extra raw identity outside the authorized roles')
    return allowed

def evidence_export():
    root=Path(REMOTE_ROOT);out=root/'export'
    c=read(root/'contract.json');validate_contract(c)
    for phase in PHASES:phase_complete(c,phase)
    if (out/'RESULT.json').is_file():
        previous=read(out/'RESULT.json')
        require(previous['task_id']==TASK_ID and previous['contract_sha256']==c['contract_sha256'],'existing export belongs to another contract')
        require(previous['archive']==reference(out/'evidence.tar.gz'),'existing export changed')
        return previous
    require(not out.exists(),'incomplete export preserved; inspect before repair')
    allowed=raw_export_files(c)
    for directory in (root/'short',root/'formal',root/'trace-cache',root/'trace-analysis'):
        for p in directory.rglob('*'):
            if not p.is_file() or p.suffix=='.pth':continue
            require(not p.is_symlink(),'symlink output cannot enter evidence')
            if p.name.endswith(('.partial','.publishing','.next')):raise RuntimeError('incomplete output remains: '+str(p))
            allowed.append(p)
    for p in root.glob('*.json'):allowed.append(p)
    for p in root.glob('*-dispatcher.log'):allowed.append(p)
    for directory in root.glob('raw-preflight-*'):
        if (directory/'RESULT.json').is_file():allowed.append(directory/'RESULT.json')
    allowed.extend(p for p in (root/'materials').glob('*.json') if p.is_file())
    allowed.append(Path(c['source_manifest']))
    local=Path(c['local_checks']['path']);verify_local_checks(c);allowed.append(local)
    allowed.extend(scoped(local.parent,row['path']) for row in read(local)['evidence_files'])
    require(all(p.resolve().is_relative_to(root.resolve()) and not p.is_symlink() for p in allowed),'evidence path escapes task')
    listing=[{'path':p.relative_to(root).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(set(allowed))]
    out.mkdir(exist_ok=False)
    write(out/'manifest.json',{'task_id':TASK_ID,'contract_sha256':c['contract_sha256'],'source_sha256':c['source_sha256'],
          'data_sha256':c['data_sha256'],'files':listing,'checkpoints':'six formal key states and four short states transported separately; every epoch retained remote'})
    archive=out/'evidence.tar.gz'
    with tarfile.open(archive,'x:gz') as tar:
        tar.add(out/'manifest.json',arcname='manifest.json',recursive=False)
        for row in listing:tar.add(root/row['path'],arcname=row['path'],recursive=False)
    result={'task_id':TASK_ID,'contract_sha256':c['contract_sha256'],'source_sha256':c['source_sha256'],
            'archive':reference(archive),'files':len(listing),'finished_epoch':time.time()}
    write(out/'RESULT.json',result);return result

def main(argv=None):
    argv=sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ('prepare-recovery','launch-recovery-check','launch-resume','launch-recovery-trace',
                            'recovery-supervise','recovery-worker','recovery-status','recovery-export',
                            'remote-recovery','watch-recovery-backups','sync-recovery'):
        from .recovery import main as recovery_main
        return recovery_main(argv)
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    plan=sub.add_parser('plan');plan.add_argument('--output')
    exp=sub.add_parser('export-source');exp.add_argument('--out',required=True)
    raw=sub.add_parser('raw-preflight');raw.add_argument('--request',required=True);raw.add_argument('--out',required=True)
    fr=sub.add_parser('freeze');fr.add_argument('--request',required=True);fr.add_argument('--out',required=True)
    sr=sub.add_parser('stage-raw');sr.add_argument('--request',required=True)
    for name in ('launch','supervise','worker'):
        cmd=sub.add_parser(name);cmd.add_argument('--contract',required=True);cmd.add_argument('--mode',required=True,choices=['short','formal','trace-cache','trace-analysis'])
    sub.add_parser('status');sub.add_parser('export-evidence');sub.add_parser('environment')
    rest=sub.add_parser('storage-restore');rest.add_argument('--request',required=True)
    pr=sub.add_parser('remote-probe');pr.add_argument('--receipt',required=True)
    st=sub.add_parser('stage');st.add_argument('--export',required=True);st.add_argument('--materials-plan',required=True);st.add_argument('--local-ready',required=True);st.add_argument('--out',required=True);st.add_argument('--release',default='release-001')
    rc=sub.add_parser('remote-call');rc.add_argument('--deployment',required=True);rc.add_argument('--action',dest='remote_action',required=True,choices=['environment','stage-raw','raw-preflight','storage-restore','freeze','launch','status','export-evidence']);rc.add_argument('--receipt',required=True);rc.add_argument('--mode',choices=['short','formal','trace-cache','trace-analysis']);rc.add_argument('--request');rc.add_argument('--contract')
    for name in ('watch-backups','sync-final'):
        cmd=sub.add_parser(name);cmd.add_argument('--deployment',required=True);cmd.add_argument('--out',required=True)
    a=p.parse_args(argv)
    if a.action=='plan':
        value={'state':'PLAN_ONLY','task_id':TASK_ID,'job_root':REMOTE_ROOT,'python_historical_path':PYTHON_REMOTE,'workload':WORKLOAD,
               'required_before_gpu':['source/local-interface verification','raw4933 closure','fresh resources/environment and eligible-F0 audit','frozen contract'],
               'gpu_started':False,'autodl_actual_state':'NOT_VERIFIED_THIS_STAGE'}
        if a.output:write(a.output,value)
    elif a.action=='export-source':value=export_source(a.out)
    elif a.action=='raw-preflight':value=raw_preflight(read(a.request),a.out)
    elif a.action=='stage-raw':value=stage_raw(read(a.request))
    elif a.action=='freeze':value=freeze(a.request,a.out)
    elif a.action=='launch':value=dispatch(a.contract,a.mode)
    elif a.action=='supervise':return supervise(a.contract,a.mode)
    elif a.action=='worker':return worker(a.contract,a.mode)
    elif a.action=='status':value=status()
    elif a.action=='export-evidence':value=evidence_export()
    elif a.action=='environment':value=environment()
    elif a.action=='storage-restore':
        from .worker import storage_restore
        request=read(a.request)
        require(request['job_root']==REMOTE_ROOT and request['source_root']==str(REPO),'storage gate request root differs')
        verify_source(request)
        value=storage_restore(request['source_sha256'],Path(REMOTE_ROOT)/'storage-restore-001')
    else:
        from . import client
        if a.action=='remote-probe':value=client.probe(a.receipt)
        elif a.action=='stage':value=client.stage(a.export,a.materials_plan,a.local_ready,a.out,a.release)
        elif a.action=='remote-call':value=client.remote_call(a.deployment,a.remote_action,a.receipt,mode=a.mode,request=a.request,contract=a.contract)
        elif a.action=='watch-backups':value=client.watch(a.deployment,a.out)
        else:value=client.sync_final(a.deployment,a.out)
    print(json.dumps(value,ensure_ascii=False,allow_nan=False));return 0
