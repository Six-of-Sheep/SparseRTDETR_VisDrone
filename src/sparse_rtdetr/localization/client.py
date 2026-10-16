"""Windows side of deploy.py: scoped SSH, backup and final SHA verification."""
import json,os,re,shutil,subprocess,tarfile,time
from pathlib import Path
from .runio import TASK_ID,REMOTE_ROOT,KEY_EPOCHS,SHORT_STATES,PHASES,SELECTION_SHA,PRETRAINED_SHA,require,read,write,sha,reference,scoped,original_process_exited,request_id

PROBE_REMOTE='/root/autodl-tmp/c2-cloud-formal-20261008-001/release-003/reused/monitoring/cloud_probe.py'
PROBE_SHA='b6a307267ceaea8b14bd31d08319e1dd7b0a5202fce254276ec4bac596ffe95e'

def helpers():
    from .control import transport
    return transport()

def probe(receipt):
    # This only executes the already inspected, byte-bound read-only probe.
    code=f'''import hashlib,json,pathlib,runpy,subprocess,time
p=pathlib.Path({PROBE_REMOTE!r})
assert hashlib.sha256(p.read_bytes()).hexdigest()=={PROBE_SHA!r},'historical probe source changed'
probe=runpy.run_path(str(p))
value=probe['snapshot'](['/root/autodl-tmp'])
value['boot_id']=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
r=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=12)
value['compute_apps']={{'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}}
roots=[p for p in pathlib.Path('/root/autodl-tmp').iterdir() if p.is_dir() and any(k in p.name.lower() for k in ('localization','distribution','f0'))]
candidates=[]
for root in roots:
 for item in root.rglob('endpoint-lock.json'):
  if any(k in str(item).lower() for k in ('holdout','confirmatory','test-dev','test_dev')):continue
  candidates.append({{'path':str(item),'bytes':item.stat().st_size,'metadata':json.loads(item.read_text())}})
value['reusable_f0_audit']={{'status':'REVIEW_CANDIDATES' if candidates else 'NO_ELIGIBLE_F0_FOUND','scope':[str(p) for p in roots],
 'search_root':'/root/autodl-tmp','metadata':'endpoint-lock.json in F0/localization/distribution experiment roots','candidates':candidates,
 'checkpoint_deserialized':False,'all_machines_searched':False,'observed_epoch':time.time()}}
value['raw_directory_candidates']=[{{'path':str(p),'exists':p.is_dir()}} for p in [
 pathlib.Path('/root/autodl-tmp/c2-autodl-20261008-p1/bundle/data/train_core/annotations'),
 pathlib.Path('/root/autodl-tmp/c2-cloud-formal-20261008-001/materials/data/development/annotations'),
 pathlib.Path({REMOTE_ROOT!r})/'materials/train_core/annotations',pathlib.Path({REMOTE_ROOT!r})/'materials/development/annotations']]
value['cloud_contact']='READ_ONLY';value['model_calls']=0
print(json.dumps(value,allow_nan=False))
'''
    return json.loads(helpers().remote_python(code,receipt,60))

def archive_members(tar):
    members=tar.getmembers();names=[m.name for m in members]
    require(len(names)==len(set(names)),'duplicate archive members')
    for m in members:
        path=Path(m.name)
        require(m.isfile() and not path.is_absolute() and '..' not in path.parts and ':' not in m.name and '\\' not in m.name,'unsafe evidence/source archive')
    return members,names

def safe_extract(archive,out,*,expected_members=None):
    out=Path(out)
    with tarfile.open(archive,'r:gz') as tar:
        members,names=archive_members(tar)
        if expected_members is not None:require(set(names)==set(expected_members),'archive member set mismatch')
        out.mkdir(parents=True,exist_ok=False)
        for m in members:
            path=Path(m.name)
            destination=out/path
            require(destination.resolve().is_relative_to(out.resolve()),'archive traversal')
            destination.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(m) as source,destination.open('xb') as target:shutil.copyfileobj(source,target)
    return names

def upload(archive,receipt_dir):
    ops=helpers();receipt_dir=Path(receipt_dir);receipt_dir.mkdir(parents=True,exist_ok=False)
    record=reference(archive);remote=REMOTE_ROOT+'/incoming-'+record['sha256'][:16]+'-'+request_id()+'.tar.gz'
    ops.remote_python(f"import pathlib\np=pathlib.Path({REMOTE_ROOT!r});p.mkdir(exist_ok=True)\nassert not pathlib.Path({remote!r}).exists()\nprint('READY')\n",receipt_dir/'prepare.json',30)
    native=Path(ops.native_ssh()).with_name('scp.exe');started=time.time()
    options=[x for x in ops.SSH_OPTIONS if x!='-T']
    with (receipt_dir/'scp.stdout').open('xb') as stdout,(receipt_dir/'scp.stderr').open('xb') as stderr:
        child=subprocess.Popen([str(native),*options,str(archive),'autodl-c2:'+remote],stdout=stdout,stderr=stderr,stdin=subprocess.DEVNULL)
        timed_out=False
        try:rc=child.wait(timeout=300)
        except subprocess.TimeoutExpired:timed_out=True;child.kill();rc=child.wait(timeout=10)
    write(receipt_dir/'scp-exit.json',{'pid':child.pid,'returncode':rc,'timed_out':timed_out,'elapsed_seconds':time.time()-started,'child_exited':True,'automatic_retry':False})
    require(not timed_out and rc==0,'source/material upload failed; preserve incoming, no automatic replay')
    code=f"import pathlib,hashlib,json\np=pathlib.Path({remote!r});h=hashlib.sha256()\nwith p.open('rb') as f:\n for b in iter(lambda:f.read(4194304),b''):h.update(b)\nassert h.hexdigest()=={record['sha256']!r} and p.stat().st_size=={record['bytes']}\nprint(json.dumps({{'path':str(p),'sha256':h.hexdigest(),'bytes':p.stat().st_size}}))\n"
    return json.loads(ops.remote_python(code,receipt_dir/'remote-sha.json',120))

def stage(export,materials_plan,local_ready,out,release='release-001'):
    from .material_paths import metadata_path,MANIFESTS
    from .local_ready import verify
    require(re.fullmatch(r'release-\d{3}',release) is not None,'invalid release name')
    export=Path(export);spec=read(export/'EXPORT.json')
    require(spec['task_id']==TASK_ID,'wrong source export task')
    require(sha(export/'source-manifest.json')==spec['source_sha256'] and sha(export/'source.tar.gz')==spec['archive']['sha256'],'local source export changed')
    ready=verify(local_ready,spec['source_sha256']);local_ready=Path(local_ready)
    plan=read(materials_plan);bindings=[]
    for role in ('train_core','development'):
        value=plan['roles'][role];path=metadata_path(role,value['manifest_path'],value['manifest_sha256'])
        require(value['members']==MANIFESTS[role][1],'role manifest count changed')
        bindings.append((role+'_manifest.json',path))
    selected=metadata_path('selection',Path(materials_plan).parent/'trace-selection.json',SELECTION_SHA)
    for role,(_,path) in zip(('train_core','development'),bindings):require(sha(path)==MANIFESTS[role][0],'role manifest changed')
    require(sha(selected)==SELECTION_SHA,'trace selection changed')
    weight=Path(plan['pretrained']['path']);require(sha(weight)==PRETRAINED_SHA,'pretrained changed')
    out=Path(out);out.mkdir(parents=True,exist_ok=False);ops=helpers();package=out/'materials.tar.gz'
    bindings.extend([('trace-selection.json',selected),('pretrained/ResNet18_vd_pretrained_from_paddle.pth',weight)])
    material_rows=[{'path':name,**{k:v for k,v in reference(path).items() if k!='path'}} for name,path in bindings]
    write(out/'materials-manifest.json',material_rows)
    with tarfile.open(package,'x:gz') as tar:
        tar.add(out/'materials-manifest.json',arcname='materials-manifest.json',recursive=False)
        for name,path in bindings:tar.add(path,arcname=name,recursive=False)
    checks_package=out/'local-checks.tar.gz'
    with tarfile.open(checks_package,'x:gz') as tar:
        tar.add(local_ready,arcname='LOCAL_READY.json',recursive=False)
        for row in ready['evidence_files']:tar.add(scoped(local_ready.parent,row['path']),arcname=row['path'],recursive=False)
    source_names=['source-manifest.json']+['source/'+r['path'] for r in read(export/'source-manifest.json')['files']]
    material_names=['materials-manifest.json']+[name for name,_ in bindings]
    checks_names=['LOCAL_READY.json']+[r['path'] for r in ready['evidence_files']]
    source_remote=upload(export/'source.tar.gz',out/'source-upload');material_remote=upload(package,out/'materials-upload')
    checks_remote=upload(checks_package,out/'checks-upload')
    code=f'''import hashlib,json,pathlib,tarfile
def extract(archive,target,expected,reuse=False):
 target=pathlib.Path(target)
 with tarfile.open(archive,'r:gz') as tar:
  members=tar.getmembers();assert len(members)==len({{m.name for m in members}})
  assert {{m.name for m in members}}==set(expected),'archive membership changed'
  for m in members:
   name=pathlib.PurePosixPath(m.name);assert m.isfile() and not name.is_absolute() and '..' not in name.parts and ':' not in m.name and chr(92) not in m.name
  if target.exists():
   assert reuse and target.is_dir() and not target.is_symlink(),'existing/incomplete target is preserved'
   return target
  target.mkdir(exist_ok=False)
  for m in members:
   name=pathlib.PurePosixPath(m.name)
   p=target/name;assert p.resolve().is_relative_to(target.resolve());p.parent.mkdir(parents=True,exist_ok=True)
   with tar.extractfile(m) as src,p.open('xb') as dst:
    while True:
     data=src.read(4194304)
     if not data:break
     dst.write(data)
 return target
source=extract({source_remote['path']!r},{(REMOTE_ROOT+'/'+release)!r},{source_names!r})
assert hashlib.sha256((source/'source-manifest.json').read_bytes()).hexdigest()=={spec['source_sha256']!r}
for row in json.loads((source/'source-manifest.json').read_text())['files']:
 p=source/'source'/row['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'] and p.stat().st_size==row['bytes']
material=extract({material_remote['path']!r},{(REMOTE_ROOT+'/materials')!r},{material_names!r},reuse=True)
assert hashlib.sha256((material/'materials-manifest.json').read_bytes()).hexdigest()=={sha(out/'materials-manifest.json')!r},'staged material manifest differs'
for row in json.loads((material/'materials-manifest.json').read_text()):
 p=material/row['path'];assert p.resolve().is_relative_to(material.resolve()) and not any(x.is_symlink() for x in (p,*p.parents))
 assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'] and p.stat().st_size==row['bytes']
checks=extract({checks_remote['path']!r},str(source/'local-checks'),{checks_names!r})
assert hashlib.sha256((checks/'LOCAL_READY.json').read_bytes()).hexdigest()=={sha(local_ready)!r}
for row in json.loads((checks/'LOCAL_READY.json').read_text())['evidence_files']:
 p=checks/row['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'] and p.stat().st_size==row['bytes']
print(json.dumps({{'status':'STAGED_SHA_PASS','task_id':{TASK_ID!r},'release_root':str(source),'source_root':str(source/'source'),'source_sha256':{spec['source_sha256']!r},'material_root':str(material),'local_checks':{{'path':str(checks/'LOCAL_READY.json'),'sha256':{sha(local_ready)!r},'source_sha256':{spec['source_sha256']!r},'status':'LOCAL_READY'}},'model_calls':0}}))
'''
    result=json.loads(ops.remote_python(code,out/'extract.json',180));write(out/'DEPLOYED.json',result);return result

def remote_call(deployment,action,receipt,*,mode=None,request=None,contract=None):
    from .control import PYTHON_REMOTE
    dep=read(deployment);require(dep['task_id']==TASK_ID and dep['status']=='STAGED_SHA_PASS','invalid deployment reference')
    require(action in ('environment','stage-raw','raw-preflight','storage-restore','freeze','launch','status','export-evidence'),'remote action out of scope')
    deploy=dep['source_root']+'/tools/localization/deploy.py';args=[PYTHON_REMOTE,'-B',deploy,action]
    remote_request=None
    code='import json,os,pathlib,subprocess\n'
    if request:
        value=read(request);remote_request=REMOTE_ROOT+'/requests/'+request_id()+'.json'
        code+=f"p=pathlib.Path({remote_request!r});p.parent.mkdir(exist_ok=True)\nwith p.open('x') as f:json.dump({value!r},f,allow_nan=False)\n"
        args+=['--request',remote_request]
    if action=='launch':
        require(mode in ('short','formal','trace-cache','trace-analysis'),'missing explicit phase')
        args+=['--mode',mode,'--contract',contract or REMOTE_ROOT+'/contract.json']
    if action=='freeze':args+=['--out',contract or REMOTE_ROOT+'/contract.json']
    if action=='raw-preflight':args+=['--out',REMOTE_ROOT+'/raw-preflight-'+request_id()]
    process_timeout=300 if action=='storage-restore' else 120
    code+=f"env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MKL_THREADING_LAYER='GNU',PYTHONDONTWRITEBYTECODE='1')\nr=subprocess.run({args!r},env=env,stdin=subprocess.DEVNULL,timeout={process_timeout})\nraise SystemExit(r.returncode)\n"
    return json.loads(helpers().remote_python(code,receipt,process_timeout+30))

def download(remote,expected_sha,dest,receipts,expected_size=None):
    import importlib
    helpers().native_ssh();module=importlib.import_module('parallel_transfer')
    return module.download(remote,expected_sha,dest,receipts,expected_size=expected_size)

def sync_keys(status_value,out):
    out=Path(out);receipts=[]
    for item in status_value['key_checkpoints']:
        epoch=item['epoch'];cp=item['receipt'];require(epoch in KEY_EPOCHS,'unexpected checkpoint epoch')
        expected=REMOTE_ROOT+f'/formal/checkpoint-epoch-{epoch:03d}.pth'
        require(cp['path']==expected and cp['scope']=='formal_F0_seed0_e120' and cp['completed_updates']==epoch*304,'checkpoint backup scope mismatch')
        target=out/'checkpoints'/Path(expected).name
        receipts.append(download(expected,cp['sha256'],target,out/'transfer'/f'e{epoch:03d}',cp['size_bytes']))
        side=item['receipt_file'];require(side['path']==expected+'.complete.json','checkpoint sidecar path mismatch')
        side_target=target.with_suffix('.pth.complete.json')
        receipts.append(download(side['path'],side['sha256'],side_target,out/'transfer'/f'e{epoch:03d}-sidecar',side['bytes']))
        require(read(side_target)==cp,'downloaded checkpoint sidecar differs from admitted receipt')
    return receipts

def sync_short(status_value,out):
    items=status_value['short_checkpoints'];names=[x['name'] for x in items]
    require(len(names)==len(SHORT_STATES) and set(names)==set(SHORT_STATES),'short checkpoint evidence incomplete/duplicate')
    out=Path(out);receipts=[]
    for item in items:
        name=item['name'];cp=item['receipt'];side=item['receipt_file'];expected=REMOTE_ROOT+'/short/'+name
        require(cp['path']==expected and cp['scope']=='gpu_short_window' and cp['completed_updates']==SHORT_STATES[name],'short checkpoint scope/clock mismatch')
        require(side['path']==expected+'.complete.json','short checkpoint sidecar path mismatch')
        target=out/'short-checkpoints'/name
        receipts.append(download(expected,cp['sha256'],target,out/'transfer'/('short-'+name),cp['size_bytes']))
        side_target=target.with_suffix('.pth.complete.json')
        receipts.append(download(side['path'],side['sha256'],side_target,out/'transfer'/('short-'+name+'-sidecar'),side['bytes']))
        require(read(side_target)==cp,'short checkpoint sidecar differs from admitted receipt')
    return receipts

def watch(deployment,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);dep=read(deployment);ops=helpers();started=time.time();last_ready=0;number=0;done=set()
    write(out/'watcher-start.json',{'task_id':TASK_ID,'pid':os.getpid(),'started_epoch':started,'epochs':list(KEY_EPOCHS),'training_control':False})
    try:
        while True:
            number+=1
            if time.time()-last_ready>=30:
                ready={'task_id':TASK_ID,'source_sha256':dep['source_sha256'],'epochs':list(KEY_EPOCHS),'observed_epoch':time.time(),
                       'free_bytes':shutil.disk_usage(out).free,'windows_pid':os.getpid(),'training_control':False}
                code=f"import json,os,pathlib\np=pathlib.Path({(REMOTE_ROOT+'/windows-backup-ready.json')!r});tmp=p.with_suffix('.next')\nwith tmp.open('w') as f:json.dump({ready!r},f);f.flush();os.fsync(f.fileno())\nos.replace(tmp,p)\nprint(json.dumps({{'ready':True}}))\n"
                ops.remote_python(code,out/f'ready-{number:06d}.json',30);last_ready=time.time()
            s=remote_call(deployment,'status',out/f'status-{number:06d}.json')
            new=[x for x in s['key_checkpoints'] if x['epoch'] not in done]
            receipts=sync_keys({**s,'key_checkpoints':new},out)
            if receipts:write(out/f'backup-{number:06d}.json',{'files':receipts})
            done.update(x['epoch'] for x in new)
            phase=s['phases'].get('formal',{})
            require(bool(phase) or time.time()-started<300,'formal launch not observed within backup-watcher setup window')
            if 'failure.json' in phase or phase.get('supervisor-exit.json',{}).get('status')=='STOP':raise RuntimeError('formal STOP; watcher cannot restart it')
            if done==set(KEY_EPOCHS):break
            time.sleep(15)
        result={'status':'PASS','epochs':sorted(done),'training_control':False,'elapsed_seconds':time.time()-started}
    except BaseException as error:
        result={'status':'STOP_PRESERVE','epochs':sorted(done),'error':repr(error),'training_control':False,'elapsed_seconds':time.time()-started}
        write(out/'watcher-result.json',result);raise
    write(out/'watcher-result.json',result);return result

def sync_final(deployment,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    require(not (out/'VERIFICATION.json').exists(),'delivery already completed; inspect existing SHA verification')
    attempt=out/'requests'/request_id();attempt.mkdir(parents=True)
    s=remote_call(deployment,'status',attempt/'status.json')
    for mode in PHASES:
        phase=s['phases'][mode]
        require(phase['supervisor-exit.json']['status']=='PASS','phase not closed: '+mode)
        require(original_process_exited(phase['dispatch'],phase['supervisor_current_identity']),'supervisor still alive at final sync: '+mode)
    epochs=[x['epoch'] for x in s['key_checkpoints']]
    require(len(epochs)==len(KEY_EPOCHS) and set(epochs)==set(KEY_EPOCHS),'final delivery requires each key checkpoint exactly once')
    names=[x['name'] for x in s['short_checkpoints']]
    require(len(names)==len(SHORT_STATES) and set(names)==set(SHORT_STATES),'final delivery requires four short checkpoint states')
    keys=sync_keys(s,out);short_keys=sync_short(s,out)
    exported=remote_call(deployment,'export-evidence',attempt/'export.json');remote=exported['archive']
    require(exported['task_id']==TASK_ID and exported['source_sha256']==read(deployment)['source_sha256'],'remote export source/task mismatch')
    archive=out/'evidence.tar.gz';receipt=download(remote['path'],remote['sha256'],archive,out/'archive-transfer',remote['bytes'])
    require(sha(archive)==remote['sha256'] and archive.stat().st_size==remote['bytes'],'returned archive changed')
    with tarfile.open(archive,'r:gz') as tar:
        _,names=archive_members(tar);require('manifest.json' in names,'archive manifest absent')
        raw_manifest=tar.extractfile('manifest.json').read();manifest=json.loads(raw_manifest)
    expected_rows={r['path']:r for r in manifest['files']};expected=set(expected_rows)|{'manifest.json'}
    require(len(expected_rows)==len(manifest['files'])==exported['files'] and set(names)==expected,'authenticated archive member set mismatch')
    require(manifest['task_id']==TASK_ID and manifest['source_sha256']==exported['source_sha256'] and manifest['contract_sha256']==exported['contract_sha256'],'archive task/source/contract mismatch')
    evidence=out/'evidence'
    if evidence.exists():
        present=set()
        for file in evidence.rglob('*'):
            require(not file.is_symlink(),'symlink in existing evidence')
            if not file.is_file():continue
            name=file.relative_to(evidence).as_posix();present.add(name)
            require(name in expected,'unrecognized existing evidence; preserve it')
            if name=='manifest.json':require(file.read_bytes()==raw_manifest,'existing manifest differs from authenticated archive')
            else:
                row=expected_rows[name]
                require(file.stat().st_size==row['bytes'] and sha(file)==row['sha256'],'existing evidence differs from authenticated archive')
        if present!=expected:evidence=attempt/'evidence' # Preserve interrupted extraction and create a complete copy.
    if not evidence.exists():safe_extract(archive,evidence,expected_members=expected)
    require((evidence/'manifest.json').read_bytes()==raw_manifest,'extracted manifest differs from authenticated archive')
    for row in manifest['files']:
        path=scoped(evidence,row['path']);require(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Windows returned evidence SHA mismatch')
    result={'status':'WINDOWS_ALL_RETURNED_SHA_PASS','archive':receipt,'evidence_files':len(manifest['files']),
            'evidence_root':str(evidence),'contract_sha256':exported['contract_sha256'],'source_sha256':exported['source_sha256'],
            'checkpoint_files':keys,'short_checkpoint_files':short_keys,'finished_epoch':time.time(),'instance_shutdown':False,'billing_state':'UNKNOWN'}
    write(out/'VERIFICATION.json',result);return result
