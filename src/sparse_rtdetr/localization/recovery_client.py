"""Windows observer/backups/SHA return for the separately authorized recovery."""
import json,os,shutil,tarfile,time
from pathlib import Path
from . import client
from .recovery import ROOT,PHASES,PARENT_SHA,PARENT_SOURCE,PARENT_PATH,WORKLOAD,ACCEPTANCE_ROOT
from .runio import (TASK_ID,REMOTE_ROOT,require,read,write,sha,reference,request_id,original_process_exited,scoped)

def remote(deployment,action,receipt,*,request=None):
    from .control import PYTHON_REMOTE
    dep=read(deployment);require(dep['task_id']==TASK_ID and dep['status']=='STAGED_SHA_PASS','wrong recovery deployment')
    commands={'prepare':'prepare-recovery','acceptance':'launch-recovery-check','resume':'launch-resume',
              'trace-cache':'launch-recovery-trace','trace-analysis':'launch-recovery-trace','status':'recovery-status','export':'recovery-export'}
    require(action in commands,'recovery action out of scope')
    require(action!='acceptance','existing3-update acceptance reused; new GPU acceptance prohibited')
    args=[PYTHON_REMOTE,'-B',dep['source_root']+'/tools/localization/deploy.py',commands[action]]
    code='import json,os,pathlib,subprocess\n'
    if action=='prepare':
        require(request is not None,'prepare requires frozen-source request')
        value=read(request);path=REMOTE_ROOT+'/requests/recovery-'+request_id()+'.json'
        code+=f"p=pathlib.Path({path!r});p.parent.mkdir(exist_ok=True)\nwith p.open('x') as f:json.dump({value!r},f,allow_nan=False)\n"
        args+=['--request',path]
    if action in ('acceptance','resume','trace-cache','trace-analysis'):args+=['--contract',ROOT+'/contract.json']
    if action in ('trace-cache','trace-analysis'):args+=['--mode',action]
    timeout=1800 if action=='export' else 180
    code+=f"env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MKL_THREADING_LAYER='GNU',PYTHONDONTWRITEBYTECODE='1')\nr=subprocess.run({args!r},env=env,stdin=subprocess.DEVNULL,timeout={timeout})\nraise SystemExit(r.returncode)\n"
    text=client.helpers().remote_python(code,receipt,timeout+30)
    # Export writes bounded progress lines before its final immutable receipt.
    return json.loads(text.strip().splitlines()[-1])

def sync_keys(s,out,*,final=False):
    out=Path(out);items=s['key_checkpoints'];epochs=[x['epoch'] for x in items]
    expected={0,1,3,30,60,90,120}
    require(len(epochs)==len(set(epochs)) and set(epochs)<=expected,'duplicate/unexpected recovery key epoch')
    if final:require(set(epochs)==expected,'missing recovery key checkpoint')
    result=[]
    for item in items:
        epoch=item['epoch'];cp=item['receipt'];side=item['receipt_file']
        base=REMOTE_ROOT+'/formal' if epoch<=3 else ROOT+'/formal';path=base+f'/checkpoint-epoch-{epoch:03d}.pth'
        require(cp['path']==path and cp['scope']=='formal_F0_seed0_e120' and cp['completed_updates']==epoch*304,'wrong recovery checkpoint scope/path/clock')
        require(side['path']==path+'.complete.json','wrong checkpoint sidecar path')
        target=out/'checkpoints'/Path(path).name
        result.append(client.download(path,cp['sha256'],target,out/'transfer'/f'e{epoch:03d}',cp['size_bytes']))
        side_target=target.with_suffix('.pth.complete.json')
        result.append(client.download(side['path'],side['sha256'],side_target,out/'transfer'/f'e{epoch:03d}-sidecar',side['bytes']))
        require(read(side_target)==cp,'returned recovery sidecar differs')
        if epoch==3:require(cp['sha256']==PARENT_SHA,'returned parente003 changed')
    return result

def sync_acceptance(s,out):
    items=s['acceptance_checkpoints'];names=[x['name'] for x in items]
    expected={'C913.pth':913,'C914-direct.pth':914,'C914-replay.pth':914}
    require(len(names)==3 and set(names)==set(expected),'acceptance states missing/duplicate')
    result=[];out=Path(out)
    for item in items:
        name=item['name'];cp=item['receipt'];side=item['receipt_file'];path=ACCEPTANCE_ROOT+'/'+name
        require(cp['path']==path and cp['scope']=='gpu_short_window' and cp['completed_updates']==expected[name],'acceptance scope/path/clock invalid')
        require(side['path']==path+'.complete.json','acceptance sidecar path changed')
        target=out/'acceptance-checkpoints'/name
        result.append(client.download(path,cp['sha256'],target,out/'transfer'/('acceptance-'+name),cp['size_bytes']))
        result.append(client.download(side['path'],side['sha256'],target.with_suffix('.pth.complete.json'),out/'transfer'/('acceptance-'+name+'-sidecar'),side['bytes']))
        require(read(target.with_suffix('.pth.complete.json'))==cp,'returned acceptance sidecar differs')
    return result

def watch(deployment,out):
    dep=read(deployment);out=Path(out);out.mkdir(exist_ok=False,parents=True);started=time.time();last_ready=0;n=0;done=set()
    write(out/'watcher-start.json',{'task_id':TASK_ID,'pid':os.getpid(),'started_epoch':started,'epochs':[0,1,3,30,60,90,120],'training_control':False})
    try:
        while True:
            n+=1
            if time.time()-last_ready>=30:
                ready={'task_id':TASK_ID,'source_sha256':dep['source_sha256'],'contract_sha256':dep['recovery_contract_sha256'],
                       'observed_epoch':time.time(),'free_bytes':shutil.disk_usage(out).free,'windows_pid':os.getpid(),'training_control':False}
                code=f"import json,pathlib,os\np=pathlib.Path({(ROOT+'/windows-backup-ready.json')!r});tmp=p.with_suffix('.next')\nwith tmp.open('w') as f:json.dump({ready!r},f);f.flush();os.fsync(f.fileno())\nos.replace(tmp,p)\nprint(json.dumps({{'ready':True}}))\n"
                client.helpers().remote_python(code,out/f'ready-{n:06d}.json',30);last_ready=time.time()
            s=remote(deployment,'status',out/f'status-{n:06d}.json')
            items=[x for x in s['key_checkpoints'] if x['epoch'] not in done]
            receipts=sync_keys({**s,'key_checkpoints':items},out)
            if receipts:write(out/f'backup-{n:06d}.json',{'files':receipts})
            done.update(x['epoch'] for x in items)
            p=s['phases'].get('formal',{})
            require(bool(p) or time.time()-started<300,'recovery formal not observed in setup window')
            require('failure.json' not in p and 'mirror-error.json' not in p and p.get('supervisor-exit.json',{}).get('status')!='STOP','recovery formal STOP; no retry')
            if done=={0,1,3,30,60,90,120}:break
            time.sleep(15)
        result={'status':'PASS','epochs':sorted(done),'elapsed_seconds':time.time()-started,'training_control':False}
    except BaseException as error:
        result={'status':'STOP_PRESERVE','epochs':sorted(done),'elapsed_seconds':time.time()-started,'error':repr(error),'training_control':False}
        write(out/'watcher-result.json',result);raise
    write(out/'watcher-result.json',result);return result

def sync(deployment,out):
    out=Path(out);out.mkdir(exist_ok=True,parents=True);require(not (out/'VERIFICATION.json').exists(),'recovery delivery already closed')
    require(shutil.disk_usage(out).free>=12*1024**3,'Windows final-return reserve below12GiB')
    attempt=out/'requests'/request_id();attempt.mkdir(parents=True)
    s=remote(deployment,'status',attempt/'status.json')
    for mode in PHASES:
        p=s['phases'][mode];require(p['supervisor-exit.json']['status']=='PASS' and original_process_exited(p['dispatch'],p['supervisor_current_identity']),'recovery phase not fully exited/PASS')
    key=sync_keys(s,out,final=True);acceptance=sync_acceptance(s,out)
    exported=remote(deployment,'export',attempt/'export.json');archive=out/'evidence.tar.gz';r=exported['archive']
    require(exported['task_id']==TASK_ID and exported['source_sha256']==read(deployment)['source_sha256'] and exported['contract_sha256']==read(deployment)['recovery_contract_sha256'],'export source/task/contract mismatch')
    transfer=client.download(r['path'],r['sha256'],archive,out/'archive-transfer',r['bytes'])
    require(sha(archive)==r['sha256'] and archive.stat().st_size==r['bytes'],'returned archive changed')
    with tarfile.open(archive,'r:gz') as tar:
        _,names=client.archive_members(tar);require('manifest.json' in names,'recovery archive lacks manifest')
        raw=tar.extractfile('manifest.json').read();manifest=json.loads(raw)
    rows={x['path']:x for x in manifest['files']};expected=set(rows)|{'manifest.json'}
    require(len(rows)==len(manifest['files'])==exported['files'] and set(names)==expected,'recovery archive member set differs')
    require(manifest['source_sha256']==exported['source_sha256'] and manifest['contract_sha256']==exported['contract_sha256'],'manifest bound to wrong recovery')
    verify_epoch_lineage(manifest,s)
    evidence=out/'evidence'
    if evidence.exists():
        present=set()
        for p in evidence.rglob('*'):
            require(not p.is_symlink(),'existing recovery evidence symlink')
            if not p.is_file():continue
            name=p.relative_to(evidence).as_posix();present.add(name);require(name in expected,'foreign existing recovery evidence')
            if name=='manifest.json':require(p.read_bytes()==raw,'existing manifest differs from authenticated recovery archive')
            else:require(p.stat().st_size==rows[name]['bytes'] and sha(p)==rows[name]['sha256'],'existing recovery evidence changed')
        if present!=expected:evidence=attempt/'evidence'
    if not evidence.exists():client.safe_extract(archive,evidence,expected_members=expected)
    require((evidence/'manifest.json').read_bytes()==raw,'extracted recovery manifest differs')
    for row in rows.values():
        p=scoped(evidence,row['path']);require(p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'recovery evidence Windows SHA mismatch')
    # The authenticated manifest must include every fixed cache/report; remote
    # absolute paths are mapped explicitly to the returned evidence root.
    prefix='recovery-002/'
    collection=read(evidence/(prefix+'trace-cache/RESULT.json'));analysis=read(evidence/(prefix+'trace-analysis/RESULT.json'))
    require(len(collection['files'])==len(analysis['image_reports'])==64,'fixed64 return collection missing')
    for i,(cache,report) in enumerate(zip(collection['files'],analysis['image_reports'])):
        for kind,item,name in [('trace-cache',cache,f'trace-{i:03d}.npz'),('trace-analysis',report,f'image-{i:03d}.json')]:
            rel=prefix+kind+'/'+name;require(rel in rows and item['path']==ROOT+'/'+kind+'/'+name,'fixed64 member path missing')
            require(rows[rel]['sha256']==item['sha256'] and rows[rel]['bytes']==item['bytes'],'returned fixed64 SHA/size differs from completed collection')
    result={'status':'RECOVERY_ALL_WINDOWS_SHA_PASS','archive':transfer,'evidence_root':str(evidence),'evidence_files':len(rows),
            'checkpoint_files':key,'acceptance_checkpoint_files':acceptance,'logical_epoch_states':121,
            'source_sha256':exported['source_sha256'],'contract_sha256':exported['contract_sha256'],'finished_epoch':time.time(),
            'instance_shutdown':False,'actual_bill_cny':None}
    write(out/'VERIFICATION.json',result);return result

def verify_epoch_lineage(manifest,status_value):
    epochs=manifest['epoch_lineage'];require([x['epoch'] for x in epochs]==list(range(121)),'logical epoch lineage incomplete/duplicated')
    by_epoch={r['epoch']:r['checkpoint'] for r in epochs}
    for epoch,cp in by_epoch.items():
        base=REMOTE_ROOT+'/formal' if epoch<=3 else ROOT+'/formal'
        require(cp['path']==base+f'/checkpoint-epoch-{epoch:03d}.pth' and cp['completed_updates']==epoch*304 and cp['scope']=='formal_F0_seed0_e120','manifest epoch scope/path/clock changed')
    for row in status_value['key_checkpoints']:
        require(by_epoch[row['epoch']]==row['receipt'],'Windows returned key differs from authenticated epoch lineage')
