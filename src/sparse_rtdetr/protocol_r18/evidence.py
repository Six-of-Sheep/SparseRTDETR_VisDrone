"""PROJECT Windows whitelist: no science, model state, raw predictions or bulk logs."""
import hashlib,json
from pathlib import Path
from ..localization.runio import require,read,write,reference

PHASES=('cpu-monitor','gpu-recovery','formal','inference','evaluation','comparison')
ROOT_NAMES={'execution-contract.json','CPU_GATE.json','E088_CPU_AUDIT.json','EXECUTION_FREEZE.json','CHAIN_STOP.json','CHAIN_RESULT.json',
 'chain-dispatch.json','TERMINAL_AUDIT.json','SOURCE_DELTA.json','CPU_TEST_SUMMARY.json'}
SMALL_NAMES={'RESULT.json','binding.json','execution-binding.json','ORIGINAL_E088_STRICT_RESTORE.json','RESTORE_C26753.json',
 'endpoint-lock.json','LIFECYCLE_CHECK.json','failure.json','top600-metrics.json','top300-metrics.json','supervisor-start.json','supervisor-exit.json','worker-exit.json','identity.json','runtime-archive-index.json'}

def allowed(relative,size):
    p=Path(relative);parts=p.parts
    if p.is_absolute() or '..' in parts:return False
    if 'audits' in parts or 'runtime-archive' in parts or p.suffix=='.pth':return False
    if p.name in ('progress.jsonl','resources.jsonl','activity.jsonl','worker.log') or 'predictions' in p.name or 'raw-queries' in p.name:return False
    return (len(parts)==1 and p.name in ROOT_NAMES and size<=1024**2) or (len(parts)==2 and parts[0] in PHASES and (p.name in SMALL_NAMES or p.name.endswith('.pth.complete.json') or p.name.startswith('epoch-') and p.name.endswith('-complete.json')) and size<=1024**2) or (len(parts)==2 and parts[0]=='indices' and p.name in ('STATE_INDEX.json','AUDIT_INDEX.json','REMOTE_ONLY.json') and size<=16*1024**2) or (len(parts)==2 and parts[0]=='fault-excerpts' and p.suffix=='.excerpt' and size<=1024**2)

def indices(c):
    root=Path(c['job_root']);dest=root/'indices';dest.mkdir(exist_ok=True)
    states=[]
    for phase in ('gpu-recovery','formal'):
        for p in sorted((root/phase).glob('*.pth.complete.json')) if (root/phase).exists() else ():
            cp=read(p);states.append({'phase':phase,'checkpoint':cp,'sidecar':reference(p),'binding_path':str(root/phase/'binding.json'),'storage':'SERVER_ONLY'})
    write(dest/'STATE_INDEX.json',{'states':states,'state_contents_rehashed':False,'SHA_origin':'existing complete sidecars'})
    audits=[]
    runtime=Path(c['runtime_root'])/'formal';stream=runtime/'progress.jsonl'
    if stream.exists():
        with stream.open() as f:
            for line in f:
                record=json.loads(line);p=record['payload']
                if p['event']=='update_complete':audits.append({'update':p['update'],'role':p['audit_role'],**p['audit'],'input_sha256':p['input_sha256'],'event_record_sha256':record['sha256']})
    write(dest/'AUDIT_INDEX.json',{'rows':audits,'count':len(audits),'declared_bytes':sum(r['bytes'] for r in audits),'SHA_origin':'authenticated original update events; no science reread','storage':'SERVER_ONLY','source_sha256':c['source_sha256'],'contract_sha256':c['contract_sha256']})
    write(dest/'REMOTE_ONLY.json',{'policy':'complete audits/logs/state sequence/GPU acceptance states/raw predictions remain server only',
        'job_root':str(root),'runtime_root':c['runtime_root'],'old_source_reused':c['resume_parent']['manifest'],'old_state_parent':c['resume_parent']['checkpoint'],
        'raw_prediction_reference':read(root/'inference/RESULT.json').get('raw') if (root/'inference/RESULT.json').exists() else None})

def plan(c):
    root=Path(c['job_root']);files=[]
    for p in sorted(root.iterdir()):
        if p.is_file() and allowed(p.name,p.stat().st_size):files.append(reference(p))
    for phase in PHASES:
        for p in sorted((root/phase).iterdir()) if (root/phase).exists() else ():
            if p.is_file() and allowed(phase+'/'+p.name,p.stat().st_size):files.append(reference(p))
        runtime=Path(c['runtime_root'])/phase
        for name in ('supervisor-start.json','supervisor-exit.json','worker-exit.json','identity.json'):
            p=runtime/name
            if p.exists():
                require(p.is_file() and not p.is_symlink() and p.stat().st_size<=1024**2,'invalid SSD receipt')
                files.append({**reference(p),'Windows_relative':phase+'/SSD-'+name})
    for folder in ('indices','fault-excerpts'):
        for p in sorted((root/folder).iterdir()) if (root/folder).exists() else ():
            require(allowed(folder+'/'+p.name,p.stat().st_size),'unexpected evidence member');files.append(reference(p))
    total=sum(r['bytes'] for r in files)
    require(len({r['path'] for r in files})==len(files),'duplicate whitelist path')
    require(len(files)<=350 and total<=32*1024**2,'key bundle exceeds frozen local retention budget')
    return {'status':'WHITELIST_PLAN','files':files,'count':len(files),'bytes':total,'local_peak_bytes_upper_bound':2*total,
        'checkpoint_content_downloads':0,'full_science_or_log_downloads':0,'temporary_full_archive':False,'same_SHA_local_file_policy':'reference existing file; do not duplicate'}

def cpu_check(root):
    cases=[('formal/checkpoint-epoch-120.pth',400000000,False),('formal/audits/update-26753-direct.json',500000,False),
        ('formal/progress.jsonl',5000,False),('formal/runtime-archive/progress.jsonl',5000,False),('inference/raw-queries-native-R18-e120.pth',23000000,False),
        ('evaluation/top600-predictions.json',1000,False),('../outside.json',10,False),('/etc/passwd',1,False),('formal/RESULT.json',2*1024**2,False),
        ('formal/endpoint-lock.json',1000,True),('formal/checkpoint-epoch-120.pth.complete.json',1000,True),('evaluation/top600-metrics.json',10000,True),('indices/AUDIT_INDEX.json',8000000,True)]
    for name,size,value in cases:require(allowed(name,size)==value,'whitelist test differs: '+name)
    return {'status':'PASS','count':len(cases),'cases':cases,'complete_state_or_science_opened':False,'full_exporter_or_backup_imported':False}
