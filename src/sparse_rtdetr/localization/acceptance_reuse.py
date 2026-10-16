"""Explicit reuse of immutable release003 GPU acceptance, never relabelled PASS."""
import ast
from pathlib import Path
from .runio import require,read,sha,digest,reference,REMOTE_ROOT,execution_binding,original_process_exited

OLD_SOURCE='c2cf1405b0cd06e8fafb9f85348f58099ed8ed8fec57a69f05397c8b3b102842'
OLD_CONTRACT='2810725054b3f20f24c96c1a927e7432bd0e5efd5982bed1e8e7eb4f37623c5b'
OLD_ROOT=REMOTE_ROOT+'/recovery-001'
FUNCTIONS={'worker.py':('dataset','plan_epoch','check_inputs','prepare_cuda','make_engine','save_state','run_trace_cache','run_trace_analysis'),
           'recovery.py':('restored_engine','comparable','acceptance','resume')}

def runtime_equal(frozen,actual):
    require(set(frozen)=={'torch','torchvision','cuda_build','cudnn'},'frozen runtime field set changed')
    require(all(actual[k]==v for k,v in frozen.items()),'actual runtime differs from reused acceptance')
    expected={'python_version':'3.10.16','numpy':'1.26.4','scipy':'1.13.1','pillow':'12.1.0'}
    from .control import PYTHON_REMOTE
    expected['python']=PYTHON_REMOTE
    require(all(actual[k]==v for k,v in expected.items()),'complete actual runtime differs from passed acceptance')

def computation(path,name):
    tree=ast.parse(Path(path).read_text(encoding='utf-8'));node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    # Sole permitted construction difference is the output provenance label;
    # the entire numerical construction before this metadata remains exact.
    if name=='make_engine':
        allowed={"'recovery-001'", "contract.get('attempt', 'recovery-001')"}
        assignment=next(n for n in ast.walk(node) if isinstance(n,ast.Assign) and len(n.targets)==1 and ast.unparse(n.targets[0])=="binding['lineage']")
        require(isinstance(assignment.value,ast.Dict),'lineage metadata structure changed')
        index=next(i for i,k in enumerate(assignment.value.keys) if isinstance(k,ast.Constant) and k.value=='attempt')
        require(ast.unparse(assignment.value.values[index]) in allowed,'attempt metadata expression changed')
        assignment.value.values[index]=ast.Constant(value='ATTEMPT_LABEL_ONLY')
    return digest(ast.dump(node,include_attributes=False))

def compatibility(old_root,new_root):
    old_root,new_root=Path(old_root),Path(new_root)
    from .control import source_files
    allowed={'runio.py','guard.py','worker.py','recovery.py','recovery_client.py','control.py'}
    unchanged=[]
    for p in source_files(old_root):
        rel=p.relative_to(old_root)
        if rel.parts[0]=='tests' or (rel.parts[:3]==('src','sparse_rtdetr','localization') and rel.name in allowed):continue
        require(sha(p)==sha(new_root/rel),'acceptance numerical dependency changed: '+str(rel))
        unchanged.append({'path':rel.as_posix(),'sha256':sha(p)})
    functions=[]
    for file,names in FUNCTIONS.items():
        for name in names:
            a=computation(old_root/'src/sparse_rtdetr/localization'/file,name)
            b=computation(new_root/'src/sparse_rtdetr/localization'/file,name)
            require(a==b,'acceptance computation changed: '+file+':'+name)
            functions.append({'file':file,'function':name,'ast_sha256':a})
    return {'unchanged_dependencies':unchanged,'unchanged_computation':functions,
            'permitted_metadata_change':'make_engine.lineage.attempt only; full diff reviewed',
            'new_gpu_acceptance_updates':0}

def verify(c):
    from .recovery import PARENT_SHA
    from .guard import identity
    old=read(Path(OLD_ROOT)/'contract.json')
    require(old['source_sha256']==OLD_SOURCE and old['contract_sha256']==OLD_CONTRACT and digest({k:v for k,v in old.items() if k!='contract_sha256'})==OLD_CONTRACT,'reused acceptance source/contract changed')
    certificate=c['reused_acceptance']
    require(certificate['old_source_sha256']==OLD_SOURCE and certificate['old_contract_sha256']==OLD_CONTRACT and certificate['new_gpu_acceptance_updates']==0,'acceptance reuse scope changed')
    base=Path(OLD_ROOT)/'acceptance'
    require(set(certificate['receipts'])=={'RESULT.json','supervisor-exit.json','execution-binding.json','worker-exit.json','binding.json'},'reuse receipt set changed')
    require(sha(Path(REMOTE_ROOT)/'release-003/source-manifest.json')==OLD_SOURCE,'reused source manifest bytes changed')
    for row in read(Path(REMOTE_ROOT)/'release-003/source-manifest.json')['files']:
        require(sha(Path(REMOTE_ROOT)/'release-003/source'/row['path'])==row['sha256'],'reused source bytes changed')
    for name,row in certificate['receipts'].items():require(row==reference(base/name),'reused acceptance receipt bytes changed')
    ex=read(base/'supervisor-exit.json');dispatch=read(Path(OLD_ROOT)/'acceptance-dispatch.json')
    require(ex['status']=='PASS' and ex['returncode']==0 and not ex['remaining_live'] and not ex['unknown_pids'] and not ex['cleanup_errors'] and not ex['residual_after_natural_drain'],'reused acceptance not clean PASS')
    require(dispatch['contract_sha256']==OLD_CONTRACT and original_process_exited(dispatch,identity(dispatch['pid'])),'old acceptance process still active/identity unavailable')
    for name in ('RESULT.json','execution-binding.json','worker-exit.json','binding.json'):
        require(ex['completion_receipts'][name]==reference(base/name),'reused completion authentication changed')
    require(read(base/'execution-binding.json')==execution_binding(old,'acceptance'),'reused execution binding changed')
    result=read(base/'RESULT.json');b=read(base/'binding.json')
    require(result['status']=='PASS' and result['parent_sha256']==PARENT_SHA and result['actual_updates']==3 and result['direct_updates']==[913,914] and result['replayed_update']==914 and result['same_next_state_exact'] and result['same_next_window_exact'],'old fixed3 acceptance not exact')
    require(result['original_restore']['full_state_exact'] and result['original_restore']['completed_updates']==912 and result['replay_restore']['full_state_exact'] and result['replay_restore']['completed_updates']==913,'old complete restore proof missing')
    require(b['source_sha256']==OLD_SOURCE and b['contract_sha256']==OLD_CONTRACT and b['scope']=='gpu_short_window' and b['lineage']['attempt']=='recovery-001' and b['lineage']['acceptance_state'] is True,'old acceptance lineage changed')
    for key in ('environment','gpu_uuid','boot_id','config','data_sha256'):
        require(c[key]==old[key],'reuse runtime/recipe changed: '+key)
    require(certificate['checkpoints']==result['checkpoints'],'reused acceptance state set changed')
    for cp in result['checkpoints']:
        require(cp==read(cp['path']+'.complete.json') and cp['scope']=='gpu_short_window' and cp['size_bytes']==Path(cp['path']).stat().st_size,'old acceptance checkpoint/sidecar missing')
        # Full bytes SHA already returned and verified; verify anew at freeze,
        # immutable sidecars and file sizes remain strict on each phase load.
    expected=compatibility(Path(REMOTE_ROOT)/'release-003/source',c['source_root'])
    require(expected==certificate['compatibility'],'computation reuse certificate mismatch')
    simulation=c['channel_simulation'];require(reference(simulation['path'])==simulation and read(simulation['path'])['status']=='PASS','cloud no-GPU channel simulation absent/changed')
    require(read(simulation['path'])['source_sha256']==c['source_sha256'],'channel simulation belongs to other source')
    return {'status':'REUSED_RELEASE003_ACCEPTANCE','new_gpu_acceptance_updates':0}

def freeze_bytes(c):
    result=read(Path(OLD_ROOT)/'acceptance/RESULT.json')
    for cp in result['checkpoints']:require(sha(cp['path'])==cp['sha256'],'old acceptance complete bytes SHA differs')
    old_driver=read(Path(OLD_ROOT)/'acceptance/resource-entry.json')['snapshot']['gpu']['devices'][0]['driver_version']
    new_driver=c['resource_observation']['snapshot']['gpu']['devices'][0]['driver_version']
    require(old_driver==new_driver,'GPU driver changed since reused acceptance')
    return {'status':'FULL_ACCEPTANCE_BYTES_SHA_PASS','checkpoints':[reference(cp['path']) for cp in result['checkpoints']],
            'driver_version':new_driver,'new_gpu_acceptance_updates':0}
