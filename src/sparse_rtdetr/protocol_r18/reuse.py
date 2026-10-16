"""Strict cross-release evidence binding. Never load acceptance state into a model."""
from pathlib import Path
from ..localization import channel,guard
from ..localization.runio import require,read,reference,sha,digest,original_process_exited
from . import transport
OLD_SOURCE_SHA='da8ceab59a83619750b81578d9ab29b6d6222f0805ab73c14403b77e1ec477d7'
OLD_CONTRACT_SHA='03ecf4b106482c183d79d1df3c639d36f61c618099a695e155b2a0c6ebe3ced6'
OLD_ROOT=Path('/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace/experiments/localization-distribution/rtdetrv2-r18-common-protocol-seed0-e120-20261013-001')
WIRE_FILES={'src/sparse_rtdetr/protocol_r18/control.py','src/sparse_rtdetr/protocol_r18/lifecycle.py','src/sparse_rtdetr/protocol_r18/reuse.py'}

def verify(c):
    transport.install();binding=c['acceptance_reuse']
    require(binding['old_job_root']==str(OLD_ROOT) and binding['old_source_sha256']==OLD_SOURCE_SHA and binding['old_contract_sha256']==OLD_CONTRACT_SHA,'reused acceptance identity differs')
    old=read(OLD_ROOT/'execution-contract-001.json')
    require(old['contract_sha256']==OLD_CONTRACT_SHA and digest({k:v for k,v in old.items() if k!='contract_sha256'})==OLD_CONTRACT_SHA,'old contract bytes/identity changed')
    require(sha(old['source_manifest'])==OLD_SOURCE_SHA and sha(c['source_manifest'])==c['source_sha256'],'old/new source manifests differ')
    a=read(old['source_manifest'])['files'];b=read(c['source_manifest'])['files'];by_a={r['path']:r for r in a};by_b={r['path']:r for r in b}
    require(set(by_a)<=set(by_b) and set(by_b)-set(by_a)=={'src/sparse_rtdetr/protocol_r18/reuse.py'},'source members added/removed beyond protocol wire')
    changed={p for p in by_b if p not in by_a or by_a[p]['sha256']!=by_b[p]['sha256']}
    require(changed==WIRE_FILES,'non-wire source changed or repair footprint differs')
    for rows,root in ((a,Path(old['source_root'])),(b,Path(c['source_root']))):
        for r in rows:
            p=root/r['path'];require(p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],'frozen source file bytes changed: '+r['path'])
    require(c['config']==old['config'] and c['environment']==old['environment'] and c['gpu_uuid']==old['gpu_uuid'],'model recipe/runtime/GPU changed')
    require(c['data']==old['data'] and c['fixed_ids']==old['fixed_ids'] and c['pretrained']==old['pretrained'] and c['evaluation_protocol_sha256']==old['evaluation_protocol_sha256'],'input/pretrained/readout semantics changed')
    for ref in binding['evidence']:require(reference(ref['path'])==ref,'old acceptance evidence changed: '+ref['path'])
    runtime=Path(old['runtime_root'])/'gpu-acceptance';out=OLD_ROOT/'gpu-acceptance';d=read(OLD_ROOT/'gpu-acceptance-dispatch.json');ex=read(runtime/'supervisor-exit.json')
    require(original_process_exited(d,guard.identity(d['pid'])) and ex['status']=='PASS' and ex['returncode']==0 and not any(ex[k] for k in ('remaining_live','unknown_pids','cleanup_errors')),'old GPU acceptance not clean closed')
    identity=read(runtime/'identity.json')
    require(identity['task_id']==old['task_id'] and identity['source_sha256']==OLD_SOURCE_SHA and identity['contract_sha256']==OLD_CONTRACT_SHA and identity['mode']=='gpu-acceptance' and identity['supervisor_pid']==d['pid'],'old runtime identity changed')
    channel.verify_archive(runtime,out,identity)
    require(read(out/'execution-binding.json')==identity and read(out/'LIFECYCLE_CHECK.json')['status']=='PASS','old execution/outer archive not bound')
    for _,ref in ex['completion_receipts'].items():require(reference(ref['path'])==ref,'old PASS receipt changed')
    result=read(out/'RESULT.json')
    require(result['status']=='PASS' and result['actual_updates']==3 and result['sequence']==[1,2,2] and result['same_next_result_exact'] and result['complete_state_RNG_exact'] and result['state_usable_for_formal'] is False,'old fixed3 acceptance not exact')
    for ref in result['checkpoint_receipts']:
        require(Path(ref['path']).parent==out and sha(ref['path'])==ref['sha256'] and Path(ref['path']).stat().st_size==ref['size_bytes'],'old complete checkpoint SHA changed')
        require(read(ref['path']+'.complete.json')==ref,'old checkpoint sidecar changed')
    require(reference(Path(c['job_root'])/'gpu-acceptance/RESULT.json')['sha256']==reference(out/'RESULT.json')['sha256'],'new read-only acceptance result mirror differs')
    return result
