"""Explicit small-file whitelist; never exports checkpoint/prediction/audit/log bytes."""
import hashlib,json,os,time
from pathlib import Path

def server_export(execution):
 from sparse_rtdetr.localization.runio import publish,reference,read
 root=Path(execution['job_root']);index=[];allowed=[]
 manifest=Path(execution['source_root']).parent/'source-manifest.json'
 if manifest.is_file():allowed.append(manifest)
 for n in ('PLAN-v002.json','EXECUTION.json','INVENTORY.json','CPU_GATE.json','CHAIN_STARTED.json','CHAIN_RESULT.json','STOP.json','LAUNCH_RECEIPT.json',
           'PROTOCOL_DELTA.json','PARENT_REUSE.json','PREPARED.json','CPU_INTEGRATION.json','CPU_UNIT_GATE.json','SOURCE_GIT.json'):
  if (root/n).is_file():allowed.append(root/n)
 for n in ('AGGREGATE.json','PAIRED_SUMMARY.csv','REPORT.md','RESULT.json'):
  if (root/'summary'/n).is_file():allowed.append(root/'summary'/n)
 for seed in (1,2):
  d=root/f'seed{seed}'
  contract_path=next(Path(r['path']) for r in execution['contracts'] if Path(r['path']).parent==d)
  allowed.append(contract_path)
  actual=read(contract_path)
  if actual.get('parent_reuse'):
   reuse=read(actual['parent_reuse']['certification']['path'])
   cp=reuse['checkpoint']
   index.append({'kind':'reused_original_seed1_checkpoint','path':cp['path'],'bytes':cp['size_bytes'],'sha256':cp['sha256'],
                 'sha_source':actual['parent_reuse']['certification'],'location':'SERVER_ONLY','new_GPU_fullstate_restore_required':True})
   index.append({'kind':'reused_original_A548','references':reuse['old_A_raw_and_prediction_SHA_verified'],'location':'SERVER_ONLY'})
  for n in ('execution-contract.json','CPU_GATE.json','FORMAL_PREFLIGHT.json'):
   if (d/n).is_file():allowed.append(d/n)
  for mode in ('gpu-acceptance','formal','infer-A','accept-A','accept-B','accept-C','full-C','evaluation'):
   for n in ('RESULT.json','CLOSED.json','failure.json'):
    if (d/mode/n).is_file():allowed.append(d/mode/n)
  for n in ('formal/binding.json','formal/endpoint-lock.json','evaluation/COMPARISON.json','evaluation/TABLE.csv'):
   if (d/n).is_file():allowed.append(d/n)
  for n in ('formal/checkpoint-epoch-000.pth.complete.json','formal/checkpoint-epoch-120.pth.complete.json'):
   if (d/n).is_file():allowed.append(d/n)
  for mode in ('gpu-acceptance','formal','infer-A','accept-A','accept-B','accept-C','full-C','evaluation'):
   for n in ('supervisor-exit.json','runtime-archive-index.json'):
    p=(d/mode/'runtime-archive'/n) if n=='supervisor-exit.json' else d/mode/n
    if p.is_file():allowed.append(p)
  # Only existing authenticated sidecars are indexed, large originals stay remote.
  if (d/'formal').exists():
   for ep in range(121):
    p=d/'formal'/f'checkpoint-epoch-{ep:03d}.pth.complete.json'
    if p.is_file():
     cp=read(p);index.append({'kind':'checkpoint','path':cp['path'],'bytes':cp['size_bytes'],'sha256':cp['sha256'],'sha_source':str(p),'SHA_scope':'checkpoint atomic write sidecar; not rehashed at export','location':'SERVER_ONLY'})
  for mode in ('infer-A','full-C','evaluation'):
   p=d/mode/'RESULT.json'
   if p.is_file():index.append({'kind':'result_index','path':str(p),'reference':reference(p),'location':'SERVER_ONLY large raw/prediction references in result; no local large byte copy'})
 publish(root/'SCIENTIFIC_SERVER_INDEX.json',{'recorded_epoch':time.time(),'files':index,'full_audits_and_logs':'SERVER_ONLY at seedN/{mode}; original frozen path retained','not_independent_backup':True})
 allowed.append(root/'SCIENTIFIC_SERVER_INDEX.json')
 refs=[reference(p) for p in dict.fromkeys(allowed)]
 total=sum(r['bytes'] for r in refs)
 if len(refs)>200 or total>32*1024**2:raise RuntimeError('slim export whitelist budget exceeded')
 value={'task_id':execution['task_id'],'status':'READY_FOR_WINDOWS_SLIM_RETURN','files':refs,'total_bytes':total,'files_count':len(refs),'max_bytes':32*1024**2,'large_scientific_originals':'SERVER_ONLY','not_full_audit_delivery':True,'new_checkpoint_SHA_pass_claim':False}
 publish(root/'SLIM_EXPORT.json',value)
 return value
