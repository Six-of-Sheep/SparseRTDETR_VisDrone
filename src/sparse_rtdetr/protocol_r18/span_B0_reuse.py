"""Explicit completed-worker reuse after failed supervision; never forge CLOSED.

Full authentication is performed once by the tracked lab/CPU reconcile action.
An exact-source-bound witness retains full SHA plus inode/size/mtime/ctime for
each immutable parent artifact. Every new process checks all fingerprints and
bound small metadata. Checkpoint content is additionally rehashed at inference
by the original tensor loader. No old file or binding is rewritten.
"""
import ast,copy,json
from pathlib import Path
from . import span_runtime as rt
from .span_event_stream import fingerprint

OLD_COMMIT='ab88a9b3b2c8fc1f3203269f4ae022d52b0dd8cb'
OLD_SOURCE='bb073d33bdd4a935d30cf63bfc936dd9940ac85c6eed1a65a476f8f1a2af5197'
OLD_CONTRACT='08a0835fd9df656aae8e87af065e7a528f4037e9382ae854211a5ab592769a7b'
ALLOWED={'src/sparse_rtdetr/protocol_r18/span_runtime.py',
         'src/sparse_rtdetr/protocol_r18/span_job.py',
         'src/sparse_rtdetr/protocol_r18/span_formal.py',
         'src/sparse_rtdetr/protocol_r18/span_formal_admission.py',
         'src/sparse_rtdetr/protocol_r18/span_external_acceptance.py',
         'src/sparse_rtdetr/protocol_r18/span_worker.py',
         'configs/research/r18_sampling_span_formal_v1.json',
         '.github/workflows/repository-checks.yml','README.md','manifests/experiments.json'}


def worker_binding_only(before,after):
    """Permit exactly two mechanical identity/export substitutions, no math."""
    old=before.decode('utf-8')
    pairs=[("state['binding']['contract_sha256']==c['contract_sha256']",
            "state['binding']['contract_sha256']==rt.training_contract(c,arm)['contract_sha256']"),
           ("sources.append((stage+'-CLOSED.json',rt.stage_path(c,stage)/'CLOSED.json'))",
            "sources.append(rt.export_stage_evidence(c,stage))")]
    for a,b in pairs:
        rt.require(old.count(a)==1,'original worker binding seam differs');old=old.replace(a,b,1)
    rt.require(ast.dump(ast.parse(old),include_attributes=False)==ast.dump(ast.parse(after.decode('utf-8')),include_attributes=False),
               'worker computation changed beyond endpoint binding/export')


def checkpoint_health(state):
    """Allow only the native invalid-anchor +inf sentinel, never arbitrary inf.

    The original save-time geometry check authenticates generated cache values.
    Reconciliation additionally checks their named masks and model/EMA equality;
    every tensor outside these two exact cache paths must remain finite.
    """
    import torch
    caches=[];summary={}
    for label,component in (('model',state['model']),('ema',state['ema']['module'])):
        anchors=component['decoder.anchors'];mask=component['decoder.valid_mask']
        rt.require(isinstance(anchors,torch.Tensor) and isinstance(mask,torch.Tensor)
            and anchors.dtype==torch.float32 and mask.dtype==torch.bool
            and anchors.ndim==3 and anchors.shape[0]==1 and anchors.shape[-1]==4
            and mask.shape==anchors.shape[:-1]+(1,), 'B0 native anchor cache schema differs')
        valid=mask.expand_as(anchors)
        rt.require(bool(anchors[valid].isfinite().all())
            and bool(anchors[~valid].isposinf().all()), 'B0 native anchor sentinel/mask differs')
        caches.append((anchors,mask))
        summary[label]=dict(invalid_coordinates=int((~valid).sum()),
            valid_coordinates=int(valid.sum()),invalid_values='positive_infinity_only')
    rt.require(all(torch.equal(a,b) for a,b in zip(caches[0],caches[1])),
        'B0 model/EMA native caches differ')
    count=0
    def check(value,path=()):
        nonlocal count
        if isinstance(value,torch.Tensor):
            count+=1
            if path not in {('model','decoder.anchors'),('ema','module','decoder.anchors')}:
                rt.require(bool(value.isfinite().all()),'B0 state nonfinite: '+ '/'.join(map(str,path)))
        elif isinstance(value,dict):
            for key,item in value.items():check(item,path+(key,))
        elif isinstance(value,(list,tuple)):
            for key,item in enumerate(value):check(item,path+(key,))
    check(state)
    return dict(status='PASS',tensor_count=count,native_anchor_caches=summary,
                all_other_tensors_finite=True)


def build(parent_contract,new_source,output):
    contract_ref=rt.reference(Path(parent_contract).resolve());old=rt.exact(contract_ref)
    rt.require(old['protocol']==rt.FORMAL_PROTOCOL and old['task_id']==rt.TASK and 'external_B0' not in old,'only original formal B0 may be reused')
    rt.require(old['contract_sha256']==rt.digest({k:v for k,v in old.items() if k!='contract_sha256'}),'parent contract digest differs')
    if old['profile']=='LAB_GPU_ONCE':
        rt.require(contract_ref['sha256']==OLD_CONTRACT and old['source']['sha256']==OLD_SOURCE and old['commit']==OLD_COMMIT,'original B0 provenance differs')
    rt.require(old['science_sha256']==rt.digest(rt.recipe()) and old['execution_sha256']==rt.digest(rt.execution()),'parent science changed')
    source=rt.exact(old['source']);new=rt.exact(new_source);root=Path(old['source_root'])
    before={x['path']:x for x in source['files']};after={x['path']:x for x in new['files']}
    rt.require(set(before)<=set(after),'new source dropped parent files')
    changes=[n for n in before if before[n]!=after[n]]
    rt.require(set(changes)<=ALLOWED,'B0 computation differs: '+','.join(sorted(set(changes)-ALLOWED)))
    watched=[]
    for n,e in before.items():
        p=root/n
        rt.require(p.resolve().is_relative_to(root) and p.is_file() and not p.is_symlink(),'unsafe old source member')
        rt.require(rt.sha(p)==e['sha256'] and p.stat().st_size==e['bytes'],'parent source SHA differs: '+n)
        watched.append(fingerprint(p))
    worker='src/sparse_rtdetr/protocol_r18/span_worker.py'
    if worker in changes:worker_binding_only((root/worker).read_bytes(),(rt.ROOT/worker).read_bytes())
    stage=Path(old['job_root'])/'train-B0';stop_ref=rt.reference(Path(old['job_root'])/'STOP.json');stop=rt.exact(stop_ref)
    rt.require(stop['status']=='STOP_NO_RETRY' and stop['completed_stages']==['admission'] and 'train-B0' in stop['error'],'parent failed beyond permitted point')
    proc_ref=rt.reference(Path(old['job_root'])/'train-B0-supervisor-process/PROCESS.json');proc=rt.exact(proc_ref)
    rt.require(proc['returncode']==-9,'parent supervision failure differs')
    rt.require(not(stage/'CLOSED.json').exists() and not(stage/'supervisor-exit.json').exists(),'parent supervision state not the reviewed failure')
    result_ref=rt.reference(stage/'RESULT.json');result=rt.exact(result_ref)
    exit_ref=rt.reference(stage/'worker-exit.json');ex=rt.exact(exit_ref)
    rt.require(result['status']==ex['status']=='PASS' and result['identity']==ex['identity']==rt.identity(old,'train-B0') and not ex['remaining_children'],'parent worker incomplete')
    profile=rt.profile(old);updates=profile['epochs']*profile['windows']
    rt.require(result['actual_updates']==updates and result['forward_calls']==2*updates and not result['used_acceptance_state'],'B0 endpoint counts/scope differ')
    scan=rt.verify_events(old,'train-B0',stage/'events.jsonl')
    rt.require(scan[-1]['fields']['result']==result_ref and scan.pid==ex['pid'] and scan.input_count==updates
        and scan.input_sha256==result['student_inputs_sha256'],'B0 terminal/input/writer chain differs')
    ep=rt.exact(result['endpoint']);binding=rt.exact(ep['binding']);cp=ep['checkpoint'];path=Path(cp['path'])
    rt.require(path==stage/f"checkpoint-e{profile['epochs']:03d}.pth" and ep['epoch']==profile['epochs'] and ep['updates']==updates and ep['EMA'] and not ep['used_acceptance_state'],'wrong fixed endpoint')
    rt.require(binding['contract_sha256']==old['contract_sha256'] and binding['source_sha256']==old['source']['sha256']
        and binding['arm']=='r18-sampling-span-v1/B0','B0 state binding differs')
    rt.require(rt.read(path.with_suffix('.pth.complete.json'))==cp and rt.sha(path)==cp['sha256'] and path.stat().st_size==cp['size_bytes'],'B0 checkpoint SHA/sidecar differs')
    init_ref=rt.reference(Path(old['job_root'])/'admission/initialization.json');initial=rt.exact(init_ref)
    rt.require(ep['initial_model_sha256']==result['initial_model_sha256']==initial['B0']['full']==initial['P']['full'],'B0 common initial state differs')
    # Inspect the certified endpoint on CPU only; no model construction/forward.
    from . import span_checkpoint as checkpoint
    import torch
    torch.set_num_threads(2)
    state=checkpoint.read(path,cp['sha256'],binding)
    rt.require(state['engine']['completed_updates']==state['ema']['updates']==updates and state['binding_sha256']==cp['binding_sha256'],'B0 optimizer/EMA clocks differ')
    health=checkpoint_health(state)
    del state
    cost_ref=rt.reference(Path(old['job_root'])/'COST_LEDGER.json');ledger=rt.exact(cost_ref)
    rt.require(ledger['active'] is None and ledger['forward_calls']==18+2*updates and ledger['completed_GPU_seconds']>=ledger['prior']['GPU_seconds'],'parent cost incomplete/reset')
    if old['profile']=='LAB_GPU_ONCE':
        rt.require(ledger['forward_calls']==72978 and abs(ledger['completed_GPU_seconds']-59334.44017676904)<1e-6,'reviewed cumulative cost differs')
    watched.extend(fingerprint(p) for p in [Path(parent_contract),Path(old['source']['path']),Path(stop_ref['path']),Path(proc_ref['path']),
        Path(result_ref['path']),Path(exit_ref['path']),Path(result['endpoint']['path']),Path(ep['binding']['path']),
        path,path.with_suffix('.pth.complete.json'),Path(cost_ref['path']),Path(init_ref['path'])])
    watched.append(scan.file_identity)
    # Check source/stat identities again after all expensive reads.
    rt.require(all(fingerprint(x['path'])==x for x in watched),'parent changed during reconciliation')
    proof=dict(schema='sampling-span-completed-B0-with-failed-supervision-v1',
        status='AUTHENTICATED_OUTPUT_NOT_ORIGINAL_STAGE_CLOSURE',contract=contract_ref,source=old['source'],new_source=new_source,
        result=result_ref,worker_exit=exit_ref,endpoint=result['endpoint'],stop=stop_ref,failed_supervisor_process=proc_ref,
        ledger=cost_ref,initialization=init_ref,events=dict(path=str(stage/'events.jsonl'),bytes=scan.bytes,sha256=scan.sha256),
        events_count=len(scan),student_inputs_sha256=scan.input_sha256,watched=watched,changed_source_files=changes,state_health=health,
        old_closure_created=False,checkpoint_copied=False,training_repeated=False,checkpoint_loaded_for_training=False)
    rt.publish(output,proof);return rt.reference(output)


def verify(c):
    rt.require(c['protocol']==rt.FORMAL_PROTOCOL and 'external_B0' in c,'B0 reuse outside explicit formal contract')
    proof=rt.exact(c['external_B0']);old=rt.exact(proof['contract'])
    rt.require(old['contract_sha256']==rt.digest({k:v for k,v in old.items() if k!='contract_sha256'})
        and proof['source']==old['source'],'B0 original contract/source digest differs')
    rt.require(all(proof[k] is False for k in ('old_closure_created','checkpoint_copied','training_repeated','checkpoint_loaded_for_training')),
        'B0 reuse scope differs')
    rt.require(proof['schema']=='sampling-span-completed-B0-with-failed-supervision-v1' and
        proof['status']=='AUTHENTICATED_OUTPUT_NOT_ORIGINAL_STAGE_CLOSURE' and proof['new_source']==c['source'],'B0 witness identity/source differs')
    rt.require(old['attempt']!=c['attempt'] and old['job_root']!=c['job_root'] and old['profile']==c['profile'],'B0 attempt/profile identity differs')
    rt.require(old['data']==c['data'] and old['pretrained']==c['pretrained'] and old['science_sha256']==c['science_sha256']
        and old['execution_sha256']==c['execution_sha256'] and old['seed']==c['seed'],'reused B0 scientific material differs')
    rt.require(all(fingerprint(x['path'])==x for x in proof['watched']),'authenticated B0 evidence changed since full scan')
    result=rt.exact(proof['result']);ledger=rt.exact(proof['ledger']);prior=rt.exact(c['prior_cost'])
    rt.require(result['identity']==rt.identity(old,'train-B0') and result['status']=='PASS','B0 result identity differs')
    rt.require(prior['forward_calls']==ledger['forward_calls'] and prior['GPU_seconds']==ledger['completed_GPU_seconds']
        and prior['parent_ledger']==proof['ledger'] and prior['task_id']==rt.TASK and prior['profile']==c['profile'],'B0 cumulative cost reset/conflict')
    if c['profile']=='LAB_GPU_ONCE':
        rt.require(proof['contract']['sha256']==OLD_CONTRACT and proof['source']['sha256']==OLD_SOURCE,'B0 original provenance differs')
    rt.require(not(Path(old['job_root'])/'train-B0/CLOSED.json').exists(),'old closure must remain absent')
    c['_verified_B0']=dict(parent=old,proof=proof,result=result)
    return c['_verified_B0']
