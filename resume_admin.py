"""CPU certification, freeze and one-shot launch for the authorized continuation."""
import argparse,copy,json,os,secrets,subprocess,sys,time
from pathlib import Path

SOURCE=Path(__file__).resolve().parent
sys.path.insert(0,str(SOURCE/'src'))
import new_runtime as rt
rt.install()
from sparse_rtdetr.localization.runio import read,reference,require,publish,digest,sha


def refcheck(ref):
    actual=reference(ref['path'])
    require(actual['sha256']==ref['sha256'] and actual['bytes']==ref.get('bytes',ref.get('size_bytes')), 'frozen bytes differ: '+ref['path'])


def closure(parent,mode):
    """Reuse original closed evidence; do not replay all historical training audits."""
    d=Path(parent['job_root'])/mode;closed=read(d/'CLOSED.json');r=read(d/'RESULT.json')
    require(closed['status']=='PASS' and closed['mode']==mode and closed['seed']==1
            and closed['result']==reference(d/'RESULT.json') and closed['runtime_index']==reference(d/'runtime-archive-index.json'), 'original phase closure differs')
    archive=d/'runtime-archive';index=read(d/'runtime-archive-index.json');identity=read(archive/'identity.json')
    require(all(identity[k]==parent[k] for k in ('task_id','seed','source_sha256','contract_sha256')) and identity['mode']==mode, 'original closed phase identity differs')
    for name in ('identity.json','worker-exit.json','supervisor-exit.json'):
        require(sha(archive/name)==index['files'][name], 'original closure archive receipt corrupt')
    ex=read(archive/'supervisor-exit.json');wx=read(archive/'worker-exit.json')
    require(ex['status']=='PASS' and ex['returncode']==0 and wx['success']
            and not any(ex[k] for k in ('remaining_live','unknown_pids','cleanup_errors')), 'original process closure not clean')
    require(all(r[k]==parent[k] for k in ('seed','source_sha256','contract_sha256')), 'original phase result binding differs')
    return {'closed':reference(d/'CLOSED.json'),'result':reference(d/'RESULT.json'),'archive_index':reference(d/'runtime-archive-index.json'),
            'identity':reference(archive/'identity.json'),'worker_exit':reference(archive/'worker-exit.json'),'supervisor_exit':reference(archive/'supervisor-exit.json'),
            'scope':'existing closed phase plus frozen index and clean exit receipt; no historical per-update audit replay'}


def certify(root,old,manifest,commit):
    import torch
    require(not torch.cuda.is_initialized(),'CPU certification initialized CUDA')
    plan=read(SOURCE/'configs/sahi-protocol-resume.json');oldmanifest=old/'release-002/source-manifest.json'
    require(sha(oldmanifest)==plan['parent_source_sha256'] and sha(old/'STOP.json')==plan['parent_STOP_sha256'], 'wrong stopped parent')
    allowed=set(plan['allowed_changed_parent_files']);changed=[]
    for row in read(oldmanifest)['files']:
        p=old/'release-002/source'/row['path'];q=SOURCE/row['path']
        refcheck({'path':str(p),'bytes':row['bytes'],'sha256':row['sha256']})
        require(q.is_file(),'missing parent source '+row['path'])
        if sha(q)!=row['sha256']:
            require(row['path'] in allowed,'science source changed: '+row['path']);changed.append(row['path'])
    delta={'status':'PASS','old_manifest':reference(oldmanifest),'new_source_sha256':sha(manifest),'commit':commit,'changed_parent_files':changed,
           'all_other_parent_files_byte_exact':True,'science_vendor_model_loss_slice_data_checkpoint_exact':True}
    publish(root/'PROTOCOL_DELTA.json',delta)
    contract=reference(old/'seed1/execution-contract.json');parent=read(contract['path'])
    require(parent['contract_sha256']==digest({k:v for k,v in parent.items() if k!='contract_sha256'}),'old contract digest changed')
    formal=closure(parent,'formal');aclose=closure(parent,'infer-A')
    ep=read(old/'seed1/formal/endpoint-lock.json');binding=read(old/'seed1/formal/binding.json')
    require(ep['status']=='R18_COMMON_E120_EMA_LOCKED' and ep['seed']==1 and ep['epoch']==120 and ep['updates']==ep['actual_updates']==36480 and not ep['used_acceptance_state'], 'original e120 endpoint incomplete')
    require(ep['binding']==reference(old/'seed1/formal/binding.json') and ep['checkpoint']['sha256']==plan['parent_checkpoint_sha256'], 'e120 checkpoint/binding identity differs')
    from sparse_rtdetr.protocol_r18 import checkpoint
    payload=checkpoint.read(ep['checkpoint']['path'],ep['checkpoint']['sha256'],binding)
    engine=payload['engine'];require(engine['config']==parent['config'] and engine['completed_updates']==payload['ema']['updates']==36480
        and engine['completed_epochs']==120 and engine['microsteps']==72960 and engine['phase']==0 and not engine['failed'], 'original fullstate clock/EMA differs')
    require(set(payload['rng'])=={'python','numpy','torch_cpu','cuda'} and payload['rng']['cuda'] is not None, 'original full RNG missing')
    for label in ('model','ema'):
        state=payload['model'] if label=='model' else payload['ema']['module']
        for name,tensor in state.items():
            if tensor.is_floating_point():
                valid=~(tensor.isnan()|tensor.isneginf()).any() if name.rsplit('.',1)[-1]=='anchors' else tensor.isfinite().all()
                require(bool(valid),'nonfinite parent '+label+'/'+name)
    a=read(old/'seed1/infer-A/RESULT.json')
    require(a['status']=='PASS' and a['images']==548 and a['model_batch_forwards']==137 and a['state_RNG_exact'] is True and a['checkpoint_sha256']==ep['checkpoint']['sha256'], 'old A result differs')
    for ref in [a['raw'],*a['predictions'].values()]:refcheck(ref)
    raw=torch.load(a['raw']['path'],map_location='cpu',weights_only=True)
    require(raw['EMA'] is True and raw['epoch']==120 and raw['image_ids']==parent['development_ids'] and raw['checkpoint_sha256']==ep['checkpoint']['sha256']
            and raw['source_sha256']==parent['source_sha256'] and raw['contract_sha256']==parent['contract_sha256'], 'old A raw identity differs')
    from sparse_rtdetr.localization.consumer_geometry import positive_box
    positive_box('certified-resume/old-A',raw['pred_boxes']);require(bool(torch.isfinite(raw['pred_logits']).all()),'old A nonfinite logits')
    reuse={'kind':'certified_closed_seed1_e120_and_A548','contract':contract,'endpoint':reference(old/'seed1/formal/endpoint-lock.json'),
        'formal_closed':formal['closed'],'A_result':aclose['result'],'A_closed':aclose['closed'],'A_archive_index':aclose['archive_index']}
    proof={'status':'PASS','old_contract':contract,'old_source_sha256':parent['source_sha256'],'new_source_sha256':sha(manifest),
           'protocol_only_source_delta':True,'reuse_references':{k:v for k,v in reuse.items() if isinstance(v,dict)},'formal_closure':formal,'A_closure':aclose,
           'checkpoint':ep['checkpoint'],'binding':ep['binding'],'complete_checkpoint_SHA_and_CPU_schema_verified':True,'EMA_verified':True,
           'old_A_raw_and_prediction_SHA_verified':[a['raw'],*a['predictions'].values()], 'GPU_restore_still_required':True,
           'failed_accept_A_not_reused':True,'model_constructed':False,'CUDA_initialized':False,'delta':reference(root/'PROTOCOL_DELTA.json')}
    publish(root/'PARENT_REUSE.json',proof);reuse['certification']=reference(root/'PARENT_REUSE.json')
    return reuse


def prepare(args):
    root=Path(args.attempt);old=root.parent;root.mkdir(exist_ok=True)
    manifest=SOURCE.parent/'source-manifest.json';require(manifest.is_file(),'source must be a manifest-bound frozen release')
    reuse=certify(root,old,manifest,args.commit)
    root_runtime=Path('/home/lyy/.cache')/old.name/root.name
    refs=[]
    for seed in (1,2):
        c=copy.deepcopy(read(old/f'seed{seed}/execution-contract.json'));c.pop('contract_sha256')
        d=root/f'seed{seed}';d.mkdir(exist_ok=False)
        c.update(job_root=str(d),runtime_root=str(root_runtime/f'seed{seed}'),source_root=str(SOURCE),source_manifest=str(manifest),source_sha256=sha(manifest),
                 contract_path=str(d/'execution-contract.json'),execution_id=root.name+f'-seed{seed}',launch_credential=secrets.token_hex(32),
                 git={'repository':'https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone','branch':'codex/sahi-seed12-protocol-resume','commit':args.commit,'pushed':True},
                 GPU_execution_authorized=False,formal_training_authorized=False,CPU_preflight_only=True)
        if seed==1:c['parent_reuse']=reuse
        gate={'status':'PASS','seed':seed,'source_sha256':c['source_sha256'],'integration':reference(root/'CPU_INTEGRATION.json'),
              'reused_unaffected_CPU_gate':reference(old/f'seed{seed}/CPU_GATE.json'),'no_scientific_code_change':reference(root/'PROTOCOL_DELTA.json'),
              'new_attempt_engineering_tests':reference(root/'CPU_UNIT_GATE.json'),'CPU_only':True}
        publish(d/'CPU_GATE.json',gate);c['CPU_gate']=reference(d/'CPU_GATE.json')
        c['contract_sha256']=digest(c);publish(Path(c['contract_path']),c);rt.load(c['contract_path']);refs.append(reference(c['contract_path']))
    publish(root/'PREPARED.json',{'status':'CPU_CERTIFIED_NOT_GPU_LAUNCHED','commit':args.commit,'source_sha256':sha(manifest),'contracts':refs,'parent_reuse':reference(root/'PARENT_REUSE.json')})
    print(json.dumps(read(root/'PREPARED.json')))


def freeze(args):
    from sparse_rtdetr.localization import guard
    from sparse_rtdetr.localization.storage import identity
    root=Path(args.attempt);prepared=read(root/'PREPARED.json');contracts=[]
    require(read(root/'CPU_INTEGRATION.json')['status']=='PASS' and read(root/'CPU_UNIT_GATE.json')['status']=='PASS','CPU gates not passed')
    for ref in prepared['contracts']:
        refcheck(ref);c=rt.load(ref['path']);c.pop('contract_sha256');c.pop('CPU_preflight_only',None)
        # New immutable execution contract; never overwrite the CPU parent.
        path=Path(c['job_root'])/'gpu-execution-contract.json'
        c.update(contract_path=str(path),GPU_execution_authorized=True,formal_training_authorized=True)
        c['contract_sha256']=digest(c);publish(path,c);rt.load(path);contracts.append(reference(path))
    sample=guard.snapshot(root)
    require(sample['boot_id']==c['boot_id'] and identity(sample['storage'])==c['storage']['identity'],'resume boot/storage differs')
    guard.screen(sample,gpu_uuid=c['gpu_uuid'],entry=True,gpu=True,disk_bytes=350*1024**3)
    oldexec=read(root.parent/'EXECUTION.json')
    e={'kind':'remaining_seed12_protocol_repair','task_id':root.name,'job_root':str(root),'source_root':str(SOURCE),'source_sha256':c['source_sha256'],
       'git':c['git'],'seeds':[1,2],'contracts':contracts,'python':c['python'],'plan':c['plan'],'CPU_gate':reference(root/'CPU_INTEGRATION.json'),
       'parent_reuse':reference(root/'PARENT_REUSE.json'),'seed0_comparison':oldexec['seed0_comparison'],'seed0_cost':oldexec['seed0_cost'],
       'launch_credential':secrets.token_hex(32),'no_retry':True,'holdout':False,'total_wallclock_cap':None,'frozen_epoch':time.time(),'resource_observation':sample}
    e['execution_sha256']=digest(e);publish(root/'EXECUTION.json',e)
    print(json.dumps({'status':'READY','execution':reference(root/'EXECUTION.json'),'source_sha256':c['source_sha256'],'contracts':contracts}))


def launch(args):
    from sparse_rtdetr.localization import guard
    from sparse_rtdetr.localization.storage import identity
    root=Path(args.attempt);e=read(root/'EXECUTION.json');c=rt.load(e['contracts'][0]['path'])
    require(not any((root/n).exists() for n in ('RESUME_CONSUMED.json','LAUNCH_RECEIPT.json','STOP.json','CHAIN_STARTED.json')), 'resume launch already consumed')
    s=guard.snapshot(root);require(s['boot_id']==c['boot_id'] and identity(s['storage'])==c['storage']['identity'],'launch boot/storage differs')
    guard.screen(s,gpu_uuid=c['gpu_uuid'],entry=True,gpu=True,disk_bytes=350*1024**3)
    command=[c['python'],'-B',str(SOURCE/'resume_chain.py'),'--execution',str(root/'EXECUTION.json')]
    publish(root/'RESUME_CONSUMED.json',{'execution_sha256':e['execution_sha256'],'command':command,'consumed_epoch':time.time(),'old_launcher_replayed':False})
    with (root/'chain.log').open('xb') as log:
        p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':str(SOURCE/'src'),'PYTHONDONTWRITEBYTECODE':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8','MKL_THREADING_LAYER':'GNU'})
    receipt={'status':'STARTED_NOT_COMPLETION','pid':p.pid,'process_identity':guard.identity(p.pid),'execution_sha256':e['execution_sha256'],'command':command,'started_epoch':time.time(),'resource':s,'old_launcher_replayed':False}
    publish(root/'LAUNCH_RECEIPT.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','freeze','launch'));p.add_argument('--attempt',required=True);p.add_argument('--commit');a=p.parse_args()
    {'prepare':prepare,'freeze':freeze,'launch':launch}[a.action](a)
