"""Separate formal contract, authenticated external acceptance and historical N.

No acceptance worker or checkpoint is a formal training predecessor.
"""
import os
from pathlib import Path, PurePosixPath
import re
import sys
from ..localization.runio import require,read,publish,reference,digest,sha
from . import h2_contract as accepted
from .h2_formal_resources import LIMITS
from .h2_history_metadata import source_rows

ROOT=accepted.ROOT
SCOPE='H2_FORMAL_UD_HISTORICAL_N_V1'
TASK='r18-h2-formal-20261024-001'
ARMS=('N','U','D')
STAGES=('train-U','train-D','infer-U','infer-D','score','cost','compare')
PRIOR_SECONDS=77.11586332297884
POLICY=dict(scope=SCOPE,training_arms=['U','D'],historical_N=True,epochs=120,windows=304,
    acceptance_replayed=False,acceptance_state_used=False,new_forwards=146578,total_forward_cap=146596,
    prior_forwards=18,prior_GPU_seconds=PRIOR_SECONDS,endpoint='e120_EMA_only',budgets=[600,300],
    cost_orders=[['N','U','D'],['N','D','U'],['U','N','D'],['U','D','N'],['D','N','U'],['D','U','N']],
    supplemental1602=False,on_anomaly='STOP_NO_RETRY')
exact=accepted.exact
science=accepted.science
environment=accepted.environment

def profile(c):
    return dict(epochs=120,windows=304,images=548,fixed_count=64) if c['profile']=='LAB_GPU_ONCE' else dict(epochs=2,windows=1,images=8,fixed_count=4)

def stage_budget(c,s):
    p=profile(c)
    return p['epochs']*p['windows']*2 if s.startswith('train-') else (p['images']+3)//4 if s.startswith('infer-') else 3*8*((p['fixed_count']+3)//4) if s=='cost' else 0

def total_new(c):return sum(stage_budget(c,s) for s in STAGES)

def fixed_ids(c):
    sel=exact(c['selection'])
    return [r['stable_image_id'] for r in sel['selected']] if 'selected' in sel else sel['fixed_ids']

def verify_source(ref):
    src=accepted.verify_source(ref)
    rows=src.get('source_files',src.get('files'));names={x['path'] for x in rows}
    require({'h2_formal.py','src/sparse_rtdetr/research/h2_history_metadata.py',*(f'src/sparse_rtdetr/research/h2_formal{n}.py' for n in ('','_contract','_resources','_worker'))}<=names,'formal helper missing from source inventory')
    return src

def original_reference(ref,root):
    """Resolve an old relative reference against its own immutable release."""
    p=Path(ref['path'])
    if not p.is_absolute():p=Path(root)/p
    require(Path(root).is_absolute(),'old release root is not absolute')
    return dict(ref,path=str(p.resolve()))

def acceptance(c):
    refs=c['acceptance'];result=exact(refs['result']);closed=exact(refs['closed']);exit_=exact(refs['exit']);cost=exact(refs['cost']);contract=exact(refs['contract'])
    require(result['status']==closed['status']==exit_['status']=='PASS' and exit_['returncode']==0 and exit_['error'] is None,'external acceptance not closed')
    require(result['contract_sha256']==closed['contract_sha256']==exit_['contract_sha256']==cost['contract_sha256']==contract['contract_sha256'],'acceptance contract bindings differ')
    require(digest({k:v for k,v in contract.items() if k!='contract_sha256'})==contract['contract_sha256'],'acceptance canonical SHA differs')
    require(exact(closed['result'])==result and exact(closed['export_index'])['status']=='PASS','acceptance closure references differ')
    require(contract['science']==science() and contract['data']==c['data'] and contract['pretrained']==c['pretrained'] and contract['environment']==c['environment'],'accepted science/data/environment changed')
    require(cost['active'] is None and cost['forward_calls']==cost['forwards_completed']==18 and cost['actual_updates']==cost['updates_completed']==9,'acceptance counts incomplete')
    source=exact(original_reference(contract['source'],contract['source_root']))
    if c['profile']=='LAB_GPU_ONCE':
        expected=read(ROOT/'docs/research/r18_local_detail_context/GPU_ACCEPTANCE_REUSE.json')['reference_sha256']
        require(set(refs)==set(expected) and all(refs[k]['sha256']==v for k,v in expected.items()),'external acceptance receipt identity changed')
        require(not result['fixture_only'] and contract['profile']=='LAB_GPU_ONCE' and contract['commit']=='7527c4646a7e86138b197d7e4dfabfb1cd4d4100','wrong external GPU acceptance')
        require(cost['GPU_seconds']==PRIOR_SECONDS and refs['contract']['sha256']=='2dd6ffc91a2a9e02c1baff01b394d6e088cf48fb1faaa8ed31ad4fa57a4c847d','acceptance cost/original contract differs')
        require(contract['gpu_uuid']==c['gpu_uuid'] and contract['source']['sha256']=='3d5fc8af11f61b020672cd3056bd3a9bb31e0cb4e207f3b66dc76858ae3d841b','acceptance GPU/source differs')
        for row in source['source_files']:
            if row['path'].startswith(('src/','vendor/','configs/')):
                require(sha(ROOT/row['path'])==row['sha256'],'accepted scientific/input dependency changed: '+row['path'])
    for arm in ARMS:
        cl=exact(refs[arm]);r=exact(cl['result']);e=exact(cl['supervisor_exit'])
        require(cl['contract_sha256']==contract['contract_sha256'] and r['arm']==arm and r['forward_calls']==6 and r['actual_updates']==3 and r['full_state_RNG_exact'] and not r['state_usable_for_formal'],'acceptance arm incomplete')
        require(set(r['logits_dtypes'])=={'torch.bfloat16'} and len(r['logits_dtypes'])==6 and e['status']=='PASS' and not e['cleanup_errors'] and not e['remaining_live'],'acceptance BF16/exit invalid')
    return result

def historical(c,*,physical=False,references=None):
    # Only the explicit metadata audit CLI supplies a read-only snapshot resolver.
    # Every production prepare/bind/load call retains native exact()/physical IO.
    require(not (physical and references is not None),'snapshot cannot certify physical checkpoint')
    exact = accepted.exact if references is None else references.exact
    n=c['historical_N'];ep=exact(n['endpoint']);b=exact(ep['binding']);proof=exact(n['authentication']);stop=exact(n['original_STOP'])
    require(ep['updates']==36480 and ep['epoch']==120 and ep['EMA'] and not ep['used_acceptance_state'],'historical N endpoint differs')
    require(not Path(n['original_CLOSED_path']).exists() if references is None else references.absent(n['original_CLOSED_path']),'historical missing CLOSED changed')
    require(n['status']=='AUTHENTICATED_OUTPUT_NOT_ORIGINAL_STAGE_CLOSURE' and proof and stop,'historical provenance missing')
    if 'source' in n:
        old=exact(n['source']);by={r['path']:r for r in source_rows(old)}
    if c['profile']=='LAB_GPU_ONCE':
        cfg=read(ROOT/'configs/research/r18_h2_nud_v1.json')
        require(ep['checkpoint']['sha256']==cfg['N_checkpoint_sha256'] and ep['initial_model_sha256']==cfg['N_initial_model_sha256'],'N checkpoint/initial identity differs')
        require(n['authentication']['sha256']=='3e0da19564a8b52cf5ab8156ff39bd08cdc843cea0202216a2c7238efa9898d7' and n['original_STOP']['sha256']=='b501b494728af6b55cd8fd88aebaa84874ba6d8433e4f01a2f906e791e92c40d','N original authentication/STOP changed')
        if physical:
            cp=ep['checkpoint'];require(Path(cp['path']).stat().st_size==cp['size_bytes'] and sha(cp['path'])==cp['sha256'],'N physical checkpoint changed')
        # The audited scientific closure binds the two score receipts to this endpoint.
        score_result=exact(n['score_result']);inference=exact(n['infer_result'])
        require(score_result['results']['B0']==n['scores'] and inference['EMA'] and inference['images']==548,'N score origin/coverage differs')
        require(n['source']['sha256']=='c91451b3db676122b9f705ed3e4647f2318c4f4c1437e35107f856e383c7ecf1','N source reference changed')
        require(proof['endpoint']==n['endpoint'] and proof['stop']==n['original_STOP'] and proof['new_source']==n['source']
            and proof['status']==n['status'] and proof['old_closure_created'] is False
            and proof['training_repeated'] is False and proof['checkpoint_loaded_for_training'] is False,'N authentication references differ')
        origin=exact(proof['contract']);origin_source=exact(proof['source']);source_rows(origin_source)
        require(origin['source']==proof['source'] and origin['commit']==origin_source['commit']
            and b['source_sha256']==proof['source']['sha256'] and b['contract_sha256']==origin['contract_sha256']
            and b['data_sha256']==digest(c['data']) and b['pretrained_sha256']==c['pretrained']['sha256']
            and b['arm']=='r18-sampling-span-v1/B0' and b['scope']=='span_formal_e120'
            and digest(b)==ep['checkpoint']['binding_sha256'],'N checkpoint binding identity differs')
        result=exact(proof['result']);worker_exit=exact(proof['worker_exit'])
        require(result['endpoint']==n['endpoint'] and result['actual_updates']==36480 and result['status']=='PASS'
            and worker_exit['status']=='PASS' and not worker_exit['remaining_children']
            and result['identity']==worker_exit['identity']
            and result['identity']['contract_sha256']==origin['contract_sha256'],'N original endpoint/worker references differ')
        score_dir=PurePosixPath(n['score_result']['path']).parent;infer_dir=PurePosixPath(n['infer_result']['path']).parent
        # CLOSED paths are derived from immutable, pinned RESULT paths. Snapshot
        # lookup still requires their original authenticated size/SHA entries.
        def related(path):
            ref=reference(path) if references is None else references.rows[str(path)][0]
            return exact(ref)
        for directory,ref,stage in ((score_dir,n['score_result'],'score'),(infer_dir,n['infer_result'],'infer-B0')):
            cl=related(directory/'CLOSED.json');sx=exact(cl['supervisor_exit']);wx=exact(cl['worker_exit'])
            require(cl['status']=='PASS' and cl['result']==ref and sx['status']=='PASS' and wx['status']=='PASS'
                and not sx['remaining_live'] and not sx['cleanup_errors'] and not wx['remaining_children']
                and cl['identity']==wx['identity'] and cl['identity']['stage']==stage
                and cl['identity']['source_sha256']==n['source']['sha256'],'N inference/score closure differs')
        require(score_result['identity']==related(score_dir/'CLOSED.json')['identity']
            and inference['identity']==related(infer_dir/'CLOSED.json')['identity']
            and score_result['identity']['contract_sha256']==inference['identity']['contract_sha256'],'N score/inference identity differs')
        audit=read(ROOT/'docs/research/r18_local_detail_context/full_cpu/B0_N_REUSE_AUDIT.json')
        require(audit['status']=='PASS_LOCAL_REFERENCE_AUDIT','N scientific reuse audit absent')
        # All common model/data/loss/evaluator dependencies are compared to the authenticated source.
        for row in audit['source_files_equal']:
            path=row['path'];require(path in by and by[path]['sha256']==row['sha256'] and by[path]['bytes']==row['bytes']
                and (ROOT/path).stat().st_size==row['bytes'] and sha(ROOT/path)==row['sha256'],'N shared scientific dependency differs: '+path)
    for budget in ('600','300'):
        if c['profile']=='LAB_GPU_ONCE':
            require(n['scores'][budget]['sha256']=={'600':'86be4fcc1478442bd4bb6f3b6e5b0c421e29f854e6977c54ba906814e79dfc7e','300':'5a829d1ed3c68022053605ab5f143d3dd988eb277960918b898ffcaacd7e3129'}[budget],'N frozen score SHA differs')
        r=exact(n['scores'][budget]);require('COCO_bridge' in r and 'official' in r,'N evaluator schema differs')
    return ep,b

def base(spec,source):
    c=read(spec);src=verify_source(source)
    fixed=dict(task_id=TASK,scope=SCOPE,policy=POLICY,resources=LIMITS,science=science(),source_root=str(ROOT),source=source,commit=src['commit'],seed=0)
    for k,v in fixed.items():require(k not in c or c[k]==v,'raw spec conflicts: '+k)
    c.update(fixed);validate(c,preparation=True);return c

def validate(c,*,preparation=False):
    require(c['scope']==SCOPE and c['policy']==POLICY and c['resources']==LIMITS and c['science']==science(),'formal frozen scope/science/resources differ')
    require(c['profile'] in ('LAB_GPU_ONCE','CPU_FIXTURE_ONLY') and c['attempt']=='attempt-formal-001' and c['seed']==0,'profile/attempt/seed differs')
    root=Path(c['task_root']);require(root.is_absolute() and root==root.resolve() and Path(c['job_root'])==root/c['attempt'] and not any(p.is_symlink() for p in (root,*root.parents)),'formal root escapes')
    require(c['source_root']==str(ROOT) and Path(c['python']).resolve()==Path(sys.executable).resolve() and len(c['nonce'])>=16,'runtime/nonce differs')
    if c['profile']=='LAB_GPU_ONCE':
        require(root.name==TASK and root.parent==Path('/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace/experiments/localization-distribution'),'production task root differs')
        require('fixture' not in c and re.fullmatch('[0-9a-f]{40}',c['commit']) and re.fullmatch(r'GPU-[0-9a-fA-F-]{36}',c['gpu_uuid']),'production identity/fixture invalid')
        cfg=read(ROOT/'configs/research/r18_h2_nud_v1.json')
        require(c['pretrained']['sha256']==cfg['ImageNet_sha256'],'pretrained differs')
        for role,key,count in (('train_core','train_manifest_sha256',4869),('development','development_manifest_sha256',548)):
            require(c['data'][role]['manifest']['sha256']==cfg[key] and c['data'][role]['count']==count,'data manifest/count changed')
        from ..localization.material_paths import metadata_path,SELECTION_SHA
        require(c['selection']['sha256']==SELECTION_SHA,'fixed64 selection differs');metadata_path('selection',c['selection']['path'],SELECTION_SHA)
    else:
        require(c['gpu_uuid']=='CPU_FIXTURE_NO_GPU','CPU fixture exposed GPU')
        require(c.get('fixture',{}).get('fault') in ('none','worker_exception','forward_overrun','update_overrun','time_overrun','allocated_overrun'),'unknown CPU fault')
    acceptance(c);historical(c)
    prior=exact(c['prior_cost'])
    require(prior['task_id']==TASK and prior['forward_calls']==18 and prior['GPU_seconds']==PRIOR_SECONDS and prior['acceptance']==c['acceptance']['cost'] and prior['total_forward_cap']==146596,'prior cost reset/identity mismatch')
    if not preparation:
        gate=exact(c['CPU_gate']);prep=exact(c['prepare_gate']);auth=exact(c['authorization'])
        require(gate['status']=='PASS' and gate['skips']==0 and gate['tests']>=12 and (c['profile']=='CPU_FIXTURE_ONLY' or not gate.get('bootstrap_fixture',False)),'actual CLI CPU gate absent')
        for row in gate['source_files']:require(sha(ROOT/row['path'])==row['sha256'],'CPU checked source differs')
        require(prep['status']=='PREPARE_PASS' and prep['source_sha256']==c['source']['sha256'] and prep['data_sha256']==digest(c['data']) and prep['profile']==c['profile'],'prepare binding differs')
        require(auth['scope']==SCOPE and auth['policy']==POLICY and auth['source_sha256']==c['source']['sha256'] and (c['profile']=='CPU_FIXTURE_ONLY' or not auth.get('CPU_fixture_not_human_GPU_authorization',False)),'authorization binding differs')
    return c

seal=accepted.seal
def load(path):
    c=read(path);require(c['contract_sha256']==digest({k:v for k,v in c.items() if k!='contract_sha256'}),'contract digest differs')
    validate(c);verify_source(c['source']);c['_path']=str(Path(path).resolve());return c

def command(c,action,arm=None):
    argv=[c['python'],'-B',str(ROOT/'h2_formal.py'),action,'--contract',c['_path']]
    if arm is not None:argv+=['--arm',arm]
    return argv
