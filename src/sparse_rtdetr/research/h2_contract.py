"""Immutable acceptance-only contracts; no formal or scoring scope exists."""
import os
from pathlib import Path
import re
import sys
from ..localization.runio import require,read,publish,reference,digest,sha
from .h2_resources import LIMITS

ROOT=Path(__file__).resolve().parents[3]
SCOPE='H2_GPU_ACCEPTANCE_ONLY_V1'
TASK='r18-h2-acceptance-20261023-001'
ARMS=('N','U','D')
POLICY=dict(scope=SCOPE,arms=list(ARMS),actual_updates=9,forward_calls=18,
    updates_per_arm=3,forwards_per_arm=6,sequence=[1,2,2],future_total_forward_budget=146596,
    formal_training=False,scientific_scoring=False,cost_benchmark=False,state_usable_for_formal=False)


def exact(ref):
    require(isinstance(ref,dict) and set(ref)=={'path','bytes','sha256'},'invalid file reference')
    p=Path(ref['path']);require(p.is_file() and not p.is_symlink() and reference(p)==ref,'file reference SHA/size differs: '+str(p))
    return read(p)


def science():
    rows=read(ROOT/'docs/research/r18_local_detail_context/full_cpu/CPU_SOURCE.json')
    rows=[r for r in rows if '/research/local_detail_' in r['path']]
    require(len(rows)==4,'accepted H2 scientific source inventory differs')
    for r in rows:
        p=ROOT/r['path'];require(p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],'H2 accepted science changed: '+r['path'])
    return dict(parent_commit='ab951f245808359c8de672a1b6fabd741531ad25',files=rows,
                config_sha256=sha(ROOT/'configs/research/r18_h2_nud_v1.json'))


def verify_source(ref):
    source=exact(ref);rows=source.get('source_files',source.get('files'))
    require(isinstance(rows,list) and rows,'source manifest has no files')
    require(len({r['path'] for r in rows})==len(rows),'duplicate source files')
    for row in rows:
        p=ROOT/row['path']
        require(not Path(row['path']).is_absolute() and '..' not in Path(row['path']).parts,'source escape')
        require(p.is_file() and not p.is_symlink() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'frozen source differs: '+row['path'])
    required={'h2_acceptance.py',*(f'src/sparse_rtdetr/research/{n}.py' for n in
        ('h2_acceptance','h2_contract','h2_resources','h2_worker','h2_cuda_identity','local_detail_model','local_detail_engine','local_detail_checkpoint'))}
    require(required<={r['path'] for r in rows},'execution helper absent from manifest')
    return source


def base(spec,source):
    c=read(spec)
    needed={'profile','attempt','task_root','job_root','python','boot_id','gpu_uuid','storage_identity','data','pretrained','environment','prior_cost','nonce'}
    require(not needed-set(c),'raw H2 spec missing fields: '+','.join(sorted(needed-set(c))))
    require(c['profile'] in ('CPU_FIXTURE_ONLY','LAB_GPU_ONCE'),'unknown profile')
    src=verify_source(source)
    fixed=dict(task_id=TASK,scope=SCOPE,policy=POLICY,resources=LIMITS,science=science(),
               source_root=str(ROOT),source=source,commit=src['commit'],seed=0)
    for k,v in fixed.items():require(k not in c or c[k]==v,'spec conflicts with frozen field: '+k)
    c.update(fixed)
    validate(c,preparation=True)
    return c


def validate(c,*,preparation=False):
    require(c['scope']==SCOPE and c['policy']==POLICY,'acceptance scope/workload changed')
    require(c['resources']==LIMITS,'resource contract changed/missing')
    require(c['science']==science(),'accepted scientific bytes differ')
    require(c['source_root']==str(ROOT) and str(Path(c['python']).resolve())==str(Path(sys.executable).resolve()),'source/interpreter binding differs')
    require(c['seed']==0 and c['attempt']=='attempt-acceptance-001','seed/attempt differs')
    root=Path(c['task_root']);job=Path(c['job_root'])
    require(root.is_absolute() and job==root/c['attempt'] and root==root.resolve(),'job root escape')
    require(not any(p.is_symlink() for p in (root,*root.parents)),'task root symlink')
    require(c['nonce'] and len(c['nonce'])>=16,'nonce absent')
    require(set(c['data'])=={'train_core','development'},'data role inventory differs')
    for role,count in (('train_core',4869),('development',548)):
        d=c['data'][role];require(d['role']==role and d['count']==count,'data member count/role differs')
    if c['profile']=='LAB_GPU_ONCE':
        require(root.name==TASK and root.parent==Path('/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace/experiments/localization-distribution'),'production task root differs')
        require(bool(re.fullmatch(r'[0-9a-f]{40}',c['commit'])),'production commit absent')
        require(re.fullmatch(r'GPU-[0-9a-fA-F-]{36}',c['gpu_uuid']) is not None,'full GPU UUID absent')
        require('fixture' not in c,'production refuses fixture options')
        conf=read(ROOT/'configs/research/r18_h2_nud_v1.json')
        require(c['pretrained']['sha256']==conf['ImageNet_sha256'],'ImageNet identity differs')
        for role,key in (('train_core','train_manifest_sha256'),('development','development_manifest_sha256')):
            require(c['data'][role]['manifest']['sha256']==conf[key],'data manifest differs')
            from ..localization.material_paths import metadata_path
            metadata_path(role,c['data'][role]['manifest']['path'],conf[key])
        require(c['environment']==dict(torch='2.4.1',torchvision='0.19.1',cuda_build='12.4',cudnn=90100),'production environment changed')
    else:
        require(c['gpu_uuid']=='CPU_FIXTURE_NO_GPU' and c['boot_id']=='CPU_FIXTURE_BOOT','fixture identity differs')
        require(c.get('fixture',{}).get('fault','none') in ('none','worker_exception','forward_overrun','update_overrun','time_overrun','wall_overrun','allocated_overrun','watchdog_deadline'),'unknown fault fixture')
    prior=exact(c['prior_cost'])
    require(prior['task_id']==TASK and prior['forward_calls']==0 and prior['GPU_seconds']==0 and prior['total_forward_cap']==146596,'prior H2 ledger absent/nonzero/reset attempt')
    if not preparation:
        gate=exact(c['CPU_gate']);prep=exact(c['prepare_gate']);auth=exact(c['authorization'])
        require(gate['status']=='PASS' and gate['skips']==0 and gate['tests']>=12,'CPU CLI gate missing')
        require(c['profile']=='CPU_FIXTURE_ONLY' or not gate.get('bootstrap_fixture',False),'production refuses bootstrap CPU fixture gate')
        for row in gate['source_files']:
            require(sha(ROOT/row['path'])==row['sha256'],'CPU tested source differs')
        require(prep['status']=='PREPARE_PASS' and prep['source_sha256']==c['source']['sha256'] and prep['data_sha256']==digest(c['data']),'prepare gate binding differs')
        require(prep['profile']==c['profile'],'prepare gate profile differs')
        require(auth['scope']==SCOPE and auth['policy']==POLICY and auth['source_sha256']==c['source']['sha256'],'authorization source/scope differs')
        require(c['profile']=='CPU_FIXTURE_ONLY' or not auth.get('CPU_fixture_not_human_GPU_authorization',False),'production refuses fixture authorization')
    return c


def seal(c,path):
    c=dict(c);c.pop('_path',None);c.pop('contract_sha256',None)
    c['contract_sha256']=digest(c);publish(path,c);return c


def load(path,*,preparation=False):
    c=read(path)
    require(c['contract_sha256']==digest({k:v for k,v in c.items() if k!='contract_sha256'}),'contract digest differs')
    validate(c,preparation=preparation);verify_source(c['source'])
    c['_path']=str(Path(path).resolve())
    return c


def environment(c,*,gpu=False):
    e=dict(os.environ,CUDA_VISIBLE_DEVICES=c['gpu_uuid'] if gpu else '',CUBLAS_WORKSPACE_CONFIG=':4096:8',
           PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_THREADING_LAYER='GNU')
    e.pop('PYTHONPATH',None)
    return e


def command(c,action,arm=None):
    result=[c['python'],'-B',str(ROOT/'h2_acceptance.py'),action,'--contract',c['_path']]
    if arm is not None:result+=['--arm',arm]
    return result
