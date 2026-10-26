"""Build a small source bundle and exact diagnostic contract from existing receipts.

Reads metadata only. Does not connect, deploy, load predictions or execute.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from r18_cache_diag import publish, reference

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ['tools/r18_cache_diag.py','tools/r18_cache_worker.py','tools/r18_cache_core.py',
           'tools/deploy_h2_source.py','configs/research/r18_cache_diag_v1.json']


def metadata(path):
    return json.loads(path.read_text(encoding='utf-8'))


def bound_metadata(base, name):
    p = base / name
    idx = metadata(base/'DELIVERY_INDEX.json')
    wanted = next(r for r in idx['files'] if r['path'].replace('\\','/') == name)
    actual = reference(p)
    if actual['bytes'] != wanted['bytes'] or actual['sha256'] != wanted['sha256']:
        raise ValueError('existing local metadata index differs: '+name)
    return metadata(p)


def build(delivery, output, remote):
    output.mkdir(parents=False,exist_ok=False)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise ValueError('source must be committed and clean before freeze')
    source=[]
    for name in RUNTIME:
        data=subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT)
        source.append(dict(path=name,mode='100644',bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    subprocess.run(['git','archive','--format=tar','--output='+str(output/'source.tar'),commit,*RUNTIME],cwd=ROOT,check=True)
    publish(output/'RUN_SOURCE_MANIFEST.json',dict(commit=commit,source_files=source))
    h=delivery/'r18-h2-formal-20261024-001/terminal-delivery-001'
    s=delivery/'r18-sampling-span-20261020-001/terminal-delivery-001'
    hc=bound_metadata(h,'receipts/contract.json')
    nc=bound_metadata(s,'receipts/formal-execution-003/contract.json')
    physical=bound_metadata(h,'PHYSICAL_ENDPOINT_SHA.json')
    score=bound_metadata(h,'receipts/score-RESULT.json')
    policy=json.loads(subprocess.check_output(['git','show',commit+':configs/research/r18_cache_diag_v1.json'],cwd=ROOT))
    c={k:policy[k] for k in ('arms_order','images','thresholds','budgets','no_GPU_forward_training_AP_AR','supplemental1602','limits')}
    c.update(schema='r18-cache-diagnostic-v1',profile='REAL_NUD_548',runtime_commit=commit,
             native_root=hc['source_root'],native_commit=hc['commit'],native_manifest=hc['source'],
             N_source=dict(root=nc['source_root'],manifest=nc['source']),
             manifest=hc['data']['development']['manifest'],annotation_root=hc['data']['development']['roots']['annotations'],
             output=remote+'/attempt-cpu-001',arms={},
             original_N_STOP=hc['historical_N']['original_STOP'],original_N_CLOSED=hc['historical_N']['original_CLOSED_path'],
             runtime_files={Path(r['path']).name:r['sha256'] for r in source if r['path'].startswith('tools/r18_cache_')})
    # Preserve original remote paths: metadata hashes bind the local verified copies.
    def original_ref(local, path):
        r=reference(local)
        r['path']=path
        return r
    taskbase='/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace/experiments/localization-distribution/'
    c['origin_contracts']=[original_ref(h/'receipts/contract.json',taskbase+'r18-h2-formal-20261024-001/metadata-003/contract.json'),
                           original_ref(s/'receipts/formal-execution-003/contract.json',taskbase+'r18-sampling-span-20261020-001/formal-execution-003/contract.json')]
    for p in physical['endpoints']:
        arm=p['arm']
        if arm=='N':
            closed=bound_metadata(s,'receipts/attempt-formal-003/infer-B0/CLOSED.json')
            closedpath=taskbase+'r18-sampling-span-20261020-001/attempt-formal-003/infer-B0/CLOSED.json'
            closedlocal=s/'receipts/attempt-formal-003/infer-B0/CLOSED.json'
        else:
            closed=bound_metadata(h,f'receipts/infer-{arm}-CLOSED.json')
            closedpath=taskbase+f'r18-h2-formal-20261024-001/attempt-formal-001/infer-{arm}/CLOSED.json'
            closedlocal=h/f'receipts/infer-{arm}-CLOSED.json'
        c['arms'][arm]=dict(endpoint=p['endpoint'],binding=p['binding'],checkpoint=p['checkpoint'],
                           infer_result=closed['result'],infer_closed=original_ref(closedlocal,closedpath),
                           scores=score['results'][arm])
    publish(output/'contract.json',c)
    publish(output/'PACKAGE.json',dict(commit=commit,source=reference(output/'source.tar'),manifest=reference(output/'RUN_SOURCE_MANIFEST.json'),
              contract=reference(output/'contract.json'),remote_root=remote,release=remote+'/release-'+commit[:12],
              files=len(source),source_bytes=sum(r['bytes'] for r in source)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--delivery-research',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--remote-root',required=True)
    a=p.parse_args()
    build(a.delivery_research,a.output.resolve(),a.remote_root)
