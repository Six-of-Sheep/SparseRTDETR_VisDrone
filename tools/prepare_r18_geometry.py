"""Freeze committed CPU-only geometry source and bind existing diagnostic receipts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from r18_cache_diag import publish, reference

RUNTIME=['tools/r18_cache_diag.py','tools/r18_cache_worker.py','tools/r18_cache_core.py',
         'tools/r18_geometry_diag.py','tools/r18_geometry_core.py','tools/deploy_h2_source.py','configs/research/r18_geometry_v1.json']


def build(prior,output,remote):
    root=Path(__file__).parents[1]
    require_clean=subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()
    if require_clean:raise ValueError('freeze requires committed clean source')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    output.mkdir(exist_ok=False)
    source=[]
    for name in RUNTIME:
        data=subprocess.check_output(['git','show',commit+':'+name],cwd=root)
        source.append(dict(path=name,mode='100644',bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    subprocess.run(['git','archive','--format=tar','--output='+str(output/'source.tar'),commit,*RUNTIME],cwd=root,check=True)
    publish(output/'RUN_SOURCE_MANIFEST.json',dict(commit=commit,source_files=source))
    base=json.loads((prior/'contract.json').read_text())
    closed=json.loads((prior/'CLOSED.json').read_text())
    # Prior slim export had independent local SHA verification; recheck those indexed bytes.
    index=json.loads((prior/'SHA_INDEX.json').read_text())
    for name in ('contract.json','CLOSED.json'):
        wanted=next(r for r in index['files'] if r['name']==name)
        actual=reference(prior/name)
        if (wanted['bytes'],wanted['sha256'])!=(actual['bytes'],actual['sha256']):raise ValueError('prior local indexed bytes differ')
    oldroot=str(Path(base['output']).parent).replace('\\','/')
    release=remote+'/release-'+commit[:12]
    def remote_ref(local,path):
        r=reference(local);r['path']=path;return r
    policy=next(r for r in source if r['path']=='configs/research/r18_geometry_v1.json')
    c=dict(schema='r18-fixed-pair-geometry-execution-v1',profile='REAL_N_548',runtime_commit=commit,
        base_contract=remote_ref(prior/'contract.json',oldroot+'/incoming/contract.json'),
        prior_closed=remote_ref(prior/'CLOSED.json',base['output']+'/CLOSED.json'),prior_records=closed['files']['GT_RECORDS.csv.gz'],
        policy=dict(path=release+'/'+policy['path'],bytes=policy['bytes'],sha256=policy['sha256']),
        output=remote+'/attempt-cpu-001',runtime_files={Path(r['path']).name:r['sha256'] for r in source if r['path'].startswith('tools/r18_geometry_')})
    publish(output/'contract.json',c)
    publish(output/'PACKAGE.json',dict(commit=commit,release=release,remote_root=remote,
        source=reference(output/'source.tar'),manifest=reference(output/'RUN_SOURCE_MANIFEST.json'),contract=reference(output/'contract.json'),
        installer_sha256=next(r['sha256'] for r in source if r['path']=='tools/deploy_h2_source.py')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prior',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--remote-root',required=True)
    a=p.parse_args();build(a.prior,a.output,a.remote_root)
