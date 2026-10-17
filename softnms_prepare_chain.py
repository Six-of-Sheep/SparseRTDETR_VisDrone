"""Preparation-only CLI; resolve historical inputs from authenticated references."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main(argv=None):
    import softnms_job as j
    from softnms_input_check import PARENTS
    from softnms_parent_check import Snapshot
    p=argparse.ArgumentParser()
    p.add_argument('--request',required=True);p.add_argument('--output',required=True)
    a=p.parse_args(argv);request=Path(a.request);out=Path(a.output)
    j.require(not out.exists(),'preparation chain output already used')
    # No input/output is inferred from a directory name. Verify all bindings first.
    req=j.read(request)
    old=json.loads(j.exact(req['historical_contract'],contents=True))
    old_gate=json.loads(j.exact(old['parent_schema_cpu_gate'],contents=True))
    j.require(old_gate['status']=='PASS' and old_gate['seeds']==[0,1,2], 'historical parent gate not PASS')
    snapshot_ref=old_gate['real_snapshot']
    snapshot_path=j.exact(snapshot_ref)
    snapshot=Snapshot(snapshot_path)
    parents={k:old[k] for k in PARENTS}
    j.require(snapshot.value['request']==parents,'snapshot parent binding differs')
    prior_manifest=json.loads(j.exact(req['prior_source_manifest'],contents=True))
    j.require(prior_manifest['commit']==req['prior_runtime_commit'],'inherited CPU source commit differs')
    source_comparison=[]
    for row in prior_manifest['files']:
        # Only a new preparation driver / documentation may differ. All code
        # exercised by the inherited synthetic CPU gate remains byte-identical.
        if row['path'].endswith('.py') or row['path']=='configs/sahi_softnms_fixed.json':
            j.require(j.sha(j.ROOT/row['path'])==row['sha256'],'inherited CPU code changed: '+row['path'])
            source_comparison.append(row['path'])
    for ref in req['prior_synthetic_CPU']:
        value=json.loads(j.exact(ref,contents=True))
        j.require(value['status']=='PASS','inherited synthetic CPU not PASS')
    checked={}
    def check_ref(ref):
        path=Path(ref['path'])
        if str(path) in checked:
            j.require(checked[str(path)]==ref,'conflicting repeated input reference')
            return
        j.require(path.is_absolute() and path.is_file()
                  and not any(x.is_symlink() for x in (path,*path.parents)), 'missing/nonregular/symlink input: '+str(path))
        j.require(not any(x in str(path).lower().replace('-','_') for x in ('holdout','confirmatory','test_dev')),
                  'forbidden input role')
        j.require(path.stat().st_size==ref.get('bytes',ref.get('size_bytes')),'preflight size differs: '+str(path))
        checked[str(path)]=ref
    def scan(value):
        if isinstance(value,dict):
            if 'path' in value and 'sha256' in value and ('bytes' in value or 'size_bytes' in value):
                ref={'path':value['path'],'bytes':value.get('bytes',value.get('size_bytes')),'sha256':value['sha256']}
                # Relative manifest entries describe source members; their
                # owning bound source root is checked by j.context, not cwd.
                if Path(ref['path']).is_absolute():check_ref(ref)
            for x in value.values():scan(x)
        elif isinstance(value,list):
            for x in value:scan(x)
    for item in snapshot.files.values():
        scan(item['reference']);scan(item['value'])
        j.exact(item['reference'])
    scan(old)
    j.require(j.parent_ready(parents),'original seed chain not closed')
    specs,rows,data=j.context(parents)
    scan(specs)
    for row in rows:
        path=Path(data['roots']['annotations'])/Path(row['annotation_relative_path']).name
        check_ref({'path':str(path),'bytes':row['annotation_size_bytes'],'sha256':row['annotation_sha256']})
    j.require(out.is_absolute() and out.parent.is_dir()
              and not any(q.is_symlink() for q in (out,*out.parents)), 'noncanonical output')
    targets=[out/'parent-check',out/'real-data']
    j.require(all(not t.exists() for t in targets),'new stage output already used')
    j.resources(out.parent,j.read(j.PLAN))
    out.mkdir()
    j.publish(out/'PREFLIGHT.json',{'status':'PASS','request':j.reference(request),
        'historical_contract':req['historical_contract'],'historical_parent_gate':old['parent_schema_cpu_gate'],
        'snapshot':snapshot_ref,'inputs':list(checked.values()),'source_files_compared':source_comparison,
        'prior_synthetic_CPU':req['prior_synthetic_CPU'],'preflight_SHA_scope':'all small snapshot files; large raw/predictions checked fully at their stage',
        'new_outputs':[str(t) for t in targets],'scoring_allowed':False})
    stages=[]
    try:
        def stage(name,args):
            started=time.monotonic()
            with (out/(name+'.log')).open('xb') as log:
                process=subprocess.run([sys.executable,'-B',str(j.ROOT/'softnms_job.py'),*args],stdout=log,stderr=subprocess.STDOUT)
            result={'stage':name,'returncode':process.returncode,'elapsed_seconds':time.monotonic()-started}
            stages.append(result);j.publish(out/(name+'-EXIT.json'),result)
            j.require(process.returncode==0,'preparation stage failed: '+name)
        metadata_request=j.publish(out/'METADATA_REQUEST.json',{'parent_bindings':parents})
        stage('metadata',['parent-check','--request',metadata_request['path'],'--snapshot',str(snapshot_path),
                          '--output',str(targets[0])])
        preparation_request=j.publish(out/'PREPARATION_REQUEST.json',{'parent_bindings':parents,
            'parent_schema_cpu_gate':j.reference(targets[0]/'PARENT_CPU_GATE.json')})
        stage('real-input-and-cache',['input-check','--request',preparation_request['path'],'--output',str(targets[1])])
        j.publish(out/'PREPARATION_CLOSED.json',{'status':'PASS','stages':stages,
            'parent_schema_cpu_gate':j.reference(targets[0]/'PARENT_CPU_GATE.json'),
            'real_data_gate':j.reference(targets[1]/'REAL_DATA_GATE.json'),'scoring_attempt_started':False})
        return 0
    except BaseException as e:
        j.publish(out/'STOP.json',{'status':'STOP_PREPARATION','error':repr(e),'traceback':traceback.format_exc(),'stages':stages,
            'retry_policy':'only positively identified new-tool path/serialization/schema errors may be repaired under current user scope; real evidence conflicts stop'})
        return 2


if __name__=='__main__':
    raise SystemExit(main())
