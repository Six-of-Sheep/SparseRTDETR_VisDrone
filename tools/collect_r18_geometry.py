"""Single terminal read-only audit and bounded whitelist export. Never execute."""
import argparse
import json
from pathlib import Path
import shutil
from r18_cache_diag import require, read, exact, reference, sha, publish, output_bytes, LIMITS


def collect(root):
    out=root/'attempt-cpu-001';closed=json.loads((out/'CLOSED.json').read_text())
    require(closed['status']=='PASS' and not (out/'STOP.json').exists(),'diagnostic did not close PASS')
    for ref in closed['files'].values():exact(ref)
    result=read(closed['result']);exit_=read(closed['exit'])
    require(result['status']==exit_['status']=='PASS' and exit_['returncode']==0 and exit_['error'] is None,'exit/result differs')
    require(result['selected_GT']==10503 and result['small_B_denominator']==26441 and result['images']==548 and not result['synthetic'],'coverage differs')
    require(all(result[k]==0 for k in ('GPU_calls','model_forward_calls','training_updates','AP_AR_calls','native_matcher_calls')),'scope differs')
    require(exit_['elapsed_seconds']<1800 and exit_['peak_total_rss_bytes']<=LIMITS['rss_bytes'] and len(exit_['cpu_affinity'])==2,'resource limit differs')
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:argv=p.read_bytes().split(b'\0')
        except OSError:continue
        require(not(any(x.endswith(b'/r18_geometry_diag.py') for x in argv) and (b'execute' in argv or b'_worker' in argv)),'owned diagnostic remains live')
    c=json.loads((root/'incoming/contract.json').read_text())
    require(sha(root/'incoming/contract.json')==json.loads((out/'CONSUMED.json').read_text())['contract_sha256'],'consumed contract differs')
    base=read(c['base_contract']);read(c['prior_closed']);exact(c['prior_records'])
    exact(base['original_N_STOP']);require(not Path(base['original_N_CLOSED']).exists(),'historical closure changed')
    manifest=json.loads((root/'incoming/RUN_SOURCE_MANIFEST.json').read_text())
    release=root/('release-'+c['runtime_commit'][:12])
    for r in manifest['source_files']:exact(dict(path=str(release/r['path']),bytes=r['bytes'],sha256=r['sha256']))
    names=('CLOSED.json','CONSUMED.json','RESULT.json','EXIT.json','ADMISSION.json','SUMMARY.json','worker.stdout','worker.stderr')
    files={n:out/n for n in names}
    files.update({n:root/n for n in ('INSTALL.json','LAUNCH.json','IMMEDIATE.json')})
    files.update({'contract.json':root/'incoming/contract.json','RUN_SOURCE_MANIFEST.json':root/'incoming/RUN_SOURCE_MANIFEST.json'})
    require(sum(p.stat().st_size for p in files.values())<LIMITS['return_bytes'],'slim return exceeds budget')
    require(output_bytes(root)+sum(p.stat().st_size for p in files.values())+65536<LIMITS['output_bytes'],'export would exceed server storage')
    target=root/'slim-export';target.mkdir(exist_ok=False);refs=[]
    for n,p in files.items():
        shutil.copyfile(p,target/n);require(sha(target/n)==sha(p),'export copy differs');r=reference(target/n);r.update(name=n,original_path=str(p));refs.append(r)
    publish(target/'SHA_INDEX.json',dict(status='PASS',files=refs,original_results_preserved=True,diagnostic_executions=1,
        original_N_STOP_preserved=True,original_N_CLOSED_absent=True,owned_worker_absent=True,server_only_pairs=reference(out/'PAIRS.csv.gz')))
    require(output_bytes(target)<LIMITS['return_bytes'] and output_bytes(root)<LIMITS['output_bytes'],'final export budget exceeded')
    print(json.dumps(dict(status='PASS',files=len(refs)+1,return_bytes=output_bytes(target),server_bytes=output_bytes(root),index=reference(target/'SHA_INDEX.json'),exit=exit_)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task-root',type=Path,required=True);collect(p.parse_args().task_root)
