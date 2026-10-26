"""Read-only terminal audit and bounded whitelist export; never executes diagnosis."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def verify(ref):
    p=Path(ref['path'])
    assert p.is_file() and not p.is_symlink() and p.stat().st_size==ref['bytes'] and sha(p)==ref['sha256'],'terminal reference SHA/size differs'
    return p


def collect(root):
    out=root/'attempt-cpu-001'
    closed=json.loads((out/'CLOSED.json').read_text())
    assert closed['status']=='PASS' and not (out/'STOP.json').exists()
    for ref in closed['files'].values():verify(ref)
    result=json.loads(verify(closed['result']).read_text())
    exit_=json.loads(verify(closed['exit']).read_text())
    assert result['status']==exit_['status']=='PASS' and exit_['returncode']==0 and exit_['error'] is None
    assert result['images_per_arm']==548 and result['actual_raw_batches_per_arm']==137 and not result['synthetic']
    assert result['detector_forward_calls']==result['GPU_calls']==result['AP_AR_calls']==result['training_updates']==0
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:args=p.read_bytes().split(b'\0')
        except OSError:continue
        assert not (any(a.endswith(b'/r18_cache_diag.py') for a in args) and (b'execute' in args or b'_worker' in args)), 'owned diagnostic process still running'
    assert exit_['elapsed_seconds']<1800 and exit_['peak_total_rss_bytes']<=4*1024**3 and len(exit_['cpu_affinity'])==2
    c=json.loads((root/'incoming/contract.json').read_text())
    assert sha(root/'incoming/contract.json')==json.loads((out/'CONSUMED.json').read_text())['contract_sha256']
    old=Path(c['original_N_STOP']['path']);assert sha(old)==c['original_N_STOP']['sha256'] and not Path(c['original_N_CLOSED']).exists()
    for name,s in c['runtime_files'].items():assert sha(root/('release-'+c['runtime_commit'][:12])/'tools'/name)==s
    summary=json.loads(gzip.decompress((out/'SUMMARY.json.gz').read_bytes()))
    assert summary['native_primary_original_count_match'] is True
    files={n:out/n for n in ('CLOSED.json','CONSUMED.json','RESULT.json','EXIT.json','ADMISSION.json','SUMMARY.json.gz','OVERVIEW.json','worker.stdout','worker.stderr')}
    files.update({n:root/n for n in ('INSTALL.json','LAUNCH.json','IMMEDIATE.json')})
    files.update({'contract.json':root/'incoming/contract.json','RUN_SOURCE_MANIFEST.json':root/'incoming/RUN_SOURCE_MANIFEST.json'})
    assert sum(p.stat().st_size for p in files.values())<5*1024**2
    assert sum(p.stat().st_size for p in root.rglob('*') if p.is_file())<20*1024**2
    export=root/'slim-export';export.mkdir(exist_ok=False)
    refs=[]
    for name,p in files.items():
        shutil.copyfile(p,export/name)
        assert sha(export/name)==sha(p)
        refs.append(dict(name=name,bytes=p.stat().st_size,sha256=sha(p),original_path=str(p)))
    audit=dict(status='PASS',runtime_commit=c['runtime_commit'],files=refs,
               diagnostic_executions=1,original_N_STOP_preserved=True,original_N_CLOSED_absent=True,
               owned_diagnostic_processes_absent=True,
               large_GT_records='SERVER_ONLY: '+str(out/'GT_RECORDS.csv.gz'),
               source_and_stage_closure_SHA_verified=True,AP_AR_recomputed=False)
    (export/'SHA_INDEX.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    total=sum(p.stat().st_size for p in export.iterdir())
    assert total<5*1024**2 and sum(p.stat().st_size for p in root.rglob('*') if p.is_file())<20*1024**2
    print(json.dumps(dict(status='PASS',files=len(refs)+1,bytes=total,index_sha256=sha(export/'SHA_INDEX.json'),
                          elapsed_seconds=exit_['elapsed_seconds'],peak_total_rss_bytes=exit_['peak_total_rss_bytes'])))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task-root',type=Path,required=True)
    collect(p.parse_args().task_root)
