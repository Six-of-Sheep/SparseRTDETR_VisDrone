"""Constructed algebra and actual geometry CLI acceptance; never real data."""
import argparse
import copy
import csv
import gzip
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from r18_cache_diag import publish, reference, sha


def algebra():
    import numpy as np
    from r18_geometry_core import analyze, aggregate
    gt=[0,0,10,10]
    examples=[([2,0,12,10],'center_only'),([-1,-1,11,11],'size_only'),
              ([1.5,.5,10.5,9.5],'both_individually'),([2.5,1,10.5,9],'neither_individually')]
    for b,group in examples:
        v=analyze([b,b],gt)
        assert v['query']==0 and v['exact_ties']==2 and v['group']==group,(v,group)
        assert abs(v['metrics']['both_IoU']-1)<1e-12
    # Fixed selection must not turn into choosing a replacement-specific winner.
    boxes=[[2,0,12,10],[-1,-1,11,11]]
    v=analyze(boxes,gt)
    assert v['query']==1 and v['metrics']['center_IoU']<.75
    assert analyze([boxes[0]],gt)['metrics']['center_IoU']==1
    # Translation/scaling and asymmetric x/y units preserve geometric decisions.
    original=analyze([[2,0,12,10]],gt)
    scaled=analyze([[104,-40,124,-10]],[100,-40,120,-10])
    assert original['group']==scaled['group']
    assert abs(original['metrics']['original_IoU']-scaled['metrics']['original_IoU'])<1e-12
    r=analyze([[-2,1,8,9]],gt)
    assert r['metrics']['dx_over_GT_width']==-.2 and r['metrics']['height_ratio']==.8
    # Correction components interact: their gains need not add up to both correction.
    assert abs(analyze([[2.5,1,10.5,9]],gt)['metrics']['interaction'])>0
    for bad in ([0,0,0,1],[0,0,float('nan'),1]):
        try:analyze([bad],gt)
        except ValueError:pass
        else:raise AssertionError('bad geometry accepted')
    original.update(image_id='CONSTRUCTED',category='1',sequence='S',area_bin='(64,256]')
    agg=aggregate([original],range(1,11),['S','EMPTY'])
    assert agg['groups']['sequence']['EMPTY']['denominator']==0
    assert sum(agg['overall']['groups'].values())==1


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--native-root',type=Path)
    a=p.parse_args();root=a.output.resolve();root.mkdir(exist_ok=False)
    algebra()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    os.environ.update({k:env[k] for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')})
    temporary=None
    if a.native_root:native_root=a.native_root.resolve()
    else:
        temporary=tempfile.TemporaryDirectory(prefix='geometry-native-');native_root=Path(temporary.name)
        raw=subprocess.check_output(['git','archive','e50bedbe8e41bdfd3c47d09f7076d9644e438e26'],cwd=Path(__file__).parents[1])
        with tarfile.open(fileobj=io.BytesIO(raw)) as f:f.extractall(native_root)
    from check_r18_cache_cli import fixture
    b=root/'fixture';b.mkdir();base=fixture(b,native_root,geometry_fixture=True)
    log=0
    def run(script, mode, cfg, wanted=0, message=None):
        nonlocal log
        r=subprocess.run([sys.executable,'-B',str(Path(__file__).with_name(script)),mode,'--contract',str(cfg),'--sha256',sha(cfg)],env=env,capture_output=True,text=True)
        (root/f'cli-{log}.txt').write_text(r.stdout+r.stderr);log+=1
        assert r.returncode==wanted and (not message or message in r.stdout+r.stderr),(r.returncode,r.stdout,r.stderr)
    run('r18_cache_diag.py','execute',b/'contract.json')
    prior=json.loads((b/'attempt/CLOSED.json').read_text())
    c=dict(schema='r18-fixed-pair-geometry-execution-v1',profile='SYNTHETIC_FIXTURE',runtime_commit='CONSTRUCTED_NOT_SCIENTIFIC',
        base_contract=reference(b/'contract.json'),prior_closed=reference(b/'attempt/CLOSED.json'),prior_records=prior['files']['GT_RECORDS.csv.gz'],
        policy=reference(Path(__file__).parents[1]/'configs/research/r18_geometry_v1.json'),output=str(root/'geometry'),
        runtime_files={n:sha(Path(__file__).with_name(n)) for n in ('r18_geometry_diag.py','r18_geometry_core.py')})
    cfg=root/'contract.json';publish(cfg,c)
    run('r18_geometry_diag.py','check',cfg);assert not Path(c['output']).exists()
    run('r18_geometry_diag.py','execute',cfg)
    out=Path(c['output']);summary=json.loads((out/'SUMMARY.json').read_text())
    assert summary['synthetic'] and summary['selected_GT']==4 and summary['overall']['groups']['center_only']==4
    assert (out/'CLOSED.json').is_file() and not (out/'STOP.json').exists()
    original=sha(out/'SUMMARY.json');run('r18_geometry_diag.py','execute',cfg,1,'FileExistsError');assert sha(out/'SUMMARY.json')==original
    cases=3
    def changed(name,edit):
        nonlocal cases
        x=copy.deepcopy(c);x['output']=str(root/('rejected-'+name));edit(x)
        p=root/('contract-'+name+'.json');publish(p,x);run('r18_geometry_diag.py','check',p,1)
        assert not Path(x['output']).exists();cases+=1
    changed('missing-records',lambda x:x['prior_records'].update(path=str(root/'absent.gz')))
    changed('bad-records-sha',lambda x:x['prior_records'].update(sha256='0'*64))
    changed('profile',lambda x:x.update(profile='REAL_N_548'))
    changed('source',lambda x:x['runtime_files'].update({'r18_geometry_core.py':'0'*64}))
    bad=copy.deepcopy(json.loads(Path(c['policy']['path']).read_text()));bad['limits']['wall_seconds']=1801
    publish(root/'bad-policy.json',bad);changed('policy',lambda x:x.update(policy=reference(root/'bad-policy.json')))
    # Authenticated constructed prior with a changed selected query must STOP during worker.
    with gzip.open(b/'attempt/GT_RECORDS.csv.gz','rt') as f:rs=list(csv.DictReader(f))
    for r in rs:
        if r['arm']=='N' and .5 <= float(r['best_raw_IoU']) < .75:r['best_query']=str((int(r['best_query'])+1)%300)
    with gzip.open(root/'wrong-pairs.gz','wt',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
    pr=copy.deepcopy(prior);pr['files']['GT_RECORDS.csv.gz']=reference(root/'wrong-pairs.gz');publish(root/'wrong-prior.json',pr)
    x=copy.deepcopy(c);x.update(prior_closed=reference(root/'wrong-prior.json'),prior_records=reference(root/'wrong-pairs.gz'),output=str(root/'geometry-STOP'))
    publish(root/'contract-STOP.json',x);run('r18_geometry_diag.py','execute',root/'contract-STOP.json',2)
    assert (Path(x['output'])/'STOP.json').is_file() and not (Path(x['output'])/'CLOSED.json').exists();cases+=1
    publish(root/'CPU_ACCEPTANCE.json',dict(status='PASS',actual_geometry_CLI_cases=cases,algebra_pass=True,synthetic_only=True,GPU=0,model_forward=0,AP_AR=0,python=sys.version.split()[0]))
    if temporary:temporary.cleanup()
    print(json.dumps(dict(status='PASS',cases=cases,output=str(root))))


if __name__=='__main__':main()
