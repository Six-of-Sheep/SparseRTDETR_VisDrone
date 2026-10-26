"""Actual CPU CLI acceptance on constructed full-query caches; no detectors."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import io

from r18_cache_diag import LIMITS, reference, publish, sha


def fixture(root, native_root, geometry_fixture=False):
    import torch
    from r18_cache_worker import native
    c = dict(schema='r18-cache-diagnostic-v1', profile='SYNTHETIC_FIXTURE', limits=LIMITS,
             arms_order=['N','U','D'], arms={}, images=4, thresholds=[.5,.75], budgets=[600,300],
             no_GPU_forward_training_AP_AR=True, supplemental1602=False, native_root=str(native_root),
             native_commit='e50bedbe8e41bdfd3c47d09f7076d9644e438e26', output=str(root/'attempt'))
    source = []
    for name in ('src/sparse_rtdetr/r0/evaluation.py','src/sparse_rtdetr/localization/data.py',
                 'src/sparse_rtdetr/baseline/primary_evaluator.py','vendor/rtdetrv2_pytorch/src/zoo/rtdetr/matcher.py'):
        p = native_root/name
        source.append(dict(path=name, bytes=p.stat().st_size, sha256=sha(p)))
    publish(root/'native.json',dict(commit=c['native_commit'],source_files=source))
    c['native_manifest'] = reference(root/'native.json')
    ann = root/'development/annotations'
    ann.mkdir(parents=True)
    c['annotation_root'] = str(ann)
    rows = []
    raw = b'20,30,8,8,1,1,0,0\n60,60,16,16,1,1,0,0\n150,80,50,50,1,2,0,0\n260,130,100,100,1,3,0,0\n2,2,6,6,0,0,0,0\n80,30,10,10,0,1,0,0\n'
    for i in range(4):
        p = ann/f'constructed_{i}.txt'
        p.write_bytes(raw)
        rows.append(dict(stable_image_id=hashlib.sha256(f'CONSTRUCTED/{i}'.encode()).hexdigest(), split='val',
                         sequence_key=f'CONSTRUCTED_SEQ_{i//2}', width=384, height=256,
                         annotation_relative_path='val/annotations/'+p.name, annotation_sha256=sha(p),
                         annotation_size_bytes=len(raw),raw_annotation_rows=6))
    publish(root/'members.json',dict(records=rows))
    c['manifest']=reference(root/'members.json')
    publish(root/'oldSTOP.json',dict(status='SYNTHETIC_HISTORICAL_STOP'))
    c['original_N_STOP']=reference(root/'oldSTOP.json')
    c['original_N_CLOSED']=str(root/'never-created-old-CLOSED.json')
    torch, parse, _, rank, primary, ImageRow, GTRow, _, _ = native(c)
    for arm in c['arms_order']:
        base=root/('infer-B0' if arm=='N' else 'infer-'+arm)
        base.mkdir()
        cp=root/(arm+'-checkpoint-fixture.pth')
        cp.write_bytes(b'CONSTRUCTED_METADATA_NOT_A_MODEL_'+arm.encode())
        binding=dict(arm='r18-sampling-span-v1/B0' if arm=='N' else arm, config=dict(input_hw=[768,1344],seed=0),
                     contract_sha256='CONSTRUCTED',source_sha256=c['native_manifest']['sha256'])
        publish(base/'binding.json',binding)
        publish(base/'endpoint.json',dict(checkpoint=dict(path=str(cp),size_bytes=cp.stat().st_size,sha256=sha(cp)),
                    binding=reference(base/'binding.json'),epoch=120,updates=36480,EMA=True,used_acceptance_state=False))
        logits=torch.full((4,300,10),-9.,dtype=torch.float32)
        boxes=torch.zeros((4,300,4),dtype=torch.float32)
        boxes[:]=torch.tensor([.9,.9,.01,.02])
        for q,b,cat,z in [(0,[24,34,8,8],0,5.),(1,[72,68,16,16],0,3.),
                           (2,[175,105,50,50],1,4.),(3,[310,180,100,100],2,6.),(299,[68,68,16,16],0,-12.)]:
            boxes[:,q]=torch.tensor(b)/torch.tensor([384,256,384,256])
            logits[:,q,cat]=z
        if arm=='U':
            boxes[:,0,0] += 2/384
        if geometry_fixture and arm=='N':
            boxes[:,0,0] += 2/384  # constructed .50--.75 population, never real caches
        cache=dict(ids=[r['stable_image_id'] for r in rows],logits=logits,boxes=boxes)
        if arm!='N':
            cache['coarse']=boxes.clone()
        torch.save(cache,base/'raw-000.pth')
        result=dict(status='PASS',images=4,EMA=True,forward_calls=1,raw_batches=[reference(base/'raw-000.pth')],predictions={},contract_sha256='CONSTRUCTED')
        scores={}
        for budget in (600,300):
            pred=[]
            table=[[[[0,0,0,0] for _ in range(4)] for _ in range(10)] for _ in range(10)]
            for i,row in enumerate(rows):
                ranked=rank(logits[i],boxes[i],row,budget)
                pred.extend(ranked)
                gt=parse(raw,row['stable_image_id'],expected_rows=6)
                image=ImageRow(row['stable_image_id'],384,256)
                regions=[GTRow(g['id'],row['stable_image_id'],0,tuple(g['box']),g['raw_area'],True,False) for g in gt['rows'] if g['category']==0]
                integral=primary._build_integral(image,regions)
                kept=[p for p in ranked if primary._ignored_fraction(integral,image,tuple(p['bbox_xyxy']))<.5][:500]
                for t,ti in ((.5,0),(.75,5)):
                    for cat in range(1,11):
                        gs=[g for g in gt['rows'] if g['category']==cat and primary._ignored_fraction(integral,image,tuple(g['box']))<.5]
                        ds=[p for p in kept if p['category_id']==cat]
                        gtstate,detstate=primary._match([(p['score'],primary._official_xywh(tuple(p['bbox_xyxy']))) for p in ds],
                                                       [(primary._gt_xywh(tuple(g['box'])),g['score']==0) for g in gs],t)
                        v=[len(gtstate),sum(s==1 for s in detstate),sum(s==0 for s in detstate),sum(s==-1 for s in detstate)]
                        table[cat-1][ti][3]=[a+b for a,b in zip(table[cat-1][ti][3],v)]
            publish(base/f'predictions-{budget}.json',pred)
            result['predictions'][str(budget)]=reference(base/f'predictions-{budget}.json')
            publish(base/f'score-{budget}.json',dict(official=dict(match_counts_by_class_iou_max_dets=table)))
            scores[str(budget)]=reference(base/f'score-{budget}.json')
        publish(base/'RESULT.json',result)
        publish(base/'CLOSED.json',dict(status='PASS',result=reference(base/'RESULT.json')))
        c['arms'][arm]=dict(endpoint=reference(base/'endpoint.json'),binding=reference(base/'binding.json'),
                            checkpoint=reference(cp),infer_result=reference(base/'RESULT.json'),
                            infer_closed=reference(base/'CLOSED.json'),scores=scores)
    c['runtime_files']={n:sha(Path(__file__).parent/n) for n in ('r18_cache_diag.py','r18_cache_worker.py','r18_cache_core.py')}
    publish(root/'contract.json',c)
    return c


def run(args, root, wanted=0, message=None):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
             OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    result=subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('r18_cache_diag.py')),*args],
                          env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    (root/f'cli-{len(list(root.glob("cli-*.txt")))}.txt').write_text(result.stdout+result.stderr)
    if (result.returncode==0) != (wanted==0) or (message and message not in result.stdout+result.stderr):
        raise AssertionError(f'actual CLI expected {wanted}: {result.returncode}\n{result.stdout}\n{result.stderr}')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-root')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    root=Path(args.output).resolve()
    root.mkdir(exist_ok=False,parents=False)
    base=root/'fixture'
    base.mkdir()
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    exported = None
    if args.native_root:
        native_root = Path(args.native_root).resolve()
    else:
        exported = tempfile.TemporaryDirectory(prefix='r18-native-source-')
        native_root = Path(exported.name)
        raw = subprocess.check_output(['git','archive','e50bedbe8e41bdfd3c47d09f7076d9644e438e26'],cwd=Path(__file__).resolve().parents[1])
        with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
            archive.extractall(native_root)
    c=fixture(base,native_root)
    cfg=base/'contract.json'
    argv=['--contract',str(cfg),'--sha256',sha(cfg)]
    run(['check',*argv],root)
    assert not Path(c['output']).exists()
    run(['execute',*argv],root)
    out=Path(c['output'])
    result=json.loads((out/'RESULT.json').read_text())
    assert result['images_per_arm']==4 and result['native_matcher_CPU_calls']==3 and result['GPU_calls']==0
    assert (out/'CLOSED.json').is_file() and not (out/'STOP.json').exists()
    before=sha(out/'RESULT.json')
    run(['execute',*argv],root,wanted=1,message='FileExistsError')
    assert sha(out/'RESULT.json')==before
    run(['check','--contract',str(cfg),'--sha256','0'*64],root,wanted=1,message='contract SHA differs')
    cases=3
    def changed(label,edit):
        nonlocal cases
        x=copy.deepcopy(c)
        x['output']=str(base/('attempt-'+label))
        edit(x)
        p=base/('contract-'+label+'.json');publish(p,x)
        run(['check','--contract',str(p),'--sha256',sha(p)],root,wanted=1)
        assert not Path(x['output']).exists()
        cases+=1
    changed('missing',lambda x:x['arms']['N']['checkpoint'].update(path=str(base/'missing.pth')))
    changed('bad-sha',lambda x:x['arms']['N']['checkpoint'].update(sha256='0'*64))
    changed('wrong-endpoint',lambda x:x['arms']['N'].update(endpoint=x['arms']['U']['endpoint']))
    changed('nondev-root',lambda x:x.update(annotation_root=str(base/'holdout/annotations')))
    changed('budget',lambda x:x['limits'].update(wall_seconds=1801))
    changed('scope',lambda x:x.update(supplemental1602=True))
    changed('source',lambda x:x['runtime_files'].update({'r18_cache_worker.py':'0'*64}))
    members=json.loads((base/'members.json').read_text())
    members['records'][1]['stable_image_id']=members['records'][0]['stable_image_id']
    publish(base/'duplicate-members.json',members)
    changed('duplicate-ID',lambda x:x.update(manifest=reference(base/'duplicate-members.json')))
    members=json.loads((base/'members.json').read_text())
    members['records'][1]['split']='train'
    publish(base/'wrong-role-members.json',members)
    changed('nondev-member',lambda x:x.update(manifest=reference(base/'wrong-role-members.json')))
    # Shape corruption is authenticated as a constructed fixture, then rejected by actual worker.
    import torch
    rawp=base/'infer-B0/raw-000.pth'
    payload=torch.load(rawp,map_location='cpu',weights_only=True)
    payload['logits']=payload['logits'][:,:299]
    torch.save(payload,rawp)
    rpath=base/'infer-B0/RESULT.json';r=json.loads(rpath.read_text());r['raw_batches']=[reference(rawp)]
    rpath.write_text(json.dumps(r)+'\n')
    clpath=base/'infer-B0/CLOSED.json';clpath.write_text(json.dumps(dict(status='PASS',result=reference(rpath)))+'\n')
    bad=copy.deepcopy(c);bad['output']=str(base/'attempt-bad-shape')
    bad['arms']['N']['infer_result']=reference(rpath);bad['arms']['N']['infer_closed']=reference(clpath)
    badp=base/'contract-bad-shape.json';publish(badp,bad)
    run(['execute','--contract',str(badp),'--sha256',sha(badp)],root,wanted=2)
    assert (Path(bad['output'])/'STOP.json').is_file() and not (Path(bad['output'])/'CLOSED.json').exists()
    cases+=1
    from r18_cache_core import overlap,stages,cardinality
    import numpy as np
    assert cardinality(np.array([[True,True],[True,False]]))==2
    assert overlap([[0,0,8,8]],[[1,0,9,8]])[0,0]==7/9
    publish(root/'CPU_ACCEPTANCE.json',dict(status='PASS',actual_CLI_cases=cases,
             synthetic_only=True,detector_forward=0,GPU=0,production_input_used=False,
             python=sys.version.split()[0],torch=torch.__version__))
    if exported is not None:
        exported.cleanup()
    print(json.dumps(dict(status='PASS',actual_CLI_cases=cases,output=str(root))))


if __name__=='__main__':
    main()
