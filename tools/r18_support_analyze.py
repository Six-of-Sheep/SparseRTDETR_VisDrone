"""Display already-saved diagnostic scalars. No model/data/evaluator imports."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def describe(values):
    v=[x for x in values if x is not None]
    return dict(total=len(values),defined=len(v),undefined=len(values)-len(v),
                positive=sum(x>0 for x in v),zero=sum(x==0 for x in v),negative=sum(x<0 for x in v),
                mean=statistics.mean(v) if v else None,median=statistics.median(v) if v else None,
                min=min(v) if v else None,max=max(v) if v else None)


def analyze(receipts):
    p=Path(receipts)
    exit=json.loads((p/'EXIT.json').read_text(encoding='utf-8'))
    if exit['status']!='PASS':
        return dict(status='STOP_NO_RETRY',exit=exit,mechanism='UNKNOWN; incomplete run'),[]
    assert json.loads((p/'CLOSED.json').read_text(encoding='utf-8'))['status']=='PASS'
    r=json.loads((p/'WORKER_RESULT.json').read_text(encoding='utf-8'))
    for ref in (r['result'],r['inventory']):
        local=p/Path(ref['path']).name
        assert local.stat().st_size==ref['bytes'] and sha(local)==ref['sha256']
    m=json.loads((p/'MEASUREMENTS.json').read_text(encoding='utf-8'));rows=[dict(zip(m['columns'],v)) for v in m['rows']]
    inv=json.loads((p/'INVENTORY.json').read_text(encoding='utf-8'))
    coverage={};layers=[]
    for arm in ('P','B0'):
        images=[v for v in inv['images'] if v['arm']==arm]
        q=[v for v in inv['queries'] if v['arm']==arm]
        r=[v for v in rows if v['arm']==arm]
        assert len(images)==32 and len({v['id'] for v in images})==32
        coverage[arm]=dict(images=len(images),valid_GT=sum(v['valid_GT'] for v in images),
            small_GT=sum(v['small_GT'] for v in images),matched_GT=sum(v['matched_GT'] for v in images),
            matched_small=sum(v['matched_small'] for v in images),
            unmatched_small=sum(v['small_GT']-v['matched_small'] for v in images),
            ordinary_query_layer_instances=sum(v['ordinary'] for v in q),
            below_cell_query_layer_instances=sum(len(v['below_cell_queries']) for v in q),
            below_cell_unique_image_query=len({(v['image_id'],n) for v in q for n in v['below_cell_queries']}),
            matched_small_query_layer_instances=len(r),
            matched_small_below_cell_instances=sum(v['triggered'] for v in r),
            matched_small_below_cell_unique_image_query=len({(v['image_id'],v['query']) for v in r if v['triggered']}),
            floor_applied_instances=sum(v['floor_applied'] for v in r))
        assert len(r)==coverage[arm]['matched_small']*3
        for layer in range(3):
            for group in ('all_matched_small','below_cell_matched_small'):
                subset=[v for v in r if v['layer']==layer and (group=='all_matched_small' or v['triggered'])]
                layers.append(dict(arm=arm,layer=layer,group=group,n=len(subset),
                    actual_I=describe([v['actual_I'] for v in subset]),native_I=describe([v['native_I'] for v in subset]),
                    actual_O=describe([v['actual_O'] for v in subset]),native_O=describe([v['native_O'] for v in subset]),
                    actual_V=describe([v['actual_V'] for v in subset]),native_V=describe([v['native_V'] for v in subset]),
                    actual_M=describe([v['actual_M'] for v in subset]),native_M=describe([v['native_M'] for v in subset]),
                    delta_I=describe([v['delta_I'] for v in subset]),delta_O=describe([v['delta_O'] for v in subset]),
                    delta_V=describe([v['delta_V'] for v in subset]),delta_M=describe([v['delta_M'] for v in subset]),
                    stencil_changed=sum(v['stencil_changed'] for v in subset),
                    effective_interpolation_changed=sum(v['effective_interpolation_changed'] for v in subset),
                    local_core_output_abs_mean=describe([v['local_core_output_abs_mean'] for v in subset])))
    matched=lambda arm:{(v['image_id'],v['gt_id']) for v in rows if v['arm']==arm}
    return dict(status='PASS',coverage=coverage,layers=layers,
        common_matched_small_GT=len(matched('P')&matched('B0')),P_only_small_GT=len(matched('P')-matched('B0')),
        B0_only_small_GT=len(matched('B0')-matched('P')),peaks=inv['peaks'],state_checks=inv['state_checks'],
        forward_calls=16,GPU_stage_seconds=exit['GPU_stage_seconds'],
        DN='UNKNOWN: eval has no DN; not reconstructed',training_cause='UNKNOWN',AP='NOT_RECOMPUTED'),rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--receipts',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    result,rows=analyze(a.receipts)
    (out/'SUMMARY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    if not rows:
        return
    with (out/'PAIRED.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# 32图终点诊断原始数值展示','',
           'I/O/V以全部最细attention质量A为分母，M以有效质量Z为分母；差值为P实际减同状态去下限。',
           'B0是自身原生分布背景，其配对差值结构为零，不是与P同状态。无AP/训练根因裁决。','',
           '|模型|层|组|配对数|ΔI均值|ΔO均值|ΔV均值|ΔM均值|ΔM正/零/负/未定义|有效插值变化数|局部输出非零数|',
           '|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|']
    fmt=lambda x:'UNDEFINED' if x is None else f'{x:.9f}'
    for v in result['layers']:
        m=v['delta_M']
        lines.append('|'+ '|'.join(map(str,[v['arm'],v['layer'],v['group'],v['n'],
            fmt(v['delta_I']['mean']),fmt(v['delta_O']['mean']),fmt(v['delta_V']['mean']),fmt(m['mean']),
            f"{m['positive']}/{m['zero']}/{m['negative']}/{m['undefined']}",v['effective_interpolation_changed'],
            v['local_core_output_abs_mean']['positive']]))+'|')
    (out/'TABLES.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
