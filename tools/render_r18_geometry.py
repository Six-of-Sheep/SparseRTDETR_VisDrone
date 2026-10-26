"""Render authenticated precomputed geometric summaries. No analysis of cache payloads."""
import argparse
import csv
import json
from pathlib import Path
from r18_cache_diag import sha, require, publish, reference
from r18_geometry_core import GROUPS, QUANTILES


def render(summary,digest,out):
    require(sha(summary)==digest,'summary SHA differs')
    s=json.loads(summary.read_text());require(not s['synthetic'] and s['selected_GT']==10503,'not original real population')
    require(s['prior_population_exact'] and s['AP_AR_calls']==s['model_forward_calls']==s['GPU_calls']==0,'scope differs')
    out.mkdir(exist_ok=False)
    rows=[('overall','all',s['overall'])]+[(dim,k,v) for dim,d in s['groups'].items() for k,v in d.items()]
    with (out/'GEOMETRY_GROUPS.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['dimension','key','denominator','images',*GROUPS,*[g+'_fraction' for g in GROUPS]])
        for dim,k,v in rows:w.writerow([dim,k,v['denominator'],v.get('images',0),*[v['groups'][g] for g in GROUPS],*[v.get('proportions',{}).get(g,'UNKNOWN') for g in GROUPS]])
    with (out/'GEOMETRY_DISTRIBUTIONS.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['dimension','key','metric','denominator','mean',*[str(q) for q in QUANTILES]])
        for dim,k,v in rows:
            for metric,d in v['distributions'].items():w.writerow([dim,k,metric,d['n'],d['mean'],*[d['quantiles'][str(q)] for q in QUANTILES]])
    with (out/'GEOMETRY_CONCENTRATION.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['dimension','excluded_key','remaining_denominator',*GROUPS])
        for dim,d in s['concentration'].items():
            for k,v in d['leave_one_group_out'].items():w.writerow([dim,k,v['denominator'],*[v['groups'][g] for g in GROUPS]])
    lines=['# N固定框几何敏感性表','',
        '真实548图缓存；10503个B小GT；同一原始最大IoU框，完全并列取最小query ID。GT辅助配对，不是检测匹配或可部署选框。','',
        '|互斥组|数量|分母|比例|','|---|---:|---:|---:|']
    labels=dict(center_only='仅修中心可达0.75',size_only='仅修尺寸可达0.75',both_individually='两者单独均可',neither_individually='两者单独均不可')
    v=s['overall']
    for g in GROUPS:lines.append(f"|{labels[g]}|{v['groups'][g]}|{v['denominator']}|{v['proportions'][g]*100:.4f}%|")
    lines+=['','|量|均值|p05|p25|p50|p75|p95|','|---|---:|---:|---:|---:|---:|---:|']
    for m,d in v['distributions'].items():
        lines.append('|'+m+'|'+f"{d['mean']:.8f}|"+'|'.join(f"{d['quantiles'][str(q)]:.8f}" for q in (.05,.25,.5,.75,.95))+'|')
    lines+=['','全类别、既有序列、面积分箱及min/max见三个CSV；空分母为UNKNOWN。',
        '两种单独改善不可相加当贡献；负值与交互保留。联合替换为1只是计算检查。',
        '不推断可学习修复、AP收益或训练根因，不自动恢复H2或选择回归损失。','']
    (out/'GEOMETRY_TABLES.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    refs=[dict(name=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir())]
    publish(out/'OUTPUT_INDEX.json',dict(schema='r18-geometry-display-v1',source=reference(summary),files=refs,AP_AR_recomputed=False))
    print(json.dumps(dict(status='PASS',files=len(refs),output=str(out))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--summary',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();render(a.summary,a.sha256,a.output)
