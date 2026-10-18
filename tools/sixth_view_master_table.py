"""Compile existing small endpoint records, never recalculate AP or read models."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

def build(project,out):
    base=project/'workspace_management/delivery/experiments/localization-distribution'
    inputs=[]
    def read(rel):
        p=base/rel;raw=p.read_bytes();inputs.append({'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
        return json.loads(raw)
    vote=read('sahi-box-vote-fixed-20261018-001/returned/production-001/RESULT.json')
    soft=read('sahi-softnms-fixed-20261017-001/revision-007/returned/production-20261018-004/RESULT.json')
    ft=read('r18-slice-finetune-20261018-001/execution-001/terminal/job/evaluation/COMPARISON.json')
    costs=read('sahi-sixth-view-20261018-001/EXISTING_COST_INDEX.json')
    assert not vote['quality_passed'] and not soft['quality_passed'] and ft['quality']=='FAIL'
    costmap={(r['seed'],r['arm']):r for r in costs['rows']}
    rows=[]
    reports={'ABC':'r18-sahi-seed12-20261016-001/protocol-resume-20261017-001/terminal-review-20261018-001/summary/REPORT.md',
             'Soft-NMS':'sahi-softnms-fixed-20261017-001/revision-007/REPORT.md',
             'FT':'r18-slice-finetune-20261018-001/execution-001/terminal/REPORT.md',
             'Vote':'sahi-box-vote-fixed-20261018-001/REPORT.md'}
    def add(family,seed,arm,scores,quality,cost,source,scope):
        assert (base/reports[family]).is_file()
        for budget,ms in scores.items():
            rows.append({'family':family,'seed':seed,'seed_scope':scope,'arm':arm,'budget':int(budget),
               'quality':quality,'cost':cost,'source':source,'report':reports[family],
               'COCO_bridge_original_0_1':ms})
    for s in vote['seed_results']:
        seed=s['seed']
        for arm in ('A','B','C'):
            c=costmap[seed,arm];reference=costmap[seed,'A']['e2e_total_seconds']
            add('ABC',seed,arm,s['scores'][arm],{'A':'REFERENCE','B':'CONTROL_NO_INDEPENDENT_PASS_CLAIM','C':'PASS_C_MINUS_A_SEVEN'}[arm],
               {'measured':True,'same_seed_64_e2e_seconds':c['e2e_total_seconds'],'over_A':c['e2e_total_seconds']/reference,
                'C_gate_passed':c['e2e_total_seconds']/reference<=6 if arm=='C' else None,
                'peak_reserved_bytes':c['peak_reserved_bytes'],'record_ref':{k:c[k] for k in ('path','bytes','sha256')}},
               {'runtime_commit':None if seed==0 else 'bcbda80b3f3af7250d2b47bea014f85b76a3f86c',
                'source_manifest_sha256':c['source_sha256'],'checkpoint_sha256':c['checkpoint_sha256'],
                'seed0_commit_note':'legacy source manifest, not inventing an original launch commit'},'three independent e120 seeds')
        for arm in ('V-whole','V-five'):
            add('Vote',seed,arm,s['scores'][arm],'CONTROL_ONLY' if arm=='V-whole' else 'FAIL_CLOSED',
                {'measured':False,'reason':'quality gate failed; GPU cost not executed'},
                {'runtime_commit':'f563dfb91d87e13e5a7287ea472049e1deb45810'},'same three e120 caches; no retraining')
    for s in soft['seed_results']:
        for arm in ('E','D'):
            add('Soft-NMS',s['seed'],arm,s['scores'][arm],'CONTROL_ONLY' if arm=='E' else 'FAIL_CLOSED',
                {'measured':False,'reason':'end-to-end GPU not measured'},
                {'runtime_commit':'c4a8f124d655fa6c4d5a20506b0fb191f49f2c18'},'same three e120 caches; no retraining')
    for arm,by_view in ft['scores'].items():
        for view,values in by_view.items():
            add('FT',0,arm+'/'+view,values,'CONTROL_ONLY' if arm=='T-control' else 'FAIL_CLOSED',
                {'measured':True,'protocol_cost_passed':ft['cost_passed'],'arm_cost':ft['cost'][arm]},
                {'runtime_commit':'57ae4d1990ef059818825e6d8813435b04046505'},'seed0 only; two paired 12-pass arms, not multi-seed')
    result={'status':'EXISTING_RESULTS_COMPILED','no_new_AP':True,'no_new_bootstrap':True,'rows':rows,'input_refs':inputs,
            'limits':'repeated development; original scientific decisions unchanged; A/B controls not promoted; FT only seed0'}
    out.mkdir(parents=True,exist_ok=False)
    (out/'MASTER.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    with (out/'MASTER.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['family','seed','arm','budget','AP_percent','APS_percent','AP75_small_percent','APM_percent','APL_percent','quality','cost_measured','source_commit','report'])
        for r in rows:
            w.writerow([r['family'],r['seed'],r['arm'],r['budget'],*[r['COCO_bridge_original_0_1'][m]*100 for m in ('AP','APS','AP75_small','APM','APL')],r['quality'],r['cost']['measured'],r['source']['runtime_commit'],r['report']])
    text=['# 当前实验主表（原记录汇总，不重评分）','','原C保留，所有已关闭配方不重开。原共同协议适配R18三seed；同缓存的Soft-NMS/投票不是新训练seed。微调仅seed0双臂。完整两预算分数、成本/源码/报告与小文件SHA见MASTER.json/CSV。','',
          '| 系列 | seed | 主600方案 | AP% | APS% | 质量状态 | 成本 |','|---|---:|---|---:|---:|---|---|']
    for r in rows:
        if r['budget']!=600:continue
        cost=f"{r['cost']['over_A']:.6f}×A（实测）" if r['family']=='ABC' else ('协议成本PASS（实测）' if r['cost']['measured'] else '未测/未裁决')
        text.append(f"| {r['family']} | {r['seed']} | {r['arm']} | {r['COCO_bridge_original_0_1']['AP']*100:.6f} | {r['COCO_bridge_original_0_1']['APS']*100:.6f} | {r['quality']} | {cost} |")
    text+=['','A是参考而非新增方法PASS；B未按C的七项政策独立通过，不追认。原C逐seed质量及≤6成本通过。Soft-NMS42项7FAIL；框投票42项21FAIL；固定微调全局FAIL、成本PASS。未测速不写成本FAIL或PASS。本轮新C-center/C-flip均未GPU/评分，质量和成本均未评价。','',
           '源码：原seed0仅原manifest身份（2c9c37...），不伪造原启动commit；seed1/2实际release bcbda80b...及eee776...，与后续观察器/文档head区别。Soft-NMS c4a8f124...、FT57ae4d19...、投票f563dfb9...；完整SHA见JSON。','',
           '报告索引：']+[f'- {k}: {v}' for k,v in reports.items()]
    (out/'MASTER.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    x=build(a.project,a.output);print(json.dumps({'status':x['status'],'rows':len(x['rows']),'old_scores_recomputed':0}))
