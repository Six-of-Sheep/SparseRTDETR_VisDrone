"""Reduce authenticated slim receipts to tables; never imports a model or evaluator."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from two_of_four_core import PAIRS,screen,select

def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pub(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def table(p,rows):
    with p.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--delivery',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();returned=a.delivery/'returned';run=returned/'attempt-001'
    idx=read(a.delivery/'TERMINAL_RETURN_INDEX.json')
    remote='/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace/experiments/localization-distribution/sahi-two-of-four-20261019-001'
    assert idx['status']=='TERMINAL_VERIFIED' and idx['worker_exited']
    for r in idx['files']:
        p=returned/Path(r['path']).relative_to(remote)
        assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
    s=read(run/'SUMMARY.json');result=read(run/'RESULT.json');rebuild=read(run/'REBUILD.json')
    pairs=['P'+''.join(map(str,p)) for p in PAIRS]
    assert result['complete_pairs']==pairs and rebuild['original_exact_images']==dict(A=548,B=548,C=548)
    gate=read(returned/'CPU_GATE.json');assert gate['tests']==13 and gate['skips']==0 and gate['status']=='PASS'
    for name,r in s['results'].items():
        check=screen(r['metrics'],s['baseline'])
        assert all(r[k]==check[k] for k in check)
    assert select(s['results'])==s['selected']==result['selected']
    old=read(ROOT/'docs/research/abc_results_20261019/SOURCE_SCALARS.json')
    old={r['arm']:r for r in old['records'] if r['collection']=='development548' and r['seed']==0 and r['budget']==600}
    proofs=read(run/'INPUT_PROOFS.json')
    for p in proofs:
        assert p['metric']==old[p['arm']]['original_metrics']
        assert s['baseline'][p['arm']]==old[p['arm']]['COCO_0_1']
    out=a.output;out.mkdir(parents=True,exist_ok=False)
    metrics={**s['baseline'],**{n:r['metrics'] for n,r in s['results'].items()}}
    table(out/'ABSOLUTE_PERCENT.csv',[dict(arm=n,views=1 if n in ('A','B') else 5 if n=='C' else 3,
        **{k:100*v for k,v in m.items()}) for n,m in metrics.items()])
    table(out/'DIFFERENCES_PP.csv',[dict(arm=n,reference=ref,**ds) for n,r in s['results'].items() for ref,ds in r['differences_pp'].items()])
    gs=[]
    for n,r in s['results'].items():
        for name,g in r['gates'].items():
            gs.append(dict(arm=n,gate=name,actual_pp=g.get('actual_gain_pp',g.get('difference_pp')),
                minimum_pp=g.get('minimum_gain_pp',g.get('minimum_pp')),retained_fraction=g.get('retained_fraction',''),passed=g['passed']))
    table(out/'GATES.csv',gs)
    table(out/'FILTER_COUNTS.csv',[
        *[dict(arm=p['arm'],**p['counts']) for p in proofs],
        *[dict(arm=n,**r['spatial_filter_counts']) for n,r in s['results'].items()]])
    official={**{n:r['official_pp'] for n,r in old.items()},**{n:{m:r['official'][m] for m in old['A']['official_pp']} for n,r in s['results'].items()}}
    table(out/'OFFICIAL_PERCENT.csv',[dict(arm=n,**v) for n,v in official.items()])
    table(out/'OFFICIAL_DIFFERENCES_PP.csv',[dict(arm=n,reference=ref,**{m:v[m]-official[ref][m] for m in v})
        for n,v in official.items() if n in pairs for ref in ('A','B','C')])
    seq=read(run/'SEQUENCES.json')['rows'];lookup={(r['arm'],r['sequence'],r['area'],r['iou']):r for r in seq}
    aggregates=defaultdict(list)
    for r in seq:
        ref=lookup[('C',r['sequence'],r['area'],r['iou'])]
        assert r['GT']==ref['GT'] and r['images']==ref['images']
        assert r['covered']==r['gained']+r['retained'] and ref['covered']==r['lost']+r['retained']
        aggregates[(r['arm'],r['area'],r['iou'])].append(r)
    assert len(seq)==9*76*3*2
    table(out/'SEQUENCES.csv',seq)
    ag=[]
    for (arm,area,t),rs in sorted(aggregates.items()):
        assert sum(r['images'] for r in rs)==548 and len(rs)==76
        valid=[r for r in rs if r['GT']>0];gt=sum(r['GT'] for r in rs)
        ag.append(dict(arm=arm,area=area,iou=t,GT=gt,covered=sum(r['covered'] for r in rs),
            gained=sum(r['gained'] for r in rs),lost=sum(r['lost'] for r in rs),
            target_equal_coverage=sum(r['covered'] for r in rs)/gt,
            sequence_equal_coverage=statistics.mean(r['covered']/r['GT'] for r in valid),
            sequences=76,sequences_with_GT=len(valid),zero_GT_sequences=76-len(valid),
            sequences_net_gained=sum(r['gained']>r['lost'] for r in rs),
            sequences_net_lost=sum(r['gained']<r['lost'] for r in rs),
            sequences_net_equal=sum(r['gained']==r['lost'] for r in rs)))
    table(out/'SEQUENCE_AGGREGATES.csv',ag)
    cost=read(run/'COST.json');res=read(run/'RESOURCE.json')
    cpu_log=(returned/'CPU_GATE.log').read_text()
    import re
    seconds=float(re.search(r'Ran 13 tests in ([0-9.]+)s',cpu_log).group(1))
    admission=read(returned/'admission-001/RESULT.json')
    cumulative=dict(CPU_gate_seconds=seconds,admission_seconds=admission['elapsed_seconds'],
        local_11_core_checks_seconds=.032,
        formal_entry_seconds=result['elapsed_seconds'],task_measured_wall_seconds=seconds+admission['elapsed_seconds']+result['elapsed_seconds'],
        accounting_scope='timed server gates/admission/formal subtotal; local core tests separately; development, deployment, CI and uninstrumented checks excluded; not project grand total',
        previous_tasks_costs='preserved separately; not reset and no invented project grand total',
        new_forward_calls=0,new_training_updates=0,new_GPU_calls=0,new_score_groups=6,new_evaluator_calls=12)
    pub(out/'COST.json',cumulative)
    threshold={m:100*(s['baseline']['A'][m]+.8*(s['baseline']['C'][m]-s['baseline']['A'][m])) for m in ('AP','APS')}
    selected=s['selected'];passed=[n for n,r in s['results'].items() if r['passed']]
    text=['# 固定二选四切片可行性筛选','',
        ('**筛选通过：'+', '.join(passed)+'；按预冻结字典序保留'+selected+'，仅供后续另行授权的跨seed和真实测速。**') if selected else '**六种固定组合均未通过，关闭本次固定二选四方案，不扩展搜索。**','',
        'seed0／548 development全76序列／top600。原A/B/C与旧门槛、旧失败配方保持。此结果是development配置筛选，非跨seed确认、独立验证或论文充分性裁决。没有逐图GT选择，没有训练、新forward或GPU测速。','',
        '## 全部结果','',
        'COCO桥接，绝对值为百分数；保留率为(P−A)/(C−A)。未舍入值用于8项筛选；报告小数仅为显示。',
        f"80%要求等价于AP≥{threshold['AP']:.8f}、APS≥{threshold['APS']:.8f}。",'',
        '|方案/角视图|AP|APS|AP75-small|APM|APL|AR-large|AP增益保留%|APS增益保留%|裁决|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for n,m in metrics.items():
        vals='|'.join(f'{100*m[k]:.4f}' for k in ('AP','APS','AP75_small','APM','APL','AR_large'))
        if n in s['results']:
            r=s['results'][n];ap=r['gates']['AP_retention']['retained_fraction']*100;aps=r['gates']['APS_retention']['retained_fraction']*100
            status='PASS' if r['passed'] else 'FAIL: '+', '.join(k for k,v in r['gates'].items() if not v['passed'])
            text.append(f'|{n}|{vals}|{ap:.2f}|{aps:.2f}|{status}|')
        else:text.append(f'|{n}|{vals}|—|—|原结果复用|')
    if selected:
        d=s['results'][selected]['differences_pp']['C']
        text+=['',f"保留{selected}的代价：相对C，APM {d['APM']:+.6f}pp（保护下限−.50pp），AR-large {d['AR_large']:+.6f}pp。AR-large本轮未预注册新硬门槛，不能事后改筛选裁决，但必须完整披露，不能将通过写成所有尺度无损。"]
    text+=['','1/2/3/4分别为左上/右上/左下/右下，所有P均保留整图view0。六组合均为3视图。','',
        '完整13项绝对指标见ABSOLUTE_PERCENT.csv；对A/B/C各项增量见DIFFERENCES_PP.csv；8门槛原值见GATES.csv；过滤前后计数见FILTER_COUNTS.csv。官方主评价器的7项百分数及全部成对增量另列OFFICIAL_PERCENT.csv／OFFICIAL_DIFFERENCES_PP.csv，不和COCO桥接单位混用。','',
        '## 序列差异（描述，不是路由证据）','',
        '固定IoU.50/.75与小/中/大面积层，原raw-ignore后正确类别几何覆盖；不含一对一匹配或序列AP。各层零GT序列保留，序列等权率只在该层有GT的序列计算并明确分母；预测数包含该序列全部类别/尺寸，不能当分层FP数量。', '',
        '|组合|小目标.75相对C新增/丢失GT|净增/净损/相同的序列数（76）|目标等权覆盖%|序列等权覆盖%|',
        '|---|---:|---:|---:|---:|']
    for n in pairs:
        v=next(v for v in ag if v['arm']==n and v['area']=='small' and v['iou']==.75)
        text.append(f"|{n}|{v['gained']}/{v['lost']}|{v['sequences_net_gained']}/{v['sequences_net_lost']}/{v['sequences_net_equal']}|{100*v['target_equal_coverage']:.3f}|{100*v['sequence_equal_coverage']:.3f}|")
    text+=['','全部序列、阈值、面积层见SEQUENCES.csv；54组目标/序列等权汇总见SEQUENCE_AGGREGATES.csv。差异不能证明测试时可无GT识别值得执行的视图；不计算逐图oracle，不追加自适应选择器。','',
        '## 工程与身份','',
        'CPU13项0跳过；真实CLI走通双评价器→RESULT/CLOSED→精简导出→退出，含worker异常和重复拒绝。370文件冻结依赖与421文件执行release逐SHA；548标注、137批抑制前原始缓存逐SHA，原A/B/C三组canonical输入/过滤计数及各548图逐候选精确重建通过。所有六组预测先生成才开始评分；新评分后再次验证canonical输入、原确定性过滤及评价计数。','',
        f"执行commit `{result['execution']['commit']}`；manifest SHA `{result['execution']['run_source_sha256']}`；合同SHA `{result['execution']['contract_sha256']}`。原seed0 e120 EMA SHA `13a35283a02d2b728e3449917586ded5cdf2d578ebe6847510748e0dc4d21757`。",'',
        f"正式入口耗时{result['elapsed_seconds']:.3f}s（{result['elapsed_seconds']/60:.2f}分钟）；服务器CPU门禁+准入+正式阶段计时小计{cumulative['task_measured_wall_seconds']:.3f}s，本地11项核心检查另计.032s；未含开发、部署、CI及未计时检查，是已计时下界而非项目总成本。6组/12评价器调用，训练/forward/GPU=0；最高CPU温度{max(v['CPU_temperature_max'] for v in res):.1f}℃，最低可用RAM{min(v['RAM_available'] for v in res)/1e9:.2f}GB。",'',
        '3/5仅是每图视图数比例，未测实际端到端节省。历史成本原样保留，不将本任务耗时当作推理延迟，也不把CPU缓存评分算作部署成本通过。进程终态与SHA精简回传已核验；原预测、逐图详情和全日志留服务器。','',
        '## 先例与研究决策','',
        'SAHI已覆盖切片/整图及质量成本取舍；AutoFocus与ROI-Gated有选择局部视图先例。后者是COCO128预印本、不是航拍验证。IEEE Access航拍SAHI接受作者稿已覆盖VisDrone上的RT-DETR与尺度/成本取舍。固定二选四是静态预算控制，不称原创；目前查新范围有限，新颖性未知。旧“整图引导二选四”只曾作备用设想，已查本地登记未发现本轮同协议六组合结果。','',
        ('本轮支持将保留方案作为“能否以较少固定视图保留大部分增益”的后续验证对象；当前不能宣称可重复计算节省。先做冻结后的跨seed验证和真实测速，需另行授权；没有理由由当前通过结果直接升级自适应。' if selected else '本轮不支持固定二选四达到预设80%双增益及保护目标，按方案关闭。序列差异本身不足以支持自适应分配，本轮不默认转向选择器或扩展组合搜索。'),'',
        '不访问补充/封存集合，不改旧结果，不合并main。PR工程CI与本次科学筛选分别报告。']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    pub(out/'SOURCE_INDEX.json',dict(execution=result['execution'],parent_sha256='b2ac33e7a99cfa4e8ea3ccce7297f5088272d839f98fae7b7e13c4e2a7f46c28',
        return_index_sha256=sha(a.delivery/'TERMINAL_RETURN_INDEX.json'),summary_sha256=sha(run/'SUMMARY.json'),
        original_scalar_index_sha256=sha(ROOT/'docs/research/abc_results_20261019/SOURCE_SCALARS.json'),
        server_inventory=idx['server_inventory'],returned_files=idx['files'],
        latest_large_NMS_direction='closed; PR11 aa4398c1/88dff81 retained',literature='PR10 PDF_REVIEW_006 / DIRECT_PRECEDENT_DECISION_008; prior SAHI_NEXT_DECISION_20261017'))
    print(json.dumps(dict(status='REPORT_VALIDATED',pairs=6,sequence_rows=len(seq),passed=passed,selected=selected,report=str(out/'REPORT.md'))))

if __name__=='__main__':main()
