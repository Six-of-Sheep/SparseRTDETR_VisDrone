"""Build paper artifacts from authenticated scalar summaries only.

No model, image, prediction, evaluator, CUDA, SSH or training dependency.
Each collection stays separate; SD is descriptive sample SD across3 seeds.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

METRICS=('AP','AP50','AP75','APS','APM','APL','AP75_small','AR1','AR10','AR100','AR_small','AR_medium','AR_large')
COLLECTIONS=('development548','supplemental1602_history_unverified')
def run(source, output):
    b=source.read_bytes();s=json.loads(b);output.mkdir(parents=True,exist_ok=True)
    assert s['new_model_calls']==s['new_AP_calls']==0 and s['no_pooled_AP'] is True
    raw=s['records'];assert len(raw)==36
    by={(r['collection'],r['seed'],r['arm'],r['budget']):r for r in raw}
    assert len(by)==36
    absolute=[];paired=[];means=[];delta_stats=[];gates=[]
    official_keys=sorted({k for r in raw for k in r['official_pp']})
    for r in raw:
        assert set(r['COCO_0_1'])==set(METRICS)
        assert all(0<=v<=1 for v in r['COCO_0_1'].values())
        absolute.append({k:r[k] for k in ('collection','images','seed','arm','budget')} | {k+'_percent':r['COCO_0_1'][k]*100 for k in METRICS} | {'official_'+k+'_pp':r['official_pp'].get(k,'NOT_IN_CERTIFIED_SUMMARY') for k in official_keys})
    for col in COLLECTIONS:
        for budget in (600,300):
            for arm in 'ABC':
                for key in METRICS:
                    vals=[by[col,i,arm,budget]['COCO_0_1'][key]*100 for i in (0,1,2)]
                    means.append(dict(collection=col,budget=budget,arm=arm,metric=key,mean_percent=statistics.mean(vals),sample_SD_percent=statistics.stdev(vals),minimum_percent=min(vals),maximum_percent=max(vals),n=3))
            for new,old in (('C','A'),('B','A'),('C','B')):
                rows=[]
                for seed in (0,1,2):
                    x,y=by[col,seed,new,budget],by[col,seed,old,budget]
                    row=dict(collection=col,budget=budget,comparison=new+'-'+old,seed=seed)
                    row.update({k:(x['COCO_0_1'][k]-y['COCO_0_1'][k])*100 for k in METRICS})
                    row.update({'official_'+k+'_pp':x['official_pp'][k]-y['official_pp'][k] if k in x['official_pp'] and k in y['official_pp'] else 'NOT_IN_CERTIFIED_SUMMARY' for k in official_keys})
                    paired.append(row);rows.append(row)
                for key in METRICS:
                    vals=[x[key] for x in rows]
                    delta_stats.append(dict(collection=col,budget=budget,comparison=new+'-'+old,metric=key,seed0_pp=vals[0],seed1_pp=vals[1],seed2_pp=vals[2],mean_pp=statistics.mean(vals),sample_SD_pp=statistics.stdev(vals),worst_pp=min(vals),n=3))
        for seed in (0,1,2):
            for name,minimum in s['quality_minimum_pp'].items():
                budget,key=name.split('_',1)
                row=next(x for x in paired if x['collection']==col and x['seed']==seed and x['budget']==int(budget) and x['comparison']=='C-A')
                gates.append(dict(collection=col,seed=seed,condition=name,delta_pp=row[key],minimum_pp=minimum,passed=row[key]>=minimum))
    assert len(gates)==42 and all(x['passed'] for x in gates)
    def write_csv(name, rows):
        with (output/name).open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    official_means=[];official_deltas=[]
    for col in COLLECTIONS:
        for budget in (600,300):
            available=set.intersection(*(set(by[col,i,a,budget]['official_pp']) for i in (0,1,2) for a in 'ABC'))
            for key in sorted(available):
                for arm in 'ABC':
                    vals=[by[col,i,arm,budget]['official_pp'][key] for i in (0,1,2)]
                    official_means.append(dict(collection=col,budget=budget,arm=arm,metric=key,mean_original_pp=statistics.mean(vals),sample_SD_original_pp=statistics.stdev(vals),n=3))
                for new,old in (('C','A'),('B','A'),('C','B')):
                    vals=[by[col,i,new,budget]['official_pp'][key]-by[col,i,old,budget]['official_pp'][key] for i in (0,1,2)]
                    official_deltas.append(dict(collection=col,budget=budget,comparison=new+'-'+old,metric=key,seed0_pp=vals[0],seed1_pp=vals[1],seed2_pp=vals[2],mean_pp=statistics.mean(vals),sample_SD_pp=statistics.stdev(vals),worst_pp=min(vals),n=3))
    write_csv('OFFICIAL_MEAN_SD_PP.csv',official_means);write_csv('OFFICIAL_PAIRED_MEAN_SD_PP.csv',official_deltas)
    write_csv('ABSOLUTE_ALL.csv',absolute);write_csv('ABSOLUTE_MEAN_SD.csv',means)
    write_csv('PAIRED_ALL_PP.csv',paired);write_csv('PAIRED_MEAN_SD_PP.csv',delta_stats);write_csv('FROZEN_QUALITY_GATES.csv',gates)
    costrows=[]
    for r in s['benchmarks']:
        costrows.append({k:v for k,v in r.items() if k!='original_ref'})
    write_csv('COST_FIXED64.csv',costrows)
    coststats=[]
    for measurement in sorted({x['measurement'] for x in costrows}):
        for arm in 'ABC':
            rows=[x for x in costrows if x['measurement']==measurement and x['arm']==arm]
            for key in ('seconds_64','over_A'):
                vals=[x[key] for x in rows];coststats.append(dict(measurement=measurement,arm=arm,quantity=key,mean=statistics.mean(vals),sample_SD=statistics.stdev(vals),worst_max=max(vals),n=3))
    write_csv('COST_MEAN_SD.csv',coststats)
    candidates=[]
    for x in s['prior_candidate_rows']:
        candidates.append(dict(family=x['family'],seed=x['seed'],arm=x['arm'],budget=x['budget'],quality=x['quality'],cost_status='MEASURED' if x['cost'].get('measured') else 'NOT_MEASURED',AP_percent=x['COCO_bridge_original_0_1']['AP']*100,APS_percent=x['COCO_bridge_original_0_1']['APS']*100,AP75_small_percent=x['COCO_bridge_original_0_1']['AP75_small']*100,APM_percent=x['COCO_bridge_original_0_1']['APM']*100,APL_percent=x['COCO_bridge_original_0_1']['APL']*100,report=x['report']))
    write_csv('CLOSED_CANDIDATES.csv',candidates)
    main=['# A/B/C实验主表（冻结整理）','','A为共同协议基线，B为常规后处理控制，C为保留的SAHI式实用方案。两集合分别汇总，不合并AP；所有数值来自认证标量，未重跑模型或AP。','',
          '百分数用于绝对分数，pp用于成对差；均值±样本SD仅描述3个训练seed，非CI。补充集历史使用未完全核实，独立资格仍BLOCKED。','']
    for col in COLLECTIONS:
        main+=['## '+col,'','|预算|方案|AP均值±SD|APS均值±SD|AP75-small均值±SD|APM均值±SD|APL均值±SD|','|---|---|---:|---:|---:|---:|---:|']
        for budget in (600,300):
            for arm in 'ABC':
                cells=[]
                for key in ('AP','APS','AP75_small','APM','APL'):
                    v=next(x for x in means if x['collection']==col and x['budget']==budget and x['arm']==arm and x['metric']==key)
                    cells.append(f"{v['mean_percent']:.4f} ± {v['sample_SD_percent']:.4f}")
                main+=['|'+str(budget)+'|'+arm+'|'+'|'.join(cells)+'|']
        main+=['','|预算|比较|指标|seed0|seed1|seed2|均值|样本SD|最差|','|---|---|---|---:|---:|---:|---:|---:|---:|']
        for row in delta_stats:
            if row['collection']==col:
                main+=['|'+str(row['budget'])+'|'+row['comparison']+'|'+row['metric']+'|'+'|'.join(f'{row[k]:+.6f}' for k in ('seed0_pp','seed1_pp','seed2_pp','mean_pp','sample_SD_pp','worst_pp'))+'|']
    main+=['','## 成本口径','','COST_FIXED64.csv为两轮development固定64工程样本的匹配端到端测量，均batch4/FP32/包含真实视图和后处理；不是548或1602全量吞吐，页缓存未清理，不是冷启动。成本是两个输出预算共享的端到端流程，不把两档算两次测量。','',
           '|测量|seed|A秒|B秒|C秒|C/A|','|---|---|---:|---:|---:|---:|']
    for measure in sorted({x['measurement'] for x in costrows}):
        for seed in (0,1,2):
            rr={x['arm']:x for x in costrows if x['measurement']==measure and x['seed']==seed}
            main+=['|'+measure+'|'+str(seed)+'|'+'|'.join(f"{rr[a]['seconds_64']:.6f}" for a in 'ABC')+'|'+f"{rr['C']['over_A']:.6f}"+'|']
    main+=['','完整逐seed两预算绝对值含全部AP/AR和主评价器字段见ABSOLUTE_ALL.csv，成对差/样本SD见PAIRED_*。认证摘要未提供的主评价器AR字段明确标记NOT_IN_CERTIFIED_SUMMARY，不用零填充。','',
           '42项C−A质量检查全部通过，不给B追认同一质量资格。C−B APL下降及C−A AR-large下降逐seed完整保留；未用均值抵消失败。','',
           '旧候选完整数字见CLOSED_CANDIDATES.csv；Soft-NMS与框投票质量FAIL、GPU端到端成本未测；固定微调仅seed0质量FAIL/成本PASS；中心第六视图仅seed0质量FAIL/成本PASS，flip保持控制，不升格。所有旧实验原件/源码/失败/STOP/成本不改，所有扩展到此结束，默认无下一训练。']
    (output/'MASTER.md').write_text('\n'.join(main)+'\n',encoding='utf-8')
    tex=['% Existing scalar scores; mean +/- sample SD, not confidence intervals.',r'\begin{tabular}{llrrrrr}',r'Dataset/budget & Arm & AP & APS & AP75-small & APM & APL \\',r'\hline']
    for col in COLLECTIONS:
        for budget in (600,300):
            for arm in 'ABC':
                cells=[]
                for key in ('AP','APS','AP75_small','APM','APL'):
                    v=next(x for x in means if x['collection']==col and x['budget']==budget and x['arm']==arm and x['metric']==key)
                    cells.append(f"${v['mean_percent']:.3f} \\pm {v['sample_SD_percent']:.3f}$")
                tex.append(('Dev548' if col==COLLECTIONS[0] else 'Supp1602')+f'/{budget} & '+arm+' & '+' & '.join(cells)+r' \\')
    tex+=[r'\end{tabular}'];(output/'TABLE_MAIN.tex').write_text('\n'.join(tex)+'\n',encoding='utf-8')
    plots(output,by,delta_stats,costrows)
    validation=dict(status='PASS',source_sha256=hashlib.sha256(b).hexdigest(),absolute_rows=len(absolute),paired_rows=len(paired),absolute_stat_rows=len(means),paired_stat_rows=len(delta_stats),gates=len(gates),all_gates_passed=True,cost_measurements=len(costrows),closed_candidate_rows=len(candidates),new_model_AP_calls=0,no_pooled_AP=True,unit_conversion='COCO0..1 to percent/pp once; official pp unchanged',SD='sample ddof1 over seed, not CI',forbidden_imports='no model/evaluator/torch imports')
    (output/'VALIDATION.json').write_text(json.dumps(validation,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(validation))

def plots(out,by,stats,costs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    colors=['#377eb8','#e58428','#329b66'];labels=['Development548','Supplemental1602 (history unverified)']
    fig,axs=plt.subplots(2,2,figsize=(9.2,6.1),layout='constrained')
    for i,col in enumerate(COLLECTIONS):
        for j,key in enumerate(('AP','APS')):
            ax=axs[i,j]
            for a,arm in enumerate('ABC'):
                vals=[by[col,seed,arm,600]['COCO_0_1'][key]*100 for seed in (0,1,2)]
                ax.errorbar(a,statistics.mean(vals),yerr=statistics.stdev(vals),fmt='s',color=colors[a],capsize=4)
                ax.scatter([a-.08,a,a+.08],vals,color=colors[a],s=18,alpha=.8)
            ax.set_xticks(range(3),['A: whole','B: whole+NMS','C: five views']);ax.set_ylabel(key+' (%)');ax.set_title(labels[i]);ax.grid(axis='y',alpha=.25)
    fig.suptitle('Fixed top600 endpoints: all3 seeds; bars = sample SD (not CI)')
    save(fig,out,'FIG1_ABSOLUTE');plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(9.2,6.0),layout='constrained')
    for i,col in enumerate(COLLECTIONS):
        for j,(comparison,key) in enumerate((('C-B','APL'),('C-A','AR_large'))):
            ax=axs[i,j]
            for n,budget in enumerate((600,300)):
                row=next(x for x in stats if x['collection']==col and x['budget']==budget and x['comparison']==comparison and x['metric']==key)
                vals=[row['seed0_pp'],row['seed1_pp'],row['seed2_pp']]
                ax.plot(range(3),vals,marker='o' if n==0 else 's',label=f'top{budget}',color=('#6a51a3' if n==0 else '#dd4b39'))
            ax.axhline(0,color='black',lw=.7);ax.set_xticks(range(3),['seed0','seed1','seed2']);ax.set_ylabel(f'{comparison} {key} (pp)');ax.set_title(labels[i]);ax.grid(axis='y',alpha=.25);ax.legend(fontsize=8)
    fig.suptitle('Predefined tradeoffs: negative and positive deltas retained')
    save(fig,out,'FIG2_TRADEOFFS');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.4,3.6),layout='constrained')
    measures=sorted({x['measurement'] for x in costs})
    for k,measure in enumerate(measures):
        vals=[next(x['over_A'] for x in costs if x['measurement']==measure and x['seed']==seed and x['arm']=='C') for seed in (0,1,2)]
        ax.bar([i+(k-.5)*.32 for i in range(3)],vals,width=.3,label='Historical64' if k==0 else 'Supplemental execution64 recheck',color=('#377eb8' if k==0 else '#e58428'))
    ax.axhline(6,color='#a00',ls='--',lw=1,label='Frozen upper limit6');ax.set_ylim(0,6.6);ax.set_ylabel('C/A end-to-end time ratio');ax.set_xticks(range(3),['seed0','seed1','seed2']);ax.legend(fontsize=8,ncol=2);ax.set_title('Development fixed64, batch4; real views + preprocessing + NMS')
    save(fig,out,'FIG3_COST64');plt.close(fig)

def save(fig,out,name):
    for ext in ('pdf','svg','png'):
        fig.savefig(out/(name+'.'+ext),dpi=180,bbox_inches='tight')
        if ext=='svg':
            p=out/(name+'.svg')
            p.write_text('\n'.join(line.rstrip() for line in p.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.source,a.output)
