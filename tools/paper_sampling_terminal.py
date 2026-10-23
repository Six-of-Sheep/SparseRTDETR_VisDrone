"""Read only the authenticated formal-003 continuation; no evaluator/model imports.

Reuses PR17 table arithmetic/plotting and literal mechanism normalization. PR17's
original validator stays unchanged: its formal-001 binding is not this attempt.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import paper_sampling_tables as tables
import paper_sampling_mechanism as mechanism

ROOT=Path(__file__).resolve().parents[1]
LOCK=ROOT/'docs/papers/r18_sampling_scale/TERMINAL_BINDING.json'
STAGES=('admission','train-B1','train-P','infer-B0','infer-B1','infer-P','score','cost','compare','export')

def check(ok, message):
    if not ok: raise ValueError(message)

def load(folder):
    check(not (Path(folder)/'attempt-formal-003/STOP.json').exists(),'current attempt STOP present')
    lock=json.loads(LOCK.read_text(encoding='utf-8')); store=tables.Receipts(folder)
    def fixed(name):
        ref=lock['receipts'][name]
        return store.exact({'path':name,'sha256':ref['sha256'],'bytes':ref['bytes']})
    closed=fixed('attempt-formal-003/CLOSED.json'); root=store.exact(closed['result'])
    contract=fixed('formal-execution-003/contract.json'); source=store.exact(contract['source'])
    check(tables.digest({k:v for k,v in contract.items() if k!='contract_sha256'})==lock['contract_digest']==contract['contract_sha256']==root['contract_sha256'],'canonical contract identity')
    check(contract['attempt']==lock['attempt'] and contract['commit']==source['commit']==lock['commit'],'attempt/commit')
    check(contract['profile']==root['profile']=='LAB_GPU_ONCE','formal production only')
    check(contract['source']['sha256']==root['source_sha256']==lock['source_sha256'],'source identity')
    check(root['task_id']==lock['task_id'] and root['status']==closed['status']=='PASS','root not closed')
    check(tuple(root['stages'])==STAGES,'continuation stage sequence')
    check(root['external_B0']==closed['external_B0']==contract['external_B0'],'B0 reference')
    b0=store.exact(root['external_B0'])
    check(b0['status']=='AUTHENTICATED_OUTPUT_NOT_ORIGINAL_STAGE_CLOSURE','B0 provenance')
    for key in ('old_closure_created','training_repeated','checkpoint_loaded_for_training','checkpoint_copied'):
        check(b0[key] is False,'B0 original failure must not be rewritten')
    old=store.exact(b0['contract']); old_source=store.exact(b0['source'])
    check(old_source['commit']==lock['B0_commit'] and b0['source']['sha256']==lock['B0_source_sha256'],'B0 source identity')
    check(old['science_sha256']==contract['science_sha256'] and old['execution_sha256']==contract['execution_sha256'],'unchanged science/execution')
    check(root['external_acceptance']==closed['external_acceptance']==contract['external_acceptance'],'acceptance reference')
    store.exact(root['external_acceptance'])
    stages={}
    for ref in closed['stage_closures']:
        c=store.exact(ref); identity=c['identity']; stage=identity['stage']
        check(stage not in stages and stage in STAGES,'stage coverage')
        check(c['status']=='PASS' and identity['attempt']==lock['attempt'] and identity['source_sha256']==lock['source_sha256'] and identity['contract_sha256']==lock['contract_digest'],'stage identity')
        check(identity['science_sha256']==contract['science_sha256'] and identity['initialization_sha256']==contract['pretrained']['sha256'],'science/initialization identity')
        value=store.exact(c['result']);check(value['status']=='PASS' and value['identity']==identity,'stage result')
        for name in ('supervisor_exit','worker_exit'):
            ex=store.exact(c[name]);check(ex['status']=='PASS' and ex['identity']==identity,'exit identity')
            if name=='supervisor_exit':check(ex['returncode']==0 and not ex['remaining_live'] and not ex['cleanup_errors'] and ex['error'] is None,'supervisor failure')
        stages[stage]=value
    check(tuple(stages)==STAGES,'all ten actual stages closed')
    ex=fixed('formal-execution-003/EXECUTION_EXIT.json')
    check(ex['status']=='PASS' and ex['returncode']==0 and ex['contract']['sha256']==lock['receipts']['formal-execution-003/contract.json']['sha256'],'execute exit')
    stages['train-B0']=store.exact(b0['result'])
    endpoints={}
    for arm in tables.ARMS:
        train=stages['train-'+arm]
        check(train['actual_updates']==36480 and train['forward_calls']==72960 and train['used_acceptance_state'] is False,'fixed fresh training')
        endpoint=store.exact(train['endpoint']);endpoints[arm]=endpoint
        check(endpoint['epoch']==120 and endpoint['updates']==36480 and endpoint['EMA'] is True and endpoint['used_acceptance_state'] is False,'fixed e120 EMA')
        check(endpoint['checkpoint']['completed_updates']==36480,'checkpoint cursor')
    check(len({stages['train-'+a]['student_inputs_sha256'] for a in tables.ARMS})==1,'training input identity')
    check(endpoints['B0']['initial_model_sha256']==endpoints['P']['initial_model_sha256'],'B0/P common initial state')
    comp=stages['compare']; score=stages['score'];metrics={}
    check(score['groups']==6 and score['evaluator_calls']==12 and comp['inputs_equal'] is True,'score/input coverage')
    for arm in tables.ARMS:
        for budget in tables.BUDGETS:
            value=store.exact(score['results'][arm][budget]);metrics[arm,budget]=value
            check(set(value['COCO_bridge']['metrics'])==set(tables.METRICS),'13 COCO fields')
            check(value['official']['evaluated_image_count']==548,'548 images')
            for name in tables.METRICS:check(0<=tables.finite(value['COCO_bridge']['metrics'][name])<=1,'COCO range')
            for name in tables.PRIMARY:check(0<=tables.finite(value['official'][name])<=100,'primary range')
    expected=tables.gate_keys(lock);seen={}
    for g in comp['quality_gates']:
        key=g['pair'],g['budget'],g['metric']
        check(key not in seen and key in expected and g['minimum']==expected[key] and type(g['passed']) is bool,'frozen gate schema')
        seen[key]=g
    check(seen.keys()==expected.keys(),'22 original gates')
    # Arithmetic cross-checks are validation of stored comparisons, not new AP or gates.
    for pair,base in [('P-B0','B0'),('P-B1','B1')]:
        for budget in tables.BUDGETS:
            for name in tables.METRICS:
                literal=comp['deltas_pp'][pair][budget][name]
                delta=100*(metrics['P',budget]['COCO_bridge']['metrics'][name]-metrics[base,budget]['COCO_bridge']['metrics'][name])
                check(math.isclose(literal,delta,abs_tol=1e-10),'stored delta inconsistent')
    check(comp['cost']==stages['cost']['summary'] and comp['scope']=='seed0_development548_exploration','comparison/cost scope')
    for key in ('quality_passed','cost_passed'):check(comp[key]==root[key] and type(root[key]) is bool,'terminal decision mismatch')
    ledger=fixed('attempt-formal-003/COST_LEDGER.json')
    check(ledger['active'] is None and ledger['forward_calls']==root['forward_calls_total']==comp['total_forward_calls']==219693,'final forward ledger')
    check(root['inherited_forward_calls']==ledger['prior']['forward_calls']==72978 and root['new_forward_calls']==146715 and comp['train_updates']==109440,'inherited costs not reset')
    check(math.isclose(ledger['completed_GPU_seconds'],ledger['prior']['GPU_seconds']+sum(s['GPU_seconds'] for s in ledger['stages']),abs_tol=1e-7),'cumulative seconds mismatch')
    check(ledger['completed_GPU_seconds']<=60*3600,'60h cap')
    return dict(lock=lock,store=store,root=root,comparison=comp,metrics=metrics,fixture=False,endpoints=endpoints,ledger=ledger),fixed

def render(folder,output,make_plots=True):
    data,fixed=load(folder);out=Path(output)
    check(not out.exists(),'new output folder required');out.mkdir(parents=True)
    absolute,deltas,cost,gates=tables.rows_for(data)
    for name,rows in [('absolute_metrics',absolute),('paired_deltas',deltas),('cost',cost),('gates',gates)]:tables.write_csv(out/(name+'.csv'),rows)
    content=['# 正式终态完整结果','', '固定e120 EMA / seed0 / development548。绝对值为百分数，差值为百分点；非独立验证，无显著性或跨seed结论。','']
    for budget in tables.BUDGETS:
        for evaluator,names in [('COCO_bridge',tables.METRICS),('primary',tables.PRIMARY)]:
            content+=['## top'+budget+' / '+evaluator,'']
            rows=[]
            for arm in tables.ARMS:rows.append([arm]+[next(r['value_percent'] for r in absolute if r['arm']==arm and r['budget']==int(budget) and r['evaluator']==evaluator and r['metric']==n) for n in names])
            for pair in ('P-B0','P-B1','B1-B0'):rows.append([pair]+[next(r['delta_pp'] for r in deltas if r['pair']==pair and r['budget']==int(budget) and r['evaluator']==evaluator and r['metric']==n) for n in names])
            content +=[tables.markdown(['方案/比较',*names],rows),'']
    content+=['## 全部冻结门槛（照录裁决）','',tables.markdown(list(gates[0]),[list(r.values()) for r in gates]),'','## 固定64图成本','',tables.markdown(list(cost[0]),[list(r.values()) for r in cost]),'']
    measured=fixed('attempt-formal-003/cost/RESULT.json')['measurements']
    check(set(measured)=={'A','B0','B1','P'} and all(len(rows)==4 for rows in measured.values()),'four measured passes/model')
    check(all(r['batch']==4 and r['images']==64 and r['dtype']=='FP32' for rows in measured.values() for r in rows),'timing boundary')
    tables.write_csv(out/'cost_trials.csv',[{'arm':arm,'trial':i+1,'seconds':r['seconds'],'peak_allocated_bytes':r['peak_allocated_bytes'],'peak_reserved_bytes':r['peak_reserved_bytes']} for arm,rows in measured.items() for i,r in enumerate(rows)])
    (out/'TABLES.md').write_text('\n'.join(content),encoding='utf-8')
    if make_plots:tables.plots(out,data,absolute,deltas,cost)
    # PR17 plots top600; add the same literal deltas for both budgets, no new statistics.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for ax,name in zip(axes.flat,('AP','APS','AP75_small','APM','APL','AR_large')):
        for i,pair in enumerate(('P-B0','P-B1','B1-B0')):
            for j,budget in enumerate((600,300)):
                v=next(r['delta_pp'] for r in deltas if r['pair']==pair and r['budget']==budget and r['evaluator']=='COCO_bridge' and r['metric']==name)
                ax.bar(i+(j-.5)*.34,v,.32,label='top'+str(budget) if i==0 else None,color=('#245f81','#d48a44')[j])
        ax.axhline(0,color='#555',lw=.8);ax.set_xticks(range(3),('P-B0','P-B1','B1-B0'));ax.set_title(name);ax.set_ylabel('Delta (pp)')
    axes.flat[0].legend();fig.suptitle('seed0 development548 / fixed e120 EMA / no uncertainty estimate')
    for ext in ('svg','png'):fig.savefig(out/('both_budgets.'+ext),dpi=170)
    plt.close(fig)
    # Existing bounded mechanism summaries. Identity remains attached to filename/layer.
    counts=[];values=[];window_rows=[];provenance=[]
    for arm,attempt in [('B0','attempt-formal-001'),('B1','attempt-formal-003'),('P','attempt-formal-003')]:
        for epoch in (1,40,80,117,118,120):
            name=f'{attempt}/train-{arm}/sampling-e{epoch:03d}-w000.json';payload=fixed(name)
            check(len(payload)==6 and {(r['layer'],r['group']) for r in payload}=={(l,g) for l in (0,1,2) for g in ('ordinary','DN')},'mechanism coverage')
            check(all(r['arm']==arm for r in payload),'mechanism arm')
            c,v,_=mechanism.normalize(payload,scope='selected-training-windows',source=name);counts+=c;values+=v
            for group in ('ordinary','DN'):
                rr=[r for r in payload if r['group']==group];n=sum(r['reference_instances'] for r in rr);below=sum(r['reference_below_cell_any'] for r in rr)
                check(all(r['DN_includes_native_padding']==(group=='DN') for r in rr),'padding flag')
                window_rows.append(dict(arm=arm,epoch=epoch,group=group,reference_layer_occurrences=n,below_cell_any=below,descriptive_fraction=below/n,floor_applied=sum(r['floor_applied_instances'] for r in rr),weight_means_by_layer=';'.join(str(r['bilinear_effective_weight_abs_change']['mean']) for r in rr),output_means_by_layer=';'.join(str(r['attention_aggregate_output_abs_change']['mean']) for r in rr),same_stencil_by_layer=';'.join(str(r['same_stencil_fraction']) for r in rr),sampled_queries_per_image=8,GT_size_linkage='UNKNOWN',nonpadding_DN_effect='UNKNOWN',gradient_effect='UNKNOWN'))
            provenance.append(name)
    mechanism.write_csv(out/'reference_counts.csv',counts,('source','arm','group','record')+mechanism.COUNT_FIELDS)
    mechanism.write_csv(out/'mechanism_measurements.csv',values,('source','arm','group','record','metric','statistic','value','count','population'))
    tables.write_csv(out/'mechanism_windows.csv',window_rows)
    (out/'MECHANISM.md').write_text('# 已保存正式机制窗口\n\n三层reference计数相加是重复层实例，不是独立目标数。全部逐层分位数、位置、跨度与有效权重统计见CSV，未合并分位数。\n\n'+tables.markdown(list(window_rows[0]),[list(r.values()) for r in window_rows]),encoding='utf-8')
    if make_plots:mechanism.plots(out,counts,values,'Selected training windows only; first <=8 queries/image; DN includes padding')
    fig,axes=plt.subplots(2,3,figsize=(12,6.5),layout='constrained')
    for gi,group in enumerate(('ordinary','DN')):
        for layer in range(3):
            selected=[]
            for epoch in (1,40,80,117,118,120):
                payload=fixed(f'attempt-formal-003/train-P/sampling-e{epoch:03d}-w000.json')
                selected.append(next(r for r in payload if r['group']==group and r['layer']==layer))
            for col,ys in enumerate(([r['reference_floor_trigger_fraction_any']*100 for r in selected],[r['bilinear_effective_weight_abs_change']['mean'] for r in selected],[r['attention_aggregate_output_abs_change']['mean'] for r in selected])):
                axes[gi,col].scatter(range(6),ys,label='decoder '+str(layer),marker=('o','x','+')[layer],s=45)
                axes[gi,col].set_xticks(range(6),('e1','e40','e80','e117','e118','e120'));axes[gi,col].set_ylabel(group)
        axes[gi,0].set_ylim(0,100)
    for ax,title in zip(axes[0],('All reference occurrences: below cell (%)','First <=8 queries: effective weight |change|','First <=8 queries: aggregate output |change|')):ax.set_title(title,fontsize=10)
    axes[0,0].legend(fontsize=8);fig.suptitle('P / selected first-microbatch windows only / DN includes native padding\nNo GT-size linkage or loss-gradient evidence; per-record means, no pooled quantiles',fontsize=11)
    for ext in ('svg','png'):fig.savefig(out/('mechanism_selected_P.'+ext),dpi=170)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for i,arm in enumerate(('A','B0','B1','P')):
        ys=[r['seconds'] for r in measured[arm]]
        ax.scatter([i-.12+j*.08 for j in range(4)],ys,color='#2a6f8e',s=30)
        ax.errorbar(i,data['comparison']['cost'][arm]['mean_seconds'],yerr=data['comparison']['cost'][arm]['sample_SD_seconds'],fmt='_',color='#222',capsize=4)
    ax.set_xticks(range(4),('A (historical)','B0','B1','P'));ax.set_ylabel('Seconds per fixed64 pass');ax.set_title('Four FP32 batch4 measured passes + stored mean/sample SD\nTiming variability is not across-seed uncertainty',fontsize=11)
    for ext in ('svg','png'):fig.savefig(out/('cost_trials.'+ext),dpi=170)
    plt.close(fig)
    verification=dict(status='READONLY_TERMINAL_VERIFIED',binding=data['lock'],original_root=data['root'],original_comparison=data['comparison'],endpoints=data['endpoints'],ledger=data['ledger'],referenced_JSON=data['store'].used,mechanism_sources=provenance,recomputed_AP=False,recomputed_scientific_decisions=False,large_artifacts_rehashed=False,B0_original_CLOSED_created=False)
    (out/'VERIFICATION.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
    # Matplotlib paths contain cosmetic trailing spaces; normalize without altering geometry.
    for svg in out.glob('*.svg'):
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    print(json.dumps({'absolute_rows':len(absolute),'delta_rows':len(deltas),'gate_rows':len(gates),'mechanism_records':len(counts),'quality_passed':data['root']['quality_passed'],'cost_passed':data['root']['cost_passed']}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--receipts',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();render(a.receipts,a.output)
