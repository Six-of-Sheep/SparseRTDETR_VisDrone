"""Actual formal CLI lifecycle on explicitly synthetic CPU inputs, no lab IO."""
import argparse,copy,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sparse_rtdetr.localization.runio import read,publish,reference,sha
from sparse_rtdetr.research import h2_formal_contract as ct

FILES=['h2_formal.py',*[f'src/sparse_rtdetr/research/h2_formal{x}.py' for x in ('','_contract','_resources','_worker')],
    'src/sparse_rtdetr/research/h2_history_metadata.py','tools/check_h2_formal_cli.py']

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--acceptance-fixture',type=Path,required=True);p.add_argument('--source-inventory',type=Path);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);old=a.acceptance_fixture.resolve();ac=read(old/'success/contract.json');ar=old/'success/attempt-acceptance-001'
    refs={k:reference(v) for k,v in dict(contract=old/'success/contract.json',result=ar/'RESULT.json',closed=ar/'CLOSED.json',exit=ar/'EXECUTION_EXIT.json',cost=old/'success/CUMULATIVE_COST.json',**{arm:ar/('accept-'+arm)/'CLOSED.json' for arm in ct.ARMS}).items()}
    names=({x['path'] for x in read(a.source_inventory)['source_files']} if a.source_inventory else set(subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()))|set(FILES)
    source=out/'source.json';publish(source,dict(commit='CPU_WORKING_TREE_NOT_PRODUCTION',files=[dict(path=n,bytes=(ROOT/n).stat().st_size,sha256=sha(ROOT/n)) for n in sorted(names)]))
    gate=out/'BOOTSTRAP_NOT_FINAL_GATE.json';publish(gate,dict(status='PASS',tests=12,skips=0,bootstrap_fixture=True,source_files=[dict(path=n,sha256=sha(ROOT/n)) for n in FILES]))
    auth=out/'authorization.json';publish(auth,dict(scope=ct.SCOPE,policy=ct.POLICY,source_sha256=sha(source),CPU_fixture_not_human_GPU_authorization=True))
    prior=out/'prior.json';publish(prior,dict(task_id=ct.TASK,forward_calls=18,GPU_seconds=ct.PRIOR_SECONDS,total_forward_cap=146596,acceptance=refs['cost']))
    ep=out/'N-endpoint.json';b=out/'N-binding.json';pr=out/'N-authentication.json';stop=out/'N-original-STOP.json'
    for path,value in ((b,dict(CPU_fixture=True)),(pr,dict(CPU_fixture=True,not_historical_evidence=True)),(stop,dict(status='SYNTHETIC_ORIGINAL_STOP'))):publish(path,value)
    publish(ep,dict(updates=36480,epoch=120,EMA=True,used_acceptance_state=False,binding=reference(b)))
    # N fixture scores are made by the SAME certified evaluator on constructed
    # logits and annotations; they are never reported as real detector results.
    from sparse_rtdetr.research.h2_formal_worker import development
    from sparse_rtdetr.r0.evaluation import rank_candidates,score_budget
    import torch
    rows,reader=development(dict(profile='CPU_FIXTURE_ONLY'));scores={}
    for budget in ('600','300'):
        preds=[]
        for row in rows:preds+=rank_candidates(torch.zeros(300,10),torch.full((300,4),.2),row,int(budget))
        v=score_budget(rows,[reader.raw(r) for r in rows],preds,str(ROOT));q=out/f'N-{budget}.json';publish(q,v);scores[budget]=reference(q)
    sel=out/'selection.json';publish(sel,dict(fixture_only=True,fixed_ids=[r['stable_image_id'] for r in rows[:4]]))
    history=dict(endpoint=reference(ep),authentication=reference(pr),original_STOP=reference(stop),original_CLOSED_path=str(out/'ORIGINAL_CLOSED_MUST_REMAIN_ABSENT.json'),status='AUTHENTICATED_OUTPUT_NOT_ORIGINAL_STAGE_CLOSURE',scores=scores)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');counter=0;cases=[]
    def cli(argv,ok=True):
        nonlocal counter
        counter+=1;r=subprocess.run([sys.executable,'-B',str(ROOT/'h2_formal.py'),*map(str,argv)],cwd=ROOT,env=env,capture_output=True)
        (out/f'CLI-{counter:03d}.stdout').write_bytes(r.stdout);(out/f'CLI-{counter:03d}.stderr').write_bytes(r.stderr)
        if (r.returncode==0)!=ok:raise RuntimeError(f'CLI{counter} rc{r.returncode} '+r.stderr.decode(errors='replace')[-2500:])
        return r
    prepared=None
    def bind(name,fault='none'):
        nonlocal prepared
        root=out/name;root.mkdir();spec=root/'spec.json';v=dict(ac)
        for k in ('contract_sha256','CPU_gate','prepare_gate','authorization','scope','policy','resources','science','source_root','source','commit','task_id'):v.pop(k,None)
        v.update(attempt='attempt-formal-001',task_root=str(root),job_root=str(root/'attempt-formal-001'),python=sys.executable,prior_cost=reference(prior),acceptance=refs,historical_N=history,selection=reference(sel),nonce='SYNTHETIC-FORMAL-NONCE-'+name,fixture=dict(fault=fault))
        publish(spec,v)
        if prepared is None:
            cli(['prepare','--spec',spec,'--source',source,'--output',root/'prepare']);prepared=root/'prepare/PREPARE_PASS.json'
        contract=root/'contract.json';cli(['bind','--spec',spec,'--source',source,'--output',contract,'--cpu-gate',gate,'--prepare-gate',prepared,'--authorization',auth]);return root,spec,contract
    relative=dict(ac['source'],path=os.path.relpath(ac['source']['path'],ac['source_root']))
    assert ct.exact(ct.original_reference(relative,ac['source_root']))==ct.exact(ac['source']);cases.append('external_relative_reference_uses_original_release_root')
    root,spec,contract=bind('normal');cli(['contract-check','--contract',contract]);cases.append('actual_raw_prepare_bind_and_hidden_contract')
    cli(['launch','--contract',contract,'--session','cpu-h2-formal','--foreground']);cli(['verify','--contract',contract]);cases.append('actual_full_CLI_cross_epoch_train_endpoint_dual_score_cost_compare_export')
    job=root/'attempt-formal-001';cost=read(root/'CUMULATIVE_COST.json');conf=read(contract)
    assert cost['forward_calls']==cost['forwards_completed']==18+ct.total_new(conf)==54 and cost['GPU_seconds']>=ct.PRIOR_SECONDS and cost['active'] is None;cases.append('inherited_18_forward_and_77_seconds_not_reset')
    assert cost['actual_updates']==cost['updates_completed']==4;cases.append('U_D_independent_complete_update_clocks')
    for arm in ('U','D'):
        rr=read(job/('train-'+arm)/'RESULT.json');assert rr['actual_updates']==2 and rr['BF16_forward_calls']==4
        assert read(job/('train-'+arm)/'CROSS_EPOCH_RESTORE.json')['full_state_exact']
        ep_=read(job/('train-'+arm)/'endpoint.json');assert not ep_['used_acceptance_state'] and not ep_['used_old_endpoint'] and ep_['epoch']==2
    cases.append('actual_fullstate_restore_across_epochs_fresh_initial_and_EMA_endpoint_load')
    assert read(job/'score/RESULT.json')['new_evaluator_calls']==8 and not read(job/'score/RESULT.json')['N_recomputed'];cases.append('four_new_groups_eight_evaluator_calls_N_scores_exact_reuse')
    for arm in ct.ARMS:
        for budget in ('600','300'):
            v=read(job/f'score/{arm}-{budget}.json');assert len(v['COCO_bridge']['metrics'])==13 and len([k for k in v['official'] if k.startswith(('AP','AR'))])>=7
    cases.append('six_score_JSON_13_COCO_and_7_primary_metrics')
    for arm in ('U','D'):
        cs=read(job/('infer-'+arm)/'CENTER_SUMMARY.json');assert cs['images']==8 and cs['matched_small']>0 and sum(cs[x] for x in ('improved','unchanged','worsened'))==cs['matched_small']
    cases.append('same_forward_coarse_final_native_matching_small_GT_denominators')
    timing=read(job/'cost/RESULT.json');assert timing['forward_calls']==24 and all(len(x)==6 for x in timing['measurements'].values());cases.append('three_arm_two_warmup_six_all_permutations_FP32_E2E')
    comp=read(job/'compare/RESULT.json');assert set(comp['deltas_pp'])=={'D-N','D-U','U-N'} and len(comp['quality_gates'])==22 and comp['quality_passed'] is None;cases.append('all_fixed_gates_full_deltas_no_fixture_scientific_claim')
    idx=read(job/'slim-export/SHA_INDEX.json');assert sum(x['returned']['bytes'] for x in idx['files'])<=20*1024**2
    for x in idx['files']:assert reference(x['returned']['path'])==x['returned'] and not x['name'].endswith('.pth')
    cases.append('slim_whitelist_SHA_no_state_or_prediction_export')
    before=sha(root/'CUMULATIVE_COST.json');cli(['launch','--contract',contract,'--session','cpu-duplicate','--foreground'],False);cli(['run','--contract',contract],False);assert sha(root/'CUMULATIVE_COST.json')==before;cases.append('duplicate_launch_run_no_budget_reset')
    for name,field,value in (('changed_policy','policy',dict(ct.POLICY,epochs=12)),('changed_resources','resources',{}),('changed_scope','scope','H2_GPU_ACCEPTANCE_ONLY_V1')):
        v=read(spec);v[field]=value;bad=root/(name+'.json');publish(bad,v);cli(['prepare','--spec',bad,'--source',source,'--output',root/(name+'-output')],False);cases.append(name+'_actual_CLI_rejected')
    r2,s2,c2=bind('worker-fault','worker_exception');cli(['launch','--contract',c2,'--session','cpu-fault','--foreground'],False)
    jj=r2/'attempt-formal-001';assert (jj/'STOP.json').is_file() and not (jj/'CLOSED.json').exists() and (jj/'slim-export/SHA_INDEX.json').is_file()
    for file in (jj/'train-U/supervisor-exit.json',jj/'train-U-PROCESS-EXIT.json'):
        v=read(file);assert not v['remaining_live'] and not v['cleanup_errors']
    cases.append('new_anomaly_actual_worker_exit_own_tree_cleanup_STOP_slim_no_retry')
    for fault in ('forward_overrun','update_overrun','time_overrun','allocated_overrun'):
        rf,sf,cf=bind(fault,fault);cli(['launch','--contract',cf,'--session','cpu-'+fault.replace('_','-'),'--foreground'],False)
        jf=rf/'attempt-formal-001';assert (jf/'STOP.json').is_file() and not (jf/'CLOSED.json').exists() and (jf/'slim-export/SHA_INDEX.json').is_file()
        v=read(jf/'train-U/supervisor-exit.json');assert not v['remaining_live'] and not v['cleanup_errors'];cases.append(fault+'_actual_CLI_hard_refusal_and_exit')
    # Identity/byte corruption must be rejected by actual terminal CLI verification.
    q=job/'train-U/events.jsonl';original=q.read_bytes();q.write_bytes(original+b'{}\n');cli(['verify','--contract',contract],False);q.write_bytes(original);cases.append('bounded_full_event_stream_tamper_rejected')
    q=job/'train-D/checkpoint-e002.pth';before=q.read_bytes();q.write_bytes(before+b'x')
    from sparse_rtdetr.research.h2_formal_worker import endpoint
    try:endpoint(conf,'D',torch.device('cpu'));raise AssertionError('endpoint SHA bypass')
    except ValueError as exc:assert 'SHA' in str(exc)
    q.write_bytes(before);cases.append('wrong_endpoint_SHA_rejected_before_weights_loaded')
    report=dict(status='PASS',tests=len(cases),skips=0,cases=cases,actual_CLI_invocations=counter,CPU_only=True,bootstrap_fixture=False,CUDA_initialized=torch.cuda.is_initialized(),GPU_forward_calls=0,normal_new_CPU_forwards=ct.total_new(conf),fixture_not_science=True,
        source_files=[dict(path=n,bytes=(ROOT/n).stat().st_size,sha256=sha(ROOT/n)) for n in FILES],python=sys.version.split()[0],time_epoch=time.time())
    publish(out/'CPU_CLI_GATE.json',report)
    removed=[]
    for q in out.rglob('*.pth'):
        assert q.resolve().is_relative_to(out);removed.append(dict(path=str(q.relative_to(out)),bytes=q.stat().st_size));q.unlink()
    publish(out/'TEMPORARY_CPU_STATES_REMOVED.json',dict(status='PASS',files=removed));print(json.dumps(dict(status='PASS',tests=len(cases),CLI_invocations=counter)))

if __name__=='__main__':main()
