"""Actual compressed CPU formal CLI, including real scorer and balanced cost branches."""
import os,sys,subprocess,json,tempfile,shutil,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
from sabl_corrected_formal import BASE,BASE_MANIFEST,sha,read,write,validate,forwards,N_config_matches,cost_record,GPU_memory_gate,allocator_policy,RUNTIME

def main(output):
    out=Path(output);out.mkdir();start=time.monotonic();checks=[];calls=[];peak=0;total_cpu_forward=0;total_cpu_updates=0
    def check(name,value):
        checks.append({'name':name,'pass':bool(value)});(out/'CHECKS.json').write_text(json.dumps(checks,indent=2))
        if not value:raise AssertionError(name)
    def rss(pid):
        try:
            s=Path('/proc/%d/status'%pid).read_text();n=int(next(x.split()[1] for x in s.splitlines() if x.startswith('VmRSS:')))*1024
            return n+sum(rss(int(p)) for p in Path('/proc/%d/task/%d/children'%(pid,pid)).read_text().split())
        except (OSError,StopIteration):return 0
    with tempfile.TemporaryDirectory(prefix='corrected-formal-CPU-',dir=out) as temp:
        root=Path(temp);release=root/'release';release.mkdir()
        names=set(subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()) if (ROOT/'.git').exists() else {x['path'] for x in read(ROOT/'RELEASE_MANIFEST.json')['files']}
        names|={'tools/sabl_corrected_formal.py','tools/check_sabl_corrected_formal.py','tests/fixtures/sabl_corrected_formal_base.json','tools/sabl_ipc.py','tools/check_sabl_ipc.py'}
        rows=[]
        for n in sorted(names):
            p=ROOT/n;q=release/n;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q);rows.append({'path':n,'bytes':q.stat().st_size,'sha256':sha(q)})
        manifest={'commit':'c'*40,'candidate_not_for_real_execution':True,'files':rows,'computation_base':BASE,'computation_base_manifest':BASE_MANIFEST}
        mf=release/'RELEASE_MANIFEST.json';write(mf,manifest);write(release/'DEPLOYED.json',{'status':'PASS','manifest_sha256':sha(mf),**manifest})
        entry=release/'tools/sabl_corrected_formal.py';env=dict(os.environ);env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
        def cli(name,args,ok=True):
            nonlocal peak
            begin=time.monotonic();p=subprocess.Popen([sys.executable,'-B',str(release/'tools/sabl_ipc.py'),'exec','--task',str(Path(args[args.index('--contract')+1]).parent),'--',sys.executable,'-B',str(entry),*args] if args[0]=='launch' else [sys.executable,'-B',str(entry),*args],cwd=release,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
            while p.poll() is None:
                peak=max(peak,rss(os.getpid()))
                if peak>4<<30 or time.monotonic()-start>590:
                    os.killpg(p.pid,signal.SIGKILL);a,b=p.communicate();raise RuntimeError('CPU RSS/wall budget')
                time.sleep(.05)
            a,b=p.communicate();(out/(name+'.stdout')).write_bytes(a);(out/(name+'.stderr')).write_bytes(b)
            calls.append({'name':name,'returncode':p.returncode,'seconds':time.monotonic()-begin,'stdout_bytes':len(a),'stderr_bytes':len(b)})
            (out/'CLI_COST.json').write_text(json.dumps(calls,indent=2));check('CLI '+name,(p.returncode==0)==ok)
            return a,b
        def plan(name,fault='none'):
            task=root/name;task.mkdir();c=task/'CONTRACT.json'
            cli(name+'-plan',['plan','--synthetic','--manifest',str(mf),'--attempt-root',str(task/'attempt'),'--out',str(c),'--fault',fault]);return task,c
        def launch(name,task,c,ok=True,extra=None):
            return cli(name,['launch','--contract',str(c),'--manifest',str(mf),'--approve-commit','c'*40,'--approve-manifest',sha(mf),'--approve-contract-sha',sha(c),'--authorize-cpu',*(extra or [])],ok)
        def retire(task):
            nonlocal total_cpu_forward,total_cpu_updates
            p=task/'attempt/ledger.json'
            if p.exists():
                l=read(p);total_cpu_forward+=l['new_forward'];total_cpu_updates+=l['actual_updates']
            refs=[]
            for p in task.rglob('*.pt'):
                assert p.resolve().is_relative_to(root.resolve());refs.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)});p.unlink()
            if refs:(out/(task.name+'-SYNTHETIC_TENSOR_INDEX.json')).write_text(json.dumps(refs,indent=2))
        task,c=plan('positive');launch('positive-run',task,c)
        r=read(task/'attempt/slim/RESULT.json');l=r['ledger'];contract=read(c)
        check('40 forwards 6 actual updates',l['new_forward']==40 and l['actual_updates']==l['reserved_updates']==6)
        check('inherited budget preserved',l['total_forward']==57 and l['GPU_stage_seconds']==2.0 and l['inherited']==contract['prior'])
        check('three compressed CPU epochs only S A',all(read(task/'attempt'/('train-'+a)/'endpoint.json')['updates']==3 for a in ['S','A']) and not (task/'attempt/train-N').exists())
        check('strict cross-phase full-state restore completed',all(read(task/'attempt'/('train-'+a)/'ENGINEERING_PASS.json')['checkpoint']['updates']==2 for a in ['S','A']))
        score=read(task/'attempt/score/RESULT.json');cost=read(task/'attempt/cost/RESULT.json');comp=read(task/'attempt/compare/COMPARISON.json')
        check('two budgets four scores eight actual evaluator calls',score['new_score_groups']==4 and score['evaluator_calls']==8 and len(list((task/'attempt/score').glob('[SA]-*.json')))==4)
        check('N and original N not rescored',not score['N24_recomputed'] and not score['original_N_recomputed'] and score['CPU_historical_scores_are_explicit_alias_fixture'])
        check('balanced N S A cost',len(cost['warmups'])==6 and set(cost['measurements'])=={'N','S','A'} and all(len(v)==6 for v in cost['measurements'].values()))
        check('N24 no synthetic training claimed',r['N24_retrained'] is False and r['CPU_synthetic'] and cost['scope']=='CPU_SYNTHETIC_NOT_COST_EVIDENCE')
        check('full absolute and differences',all(x in (task/'attempt/compare/TABLE.md').read_text() for x in ['S-N24','S-A24','S-originalN','A24-N24']))
        check('quality negative still exported',comp['quality_controls_pass'] is False and (task/'attempt/CLOSED.json').exists() and r['scientific_status']=='NOT_SCIENTIFIC')
        index=read(task/'attempt/slim/SHA_INDEX.json');check('slim SHA closure',all(sha(task/'attempt/slim'/k)==v['sha256'] for k,v in index.items()))
        check('no followup or short state reuse',r['no_followup'] and not r['short_state_for_formal'] and r['must_exit'])
        (out/'POSITIVE_RESULT.json').write_text(json.dumps(r,indent=2));(out/'POSITIVE_COMPARISON.json').write_text(json.dumps(comp,indent=2))
        before=sha(task/'attempt/CLOSED.json');launch('duplicate-refused',task,c,False);check('duplicate preserves closure',sha(task/'attempt/CLOSED.json')==before);retire(task)
        for fault,want in [('worker_exception','SYNTHETIC_WORKER_EXCEPTION'),('forward_exhausted','new forward cap'),('nonfinite','non-finite'),('bad_endpoint','fixed e24 endpoint provenance'),('missing_close','CLOSED.json'),('bad_input','score membership/budget')]:
            t,d=plan(fault,fault);launch(fault+'-run',t,d,False);stop=read(t/'attempt/STOP.json')
            (out/(fault+'-STOP.json')).write_text(json.dumps(stop,indent=2))
            check(fault+' STOP and no CLOSED',stop['no_retry'] and not (t/'attempt/CLOSED.json').exists())
            # Nonfinite wording is owned by the existing Engine; never broaden actual criteria.
            check(fault+' exact intended rejection',(want in stop['error_message']) if fault!='nonfinite' else ('training images' in stop['error'] and 'finite' in stop['error']))
            (out/(fault+'-STOP.json')).write_text(json.dumps(stop,indent=2));retire(t)
        t,d=plan('wrong-policy');v=read(d);v['policy']['epochs']=25;d.write_text(json.dumps(v));launch('wrong-policy-run',t,d,False);check('bad policy no attempt',not (t/'attempt').exists())
        t,d=plan('wrong-contract-bytes');launch('wrong-contract-bytes-run',t,d,False,['--approve-contract-sha','0'*64]);check('wrong contract bytes no attempt/authority',not (t/'attempt').exists() and not (t/'authorization.json').exists())
        t,d=plan('wrong-authority');launch('wrong-authority-run',t,d,False,['--authorize-gpu']);check('mixed authority no attempt',not (t/'attempt').exists())
        current={'arm':'r18-sabl-control-lrtb-logsigmoid-v3','epochs':24,'learning_rate':1e-5}
        saved=dict(current,arm='r18-sabl-control-v1')
        check('N-only registered identity mapping',N_config_matches(saved,current))
        check('N bridge rejects numerical config drift',not N_config_matches(dict(saved,learning_rate=2e-5),current))
        check('N bridge rejects unknown field',not N_config_matches(dict(saved,unknown=1),current))
        fake=json.loads(json.dumps(contract));fake['prior']={'forward':32133,'GPU_seconds':37340.53682075267};fake['requested_cumulative']={'forward':61975,'GPU_seconds':'94472.36945352436'}
        cost_recorded=cost_record(fake,{'new_forward':29842,'actual_updates':14592},50.0)
        check('cost schema inherits consumed failures',cost_recorded['cumulative']['forward']==61975 and cost_recorded['remaining']['forward']==0 and cost_recorded['previous']['GPU_stage_seconds']==37340.53682075267 and cost_recorded['failed_cost_refunded'] is False)
        check('21GiB allocator only runtime; native',contract['caps']['allocated_bytes']==21<<30 and contract['runtime']==RUNTIME and contract['policy']['new_forwards']==29842)
        check('fresh resource memory gate positive',GPU_memory_gate(24<<30,2<<30,22<<30)['free_bytes']==22<<30)
        for postinit,free in [(False,(22<<30)-1),(True,(21<<30)+(512<<20)-1)]:
            try:GPU_memory_gate(24<<30,(24<<30)-free,free,postinit)
            except ValueError:check('low free rejected '+str(postinit),True)
            else:check('low free rejected '+str(postinit),False)
        from unittest.mock import patch
        import types
        for env,backend,ok in [({},'native',True),({'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'},'native',False),({},'cudaMallocAsync',False)]:
            fake_torch=types.SimpleNamespace(cuda=types.SimpleNamespace(memory=types.SimpleNamespace(get_allocator_backend=lambda:backend)))
            with patch.dict(os.environ,env,clear=True):
                try:allocator_policy(fake_torch);passed=True
                except ValueError:passed=False
            check('allocator policy '+str(env)+backend,passed==ok)
        fake['independent_debug']={'spent':{'forward':120,'GPU_stage_seconds_conservative_process_wall':199.9896325499285},'unused_allowance_for_formal':False}
        separated=cost_record(fake,{'new_forward':29842,'actual_updates':14592},50.0)
        check('independent debug not borrowed or refunded',separated['cumulative']['forward']==61975 and separated['cumulative_all_research']['forward']==62095 and separated['remaining']['forward']==0 and separated['debug_unused_budget_for_formal'] is False)
        for key in CAPS_KEYS:
            v=json.loads(json.dumps(contract));v['caps'][key]=float(v['caps'][key])+1
            try:validate(v)
            except ValueError:check('tamper '+key,True)
            else:check('tamper '+key,False)
    result={'status':'PASS_ACTUAL_CORRECTED_FORMAL_CPU_CLI','checks':len(checks),'CPU_detector_forward':total_cpu_forward,'CPU_actual_updates':total_cpu_updates,'GPU_forward':0,'real_images_or_weights_loaded':False,'compressed_profile':'S/A 3 epochs x1 window; no claim of GPU304-window verification','actual_synthetic_dual_evaluator_calls':8,'actual_synthetic_NSA_cost_calls':24,'historical_scores_are_explicit_alias_fixture':True,'CPU_seconds':time.monotonic()-start,'max_tree_RSS_bytes':peak,'RSS_cap_bytes':4<<30,'synthetic_tensor_states_removed':True,'production_authorization_created':False,'production_attempt_created':False,'CUDA_path_status':'UNKNOWN_NOT_EXECUTED'}
    write(out/'RESULT.json',result);print(json.dumps(result))

CAPS_KEYS=['new_forward','new_GPU_seconds','new_wall_seconds','allocated_bytes','new_storage_bytes','family_storage_bytes','return_bytes']
if __name__=='__main__':main(sys.argv[1])
