"""Actual two-arm production CLI with synthetic CPU material, never a GPU authorization."""
import os,sys,subprocess,json,tempfile,shutil,hashlib,time,resource,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
from sabl_corrected_short import BASE,BASE_MANIFEST,sha,read,write,ref_exact
from types import SimpleNamespace

def main(out):
    out=Path(out);out.mkdir();begin=time.monotonic();checks=[];cost=[];peak=0
    def check(name,ok):
        checks.append({'name':name,'pass':bool(ok)})
        (out/'CHECKS.json').write_text(json.dumps(checks,indent=2))
        if not ok:raise AssertionError(name)
    def rss(pid):
        try:
            s=Path('/proc/%d/status'%pid).read_text();n=int(next(x.split()[1] for x in s.splitlines() if x.startswith('VmRSS:')))*1024
            children=Path('/proc/%d/task/%d/children'%(pid,pid)).read_text().split()
            return n+sum(rss(int(c)) for c in children)
        except (OSError,StopIteration):return 0
    with tempfile.TemporaryDirectory(prefix='corrected-short-CPU-',dir=out) as temp:
        root=Path(temp);release=root/'release';release.mkdir()
        # Copy the actual edited checkout and new entry, no import-environment shim.
        tracked=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines() if (ROOT/'.git').exists() else [x['path'] for x in read(ROOT/'RELEASE_MANIFEST.json')['files']]
        tracked=set(tracked)|{'tools/sabl_corrected_short.py','tools/check_sabl_corrected_short.py','tests/fixtures/sabl_short_scientific_base.json'}
        rows=[]
        for name in sorted(tracked):
            p=ROOT/name
            if not p.is_file():continue
            q=release/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
            rows.append({'path':name,'bytes':q.stat().st_size,'sha256':sha(q)})
        manifest={'commit':'f'*40,'candidate_not_for_real_execution':True,'files':rows,'scientific_base_commit':BASE,'scientific_base_manifest':BASE_MANIFEST}
        write(release/'RELEASE_MANIFEST.json',manifest);write(release/'DEPLOYED.json',{'status':'PASS','manifest_sha256':sha(release/'RELEASE_MANIFEST.json'),**manifest})
        entry=release/'tools/sabl_corrected_short.py';mf=release/'RELEASE_MANIFEST.json'
        env=dict(os.environ);env.pop('PYTHONPATH',None)
        def cli(name,args,want=0):
            nonlocal peak
            t=time.monotonic();p=subprocess.Popen([sys.executable,'-B',str(entry),*args],cwd=release,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
            # This is CPU-test-only isolation, not a production process controller.
            while p.poll() is None:
                used=rss(os.getpid());peak=max(peak,used)
                if used>4<<30 or time.monotonic()-begin>290:
                    os.killpg(p.pid,signal.SIGKILL);a,b=p.communicate();raise RuntimeError('CPU tree RSS/wall exceeded')
                time.sleep(.05)
            a,b=p.communicate();(out/(name+'.stdout')).write_bytes(a);(out/(name+'.stderr')).write_bytes(b)
            cost.append({'name':name,'returncode':p.returncode,'seconds':time.monotonic()-t,'stdout_bytes':len(a),'stderr_bytes':len(b)})
            check('CLI '+name+' expected returncode',p.returncode==want if want==0 else p.returncode!=0)
            return a,b
        def prepared(name,fault='none'):
            task=root/name;task.mkdir();fresh=task/'cpu-admission'
            cli(name+'-prepare',['prepare','--manifest',str(mf),'--out',str(fresh),'--synthetic','--fault',fault])
            return task,fresh/'DEFINITION.json'
        def launch(name,task,definition,want=0,extra=None):
            return cli(name,['launch','--manifest',str(mf),'--definition',str(definition),'--approve-commit','f'*40,'--approve-manifest',sha(mf),'--authorize-cpu',*(extra or [])],want)
        task,definition=prepared('positive')
        check('synthetic definition clearly separated',read(definition)['mode']=='CPU_SYNTHETIC_CORRECTED_SHORT' and read(definition)['materials'] is None)
        launch('positive-run',task,definition)
        result=read(task/'attempt-short-001/slim/RESULT.json');closed=read(task/'attempt-short-001/CLOSED.json')
        check('CLOSED result SHA',closed['result_sha256']==sha(task/'attempt-short-001/slim/RESULT.json'))
        check('12 CPU forwards 6 real synthetic updates',result['ledger']=={'forward':12,'actual_updates':6,'reserved_updates':6})
        check('S A initial states exactly equal',result['arms']['S']['initial']==result['arms']['A']['initial'])
        for arm in ['S','A']:
            r=result['arms'][arm]
            check(arm+' model optimizer EMA RNG replay exact',r['replay_full_state_exact'] and r['loss_output_gradient_DN_exact'] and r['next_window_replay_exact'])
            check(arm+' actual supervision called every microbatch',all(len(x['supervision']['encode_calls'])==2 for x in r['updates']))
            check(arm+' expected parameter groups updated',all(x['changed_parameter_count']>0 for x in r['updates']))
            check(arm+' FP32 CPU synthetic explicitly marked',all(x['logits_dtype']==x['boundary_dtype']=='torch.float32' for x in r['updates']))
        j=read(task/'attempt-short-001/slim/SHA_INDEX.json')
        check('slim export SHA index',all(sha(task/'attempt-short-001/slim'/k)==v for k,v in j.items()))
        launch('duplicate-refused',task,definition,1)
        for checkpoint in task.rglob('*.pt'):checkpoint.unlink()  # owned synthetic fixture tensors only
        # Full receipt retained; synthetic tensor states are ephemeral fixtures only.
        (out/'POSITIVE_RESULT.json').write_text(json.dumps(result,indent=2))
        for fault in ['forward_exhausted','update_exhausted','nonfinite','bad_input']:
            t,d=prepared(fault,fault);launch(fault+'-run',t,d,1)
            stop=read(t/'attempt-short-001/STOP.json')
            check(fault+' closes exception without CLOSED',stop['no_retry'] and not (t/'attempt-short-001/CLOSED.json').exists())
        t,d=prepared('missing-export','missing_export');launch('missing-export-run',t,d,1)
        check('missing export never CLOSED','closed export fixture' in read(t/'attempt-short-001/STOP.json')['error'] and not (t/'attempt-short-001/CLOSED.json').exists())
        for checkpoint in t.rglob('*.pt'):checkpoint.unlink()
        t,d=prepared('wrong-identity');v=read(d);v['policy']['seed']=1;d.write_text(json.dumps(v));launch('wrong-policy-refused',t,d,1)
        check('invalid contract creates no attempt',not (t/'attempt-short-001').exists())
        t,d=prepared('wrong-authority');launch('GPU authority on CPU fixture refused',t,d,1,['--authorize-gpu'])
        # Legacy bytes/size_bytes forms use real SHA binding and strict conflict refusal.
        p=root/'metadata.bin';p.write_bytes(b'authenticated-shape-fixture')
        ref={'path':str(p),'sha256':sha(p)}
        for key in ['bytes','size_bytes']:
            check('legacy '+key,ref_exact({**ref,key:p.stat().st_size})['bytes']==p.stat().st_size)
        bad=[ref,{**ref,'bytes':True},{**ref,'bytes':-1},{**ref,'size_bytes':'27'},{**ref,'bytes':27,'size_bytes':28},{**ref,'bytes':p.stat().st_size,'sha256':'0'*64}]
        for i,r in enumerate(bad):
            try:ref_exact(r)
            except ValueError:check('invalid metadata refused '+str(i),True)
            else:check('invalid metadata refused '+str(i),False)
        # Fixed caps reject any production-budget widening.
        from sabl_corrected_short import validate
        c=read(definition)
        for field in ['forward','actual_updates','GPU_seconds','wall_seconds','allocated_bytes','new_storage_bytes','return_bytes']:
            v=json.loads(json.dumps(c));v['caps'][field]+=1
            try:validate(v)
            except ValueError:check('cap tamper refused '+field,True)
            else:check('cap tamper refused '+field,False)
        (out/'CLI_COST.json').write_text(json.dumps(cost,indent=2))
    result={'status':'PASS_ACTUAL_CORRECTED_SHORT_CPU_CLI','checks':len(checks),'CPU_synthetic_detector_forward':24,'CPU_synthetic_actual_updates':12,'GPU_forward':0,'real_images_or_weights_loaded':False,'CUDA_validation':'NOT_EXECUTED','seconds':time.monotonic()-begin,'max_tree_RSS_bytes':peak,'RSS_cap_bytes':4<<30,'temporary_synthetic_states_removed':True,'failures_preserved_as_stderr_and_receipts':True,'production_contract_or_attempt_created':False}
    write(out/'RESULT.json',result);print(json.dumps(result))

if __name__=='__main__':main(sys.argv[1])
