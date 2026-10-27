"""Actual package/deploy/formal CLI on constructed CPU tensors only."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main(base):
    base=Path(base); base.mkdir(); checks=[]; commands=[]; began=time.monotonic()
    def read(p): return json.loads(Path(p).read_text())
    def write(p,v): Path(p).write_text(json.dumps(v))
    def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def call(script,*args):
        env=dict(os.environ); env.pop('PYTHONPATH',None); env.pop('PYTHONHOME',None)
        r=subprocess.run([sys.executable,'-B',str(script),*map(str,args)],cwd=base,env=env,capture_output=True,text=True,timeout=300)
        commands.append({'script':str(script),'args':list(map(str,args)),'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
        write(base/'COMMANDS.json',commands)
        return r.returncode
    def check(name,ok):
        checks.append({'name':name,'pass':bool(ok)})
        write(base/'CHECKS.json',checks)
        if not ok:
            print(json.dumps(commands[-1])); raise AssertionError(name)
    def retire_constructed_tensors(folder):
        # Only disposable tensors created by this checker. Never historical or
        # production states. Retain SHA/index plus all negative STOP receipts.
        refs=[]
        for p in folder.rglob('*.pt'):
            assert p.resolve().is_relative_to(base.resolve())
            refs.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'synthetic_tensor_removed':True})
            p.unlink()
        if refs:write(folder/'SYNTHETIC_TENSOR_INDEX.json',refs)
    pkg=base/'package'; release=base/'release'
    check('package_formal_actual_entry',call(ROOT/'tools/package_sabl.py',pkg,'--candidate','--formal')==0)
    h=sha(pkg/'MANIFEST.json'); entry=release/'tools/sabl_formal.py'
    check('deploy_actual_entry',call(pkg/'deploy.py','--bundle',pkg/'source.tar.gz','--manifest',pkg/'MANIFEST.json','--sha256',h,'--out',release)==0)
    contract=base/'contract.json'; run=base/'attempt'
    check('plan_actual_entry',call(entry,'plan','--synthetic','--manifest-sha',h,'--attempt-root',run,'--out',contract)==0)
    check('full_formal_CLI_constructed_only',call(entry,'run','--contract',contract,'--out',run)==0)
    result=read(run/'slim/RESULT.json'); ledger=result['ledger']; c=read(contract)
    check('cross_phase_three_epochs_all_arms',all(read(run/('train-'+a)/'endpoint.json')['updates']==3 for a in ('N','S','A')))
    check('exact_48_forward_9_updates_synthetic',ledger['new_forward']==48 and ledger['actual_updates']==ledger['reserved_updates']==9)
    check('prior_budget_not_reset',ledger['inherited']=={'detector_forward':18,'GPU_stage_seconds':53.45389260002412} and ledger['total_forward']==66 and ledger['GPU_stage_seconds']==53.45389260002412)
    check('dual_evaluators_two_budgets',read(run/'score/RESULT.json')['evaluator_calls']==12 and len(list((run/'score').glob('[NSA]-*.json')))==6)
    check('six_images_partial_batch_coverage',all(read(run/('infer-'+a)/'RESULT.json')['forward_calls']==2 for a in ('N','S','A')))
    check('balanced_same_session_cost',len(read(run/'cost/RESULT.json')['warmups'])==6 and all(len(v)==6 for v in read(run/'cost/RESULT.json')['measurements'].values()))
    check('all_comparisons_and_original_N',all(s in (run/'compare/TABLE.md').read_text() for s in ('S-N24','S-A24','S-originalN','N24-originalN','A24-N24')))
    check('not_scientific_no_followup',result['scientific_status']=='NOT_SCIENTIFIC' and result['must_exit'] and not result['acceptance_state_used'])
    check('closed_and_exit',read(run/'CLOSED.json')['status']=='PASS_ENGINEERING' and read(run/'EXIT.json')['returncode']==0 and not (run/'STOP.json').exists())
    check('slim_SHA_and_cap',all(sha(run/'slim'/k)==v['sha256'] for k,v in read(run/'slim/SHA_INDEX.json').items()) and sum(p.stat().st_size for p in (run/'slim').rglob('*') if p.is_file())<5<<20)
    before=sha(run/'CLOSED.json'); check('duplicate_attempt_no_overwrite',call(entry,'run','--contract',contract,'--out',run)!=0 and sha(run/'CLOSED.json')==before)
    retire_constructed_tensors(run)
    for name,key,value in [('scope','mode','bad'),('parent','N_checkpoint_SHA','0'*64),('arms','arms',['N','S','D']),('prior','prior',{'detector_forward':0,'GPU_stage_seconds':0}),('cap','caps',dict(c['caps'],detector_forward=44590)),('policy','policy',dict(c['policy'],epochs=25))]:
        x=copy.deepcopy(c); target=base/name; x.update(attempt_root=str(target)); x[key]=value; p=base/(name+'.json'); write(p,x)
        check(name+'_actual_CLI_refusal',call(entry,'run','--contract',p,'--out',target)!=0 and (target/'STOP.json').exists())
    for fault in ('worker_exception','forward_exhausted','nonfinite','bad_endpoint','missing_close','bad_input'):
        target=base/fault; x=copy.deepcopy(c); x.update(attempt_root=str(target),fixture={'fault':fault}); p=base/(fault+'.json'); write(p,x)
        check(fault+'_actual_CLI_STOP',call(entry,'run','--contract',p,'--out',target)!=0 and read(target/'STOP.json')['no_retry'] and not (target/'CLOSED.json').exists())
        retire_constructed_tensors(target)
    actual_forward=sum(read(p)['new_forward'] for p in base.glob('*/ledger.json'))
    actual_updates=sum(read(p)['actual_updates'] for p in base.glob('*/ledger.json'))
    report={'status':'PASS','checks':checks,'scope':'CPU_SYNTHETIC_NOT_SCIENTIFIC','seconds':time.monotonic()-began,
            'CPU_detector_forward_reserved':actual_forward,'CPU_optimizer_updates':actual_updates,'real_GPU_forward':0,'manifest_sha256':h}
    write(base/'ACCEPTANCE.json',report); print(json.dumps(report))


if __name__=='__main__': main(sys.argv[1])
