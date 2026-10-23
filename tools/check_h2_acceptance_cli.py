"""Exercise the ACTUAL acceptance CLI, subprocess supervision and full model.

All inputs are constructed CPU fixtures. No CUDA, real images, AP or detector
substitution. Successful temporary tensors are removed; failure receipts stay.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sparse_rtdetr.research import h2_contract as ct
from sparse_rtdetr.localization.runio import read,publish,reference,sha,digest

FILES=['h2_acceptance.py',*[f'src/sparse_rtdetr/research/{n}.py' for n in
    ('h2_acceptance','h2_contract','h2_resources','h2_worker','h2_cuda_identity')],
    'tools/check_h2_acceptance_cli.py']


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-inventory',type=Path)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    names=({x['path'] for x in read(args.source_inventory)['source_files']} if args.source_inventory else
           set(subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()))|set(FILES)
    source=out/'source.json';publish(source,dict(commit='CPU_WORKING_TREE_NOT_A_PRODUCTION_COMMIT',
        files=[dict(path=n,bytes=(ROOT/n).stat().st_size,sha256=sha(ROOT/n)) for n in sorted(names)]))
    gate=out/'BOOTSTRAP_DEPENDENCY_NOT_FINAL_GATE.json';publish(gate,dict(status='PASS',tests=13,skips=0,bootstrap_fixture=True,
        source_files=[dict(path=n,sha256=sha(ROOT/n)) for n in FILES]))
    auth=out/'authorization-fixture.json';publish(auth,dict(scope=ct.SCOPE,policy=ct.POLICY,source_sha256=sha(source),CPU_fixture_not_human_GPU_authorization=True))
    input_file=out/'CONSTRUCTED_INPUT_RECIPE.json';publish(input_file,dict(seed=1731,windows=[0,1],real_images=False))
    tests=[];counter=0
    def cli(argv,expected=0):
        nonlocal counter
        counter+=1;proc=subprocess.run([sys.executable,'-B',str(ROOT/'h2_acceptance.py'),*map(str,argv)],cwd=ROOT,env=env,capture_output=True)
        (out/f'CLI-{counter:03d}.stdout').write_bytes(proc.stdout);(out/f'CLI-{counter:03d}.stderr').write_bytes(proc.stderr)
        if (proc.returncode==0)!=(expected==0):raise RuntimeError(f'actual CLI {counter} wrong rc={proc.returncode}: '+proc.stderr.decode(errors='replace')[-2500:])
        return proc
    def spec(name,fault='none'):
        root=out/name;root.mkdir()
        prior=root/'prior.json';publish(prior,dict(task_id=ct.TASK,forward_calls=0,GPU_seconds=0,total_forward_cap=146596))
        d={role:dict(role=role,count=count,manifest=reference(input_file),roots={}) for role,count in (('train_core',4869),('development',548))}
        c=dict(profile='CPU_FIXTURE_ONLY',attempt='attempt-acceptance-001',task_root=str(root),job_root=str(root/'attempt-acceptance-001'),
            python=sys.executable,boot_id='CPU_FIXTURE_BOOT',gpu_uuid='CPU_FIXTURE_NO_GPU',
            storage_identity=dict(canonical_mount='CPU_FIXTURE',mount=dict(source='CPU_FIXTURE',rw=True),st_dev=0),
            data=d,pretrained=dict(path='CPU_SYNTHETIC_NO_PRETRAINED',bytes=0,sha256='0'*64),environment=dict(CPU_fixture=True),
            prior_cost=reference(prior),nonce='CPU-CONSTRUCTED-NONCE-'+name,fixture=dict(fault=fault))
        p=root/'spec.json';publish(p,c);return p,c
    prepared_gate=None
    def prepare_bind(name,fault='none'):
        nonlocal prepared_gate
        sp,c=spec(name,fault);root=Path(c['task_root']);prep=root/'prepare'
        if prepared_gate is None:
            cli(['prepare','--spec',sp,'--source',source,'--output',prep]);prepared_gate=prep/'PREPARE_PASS.json'
        contract=root/'contract.json'
        cli(['bind','--spec',sp,'--source',source,'--output',contract,'--cpu-gate',gate,'--prepare-gate',prepared_gate,'--authorization',auth])
        return sp,c,contract
    def record(name):tests.append(name)
    sp,c,contract=prepare_bind('success')
    cli(['contract-check','--contract',contract]);record('actual_raw_spec_prepare_bind_contract_resource_fields')
    cli(['launch','--contract',contract,'--session','cpu-h2-acceptance','--foreground'])
    cli(['verify','--contract',contract]);record('actual_CLI_three_arm_full_model_closed_and_verified')
    root=Path(c['job_root']);result=read(root/'RESULT.json');cost=read(Path(c['task_root'])/'CUMULATIVE_COST.json')
    assert result['actual_updates']==9 and result['forward_calls']==18 and cost['forward_calls']==18
    assert cost['active'] is None and all(cost['arms'][a]==dict(forward_calls=6,actual_updates=3,forwards_completed=6,updates_completed=3) for a in ct.ARMS)
    record('exact_actual_update_forward_and_future_nonreset_ledger')
    assert not any((root/n).exists() for n in ('formal','train-U','train-D','score','cost','inference'))
    record('independent_exit_no_formal_score_or_cost_dispatch')
    for a in ct.ARMS:
        v=read(root/('accept-'+a)/'RESULT.json');assert v['full_state_RNG_exact'] and set(v['logits_dtypes'])=={'torch.bfloat16'} and not v['state_usable_for_formal']
    record('actual_BF16_fullstate_RNG_next_input_output_DN_gradient_replay')
    index=read(root/'slim-export/SHA_INDEX.json')
    assert sum(x['returned']['bytes'] for x in index['files'])<=20*1024**2
    for x in index['files']:
        assert reference(x['returned']['path'])==x['returned'] and not x['name'].endswith('.pth')
    record('actual_whitelist_export_SHA_no_large_state')
    before=sha(Path(c['task_root'])/'CUMULATIVE_COST.json')
    cli(['launch','--contract',contract,'--session','cpu-h2-duplicate','--foreground'],2)
    cli(['run','--contract',contract],2)
    assert sha(Path(c['task_root'])/'CUMULATIVE_COST.json')==before
    record('duplicate_launch_and_run_refused_without_budget_reset')
    original=read(sp)
    for name,change in [('missing_resource',lambda x:x.update(resources={})),('changed_resource',lambda x:x.update(resources=dict(ct.LIMITS,GPU_stage_seconds_cap=901))),
        ('expanded_scope',lambda x:x.update(scope='formal_H2_seed0_e120')),('expanded_forward',lambda x:x.update(policy=dict(ct.POLICY,forward_calls=19)))]:
        changed=copy.deepcopy(original);change(changed);bad=Path(c['task_root'])/(name+'.json');publish(bad,changed)
        cli(['prepare','--spec',bad,'--source',source,'--output',Path(c['task_root'])/(name+'-output')],2)
        record(name+'_actual_CLI_rejection')
    for name in ('formal','score','cost'):
        cli([name,'--contract',contract],2)
    record('parser_has_no_unauthorized_stage')
    # Alter a bound input and prove the actual launcher refuses BEFORE token/root creation.
    sp2,c2,contract2=prepare_bind('input-mismatch')
    old=input_file.read_bytes();input_file.write_bytes(old+b' ')
    cli(['launch','--contract',contract2,'--session','cpu-input-mismatch','--foreground'],2)
    assert not Path(c2['job_root']).exists() and not (Path(c2['task_root'])/'LAUNCH_CONSUMED.json').exists()
    input_file.write_bytes(old);record('bound_input_tamper_refused_before_consumption')
    for fault in ('worker_exception','forward_overrun','update_overrun','time_overrun','wall_overrun','allocated_overrun','watchdog_deadline'):
        sf,cf,con=prepare_bind(fault,fault)
        cli(['launch','--contract',con,'--session','cpu-'+fault.replace('_','-'),'--foreground'],2)
        job=Path(cf['job_root']);assert (job/'STOP.json').is_file() and not (job/'CLOSED.json').exists()
        exit_record=read(job/'accept-N/supervisor-exit.json')
        assert exit_record['status']=='STOP' and not exit_record['remaining_live'] and not exit_record['cleanup_errors']
        assert (job/'slim-export/SHA_INDEX.json').is_file()
        outer=read(job/'accept-N-PROCESS-EXIT.json');assert not outer['remaining_live'] and not outer['cleanup_errors']
        if fault=='watchdog_deadline':assert 'TimeoutError' in exit_record['error']['error'] and exit_record['tracked']
        record(fault+'_actual_CLI_STOP_owned_exit_and_slim_export')
    sources=[dict(path=n,bytes=(ROOT/n).stat().st_size,sha256=sha(ROOT/n)) for n in FILES]
    report=dict(status='PASS',tests=len(tests),skips=0,cases=tests,actual_CLI_invocations=counter,
        CPU_only=True,production_GPU_validated=False,GPU_forward_calls=0,bootstrap_fixture=False,source_files=sources,
        no_real_images_or_old_checkpoint=True,normal_acceptance_actual_updates=9,normal_acceptance_forwards=18,
        time_epoch=time.time(),python=sys.version.split()[0])
    publish(out/'CPU_CLI_GATE.json',report)
    # Remove only this check's newly created tensors after all receipts pass.
    removed=[]
    for p in out.rglob('*.pth'):
        resolved=p.resolve();assert resolved.is_relative_to(out)
        removed.append(dict(path=str(p.relative_to(out)),bytes=p.stat().st_size));p.unlink()
    publish(out/'TEMPORARY_CPU_STATES_REMOVED.json',dict(status='PASS',files=removed,old_states_touched=False))
    print(json.dumps(dict(status='PASS',tests=len(tests),CLI_invocations=counter,CPU_only=True,GPU_forward_calls=0)))


if __name__=='__main__':main()
