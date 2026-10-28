"""Isolated real Python/stdin/cwd integration; zero production writes or GPU."""
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sparse_rtdetr.research import sabl_formal_contract as ct


def main(output):
    resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
    checks, commands = [], []
    def check(name, ok):
        checks.append({'name':name, 'pass':bool(ok)})
        if not ok:
            raise AssertionError(name)
    def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
    def write(p, x): p.write_text(json.dumps(x, indent=2) + '\n')
    with tempfile.TemporaryDirectory(prefix='pr26-bind-stdin-') as temp:
        base = Path(temp).resolve(); task = base / 'SYNTHETIC_TASK'; task.mkdir()
        (task / 'releases').mkdir(); release = task / 'releases/CONSTRUCTED_RELEASE'; release.mkdir()
        pre = task / 'SYNTHETIC_PREFLIGHT'; pre.mkdir(); (task / 'preflight').mkdir()
        # Exact known STOP bytes, metadata only; no live historical IO.
        (task / 'preflight/STOP.json').write_bytes(b'{"status": "STOP", "gate": "MATERIAL_HIDDEN", "error": "KeyError(\'bytes\')", "no_retry": true}\n')
        check('registered_old_STOP_fixture_SHA', sha(task / 'preflight/STOP.json') == 'd042caa937755513ddca494cacac26d5da5c56562f6caf721261f687b4744178')
        names = ['tools/sabl_bind.py','tools/sabl_formal.py','tools/sabl_acceptance.py','tools/r18_deploy.py',
                 'src/sparse_rtdetr/__init__.py','src/sparse_rtdetr/research/sabl_formal_contract.py']
        rows = []
        for name in names:
            p = release / name; p.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / name, p)
            rows.append({'path':name, 'bytes':p.stat().st_size, 'sha256':sha(p)})
        commit = 'a' * 40  # Explicitly synthetic source identity, not a release claim.
        manifest = {'commit':commit,'candidate_not_for_real_execution':False,'files':rows,
                    'scope':'SYNTHETIC_MODULE_IMPORT_FIXTURE_ONLY'}
        write(release / 'RELEASE_MANIFEST.json', manifest)
        manifest_sha = sha(release / 'RELEASE_MANIFEST.json')
        write(release / 'DEPLOYED.json', {'status':'PASS','commit':commit,'candidate_not_for_real_execution':False,'manifest_sha256':manifest_sha,'files':rows})
        c = {'schema':ct.SCOPE,'mode':ct.GPU,'policy':copy.deepcopy(ct.POLICY),'caps':copy.deepcopy(ct.CAPS),
             'prior':copy.deepcopy(ct.PRIOR),'arms':list(ct.ARMS),'N_checkpoint_SHA':ct.N_SHA,
             'manifest_sha256':manifest_sha,'attempt_root':str(task / 'attempt-formal-001'),
             'metadata':{'python':sys.executable,'prior_storage_bytes':2065342933},
             'acceptance_cost':None,'gpu_uuid':'SYNTHETIC_NOT_A_GPU','fixture':None,'planned_new_forwards':44571}
        ct.validate(c)
        source = {'status':'PASS','commit':commit,'manifest_sha256':manifest_sha}
        material = {'status':'PASS','real_forward':0,'material':{'all_member_bytes_verified':{'train_core':4869,'development':548},'accepted_computation_files':300},
                    'fixed64_count':64,'N_e120_EMA':{'sha256':ct.N_SHA,'updates':36480,'epoch':120},'proposal_digest':ct.digest(c)}
        resource_receipt = {'status':'PASS','probe_forward':0,'inherited_storage_bytes':2065342933,'storage_cap_bytes':30 << 30}
        for name, value in [('PROPOSED_CONTRACT.json',c),('SOURCE.json',source),('MATERIAL_HIDDEN.json',material),('RESOURCE_VISIBLE.json',resource_receipt)]:
            write(pre / name, value)
        env = dict(os.environ); env.pop('PYTHONPATH',None); env.pop('PYTHONHOME',None)
        env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
        source_bytes = (release / 'tools/sabl_bind.py').read_bytes()
        args = ['--release',str(release),'--python',sys.executable,'--task',str(task),'--preflight',str(pre),
                '--commit',commit,'--manifest-sha',manifest_sha,'--old-stop-sha',sha(task / 'preflight/STOP.json'),'--check-only']
        def inventory(): return {p.relative_to(base).as_posix():sha(p) for p in base.rglob('*') if p.is_file()}
        def call(label, argv=None, cwd=None, payload=None, expected_error=None):
            before = inventory()
            r = subprocess.run([sys.executable,'-B','-',*(args if argv is None else argv)],
                input=source_bytes if payload is None else payload, cwd=release if cwd is None else cwd,
                env=env,capture_output=True,timeout=30)
            commands.append({'name':label,'interpreter':sys.executable,'argv':['-B','-',*(args if argv is None else argv)],
                             'cwd':str(release if cwd is None else cwd),'returncode':r.returncode,'stdout':r.stdout.decode(),'stderr':r.stderr.decode()})
            check(label + '_read_only_tree', before == inventory())
            if expected_error:
                check(label + '_refused', r.returncode != 0 and expected_error in r.stderr.decode())
            else:
                check(label + '_PASS', r.returncode == 0)
            return r
        def altered(key, value):
            value_args = list(args); value_args[value_args.index(key)+1] = value; return value_args
        # Reproduce the actual historical import failure as an expected control.
        call('unbootstrapped_stdin_negative_control',cwd=base,payload=b'import sabl_formal\n',expected_error="No module named 'sabl_formal'")
        r = call('actual_stdin_binding_check')
        result = json.loads(r.stdout)
        check('actual_stdin_interpreter_cwd_and_origin', result['stdin_script'] and result['interpreter']==sys.executable and result['cwd']==str(release) and result['formal_import']==str(release / 'tools/sabl_formal.py'))
        check('same_canonical_contract_digest', result['contract_canonical_sha256']==ct.digest(c))
        check('check_only_no_production_writes', not result['production_files_written'] and not result['job_started'] and not result['material_resource_admission_performed'])
        call('wrong_cwd',cwd=base,expected_error='release working directory')
        call('wrong_interpreter',argv=altered('--python','/not/the/certified/python'),expected_error='binding interpreter')
        call('wrong_commit',argv=altered('--commit','b'*40),expected_error='binding release version/candidate')
        call('wrong_manifest',argv=altered('--manifest-sha','0'*64),expected_error='release identity')
        call('wrong_STOP',argv=altered('--old-stop-sha','1'*64),expected_error='registered old STOP identity')
        def bad_json(name, value, label, expected):
            p=pre/name;original=p.read_bytes();write(p,value)
            call(label,expected_error=expected);p.write_bytes(original)
        v=copy.deepcopy(c);v['policy']['epochs']=25
        bad_json('PROPOSED_CONTRACT.json',v,'changed_scientific_policy','frozen policy/budget inheritance')
        v=copy.deepcopy(c);v['metadata']['prior_storage_bytes']+=1
        bad_json('PROPOSED_CONTRACT.json',v,'proposal_digest_changed','binding proposal digest')
        v=copy.deepcopy(material);v['N_e120_EMA']['updates']=1
        bad_json('MATERIAL_HIDDEN.json',v,'wrong_parent','binding parent EMA receipt')
        v=copy.deepcopy(material);v['material']['all_member_bytes_verified']['development']=547
        bad_json('MATERIAL_HIDDEN.json',v,'wrong_members','binding data receipt')
        v=copy.deepcopy(resource_receipt);v['storage_cap_bytes']-=1
        bad_json('RESOURCE_VISIBLE.json',v,'changed_storage_cap','binding storage inheritance')
        for name in ('contract.json','authorization.json','FORMAL_ONCE.json','attempt-formal-001'):
            p=task/name;p.write_text('constructed existing-output sentinel')
            call('existing_'+name,expected_error='binding existing output');p.unlink()
        (pre / 'STOP.json').write_text('{}')
        call('current_STOP',expected_error='binding current preflight STOP');(pre / 'STOP.json').unlink()
        p=release/'tools/sabl_acceptance.py';original=p.read_bytes();p.write_bytes(original+b'\n# tamper\n')
        call('source_SHA_tamper',expected_error='deployed file changed');p.write_bytes(original)
        # No successful bind/write branch is executed, even in the temporary tree.
        check('no_contract_authorization_marker_or_attempt_created', all(not (task / name).exists() for name in ('contract.json','authorization.json','FORMAL_ONCE.json','attempt-formal-001')))
    Path(output).write_text(json.dumps({'status':'PASS','scope':'SYNTHETIC_STDIN_RELEASE_BINDING_NO_PRODUCTION_IO',
        'checks':checks,'commands':commands,'actual_interpreter':sys.executable,'python':sys.version,
        'production_material_admission_exercised':False,'production_bind_executed':False,
        'model_forward':0,'optimizer_updates':0,'GPU_seconds':0,'server_connections':0},indent=2)+'\n')
    print(json.dumps({'status':'PASS','checks':len(checks),'stdin_subprocesses':len(commands),'production_bind_executed':False}))


if __name__ == '__main__':
    main(sys.argv[1])
