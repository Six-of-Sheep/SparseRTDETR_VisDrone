"""Actual generator + binary sender + real bash -s + frozen binder check-only."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from sparse_rtdetr.research import sabl_formal_contract as ct
import lf_transport as transport


def main(a):
    resource.setrlimit(resource.RLIMIT_AS,(4<<30,4<<30))
    checks=[]
    def check(name,ok):
        checks.append({'name':name,'pass':bool(ok)})
        if not ok:raise AssertionError(name)
    def write(p,v):p.write_bytes((json.dumps(v,indent=2)+'\n').encode('utf-8'))
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix='pr26-lf-real-bash-') as temp:
        root=Path(temp).resolve();task=root/'SYNTHETIC_TASK';task.mkdir()
        release=task/'releases/CONSTRUCTED_RELEASE';release.mkdir(parents=True)
        pre=task/'SYNTHETIC_PREFLIGHT';pre.mkdir();(task/'preflight').mkdir()
        (task/'preflight/STOP.json').write_bytes(b'{"status": "STOP", "gate": "MATERIAL_HIDDEN", "error": "KeyError(\'bytes\')", "no_retry": true}\n')
        rows=[]
        for name in ('tools/sabl_bind.py','tools/sabl_formal.py','tools/sabl_acceptance.py','tools/r18_deploy.py','src/sparse_rtdetr/__init__.py','src/sparse_rtdetr/research/sabl_formal_contract.py'):
            p=release/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,p)
            rows.append({'path':name,'bytes':p.stat().st_size,'sha256':sha(p)})
        commit='a'*40
        write(release/'RELEASE_MANIFEST.json',{'commit':commit,'candidate_not_for_real_execution':False,'files':rows,'scope':'SYNTHETIC_TRANSPORT_IMPORT_ONLY'})
        manifest=sha(release/'RELEASE_MANIFEST.json')
        write(release/'DEPLOYED.json',{'status':'PASS','commit':commit,'candidate_not_for_real_execution':False,'files':rows,'manifest_sha256':manifest})
        c={'schema':ct.SCOPE,'mode':ct.GPU,'policy':copy.deepcopy(ct.POLICY),'caps':copy.deepcopy(ct.CAPS),'prior':copy.deepcopy(ct.PRIOR),
           'arms':list(ct.ARMS),'N_checkpoint_SHA':ct.N_SHA,'manifest_sha256':manifest,'attempt_root':str(task/'attempt-formal-001'),
           'metadata':{'python':sys.executable,'prior_storage_bytes':2065342933},'acceptance_cost':None,'gpu_uuid':'SYNTHETIC_NOT_GPU','fixture':None,'planned_new_forwards':44571}
        ct.validate(c)
        write(pre/'PROPOSED_CONTRACT.json',c)
        write(pre/'SOURCE.json',{'status':'PASS','commit':commit,'manifest_sha256':manifest})
        write(pre/'MATERIAL_HIDDEN.json',{'status':'PASS','real_forward':0,'material':{'all_member_bytes_verified':{'train_core':4869,'development':548},'accepted_computation_files':300},'fixed64_count':64,'N_e120_EMA':{'sha256':ct.N_SHA,'updates':36480,'epoch':120},'proposal_digest':ct.digest(c)})
        write(pre/'RESOURCE_VISIBLE.json',{'status':'PASS','probe_forward':0,'inherited_storage_bytes':2065342933,'storage_cap_bytes':30<<30})
        argv=[sys.executable,'-B','-','--release',str(release),'--python',sys.executable,'--task',str(task),'--preflight',str(pre),'--commit',commit,'--manifest-sha',manifest,'--old-stop-sha',sha(task/'preflight/STOP.json'),'--check-only']
        os.environ.pop('PYTHONPATH',None);os.environ.pop('PYTHONHOME',None)
        os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
        before={p.relative_to(task).as_posix():sha(p) for p in task.rglob('*') if p.is_file()}
        payload=transport.shell(release,argv,release/'tools/sabl_bind.py')
        script=root/'binding.sh';digest=transport.write_payload(script,payload)
        check('explicit_UTF8_no_BOM_LF_bytes',not payload.startswith(b'\xef\xbb\xbf') and b'\r' not in payload and payload.decode('utf-8').endswith('\n'))
        result=transport.send(script,['bash','-s'],root/'bind-receipt')
        check('real_bash_s_actual_sender_success',result['returncode']==0)
        output=json.loads(result['stdout'])
        check('cwd_actual_interpreter_and_import',output['cwd']==str(release) and output['interpreter']==sys.executable and output['formal_import']==str(release/'tools/sabl_formal.py'))
        check('Python_received_stdin_script',output['stdin_script'])
        check('canonical_binding_validation',output['contract_canonical_sha256']==ct.digest(c) and output['status']=='PASS_BINDING_CHECK_ONLY')
        check('actual_sent_payload_SHA',result['sent_payload_sha256']==digest==sha(script))
        check('no_fixture_or_production_binding_writes',before=={p.relative_to(task).as_posix():sha(p) for p in task.rglob('*') if p.is_file()} and not output['production_files_written'])
        # Shell quoting, Unicode/space/quote/$ arguments through the same generator/sender.
        args=['含 空格',"quote'\"",'$HOME; $(false)']
        probe=transport.shell(release,[sys.executable,'-B','-c','import json,os,sys;print(json.dumps([os.getcwd(),sys.argv[1:]]))',*args])
        p=root/'args.sh';transport.write_payload(p,probe);r=transport.send(p,['bash','-s'],root/'args-receipt')
        check('arguments_and_working_directory_exact',r['returncode']==0 and json.loads(r['stdout'])==[str(release),args])
        p=root/'exit.sh';transport.write_payload(p,transport.shell(release,['bash','-c','exit 7']))
        r=transport.send(p,['bash','-s'],root/'exit-receipt')
        check('nonzero_returncode_preserved',r['returncode']==7)
        wrong=list(argv);wrong[wrong.index('--python')+1]='/not/the/certified/python'
        p=root/'wrong.sh';transport.write_payload(p,transport.shell(release,wrong,release/'tools/sabl_bind.py'))
        r=transport.send(p,['bash','-s'],root/'wrong-receipt')
        check('binder_rejection_returned_through_bash',r['returncode']!=0 and 'binding interpreter' in r['stderr'])
        legacy=Path(a.legacy_payload).read_bytes() if a.legacy_payload else b'set -euo pipefail\r\nexit 0\r\n'
        check('legacy_payload_contains_CRLF',b'\r\n' in legacy)
        rejected=[]
        for label,blob in [('legacy_CRLF',legacy),('BOM',b'\xef\xbb\xbfexit 0\n'),('bare_CR',b'exit 0\r'),('invalid_UTF8',b'\xff\n')]:
            p=root/(label+'.sh');p.write_bytes(blob);receipt=root/(label+'-receipt')
            try:transport.send(p,['bash','-s'],receipt)
            except (ValueError,UnicodeError):rejected.append(label)
            else:raise AssertionError('invalid payload sent: '+label)
            check(label+'_rejected_before_invocation',not Path(str(receipt)+'.begin.json').exists())
        check('legacy_original_bytes_unchanged',not a.legacy_payload or Path(a.legacy_payload).read_bytes()==legacy)
        try:transport.send(script,['bash','-s'],root/'bind-receipt')
        except FileExistsError:check('same_receipt_replay_refused',True)
        else:check('same_receipt_replay_refused',False)
        check('no_attempt_authorization_or_marker',all(not (task/x).exists() for x in ('contract.json','authorization.json','FORMAL_ONCE.json','attempt-formal-001')))
    report={'status':'PASS','checks':checks,'real_bash_s_invocations':4,'legacy_payload_SHA':hashlib.sha256(legacy).hexdigest(),'legacy_payload_from_actual_failure':bool(a.legacy_payload),'generator_and_binary_sender_exercised':True,'production_binding_executed':False,'model_forward':0,'updates':0,'GPU_seconds':0,'server_connections':0,'python':sys.version,'interpreter':sys.executable}
    Path(a.output).write_bytes((json.dumps(report,indent=2)+'\n').encode('utf-8'))
    print(json.dumps({'status':'PASS','checks':len(checks),'real_bash_s_invocations':4,'legacy_payload_SHA':report['legacy_payload_SHA']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('--legacy-payload');main(p.parse_args())
