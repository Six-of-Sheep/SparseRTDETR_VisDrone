"""Isolated synthetic data backend for the actual launch/run/supervisor/worker CLI."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def make_contract(request, revision, source, manifest, attempt):
    import boxvote_job as j
    req=j.read(request)
    c={'schema':'boxvote-entry-synthetic-contract-v1','task_id':j.TASK,
       'scope':'synthetic-only-no-development','synthetic_validation':True,
       'synthetic_request':j.reference(request),'fixture_context':req['context'],
       'suite_root':req['suite_root'],'fail_worker':req['fail_worker'],
       'attempt_id':attempt,'output_root':str(revision.parent/attempt),
       'source_root':str(source),'source_files':manifest['files'],'source_commit':manifest['commit'],
       'source_manifest':j.reference(revision/'FINAL_SOURCE_MANIFEST.json'),
       'plan_sha256':j.sha(j.PLAN),'python':sys.executable,'python_sha256':j.sha(sys.executable),
       'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    validate_contract(c)
    return c


def validate_contract(c):
    import boxvote_job as j
    j.require(c['synthetic_validation'] is True and c['schema']=='boxvote-entry-synthetic-contract-v1'
              and c['scope']=='synthetic-only-no-development','synthetic scope invalid')
    root=Path(c['suite_root'])
    j.require(root.is_absolute() and j.read(root/'SYNTHETIC_ONLY.json')=={'synthetic_only':True},'synthetic suite not isolated')
    req=json.loads(j.exact(c['synthetic_request'],contents=True))
    j.require(req['schema']=='boxvote-entry-fixture-request-v1' and req['suite_root']==str(root)
              and req['context']==c['fixture_context'] and type(req['fail_worker']) is bool
              and req['fail_worker']==c['fail_worker'],'synthetic request binding differs')
    j.require(Path(c['synthetic_request']['path']).is_relative_to(root)
              and Path(c['output_root']).is_relative_to(root),'synthetic path escapes suite')
    value=json.loads(j.exact(c['fixture_context'],contents=True))
    ids=[f'cpu-synthetic-{i}' for i in range(4)]
    j.require(value['synthetic_only'] is True and [r['stable_image_id'] for r in value['rows']]==ids
              and len(value['parsed'])==4 and [s['seed'] for s in value['specs']]==[0,1,2]
              and all(s['ids']==ids and s['identity']['task_id']=='cpu-synthetic' for s in value['specs']),
              'synthetic members/source differ')
    def scan(x):
        if isinstance(x,dict):
            if 'path' in x and 'sha256' in x and 'bytes' in x:
                j.require(Path(x['path']).is_relative_to(root/'data'),'fixture input escapes synthetic data')
                j.exact(x)
            for item in x.values():scan(item)
        elif isinstance(x,list):
            for item in x:scan(item)
    j.require(Path(c['fixture_context']['path']).is_relative_to(root/'data'),'fixture context escapes synthetic data')
    scan(value)
    return value


def worker(c,out):
    import boxvote_job as j
    j.cpu_only();value=validate_contract(c);plan=j.read(j.PLAN)
    j.publish(out/'RESOURCE_START.json',j.resources(out,plan))
    if c['fail_worker']:raise RuntimeError('EXPECTED_SYNTHETIC_WORKER_FAILURE')
    def progress(**fields):
        j.resources(out,plan)
        temp=out/'heartbeat.tmp';temp.write_bytes(j.encoded({'time':time.time(),**fields}));os.replace(temp,out/'heartbeat.json')
    j.process_inputs(out,value['specs'],value['rows'],j.digest(c),progress,lambda:value['parsed'])


def suite(out):
    import boxvote_job as j
    from softnms_cpu_fixture import run as build_fixture
    start=time.monotonic()
    j.require(not out.exists(),'entry CPU output already used')
    out.mkdir();j.publish(out/'SYNTHETIC_ONLY.json',{'synthetic_only':True})
    context=build_fixture(out/'data',build_only=True)
    from boxvote_flow_checks import run as flow_checks
    flow_gate=flow_checks()
    j.publish(out/'FILTER_ORDER_CPU_GATE.json',flow_gate)
    manifest=j.read(j.ROOT.parent.parent/'FINAL_SOURCE_MANIFEST.json')
    records=[]
    for name,failure in [('success',False),('worker-exception',True)]:
        revision=out/(name+'-revision');revision.mkdir()
        j.publish(revision/'FINAL_SOURCE_MANIFEST.json',manifest)
        request=j.publish(revision/'SYNTHETIC_REQUEST.json',{'schema':'boxvote-entry-fixture-request-v1',
                          'suite_root':str(out),'context':context,'fail_worker':failure})
        cmd=[sys.executable,'-B',str(j.ROOT/'boxvote_score_launch.py'),'--revision',str(revision),
             '--attempt',name,'--synthetic-request',request['path'],'--wait']
        launch=subprocess.run(cmd,capture_output=True)
        j.require(launch.returncode==(2 if failure else 0),'actual synthetic launcher exit differs: '+launch.stderr.decode('utf-8','replace'))
        receipt=j.read(revision/'SCORING_LAUNCH_RECEIPT.json');identity=receipt['start']
        deadline_progress=time.monotonic();work=out/name
        while True:
            process=j.proc(receipt['pid'])
            if process is None or process['state'] in ('Z','X'):break
            j.require(process['start']==identity,'synthetic supervisor PID reused')
            j.resources(out,j.read(j.PLAN))
            touched=max([deadline_progress]+[time.monotonic()-max(0,time.time()-p.stat().st_mtime)
                        for p in [work/'events.jsonl',work/'heartbeat.json'] if p.exists()])
            j.require(time.monotonic()-touched<j.read(j.PLAN)['progress_timeout_seconds'],'synthetic process no progress')
            time.sleep(1)
        # The actual supervisor closes its worker, then exports and exits.
        worker_exit=j.read(work/'WORKER_EXIT.json');started=j.read(work/'WORKER_STARTED.json')
        supervisor_exit=j.read(revision/'SUPERVISOR_EXIT.json')
        j.require(supervisor_exit['returncode']==launch.returncode,'launcher/supervisor exit differs')
        j.require(started['parent_pid']==receipt['pid'] and started['pid']==worker_exit['pid'],
                  'launch/supervisor/worker PID chain differs')
        live=j.proc(started['pid']);j.require(live is None or live['state'] in ('Z','X'),'worker did not exit')
        exported=j.read(work/'SLIM_EXPORT.json')
        for ref in exported['files']:j.exact(ref)
        if failure:
            j.require(worker_exit['returncode']!=0 and 'EXPECTED_SYNTHETIC_WORKER_FAILURE' in j.read(work/'STOP.json')['error']
                      and not (work/'RESULT.json').exists() and not (work/'CLOSED.json').exists(), 'worker failure not preserved/rejected')
        else:
            j.require(worker_exit['returncode']==0 and j.read(work/'CLOSED.json')['status']=='PASS','success not closed')
            result=j.read(work/'RESULT.json');j.require(result['new_score_budget_calls']==12,'synthetic dual scoring incomplete')
            j.Journal.validate(work/'events.jsonl',j.digest(j.read(revision/'execution-contract.json')),complete=True)
            refs=[r for r in exported['files'] if r['path'].endswith('-metrics.json')]
            j.require(len(refs)==12 and all('official' in json.loads(j.exact(r,contents=True))
                      and 'COCO_bridge' in json.loads(j.exact(r,contents=True)) for r in refs),'synthetic evaluators missing')
            original=(work/'events.jsonl').read_bytes().splitlines(keepends=True)
            for bad,lines in [('missing',original[:-1]),('duplicate',original+[original[-1]]),('reordered',original[:2]+[original[3],original[2]]+original[4:])]:
                path=out/(bad+'-events.jsonl');path.write_bytes(b''.join(lines))
                try:j.Journal.validate(path,j.digest(j.read(revision/'execution-contract.json')),complete=True)
                except ValueError:pass
                else:raise ValueError('bad lifecycle accepted: '+bad)
            # Reject both the actual launch script and run CLI on consumed state.
            before={str(p):j.sha(p) for p in work.rglob('*') if p.is_file()}
            duplicate_launch=subprocess.run(cmd,capture_output=True)
            duplicate_run=subprocess.run([sys.executable,'-B',str(j.ROOT/'boxvote_job.py'),'run','--contract',
                                          str(revision/'execution-contract.json')],capture_output=True)
            after={str(p):j.sha(p) for p in work.rglob('*') if p.is_file()}
            j.require(duplicate_launch.returncode!=0 and duplicate_run.returncode!=0 and before==after,
                      'duplicate startup accepted or changed consumed output')
        records.append({'case':name,'launch_command':cmd,'supervisor':receipt,'supervisor_exit':supervisor_exit,'worker':started,'worker_exit':worker_exit,
                        'export_files':len(exported['files']),'supervisor_and_worker_exited':True,
                        'result_closed':not failure,'duplicate_launcher_and_run_rejected':not failure})
    gate={'status':'PASS','synthetic_only':True,'actual_chain':'score_launch -> run -> real supervisor -> real worker -> original dual evaluators -> RESULT/CLOSED -> export -> exit',
          'cases':records,'source_manifest_sha256':j.sha(j.ROOT.parent.parent/'FINAL_SOURCE_MANIFEST.json'),
          'source_files':{r['path']:r['sha256'] for r in manifest['files']},'filter_order_gate':j.reference(out/'FILTER_ORDER_CPU_GATE.json'),'science_scores':0,'synthetic_score_budget_calls':12,
          'GPU_calls':0,'elapsed_seconds':time.monotonic()-start}
    j.publish(out/'FINAL_ENTRY_CPU_GATE.json',gate)
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    args=p.parse_args();raise SystemExit(suite(Path(args.output)))
