"""Preparation and immutable scoring contract, separate from scientific execution."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

PARENT_SHA='b2ac33e7a99cfa4e8ea3ccce7297f5088272d839f98fae7b7e13c4e2a7f46c28'


def prepare(parent,out):
    import boxvote_job as j
    j.require(not out.exists(),'preparation output exists');out.mkdir(parents=True)
    started=time.monotonic();j.publish(out/'START.json',{'parent':j.reference(parent),'pid':os.getpid()})
    try:
        j.require(j.sha(parent)==PARENT_SHA,'certified parent contract SHA differs')
        old=j.read(parent)
        for row in old['source_files']:
            j.exact({**row,'path':str(Path(old['source_root'])/row['path'])})
        # Use the original strict verifier against original source, not new bytes.
        script="""import sys,json
from pathlib import Path
c=json.loads(Path(sys.argv[1]).read_bytes());sys.path.insert(0,c['source_root'])
import softnms_job as j
c=j.load_contract(sys.argv[1]);s,r,d=j.context(c)
from softnms_reuse import validate_reuse
v=validate_reuse(c)
j.require(j.digest(s)==v['context_sha256'] and j.digest(r)==v['rows_sha256'],'old certified context changed')
proof=json.loads(j.exact(c['real_data_gate']['input_proof'],contents=True))
j.require(proof['status']=='PASS' and len(proof['proofs'])==18,'original input proof incomplete')
print(json.dumps({'status':'PASS','context':j.digest(s),'rows':j.digest(r),'input_proof':c['real_data_gate']['input_proof'],'old_source_manifest':c['source_manifest']}))
"""
        result=subprocess.run([sys.executable,'-B','-c',script,str(parent)],capture_output=True)
        (out/'original-verifier.log').write_bytes(result.stdout+result.stderr)
        j.require(result.returncode==0,'original strict verifier failed')
        verified=json.loads(result.stdout);j.publish(out/'ORIGINAL_GATE_REVERIFIED.json',verified)
        torch=j.cpu_only();before=torch.get_rng_state().clone()
        specs,rows,data=j.context(old)
        j.require(j.digest(specs)==verified['context'] and j.digest(rows)==verified['rows'],'new context differs')
        parsed=j.parsed_gt(rows,data)
        j.publish(out/'METADATA_GATE.json',{'status':'PASS','context_sha256':j.digest(specs),'rows_sha256':j.digest(rows),
                  'images':len(rows),'seeds':[s['seed'] for s in specs],'original_gate':j.reference(out/'ORIGINAL_GATE_REVERIFIED.json')})
        prepared=[];filter_counts=[]
        def progress(**fields):
            j.resources(out,j.read(j.PLAN))
            temp=out/'heartbeat.tmp';temp.write_bytes(j.encoded({'time':time.time(),**fields}));os.replace(temp,out/'heartbeat.json')
            print(json.dumps(fields),flush=True)
        from sparse_rtdetr.r0.evaluation import bridge_inputs
        for spec in specs:
            dest=out/f'seed{spec["seed"]}';dest.mkdir()
            p=j.prepare_seed(spec,rows,dest,progress)
            for arm in ('V-whole','V-five'):
                for budget in ('600','300'):
                    preds=json.loads(j.exact(p['predictions'][arm][budget],contents=True))
                    _,_,counts=bridge_inputs(rows,parsed,preds)
                    filter_counts.append({'seed':spec['seed'],'arm':arm,'budget':int(budget),**counts})
            prepared.append(j.publish(dest/'PREPARED.json',p))
        j.publish(out/'FILTER_COUNTS.json',filter_counts)
        j.require(not torch.cuda.is_initialized() and torch.equal(before,torch.get_rng_state()),'preparation RNG/CUDA changed')
        manifest=j.read(j.ROOT.parent.parent/'FINAL_SOURCE_MANIFEST.json')
        gate={'status':'PASS','parent_contract':j.reference(parent),'plan_sha256':j.sha(j.PLAN),
              'source_files':manifest['files'],'context_sha256':j.digest(specs),'rows_sha256':j.digest(rows),
              'metadata_gate':j.reference(out/'METADATA_GATE.json'),'input_proof':verified['input_proof'],
              'input_proof_status':'REUSED_18_ORIGINAL_SHA_AND_BEFORE_AFTER_CERTIFIED',
              'cache_rebuild_status':'411_RAW_BATCHES_A_B_C_EXACT','prepared':prepared,
              'filter_counts':j.reference(out/'FILTER_COUNTS.json'),'score_budget_calls':0,'GPU_calls':0,
              'elapsed_seconds':time.monotonic()-started}
        j.publish(out/'DATA_GATE.json',gate)
        return 0
    except BaseException as e:
        j.publish(out/'STOP.json',{'error':repr(e),'traceback':traceback.format_exc(),'elapsed_seconds':time.monotonic()-started,'automatic_retry':False})
        return 2


def make_contract(root,source,manifest,attempt):
    import boxvote_job as j
    gate=j.reference(root/'admission/DATA_GATE.json');g=json.loads(j.exact(gate,contents=True))
    cpu=j.reference(root/'entry-CPU/FINAL_ENTRY_CPU_GATE.json');v=json.loads(j.exact(cpu,contents=True))
    j.require(g['status']==v['status']=='PASS' and g['source_files']==manifest['files'],'gates incomplete')
    j.require(v['source_files']=={r['path']:r['sha256'] for r in manifest['files']},'CPU/source binding differs')
    j.require(all(x['supervisor_and_worker_exited'] for x in v['cases']) and v['cases'][0]['duplicate_launcher_and_run_rejected'],'entry not accepted')
    return {'schema':'boxvote-scoring-contract-v1','task_id':j.TASK,'attempt_id':attempt,
            'output_root':str(root.parent/attempt),'source_root':str(source),'source_files':manifest['files'],
            'source_commit':manifest['commit'],'source_manifest':j.reference(root/'FINAL_SOURCE_MANIFEST.json'),
            'admission_gate':gate,'final_entry_CPU_gate':cpu,'plan_sha256':j.sha(j.PLAN),
            'python':sys.executable,'python_sha256':j.sha(sys.executable),
            'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'scope':'one fixed three-seed coordinate-vote CPU scoring; 12 dual evaluations; conditional later GPU cost only',
            'automatic_retry':False,'training_updates':0,'GPU_calls':0}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();raise SystemExit(prepare(Path(a.parent),Path(a.output)))
