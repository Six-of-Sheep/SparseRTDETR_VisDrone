"""Explicit old/new source certification; original gates stay immutable and strict."""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

PARENTS=('parent_root','parent_execution','parent_execution_digest','parent_launch','seed0_contract')
ENTRY_FUNCTIONS={'parent_ready','pipeline','load_contract','main','export'}


def compatibility(old_contract,new_manifest_ref):
    import softnms_job as j
    old_manifest=json.loads(j.exact(old_contract['source_manifest'],contents=True))
    new_manifest=json.loads(j.exact(new_manifest_ref,contents=True))
    old_root=Path(old_contract['source_root']);new_root=j.ROOT
    old={x['path']:x for x in old_manifest['files']};new={x['path']:x for x in new_manifest['files']}
    changed=[]
    allowed={'softnms_job.py','softnms_cpu_fixture.py','softnms_score_launch.py','.github/workflows/repository-checks.yml'}
    for name,row in old.items():
        j.require(name in new,'old source member removed: '+name)
        j.exact({**row,'path':str(old_root/name)})
        j.exact({**new[name],'path':str(new_root/name)})
        if row!=new[name]:
            j.require(name in allowed,'unexpected changed old source member: '+name)
            changed.append({'path':name,'old_sha256':row['sha256'],'new_sha256':new[name]['sha256']})
    additions=sorted(set(new)-set(old))
    j.require(all(n in {'softnms_entry_fixture.py','softnms_reuse.py','tests/test_softnms_entry.py'}
                  or n.startswith('docs/research/') for n in additions),'unexpected new source member')
    for name in additions:j.exact({**new[name],'path':str(new_root/name)})
    old_ast=ast.parse((old_root/'softnms_job.py').read_bytes())
    new_ast=ast.parse((new_root/'softnms_job.py').read_bytes())
    def strip(module):
        module=copy.deepcopy(module)
        module.body=[x for x in module.body if not isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) or x.name not in ENTRY_FUNCTIONS]
        return ast.dump(module,include_attributes=False)
    j.require(strip(old_ast)==strip(new_ast),'scientific/helper definitions or global state changed')
    # Synthetic builder extension cannot alter the existing fixture path.
    fixture_old=ast.parse((old_root/'softnms_cpu_fixture.py').read_bytes())
    fixture_new=ast.parse((new_root/'softnms_cpu_fixture.py').read_bytes())
    fn=next(x for x in fixture_new.body if isinstance(x,ast.FunctionDef) and x.name=='run')
    fn.args.kwonlyargs=[];fn.args.kw_defaults=[]
    fn.body=[x for x in fn.body if not (isinstance(x,ast.If) and isinstance(x.test,ast.Name) and x.test.id=='build_only')]
    j.require(ast.dump(fixture_old,include_attributes=False)==ast.dump(fixture_new,include_attributes=False),'synthetic builder changed existing fixture calculation')
    return {'status':'PASS','old_source_manifest':old_contract['source_manifest'],
            'new_source_manifest':new_manifest_ref,'old_commit':old_manifest['commit'],'new_commit':new_manifest['commit'],
            'modified_existing_files':changed,'added_files':additions,'job_AST_unchanged_except':sorted(ENTRY_FUNCTIONS),
            'old_gate_source_checks':'run unchanged against original old source; never compare old SHA to new bytes',
            'science_unchanged':True}


def old_verify(request,out):
    # This is a separate interpreter; import the original verifier from its exact
    # historical source, never patch its functions, hashes or recorded gates.
    req=json.loads(request.read_bytes());ref=req['old_contract'];raw=Path(ref['path']).read_bytes()
    if len(raw)!=ref['bytes'] or hashlib.sha256(raw).hexdigest()!=ref['sha256']:raise ValueError('old contract ref differs')
    c=json.loads(raw);sys.path.insert(0,c['source_root'])
    import softnms_job as old
    from softnms_input_check import admitted_preparation as original_admission
    old.require(Path(old.__file__).parent==Path(c['source_root']),'old verifier module path differs')
    c=old.load_contract(ref['path']);old.cpu_only()
    manifest=json.loads(old.exact(c['source_manifest'],contents=True))
    old.require(manifest['files']==c['source_files'],'old manifest/contract differs')
    specs,rows,_=old.context(c);prepared=original_admission(c,specs,rows)
    cache=json.loads(old.exact(c['real_data_gate']['cache_gate'],contents=True))
    old.publish(out,{'status':'PASS','old_contract':ref,'old_source_manifest':c['source_manifest'],
        'old_source_root':c['source_root'],'old_gate_source_checked':True,'old_verifier_module':old.__file__,
        'real_data_gate':c['real_data_gate'],'parent_schema_cpu_gate':c['parent_schema_cpu_gate'],
        'context_sha256':old.digest(specs),'rows_sha256':old.digest(rows),'prepared_refs':cache['prepared'],
        'prepared_seed_identities':[p['identity'] for p in prepared],'images_each':548,'input_proofs':18,
        'prior_raw_batches_certified':411,'E_D_prediction_SHA_verified':12,'score_budget_calls':0,'GPU_calls':0})
    return 0


def prepare(request,out):
    import softnms_job as j
    started=time.monotonic();j.require(not out.exists(),'reuse output exists');out.mkdir()
    req=j.read(request);old=json.loads(j.exact(req['old_contract'],contents=True))
    new_ref=j.reference(j.ROOT.parent.parent/'FINAL_SOURCE_MANIFEST.json')
    difference=compatibility(old,new_ref)
    j.publish(out/'SOURCE_COMPATIBILITY.json',difference)
    with (out/'old-verifier.log').open('xb') as log:
        child=subprocess.run([sys.executable,'-B',str(j.ROOT/'softnms_reuse.py'),'old-verify','--request',str(request),
                              '--output',str(out/'ORIGINAL_GATE_REVERIFIED.json')],stdout=log,stderr=subprocess.STDOUT)
    j.require(child.returncode==0,'original strict source/gate verifier failed')
    old_gate=j.read(out/'ORIGINAL_GATE_REVERIFIED.json')
    j.require(old_gate['status']=='PASS' and old_gate['old_contract']==req['old_contract'],'old gate receipt differs')
    certificate={'status':'PASS','schema':'softnms-explicit-source-reuse-v1','old_contract':req['old_contract'],
        'old_verification':j.reference(out/'ORIGINAL_GATE_REVERIFIED.json'),
        'source_compatibility':j.reference(out/'SOURCE_COMPATIBILITY.json'),'new_source_manifest':new_ref,
        'scope':'18 original input proofs, 411 raw-cache rebuild certifications and prepared E/D predictions; no old source/gate mutation',
        'score_budget_calls':0,'elapsed_seconds':time.monotonic()-started}
    j.publish(out/'REUSE_CERTIFICATE.json',certificate)
    return 0


def validate_reuse(c):
    import softnms_job as j
    cert=json.loads(j.exact(c['reuse_certificate'],contents=True))
    j.require(cert['schema']=='softnms-explicit-source-reuse-v1' and cert['status']=='PASS','reuse certificate invalid')
    old=json.loads(j.exact(cert['old_contract'],contents=True))
    j.require(cert['new_source_manifest']==c['source_manifest'],'reuse new source binding differs')
    proof=json.loads(j.exact(cert['source_compatibility'],contents=True))
    j.require(compatibility(old,c['source_manifest'])==proof,'explicit source compatibility differs')
    verified=json.loads(j.exact(cert['old_verification'],contents=True))
    j.require(verified['status']=='PASS' and verified['old_contract']==cert['old_contract']
              and verified['old_gate_source_checked'] is True and verified['score_budget_calls']==0,
              'old source gate receipt invalid')
    j.require(all(c[k]==old[k] for k in PARENTS) and c['real_data_gate']==old['real_data_gate']
              and c['parent_schema_cpu_gate']==old['parent_schema_cpu_gate'],'reuse changed parent/data bindings')
    for ref in c['real_data_gate'].values():j.exact(ref)
    return verified


def admitted_reuse(c,specs,rows):
    import softnms_job as j
    verified=validate_reuse(c)
    j.require(j.digest(specs)==verified['context_sha256'] and j.digest(rows)==verified['rows_sha256'],
              'reused context/membership differs')
    result=[]
    for seed,ref in enumerate(verified['prepared_refs']):
        p=json.loads(j.exact(ref,contents=True))
        j.require(p['seed']==seed and p['identity']==specs[seed]['identity'] and p['images']==548
                  and p['raw_batches']==137 and p['all_A_B_C_exact'] is True,'reused seed preparation differs')
        j.exact(p['counts'])
        for arm in ('E','D'):
            for budget in ('600','300'):j.exact(p['predictions'][arm][budget])
        result.append(p)
    j.require(len(result)==3,'reused seeds incomplete')
    return result


def make_contract(root,source,manifest,attempt):
    import softnms_job as j
    cert_ref=j.reference(root/'reuse/REUSE_CERTIFICATE.json');cert=json.loads(j.exact(cert_ref,contents=True))
    old=json.loads(j.exact(cert['old_contract'],contents=True))
    gate_ref=j.reference(root/'entry-CPU/FINAL_ENTRY_CPU_GATE.json');gate=json.loads(j.exact(gate_ref,contents=True))
    j.require(gate['status']=='PASS' and gate['synthetic_only'] is True
              and gate['source_manifest_sha256']==j.sha(root/'FINAL_SOURCE_MANIFEST.json')
              and gate['source_files']=={r['path']:r['sha256'] for r in manifest['files']},'final real-entry CPU gate source differs')
    j.require([r['case'] for r in gate['cases']]==['success','worker-exception']
              and gate['cases'][0]['duplicate_launcher_and_run_rejected'] is True
              and all(r['supervisor_and_worker_exited'] is True for r in gate['cases']), 'final CPU lifecycle incomplete')
    c=copy.deepcopy(old)
    c.update(schema='softnms-scoring-contract-v3-explicit-reuse',attempt_id=attempt,
             output_root=str(root.parent/attempt),source_root=str(source),source_files=manifest['files'],
             source_commit=manifest['commit'],source_manifest=j.reference(root/'FINAL_SOURCE_MANIFEST.json'),
             reuse_certificate=cert_ref,final_entry_CPU_gate=gate_ref,
             scope='one fixed three-seed E/D two-budget dual-evaluator CPU scoring; unchanged scientific recipe')
    j.require('synthetic_validation' not in c,'synthetic scope leaked into scientific contract')
    specs,rows,_=j.context(c);admitted_reuse(c,specs,rows)
    return c


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','old-verify'])
    p.add_argument('--request',required=True);p.add_argument('--output',required=True)
    args=p.parse_args()
    raise SystemExit((old_verify if args.mode=='old-verify' else prepare)(Path(args.request),Path(args.output)))
