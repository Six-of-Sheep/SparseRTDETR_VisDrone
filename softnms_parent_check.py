"""Full real-parent snapshot regression, no model/cache/scoring execution."""
from __future__ import annotations
import base64
import copy
import hashlib
import json
from pathlib import Path
import time
import traceback
from unittest.mock import patch


class Snapshot:
    def __init__(self, path):
        self.value = json.loads(Path(path).read_bytes())
        self.files = self.value['files']
        for name, entry in self.files.items():
            b = base64.b64decode(entry['payload_b64'], validate=True)
            r = entry['reference']
            if name != r['path'] or len(b) != r['bytes'] or hashlib.sha256(b).hexdigest() != r['sha256']:
                raise ValueError('snapshot bytes do not match original reference: '+name)
        self.accessed = set()
        self.fault = None

    def reference(self, path):
        return copy.deepcopy(self.files[str(path)]['reference'])

    def exact(self, ref, *, contents=False):
        name = ref['path']; entry = self.files[name]; self.accessed.add(name)
        r = entry['reference']
        if r['sha256'] != ref['sha256'] or r['bytes'] != ref.get('bytes',ref.get('size_bytes')):
            raise ValueError('snapshot exact SHA/size differs: '+name)
        b = base64.b64decode(entry['payload_b64'])
        if contents:
            obj = json.loads(b) if Path(name).suffix == '.json' else b.decode('utf-8')
            if self.fault:self.fault(name, obj)
            return (json.dumps(obj, ensure_ascii=False, allow_nan=False).encode()
                    if Path(name).suffix == '.json' else obj.encode())
        return Path(name)

    def read(self, path):
        name = str(path); self.accessed.add(name)
        obj = copy.deepcopy(self.files[name]['value'])
        if self.fault:self.fault(name, obj)
        return obj


def run(request, out, snapshot=None, diagnostic=False):
    import softnms_job as j
    j.require(snapshot is not None, 'explicit authenticated metadata snapshot required')
    j.require(not out.exists(), 'parent-check output already exists')
    out.mkdir(parents=True)
    start = time.monotonic()
    try:
        torch = j.cpu_only()
        store = Snapshot(snapshot)
        c = store.value['request']
        j.require(j.read(request)['parent_bindings'] == c, 'snapshot/request differs')
        real_sha, real_reference = j.sha, j.reference
        def snap_ref(path):
            if str(path) in store.files:return store.reference(path)
            j.require(Path(path).is_relative_to(out), 'unexpected output reference')
            return real_reference(path)
        def snap_sha(path):
            if str(path) in store.files:return store.reference(path)['sha256']
            j.require(Path(path).is_relative_to(j.ROOT), 'snapshot missing requested parent source')
            return real_sha(path)
        def forbidden(*args, **kwargs):raise AssertionError('metadata regression forbids scientific execution')
        findings=[];rejected=[]
        with patch.object(j,'read',store.read),patch.object(j,'exact',store.exact), \
             patch.object(j,'reference',snap_ref),patch.object(j,'sha',snap_sha), \
             patch.object(torch,'load',forbidden),patch.object(j,'process_inputs',forbidden):
            ex=store.read(c['parent_execution']['path'])
            contracts=[(0,json.loads(store.exact(c['seed0_contract'],contents=True)))]+[
                (i+1,json.loads(store.exact(r,contents=True))) for i,r in enumerate(ex['contracts'])]
            # Diagnostic gathers all schema paths before an acceptance attempt exists.
            for seed,ct in contracts:
                for name,call in [('full-C',lambda:j.closed(ct['job_root'],'full-C',ct)),
                                  ('evaluation',lambda:j.closed(ct['job_root'],'evaluation',ct)),
                                  ('endpoint',lambda:j.endpoint_checkpoint(ct,seed)),
                                  ('generating-sources',lambda:j.authenticate_parent_sources(ct,seed))]:
                    try:call()
                    except Exception as e:findings.append({'seed':seed,'path':name,'error':repr(e)})
            try:specs,rows,data=j.context(c)
            except Exception as e:findings.append({'path':'complete-context','error':repr(e)})
            if diagnostic:
                j.publish(out/'DIAGNOSIS.json',{'status':'FINDINGS' if findings else 'COMPLETE_NO_UNRESOLVED_FINDINGS',
                          'findings':findings,'seeds':[0,1,2],'accessed_small_files':len(store.accessed),
                          'snapshot':str(snapshot),'score_budget_calls':0,
                          'elapsed_seconds':time.monotonic()-start})
                print(json.dumps({'diagnosis_findings':findings,'small_files':len(store.accessed)}))
                return 0
            j.require(not findings,'unresolved real-parent diagnostic findings: '+repr(findings))
            j.require([s['seed'] for s in specs]==[0,1,2] and len(rows)==548,'real context incomplete')
            def reject(name,target,mutation):
                def fault(path,obj):
                    if path==target:mutation(obj)
                store.fault=fault
                try:
                    j.context(c)
                except (ValueError,KeyError,FileNotFoundError) as e:rejected.append({'name':name,'error':repr(e)})
                else:raise AssertionError('invalid snapshot accepted: '+name)
                finally:store.fault=None
            mutations={'missing':lambda x:x[:-1], 'duplicate':lambda x:x[:-1]+[x[0]],
                       'order':lambda x:[x[1],x[0],*x[2:]]}
            for seed,ct in contracts:
                root=Path(ct['job_root'])
                for phase in ('full-C','evaluation'):
                    target=str(root/phase/'runtime-archive/identity.json')
                    for key in ('task_id','source_sha256','contract_sha256','output_root','runtime_root','plan_sha256'):
                        reject(f's{seed}/{phase}/{key}',target,lambda x,k=key:x.__setitem__(k,'WRONG'))
                    reject(f's{seed}/{phase}/missing-task',target,lambda x:x.pop('task_id'))
                    reject(f's{seed}/{phase}/phase',target,lambda x:x.__setitem__('phase' if seed==0 else 'mode','accept-A'))
                    for label,fn in mutations.items():
                        def mutate(x,f=fn):
                            y=x if seed==0 else x['evaluation_authorization'];y['image_ids']=f(y['image_ids'])
                        reject(f's{seed}/{phase}/members-{label}',target,mutate)
                    def cp_mutation(x):
                        y=x if seed==0 else x['evaluation_authorization'];y['checkpoint_sha256']='0'*64
                    reject(f's{seed}/{phase}/checkpoint',target,cp_mutation)
                    if seed:reject(f's{seed}/{phase}/cross-seed',target,lambda x:x.__setitem__('seed',3-seed))
                cr=c['seed0_contract'] if seed==0 else ex['contracts'][seed-1]
                for key in (('full_image_ids',) if seed==0 else ('full_image_ids','development_ids')):
                    for label,fn in mutations.items():
                        reject(f's{seed}/contract/{key}-{label}',cr['path'],lambda x,k=key,f=fn:x.__setitem__(k,f(x[k])))
                target=str(root/'evaluation/RESULT.json')
                for key in ('task_id','checkpoint_sha256','schema'):
                    reject(f's{seed}/result/{key}',target,lambda x,k=key:x.__setitem__(k,'WRONG'))
                if seed==0:reject('s0/full-C-association',target,lambda x:x.__setitem__('full_C_result',{}))
                for arm in ('A','B','C'):
                    ref=specs[seed]['evaluation']['metrics'][arm]['600']
                    reject(f's{seed}/{arm}/score-conflict',ref['path'],lambda x:x['COCO_bridge']['metrics'].__setitem__('AP',.123456789))
                    if seed or arm!='A':
                        reject(f's{seed}/{arm}/prediction-binding',ref['path'],lambda x:x['sahi_binding'].__setitem__('prediction',{}))
            for label,fn in mutations.items():
                reject('manifest/'+label,data['manifest'],lambda x,f=fn:x.__setitem__('records',f(x['records'])))
            for seed in range(3):
                ref=c['seed0_contract'] if seed==0 else ex['contracts'][seed-1]
                bad={**ref,'sha256':'0'*64}
                try:j.exact(bad)
                except ValueError as e:rejected.append({'name':f's{seed}/SHA-conflict','error':repr(e)})
                else:raise AssertionError('wrong snapshot SHA accepted')
            bad=copy.deepcopy(c);bad['seed0_contract']=ex['contracts'][0]
            try:j.context(bad)
            except (ValueError,KeyError) as e:rejected.append({'name':'cross-seed-contract','error':repr(e)})
            else:raise AssertionError('cross-seed contract accepted')
        # Real filesystem corruption, separate from authenticated snapshot semantic faults.
        p=out/'isolated-SHA.json';p.write_text('{}\n');ref=j.reference(p);p.write_text('[]\n')
        try:j.exact(ref)
        except ValueError as e:rejected.append({'name':'real-disk-SHA-corruption','error':repr(e)})
        else:raise AssertionError('corrupt bytes accepted')
        j.require(not torch.cuda.is_initialized(),'CUDA initialized')
        j.publish(out/'PARENT_CPU_GATE.json',{'status':'PASS','seeds':[0,1,2],
                  'source_files':{n:j.sha(j.ROOT/n) for n in
                                  ('softnms_job.py','softnms_parent_check.py','softnms_core.py','slice_core.py')},
                  'plan_sha256':j.sha(j.PLAN),
                  'real_snapshot':j.reference(snapshot),'rejections':rejected,'rejection_count':len(rejected),
                  'positive':'complete parent_metadata/context plus generating source and metric bindings',
                  'semantic_faults':'after SHA authentication, immutable snapshot read injection',
                  'evaluation_548_input_digest_proof':'REQUIRED_AT_PRODUCTION_ADMISSION_BEFORE_CANDIDATES',
                  'unaffected_algorithm_gate':'REUSED','raw_candidate_reads':0,'score_budget_calls':0,
                  'GPU_calls':0,'training_updates':0,'elapsed_seconds':time.monotonic()-start})
        print(json.dumps({'status':'PASS','rejections':len(rejected),'seconds':time.monotonic()-start}))
        return 0
    except BaseException as e:
        j.publish(out/'STOP.json',{'status':'STOP_NO_RETRY','error':repr(e),'traceback':traceback.format_exc(),
                  'elapsed_seconds':time.monotonic()-start})
        print(traceback.format_exc())
        return 2
