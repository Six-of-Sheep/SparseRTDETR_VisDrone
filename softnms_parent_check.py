"""Real-parent, metadata-only preflight; faults never edit original evidence."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import time
import traceback
from unittest.mock import patch


def run(request, out):
    import softnms_job as j
    j.require(not out.exists(), 'parent-check attempt already exists')
    out.mkdir(parents=True)
    start = time.monotonic()
    try:
        torch = j.cpu_only()
        c = j.read(request)['parent_bindings']
        j.require(j.parent_ready(c), 'original parent chain not complete')
        j.publish(out/'RESOURCE.json', j.resources(out, j.read(j.PLAN)))
        accessed = {}
        real_read, real_exact, real_sha = j.read, j.exact, j.sha

        def safe_read(path):
            p = Path(path)
            j.require(p.suffix == '.json', 'metadata check forbids non-JSON read')
            accessed[str(p)] = j.reference(p)
            return real_read(p)

        def safe_exact(ref, **kwargs):
            j.require(Path(ref['path']).suffix == '.json', 'metadata check forbids raw/model access')
            accessed[ref['path']] = dict(ref)
            return real_exact(ref, **kwargs)

        def forbidden(*args, **kwargs):
            raise AssertionError('metadata check attempted scientific execution')

        with patch.object(j, 'read', safe_read), patch.object(j, 'exact', safe_exact), \
                patch.object(torch, 'load', forbidden), patch.object(j, 'process_inputs', forbidden):
            specs, rows, data = j.context(c)
        j.require([s['seed'] for s in specs] == [0,1,2] and len(rows) == 548, 'real context incomplete')
        j.publish(out/'REAL_CONTEXT.json', {'status':'PASS','seeds':[0,1,2], 'images':len(rows),
                  'identities':[s['identity'] for s in specs], 'metadata':list(accessed.values()),
                  'raw_cache_reads':0,'score_budget_calls':0,'model_constructions':0,'GPU_calls':0})
        rejected = []

        def reject(name, call):
            try:
                call()
            except (ValueError, KeyError, FileNotFoundError) as e:
                rejected.append({'name':name,'error':repr(e)})
            else:
                raise AssertionError('invalid metadata was accepted: '+name)

        # Semantic mutations are injected AFTER real archive SHA verification.
        # Originals remain immutable. The disk corruption checks below exercise SHA itself.
        def read_fault(seed, phase, mutate):
            target = Path(specs[seed]['root'])/phase/'runtime-archive/identity.json'
            def altered(path):
                value = real_read(path)
                if Path(path) == target:
                    mutate(value)
                return value
            with patch.object(j, 'read', altered):
                j.parent_metadata(c)

        for seed in range(3):
            for phase in ('full-C','evaluation'):
                for key in ('task_id','source_sha256','contract_sha256','output_root','runtime_root','plan_sha256'):
                    reject(f's{seed}/{phase}/conflict/{key}', lambda s=seed,p=phase,k=key:
                           read_fault(s,p,lambda x:x.__setitem__(k,'WRONG')))
                reject(f's{seed}/{phase}/missing-task', lambda s=seed,p=phase:
                       read_fault(s,p,lambda x:x.pop('task_id')))
                key='phase' if seed==0 else 'mode'
                reject(f's{seed}/{phase}/wrong-phase',lambda s=seed,p=phase,k=key:
                       read_fault(s,p,lambda x:x.__setitem__(k,'accept-A')))
                if seed:
                    reject(f's{seed}/{phase}/cross-seed',lambda s=seed,p=phase:
                           read_fault(s,p,lambda x:x.__setitem__('seed',3-s)))
                    reject(f's{seed}/{phase}/checkpoint',lambda s=seed,p=phase:
                           read_fault(s,p,lambda x:x['evaluation_authorization'].__setitem__('checkpoint_sha256','0'*64)))
                else:
                    reject(f's0/{phase}/checkpoint',lambda p=phase:
                           read_fault(0,p,lambda x:x.__setitem__('checkpoint_sha256','0'*64)))
        # Preserve explicit legacy-result conflict and full-C association rejection.
        target=Path(specs[0]['root'])/'evaluation/RESULT.json'
        for key in ('task_id','checkpoint_sha256','full_C_result','schema'):
            def fault(k=key):
                def altered(path):
                    value=real_read(path)
                    if Path(path)==target:value[k]='WRONG'
                    return value
                with patch.object(j,'read',altered):j.parent_metadata(c)
            reject('legacy-result/'+key,fault)
        # Real bytes, real exact(); no mocks and no parent writes.
        original=Path(specs[0]['root'])/'evaluation/runtime-archive/identity.json'
        p=out/'isolated-identity.json';p.write_bytes(original.read_bytes());ref=j.reference(p)
        raw=bytearray(p.read_bytes());raw[-2]^=1;p.write_bytes(raw)
        reject('real-disk-corruption',lambda:j.exact(ref))
        missing=copy.deepcopy(c);missing['seed0_contract']['path']=str(out/'absent.json')
        reject('missing-parent-file',lambda:j.parent_metadata(missing))
        wrong=copy.deepcopy(c);wrong['seed0_contract']['sha256']='0'*64
        reject('real-contract-SHA-mismatch',lambda:j.parent_metadata(wrong))
        swapped=copy.deepcopy(c);swapped['seed0_contract']=j.read(c['parent_execution']['path'])['contracts'][0]
        reject('cross-seed-contract-substitution',lambda:j.parent_metadata(swapped))
        # Original metadata remained byte-identical through all injected faults.
        for ref in accessed.values():real_exact(ref)
        j.require(not torch.cuda.is_initialized(),'CUDA initialized')
        j.publish(out/'PARENT_CPU_GATE.json',{'status':'PASS','real_context_seeds':[0,1,2],
                  'rejections':rejected,'rejection_count':len(rejected),'elapsed_seconds':time.monotonic()-start,
                  'semantic_fault_method':'post-authentication read injection; originals unchanged',
                  'SHA_fault_method':'real corrupted isolated bytes and wrong reference SHA',
                  'old_algorithm_gate':'REUSED_UNCHANGED','raw_cache_reads':0,'score_budget_calls':0,
                  'model_constructions':0,'GPU_calls':0,'training_updates':0})
        print(json.dumps({'status':'PASS','rejections':len(rejected),'seconds':time.monotonic()-start}))
        return 0
    except BaseException as e:
        j.publish(out/'STOP.json',{'status':'STOP_NO_RETRY','error':repr(e),'traceback':traceback.format_exc(),
                  'elapsed_seconds':time.monotonic()-start})
        print(traceback.format_exc())
        return 2
