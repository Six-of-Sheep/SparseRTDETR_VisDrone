"""Fast stdlib policy/resource-unit checks; no CUDA or model import."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from sparse_rtdetr.research import sabl_formal_contract as ct
import sabl_formal as cli
import sabl_formal_preflight as admission


def main(output):
    checks=[]
    def check(name,ok):
        checks.append({'name':name,'pass':bool(ok)})
        if not ok: raise AssertionError(name)
    def rejects(fn):
        try: fn()
        except Exception: return True
        return False
    with tempfile.TemporaryDirectory(prefix='pr26-formal-policy-') as temp:
        root=Path(temp); c={'schema':ct.SCOPE,'mode':ct.CPU,'policy':ct.POLICY,'caps':ct.CAPS,'prior':ct.PRIOR,
            'arms':list(ct.ARMS),'N_checkpoint_SHA':ct.N_SHA,'manifest_sha256':'0'*64,'attempt_root':str(root/'attempt'),
            'metadata':None,'acceptance_cost':None,'gpu_uuid':None,'fixture':{'fault':'none'}}
        c['planned_new_forwards']=ct.new_forwards(c);ct.validate(c)
        check('CPU_exact_count',c['planned_new_forwards']==48)
        g=copy.deepcopy(c);g.update(mode=ct.GPU,fixture=None);g['planned_new_forwards']=ct.new_forwards(g)
        check('frozen_forward_arithmetic',ct.new_forwards(g)==43776+411+384==44571 and ct.CAPS['detector_forward']-ct.PRIOR['detector_forward']==44571)
        check('inherited_seconds',ct.CAPS['GPU_stage_seconds']-ct.PRIOR['GPU_stage_seconds']==86346.54610739998)
        x=copy.deepcopy(c);x['policy']=dict(x['policy'],seed=False)
        check('typed_policy_refusal',rejects(lambda:ct.validate(x)))
        out=root/'meter';out.mkdir();meter=cli.Meter(c,out,time.monotonic(),None)
        meter.begin('one');meter.forward();meter.finish();meter.begin('two');meter.forward();meter.finish()
        check('stage_no_reset',meter.ledger['new_forward']==2 and meter.ledger['total_forward']==20 and meter.ledger['stage_forwards']=={'one':1,'two':1})
        meter.limit=2;check('forward_cap_guard',rejects(meter.forward))
        meter.start-=301;check('CPU_deadline_guard',rejects(meter.guard));meter.start+=301
        sparse=out/'sparse_budget_unit_fixture.bin'
        sparse.write_bytes(b'constructed storage budget unit fixture')
        with patch.dict(ct.CAPS,storage_bytes=16):
            check('storage_stat_guard_unit',rejects(meter.guard))
        sparse.unlink()
        class FakeCUDA:
            peak=0
            def synchronize(self):pass
            def max_memory_allocated(self):return self.peak
            def reset_peak_memory_stats(self):self.peak=0
        fake=FakeCUDA();meter.telemetry.cuda=fake;meter.cuda=fake
        meter.begin('A');fake.peak=ct.CAPS['allocated_bytes']+1
        check('allocated_cap_guard',rejects(meter.guard))
        meter.telemetry.observe_peak();fake.peak=1
        check('peak_cannot_be_erased_by_reset',rejects(meter.guard))
        stage=root/'stage';stage.mkdir();cli.stage_close(stage,{'status':'PASS','endpoint':'fixture'},c)
        check('bound_stage_result',cli.closed(stage,c)['endpoint']=='fixture')
        (stage/'RESULT.json').write_text('{}')
        check('stage_SHA_conflict_refusal',rejects(lambda:cli.closed(stage,c)))
        (stage/'CLOSED.json').unlink();check('missing_stage_closure_refusal',rejects(lambda:cli.closed(stage,c)))
        fake_query='GPU-1faee6f0-1da7-4ede-2475-67a5a00274a8, NVIDIA GeForce RTX 4090 D, 00000000:01:00.0, 24064\n'
        env={'metadata':{'python':sys.executable},'gpu_uuid':'GPU-1faee6f0-1da7-4ede-2475-67a5a00274a8'}
        for name,returns,pass_expected in [('resource_unit',[fake_query,''],True),('busy_unit',[fake_query,'123\n'],False),('wrong_GPU_unit',[fake_query.replace('1faee6f0','00000000')],False)]:
            folder=root/name;folder.mkdir();attempt=folder/'attempt';attempt.mkdir()
            with patch.object(admission.subprocess,'check_output',side_effect=returns):
                ok=not rejects(lambda:admission.resources(env,attempt))
            check(name,ok==pass_expected)
        with patch.object(admission.subprocess,'check_output',side_effect=[fake_query,'']):
            check('parent_once_marker_refusal',rejects(lambda:admission.resources(env,root/'resource_unit/attempt')))
    reuse=ct.read(ROOT/'docs/research/r18_sabl_control/GPU_EVIDENCE_REUSE.json')
    check('accepted_worktree_source_equivalence_EOL',all(hashlib.sha256((ROOT/r['path']).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==r['worktree_lf_sha256'] for r in reuse['unchanged_files']))
    check('native_sources_not_replaced',len(reuse['unchanged_files'])==300 and not reuse['new_GPU_acceptance'])
    check('no_torch_or_CUDA_import', 'torch' not in sys.modules)
    result={'status':'PASS','scope':'STDLIB_CONSTRUCTED_RESOURCE_UNITS_NOT_REAL_ADMISSION','checks':checks,'model_forward':0,'GPU_cost':0}
    Path(output).write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main(sys.argv[1])
