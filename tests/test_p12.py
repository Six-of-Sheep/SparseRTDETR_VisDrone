import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import hashlib
import shutil

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tools'), str(ROOT), str(ROOT/'src')]
from p12_job import validate_plan, timing_eligible, read, evaluate


class Tests(unittest.TestCase):
    def production_release(self, root):
        # Minimal own-source snapshot, no fake parent identity or loosened parent gate.
        names = ['tools/p12_job.py','tools/two_of_four_job.py','tools/two_of_four_core.py',
                 'slice_core.py','configs/p12_cross_seed.json']
        refs = []
        for name in names:
            p = root/name; p.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,p)
            refs.append(dict(path=name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        (root/'RUN_SOURCE.json').write_text(json.dumps(dict(experiment_id='sahi-p12-cross-seed-20261019-001',
            commit='test-only-own-source',config_sha256=hashlib.sha256((root/names[-1]).read_bytes()).hexdigest(),
            source_files=refs)),encoding='utf-8')
        return root

    def prepare_process(self, root, out):
        env = dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        env.pop('PYTHONPATH',None)
        parent=root/'untrusted-parent.json';parent.write_text('{}',encoding='utf-8')
        return subprocess.run([sys.executable,'-B',str(root/'tools/p12_job.py'),'prepare',
              '--parent',str(parent),'--output',str(out)],cwd=root,env=env,capture_output=True)

    def test_real_prepare_child_bootstrap_reaches_strict_parent_refusal(self):
        with tempfile.TemporaryDirectory() as t:
            root=self.production_release(Path(t)/'release')
            out=Path(t)/'sahi-p12-cross-seed-20261019-001'/'rejected-parent'
            r=self.prepare_process(root,out)
            self.assertEqual(r.returncode,2,r.stderr.decode(errors='replace'))
            stop=read(out/'STOP.json')['traceback']
            self.assertIn('parent contract SHA',stop)
            self.assertNotIn('ModuleNotFoundError',stop)
            identity=read(out/'ENTRY_IDENTITY.json')
            self.assertEqual(Path(identity['cwd']),root)
            self.assertIsNone(identity['PYTHONPATH'])
            self.assertEqual(set(identity['modules']),{'two_of_four_job','two_of_four_core','slice_core'})
            self.assertFalse((out/'SUMMARY.json').exists())

    def test_real_prepare_child_missing_source_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=self.production_release(Path(t)/'release')
            (root/'slice_core.py').unlink()
            out=Path(t)/'sahi-p12-cross-seed-20261019-001'/'missing'
            self.assertEqual(self.prepare_process(root,out).returncode,2)
            self.assertIn('own source missing/identity differs: slice_core.py',read(out/'STOP.json')['traceback'])
            self.assertFalse((out/'ENTRY_IDENTITY.json').exists())

    def test_real_prepare_child_wrong_source_identity_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=self.production_release(Path(t)/'release')
            p=root/'tools/two_of_four_core.py';b=p.read_bytes();p.write_bytes(b.replace(b'PROTECTION',b'PROTECTIOX',1))
            out=Path(t)/'sahi-p12-cross-seed-20261019-001'/'wrong-identity'
            self.assertEqual(self.prepare_process(root,out).returncode,2)
            self.assertIn('own source missing/identity differs: tools/two_of_four_core.py',read(out/'STOP.json')['traceback'])
            self.assertFalse((out/'ENTRY_IDENTITY.json').exists())

    def test_fixed_plan_and_no_new_pair(self):
        p = read(ROOT/'configs/p12_cross_seed.json'); validate_plan(p)
        for k, v in [('pair',[1,3]),('seeds',[0,1]),('budget',300),('gain_retention',.7),
                     ('maximum_P12_over_C_seconds',.8),('training_updates',1)]:
            q = copy.deepcopy(p); q[k] = v
            with self.assertRaises(ValueError): validate_plan(q)

    def test_each_seed_gates_timing(self):
        self.assertTrue(timing_eligible({'1':dict(passed=True),'2':dict(passed=True)}))
        for a, b in ((False,True),(True,False),(False,False)):
            self.assertFalse(timing_eligible({'1':dict(passed=a),'2':dict(passed=b)}))
        for v in ({'1':dict(passed=True)}, {'2':dict(passed=True),'1':dict(passed=True)},
                  {'1':dict(passed=1),'2':dict(passed=True)}):
            with self.assertRaises(ValueError): timing_eligible(v)

    def test_actual_entry_both_dual_evaluators_close_export_exit(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t)/'ok'
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            cmd = [sys.executable,'-B',str(ROOT/'tools/p12_job.py'),'fixture','--output',str(out)]
            r = subprocess.run(cmd,capture_output=True,env=env)
            self.assertEqual(r.returncode,0,r.stderr.decode(errors='replace'))
            result = read(out/'RESULT.json')
            self.assertEqual(result['status'],'QUALITY_PASS_TIMING_PENDING')
            self.assertEqual(result['seeds_complete'],[1,2])
            self.assertEqual(result['evaluator_calls'],4)
            for s in (1,2):
                v = read(out/f'seed{s}-P12-metrics.json')
                self.assertEqual(v['official']['evaluated_image_count'],5)
            self.assertTrue((out/'SLIM_EXPORT.json').exists())
            before = (out/'CLOSED.json').read_bytes()
            self.assertNotEqual(subprocess.run(cmd,capture_output=True,env=env).returncode,0)
            self.assertEqual(before,(out/'CLOSED.json').read_bytes())
            cmd[-1] = str(Path(t)/'fail')
            r = subprocess.run(cmd+['--inject-failure'],capture_output=True,env=env)
            self.assertEqual(r.returncode,2)
            self.assertTrue((Path(t)/'fail/STOP.json').exists())
            self.assertFalse((Path(t)/'fail/CLOSED.json').exists())

    def test_quality_failure_still_finishes_second_seed(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as t:
            from two_of_four_job import publish
            out = Path(t); prepared = {}
            for s in (1,2):
                base = {a:dict(AP=.2,APS=.2,AP75_small=.2,APM=.2,APL=.2) for a in ('A','B','C')}
                base['C']['AP'] = base['C']['APS'] = .3
                prepared[str(s)] = dict(seed=s,baseline=base,predictions=publish(out/f'p{s}.json',[]))
            record = dict(COCO_bridge=dict(metrics=dict(AP=.1,APS=.1,AP75_small=.2,APM=.2,APL=.2)),
                          official={},spatial_filter_counts={})
            with patch('p12_job.score',return_value=record) as scorer:
                result = evaluate(prepared,[],[],ROOT,out)
            self.assertEqual(scorer.call_count,2)
            self.assertEqual(result['status'],'QUALITY_FAIL_CLOSED_NO_GPU')
            self.assertFalse(read(out/'SUMMARY.json')['GPU_timing_eligible'])

    def test_cross_seed_prepared_refused_before_score(self):
        with tempfile.TemporaryDirectory() as t:
            from unittest.mock import patch
            with patch('p12_job.score') as scorer:
                with self.assertRaises(ValueError):
                    evaluate({'1':dict(seed=2),'2':dict(seed=1)},[],[],ROOT,Path(t))
            scorer.assert_not_called()


if __name__ == '__main__': unittest.main()
