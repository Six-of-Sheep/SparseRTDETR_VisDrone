import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tools'), str(ROOT), str(ROOT/'src')]
from p12_job import validate_plan, timing_eligible, read, evaluate


class Tests(unittest.TestCase):
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
