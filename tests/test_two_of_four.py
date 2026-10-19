import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT),str(ROOT/'src')]
from slice_core import Candidate,classwise_nms
from two_of_four_core import PAIRS,merge_pair,screen,select,coverage
from two_of_four_job import validate_plan,read

def cand(v,q=0,score=.5,box=(0,0,3,1),cat=1):
    return Candidate(box,score,v,q,cat,1)

class Tests(unittest.TestCase):
    def test_six_pairs_exact_scope(self):
        self.assertEqual(len(PAIRS),6)
        pool=[cand(v,box=(v*10,0,v*10+3,1)) for v in range(5)]
        for p in PAIRS:self.assertEqual({x.view_id for x in merge_pair(pool,p)},{0,*p})
        for p in ((0,1),(1,1),(2,1),(1,5),()):
            with self.assertRaises(ValueError):merge_pair(pool,p)

    def test_subset_must_precede_NMS(self):
        chosen=cand(1,score=.8);omitted=cand(3,score=.9)
        self.assertNotIn(chosen,classwise_nms([chosen,omitted]))
        self.assertIn(chosen,merge_pair([chosen,omitted],(1,2)))

    def test_original_stable_ties_and_identity(self):
        a,b=cand(1),cand(0)
        self.assertEqual(merge_pair([a,b],(1,2)),[b])
        a,b=cand(1,cat=1),cand(2,cat=2)
        self.assertEqual(merge_pair([b,a],(1,2)),[a,b])
        self.assertIs(merge_pair([a],(1,2))[0],a)

    def test_NMS_strict_boundary(self):
        a=cand(1);b=cand(2,box=(1,0,4,1))
        self.assertEqual(merge_pair([a,b],(1,2)),[a,b])

    def test_budget_is600_no_new_cutoff(self):
        pool=[cand(v,q,score=0.,box=(1000*v+q*3,0,1000*v+q*3+1,1)) for v in (0,1,2) for q in range(300)]
        result=merge_pair(pool,(1,2))
        self.assertEqual(len(result),600)
        self.assertTrue(all(p.view_id in (0,1) for p in result))

    def test_duplicate_and_invalid_geometry_refused(self):
        with self.assertRaises(ValueError):merge_pair([cand(1),cand(1)],(1,2))
        for box in ((0,0,0,1),(0,0,float('nan'),1)):
            with self.assertRaises(ValueError):cand(1,box=box)

    def test_screen_exact_retention_and_protections(self):
        base={a:{m:.25 for m in ('AP','APS','AP75_small','APM','APL')} for a in ('A','B','C')}
        base['C']['AP']=base['C']['APS']=.375
        p=dict(base['A'],AP=.35,APS=.35)
        # Use comfortably above the frozen boundary; comparisons use unrounded floats.
        p['AP']=p['APS']=.351
        self.assertTrue(screen(p,base)['passed'])
        for key in ('AP','APS','AP75_small','APM','APL'):
            bad=p.copy();bad[key]=0.
            self.assertFalse(screen(bad,base)['passed'])

    def test_screen_bad_values_and_reference_gain(self):
        b={a:{m:.25 for m in ('AP','APS','AP75_small','APM','APL')} for a in ('A','B','C')}
        with self.assertRaises(ValueError):screen(b['A'],b)
        b['C']['AP']=b['C']['APS']=.3
        for value in (float('nan'),float('inf'),31.):
            bad=b['C'].copy();bad['AP']=value
            with self.assertRaises(ValueError):screen(bad,b)

    def test_selection_canonical_and_all_required(self):
        d={'P'+''.join(map(str,p)):dict(passed=True) for p in PAIRS}
        self.assertEqual(select(d),'P12')
        d['P12']['passed']=False;self.assertEqual(select(d),'P13')
        for v in d.values():v['passed']=False
        self.assertIsNone(select(d))
        d.pop('P34')
        with self.assertRaises(ValueError):select(d)

    def test_frozen_plan(self):
        p=read(ROOT/'configs/sahi_two_of_four.json');validate_plan(p)
        for key,v in (('gain_retention',.7),('seed',1),('budget',300),('forward_calls',1)):
            bad=copy.deepcopy(p);bad[key]=v
            with self.assertRaises(ValueError):validate_plan(bad)

    def test_geometric_coverage_not_unique_matching(self):
        g=[[0,0,3,1],[0,0,3,1]]
        self.assertEqual(coverage([g[0]],[1],g,[1,1]).tolist(),[1.,1.])
        self.assertEqual(coverage([g[0]],[2],g,[1,1]).tolist(),[0.,0.])
        self.assertEqual(coverage([],[],g,[1,1]).tolist(),[0.,0.])

    def test_production_filter_after_budget_no_refill(self):
        from sparse_rtdetr.localization.data import parse_raw
        from sparse_rtdetr.r0.evaluation import bridge_inputs
        row=dict(stable_image_id='a',width=100,height=100)
        parsed=parse_raw(b'0,0,10,10,0,0,0,0\n20,20,10,10,1,1,0,0\n','a',expected_rows=2)
        predictions=[dict(image_id='a',category_id=1,score=s,bbox_xyxy=b) for s,b in ((.9,[0,0,5,5]),(.8,[20,20,30,30]))]
        ds,ps,c=bridge_inputs([row],[parsed],predictions[:1])
        self.assertEqual(c['detections_before'],1);self.assertEqual(c['detections_after'],0)
        self.assertEqual(bridge_inputs([row],[parsed],predictions)[2]['detections_after'],1)
        self.assertEqual(bridge_inputs([row],[parsed],predictions[1:])[2]['detections_after'],1)

    def test_actual_entry_dual_close_export_failure_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'ok'
            cmd=[sys.executable,'-B',str(ROOT/'tools/two_of_four_job.py'),'fixture','--output',str(out)]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            r=subprocess.run(cmd,capture_output=True,env=env)
            self.assertEqual(r.returncode,0,r.stderr.decode(errors='replace'))
            result=read(out/'RESULT.json');self.assertTrue(result['dual_evaluators'])
            self.assertEqual(read(out/'SLIM_EXPORT.json')['status'],'READY')
            before=(out/'CLOSED.json').read_bytes()
            self.assertNotEqual(subprocess.run(cmd,capture_output=True,env=env).returncode,0)
            self.assertEqual(before,(out/'CLOSED.json').read_bytes())
            cmd[-1]=str(Path(tmp)/'fail')
            self.assertEqual(subprocess.run(cmd+['--inject-failure'],capture_output=True,env=env).returncode,2)
            self.assertTrue((Path(tmp)/'fail/STOP.json').is_file())
            self.assertFalse((Path(tmp)/'fail/CLOSED.json').exists())

if __name__=='__main__':unittest.main()
