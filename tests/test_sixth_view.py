import unittest
import math
from slice_core import Candidate,merge_candidates,box_iou64
from sixth_view_core import (SixthCandidate,center_window,map_extra_xyxy,
                             merge_six,extra_candidates)

def pool(view,cls=Candidate):
    return [cls((q*20.,cat*20.,q*20.+10,cat*20.+10),.5,view,q,cat,(cat-1)*300+q+1)
            for cat in (1,2) for q in range(300)]

class SixthTests(unittest.TestCase):
    def test_odd_even_center_exact_halfopen(self):
        for h,w in [(768,1344),(769,1345),(1,1),(2,3),(9,9),(1080,1920)]:
            x,y,x2,y2=center_window(h,w)
            self.assertEqual(x2-x,math.ceil(5*w/9));self.assertEqual(y2-y,math.ceil(5*h/9))
            self.assertEqual(x,(w-(x2-x))//2);self.assertEqual(y,(h-(y2-y))//2)
            self.assertTrue(0<=x<x2<=w and 0<=y<y2<=h)
    def test_center_outside_not_clipped(self):
        x,y,_,_=center_window(101,203)
        self.assertEqual(map_extra_xyxy([(-20.,-5.,300.,150.)],101,203,'C-center'),[(x-20.,y-5.,x+300.,y+150.)])
    def test_flip_roundtrip_continuous_and_outside(self):
        for w in (1,2,1345):
            b=[(-30.,-20.,1500.,99.),(0.,0.,float(w),20.),(.125,.25,.875,10.5)]
            self.assertEqual(map_extra_xyxy(map_extra_xyxy(b,100,w,'C-flip'),100,w,'C-flip'),b)
    def test_disabled_C_exact(self):
        b=sum([pool(v) for v in range(5)],[])
        self.assertEqual(merge_six(b),merge_candidates(b))
        with self.assertRaises(ValueError):merge_six(b,[])
    def test_sixth_ties_original_ids_win_prefix_zero_retained(self):
        b=sum([pool(v) for v in range(5)],[]);e=pool(5,SixthCandidate)
        for variant in ('C-center','C-flip'):
            x=merge_six(b,e,variant=variant)
            self.assertTrue(all(p.view_id==0 for p in x['top600']))
            self.assertEqual(x['top300'],x['top600'][:300]);self.assertEqual(x['candidate_count'],3600)
        z=[SixthCandidate((q*20.,cat*20.,q*20.+10,cat*20.+10),0.,5,q,cat,(cat-1)*300+q+1) for cat in (1,2) for q in range(300)]
        self.assertEqual(len(merge_six(b,z,variant='C-center')['top600']),600)
    def test_extra_top600_ranking_and_identity(self):
        boxes=[(q*20.,0.,q*20.+10.,10.) for q in range(300)];scores=[[0.]*10 for _ in range(300)]
        ps=extra_candidates(boxes,scores)
        self.assertEqual(len(ps),600);self.assertTrue(all(p.view_id==5 for p in ps))
        self.assertEqual([(p.query_id,p.category_id) for p in ps[:11]],[(0,i) for i in range(1,11)]+[(1,1)])
    def test_old_class_unchanged_versioned_extension(self):
        with self.assertRaises(ValueError):pool(5)
        self.assertEqual(SixthCandidate.from_dict(pool(5,SixthCandidate)[0].as_dict()),pool(5,SixthCandidate)[0])
        with self.assertRaises(ValueError):pool(6,SixthCandidate)
    def test_bad_fields_missing_pool_and_geometry_refused(self):
        with self.assertRaises(ValueError):center_window(0,9)
        for b in [(0.,0.,0.,1.),(0.,0.,math.nan,1.)]:
            with self.assertRaises(ValueError):map_extra_xyxy([b],100,100,'C-flip')
        with self.assertRaises(ValueError):map_extra_xyxy([],100,100,'scan')
        with self.assertRaises(ValueError):merge_six([],pool(5,SixthCandidate),variant='C-center')
    def test_pre_registered_quality_control_and_units(self):
        from sixth_view_decision import assess_quality,METRICS
        scores={a:{b:{m:.3 for m in METRICS} for b in ('600','300')} for a in ('A','C','C-center','C-flip')}
        for b in ('600','300'):
            scores['C-center'][b]={m:.31 for m in METRICS}
            scores['C-flip'][b]={m:.32 for m in METRICS}
        x=assess_quality(scores)
        self.assertTrue(x['main_14_quality_passed']);self.assertFalse(x['quality_permits_next_seeds'])
        self.assertFalse(x['center_position_extra_value_supported']);self.assertTrue(x['flip_must_not_be_promoted'])
        scores['C-flip']['600']={m:.305 for m in METRICS}
        self.assertTrue(assess_quality(scores)['quality_permits_next_seeds'])
        scores['C-flip']['600']['AP']=scores['C-center']['600']['AP']
        self.assertFalse(assess_quality(scores)['quality_permits_next_seeds'])
        scores['C-center']['600']['AP']=31.
        with self.assertRaises(ValueError):assess_quality(scores)

if __name__=='__main__':unittest.main()
