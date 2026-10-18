import json
import math
import unittest
from dataclasses import replace
from slice_core import Candidate,box_iou64,classwise_nms
from boxvote_core import coordinate_vote,compare_scores,METRICS

def c(q,box=(0.,0.,10.,10.),score=1.,cat=1,view=0):
    return Candidate(box,score,view,q,cat,q+1)

class VoteTests(unittest.TestCase):
    def test_hand_computation(self):
        x=coordinate_vote([c(0,score=1.),c(1,(2.,0.,12.,10.),.5)])
        self.assertEqual(x['all'][0].box_xyxy,(2/3,0.,32/3,10.))
        self.assertEqual(x['all'][0].score,1.)
    def test_strict_half_boundary(self):
        a=c(0,(0.,0.,3.,1.));b=c(1,(1.,0.,4.,1.))
        self.assertEqual(box_iou64(a.box_xyxy,b.box_xyxy),.5)
        self.assertEqual(coordinate_vote([a,b])['all'],[a,b])
    def test_zero_fallback_and_empty(self):
        x=coordinate_vote([c(0,score=0.),c(1,score=0.)]);self.assertEqual(x['stats']['zero_weight_fallbacks'],1)
        self.assertEqual(x['all'],[c(0,score=0.)]);self.assertEqual(coordinate_vote([])['all'],[])
        json.loads(json.dumps(x['stats'],allow_nan=False))
    def test_ties_classes_identity_order(self):
        pool=[c(3,cat=2),c(2,cat=1),c(0,view=1)]
        x=coordinate_vote(pool)['all'];self.assertEqual([(p.identity,p.score) for p in x],[(p.identity,p.score) for p in classwise_nms(pool)])
        self.assertEqual(x,coordinate_vote(pool[::-1])['all'])
    def test_original_neighborhood_can_vote_to_two_survivors(self):
        a=c(0,(0.,0.,10.,10.),1.);b=c(1,(5.,0.,15.,10.),.9);p=c(2,(2.5,0.,12.5,10.),.5)
        x=coordinate_vote([a,b,p]);self.assertEqual(len(x['all']),2)
        self.assertEqual(x['stats']['neighborhood_memberships'],4)
        self.assertAlmostEqual(x['all'][1].box_xyxy[0],(5*.9+2.5*.5)/1.4)
    def test_no_clip_no_rescore_prefix(self):
        pool=[c(q,(-10.+q*20.,-3.,q*20.,7.),.9) for q in range(300)]
        x=coordinate_vote(pool);self.assertEqual(x['top300'],x['top600'][:300]);self.assertLess(x['all'][0].box_xyxy[0],0)
    def test_reject_bad_geometry_score_duplicates(self):
        for box in [(0.,0.,0.,1.),(0.,0.,math.inf,1.),(1e308,0.,-1e308,1.)]:
            with self.assertRaises(ValueError):c(0,box)
        with self.assertRaises(ValueError):coordinate_vote([c(0),c(0)])
        with self.assertRaises(ValueError):c(0,score=math.nan)
    def test_gate_units_and_all_seeds_cannot_compensate(self):
        scores={a:{b:{m:.3 for m in METRICS} for b in ('600','300')} for a in ('A','B','C','V-whole','V-five')}
        result=compare_scores(scores);self.assertFalse(result['quality_passed'])
        self.assertEqual(result['gates']['V-five-C']['600']['AP']['minimum_pp'],.2)

if __name__=='__main__':unittest.main()
