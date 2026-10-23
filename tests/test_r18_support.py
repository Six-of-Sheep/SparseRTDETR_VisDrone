"""Constructed CPU cases only; no detector/checkpoint/images or backward."""
import functools
import sys
import types
import unittest
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'tools'),
                str(Path(__file__).resolve().parents[1]/'src')]
import torch
from torch.nn import functional as F
from r18_support_math import support, stencil, capture, coordinates
from sparse_rtdetr.research.sampling_span import FinestSpanAttention, bind_common_core, MSDeformableAttention
from sparse_rtdetr.baseline import training_v2b as vendor


class SupportTests(unittest.TestCase):
    def test_invalid_stencil_motion_is_not_effective_change(self):
        from r18_support_math import effective_change
        left=torch.tensor([[[-1.,-1.]]]);right=torch.tensor([[[-2.,-2.]]])
        self.assertFalse(effective_change(left,right,(2,4)))
        self.assertTrue(effective_change(torch.tensor([[[.125,.25]]]),right,(2,4)))

    def test_budget_boundary_refusals(self):
        from r18_support_launch import bounds
        bounds(599.9,20*1024**2,16,18*1024**3)
        for values in ((600,0,0,0),(1,20*1024**2+1,0,0),(1,0,17,0),(1,0,0,18*1024**3+1)):
            with self.assertRaises(RuntimeError):
                bounds(*values)

    def test_immutable_output_and_input_reference(self):
        import tempfile
        from r18_support_diagnostic import publish,reference,exact
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'receipt.json'
            publish(p,{'status':'PASS'})
            r=reference(p)
            self.assertEqual(exact(r)['status'],'PASS')
            with self.assertRaises(FileExistsError):
                publish(p,{'status':'FAIL'})
            p.write_text('{"status":"FAIL"}')
            with self.assertRaises(RuntimeError):
                exact(r)

    def test_center_boundary_and_empty_support(self):
        loc = torch.tensor([[[.125,.25]], [[0.,0.]], [[-1.,-1.]]])
        a = torch.ones(3,1)
        s = support(loc,a,(2,4),[0,0,.25,.5])
        self.assertAlmostEqual(s['I'],1.25/3)
        self.assertAlmostEqual(s['V'],1.75/3)
        self.assertEqual(s['M'],1.)
        z = support(loc[2:],a[2:],(2,4),[0,0,1,1])
        self.assertIsNone(z['M'])
        self.assertEqual(z['V'],1.)
        self.assertIsNone(support(loc,a*0,(2,4),[0,0,1,1])['I'])

    def test_half_open_gt_center(self):
        loc = torch.tensor([[[.125,.25]]])
        a = torch.ones(1,1)
        self.assertEqual(support(loc,a,(2,4),[0,0,.125,1])['I'],0.)
        self.assertEqual(support(loc,a,(2,4),[.125,.25,.2,.3])['I'],1.)

    def test_manual_interpolation_matches_grid_sample(self):
        loc = torch.tensor([[[.4,.6],[0.,0.],[1.,1.],[-1.,.5]]],dtype=torch.float64)
        n,w,v = stencil(loc,(2,4))
        field = torch.arange(8,dtype=torch.float64).reshape(1,1,2,4)
        expected = F.grid_sample(field,loc.unsqueeze(0)*2-1,align_corners=False,padding_mode='zeros')
        acc = torch.zeros_like(w[...,0])
        for j in range(4):
            x = n[...,j,0].clamp(0,3).long()
            y = n[...,j,1].clamp(0,1).long()
            acc += field[0,0,y,x]*w[...,j]*v[...,j]
        torch.testing.assert_close(acc,expected[0,0])

    def test_pool_before_dividing(self):
        loc = torch.tensor([[[.125,.25]], [[.875,.75]]])
        s = support(loc,torch.tensor([[.9],[.1]]),(2,4),[0,0,.5,1])
        self.assertAlmostEqual(s['M'],.9,places=6)
        self.assertNotEqual(s['M'],.5)

    def test_unavailable_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            support(torch.full((1,1,2),float('nan')),torch.ones(1,1),(2,4),[0,0,1,1])

    def test_floor_zero_offset_not_span_guarantee(self):
        from sparse_rtdetr.research.sampling_span import sampling_references
        ref = torch.tensor([[[[.5,.5,.1,.8]],[[.5,.5,.3,.2]]]])
        shapes = [(2,4),(1,2),(1,1)]
        expanded = sampling_references(ref,shapes)
        torch.testing.assert_close(expanded[0,0,0,2:],torch.tensor([.25,.8]))
        torch.testing.assert_close(expanded[0,1,0,2:],torch.tensor([.3,.5]))
        zero = torch.zeros(1,2,1,12,2)
        out = coordinates(expanded[:,:,:1],zero,shapes[0],torch.tensor(1/4))
        self.assertEqual(float((out.amax(-2)-out.amin(-2)).max()),0.)

    def test_capture_native_leaf_output_state_rng_invariant(self):
        torch.manual_seed(33)
        for arm in ('P','B0'):
            layers = torch.nn.ModuleList()
            for _ in range(3):
                leaf = FinestSpanAttention(32,8) if arm=='P' else MSDeformableAttention(32,8,3,[4,4,4])
                layer = torch.nn.Module()
                layer.cross_attn = bind_common_core(leaf)
                layers.append(layer)
            model = torch.nn.Module().eval()
            model.decoder = types.SimpleNamespace(decoder=types.SimpleNamespace(layers=layers))
            layers.eval()
            q=torch.randn(1,300,32);value=torch.randn(1,11,32)
            r=torch.rand(1,300,1,4)*.2+.01
            r[:,:,:,0:2]=.5
            shapes=[(2,4),(1,2),(1,1)]
            old_rng=torch.get_rng_state().clone()
            old_state=[{k:v.clone() for k,v in x.state_dict().items()} for x in layers]
            with torch.inference_mode():
                expected=[x.cross_attn(q,r,value,shapes) for x in layers]
                buf=[]
                with capture(model,arm,buf):
                    actual=[x.cross_attn(q,r,value,shapes) for x in layers]
            for a,b in zip(expected,actual):
                self.assertTrue(torch.equal(a,b))
            for before,layer in zip(old_state,layers):
                self.assertTrue(all(torch.equal(v,layer.state_dict()[k]) for k,v in before.items()))
            self.assertTrue(torch.equal(old_rng,torch.get_rng_state()))
            self.assertEqual(len(buf),3)
            if arm=='P':
                self.assertGreater(float(buf[0]['output_abs_mean'].max()),0)

    def test_native_matching_final_queries_and_empty(self):
        with vendor._vendor_path(vendor._vendor_root(Path(__file__).resolve().parents[1])):
            from src.zoo.rtdetr.matcher import HungarianMatcher
        matcher=HungarianMatcher(dict(cost_class=2,cost_bbox=5,cost_giou=2),True,.25,2.)
        box=torch.tensor([[[.2,.2,.1,.1],[.8,.8,.1,.1]]])
        out=dict(pred_logits=torch.tensor([[[8.,-8.],[-8.,8.]]]),pred_boxes=box)
        t=[dict(labels=torch.tensor([1,0]),boxes=box[0].flip(0))]
        i,j=matcher(out,t)['indices'][0]
        self.assertEqual(i.tolist(),[0,1]); self.assertEqual(j.tolist(),[1,0])
        empty=matcher(out,[dict(labels=torch.empty(0,dtype=torch.long),boxes=torch.empty(0,4))])
        self.assertEqual(len(empty['indices'][0][0]),0)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
