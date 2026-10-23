"""Constructed CPU features only; gradients are checks, no optimizer or training."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from sparse_rtdetr.research.local_detail_center import (
    locations,local_samples,odd_even,compose_box,LocalDetailCenter,inverse_sigmoid)


class DetailTests(unittest.TestCase):
    def setUp(self):
        self.ref=torch.tensor([[[.5,.5,.4,.2]]],dtype=torch.float64)
        self.ctx=torch.tensor([[[.7,-.6,.2,-.3]]],dtype=torch.float64)
        self.feature=torch.arange(5*9,dtype=torch.float64).reshape(1,1,5,9)

    def test_cardinal_nonsquare_and_no_floor(self):
        expected=torch.tensor([[[[.6,.5],[.4,.5],[.5,.55],[.5,.45]]]],dtype=torch.float64)
        torch.testing.assert_close(locations(self.ref),expected)
        tiny=self.ref.clone();tiny[...,2:]=1e-5
        span=locations(tiny).amax(-2)-locations(tiny).amin(-2)
        self.assertLess(float(span.max()),1e-4)

    def test_multibatch_multichannel_readout_state(self):
        torch.manual_seed(4)
        leaf=LocalDetailCenter(5).double()
        with torch.no_grad():leaf.readout.weight.normal_()
        feature=torch.randn(2,5,6,10,dtype=torch.float64)
        ref=self.ref.expand(2,3,4).clone();ctx=self.ctx.expand(2,3,4).clone()
        value=leaf(feature,ref,ctx)
        self.assertEqual(tuple(value['sampled'].shape),(2,3,4,5))
        self.assertEqual(tuple(value['box'].shape),(2,3,4))
        other=LocalDetailCenter(5).double();other.load_state_dict(leaf.state_dict())
        self.assertTrue(torch.equal(value['box'],other(feature,ref,ctx)['box']))

    def test_sampler_matches_original_cpu_grid(self):
        from torch.nn.functional import grid_sample
        actual=local_samples(self.feature,self.ref)
        expected=grid_sample(self.feature,locations(self.ref)*2-1,align_corners=False,padding_mode='zeros').permute(0,2,3,1)
        torch.testing.assert_close(actual,expected,rtol=0,atol=1e-14)

    def test_even_odd_are_reversible_not_new_information(self):
        s=local_samples(self.feature,self.ref);d,m=odd_even(s)
        reconstructed=torch.stack((m[...,:1]+d[...,:1],m[...,:1]-d[...,:1],
                                   m[...,1:]+d[...,1:],m[...,1:]-d[...,1:]),-2)
        torch.testing.assert_close(s,reconstructed)

    def test_fixed_input_common_mode_cancels_only_interior(self):
        d,_=odd_even(local_samples(self.feature,self.ref))
        shifted,_=odd_even(local_samples(self.feature+19,self.ref))
        torch.testing.assert_close(d,shifted,atol=1e-14,rtol=0)

    def test_signed_information_changes_under_pair_swap(self):
        s=local_samples(self.feature,self.ref);d,m=odd_even(s)
        d2,m2=odd_even(s[:,:,[1,0,3,2]])
        torch.testing.assert_close(d2,-d);torch.testing.assert_close(m2,m)

    def test_pdc_folds_to_conventional_linear_weights(self):
        torch.manual_seed(7)
        sample=torch.randn(2,3,4,5,dtype=torch.float64)
        w=torch.randn(2,10,dtype=torch.float64);d,_=odd_even(sample)
        folded=torch.cat((w[:,:5]/2,-w[:,:5]/2,w[:,5:]/2,-w[:,5:]/2),-1)
        torch.testing.assert_close(d@w.T,sample.flatten(-2)@folded.T)
        # Thus operator novelty is explicitly NOT claimed.

    def test_zero_detail_blocks_context_center_but_not_size(self):
        leaf=LocalDetailCenter(1).double()
        out=leaf(torch.ones_like(self.feature),self.ref,self.ctx)
        torch.testing.assert_close(out['box'][...,:2],self.ref[...,:2])
        self.assertFalse(torch.equal(out['box'][...,2:],self.ref[...,2:]))
        control=leaf(torch.ones_like(self.feature),self.ref,self.ctx,additive_control=True)
        self.assertFalse(torch.equal(control['box'][...,:2],self.ref[...,:2]))

    def test_flat_feature_wrong_anchor_is_a_real_counterexample(self):
        leaf=LocalDetailCenter(1).double()
        with torch.no_grad():leaf.readout.weight.fill_(1)
        pred=leaf(torch.ones_like(self.feature),self.ref,self.ctx)['box']
        gt=torch.tensor([[[.6,.5]]],dtype=torch.float64)
        self.assertGreater(float((pred[...,:2]-gt).abs().sum()),.09)
        torch.testing.assert_close(pred[...,:2],self.ref[...,:2])

    def test_background_ramp_can_generate_false_center_motion(self):
        leaf=LocalDetailCenter(1).double()
        with torch.no_grad():leaf.readout.weight.copy_(torch.eye(2,dtype=torch.float64))
        out=leaf(self.feature,self.ref,self.ctx)
        self.assertGreater(float(out['detail_delta'].abs().max()),0)

    def test_zero_padding_breaks_common_mode_property_at_boundary(self):
        ref=torch.tensor([[[.02,.5,.4,.2]]],dtype=torch.float64)
        odd,_=odd_even(local_samples(torch.ones_like(self.feature),ref))
        self.assertGreater(float(odd.abs().max()),0)

    def test_sampling_detach_and_feature_gradients(self):
        f=self.feature.clone().requires_grad_();r=self.ref.clone().requires_grad_()
        d,_=odd_even(local_samples(f,r));d.square().sum().backward()
        self.assertIsNone(r.grad)
        self.assertTrue(torch.isfinite(f.grad).all());self.assertGreater(float(f.grad.abs().sum()),0)

    def test_native_aux_output_reference_gradient_is_separate(self):
        ref=self.ref.clone().requires_grad_();ctx=self.ctx.clone().requires_grad_()
        detail=torch.tensor([[[.1,.2]]],dtype=torch.float64,requires_grad=True)
        compose_box(ref,detail,ctx).sum().backward()
        self.assertGreater(float(ref.grad.abs().sum()),0)
        torch.testing.assert_close(ctx.grad[...,:2],torch.zeros_like(ctx.grad[...,:2]))
        self.assertGreater(float(ctx.grad[...,2:].abs().sum()),0)
        self.assertTrue(torch.isfinite(detail.grad).all())

    def test_zero_init_is_not_a_dead_readout(self):
        leaf=LocalDetailCenter(1).double()
        out=leaf(self.feature,self.ref,self.ctx)
        torch.testing.assert_close(out['detail_delta'],torch.zeros_like(out['detail_delta']))
        out['box'][...,:2].sum().backward()
        self.assertGreater(float(leaf.readout.weight.grad.abs().sum()),0)

    def test_invalid_shape_nonfinite_negative_extent_rejected(self):
        for r in [torch.zeros(1,4),torch.full((1,1,4),float('nan')),torch.tensor([[[.5,.5,-.1,.2]]])]:
            with self.assertRaises(ValueError):locations(r)
        with self.assertRaises(ValueError):local_samples(torch.full_like(self.feature,float('nan')),self.ref)

    def test_native_inverse_sigmoid_expression(self):
        x=torch.tensor([0.,1e-7,.5,1-1e-7,1.],dtype=torch.float64)
        expected=(x.clip(0,1).clip(min=1e-5)/(1-x.clip(0,1)).clip(min=1e-5)).log()
        self.assertTrue(torch.equal(inverse_sigmoid(x),expected))


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
