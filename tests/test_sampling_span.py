"""Synthetic CPU geometry/integration checks; no data, model weights or AP."""
import copy
import io
import unittest

import torch

from sparse_rtdetr.research.sampling_span import (
    MSDeformableAttention, FinestSpanAttention, adapt_cpu_model, bind_common_core,
    from_baseline, sampling_references,
)
from sparse_rtdetr.baseline import training_v2b as vendor
from sparse_rtdetr.protocol_r18.model import Config, build


class SamplingSpanTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.shapes = [(8, 12), (4, 6), (2, 3)]
        self.base = bind_common_core(MSDeformableAttention(32, 4, 3, [4, 4, 4]))
        self.q = torch.randn(2, 5, 32)
        self.v = torch.randn(2, 126, 32)
        self.ref = torch.tensor([0.35, 0.45, 0.005, 0.01]).repeat(2, 5, 1, 1)

    def test_disabled_exact_output_and_input_unchanged(self):
        p = from_baseline(self.base, 'P'); p.span_enabled = False
        original = self.ref.clone()
        self.assertTrue(torch.equal(self.base(self.q,self.ref,self.v,self.shapes),
                                    p(self.q,self.ref,self.v,self.shapes)))
        self.assertTrue(torch.equal(original, self.ref))

    def test_above_one_cell_exact_output(self):
        ref = self.ref.clone(); ref[...,2:] = 0.3
        p = from_baseline(self.base,'P')
        self.assertTrue(torch.equal(self.base(self.q,ref,self.v,self.shapes),
                                    p(self.q,ref,self.v,self.shapes)))

    def test_point_order_and_anisotropic_floor(self):
        original = self.ref.clone(); x = sampling_references(self.ref,self.shapes)
        self.assertEqual(tuple(x.shape), (2,5,12,4))
        self.assertTrue(torch.equal(x[...,:2],self.ref[...,:2].expand(-1,-1,12,-1)))
        torch.testing.assert_close(x[:,:,0:4,2],torch.full((2,5,4),1/12))
        torch.testing.assert_close(x[:,:,0:4,3],torch.full((2,5,4),1/8))
        self.assertTrue(torch.equal(x[:,:,4:,2:],self.ref[...,2:].expand(-1,-1,8,-1)))
        self.assertTrue(torch.equal(original,self.ref))

    def test_mixed_spans_axiswise_and_empty_queries(self):
        ref = self.ref.clone(); ref[...,2] = 0.5
        x = sampling_references(ref,self.shapes)
        self.assertTrue(torch.equal(x[...,2],ref[...,2].expand(-1,-1,12)))
        self.assertEqual(sampling_references(ref[:,:0],self.shapes).shape[1],0)

    def test_original_outside_centers_not_clipped(self):
        ref = self.ref.clone(); ref[...,:2] = -2
        x = sampling_references(ref,self.shapes)
        self.assertTrue(torch.equal(x[...,:2],ref[...,:2].expand(-1,-1,12,-1)))
        out = from_baseline(self.base,'P')(self.q,ref,self.v,self.shapes)
        self.assertTrue(torch.isfinite(out).all())
        # All gathers are outside support; projection bias is initialized zero.
        self.assertEqual(out.count_nonzero().item(),0)

    def test_real_native_core_receives_only_finest_location_change(self):
        p = from_baseline(self.base,'P'); captured = []
        for module in (self.base,p):
            core = module.ms_deformable_attn_core
            def observe(*args, _core=core, **kwargs):
                captured.append(args[2].detach().clone())
                return _core(*args, **kwargs)
            module.ms_deformable_attn_core = observe
            module(self.q,self.ref,self.v,self.shapes)
        self.assertFalse(torch.equal(captured[0][...,:4,:],captured[1][...,:4,:]))
        self.assertTrue(torch.equal(captured[0][...,4:,:],captured[1][...,4:,:]))

    def test_finite_cpu_autograd_not_scientific_probe(self):
        p = from_baseline(self.base,'P')
        q=self.q.clone().requires_grad_();v=self.v.clone().requires_grad_()
        out=p(q,self.ref,v,self.shapes);out.square().sum().backward()
        for gradient in (q.grad,v.grad,p.sampling_offsets.weight.grad,p.value_proj.weight.grad):
            self.assertIsNotNone(gradient);self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(p.sampling_offsets.weight.grad.abs().sum().item(),0)

    def test_variable_query_counts_including_dn_and_native_dims(self):
        native=bind_common_core(MSDeformableAttention(256,8,3,[4,4,4]))
        p=from_baseline(native,'P')
        for count in (1,300,500):
            q=torch.zeros(1,count,256);v=torch.randn(1,21,256)
            r=torch.tensor([0.5,0.5,0.01,0.01]).repeat(1,count,1,1)
            self.assertEqual(p(q,r,v,[(4,4),(2,2),(1,1)]).shape,(1,count,256))

    def test_state_roundtrip_and_parameters_equal(self):
        p=from_baseline(self.base,'P');b1=from_baseline(self.base,'B1')
        count=sum(x.numel() for x in self.base.parameters())
        for module in (p,b1):
            self.assertEqual(sum(x.numel() for x in module.parameters()),count)
            self.assertEqual(list(module.state_dict()),list(self.base.state_dict()))
        for k,v in p.state_dict().items():self.assertTrue(torch.equal(v,self.base.state_dict()[k]))
        buf=io.BytesIO();torch.save(p.state_dict(),buf);buf.seek(0)
        restored=FinestSpanAttention(32,4)
        restored.load_state_dict(torch.load(buf,weights_only=True));bind_common_core(restored)
        self.assertTrue(torch.equal(p(self.q,self.ref,self.v,self.shapes),
                                    restored(self.q,self.ref,self.v,self.shapes)))

    def test_native_conventional_control_and_rng(self):
        rng=torch.get_rng_state().clone();b1=from_baseline(self.base,'B1')
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(b1.num_points_list,[8,2,2]);self.assertEqual(sum(b1.num_points_list),12)
        differs={k for k,v in b1.state_dict().items() if not torch.equal(v,self.base.state_dict()[k])}
        self.assertEqual(differs,{'sampling_offsets.bias','num_points_scale'})
        self.assertTrue(torch.isfinite(b1(self.q,self.ref,self.v,self.shapes)).all())

    def test_synthetic_cell_support_example(self):
        # Native align_corners=False cell coordinate is normalized*W-.5.
        # A +x head with four radial offsets and offset_scale=.5.
        native=torch.tensor([1.,2.,3.,4.])/4*.5*.125 + 3.25
        floor=torch.tensor([1.,2.,3.,4.])/4*.5 + 3.25
        # Positive direction alone stays in cell 3. Negative direction crosses.
        native_neg=3.25-(native-3.25);floor_neg=3.25-(floor-3.25)
        self.assertEqual(torch.cat((native,native_neg)).floor().unique().numel(),1)
        self.assertEqual(torch.cat((floor,floor_neg)).floor().unique().numel(),2)

    def test_reject_invalid_inputs(self):
        for index,value in ((2,-1.),(3,0.),(0,float('nan')),(2,float('inf'))):
            ref=self.ref.clone();ref[...,index]=value
            with self.assertRaises(ValueError):sampling_references(ref,self.shapes)
        for shapes in ([(8,12)],[(8,12),(0,6),(2,3)],[(2,3),(4,6),(8,12)]):
            with self.assertRaises(ValueError):sampling_references(self.ref,shapes)
        with self.assertRaises(ValueError):sampling_references(self.ref[...,:2],self.shapes)
        with self.assertRaises(ValueError):from_baseline(self.base,'unknown')

    def test_full_common_model_construction_state_diff_and_old_guard(self):
        baseline,_,_=build(Config(input_hw=(128,128),physical_batch_size=1,amp_dtype='float32'),synthetic=True)
        p=adapt_cpu_model(baseline,'P');b1=adapt_cpu_model(baseline,'B1')
        self.assertEqual(sum(x.numel() for x in p.parameters()),20094584)
        self.assertEqual(sum(x.numel() for x in b1.parameters()),20094584)
        a=baseline.state_dict()
        self.assertTrue(all(torch.equal(v,a[k]) for k,v in p.state_dict().items()))
        differs={k for k,v in b1.state_dict().items() if not torch.equal(v,a[k])}
        expected={f'decoder.decoder.layers.{i}.cross_attn.{key}' for i in range(3)
                  for key in ('sampling_offsets.bias','num_points_scale')}
        self.assertEqual(differs,expected)
        # Do NOT loosen production identity guard to make this prototype run.
        with self.assertRaises(vendor.V2BConfigurationError):
            vendor.validate_model_sampling(p,sampling_backend='deterministic_gather')


if __name__=='__main__':
    unittest.main()
