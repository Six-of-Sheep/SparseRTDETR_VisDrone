"""Full synthetic CPU acceptance; never read detection data or old states."""
import copy
import ast
import gc
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from sparse_rtdetr.protocol_r18.model import Config, build as common_build, CommonEMA, dn_capture, fp32_outputs
from sparse_rtdetr.protocol_r18.engine import tensor_digest, output_digest
from sparse_rtdetr.baseline.training_v2b import _optimizer
from sparse_rtdetr.research import local_detail_model as h2
from sparse_rtdetr.research.local_detail_engine import Engine, assert_healthy
from sparse_rtdetr.research import local_detail_checkpoint as cp

torch.set_num_threads(2)


def config(dtype='float32'):
    return Config(input_hw=(128,160), physical_batch_size=1, amp_dtype=dtype)


def targets():
    result=[]
    for i, n in enumerate((2,0)):
        result.append(dict(boxes=torch.tensor([[.4,.4,.15,.2],[.7,.7,.1,.12]])[:n],
            labels=torch.tensor([1,4],dtype=torch.int64)[:n],gt_ids=[f'constructed/{i}/{j}' for j in range(n)],
            ignore_union=torch.empty(0,4,dtype=torch.float64),ignore_class_boxes=torch.empty(0,4,dtype=torch.float64),
            ignore_class_labels=torch.empty(0,dtype=torch.int64)))
    return result


def engine(arm, dtype='float32'):
    conf=config(dtype)
    model, criterion, info=h2.build(conf,arm,synthetic=True)
    e=Engine(model,criterion,_optimizer(model,conf),CommonEMA(model),conf)
    binding=dict(arm=arm,scope='cpu_H2_synthetic',config=conf.binding(),model=info,source_sha256='a'*64)
    return e,binding


class FullModelTests(unittest.TestCase):
    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())
        gc.collect()

    def test_initialization_inventory_and_shared_tensor_rows(self):
        before=torch.get_rng_state()
        n,_,ni=h2.build(config(),'N',synthetic=True)
        u,uc,ui=h2.build(config(),'U',synthetic=True)
        d,dc,di=h2.build(config(),'D',synthetic=True)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertEqual([ni['trainable_parameters'],ui['trainable_parameters'],di['trainable_parameters']],
                         [20094584,20079160,20078646])
        self.assertEqual(uc.recipe(),dc.recipe())
        self.assertEqual(ui['config'],di['config']); self.assertEqual(ui['effective_config'],di['effective_config'])
        ns,us,ds=n.state_dict(),u.state_dict(),d.state_dict()
        self.assertEqual(set(us),set(ds))
        for key in us:
            expected=us[key][2:] if key.startswith('decoder.dec_bbox_head.2.layers.2.') else us[key]
            self.assertTrue(torch.equal(expected,ds[key]),key)
            if 'detail_readout' in key:
                self.assertEqual(int(us[key].count_nonzero()),0)
            elif 'decoder.decoder.layers.2.cross_attn.sampling_offsets.' in key:
                rows=torch.arange(192).reshape(8,12,2)[:,4:].flatten()
                self.assertTrue(torch.equal(us[key],ns[key][rows]),key)
            else:
                self.assertTrue(torch.equal(us[key],ns[key]),key)
        self.assertEqual(sum(us[k].numel()-ds[k].numel() for k in us),514)

    def test_N_exact_native_training_outputs_DN_and_gradients(self):
        n,c,_=h2.build(config(),'N',synthetic=True)
        native,nc,_=common_build(config(),synthetic=True)
        self.assertEqual(tensor_digest(n.state_dict().items()),tensor_digest(native.state_dict().items()))
        images=torch.rand(2,3,128,160); t=targets(); rng=torch.get_rng_state()
        with dn_capture(n) as noise:a=n(images,t)
        torch.set_rng_state(rng)
        with dn_capture(native) as noise2:b=native(images,t)
        self.assertEqual(output_digest(a),output_digest(b));self.assertEqual(output_digest(noise),output_digest(noise2))
        la=c(fp32_outputs(a),t); lb=nc(fp32_outputs(b),t)
        self.assertEqual(len(la),21);self.assertTrue(all(torch.equal(la[k],lb[k]) for k in la))
        sum(la.values()).backward();sum(lb.values()).backward()
        self.assertEqual(tensor_digest((k,p.grad) for k,p in n.named_parameters()),
                         tensor_digest((k,p.grad) for k,p in native.named_parameters()))

    def test_shared_sampler_exact_context_and_single_fine_call(self):
        value=torch.randn(2,27,8,32,requires_grad=True)
        shapes=[(3,5),(2,4),(2,2)]
        locations=torch.rand(2,7,8,12,2)*1.4-.2
        weights=torch.softmax(torch.randn(2,7,8,12),-1)
        with patch.object(h2,'bilinear_grid_sample',wraps=h2.bilinear_grid_sample) as sampler:
            output,fine=h2.shared_sample_core(value,shapes,locations,weights)
        self.assertEqual(sampler.call_count,3)
        expected=h2.deformable_attention_core_func_v2(value,shapes,locations,weights,[4,4,4])
        self.assertTrue(torch.equal(output,expected))
        self.assertEqual(fine.shape,(2,7,4,256))
        fine.square().sum().backward();self.assertGreater(float(value.grad.abs().sum()),0)

    def test_UD_same_inputs_supervision_DN_aux_initial_outputs(self):
        u,uc,_=h2.build(config(),'U',synthetic=True);d,dc,_=h2.build(config(),'D',synthetic=True)
        images=torch.rand(2,3,128,160);t=targets();rng=torch.get_rng_state()
        with dn_capture(u) as nu:a=u(images,t)
        torch.set_rng_state(rng)
        with dn_capture(d) as nd:b=d(images,t)
        self.assertEqual(output_digest(nu),output_digest(nd));self.assertEqual(output_digest(a),output_digest(b))
        self.assertEqual(a['pred_boxes'].shape,(2,300,4))
        self.assertEqual(len(a['aux_outputs']),2);self.assertEqual(len(a['dn_aux_outputs']),3)
        self.assertGreater(a['dn_meta']['dn_num_split'][0],0)
        la,lb=uc(fp32_outputs(a),t),dc(fp32_outputs(b),t)
        self.assertEqual(len(la),21); self.assertTrue(all(torch.equal(la[k],lb[k]) for k in la))
        sum(lb.values()).backward()
        self.assertGreater(float(d.decoder.decoder.layers[2].cross_attn.detail_readout.weight.grad.abs().sum()),0)
        self.assertIsNone(d.decoder.decoder.layers[2].cross_attn._pending_detail)
        self.assertTrue(all(p.grad is None or p.grad.isfinite().all() for p in d.parameters()))

    def test_nonzero_readout_formula_covers_actual_ordinary_and_DN_queries(self):
        from sparse_rtdetr.research.local_detail_center import inverse_sigmoid
        for arm in ('U','D'):
            m,_,_=h2.build(config(),arm,synthetic=True)
            att=m.decoder.decoder.layers[2].cross_attn
            with torch.no_grad():
                att.detail_readout.weight.normal_(std=.01)
                m.decoder.dec_bbox_head[2].layers[-1].bias.fill_(.125)
            refs=[];details=[];contexts=[]
            handles=[att.register_forward_pre_hook(lambda mod,args:refs.append(args[1])),
                att.detail_readout.register_forward_hook(lambda mod,args,out:details.append(out)),
                m.decoder.dec_bbox_head[2].register_forward_hook(lambda mod,args,out:contexts.append(out))]
            try:out=m(torch.rand(2,3,128,160),targets())
            finally:
                for handle in handles:handle.remove()
            center=details[0]+contexts[0][...,:2] if arm=='U' else details[0]
            expected=torch.sigmoid(torch.cat((center,contexts[0][...,-2:]),-1)+inverse_sigmoid(refs[0][:,:,0,:]))
            dn=out['dn_meta']['dn_num_split'][0]
            self.assertGreater(dn,0);self.assertGreater(float(details[0].abs().sum()),0)
            self.assertTrue(torch.equal(out['pred_boxes'],expected[:,dn:]))
            self.assertTrue(torch.equal(out['dn_aux_outputs'][-1]['pred_boxes'],expected[:,:dn]))
            with self.assertRaises(ValueError):
                m.decoder.label_noise_ratio=.1;h2.validate_model(m,arm)

    def test_detached_sampling_but_native_aux_reference_gradient(self):
        d,_,_=h2.build(config(),'D',synthetic=True);seen=[];middle=[]
        def reference(m,args):seen.append(args[1])
        def head(m,args,result):result.retain_grad();middle.append(result)
        a=d.decoder.decoder.layers[2].cross_attn.register_forward_pre_hook(reference)
        b=d.decoder.dec_bbox_head[1].register_forward_hook(head)
        try:
            out=d(torch.rand(2,3,128,160),targets())
            out['pred_boxes'][...,:2].sum().backward()
        finally:a.remove();b.remove()
        self.assertFalse(seen[0].requires_grad)
        self.assertGreater(float(middle[0].grad[...,:2].abs().sum()),0)

    def test_full_attention_coordinates_no_floor_and_mask(self):
        u,_,_=h2.build(config(),'U',synthetic=True);a=u.decoder.decoder.layers[2].cross_attn
        refs=torch.tensor([[[[.01,.98,.001,.8]],[[.5,.5,.6,.002]]]])
        q=torch.randn(1,2,256);value=torch.randn(1,27,256);shapes=[(3,5),(2,4),(2,2)]
        real=h2.shared_sample_core; observed=[]
        def inspect(v,s,l,w):observed.append((v,l));return real(v,s,l,w)
        with patch.object(h2,'shared_sample_core',side_effect=inspect):
            a(q,refs,value,shapes,torch.zeros(1,27,dtype=torch.bool))
        projected,loc=observed[0]
        expected=(refs[...,:2]+refs[...,2:]*refs.new_tensor([[.25,0],[-.25,0],[0,.25],[0,-.25]])).unsqueeze(2).expand(-1,-1,8,-1,-1)
        self.assertTrue(torch.equal(loc[...,:4,:],expected));self.assertEqual(float(projected.abs().sum()),0)
        a._pending_detail=None
        with self.assertRaises(ValueError):a(q,refs.requires_grad_(),value,shapes)

    def test_zero_difference_and_U_direct_center_counterexample(self):
        for arm in ('U','D'):
            m,_,_=h2.build(config(),arm,synthetic=True);m.eval()
            refs=[]
            handle=m.decoder.decoder.layers[2].cross_attn.register_forward_pre_hook(lambda mod,args:refs.append(args[1]))
            with torch.no_grad():
                m.decoder.dec_bbox_head[2].layers[-1].bias.fill_(.5)
                att=m.decoder.decoder.layers[2].cross_attn
                att.value_proj.weight.zero_();att.value_proj.bias.zero_()
                result=m(torch.rand(1,3,128,160))
                self.assertTrue(result['pred_boxes'].isfinite().all())
                from sparse_rtdetr.research.local_detail_center import inverse_sigmoid
                delta=refs[0].new_tensor([.5,.5,.5,.5] if arm=='U' else [0,0,.5,.5])
                expected=torch.sigmoid(inverse_sigmoid(refs[0][:,:,0,:])+delta)
                self.assertTrue(torch.equal(result['pred_boxes'],expected))
            handle.remove()
            # All four sampled vectors are zero; final D center delta is identically zero.
            self.assertIsNone(att._pending_detail)
            self.assertEqual(att.detail_readout.bias,None)

    def test_full_state_save_replay_all_arms_both_CPU_dtypes(self):
        for dtype in ('float32','bfloat16'):
            for arm in h2.ARMS:
                with self.subTest(dtype=dtype,arm=arm),tempfile.TemporaryDirectory(prefix='H2-CPU-') as tmp:
                    e,b=engine(arm,dtype);torch.manual_seed(42)
                    images=torch.rand(2,3,128,160); t=targets()
                    first=e.train_window(images,t,capture=True)
                    receipt=cp.save(e,b,Path(tmp)/'C1.pth')
                    direct=e.train_window(images,t,capture=True);expected=cp.capture(e,b)
                    clone,b2=engine(arm,dtype);self.assertEqual(b,b2)
                    restored=cp.restore(clone,b,receipt['path'],receipt['sha256'])
                    replay=clone.train_window(images,t,capture=True)
                    self.assertTrue(restored['full_state_exact'])
                    self.assertEqual({k:v for k,v in direct.items() if k!='elapsed_seconds'},
                                     {k:v for k,v in replay.items() if k!='elapsed_seconds'})
                    self.assertTrue(cp.exact_equal(expected,cp.capture(clone,b)))
                    self.assertEqual(clone.completed_updates,clone.ema.updates)
                    self.assertEqual(first['coefficients'],[.5,.5])
                    self.assertTrue(all(len(x)==(21 if i==0 else 12) for i,x in enumerate(first['microbatch_losses'])))
                    with self.assertRaises(FileExistsError):cp.save(clone,b,receipt['path'])
                    del e,clone,expected;gc.collect()

    def test_identity_cache_nonfinite_scope_and_SHA_rejections(self):
        e,b=engine('D')
        with tempfile.TemporaryDirectory(prefix='H2-CPU-reject-') as tmp:
            receipt=cp.save(e,b,Path(tmp)/'E000.pth')
            wrong=copy.deepcopy(b);wrong['arm']='U'
            with self.assertRaises(ValueError):cp.read(receipt['path'],receipt['sha256'],wrong)
            with self.assertRaises(ValueError):cp.read(receipt['path'],'0'*64,b)
            wrong=copy.deepcopy(b);wrong['scope']='span_formal_e120'
            with self.assertRaises(ValueError):cp.capture(e,wrong)
            bad,_=engine('U')
            with self.assertRaises(ValueError):cp.restore(bad,b,receipt['path'],receipt['sha256'])
            self.assertTrue(bad.failed)
        e.model.decoder.anchors[0,0,0]=float('inf')
        with self.assertRaises(ValueError):assert_healthy(e.model)

    def test_native_update_math_unchanged_and_legacy_guards_stay_strict(self):
        import inspect
        from sparse_rtdetr.protocol_r18.engine import Engine as NativeEngine
        from sparse_rtdetr.protocol_r18.geometry import validate_geometry
        import textwrap
        def tree(method):return ast.dump(ast.parse(textwrap.dedent(inspect.getsource(method))),include_attributes=False)
        self.assertEqual(tree(Engine.train_window),tree(NativeEngine.train_window))
        for arm in ('U','D'):
            e,b=engine(arm)
            self.assertEqual(e.completed_updates,0);self.assertEqual(e.ema.updates,0)
            self.assertEqual(len(e.optimizer.state),0)
            self.assertEqual(tensor_digest(e.model.state_dict().items()),tensor_digest(e.ema.module.state_dict().items()))
            with self.assertRaises(ValueError):validate_geometry(e.model)
            self.assertIsNone(e.model.decoder.decoder.layers[2].cross_attn._pending_detail)

    def test_production_canvas_cache_and_invalid_anchor_exception_is_narrow(self):
        conf=Config();m,c,info=h2.build(conf,'D',synthetic=True)
        e=Engine(m,c,_optimizer(m,conf),CommonEMA(m),conf)
        b=dict(arm='D',scope='cpu_H2_synthetic',config=conf.binding(),model=info)
        payload=cp.capture(e,b)
        self.assertGreater(int((~m.decoder.valid_mask).sum()),0)
        self.assertTrue(cp.exact_equal(payload['model'],payload['ema']['module']))
        p=next(m.parameters())
        with torch.no_grad():p.flatten()[0]=float('inf')
        with self.assertRaises(RuntimeError):assert_healthy(m)

    def test_historical_N_import_is_explicit_eval_only_and_not_a_resume(self):
        from sparse_rtdetr.research.local_detail_reference import load_N_for_cost
        import hashlib
        conf=Config();m,_,_=h2.build(conf,'N',synthetic=True)
        binding=dict(arm='r18-sampling-span-v1/B0',scope='span_formal_e120',config=conf.binding(),
                     pretrained_sha256='911a745b62e173c8f4b9af513c2ea295428cf23f1bbfe9048381500f140fd720')
        fixture=dict(format='sparse_rtdetr.sampling_span_fullstate_v1',binding=binding,
            binding_sha256=hashlib.sha256(cp.canonical(binding)).hexdigest(),
            engine=dict(completed_updates=36480),ema=dict(updates=36480,module=m.state_dict()))
        with tempfile.TemporaryDirectory(prefix='H2-constructed-old-format-') as tmp:
            path=Path(tmp)/'CONSTRUCTED_NOT_REAL_B0.pth';torch.save(fixture,path)
            sha=cp.file_sha(path)
            with self.assertRaises(ValueError):load_N_for_cost(m,path,expected_sha=sha,expected_binding=binding)
            m.eval()
            with self.assertRaises(ValueError):load_N_for_cost(m,path,expected_sha='0'*64,expected_binding=binding)
            receipt=load_N_for_cost(m,path,expected_sha=sha,expected_binding=binding)
            self.assertFalse(receipt['old_optimizer_or_RNG_restored'])
            wrong=copy.deepcopy(binding);wrong['scope']='span_GPU_acceptance'
            with self.assertRaises(ValueError):load_N_for_cost(m,path,expected_sha=sha,expected_binding=wrong)


if __name__=='__main__':unittest.main()
