"""Synthetic CPU only. Exercises the actual vendor DN function through the research adapter."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, types, importlib, json, unittest, time, resource
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import torch
torch.set_num_threads(2)
from sparse_rtdetr.dn_block_replay import *
pkg=types.ModuleType("dn_vendor")
pkg.__path__=[str(ROOT/"vendor/rtdetrv2_pytorch/src/zoo/rtdetr")]
sys.modules["dn_vendor"]=pkg
native=importlib.import_module("dn_vendor.denoising").get_contrastive_denoising_training_group

class Checks(unittest.TestCase):
    def test_independent_geometry(self):
        gt=torch.tensor([[.4,.6,.2,.1],[.6,.4,.1,.3]],dtype=torch.float64)
        pred=torch.tensor([[.42,.58,.3,.05],[.55,.46,.05,.6]],dtype=torch.float64)
        expected=torch.tensor([[.1,-.2,1.5,.5],[-.5,.2,.5,2.]],dtype=torch.float64)
        e=encode(pred,gt)
        torch.testing.assert_close(e[:,:2],expected[:,:2],rtol=0,atol=1e-14)
        torch.testing.assert_close(e[:,2:].exp(),expected[:,2:],rtol=0,atol=1e-14)
        torch.testing.assert_close(reconstruct(gt,e),pred,rtol=0,atol=1e-14)
        self.assertFalse(e.requires_grad)
    def test_crossblock(self):
        e=torch.arange(24,dtype=torch.float64).reshape(6,4)/100
        g=torch.tensor([0,0,1,1,2,2])
        state=torch.get_rng_state().clone()
        j,b,d,p=select(e,g,g.repeat(5),77)
        self.assertTrue(torch.equal(torch.get_rng_state(),state))
        self.assertTrue(torch.equal(j[:,:2],b[:,:2]))
        self.assertTrue(torch.equal(j[p,2:],b[:,2:]))
        self.assertTrue(torch.equal(g.repeat(5),g.repeat(5)[p]))
        self.assertTrue(torch.equal(j,e[d]))
    def test_same_seed_and_singleton(self):
        e=torch.zeros(3,4);g=torch.arange(3)
        j,b,d,p=select(e,g,g,5)
        self.assertTrue(torch.equal(j,b))
        self.assertTrue(torch.equal(p,torch.arange(3)))
        self.assertTrue(torch.equal(select(e,g,g,5)[2],d))
    def test_interior_control(self):
        gt=torch.tensor([[.5,.5,.05,.08]]*4)
        j=torch.tensor([[.1,.1,.1,.1],[-.1,-.1,-.1,-.1],[.2,-.2,.2,-.2],[-.2,.2,-.2,.2]])
        p=torch.tensor([1,2,3,0]);b=j.clone();b[:,2:]=j[p,2:]
        a=control_audit(gt,j,b,p)
        self.assertTrue(a["joint_only_admissible"],a)
    def test_clipping_counterexample(self):
        gt=torch.tensor([[.1,.5,.2,.2],[.1,.5,.2,.2]],dtype=torch.float64)
        j=torch.tensor([[-.3,0,.6,0],[.3,0,-.6,0]],dtype=torch.float64)
        p=torch.tensor([1,0]);b=j.clone();b[:,2:]=j[p,2:]
        a=control_audit(gt,j,b,p)
        self.assertTrue(a["pre_size_block_equal"])
        self.assertFalse(a["joint_only_admissible"])
        print("EXPECTED_COUNTEREXAMPLE",json.dumps(a))
    def test_invalid_is_not_repaired(self):
        gt=torch.tensor([[.5,.5,.1,.1]])
        e=torch.tensor([[20.,0,0,0]])
        self.assertTrue(bool(effective(gt,e)[5][0]))
        with self.assertRaises(ValueError): encode(torch.zeros(1,4),gt)
        with self.assertRaises(ValueError): reconstruct(gt,torch.full((1,4),float("nan")))
    def test_inverse_sigmoid_not_identity(self):
        gt=torch.tensor([[.000001,.5,.000001,.1]],dtype=torch.float64)
        v=effective(gt,torch.zeros_like(gt))
        self.assertGreater(float((v[2]-v[1]).abs().max()),1e-6)
    def test_zero_iou_kept(self):
        gt=torch.tensor([[.2,.2,.1,.1]])
        pred=torch.tensor([[.8,.8,.1,.1]])
        self.assertTrue(torch.isfinite(encode(pred,gt)).all())
        torch.testing.assert_close(reconstruct(gt,encode(pred,gt)),pred)
    def test_semantics_rejects_hard_positive_not_rebalanced(self):
        gt=torch.tensor([[.2,.2,.1,.1]])
        far=torch.tensor([[.8,.8,.1,.1]])
        a=semantic_audit(gt,gt,far,far)
        self.assertFalse(a["admissible"])
        self.assertEqual(a["replay_positive_zero_iou"],1.)
    def test_bad_library(self):
        with self.assertRaises(ValueError): select(torch.zeros(1,4),torch.tensor([0]),torch.tensor([1]),1)
        with self.assertRaises(ValueError): select(torch.zeros(1,4),torch.tensor([4]),torch.tensor([0]),1)
    def setup_entry(self):
        targets=[{"labels":torch.tensor([0,1,2,1]),"boxes":torch.tensor([[.4,.4,.03,.02],[.6,.6,.03,.02],[.4,.6,.04,.04],[.6,.4,.04,.04]])},
                 {"labels":torch.tensor([1]),"boxes":torch.tensor([[.5,.5,.03,.02]])}]
        with torch.random.fork_rng(): torch.manual_seed(4);emb=torch.nn.Embedding(4,8)
        e=torch.tensor([[.05,.03,.04,.02],[-.05,-.03,-.04,-.02],[.08,-.01,.02,-.01],[-.08,.01,-.02,.01]])
        groups=torch.tensor([0,0,1,1])
        return targets,emb,e,groups
    def run_entry(self,arm):
        t,emb,e,g=self.setup_entry()
        torch.manual_seed(19)
        ctx=None if arm is None else Replay(e,g,arm,17)
        out=dn_entry(native,t,3,300,emb,num_denoising=8,replay=ctx)
        return out,torch.get_rng_state(),ctx
    def test_dense_group_and_empty_row(self):
        _,emb,e,g=self.setup_entry()
        targets=[{"labels":torch.zeros(101,dtype=torch.long),
                  "boxes":torch.tensor([[.5,.5,.03,.02]]).repeat(101,1)},
                 {"labels":torch.zeros(0,dtype=torch.long),"boxes":torch.zeros(0,4)}]
        ctx=Replay(e,g,"B",9)
        out=dn_entry(native,targets,3,300,emb,100,replay=ctx)
        self.assertEqual(out[3]["dn_num_group"],1)
        self.assertEqual(ctx.audit["replaced_positives"],51)
        self.assertEqual(ctx.audit["padding_slots"],202)
    def test_actual_native_bypass(self):
        t,emb,e,g=self.setup_entry()
        torch.manual_seed(19);expected=native(t,3,300,emb,8,.5,1.)
        actual,_,_=self.run_entry(None)
        for i in range(3): self.assertTrue(torch.equal(expected[i],actual[i]))
    def test_actual_positive_negative_padding_mask_rng(self):
        n,rng,_=self.run_entry(None)
        j,jrng,jctx=self.run_entry("J");b,brng,bctx=self.run_entry("B")
        self.assertTrue(torch.equal(rng,jrng));self.assertTrue(torch.equal(rng,brng))
        keep=torch.ones(n[1].shape[:2],dtype=torch.bool)
        for i,ids in enumerate(n[3]["dn_positive_idx"]): keep[i,ids]=False
        valid=(~keep).nonzero()
        untouched=valid[1::2]
        for out in (j,b):
            self.assertTrue(torch.equal(n[1][untouched[:,0],untouched[:,1]],out[1][untouched[:,0],untouched[:,1]]))
            self.assertTrue(torch.equal(n[0],out[0]))
            self.assertTrue(torch.equal(n[2],out[2]))
            self.assertTrue(torch.equal(n[1][keep],out[1][keep]))
            self.assertEqual(n[3]["dn_num_group"],out[3]["dn_num_group"])
            self.assertEqual(n[3]["dn_num_split"],out[3]["dn_num_split"])
            for a,z in zip(n[3]["dn_positive_idx"],out[3]["dn_positive_idx"]):
                self.assertTrue(torch.equal(a,z))
        self.assertEqual(jctx.audit["donors"],bctx.audit["donors"])
        self.assertTrue(jctx.audit["joint_only_admissible"])
        self.assertTrue(jctx.audit["positive_negative"]["admissible"])
        self.assertEqual(jctx.audit["positive_negative"]["n"],5)
    def test_empty_and_disabled(self):
        _,emb,e,g=self.setup_entry()
        t=[{"boxes":torch.zeros(0,4),"labels":torch.zeros(0,dtype=torch.long)}]
        self.assertEqual(dn_entry(native,t,3,300,emb,replay=Replay(e,g,"J",0)),(None,)*4)
        with self.assertRaises(ValueError):
            dn_entry(native,t,3,300,emb,box_noise_scale=0,replay=Replay(e,g,"J",0))
    def test_actual_entry_rejects_clipped_control(self):
        t,emb,e,g=self.setup_entry()
        # Shared fixed errors shift every box outside the image: rejects, no resampling.
        bad=torch.tensor([[100.,0,0,0],[101.,0,0,0]])
        ctx=Replay(bad,torch.tensor([0,1]),"J",1)
        with self.assertRaises(ValueError): dn_entry(native,t,3,300,emb,8,replay=ctx)
        self.assertGreater(ctx.audit["invalid_J"],0)

if __name__=="__main__":
    start=time.monotonic()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    receipt={"synthetic_only":True,"tests":result.testsRun,"success":result.wasSuccessful(),
             "seconds":time.monotonic()-start,"max_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             "cuda_initialized":torch.cuda.is_initialized(),"detector_forwards":0,"optimizer_updates":0}
    print(json.dumps(receipt,sort_keys=True))
    if receipt["seconds"]>300 or receipt["max_rss_kib"]>4*1024*1024 or receipt["cuda_initialized"]:
        raise SystemExit("CPU budget exceeded")
    raise SystemExit(0 if result.wasSuccessful() else 1)
