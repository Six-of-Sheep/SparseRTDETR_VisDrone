"""Fixed CPU rules; no scientific evaluation or detector inference."""
import copy
import unittest
import numpy as np
import torch
from PIL import Image

from sparse_rtdetr.localization.data import RawAugmentation, parse_raw
from sparse_rtdetr.protocol_r18 import slice_finetune as ft
from sparse_rtdetr.protocol_r18.checkpoint import exact_equal
from sparse_rtdetr.protocol_r18.engine import Engine, learning_rate as original_lr


def fixture():
    im=Image.fromarray(np.arange(100*100*3,dtype=np.uint8).reshape(100,100,3))
    raw=parse_raw(b'10,10,10,10,1,1,0,0\n50,10,20,20,1,2,0,0\n80,80,10,10,1,3,0,0\n52,0,20,30,0,0,0,0\n52,10,20,10,0,4,0,0\n', 'fixture')
    return im,raw


class FixedRules(unittest.TestCase):
    def test_balanced_views(self):
        for start in range(0,4864,16):
            views=[ft.view_id('T-slice',i) for i in range(start,start+16)]
            self.assertEqual([views.count(i) for i in range(5)],[8,2,2,2,2])
            for micro in (views[:8],views[8:]):self.assertEqual(micro.count(0),4)

    def test_control_always_whole(self):
        self.assertEqual({ft.view_id('T-control',i) for i in range(4864)},{0})

    def test_budget_clock(self):
        self.assertEqual([ft.clock(i) for i in (1,9,10,12)],[109,117,118,120])
        self.assertEqual(ft.PASSES*304,ft.UPDATES)
        self.assertEqual(ft.UPDATES*16,58368)

    def test_bad_clocks(self):
        for f,args in [(ft.clock,(0,)),(ft.clock,(13,)),(ft.view_id,('bad',0)),(ft.view_id,('T-slice',4864)),(ft.learning_rate,(0,)),(ft.learning_rate,(3649,))]:
            with self.assertRaises(ValueError):f(*args)

    def test_schedule(self):
        self.assertEqual(ft.learning_rate(200),1e-5)
        self.assertEqual(ft.learning_rate(3648),1e-6)
        values=[ft.learning_rate(i) for i in range(1,3649)]
        self.assertTrue(all(a<=b for a,b in zip(values[:199],values[1:200])))
        self.assertTrue(all(a>=b for a,b in zip(values[199:],values[200:])))

    def test_native_schedule_hook_unchanged(self):
        class Cursor:pass
        c=Cursor()
        for u in (0,1,1999,2000,36479):
            c.completed_updates=u
            self.assertEqual(Engine.next_learning_rate(c),original_lr(u+1))

    def test_clip_intersection_not_center_gate(self):
        im,r=fixture();_,t,sel=ft.crop_raw(im,r,1)
        self.assertEqual(sel['crop_box'],[0,0,56,56])
        self.assertEqual(len(t['gt']),2)
        self.assertEqual(t['gt'][1]['box'],[50.,10.,56.,30.])
        self.assertEqual(t['gt'][1]['raw_area'],400)
        self.assertEqual(t['gt'][1]['id'],r['gt'][1]['id'])

    def test_no_mutation(self):
        im,r=fixture();old=copy.deepcopy(r);ft.crop_raw(im,r,1)
        self.assertEqual(r,old)

    def test_ignore_geometry(self):
        im,r=fixture();_,t,_=ft.crop_raw(im,r,1)
        self.assertEqual(t['spatial_ignore'][0]['box'],[52.,0.,56.,30.])
        self.assertEqual(t['class_ignore'][0]['box'],[52.,10.,56.,20.])
        self.assertEqual(t['class_ignore'][0]['category'],4)

    def test_empty_view_no_resample(self):
        im,r=fixture();r['gt']=[]
        x,t,tr=ft.Transform('T-slice',(128,128))(im,r,epoch=12,position=1,image_id='fixture')
        self.assertEqual(t['boxes'].shape,(0,4));self.assertEqual(tr['fixed_view']['view_id'],1)
        self.assertTrue(torch.isfinite(x).all())

    def test_control_pixels_targets_native_exact(self):
        for e in (1,9,10,12):
            im,r=fixture()
            old=RawAugmentation((128,128))(im,r,epoch=ft.clock(e),seed=0,position=0,image_id='fixture')
            new=ft.Transform('T-control',(128,128))(im,r,epoch=e,position=0,image_id='fixture')
            self.assertTrue(torch.equal(old[0],new[0]));self.assertTrue(exact_equal(old[1],new[1]))
            self.assertEqual(old[2],{k:new[2][k] for k in old[2]})

    def test_raw_identity_and_positive_geometry(self):
        im,r=fixture()
        for p in (1,3,5,7):
            _,t,tr=ft.Transform('T-slice',(128,128))(im,r,epoch=12,position=p,image_id='fixture')
            self.assertTrue(set(t['gt_ids'])<=set(v['id'] for v in r['gt']))
            self.assertTrue((t['boxes'][:,2:]>0).all())
            self.assertTrue(torch.isfinite(t['ignore_union']).all())
            self.assertFalse(tr['augmentation_strong'])

    def test_sample_rng_isolation(self):
        im,r=fixture();before=ft.native_state.capture_rng(torch.device('cpu'))
        ft.Transform('T-slice',(128,128))(im,r,epoch=1,position=1,image_id='fixture')
        self.assertTrue(exact_equal(before,ft.native_state.capture_rng(torch.device('cpu'))))

    def test_all_outside_and_subpixel_sanitize(self):
        im,r=fixture();r['gt']=[dict(r['gt'][0],box=[55.9,10,56.1,11])]
        _,t,_=ft.Transform('T-slice',(128,128))(im,r,epoch=12,position=1,image_id='fixture')
        self.assertEqual(len(t['boxes']),0)  # original min_size1 before resize

    def test_tile_selection_has_no_gt_input(self):
        im,r=fixture();sel1=ft.crop_raw(im,r,2)[2];r['gt']=[]
        self.assertEqual(sel1,ft.crop_raw(im,r,2)[2])


if __name__=='__main__':unittest.main()
