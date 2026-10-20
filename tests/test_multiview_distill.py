"""Small deterministic production-interface checks; never evaluates real AP."""
import copy
import json
import io
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

from slice_core import Candidate, tile_windows
from sparse_rtdetr.localization.data import RawAugmentation, parse_raw
from sparse_rtdetr.protocol_r18 import distill as kd
from sparse_rtdetr.protocol_r18.checkpoint import exact_equal, capture_rng
from sparse_rtdetr.protocol_r18.model import Config, build


def fixture():
    image = Image.fromarray(np.arange(101*157*3, dtype=np.uint8).reshape(101,157,3))
    raw = parse_raw(b'10,10,20,20,1,1,0,0\n80,40,15,30,1,2,0,0\n120,0,37,40,0,0,0,0\n50,50,10,10,0,4,0,0\n', 'CPU_FIXTURE')
    return image, raw


def candidate(box=(10.,10.,30.,30.), score=.75, view=0, query=0, category=1):
    return Candidate(box,score,view,query,category,1)


def empty_teacher():
    return dict(boxes=torch.empty(0,4), labels=torch.empty(0,dtype=torch.long),
                scores=torch.empty(0), identities=[])


class DataChecks(unittest.TestCase):
    def test_teacher_view_geometry(self):
        for h,w in ((100,156),(101,157),(1,1)):
            tiles=tile_windows(h,w)
            self.assertEqual(tiles[0].height, min(h,(5*h+8)//9))
            self.assertEqual(tiles[3].crop_box, (w-tiles[3].width,h-tiles[3].height,w,h))

    def test_raw_cache_pipeline_and_binding(self):
        row=dict(stable_image_id='CPU_ONLY',height=101,width=157,image_sha256='a'*64,annotation_sha256='b'*64)
        binding=kd.cache_binding(source_manifest_sha256='c'*64,contract_sha256='d'*64)
        views={i:dict(pred_logits=torch.linspace(-8,4,3000).reshape(300,10),
                      pred_boxes=torch.tensor([[.5,.5,.1,.1]]*300)) for i in range(5)}
        record=kd.from_raw_views(row,views,binding)
        self.assertEqual(len(record['views']),5)
        self.assertEqual(kd.select_teacher(record,'T0'),[])
        self.assertTrue(all(c.view_id==0 for c in kd.select_teacher(record,'T1')))
        self.assertLessEqual(len(kd.select_teacher(record,'T2')),300)
        for field in ('checkpoint_sha256','manifest_sha256','source_manifest_sha256','seed','role'):
            bad=copy.deepcopy(record);bad['binding'][field]='wrong'
            with self.assertRaises(ValueError):kd.verify_record(bad,row,binding)
        bad=copy.deepcopy(record);bad['member']['annotation_sha256']='e'*64
        with self.assertRaises(ValueError):kd.verify_record(bad,row,binding)
        bad=copy.deepcopy(record);bad['views'][0]['candidates'].pop()
        with self.assertRaises(ValueError):kd.verify_record(bad,row,binding)

    def test_tie_NMS_category_and_threshold(self):
        a=candidate(); b=candidate(view=1); c=candidate(category=2)
        rec=dict(views=[dict(view_id=1,candidates=[b.as_dict()]),
                        dict(view_id=0,candidates=[c.as_dict(),a.as_dict()])])
        out=kd.select_teacher(rec,'T2')
        self.assertEqual([r.identity for r in out],[a.identity,c.identity])
        rec=dict(views=[dict(view_id=0,candidates=[candidate(score=.5).as_dict(),
                    candidate(box=(60,60,70,70),score=.499,query=1).as_dict()])])
        self.assertEqual(len(kd.select_teacher(rec,'T1')),1)

    def test_exact_NMS_boundary_and_target_budget(self):
        a=candidate((0.,0.,3.,1.),query=0)
        edge=candidate((1.,0.,4.,1.),query=1)
        over=candidate((.9,0.,3.9,1.),query=2)
        def select(cs):
            return kd.select_teacher(dict(views=[dict(view_id=0,candidates=[c.as_dict() for c in cs])]),'T1')
        self.assertEqual(len(select([a,edge])),2)  # IoU exactly .5 is retained.
        self.assertEqual(len(select([a,over])),1)
        cs=[candidate((i*10.,0.,i*10.+1.,1.),score=.9,query=i//10,category=i%10+1) for i in range(600)]
        chosen=select(cs)
        self.assertEqual([c.identity for c in chosen],[c.identity for c in cs[:300]])

    def test_cache_raw_SHA_and_reconstruction_rejection(self):
        from sparse_rtdetr.protocol_r18.checkpoint import file_sha
        row=dict(stable_image_id='CPU_RAW_SHA',height=101,width=157,image_sha256='a'*64,annotation_sha256='b'*64)
        binding=kd.cache_binding(source_manifest_sha256='c'*64,contract_sha256='d'*64)
        views={i:dict(pred_logits=torch.full((300,10),-8.),pred_boxes=torch.tensor([[.5,.5,.1,.1]]*300)) for i in range(5)}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);member=kd.write_cache_member(root/'member',row,views,binding)
            index=root/'index.json';index.write_bytes(kd.canonical(dict(closed=True,binding=binding,records=[member])))
            ref=dict(path=str(index),bytes=index.stat().st_size,sha256=file_sha(index))
            self.assertEqual(kd.CacheReader(ref,[row],binding)(row)['member'],row)
            compact=json.loads(Path(member['record']['path']).read_bytes())
            raw_path=Path(compact['raw_reference']['path']);saved=raw_path.read_bytes()
            raw_path.write_bytes(b'corrupted raw')
            with self.assertRaises(ValueError):kd.CacheReader(ref,[row],binding)
            raw_path.write_bytes(saved)
            # Even a freshly hashed compact record must agree with bound raw.
            compact['views'][0]['candidates'][0]['score']=.9
            compact_path=Path(member['record']['path']);compact_path.write_bytes(kd.canonical(compact))
            member['record'].update(bytes=compact_path.stat().st_size,sha256=file_sha(compact_path))
            index.write_bytes(kd.canonical(dict(closed=True,binding=binding,records=[member])))
            ref.update(bytes=index.stat().st_size,sha256=file_sha(index))
            with self.assertRaises(ValueError):kd.CacheReader(ref,[row],binding)

    def test_alignment_actual_augmentation_and_rng(self):
        im,raw=fixture(); aug=RawAugmentation((128,128))
        before=capture_rng(torch.device('cpu'))
        # Pseudo copies of GT check the SAME real geometry/row admission.
        candidates=[candidate(tuple(r['box']),query=i,category=r['category']) for i,r in enumerate(raw['gt'])]
        for epoch in (118,119,126,129):
            for pos in range(8):
                x,t,tr=aug(im,raw,epoch=epoch,seed=0,position=pos,image_id='CPU_FIXTURE')
                pseudo=kd.align_teacher(candidates,(101,157),tr,t,(128,128))
                self.assertTrue(torch.allclose(pseudo['boxes'],t['boxes'],atol=2e-7,rtol=2e-6))
                self.assertTrue(torch.equal(pseudo['labels'],t['labels']))
        self.assertTrue(exact_equal(before,capture_rng(torch.device('cpu'))))

    def test_manual_zoom_crop_flip_and_empty(self):
        im,raw=fixture(); _,t,_=RawAugmentation((128,128))(im,raw,epoch=120,seed=0,position=0,image_id='CPU_FIXTURE')
        tr=dict(augmentation_strong=False,image_id=t['image_id'],geometry_steps=[dict(operation='zoom',padding=[5,7,2,3]),
                    dict(operation='crop',top=10,left=10,height=50,width=60),
                    dict(operation='horizontal_flip')])
        a=kd.align_teacher([candidate()],(101,157),tr,t,(128,128))
        # [10,10,30,30]+[5,7] -> crop [5,7,25,27] -> flip [35,7,55,27].
        expected=torch.tensor([[45/60,17/50,20/60,20/50]])
        self.assertTrue(torch.allclose(a['boxes'],expected))
        self.assertEqual(len(kd.align_teacher([], (101,157),tr,t)['boxes']),0)
        # Out-of-image pseudo is clipped then removed, no resampling.
        self.assertEqual(len(kd.align_teacher([candidate((200,0,220,20))],(101,157),dict(augmentation_strong=False,image_id=t['image_id'],geometry_steps=[]),t)['boxes']),0)

    def test_ignore_same_training_implementation(self):
        im,raw=fixture(); _,t,tr=RawAugmentation((128,128))(im,raw,epoch=120,seed=0,position=0,image_id='CPU_FIXTURE',training=False)
        cs=[candidate((121,1,140,20)),candidate((51,51,59,59),query=1,category=4),candidate((51,51,59,59),query=2,category=5)]
        p=kd.align_teacher(cs,(101,157),tr,t,(128,128))
        self.assertEqual(p['labels'].tolist(),[4])

    def test_native_input_no_teacher_mutation(self):
        im,raw=fixture(); old=copy.deepcopy(raw)
        first=RawAugmentation((128,128))(im,raw,epoch=118,seed=0,position=0,image_id='CPU_FIXTURE')
        for arm in kd.ARMS:
            new=RawAugmentation((128,128))(im,raw,epoch=118,seed=0,position=0,image_id='CPU_FIXTURE')
            self.assertTrue(exact_equal(first,new))
        self.assertEqual(old,raw)

    def test_strong_or_wrong_image_alignment_refused(self):
        im,raw=fixture();_,t,tr=RawAugmentation((128,128))(im,raw,epoch=109,seed=0,position=0,image_id='CPU_FIXTURE')
        with self.assertRaises(ValueError):kd.align_teacher([], (101,157),tr,t)
        tr['augmentation_strong']=False;tr['image_id']='wrong'
        with self.assertRaises(ValueError):kd.align_teacher([], (101,157),tr,t)

    def test_schedule_and_budget(self):
        self.assertEqual(kd.learning_rate(200),1e-5)
        self.assertEqual(kd.learning_rate(3648),1e-6)
        for u in (0,3649,True):
            with self.assertRaises(ValueError):kd.learning_rate(u)
        self.assertEqual(12*304*16*3,175104)

    def test_quality_conditions_unrounded_and_complete(self):
        plan=kd.read_plan(Path(__file__).resolve().parents[1]/'configs/research/r18_multiview_distill_v1.json')
        names=('AP','APS','AP75_small','AP75','APM','APL','AR_large')
        metrics={arm:{budget:{name:.2 for name in names} for budget in ('600','300')} for arm in ('T0','T1','T2','A')}
        for budget in metrics['T2']:
            for name in names:metrics['T2'][budget][name]=.21
        result=kd.quality_gates(metrics,plan)
        self.assertEqual(len(result['conditions']),27);self.assertTrue(result['passed'])
        metrics['T2']['600']['APS']=.20599
        self.assertFalse(kd.quality_gates(metrics,plan)['passed'])
        metrics['T2']['600']['APS']=float('nan')
        with self.assertRaises(ValueError):kd.quality_gates(metrics,plan)

    def test_actual_dataset_getitem_training_bytes_and_teacher(self):
        # Files and one member are synthetic; the actual production getitem and
        # inherited byte/root verifier run unchanged. Frozen4869 constructor
        # identity is separately checked by the real metadata prepare entry.
        im,raw=fixture();encoded=io.BytesIO();im.save(encoded,format='PNG')
        image_bytes=encoded.getvalue();annotation=b'10,10,20,20,1,1,0,0\n'
        from sparse_rtdetr.data_protocol.schema import stable_image_id
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'train_core';images=root/'images';annotations=root/'annotations'
            images.mkdir(parents=True);annotations.mkdir()
            (images/'fixture.png').write_bytes(image_bytes);(annotations/'fixture.txt').write_bytes(annotation)
            row=dict(stable_image_id=stable_image_id('train','train/images/fixture.png'),split='train',
                     relative_path='train/images/fixture.png',annotation_relative_path='train/annotations/fixture.txt',
                     image_sha256=hashlib.sha256(image_bytes).hexdigest(),image_size_bytes=len(image_bytes),
                     annotation_sha256=hashlib.sha256(annotation).hexdigest(),annotation_size_bytes=len(annotation),
                     width=157,height=101,raw_annotation_rows=1)
            binding=kd.cache_binding(source_manifest_sha256='c'*64,contract_sha256='d'*64)
            raw_views={i:dict(pred_logits=torch.full((300,10),-8.),pred_boxes=torch.tensor([[.5,.5,.1,.1]]*300)) for i in range(5)}
            record=kd.from_raw_views(row,raw_views,binding)
            results=[]
            for arm in kd.ARMS:
                dataset=kd.Dataset.__new__(kd.Dataset)
                dataset.arm=arm;dataset.records=[row];dataset.by_id={row['stable_image_id']:row}
                dataset.role='train_core';dataset.seed=0;dataset.images_path=images;dataset.annotations_path=annotations
                dataset.root_binding=dict(role='train_core',images=str(images),annotations=str(annotations))
                dataset.transform=RawAugmentation((128,128));dataset.cache_identity=binding
                dataset.cache_reader=lambda row: record
                result=dataset[(109,0,0)];results.append(result)
                self.assertFalse(result[2]['augmentation_strong'])
                self.assertEqual(result[2]['epoch'],118)
                self.assertEqual(result[2]['image_sha256'],row['image_sha256'])
                if arm!='T0':self.assertEqual(len(result[1]['teacher']['boxes']),0)
            for x,t,tr in results[1:]:
                self.assertTrue(torch.equal(x,results[0][0]))
                self.assertTrue(exact_equal({k:v for k,v in t.items() if k!='teacher'},results[0][1]))
            (annotations/'fixture.txt').write_bytes(b'corruption')
            with self.assertRaises(ValueError):dataset[(109,0,0)]


class NativeLossChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.model,cls.base,_=build(Config(input_hw=(128,128),physical_batch_size=1,amp_dtype='float32'),synthetic=True)

    def test_matching_permutation_not_query_ID(self):
        boxes=torch.tensor([[[.8,.8,.1,.1],[.2,.2,.1,.1]]],requires_grad=True)
        logits=torch.zeros(1,2,10,requires_grad=True)
        t=dict(boxes=torch.tensor([[.2,.2,.1,.1],[.8,.8,.1,.1]]), labels=torch.tensor([0,1]),scores=torch.tensor([.75,.75]))
        loss,pairs=kd.kd_loss(dict(pred_boxes=boxes,pred_logits=logits),[t],self.base.matcher)
        self.assertEqual(pairs,[[(0,1),(1,0)]])
        loss.backward();self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.isfinite(boxes.grad).all())

    def test_KL_hand_value_and_teacher_detach(self):
        boxes=torch.tensor([[[.2,.2,.1,.1]]],requires_grad=True)
        logits=torch.zeros(1,1,10,requires_grad=True)
        t=dict(boxes=boxes[0].detach().clone().requires_grad_(),labels=torch.tensor([0]),scores=torch.tensor([1.],requires_grad=True))
        loss,_=kd.kd_loss(dict(pred_boxes=boxes,pred_logits=logits),[t],self.base.matcher)
        self.assertAlmostEqual(float(loss.detach()),np.log(2),places=6)
        loss.backward();self.assertIsNone(t['scores'].grad);self.assertIsNone(t['boxes'].grad)

    def test_empty_teacher_and_invalid(self):
        boxes=torch.tensor([[[.2,.2,.1,.1]]],requires_grad=True);logits=torch.zeros(1,1,10,requires_grad=True)
        out=dict(pred_boxes=boxes,pred_logits=logits)
        loss,pairs=kd.kd_loss(out,[empty_teacher()],self.base.matcher)
        self.assertEqual(float(loss.detach()),0);self.assertEqual(pairs,[[]]);loss.backward()
        for field,value in [('scores',torch.tensor([float('nan')])),('scores',torch.tensor([.4])),('boxes',torch.tensor([[.2,.2,-.1,.1]]))]:
            t=dict(boxes=torch.tensor([[.2,.2,.1,.1]]),labels=torch.tensor([0]),scores=torch.tensor([.7]));t[field]=value
            with self.assertRaises(ValueError):kd.kd_loss(out,[t],self.base.matcher)

    def test_native_supervised_terms_unchanged(self):
        im,raw=fixture();x,t,_=RawAugmentation((128,128))(im,raw,epoch=120,seed=0,position=0,image_id='CPU_FIXTURE')
        self.model.train();out=self.model(x[None],targets=[t])
        old=self.base(out,[t]);nt=dict(t,teacher=empty_teacher())
        new=kd.Criterion(self.base,'T1')(out,[nt])
        for key in old:self.assertTrue(torch.equal(old[key],new[key]))
        self.assertEqual(float(new['loss_prediction_kd'].detach()),0)
        t0=kd.Criterion(self.base,'T0')(out,[t])
        self.assertEqual(set(old),set(t0))
        with self.assertRaises(ValueError):kd.Criterion(self.base,'T0')(out,[nt])


if __name__=='__main__':unittest.main()
