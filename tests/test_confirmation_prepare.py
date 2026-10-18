import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import confirmation_prepare as p


class PreparationTests(unittest.TestCase):
    def test_partial_batches(self):
        for n in (1,2,3,4,5,547,548,1602):
            rows=list(range(n));chunks=p.batches(rows)
            self.assertEqual(sum(chunks,[]),rows)
            self.assertEqual(len(chunks),(n+3)//4)
            self.assertEqual(len(chunks[-1]),(n-1)%4+1)
        self.assertEqual(p.batches([]),[])

    def test_invalid_batch_size(self):
        for size in (0,-1,True,1.5):
            with self.assertRaises(RuntimeError):p.batches([1],size)

    def test_members(self):
        rows=[{'stable_image_id':'a'},{'stable_image_id':'b'}]
        p.members(rows,['a','b'])
        for ids in (['b','a'],['a'],['a','a'],[]):
            with self.assertRaises(RuntimeError):p.members(rows,ids)

    def test_config_is_not_parsed_gt(self):
        with self.assertRaisesRegex(RuntimeError,'parsed GT list'):
            p.validate_parsed([{'stable_image_id':'a'}],{'roots':{}})

    def test_seed_and_checkpoint_binding(self):
        parents=[dict(seed=s,checkpoint_sha256=str(s)*64,identity={'checkpoint_sha256':str(s)*64}) for s in (0,1,2)]
        expected=[s['checkpoint_sha256'] for s in parents];p.validate_parents(parents,expected)
        for bad in (parents[:2], [parents[1],parents[0],parents[2]], [parents[0],parents[0],parents[2]]):
            with self.assertRaises(RuntimeError):p.validate_parents(bad,expected)
        bad=copy.deepcopy(parents);bad[1]['identity']['checkpoint_sha256']=expected[2]
        with self.assertRaises(RuntimeError):p.validate_parents(bad,expected)


class ScientificStackTests(unittest.TestCase):
    def setUp(self):
        try:import torch
        except ImportError:self.skipTest('remote certified scientific CPU stack required')

    def test_raw_ignore_and_budget_inputs(self):
        from sparse_rtdetr.localization.data import parse_raw
        row={'stable_image_id':'a','width':20,'height':20,'raw_annotation_rows':2}
        # Full-image ignore; caller already selected the output budget.
        raw=parse_raw(b'0,0,20,20,0,0,0,0\n2,2,3,3,1,1,0,0\n','a',expected_rows=2)
        pred=[dict(image_id='a',category_id=1,score=.5,bbox_xyxy=[2,2,5,5])]
        proof=p.evaluator_inputs([row],[raw],pred,300)
        self.assertEqual(proof['counts'],dict(GT_before=1,GT_after=0,detections_before=1,detections_after=0))
        self.assertEqual(proof['primary_GT_including_ignore'],2)
        self.assertFalse(proof['AP_scored'])
        self.assertEqual(p.evaluator_inputs([row],[raw],[],300)['members'],1)

    def test_wrong_GT_identity(self):
        from sparse_rtdetr.localization.data import parse_raw
        row={'stable_image_id':'a','raw_annotation_rows':1}
        raw=parse_raw(b'2,2,3,3,1,1,0,0\n','b',expected_rows=1)
        with self.assertRaisesRegex(RuntimeError,'GT image/line identity'):p.validate_parsed([row],[raw])

    def test_channel_refusals(self):
        p.install()
        from sparse_rtdetr.localization import channel
        with tempfile.TemporaryDirectory() as tmp:
            b=channel.create(Path(tmp)/'runtime',dict(task_id=p.TASK,seed=0,progress_protocol=p.PROTOCOL,
                source_sha256='1'*64,contract_sha256='2'*64,checkpoint_sha256='3'*64))
            writer=channel.Writer(b,'progress')
            def payload(step,**kw):
                return dict(event='confirmation_prepare_step',step=step,label=p.LABELS[step],seed=0,
                            checkpoint_sha256='3'*64,monotonic=float(step+10),**kw)
            for bad in (payload(1),{**payload(0),'seed':1},{**payload(0),'checkpoint_sha256':'4'*64}):
                with self.assertRaises(RuntimeError):writer.append(bad)
            for step in range(len(p.LABELS)-1):writer.append(payload(step))
            with self.assertRaisesRegex(RuntimeError,'incomplete protocol'):p.closed_stream(b['runtime_root'],b)
            writer.append(payload(len(p.LABELS)-1))
            before=(Path(b['runtime_root'])/'progress.jsonl').read_bytes()
            with self.assertRaises(RuntimeError):writer.append(payload(len(p.LABELS)-1))
            self.assertEqual(before,(Path(b['runtime_root'])/'progress.jsonl').read_bytes())
            r=channel.Reader(Path(b['runtime_root'])/'progress.jsonl',b,'progress');r.poll()
            self.assertEqual(r.clock['_confirmation_next'],len(p.LABELS))
            # A new transport record with a prior phase cannot hide as replay.
            with self.assertRaises(RuntimeError):writer.append({**payload(0),'monotonic':100.})
            damaged=copy.deepcopy(b);damaged['seed']=1
            with self.assertRaises(RuntimeError):p.closed_stream(b['runtime_root'],damaged)
            target=Path(b['runtime_root'])/'progress.jsonl'
            data=target.read_text();target.write_text(data.replace('confirmation_prepare_step','confirmation_prepare_stex',1))
            with self.assertRaises(RuntimeError):p.closed_stream(b['runtime_root'],b)


if __name__=='__main__':unittest.main()
