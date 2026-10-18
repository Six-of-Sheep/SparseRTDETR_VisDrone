"""Targeted parsed-GT schema and identity refusals; no detector or scoring."""
import copy
import sys
from pathlib import Path
import unittest
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'src')]
from sixth_view_score_reuse import validate_annotations
from sparse_rtdetr.localization.data import parse_raw


class Annotations(unittest.TestCase):
    def setUp(self):
        self.ids=[f'fixture-{i}' for i in range(548)]
        self.rows=[{'stable_image_id':i,'split':'val','raw_annotation_rows':3,'annotation_sha256':'a'*64,'annotation_size_bytes':72} for i in self.ids]
        raw=b'0,0,10,10,1,1,0,0\n10,0,2,2,0,0,0,0\n0,10,3,3,0,2,0,0\n'
        self.parsed=[parse_raw(raw,i) for i in self.ids]
    def test_valid(self):self.assertEqual(validate_annotations(self.rows,self.parsed,self.ids)['physical_annotations'],1644)
    def test_config_not_annotations(self):
        with self.assertRaises(RuntimeError):validate_annotations(self.rows,{'roots':{}},self.ids)
    def test_missing_member(self):
        with self.assertRaises(RuntimeError):validate_annotations(self.rows,self.parsed[:-1],self.ids)
    def test_reordered_gt(self):
        with self.assertRaises(RuntimeError):validate_annotations(self.rows,self.parsed[::-1],self.ids)
    def test_duplicate_member(self):
        with self.assertRaises(RuntimeError):validate_annotations(self.rows,self.parsed,self.ids[:-1]+self.ids[:1])
    def test_ignore_partition(self):
        p=copy.deepcopy(self.parsed);p[0]['spatial_ignore']=[]
        with self.assertRaises(RuntimeError):validate_annotations(self.rows,p,self.ids)
    def test_counts_conflict(self):
        p=copy.deepcopy(self.parsed);p[0]['counts']['valid_gt']=0
        with self.assertRaises(RuntimeError):validate_annotations(self.rows,p,self.ids)


if __name__=='__main__':unittest.main()
