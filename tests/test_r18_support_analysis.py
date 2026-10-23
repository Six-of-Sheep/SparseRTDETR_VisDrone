"""Display/provenance tests only; all values are artificial and temporary."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from r18_support_analyze import analyze, describe


class AnalysisTests(unittest.TestCase):
    def test_unknown_not_zero_and_negative_retained(self):
        s=describe([None,-.2,0.,.1])
        self.assertEqual((s['defined'],s['undefined'],s['negative'],s['zero'],s['positive']),(3,1,1,1,1))
        self.assertLess(s['mean'],0)
        self.assertIsNone(describe([None])['mean'])

    def test_stop_has_no_scientific_summary(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'EXIT.json').write_text(json.dumps(dict(status='STOP_NO_RETRY')),encoding='utf-8')
            s,r=analyze(p)
            self.assertEqual(s['status'],'STOP_NO_RETRY');self.assertEqual(r,[])
            self.assertNotIn('coverage',s)

    def test_full_population_and_tamper_refusal(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            def write(n,v):
                path=p/n;path.write_text(json.dumps(v),encoding='utf-8')
                return dict(path='/server/'+n,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            write('EXIT.json',dict(status='PASS',GPU_stage_seconds=99))
            write('CLOSED.json',dict(status='PASS'))
            images=[];queries=[];rows=[]
            for arm in ('P','B0'):
                for i in range(32):
                    images.append(dict(arm=arm,id=str(i),valid_GT=1,small_GT=1,matched_GT=1,matched_small=1))
                    for layer in range(3):
                        queries.append(dict(arm=arm,image_id=str(i),ordinary=300,below_cell_queries=[0],layer=layer))
                        rows.append(dict(arm=arm,image_id=str(i),gt_id='gt',query=0,layer=layer,triggered=True,
                            floor_applied=arm=='P',actual_I=.4,native_I=.5,delta_I=-.1,actual_O=.5,native_O=.4,delta_O=.1,
                            actual_V=.1,native_V=.1,delta_V=0.,actual_M=None,native_M=.5,delta_M=None,
                            stencil_changed=True,effective_interpolation_changed=True,local_core_output_abs_mean=.01))
            columns=list(rows[0])
            r=write('MEASUREMENTS.json',dict(columns=columns,rows=[[r[k] for k in columns] for r in rows]))
            v=write('INVENTORY.json',dict(images=images,queries=queries,peaks=[],state_checks=[]))
            write('WORKER_RESULT.json',dict(result=r,inventory=v))
            s,_=analyze(p)
            self.assertEqual(s['coverage']['P']['ordinary_query_layer_instances'],28800)
            self.assertEqual(s['coverage']['P']['matched_small'],32)
            self.assertEqual(s['common_matched_small_GT'],32)
            self.assertEqual(s['layers'][0]['delta_M']['undefined'],32)
            self.assertLess(s['layers'][0]['delta_I']['mean'],0)
            (p/'MEASUREMENTS.json').write_text('{}',encoding='utf-8')
            with self.assertRaises(AssertionError):
                analyze(p)


if __name__=='__main__':
    unittest.main()
