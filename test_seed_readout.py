"""Only affected pairing/gate/readout interface tests; no detector construction."""
import copy
import json
from pathlib import Path
import unittest

import aggregate_seeds as aggregate


def fixture(seed=0, change=.01, cost=4.5):
    scores = {a: {b: {m: .2 + (change if a == 'C' else .001 if a == 'B' else 0)
                        for m in aggregate.METRICS} for b in aggregate.BUDGETS}
              for a in aggregate.ARMS}
    differences = {b: {a + '-' + z: {m: (scores[a][b][m] - scores[z][b][m]) * 100
                           for m in aggregate.METRICS} for a, z in aggregate.PAIRS}
                   for b in aggregate.BUDGETS}
    gates = {f'top{b}_{m}': {'difference_pp': differences[b]['C-A'][m], 'minimum_pp': limit,
                           'passed': differences[b]['C-A'][m] >= limit}
             for b, limits in aggregate.LIMITS.items() for m, limit in limits.items()}
    science = all(row['passed'] for row in gates.values())
    return {'images':548, 'seed':seed, 'scope':f'formal_R18_seed{seed}_e120',
            'checkpoint_sha256':str(seed) * 64, 'COCO_bridge_original_score_units':'0..1',
            'difference_units':'percentage points', 'scores':scores, 'differences_pp':differences,
            'gates_C_minus_A':gates, 'cost':{'seconds':{'A':2.,'B':2.2,'C':2 * cost},
             'C_over_A':cost,'maximum_C_over_A':6.,'passed':cost <= 6.},
            'benefit_and_protection_passed':science, 'screen_passed':science and cost <= 6.,
            'official_separate':{}}


class SeedReadoutTests(unittest.TestCase):
    def test_sample_sd_and_worst(self):
        s = aggregate.paired_stats([-1,0,1])
        self.assertEqual((s['mean_pp'],s['sample_sd_pp'],s['worst_pp']), (0,1,-1))
        self.assertEqual(s['sample_sd_ddof'],1)

    def test_no_missing_or_extra_seeds(self):
        for values in ([1,2], [0,1,2,3]):
            with self.assertRaises(ValueError): aggregate.paired_stats(values)
        with self.assertRaises(ValueError): aggregate.summarize({'0':fixture()})

    def test_scientific_fail_not_masked_by_mean(self):
        records = {'0':fixture(0,.04),'1':fixture(1,.04),'2':fixture(2,0)}
        result = aggregate.summarize(records)
        self.assertGreater(result['paired_delta_statistics_pp']['600']['C-A']['AP']['mean_pp'], .3)
        self.assertEqual(result['decision'],'FAIL')

    def test_cost_fail_separate_from_science(self):
        records = {str(s):fixture(s) for s in range(3)}
        records['2'] = fixture(2,cost=6.01)
        result = aggregate.summarize(records)
        self.assertTrue(result['strict_benefit_and_protection_passed'])
        self.assertFalse(result['strict_cost_passed'])
        self.assertEqual(result['decision'],'FAIL')

    def test_exact_cost_boundary(self):
        result = aggregate.summarize({str(s):fixture(s,cost=6.) for s in range(3)})
        self.assertTrue(result['strict_repeatability_passed'])

    def test_changed_unit_and_delta_rejected(self):
        for mutation in ('units','delta','threshold','seed'):
            records = {str(s):fixture(s) for s in range(3)}
            if mutation == 'units': records['1']['difference_units']='score'
            elif mutation == 'delta': records['1']['differences_pp']['600']['C-A']['AP'] *= 100
            elif mutation == 'threshold': records['1']['gates_C_minus_A']['top600_AP']['minimum_pp']=0
            else: records['1']['seed']=2
            with self.subTest(mutation=mutation),self.assertRaises(ValueError): aggregate.summarize(records)

    def test_three_pass_all(self):
        result = aggregate.summarize({str(s):fixture(s) for s in range(3)})
        self.assertEqual(result['decision'],'PASS')
        self.assertEqual(len(result['paired_delta_statistics_pp']['600']['C-A']),7)

    def test_unavailable_nonfinite_refused(self):
        for value in (-1,float('nan'),float('inf'),True):
            records = {str(s):fixture(s) for s in range(3)}
            records['1']['scores']['A']['600']['AP']=value
            with self.subTest(value=value),self.assertRaises(ValueError): aggregate.summarize(records)


if __name__ == '__main__':
    unittest.main()
