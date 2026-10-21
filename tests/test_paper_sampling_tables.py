"""Tiny JSON fixtures only. No models, datasets, predictions, training or AP calls."""
import copy
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('paper_tables', ROOT / 'tools/paper_sampling_tables.py')
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)


def fixture(root, alter=None):
    """Equal arbitrary 25% values check formatting only; never detector outputs."""
    root.mkdir(parents=True, exist_ok=True)

    def put(name, value):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
        raw = path.read_bytes()
        return dict(path='/SYNTHETIC_TEST_ONLY/' + name, sha256=t.hashlib.sha256(raw).hexdigest(), bytes=len(raw))

    lock = t.read(t.BINDING)
    source = put('source.json', dict(commit='test-commit', files=[], scope='SYNTHETIC_TEST_ONLY'))
    c = dict(protocol=t.PROTOCOL, profile='CPU_FIXTURE_ONLY', attempt='SYNTHETIC_TEST_ONLY',
             source=source, commit='test-commit', science_sha256='test-science', pretrained=dict(sha256='test-init'))
    c['contract_sha256'] = t.digest(c)
    put('contract.json', c)
    stages, closures = {}, []
    for stage in lock['stages']:
        identity = dict(task_id=lock['task_id'], protocol=t.PROTOCOL, attempt=c['attempt'], stage=stage,
                        source_sha256=source['sha256'], contract_sha256=c['contract_sha256'],
                        science_sha256=c['science_sha256'], profile=c['profile'], initialization_sha256='test-init')
        value = dict(status='PASS', identity=identity)
        if stage.startswith('train-'):
            value.update(used_acceptance_state=False, actual_updates=2)
        if stage == 'score':
            results = {}
            for arm in t.ARMS:
                results[arm] = {}
                for budget in t.BUDGETS:
                    score = dict(official={**{n:25.0 for n in t.PRIMARY}, 'evaluated_image_count':6},
                                 COCO_bridge=dict(metrics={n:0.25 for n in t.METRICS}),
                                 synthetic_test_identity=arm+'-'+budget)
                    results[arm][budget] = put(f'score/{arm}-{budget}.json', score)
            value.update(groups=6, evaluator_calls=12, results=results)
        if stage == 'cost':
            value['summary'] = {a: dict(mean_seconds=1., sample_SD_seconds=.1, min_seconds=.9, max_seconds=1.1,
                                       peak_allocated_bytes=0, peak_reserved_bytes=0) for a in ('A', *t.ARMS)}
        if stage == 'compare':
            value.update(quality_passed=None, cost_passed=None, quality_gates=[dict(pair=p, budget=b,
                metric=m, minimum=v, delta_pp=0., passed=0. >= v) for (p,b,m),v in t.gate_keys(lock).items()],
                deltas_pp={p:{b:{m:0. for m in t.METRICS} for b in t.BUDGETS} for p in ('P-B0','P-B1')},
                cost=stages['cost']['summary'], scope='CPU_FIXTURE_ONLY_NOT_SCIENCE', total_forward_calls=84,
                train_updates=6, inputs_equal=True, P_over_B0_E2E=1., P_over_B0_peak_allocated=None)
        if alter:
            alter(stage, value)
        result_ref = put(f'{stage}/RESULT.json', value)
        exit_value = dict(status='PASS', identity=identity, returncode=0, remaining_live=[], cleanup_errors=[])
        sup = put(f'{stage}/supervisor-exit.json', exit_value)
        worker = put(f'{stage}/worker-exit.json', exit_value)
        closures.append(put(f'{stage}/CLOSED.json', dict(status='PASS', identity=identity, result=result_ref,
                            supervisor_exit=sup, worker_exit=worker)))
        stages[stage] = value
    root_result = dict(status='PASS', task_id=lock['task_id'], profile=c['profile'], stages=lock['stages'],
        source_sha256=source['sha256'], contract_sha256=c['contract_sha256'], external_acceptance={'test_only':True},
        forward_calls_total=84, inherited_forward_calls=18, new_forward_calls=66, quality_passed=None, cost_passed=None)
    result_ref = put('RESULT.json', root_result)
    put('CLOSED.json', dict(status='PASS', result=result_ref, stage_closures=closures, external_acceptance={'test_only':True}))
    return root / 'CLOSED.json', root / 'contract.json'


class Tables(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.receipts = self.root / 'receipts'

    def tearDown(self):
        self.temp.cleanup()

    def load(self, alter=None):
        closed, contract = fixture(self.receipts, alter)
        return t.load_closed(self.receipts, closed, contract, test_fixture=True)

    def test_format_units_and_descriptive_delta(self):
        data = self.load()
        absolute, delta, _, gates = t.rows_for(data)
        self.assertEqual(len(absolute), 120)
        self.assertTrue(all(r['value_percent'] == 25 for r in absolute))
        self.assertEqual(len(delta), 120)
        self.assertTrue(all(r['delta_pp'] == 0 for r in delta))
        self.assertEqual(len(gates), 22)
        self.assertTrue(all(g['display_status'] == 'TEST_ONLY_NOT_SCIENCE' for g in gates))

    def test_default_refuses_cpu_receipts(self):
        closed, contract = fixture(self.receipts)
        with self.assertRaisesRegex(ValueError, 'test-fixture'):
            t.load_closed(self.receipts, closed, contract)

    def test_cli_real_json_closure(self):
        closed, contract = fixture(self.receipts)
        output = self.root / 'test-output'
        p = subprocess.run([sys.executable, '-B', str(ROOT/'tools/paper_sampling_tables.py'), 'render',
            '--receipts', str(self.receipts), '--closed', str(closed), '--contract', str(contract),
            '--output', str(output), '--test-fixture'], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('SYNTHETIC TEST ONLY', (output/'TABLES.md').read_text())
        self.assertFalse(t.read(output/'PROVENANCE.json')['recalculated_AP'])
        with self.assertRaisesRegex(ValueError, 'never overwrite'):
            t.render(self.receipts, closed, contract, output, test_fixture=True)

    def test_refuses_wrong_sha_and_preserves_input(self):
        closed, contract = fixture(self.receipts)
        path = self.receipts/'score/P-600.json'
        path.write_text(path.read_text()+' ')
        with self.assertRaisesRegex(ValueError, 'SHA/size missing'):
            t.load_closed(self.receipts, closed, contract, test_fixture=True)

    def test_identity_and_initialization_rejections(self):
        for field in ('used_acceptance_state', 'actual_updates', 'identity'):
            with self.subTest(field=field):
                def alter(stage, value):
                    if stage == 'train-P':
                        value[field] = True if field == 'used_acceptance_state' else (3 if field == 'actual_updates' else {})
                with self.assertRaises((ValueError, KeyError)):
                    self.load(alter)

    def test_gate_values_are_copied_not_redecided(self):
        def alter(stage, value):
            if stage == 'compare':
                value['quality_gates'][0]['passed'] = True
        data = self.load(alter)
        self.assertIs(t.rows_for(data)[3][0]['passed'], True)
        self.assertIsNone(data['comparison']['quality_passed'])

    def test_omitted_or_changed_threshold_rejected(self):
        for mode in ('omit', 'minimum', 'NaN'):
            def alter(stage, value):
                if stage == 'compare':
                    if mode == 'omit': value['quality_gates'].pop()
                    if mode == 'minimum': value['quality_gates'][0]['minimum'] = -.5
                    if mode == 'NaN': value['quality_gates'][0]['delta_pp'] = float('nan')
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.load(alter)

    def test_missing_root_stop_and_exit_refusal(self):
        closed, contract = fixture(self.receipts)
        (self.receipts/'STOP.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'STOP'):
            t.load_closed(self.receipts, closed, contract, test_fixture=True)

    def test_pending_has_no_scores(self):
        output = self.root/'pending'
        t.template(output)
        with (output/'PENDING_absolute_metrics.csv').open(newline='') as f:
            rows = list(csv.DictReader(f))
        self.assertTrue(all(r['value_percent'] == '' for r in rows))
        self.assertIn('PENDING CLOSED RESULTS', (output/'RESULTS_PENDING.svg').read_text())

    def test_never_imports_model_or_evaluator(self):
        code = (ROOT/'tools/paper_sampling_tables.py').read_text()
        self.assertNotIn('import torch', code)
        self.assertNotIn('import sparse_rtdetr', code)


if __name__ == '__main__':
    unittest.main()
