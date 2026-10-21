"""Read closed sampling-span receipts; format existing scores, never evaluate AP.

Only local JSON is read. Remote paths in references are labels, never opened.
Receipt relocation is resolved by exact SHA and size, without rewriting evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINDING = ROOT / 'docs/research/r18_sampling_paper_tools_20261021/FROZEN_OUTPUT_BINDING.json'
ARMS = ('B0', 'B1', 'P')
BUDGETS = ('600', '300')
METRICS = ('AP', 'AP50', 'AP75', 'APS', 'APM', 'APL', 'AR1', 'AR10',
           'AR100', 'AR_small', 'AR_medium', 'AR_large', 'AP75_small')
PRIMARY = ('AP', 'AP50', 'AP75', 'AR1', 'AR10', 'AR100', 'AR500')
PROTOCOL = 'r18-sampling-span-formal-external-acceptance-v1'
MAX_JSON_BYTES = 2 * 1024 * 1024


def require(ok, message):
    if not ok:
        raise ValueError(message)


def finite(value):
    require(type(value) in (float, int) and math.isfinite(value), 'missing/nonfinite metric')
    return value


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


class Receipts:
    """Bounded JSON-only inventory of an already-returned, small receipt folder."""
    def __init__(self, folder):
        self.root = Path(folder).resolve()
        require(self.root.is_dir(), 'receipt folder missing')
        self.entries = {}
        self.used = {}
        files = sorted(self.root.rglob('*.json'))
        require(len(files) <= 2000, 'use a slim receipt directory, not a full experiment tree')
        for path in files:
            require(not path.is_symlink() and path.resolve().is_relative_to(self.root), 'receipt link refused')
            size = path.stat().st_size
            require(size <= MAX_JSON_BYTES, 'large JSON refused; supply necessary receipts only')
            raw = path.read_bytes()
            key = (hashlib.sha256(raw).hexdigest(), len(raw))
            self.entries[key] = path

    def exact(self, ref):
        require(isinstance(ref, dict) and set(ref) == {'path', 'sha256', 'bytes'}, 'invalid receipt reference')
        key = (ref['sha256'], ref['bytes'])
        require(key in self.entries, 'referenced JSON SHA/size missing: ' + str(ref['path']))
        path = self.entries[key]
        self.used[str(path.relative_to(self.root))] = dict(sha256=key[0], bytes=key[1], original_path=ref['path'])
        return read(path)

    def local(self, path):
        path = Path(path).resolve()
        require(path.is_relative_to(self.root), 'closed receipt must be inside receipt folder')
        require(not (path.parent / 'STOP.json').exists(), 'root STOP present')
        raw = path.read_bytes()
        return self.exact(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw)))


def gate_keys(lock):
    result = {}
    for pair in ('P-B0', 'P-B1'):
        for key, minimum in lock['gates'].items():
            budget, rest = key.split('_', 1)
            evaluator = 'primary' if rest.startswith('primary_') else 'COCO'
            name = rest[len(evaluator) + 1:-4]
            if evaluator == 'primary':
                name = 'primary_' + name
            result[(pair, int(budget[3:]), name)] = minimum
    return result


def load_closed(folder, closed_path, contract_path, *, test_fixture=False):
    lock = read(BINDING)
    store = Receipts(folder)
    closed = store.local(closed_path)
    result = store.exact(closed['result'])
    contract = store.local(contract_path)
    unsigned = {k: v for k, v in contract.items() if k != 'contract_sha256'}
    require(digest(unsigned) == contract['contract_sha256'] == result['contract_sha256'], 'contract digest differs')
    source = store.exact(contract['source'])
    require(contract['source']['sha256'] == result['source_sha256'] and source['commit'] == contract['commit'], 'source manifest identity differs')
    require(closed['status'] == result['status'] == 'PASS', 'root incomplete')
    fixture = result.get('profile') == 'CPU_FIXTURE_ONLY'
    require(fixture == test_fixture, 'CPU fixture requires --test-fixture; production refuses test mode')
    require(result['profile'] in ('LAB_GPU_ONCE', 'CPU_FIXTURE_ONLY'), 'unrecognized profile')
    require(result['task_id'] == lock['task_id'] and result['stages'] == lock['stages'], 'wrong task/stage sequence')
    if not fixture:
        require(result['source_sha256'] == lock['source_sha256'] and
                result['contract_sha256'] == lock['contract_sha256'], 'wrong frozen source/contract')
        require(source['commit'] == lock['commit'] and contract['attempt'] == lock['attempt'], 'wrong frozen commit/attempt')
    require(contract['protocol'] == PROTOCOL and contract['profile'] == result['profile'], 'contract profile/protocol differs')
    require(closed['external_acceptance'] == result['external_acceptance'], 'acceptance reference differs')
    stages = {}
    for closure in closed['stage_closures']:
        c = store.exact(closure)
        identity = c['identity']
        stage = identity['stage']
        require(stage not in stages and stage in lock['stages'], 'duplicate/unexpected stage')
        require(c['status'] == 'PASS' and identity['task_id'] == lock['task_id'] and
                identity['protocol'] == PROTOCOL and identity['profile'] == result['profile'] and
                identity['source_sha256'] == result['source_sha256'] and
                identity['contract_sha256'] == result['contract_sha256'], 'stage identity mismatch')
        if not fixture:
            require(identity['attempt'] == lock['attempt'], 'wrong attempt')
        require(identity['attempt'] == contract['attempt'] and identity['science_sha256'] == contract['science_sha256'] and
                identity['initialization_sha256'] == contract['pretrained']['sha256'], 'stage science/init identity differs')
        value = store.exact(c['result'])
        require(value['status'] == 'PASS' and value['identity'] == identity, 'stage result identity/closure differs')
        # Worker/supervisor receipts are small; event streams remain server-side.
        for key in ('supervisor_exit', 'worker_exit'):
            exited = store.exact(c[key])
            require(exited['status'] == 'PASS' and exited['identity'] == identity, 'stage process identity/exit differs')
            if key == 'supervisor_exit':
                require(exited['returncode'] == 0 and not exited['remaining_live'] and
                        not exited['cleanup_errors'], 'stage process not closed')
        stages[stage] = value
    require(list(stages) == lock['stages'], 'root stage closures incomplete/out of order')
    comparison = stages['compare']
    require(comparison['scope'] == ('CPU_FIXTURE_ONLY_NOT_SCIENCE' if fixture else
                                  'seed0_development548_exploration'), 'comparison scope differs')
    require(comparison['inputs_equal'] is True, 'three-arm input identity failed')
    for key in ('quality_passed', 'cost_passed'):
        require(comparison[key] == result[key] and (comparison[key] is None if fixture else
                type(comparison[key]) is bool), 'missing/inconsistent terminal decision')
    expected_calls = 84 if fixture else 219693
    require(result['forward_calls_total'] == comparison['total_forward_calls'] == expected_calls and
            result['inherited_forward_calls'] == 18 and result['new_forward_calls'] == expected_calls - 18,
            'cumulative forward totals differ')
    require(comparison['train_updates'] == (6 if fixture else 109440), 'training update total differs')
    metrics = {}
    require(stages['score']['groups'] == 6 and stages['score']['evaluator_calls'] == 12, 'score coverage incomplete')
    for arm in ARMS:
        train = stages['train-' + arm]
        require(train['used_acceptance_state'] is False and
                train['actual_updates'] == (2 if fixture else 36480), 'wrong formal initialization/endpoint receipt')
        for budget in BUDGETS:
            score = store.exact(stages['score']['results'][arm][budget])
            coco, official = score['COCO_bridge']['metrics'], score['official']
            require(set(coco) == set(METRICS), 'COCO metric schema differs')
            for name in METRICS:
                finite(coco[name])
                require(fixture or 0 <= coco[name] <= 1, 'undefined/out-of-range COCO metric')
            for name in PRIMARY:
                finite(official[name])
                require(fixture or 0 <= official[name] <= 100, 'invalid primary metric')
            require(official['evaluated_image_count'] == (6 if fixture else 548), 'score image coverage differs')
            metrics[arm, budget] = score
    gates = comparison['quality_gates']
    expected, observed = gate_keys(lock), {}
    for g in gates:
        key = g['pair'], g['budget'], g['metric']
        require(key in expected and key not in observed and g['minimum'] == expected[key] and
                type(g['passed']) is bool, 'gate schema/threshold differs')
        finite(g['delta_pp'])
        observed[key] = g
    require(observed.keys() == expected.keys(), 'quality gates incomplete')
    for pair in ('P-B0', 'P-B1'):
        for b in BUDGETS:
            require(set(comparison['deltas_pp'][pair][b]) == set(METRICS), 'stored comparison metrics incomplete')
            for value in comparison['deltas_pp'][pair][b].values():
                finite(value)
    require(stages['cost']['summary'] == comparison['cost'], 'cost result/compare differ')
    require(fixture or stages['cost']['scope'] == 'FIXED64_GPU_E2E', 'wrong timing scope')
    for value in comparison['cost'].values():
        for name in ('mean_seconds', 'sample_SD_seconds', 'min_seconds', 'max_seconds',
                     'peak_allocated_bytes', 'peak_reserved_bytes'):
            require(finite(value[name]) >= 0, 'invalid cost summary')
    return dict(lock=lock, store=store, root=result, comparison=comparison,
                metrics=metrics, fixture=fixture)


def rows_for(data):
    comparison, metrics = data['comparison'], data['metrics']
    absolute, deltas = [], []
    for budget in BUDGETS:
        for arm in ARMS:
            value = metrics[arm, budget]
            for evaluator, values, names, scale in (
                    ('COCO_bridge', value['COCO_bridge']['metrics'], METRICS, 100),
                    ('primary', value['official'], PRIMARY, 1)):
                for name in names:
                    absolute.append(dict(arm=arm, budget=int(budget), evaluator=evaluator,
                                         metric=name, value_percent=finite(values[name]) * scale))
        for pair, baseline in (('P-B0', 'B0'), ('P-B1', 'B1'), ('B1-B0', 'B0')):
            for name in METRICS:
                if pair == 'B1-B0':
                    value = (metrics['B1', budget]['COCO_bridge']['metrics'][name] -
                             metrics['B0', budget]['COCO_bridge']['metrics'][name]) * 100
                    origin = 'descriptive subtraction of already-scored metrics'
                else:
                    value = comparison['deltas_pp'][pair][budget][name]
                    origin = 'frozen comparison.deltas_pp'
                deltas.append(dict(pair=pair, budget=int(budget), evaluator='COCO_bridge',
                                   metric=name, delta_pp=value, origin=origin))
            for name in PRIMARY:
                if pair != 'B1-B0' and name == 'AP':
                    value = next(g['delta_pp'] for g in comparison['quality_gates'] if
                                 g['pair'] == pair and g['budget'] == int(budget) and g['metric'] == 'primary_AP')
                    origin = 'frozen comparison.quality_gates'
                else:
                    target = 'B1' if pair == 'B1-B0' else 'P'
                    value = metrics[target, budget]['official'][name] - metrics[baseline, budget]['official'][name]
                    origin = 'descriptive subtraction of already-scored metrics'
                deltas.append(dict(pair=pair, budget=int(budget), evaluator='primary',
                                   metric=name, delta_pp=value, origin=origin))
    cost = [dict(arm=arm, **value) for arm, value in comparison['cost'].items()]
    gates = [dict(g, display_status='TEST_ONLY_NOT_SCIENCE' if data['fixture'] else
                  ('PASS' if g['passed'] else 'FAIL')) for g in comparison['quality_gates']]
    return absolute, deltas, cost, gates


def write_csv(path, rows):
    require(rows, 'no table rows')
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown(headers, rows):
    def cell(v):
        return f'{v:.4f}' if isinstance(v, float) else str(v)
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |'] +
                     ['| ' + ' | '.join(cell(v) for v in row) + ' |' for row in rows])


def plots(folder, data, absolute, deltas, cost):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    title = 'SYNTHETIC TEST ONLY - NOT EXPERIMENT RESULTS' if data['fixture'] else 'seed0 development548, fixed e120 EMA'
    names = ('AP', 'APS', 'AP75_small', 'APM', 'APL', 'AR_large')
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout='constrained')
    for ax, name in zip(axes.flat, names):
        for i, (pair, color) in enumerate(zip(('P-B0', 'P-B1', 'B1-B0'), ('#176582', '#cb7433', '#807987'))):
            value = next(r['delta_pp'] for r in deltas if r['pair'] == pair and r['budget'] == 600 and
                         r['evaluator'] == 'COCO_bridge' and r['metric'] == name)
            ax.bar(i, value, color=color)
        ax.axhline(0, color='#555555', lw=.8)
        ax.set_xticks(range(3), ('P-B0', 'P-B1', 'B1-B0'))
        ax.set_title(name)
        ax.set_ylabel('Delta (pp)')
    fig.suptitle(title + '\nMain top600; negative results retained; no uncertainty estimate', fontsize=12)
    for extension in ('svg', 'png'):
        fig.savefig(folder / ('metric_deltas.' + extension), dpi=190)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.bar([r['arm'] for r in cost], [r['mean_seconds'] for r in cost],
           yerr=[r['sample_SD_seconds'] for r in cost], capsize=4, color='#4d7f93')
    ax.set_ylabel('Seconds per fixed64 pass (mean +/- sample SD)')
    ax.set_title(title + '\nFP32 batch4; four measured passes', fontsize=11)
    for extension in ('svg', 'png'):
        fig.savefig(folder / ('cost.' + extension), dpi=190)
    plt.close(fig)


def render(folder, closed_path, contract_path, output, *, test_fixture=False, make_plots=False):
    data = load_closed(folder, closed_path, contract_path, test_fixture=test_fixture)
    absolute, deltas, cost, gates = rows_for(data)
    output = Path(output)
    require(not output.exists(), 'new output directory required; never overwrite prior figures')
    output.mkdir(parents=True)
    for name, rows in (('absolute_metrics', absolute), ('paired_deltas', deltas), ('cost', cost), ('gates', gates)):
        write_csv(output / (name + '.csv'), rows)
    banner = 'SYNTHETIC TEST ONLY - NOT EXPERIMENT RESULTS' if data['fixture'] else 'Closed formal results: seed0 / development548'
    content = ['# ' + banner, '', 'COCO absolute values are percent; deltas are percentage points. Primary values are already percent.',
               'Fixed e120 EMA; single seed, repeatedly used development set. No seed SD or significance claim.', '']
    for budget in BUDGETS:
        content += ['## top' + budget, '', markdown(['arm', *METRICS],
                    [[a] + [r['value_percent'] for m in METRICS for r in absolute if r['arm'] == a and
                     r['budget'] == int(budget) and r['evaluator'] == 'COCO_bridge' and r['metric'] == m] for a in ARMS]), '',
                    markdown(['pair', *METRICS], [[p] + [r['delta_pp'] for m in METRICS for r in deltas if
                     r['pair'] == p and r['budget'] == int(budget) and r['evaluator'] == 'COCO_bridge' and r['metric'] == m]
                     for p in ('P-B0', 'P-B1', 'B1-B0')]), '',
                    markdown(['arm', *PRIMARY], [[a] + [r['value_percent'] for m in PRIMARY for r in absolute if
                     r['arm'] == a and r['budget'] == int(budget) and r['evaluator'] == 'primary' and r['metric'] == m]
                     for a in ARMS]), '']
    content += ['## Frozen gate decisions (not recomputed)', '', markdown(list(gates[0]), [list(r.values()) for r in gates]), '',
                '## Cost', '', markdown(list(cost[0]), [list(r.values()) for r in cost]), '',
                'Timing: fixed64, FP32 batch4, two warmup and four measured full passes. Model load and GT parse excluded;',
                'decode, resize, H2D, forward, ranking, raw-ignore and CUDA synchronization included. Not batch1 deployment.',
                'Cost ratios and overall flags are copied from the frozen comparison; no gate is decided by rounded display values.', '']
    comp = data['comparison']
    content += [f"Stored quality_passed={comp['quality_passed']}; cost_passed={comp['cost_passed']}; "
                f"P/B0 E2E={comp['P_over_B0_E2E']}; peak allocated={comp['P_over_B0_peak_allocated']}.", '',
                'Historical P-A / P-C deltas remain in the original comparison, not same-run causal controls.',
                'Reporting verifies local root/stage result and exit references. Full event streams, data and tensors are not re-audited here.', '']
    (output / 'TABLES.md').write_text('\n'.join(content), encoding='utf-8')
    (output / 'DECISIONS.json').write_text(json.dumps(dict(
        notice=banner, quality_passed=comp['quality_passed'], cost_passed=comp['cost_passed'],
        original_quality_gates=comp['quality_gates'], original_cost_ratios=dict(
            P_over_B0_E2E=comp['P_over_B0_E2E'], P_over_B0_peak_allocated=comp['P_over_B0_peak_allocated']),
        frozen_cost_targets=data['lock']['cost_targets'],
        individual_cost_gate_status='UNKNOWN: not separately stored; combined cost_passed copied above',
        recalculated_decisions=False), indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (output / 'PROVENANCE.json').write_text(json.dumps(dict(
        status=banner, binding=read(BINDING), result=data['root'],
        inputs=data['store'].used, recalculated_AP=False, recalculated_gate_decisions=False,
        B1_minus_B0='descriptive subtraction only', full_runtime_certification=False), indent=2) + '\n', encoding='utf-8')
    if make_plots:
        plots(output, data, absolute, deltas, cost)
    return output


def template(output):
    output = Path(output)
    require(not output.exists(), 'new output directory required')
    output.mkdir(parents=True)
    rows = [dict(arm=a, budget=int(b), evaluator='COCO_bridge', metric=m, value_percent='', status='PENDING_CLOSED_RESULTS')
            for a in ARMS for b in BUDGETS for m in METRICS]
    write_csv(output / 'PENDING_absolute_metrics.csv', rows)
    write_csv(output / 'PENDING_paired_deltas.csv', [dict(pair=p, budget=int(b), metric=m, delta_pp='', status='PENDING_CLOSED_RESULTS')
              for p in ('P-B0', 'P-B1', 'B1-B0') for b in BUDGETS for m in METRICS])
    write_csv(output / 'PENDING_cost.csv', [dict(arm=a, mean_seconds='', sample_SD_seconds='', peak_allocated_bytes='', status='PENDING_CLOSED_RESULTS')
                                         for a in ('A', *ARMS)])
    lock = read(BINDING)
    write_csv(output / 'PENDING_gates.csv', [dict(pair=p, budget=b, metric=m, minimum=v, passed='', status='PENDING_CLOSED_RESULTS')
                                           for (p, b, m), v in gate_keys(lock).items()])
    (output / 'RESULTS_PENDING.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" width="960" height="360">'
        '<rect width="960" height="360" fill="white"/><text x="480" y="160" text-anchor="middle" '
        'font-family="sans-serif" font-size="27">PENDING CLOSED RESULTS</text><text x="480" y="215" '
        'text-anchor="middle" font-family="sans-serif" font-size="19">No experimental scores or benefits shown</text></svg>', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    pending = sub.add_parser('template')
    pending.add_argument('--output', required=True, type=Path)
    render_parser = sub.add_parser('render')
    render_parser.add_argument('--receipts', required=True, type=Path)
    render_parser.add_argument('--closed', required=True, type=Path)
    render_parser.add_argument('--contract', required=True, type=Path)
    render_parser.add_argument('--output', required=True, type=Path)
    render_parser.add_argument('--test-fixture', action='store_true')
    render_parser.add_argument('--plots', action='store_true')
    args = parser.parse_args()
    if args.action == 'template':
        template(args.output)
    else:
        render(args.receipts, args.closed, args.contract, args.output, test_fixture=args.test_fixture, make_plots=args.plots)


if __name__ == '__main__':
    main()
