"""Affected production reader/current paths acceptance; never scores or uses CUDA."""
import argparse
import copy
import json
import os
from pathlib import Path
import time
import supplemental_runtime as rt
from supplemental_runtime import exact, reference, require, publish


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--context', required=True)
    p.add_argument('--sha256', required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--prior-source', required=True)
    p.add_argument('--prior-gate', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args(); rt.install()
    import torch
    from supplemental_data import dataset, current_paths, member_path, read_bytes
    from confirmation_prepare import evaluator_inputs
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU-only check')
    torch.set_num_threads(1)
    out = Path(a.output); out.mkdir(exist_ok=False)
    start = time.monotonic()
    cr = reference(a.context); require(cr['sha256'] == a.sha256, 'context SHA')
    c = exact(cr)
    data, rows, parsed = dataset(c)
    require(len(rows) == len(parsed) == 1602, 'real member coverage')
    # Exercise the unchanged production inputs for both budgets without scoring.
    proofs = {str(b): evaluator_inputs(rows, parsed, [], b) for b in (600, 300)}
    require(all(x['counts']['detections_before'] == x['counts']['detections_after'] == 0 for x in proofs.values()), 'empty check predictions')
    tests = []
    def rejects(name, call):
        try:
            call()
        except (RuntimeError, ValueError, FileNotFoundError, KeyError):
            tests.append(name)
        else:
            raise RuntimeError('invalid current path admitted: ' + name)
    for key in ('images', 'annotations'):
        bad = copy.deepcopy(c); bad['evaluation_data'][key] += '-missing'
        rejects('unbound_' + key + '_root', lambda: current_paths(bad, rows))
    bad = copy.deepcopy(c); bad['evaluation_data']['path_binding']['sha256'] = '0' * 64
    rejects('binding_SHA', lambda: current_paths(bad, rows))
    rejects('member_order', lambda: current_paths(c, list(reversed(rows))))
    rejects('member_missing', lambda: current_paths(c, rows[:-1]))
    rejects('member_duplicate', lambda: current_paths(c, rows[:-1] + [rows[0]]))
    missing = out / 'missing'; missing.mkdir()
    rejects('file_missing', lambda: member_path(missing, rows[0], 'annotations'))
    actual = member_path(c['evaluation_data']['annotations'], rows[0], 'annotations')
    rejects('annotation_SHA', lambda: read_bytes(actual, '0' * 64, rows[0]['annotation_size_bytes']))
    rejects('annotation_size', lambda: read_bytes(actual, rows[0]['annotation_sha256'], rows[0]['annotation_size_bytes'] + 1))
    wrong = dict(rows[0], annotation_relative_path='../other.txt')
    rejects('member_path_escape', lambda: member_path(c['evaluation_data']['annotations'], wrong, 'annotations'))
    source = rt.read(a.source); prior = rt.read(a.prior_source); gate = rt.read(a.prior_gate)
    require(gate['status'] == 'PASS' and gate['source_sha256'] == reference(a.prior_source)['sha256'], 'prior CPU/source binding')
    old = {x['path']: x for x in prior['files']}; new = {x['path']: x for x in source['files']}
    changed = [x for x in old if x not in new or old[x] != new[x]]
    require(changed == ['supplemental_data.py'], 'unexpected prior runtime change: ' + repr(changed))
    for r in source['files']:
        require(reference(rt.ROOT / r['path']) == dict(path=str(rt.ROOT/r['path']), bytes=r['bytes'], sha256=r['sha256']), 'deployed source changed')
    require(not torch.cuda.is_initialized(), 'CUDA initialized during path checks')
    result = dict(gate)
    result.update(status='PASS', source_sha256=reference(a.source)['sha256'],
        prior_CPU_gate=reference(a.prior_gate), prior_source=reference(a.prior_source),
        reuse_scope='unchanged launcher/supervisor/worker/evaluators/restore/closure/export/exit and fixture checks; prior gate not rewritten',
        changed_prior_runtime_files=changed, real_current_reader_context=cr,
        actual_production_annotations=1602, actual_physical_rows=sum(len(p['rows']) for p in parsed),
        real_evaluator_input_proofs=proofs, current_path_refusals=tests,
        sealed_content_access=True, GPU_forward_calls=0, scientific_evaluation_groups=0,
        production_sealed_reader_not_yet_opened=False, elapsed_seconds=time.monotonic()-start)
    publish(out/'CPU_GATE.json', result)
    print(json.dumps({'status':'PASS','CPU_gate':reference(out/'CPU_GATE.json'),'refusals':tests,'annotations':1602}))


if __name__ == '__main__':
    main()
