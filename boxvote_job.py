"""Independent CPU-only cache job. Never imports a detector or old launcher.

Production reads parent closure, authenticates pre-NMS views, reproduces A/B/C,
then scores exactly E/D x two budgets x three seeds. A server waiter may start
this once after the original seed chain exports its terminal evidence.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from slice_core import Candidate, merge_candidates, select_view_candidates, tile_windows, map_crop_xyxy
from boxvote_core import coordinate_vote, compare_scores, LIMITS, METRICS, PAIRS

TASK = 'sahi-box-vote-fixed-20261018-001'
PLAN = ROOT / 'configs/sahi_boxvote_fixed.json'
MAX_RECORD = 1024 ** 2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + '\n').encode()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 ** 2), b''):
            h.update(b)
    return h.hexdigest()


def reference(path):
    p = Path(path)
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)}


def exact(ref, *, contents=False):
    p = Path(ref['path'])
    require(p.is_absolute() and not p.is_symlink(), 'non-absolute/symlink reference')
    require(not any(x.is_symlink() for x in p.parents), 'reference traverses symlink')
    require(not any(k in str(p).lower().replace('-', '_') for k in ('holdout', 'confirmatory', 'test_dev')),
            'forbidden data role before read')
    require(p.stat().st_size == ref.get('bytes', ref.get('size_bytes')), 'reference size differs: ' + str(p))
    require(sha(p) == ref['sha256'], 'reference SHA differs: ' + str(p))
    return p.read_bytes() if contents else p


def publish(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    b = encoded(value)
    temporary = p.with_name(p.name + '.partial')
    require(not p.exists(), 'immutable output already exists: ' + str(p))
    with temporary.open('xb') as f:
        f.write(b); f.flush(); os.fsync(f.fileno())
    os.link(temporary, p)
    temporary.unlink()
    return reference(p)


def expected_events():
    return (['started', 'parents_closed'] + [f'cache_seed{s}' for s in range(3)] + ['all_caches_admitted']
            + [f'score_seed{s}_{a}_{b}' for s in range(3) for a in ('V-whole', 'V-five') for b in (600, 300)]
            + ['complete'])


class Journal:
    def __init__(self, out, binding):
        self.path = Path(out) / 'events.jsonl'
        self.binding = binding
        self.number = 0
        self.previous = None

    def emit(self, event, **fields):
        require(self.number < len(expected_events()) and event == expected_events()[self.number],
                'event missing/duplicate/out of order: ' + event)
        rec = {'seq': self.number + 1, 'event': event, 'binding': self.binding,
               'previous': self.previous, 'time': time.time(), 'fields': fields}
        rec['sha256'] = digest(rec)
        b = encoded(rec).replace(b'\n', b'') + b'\n'
        require(len(b) <= MAX_RECORD, 'progress record exceeds final 1MiB cap')
        with self.path.open('ab') as f:
            f.write(b); f.flush(); os.fsync(f.fileno())
        self.number += 1
        self.previous = rec['sha256']

    @staticmethod
    def validate(path, binding, *, complete=False):
        previous = None; names = []
        for line in Path(path).read_bytes().splitlines():
            require(len(line) + 1 <= MAX_RECORD, 'oversized journal')
            r = json.loads(line); proof = r.pop('sha256')
            require(proof == digest(r) and r['previous'] == previous and r['binding'] == binding,
                    'journal SHA/binding/chain differs')
            require(r['seq'] == len(names) + 1, 'journal sequence differs')
            names.append(r['event']); previous = proof
        require(names == expected_events()[:len(names)], 'missing/duplicate/reordered terminal')
        require(not complete or names == expected_events(), 'journal is not closed')
        return {'events': len(names), 'last_sha256': previous}


def cpu_only():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CUDA must be explicitly hidden')
    import torch
    torch.set_num_threads(1)
    require(not torch.cuda.is_initialized(), 'CUDA was initialized')
    def forbidden(*args, **kwargs):
        raise RuntimeError('CPU cache job forbids model construction, CUDA and backward')
    torch.cuda._lazy_init = forbidden
    torch.nn.Module.__init__ = forbidden
    torch.autograd.backward = forbidden
    return torch


def resources(out, plan):
    # No GPU API, power write or access to another process. Missing counters fail.
    info = {line.split(':')[0]: int(line.split()[1]) * 1024
            for line in Path('/proc/meminfo').read_text().splitlines()}
    free = shutil.disk_usage(out).free
    temps = []
    for p in Path('/sys/class/hwmon').glob('hwmon*/temp*_input'):
        if (p.parent / 'name').read_text().strip() in ('coretemp', 'k10temp'):
            temps.append(float(p.read_text()) / 1000.)
    require(temps and all(math.isfinite(v) for v in temps), 'CPU temperature unavailable')
    require(info['MemAvailable'] >= plan['minimum_available_RAM_GiB'] * 1024 ** 3, 'RAM insufficient')
    require(free >= plan['minimum_free_disk_GiB'] * 1024 ** 3, 'disk insufficient')
    require(max(temps) < plan['max_CPU_temperature_C'], 'CPU temperature exceeds cap')
    return {'time': time.time(), 'RAM_available': info['MemAvailable'], 'disk_free': free,
            'CPU_temperature_max': max(temps), 'GPU_access': False}


def proc(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return {'start': fields[19], 'state': fields[0]}
    except FileNotFoundError:
        return None


def parent_ready(c):
    if c.get('synthetic_validation') is True:
        from softnms_entry_fixture import validate_contract as validate_fixture_contract
        validate_fixture_contract(c)
        return True
    parent = Path(c['parent_root'])
    exact(c['parent_execution'])
    execution = read(c['parent_execution']['path'])
    require(execution['execution_sha256'] == c['parent_execution_digest'], 'parent execution differs')
    require(execution['execution_sha256'] == digest({k:v for k,v in execution.items() if k!='execution_sha256'}),
            'parent execution canonical digest differs')
    require(not (parent / 'STOP.json').exists() and not (parent / 'EXPORT_ERROR.json').exists(), 'parent STOP/export error')
    launch = json.loads(exact(c['parent_launch'], contents=True))
    live = proc(launch['pid'])
    require(live is None or live['start'] == launch['process_identity']['start'], 'parent PID identity changed')
    result, export = parent / 'CHAIN_RESULT.json', parent / 'SLIM_EXPORT.json'
    if not result.exists() or not export.exists():
        require(live is not None and live['state'] not in ('Z', 'X'), 'parent exited before complete evidence')
        return False
    # Wait for original launcher exit so this task cannot contend with its export.
    if live is not None and live['state'] not in ('Z', 'X'):
        return False
    r, e = read(result), read(export)
    require(r['status'] == 'FIXED_PAIRED_EXECUTION_COMPLETE' and r['seeds'] == [1,2], 'parent incomplete')
    require(e['status'] == 'READY_FOR_WINDOWS_SLIM_RETURN', 'parent export incomplete')
    refs = {v['path']:v for v in e['files']}
    require(refs.get(str(result)) == reference(result), 'parent terminal not bound by slim export')
    require([v['seed'] for v in r['completed']] == [1,2], 'parent seed list differs')
    for item in r['completed']:
        exact(item['comparison']); exact(item['result'])
    return True


def closed(root, phase, contract):
    d = Path(root) / phase
    closure = read(d / 'CLOSED.json')
    require(closure['status'] == 'PASS', 'parent phase not CLOSED PASS')
    require(closure['result'] == reference(d / 'RESULT.json'), 'closed result differs')
    require(closure['runtime_index'] == reference(d / 'runtime-archive-index.json'),
            'parent archive index path/binding differs')
    index = json.loads(exact(closure['runtime_index'], contents=True))
    archived = d / 'runtime-archive'
    for name in ('identity.json', 'worker-exit.json', 'supervisor-exit.json'):
        require(sha(archived/name) == index['files'][name], 'parent archive key damaged: '+name)
    require(index['identity_sha256'] == index['files']['identity.json'], 'archive identity digest conflict')
    identity = read(archived / 'identity.json')
    require(identity['schema'] == 'f0-ssd-channel-v4', 'unknown archive identity schema')
    for key in ('task_id', 'source_sha256', 'contract_sha256'):
        require(identity[key] == contract[key], 'archive binding differs: ' + key)
    require(identity['output_root'] == str(d)
            and identity['runtime_root'] == str(Path(contract['runtime_root']) / phase),
            'archive output/runtime root differs')
    require(identity['plan_sha256'] == contract['plan']['sha256'], 'archive plan differs')
    if 'seed' in contract:
        require(identity['seed'] == contract['seed'] and identity['mode'] == phase,
                'archive seed/phase differs')
        require(identity['progress_protocol'] == 'r18-sahi-paired-seed-progress-v1', 'repeat protocol differs')
        require(closure['seed'] == contract['seed'] and closure['mode'] == phase, 'closure seed/phase differs')
        for key in ('fixed64','fixed_ids','arm','data_sha256','material_binding_sha256'):
            require(identity[key] == contract[key], 'archive shared binding differs: '+key)
    else:
        require(identity['phase'] == phase and identity['mode'] == ('cpu' if phase == 'evaluation' else 'gpu'),
                'legacy archive phase/mode differs')
        require(identity['progress_protocol'] == 'sahi-fixed-progress-v1', 'legacy protocol differs')
    require(read(archived / 'worker-exit.json')['success'] is True, 'parent worker did not succeed')
    ex = read(archived / 'supervisor-exit.json')
    require(ex['status'] == 'PASS' and ex['returncode'] == 0 and not ex['remaining_live']
            and not ex['unknown_pids'] and not ex['cleanup_errors'], 'parent exit not clean')
    result = read(d/'RESULT.json')
    require(result['status'] == ('R18_COMMON_E120_EMA_LOCKED' if phase == 'formal' else 'PASS'),
            'parent result not closed successfully')
    return result, identity


def endpoint_checkpoint(contract, seed):
    """Read e120 metadata only; never construct/load a detector or checkpoint."""
    if seed == 0:
        endpoint = json.loads(exact(contract['endpoint'], contents=True))
        require(endpoint['checkpoint'] == contract['checkpoint'], 'legacy checkpoint/endpoint differs')
        training = json.loads(exact(contract['parent_contract'], contents=True))
        require(endpoint['source_sha256'] == contract['parent_source_sha256'] == training['source_sha256']
                and endpoint['contract_sha256'] == contract['expected_parent_contract_digest']
                == training['contract_sha256'], 'legacy training lineage differs')
        require(endpoint['seed'] == 0 and endpoint['original_completed_updates'] == 26752
                and endpoint['actual_updates'] == 9728, 'legacy resumed update accounting differs')
    else:
        training = contract
        if seed == 1:
            reuse = contract['parent_reuse']
            require(reuse['kind'] == 'certified_closed_seed1_e120_and_A548', 'unknown reuse schema')
            training = json.loads(exact(reuse['contract'], contents=True))
            certification = json.loads(exact(reuse['certification'], contents=True))
            require(certification['status'] == 'PASS'
                    and certification['old_contract'] == reuse['contract']
                    and certification['old_source_sha256'] == training['source_sha256']
                    and certification['new_source_sha256'] == contract['source_sha256']
                    and certification['protocol_only_source_delta'] is True, 'reuse certification differs')
            for key, value in certification['reuse_references'].items():
                require(reuse[key] == value, 'reuse reference conflict: ' + key)
            exact(reuse['formal_closed'])
            endpoint = json.loads(exact(reuse['endpoint'], contents=True))
        else:
            require(seed == 2 and 'parent_reuse' not in contract, 'unexpected seed/reuse branch')
            endpoint = read(Path(contract['job_root']) / 'formal/endpoint-lock.json')
        formal, _ = closed(training['job_root'], 'formal', training)
        require(formal == endpoint, 'formal CLOSED result differs from endpoint')
        require(training['seed'] == endpoint['seed'] == seed and training['scope'] == contract['scope'],
                'training endpoint seed/scope differs')
        for key in ('source_sha256', 'contract_sha256'):
            require(endpoint[key] == training[key], 'endpoint training binding differs: ' + key)
        require(endpoint['actual_updates'] == 36480, 'independent seed update accounting differs')
        if seed == 1:
            require(certification['checkpoint'] == endpoint['checkpoint'], 'reuse checkpoint differs')
    require(endpoint['status'] == 'R18_COMMON_E120_EMA_LOCKED' and endpoint['epoch'] == 120
            and endpoint['updates'] == 36480
            and endpoint['used_acceptance_state'] is False, 'endpoint is not independent e120')
    cp = endpoint['checkpoint']
    require(cp['completed_updates'] == 36480 and cp['scope'] == f'formal_R18_seed{seed}_e120',
            'checkpoint update/scope differs')
    return cp


def parent_metadata(c):
    """Authenticate all three real schema variants before touching raw candidates."""
    execution = json.loads(exact(c['parent_execution'], contents=True))
    require(execution['execution_sha256'] == c['parent_execution_digest']
            == digest({k:v for k,v in execution.items() if k != 'execution_sha256'}), 'execution digest differs')
    contracts = [json.loads(exact(r, contents=True)) for r in execution['contracts']]
    require([x['seed'] for x in contracts] == [1, 2], 'parent contract seeds differ')
    s0 = json.loads(exact(c['seed0_contract'], contents=True))
    specs = []
    for seed, contract in [(0, s0), *[(x['seed'], x) for x in contracts]]:
        root = Path(contract['job_root'])
        exact(contract['plan'])
        full, fi = closed(root, 'full-C', contract)
        ev, ei = closed(root, 'evaluation', contract)
        cp = endpoint_checkpoint(contract, seed)
        ids = contract['full_image_ids'] if seed == 0 else contract['development_ids']
        require(len(ids) == len(set(ids)) == 548, 'development IDs invalid')
        identity = {k:full[k] for k in ('task_id','source_sha256','contract_sha256','checkpoint_sha256')}
        require(full['phase'] == 'full-C', 'full-C phase differs')
        for key in ('task_id','source_sha256','contract_sha256'):
            require(identity[key] == contract[key] == fi[key] == ei[key], 'parent identity conflict: ' + key)
        require(identity['checkpoint_sha256'] == cp['sha256'], 'full-C checkpoint differs')
        if seed == 0:
            require(ev['schema'] == 'r18-sahi-fixed-evaluation-v1', 'unknown legacy evaluation schema')
            require(fi['checkpoint_sha256'] == ei['checkpoint_sha256'] == cp['sha256'],
                    'legacy archive checkpoint differs')
            require(fi['image_ids'] == ids, 'legacy full-C members differ')
            require(len(contract['fixed64']) == len(set(contract['fixed64'])) == 64
                    and set(contract['fixed64']).issubset(ids), 'legacy fixed64 members invalid')
            require(ei['image_ids'] == contract['fixed64'], 'legacy archive fixed64 differs')
            # Missing legacy fields come only from authenticated archive, never defaults.
            for key in ('task_id','checkpoint_sha256'):
                if key in ev:
                    require(ev[key] == ei[key], 'legacy optional result identity conflict: ' + key)
            for key in ('source_sha256','contract_sha256','plan_sha256'):
                require(ev[key] == ei[key] == full[key], 'legacy result binding differs: ' + key)
            require(ev['full_C_result'] == reference(root/'full-C/RESULT.json'), 'legacy full-C association differs')
            exact(ev['full_C_result'])
        else:
            require(full['schema'] == 'r18-sahi-repeat-inference-v1'
                    and ev['schema'] == 'r18-sahi-repeat-evaluation-v1', 'unknown repeat schema')
            identity.update(seed=seed, scope=contract['scope'])
            require(contract['scope'] == f'formal_R18_seed{seed}_e120', 'contract scope differs')
            for result in (full, ev):
                require(all(result[k] == v for k,v in identity.items()), 'repeat result identity differs')
            for archive in (fi, ei):
                auth = archive['evaluation_authorization']
                require(archive['starting_cursor'] == 36480 and auth['checkpoint_sha256'] == cp['sha256']
                        and auth['image_ids'] == ids and auth['budgets'] == [600,300],
                        'repeat archive endpoint/images/budgets differ')
        require(ev['budgets'] == [600,300], 'evaluation budgets differ')
        require(full['images'] == ev['images'] == 548 and full['whole_output_exact'] == 548,
                'parent does not cover full development')
        require(full['detector_updates'] == 0 and full['state_RNG_exact'] is True,
                'parent inference mutated detector')
        require(len(full['raw_view_refs']) == 137, 'pre-NMS raw-view batches incomplete')
        require(ev['predictions']['C'] == full['predictions'], 'C prediction refs differ')
        comparison = json.loads(exact(ev['comparison'], contents=True))
        require(comparison['task_id'] == identity['task_id'], 'comparison task differs')
        if seed:
            require(all(comparison[k] == v for k,v in identity.items()), 'comparison identity differs')
        require(comparison['COCO_bridge_original_score_units'] == '0..1'
                and comparison['difference_units'] == 'percentage points', 'original units differ')
        require(set(comparison['scores']) == {'A','B','C'}, 'original arms differ')
        sources = authenticate_parent_sources(contract, seed)
        metrics = parent_metric_metadata(ev, comparison, identity, seed)
        specs.append({'seed':seed,'identity':identity,'ids':ids,'full':full,'evaluation':ev,
                      'comparison':comparison,'root':str(root),'contract':contract,
                      'metrics':metrics,'generating_sources':sources})
    require(all(s['ids']==specs[0]['ids'] for s in specs), 'cross-seed image identity/order differs')
    return specs, contracts


def authenticate_parent_sources(contract, seed):
    """Only the generating entry, data and evaluator code; never a model import."""
    common = ['src/sparse_rtdetr/r0/evaluation.py', 'src/sparse_rtdetr/r0/evaluation_data.py',
              'src/sparse_rtdetr/baseline/primary_evaluator.py', 'src/sparse_rtdetr/data_protocol/evaluation.py',
              'src/sparse_rtdetr/data_protocol/schema.py', 'src/sparse_rtdetr/localization/material_paths.py']
    refs = []
    if seed == 0:
        manifest = json.loads(exact(contract['source_manifest'], contents=True))
        require(contract['source_manifest']['sha256'] == contract['source_sha256'], 'legacy source manifest differs')
        require(manifest['files'] == contract['source_files'], 'legacy source list differs')
        for name in ('deploy.py','evaluate_cached.py','chain.py'):
            hits = [r for r in manifest['files'] if Path(r['path']).name == name]
            require(len(hits) == 1, 'legacy generator missing/duplicate')
            exact(hits[0]); refs.append(hits[0])
        parent = json.loads(exact(contract['parent_contract'], contents=True))
        root = Path(parent['source_root']); mp = Path(parent['source_manifest'])
        require(sha(mp) == contract['parent_source_sha256'], 'legacy evaluation source differs')
    else:
        root = Path(contract['source_root']); mp = Path(contract['source_manifest'])
        require(sha(mp) == contract['source_sha256'], 'repeat source manifest differs')
        common += ['infer_seed.py', 'new_entry.py', 'new_runtime.py']
    manifest = read(mp)
    for name in common:
        hits = [r for r in manifest['files'] if r['path'] == name]
        require(len(hits) == 1, 'generating source missing/duplicate: '+name)
        ref = {**hits[0], 'path':str(root/name)}
        exact(ref); refs.append(ref)
        if name.startswith('src/'):
            require(sha(ROOT/name) == ref['sha256'], 'local canonical input implementation differs: '+name)
    return refs


def parent_metric_metadata(ev, comparison, identity, seed):
    metrics = {}
    require(set(ev['predictions']) == set(ev['metrics']) == {'A','B','C'}, 'parent arms differ')
    for arm in ('A','B','C'):
        metrics[arm] = {}
        require(set(ev['predictions'][arm]) == set(ev['metrics'][arm]) == {'600','300'}, 'parent budgets differ')
        for b in ('600','300'):
            m = json.loads(exact(ev['metrics'][arm][b], contents=True))
            require(m['COCO_bridge']['units'] == '0..1 AP, -1 means no GT in an area/class aggregation',
                    'parent metric units differ')
            require({k:m['COCO_bridge']['metrics'][k] for k in METRICS} == comparison['scores'][arm][b],
                    'original comparison/metric scores differ')
            official = m['official']
            require(official['evaluated_image_count'] == 548
                    and len(official['canonical_input_sha256']) == 64, 'missing canonical evaluation input proof')
            if seed or arm != 'A':
                binding = m['sahi_binding']
                for key in ('task_id','source_sha256','contract_sha256'):
                    require(binding[key] == identity[key], 'metric source identity differs: '+key)
                if seed:
                    require(all(binding[k] == v for k,v in identity.items()), 'metric seed/checkpoint differs')
                require(binding['arm'] == arm and binding['budget'] == int(b)
                        and binding['prediction'] == ev['predictions'][arm][b], 'metric prediction binding differs')
            metrics[arm][b] = m
    return metrics


def context(c):
    specs, contracts = parent_metadata(c)
    spec = contracts[0]['data']['development']
    require(all(x['data']['development'] == spec for x in contracts), 'GT/data roots differ')
    from sparse_rtdetr.localization.material_paths import MANIFESTS, metadata_path
    p = metadata_path('development', spec['manifest'], MANIFESTS['development'][0])
    require(sha(p) == MANIFESTS['development'][0], 'development manifest SHA differs')
    rows = read(p)['records']
    require([r['stable_image_id'] for r in rows] == specs[0]['ids'] and all(r['split']=='val' for r in rows),
            'development membership differs')
    for s in specs:
        contract = s['contract']
        data = (json.loads(exact(contract['parent_contract'], contents=True))['data']['development']
                if s['seed'] == 0 else contract['data']['development'])
        require(data == spec, 'parent evaluation data source differs')
        require(contract['full_image_ids'] == s['ids'], 'full/development member order differs')
        if s['seed']:
            require(contract['development_ids'] == s['ids'], 'repeat evaluation membership differs')
    from sparse_rtdetr.r0.evaluation import POLICY
    for x in contracts:
        require(sha(x['evaluation_protocol']) == x['evaluation_protocol_sha256']
                and read(x['evaluation_protocol']) == POLICY, 'evaluator policy differs')
    return specs, rows, spec


def evaluation_count_proof(rows, parsed, predictions, metric):
    """Authenticate the pre-filter input and reuse the frozen raster filter, no AP."""
    from sparse_rtdetr.r0.evaluation import primary_inputs, bridge_inputs
    from sparse_rtdetr.baseline.primary_evaluator import _input_payload
    from sparse_rtdetr.data_protocol.schema import canonical_json_bytes
    payload = _input_payload(primary_inputs(rows, parsed, predictions))
    actual = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    official = metric['official']
    require(actual == official['canonical_input_sha256'], 'saved evaluation canonical input SHA differs')
    dataset, filtered, counts = bridge_inputs(rows, parsed, predictions)
    saved = metric['spatial_filter_counts']
    require(all(type(official[k]) is int for k in
                ('evaluated_image_count','evaluated_detection_count','evaluated_ground_truth_count')),
            'saved evaluated counts are not native integers')
    require(set(saved) == set(counts), 'saved spatial count fields differ')
    require(all(type(saved[k]) is int and saved[k] == counts[k] for k in counts),
            'saved spatial before/after counts differ')
    require(counts['detections_before'] == len(payload['detections']) == len(predictions),
            'before count is not the budget-truncated input')
    require(counts['GT_before'] == sum(g['category_id'] in range(1,11) for g in payload['ground_truth']),
            'before GT count includes excluded categories')
    require(counts['detections_after'] == len(filtered) == official['evaluated_detection_count']
            and counts['GT_after'] == len(dataset['annotations']) == official['evaluated_ground_truth_count']
            and len(rows) == official['evaluated_image_count'], 'saved evaluated after counts differ')
    return {'canonical_input_sha256':actual, 'counts':counts,
            'raw_input_GT_including_regions_and_excluded_categories':len(payload['ground_truth']),
            'official_evaluated_detection_count':official['evaluated_detection_count'],
            'official_evaluated_ground_truth_count':official['evaluated_ground_truth_count'],
            'filter_implementation':'sparse_rtdetr.r0.evaluation.bridge_inputs (frozen official raster primitive)',
            'order':['original candidates','postprocess','budget truncation','raw-ignore','evaluation'],
            'no_refill':True,'rescored':False}


def evaluation_input_proof(spec, rows, parsed, predictions, arm, budget):
    """Recreate the saved evaluator INPUT hash; never evaluates detections."""
    ids = spec['ids']
    require([r['stable_image_id'] for r in rows] == ids and len(set(ids)) == len(ids) == 548,
            'evaluation input members missing/duplicate/reordered')
    grouped = group(predictions, ids, budget)
    require(list(dict.fromkeys(p['image_id'] for p in predictions)) == ids,
            'prediction members missing/duplicate-block/reordered')
    require(all(grouped[i] for i in ids), 'prediction member is absent')
    counts = evaluation_count_proof(rows, parsed, predictions, spec['metrics'][arm][str(budget)])
    return {'arm':arm,'budget':budget,'members':len(ids),'ordered_members_sha256':digest(ids),
            **counts,'metric':spec['evaluation']['metrics'][arm][str(budget)],
            'prediction':spec['evaluation']['predictions'][arm][str(budget)],'rescored':False}


def prove_all_evaluation_inputs(specs, rows, parsed, progress):
    proofs = []
    for spec in specs:
        for arm in ('A','B','C'):
            for budget in (600,300):
                predictions = json.loads(exact(spec['evaluation']['predictions'][arm][str(budget)], contents=True))
                proofs.append({'seed':spec['seed'], **evaluation_input_proof(spec,rows,parsed,predictions,arm,budget)})
                progress(seed=spec['seed'],stage='evaluation_input_SHA',arm=arm,budget=budget)
    return {'status':'PASS','proofs':proofs,'score_budget_calls':0,
            'ordered_image_ids':[r['stable_image_id'] for r in rows],
            'data_manifest_rows_sha256':digest(rows),
            'legacy_archive_fixed64_is_not_evaluation_membership':True}


def prediction(image, candidate, whole=False):
    p = {'image_id':image,'query_id':candidate.query_id,'category_id':candidate.category_id,
         'score':candidate.score,'bbox_xyxy':list(candidate.box_xyxy)}
    if not whole:
        p.update(view_id=candidate.view_id, view_candidate_rank=candidate.view_candidate_rank)
    return p


def decode_raw(raw, rows, identity, seed):
    import torch
    from sparse_rtdetr.localization.consumer_geometry import corners, positive_box
    require(raw['schema'] == ('sahi-fixed-raw-views-v1' if seed==0 else 'sahi-repeat-raw-views-v1'), 'raw schema differs')
    required = {'checkpoint_sha256':identity['checkpoint_sha256']} if seed==0 else identity
    require(all(raw.get(k) == v for k,v in required.items()), 'raw views wrong source/seed/checkpoint')
    require(raw['image_ids'] == [r['stable_image_id'] for r in rows], 'raw image order differs')
    require(raw['geometry'] == [[t.as_dict() for t in tile_windows(r['height'],r['width'])] for r in rows], 'crop geometry differs')
    require([v['view_id'] for v in raw['views']] == list(range(5)), 'missing/duplicate/reordered view')
    pools = [[] for _ in rows]
    for view in raw['views']:
        view_id = view['view_id']; logits, boxes = view['pred_logits'], view['pred_boxes']
        require(logits.device.type == boxes.device.type == 'cpu' and logits.dtype == boxes.dtype == torch.float32,
                'raw precision/device differs')
        require(tuple(logits.shape) == (len(rows),300,10) and tuple(boxes.shape) == (len(rows),300,4), 'raw shape differs')
        require(torch.isfinite(logits).all().item() and not logits.requires_grad and not boxes.requires_grad, 'raw logits invalid')
        positive_box('SoftNMS/raw', boxes)
        for j,row in enumerate(rows):
            tile = tile_windows(row['height'],row['width'])[view_id-1] if view_id else None
            width,height = (tile.width,tile.height) if tile else (row['width'],row['height'])
            b = boxes[j]
            xy = torch.cat((b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2),-1)
            corners('SoftNMS/normalized',xy)
            xy *= torch.tensor([width,height]*2,dtype=torch.float32)
            corners('SoftNMS/view-pixel',xy)
            mapped = map_crop_xyxy(xy.tolist(),tile) if tile else xy.tolist()
            pools[j].extend(select_view_candidates(mapped,logits[j].sigmoid().numpy(),view_id=view_id))
    return pools


def group(predictions, ids, budget):
    result = {i:[] for i in ids}
    for p in predictions:
        require(p['image_id'] in result, 'prediction outside development')
        result[p['image_id']].append(p)
    for rows in result.values():
        require(len(rows)<=budget, 'output exceeds budget')
        keys=[]
        for p in rows:
            obj=Candidate(tuple(p['bbox_xyxy']),p['score'],p.get('view_id',0),p['query_id'],p['category_id'],p.get('view_candidate_rank',1))
            keys.append(obj.identity)
        require(len(keys)==len(set(keys)), 'duplicate output identity')
    return result


def prepare_seed(spec, rows, out, progress):
    import torch
    ids=spec['ids']; expected={}
    for arm in ('A','B','C'):
        expected[arm]={}
        for budget in (600,300):
            ref=spec['evaluation']['predictions'][arm][str(budget)]
            expected[arm][budget]=group(json.loads(exact(ref,contents=True)),ids,budget)
        require(all(expected[arm][300][i]==expected[arm][600][i][:300] for i in ids), 'original budget prefix differs')
    outputs={a:{b:[] for b in (600,300)} for a in ('V-whole','V-five')}; stats=[]
    for k,ref in enumerate(spec['full']['raw_view_refs']):
        raw=torch.load(io.BytesIO(exact(ref,contents=True)),map_location='cpu',weights_only=True)
        part=rows[4*k:4*k+4]
        pools=decode_raw(raw,part,spec['identity'],spec['seed'])
        for row,pool in zip(part,pools):
            image=row['stable_image_id'];whole=[p for p in pool if p.view_id==0]
            rebuilt={'A':{'top600':whole,'top300':whole[:300]},'B':merge_candidates(whole),'C':merge_candidates(pool)}
            for arm,values in rebuilt.items():
                for budget in (600,300):
                    actual=[prediction(image,p,whole=arm=='A') for p in values[f'top{budget}']]
                    require(actual==expected[arm][budget][image],f'seed{spec["seed"]} {arm}{budget} rebuild differs: '+image)
            counts={'image_id':image,'original_stages':{a:{
                    'raw_query_class_scores':int(sum(v['pred_logits'][0].numel() for v in raw['views']
                                                     if a=='C' or v['view_id']==0)),
                    'selected_view_pool':len(pool) if a=='C' else len(whole),
                    'after_postprocess_before_budget':values.get('kept_count',len(whole)),
                    'before_raw_ignore':{str(b):len(values[f'top{b}']) for b in (600,300)}}
                    for a,values in rebuilt.items()}}
            for arm,values in [('V-whole',whole),('V-five',pool)]:
                merged=coordinate_vote(values);counts[arm]=merged['stats']
                for budget in (600,300):
                    selected=merged[f'top{budget}']
                    outputs[arm][budget].extend(prediction(image,p) for p in selected)
                    counts[arm][f'top{budget}_view_counts']={str(v):int(sum(p.view_id==v for p in selected)) for v in range(5)}
            stats.append(counts)
        progress(seed=spec['seed'],batch=k+1,stage='cache_rebuild')
    refs={}
    for arm in ('V-whole','V-five'):
        refs[arm]={}
        for budget in (600,300):
            refs[arm][str(budget)]=publish(out/f'{arm}-top{budget}-predictions.json',outputs[arm][budget])
        a=group(outputs[arm][600],ids,600);b=group(outputs[arm][300],ids,300)
        require(all(b[i]==a[i][:300] for i in ids),'new budget prefix differs')
    statref=publish(out/'COUNTS.json',stats)
    return {'seed':spec['seed'],'predictions':refs,'counts':statref,'images':len(rows),
            'all_A_B_C_exact':True,'raw_batches':len(spec['full']['raw_view_refs']),
            'original_comparison':spec['evaluation']['comparison'],'identity':spec['identity']}


def parsed_gt(rows, spec):
    from sparse_rtdetr.localization.data import parse_raw, validate_role_roots
    validate_role_roots('development',spec['roots']['images'],spec['roots']['annotations'],spec['roots'])
    result=[]
    for row in rows:
        p=Path(spec['roots']['annotations'])/Path(row['annotation_relative_path']).name
        raw=exact({'path':str(p),'bytes':row['annotation_size_bytes'],'sha256':row['annotation_sha256']},contents=True)
        parsed=parse_raw(raw,row['stable_image_id'],expected_rows=row['raw_annotation_rows'])
        require(not parsed['counts'].get('nonpositive_area',0),'nonpositive GT area')
        result.append(parsed)
    return result


def evaluate_seed(seed, prepared, rows, parsed, old, out, journal):
    from sparse_rtdetr.r0.evaluation import score_budget
    scores={a:old['scores'][a] for a in ('A','B','C')}; refs={}; official={}
    for arm in ('V-whole','V-five'):
        scores[arm]={};refs[arm]={};official[arm]={}
        for budget in (600,300):
            ref=prepared['predictions'][arm][str(budget)]
            predictions=json.loads(exact(ref,contents=True))
            record=score_budget(rows,parsed,predictions,ROOT)
            require(record['COCO_bridge']['units']=='0..1 AP, -1 means no GT in an area/class aggregation','new metric units differ')
            scores[arm][str(budget)]={m:record['COCO_bridge']['metrics'][m] for m in METRICS}
            record['boxvote_binding']={'seed':seed,'arm':arm,'budget':budget,'prediction':ref,'plan_sha256':sha(PLAN)}
            refs[arm][str(budget)]=publish(out/f'{arm}-top{budget}-metrics.json',record)
            official[arm][str(budget)]=record['official']
            journal.emit(f'score_seed{seed}_{arm}_{budget}',metric=refs[arm][str(budget)])
    result={'seed':seed,'scores':scores,**compare_scores(scores),'new_metrics':refs,'official_new_separate':official,
            'original_comparison':prepared.get('original_comparison'),'new_score_budget_calls':4}
    publish(out/'COMPARISON.json',result)
    return result


def aggregate(results):
    require([r['seed'] for r in results]==[0,1,2],'aggregate requires all fixed seeds')
    stats={b:{pair:{m:{'values':[r['differences_pp'][b][pair][m] for r in results],
                              'mean':statistics.mean(r['differences_pp'][b][pair][m] for r in results),
                              'sample_SD':statistics.stdev(r['differences_pp'][b][pair][m] for r in results),
                              'minimum':min(r['differences_pp'][b][pair][m] for r in results)} for m in METRICS}
                       for pair in PAIRS} for b in ('600','300')}
    passed=all(r['quality_passed'] for r in results)
    return {'status':'QUALITY_PASS_COST_PENDING' if passed else 'QUALITY_FAIL_CLOSED','quality_passed':passed,
            'end_to_end_cost':'NOT_MEASURED','fixed_candidate_closed':not passed,'seeds':[0,1,2],
            'paired_differences_pp':stats,'seed_results':results,'new_score_budget_calls':12,
            'GPU_calls':0,'training_updates':0,'bootstrap_calls':0,'holdout_accessed':False}


def export(out):
    names=['RESULT.json','STOP.json','CLOSED.json','CONSUMED.json','WAITING.json','ADMISSION.json',
           'WORKER_EXIT.json','WORKER_STARTED.json','SUMMARY.csv','REPORT.md','RESOURCE_START.json','EVALUATION_INPUT_PROOF.json']
    paths=[out/n for n in names if (out/n).is_file()]
    for seed in range(3):
        paths += [p for p in (out/f'seed{seed}').glob('*metrics.json')]
        paths += [p for p in [out/f'seed{seed}/COMPARISON.json',out/f'seed{seed}/PREPARED.json'] if p.is_file()]
    refs=[reference(p) for p in paths]
    require(sum(r['bytes'] for r in refs)<=16*1024**2 and len(refs)<=40,'slim export exceeded')
    publish(out/'SLIM_EXPORT.json',{'status':'READY','files':refs,'total_bytes':sum(r['bytes'] for r in refs),
             'large_predictions_and_full_logs':'SERVER_ONLY','not_full_delivery':True})


def process_inputs(out, specs, rows, binding, progress, parsed_factory, certified_prepared=None):
    import torch
    journal=Journal(out,binding)
    before=torch.get_rng_state().clone()
    journal.emit('started',contract_sha256=binding)
    journal.emit('parents_closed')
    prepared=[]
    for spec in specs:
        dest=out/f'seed{spec["seed"]}';dest.mkdir(exist_ok=False)
        p=(prepare_seed(spec,rows,dest,progress) if certified_prepared is None
           else certified_prepared[spec['seed']])
        publish(dest/'PREPARED.json',p);prepared.append(p)
        journal.emit(f'cache_seed{spec["seed"]}',prepared=reference(dest/'PREPARED.json'))
    publish(out/'ADMISSION.json',{'status':'PASS','all_seeds':[0,1,2],'images_each':len(rows),
                                'raw_SHA_identity_and_A_B_C_exact':True,'scored_before_gate':False})
    journal.emit('all_caches_admitted');parsed=parsed_factory()
    results=[]
    for spec,p in zip(specs,prepared):
        progress(seed=spec['seed'],stage='scoring')
        results.append(evaluate_seed(spec['seed'],p,rows,parsed,spec['comparison'],out/f'seed{spec["seed"]}',journal))
    result=aggregate(results)
    require(not torch.cuda.is_initialized() and torch.equal(before,torch.get_rng_state()),'CPU RNG/CUDA state changed')
    result.update(plan_sha256=sha(PLAN),contract_sha256=binding,finished_epoch=time.time())
    publish(out/'RESULT.json',result)
    with (out/'SUMMARY.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['seed','budget','contrast','metric','difference_pp'])
        for r in results:
            for b,cs in r['differences_pp'].items():
                for contrast,ms in cs.items():
                    for metric,value in ms.items():w.writerow([r['seed'],b,contrast,metric,value])
    (out/'REPORT.md').write_text('# 固定坐标投票 CPU 对照\n\n'+result['status']+
        '\n\n三seed固定比较完成。质量与成本分开；端到端成本未验证。原ABC结果/裁决不改。详见RESULT.json与SUMMARY.csv。'
        '\n\n检测器/训练/GPU/新推理/holdout均为0；预测与完整日志仅服务器。\n',encoding='utf-8')
    journal.emit('complete',result=reference(out/'RESULT.json'))
    closure=Journal.validate(out/'events.jsonl',binding,complete=True)
    publish(out/'CLOSED.json',{'status':'PASS','result':reference(out/'RESULT.json'),'journal':closure,
                            'engineering_only':True,'quality_passed':result['quality_passed']})


def pipeline(c,out):
    if c.get('synthetic_validation') is True:
        from boxvote_entry_fixture import worker as fixture_worker
        return fixture_worker(c,out)
    cpu_only();plan=read(PLAN)
    publish(out/'RESOURCE_START.json',resources(out,plan))
    gate=json.loads(exact(c['admission_gate'],contents=True))
    require(gate['status']=='PASS' and gate['plan_sha256']==sha(PLAN),'admission incomplete')
    parent=json.loads(exact(gate['parent_contract'],contents=True))
    specs,rows,data=context(parent)
    require(digest(specs)==gate['context_sha256'] and digest(rows)==gate['rows_sha256'],'context changed')
    prepared=[]
    for seed,ref in enumerate(gate['prepared']):
        p=json.loads(exact(ref,contents=True))
        require(p['seed']==seed and p['identity']==specs[seed]['identity'] and p['all_A_B_C_exact'], 'prepared identity')
        for arm in ('V-whole','V-five'):
            for ref in p['predictions'][arm].values():exact(ref)
        exact(p['counts']);prepared.append(p)
    publish(out/'EVALUATION_INPUT_PROOF.json',json.loads(exact(gate['input_proof'],contents=True)))
    def progress(**fields):
        resources(out,plan)
        tmp=out/'heartbeat.tmp';tmp.write_bytes(encoded({'time':time.time(),**fields}));os.replace(tmp,out/'heartbeat.json')
    process_inputs(out,specs,rows,digest(c),progress,lambda:parsed_gt(rows,data),prepared)


def load_contract(path):
    c=read(path)
    require(c['task_id']==TASK and c['plan_sha256']==sha(PLAN),'task/plan differs')
    require(read(PLAN)['gates_pp']==LIMITS,'plan gates differ')
    for row in c['source_files']:
        p=Path(row['path']);require(not p.is_absolute() and '..' not in p.parts,'source escapes')
        require((ROOT/p).stat().st_size==row['bytes'] and sha(ROOT/p)==row['sha256'],'source changed: '+str(p))
    require(c['source_root']==str(ROOT) and c['python']==sys.executable and sha(sys.executable)==c['python_sha256'],'runtime differs')
    require(Path('/proc/sys/kernel/random/boot_id').read_text().strip()==c['boot_id'],'boot changed')
    if c.get('synthetic_validation') is True:
        from boxvote_entry_fixture import validate_contract
        validate_contract(c)
    else:
        gate=json.loads(exact(c['admission_gate'],contents=True))
        require(gate['source_files']==c['source_files'] and gate['status']=='PASS','admission source differs')
        cpu=json.loads(exact(c['final_entry_CPU_gate'],contents=True))
        require(cpu['status']=='PASS' and cpu['source_files']=={r['path']:r['sha256'] for r in c['source_files']},'entry CPU source differs')
    return c


def supervise(c,contract_path,out):
    plan=read(PLAN)
    # Reverify after waiting, before any cache read, with no retry of child.
    load_contract(contract_path);resources(out,plan)
    log=(out/'worker.log').open('xb')
    started=time.monotonic()
    child=subprocess.Popen([sys.executable,'-B',str(ROOT/'boxvote_job.py'),'worker','--contract',str(contract_path)],
                           stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',
                           OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),start_new_session=True)
    try:
        while child.poll() is None:
            time.sleep(10);resources(out,plan)
            touched=max([started]+[time.monotonic()-max(0,time.time()-p.stat().st_mtime)
                        for p in [out/'heartbeat.json',out/'events.jsonl'] if p.exists()])
            require(time.monotonic()-touched<plan['progress_timeout_seconds'],'CPU worker no progress')
        require(child.returncode==0,'CPU worker failed; no retry')
        Journal.validate(out/'events.jsonl',digest(c),complete=True)
        require(read(out/'CLOSED.json')['status']=='PASS','worker closure absent')
    except BaseException:
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        raise
    finally:
        log.close()
        publish(out/'WORKER_EXIT.json',{'pid':child.pid,'returncode':child.returncode,
                                      'elapsed_seconds':time.monotonic()-started,'automatic_retry':False})


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','worker'])
    p.add_argument('--contract',required=True);a=p.parse_args(argv)
    c=load_contract(a.contract);out=Path(c['output_root'])
    if a.mode=='run':
        require(not out.exists(),'output already exists; consumed attempt cannot rerun')
        out.mkdir(parents=True)
        publish(out/'CONSUMED.json',{'contract_sha256':digest(c),'time':time.time(),'pid':os.getpid(),'start':proc(os.getpid())})
    else:
        require(read(out/'CONSUMED.json')['contract_sha256']==digest(c),'foreign worker')
        require(not (out/'events.jsonl').exists() and not (out/'STOP.json').exists(),'worker already used')
    try:
        if a.mode=='run':supervise(c,Path(a.contract),out)
        else:
            publish(out/'WORKER_STARTED.json',{'pid':os.getpid(),'parent_pid':os.getppid(),'identity':proc(os.getpid()),'contract_sha256':digest(c),'time':time.time()})
            pipeline(c,out)
        if a.mode=='run':export(out)
        return 0
    except BaseException as e:
        if not (out/'STOP.json').exists():publish(out/'STOP.json',{'status':'STOP_NO_RETRY','error':repr(e),'traceback':traceback.format_exc(),'time':time.time(),'mode':a.mode,'contract_sha256':digest(c)})
        if a.mode=='run' and not (out/'SLIM_EXPORT.json').exists():export(out)
        return 2

if __name__=='__main__':raise SystemExit(main())
