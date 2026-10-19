"""Scoped P12 continuation. Old six-pair launcher and old evidence stay immutable."""
from __future__ import annotations
import argparse
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback

from two_of_four_job import ROOT, PARENT_SHA, read, sha, publish, require, score

TASK = 'sahi-p12-cross-seed-20261019-001'
CONFIG = ROOT / 'configs/p12_cross_seed.json'


def validate_plan(p):
    from two_of_four_core import PROTECTION
    require(p['task_id'] == TASK and p['seeds'] == [1, 2] and p['pair'] == [1, 2]
            and p['views'] == [0, 1, 2] and p['images'] == 548 and p['budget'] == 600, 'P12 scope')
    require(p['gain_retention'] == .8 and p['protection_pp'] == PROTECTION
            and p['protection_references'] == ['C', 'A'], 'eight frozen quality rules')
    require(p['training_updates'] == p['CPU_phase_forward_calls'] == 0
            and p['new_score_groups'] == 2 and p['dual_evaluator_calls'] == 4, 'work budget')
    require(p['complete_both_seeds_before_quality_verdict'] is True
            and p['GPU_timing_requires_both_quality_pass'] is True
            and p['maximum_P12_over_C_seconds'] == .75 and p['automatic_retry'] is False, 'stage policy')


def timing_eligible(results):
    require(list(results) == ['1', '2'], 'both fixed seed results required')
    require(all(type(r['passed']) is bool for r in results.values()), 'quality type')
    return all(r['passed'] for r in results.values())


def verify_source():
    m = read(ROOT / 'RUN_SOURCE.json')
    require(m['experiment_id'] == TASK and m['config_sha256'] == sha(CONFIG), 'source/config task binding')
    for r in m['source_files']:
        p = Path(r['path'])
        require(not p.is_absolute() and '..' not in p.parts and not (ROOT / p).is_symlink()
                and (ROOT / p).stat().st_size == r['bytes'] and sha(ROOT / p) == r['sha256'], 'own source SHA: ' + str(p))
    return m


def dependency(parent):
    # New adapter explicitly authenticates both seed specs. Old seed0 adapter is not relaxed.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ[k] = '1'
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    require(sha(parent) == PARENT_SHA, 'parent contract SHA')
    c = read(parent); root = Path(c['source_root'])
    for r in c['source_files']:
        p = Path(r['path'])
        require(not p.is_absolute() and '..' not in p.parts and sha(root / p) == r['sha256'], 'parent source SHA')
    require(sha(root / 'slice_core.py') == sha(ROOT / 'slice_core.py'), 'original merge source differs')
    sys.path.insert(0, str(root))
    import softnms_job as j
    torch = j.cpu_only(); c = j.load_contract(parent)
    specs, rows, data = j.context(c)
    require([s['seed'] for s in specs] == [0, 1, 2] and len(rows) == 548, 'parent seed/member scope')
    specs = [s for s in specs if s['seed'] in (1, 2)]
    require(all(s['ids'] == [r['stable_image_id'] for r in rows]
                and len(s['full']['raw_view_refs']) == 137 for s in specs), 'member order/raw budget')
    return j, torch, specs, rows, j.parsed_gt(rows, data)


def prepare(parent, plan, out):
    from slice_core import classwise_nms
    from two_of_four_core import merge_pair
    j, torch, specs, rows, parsed = dependency(parent)
    rng = torch.get_rng_state().clone()
    resources = [j.resources(out, plan)]
    inputs, rebuilt, prepared = {}, {}, {}
    for spec in specs:
        seed = spec['seed']; key = str(seed)
        baseline, expected, proofs = {}, {}, []
        for arm in ('A', 'B', 'C'):
            ps = json.loads(j.exact(spec['evaluation']['predictions'][arm]['600'], contents=True))
            proofs.append(j.evaluation_input_proof(spec, rows, parsed, ps, arm, 600))
            expected[arm] = j.group(ps, spec['ids'], 600)
            baseline[arm] = spec['metrics'][arm]['600']['COCO_bridge']['metrics']
        inputs[key] = proofs
        predictions = []; counts = dict(A=0, B=0, C=0)
        for batch, ref in enumerate(spec['full']['raw_view_refs']):
            resources.append(j.resources(out, plan))
            raw = torch.load(io.BytesIO(j.exact(ref, contents=True)), map_location='cpu', weights_only=True)
            part = rows[batch * 4:batch * 4 + 4]
            pools = j.decode_raw(raw, part, spec['identity'], seed)
            require(len(pools) == len(part), 'decoded image count')
            for row, pool in zip(part, pools):
                image = row['stable_image_id']; whole = [p for p in pool if p.view_id == 0]
                for arm, items in (('A', sorted(whole, key=lambda p: p.sort_key)),
                                   ('B', classwise_nms(whole)), ('C', classwise_nms(pool))):
                    ps = [j.prediction(image, p, whole=arm == 'A') for p in items[:600]]
                    require(ps == expected[arm][image], f'seed{seed} original {arm} exact rebuild differs')
                    counts[arm] += 1
                predictions.extend(j.prediction(image, p) for p in merge_pair(pool, (1, 2)))
            print(json.dumps(dict(event='cache_rebuilt', seed=seed, batch=batch + 1, batches=137)), flush=True)
        require(counts == dict(A=548, B=548, C=548), 'rebuild scope')
        pref = publish(out / f'seed{seed}-P12-predictions.json', predictions)
        rebuilt[key] = dict(status='PASS', original_exact_images=counts, raw_refs=spec['full']['raw_view_refs'])
        prepared[key] = dict(seed=seed, identity=spec['identity'], predictions=pref, baseline=baseline,
                             original_metric_refs=spec['evaluation']['metrics'])
    require(torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized(), 'CPU RNG/CUDA changed')
    publish(out / 'INPUT_PROOFS.json', dict(status='PASS', seeds=inputs))
    publish(out / 'REBUILD.json', dict(status='PASS', seeds=rebuilt))
    publish(out / 'RESOURCE.json', resources)
    return dict(status='PREPARED_PASS', seeds=prepared, images=548, raw_batches=274,
                parent_sha256=sha(parent), source_manifest_sha256=sha(ROOT / 'RUN_SOURCE.json'),
                scores_executed=0, forward_calls=0, training_updates=0)


def evaluate(prepared, rows, parsed, repo, out, *, proof=None, inject_failure=False, resource=None):
    from two_of_four_core import screen
    require(list(prepared) == ['1', '2'], 'all fixed seeds before score')
    results = {}; refs = {}
    for key, p in prepared.items():
        require(p['seed'] == int(key), 'cross-seed prepared identity')
        if resource: resource()
        print(json.dumps(dict(event='score_started', seed=int(key), arm='P12', budget=600)), flush=True)
        if inject_failure: raise RuntimeError('injected worker failure')
        ps = read(p['predictions']['path'])
        require(sha(p['predictions']['path']) == p['predictions']['sha256'], 'prediction SHA')
        record = score(rows, parsed, ps, repo)
        if proof: record['input_proof'] = proof(rows, parsed, ps, record)
        refs[key] = publish(out / f'seed{key}-P12-metrics.json', record)
        metrics = record['COCO_bridge']['metrics']
        results[key] = dict(seed=int(key), metrics=metrics, baseline=p['baseline'], official=record['official'],
                            spatial_filter_counts=record['spatial_filter_counts'], **screen(metrics, p['baseline']))
        print(json.dumps(dict(event='score_complete', seed=int(key), metric=refs[key])), flush=True)
    eligible = timing_eligible(results)
    publish(out / 'SUMMARY.json', dict(results=results, both_seeds_complete=True,
            GPU_timing_eligible=eligible, cost_status='NOT_MEASURED',
            scope='same development cross-training-seed repetition; seed0 selected previously'))
    return dict(status='QUALITY_PASS_TIMING_PENDING' if eligible else 'QUALITY_FAIL_CLOSED_NO_GPU',
                seeds_complete=[1, 2], metric_refs=refs, GPU_timing_eligible=eligible,
                score_groups=2, evaluator_calls=4, forward_calls=0, training_updates=0)


def fixture(out, fail):
    sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
    from slice_core import Candidate
    from two_of_four_core import merge_pair
    from sparse_rtdetr.localization.data import parse_raw
    rows = [dict(stable_image_id=f'fixture{i}', width=1000, height=1000) for i in range(5)]
    raw = b'10,10,10,10,1,1,0,0\n100,100,50,50,1,1,0,0\n300,300,100,100,1,1,0,0\n'
    parsed = [parse_raw(raw, r['stable_image_id'], expected_rows=3) for r in rows]
    prepared = {}
    for seed in (1, 2):
        pool = [Candidate(box, .9, v, q, 1, q+1) for v in range(5)
                for q, box in enumerate(((10,10,20,20),(100,100,150,150),(300,300,400,400)))]
        ps = [dict(image_id=r['stable_image_id'], category_id=p.category_id, score=p.score, bbox_xyxy=list(p.box_xyxy))
              for r in rows for p in merge_pair(pool, (1, 2))]
        base = {a: dict(AP=.2, APS=.2, AP75_small=.2, APM=.2, APL=.2) for a in ('A', 'B', 'C')}
        base['C']['AP'] = base['C']['APS'] = .3
        prepared[str(seed)] = dict(seed=seed, baseline=base,
                predictions=publish(out / f'fixture-seed{seed}.json', ps))
    return evaluate(prepared, rows, parsed, ROOT, out, inject_failure=fail)


def close(out, result):
    ref = publish(out / 'RESULT.json', result)
    publish(out / 'CLOSED.json', dict(result=ref, returncode=0))
    names = ['RESULT.json', 'CLOSED.json', 'WORKER_STARTED.json', 'INPUT_PROOFS.json', 'REBUILD.json',
             'RESOURCE.json', 'SUMMARY.json', 'COST.json', 'seed1-P12-metrics.json', 'seed2-P12-metrics.json']
    files = [dict(path=str(out / n), bytes=(out / n).stat().st_size, sha256=sha(out / n))
             for n in names if (out / n).is_file()]
    require(sum(r['bytes'] for r in files) < 8 * 1024 ** 2, 'slim cap')
    publish(out / 'SLIM_EXPORT.json', dict(status='READY', files=files, predictions='SERVER_ONLY'))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('prepare', 'run', 'fixture'))
    p.add_argument('--parent', type=Path); p.add_argument('--contract', type=Path)
    p.add_argument('--output', required=True, type=Path); p.add_argument('--inject-failure', action='store_true')
    a = p.parse_args(argv)
    if a.mode != 'fixture' and (a.parent is None or a.inject_failure or TASK not in a.output.resolve().parts):
        p.error('production scope')
    if a.mode == 'run' and a.contract is None: p.error('contract required')
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    publish(out / 'WORKER_STARTED.json', dict(pid=os.getpid(), time=time.time(), mode=a.mode))
    try:
        plan = read(CONFIG); validate_plan(plan)
        if a.mode == 'fixture': result = fixture(out, a.inject_failure)
        else:
            m = verify_source()
            if a.mode == 'prepare': result = prepare(a.parent, plan, out)
            else:
                c = read(a.contract)
                require(c['task_id'] == TASK and c['commit'] == m['commit']
                        and c['source_manifest_sha256'] == sha(ROOT / 'RUN_SOURCE.json')
                        and c['plan_sha256'] == sha(CONFIG) and sha(a.parent) == PARENT_SHA, 'contract binding')
                j, torch, specs, rows, parsed = dependency(a.parent)
                gate = json.loads(j.exact(c['CPU_GATE'], contents=True))
                require(gate['status'] == 'PASS' and gate['skips'] == 0 and gate['commit'] == m['commit'], 'CPU gate')
                prepared = json.loads(j.exact(c['prepared'], contents=True))
                require(prepared['status'] == 'PREPARED_PASS' and prepared['source_manifest_sha256'] == sha(ROOT / 'RUN_SOURCE.json'), 'prepared binding')
                for spec in specs:
                    q = prepared['seeds'][str(spec['seed'])]
                    require(q['identity'] == spec['identity'], 'prepared checkpoint/source identity')
                    j.exact(q['predictions'])
                for name, ref in c['preparation_proofs'].items():
                    require(json.loads(j.exact(ref, contents=True))['status'] == 'PASS', 'preparation proof ' + name)
                token = Path(c['token']); require(read(token)['task_id'] == TASK, 'token task')
                publish(token.with_name(token.name + '.consumed'), dict(contract_sha256=sha(a.contract), output=str(out), pid=os.getpid()))
                resources = [j.resources(out, plan)]
                result = evaluate(prepared['seeds'], rows, parsed, j.ROOT, out,
                                  proof=j.evaluation_count_proof, resource=lambda: resources.append(j.resources(out, plan)))
                publish(out / 'RESOURCE.json', resources)
                result['execution'] = dict(commit=m['commit'], contract_sha256=sha(a.contract),
                            run_source_sha256=sha(ROOT / 'RUN_SOURCE.json'), plan_sha256=sha(CONFIG))
        result['elapsed_seconds'] = time.monotonic() - start
        publish(out / 'COST.json', dict(elapsed_seconds=result['elapsed_seconds'], forward_calls=0,
                    training_updates=0, score_groups=result.get('score_groups', 0), evaluator_calls=result.get('evaluator_calls', 0),
                    prior_costs='preserved; this stage only'))
        close(out, result)
        print(json.dumps(dict(status=result['status'], elapsed_seconds=result['elapsed_seconds'])), flush=True)
        return 0
    except Exception:
        publish(out / 'STOP.json', dict(traceback=traceback.format_exc(), elapsed_seconds=time.monotonic()-start, automatic_retry=False))
        traceback.print_exc(); return 2


if __name__ == '__main__': sys.exit(main())
