"""PR26 fixed 24-epoch chain. Production requires a separate one-shot contract."""
import argparse
import gc
import json
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import sys
import time
import traceback
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
for k, v in {'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
             'OPENBLAS_NUM_THREADS': '2', 'MKL_THREADING_LAYER': 'GNU', 'PYTHONNOUSERSITE': '1'}.items():
    os.environ[k] = v
from sparse_rtdetr.research import sabl_formal_contract as ct
from sabl_acceptance import Telemetry
from r18_deploy import verify_release
need, read, sha, digest, reference = ct.need, ct.read, ct.sha, ct.digest, ct.reference


def write(path, obj):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())


def atomic(path, obj):
    p = Path(path); q = p.with_suffix('.tmp')
    with q.open('w', encoding='utf-8') as f:
        json.dump(obj, f, allow_nan=False); f.flush(); os.fsync(f.fileno())
    os.replace(q, p)
    from sparse_rtdetr.protocol_r18.runio import fsync_directory
    fsync_directory(p.parent)


def plan(a):
    c = {'schema': ct.SCOPE, 'mode': ct.CPU if a.synthetic else ct.GPU, 'policy': ct.POLICY,
         'caps': ct.CAPS, 'prior': ct.PRIOR, 'arms': list(ct.ARMS), 'N_checkpoint_SHA': ct.N_SHA,
         'manifest_sha256': a.manifest_sha, 'attempt_root': str(Path(a.attempt_root).resolve()),
         'metadata': None, 'acceptance_cost': None, 'gpu_uuid': a.gpu_uuid,
         'fixture': {'fault': a.fault} if a.synthetic else None}
    if not a.synthetic:
        need(a.metadata and a.acceptance_cost and a.gpu_uuid, 'production records/UUID required')
        old = read(a.metadata)
        need(sha(a.metadata) == 'cbf9fab7c6a9de54293716091064715e27bff5b5905b624fade5cc1fe0f12d88', 'registered parent metadata SHA')
        n = ct.exact(old['historical_N']['endpoint'])
        need(n['checkpoint']['sha256'] == ct.N_SHA, 'registered N parent')
        history = read(ROOT / 'docs/research/r18_sabl_control/HISTORICAL_N.json')['historical_N']
        need(old['historical_N'] == history, 'historical N metadata identity')
        c['metadata'] = {'parent_record': reference(a.metadata), 'checkpoint': n['checkpoint'],
                         'data': old['data'], 'selection': old['selection'], 'python': old['python'],
                         'environment': old['environment'], 'historical_N': history}
        c['acceptance_cost'] = reference(a.acceptance_cost)
        ct.verify_acceptance(c['acceptance_cost'])
        c['metadata']['prior_storage_bytes'] = ct.accepted_storage(c['acceptance_cost'])
        c['metadata']['fixed_ids'] = [r['stable_image_id'] for r in ct.exact(old['selection'])['selected']]
    c['planned_new_forwards'] = ct.new_forwards(c)
    ct.validate(c); write(a.out, c)


class Meter:
    """Durable reservations before calls, no rollback and no budget resets."""
    def __init__(self, c, out, start, cuda):
        self.c, self.out, self.start, self.cuda = c, out, start, cuda
        self.telemetry = Telemetry(cuda)
        self.stage = 'admission'; self.stage_count = 0
        self.ledger = {'inherited': dict(ct.PRIOR), 'new_forward': 0, 'actual_updates': 0,
                       'reserved_updates': 0, 'stage_forwards': {}, 'total_forward': 18,
                       'GPU_stage_seconds': ct.PRIOR['GPU_stage_seconds'], 'CPU_fixture': c['mode'] == ct.CPU}
        self.limit = c['planned_new_forwards']
        if c['mode'] == ct.CPU and c['fixture']['fault'] == 'forward_exhausted':
            self.limit = 1
        self.publish()

    def elapsed(self):
        return time.monotonic() - self.start

    def publish(self):
        self.ledger['GPU_stage_seconds'] = ct.PRIOR['GPU_stage_seconds'] + (0 if self.c['mode'] == ct.CPU else self.elapsed())
        self.ledger['CPU_fixture_seconds'] = self.elapsed() if self.c['mode'] == ct.CPU else None
        atomic(self.out / 'ledger.json', self.ledger)

    def guard(self):
        elapsed = self.elapsed()
        need(elapsed < (300 if self.c['mode'] == ct.CPU else ct.CAPS['GPU_stage_seconds'] - ct.PRIOR['GPU_stage_seconds']), 'cumulative GPU/CPU time cap')
        need(elapsed < ct.CAPS['wall_seconds'], 'cumulative wall cap')
        prior_storage = 0 if self.c['mode'] == ct.CPU else self.c['metadata']['prior_storage_bytes']
        need(prior_storage + sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file()) <= ct.CAPS['storage_bytes'], 'cumulative task storage cap')
        if self.cuda:
            need(self.telemetry.observe_peak() <= ct.CAPS['allocated_bytes'], 'cumulative allocated cap')

    def begin(self, stage):
        self.guard(); self.stage = stage
        self.stage_count = self.ledger['new_forward']
        self.telemetry.begin(stage)

    def finish(self):
        self.guard(); self.telemetry.finish()
        self.ledger['stage_forwards'][self.stage] = self.ledger['new_forward'] - self.stage_count
        self.publish()

    def forward(self, *_):
        self.guard()
        need(self.ledger['new_forward'] < self.limit, 'forward hard cap')
        self.ledger['new_forward'] += 1
        self.ledger['total_forward'] = 18 + self.ledger['new_forward']
        need(self.ledger['total_forward'] <= ct.CAPS['detector_forward'], 'inherited forward hard cap')
        self.publish()

    def update(self, done=False):
        key = 'actual_updates' if done else 'reserved_updates'
        self.ledger[key] += 1
        need(self.ledger[key] <= 3 * ct.profile(self.c)['epochs'] * ct.profile(self.c)['windows'], 'update hard cap')
        self.publish()


def seed(device):
    import numpy as np
    import torch
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    if device == 'cuda':
        torch.cuda.manual_seed_all(0)


def fresh(c, arm, device):
    from sparse_rtdetr.research.sabl_control import make_engine, VERSION
    from sparse_rtdetr.protocol_r18.engine import tensor_digest
    seed(device)
    e = make_engine(arm, c['mode'] == ct.CPU, None if c['mode'] == ct.CPU else c['metadata']['checkpoint']['path'], device)
    need(not e.optimizer.state and e.completed_updates == e.ema.updates == 0, 'fresh formal optimizer/EMA')
    b = {'scope': c['mode'], 'arm': arm, 'version': VERSION, 'contract': digest(c),
         'release_manifest': c['manifest_sha256'], 'N_SHA': ct.N_SHA}
    shared = tensor_digest(e.model.native.state_dict().items())
    head = None if arm == 'N' else tensor_digest(e.model.boundary.state_dict().items())
    seed(device)
    return e, b, shared, head


def stage_close(folder, result, c):
    write(folder / 'RESULT.json', result)
    write(folder / 'CLOSED.json', {'status': 'PASS_ENGINEERING', 'contract': digest(c), 'result': reference(folder / 'RESULT.json')})


def closed(folder, c):
    meta = read(folder / 'CLOSED.json')
    need(meta['status'] == 'PASS_ENGINEERING' and meta['contract'] == digest(c), 'stage closure binding')
    return ct.exact(meta['result'])


def train(c, arm, first, out, meter, device):
    import torch
    from sparse_rtdetr.research import sabl_formal_state as state, sabl_formal_data as data
    from sparse_rtdetr.protocol_r18.engine import tensor_digest
    from sparse_rtdetr.baseline.training_v2b_checkpoint import _equal
    profile = ct.profile(c); folder = out / ('train-' + arm); folder.mkdir(exist_ok=not first)
    stage = ('engineering-' if first else 'remaining-') + arm
    meter.begin(stage)
    e, binding, shared, head = fresh(c, arm, device)
    if first:
        write(folder / 'INITIAL.json', {'native_SHA': shared, 'head_SHA': head, 'optimizer_entries': 0, 'EMA_updates': 0,
                                       'acceptance_state_used': False, 'parent': ct.N_SHA, 'RNG': 'independently_reset_seed0'})
        state.save(e, binding, folder / 'checkpoint-e000.pt')
        input_hash = __import__('hashlib').sha256()
        start_epoch = 1; end_epoch = profile['engineering_epochs']
    else:
        gate = read(folder / 'ENGINEERING_PASS.json')
        need(gate['arm'] == arm and gate['epochs'] == 2 and gate['contract'] == digest(c), 'engineering gate binding')
        ref = gate['checkpoint']; audit = state.restore(e, binding, ref['path'], ref['sha256'])
        need(audit['full_state_exact'] and e.completed_updates == 2 * profile['windows'], 'cross-phase full restoration')
        # Continue the identity hash by hashing the ordered per-epoch digests at close.
        start_epoch = 3; end_epoch = profile['epochs']
    e.model.register_forward_pre_hook(meter.forward)
    def actual_dtype(module, args, value):
        need(str(value['pred_logits'].dtype) == ('torch.float32' if c['mode'] == ct.CPU else 'torch.bfloat16'), 'training precision')
        need(module.last_boundary_dtype == (None if arm == 'N' else 'torch.float32'), 'boundary precision')
    e.model.register_forward_hook(actual_dtype)
    ds = data.Fixture() if c['mode'] == ct.CPU else data.Dataset(c['metadata']['data']['train_core'])
    rows = []
    gradient_evidence = {'head': arm == 'N', 'F8': arm == 'N', 'backbone': False}
    handles = []
    if first:
        def observe(name):
            def f(g):
                # One witnessed nonzero path is enough; native Engine.health
                # still checks all gradients at every update. Avoid recurring
                # per-parameter CUDA synchronizations from an observation hook.
                if not gradient_evidence[name]:
                    gradient_evidence[name] = bool(g.isfinite().all() and (g != 0).any())
            return f
        for p in e.model.backbone.parameters():
            if p.requires_grad:
                handles.append(p.register_hook(observe('backbone')))
        if arm != 'N':
            for p in e.model.boundary.parameters():
                handles.append(p.register_hook(observe('head')))
            def feature(module, args, value):
                value[0].register_hook(observe('F8'))
            handles.append(e.model.encoder.register_forward_hook(feature))
    try:
        with (folder / ('engineering-events.jsonl' if first else 'remaining-events.jsonl')).open('x') as events:
            for epoch in range(start_epoch, end_epoch + 1):
                need(e.completed_updates == (epoch - 1) * profile['windows'], 'cross-epoch update cursor')
                began = time.monotonic(); epoch_inputs = __import__('hashlib').sha256(); updates = 0; update_seconds = []
                iterable = [ds.window(epoch)] if c['mode'] == ct.CPU else data.windows(ds, epoch)
                for images, targets, identity in iterable:
                    meter.update()
                    if c['mode'] == ct.CPU and c['fixture']['fault'] == 'nonfinite':
                        images[0, 0, 0, 0] = float('nan')
                    r = e.train_window(images, targets, capture=c['mode'] == ct.CPU)
                    meter.update(done=True); updates += 1; update_seconds.append(r['elapsed_seconds'])
                    need(r['finite'] and e.completed_updates == (epoch - 1) * profile['windows'] + updates, 'update health/cursor')
                    epoch_inputs.update(digest(identity).encode())
                    # Stream only bounded per-update records; never load the full log.
                    events.write(json.dumps({'epoch': epoch, 'update': e.completed_updates, 'input': identity, 'result': r}, allow_nan=False) + '\n'); events.flush()
                    atomic(folder / 'progress.json', {'epoch': epoch, 'update': e.completed_updates, 'maximum': profile['epochs'] * profile['windows']})
                    meter.guard()
                need(updates == profile['windows'], 'incomplete epoch')
                row = {'epoch': epoch, 'updates': updates, 'input_SHA': epoch_inputs.hexdigest(),
                       'stage_seconds': time.monotonic() - began, 'update_seconds_sum': sum(update_seconds),
                       'update_seconds_min': min(update_seconds), 'update_seconds_max': max(update_seconds)}
                rows.append(row); write(folder / ('epoch-%03d.json' % epoch), row)
        for h in handles:
            h.remove()
        need(e.completed_updates == e.ema.updates == end_epoch * profile['windows'], 'EMA/endpoint clock')
        final = state.save(e, binding, folder / ('checkpoint-e%03d.pt' % end_epoch))
        if first:
            need(all(gradient_evidence.values()), 'engineering gradient coverage')
            write(folder / 'ENGINEERING_PASS.json', {'arm': arm, 'epochs': 2, 'contract': digest(c),
                    'checkpoint': final, 'gradient_evidence': gradient_evidence, 'no_intermediate_AP': True})
        else:
            # Full-state read-back through the same strict restore primitive, no forward.
            audit = state.restore(e, binding, final['path'], final['sha256'])
            need(audit['full_state_exact'], 'endpoint full readback')
            ep = {'arm': arm, 'contract': digest(c), 'manifest': c['manifest_sha256'], 'checkpoint': final,
                  'binding': binding, 'epochs': end_epoch, 'updates': e.completed_updates, 'EMA': True,
                  'parent_N_SHA': ct.N_SHA, 'acceptance_state_used': False, 'synthetic': c['mode'] == ct.CPU}
            write(folder / 'endpoint.json', ep)
            all_rows = [read(folder / ('epoch-%03d.json' % i)) for i in range(1, end_epoch + 1)]
            result = {'status': 'PASS', 'endpoint': reference(folder / 'endpoint.json'),
                      'training_input_SHA': digest([x['input_SHA'] for x in all_rows]),
                      'epochs': end_epoch, 'updates': e.completed_updates, 'initial': read(folder / 'INITIAL.json')}
            stage_close(folder, result, c)
    finally:
        del e; gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
    meter.finish()


def endpoint(c, arm, out, device):
    import torch
    from sparse_rtdetr.research import sabl_formal_state as state
    folder = out / ('train-' + arm)
    result = closed(folder, c); ep = ct.exact(result['endpoint'])
    e, binding, _, _ = fresh(c, arm, device)
    need(ep['arm'] == arm and ep['binding'] == binding and ep['contract'] == digest(c) and ep['manifest'] == c['manifest_sha256'], 'endpoint identity')
    need(ep['epochs'] == ct.profile(c)['epochs'] and ep['updates'] == ct.profile(c)['epochs'] * ct.profile(c)['windows'] and ep['EMA'] and not ep['acceptance_state_used'], 'fixed e24 endpoint provenance')
    state.restore(e, binding, ep['checkpoint']['path'], ep['checkpoint']['sha256'])
    model = e.ema.module.eval().requires_grad_(False)
    need(e.ema.updates == ep['updates'], 'endpoint EMA updates')
    del e
    return model, result['endpoint']


def predictions(c, model, rows, reader, device, meter, *, cost_only=False):
    import torch
    from sparse_rtdetr.research.sabl_evaluation import rank_refined
    from sparse_rtdetr.r0.evaluation import rank_candidates
    from sparse_rtdetr.protocol_r18.geometry import forward_guards
    pred = {600: []} if cost_only else {600: [], 300: []}; raw = []; invalid = 0
    with torch.inference_mode(), forward_guards(model):
        for start in range(0, len(rows), 4):
            batch = rows[start:start + 4]
            images = torch.stack([reader.item(row, start + i)[0] for i, row in enumerate(batch)]).to(device)
            meter.forward(); value = model(images); meter.guard()
            need(value['pred_logits'].dtype == torch.float32, 'endpoint/cost FP32')
            logits = value['pred_logits'].float().cpu(); boxes = value['pred_boxes'].float().cpu()
            valid = value.get('refined_valid', torch.ones(boxes.shape[:2], dtype=torch.bool, device=boxes.device)).cpu()
            invalid += int((~valid).sum())
            if not cost_only:
                raw.append({'ids': [r['stable_image_id'] for r in batch], 'logits': logits, 'boxes': boxes, 'valid': valid})
            for i, row in enumerate(batch):
                for budget in pred:
                    pred[budget] += (rank_candidates(logits[i], boxes[i], row, budget) if model.arm == 'N'
                                     else rank_refined(logits[i], boxes[i], valid[i], row, budget))
    return pred, raw, invalid


def infer(c, arm, out, meter, device):
    import torch
    from sparse_rtdetr.research.sabl_formal_data import development
    folder = out / ('infer-' + arm); folder.mkdir(); meter.begin('infer-' + arm)
    model, ep = endpoint(c, arm, out, device); rows, reader = development(c)
    pred, raw, invalid = predictions(c, model, rows, reader, device, meter)
    need([i for b in raw for i in b['ids']] == [r['stable_image_id'] for r in rows], 'evaluation input coverage/order')
    cache = folder / 'raw-query-cache.pt'; torch.save(raw, cache)
    refs = {}
    for budget in pred:
        path = folder / ('predictions-%d.json' % budget); write(path, pred[budget]); refs[str(budget)] = reference(path)
    stage_close(folder, {'status': 'PASS', 'arm': arm, 'endpoint': ep, 'predictions': refs,
                         'image_ids': [r['stable_image_id'] for r in rows], 'raw_cache': reference(cache),
                         'invalid_ordinary_queries': invalid, 'FP32': True, 'forward_calls': (len(rows) + 3) // 4}, c)
    del model, raw; gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    meter.finish()


def score(c, out, meter):
    from sparse_rtdetr.research.sabl_formal_data import development
    from sparse_rtdetr.r0.evaluation import score_budget
    folder = out / 'score'; folder.mkdir(); meter.begin('score')
    rows, reader = development(c); truth = [reader.raw(row) for row in rows]; refs = {}
    for arm in ct.ARMS:
        source = closed(out / ('infer-' + arm), c)
        need(source['image_ids'] == [r['stable_image_id'] for r in rows], 'score data ID binding')
        need(ct.exact(source['endpoint']) == ct.exact(closed(out / ('train-' + arm), c)['endpoint']), 'score endpoint weight binding')
        refs[arm + '24'] = {}
        allowed = set(source['image_ids'])
        for budget in ('600', '300'):
            pred = ct.exact(source['predictions'][budget])
            need(all(p['image_id'] in allowed for p in pred), 'unknown prediction ID')
            need(all(n <= int(budget) for n in Counter(p['image_id'] for p in pred).values()), 'post-truncation budget')
            result = score_budget(rows, truth, pred, str(ROOT))
            result['PR26_binding'] = {'contract': digest(c), 'endpoint': source['endpoint'], 'arm': arm,
                                     'image_ids_SHA': digest(source['image_ids']), 'budget': int(budget),
                                     'predictions': source['predictions'][budget], 'invalid_ordinary_queries': source['invalid_ordinary_queries']}
            path = folder / (arm + '-' + budget + '.json'); write(path, result); refs[arm + '24'][budget] = reference(path)
            meter.guard()
    if c['mode'] == ct.CPU:
        # An explicit synthetic historical row tests table semantics, not N quality.
        refs['N_original'] = dict(refs['N24'])
    else:
        refs['N_original'] = c['metadata']['historical_N']['scores']
        for r in refs['N_original'].values():
            ct.exact(r)
    stage_close(folder, {'status': 'PASS', 'scores': refs, 'new_score_groups': 6,
                         'evaluator_calls': 12, 'original_N_recomputed': False, 'forward_calls': 0}, c)
    meter.finish()


def cost(c, out, meter, device):
    import torch
    from sparse_rtdetr.research.sabl_formal_data import development
    from sparse_rtdetr.r0.evaluation import bridge_inputs
    folder = out / 'cost'; folder.mkdir(); meter.begin('cost')
    rows, reader = development(c); by_id = {r['stable_image_id']: r for r in rows}
    ids = [r['stable_image_id'] for r in rows[:4]] if c['mode'] == ct.CPU else c['metadata']['fixed_ids']
    need(len(ids) == ct.profile(c)['fixed'] and len(set(ids)) == len(ids) and set(ids) <= set(by_id), 'fixed cost members')
    fixed = [by_id[k] for k in ids]; truth = [reader.raw(r) for r in fixed]
    measured = {a: [] for a in ct.ARMS}; warmups = []
    for round_id, order in enumerate([ct.ARMS, tuple(reversed(ct.ARMS)), *ct.COST_ORDERS]):
        for arm in order:
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
            model, ep = endpoint(c, arm, out, device)
            if device == 'cuda':
                torch.cuda.synchronize(); meter.telemetry.observe_peak(); torch.cuda.reset_peak_memory_stats()
            began = time.perf_counter()
            pred, raw, invalid = predictions(c, model, fixed, reader, device, meter, cost_only=True)
            _, _, counts = bridge_inputs(fixed, truth, pred[600])
            if device == 'cuda':
                torch.cuda.synchronize(); meter.telemetry.observe_peak()
            measurement = {'arm': arm, 'round': round_id, 'seconds': time.perf_counter() - began,
                           'images': len(fixed), 'dtype': 'FP32', 'batch': 4, 'endpoint': ep,
                           'peak_allocated_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else None,
                           'peak_reserved_bytes': torch.cuda.max_memory_reserved() if device == 'cuda' else None,
                           'filter_counts': counts, 'invalid_ordinary_queries': invalid}
            if round_id < 2:
                warmups.append(measurement)
            else:
                measured[arm].append(measurement)
            del model, raw; gc.collect()
    summary = {a: {'mean_seconds': statistics.mean(x['seconds'] for x in values),
                   'sample_SD_seconds': statistics.stdev(x['seconds'] for x in values),
                   'peak_allocated_bytes': max(x['peak_allocated_bytes'] for x in values) if device == 'cuda' else None,
                   'peak_reserved_bytes': max(x['peak_reserved_bytes'] for x in values) if device == 'cuda' else None}
               for a, values in measured.items()}
    stage_close(folder, {'status': 'PASS', 'scope': 'CPU_SYNTHETIC_NOT_COST_EVIDENCE' if c['mode'] == ct.CPU else 'FIXED64_FP32_BATCH4_E2E',
                         'orders': ct.COST_ORDERS, 'warmups': warmups, 'measurements': measured, 'summary': summary,
                         'fixed_ids': ids, 'boundary': 'decode_resize_H2D_forward_rank_raw_ignore_sync; excludes model load/GT parsing'}, c)
    meter.finish()


def compare(c, out, meter):
    from sabl_results import metrics, render
    meter.begin('compare'); folder = out / 'compare'; folder.mkdir()
    scores = closed(out / 'score', c)['scores']
    rows = {a: {b: metrics(ct.exact(ref)) for b, ref in v.items()} for a, v in scores.items()}
    result, table = render(rows)
    inputs = [closed(out / ('train-' + a), c)['training_input_SHA'] for a in ct.ARMS]
    need(len(set(inputs)) == 1, 'three-arm ordered input/GT/augmentation mismatch')
    measurements = closed(out / 'cost', c)
    s, n = measurements['summary']['S'], measurements['summary']['N']
    if c['mode'] == ct.CPU:
        costs = {'status': 'NOT_SCIENTIFIC', 'E2E_S_over_N': None, 'allocated_S_over_N': None, 'passed': None}
    else:
        ratios = {'E2E_S_over_N': s['mean_seconds'] / n['mean_seconds'],
                  'allocated_S_over_N': s['peak_allocated_bytes'] / n['peak_allocated_bytes']}
        costs = {**ratios, 'thresholds': {'E2E_S_over_N': 1.30, 'allocated_S_over_N': 1.20},
                 'passed': ratios['E2E_S_over_N'] <= 1.30 and ratios['allocated_S_over_N'] <= 1.20}
    result.update(scope='SYNTHETIC_NOT_SCIENTIFIC' if c['mode'] == ct.CPU else 'FORMAL_FIXED_E24',
                  cost=costs, scientific_status='NOT_SCIENTIFIC' if c['mode'] == ct.CPU else
                  ('PASS_SINGLE_SEED_SHORT_SCHEDULE' if result['quality_controls_pass'] and costs['passed'] else 'FAIL_FIXED_RECIPE_END'),
                  original_N_status='AUTHENTICATED_OUTPUT_NOT_ORIGINAL_STAGE_CLOSURE', no_followup=True)
    write(folder / 'COMPARISON.json', result)
    (folder / 'TABLE.md').write_text('# ' + result['scope'] + '\n\n' + table, encoding='utf-8')
    stage_close(folder, {'status': 'PASS', 'comparison': reference(folder / 'COMPARISON.json'), 'table': reference(folder / 'TABLE.md')}, c)
    meter.finish()


def admit(c, a, out):
    ct.validate(c); need(str(out.resolve()) == c['attempt_root'], 'attempt path binding')
    release = verify_release(ROOT, c['manifest_sha256'])
    if c['mode'] == ct.CPU:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
        return 'cpu'
    need(not release['candidate_not_for_real_execution'], 'candidate cannot execute production')
    need(read(ROOT / 'RELEASE_MANIFEST.json')['scope'] == 'PR26 formal24; separate authorization required; no automatic launch', 'formal release scope')
    need(a.authorization and read(a.authorization) == {'contract': digest(c), 'scope': ct.SCOPE, 'authorize_once': True}, 'formal authorization missing/mismatch')
    ct.verify_acceptance(c['acceptance_cost'])
    os.environ['CUDA_VISIBLE_DEVICES'] = c['gpu_uuid']
    from sabl_formal_preflight import material_admission, resources
    write(out / 'MATERIAL_ADMISSION.json', material_admission(c, out))
    write(out / 'RESOURCE_ADMISSION.json', resources(c, out))
    return 'cuda'


def run(a):
    out = Path(a.out); out.mkdir(); start = time.monotonic(); meter = None
    try:
        c = read(a.contract); ct.validate(c)
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('cumulative chain deadline')))
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(RuntimeError('external termination; no retry')))
        signal.signal(signal.SIGINT, lambda *_: (_ for _ in ()).throw(RuntimeError('external interrupt; no retry')))
        signal.alarm(300 if c['mode'] == ct.CPU else int(ct.CAPS['GPU_stage_seconds'] - ct.PRIOR['GPU_stage_seconds']))
        device = admit(c, a, out)
        import torch, torchvision
        need(str(torch.__version__).startswith('2.4.1') and str(torchvision.__version__).startswith('0.19.1'), 'fixed runtime')
        torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        if device == 'cuda':
            need(torch.cuda.is_available() and torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported(), 'CUDA/BF16 identity')
            expected = c['metadata']['environment']
            need(str(torch.__version__) == expected['torch'] and str(torchvision.__version__) == expected['torchvision']
                 and torch.version.cuda == expected['cuda_build'] and torch.backends.cudnn.version() == expected['cudnn'], 'certified environment')
        meter = Meter(c, out, start, torch.cuda if device == 'cuda' else None)
        write(out / 'CONSUMED.json', {'contract': digest(c), 'scope': c['mode'], 'no_retry': True})
        if c['mode'] == ct.CPU and c['fixture']['fault'] == 'worker_exception':
            raise RuntimeError('SYNTHETIC_WORKER_EXCEPTION')
        for arm in ct.ARMS:
            train(c, arm, True, out, meter, device)
        initial = [read(out / ('train-' + a) / 'INITIAL.json') for a in ct.ARMS]
        need(len({r['native_SHA'] for r in initial}) == 1 and initial[1]['head_SHA'] == initial[2]['head_SHA'], 'formal independent initial states')
        write(out / 'ENGINEERING_ALL_PASS.json', {'epochs_per_arm': 2, 'no_AP': True, 'same_initial': True,
                                                'same_heads': True, 'future_authorization_continues_to_24': True})
        if device == 'cuda':
            # Prespecified conservative update-based feasibility screen, not a quality gate.
            engineering = {a: [read(out / ('train-' + a) / ('epoch-%03d.json' % i)) for i in (1, 2)] for a in ct.ARMS}
            seconds_per_update = {a: max(x['stage_seconds'] / x['updates'] for x in values) for a, values in engineering.items()}
            projected = 1.25 * sum(v * 22 * 304 for v in seconds_per_update.values()) + 1800
            write(out / 'BUDGET_PROJECTION.json', {'basis': 'slowest of two full epochs per arm; 25pct margin plus 1800s endpoint/cost allowance',
                  'remaining_projected_seconds': projected, 'remaining_cap_seconds': ct.CAPS['GPU_stage_seconds'] - meter.ledger['GPU_stage_seconds'],
                  'not_quality_prediction': True})
            need(projected <= ct.CAPS['GPU_stage_seconds'] - meter.ledger['GPU_stage_seconds'], 'engineering budget feasibility STOP')
        for arm in ct.ARMS:
            train(c, arm, False, out, meter, device)
        if c['mode'] == ct.CPU and c['fixture']['fault'] == 'bad_endpoint':
            ep_path = out / 'train-N/endpoint.json'; ep = read(ep_path); ep['arm'] = 'S'; atomic(ep_path, ep)
        if c['mode'] == ct.CPU and c['fixture']['fault'] == 'missing_close':
            (out / 'train-N/CLOSED.json').rename(out / 'train-N/CLOSED_REMOVED_FOR_NEGATIVE_TEST.json')
        for arm in ct.ARMS:
            infer(c, arm, out, meter, device)
        if c['mode'] == ct.CPU and c['fixture']['fault'] == 'bad_input':
            q = out / 'infer-N/predictions-600.json'; x = read(q); x[0]['image_id'] = 'WRONG'; atomic(q, x)
        score(c, out, meter)
        cost(c, out, meter, device)
        compare(c, out, meter)
        need(meter.ledger['new_forward'] == c['planned_new_forwards'], 'terminal forward budget mismatch')
        need(meter.ledger['actual_updates'] == meter.ledger['reserved_updates'] == 3 * ct.profile(c)['epochs'] * ct.profile(c)['windows'], 'terminal update budget mismatch')
        slim = out / 'slim'; slim.mkdir()
        wanted = ['score/RESULT.json', 'cost/RESULT.json', 'compare/COMPARISON.json', 'compare/TABLE.md',
                  'ENGINEERING_ALL_PASS.json', *['score/' + a + '-' + b + '.json' for a in ct.ARMS for b in ('600', '300')]]
        for arm in ct.ARMS:
            wanted += ['train-' + arm + '/INITIAL.json', 'train-' + arm + '/ENGINEERING_PASS.json',
                       'train-' + arm + '/endpoint.json', 'train-' + arm + '/CLOSED.json', 'train-' + arm + '/RESULT.json',
                       'infer-' + arm + '/RESULT.json', 'infer-' + arm + '/CLOSED.json']
        for rel in wanted:
            q = slim / rel; q.parent.mkdir(parents=True, exist_ok=True); q.write_bytes((out / rel).read_bytes())
        meter.guard(); meter.publish()
        result = {'status': 'PASS_ENGINEERING', 'scope': c['mode'], 'scientific_status': read(out / 'compare/COMPARISON.json')['scientific_status'],
                  'ledger': meter.ledger, 'telemetry': meter.telemetry.arms,
                  'peak_allocated_bytes_all_stages': None if device == 'cpu' else meter.telemetry.observe_peak(),
                  'GPU_elapsed_definition': 'entire formal entry elapsed including CPU/IO; conservative allocation, no utilization integral',
                  'acceptance_state_used': False, 'no_followup': True, 'must_exit': True}
        write(slim / 'RESULT.json', result); write(slim / 'CONTRACT.json', c)
        write(slim / 'SHA_INDEX.json', {p.relative_to(slim).as_posix(): reference(p) for p in slim.rglob('*') if p.is_file()})
        need(sum(p.stat().st_size for p in slim.rglob('*') if p.is_file()) <= ct.CAPS['return_bytes'], 'slim return cap')
        meter.publish(); write(out / 'EXIT.json', {'returncode': 0, 'elapsed_seconds': meter.elapsed(), 'ledger': meter.ledger})
        write(out / 'CLOSED.json', {'status': 'PASS_ENGINEERING', 'contract': digest(c), 'result': reference(slim / 'RESULT.json'),
                                  'exit': reference(out / 'EXIT.json'), 'no_followup': True})
        return 0
    except Exception as e:
        (out / 'stderr.txt').write_text(traceback.format_exc(), encoding='utf-8')
        if meter:
            meter.publish()
        write(out / 'STOP.json', {'error': repr(e), 'no_retry': True, 'elapsed_seconds': time.monotonic() - start,
                                 'ledger': None if meter is None else meter.ledger})
        if (out / 'EXIT.json').exists():
            (out / 'EXIT.json').rename(out / 'EXIT_BEFORE_CLOSE_FAILURE.json')
        write(out / 'EXIT.json', {'returncode': 1}); print(repr(e), file=sys.stderr)
        return 1
    finally:
        signal.alarm(0)


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('plan'); a.add_argument('--synthetic', action='store_true'); a.add_argument('--fault', default='none')
    a.add_argument('--metadata'); a.add_argument('--acceptance-cost'); a.add_argument('--gpu-uuid')
    a.add_argument('--manifest-sha', required=True); a.add_argument('--attempt-root', required=True); a.add_argument('--out', required=True)
    a = sub.add_parser('run'); a.add_argument('--contract', required=True); a.add_argument('--authorization'); a.add_argument('--out', required=True)
    a = p.parse_args()
    if a.command == 'plan':
        plan(a)
    else:
        raise SystemExit(run(a))


if __name__ == '__main__':
    main()
