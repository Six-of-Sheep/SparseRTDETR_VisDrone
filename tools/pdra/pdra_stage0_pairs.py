"""PDRA stage 0: exact original-C reconstruction and unified pairing-domain material.

CPU only.  Reuses the executed c_stage_audit admission, raw decoding and exact original-C
reconstruction verbatim (``tools/c_stage_audit.py`` and ``tools/c_stage_core.py`` are
imported byte for byte from ``codex/r18-nms-suppressor-attribution`` and SHA-pinned
below).  Adds the rev-3 PDRA material: the cluster-internal pairing domain shared by
training and inference, GT label routing with the ambiguity ledger, the frozen module
train/eval sequence split, the frozen hyperparameters, a deterministic sampling manifest,
the measured data scale and the per-bucket evidence table.

Immutable outputs.  Any identity or reconstruction mismatch, or any cap excess, STOPs
without retry.  No GPU, model construction, detector forward, feature extraction,
comparator training or AP/AR scoring.  Seeds 1 and 2 are not reachable from this entry.
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent          # tools/pdra
TOOLS = HERE.parent                              # tools
for _p in (str(HERE), str(TOOLS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from c_stage_audit import PARENT_SHA, sha, publish, geometry  # noqa: E402
import pdra_core as P  # noqa: E402

TASK = 'r18-pdra-stage0-20261030-001'
SEED = 0
MODULE_VERSION = 'pdra-stage0-material-v1'
CAPS = {'wall_seconds': 1800, 'rss_bytes': 6 * 1024 ** 3, 'output_bytes': 1024 ** 3,
        'threads': 1, 'processes': 1}
OWN = ('pdra_stage0_pairs.py', 'pdra_core.py')
IMPORTED = {'c_stage_audit.py': '0592a54f436732c57eacca5da0e903c14f790f1be8bf6f6354292c0d6be13c99',
            'c_stage_core.py': '30cbf592e0696a51f24f0a4196d4f9eb7c491d89571049cafbc47bacd2ab9744'}
# Frozen seed0 pool/cluster totals measured on the 2026-10-30 attribution COUNTS (548 images).
EXPECTED_TOTALS = {'images': 548, 'pool': 1644000, 'clusters': 633230, 'suppressed': 1010770}
PAIR_LOWER_BOUND_REV2 = 1381849     # rev-2 convexity bound; the true count is measured here
SPLITS = ('train', 'eval')
RETURN_WHITELIST = ('START.json', 'ADMISSION.json', 'INPUT_PROOFS.json', 'SPLIT_SEQUENCES.json',
                    'HPARAMS_FROZEN.json', 'SAMPLING_MANIFEST.json', 'STAGE0_SUMMARY.json',
                    'seed0-IMAGE_COUNTS.json', 'RESULT.json', 'CLOSED.json', 'STOP.json')
SERVER_ONLY = ('seed0-pairs.jsonl', 'train-sampling-ids.txt')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rss_high_water():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmHWM:'):
            return int(line.split()[1]) * 1024
    raise ValueError('VmHWM unavailable')


def caps(start, out):
    elapsed = time.monotonic() - start
    hwm = rss_high_water()
    size = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    require(elapsed <= CAPS['wall_seconds'], f'wall cap exceeded: {elapsed:.1f}s')
    require(hwm <= CAPS['rss_bytes'], f'RSS cap exceeded: {hwm}')
    require(size <= CAPS['output_bytes'], f'output cap exceeded: {size}')
    return {'elapsed_seconds': elapsed, 'rss_high_water_bytes': hwm, 'output_bytes': size}


def text_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode('utf-8')).hexdigest()


def xyxy(gt):
    x, y, w, h = gt['bbox']
    return (float(x), float(y), float(x + w), float(y + h))


class _View:
    __slots__ = ('view_id',)

    def __init__(self, view_id):
        self.view_id = view_id


def view_affines(image_hw, tiles):
    return {v: P.candidate_affine(_View(v), image_hw, tiles) for v in range(5)}


def pair_id(image, a, b):
    return (f'{image}|{a.view_id},{a.query_id},{a.category_id}'
            f'|{b.view_id},{b.query_id},{b.category_id}')


def gt_matches(pool, gts_by_cat):
    """identity -> tuple((iou, GTItem)) at IoU >= MATCH_IOU, vectorised per category."""
    matches = {}
    by_cat = defaultdict(list)
    for p in pool:
        by_cat[p.category_id].append(p)
    for cat, cands in by_cat.items():
        items = gts_by_cat.get(cat, [])
        if not items:
            continue
        matrix = P.iou_matrix(np.asarray([c.box_xyxy for c in cands], dtype=np.float64),
                              np.asarray([g.box_xyxy for g in items], dtype=np.float64))
        for ci, c in enumerate(cands):
            row = matrix[ci]
            hits = [(float(row[gi]), g) for gi, g in enumerate(items) if row[gi] >= P.MATCH_IOU]
            if hits:
                hits.sort(key=lambda h: (-h[0], h[1].gt_id))
                matches[c.identity] = tuple(hits)
    return matches


class Ledger:
    """Descriptive counts for one split (or for all images).  Not AP, not a claim."""

    def __init__(self):
        self.images = 0
        self.pool = 0
        self.clusters = 0
        self.suppressed = 0
        self.cluster_sizes = Counter()
        self.pairs = 0
        self.pairs_per_image = []
        self.destination = Counter()
        self.feasible = Counter()
        self.infeasible_reason = Counter()
        self.ambiguous = 0
        self.would_be = Counter()
        self.multigt = 0
        self.view_relation = Counter()
        self.pref_feasible = Counter()        # (diou, area, view) -> pairs
        self.pref_feasible_images = defaultdict(set)
        self.pref_all = Counter()
        self.abstain_view = Counter()
        self.abstain_feasible = 0
        self.clear_pairs_feasible = 0

    def add_image(self, pool, clusters, sizes):
        self.images += 1
        self.pool += pool
        self.clusters += clusters
        self.suppressed += pool - clusters
        self.cluster_sizes.update(sizes)

    def add_pair(self, image, rec, feasible, reason, buckets):
        self.pairs += 1
        dest = rec['destination']
        self.destination[dest] += 1
        self.view_relation[buckets['view']] += 1
        if rec.get('multigt'):
            self.multigt += 1
        if rec.get('ambiguous'):
            self.ambiguous += 1
            for w in (rec.get('would_be') or ['<none>']):
                self.would_be[w] += 1
        if feasible:
            self.feasible[dest] += 1
        else:
            self.infeasible_reason[reason] += 1
        if dest == 'pref':
            key = (buckets['diou'], buckets['area'], buckets['view'])
            self.pref_all[key] += 1
            if feasible:
                self.pref_feasible[key] += 1
                self.pref_feasible_images[key].add(image)
                if abs(rec['delta_iou']) >= P.TAU_CLEAR:
                    self.clear_pairs_feasible += 1
        elif dest == 'abstain':
            self.abstain_view[buckets['view']] += 1
            if feasible:
                self.abstain_feasible += 1

    def evidence(self):
        """Per-bucket feasible preference evidence with the frozen N_min flags."""
        def table(selector):
            rows = {}
            for key, count in self.pref_feasible.items():
                name = selector(key)
                row = rows.setdefault(name, {'pairs': 0, 'images': set()})
                row['pairs'] += count
                row['images'] |= self.pref_feasible_images[key]
            out = {}
            for name, row in rows.items():
                images = len(row['images'])
                out[name] = {'pairs': row['pairs'], 'images': images,
                             'status': ('ok' if row['pairs'] >= P.N_MIN_PAIRS
                                        and images >= P.N_MIN_IMAGES else P.EVIDENCE_INSUFFICIENT)}
            return dict(sorted(out.items()))
        by_diou = table(lambda k: k[0])
        by_area = table(lambda k: k[1])
        by_diou_area = table(lambda k: f'{k[0]}|{k[1]}')
        by_view = table(lambda k: k[2])
        for bucket in P.DIOU_BUCKETS:
            by_diou.setdefault(bucket, {'pairs': 0, 'images': 0, 'status': P.EVIDENCE_INSUFFICIENT})
        for bucket in P.AREA_BUCKETS:
            by_area.setdefault(bucket, {'pairs': 0, 'images': 0, 'status': P.EVIDENCE_INSUFFICIENT})
        return {'n_min_pairs': P.N_MIN_PAIRS, 'n_min_images': P.N_MIN_IMAGES,
                'by_diou': by_diou, 'by_area': by_area, 'by_view': by_view,
                'by_diou_area': by_diou_area}

    def summary(self):
        ppi = sorted(self.pairs_per_image)
        return {
            'images': self.images, 'pool': self.pool, 'clusters': self.clusters,
            'suppressed': self.suppressed, 'pairs': self.pairs,
            'cluster_size_histogram': {str(k): v for k, v in sorted(self.cluster_sizes.items())},
            'cluster_size_max': max(self.cluster_sizes) if self.cluster_sizes else 0,
            'pairs_per_image': {'min': ppi[0] if ppi else 0,
                                'median': ppi[len(ppi) // 2] if ppi else 0,
                                'max': ppi[-1] if ppi else 0},
            'destination_counts': {d: self.destination.get(d, 0) for d in P.PAIR_DESTINATIONS},
            'feasible_counts': {d: self.feasible.get(d, 0) for d in P.PAIR_DESTINATIONS},
            'infeasible_reasons': dict(sorted(self.infeasible_reason.items())),
            'ambiguous_pairs': self.ambiguous, 'ambiguous_would_be': dict(self.would_be),
            'multigt_pairs': self.multigt, 'view_relation': dict(self.view_relation),
            'pref_pairs_all': sum(self.pref_all.values()),
            'pref_pairs_feasible': sum(self.pref_feasible.values()),
            'clear_pairs_feasible': self.clear_pairs_feasible,
            'abstain_pairs_feasible': self.abstain_feasible,
            'abstain_by_view': dict(self.abstain_view),
            'evidence': self.evidence(),
        }


class Sampling:
    """Deterministic train sampling: all feasible pref pairs + abstain pairs by hash order."""

    def __init__(self):
        self.pref = []
        self.abstain = []

    def add(self, destination, identifier):
        if destination == 'pref':
            self.pref.append(identifier)
        elif destination == 'abstain':
            self.abstain.append(identifier)

    def manifest(self, out):
        k = min(len(self.abstain), len(self.pref))
        ranked = sorted(self.abstain, key=lambda s: hashlib.sha256(s.encode()).hexdigest())
        chosen = sorted(ranked[:k])
        lines = [f'pref\t{s}' for s in sorted(self.pref)] + [f'abstain\t{s}' for s in chosen]
        path = out / 'train-sampling-ids.txt'
        with path.open('x', encoding='utf-8', newline='\n') as f:
            f.write('\n'.join(lines) + ('\n' if lines else ''))
        spec = {'rule': 'all feasible train pref pairs; feasible train abstain pairs ranked by '
                        'sha256(pair_id) ascending, first min(#abstain, #pref) taken; no randomness',
                'mix_ratio_abstain_to_pref': 1.0 if self.pref else None,
                'n_pair_pref': len(self.pref), 'n_pair_abstain_available': len(self.abstain),
                'n_pair_abstain_selected': k, 'n_pair_total': len(self.pref) + k,
                'feasibility': 'only pairs whose two exclusive-region reads are valid under theta_region'}
        return {**spec, 'sampling_spec_sha256': text_sha(spec),
                'ids_file': {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path),
                             'server_only': True}}


def materialise_image(image, sequence, split, pool, clusters, gts_by_cat, affines,
                      ledgers, sink, sampling):
    matches = gt_matches(pool, gts_by_cat)
    sizes = [1 + len(cl['suppressed']) for cl in clusters]
    require(sum(sizes) == len(pool), 'cluster partition incomplete')
    for ledger in ledgers:
        ledger.add_image(len(pool), len(clusters), sizes)
    per_dest = Counter()
    pairs = 0
    for cluster in clusters:
        for a, b in P.cluster_pairs(cluster):
            pairs += 1
            rec = P.route_pair(matches.get(a.identity, ()), matches.get(b.identity, ()))
            rec['same_view'] = a.view_id == b.view_id
            pair = P.build_pair_input(a.box_xyxy, affines[a.view_id], b.box_xyxy, affines[b.view_id])
            buckets = P.pair_buckets(rec)
            for ledger in ledgers:
                ledger.add_pair(image, rec, pair.valid, pair.reason, buckets)
            per_dest[rec['destination']] += 1
            if split == 'train' and pair.valid:
                sampling.add(rec['destination'], pair_id(image, a, b))
            sink.write(json.dumps([
                image, sequence, split, a.category_id, [a.view_id, a.query_id],
                [b.view_id, b.query_id], int(rec['same_view']), round(pair.geometry[-1], 6),
                rec['destination'], rec.get('label'),
                None if rec.get('delta_iou') is None else round(rec['delta_iou'], 6),
                rec['gt_ids'][0] if rec['destination'] in ('pref', 'excluded_gap') else None,
                rec.get('gt_area'), int(pair.valid), pair.reason], allow_nan=False) + '\n')
    for ledger in ledgers:
        ledger.pairs_per_image.append(pairs)
    return {'image_id': image, 'sequence': sequence, 'split': split, 'pool': len(pool),
            'clusters': len(clusters), 'suppressed': len(pool) - len(clusters), 'pairs': pairs,
            'max_cluster_size': max(sizes) if sizes else 0,
            **{d: per_dest.get(d, 0) for d in P.PAIR_DESTINATIONS}}


def finalize(out, ledgers, image_rows, sequence_images, sampling, pool_hash, proofs,
             reconstruction, start, extra):
    split = P.freeze_sequence_split(sequence_images)
    split_sha = text_sha(split['assignment'])
    eval_images = sorted(i for k in split['eval_sequences'] for i in sequence_images[k])
    eval_sha = text_sha(eval_images)
    split_record = {**split, 'split_sha256': split_sha, 'eval_image_ids': eval_images,
                    'eval_members_sha256': eval_sha,
                    'sequence_images': {k: len(v) for k, v in sorted(sequence_images.items())},
                    'sequence_pairs': {k: sum(r['pairs'] for r in image_rows if r['sequence'] == k)
                                       for k in sorted(sequence_images)}}
    publish(out / 'SPLIT_SEQUENCES.json', split_record)
    manifest = sampling.manifest(out)
    publish(out / 'SAMPLING_MANIFEST.json', manifest)
    hparams = {**P.frozen_hyperparameters(), 'split_sha256': split_sha,
               'eval_members_sha256': eval_sha,
               'sampling_spec_sha256': manifest['sampling_spec_sha256'],
               'candidate_pool_sha256': pool_hash}
    publish(out / 'HPARAMS_FROZEN.json', hparams)
    arms = [P.ArmSpec(key, reads, pool_hash, split_sha256=split_sha, eval_members_sha256=eval_sha,
                      sampling_spec_sha256=manifest['sampling_spec_sha256'])
            for key, reads in (('a_original_c', 'score'), ('b_single_box_quality', 'whole_box_quality'),
                               ('c_whole_box_pairwise', 'whole_box_pairwise'),
                               ('pdra', 'difference_strips'))]
    fairness = P.assert_control_fairness(arms)
    require(fairness['fair'], 'declared arms violate control fairness')
    summaries = {name: ledger.summary() for name, ledger in ledgers.items()}
    for name in SPLITS:
        require(summaries[name]['pairs'] == sum(r['pairs'] for r in image_rows if r['split'] == name),
                f'{name} pair total differs from image rows')
    require(summaries['train']['pairs'] + summaries['eval']['pairs'] == summaries['all']['pairs'],
            'split pair totals do not add up')
    for name, summary in summaries.items():
        require(sum(summary['destination_counts'].values()) == summary['pairs'],
                f'{name} destination partition incomplete')
    publish(out / 'seed0-IMAGE_COUNTS.json', image_rows)
    summary = {
        'task': TASK, 'seed': SEED, 'module_version': MODULE_VERSION,
        'core_version': P.CORE_VERSION, 'pair_domain': P.PAIR_DOMAIN,
        'reconstruction': reconstruction,
        'scale': {name: {k: summaries[name][k] for k in ('images', 'pool', 'clusters', 'suppressed',
                                                          'pairs', 'pairs_per_image',
                                                          'cluster_size_max')}
                  for name in summaries},
        'pair_count_vs_rev2_lower_bound': {'measured': summaries['all']['pairs'],
                                           'rev2_lower_bound': PAIR_LOWER_BOUND_REV2,
                                           'bound_holds': summaries['all']['pairs'] >= PAIR_LOWER_BOUND_REV2},
        'ledgers': summaries,
        'main_verdict_split': P.MAIN_VERDICT_SPLIT,
        'main_verdict_evidence': summaries['eval']['evidence'],
        'split': {k: split_record[k] for k in ('rule', 'sequence_count', 'train_sequences',
                                               'eval_sequences', 'train_images', 'eval_images',
                                               'split_sha256', 'eval_members_sha256')},
        'sampling': {k: manifest[k] for k in ('rule', 'n_pair_pref', 'n_pair_abstain_available',
                                              'n_pair_abstain_selected', 'n_pair_total',
                                              'sampling_spec_sha256')},
        'control_fairness': fairness,
        'hyperparameters_sha256': sha(out / 'HPARAMS_FROZEN.json'),
        'units': 'descriptive counts; not AP/recall/recoverable TP; no method or validity claim',
        'return_whitelist': list(RETURN_WHITELIST), 'server_only': list(SERVER_ONLY),
        **extra,
    }
    summary['caps'] = caps(start, out)
    return publish(out / 'STAGE0_SUMMARY.json', summary), summary


def run(parent, out, source_commit, prepare_only=False):
    start = time.monotonic()
    for name, expected in IMPORTED.items():
        require(sha(TOOLS / name) == expected, f'imported {name} differs from the executed byte-identical copy')
    require(sha(parent) == PARENT_SHA, 'parent contract SHA differs')
    c = json.loads(parent.read_bytes())
    source = Path(c['source_root'])
    for item in c['source_files']:
        rel = Path(item['path'])
        if rel.is_absolute() or '..' in rel.parts or sha(source / rel) != item['sha256']:
            raise ValueError('external frozen source mismatch')
    sys.path.insert(0, str(source))
    import softnms_job as j
    torch = j.cpu_only()
    c = j.load_contract(parent)
    specs, rows, data_spec = j.context(c)
    from slice_core import classwise_nms, NMS_IOU
    require(NMS_IOU == 0.5 and P.NMS_IOU == NMS_IOU, 'frozen NMS IoU differs')
    from sparse_rtdetr.r0 import evaluation as ev
    from sparse_rtdetr.baseline import primary_evaluator as primary

    def forbidden(*args, **kwargs):
        raise RuntimeError('stage 0 forbids scoring')
    ev.score_budget = ev.bridge_metrics = primary.evaluate_primary_v1 = forbidden
    parsed = j.parsed_gt(rows, data_spec)
    plan = j.read(j.PLAN)
    j.resources(out, plan)
    publish(out / 'ADMISSION.json', {
        'task': TASK, 'parent_contract_sha256': PARENT_SHA, 'external_source': str(source),
        'source_files_verified': len(c['source_files']), 'images': len(rows),
        'seeds_available': [s['seed'] for s in specs], 'seed_processed': SEED,
        'own_source': {n: sha(HERE / n) for n in OWN},
        'imported_source': {n: {'sha256': sha(TOOLS / n), 'expected': IMPORTED[n]} for n in IMPORTED},
        'source_commit': source_commit, 'module_version': MODULE_VERSION,
        'core_version': P.CORE_VERSION, 'hyperparameters': P.frozen_hyperparameters(),
        'nms_iou': NMS_IOU, 'caps': CAPS, 'expected_totals': EXPECTED_TOTALS,
        'GPU': 0, 'model_constructions': 0, 'forward_passes': 0, 'feature_reads': 0,
        'training_updates': 0, 'AP_calls': 0})
    if prepare_only:
        return {'status': 'ADMISSION_PASS', 'images': len(rows), 'parsed_annotations': len(parsed),
                'cache_batches_processed': 0, 'model_calls': 0, 'AP_calls': 0, 'seed': SEED}
    spec = next((s for s in specs if s['seed'] == SEED), None)
    require(spec is not None, 'seed0 spec absent')
    proofs, expected = [], {}
    for budget in (600, 300):
        preds = json.loads(j.exact(spec['evaluation']['predictions']['C'][str(budget)], contents=True))
        proofs.append({'seed': SEED, **j.evaluation_input_proof(spec, rows, parsed, preds, 'C', budget)})
        expected[budget] = j.group(preds, spec['ids'], budget)
    publish(out / 'INPUT_PROOFS.json', proofs)
    # the frozen split is decided from the sequence keys of the 548 development rows
    sequence_images = defaultdict(list)
    for row in rows:
        sequence_images[Path(row['annotation_relative_path']).stem.split('_')[0]].append(row['stable_image_id'])
    split = P.freeze_sequence_split(sequence_images)
    ledgers = {'all': Ledger(), 'train': Ledger(), 'eval': Ledger()}
    sampling = Sampling()
    image_rows = []
    pool_hash = hashlib.sha256()
    comparisons = boundary = 0
    batches = spec['full']['raw_view_refs']
    with (out / 'seed0-pairs.jsonl').open('x', encoding='utf-8', newline='\n') as sink:
        for batch, ref in enumerate(batches):
            j.resources(out, plan)
            state = caps(start, out)
            raw = torch.load(io.BytesIO(j.exact(ref, contents=True)), map_location='cpu', weights_only=True)
            part = rows[4 * batch: 4 * batch + 4]
            pools = j.decode_raw(raw, part, spec['identity'], SEED)
            maps = geometry(raw, part, j, torch)
            for local, (row, pool, geomap) in enumerate(zip(part, pools, maps)):
                image = row['stable_image_id']
                ann = parsed[4 * batch + local]
                require(len(geomap) == 1500, 'raw geometry incomplete')
                require(all(tuple(p.box_xyxy) == geomap[(p.view_id, p.query_id)] for p in pool),
                        'raw mapping mismatch')
                kept = classwise_nms(pool)
                for budget in (600, 300):
                    require([j.prediction(image, p) for p in kept[:budget]] == expected[budget][image],
                            'original C exact reconstruction differs')
                clusters, stats = P.nms_clusters(pool, with_stats=True)
                require([cl['survivor'].identity for cl in clusters] == [p.identity for p in kept],
                        'cluster survivors differ from frozen NMS')
                comparisons += stats['comparisons']
                boundary += stats['boundary_within_eps']
                for p in pool:
                    pool_hash.update(f'{image}|{p.view_id},{p.query_id},{p.category_id}|{p.score!r}\n'.encode())
                tiles = {t['view_id']: tuple(float(v) for v in t['crop_box']) for t in raw['geometry'][local]}
                affines = view_affines((row['height'], row['width']), tiles)
                dataset, _, _ = ev.bridge_inputs([row], [ann], [])
                gts_by_cat = defaultdict(list)
                for g in dataset['annotations']:
                    if g['iscrowd'] == 0:
                        gts_by_cat[g['category_id']].append(
                            P.GTItem(g['raw_annotation_id'], g['category_id'], xyxy(g), float(g['area'])))
                sequence = Path(row['annotation_relative_path']).stem.split('_')[0]
                part_name = split['assignment'][sequence]
                image_rows.append(materialise_image(image, sequence, part_name, pool, clusters, gts_by_cat,
                                                    affines, [ledgers['all'], ledgers[part_name]],
                                                    sink, sampling))
            print(json.dumps({'seed': SEED, 'batch': batch + 1, 'batches': len(batches),
                              'pairs': ledgers['all'].pairs, **state}), flush=True)
    require(len(image_rows) == EXPECTED_TOTALS['images'], 'image count differs from 548')
    totals = {'images': ledgers['all'].images, 'pool': ledgers['all'].pool,
              'clusters': ledgers['all'].clusters, 'suppressed': ledgers['all'].suppressed}
    require(totals == EXPECTED_TOTALS, f'pool/cluster totals differ from the frozen COUNTS: {totals}')
    require(not torch.cuda.is_initialized(), 'CUDA initialized')
    reconstruction = {'original_C_exact_both_budgets': True, 'cluster_survivors_equal_frozen_nms': True,
                      'raw_batches': len(batches), 'nms_comparisons': comparisons,
                      'nms_boundary_within_1e-9': boundary, 'totals': totals,
                      'expected_totals': EXPECTED_TOTALS, 'totals_reproduced': True,
                      'raw_views_reconstructible': True,
                      'note': 'the pool is rebuilt exactly from the frozen raw views; the rev-2 statement '
                              'that cluster sizes could not be recovered offline was wrong and is withdrawn'}
    ref, summary = finalize(out, ledgers, image_rows, sequence_images, sampling, pool_hash.hexdigest(),
                            proofs, reconstruction, start,
                            {'pairs_jsonl': {'path': str(out / 'seed0-pairs.jsonl'),
                                             'bytes': (out / 'seed0-pairs.jsonl').stat().st_size,
                                             'sha256': sha(out / 'seed0-pairs.jsonl'), 'server_only': True}})
    return {'status': 'COMPLETE', 'task': TASK, 'seed': SEED, 'seeds_processed': [SEED],
            'seeds_1_2_unreachable_from_entry': True, 'automatic_next': False,
            'original_C_exact': True, 'cluster_survivors_equal_frozen_nms': True,
            'totals_reproduced': True, 'pairs_measured': summary['scale']['all']['pairs'],
            'AP_calls': 0, 'GPU_calls': 0, 'model_calls': 0, 'forward_passes': 0,
            'feature_reads': 0, 'training_updates': 0, 'stage0_summary': ref, 'caps': CAPS}


def fixture(out):
    """Synthetic end-to-end run of the materialiser: no contract, no torch, no raw views."""
    sys.path.insert(0, str(TOOLS.parent))
    import slice_core
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    image_hw = (1500, 2000)
    tiles = {t.view_id: t.crop_box for t in slice_core.tile_windows(*image_hw)}
    affines = view_affines(image_hw, tiles)
    def cand(box, score, view, query, cat, rank):
        return slice_core.Candidate(box, score, view, query, cat, rank)
    images = {
        ('img_a', 'seqA'): ([cand((100., 100., 200., 200.), .95, 0, 1, 1, 1),
                             cand((105., 100., 205., 200.), .90, 1, 2, 1, 2),
                             cand((100., 100., 200., 290.), .85, 2, 3, 1, 3),
                             cand((500., 500., 600., 600.), .80, 0, 4, 2, 4),
                             cand((505., 505., 605., 605.), .70, 3, 5, 2, 5)],
                            {1: [P.GTItem(1, 1, (100., 100., 200., 200.), 10000.)],
                             2: [P.GTItem(2, 2, (500., 500., 600., 600.), 10000.)]}),
        ('img_b', 'seqB'): ([cand((100., 100., 200., 200.), .95, 0, 1, 1, 1),
                             cand((205., 100., 305., 200.), .90, 0, 2, 1, 2),
                             cand((130., 100., 230., 200.), .60, 4, 3, 1, 3)],
                            {1: [P.GTItem(1, 1, (100., 100., 190., 200.), 9000.),
                                 P.GTItem(2, 1, (160., 100., 260., 200.), 10000.)]}),
        ('img_c', 'seqC'): ([cand((900., 900., 950., 950.), .95, 0, 1, 3, 1),
                             cand((905., 900., 955., 950.), .50, 1, 2, 3, 2)], {}),
    }
    sequence_images = defaultdict(list)
    for (image, sequence) in images:
        sequence_images[sequence].append(image)
    split = P.freeze_sequence_split(sequence_images)
    ledgers = {'all': Ledger(), 'train': Ledger(), 'eval': Ledger()}
    sampling = Sampling()
    image_rows = []
    pool_hash = hashlib.sha256()
    enumerated = 0
    with (out / 'seed0-pairs.jsonl').open('x', encoding='utf-8', newline='\n') as sink:
        for (image, sequence), (pool, gts) in images.items():
            kept = slice_core.classwise_nms(pool)
            clusters = P.nms_clusters(pool)
            require([cl['survivor'].identity for cl in clusters] == [p.identity for p in kept],
                    'synthetic cluster survivors differ from frozen NMS')
            enumerated += len(P.cluster_pair_records(pool))
            for p in pool:
                pool_hash.update(f'{image}|{p.view_id},{p.query_id},{p.category_id}|{p.score!r}\n'.encode())
            part_name = split['assignment'][sequence]
            image_rows.append(materialise_image(image, sequence, part_name, pool, clusters, gts, affines,
                                                [ledgers['all'], ledgers[part_name]], sink, sampling))
    reconstruction = {'synthetic': True, 'cluster_survivors_equal_frozen_nms': True}
    ref, summary = finalize(out, ledgers, image_rows, sequence_images, sampling, pool_hash.hexdigest(),
                            [], reconstruction, start, {'synthetic': True})
    ledger = summary['ledgers']['all']
    require(ledger['pairs'] == enumerated, 'ledger pair count differs from the enumeration')
    require(ledger['destination_counts']['pref'] >= 1 and ledger['destination_counts']['abstain'] >= 1
            and ledger['destination_counts']['excluded_empty'] >= 1, 'fixture did not exercise routing')
    return {'status': 'COMPLETE', 'synthetic': True, 'module_version': MODULE_VERSION,
            'pairs_enumerated': enumerated, 'ledger': {'total_pairs': ledger['pairs'],
                                                      'destination_counts': ledger['destination_counts']},
            'split': summary['split'] | {'main_verdict_split': summary['main_verdict_split']},
            'stage0_summary': ref}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('run', 'admit', 'fixture'))
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', default='UNRECORDED')
    parser.add_argument('--inject-failure', action='store_true')
    args = parser.parse_args(argv)
    if args.mode != 'fixture' and (args.parent is None or args.inject_failure):
        parser.error('invalid production options')
    out = args.output.resolve()
    if args.mode != 'fixture' and TASK not in out.parts:
        parser.error('production output is outside task directory')
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    publish(out / 'START.json', {'task': TASK, 'mode': args.mode, 'pid': os.getpid(),
                                 'started_unix': time.time(), 'seed': SEED,
                                 'source_commit': args.source_commit})
    try:
        if args.mode == 'fixture':
            if args.inject_failure:
                raise RuntimeError('expected fixture failure')
            result = fixture(out)
        else:
            result = run(args.parent, out, args.source_commit, prepare_only=args.mode == 'admit')
        result['elapsed_seconds'] = time.monotonic() - start
        ref = publish(out / 'RESULT.json', result)
        publish(out / 'CLOSED.json', {'result': ref, 'returncode': 0})
        print(json.dumps({'status': 'COMPLETE', 'result': ref}), flush=True)
        return 0
    except Exception:
        publish(out / 'STOP.json', {'traceback': traceback.format_exc(),
                                    'elapsed_seconds': time.monotonic() - start,
                                    'automatic_retry': False})
        traceback.print_exc()
        return 2


if __name__ == '__main__':
    sys.exit(main())
