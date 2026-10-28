"""Fresh read-only production admission. No training, forward, or repair."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
from sparse_rtdetr.research import sabl_formal_contract as ct

ROOT = Path(__file__).resolve().parents[1]


def checkpoint_bytes(ref):
    """Read registered size aliases without mutating the authenticated record."""
    ct.need(isinstance(ref, dict), 'checkpoint reference must be a mapping')
    fields = [key for key in ('bytes', 'size_bytes') if key in ref]
    ct.need(bool(fields), 'checkpoint size field missing')
    for key in fields:
        ct.need(type(ref[key]) is int and ref[key] > 0,
                'checkpoint size must be a positive integer: ' + key)
    ct.need(len(fields) == 1 or ref['bytes'] == ref['size_bytes'],
            'checkpoint size aliases conflict')
    return ref[fields[0]]


def material_admission(c, out):
    from sparse_rtdetr.localization.material_paths import MANIFESTS, metadata_path, SELECTION_SHA
    from sparse_rtdetr.localization.data import RawRoleDataset, read_bound, parse_raw, validate_role_roots
    from PIL import Image
    import io
    m = c['metadata']
    ct.need(m['parent_record']['sha256'] == 'cbf9fab7c6a9de54293716091064715e27bff5b5905b624fade5cc1fe0f12d88', 'parent metadata identity')
    parent = ct.exact(m['parent_record'])
    ct.need(all(m[k] == parent[k] for k in ('data','selection','python','environment','historical_N')), 'parent metadata binding')
    ct.need(m['historical_N'] == ct.read(ROOT / 'docs/research/r18_sabl_control/HISTORICAL_N.json')['historical_N'], 'history identity')
    history = m['historical_N']
    values = {k: ct.exact(history[k]) for k in ('endpoint','authentication','source','original_STOP','score_result','infer_result')}
    ct.need(values['endpoint']['checkpoint'] == m['checkpoint'] and m['checkpoint']['sha256'] == ct.N_SHA, 'parent checkpoint binding')
    ct.need(not Path(history['original_CLOSED_path']).exists(), 'historical closure status changed; retain original STOP')
    for ref in history['scores'].values():
        score = ct.exact(ref)
        ct.need(score['official']['evaluated_image_count'] == 548 and score['COCO_bridge']['maxDets'] == [1,10,100], 'historical score protocol')
    cp = Path(m['checkpoint']['path'])
    expected_bytes = checkpoint_bytes(m['checkpoint'])
    ct.need(cp.is_absolute() and cp.is_file() and not cp.is_symlink() and cp.stat().st_size == expected_bytes and ct.sha(cp) == ct.N_SHA, 'physical certified N checkpoint')
    ct.need(m['prior_storage_bytes'] == ct.accepted_storage(c['acceptance_cost']), 'prior storage inheritance')
    ct.need(shutil.disk_usage(out).free >= ct.CAPS['storage_bytes'] - m['prior_storage_bytes'], 'fresh free disk')
    seen = {}
    for role, spec in m['data'].items():
        ct.need(role in MANIFESTS and spec['role'] == role and spec['count'] == MANIFESTS[role][1], 'data role/count')
        ct.need(spec['manifest']['sha256'] == MANIFESTS[role][0], 'data manifest digest')
        metadata_path(role, spec['manifest']['path'], MANIFESTS[role][0])
        records = ct.exact(spec['manifest'])['records']
        ct.need(len(records) == spec['count'] and len({r['stable_image_id'] for r in records}) == len(records), 'data members')
        roots = spec['roots']; validate_role_roots(role, roots['images'], roots['annotations'], roots)
        seen[role] = records
        for row in records:
            ct.need(row['split'] == MANIFESTS[role][2], 'member split')
            blob = read_bound(Path(roots['images']) / Path(row['relative_path']).name, row['image_sha256'], row['image_size_bytes'])
            annotation = read_bound(Path(roots['annotations']) / Path(row['annotation_relative_path']).name, row['annotation_sha256'], row['annotation_size_bytes'])
            parsed = parse_raw(annotation, row['stable_image_id'], expected_rows=row['raw_annotation_rows'])
            if role == 'development':
                # Match Development548 exactly; train_core keeps the original
                # parser's nonpositive-row exclusion rather than a new gate.
                ct.need(not parsed['counts'].get('nonpositive_area', 0), 'unsupported nonpositive development GT')
            with Image.open(io.BytesIO(blob)) as image:
                ct.need(image.size == (row['width'],row['height']), 'member dimensions')
                image.verify()
    ct.need(set(seen) == {'train_core','development'}, 'exact data roles')
    selection = ct.exact(m['selection'])
    metadata_path('selection', m['selection']['path'], SELECTION_SHA)
    ct.need(m['selection']['sha256'] == SELECTION_SHA, 'selection digest')
    ordered = sorted(seen['development'], key=lambda r: (__import__('hashlib').sha256(('loc-trace-v1/'+r['stable_image_id']).encode()).hexdigest(),r['stable_image_id']))[:64]
    ct.need(selection['selected'] == [dict(r, selection_key=__import__('hashlib').sha256(('loc-trace-v1/'+r['stable_image_id']).encode()).hexdigest()) for r in ordered], 'fixed64 selection rule')
    ct.need(m['fixed_ids'] == [r['stable_image_id'] for r in ordered], 'fixed64 contract IDs')
    reuse = ct.read(ROOT / 'docs/research/r18_sabl_control/GPU_EVIDENCE_REUSE.json')
    ct.need(reuse['accepted_commit'] == ct.ACCEPT_COMMIT and reuse['accepted_manifest'] == ct.ACCEPT_MANIFEST, 'accepted evidence identity')
    for row in reuse['unchanged_files']:
        ct.need(ct.sha(ROOT / row['path']) == row['sha256'], 'accepted computation bytes changed: '+row['path'])
    return {'all_member_bytes_verified': {k:len(v) for k,v in seen.items()}, 'checkpoint_SHA':ct.N_SHA,
            'historical_status':history['status'], 'no_forward':True, 'accepted_computation_files':len(reuse['unchanged_files'])}


def resources(c, out):
    ct.need(Path(sys.executable).resolve() == Path(c['metadata']['python']).resolve(), 'certified interpreter')
    lines = subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,pci.bus_id,memory.free','--format=csv,noheader,nounits'], text=True).strip().splitlines()
    ct.need(len(lines)==1, 'single physical GPU')
    uuid,name,pci,free = [v.strip() for v in lines[0].split(',')]
    ct.need(uuid == c['gpu_uuid'] == 'GPU-1faee6f0-1da7-4ede-2475-67a5a00274a8' and name == 'NVIDIA GeForce RTX 4090 D'
            and pci == '00000000:01:00.0' and int(free)>=18432, 'fresh GPU identity/free memory')
    ct.need(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(), 'existing GPU job')
    # Atomic parent marker also rejects competing plans/attempts in this formal task.
    marker = out.parent / 'FORMAL_ONCE.json'
    with marker.open('x') as f:
        json.dump({'contract':ct.digest(c),'attempt':str(out),'no_retry':True},f)
    return {'uuid':uuid,'name':name,'pci':pci,'free_MiB':int(free)}
