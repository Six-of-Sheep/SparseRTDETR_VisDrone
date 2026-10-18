"""Explicit purpose-bound reader. Existing role/confirmation guards stay intact."""
import io
import json
from pathlib import Path
from types import SimpleNamespace
from supplemental_runtime import exact, require, digest, reference, SEALED_SHA, TASK, PURPOSE


def regular(path):
    p = Path(path)
    require(p.is_absolute() and not any(x.is_symlink() for x in (p, *p.parents)), 'payload path symlink/relative')
    require(p.is_file(), 'payload missing: ' + str(p))
    return p


def member_path(root, row, kind):
    field = 'relative_path' if kind == 'images' else 'annotation_relative_path'
    rel = Path(row[field])
    require(not rel.is_absolute() and len(rel.parts) == 3 and rel.parts[:2] == ('train', kind), 'member source path')
    p = Path(root) / rel.name
    require(p.resolve().parent == Path(root).resolve(), 'member escapes bound root')
    return regular(p)


def read_bytes(path, sha, size):
    from sparse_rtdetr.localization.data import read_bound
    return read_bound(regular(path), sha, size)


def current_paths(c, rows):
    """New explicit location binding; never rewrite a historical source root."""
    if c['synthetic']:
        return
    spec = c['evaluation_data']
    binding = exact(spec['path_binding'])
    require(binding['task_id'] == TASK and binding['manifest'] == spec['manifest'], 'current path task/manifest')
    require(binding['images'] == spec['images'] and binding['annotations'] == spec['annotations'], 'current root differs from certified mapping')
    images, annotations = exact(binding['image_map']), exact(binding['annotation_map'])
    ids = [r['stable_image_id'] for r in rows]
    require([r['image_id'] for r in images] == [r['image_id'] for r in annotations] == ids, 'current mapping member order/identity')
    for row, image, annotation in zip(rows, images, annotations):
        ip = Path(spec['images']) / Path(row['relative_path']).name
        ap = Path(spec['annotations']) / Path(row['annotation_relative_path']).name
        require(image['current_path'] == str(ip) and image['bytes'] == row['image_size_bytes'] and image['sha256'] == row['image_sha256'], 'image mapping conflict')
        require(annotation['historical_relative_path'] == row['annotation_relative_path'] and annotation['restored'] == dict(path=str(ap), bytes=row['annotation_size_bytes'], sha256=row['annotation_sha256']), 'annotation mapping conflict')
        require(annotation['source_lineage'] == spec['lineage'], 'annotation recovery lineage conflict')


def dataset(c):
    """Called only after CPU/source freeze and new fresh resource admission."""
    require(c['task_id'] == TASK and c['purpose'] == PURPOSE, 'reader purpose')
    spec = c['evaluation_data']
    if c['synthetic']:
        require(c['sealed_access_authorized'] is False and spec['fixture'] is True, 'fixture scope')
        require('synthetic' in Path(spec['manifest']['path']).parts, 'fixture cannot name production data')
    else:
        require(c['sealed_access_authorized'] is True and spec['fixture'] is False, 'sealed reader authority')
        auth = exact(c['authorization'])
        require(auth['manifest_sha256'] == SEALED_SHA and auth['task_id'] == TASK and auth['purpose'] == PURPOSE, 'reader authorization binding')
        require(spec['manifest']['sha256'] == SEALED_SHA, 'exact manifest identity')
    for r in (spec['manifest'], spec['lineage'], spec['train_manifest']):
        regular(r['path'])
    manifest = exact(spec['manifest'])
    require(manifest['selection_allowed'] is False and manifest['metrics_access_allowed'] is False and manifest['single_final_access_only'] is True, 'original seal metadata changed')
    rows = manifest['records']
    expected = 6 if c['synthetic'] else 1602
    require(len(rows) == expected and len({r['stable_image_id'] for r in rows}) == expected, 'manifest count/duplicate')
    current_paths(c, rows)
    from sparse_rtdetr.data_protocol.schema import stable_image_id
    train = exact(spec['train_manifest'])['records']
    require(len(train) == (1 if c['synthetic'] else 4869), 'actual training member count')
    if not c['synthetic']:
        require(spec['train_manifest']['sha256'] == 'df87e6a377a5546d3d3fe0cd0e860d15c2fe5e8003ea87d887153579999a4e9d', 'authenticated train manifest')
        require(len({r['sequence_key'] for r in rows}) == 57, 'recorded group count')
    train_ids = {r['stable_image_id'] for r in train}
    train_sha = {r['image_sha256'] for r in train}
    train_paths = {r['relative_path'] for r in train}
    require(not any(r['stable_image_id'] in train_ids or r['image_sha256'] in train_sha or r['relative_path'] in train_paths for r in rows), 'actual training-member conflict')
    parsed = []
    from sparse_rtdetr.localization.data import parse_raw
    for r in rows:
        require(r['split'] == 'train' and r['stable_image_id'] == stable_image_id('train', r['relative_path']), 'stable member identity')
        require(type(r['width']) is int and type(r['height']) is int and min(r['width'], r['height']) > 0, 'member dimensions')
        raw = read_bytes(member_path(spec['annotations'], r, 'annotations'), r['annotation_sha256'], r['annotation_size_bytes'])
        value = parse_raw(raw, r['stable_image_id'], expected_rows=r['raw_annotation_rows'])
        if 'evaluation_input_extension' not in c:
            require(value['counts'].get('nonpositive_area', 0) == 0, 'unsupported nonpositive GT')
        parsed.append(value)
    # Immutable lineage is necessary only for physical-row/image binding. It is
    # not used to select members or to infer historical access or independence.
    ref = spec['lineage']
    require(reference(ref['path']) == ref, 'lineage SHA')
    by_id = {r['stable_image_id']: p for r, p in zip(rows, parsed)}
    expected_lines = {(r['stable_image_id'], x['physical_line']): x for r, p in zip(rows, parsed) for x in p['rows']}
    seen = set()
    for line in Path(ref['path']).read_bytes().splitlines():
        obj = json.loads(line)
        key = (obj['image_id'], obj['physical_line_number'])
        require(key in expected_lines and key not in seen, 'lineage member/missing/duplicate')
        seen.add(key); raw = expected_lines[key]
        require(obj['annotation_id'] == raw['id'] and obj['raw_line_sha256'] == raw['raw_line_sha256'] and obj['bbox_xyxy'] == raw['box'] and obj['raw_category_id'] == raw['category'], 'lineage/raw identity')
    require(seen == set(expected_lines), 'lineage physical rows incomplete')
    from confirmation_prepare import validate_parsed
    validate_parsed(rows, parsed, gt_extension=extension(c, rows, parsed))
    from torchvision.transforms import v2 as T
    data = SimpleNamespace(spec={'roots': {'images': spec['images']}}, records=rows,
                           by_id={r['stable_image_id']: r for r in rows}, transform=SimpleNamespace(resize=T.Resize([768, 1344])))
    return data, rows, parsed


def verify_images(data, c, emit=None):
    from infer_seed import load_original
    for i, row in enumerate(data.records):
        im = load_original(data, row)
        im.close()
        if emit is not None and (i % 32 == 0 or i + 1 == len(data.records)):
            emit(i + 1)
    return {'status': 'PASS', 'images': len(data.records), 'ordered_ids_sha256': digest([r['stable_image_id'] for r in data.records]),
            'raw_ignore': 'original parse_raw; original evaluator; budget before ignore, no refill',
            'historical_use': 'unknown', 'near_duplicates': 'unknown', 'flight_site': 'unknown',
            'independent_qualification': 'BLOCKED_PENDING_ACCESS_HISTORY'}


def extension(c, rows, parsed):
    if 'evaluation_input_extension' not in c:
        return None
    from sparse_rtdetr.data_protocol.supplemental_zero_gt import SupplementalZeroGT
    return SupplementalZeroGT.bind(rows, parsed, exact(c['evaluation_input_extension']))
