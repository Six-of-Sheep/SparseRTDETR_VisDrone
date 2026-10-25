"""Verify/restore certified member metadata. Stdlib only; no images or models.

The TSVs preserve all 16 fields and row order of the original manifests.
Default: verify in memory. --output-dir writes into a new directory only.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath


FIELDS = (
    'annotation_relative_path', 'annotation_sha256', 'annotation_size_bytes',
    'coco_image_id', 'filtered_rows', 'height', 'ignore_rows', 'image_sha256',
    'image_size_bytes', 'keep_rows', 'raw_annotation_rows', 'relative_path',
    'sequence_key', 'split', 'stable_image_id', 'width',
)
INTEGER_FIELDS = {
    'annotation_size_bytes', 'coco_image_id', 'filtered_rows', 'height',
    'ignore_rows', 'image_size_bytes', 'keep_rows', 'raw_annotation_rows', 'width',
}
EXPECTED = {
    'train_core': (4869, 2955809, 'df87e6a377a5546d3d3fe0cd0e860d15c2fe5e8003ea87d887153579999a4e9d'),
    'development': (548, 329199, '5f4d788591b60bd2a6223c15e548248f4c0296656373d6d98c016d912a7537dd'),
}


def restore_bytes(directory):
    """Validate both inputs before returning original JSON bytes for writing."""
    results = {}
    index = json.loads((directory / 'SPLIT_INDEX.json').read_text(encoding='utf-8'))
    if index['schema'] != 'h2-member-tsv-v1' or tuple(index['fields']) != FIELDS:
        raise ValueError('Unexpected member schema')
    types = index.get('integer_fields')
    if not isinstance(types, list) or len(types) != len(INTEGER_FIELDS) or set(types) != INTEGER_FIELDS:
        raise ValueError('Integer field declaration differs')
    if set(index['splits']) != set(EXPECTED):
        raise ValueError('Unexpected split inventory')
    for name, (count, size, expected_hash) in EXPECTED.items():
        binding = index['splits'][name]
        raw = (directory / (name + '.tsv')).read_bytes()
        if len(raw) != binding['tsv_bytes'] or hashlib.sha256(raw).hexdigest() != binding['tsv_sha256']:
            raise ValueError(name + ': TSV bytes differ')
        reader = csv.DictReader(io.StringIO(raw.decode('utf-8'), newline=''), delimiter='\t')
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(name + ': columns differ')
        rows = []
        for row in reader:
            if set(row) != set(FIELDS) or any(v is None for v in row.values()):
                raise ValueError(name + ': malformed row')
            for key in INTEGER_FIELDS:
                if not row[key].isdigit():
                    raise ValueError(name + ': invalid integer')
                row[key] = int(row[key])
            for key in ('relative_path', 'annotation_relative_path'):
                path = PurePosixPath(row[key])
                if path.is_absolute() or '..' in path.parts or ':' in row[key] or '\\' in row[key]:
                    raise ValueError(name + ': non-portable path')
            rows.append(row)
        if len(rows) != count:
            raise ValueError(name + ': member count differs')
        for key in ('relative_path', 'stable_image_id', 'coco_image_id'):
            if len({row[key] for row in rows}) != count:
                raise ValueError(name + ': duplicate member identity')
        original = json.dumps({'records': rows}, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
        actual_hash = hashlib.sha256(original).hexdigest()
        if len(original) != size or actual_hash != expected_hash:
            raise ValueError(name + ': original manifest identity differs')
        if (binding['members'], binding['original_bytes'], binding['original_sha256']) != (count, size, expected_hash):
            raise ValueError(name + ': index disagrees with fixed identity')
        results[name] = original
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'docs/papers/r18_h2_center/splits')
    parser.add_argument('--output-dir', type=Path, help='New directory only; omitted means read-only verification')
    args = parser.parse_args()
    try:
        data = restore_bytes(args.metadata_dir)
        if args.output_dir is not None:
            # Do not overwrite an existing data root or a previous output.
            args.output_dir.mkdir(parents=False, exist_ok=False)
            for name, payload in data.items():
                with (args.output_dir / (name + '_manifest.json')).open('xb') as output:
                    output.write(payload)
        print(json.dumps({
            'status': 'PASS', 'mode': 'restore' if args.output_dir else 'verify_only',
            'splits': {name: {'members': EXPECTED[name][0], 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()} for name, payload in data.items()},
            'data_files_opened': 0, 'model_executions': 0,
        }, indent=2))
    except FileExistsError:
        parser.exit(2, 'Output already exists; no existing files were overwritten.\n')
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
        parser.exit(2, 'Member metadata check failed: ' + str(error) + '\n')


if __name__ == '__main__':
    main()
