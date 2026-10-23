"""Strict registered source formats and read-only historical metadata snapshots.

Snapshots resolve original references without rewriting their bytes or paths.
They cannot verify a physical checkpoint or certify current server absence.
"""
from pathlib import Path, PurePosixPath
import re
from ..localization.runio import require, read, reference, sha


def source_rows(value):
    require(isinstance(value, dict), 'source manifest must be an object')
    require(('files' in value) != ('source_files' in value), 'unknown or ambiguous source format')
    if 'files' in value:
        require(set(value) == {'commit', 'scope', 'files'}, 'unknown legacy source fields')
        require(value['scope'] == 'actual release bytes; no execution authority', 'legacy source scope differs')
        rows = value['files']; fields = {'path', 'bytes', 'sha256'}
    else:
        require(set(value) == {'schema_version','experiment_id','commit','branch','config',
            'config_sha256','source_files','execution_authorized','scope'}, 'unknown versioned source fields')
        require(type(value['schema_version']) is int and value['schema_version'] == 1
            and value['execution_authorized'] is False, 'source schema/authority differs')
        require(all(isinstance(value[k], str) and value[k] for k in ('experiment_id','branch','config','scope')),
            'source identity fields missing')
        require(re.fullmatch('[0-9a-f]{64}', value['config_sha256']) is not None, 'config SHA invalid')
        rows = value['source_files']; fields = {'path','mode','bytes','sha256'}
    require(isinstance(value['commit'], str) and re.fullmatch('[0-9a-f]{40}', value['commit']) is not None,
        'source commit invalid')
    require(isinstance(rows, list) and rows, 'source entries missing')
    seen = set()
    for row in rows:
        require(isinstance(row, dict) and set(row) == fields, 'source entry fields differ')
        path = row['path']
        require(isinstance(path, str) and path and '\\' not in path and ':' not in path
            and not PurePosixPath(path).is_absolute() and PurePosixPath(path).as_posix() == path
            and all(p not in ('', '.', '..') for p in path.split('/')), 'source path invalid')
        require(path not in seen, 'duplicate/conflicting source path: ' + path); seen.add(path)
        require(type(row['bytes']) is int and row['bytes'] >= 0, 'source size invalid')
        require(isinstance(row['sha256'], str) and re.fullmatch('[0-9a-f]{64}', row['sha256']) is not None,
            'source SHA invalid')
        if 'mode' in fields: require(row['mode'] in ('100644','100755'), 'source mode invalid')
    if 'source_files' in value:
        by = {r['path']: r for r in rows}
        require(value['config'] in by and by[value['config']]['sha256'] == value['config_sha256'],
            'source config binding differs')
    return rows


class SnapshotReferences:
    """Explicit audit-only resolver; production prepare never selects this class."""
    def __init__(self, index, expected_sha256):
        require(sha(index) == expected_sha256, 'snapshot index SHA differs')
        self.root = Path(index).resolve().parent
        value = read(index)
        require(set(value) == {'schema','files','observed_absent'} and value['schema'] == 'H2_N_METADATA_SNAPSHOT_V1',
            'unknown snapshot index')
        self.rows = {}; self.reads = []
        for row in value['files']:
            require(set(row) == {'original','relative'}, 'snapshot entry fields differ')
            ref = row['original']; require(set(ref) == {'path','bytes','sha256'}, 'snapshot reference invalid')
            require(ref['path'] not in self.rows, 'duplicate snapshot path')
            p = self.root / row['relative']
            require(p.resolve().is_relative_to(self.root) and p.is_file() and not p.is_symlink(), 'snapshot escape/missing file')
            require(reference(p) == dict(ref, path=str(p)), 'snapshot bytes/SHA differ')
            self.rows[ref['path']] = (ref, p)
        self.absences = value['observed_absent']

    def exact(self, ref):
        require(set(ref) == {'path','bytes','sha256'} and ref['path'] in self.rows, 'snapshot reference missing')
        original, p = self.rows[ref['path']]
        require(ref == original and reference(p) == dict(ref, path=str(p)), 'snapshot reference identity differs')
        self.reads.append(ref)
        return read(p)

    def absent(self, path):
        require(path in self.absences and path not in self.rows, 'snapshot has no recorded absence')
        return True
