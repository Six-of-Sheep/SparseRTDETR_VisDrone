"""Read-only reproduction verification against the original G0 passes (no GPU).

usage: dual_ema_verify.py <new_job_root> <g0_job_root> <out_json>

Pairs (new -> original G0, both retained):
  accept-D0/C2-direct.pth      -> accept-G0/C2-direct.pth
  formal-D0/checkpoint-FT000.pth -> formal-G0/checkpoint-FT000.pth
  formal-D0/checkpoint-FT006.pth -> formal-G0/checkpoint-FT006.pth
  formal-D0/checkpoint-FT012.pth -> formal-G0/checkpoint-FT012.pth

Compared component by component with exact tensor equality: the online model
state, the complete optimizer state (ids, options, step ledger and both moments),
the original EMA-R module and counter, the engine clock/parameter-step ledger and
the behaviour identity.  Identity strings that differ by design (schema, arm
name, plan/contract/source SHAs) are reported, not compared.  The second EMA
record is reported informationally (its counter must be EMA-R's + 36,480).

Writes the report to <out_json> and prints it.  A mismatch must be preserved and
forbids attributing any observed difference to the EMA counter.
"""
import hashlib
import json
import sys
from pathlib import Path

import torch

IDENTITY_FIELDS = ('schema', 'arm', 'plan_sha256', 'parent_sha256', 'total_updates',
                   'augmentation_clock', 'endpoint', 'dual_ema', 'checkpoint_passes')
PAIRS = (('accept-D0/C2-direct.pth', 'accept-G0/C2-direct.pth'),
         ('formal-D0/checkpoint-FT000.pth', 'formal-G0/checkpoint-FT000.pth'),
         ('formal-D0/checkpoint-FT006.pth', 'formal-G0/checkpoint-FT006.pth'),
         ('formal-D0/checkpoint-FT012.pth', 'formal-G0/checkpoint-FT012.pth'))


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def compare_tensors(left, right, label, report):
    if set(left) != set(right):
        report['key_mismatch'] = {'label': label, 'only_new': sorted(set(left) - set(right)),
                                  'only_reference': sorted(set(right) - set(left))}
        return False
    differing = []
    max_diff = 0.0
    for key in sorted(left):
        a, b = left[key], right[key]
        if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
            differing.append(key)
            if a.shape == b.shape and a.is_floating_point():
                max_diff = max(max_diff, float((a.float() - b.float()).abs().max()))
    report['tensors_compared'] = len(left)
    report['differing_tensors'] = differing[:20]
    report['differing_tensor_count'] = len(differing)
    report['max_abs_diff'] = max_diff
    return not differing


def compare_optimizer(new, ref, report):
    if new['param_groups'] != ref['param_groups']:
        report['optimizer_param_groups_equal'] = False
        return False
    report['optimizer_param_groups_equal'] = True
    if set(new['state']) != set(ref['state']):
        report['optimizer_state_ids_equal'] = False
        return False
    report['optimizer_state_ids_equal'] = True
    differing = []
    steps_match = True
    max_diff = 0.0
    for key in sorted(new['state']):
        a, b = new['state'][key], ref['state'][key]
        if type(a.get('step')) is not type(b.get('step')):
            steps_match = False
        if a.get('step') != b.get('step'):
            steps_match = False
        for name in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
            if name in a or name in b:
                x, y = a.get(name), b.get(name)
                if x is None or y is None or x.dtype != y.dtype or x.shape != y.shape or not torch.equal(x, y):
                    differing.append('%s:%s' % (key, name))
                    if x is not None and y is not None and x.shape == y.shape:
                        max_diff = max(max_diff, float((x.float() - y.float()).abs().max()))
    report['optimizer_step_ledger_equal'] = steps_match
    report['optimizer_moment_differing'] = differing[:20]
    report['optimizer_moment_differing_count'] = len(differing)
    report['optimizer_moment_max_abs_diff'] = max_diff
    return steps_match and not differing


def main():
    new_root = Path(sys.argv[1]); ref_root = Path(sys.argv[2]); out = Path(sys.argv[3])
    report = {'schema': 'r18-dual-ema-reproduction-v1', 'new_job_root': str(new_root),
              'reference_job_root': str(ref_root), 'pairs': {}}
    overall = True
    for new_rel, ref_rel in PAIRS:
        new_path, ref_path = new_root / new_rel, ref_root / ref_rel
        row = {'new': {'path': str(new_path), 'bytes': new_path.stat().st_size, 'sha256': file_sha(new_path)},
               'reference': {'path': str(ref_path), 'bytes': ref_path.stat().st_size, 'sha256': file_sha(ref_path)}}
        new = torch.load(new_path, map_location='cpu', weights_only=True)
        ref = torch.load(ref_path, map_location='cpu', weights_only=True)
        row['same_file_bytes'] = row['new']['sha256'] == row['reference']['sha256']
        row['model'] = {}
        row['ema_r_module'] = {}
        row['optimizer'] = {}
        model_holds = compare_tensors(new['model'], ref['model'], 'model', row['model'])
        ema_holds = compare_tensors(new['ema']['module'], ref['ema']['module'], 'ema_r_module', row['ema_r_module'])
        row['model_holds'] = bool(model_holds)
        row['ema_r_module_holds'] = bool(ema_holds)
        row['ema_r_updates'] = [int(new['ema']['updates']), int(ref['ema']['updates'])]
        row['ema_recipe'] = {'new': [new['ema']['decay'], new['ema']['warmups']],
                             'reference': [ref['ema']['decay'], ref['ema']['warmups']]}
        recipe_holds = bool(new['ema']['updates'] == ref['ema']['updates']
                            and (new['ema']['decay'], new['ema']['warmups']) == (ref['ema']['decay'], ref['ema']['warmups']))
        row['ema_r_counter_recipe_holds'] = recipe_holds
        row['ema_c_updates'] = [int(new['ema_alt']['updates']), None]
        row['ema_c_counter_consistent'] = bool(new['ema_alt']['updates'] == new['ema']['updates'] + 36480)
        first_key = sorted(new['ema_alt']['module'])[0]
        row['ema_c_first_key'] = first_key
        row['ema_c_module_differs_from_r'] = bool(
            not torch.equal(new['ema_alt']['module'][first_key], new['ema']['module'][first_key]))
        optimizer_holds = compare_optimizer(new['optimizer'], ref['optimizer'], row['optimizer'])
        keys = ('completed_updates', 'microsteps', 'completed_epochs', 'next_epoch', 'next_window', 'parameter_optimizer_steps')
        row['engine'] = {k: (new['engine'].get(k) == ref['engine'].get(k)) for k in keys}
        row['engine_clock'] = [new['engine'].get('completed_updates'), ref['engine'].get('completed_updates')]
        row['engine_identity_fields_differ_by_design'] = sorted(
            k for k in IDENTITY_FIELDS if new['engine'].get(k) != ref['engine'].get(k))
        row['holds'] = bool(model_holds and ema_holds and recipe_holds and optimizer_holds
                            and all(row['engine'].values()))
        row['behavior_equal'] = bool(new['behavior'] == ref['behavior'])
        if not row['behavior_equal']:
            row['holds'] = False
        row['optimizer_layout_equal'] = bool(new['optimizer_layout'] == ref['optimizer_layout'])
        row['model_layout_equal'] = bool(new['model_layout'] == ref['model_layout'])
        row['rng_equal'] = bool(new['rng']['python'] == ref['rng']['python']
                                and torch.equal(new['rng']['torch_cpu'], ref['rng']['torch_cpu'])
                                and new['rng']['numpy']['position'] == ref['rng']['numpy']['position'])
        row['execution_binding_new'] = new['execution_binding']
        report['pairs'][new_rel] = row
        overall = overall and row['holds']
        del new, ref
    report['reproduction_holds'] = bool(overall)
    report['note'] = ('exact tensor equality on every compared component; identity strings that differ by design are '
                      'listed, not compared.  If reproduction_holds is false the difference is preserved as-is and may '
                      'not be attributed to the EMA counter.')
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'reproduction_holds': report['reproduction_holds'],
                      'pairs': {k: {'holds': v['holds'], 'max_abs_diff': v.get('max_abs_diff')} for k, v in report['pairs'].items()},
                      'out': str(out)}))


if __name__ == '__main__':
    main()
