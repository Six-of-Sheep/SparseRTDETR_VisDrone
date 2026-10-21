"""Real metadata -> unchanged production prepare CLI, CPU only, no retries.

The raw spec must carry both purpose lists and must NOT pre-map fixed_ids.
Invalid cases use copies; no authenticated source or old receipt is modified.
The positive case reads every real member and initializes three real CPU arms.
"""
import argparse, copy, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from sparse_rtdetr.protocol_r18 import span_runtime as rt
    from sparse_rtdetr.protocol_r18.cuda_identity import recorded_subprocess
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    original = rt.read(args.spec)
    rt.require(original['profile'] == 'LAB_GPU_ONCE' and 'fixed_ids' not in original and
               'input_list_proof' not in original, 'raw unmapped production spec required')
    rt.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only environment required')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH', None)
    cases = []
    bad = copy.deepcopy(original)
    lists = bad['input_lists']; lists['timing'], lists['training_acceptance'] = lists['training_acceptance'], lists['timing']
    cases.append(('swapped_lists', bad, 'training acceptance purpose/role differs'))
    bad = copy.deepcopy(original)
    bad['input_lists']['timing']['ids'][0] = original['input_lists']['training_acceptance']['ids'][0][0]
    cases.append(('timing_correct_count_wrong_identity', bad, 'timing member/order differs'))
    bad = copy.deepcopy(original)
    bad['input_lists']['training_acceptance']['ids'][0][0] = original['input_lists']['timing']['ids'][0]
    cases.append(('training_correct_count_wrong_identity', bad, 'historical training acceptance member/order differs'))
    bad = copy.deepcopy(original); del bad['input_lists']['timing']['source']
    cases.append(('missing_field', bad, 'input-list fields missing/extra'))
    bad = copy.deepcopy(original); del bad['input_lists']
    cases.append(('missing_lists', bad, 'explicit training_acceptance/timing input lists required'))
    bad = copy.deepcopy(original); bad['input_lists']['timing']['source']['sha256'] = '0'*64
    cases.append(('timing_wrong_SHA', bad, 'timing source identity differs'))
    bad = copy.deepcopy(original); bad['input_lists']['training_acceptance']['source']['sha256'] = '0'*64
    cases.append(('training_wrong_SHA', bad, 'training acceptance source identity differs'))
    bad = copy.deepcopy(original); bad['input_lists']['timing']['source']['bytes'] += 1
    cases.append(('certified_SHA_wrong_file_size', bad, 'bound size/SHA differs'))
    bad = copy.deepcopy(original); bad['fixed_ids'] = copy.deepcopy(original['input_lists']['training_acceptance']['ids'])
    cases.append(('old_miswired_legacy_field', bad, 'legacy fixed_ids conflicts with timing purpose'))
    checks = []
    for name, spec, reason in cases:
        path = out/(name+'-spec.json'); rt.publish(path, spec)
        target = out/name
        cmd = [sys.executable, '-B', str(ROOT/'r18_sampling_span.py'), 'prepare',
               '--spec', str(path), '--source', str(args.source.resolve()), '--output', str(target)]
        result = recorded_subprocess(out/(name+'-process'), cmd, cwd=ROOT, env=env)
        rt.require(result.returncode != 0 and reason in result.stderr.decode(errors='replace'),
                   'expected production rejection absent: '+name)
        rt.require(not target.exists() and not target.with_name(target.name+'-contract.json').exists(),
                   'invalid mapping reached reader/sealed contract')
        checks.append(dict(name=name, status='EXPECTED_REJECTION', reason=reason))
    # Pass the provided raw spec as-is; the CLI itself performs the mapping.
    cmd = [sys.executable, '-B', str(ROOT/'r18_sampling_span.py'), 'prepare',
           '--spec', str(args.spec.resolve()), '--source', str(args.source.resolve()),
           '--output', str(out/'normal')]
    result = recorded_subprocess(out/'normal-process', cmd, cwd=ROOT, env=env)
    rt.require(result.returncode == 0, 'real production CPU preparation failed; no retry')
    gate = rt.read(out/'normal/PREPARE_PASS.json')
    rt.require(gate['model_forward_calls'] == 0 and gate['CUDA_initialized'] is False,
               'CPU preparation exceeded authority')
    contract = rt.read(out/'normal-contract.json')
    rt.require(contract['fixed_ids'] == original['input_lists']['timing']['ids'] and
               len(contract['fixed_ids']) == 64 and
               contract['input_lists']['training_acceptance']['ids'] == original['input_lists']['training_acceptance']['ids'],
               'purpose mapping was not preserved')
    checks.append(dict(name='normal_actual_prepare', status='PASS'))
    rt.publish(out/'REAL_INPUT_CLI_PASS.json', dict(status='REAL_INPUT_CLI_PASS',
        source=rt.reference(args.source.resolve()), spec=rt.reference(args.spec.resolve()),
        checks=checks, checks_run=len(checks), skips=0,
        prepare_gate=rt.reference(out/'normal/PREPARE_PASS.json'),
        input_list_proof=contract['input_list_proof'], model_forward_calls=0,
        CUDA_initialized=False, no_PYTHONPATH_in_child=True))
    print(json.dumps(dict(status='REAL_INPUT_CLI_PASS', checks_run=len(checks), skips=0)))


if __name__ == '__main__':
    main()
