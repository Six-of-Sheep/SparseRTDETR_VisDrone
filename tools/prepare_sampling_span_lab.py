"""Build explicit purpose spec from certified metadata, then real CPU CLI checks.

No GPU authorization/token, CUDA initialization, forward, training or scoring.
All paths supplied explicitly; no historical output is overwritten.
"""
import argparse, json, os, sys, traceback, uuid
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent-contract', 'selection', 'historical-checkpoint', 'historical-binding', 'source', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    from sparse_rtdetr.protocol_r18 import span_runtime as rt
    from sparse_rtdetr.protocol_r18.span_inputs import PARENT_SHA, TRAIN_PURPOSE, TIMING_PURPOSE
    from sparse_rtdetr.protocol_r18.cuda_identity import recorded_subprocess
    from sparse_rtdetr.localization.thermal_resume import snapshot
    from sparse_rtdetr.localization.material_paths import SELECTION_SHA, MANIFESTS
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    try:
        rt.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only environment required')
        source = rt.reference(args.source.resolve()); rt.verify_source(source)
        parent_ref = rt.reference(args.parent_contract.resolve())
        rt.require(parent_ref['sha256'] == PARENT_SHA, 'parent contract identity differs')
        old = rt.exact(parent_ref)
        rt.require(old['python'] == sys.executable, 'original interpreter differs')
        selection_ref = rt.reference(args.selection.resolve())
        rt.require(selection_ref['sha256'] == SELECTION_SHA, 'selection identity differs')
        selection = rt.exact(selection_ref)
        data = {}
        for role in ('train_core', 'development'):
            d = old['data'][role]; ref = rt.reference(d['manifest'])
            rt.require(ref['sha256'] == MANIFESTS[role][0], 'manifest identity differs')
            data[role] = dict(role=role, count=MANIFESTS[role][1], manifest=ref, roots=d['roots'])
        pretrained = rt.reference(old['pretrained']['path'])
        rt.require(pretrained['sha256'] == rt.recipe()['initialization']['ImageNet_sha256'], 'pretrained identity differs')
        historical = dict(checkpoint=rt.reference(args.historical_checkpoint.resolve()),
                          binding=rt.reference(args.historical_binding.resolve()))
        rt.require(historical['checkpoint']['sha256'] == '13a35283a02d2b728e3449917586ded5cdf2d578ebe6847510748e0dc4d21757'
                   and historical['binding']['sha256'] == '93d50992a27583832f2ab4ae4039b02db590ffbbb0850f050507bb1a0bdbc938',
                   'historical A identity differs')
        sample = snapshot(str(out)); rt.publish(out/'resource-before.json', sample)
        rt.require(sample['snapshot']['gpu']['devices'][0]['uuid']['value'] == old['gpu_uuid'], 'resource UUID differs')
        rt.publish(out/'PRIOR_COST.json', dict(task_id=rt.TASK, GPU_seconds=0, forward_calls=0,
            attempts=['preparation-cpu-001: CPU only STOP, no forward', 'preparation-cpu-002: CPU only'],
            scope='no GPU stage allocation in this task; historical CPU/synthetic failures retained separately'))
        spec = dict(profile='LAB_GPU_ONCE', seed=0, attempt='attempt-preparation-cpu-002',
            job_root=str(out.parent/'attempt-preparation-cpu-002'), nonce=uuid.uuid4().hex,
            boot_id=sample['boot_id'], gpu_uuid=old['gpu_uuid'],
            storage_identity={k:sample['storage'][k] for k in ('canonical_mount', 'mount', 'st_dev')},
            environment=rt.execution()['environment'], data=data, pretrained=pretrained, historical_A=historical,
            selection=selection_ref, baselines=rt.reference(ROOT/'docs/research/abc_results_20261019/SOURCE_SCALARS.json'),
            prior_cost=rt.reference(out/'PRIOR_COST.json'),
            input_lists=dict(training_acceptance=dict(purpose=TRAIN_PURPOSE, role='train_core',
                    source=parent_ref, ids=old['fixed_ids']),
                timing=dict(purpose=TIMING_PURPOSE, role='development', source=selection_ref,
                    ids=[r['stable_image_id'] for r in selection['selected']])))
        rt.publish(out/'lab-spec.json', spec)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_THREADING_LAYER='GNU')
        env.pop('PYTHONPATH', None)
        cmd = [sys.executable, '-B', str(ROOT/'tools/check_sampling_span_lab_inputs.py'),
            '--spec', str(out/'lab-spec.json'), '--source', source['path'], '--output', str(out/'real-input-cli')]
        result = recorded_subprocess(out/'real-input-cli-process', cmd, cwd=ROOT, env=env)
        rt.require(result.returncode == 0, 'real input CLI checks stopped; no retry')
        rt.publish(out/'LAB_CPU_SUMMARY.json', dict(status='REAL_LAB_CPU_PREPARE_PASS',
            source=source, gate=rt.reference(out/'real-input-cli/REAL_INPUT_CLI_PASS.json'),
            model_forward_calls=0, CUDA_initialized=False, GPU_authorization_created=False,
            production_execution_contract_created=False))
        print(json.dumps(dict(status='REAL_LAB_CPU_PREPARE_PASS', source=source)))
    except BaseException as exc:
        rt.publish(out/'PREPARATION_STOP.json', dict(status='STOP_NO_RETRY', error=repr(exc), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
