"""Run required production-interface regressions; skips are a CI failure."""
import os
from contextlib import ExitStack
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def main():
    import torch
    import torchvision
    if torch.version.cuda is not None or torch.cuda.is_initialized():
        raise RuntimeError('CI requires CPU-only torch, with CUDA uninitialized')
    torch.set_num_threads(1)
    print(f'CPU stack: torch={torch.__version__}, torchvision={torchvision.__version__}', flush=True)
    def forbidden(*args, **kwargs):
        raise RuntimeError('CI forbids model/checkpoint/CUDA/backward/AP execution')
    with ExitStack() as guards:
        # Only prohibited side effects are blocked, never the input/policy code
        # under test. No replacement torch, parser, evaluator input or channel.
        for target in ('torch.load', 'torch.nn.Module.__init__',
                       'torch.cuda._lazy_init', 'torch.autograd.backward'):
            guards.enter_context(patch(target, side_effect=forbidden))
        from sparse_rtdetr.r0 import evaluation
        for target in ('score_budget', 'bridge_metrics'):
            guards.enter_context(patch.object(evaluation, target, side_effect=forbidden))
        guards.enter_context(patch.object(evaluation.primary, 'evaluate_primary_v1', side_effect=forbidden))
        suite = unittest.TestSuite()
        for pattern in ('test_confirmation_prepare.py', 'test_supplemental_plan.py',
                        'test_supplemental_zero_gt.py'):
            found = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern=pattern)
            if not found.countTestCases():
                raise RuntimeError('required suite missing: ' + pattern)
            suite.addTests(found)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    if torch.cuda.is_initialized():
        raise RuntimeError('test initialized CUDA')
    if result.skipped:
        print('Required checks skipped: ' + repr(result.skipped), file=sys.stderr)
    return 0 if result.wasSuccessful() and not result.skipped else 1


if __name__ == '__main__':
    raise SystemExit(main())
