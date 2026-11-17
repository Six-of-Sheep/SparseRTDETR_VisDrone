"""Run the explicit, data-free research regression set on CPU Linux."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
os.environ["CUDA_VISIBLE_DEVICES"] = ""

TESTS = (
    "test_research_platform", "test_shared_query_cls", "test_sqc_seed12",
    "test_r27_local_memory", "test_r27_l0_control", "test_r27_l0_common",
    "test_r27_l0_cost", "test_r27_l0_summary",
    "test_r28_warp", "test_r28_resample", "test_r28_summary", "test_r28_followup",
    "test_r28_fovea", "test_r28_fovea_control", "test_r28_fovea_cal",
    "test_r28_cal_control", "test_r28_pipeline", "test_r28_fovea_geo", "test_r28_geo_control",
)


class NoSkips:
    def __init__(self):
        self.skipped = []

    def pytest_runtest_logreport(self, report):
        if report.skipped:
            self.skipped.append(report.nodeid)

    def pytest_collectreport(self, report):
        if report.skipped:
            self.skipped.append(report.nodeid)


def main():
    import pytest
    import torch
    if not sys.platform.startswith("linux") or torch.version.cuda is not None:
        raise RuntimeError("research CI requires Linux and CPU-only torch")
    torch.set_num_threads(2)
    files = [ROOT / "tests" / (name + ".py") for name in TESTS]
    if not all(path.is_file() for path in files):
        raise RuntimeError("required research suite is missing")
    policy = NoSkips()
    code = pytest.main(["-q", "-p", "no:cacheprovider", *map(str, files)], plugins=[policy])
    if torch.cuda.is_initialized() or policy.skipped:
        print("Research CI policy failure: CUDA initialized or skipped tests", policy.skipped)
        return 1
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
