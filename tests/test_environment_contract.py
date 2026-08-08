import importlib
import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "rtdetrv2_pytorch"
CONFIG = VENDOR / "configs" / "rtdetrv2" / "rtdetrv2_r18vd_120e_coco.yml"
R2_NAME = "sparse-rtdetrv2-p3-r2"


def conda_records():
    records = {}
    for path in (Path(sys.prefix) / "conda-meta").glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        records[record["name"]] = record
    return records


class EnvironmentContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if str(VENDOR) not in sys.path:
            sys.path.insert(0, str(VENDOR))
        cls.torch = importlib.import_module("torch")
        cls.torchvision = importlib.import_module("torchvision")
        cls.numpy = importlib.import_module("numpy")
        cls.scipy = importlib.import_module("scipy")
        cls.metadata = importlib.import_module("importlib.metadata")
        cls.records = conda_records()

        from src.core import YAMLConfig
        from src.solver import TASKS

        cls.yaml_config = YAMLConfig(str(CONFIG), PResNet={"pretrained": False})
        cls.tasks = TASKS
        cls.download_calls = []

        def forbidden_load(*args, **kwargs):
            cls.download_calls.append("torch.load")
            raise AssertionError("checkpoint loading is forbidden")

        def forbidden_download(*args, **kwargs):
            cls.download_calls.append("torch.hub.load_state_dict_from_url")
            raise AssertionError("weight download is forbidden")

        def forbidden_cuda(*args, **kwargs):
            raise AssertionError("CUDA initialization is forbidden")

        old_load = cls.torch.load
        old_download = cls.torch.hub.load_state_dict_from_url
        old_lazy_init = cls.torch.cuda._lazy_init
        cls.torch.load = forbidden_load
        cls.torch.hub.load_state_dict_from_url = forbidden_download
        cls.torch.cuda._lazy_init = forbidden_cuda
        try:
            cls.model = cls.yaml_config.model
        finally:
            cls.torch.load = old_load
            cls.torch.hub.load_state_dict_from_url = old_download
            cls.torch.cuda._lazy_init = old_lazy_init

    def test_interpreter_and_hidden_cuda(self):
        self.assertEqual(Path(sys.prefix).name, R2_NAME)
        self.assertEqual(sys.version_info[:3], (3, 10, 16))
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "")
        self.assertFalse(self.torch.cuda.is_available())
        self.assertEqual(self.torch.cuda.device_count(), 0)
        self.assertFalse(self.torch.cuda.is_initialized())

    def test_core_versions_and_channels(self):
        self.assertEqual(self.torch.__version__, "2.4.1")
        self.assertEqual(self.torchvision.__version__, "0.19.1")
        self.assertEqual(self.torch.version.cuda, "12.4")
        self.assertEqual(self.numpy.__version__, "1.26.4")
        self.assertEqual(self.scipy.__version__, "1.13.1")
        expected = {
            "python": ("3.10.16", "he870216_1"),
            "pytorch": ("2.4.1", "py3.10_cuda12.4_cudnn9.1.0_0"),
            "torchvision": ("0.19.1", "py310_cu124"),
            "pytorch-cuda": ("12.4", "hc786d27_7"),
            "torchtriton": ("3.0.0", "py310"),
            "mkl": ("2023.1.0", "h213fc3f_46344"),
            "intel-openmp": ("2023.1.0", "hdb19cb5_46306"),
            "numpy": ("1.26.4", "py310h5f9d8c6_0"),
            "scipy": ("1.13.1", "py310h5f9d8c6_1"),
        }
        for name, (version, build) in expected.items():
            self.assertIn(name, self.records)
            record = self.records[name]
            self.assertEqual((record["version"], record["build"]), (version, build))
        self.assertIn("pytorch", self.records["pytorch"]["channel"])
        self.assertIn("pytorch", self.records["torchvision"]["channel"])
        self.assertIn("repo.anaconda.com", self.records["mkl"]["channel"])
        self.assertIn("repo.anaconda.com", self.records["intel-openmp"]["channel"])

    def test_required_python_dependencies(self):
        expected = {
            "pycocotools": "2.0.11",
            "faster-coco-eval": "1.6.7",
            "pytest": "8.4.2",
            "PyYAML": "6.0.3",
            "tensorboard": "2.21.0",
            "tqdm": "4.70.0",
            "fsspec": "2026.6.0",
        }
        for name, version in expected.items():
            self.assertEqual(self.metadata.version(name), version)

    def test_pip_check(self):
        result = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": ""},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_libtorch_cpu_has_no_unresolved_symbols(self):
        libtorch_cpu = Path(self.torch.__file__).parent / "lib" / "libtorch_cpu.so"
        result = subprocess.run(
            ["ldd", "-r", str(libtorch_cpu)],
            capture_output=True,
            text=True,
            check=False,
        )
        output = result.stdout + result.stderr
        self.assertNotRegex(output, re.compile(r"undefined symbol|not found"))

    def test_all_vendor_python_sources_compile_in_memory(self):
        sources = sorted(VENDOR.rglob("*.py"))
        self.assertGreater(len(sources), 0)
        for path in sources:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")

    def test_vendor_core_import_and_r18_config(self):
        self.assertIn("detection", self.tasks)
        self.assertTrue(CONFIG.is_file())
        self.assertEqual(self.yaml_config.yaml_cfg["PResNet"]["depth"], 18)
        self.assertFalse(self.yaml_config.yaml_cfg["PResNet"]["pretrained"])

    def test_r18_model_is_cpu_finite_and_no_download(self):
        params = list(self.model.parameters())
        self.assertGreater(sum(param.numel() for param in params), 0)
        self.assertTrue(all(bool(self.torch.isfinite(param.detach()).all()) for param in params))
        self.assertEqual(self.download_calls, [])
        self.assertFalse(self.torch.cuda.is_initialized())
        self.assertIsNone(getattr(self.yaml_config, "_train_dataloader", None))
        self.assertIsNone(getattr(self.yaml_config, "_val_dataloader", None))


if __name__ == "__main__":
    unittest.main()
