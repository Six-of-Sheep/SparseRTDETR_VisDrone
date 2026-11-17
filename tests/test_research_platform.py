"""Integration checks for real launch locks; never install an fcntl stub."""
import importlib
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
for folder in ("query_local", "scale_baseline"):
    sys.path.insert(0, str(ROOT / "tools" / folder))


@pytest.mark.parametrize("name", ["formal_run", "s15_formal_run", "mix_formal_run"])
def test_real_launch_lock_excludes_second_holder_and_releases(name, tmp_path, monkeypatch):
    module = importlib.import_module(name)
    monkeypatch.setattr(module, "ROUND", tmp_path)
    if sys.platform == "win32":
        with pytest.raises(ModuleNotFoundError, match="fcntl"):
            module.lock_round()
        assert not list(tmp_path.iterdir())
        return
    with module.lock_round():
        with pytest.raises(RuntimeError, match="lock is held"):
            module.lock_round()
    with module.lock_round() as handle:
        assert not handle.closed
