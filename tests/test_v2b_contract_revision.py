import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("seal_v2b_contract_revision", ROOT / "tools" / "seal_v2b_contract_revision.py")
seal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seal)


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_runtime_locator_is_not_in_authority_identity(tmp_path):
    _write(tmp_path / "a.py", b"a")
    first = seal._authority({"authority_id": "x", "logical_type": "fixture",
                             "root": str(tmp_path), "include_files": ["a.py"]})
    second = json.loads(json.dumps(first))
    second["runtime_locator"]["path"] = "/another/mount"
    assert first["identity_sha256"] == second["identity_sha256"]


def test_runtime_locator_is_not_in_evaluation_identity():
    body = {"checkpoints": [{"runtime_locator": {"path": "/a"},
                              "checkpoint": {"checkpoint_id": "x"}}]}
    assert seal.canonical_sha(seal.identity_without_runtime(body)) == seal.canonical_sha(
        seal.identity_without_runtime({"checkpoints": [{"runtime_locator": {"path": "/b"},
                                                           "checkpoint": {"checkpoint_id": "x"}}]}))


def test_inventory_is_sorted_and_content_bound(tmp_path):
    _write(tmp_path / "b.bin", b"b")
    _write(tmp_path / "a.bin", b"a")
    value = seal.inventory(tmp_path, ["b.bin", "a.bin"])
    assert [row["relative_path"] for row in value["rows"]] == ["a.bin", "b.bin"]
    assert value["file_count"] == 2 and value["total_size_bytes"] == 2


def test_missing_or_symlink_files_fail(tmp_path):
    with pytest.raises(seal.SealError):
        seal.inventory(tmp_path, ["missing.bin"])
    _write(tmp_path / "real.bin", b"x")
    (tmp_path / "link.bin").symlink_to(tmp_path / "real.bin")
    with pytest.raises(seal.SealError):
        seal.inventory(tmp_path, ["link.bin"])


def test_source_paths_are_explicit(tmp_path):
    _write(tmp_path / "src.py", b"source")
    result = seal._source(tmp_path, {"files": ["src.py"], "source_id": "s"})
    assert result["files"][0]["relative_path"] == "src.py"
    with pytest.raises(seal.SealError):
        seal._source(tmp_path, {"files": ["not-there.py"]})
