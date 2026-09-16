"""CPU subprocess tests for EOF-complete seed-2 controller log receipts."""
from __future__ import annotations

import errno
import hashlib
import os
from pathlib import Path
import sys
import threading
import time

import pytest

from sparse_rtdetr.baseline import training_v2b_seed2_process_logs as logs


def _capture(tmp_path):
    return logs.PipeLogCapture(tmp_path / "stdout.log", tmp_path / "stderr.log")


def _env():
    return dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
                PYTHONWARNINGS="ignore::ResourceWarning")


def _wait_for(predicate, seconds=3.0):
    deadline = time.monotonic() + seconds
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("CPU subprocess condition exceeded bounded wait")
        time.sleep(0.005)


def _descendant_parent(release, *, stdout=b""):
    descendant = (
        "import os,time\nfrom pathlib import Path\n"
        f"release=Path({str(release)!r})\n"
        "deadline=time.monotonic()+3\n"
        "while not release.exists() and time.monotonic()<deadline: time.sleep(.005)\n"
        "os.write(2,b'late child stderr\\n')\n"
    )
    return (
        "import os,subprocess,sys\n"
        f"subprocess.Popen([sys.executable,'-c',{descendant!r}],stdout=subprocess.DEVNULL)\n"
        f"os.write(1,{stdout!r})\n"
        "os.write(2,b'parent stderr\\n')\n"
    )


def _assert_reference(reference, path, expected):
    assert path.read_bytes() == expected
    assert reference == {
        "path": str(path), "sha256": hashlib.sha256(expected).hexdigest(),
        "size_bytes": len(expected), "sha256_scope": "complete_file_bytes"}


def test_both_pipes_drain_beyond_pipe_capacity_and_complete_bytes(tmp_path):
    capture = _capture(tmp_path)
    code = "import os\nfor _ in range(32):\n os.write(1,b'x'*8192)\n os.write(2,b'y'*8192)\n"
    process = capture.spawn([sys.executable, "-c", code], env=_env())
    assert process.wait(timeout=5) == 0
    result = capture.finish(timeout_seconds=3)
    assert result["status"] == result["capture_status"] == "PASS"
    assert result["signals_sent"] is False
    assert not result["background_capture_pending"]
    for name, byte in (("stdout", b"x"), ("stderr", b"y")):
        _assert_reference(result["references"][name], tmp_path / f"{name}.log", byte * 8192 * 32)
        assert result["streams"][name]["eof_seen"]
        assert result["streams"][name]["file_closed"]
    result["references"]["stdout"]["sha256"] = "caller mutation"
    assert capture.finish()["references"]["stdout"]["sha256"] != "caller mutation"


def test_parent_exit_waits_for_descendant_stderr_eof(tmp_path):
    release = tmp_path / "release"
    capture = _capture(tmp_path)
    process = capture.spawn([sys.executable, "-c", _descendant_parent(release)], env=_env())
    try:
        assert process.wait(timeout=3) == 0
        _wait_for(lambda: capture.status()["streams"]["stderr"]["bytes_written"] >= 14)
        assert not capture.status()["streams"]["stderr"]["eof_seen"]
        assert not capture.status()["streams"]["stderr"]["file_closed"]
        release.touch()
        result = capture.finish(timeout_seconds=3)
    finally:
        release.touch(exist_ok=True)
    assert result["capture_status"] == "PASS"
    _assert_reference(result["references"]["stderr"], tmp_path / "stderr.log",
                      b"parent stderr\nlate child stderr\n")


def test_eof_timeout_retains_late_output_and_cannot_later_become_pass(tmp_path):
    release = tmp_path / "release"
    capture = _capture(tmp_path)
    process = capture.spawn([sys.executable, "-c", _descendant_parent(release)], env=_env())
    primary = {"kind": "hardware", "reason": "foreign GPU compute observed"}
    try:
        assert process.wait(timeout=3) == 0
        result = capture.finish(timeout_seconds=0.03, primary_failure=primary)
        assert result["status"] == result["capture_status"] == "FAIL"
        assert result["references"] is None
        assert result["failure"] == result["primary_failure"] == primary
        assert any(row["operation"] == "EOF_TIMEOUT" for row in result["errors"])
        assert result["elapsed_finish_seconds"] < 0.5
        assert result["background_capture_pending"]
    finally:
        release.touch(exist_ok=True)
    _wait_for(lambda: capture.status()["finalization_done"])
    assert (tmp_path / "stderr.log").read_bytes().endswith(b"late child stderr\n")
    assert capture.finish(timeout_seconds=3) == result
    primary["reason"] = "caller changed its object"
    assert capture.finish()["failure"]["reason"] == "foreign GPU compute observed"


def test_short_writes_and_interrupted_write_preserve_full_bytes(tmp_path, monkeypatch):
    actual = logs._write_once
    calls = 0

    def short_write(fd, data):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise InterruptedError()
        return actual(fd, data[:7])

    monkeypatch.setattr(logs, "_write_once", short_write)
    capture = _capture(tmp_path)
    process = capture.spawn([sys.executable, "-c",
                             "import os; os.write(1,b'a'*4096); os.write(2,b'b'*3096)"],
                            env=_env())
    assert process.wait(timeout=3) == 0
    result = capture.finish(timeout_seconds=3)
    assert result["capture_status"] == "PASS"
    assert calls > 100
    _assert_reference(result["references"]["stdout"], tmp_path / "stdout.log", b"a" * 4096)
    _assert_reference(result["references"]["stderr"], tmp_path / "stderr.log", b"b" * 3096)


@pytest.mark.parametrize("returncode,primary", [
    (0, {"kind": "numeric", "reason": "nonfinite loss"}),
    (9, None),
])
def test_write_error_drains_without_deadlock_and_preserves_first_failure(
        tmp_path, monkeypatch, returncode, primary):
    def disk_full(fd, data):
        raise OSError(errno.ENOSPC, "synthetic CPU disk-full injection")

    monkeypatch.setattr(logs, "_write_once", disk_full)
    capture = _capture(tmp_path)
    process = capture.spawn(
        [sys.executable, "-c",
         f"import os\nfor _ in range(32):\n os.write(1,b'a'*8192)\n os.write(2,b'b'*8192)\nos._exit({returncode})"],
        env=_env())
    assert process.wait(timeout=5) == returncode
    assert capture.status()["errors"]
    result = capture.finish(timeout_seconds=3, primary_failure=primary)
    assert result["capture_status"] == "FAIL"
    assert result["references"] is None
    assert all(stream["eof_seen"] for stream in result["streams"].values())
    assert all(stream["bytes_read"] == 32 * 8192 for stream in result["streams"].values())
    assert any(row["operation"] == "LOG_WRITE_ERROR" for row in result["errors"])
    assert result["failure"] == (primary if primary else {"operation": "WORKER_EXIT", "returncode": 9})


@pytest.mark.parametrize("mutation", ["replace", "in_place"])
def test_log_path_or_byte_mutation_cannot_receive_complete_reference(tmp_path, mutation):
    release = tmp_path / "release"
    capture = _capture(tmp_path)
    process = capture.spawn(
        [sys.executable, "-c", _descendant_parent(release, stdout=b"original")], env=_env())
    try:
        assert process.wait(timeout=3) == 0
        _wait_for(lambda: capture.status()["streams"]["stdout"]["bytes_written"] == 8)
        path = tmp_path / "stdout.log"
        if mutation == "replace":
            path.rename(tmp_path / "preserved-original.log")
        path.write_bytes(b"tampered")
        release.touch()
        result = capture.finish(timeout_seconds=3)
    finally:
        release.touch(exist_ok=True)
    assert result["capture_status"] == "FAIL"
    assert result["references"] is None
    assert any(row["operation"] in {"LOG_PATH_IDENTITY_ERROR", "LOG_FINALIZATION_ERROR"}
               for row in result["errors"])


def test_worker_alive_cannot_be_certified_even_after_both_pipes_close(tmp_path):
    release = tmp_path / "release"
    capture = _capture(tmp_path)
    code = (
        "import os,time\nfrom pathlib import Path\n"
        "os.close(1);os.close(2)\n"
        f"release=Path({str(release)!r});deadline=time.monotonic()+3\n"
        "while not release.exists() and time.monotonic()<deadline: time.sleep(.005)\n"
    )
    process = capture.spawn([sys.executable, "-c", code], env=_env())
    try:
        _wait_for(lambda: capture.status()["drain_done"])
        assert process.poll() is None
        result = capture.finish(timeout_seconds=0.2)
        assert result["capture_status"] == "FAIL"
        assert result["references"] is None
        assert any(row["operation"] == "WORKER_STILL_RUNNING" for row in result["errors"])
        assert process.poll() is None  # The helper did not signal the worker.
    finally:
        release.touch(exist_ok=True)
        assert process.wait(timeout=3) == 0


def test_fsync_error_withholds_hash_and_does_not_mask_primary_failure(tmp_path, monkeypatch):
    def broken_sync(fd):
        raise OSError(errno.EIO, "synthetic CPU fsync failure")

    monkeypatch.setattr(logs, "_fsync", broken_sync)
    capture = _capture(tmp_path)
    process = capture.spawn([sys.executable, "-c", "print('complete pipe bytes')"], env=_env())
    assert process.wait(timeout=3) == 0
    result = capture.finish(timeout_seconds=3, primary_failure="native guardian failed")
    assert result["capture_status"] == "FAIL"
    assert result["failure"] == "native guardian failed"
    assert result["references"] is None
    assert all(row["file_closed"] for row in result["streams"].values())


def test_blocked_file_finalization_respects_timeout_and_never_upgrades(tmp_path, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    actual = logs._fsync

    def delayed_sync(fd):
        entered.set()
        release.wait(timeout=2)
        return actual(fd)

    monkeypatch.setattr(logs, "_fsync", delayed_sync)
    capture = _capture(tmp_path)
    process = capture.spawn([sys.executable, "-c", "print('bytes')"], env=_env())
    assert process.wait(timeout=3) == 0
    try:
        result = capture.finish(timeout_seconds=0.03)
        assert entered.is_set()
        assert result["capture_status"] == "FAIL"
        assert result["references"] is None
        assert result["elapsed_finish_seconds"] < 0.5
        assert any(row["operation"] == "FINALIZATION_TIMEOUT" for row in result["errors"])
    finally:
        release.set()
    _wait_for(lambda: capture.status()["finalization_done"])
    assert capture.finish(timeout_seconds=3) == result


def test_exclusive_open_never_overwrites_or_deletes_partial_attempt(tmp_path):
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"
    stderr.write_bytes(b"immutable previous evidence")
    with pytest.raises(FileExistsError):
        logs.PipeLogCapture(stdout, stderr)
    assert stderr.read_bytes() == b"immutable previous evidence"
    assert stdout.read_bytes() == b""
    with pytest.raises(ValueError, match="distinct"):
        logs.PipeLogCapture(tmp_path / "same", tmp_path / "same")
    assert not (tmp_path / "same").exists()


def test_symlink_output_is_rejected_without_touching_target(tmp_path):
    target = tmp_path / "history"
    target.write_bytes(b"keep")
    (tmp_path / "stdout.log").symlink_to(target)
    with pytest.raises(FileExistsError):
        _capture(tmp_path)
    assert target.read_bytes() == b"keep"
    assert not (tmp_path / "stderr.log").exists()


def test_spawn_failure_is_one_attempt_and_has_no_complete_log_refs(tmp_path):
    capture = _capture(tmp_path)
    with pytest.raises(FileNotFoundError):
        capture.spawn([str(tmp_path / "no-such-executable")], env=_env())
    result = capture.finish(timeout_seconds=3)
    assert result["capture_status"] == "FAIL"
    assert result["references"] is None
    assert result["failure"]["operation"] == "SPAWN_ERROR"
    assert all(row["file_closed"] for row in result["streams"].values())
    with pytest.raises(RuntimeError, match="one worker"):
        capture.spawn([sys.executable, "-c", "raise SystemExit(0)"], env=_env())


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
def test_invalid_timeout_is_terminal_failure(tmp_path, timeout):
    capture = _capture(tmp_path)
    process = capture.spawn([sys.executable, "-c", "pass"], env=_env())
    assert process.wait(timeout=3) == 0
    result = capture.finish(timeout_seconds=timeout, primary_failure="preserve this failure")
    assert result["capture_status"] == "FAIL"
    assert result["references"] is None
    assert result["failure"] == "preserve this failure"
    assert any(row["operation"] == "INVALID_FINISH_TIMEOUT" for row in result["errors"])
    _wait_for(lambda: capture.status()["finalization_done"])


def test_selector_creation_error_is_visible_and_fails_closed(tmp_path, monkeypatch):
    def unavailable_selector():
        raise OSError(errno.EMFILE, "synthetic CPU descriptor exhaustion")

    monkeypatch.setattr(logs.selectors, "DefaultSelector", unavailable_selector)
    capture = _capture(tmp_path)
    process = capture.spawn([sys.executable, "-c", "pass"], env=_env())
    assert process.wait(timeout=3) == 0
    result = capture.finish(timeout_seconds=3)
    assert result["capture_status"] == "FAIL"
    assert result["references"] is None
    assert any(row["operation"] == "DRAIN_ERROR" for row in result["errors"])
    assert all(row["file_closed"] for row in result["streams"].values())


def test_zero_progress_write_fails_closed_instead_of_looping(tmp_path, monkeypatch):
    monkeypatch.setattr(logs, "_write_once", lambda fd, data: 0)
    capture = _capture(tmp_path)
    process = capture.spawn([sys.executable, "-c", "print('payload')"], env=_env())
    assert process.wait(timeout=3) == 0
    result = capture.finish(timeout_seconds=3)
    assert result["capture_status"] == "FAIL"
    assert result["references"] is None
    assert any(row["operation"] == "LOG_WRITE_ERROR" for row in result["errors"])


def test_explicit_session_and_stdin_are_applied_to_the_real_child(tmp_path):
    import json
    import subprocess

    capture = _capture(tmp_path)
    code = (
        "import os,json\n"
        "print(json.dumps(dict(pid=os.getpid(),sid=os.getsid(0),stdin_eof=os.read(0,1)==b'')))"
    )
    process = capture.spawn([sys.executable, "-c", code], env=_env(),
                            stdin=subprocess.DEVNULL, start_new_session=True)
    assert process.wait(timeout=3) == 0
    result = capture.finish(timeout_seconds=3)
    observed = json.loads((tmp_path / "stdout.log").read_bytes())
    assert observed == {"pid": process.pid, "sid": process.pid, "stdin_eof": True}
    assert result["worker_pid"] == process.pid
    assert result["status"] == "PASS"


def test_post_popen_thread_start_failure_retains_owned_child_and_cleanup_bytes(tmp_path, monkeypatch):
    actual = threading.Thread.start
    calls = 0

    def first_start_fails(thread):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("synthetic CPU capture thread start failure")
        return actual(thread)

    monkeypatch.setattr(threading.Thread, "start", first_start_fails)
    capture = _capture(tmp_path)
    process = capture.spawn([sys.executable, "-c", "print('preserved buffered bytes')"], env=_env())
    assert capture.status()["worker_pid"] == process.pid
    assert any(row["operation"] == "DRAIN_THREAD_START_ERROR" for row in capture.status()["errors"])
    assert process.wait(timeout=3) == 0
    result = capture.finish(timeout_seconds=3, primary_failure="controller observed capture failure")
    assert result["worker_pid"] == process.pid
    assert result["status"] == result["capture_status"] == "FAIL"
    assert result["failure"] == "controller observed capture failure"
    assert result["references"] is None
    assert (tmp_path / "stdout.log").read_bytes() == b"preserved buffered bytes\n"
    assert calls == 2  # One failed capture startup, one cleanup-only drain.


@pytest.mark.parametrize("kwargs", [{"stdin": -1}, {"start_new_session": 1}])
def test_spawn_settings_cannot_silently_change_semantics(tmp_path, kwargs):
    capture = _capture(tmp_path)
    with pytest.raises(ValueError):
        capture.spawn([sys.executable, "-c", "pass"], env=_env(), **kwargs)
    result = capture.finish(timeout_seconds=3)
    assert result["worker_pid"] is None
    assert result["capture_status"] == "FAIL"
    assert result["references"] is None
