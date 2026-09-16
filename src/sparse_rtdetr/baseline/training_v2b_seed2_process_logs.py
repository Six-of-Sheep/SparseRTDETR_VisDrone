"""EOF-bound capture for the new seed-2 controller; no signals or GPU calls.

The controller owns worker supervision and cleanup. Call finish only after the
worker exits. Descendants may inherit its pipes: worker exit alone never
certifies a log. Timeout permanently fails this capture but leaves the daemon
drain running while this controller lives so late output is not deliberately
cut off. Controller exit can truncate a pending capture. No failed capture
returns file hashes. Historical log receipts are never repaired here.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import math
import os
from pathlib import Path
import selectors
import stat
import subprocess
import threading
import time
from typing import Any, Mapping, Sequence

_write_once = os.write
_fsync = os.fsync


@dataclass
class _Stream:
    path: Path
    fd: int | None
    inode: tuple[int, int]
    pipe: Any = None
    bytes_read: int = 0
    bytes_written: int = 0
    eof_seen: bool = False
    file_closed: bool = False
    write_failed: bool = False
    digest: Any = field(default_factory=hashlib.sha256)
    reference: dict[str, Any] | None = None


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


class PipeLogCapture:
    """Create two exclusive files, spawn once, and certify only after real EOF.

    status() exposes capture errors for the controller's supervision loop.
    finish() returns a detached, permanently terminal result. Its
    primary_failure argument is preserved, and takes precedence over worker
    exit and capture failures in failure. Capture errors are always retained
    separately. references is null unless both streams were completely
    drained, fsynced, checked against captured bytes, and closed.
    """

    def __init__(self, stdout_path: str | os.PathLike[str],
                 stderr_path: str | os.PathLike[str]) -> None:
        paths = {"stdout": Path(stdout_path).absolute(),
                 "stderr": Path(stderr_path).absolute()}
        if paths["stdout"] == paths["stderr"]:
            raise ValueError("stdout and stderr require distinct absent paths")
        for path in paths.values():
            if path.parent.resolve(strict=True) != path.parent:
                raise ValueError("log parent must be an existing canonical directory")
        self._lock = threading.Lock()
        self._drain_done = threading.Event()
        self._finalize_requested = threading.Event()
        self._done = threading.Event()
        self._streams: dict[str, _Stream] = {}
        self._errors: list[dict[str, Any]] = []
        self._process: subprocess.Popen[bytes] | None = None
        self._thread: threading.Thread | None = None
        self._spawn_attempted = False
        self._terminal: dict[str, Any] | None = None
        try:
            for name, path in paths.items():
                fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL
                             | getattr(os, "O_NOFOLLOW", 0)
                             | getattr(os, "O_CLOEXEC", 0), 0o600)
                try:
                    info = os.fstat(fd)
                except BaseException:
                    os.close(fd)
                    raise
                self._streams[name] = _Stream(path, fd, (info.st_dev, info.st_ino))
        except BaseException:
            # Keep any newly created partial file as evidence; never unlink it.
            for stream in self._streams.values():
                if stream.fd is not None:
                    os.close(stream.fd)
                    stream.fd = None
            raise

    def _error(self, operation: str, error: BaseException | str, *,
               stream: str | None = None) -> None:
        row = {"operation": operation, "stream": stream,
               "error_type": type(error).__name__ if isinstance(error, BaseException) else None,
               "detail": str(error)}
        with self._lock:
            self._errors.append(row)

    def spawn(self, argv: Sequence[str], *,
              cwd: str | os.PathLike[str] | None = None,
              env: Mapping[str, str] | None = None,
              stdin: int = subprocess.DEVNULL,
              start_new_session: bool = False) -> subprocess.Popen[bytes]:
        if self._spawn_attempted or self._terminal is not None:
            raise RuntimeError("capture permits exactly one worker spawn attempt")
        if (isinstance(argv, (str, bytes)) or not argv
                or any(type(item) is not str or not item for item in argv)):
            raise ValueError("worker argv must be a nonempty sequence of strings")
        if type(stdin) is not int or stdin != subprocess.DEVNULL:
            raise ValueError("worker stdin must remain DEVNULL")
        if type(start_new_session) is not bool:
            raise ValueError("start_new_session must be an explicit bool")
        self._spawn_attempted = True
        try:
            self._process = subprocess.Popen(
                list(argv), cwd=cwd, env=dict(env) if env is not None else None,
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, bufsize=0, close_fds=True,
                start_new_session=start_new_session)
        except BaseException as error:
            self._error("SPAWN_ERROR", error)
            self._drain_done.set()
            self._finalize_requested.set()
            self._begin_thread(self._finalize_files, "CAPTURE_CLEANUP_THREAD_START_ERROR")
            raise
        self._streams["stdout"].pipe = self._process.stdout
        self._streams["stderr"].pipe = self._process.stderr
        # Popen has succeeded: always return this owned child even if capture
        # startup fails. The supervisor sees errors and retains its PID handle.
        self._begin_thread(self._drain, "DRAIN_THREAD_START_ERROR")
        return self._process

    def _begin_thread(self, target: Any, operation: str) -> None:
        candidate = None
        try:
            candidate = threading.Thread(target=target, daemon=True)
            self._thread = candidate
            candidate.start()
        except BaseException as error:
            self._error(operation, error)
            # A rare interruption may occur after the thread really started.
            # Keep that thread if alive; never launch a duplicate consumer.
            self._thread = candidate if candidate is not None and candidate.is_alive() else None

    def _write(self, name: str, data: bytes) -> None:
        stream = self._streams[name]
        view = memoryview(data)
        while view:
            try:
                count = _write_once(stream.fd, view)
            except InterruptedError:
                continue
            if type(count) is not int or not 0 < count <= len(view):
                raise OSError("invalid or zero-length log write")
            stream.digest.update(view[:count])
            with self._lock:
                stream.bytes_written += count
            view = view[count:]

    def _close_pipe(self, name: str) -> None:
        pipe = self._streams[name].pipe
        if pipe is not None and not pipe.closed:
            try:
                pipe.close()
            except Exception as error:
                self._error("PIPE_CLOSE_ERROR", error, stream=name)

    def _drain(self) -> None:
        selector = None
        try:
            selector = selectors.DefaultSelector()
            for name, stream in self._streams.items():
                os.set_blocking(stream.pipe.fileno(), False)
                selector.register(stream.pipe.fileno(), selectors.EVENT_READ, name)
            while selector.get_map():
                for key, _ in selector.select(timeout=0.05):
                    name = key.data
                    stream = self._streams[name]
                    try:
                        data = os.read(key.fd, 65536)
                    except (BlockingIOError, InterruptedError):
                        continue
                    except Exception as error:
                        self._error("PIPE_READ_ERROR", error, stream=name)
                        selector.unregister(key.fd)
                        self._close_pipe(name)
                        continue
                    if not data:
                        with self._lock:
                            stream.eof_seen = True
                        selector.unregister(key.fd)
                        self._close_pipe(name)
                        continue
                    with self._lock:
                        stream.bytes_read += len(data)
                    if not stream.write_failed:
                        try:
                            self._write(name, data)
                        except Exception as error:
                            stream.write_failed = True
                            self._error("LOG_WRITE_ERROR", error, stream=name)
                    # After a disk error keep draining, rather than deadlocking
                    # the worker. The supervisor sees errors via status().
        except Exception as error:
            self._error("DRAIN_ERROR", error)
        finally:
            if selector is not None:
                try:
                    selector.close()
                except Exception as error:
                    self._error("SELECTOR_CLOSE_ERROR", error)
            for name in self._streams:
                self._close_pipe(name)
            self._drain_done.set()
            # Even early EOF cannot finalize before the controller asks after
            # observing worker exit. No blocking file I/O occurs in finish().
            self._finalize_requested.wait()
            self._finalize_files()

    def _finalize_stream(self, name: str) -> None:
        stream = self._streams[name]
        fd = stream.fd
        candidate = None
        verified_stat = None
        try:
            _fsync(fd)  # Writes are unbuffered os.write; this flushes file data.
            before = os.fstat(fd)
            if (not stat.S_ISREG(before.st_mode)
                    or (before.st_dev, before.st_ino) != stream.inode):
                raise OSError("opened log is no longer the original regular file")
            if stream.eof_seen and not stream.write_failed:
                if before.st_size != stream.bytes_written or stream.bytes_read != stream.bytes_written:
                    raise OSError("captured and persisted byte counts differ")
                os.lseek(fd, 0, os.SEEK_SET)
                digest = hashlib.sha256()
                count = 0
                while True:
                    data = os.read(fd, 1024 * 1024)
                    if not data:
                        break
                    digest.update(data)
                    count += len(data)
                after = os.fstat(fd)
                if (_identity(before) != _identity(after)
                        or count != stream.bytes_written
                        or digest.digest() != stream.digest.digest()):
                    raise OSError("log changed or differs from fully captured bytes")
                candidate = {"path": str(stream.path), "sha256": digest.hexdigest(),
                             "size_bytes": count, "sha256_scope": "complete_file_bytes"}
                verified_stat = after
        except Exception as error:
            self._error("LOG_FINALIZATION_ERROR", error, stream=name)
        finally:
            try:
                os.close(fd)
            except Exception as error:
                self._error("LOG_CLOSE_ERROR", error, stream=name)
            else:
                with self._lock:
                    stream.file_closed = True
            stream.fd = None
        if candidate is not None and stream.file_closed:
            try:
                visible = os.stat(stream.path, follow_symlinks=False)
                if not stat.S_ISREG(visible.st_mode) or _identity(visible) != _identity(verified_stat):
                    raise OSError("closed log path was replaced or changed")
                stream.reference = candidate
            except Exception as error:
                self._error("LOG_PATH_IDENTITY_ERROR", error, stream=name)

    def _finalize_files(self) -> None:
        try:
            for name in self._streams:
                self._finalize_stream(name)
            for parent in {stream.path.parent for stream in self._streams.values()}:
                fd = None
                try:
                    fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                    _fsync(fd)
                except Exception as error:
                    self._error("LOG_DIRECTORY_SYNC_ERROR", error)
                finally:
                    if fd is not None:
                        try:
                            os.close(fd)
                        except Exception as error:
                            self._error("LOG_DIRECTORY_CLOSE_ERROR", error)
        except Exception as error:
            self._error("FINALIZATION_ERROR", error)
        finally:
            self._done.set()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {"errors": deepcopy(self._errors),
                    "drain_done": self._drain_done.is_set(),
                    "finalization_done": self._done.is_set(),
                    "worker_pid": self._process.pid if self._process is not None else None,
                    "streams": {
                        name: {"path": str(stream.path),
                               "eof_seen": stream.eof_seen,
                               "bytes_read": stream.bytes_read,
                               "bytes_written": stream.bytes_written,
                               "file_closed": stream.file_closed}
                        for name, stream in self._streams.items()}}

    def finish(self, *, timeout_seconds: float = 10.0,
               primary_failure: str | Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Bound the wait; never signal a process or replace an earlier failure.

        A timeout is terminal even if the daemon later reaches EOF. Its files
        may keep growing and therefore receive no complete-file references.
        The controller must retain the returned failure and partial paths.
        A pending daemon can run only while its controller is alive; process
        exit may leave those files incomplete and not durably finalized.
        """
        if self._terminal is not None:
            return deepcopy(self._terminal)
        started = time.monotonic()
        if (type(timeout_seconds) not in (int, float)
                or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
            self._error("INVALID_FINISH_TIMEOUT", "timeout must be finite and positive")
            timeout_seconds = 0
        returncode = None
        if self._process is None:
            self._error("WORKER_NOT_STARTED", "no worker exists for this capture")
        else:
            try:
                returncode = self._process.poll()
            except Exception as error:
                self._error("WORKER_STATUS_ERROR", error)
            if returncode is None:
                self._error("WORKER_STILL_RUNNING", "controller must observe worker exit before finish")
        if self._thread is None:
            # At most one cleanup-only attempt after a failed thread startup.
            # The original capture error remains terminal; no worker respawns.
            if self._process is None:
                self._drain_done.set()
            self._begin_thread(self._finalize_files if self._process is None else self._drain,
                               "CAPTURE_CLEANUP_THREAD_START_ERROR")
        self._finalize_requested.set()
        remaining = max(0.0, timeout_seconds - (time.monotonic() - started))
        if not self._done.wait(remaining):
            state = self.status()
            code = ("EOF_TIMEOUT" if not all(row["eof_seen"] for row in state["streams"].values())
                    else "FINALIZATION_TIMEOUT")
            self._error(code, "capture did not reach both EOF and durable close before the deadline")
        state = self.status()
        capture_pass = (self._done.is_set() and not state["errors"]
                        and returncode is not None
                        and all(stream.eof_seen and stream.file_closed and stream.reference is not None
                                for stream in self._streams.values()))
        failure = deepcopy(primary_failure)
        if failure is None and returncode not in (None, 0):
            failure = {"operation": "WORKER_EXIT", "returncode": returncode}
        if failure is None and not capture_pass:
            failure = deepcopy(state["errors"][0]) if state["errors"] else {
                "operation": "INCOMPLETE_CAPTURE", "detail": "no complete log references"}
        self._terminal = {
            "schema_version": 1, "status": "PASS" if capture_pass and failure is None else "FAIL",
            "capture_status": "PASS" if capture_pass else "FAIL",
            "primary_failure": deepcopy(primary_failure), "failure": failure,
            "worker_pid": state["worker_pid"],
            "worker_returncode": returncode, "errors": state["errors"],
            "streams": state["streams"], "drain_done": state["drain_done"],
            "finalization_done": state["finalization_done"],
            "background_capture_pending": not self._done.is_set(),
            "pending_capture_policy": "daemon drain requires a live controller; pending files may be incomplete and receive no complete-file SHA",
            "references": {name: deepcopy(stream.reference) for name, stream in self._streams.items()}
                          if capture_pass else None,
            "elapsed_finish_seconds": time.monotonic() - started,
            "signals_sent": False}
        return deepcopy(self._terminal)
