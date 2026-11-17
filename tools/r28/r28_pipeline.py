"""Input pipeline of the R28 formal loop (ENGINEERING_INPUT_OVERLAP_NEXT_005, TASK_R28_CALIBRATED_F_CONTROL_006 s.3).

Two paths deliver the same audited logical windows in the same order:

  serial    the reference path: the consumer thread itself iterates formal_data.windows (loader, identity checks,
            per-image SHA, whole-window digests) and compares the input record with the reference ledger.
  overlap   one producer thread iterates the SAME generator and performs the SAME checks while the consumer trains on
            the previous window.  A permit protocol bounds the look-ahead: the producer may prepare the next window
            only after the consumer has taken the previous one, so at most ONE prepared window exists besides the
            one being consumed (the DataLoader's own worker prefetch is unchanged and is accounted separately).

What does not change: the spawn DataLoader with workers=2 and its generator, the augmentation inside the worker
processes, every hash and its format, the comparison with the reference ledger before a window can be consumed, the
order update by update.  The producer thread touches no model, no CUDA call and no global RNG: it only iterates the
loader (whose sampler and base seed come from its own generator), hashes CPU bytes and compares dictionaries.

Ownership: a delivered window is sealed (storage pointer and version counter of the pixel tensor and of every target
tensor) after its audit; the consumer verifies the seal before use, so the bytes that were hashed are the bytes that
are trained on.  Nothing is committed by a prepared window: the committed cursor is the engine's update clock, and a
stream opened at a committed update rebuilds every later window deterministically, whatever was prepared before.

Errors: an exception of the producer (loader failure, identity or digest mismatch, reference mismatch) is handed to
the consumer in window order and raised there before the offending window can be consumed; a producer that dies
without delivering is detected instead of waiting forever.
"""
import os
import queue
import sys
import threading
import time

VERSION = "r28-input-pipeline-v1"
MODES = ("serial", "overlap")
READY_LIMIT = 1
WORKERS = 2
WAIT_SECONDS = 900.0
_END = object()


class PipelineError(RuntimeError):
    pass


def _require(ok, message):
    if not ok:
        raise PipelineError(message)


def seal(x, targets):
    """Identity of the delivered objects: storage pointer, version counter and shape of every tensor."""
    import torch
    rows = [("pixels", x.data_ptr(), x._version, tuple(x.shape))]
    for i, t in enumerate(targets):
        for k in sorted(t):
            v = t[k]
            if isinstance(v, torch.Tensor):
                rows.append((f"{i}/{k}", v.data_ptr(), v._version, tuple(v.shape)))
    return rows


class Item:
    __slots__ = ("x", "t", "inp", "seal", "timing")

    def __init__(self, x, t, inp, timing):
        self.x, self.t, self.inp, self.timing = x, t, inp, timing
        self.seal = seal(x, t)

    def verify(self):
        _require(seal(self.x, self.t) == self.seal, f"window {self.inp.get('update')} was modified after its audit")


class Clock:
    """Times formal_data.input_record (the audit: per-image SHA, whole-window digests) inside the generator, and with
    fine=True also the two whole-window digest functions it calls.  A wrapper calls the original with the original
    arguments and returns its result; the wrappers are removed when the stream is closed."""

    def __init__(self, formal_data, fine=False):
        self.module, self.fine, self.acc, self.undo, self.depth = formal_data, fine, {}, [], {}

    def _wrap(self, owner, name, key, inside=None):
        """inside: the key of an enclosing timed function; a call made while that one runs is not timed again."""
        original = getattr(owner, name)

        def wrapper(*a, **k):
            if inside is not None and self.depth.get(inside, 0):
                return original(*a, **k)
            started = time.perf_counter()
            self.depth[key] = self.depth.get(key, 0) + 1
            try:
                return original(*a, **k)
            finally:
                self.depth[key] -= 1
                self.acc[key] = self.acc.get(key, 0.0) + time.perf_counter() - started

        wrapper.__wrapped__ = original
        setattr(owner, name, wrapper)
        self.undo.append((owner, name, original))

    def install(self):
        _require(not hasattr(self.module.input_record, "__wrapped__"), "an input pipeline clock is already installed")
        self._wrap(self.module, "input_record", "audit_s")
        if self.fine:
            from sparse_rtdetr.r0 import engine as r0_engine
            self._wrap(r0_engine, "tensor_digest", "window_pixel_digest_s", inside="window_target_digest_s")
            self._wrap(r0_engine, "output_digest", "window_target_digest_s")

    def remove(self):
        for owner, name, original in reversed(self.undo):
            setattr(owner, name, original)
        self.undo = []

    def take(self):
        acc, self.acc = self.acc, {}
        return acc


def audited(windows, data, start, stop, per_epoch, check, clock=None):
    """The window sequence of the serial loop, update start+1 .. stop, as Items.  Runs in whichever thread iterates
    it.  `check(inp)` is the comparison with the reference ledger; it raises on a difference."""
    position = start
    while position < stop:
        begin = position
        gen = windows(data, begin, limit=min(stop - begin, per_epoch - begin % per_epoch))
        try:
            while True:
                t0 = time.perf_counter()
                try:
                    x, t, inp = next(gen)
                except StopIteration:
                    break
                t1 = time.perf_counter()
                _require(inp["update"] == position + 1, f"window {inp['update']} is not the next update of the schedule ({position + 1})")
                check(inp)
                t2 = time.perf_counter()
                acc = clock.take() if clock is not None else {}
                audit = acc.get("audit_s")
                timing = {"next_s": t1 - t0, "audit_s": audit, "fetch_s": None if audit is None else (t1 - t0) - audit,
                          "reference_check_s": t2 - t1, "ready_at": t2,
                          **{k: v for k, v in acc.items() if k != "audit_s"}}
                if "window_pixel_digest_s" in acc and audit is not None:
                    timing["per_image_hash_s"] = audit - acc["window_pixel_digest_s"] - acc.get("window_target_digest_s", 0.0)
                position += 1
                yield Item(x, t, inp, timing)
        finally:
            gen.close()
        _require(position != begin, "the loader made no progress")


class Stream:
    """Context manager and iterator over the audited windows of updates start+1 .. stop.

    for item in stream: the item has passed every audit and the reference check, and its seal has been verified.
    ready()             number of prepared windows not yet taken (0 or 1 in overlap mode, always 0 in serial mode).
    wait_ready()        overlap mode: block until the next window is prepared (or the producer ended / failed).
    """

    def __init__(self, mode, windows, data, start, stop, per_epoch, check, *, formal_data=None, fine=False):
        _require(mode in MODES, f"input pipeline mode: {mode!r}")
        _require(type(start) is int and type(stop) is int and start <= stop, "input pipeline span")
        self.mode, self.start, self.stop = mode, start, stop
        self.clock = Clock(formal_data, fine) if formal_data is not None else None
        self._source = audited(windows, data, start, stop, per_epoch, check, self.clock)
        self.taken = self.produced = 0
        self.max_prepared_ahead = 0
        self.wait_s = 0.0
        self._closed = False
        self._q = queue.Queue()
        self._permit = threading.Semaphore(READY_LIMIT)
        self._halt = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

    # ---- life cycle
    def __enter__(self):
        if self.clock is not None:
            self.clock.install()
        if self.mode == "overlap":
            self._thread = threading.Thread(target=self._produce, name="r28-input-producer", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        try:
            self.close()
        except PipelineError:
            if exc[0] is None:          # never replace the error that is already leaving the loop
                raise
        return False

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._halt.set()
        if self._thread is not None:
            self._permit.release()
            self._thread.join(timeout=120)
            _require(not self._thread.is_alive(), "the input producer did not stop")
        else:
            self._source.close()
        while True:                     # prepared but unconsumed windows are dropped: they were never committed
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        if self.clock is not None:
            self.clock.remove()

    # ---- producer thread (overlap mode only)
    def _produce(self):
        try:
            while True:
                while not self._permit.acquire(timeout=0.2):
                    if self._halt.is_set():
                        return
                if self._halt.is_set():
                    return
                try:
                    item = next(self._source)
                except StopIteration:
                    self._q.put(_END)
                    return
                with self._lock:
                    self.produced += 1
                    self.max_prepared_ahead = max(self.max_prepared_ahead, self.produced - self.taken)
                self._q.put(item)
        except BaseException as exc:    # handed to the consumer in order; never swallowed
            self._q.put(exc)
        finally:
            self._source.close()

    # ---- consumer side
    def ready(self):
        return sum(1 for v in list(self._q.queue) if isinstance(v, Item)) if self.mode == "overlap" else 0

    def wait_ready(self, timeout=WAIT_SECONDS):
        _require(self.mode == "overlap", "only the overlapped path prepares windows ahead")
        deadline = time.monotonic() + timeout
        while self._q.empty():
            _require(self._thread.is_alive() or not self._q.empty(), "the input producer ended without a prepared window")
            _require(time.monotonic() < deadline, "timed out waiting for a prepared window")
            time.sleep(0.01)
        return self.ready()

    def __iter__(self):
        return self

    def __next__(self):
        _require(not self._closed, "the input stream is closed")
        started = time.perf_counter()
        if self.mode == "serial":
            item = next(self._source)       # StopIteration ends the loop
            self.produced += 1
        else:
            deadline = time.monotonic() + WAIT_SECONDS
            while True:
                try:
                    item = self._q.get(timeout=0.5)
                    break
                except queue.Empty:
                    _require(self._thread.is_alive() or not self._q.empty(),
                             "the input producer ended without delivering a window or an error")
                    _require(time.monotonic() < deadline, "timed out waiting for the input producer")
            if item is _END:
                self._q.put(_END)
                raise StopIteration
            if isinstance(item, BaseException):
                self._q.put(item)           # the stream stays failed
                raise item
            with self._lock:
                self.taken += 1
            self._permit.release()          # the producer may now prepare the following window
        if self.mode == "serial":
            self.taken += 1
        waited = time.perf_counter() - started
        self.wait_s += waited
        item.verify()
        item.timing = {**item.timing, "wait_s": waited, "taken_at": time.perf_counter(), "pipeline": self.mode}
        return item

    def account(self):
        return {"mode": self.mode, "version": VERSION, "ready_limit": READY_LIMIT if self.mode == "overlap" else 0,
                "workers": WORKERS, "start": self.start, "stop": self.stop, "taken": self.taken, "produced": self.produced,
                "max_prepared_ahead": self.max_prepared_ahead if self.mode == "overlap" else 0, "consumer_wait_seconds": self.wait_s}


def formal_stream(mode, data, start, stop, check, *, fine=False):
    """The stream of the formal schedule: formal_data.windows with its fixed epoch length."""
    import formal_data
    return Stream(mode, formal_data.windows, data, start, stop, formal_data.WINDOWS_PER_EPOCH, check,
                  formal_data=formal_data, fine=fine)


# ================================================================ memory readings
def _children(pid):
    out = []
    try:
        for tid in os.listdir(f"/proc/{pid}/task"):
            try:
                with open(f"/proc/{pid}/task/{tid}/children") as f:
                    out += [int(v) for v in f.read().split()]
            except OSError:
                pass
    except OSError:
        pass
    return out


def _memory(pid):
    """(Rss, Pss) in bytes of one process; Pss falls back to Rss when smaps_rollup is not readable."""
    rss = pss = None
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            for line in f:
                if line.startswith("Rss:"):
                    rss = int(line.split()[1]) * 1024
                elif line.startswith("Pss:"):
                    pss = int(line.split()[1]) * 1024
    except OSError:
        pass
    if rss is None:
        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss = int(line.split()[1]) * 1024
        except OSError:
            return 0, 0
    return rss or 0, pss if pss is not None else (rss or 0)


class MemorySampler:
    """Samples the resident memory of this process and of its direct children (the loader workers) in a thread.
    Pss counts a shared page once across the processes that map it (window tensors travel through shared memory)."""

    def __init__(self, interval=1.0):
        self.interval, self.samples, self._halt = interval, [], threading.Event()
        self._thread = threading.Thread(target=self._run, name="r28-memory-sampler", daemon=True)

    def _run(self):
        me = os.getpid()
        while not self._halt.is_set():
            kids = _children(me)
            own = _memory(me)
            sub = [_memory(k) for k in kids]
            self.samples.append({"t": time.perf_counter(), "main_rss": own[0], "main_pss": own[1], "children": len(kids),
                                 "children_rss": sum(s[0] for s in sub), "children_pss": sum(s[1] for s in sub)})
            self._halt.wait(self.interval)

    def __enter__(self):
        if not sys.platform.startswith("linux"):
            raise RuntimeError("MemorySampler requires Linux /proc; RSS/PSS are unavailable on this platform")
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._halt.set()
        self._thread.join(timeout=30)
        return False

    def summary(self, since=None, until=None):
        rows = [s for s in self.samples if (since is None or s["t"] >= since) and (until is None or s["t"] <= until)]
        if not rows:
            return {"samples": 0}
        return {"samples": len(rows), "interval_seconds": self.interval,
                "peak_total_pss_bytes": max(s["main_pss"] + s["children_pss"] for s in rows),
                "peak_total_rss_bytes": max(s["main_rss"] + s["children_rss"] for s in rows),
                "peak_main_rss_bytes": max(s["main_rss"] for s in rows),
                "peak_children_rss_bytes": max(s["children_rss"] for s in rows),
                "max_children": max(s["children"] for s in rows),
                "note": "Pss counts shared pages once; Rss sums count a shared-memory window in every process that maps it"}


def window_bytes(x, targets):
    import torch
    n = x.numel() * x.element_size()
    for t in targets:
        n += sum(v.numel() * v.element_size() for v in t.values() if isinstance(v, torch.Tensor))
    return n


def memory_plan(window, mode):
    """Upper bound of the window copies alive in host memory, from the fixed loader settings (not a measurement)."""
    prefetch = WORKERS * 2
    ahead = READY_LIMIT if mode == "overlap" else 0
    return {"window_bytes": window, "being_consumed": 1, "prepared_by_the_producer": ahead,
            "loader_prefetch_windows_max": prefetch, "loader_workers_in_progress_max": WORKERS,
            "windows_max": 1 + ahead + prefetch + WORKERS, "bytes_max": window * (1 + ahead + prefetch + WORKERS),
            "note": "workers=2 and prefetch_factor=2 are the unchanged loader settings; the overlapped path adds at most one "
                    "prepared window (the window in transit and the ready window are the same object)"}
