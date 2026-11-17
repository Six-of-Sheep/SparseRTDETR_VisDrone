"""Input pipeline of the R28 formal loop (ENGINEERING_INPUT_OVERLAP_NEXT_005): pure CPU checks with a synthetic window
generator.  No model, no real loader: the real loader, the reference ledger and the bitwise acceptance are the
cal-check and pipeline stages.

Covered: the serial and the overlapped path deliver the same audited windows in the same order across an epoch
boundary; the look-ahead is one prepared window; a failing audit or loader is raised to the consumer in window order
before the offending window can be consumed and is never swallowed; a window modified after its audit is refused;
a stream reopened at the committed cursor rebuilds the dropped window; closing never hangs; the clock wrappers.
"""
import sys
import threading
import time
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "r28"))
import r28_pipeline as PL  # noqa: E402

PER_EPOCH = 4


def make_windows(*, fail_at=None, delay=0.0, made=None, module=None):
    """A stand-in for formal_data.windows: the rest of one epoch from the committed update, or `limit` windows."""

    def windows(data, completed, *, limit=None):
        first = completed % PER_EPOCH
        n = PER_EPOCH - first if limit is None else limit
        for k in range(n):
            u = completed + k + 1
            if fail_at == u:
                raise RuntimeError(f"loader failed at window {u}")
            if delay:
                time.sleep(delay)
            x = torch.full((2, 3), float(u))
            t = [{"boxes": torch.full((1, 4), float(u)), "labels": torch.tensor([u]), "image_id": f"img{u}"}]
            if module is not None:
                module.input_record(x, t, None)
            if made is not None:
                made.append(u)
            yield x, t, {"update": u, "epoch": completed // PER_EPOCH + 1, "window": first + k}

    return windows


def stream(mode, start, stop, *, check=lambda inp: None, **kw):
    module = kw.pop("formal_data", None)
    return PL.Stream(mode, make_windows(module=module, **kw), None, start, stop, PER_EPOCH, check, formal_data=module)


def content(item):
    return item.inp["update"], item.inp["epoch"], item.inp["window"], item.x.clone(), item.t[0]["boxes"].clone()


def same(a, b):
    return a[:3] == b[:3] and torch.equal(a[3], b[3]) and torch.equal(a[4], b[4])


def test_serial_and_overlap_deliver_the_same_windows_across_an_epoch_boundary():
    with stream("serial", 2, 9) as st:
        serial = [content(i) for i in st]
        a = st.account()
    with stream("overlap", 2, 9) as st:
        overlap = [content(i) for i in st]
        b = st.account()
    assert [s[:3] for s in serial] == [(3, 1, 2), (4, 1, 3), (5, 2, 0), (6, 2, 1), (7, 2, 2), (8, 2, 3), (9, 3, 0)]
    assert len(overlap) == 7 and all(same(x, y) for x, y in zip(serial, overlap))
    assert a == {**a, "mode": "serial", "taken": 7, "produced": 7, "max_prepared_ahead": 0, "ready_limit": 0}
    assert b == {**b, "mode": "overlap", "taken": 7, "produced": 7, "max_prepared_ahead": 1, "ready_limit": 1, "workers": 2}
    assert PL.READY_LIMIT == 1 and PL.WORKERS == 2 and PL.MODES == ("serial", "overlap")
    with pytest.raises(PL.PipelineError):
        stream("eager", 0, 4)
    with pytest.raises(PL.PipelineError):
        stream("serial", 4, 2)
    with stream("overlap", 5, 5) as st:                      # an empty span ends at once on both paths
        assert list(st) == []


def test_the_producer_prepares_at_most_one_window_ahead():
    holder, ahead, made = {}, [], []

    def check(inp):
        ahead.append(inp["update"] - holder["st"].taken)      # windows audited so far minus windows taken by the consumer

    st = holder["st"] = stream("overlap", 0, 8, check=check, made=made)      # bound before the producer thread starts
    with st:
        for k, item in enumerate(st):
            assert st.wait_ready() == 1 if k < 7 else st.ready() == 0          # the next window is prepared while this one is used
            time.sleep(0.02)                                                    # a slow consumer does not let the producer run on
            assert len(made) <= item.inp["update"] + 1
        account = st.account()
    assert max(ahead) == 1 and min(ahead) == 1 and account["max_prepared_ahead"] == 1 and made == list(range(1, 9))
    with stream("serial", 0, 3) as st:
        assert st.ready() == 0
        with pytest.raises(PL.PipelineError):
            st.wait_ready()                                                     # only the overlapped path prepares ahead


@pytest.mark.parametrize("mode", PL.MODES)
def test_a_failing_audit_reaches_the_consumer_before_the_window_and_is_not_swallowed(mode):
    def check(inp):
        if inp["update"] == 5:
            raise ValueError("input record differs before update 5")

    consumed = []
    with pytest.raises(ValueError, match="before update 5"):
        with stream(mode, 2, 9, check=check) as st:
            for item in st:
                consumed.append(item.inp["update"])           # stands for the forward
    assert consumed == [3, 4]
    st = stream(mode, 2, 9, check=check)
    with st:
        assert [next(st).inp["update"], next(st).inp["update"]] == [3, 4]
        for _ in range(2):                                    # the stream stays failed: no later window is delivered
            with pytest.raises(ValueError, match="before update 5"):
                next(st)
            if mode == "serial":
                break


@pytest.mark.parametrize("mode", PL.MODES)
def test_a_loader_failure_is_raised_in_window_order(mode):
    consumed = []
    with pytest.raises(RuntimeError, match="loader failed at window 6"):
        with stream(mode, 2, 9, fail_at=6) as st:
            for item in st:
                consumed.append(item.inp["update"])
    assert consumed == [3, 4, 5]
    with pytest.raises(PL.PipelineError, match="not the next update"):          # a window out of schedule order

        def shifted(data, completed, *, limit=None):
            for x, t, inp in make_windows()(data, completed, limit=limit):
                yield x, t, {**inp, "update": inp["update"] + 1}

        with PL.Stream(mode, shifted, None, 0, 3, PER_EPOCH, lambda inp: None) as st:
            list(st)


def test_a_window_modified_after_its_audit_is_refused():
    with stream("overlap", 0, 4) as st:
        first = next(st)
        assert first.inp["update"] == 1 and st.wait_ready() == 1
        prepared = st._q.queue[0]
        prepared.x.add_(1.0)                                  # in-place write after the hash: the version counter moves
        with pytest.raises(PL.PipelineError, match="modified after its audit"):
            next(st)
    with stream("overlap", 0, 4) as st:
        next(st)
        st.wait_ready()
        st._q.queue[0].t[0]["boxes"].mul_(2.0)                # a target tensor
        with pytest.raises(PL.PipelineError, match="modified after its audit"):
            next(st)
    with stream("overlap", 0, 4) as st:
        next(st)
        st.wait_ready()
        item = st._q.queue[0]
        item.x = item.x.clone()                               # another storage with the same values is another object
        with pytest.raises(PL.PipelineError, match="modified after its audit"):
            next(st)
    x, t = torch.zeros(2, 3), [{"boxes": torch.zeros(1, 4), "image_id": "a"}]
    item = PL.Item(x, t, {"update": 1}, {})
    item.verify()
    t[0]["boxes"][0, 0] = 1.0
    with pytest.raises(PL.PipelineError):
        item.verify()


def test_reopening_at_the_committed_cursor_rebuilds_the_dropped_window():
    with stream("serial", 0, 6) as st:
        reference = [content(i) for i in st]
    made = []
    with stream("overlap", 0, 6, made=made) as st:
        got = [content(next(st)), content(next(st))]
        assert st.wait_ready() == 1                           # window 3 is prepared, not committed
        account = st.account()
    assert account["taken"] == 2 and account["produced"] == 3 and made[:3] == [1, 2, 3]
    with stream("overlap", 2, 6) as st:                       # the committed cursor is 2: window 3 is rebuilt, not skipped
        rest = [content(i) for i in st]
    assert all(same(a, b) for a, b in zip(got + rest, reference)) and len(got + rest) == 6
    assert st._thread is not None and not st._thread.is_alive()


def test_close_never_hangs_and_a_dead_producer_is_detected():
    st = stream("overlap", 0, 8, delay=0.05)
    with st:
        next(st)                                              # the producer is now preparing or holding window 2
    assert not st._thread.is_alive() and st._q.empty()
    st.close()                                                # idempotent
    with pytest.raises(PL.PipelineError, match="closed"):
        next(st)
    t0 = time.monotonic()
    with stream("overlap", 0, 8, delay=0.05):
        pass                                                  # closed before anything was taken
    assert time.monotonic() - t0 < 30
    dead = stream("overlap", 0, 8)
    dead._thread = threading.Thread(target=lambda: None)
    dead._thread.start()
    dead._thread.join()
    with pytest.raises(PL.PipelineError, match="without delivering"):
        next(dead)
    with pytest.raises(PL.PipelineError, match="without a prepared window"):
        dead.wait_ready()
    with pytest.raises(KeyError):                             # an error leaving the loop is not replaced by the close
        with stream("overlap", 0, 8, delay=0.01) as st2:
            next(st2)
            raise KeyError("consumer failure")


@pytest.mark.parametrize("mode", PL.MODES)
def test_clock_times_the_audit_inside_the_generator_and_is_removed(mode):
    def input_record(x, t, traces):
        time.sleep(0.02)
        return {}

    module = types.SimpleNamespace(input_record=input_record)
    with stream(mode, 0, 3, formal_data=module, delay=0.02) as st:
        assert module.input_record is not input_record and module.input_record.__wrapped__ is input_record
        with pytest.raises(PL.PipelineError):
            PL.Clock(module).install()                        # one clock at a time
        rows = [i.timing for i in st]
    assert module.input_record is input_record                # the wrapper is gone after the stream
    for r in rows:
        assert r["pipeline"] == mode and r["audit_s"] >= 0.015 and r["next_s"] >= r["audit_s"] + 0.015
        assert abs(r["fetch_s"] - (r["next_s"] - r["audit_s"])) <= 1e-9 and r["reference_check_s"] >= 0 and r["wait_s"] >= 0
    with stream(mode, 0, 2) as st:                            # without a module nothing is wrapped and the audit time is unknown
        assert all(i.timing["audit_s"] is None and i.timing["fetch_s"] is None for i in st)


def test_memory_plan_and_sampler():
    x, t = torch.zeros(16, 3, 8, 8), [{"boxes": torch.zeros(5, 4), "labels": torch.zeros(5, dtype=torch.int64), "image_id": "a"}]
    n = PL.window_bytes(x, t)
    assert n == 16 * 3 * 8 * 8 * 4 + 5 * 4 * 4 + 5 * 8
    serial, overlap = PL.memory_plan(n, "serial"), PL.memory_plan(n, "overlap")
    assert serial["windows_max"] == 1 + 4 + 2 and overlap["windows_max"] == serial["windows_max"] + 1
    assert overlap["prepared_by_the_producer"] == 1 and serial["prepared_by_the_producer"] == 0
    assert overlap["bytes_max"] == n * overlap["windows_max"] and overlap["loader_prefetch_windows_max"] == 4
    if not sys.platform.startswith("linux"):
        with pytest.raises(RuntimeError, match="requires Linux /proc"):
            with PL.MemorySampler(0.05):
                pytest.fail("unsupported sampler entered")
        return
    with PL.MemorySampler(0.05) as m:
        mark = time.perf_counter()
        time.sleep(0.25)
    whole, late = m.summary(), m.summary(since=mark + 0.1)
    assert whole["samples"] >= 2 and whole["peak_total_rss_bytes"] > 0 and whole["peak_total_pss_bytes"] > 0
    assert 1 <= late["samples"] <= whole["samples"] and m.summary(since=time.perf_counter() + 10) == {"samples": 0}


def test_the_producer_thread_leaves_the_global_rng_alone():
    import random
    import numpy as np
    torch.manual_seed(5)
    random.seed(5)
    np.random.seed(5)
    before = (torch.get_rng_state().clone(), random.getstate(), np.random.get_state()[1].copy())
    with stream("overlap", 0, 8) as st:
        list(st)
    assert torch.equal(torch.get_rng_state(), before[0]) and random.getstate() == before[1]
    assert np.array_equal(np.random.get_state()[1], before[2])
