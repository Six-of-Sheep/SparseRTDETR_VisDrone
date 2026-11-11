"""R27-002 L0 runner common layer: shared-field comparison and fixed constants (pure CPU, no model, no torch)."""
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "r27_l0"))
import l0_common as K  # noqa: E402


def row(update=18241):
    return {"input": {"update": update, "epoch": 61, "window": 0, "image_ids_sha256": "a" * 64, "gt_ids_sha256": "b" * 64,
                      "augmentation_seeds_sha256": "c" * 64, "augmentation_trace_sha256": "d" * 64,
                      "augmented_samples_sha256": "e" * 64, "input_tensor_sha256": "f" * 64, "target_sha256": "0" * 64,
                      "augmentation_strong": True},
            "gt_count": 916, "loss": 10.27, "lr": 5.499938797812981e-05, "gradient_norm_before_clip": 127.1,
            "microbatch_losses": [{"loss_vfl": 0.44}], "update_seconds": 2.3, "ema_updates": update}


def test_shared_fields_are_the_declared_list():
    assert K.SHARED_FIELDS == tuple(f"input.{k}" for k in K.INPUT_FIELDS) + ("lr", "gt_count", "ema_updates")
    assert len(K.INPUT_FIELDS) == 11 and len(K.SHARED_FIELDS) == 14
    s = K.shared(row())
    assert tuple(s) == K.SHARED_FIELDS
    assert not any(k in s for k in ("loss", "gradient_norm_before_clip", "update_seconds", "microbatch_losses", "monitor"))


def test_compare_shared_accepts_identical_rows_and_unshared_changes():
    a, b = row(), row()
    assert K.compare_shared(a, b) == K.shared(a)
    b.update(loss=9.0, gradient_norm_before_clip=1.0, update_seconds=99.0, microbatch_losses=[], monitor={"x": 1})
    assert K.compare_shared(a, b) == K.shared(a)


@pytest.mark.parametrize("field", K.SHARED_FIELDS)
def test_compare_shared_rejects_each_changed_shared_field(field):
    a, b = row(), row()
    target, key = (b["input"], field[len("input."):]) if field.startswith("input.") else (b, field)
    old = target[key]
    target[key] = (not old) if isinstance(old, bool) else (old + 1 if isinstance(old, int) else
                                                           (old * 1.0000001 if isinstance(old, float) else "9" * 64))
    with pytest.raises(ValueError, match=f"shared field differs: {field.replace('.', '[.]')} "):
        K.compare_shared(a, b)


def test_compare_shared_names_the_first_differing_field_and_checks_types():
    a, b = row(), row()
    b["input"]["target_sha256"] = "1" * 64
    b["lr"] = 0.0
    with pytest.raises(ValueError, match="input.target_sha256"):
        K.compare_shared(a, b)
    c = row()
    c["input"]["augmentation_strong"] = 1  # equal to True by value, different type
    with pytest.raises(ValueError, match="input.augmentation_strong"):
        K.compare_shared(a, c)


def test_shared_rejects_missing_or_extra_input_keys():
    a = row()
    del a["input"]["target_sha256"]
    with pytest.raises(ValueError):
        K.shared(a)
    b = row()
    b["input"]["arm"] = "L0"
    with pytest.raises(ValueError):
        K.shared(b)
    c = copy.deepcopy(row())
    del c["ema_updates"]
    with pytest.raises(KeyError):
        K.shared(c)


def test_gate_rotations_and_schedule_constants():
    assert K.GATE_D == (("600", "APS", 0.50), ("600", "AP75_small", 0.30), ("600", "AP", -0.10), ("300", "AP", -0.20),
                        ("300", "APS", -0.20), ("600", "APM", -0.50), ("600", "APL", -0.50))
    assert K.ROTATIONS == (("A0", "L0", "P", "S15"), ("P", "S15", "A0", "L0"), ("L0", "A0", "S15", "P"))
    assert all(sorted(r) == ["A0", "L0", "P", "S15"] for r in K.ROTATIONS)
    assert K.SCHEMA == "r27-l0-state-v1" and K.ARM == "L0" and K.ENDPOINT == 36480
    assert K.SAVE_EPOCHS == (60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120) and len(K.SAVE_EPOCHS) == 13
    assert K.FORMAL_SAVE_EPOCHS == (60, 75, 90, 105, 120) and set(K.FORMAL_SAVE_EPOCHS) <= set(K.SAVE_EPOCHS)
    assert (K.EVAL_BATCH, K.EVAL_WARMUP, K.EVAL_BATCHES, K.BUDGETS) == (4, 4, 137, (600, 300))
    assert (K.COST_BATCH, K.COST_WARMUP, K.COST_PASSES, K.COST_BATCHES_PER_PASS) == (4, 2, 3, 16)
    assert K.PARENT_SHA == "a2b6412e3c85de7592829f24aff09e355fdd11cb53798bc498bd04b882e1bf8a"


def test_workload_totals():
    w = K.WORKLOAD
    assert (w["cpu_fixture"]["updates"], w["cpu_fixture"]["forwards"]) == (3, 6)
    assert (w["cuda_zero_step"]["updates"], w["cuda_zero_step"]["forwards"]) == (0, 4)
    assert (w["cuda_accept"]["updates"], w["cuda_accept"]["forwards"]) == (3, 6)
    assert (w["formal"]["updates"], w["formal"]["forwards"]) == (18240, 36480)
    assert w["evaluation"]["forwards"] == 141 and w["evaluation"]["score_budget_calls"] == 2
    assert w["evaluation"]["per_class_APL_COCOeval_calls"] == 1
    assert w["cost4"]["forwards_per_endpoint"] == 50 and w["cost4"]["forwards"] == 200
    assert w["normal_gpu_total"] == {"updates": 18243, "forwards": 36831}


def test_cells_reproduce_a_difference_in_pp():
    p = {"600": {"APS": 0.2332513847416923}}
    a0 = {"600": {"APS": 0.2217618575337924}}
    out = K.cells(p, a0, (("600", "APS", 0.50),))
    assert out["top600_APS"]["difference_pp"] == 1.148952720789989 and out["top600_APS"]["passed"]


# ---------------------------------------------------------------- endpoint location (file system only)
def closed_dir(root, name, *, status="PASS", arm="L0", digest="c" * 64, closed=True):
    d = root / name
    d.mkdir()
    if closed:
        (d / "CLOSED.json").write_text(json.dumps({"status": status, "arm": arm, "contract_sha256": digest}))
    return d


@pytest.fixture
def round_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(K, "ROUND", tmp_path)
    return tmp_path


C = {"contract_sha256": "c" * 64}


def test_formal_dir_refuses_a_round_without_a_closed_formal_directory(round_dir):
    with pytest.raises(ValueError, match="exactly one closed PASS formal-L0"):
        K.formal_dir(C)
    closed_dir(round_dir, "formal-L0", closed=False)  # interrupted run: no CLOSED.json
    closed_dir(round_dir, "formal-L0-r2", status="STOP")
    with pytest.raises(ValueError, match="exactly one closed PASS formal-L0"):
        K.endpoint_path(C)


@pytest.mark.parametrize("name", ["formal-L0", "formal-L0-r2", "formal-L0-r13"])
def test_formal_dir_finds_the_single_pass_directory(round_dir, name):
    closed_dir(round_dir, "formal-L0", closed=False) if name != "formal-L0" else None
    want = closed_dir(round_dir, name)
    for other in ("formal-L0-r1", "formal-L0-r02", "formal-L0-dev", "formal-P", "accept", "xformal-L0"):
        closed_dir(round_dir, other)  # names outside formal-L0 / formal-L0-rN (N>=2) are never candidates
    assert K.formal_dir(C) == want
    assert K.endpoint_path(C) == want / "checkpoint-epoch-120.pth"
    (round_dir / "CONTRACT.json").write_text(json.dumps(C))
    assert K.endpoint_path() == want / "checkpoint-epoch-120.pth"  # no argument: digest read from the round contract


def test_formal_dir_refuses_two_pass_directories(round_dir):
    closed_dir(round_dir, "formal-L0")
    closed_dir(round_dir, "formal-L0-r2")
    with pytest.raises(ValueError, match="formal-L0', 'formal-L0-r2"):
        K.formal_dir(C)


def test_formal_dir_ignores_another_contract_or_arm(round_dir):
    closed_dir(round_dir, "formal-L0", digest="d" * 64)
    closed_dir(round_dir, "formal-L0-r2", arm="P")
    with pytest.raises(ValueError, match="exactly one closed PASS formal-L0"):
        K.formal_dir(C)
    want = closed_dir(round_dir, "formal-L0-r3")
    assert K.formal_dir(C) == want  # the other-contract and other-arm directories do not count as a second one


def test_endpoint_state_requires_the_e120_file_in_the_closed_directory(round_dir):
    closed_dir(round_dir, "formal-L0-r2")
    with pytest.raises(ValueError, match="L0 endpoint missing"):
        K.endpoint_state(C)


def test_formal_dir_never_selects_a_renamed_failure_directory(round_dir):
    # a failed run preserved as formal-L0-fail-NNN is evidence only, even if someone left a PASS CLOSED.json in it
    closed_dir(round_dir, "formal-L0-fail-001")
    closed_dir(round_dir, "formal-L0-fail-002", closed=False)
    assert K.FORMAL_DIR.fullmatch("formal-L0-fail-001") is None and K.FORMAL_DIR.fullmatch("formal-L0-r2-fail-001") is None
    with pytest.raises(ValueError, match="exactly one closed PASS formal-L0"):
        K.formal_dir(C)
    want = closed_dir(round_dir, "formal-L0")
    assert K.formal_dir(C) == want


# ---------------------------------------------------------------- release tree against its MANIFEST (file system only)
def test_release_intact_detects_changed_added_and_removed_files(tmp_path):
    src = tmp_path / "source"
    (src / "tools").mkdir(parents=True)
    (src / "tools" / "a.py").write_text("a = 1\n")
    (src / "b.txt").write_text("b\n")
    files = K.source_manifest(src)
    assert set(files) == {"tools/a.py", "b.txt"}
    assert K.release_intact(src, files) == 2
    (src / "tools" / "__pycache__").mkdir()
    (src / "tools" / "__pycache__" / "a.cpython-310.pyc").write_bytes(b"x")  # byte-code caches are not release content
    assert K.release_intact(src, files) == 2
    (src / "tools" / "a.py").write_text("a = 2\n")
    with pytest.raises(ValueError, match=r"release source differs from MANIFEST.json \(1 paths\): \['tools/a.py'\]"):
        K.release_intact(src, files)
    (src / "tools" / "a.py").write_text("a = 1\n")
    (src / "new.py").write_text("")
    with pytest.raises(ValueError, match="new.py"):
        K.release_intact(src, files)
    (src / "new.py").unlink()
    (src / "b.txt").unlink()
    with pytest.raises(ValueError, match="b.txt"):
        K.release_intact(src, files)


# ---------------------------------------------------------------- formal loop helpers (pure functions of l0_loop)
import l0_loop as LOOP  # noqa: E402


def test_formal_points_are_the_twelve_epoch_saves_after_the_parent():
    assert LOOP.FORMAL_POINTS == tuple(ep * 304 for ep in range(65, 121, 5)) and len(LOOP.FORMAL_POINTS) == 12
    assert LOOP.FORMAL_POINTS[0] == 19760 and LOOP.FORMAL_POINTS[-1] == K.ENDPOINT


@pytest.mark.parametrize("start,first,count", [(18240, 19760, 12), (19760, 21280, 11), (19761, 21280, 11),
                                               (21279, 21280, 11), (34960, 36480, 1), (36479, 36480, 1), (36480, None, 0)])
def test_points_after_drops_the_points_at_or_before_the_start(start, first, count):
    used = LOOP.points_after(LOOP.FORMAL_POINTS, start)
    assert len(used) == count and all(start < p <= K.ENDPOINT for p in used) and list(used) == sorted(used)
    assert (used[0] if used else None) == first
    assert set(used) == {p for p in LOOP.FORMAL_POINTS if p > start}
    if used:
        assert used[-1] == K.ENDPOINT  # the endpoint is always still to be written


def test_points_after_resume_at_a_save_point_excludes_that_point():
    # the reported defect: a resume from the e065 state (update 19760) used to be handed 19760 itself and stopped
    assert 19760 in LOOP.FORMAL_POINTS and 19760 not in LOOP.points_after(LOOP.FORMAL_POINTS, 19760)
    assert LOOP.points_after((18544, 18848), 18544) == (18848,)
    assert LOOP.points_after([18848, 18544, 18544], 18240) == (18544, 18848)


def test_input_differences_lists_every_differing_field():
    a, b = row()["input"], row()["input"]
    assert LOOP.input_differences(a, b) == []
    b["target_sha256"] = "1" * 64
    b["window"] = 5
    b["augmentation_strong"] = 1  # equal to True by value, different type
    del b["gt_ids_sha256"]
    b["extra"] = 0
    got = LOOP.input_differences(a, b)
    assert [d["field"] for d in got] == ["input.augmentation_strong", "input.extra", "input.gt_ids_sha256",
                                         "input.target_sha256", "input.window"]
    assert got[1] == {"field": "input.extra", "L0": "<missing>", "R27": 0}
    assert got[4] == {"field": "input.window", "L0": 0, "R27": 5}


def test_shared_differences_lists_every_differing_shared_field_and_ignores_unshared_ones():
    a, b = row(), row()
    b.update(loss=1.0, gradient_norm_before_clip=2.0, update_seconds=3.0)
    assert LOOP.shared_differences(a, b) == []
    b["input"]["target_sha256"] = "1" * 64
    b["lr"] = 0.0
    b["ema_updates"] = 18242
    assert [d["field"] for d in LOOP.shared_differences(a, b)] == ["input.target_sha256", "lr", "ema_updates"]
    with pytest.raises(ValueError, match="input.target_sha256"):  # compare_shared keeps naming the first one
        K.compare_shared(a, b)


def test_first_reading_is_an_exact_comparison():
    first = {"update": 18241, "P": 10.27, "C": 10.27}
    r = LOOP.first_reading(10.27, first)
    assert r["equal_to_P"] and r["equal_to_C"] and r["update"] == 18241 and r["L0"] == 10.27
    assert not LOOP.first_reading(10.270000000000001, first)["equal_to_P"]


def test_formal_guard_refuses_a_second_formal_run_once_one_is_closed(round_dir, monkeypatch):
    monkeypatch.setattr(K, "stage_closed", lambda c, name: {"status": "PASS"})  # the three check stages are not the subject
    assert LOOP.guard(C, round_dir / "formal-L0", None) is None
    with pytest.raises(ValueError, match="writes <round>/formal-L0"):
        LOOP.guard(C, round_dir / "formal-L0-r2", None)
    with pytest.raises(ValueError, match="directly under the round"):
        LOOP.guard(C, round_dir / "x" / "formal-L0", None)
    closed_dir(round_dir, "formal-L0-fail-001")  # a preserved failure does not block a fresh run
    closed_dir(round_dir, "formal-L0", closed=False)
    assert LOOP.guard(C, round_dir / "formal-L0", None) is None  # (l0_run then refuses the existing directory itself)
    closed_dir(round_dir, "formal-L0-r2")
    for out, resume in ((round_dir / "formal-L0", None), (round_dir / "formal-L0-r3", round_dir / "formal-L0" / "x.pth")):
        with pytest.raises(ValueError, match="already has a closed formal directory .'formal-L0-r2'."):
            LOOP.guard(C, out, resume)
