"""Affected-path CPU checks for the r24 roi-ld engine installation (no CUDA, no training, no data root).

These checks cover what the installation actually changed: the round identity of the three engine
modules, the retirement of the r19 tiny-positives item, the r24 science-plan reader (positive and
six negative cases), the unified evaluation definition, the measured-cost policy, the patched
criterion's roi declaration and its native-arm zero-selection observation, the R2 dispatch with the
CPU-fixture scaffolding, the dataset hook and the roi state round trip.  They are implementation
checks only: they prove the wiring, not any benefit.

usage: <frozen python> -I tools/roi_ld/test_roi_ld_engine.py --source-root <release>/source \
           [--receipt <round>/plan-002/PLAN_AMENDMENT_r24.json]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

RESULTS = []
PLAN_REL = "configs/research/r24_roi_ld_v1.rev2.json"


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition), detail))
    print(("PASS " if condition else "FAIL ") + name + ((" | " + str(detail)) if detail else ""))


def refused(fn):
    try:
        fn()
    except Exception:
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--receipt", default=None)
    args = parser.parse_args()
    root = Path(args.source_root).resolve()
    for path in (root, root / "src", root / "vendor/rtdetrv2_pytorch/src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    sys.path.insert(0, str(root / "tools"))

    from sparse_rtdetr.protocol_r18.model import Config, build, fp32_outputs
    from sparse_rtdetr.protocol_r18 import slice_finetune as ft
    from sparse_rtdetr.protocol_r18 import slice_ft_execution as ex
    from sparse_rtdetr.protocol_r18 import slice_ft_progress as progress
    from sparse_rtdetr.protocol_r18 import slice_ft_job as job
    from sparse_rtdetr.protocol_r18 import tiny_positives_arm, roi_ld, roi_ld_arm, roi_ld_data

    # 1. round identity of the installed engine modules
    check("round.version", ft.VERSION == "r24-roi-ld-v1", ft.VERSION)
    check("round.retired_install_map", ft.POS_INSTALL == {"A0": None, "A1": None, "A2": None}, ft.POS_INSTALL)
    check("round.roi_install_map", ft.ROI_INSTALL == {"A0": None, "A1": "R1", "A2": "R2"}, ft.ROI_INSTALL)
    check("round.arm_install_map", ft.ARM_INSTALL == {"A0": None, "A1": None, "A2": None}, ft.ARM_INSTALL)
    check("round.schedule_constants",
          ft.ARMS == ("A0", "A1", "A2") and ft.UPDATES == 3648 and ft.PASSES == 12
          and ft.ALT_OFFSET == 36480 and tuple(ft.CHECKPOINT_PASSES) == (0, 6, 12), (ft.UPDATES, ft.PASSES))
    check("round.parent_sha", ft.PARENT_SHA == "13a35283a02d2b728e3449917586ded5cdf2d578ebe6847510748e0dc4d21757")

    # 2. execution identity: round, acceptance phase, phase table and the contract invariants
    check("execution.task_and_phase",
          ex.TASK == "r24-roi-ld-20261103-001" and ex.ACCEPTANCE_PHASE == "accept-roi-ld"
          and ex.PHASES == ("accept-roi-ld", "formal-A0", "formal-A1", "formal-A2", "infer-A0-A-emaC",
                            "infer-A1-A-emaC", "infer-A2-A-emaC", "evaluation"), ex.PHASES)
    check("execution.arm_maps", ex.ROI_INSTALL == {"A0": None, "A1": "R1", "A2": "R2"}
          and ex.POS_INSTALL_RETIRED == {"A0": None, "A1": None, "A2": None})
    check("execution.phase_arm",
          ex.phase_arm("accept-roi-ld") == "roi" and ex.phase_arm("formal-A1") == "A1"
          and ex.phase_arm("infer-A2-A-emaC") == "A2" and ex.phase_arm("evaluation") == "paired")
    check("execution.evaluation_definition",
          ex.EVALUATION_DEFINITION["scorings"] == 6 and ex.EVALUATION_DEFINITION["ema"] == "C"
          and ex.EVALUATION_DEFINITION["ema_r"] == "state_only"
          and ex.EVALUATION_DEFINITION["trained_A0_required_equal_to_parent"] is False,
          ex.EVALUATION_DEFINITION)

    # 3. progress namespace: EMA-C only and exactly six scorings in a fixed order
    check("progress.namespace", progress.PROTOCOL == "r24-roi-ld-full-execution-v1" and progress.EMAS == ("C",),
          (progress.PROTOCOL, progress.EMAS))
    check("progress.evaluation_order",
          progress.EVALUATION_ORDER == [("A0", "A", 600), ("A0", "A", 300), ("A1", "A", 600),
                                        ("A1", "A", 300), ("A2", "A", 600), ("A2", "A", 300)],
          progress.EVALUATION_ORDER)

    # 4. the r24 science plan reads positively
    plan_path = root / PLAN_REL
    plan = ft.read_plan(plan_path)
    check("plan.read", plan["revision"] == 2 and plan["version"] == "r24-roi-ld-v1"
          and plan["roi_install"] == {"A0": None, "A1": "R1", "A2": "R2"}, plan["revision"])
    check("plan.cost_semantics",
          plan["cost_policy"]["predeclared_ratio"] is None
          and plan["cost_policy"]["ratio_source"] == "measured_from_the_inference_stage"
          and plan["gates"]["cost"]["metric"] == "inference_stage_seconds", plan["cost_policy"]["gate"])
    check("plan.evaluation_definition", plan["evaluation_definition"]["scorings"] == 6
          and plan["evaluation_definition"]["ema"] == "C")

    # 5. the plan reader refuses every change that would move science or cost semantics
    mutations = [
        ("version", lambda p: p.update(version="r18-tiny-positives-v2")),
        ("roi_install", lambda p: p["roi_install"].update(A2=None)),
        ("retirement", lambda p: p.update(pos_install={"A0": None, "A1": "A1", "A2": "A2"})),
        ("constants", lambda p: p["mechanism"]["constants"].update(target_iou_tau=0.5)),
        ("cost_predeclared", lambda p: p["cost_policy"].update(predeclared_ratio=1.0)),
        ("evaluation_scopings", lambda p: p["evaluation_definition"].update(scorings=12)),
    ]
    with tempfile.TemporaryDirectory() as work:
        for name, mutate in mutations:
            broken = copy.deepcopy(plan)
            mutate(broken)
            path = Path(work) / (name + ".json")
            path.write_text(json.dumps(broken, ensure_ascii=False), encoding="utf-8")
            check("plan.refuses." + name, refused(lambda q=path: ft.read_plan(q)))

    # 6. the patched criterion carries the roi declaration, native or armed
    config = Config(input_hw=(128, 128), physical_batch_size=1, amp_dtype="float32")
    model, criterion, _ = build(config, synthetic=True)
    check("criterion.native_recipe",
          criterion.recipe()["roi_ld"] == roi_ld_arm.NATIVE_DECLARATION, criterion.recipe()["roi_ld"])
    criterion.roi_ld_declaration = roi_ld_arm.declaration("R2")
    term = roi_ld_arm.install(model, criterion, "R2")
    check("criterion.armed_recipe",
          criterion.recipe()["roi_ld"] == roi_ld_arm.declaration("R2")
          and criterion.recipe()["roi_ld"]["constants_sha256"] == roi_ld.constants_sha256(),
          criterion.recipe()["roi_ld"])
    check("criterion.state_round_trip",
          roi_ld_arm.state(type("E", (), {"criterion": criterion})())["mode"] == "R2"
          and roi_ld_arm.state(type("E", (), {"criterion": criterion})())["targets"] == int(term.targets))

    # 7/8. one real forward-lite through the patched criterion, native then armed with the fixture
    def build_batch(mode):
        from PIL import Image
        images, targets = [], []
        for position in (0, 1):
            raw = ft.parse_raw(b"30,30,20,25,1,1,0,0\n8,8,4,4,1,2,0,0\n40,10,4,4,1,3,0,0\n"
                               if position % 2 == 0 else b"", "engine-check-%d" % position)
            x, t, _ = ft.Transform("A0", (128, 128))(Image.fromarray(np.full((128, 128, 3), 90 + position, np.uint8)),
                                                     raw, epoch=1, position=position,
                                                     image_id="engine-check-%d" % position)
            if mode is not None:
                job.fixture_roi_target(mode, t)
            images.append(x); targets.append(t)
        return torch.stack(images), targets

    def run(mode):
        m, c, _ = build(config, synthetic=True)
        c.roi_ld_declaration = roi_ld_arm.declaration(mode)
        if mode is not None:
            roi_ld_arm.install(m, c, mode)
        batch, targets = build_batch(mode)
        m.train(); c.train()
        torch.manual_seed(20261103)
        out = m(batch, targets=targets)
        losses = c(fp32_outputs(out), targets)
        rows = [row for diag in [c.last_diagnostics] for row in diag
                if isinstance(row, dict) and "roi_band_gt" in row]
        return losses, rows

    losses_native, rows_native = run(None)
    check("native.observes_roi_band_with_zero_selected",
          len(rows_native) > 0 and all(row["roi_selected"] == 0 and row["roi_mode"] is None
                                       and row["roi_band_gt"] >= 0 for row in rows_native),
          [row["roi_mode"] for row in rows_native])
    losses_armed, rows_armed = run("R2")
    # Every non-denoising family call produces its own ledger row over the same targets, so the
    # assertion is per row: a row that sees the fixture's two band rows must select exactly one
    # (the common-support row) and record exactly one one-sided row (the mirrored row).
    positive_rows = [row for row in rows_armed if row["roi_band_gt"] > 0]
    check("armed.dispatch_and_selection",
          len(rows_armed) > 0 and all(row["roi_mode"] == "R2" for row in rows_armed)
          and len(positive_rows) > 0
          and all(row["roi_selected"] == (row["roi_band_gt"] + 1) // 2
                  and row["roi_one_sided"] == row["roi_band_gt"] // 2
                  and row["roi_below_iou"] == 0 for row in positive_rows),
          [(row["roi_band_gt"], row["roi_selected"], row["roi_one_sided"]) for row in rows_armed])
    check("armed.ledger_closes",
          all(row["roi_selected"] + row["roi_below_iou"] + row["roi_no_candidate"] + row["roi_no_class"]
              + row["roi_one_sided"] + row["roi_unmatched_band_gt"] == row["roi_band_gt"]
              for row in rows_armed))
    check("armed.same_loss_keys_as_native", sorted(losses_armed) == sorted(losses_native),
          (sorted(losses_native), sorted(losses_armed)))

    # 9. the dataset hook is installed (source-level; the data path is covered by roi_ld_check)
    init_source = inspect.getsource(ft.Dataset.__init__)
    get_source = inspect.getsource(ft.Dataset.__getitem__)
    check("dataset.hook_present",
          "roi_mode" in init_source and "roi_cache" in init_source
          and "roi_ld_data.attach" in get_source and "ROI_INSTALL" not in get_source)

    # 10. the retirement leaves the r19 modules importable and inert
    check("retired.modules_present",
          tiny_positives_arm.declaration("A2")["installed"] is True
          and tiny_positives_arm.declaration(None) == tiny_positives_arm.NATIVE_DECLARATION)

    # 11. the plan amendment receipt proves the scientific projection did not move
    if args.receipt:
        receipt = json.loads(Path(args.receipt).read_text(encoding="utf-8"))
        check("amendment.projection_unchanged",
              receipt["scientific_projection_unchanged"] is True
              and receipt["scientific_projection_sha256_before"] == receipt["scientific_projection_sha256_after"]
              and receipt["rev2"]["sha256"] == hashlib.sha256(plan_path.read_bytes()).hexdigest(),
              receipt["scientific_projection_sha256_after"])

    check("cuda.not_initialized", not torch.cuda.is_initialized())
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print("TESTS %d/%d PASS" % (passed, len(RESULTS)))
    if passed != len(RESULTS):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
