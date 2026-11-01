"""r16 opportunity scan: 32 pre-frozen train_core images through the production term (CPU).

Read-only scan.  Same parent state as the whole line (the certified e120 EMA, loaded with
the chain's own ``initialize_from_parent``), then 32 plain forward passes: no backward,
no optimizer, no EMA, no checkpoint write, no BN or buffer change (the model is switched
to eval, which uses running statistics and never updates them, and the denoising branch
is inactive).  The image set and the augmentation seeds were frozen BEFORE the scan:

    epoch   source epoch 109 (the chain's first fine-tune pass)
    order   the chain's own logical order, windows 0 and 1, positions 0..31
    seeds   derived inside the chain's transform from seed 0, the epoch, the position
            and the stable image id, so freezing the rows freezes the augmentation

For every image the scan reports the term's own per-image record: the selected top-k
token set, the eligible counts and the owner/partner legality checks, the pairing
statistics, the GT size-band distribution, and the A1 (own CPU generator) versus A2
(highest quality) partner choice computed on the SAME forward.  The scan stops after the
frozen 32 images; images are never added and no threshold, weight or formula is touched.

Pre-registered reading (declared before the run, not tuned after it): the scan decides
only whether the three-arm comparison has a basis.  A positive choice-space count means
the A1/A2 control has something to compare; zero means the round registers
"三臂机制比较依据不足" and the three arms are NOT started mechanically, and the method is
NOT judged invalid.  No AP, coverage or benefit claim is made or required here.

usage: CUDA_VISIBLE_DEVICES= <frozen python> -I scan_opportunity_32.py
       --source-root <dev-tree>/source --contract <production-contract-003.json> --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

PLAN_SHA = "d0a3593d14878bf0e0d400e68e35fdec40c33dd742aceae0ab7dd78c4714c11c"
FROZEN_CHECKPOINT_SHA = "13a35283a02d2b728e3449917586ded5cdf2d578ebe6847510748e0dc4d21757"
FROZEN_EPOCH = 109
WINDOWS = 2
LOGICAL_BATCH = 16
SMALL_MAX_AREA = 1024.0
MEDIUM_MAX_AREA = 9216.0
DECISION_RULE = {
    "question": "does the mechanism have a pairing and a control opportunity on the frozen 32 images?",
    "pre_registered": ("choice_space_instances > 0 -> the A1/A2 comparison has a basis (magnitude is "
                       "reported, no post-hoc threshold); == 0 -> register 三臂机制比较依据不足"),
    "not_claimed": "no AP, coverage or benefit claim; thresholds, loss weight and quality formula unchanged",
}
# FROZEN before the scan, from the chain's own order (fetch_opportunity_ids_001.py, epoch 109).
FROZEN_ROWS = (
    (0, 3787, "884d9a56752085c3f2db33d1d497b04875e05971421b2935f01a0e07ea9ed7a4"),
    (1, 236, "36acf2ee64e396422c1fb4079a996d0682019ea744e1acce9965f03ead8e32bb"),
    (2, 3623, "47422fb16bd33f3b112faea6518557135d9377742158aea1bd4879ade1d636c7"),
    (3, 2218, "bc543302c6730493cc70b3f35521be48a9924836c3780b37e7abb88457c8c10b"),
    (4, 3747, "b49ea9eb437cbda22dfb59b9ba6293f51d401878c7fabfbca7dc79e575fba14c"),
    (5, 2697, "b70f9b93290ba7de178ad39b2600713e9684cb477ab0a66e0357db4a76effdae"),
    (6, 2757, "4d218e81630669893dd1be38c47d46575ab8998677313c9686d87136ad24f8ca"),
    (7, 4214, "62f5e19455c61c4bea1b98aa1f039b4066d17cd5387046f162f67b523358f26a"),
    (8, 1670, "84d4d59f2157aa4cd18a5df37035e89b7161606e83d3b800ad77ef46d6f9e242"),
    (9, 937, "62d0abeb6ea3e3de0b7f1508d60b306d321bfd53444ed5ea3e265a8813d5de2f"),
    (10, 178, "8d64995f5e6348424936dd35de6088004ead807fea58784558b169fdd219041b"),
    (11, 1374, "e20c98ed7af172f5bcfb4d7882f5ecda32eb84702b74b9e5ff9a2cb54354ffd9"),
    (12, 4569, "28ed4ac9e2e69231cb5c2d1f4f00e8ed276571d8388ca69b59eaaa4e3448c697"),
    (13, 3928, "bbc74c6eb3ffcd98d975e9b9edf7a5ac1beb1d2ef264cc80109273d25e18c27e"),
    (14, 939, "9f77f9007b2ba05123815f706059c8fa3823978f5cc3cf64c0d2386451d22ae7"),
    (15, 43, "b1289c4377622a1cbb139b2e12b967a79b5fb15a841b85022f5896fde2ee8653"),
    (16, 643, "8494d6d8a99920c9a521e03a68dcbfc331b0984f45a89e2f7f731399a7b05e21"),
    (17, 4175, "b9e8cf5739fae975eda47add24e9d5a12df74756672ca2629a1f52c8d6739a5a"),
    (18, 2994, "6dd0161fb932ca36460a0c883f063f56263af678ec861023d2ece60f5d084d16"),
    (19, 1024, "b8cf0d04b3d9cb32afc9cddc35c54ded7dfa0b24192af871c00f62397da74c1c"),
    (20, 2823, "0bfe98bed0fced29a71c57e10abfa2e16b2f4d325bb29ced6d0b342aa872e2d5"),
    (21, 992, "8cdd44ecb5443ce97c8f4ff6ac0c9a774414e17e5484eae02bccfec53eb00be4"),
    (22, 906, "8cc064ff340eec4e1fab1372a402fe1b3370eaa941b794eab062c46ce3dd6233"),
    (23, 314, "0c8d817644b49b3cd69bdd985a4cdc824026c3283df040c3c1448904e990ff68"),
    (24, 3369, "3b7c9ffd0128ae21a8bb2d76f7b9cc01066b9375933a4436471e00616d5c1f6d"),
    (25, 3044, "0f0fb17154cddb4bed2ac583dcb1f4c46bf22fbadeaf305e00f45e5cd670109d"),
    (26, 3260, "c0762d895f503071c233653b8314d0a9a5a5af371886b3ff5cd3065a9cf881f7"),
    (27, 1130, "f548c263efa80bc2b03fa529717e775a5ed2e9509279e6be1396b67c28cb37ef"),
    (28, 3870, "3c92d6686c13dcd25d87505e18a7bfe95309da6b19ca1d9cebcc75ef6ed911d5"),
    (29, 898, "aa828c8802724e89dbd2b898adbfcf0db6569db7b5a5cb1369f6298c3eba1199"),
    (30, 2033, "1b689eb530de05a111a332942bc30b4423df86ec076ed67c98fc9d538efc4824"),
    (31, 3893, "51795ab14975c24abbdd99229b6cc6bf83eb5749b43485a75d114c8631bb3775"),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def content_digest(value):
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(canonical).hexdigest()


def load_tree(root):
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root))
    import sparse_rtdetr
    require(str(Path(sparse_rtdetr.__file__).resolve()).startswith(str((root / "src").resolve())),
            "sparse_rtdetr resolved outside the staged tree")
    from sparse_rtdetr.protocol_r18.model import Config, build, fp32_outputs
    from sparse_rtdetr.protocol_r18 import checkpoint as native_state
    from sparse_rtdetr.protocol_r18 import slice_finetune as ft
    from sparse_rtdetr.protocol_r18 import selection_rank as sr
    from sparse_rtdetr.protocol_r18.engine import tensor_digest
    from sparse_rtdetr.localization import consumer_geometry as geo
    from sparse_rtdetr.localization.data import collate_window
    from sparse_rtdetr.localization.runio import read as read_json
    from sparse_rtdetr.baseline.training_v2b import logical_batch_indices
    return {"Config": Config, "build": build, "fp32_outputs": fp32_outputs,
            "native_state": native_state, "ft": ft, "sr": sr, "geo": geo,
            "collate_window": collate_window, "read_json": read_json,
            "logical_batch_indices": logical_batch_indices, "tensor_digest": tensor_digest}


def band_of(area):
    if area <= SMALL_MAX_AREA:
        return "small"
    return "medium" if area <= MEDIUM_MAX_AREA else "large"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.source_root).resolve()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    require(not torch.cuda.is_initialized(), "CUDA was initialized before the read-only scan")
    m = load_tree(root)
    print("TREE", root, flush=True)

    contract = m["read_json"](args.contract)
    require(contract.get("scope") == "PRODUCTION"
            and str(contract.get("task_id", "")).startswith("r18-dual-ema-20261031-001"),
            "unexpected contract identity")
    require(contract["contract_sha256"] == content_digest(
        {k: v for k, v in contract.items() if k != "contract_sha256"}), "contract content digest differs")
    require(file_sha(contract["plan"]) == contract["plan_sha256"] == PLAN_SHA, "the frozen plan differs")
    parent = contract["parent_checkpoint"]
    require(file_sha(parent["path"]) == parent["sha256"] == FROZEN_CHECKPOINT_SHA,
            "the parent is not the frozen e120 state")
    binding_ref = contract["parent_binding"]
    require(file_sha(binding_ref["path"]) == binding_ref["sha256"], "the parent binding file differs")
    binding = m["read_json"](binding_ref["path"])

    config = m["Config"](input_hw=(768, 1344), physical_batch_size=1, amp_dtype="float32")
    model, criterion, info = m["build"](config, synthetic=True)
    require(model.decoder.num_queries == 300 and model.decoder.query_select_method == "default",
            "the frozen selection rule differs")
    payload = m["native_state"].read(parent["path"], parent["sha256"], binding)
    initial = m["ft"].initialize_from_parent(model, payload, expected_binding=binding)
    del payload
    require(initial == contract["initial_model_sha256"], "the parent EMA did not load into the exact initial state")
    print("BUILT", info["trainable_parameters"], flush=True)

    data = contract["data"]["train_core"]
    dataset = m["ft"].Dataset("D0", manifest_path=data["manifest"], images_path=data["roots"]["images"],
                              annotations_path=data["roots"]["annotations"], root_binding=data["roots"])
    order = m["logical_batch_indices"](len(dataset), seed=0, epoch=FROZEN_EPOCH,
                                       logical_batch_size=LOGICAL_BATCH, completed_batches=0)
    derived = []
    for window in range(WINDOWS):
        indices = next(order)
        derived.extend((window * LOGICAL_BATCH + offset, int(index)) for offset, index in enumerate(indices))
    require(derived == [(row[0], row[1]) for row in FROZEN_ROWS],
            "the chain's order no longer matches the frozen rows")

    model_training_before = model.training
    state_before = m["tensor_digest"](model.state_dict().items())
    rng_before = torch.random.get_rng_state()
    model.eval()
    require(model.training is False, "the scan did not switch the model to read-only eval mode")
    grad_enabled_before = torch.is_grad_enabled()

    term_a2 = m["sr"].SelectionRankTerm("A2", seed=0, k=300)
    term_a2.attach(model, criterion, consumer=m["geo"])
    term_a2.assert_installed(model, criterion)
    term_a1 = m["sr"].SelectionRankTerm("A1", seed=0, k=300)

    per_image = []
    differences = []
    band_totals = {band: {"gt": 0, "instances": 0, "pairs": 0, "choice_space": 0}
                   for band in ("small", "medium", "large")}
    eligible_histogram = {}
    with torch.no_grad():
        for position, index, image_id in FROZEN_ROWS:
            require(not torch.is_grad_enabled(), "the scan is not running under no_grad")
            pixels, target, trace = dataset[(FROZEN_EPOCH, position, index)]
            require(trace["image_id"] == image_id, ("the loaded image differs from the frozen row", image_id))
            images, targets, traces = m["collate_window"]([(pixels, target, trace)])
            forward_started = time.monotonic()
            output = model(images, targets=targets)
            cache = dict(term_a2._cache or {})
            require(set(cache) == {"pool_logits", "pool_coord_unact", "topk_indices"},
                    "no captured pool batch on the production forward")
            criterion.selection_rank = term_a2
            losses_a2 = criterion(m["fp32_outputs"](output), targets)
            require("loss_rank" in losses_a2, "the ranking term did not run")
            stats_a2 = term_a2.last_stats
            term_a1.bind_manual(cache, consumer=m["geo"])
            criterion.selection_rank = term_a1
            rng_call_before = torch.random.get_rng_state()
            criterion(m["fp32_outputs"](output), targets)
            require(torch.equal(rng_call_before, torch.random.get_rng_state()),
                    "the A1 draw consumed the training RNG stream")
            stats_a1 = term_a1.last_stats
            criterion.selection_rank = term_a2
            row_a2 = stats_a2["per_image"][0]
            row_a1 = stats_a1["per_image"][0]
            areas = targets[0]["raw_area"].tolist()
            require(len(areas) == len(targets[0]["boxes"]), "raw area and GT rows are not aligned")
            records_a2 = {record["gt"]: record for record in row_a2["records"]}
            records_a1 = {record["gt"]: record for record in row_a1["records"]}
            require(set(records_a2) == set(records_a1), "A1 and A2 saw different instances")
            for gt, record in sorted(records_a2.items()):
                require(areas[gt] > 0, "ground truth without raw area")
                band = band_of(areas[gt])
                band_totals[band]["gt"] += 1
                band_totals[band]["instances"] += 1
                if record["partner"] is not None:
                    band_totals[band]["pairs"] += 1
                if record["eligible_count"] >= 2:
                    band_totals[band]["choice_space"] += 1
                    eligible_histogram[str(record["eligible_count"])] = \
                        eligible_histogram.get(str(record["eligible_count"]), 0) + 1
                other = records_a1[gt]
                require(other["eligible_count"] == record["eligible_count"], "A1/A2 eligible counts differ")
                if record["partner"] is not None and other["partner"] != record["partner"]:
                    differences.append({"image_id": image_id, "position": position, "gt": gt,
                                        "raw_area": areas[gt], "band": band,
                                        "eligible_count": record["eligible_count"],
                                        "a2_partner": record["partner"],
                                        "a2_partner_quality": record["quality_better"],
                                        "a1_partner": other["partner"],
                                        "a1_partner_quality": other["quality_better"],
                                        "owner_quality": record["quality_worse"]})
            per_image.append({"position": position, "index": index, "image_id": image_id,
                              "gt_per_band": {band: sum(1 for area in areas if band_of(area) == band)
                                              for band in ("small", "medium", "large")},
                              "instances": row_a2["instances"], "pairs": row_a2["pairs"],
                              "choice_space": row_a2["choice_space"],
                              "eligible_counts": row_a2["eligible_counts"],
                              "selected_count": len(row_a2["selected"]),
                              "selected": row_a2["selected"],
                              "owner_checks": row_a2["owner_checks"],
                              "records": row_a2["records"],
                              "loss_rank": float(losses_a2["loss_rank"]),
                              "a1_partners": {gt: record["partner"] for gt, record in records_a1.items()},
                              "milliseconds": round((time.monotonic() - forward_started) * 1000.0, 3)})
            print("IMAGE", position, image_id[:12], row_a2["instances"], row_a2["pairs"],
                  row_a2["choice_space"], flush=True)

    require(torch.is_grad_enabled() is grad_enabled_before, "the scan changed the global grad mode")
    state_after = m["tensor_digest"](model.state_dict().items())
    rng_after = torch.random.get_rng_state()
    require(state_after == state_before, "the scan changed model parameters or buffers")
    require(torch.equal(rng_before, rng_after), "the scan consumed the training RNG stream")
    require(all(parameter.grad is None for parameter in model.parameters()),
            "the scan left a parameter gradient")
    require(not torch.cuda.is_initialized(), "CUDA was initialized during the scan")

    instances = sum(row["instances"] for row in per_image)
    pairs = sum(row["pairs"] for row in per_image)
    choice_space = sum(row["choice_space"] for row in per_image)
    images_with_pair = sum(1 for row in per_image if row["pairs"] > 0)
    images_with_choice_space = sum(1 for row in per_image if row["choice_space"] > 0)
    observed = choice_space > 0
    report = {
        "status": "PASS",
        "scope": ("read-only CPU opportunity scan on the frozen 32 train_core images; no backward, no "
                  "optimizer, no EMA, no checkpoint, no state change; no AP or benefit claim"),
        "source_root": str(root),
        "contract": {"path": str(Path(args.contract).resolve()), "task_id": contract["task_id"],
                     "contract_sha256": contract["contract_sha256"], "plan_sha256": contract["plan_sha256"],
                     "parent_checkpoint_sha256": parent["sha256"], "initial_model_sha256": initial},
        "config": {"input_hw": [768, 1344], "physical_batch_size": 1, "amp_dtype": "float32",
                   "model_mode": "eval (read-only: running statistics used, never updated; no denoising branch)",
                   "model_training_before": bool(model_training_before)},
        "frozen_set": {"epoch": FROZEN_EPOCH, "windows": WINDOWS, "positions": [row[0] for row in FROZEN_ROWS],
                       "indices": [row[1] for row in FROZEN_ROWS], "images": len(FROZEN_ROWS)},
        "totals": {"instances": instances, "pairs": pairs, "choice_space_instances": choice_space,
                   "images": len(per_image), "images_with_pair": images_with_pair,
                   "images_with_choice_space": images_with_choice_space,
                   "zero_pair_fraction": (instances - pairs) / instances if instances else None,
                   "choice_space_fraction": choice_space / instances if instances else None},
        "size_bands": band_totals,
        "eligible_count_histogram": eligible_histogram,
        "a1_a2": {"draws": term_a1.draws, "differences": len(differences), "examples": differences[:20],
                  "note": "A1 draws uniformly from the same eligible set with its own CPU generator"},
        "read_only_invariants": {"model_state_unchanged": True, "training_rng_unchanged": True,
                                 "no_parameter_gradients": True, "no_optimizer_or_ema_constructed": True,
                                 "grad_disabled_during_the_scan": True, "grad_mode_restored": True,
                                 "cuda_initialized": False},
        "decision_rule": DECISION_RULE,
        "choice_space_observed": observed,
        "registration": ("有对照机会（幅度见 totals，不设事后门槛）" if observed
                         else "三臂机制比较依据不足"),
        "per_image": per_image,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    (out_dir / "OPPORTUNITY-32.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("totals", "size_bands", "a1_a2", "choice_space_observed",
                                                   "registration", "elapsed_seconds")},
                     ensure_ascii=False, default=str), flush=True)
    print("OPPORTUNITY-SCAN-COMPLETE", report["elapsed_seconds"], "s", flush=True)


if __name__ == "__main__":
    main()
