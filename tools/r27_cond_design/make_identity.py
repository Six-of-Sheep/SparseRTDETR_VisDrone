#!/usr/bin/env python3
"""R27 post-L0 design check -- identity / dependency list generator.

Hashes every input this task relied on, checks them against the records that
already pin them, and writes IDENTITY_001.json. Reads files only; computes
SHA256 only. No model, no CUDA, no network.

Run from the task directory:
    python make_identity.py [--repo C:/Users/liu/repos/SparseRTDETR_VisDrone]
                            [--lit  D:/Lzy_CV/.r27_cond_design_lit/verify]
"""

import argparse
import hashlib
import json
import os
import platform
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R27 = os.path.dirname(HERE)
RESEARCH = os.path.dirname(R27)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def entry(path, base=None):
    rel = os.path.relpath(path, base or R27).replace("\\", "/")
    if not os.path.isfile(path):
        return dict(path=rel, present=False)
    return dict(path=rel, present=True, bytes=os.path.getsize(path),
                sha256=sha(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="C:/Users/liu/repos/SparseRTDETR_VisDrone")
    ap.add_argument("--lit", default="D:/Lzy_CV/.r27_cond_design_lit/verify")
    args = ap.parse_args()
    problems = []

    # ---- 1. task inputs, checked against REVIEW_RECEIPT_001.json ---------
    receipt = json.load(open(os.path.join(HERE, "REVIEW_RECEIPT_001.json"),
                             encoding="utf-8"))
    inputs = []
    for rel, want in receipt["inputs_sha256_unchanged"].items():
        e = entry(os.path.join(R27, rel))
        e["receipt_sha256"] = want
        e["matches_receipt"] = (e.get("sha256") == want)
        if not e["matches_receipt"]:
            problems.append(f"input changed: {rel}")
        inputs.append(e)
    for o in receipt["outputs"]:
        e = entry(os.path.join(HERE, o["name"]))
        e["receipt_sha256"] = o["sha256"]
        e["matches_receipt"] = (e.get("sha256") == o["sha256"])
        if not e["matches_receipt"]:
            problems.append(f"task/review file changed: {o['name']}")
        inputs.append(e)

    # ---- 2. source read, checked against the R27 formal release manifest --
    manifest_path = os.path.join(R27, "return-001", "release-001", "MANIFEST.json")
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    files = manifest["files"]
    if isinstance(files, dict):
        pinned = {k: (v.get("sha256") if isinstance(v, dict) else v)
                  for k, v in files.items()}
    else:
        pinned = {(f.get("path") or f.get("relpath")): f.get("sha256")
                  for f in files}
    source_rel = [
        "vendor/rtdetrv2_pytorch/src/nn/backbone/presnet.py",
        "vendor/rtdetrv2_pytorch/src/nn/backbone/common.py",
        "vendor/rtdetrv2_pytorch/src/zoo/rtdetr/hybrid_encoder.py",
        "vendor/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml",
        "vendor/rtdetrv2_pytorch/configs/rtdetrv2/include/rtdetrv2_r50vd.yml",
        "src/sparse_rtdetr/baseline/training_v2a_contract.py",
        "src/sparse_rtdetr/baseline/training_v2b.py",
    ]
    source = []
    for rel in source_rel:
        e = entry(os.path.join(args.repo, rel), base=args.repo)
        e["release_manifest_sha256"] = pinned.get(rel)
        e["in_release_manifest"] = rel in pinned
        e["matches_release"] = (e.get("sha256") is not None
                                and e.get("sha256") == pinned.get(rel))
        if rel.startswith("vendor/") and not e["matches_release"]:
            problems.append(f"vendored source differs from release: {rel}")
        source.append(e)

    # ---- 3. measured anchors -------------------------------------------
    anchors = [
        entry(os.path.join(RESEARCH, "r26-constrained-detail-encoder-20261109-001",
                           "return-001", "cost", "CLOSED.json"), base=RESEARCH),
        entry(os.path.join(RESEARCH, "r26-constrained-detail-encoder-20261109-001",
                           "REPORT_R26_001.md"), base=RESEARCH),
        entry(os.path.join(R27, "l0-control-002", "return-001", "summary",
                           "RESULT.json"), base=RESEARCH),
        entry(os.path.join(R27, "l0-control-002", "REPORT_R27_L0_CONTROL_002.md"),
              base=RESEARCH),
        entry(manifest_path, base=RESEARCH),
    ]

    # ---- 4. external literature / code evidence fetched for this task ---
    external = dict(
        pinned_commits={
            "thomasverelst/dynconv": "19e4c583b841a372b8d2eed11fa6b6139888b499",
            "Cuogeihong/CEASC": "2abfd1a99f1b0fe1ed3d51588b64549e1584da50",
            "raoyongming/DynamicViT": "1322e626d1e9eca18cc150569908fb4af29e15f7",
        },
        papers_only_code_not_read=[
            "LASNet arXiv 2210.06223 (full text read)",
            "Dynamic Zoom-in Network arXiv 1711.05187",
            "DynamicDet arXiv 2304.05552",
            "SACT arXiv 1612.02297",
            "SBNet arXiv 1801.02108",
            "SegBlocks arXiv 2011.12025",
            "BlockCopy arXiv 2108.09376",
            "BlockDrop arXiv 1711.08393",
            "QueryDet arXiv 2103.09136",
            "ESOD arXiv 2407.16424",
            "Lite DETR arXiv 2303.07335",
            "CRED arXiv 2410.04088 (official repo holds no code)",
        ],
        fetched_files=[],
    )
    if os.path.isdir(args.lit):
        for name in sorted(os.listdir(args.lit)):
            pth = os.path.join(args.lit, name)
            if os.path.isfile(pth) and not name.endswith(".txt"):
                external["fetched_files"].append(entry(pth, base=args.lit))

    # ---- 5. this task's own programs and results -----------------------
    own = [entry(os.path.join(HERE, n), base=HERE) for n in (
        "check_backbone_shapes.py", "check_backbone_shapes.result.json",
        "check_backbone_shapes.stdout.txt",
        "check_set_risk.py", "check_set_risk.result.json",
        "check_set_risk.stdout.txt", "make_identity.py")]

    out = dict(
        task="TASK_CONDITIONAL_COMPUTE_DESIGN_001",
        scope="design verification only: source reading, literature check, "
              "synthetic CPU enumeration",
        runtime=dict(python=sys.version.split()[0],
                     implementation=platform.python_implementation(),
                     platform=platform.platform(),
                     third_party_packages_imported=[],
                     torch_imported=("torch" in sys.modules),
                     cuda_initialised=False,
                     real_model_forward_or_backward=0,
                     optimizer_updates=0,
                     scores=0,
                     teacher_cache_generated=False,
                     server_connections=0),
        release_bound_to=dict(commit=manifest.get("commit"),
                              branch=manifest.get("branch"),
                              manifest=os.path.relpath(manifest_path, RESEARCH)
                              .replace("\\", "/")),
        task_inputs=inputs,
        source_read=source,
        measured_anchors=anchors,
        external=external,
        own_programs_and_results=own,
        problems=problems,
        status="PASS" if not problems else "FAIL",
    )
    with open(os.path.join(HERE, "IDENTITY_001.json"), "w", encoding="utf-8",
              newline="\n") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print("status:", out["status"])
    for pr in problems:
        print("  PROBLEM:", pr)
    print("inputs unchanged vs receipt :",
          sum(e["matches_receipt"] for e in inputs), "/", len(inputs))
    print("vendored source == release  :",
          sum(e["matches_release"] for e in source if e["path"].startswith("vendor/")),
          "/", sum(1 for e in source if e["path"].startswith("vendor/")))
    for e in source:
        if not e["path"].startswith("vendor/"):
            print(f"  project file {e['path']}: in_release={e['in_release_manifest']}"
                  f" matches={e['matches_release']}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
