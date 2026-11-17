"""Read-only scan of the frozen audit stream for the tiny-band extra-positives round.

usage: python3 tools/tiny_verify.py <job_root> <out_json>

Scans every published update audit of the three trained arms (A0 native, A1 the native
matched-pair duplicated-regression control, A2 method), aggregates the declared observation
fields per arm and per family class (final / aux / enc / dn), checks internal consistency (the
native arm adds no extra positive and records no positives mode; the armed arms record their own
positives mode and their admission ledger closes) and writes the summary the closeout receipt
embeds.  The counts are matched-pair/instance OBSERVATION counts per family and per criterion
call, not independent ground-truth counts, and after the training fork every arm is scanned and
reported from its own run.  The scanned observation fields carry the ``pos_`` prefix of the r19
term; the output schema and keys are otherwise the r18 ones.
"""
import hashlib
import json
import sys
from pathlib import Path

ARMS = ("A0", "A1", "A2")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def family(head):
    if head == "final":
        return "final"
    for prefix in ("aux", "enc", "dn"):
        if str(head).startswith(prefix):
            return prefix
    return "other"


def main():
    job = Path(sys.argv[1])
    out = Path(sys.argv[2])
    summary = {}
    for arm in ARMS:
        audits = sorted((job / ("formal-%s" % arm) / "audits").glob("update-*.json"))
        assert audits, "no audits for " + arm
        rows = 0
        totals = {"pairs": 0, "tiny": 0, "band_gt": 0, "candidates": 0, "appended": 0,
                  "no_sibling": 0, "no_native": 0, "zero_iou": 0, "native_sum": 0.0,
                  "applied_sum": 0.0, "extra_l1_sum": 0.0, "extra_giou_sum": 0.0, "shifted": 0}
        per_family = {}
        for path in audits:
            result = load(path)["result"]
            for diag in result.get("normalizers", ()):
                for row in diag:
                    if "pos_pairs" not in row:
                        continue
                    rows += 1
                    fam = family(row.get("head"))
                    agg = per_family.setdefault(fam, {"rows": 0, "pairs": 0, "tiny": 0, "band_gt": 0,
                                                      "candidates": 0, "appended": 0, "no_sibling": 0,
                                                      "no_native": 0, "native_sum": 0.0, "applied_sum": 0.0,
                                                      "extra_l1_sum": 0.0, "extra_giou_sum": 0.0, "shifted": 0})
                    shifted = 1 if row.get("pos_extra_mode") is not None else 0
                    agg["rows"] += 1
                    pairs = int(row["pos_pairs"])
                    tiny = int(row["pos_band_pairs"])
                    agg["pairs"] += pairs
                    agg["tiny"] += tiny
                    agg["band_gt"] += int(row["pos_band_gt"])
                    agg["candidates"] += int(row["pos_extra_candidates"])
                    agg["appended"] += int(row["pos_extra_pairs"])
                    agg["no_sibling"] += int(row["pos_band_gt_no_sibling"])
                    agg["no_native"] += int(row["pos_extra_no_native"])
                    agg["native_sum"] += float(row["pos_band_native_iou_sum"])
                    agg["applied_sum"] += float(row["pos_extra_iou_sum"])
                    agg["extra_l1_sum"] += float(row["pos_extra_l1_sum"])
                    agg["extra_giou_sum"] += float(row["pos_extra_giou_sum"])
                    agg["shifted"] += shifted
                    totals["pairs"] += pairs
                    totals["tiny"] += tiny
                    totals["band_gt"] += int(row["pos_band_gt"])
                    totals["candidates"] += int(row["pos_extra_candidates"])
                    totals["appended"] += int(row["pos_extra_pairs"])
                    totals["no_sibling"] += int(row["pos_band_gt_no_sibling"])
                    totals["no_native"] += int(row["pos_extra_no_native"])
                    totals["zero_iou"] += int(row["pos_band_zero_iou"])
                    totals["native_sum"] += float(row["pos_band_native_iou_sum"])
                    totals["applied_sum"] += float(row["pos_extra_iou_sum"])
                    totals["extra_l1_sum"] += float(row["pos_extra_l1_sum"])
                    totals["extra_giou_sum"] += float(row["pos_extra_giou_sum"])
                    totals["shifted"] += shifted
        if arm == "A0":
            assert totals["applied_sum"] == 0.0, "native arm added extra positives"
            assert totals["appended"] == 0 and totals["candidates"] == 0, "native arm recorded gate work"
            assert totals["shifted"] == 0, "native arm recorded a positives mode"
        else:
            assert totals["shifted"] >= 1, "%s positives ledger empty" % arm
            assert totals["no_native"] == 0, "%s recorded a qualifying box without a native pair" % arm
            assert totals["appended"] + totals["no_sibling"] + totals["no_native"] == totals["band_gt"], \
                "%s admission accounting does not close" % arm
            assert (totals["candidates"] > 0) == (totals["appended"] > 0) \
                and totals["candidates"] >= totals["appended"], \
                "%s admissible-instance count contradicts the appended count" % arm
        totals["tiny_fraction"] = (totals["tiny"] / totals["pairs"]) if totals["pairs"] else None
        totals["appended_fraction_of_band_gt"] = (totals["appended"] / totals["band_gt"]) if totals["band_gt"] else None
        totals["appended_iou_mean"] = (totals["applied_sum"] / totals["appended"]) if totals["appended"] else None
        for agg in per_family.values():
            agg["appended_iou_mean"] = (agg["applied_sum"] / agg["appended"]) if agg["appended"] else None
        summary[arm] = {"audits_scanned": len(audits), "geo_rows": rows,
                        "totals": totals, "per_family": per_family}
    body = json.dumps({"schema": "r18-tiny-positives-training-signal-v1", "arms": summary,
                       "note": ("observation-only fields from the frozen audit stream; counts are matched-pair/"
                                "instance observation counts per family and per criterion call, not independent "
                                "ground-truth counts, and each arm is scanned and reported from its own run after "
                                "the training fork; the native arm adds no extra positive and records no positives "
                                "mode, the A1 control and A2 method each record their own positives mode and their "
                                "admission accounting closes, and the DN families carry no observations by "
                                "declaration")},
                      indent=1, ensure_ascii=False, sort_keys=True)
    out.write_bytes(body.encode("utf-8"))
    print(json.dumps({"status": "VERIFY-OK", "path": str(out), "bytes": len(body),
                      "sha256": hashlib.sha256(body.encode()).hexdigest()}))


if __name__ == "__main__":
    main()
