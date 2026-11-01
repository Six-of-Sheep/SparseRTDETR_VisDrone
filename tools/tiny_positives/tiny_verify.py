"""Read-only scan of the frozen audit stream for the tiny-band extra-positives round.

usage: python3 tools/tiny_verify.py <job_root> <out_json>

Scans every published update audit of the three trained arms (A0 native, A1 control, A2 method),
aggregates the declared observation fields per arm and per family class (final / aux / enc / dn),
checks internal consistency (the native arm adds no extra positive and records no positives mode;
the A1 and A2 arms record their own positives mode) and writes the summary the closeout receipt
embeds.  The scanned observation fields now carry the ``pos_`` prefix of the r19 term; the output
schema and keys are otherwise the r18 ones.
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
        totals = {"pairs": 0, "tiny": 0, "zero_iou": 0, "native_sum": 0.0, "applied_sum": 0.0, "shifted": 0}
        per_family = {}
        for path in audits:
            result = load(path)["result"]
            for diag in result.get("normalizers", ()):
                for row in diag:
                    if "pos_pairs" not in row:
                        continue
                    rows += 1
                    fam = family(row.get("head"))
                    agg = per_family.setdefault(fam, {"rows": 0, "pairs": 0, "tiny": 0,
                                                      "native_sum": 0.0, "applied_sum": 0.0, "shifted": 0})
                    shifted = 1 if row.get("pos_extra_mode") is not None else 0
                    agg["rows"] += 1
                    pairs = int(row["pos_pairs"])
                    tiny = int(row["pos_band_pairs"])
                    agg["pairs"] += pairs
                    agg["tiny"] += tiny
                    agg["native_sum"] += float(row["pos_band_native_iou_sum"])
                    agg["applied_sum"] += float(row["pos_extra_iou_sum"])
                    agg["shifted"] += shifted
                    totals["pairs"] += pairs
                    totals["tiny"] += tiny
                    totals["zero_iou"] += int(row["pos_band_zero_iou"])
                    totals["native_sum"] += float(row["pos_band_native_iou_sum"])
                    totals["applied_sum"] += float(row["pos_extra_iou_sum"])
                    totals["shifted"] += shifted
        if arm == "A0":
            assert totals["applied_sum"] == 0.0, "native arm added extra positives"
            assert totals["shifted"] == 0, "native arm recorded a positives mode"
        else:
            assert totals["shifted"] >= 1, "%s positives ledger empty" % arm
        totals["tiny_fraction"] = (totals["tiny"] / totals["pairs"]) if totals["pairs"] else None
        summary[arm] = {"audits_scanned": len(audits), "geo_rows": rows,
                        "totals": totals, "per_family": per_family}
    body = json.dumps({"schema": "r18-tiny-positives-training-signal-v1", "arms": summary,
                       "note": ("observation-only fields from the frozen audit stream; the native arm adds no "
                                "extra positive and records no positives mode, the A1 control and A2 method each "
                                "record their own positives mode, and the DN families carry no observations by "
                                "declaration")},
                      indent=1, ensure_ascii=False, sort_keys=True)
    out.write_bytes(body.encode("utf-8"))
    print(json.dumps({"status": "VERIFY-OK", "path": str(out), "bytes": len(body),
                      "sha256": hashlib.sha256(body.encode()).hexdigest()}))


if __name__ == "__main__":
    main()
