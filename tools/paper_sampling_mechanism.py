"""Display existing bounded sampling summaries; never run a detector/evaluator.

Accepted input: frozen sampling-eNNN-w000.json rows, MECHANISM_SUMMARY.json,
or MECHANISM_EFFECT_SUMMARY.json. Quantiles remain per record. Full reference
counts and sampled coordinate statistics are intentionally separate outputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


ARMS = ("B0", "B1", "P")
GROUPS = ("ordinary", "DN")
SCOPES = {
    "gpu-acceptance-only": "GPU engineering acceptance only, not formal training",
    "selected-training-windows": "Selected training windows only; not a population distribution",
    "test-fixture": "SYNTHETIC_TEST_ONLY - not experimental results",
}
METRICS = (
    "reference_extent_cells", "actual_location_normalized_x",
    "actual_location_normalized_y", "actual_offset_cells", "actual_span_cells_x",
    "actual_span_cells_y", "outside_image_fraction", "same_stencil_fraction",
    "bilinear_effective_weight_abs_change", "attention_aggregate_output_abs_change",
)
SAMPLED_METRICS = set(METRICS) - {"reference_extent_cells"}
STATS = ("count", "mean", "min", "p50", "p90", "max")
COUNT_FIELDS = (
    "reference_instances", "reference_below_cell_x", "reference_below_cell_y",
    "reference_below_cell_any", "reference_floor_trigger_fraction_x",
    "reference_floor_trigger_fraction_y", "reference_floor_trigger_fraction_any",
    "floor_applied_instances", "queries_per_image", "sampled_queries_per_image",
    "DN_includes_native_padding",
)
UNKNOWN = "UNKNOWN"


def checked_json(path: Path, expected: str):
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected.lower()):
        raise ValueError("expected SHA256 must contain 64 hexadecimal characters")
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected.lower():
        raise ValueError(f"SHA256 mismatch: {path.name}")
    # Reject the non-standard JSON NaN/Infinity spellings as well as nonfinite numbers.
    def bad_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    return json.loads(data, parse_constant=bad_constant), {
        "path": str(path.resolve()), "size_bytes": len(data), "sha256": actual,
    }


def number(value, label, *, integer=False, fraction=False):
    if value is None:
        return UNKNOWN
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label}: finite number required")
    if integer and (value < 0 or int(value) != value):
        raise ValueError(f"{label}: nonnegative integer required")
    if fraction and not 0 <= value <= 1:
        raise ValueError(f"{label}: fraction outside [0,1]")
    return int(value) if integer else value


def arm_name(value):
    arm = value.removeprefix("accept-") if isinstance(value, str) else value
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {value}")
    return arm


def normalize(payload, *, scope, source="input"):
    """Return literal source statistics. No equal-record or quantile aggregation."""
    if scope not in SCOPES:
        raise ValueError("explicit supported scope required")
    fixture = isinstance(payload, dict) and payload.get("data_kind") == "SYNTHETIC_TEST_ONLY"
    if fixture and scope != "test-fixture":
        raise ValueError("synthetic data cannot be used in an experimental scope")
    if isinstance(payload, dict) and "observations" in payload:
        payload = payload["observations"]
    counts, values = [], []

    def count_row(arm, group, record, obj):
        row = dict(source=source, arm=arm, group=group, record=record)
        for key in COUNT_FIELDS:
            value = obj.get(key)
            if key == "DN_includes_native_padding":
                if value is not None and not isinstance(value, bool):
                    raise ValueError("DN padding flag must be boolean")
                row[key] = UNKNOWN if value is None else value
            else:
                row[key] = number(value, key, fraction="fraction" in key,
                                  integer="fraction" not in key)
        total, below = row["reference_instances"], row["reference_below_cell_any"]
        frac = row["reference_floor_trigger_fraction_any"]
        if total != UNKNOWN and below != UNKNOWN:
            if below > total:
                raise ValueError("below-cell count exceeds reference denominator")
            if total and frac != UNKNOWN and not math.isclose(below / total, frac, abs_tol=1e-6):
                raise ValueError("trigger fraction conflicts with reference counts")
        counts.append(row)

    def metric_row(arm, group, record, metric, statistic, value, count=None):
        values.append(dict(source=source, arm=arm, group=group, record=record,
                           metric=metric, statistic=statistic,
                           value=number(value, metric),
                           count=number(count, metric + " count", integer=True),
                           population="sampled_queries_at_most_8_per_image" if metric in SAMPLED_METRICS
                           else "all_reference_instances_axes_flattened"))

    if isinstance(payload, list):
        for index, row in enumerate(payload):
            if not isinstance(row, dict):
                raise ValueError("observation row must be an object")
            arm, group = arm_name(row.get("arm")), row.get("group")
            if group not in GROUPS:
                raise ValueError("ordinary and DN are the only observation groups")
            if row.get("uses_GT") is not False:
                raise ValueError("raw observation must explicitly declare uses_GT=false")
            layer = number(row.get("layer"), "layer", integer=True)
            if layer == UNKNOWN or layer not in (0, 1, 2):
                raise ValueError("raw observation must name decoder layer 0/1/2")
            sampled = row.get("sampled_queries_per_image")
            if sampled is not None and number(sampled, "sampled_queries", integer=True) > 8:
                raise ValueError("raw sample exceeds frozen at-most-eight-query observation")
            record = f"row{index:04d}:layer{layer}"
            count_row(arm, group, record, row)
            for metric in METRICS:
                entry = row.get(metric)
                if isinstance(entry, dict):
                    for statistic in STATS[1:]:
                        metric_row(arm, group, record, metric, statistic,
                                   entry.get(statistic), entry.get("count"))
                elif entry is None or isinstance(entry, (int, float)):
                    metric_row(arm, group, record, metric, "value", entry)
                else:
                    raise ValueError(f"bad metric structure: {metric}")
        if not payload:
            raise ValueError("empty observation rows")
        return counts, values, "raw_selected_window_rows"

    if not isinstance(payload, dict) or not payload:
        raise ValueError("mechanism input must contain raw rows or arm summaries")
    if any(str(key).startswith("accept-") for key in payload) and scope == "selected-training-windows":
        raise ValueError("acceptance summary cannot be relabelled as training windows")
    for stage, groups in payload.items():
        arm = arm_name(stage)
        if not isinstance(groups, dict) or not groups:
            raise ValueError("arm summary requires ordinary/DN group objects")
        for group, obj in groups.items():
            if group not in GROUPS or not isinstance(obj, dict):
                raise ValueError("bad mechanism group")
            if "records" in obj:
                number(obj["records"], "records", integer=True)
                mapped = dict(obj)
                mapped["reference_below_cell_any"] = obj.get("below_cell_any")
                count_row(arm, group, "aggregate_reference_counts", mapped)
                for metric in METRICS:
                    means = obj.get(f"sampled_{metric}_mean_by_record")
                    if means is None:
                        metric_row(arm, group, "unavailable", metric, "UNKNOWN", None)
                    else:
                        if not isinstance(means, list) or len(means) != obj["records"]:
                            raise ValueError("record mean list length does not match records")
                        for index, mean in enumerate(means):
                            # Per-record counts were not returned in this summary. Do not pool.
                            metric_row(arm, group, f"record{index:04d}_identity_unavailable",
                                       metric, "record_mean", mean)
            else:
                count_row(arm, group, "reference_counts_unavailable", {})
                for metric in METRICS:
                    entry = obj.get(metric)
                    if entry is None:
                        metric_row(arm, group, "unavailable", metric, "UNKNOWN", None)
                    elif isinstance(entry, dict) and "weighted_mean" in entry and "count" in entry:
                        metric_row(arm, group, "source_aggregate", metric,
                                   "source_count_weighted_mean", entry["weighted_mean"], entry["count"])
                    else:
                        raise ValueError(f"bad weighted effect summary: {metric}")
    return counts, values, "existing_arm_summary"


def fmt(value):
    if value == UNKNOWN:
        return UNKNOWN
    if isinstance(value, float):
        return f"{value:.9g}"
    return str(value)


def write_csv(path, rows, columns):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def plots(output, counts, values, watermark):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"B0": "#3568a8", "B1": "#e5a239", "P": "#c34a5e"}
    for group in GROUPS:
        # Only display source-provided means/fractions. No pooled quantile/mean inference.
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        panels = [
            ("actual_span_cells_x", "Actual x span (feature cells)"),
            ("actual_span_cells_y", "Actual y span (feature cells)"),
            ("reference_trigger", "References below one finest cell (fraction)"),
            ("actual_location_normalized_x", "Actual x position (normalized)"),
            ("bilinear_effective_weight_abs_change", "Effective interpolation weight |change|"),
            ("attention_aggregate_output_abs_change", "Attention output |change|"),
        ]
        for ax, (metric, title) in zip(axes.flat, panels):
            found = False
            for j, arm in enumerate(ARMS):
                if metric == "reference_trigger":
                    series = [r["reference_floor_trigger_fraction_any"] for r in counts
                              if r["group"] == group and r["arm"] == arm
                              and r["reference_floor_trigger_fraction_any"] != UNKNOWN]
                else:
                    rows = [r for r in values if r["group"] == group and r["arm"] == arm
                            and r["metric"] == metric and r["value"] != UNKNOWN]
                    # Prefer existing aggregate, otherwise plot every record mean separately.
                    aggregates = [r for r in rows if r["statistic"] == "source_count_weighted_mean"]
                    series = [r["value"] for r in (aggregates or rows)
                              if r["statistic"] in ("source_count_weighted_mean", "record_mean", "mean", "value")]
                if series:
                    found = True
                    xs = [j] if len(series) == 1 else [j + ((i / (len(series) - 1)) - .5) * .25 for i in range(len(series))]
                    ax.scatter(xs, series, s=25, alpha=.8, color=colors[arm], label=arm)
            ax.set_title(title, fontsize=10)
            ax.set_xticks(range(3), ARMS)
            ax.grid(axis="y", alpha=.2)
            if metric == "reference_trigger":
                ax.set_ylim(0, 1)
            elif metric in ("actual_span_cells_x", "actual_span_cells_y",
                            "bilinear_effective_weight_abs_change", "attention_aggregate_output_abs_change"):
                ax.set_ylim(bottom=0)
            if not found:
                ax.text(.5, .5, "UNKNOWN\nnot recorded in supplied evidence", ha="center", va="center",
                        transform=ax.transAxes, color="#666666")
        fig.suptitle(f"Existing mechanism evidence: {group}" + (" (native DN padding included)" if group == "DN" else ""), fontsize=15)
        fig.get_layout_engine().set(rect=(0, .075, 1, .925))
        fig.text(.5, .031, "Counts: all observed references. Positions/spans/effects: at most 8 queries per image; supplied means only.",
                 ha="center", fontsize=9, color="#555555")
        fig.text(.5, .008, watermark, ha="center", fontsize=10, color="#943333", weight="bold")
        for suffix in ("svg", "png"):
            fig.savefig(output / f"mechanism-{group}.{suffix}", dpi=180, bbox_inches="tight")
        plt.close(fig)


def run(args):
    if bool(args.effects) != bool(args.effects_sha256):
        raise ValueError("--effects and --effects-sha256 are required together")
    payload, ref = checked_json(args.input, args.sha256)
    counts, values, kind = normalize(payload, scope=args.scope)
    refs = [ref]
    if args.effects:
        extra, extra_ref = checked_json(args.effects, args.effects_sha256)
        extra_counts, extra_values, _ = normalize(extra, scope=args.scope, source="effects")
        primary_ids = {(r["arm"], r["group"]) for r in counts}
        if {(r["arm"], r["group"]) for r in extra_counts} != primary_ids:
            raise ValueError("effects arm/group identity does not match input")
        counts += extra_counts
        values += extra_values
        refs.append(extra_ref)
    if args.output.exists():
        raise ValueError("output already exists; choose an independent output directory")
    args.output.mkdir(parents=True)
    write_csv(args.output / "reference-counts.csv", counts,
              ("source", "arm", "group", "record") + COUNT_FIELDS)
    write_csv(args.output / "measurements.csv", values,
              ("source", "arm", "group", "record", "metric", "statistic", "value", "count", "population"))
    manifest = dict(schema="paper-sampling-mechanism-display-v1", scope=args.scope,
                    notice=SCOPES[args.scope], input_kind=kind, sources=refs,
                    scientific_quality_evaluated=False,
                    aggregation="none; literal source statistics only",
                    missing_value=UNKNOWN)
    (args.output / "DISPLAY_PROVENANCE.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    lines = ["# Existing sampling mechanism evidence", "", SCOPES[args.scope], "",
             "Full reference counts and at-most-eight sampled queries per image are different populations.",
             "Rows, decoder layers and selected windows are repeated measurements, not independent images.",
             "DN counts include native padding when the source records that flag; summary omission remains UNKNOWN.",
             "Reference width/height floor is not a guaranteed lower bound on actual sampling span.",
             "Same stencil does not imply equal interpolation weights or sampling output.",
             "No AP, quality gate or causal mechanism claim is computed here.", "",
             "## Reference counts", "",
             "|Source|Arm|Group|Record|Reference instances|Below-cell any|Fraction|Floor applied|Sampled queries/image|",
             "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for row in counts:
        keys = ("source", "arm", "group", "record", "reference_instances", "reference_below_cell_any",
                "reference_floor_trigger_fraction_any", "floor_applied_instances", "sampled_queries_per_image")
        lines.append("|" + "|".join(fmt(row[k]) for k in keys) + "|")
    lines += ["", "## Source statistics", "",
              "Every p50/p90 remains attached to its original record. Per-record means with missing counts are not pooled.",
              "Source count-weighted means are copied as reported; this tool does not reconstruct their population.", "",
              "|Source|Arm|Group|Record|Metric|Statistic|Value|Count|", "|---|---|---|---|---|---|---:|---:|"]
    for row in values:
        lines.append("|" + "|".join(fmt(row[k]) for k in ("source", "arm", "group", "record", "metric", "statistic", "value", "count")) + "|")
    lines += ["", "## Interpretation limits", "",
              "Position/offset/quantile/stencil fields omitted by summary files remain UNKNOWN.",
              "Within-forward no-floor differences hold learned offsets, values and attention weights fixed; they are not another trained detector.",
              "Selected training windows do not establish the full-training distribution or why an AP difference occurs.",
              "The supplied SHA proves input bytes, not source authority or experiment closure. Scope is explicit user input.", ""]
    (args.output / "MECHANISM_DISPLAY.md").write_text("\n".join(lines), encoding="utf-8")
    if args.plots:
        plots(args.output, counts, values, SCOPES[args.scope])
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--scope", choices=tuple(SCOPES), required=True)
    parser.add_argument("--effects", type=Path)
    parser.add_argument("--effects-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"mechanism display refused: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
