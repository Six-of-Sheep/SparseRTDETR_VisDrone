"""Draw audited, constructed sampling geometry; never import a detector.

The scalar helpers use only the standard library.  Matplotlib is imported only
by render().  This is an illustration of frozen equations, not a model output,
implementation of an attention layer, evaluator, or mechanism logger.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ARMS = {"B0": (4, 4, 4), "B1": (8, 2, 2), "P": (4, 4, 4)}
SHAPES = ((96, 168), (48, 84), (24, 42))  # H, W at 768 x 1344.
DISCLAIMER = "CONSTRUCTED ILLUSTRATION - NOT MODEL OBSERVATIONS"


def _pair(values, name, *, positive=False):
    values = tuple(values)
    if len(values) != 2 or any(isinstance(x, bool) or not isinstance(x, (int, float))
                               or not math.isfinite(x) for x in values):
        raise ValueError(f"{name} must contain two finite numbers")
    if positive and min(values) <= 0:
        raise ValueError(f"{name} must be positive")
    return values


def _shape(shape):
    shape = tuple(shape)
    if len(shape) != 2 or any(type(x) is not int or x <= 0 for x in shape):
        raise ValueError("shape must be two positive integers in H,W order")
    return shape


def normalized_to_feature(point, shape):
    """Real-valued cell-center coordinates for align_corners=False.

    Algebraically x*W-0.5, y*H-0.5; native runtime operation association can
    differ in low precision. No rounding or coordinate clipping is applied.
    """
    x, y = _pair(point, "point")
    height, width = _shape(shape)
    return x * width - 0.5, y * height - 0.5


def sampling_extent(arm, wh, shape, level):
    if arm not in ARMS or type(level) is not int or level not in range(3):
        raise ValueError("expected a known arm and level 0,1,2")
    width, height = _pair(wh, "reference width/height", positive=True)
    feature_h, feature_w = _shape(shape)
    if arm == "P" and level == 0:
        return max(width, 1 / feature_w), max(height, 1 / feature_h)
    return width, height


def native_radial_offsets(n, *, head=1, heads=8):
    """Native *initialization* bias for one head, not learned offsets.

    d_k=k*(cos(theta),sin(theta))/max(abs(cos(theta)),abs(sin(theta))).
    """
    if any(type(x) is not int for x in (n, head, heads)) or n <= 0 or not 0 <= head < heads:
        raise ValueError("invalid point/head inventory")
    angle = head * 2 * math.pi / heads
    direction = math.cos(angle), math.sin(angle)
    divisor = max(map(abs, direction))
    return tuple((k * direction[0] / divisor, k * direction[1] / divisor)
                 for k in range(1, n + 1))


def sampling_points(center, extent, offsets, n):
    center, extent = _pair(center, "center"), _pair(extent, "extent", positive=True)
    offsets = tuple(_pair(p, "offset") for p in offsets)
    if type(n) is not int or n <= 0 or len(offsets) != n:
        raise ValueError("one explicit offset is required per point")
    return tuple(tuple(center[axis] + 0.5 * d[axis] / n * extent[axis]
                       for axis in (0, 1)) for d in offsets)


def actual_span(points):
    points = tuple(_pair(p, "sample point") for p in points)
    if not points:
        raise ValueError("sample points cannot be empty")
    return tuple(max(p[a] for p in points) - min(p[a] for p in points) for a in (0, 1))


def bilinear_neighbors(point, shape):
    """Four raw weights plus validity and effective zero-padding weights."""
    height, width = _shape(shape)
    u, v = normalized_to_feature(point, shape)
    u0, v0 = math.floor(u), math.floor(v)
    fx, fy = u - u0, v - v0
    entries = ((u0, v0, (1-fx)*(1-fy)), (u0+1, v0, fx*(1-fy)),
               (u0, v0+1, (1-fx)*fy), (u0+1, v0+1, fx*fy))
    return tuple({"u": u_i, "v": v_i, "raw_weight": weight,
                  "valid": 0 <= u_i < width and 0 <= v_i < height,
                  "effective_weight": weight if 0 <= u_i < width and 0 <= v_i < height else 0.0}
                 for u_i, v_i, weight in entries)


def example():
    """Explicit numerical illustration; there are no accuracy/result fields."""
    h0, w0 = SHAPES[0]
    center = (3.7 / w0, 3.1 / h0)  # (u,v)=(3.2,2.6).
    wh = (0.45 / w0, 0.65 / h0)
    arms = {}
    for arm, counts in ARMS.items():
        levels = []
        for level, (shape, n) in enumerate(zip(SHAPES, counts)):
            extent = sampling_extent(arm, wh, shape, level)
            offsets = native_radial_offsets(n)
            points = sampling_points(center, extent, offsets, n)
            cell_points = tuple(normalized_to_feature(p, shape) for p in points)
            levels.append({"level": level, "shape_hw": shape, "n": n,
                           "head_shown": 1, "heads_total": 8,
                           "extent_normalized_wh": extent,
                           "offsets_d_native_initialization": offsets,
                           "q_equals_d_over_n": [tuple(x/n for x in d) for d in offsets],
                           "sample_normalized_xy": points,
                           "sample_feature_uv": cell_points,
                           "actual_span_cells_uv": actual_span(cell_points),
                           "uniform_initial_attention_mass": n/12})
        arms[arm] = {"counts": counts, "levels": levels}
    p_point = arms["P"]["levels"][0]["sample_normalized_xy"][1]
    b_point = arms["B0"]["levels"][0]["sample_normalized_xy"][1]
    extent = sampling_extent("P", wh, SHAPES[0], 0)
    zero = sampling_points(center, extent, [(0, 0)]*4, 4)
    return {"schema": "constructed-sampling-diagram-v1", "status": DISCLAIMER,
            "source_commit_read_only": "ab88a9b3b2c8fc1f3203269f4ae022d52b0dd8cb",
            "normalized_reference_cxcywh": (*center, *wh), "shapes_hw": SHAPES,
            "notes": ["One native-initialization head shown, not learned arm offsets.",
                      "B1 changes native radial bias and 1/n scaling together.",
                      "Uniform softmax is over all 12 points: initial finest mass B0/P=1/3, B1=2/3.",
                      "No feature values, model measurements, quality metrics or attention outputs are invented.",
                      "All ordinary and DN queries use the same equation; no GT size is assumed."],
            "arms": arms,
            "bilinear_example_P_k2": {"point": p_point,
                                      "neighbors": bilinear_neighbors(p_point, SHAPES[0])},
            "same_stencil_baseline_B0_k2": {"point": b_point,
                                            "neighbors": bilinear_neighbors(b_point, SHAPES[0])},
            "boundary_example_x0": {"point": (0, center[1]),
                                     "neighbors": bilinear_neighbors((0, center[1]), SHAPES[0])},
            "zero_offset_counterexample": {"extent_cells_wh": (extent[0]*w0, extent[1]*h0),
                                           "sample_feature_uv": [normalized_to_feature(p, SHAPES[0]) for p in zero],
                                           "actual_span_cells_uv": actual_span(zero)}}


def render(output):
    # Imported here so geometry tests never need matplotlib, NumPy, torch or a model.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = example()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(15.6, 10.1), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=(1, 1.06))
    colors = {"B0": "#4477AA", "B1": "#CC6677", "P": "#228833"}
    center = normalized_to_feature(data["normalized_reference_cxcywh"][:2], SHAPES[0])
    ref_wh = (0.45, 0.65)

    def cell_axes(ax, *, xlim, ylim):
        ax.set_aspect("equal")
        ax.set_xticks(range(math.floor(xlim[0]), math.ceil(xlim[1])+1))
        ax.set_yticks(range(math.floor(ylim[0]), math.ceil(ylim[1])+1))
        # Setting ticks can expand limits; restore the requested local crop.
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(color="#dddddd", linewidth=0.8)
        ax.set_xlabel("u (feature-cell center coordinates)")
        ax.set_ylabel("v")
        # Grid coordinates grow downwards as in images.
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)

    for col, arm in enumerate(ARMS):
        ax = fig.add_subplot(grid[0, col])
        level = data["arms"][arm]["levels"][0]
        points = level["sample_feature_uv"]
        cell_axes(ax, xlim=(2.5, 4.05), ylim=(1.95, 3.5))
        ax.set_xticks([2.5, 3, 3.5, 4])
        ax.set_yticks([2, 2.5, 3, 3.5])
        ax.set_title(f"{arm}  |  points per level: {' / '.join(map(str, ARMS[arm]))}",
                     color=colors[arm], fontweight="bold", pad=11)
        ax.add_patch(Rectangle((center[0]-ref_wh[0]/2, center[1]-ref_wh[1]/2),
                               *ref_wh, facecolor="#eeeeee", edgecolor="#666666",
                               linewidth=1.4, linestyle="--"))
        if arm == "P":
            ax.add_patch(Rectangle((center[0]-.5, center[1]-.5), 1, 1,
                                   facecolor="none", edgecolor=colors[arm],
                                   linewidth=1.5, linestyle=":"))
        ax.plot(*center, marker="+", color="black", markersize=11, markeredgewidth=1.8)
        ax.annotate("c", center, xytext=(-14, -10), textcoords="offset points", fontsize=11)
        ax.plot([center[0], points[-1][0]], [center[1], points[-1][1]],
                color=colors[arm], alpha=.45, linewidth=1)
        ax.scatter([p[0] for p in points], [p[1] for p in points],
                   c=colors[arm], s=35, zorder=4, edgecolors="white", linewidths=.5)
        ax.annotate(r"$s_{0mk}$", points[-1], xytext=(14, 13), textcoords="offset points",
                    arrowprops={"arrowstyle": "-", "color": colors[arm]}, color=colors[arm], fontsize=12)
        ew, eh = level["extent_normalized_wh"]
        h, w = SHAPES[0]
        ax.text(.02, .035, f"extent = ({ew*w:.2f}, {eh*h:.2f}) cells\n"
                f"shown head: d[k] = k (1,1), n = {level['n']}",
                transform=ax.transAxes, fontsize=9,
                bbox={"facecolor": "white", "alpha": .9, "edgecolor": "none"})
    handles = [Line2D([0], [0], color="#666666", linestyle="--", label="unchanged reference box (not GT)"),
               Line2D([0], [0], color=colors["P"], linestyle=":", label="P sampling extent on finest level"),
               Line2D([0], [0], marker="o", color="none", markerfacecolor="#555555", label="sampling location (one head shown)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=9, frameon=False)

    ax = fig.add_subplot(grid[1, 0])
    ax.axis("off")
    ax.set_title("Coordinate and offset definitions", loc="left", fontweight="bold", pad=12)
    text = (r"Normalized reference: $r=(c_x,c_y,w,h)$" "\n\n"
            r"$s_{lmk}=c+\frac{1}{2}\,q_{lmk}\odot e_l,\quad q=d/n_l$" "\n\n"
            r"B0, B1: $e_l=(w,h)$" "\n\n"
            r"P: $e_0=(\max(w,1/W_0),\max(h,1/H_0))$" "\n"
            r"     $e_1=e_2=(w,h)$" "\n\n"
            r"$u=xW_l-\frac{1}{2},\quad v=yH_l-\frac{1}{2}$" "\n"
            "align_corners=False; feature centers are integers.\n\n"
            "Top row: native initialization, one of 8 heads.\n"
            "B1 changes d[k] and 1/n together; learned offsets\n"
            "need not match between arms. Initial attention\n"
            "mass on L0 is 1/3 (B0/P), 2/3 (B1).\n\n"
            "L0/L1/L2 grids: 96x168 / 48x84 / 24x42.\n"
            "No regression-reference or GT-size change.")
    ax.text(0, .97, text, va="top", fontsize=10, linespacing=1.25)

    ax = fig.add_subplot(grid[1, 1])
    cell_axes(ax, xlim=(2.75, 4.3), ylim=(1.7, 3.3))
    ax.set_title("Bilinear interpolation: same stencil can differ", fontweight="bold", pad=12, fontsize=10)
    p = data["bilinear_example_P_k2"]
    b = data["same_stencil_baseline_B0_k2"]
    for row in p["neighbors"]:
        ax.scatter(row["u"], row["v"], s=55, c="#555555", marker="s", zorder=3)
        dy = 16 if row["v"] == 2 else -20
        ax.annotate(f"w={row['raw_weight']:.4f}", (row["u"], row["v"]),
                    xytext=(0, dy), textcoords="offset points", ha="center", fontsize=9)
    for label, point, color in (("P k=2", p["point"], colors["P"]),
                                 ("B0 k=2", b["point"], colors["B0"])):
        uv = normalized_to_feature(point, SHAPES[0])
        ax.scatter(*uv, c=color, s=38, zorder=4)
        shift = (35, 8) if label.startswith("P") else (-46, 18)
        ax.annotate(label, uv, xytext=shift, textcoords="offset points", fontsize=9, color=color,
                    arrowprops={"arrowstyle": "-", "color": color})
    ax.text(.02, .98, "P k=2 weights; zero padding, no clip.\nNo feature values or attention outputs shown.",
            transform=ax.transAxes, fontsize=8.6, va="top",
            bbox={"facecolor": "white", "alpha": .93, "edgecolor": "none"})

    ax = fig.add_subplot(grid[1, 2])
    cell_axes(ax, xlim=(2.5, 4.05), ylim=(1.95, 3.5))
    ax.set_title("Extent floor does not bound actual span", fontweight="bold", pad=12, fontsize=10)
    ax.add_patch(Rectangle((center[0]-.5, center[1]-.5), 1, 1,
                           facecolor="#e9f3e8", edgecolor=colors["P"], linestyle=":"))
    ax.plot(*center, marker="+", color="black", markersize=13, markeredgewidth=1.7)
    ax.annotate("all 4 samples at c", center, xytext=(8, -25), textcoords="offset points",
                fontsize=10, arrowprops={"arrowstyle": "->", "color": "#555555"})
    ax.text(.03, .05, "Constructed zero-offset counterexample:\n"
            "d[k] = (0,0) for every k\n"
            "P extent = (1,1) cell; actual span = (0,0).\n"
            "actual span = max(s) - min(s), per axis.",
            transform=ax.transAxes, fontsize=9, va="bottom",
            bbox={"facecolor": "white", "alpha": .93, "edgecolor": "none"})
    fig.suptitle("Reference width/height floor in sparse sampling\n" + DISCLAIMER,
                 fontsize=15, fontweight="bold")
    paths = []
    for suffix in ("svg", "png"):
        path = output / f"sampling_schematic.{suffix}"
        fig.savefig(path, dpi=200)
        paths.append(path)
    plt.close(fig)
    path = output / "sampling_schematic_numbers.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    paths.append(path)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new local figure directory")
    args = parser.parse_args(argv)
    for path in render(args.output):
        print(path.resolve())


if __name__ == "__main__":
    main()
