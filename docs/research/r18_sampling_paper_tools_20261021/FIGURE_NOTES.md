# B0 / B1 / P sampling illustration

`tools/paper_sampling_diagram.py` reads no checkpoints or experiment results. It uses scalar geometry from the read-only frozen commit `ab88a9b3b2c8fc1f3203269f4ae022d52b0dd8cb`: native `MSDeformableAttention` initialization and four-dimensional-reference forward, `sampling_span.sampling_references`, and the coordinate convention in `span_observation.bilinear`. The plotting branch does not carry a modified copy of these execution files.

## Generate

Python 3.10+ and Matplotlib are required only for rendering. Geometry tests need the standard library.

```powershell
python -B tools/paper_sampling_diagram.py --output <local-figure-directory>
python -B -m unittest discover -s tests -p test_paper_sampling_diagram.py
```

The output directory receives `sampling_schematic.svg`, a 200-dpi PNG, and `sampling_schematic_numbers.json`. The SVG keeps editable text. Generated output is outside Git by default. All three artifacts state **CONSTRUCTED ILLUSTRATION - NOT MODEL OBSERVATIONS**. The numerical example contains no accuracy, loss, measured feature value or model-output claim.

## Reading the figure

For normalized reference `r=(cx,cy,w,h)`, point count `n_l` and learned offset `d_lmk`, the unchanged formula is `s_lmk=c+0.5*(d_lmk/n_l)*extent_l`. B0 uses 4/4/4 points; B1 uses native 8/2/2; P keeps 4/4/4 and changes only the finest sampling extent to `(max(w,1/W0),max(h,1/H0))`. Reference centers, regression references and positional encodings are unchanged. The rule applies to both ordinary and DN queries; the reference is not a GT box.

At `align_corners=False`, the real-valued feature coordinates are `u=x*W_l-0.5, v=y*H_l-0.5`. Integer coordinates identify feature-cell centers. Native computation passes through `g=2*x-1` and uses `((g+1)*W-1)/2`; this diagram is a real-valued geometric illustration, not a low-precision floating-point equivalence test. The figure shows a local crop of the actual finest 96x168 grid at input 768x1344, using a constructed 0.45x0.65-cell reference.

The top row depicts **one head's native initialization**, with radial bias `d[k]=k*(1,1)` and `q=d/n`. It does not claim learned offsets match across arms. B1's denser finest-level samples reflect both a different native radial bias and `1/n` scaling; there are 12 points per head in every arm. Native attention softmax covers all 12 points. At uniform initialization its finest-level mass is 1/3 for B0/P and 2/3 for B1, an additional conventional-control difference. The other two grids are 48x84 and 24x42; the top panels display only level 0, while the per-level counts and full numerical example identify all levels.

The interpolation panel gives the four effective weights at P's second point. B0's second point shares those four cell coordinates but has different weights. Equal stencils therefore do not establish equal interpolation or equal attention outputs. No feature values or attention aggregation outputs are supplied. Invalid neighbors have zero effective weight; sample coordinates are not clipped. The JSON includes a separate boundary example at normalized `x=0`, where the feature coordinate is `u=-0.5` and part of the bilinear support is outside the image.

The zero-offset counterexample has a 1x1-cell sampling extent and actual span 0x0. A lower bound on reference width/height does not imply a lower bound on point separation or `max(s)-min(s)`. Learned offsets may shrink, collapse or grow. Neither diagram nor this counterexample establishes the prevalence of any configuration in the running model.

## Proposed caption

**Constructed sampling geometry for B0, B1 and P.** One native-initialization head is shown on a local crop of the finest feature grid; this is not a model observation. B1 reallocates the fixed 12 samples per head from 4/4/4 to 8/2/2. P retains 4/4/4 and floors only the finest sampling reference extent at one feature cell per axis. Continuous sample positions retain bilinear interpolation, zero padding and the original coordinate mapping. Shared interpolation neighbors can have different weights; a reference-extent floor does not guarantee a nonzero actual sample span. Learned offsets and attention weights may differ between trained arms.

## Checks

The scalar tests cover coordinate mapping, finest-only adaptation, native radial initialization and `1/n`, unclipped locations, interpolation weights, boundary zero padding, equal-stencil/different-weight behavior, the zero-offset counterexample, explicit constructed-data provenance, invalid-input refusal, and import without torch or Matplotlib. Rendering and PNG visual review are separately recorded in the local delivery; unit geometry checks do not validate a detector or experimental mechanism.
