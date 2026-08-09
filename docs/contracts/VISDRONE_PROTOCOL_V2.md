# VisDrone Protocol V2

Protocol V2 is an independent feasibility-first successor to the frozen V1
split design. V1 remains preserved with its historical nearest-size selection
and permanent objective-priority failure on the corrected sequence groups.

V2 keeps the same official train/val identity, seven-digit sequence keys,
duplicate-content connected components, seed `20260808`, salt
`P3-confirmatory-v1`, target of 647 images, five percentage-point tolerance,
development split, disabled test split, and selection/metrics gates.

The V2 planner evaluates every non-empty prefix of the same hash ordering. It
first retains only prefixes that pass group disjointness, image-content
disjointness, development-sequence exclusion, all ten class-ratio checks, and
COCO-small ratio. It then chooses among feasible prefixes by absolute target
distance, not-over-target preference, fewer images, shorter prefix, and the
canonical group-list SHA-256. It never searches arbitrary group combinations.

The selected plan records the V2 policy, evaluated and feasible prefix counts,
selected prefix length, target and absolute difference, group-list SHA, and
all checks. Selection remains closed with
`selection_allowed=false`, `metrics_access_allowed=false`, and
`single_final_access_only=true`. No model outputs or confirmatory metrics can
participate in selection.
