# Fixed Soft-NMS: count stages and real-data gates

The previous production attempt incorrectly compared the original pre-filter
input size with post-filter official counts. That STOP, its source, records and
cost remain unchanged. This revision corrects verification, not the experiment.

## Stages and counts

1. Each view has Q300 x 10 original FP32 scores; the unchanged selection retains
   top600 per view. Whole-only pools contain600; five-view pools contain3000.
2. A retains whole predictions. B/C apply the original classwise hard NMS.
   E/D apply the frozen Gaussian Soft-NMS, sigma0.5 and FP64 decays. No GT enters
   selection or postprocessing. `original_stages` records actual pool sizes,
   post-NMS counts and budget lengths during exact A/B/C reconstruction.
3. Each output is truncated to top600/300 before raw-ignore, without refill.
4. `canonical_input_sha256` authenticates the entire pre-filter evaluator input,
   including original GT regions and excluded categories. It is not a digest of
   only surviving detections or target GT.
5. The frozen `r0.evaluation.bridge_inputs` directly invokes the original raster
   ignore primitive. Its `detections_before/after` and `GT_before/after` must equal
   the saved `spatial_filter_counts`. `GT_before` counts categories1..10 and does
   not equal all raw annotation rows. `GT_after` retains category score0 crowd
   annotations when spatially eligible. After counts must equal saved official
   `evaluated_detection_count` and `evaluated_ground_truth_count`.
6. The unchanged evaluator subsequently applies its original category, matching,
   maxDets and aggregation rules. Counts do not imply AP or matching equivalence.

No old evaluation is rescored. No alternative filter, inequality relaxation or
replacement of missing historical fields is used.

## Separate preparation and scoring

The real `input-check` CLI supervises one CPU worker with original RAM, disk,
temperature and progress protection; CUDA is hidden, models/backward forbidden.
It verifies the real three-seed metadata gate, authenticates all18 original
A/B/C x600/300 input hashes, regenerates original filtered counts, then verifies
every raw cache and exactly reconstructs A/B/C. E/D predictions are prepared by
the unchanged algorithm. Scoring functions are forbidden throughout preparation.

Immutable METADATA_GATE, EVALUATION_INPUT_PROOF and CACHE_REBUILD_GATE have
independent status. REAL_DATA_GATE binds all three to parents, rows, sources and
plan. Actual-consumer positive/rejection checks precede its final publication.
Only after all preparation passes can a new scoring contract/attempt be frozen.
The scoring entry authenticates that gate and its E/D files and consumes those
predictions, without rebuilding or rescoring original results.

CPU fixtures cover hand-counted none/partial/all filtering, both budget orders,
count and SHA rejection, actual dual-evaluator lifecycle/export and equivalence
of the existing preparation path versus certified-preparation consumption.
Synthetic engineering checks, real input proof, cache reconstruction, CI and
scientific quality remain separate claims. New execution failures STOP with no
automatic retry. No GPU timing, training, new inference or holdout is authorized.

Results and exact run identities belong in the new revision005 delivery report.
