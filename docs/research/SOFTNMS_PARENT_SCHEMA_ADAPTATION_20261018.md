# Fixed Soft-NMS: authenticated parent formats

This revision keeps the frozen Gaussian sigma=.5, candidate pool, precision, ranking, metrics and quality gates. It supports three actual closed parent histories without rewriting them. No detector training, construction, new inference, GPU timing or Windows polling is authorized.

| Evidence | seed0 fixed release | seed1 protocol resume | seed2 protocol resume |
|---|---|---|---|
| task/source/contract | CLOSED -> runtime-index -> archived identity; legacy evaluation RESULT has no task/checkpoint | RESULT plus archived identity, explicit seed/scope | same |
| checkpoint | contract checkpoint and original resumed e120 endpoint (26752+9728=36480) | parent_reuse certification -> original formal CLOSED endpoint | own formal CLOSED endpoint |
| archived image membership | full-C image_ids=548; evaluation image_ids=exact contract fixed64 | archived evaluation_authorization.image_ids=548; fixed64 remains a separate field | same |
| evaluated membership | authenticated original data manifest + raw GT + saved prediction refs -> recreated canonical_input_sha256 against all six saved metric records | same; metric sahi_binding also includes checkpoint/seed | same |
| generating code | source manifest authenticates deploy.py/evaluate_cached.py/chain.py; parent manifest authenticates data/evaluator/canonicalization | bound manifest authenticates infer_seed/new_entry/new_runtime and shared data/evaluator code | same |

The legacy archive field describes the original supervisor binding. It does not prove evaluation membership. The original evaluate_cached.run constructs Development548 from the bound parent development manifest, reads the authenticated prediction caches and scores those inputs. The saved official canonical input SHA binds ordered image metadata, raw GT, detections and evaluator settings. Rebuilding this SHA (without evaluating AP) proves the exact 548-member input; images=548 alone is insufficient. All 18 original A/B/C x 600/300 x three seeds must match before reading new pre-NMS batches. Original AP points are never rescored.

Admission first checks complete parent_metadata/context, exact archive references, source bytes, e120 lineage, source/metric/prediction bindings and fixed member order. The actual snapshot test reads authenticated original bytes. Semantic rejection cases alter only in-memory reads after SHA authentication; original parent files remain immutable. Real isolated corruption exercises the SHA checker separately. Member omission, duplicates, reordering, task/seed/phase/source/path/contract/checkpoint conflicts and cross-seed contract reuse must be rejected.

The production job is a new attempt, strictly consumes a new contract, verifies the real-parent CPU gate source binding, recreates all saved canonical evaluator input hashes, then checks all raw-view hashes and identities and exact original A/B/C reconstruction. Only after all three seeds are admitted does it make the fixed 12 E/D two-budget scoring calls using both original evaluators. Any unresolved identity, SHA, numerical or resource error STOPs without retry. Quality failure completes the fixed comparison and closes this candidate. Quality success is cost pending.

Previous revision-003 CPU STOP, earlier JSON/schema STOPs, source releases and cost remain unchanged. The new diagnostic inventory initially corrected a manifest-reference dict/path assumption; its complete second inventory collected 118 small metadata/source files, all three seeds, no raw/prediction/model reads. The diagnostic writer initially attempted to resolve its own new output through the parent snapshot table; this preparation-only bug was corrected. Both original diagnostic artifacts are retained. These are not production retries.

The server holds authenticated snapshots and large predictions. D receives only the report, compact gates/results, source manifest and necessary receipts. PR #4 is updated, never automatically merged. CI engineering status, parent acceptance, cache acceptance and scientific quality are reported separately.
