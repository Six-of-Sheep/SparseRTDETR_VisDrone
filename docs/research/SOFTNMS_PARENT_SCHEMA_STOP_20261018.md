# Soft-NMS real parent schema adaptation: preparation STOP

2026-10-18; branch `codex/sahi-softnms-fixed`; PR #4. This revision is NOT production-admitted.

The previous seed0 evaluation RESULT has no task_id. The adapter now reads task identity from CLOSED -> runtime-index -> archived identity and checks phase, paths, source, contract, plan, endpoint and full-C association. Seed1 resolves its certified old endpoint; seed2 resolves its own CLOSED formal endpoint. No original evidence is edited. Gaussian math, sigma, candidate pools, scoring, thresholds and output policy are unchanged.

The new actual `softnms_job.py parent-check` entry ran against real server seed metadata, with CUDA hidden and model/load/scientific processing forbidden. It stopped during seed0 metadata checks after 0.680291031 seconds: `legacy archive checkpoint/images differ`. No large raw candidate batches were read, no score_budget calls occurred, and seed1/2 validation and the new negative cases did NOT run. The previous unrelated CPU algorithm/integration gate remains valid only in its original synthetic scope.

Read-only diagnosis found that the original seed0 deploy.py line 256 assigns full_image_ids only to full-C and fixed64 to every other phase, including evaluation. The full-C archive has the correct 548 IDs; the evaluation archive has exactly the contract's fixed64 IDs. Both checkpoint hashes match. This new adapter incorrectly treated both archive lists as full evaluation membership. This observation is not a finding of corrupt historical evaluation or a change to the original SAHI PASS.

The authored checks include all three metadata formats, missing identities, phase/path/source/contract/endpoint conflicts, cross-seed misuse, full-C association and real SHA corruption. They are NOT claimed to have passed: the positive context stopped before the negative cases. Semantic mutations would be post-authentication read injection with originals unchanged; SHA checks use real isolated corrupt bytes.

A future authorized correction must explicitly distinguish the certified legacy archive workload field from 548-image evaluation membership, retain strict fixed64 checks for that legacy field, and establish full membership from authenticated full-C/evaluation inputs. It must complete actual three-seed context and all rejection cases before cache admission; merely removing the image check is insufficient. No such correction/retry was applied after this STOP.

Remote evidence: `sahi-softnms-fixed-20261017-001/revision-003/parent-check-001/STOP.json`, `SCHEMA_DIFFERENCE.json`, `PARENT_CHECK.stdout`, and PREPARATION_MANIFEST.json under the original Workspace experiment root. Four necessary receipts (4,127 bytes) were returned to D and SHA checked. Models, raw caches and predictions remain on the server. No new production token or scoring attempt was consumed. Old STOPs and costs are preserved. No quality verdict or end-to-end cost validation exists for Soft-NMS.
