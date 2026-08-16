# RT-DETRv2 R18 VisDrone Formal Training Contract V1

This document records owner decision `T1_RANDOM_INITIALIZATION` for
`rtdetrv2_r18_visdrone_baseline_training_v1`. It preserves baseline identity
`rtdetrv2_r18_visdrone_baseline_v1`: PResNet-18 is randomly initialized with
seed 0, `pretrained=false`, and `checkpoint=null`. Network downloads,
pretrained caches, external checkpoints, seed selection, retry, resume, and
overwrite are forbidden. A pretrained route requires a new baseline and
training identity.

The portable authority is
`configs/baseline/rtdetrv2_r18_visdrone_training_v1.json`. It binds the raw
and canonical training contract, baseline config, frozen vendor commit and
recipe files, the complete vendor runtime tree, its upstream manifest,
Conversion R3, category mapping, model identity, and allowed data roles. The
current training config identity is raw size 8513 bytes with SHA-256
`bf6631644d218fe65998dcd4267d93a040b709a864577f7216052dd2a9e97239`; its
canonical identity is 7227 bytes with SHA-256
`7439f0706af2bfa97d2a3b17a1b338bb00c6470b4c1268391047014370329c3f`. It
contains no runtime data root, host path, username, or AutoDL path.

## Frozen recipe

The run is one non-distributed `cuda:0` process. Train micro-batch and
effective batch are both 16 with no gradient accumulation; train workers are
4. Development batch/workers are 32/4. Train drops its incomplete last batch;
development does not. Runtime batch adaptation is forbidden: an unsuccessful
resource probe requires a new contract revision.

Training lasts 120 epochs. Development evaluation and `last` checkpoint occur
every epoch, periodic checkpoints every 10 epochs, and a final checkpoint is
required. The upstream R18 augmentation order is frozen. Photometric distort,
zoom-out, and IoU crop stop at epoch 117. Multiscale collation is disabled.

AdamW uses default LR `1e-4`, backbone non-norm LR `1e-5`, betas
`[0.9,0.999]`, default weight decay `1e-4`, zero norm/BN weight decay, and
max gradient norm `0.1`. Parameter groups must be mutually exclusive and
cover every trainable parameter by identity. Regex match counts are not proof.

Linear warmup lasts 2000 optimizer steps. MultiStepLR steps once per epoch,
has milestone 1000 and gamma 0.1, and therefore has zero decay events during
120 epochs. Milestone 1000 must never be interpreted as an iteration.

AMP and GradScaler use the owner-selected Torch 2.4.1 defaults: init scale
`65536.0`, growth factor `2.0`, backoff factor `0.5`, and growth interval
`2000`. These are exact executable parameters, not inferred runtime defaults.
Non-finite loss/gradient, overflow, and skipped optimizer steps are forbidden.
EMA uses decay 0.9999 with a 2000-optimizer-update warmup.
Development evaluation and model selection use EMA; raw and EMA states are
both retained.

## Evaluation and checkpoints

Primary evaluation is `visdrone_official_style_v1`, development-only, every
epoch. Selection maximizes unrounded float64
`AP@[0.50:0.95,maxDets=500]`, then AP50, AR500, and earlier epoch. The vendor
COCO evaluator is diagnostic only and cannot certify or select. The primary
evaluator is not yet independently certified, so training launch remains
blocked. No post-hoc accuracy threshold is defined for this first baseline.

Each checkpoint must atomically bind raw model, EMA, optimizer, scheduler,
warmup, GradScaler, epoch, global optimizer step, RNG states, and config,
source, and environment identities. File and directory fsync, readback,
loadability, SHA-256, and inventory are mandatory. Interrupted runs are
`PERMANENT_FAIL`; exactly-once run identity forbids retry and overwrite.

Runtime integrity, evidence integrity, checkpoint integrity, development
accuracy reporting, model-selection certification, confirmatory access, test
access, and speed measurement are separate states. Training certification
requires all 120 epochs and expected optimizer steps, finite loss/gradients,
no AMP skip/overflow, exit code zero, bound terminal evidence, and loadable
checkpoints. Confirmatory and test remain inaccessible.

## Public API

`sparse_rtdetr.baseline.training_contract` exposes:

- `load_training_contract(repo_root, config_path)`
- `validate_training_contract(config, baseline_config)`
- `canonical_training_contract_bytes(config)`
- `training_contract_binding(repo_root, config_path)`

The module is standard-library-only and has no import-time filesystem access.
It does not import torch, construct a model or dataset, access runtime data, or
create output. This phase does not implement a trainer, evaluator, launcher,
checkpoint writer, or training evidence writer.

## Runtime path identity boundary

Runtime loading accepts only an absolute, lexically normalized POSIX
`repo_root`. The boundary rejects NUL bytes, backslashes, empty components,
`.` and `..` before opening anything. It walks the root from `/` with retained
directory file descriptors and `O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC`; every
component is checked with `lstat` and `fstat`, and the final descriptor path
must equal the supplied lexical root. Ancestors may cross devices, but every
managed descendant after the repository root must remain on the repository
device.

Training, baseline, vendor recipe, and vendor include files are read through
the same verified root descriptor boundary. Each relative path rejects
absolute, empty, dot, dot-dot, backslash, and NUL components. Intermediate
components must be real directories. A final object must be a non-symlink
regular file with link count one and the repository device identity. It is
opened with `O_NOFOLLOW|O_CLOEXEC|O_NONBLOCK`, read through the descriptor,
and checked with stable device, inode, mode, link-count, size, mtime, and ctime
snapshots before and after the read. The root and descendant path chains are
revalidated after the read, so symlinks, hardlinks, special files, canonical
escapes, replacement, and metadata/content races fail closed before their
contents can be accepted.

## Vendor runtime and manifest binding

`training_contract_binding` validates the complete
`vendor/rtdetrv2_pytorch` tree in the same verified repository transaction as
the training config and source bindings. The frozen runtime inventory contains
124 ordinary files, 25 descendant directories, and 373735 total bytes. Its
compact rows use exactly `{relative_path,size_bytes,sha256}` relative to the
vendor root and have canonical inventory SHA-256
`0fc6803665bc4b5720e983345b2cacb0147eceead9f882588880b6f8a0e68051`.

The repository manifest at `manifests/rtdetrv2_upstream.json` is independently
bound by raw size 33264 bytes and raw SHA-256
`f65a2d475365346a5dd5ce4f46b022a135b187e21421e922cc41eee4d28d20ae`. Its
canonical five-field rows use exactly
`{relative_path,size_bytes,sha256,executable,source_role}` relative to the
repository root and have canonical inventory SHA-256
`2312c80d5b0fba88d43ffc6807c3fc150ae74b77740f2ab65072f044e033d6d7`.

The production inventory is generated from descriptor-relative `lstat` and
`openat` operations with `O_NOFOLLOW|O_CLOEXEC` (and `O_NONBLOCK` for files),
deterministically sorted directory enumeration, ordinary-file and link-count
checks, same-device checks, descriptor reads, and metadata revalidation before
and after each read and directory walk. Symlinks, hardlinks, special files,
device drift, path escape, entry changes, partial reads, and descriptor leaks
fail closed. The observed compact rows and manifest rows must independently
match the frozen configuration and manifest; updating a vendor file together
with a repacked manifest remains rejected by the frozen raw manifest and
configuration identities. The returned `vendor_runtime_binding` reports only
the verified observation, including file/directory counts, bytes, both
inventory identities, and manifest raw identity.

`load_training_contract` reads the training and baseline bytes in one verified
boundary transaction. `training_contract_binding` reuses that transaction's
validated configuration and training bytes, then reads all vendor sources
through the same boundary; it does not resolve or reopen the training path a
second time.

## Conversion R3 runtime binding

`load_training_contract` is intentionally portable: it validates the checked-in
training and baseline declarations without scanning the 304 MB Conversion R3
runtime closure. A complete runtime proof is provided only by
`training_contract_binding`. That API must observe and validate
`artifacts/data/visdrone_protocol_v2_conversion_r3` inside the same verified
repository boundary as the training, baseline, vendor, and manifest bindings;
if the closure or any authority file is absent, it fails closed.

The frozen Conversion R3 closure is a flat directory containing exactly 25
ordinary, single-link files, no descendant directories, and 304418794 total
bytes. Its observed identity is bound to the training declaration by the
completion, artifact-inventory, config, category-contract, and source-identity
SHA-256 values. `artifact_inventory.json` contains 23 strictly sorted artifact
rows and excludes exactly `artifact_inventory.json` and `completion.json`.
Every row is compared with the descriptor-based observation, while the
authority files are parsed with duplicate-key rejection, exact schemas,
builtin-type checks, canonical JSON rules, and strict cross-file references.
The returned `conversion_r3_runtime_binding` reports the observed counts,
bytes, path, and authority identities; it is not a copy of the config
declaration.

Large artifacts are hashed by streaming from stable file descriptors; only the
small authority JSON files are retained for parsing. Directory enumeration,
file identity, metadata, device, mode, link count, and containment are checked
before and after observation, and all open descriptors close on success and
failure. Missing, added, renamed, repacked, self-consistent-but-nonfrozen,
symlinked, hard-linked, special, nested, or metadata-drifting objects are
rejected. Consequently, changing a local authority file together with its
internal inventory references cannot establish a new Conversion R3 identity.

Every object role in the portable contract is closed by one recursive static
schema descriptor. Root objects, nested objects, objects inside lists, lists,
and scalar roles are validated before path/SHA semantics, cross-field rules,
and the frozen canonical digest. Missing, extra, and renamed keys fail with
their JSON pointer. Builtin dictionary insertion order is irrelevant, while
list length and order remain part of the frozen contract. Integer and boolean
roles are distinct, floats must be finite, and the four GradScaler fields use
strict builtin numeric types with exact frozen literals.

All 54 builtin numeric leaves are also covered by a static JSON-pointer
constraint registry before path, cross-field, and digest validation. Each role
declares its exact builtin type, finite requirement, basic legal range, and
frozen literal. Errors distinguish non-finite values, values outside the legal
domain, and in-range values that differ from the frozen decision. In
particular, the initialization seed is exactly integer zero, not an arbitrary
non-negative seed. Cross-field checks separately bind effective batch,
single-run policy, epoch completion, AMP/EMA requirements, evaluator launch
blocking, and exactly-once checkpoint ownership. The final canonical digest
remains a separate identity layer rather than a substitute for these semantic
checks.

The semantic layer is a single static registry of 123 in-scope non-numeric
leaves. It covers the contract and owner decisions, model and device modes,
initialization source, optimizer and scheduler roles, AMP enabled/scaler roles, EMA
weight selection, R18 transform identity, development-only evaluation, the
primary evaluator gate, checkpoint ownership, and acceptance states. Path/SHA
syntax and external source-binding values are deliberately left to their
dedicated binding layer; semantic role fields inside those binding objects are
still covered. Six ordered-list rules independently bind vendor include roles,
the R18 transform sequence, train-only stopped transforms, selection
tie-breakers, checkpoint state ownership, and acceptance-state order and set.

Twelve stable cross-field rules reject synchronized drift as well as one-sided
drift: random initialization cannot acquire a pretrained or checkpoint source;
one formal run cannot resume, retry, or overwrite; required EMA evaluation and
selection use EMA weights; AMP remains enabled with its four exact executable
parameters and zero non-finite/overflow/skip allowances; AdamW parameter-group identity is
audited; epoch scheduler semantics stay epoch-based; development-only
selection keeps confirmatory and test access sealed; the uncertified primary
evaluator keeps launch blocked; R18 augmentation remains ordered and
train-only; checkpoint evidence remains exactly-once and loadable; and
acceptance evidence remains bound to finite, complete states. Validation order
is JSON/type, closed schema, numeric, path/SHA, semantic literal/enum/list and
cross-field rules, frozen digest, then external binding. Thus changing
`/ema/development_evaluation_weights` from `ema` to `raw` fails at the semantic
layer rather than being accepted until the whole-contract digest.

The four GradScaler fields are numeric contract leaves: the three scale
parameters are exact builtin floats and growth interval is an exact builtin
integer. Numeric validation rejects non-finite, out-of-range, and in-range but
non-frozen alternatives before the canonical digest. This parameter decision
does not certify the primary evaluator and does not make training-launch ready.
