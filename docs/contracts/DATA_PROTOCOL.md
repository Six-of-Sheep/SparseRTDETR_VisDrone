# Data Protocol

The frozen implementation is documented in
`docs/contracts/VISDRONE_PROTOCOL_V1.md` and represented by
`configs/visdrone_protocol_v1.json`. Source code accepts only in-memory
bytes and relative train/val paths; it does not enumerate a dataset root.

The protocol identity is bound to the audited official train and val
inventories (6471 and 548 images respectively). The inventory algorithm is
the sorted list of `{relative_path, size_bytes, file_sha256}` records encoded
as canonical UTF-8 JSON with `ensure_ascii=true`, `sort_keys=true`, compact
separators, and no trailing newline. No production JSON or split membership
file is generated in this phase.

Historical P2 work observed the official test split and used a 5% top-k
post-hoc exploratory analysis. That evidence is not an independent unseen P3
confirmatory result. The historical official val set was also used for
selection and diagnostics. `TEST_ACCESS_ALLOWED=false` remains in force.
