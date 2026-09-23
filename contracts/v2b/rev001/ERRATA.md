# REV1 contract errata

These notes correct descriptive labels only. The sealed JSON documents in this
directory are not edited, because their self-digests (e.g.
`evaluation_contract_sha256 = 8339515233e2eeaa…`) are cited by existing
evaluation results.

## E1 (2026-09-23): `protocol.resize` in `evaluation_contract.json`

The contract labels the input transform `"per_checkpoint_square_letterbox"`.
The implemented transform is a direct, aspect-ratio-changing resize to a
square, with no padding:

- evaluation: `training_v2b_development.py` policy transform
  `{"type": "Resize", "size": [S, S]}` followed by `ConvertPILImage`;
- training: `training_v2b.py` rewrites the vendor `Resize` op to
  `[input_size, input_size]`.

For the dominant 1360×765 development images at S=640 this scales x by 0.47
and y by 0.84. Every metric produced under this contract (including
`comparison.json` of the seed2/r896 epoch-60 evaluation, which copies the
label) was computed with the square stretch. Reports and the paper must
describe the protocol as "resized directly to S×S (aspect ratio not
preserved)". New contracts should use the label `square_stretch_resize`.
