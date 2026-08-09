# Test-Split Metadata Access Incident

This record documents a metadata-only enumeration incident observed on
2026-08-09 during an interrupted converter-repair audit. Directory names and
image filenames from the disabled test split were enumerated. Content access
was not established from the retained task evidence, and prediction or metric
access was not established.

The project-level historical status remains
`P3_DATASET_TEST_SPLIT_ACCESSED=true` because P2 had previously observed test.
The process that records this incident did not access test data:
`dataset_test_accessed_by_this_process=false`. No model or evaluator was
executed and no derived test artifact was generated.

The official-train confirmatory sequence-group protocol was not compromised.
Test remains prohibited for model selection, tuning, threshold choice, or
independent confirmation.
