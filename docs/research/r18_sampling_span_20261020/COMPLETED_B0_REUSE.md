# Completed B0 output reuse after failed supervision

This is an engineering continuation, with unchanged science and execution
configuration. It does not retroactively close the original stage.

The original `attempt-formal-001` B0 worker completed e120, but its supervisor
exited with SIGKILL before writing CLOSED. The 15,509,099,568-byte event journal
was previously read, decoded and retained in full by terminal verification.
That is a confirmed unbounded-memory defect, not proof of the actual kill cause.
The replacement authenticates every record and chain SHA with a 4 MiB per-line
limit, retains only the terminal record, and incrementally checks the original
training-input digest. Oversized records fail closed. No validation is sampled.

`r18_sampling_formal.py reconcile-B0 --completed-B0-contract ORIGINAL_CONTRACT
--source NEW_MANIFEST --output NEW_WITNESS` authenticates original source,
contract, failed supervision, full events, worker exit, fixed endpoint, full
checkpoint SHA and CPU state clocks/finiteness, shared initialization and settled
costs. The witness explicitly says `AUTHENTICATED_OUTPUT_NOT_ORIGINAL_STAGE_CLOSURE`.
The old contract, source, state, STOP and missing CLOSED remain unchanged.
File identities and full SHA are retained; each new process rejects subsequent
fingerprint changes. The endpoint loader also rehashes the tensor on actual use.

The tracked `lab` action accepts the additional `--completed-B0-contract` option
and constructs this witness before normal prepare/bind. No untracked field
assembly is required. Only the single reviewed old formal attempt is allowed.
New approval binds the witness SHA and remaining budget. The original acceptance
cost is verified separately from the complete inherited ledger.

New contracts cannot execute B0 worker/supervisor stages. The new root records
ten actual closed stages and a separate reused-B0 reference; export contains
`train-B0-REUSE.json`, never a fabricated `train-B0-CLOSED.json`.
B1 and P retain independent fresh ImageNet initialization, optimizer, EMA and RNG.
No acceptance state or completed B0 checkpoint initializes either training arm.

Inherited actual totals: 72,978 forward calls and 59,334.44017676904 GPU-stage
seconds, including the original acceptance. Remaining: 146,715 calls
(145,920 B1/P training + 411 endpoint inference + 384 cost). Original total
219,693 calls and cumulative 60 GPU hours remain enforced, with no refunds.

Model, loss, augmentation, update, save/restore and scientific calculations are
unchanged. Worker changes are restricted by AST comparison to two mechanical
endpoint-contract/export-provenance substitutions. GPU acceptance is not rerun.
All new engineering, identity, numerical or resource failures STOP without retry.
CPU fixtures remain synthetic, with no CUDA/BF16 or production quality claims.
