# AutoDL Migration Sequence

1. Clone a fixed Git commit.
2. Rebuild the locked environment.
3. Configure VISDRONE_ROOT, ARTIFACT_ROOT, and PRETRAINED_ROOT.
4. Pull and verify the data manifest.
5. Pull and verify pretrained weights.
6. Run the CPU contract check.
7. Run one batch of GPU smoke.
8. Check memory, finite values, output identity, and artifact binding.
9. Stop for independent audit.
10. Allow formal training only after the audit passes.

Training hardware and formal speed-measurement hardware are recorded
separately.
