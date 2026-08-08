# Environment Policy

- P3 establishes a new environment and does not reuse the legacy visdrone
  environment as the formal RT-DETR environment.
- PyTorch and CUDA versions remain unfrozen until an upstream implementation
  is selected and audited.
- The eventual environment provides a portable core dependency file and a
  Linux-precise lock.
- pytest is a formal P3 development dependency, not an implicit borrowed
  helper from another environment.
- Local wheel paths, conda build paths, and authenticated URLs are forbidden
  in committed environment files.
- P3_ENVIRONMENT_READY=false.
