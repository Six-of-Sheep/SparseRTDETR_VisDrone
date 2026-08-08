# Environment Policy

- P3 establishes a new environment and does not reuse the legacy visdrone
  environment as the formal RT-DETR environment.
- The upstream implementation is fixed to the official `rtdetrv2_pytorch`
  snapshot recorded in `manifests/rtdetrv2_upstream.json`.
- Upstream requirements and Dockerfile are environment-design inputs only;
  they do not establish the P3 lock.
- The eventual environment provides a portable core dependency file and a
  Linux-precise lock.
- pytest is a formal P3 development dependency, not an implicit borrowed
  helper from another environment.
- Local wheel paths, conda build paths, and authenticated URLs are forbidden
  in committed environment files.
- The legacy VisDrone environment must not be reused as the formal P3
  environment.
- P3_ENVIRONMENT_READY=false.
