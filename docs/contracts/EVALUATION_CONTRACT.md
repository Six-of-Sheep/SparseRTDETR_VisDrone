# Evaluation Contract

- Primary metrics are COCO-style AP, AP50, and AP_small.
- Whether to add VisDrone official metrics waits for upstream audit.
- Baseline and candidate use fixed data, image size, thresholds, maxDets,
  and evaluation mathematics within a comparison.
- Report newly covered objects, duplicate predictions, and suppressed
  baseline true positives.
- Model selection and final confirmation are separate decisions.
- Thresholds and acceptance criteria are preregistered before selection.
- MODEL_SELECTION_READY=false.
- ACCURACY_ACCEPTANCE_READY=false.
