# Artifact Policy

Git manages source, configuration, tests, documentation, small manifests,
data identity and SHA records, and experiment contracts.

Git does not manage datasets, checkpoints, prediction dumps, runs, caches,
logs, external process evidence, secrets, .env files, or large binaries.

Future canonical data and key checkpoints may use DVC with an S3/OSS-compatible
store. Existing legacy runs may use rsync or rclone with SHA manifests. DVC
is not installed or configured in this repository, and no large artifact is
uploaded by initialization.
