# Tree SageMaker experiments

Spec: ../../specs/014-tree-sagemaker-experiments/design.md

Run tools from the repository root with `python -m experiments.tree_sagemaker.<tool>`.
All tools require explicit paths; generated configs, reports and weights belong under ignored `runs/`.

- `generate`: derive 47 schema-validated configs from a frozen baseline (E04 conditional).
- `audit`: read train/validation only, count history availability and unordered-set truncation.
- `execute`: prepare immutable submissions, verify baseline before batches, cap parallel jobs at four,
  stop new submission on failure, never retry failed training automatically. Runtime must declare train only.
- `collect`: download and verify completed job artifacts; export validation metric details as CSV.
- `diagnose`: use training-frequency buckets and validation metrics; no identifier values in output.
  Also report episode-hit and additional top-200-interest slices.

Do not run test to select parameters. Missing metrics remain null with their original diagnostics.
A partially extracted collection directory must be inspected/removed before retrying; checksum verification
will reject incomplete artifacts. Resuming execute reuses prepared job identity and never overwrites a job.
