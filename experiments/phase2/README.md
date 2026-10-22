# Phase 2 reproducible outputs

Run from repository root:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/verify_phase2_design.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_phase2.py
python scripts/report_phase2.py
```

`run_phase2.py` refuses to overwrite results.csv. Archive the entire current Phase 2 output directory first before a deliberate rerun. Original raw MultiPref bytes and split manifest must match Phase 1; no substitute download or reconstructed Phase 1 is attempted.

- frozen_protocol.json + protocol.sha256: locked before the experiment, including actual seeds.
- results.csv: every raw estimate, reference, error, acquisition cost, CPU time, status, audit hash and diagnostics. No projection.
- summary.csv: bias per category, MSE, variance, finite-population error variance and all requested group metrics. Only status=ok in main metrics; failures remain in results.csv.
- paired_comparisons.csv: complete-pair differences on exactly the same audit indices, with paired counts and win fractions.
- audit_manifest.json: sampled comparison indices, actual k/N and fold counts; identical across methods and mechanisms.
- *_fold_manifest.csv: original real partition prompt folds.
- *_fixed_reference.npz: synthetic labels/configured truth only; not real raw annotations.
- design_enumeration.csv/json + design_check.txt: independent finite-population expectation checks.
- phase1_preservation.json + preservation_check.json: original file hashes and post-run comparison.
- pytest.txt, run.log, ruff.txt: execution evidence.

Fixed-population S1/S2/S3 each have 50 audit randomizations; regenerated S1/S2/S3 each have 20 independent population seeds. Each real train/validation/test target has 30 audit randomizations. Do not pool those uncertainty mechanisms. `extra_labels` is the common allocated acquisition budget; B0 itself does not require audits. Real raw/processed data stay under ignored data/ and are not included in the deliverable archive.
