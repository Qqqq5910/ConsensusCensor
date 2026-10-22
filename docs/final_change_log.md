# Final closeout change log

Status: research closed by the research lead; no Phase 3, new estimator or large rerun.

## Preserved research evidence

All Phase 1/2 protocols, raw result CSV/JSON/NPZ products, archived summaries and historical reports are retained byte-for-byte. The pre-closeout inventory is `experiments/finalization/before_sha256.json`; final preservation checks distinguish intended packaging/documentation edits from frozen evidence. Existing estimator, selection and simulation logic is unchanged. All 54 original tests are retained.

## Explicit correctness and reporting changes

1. The historical summaries compute means over `status=ok`. They already disclose failures, but those means are conditional and cannot be interpreted as unconditional algorithmic risks. New `final_audit.py` outputs failure denominators, finite nonconverged raw-value sensitivity, identical-success-pair comparisons and a failed-run loss-assignment envelope. It does not fill in missing B5b estimates, claim a confidence interval or overwrite the old summaries.
2. Final reports consistently use half-L1 terminology for raw corrected vectors. Standard probability TV is only claimed for normalized probability vectors. Overall/group vector validity is distinguished. No estimator is clipped, normalized or changed.
3. The two randomization mechanisms stay separate: 50 audits of each fixed synthetic population versus 20 regenerated populations with common audit/selection streams. Real-data repetitions remain a third, separately identified setting. Only counts are totaled across settings; no pooled uncertainty interval is constructed.
4. Three finalization regression tests protect failure accounting, invalid-vector handling and identical-success pairing. B4 expectation enumeration, B5 audit isolation and M2 group weighting remain covered by the preserved tests and small independent design enumeration.

## Packaging and engineering changes

- Fixed `.gitignore` from `data/` to `/data/` so source modules under `src/consensuscensor/data/` are included while actual raw/processed data remain excluded. Reformatted those previously ignored source files; no logic was changed.
- Updated package description/readme/license metadata; the build minimum is setuptools 77 to support the SPDX license expression. Moved PyArrow to the `data` extra because it is used for Parquet preprocessing, not the core numerical estimators; it remains required alongside `dev` to run all preserved tests. NumPy/SciPy/pandas remain runtime dependencies. No historically used dependency was falsely described as unused.
- Added MIT LICENSE, CFF metadata with a collective contributor name, and DATA_LICENSE.md distinguishing software licensing from MultiPref ODC-BY obligations. No remote URL, personal author identity or DOI is invented.
- Added English-first README and final English report, Chinese summary, and reproducibility guide. Numerical report tables are generated from archived outputs and the separate final audit.
- Added an offline one-command reproduction: preserved/new tests, exhaustive design checks and 162 small existing-estimator runs. This is an engineering smoke check and is not new research evidence. Original large experiments were not rerun.
- Added read-only GitHub Actions CI for Python 3.11/3.12 and a wheel-content check. GitHub-hosted CI is configured but has not executed; local checks are recorded separately. No publishing/deployment workflow is added.

- Added a path-specific whitespace attribute for the historical Phase 1 CSV: one original solver message ends with a space. This keeps that raw field byte-for-byte rather than silently editing research data to satisfy a whitespace check.

## Commit boundary

The finalization ends with one conventional local commit in the existing repository. This repository previously had no commits; the commit therefore includes both preserved phases and the closeout additions. No remote repository, push, tag or GitHub Release is created. The final response supplies the actual commit hash; it is not embedded in its own tracked content.
