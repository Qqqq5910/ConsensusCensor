# Reproducibility guide

This is a completed research artifact. No Phase 3 or new algorithm study is scheduled.

## Install

Use Python 3.11+ in a fresh environment. Local final validation used Python 3.12; the CI configuration specifies 3.11 and 3.12, but no GitHub-hosted CI run has occurred yet.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,data]"
```

On Windows use `.venv\Scripts\activate`. Core runtime dependencies are NumPy, SciPy and pandas. The `data` extra supplies PyArrow for Parquet processing and the preserved preprocessing tests; `dev` supplies pytest and Ruff. No unused GPU, neural-network, web or API dependencies are included. `requirements-lock.txt` and `experiments/phase2/environment.txt` retain the historical numerical versions; the finalization environment is recorded separately. Exact timings and optimizer convergence can vary with SciPy, BLAS and platform versions.

## One-command offline reproduction

From the repository root, after installation:

```bash
python scripts/reproduce_minimal.py
```

The command performs all automated correctness tests, independent exhaustive design checks, and 162 small method runs on N=240 synthetic populations. It compares existing B1, B3, B4, B4-M2, B5a and B5b under M0/M1/M2. Only the existing statistical estimators are fitted; no new prediction model or LLM is trained. It uses no real-data download, paid API, database or GPU. These small outputs are engineering checks, not new research evidence.

Outputs go to `experiments/minimal/`: tests.txt, design_enumeration.csv, results.csv, summary.csv, protocol.json and summary.md. The command refuses a nonempty output directory. For a deliberate repeat, keep previous output and choose a different empty path:

```bash
python scripts/reproduce_minimal.py --output /tmp/consensuscensor-repeat
```

On Windows choose an equivalent new local directory. Estimated times are not used as a correctness gate.

## Inspect the archived research without re-estimation

```bash
python scripts/final_audit.py --output /tmp/consensuscensor-audit
```

This only reads the archived 28,800 Phase 2 rows and sampling manifests, reruns small exhaustive design checks, and writes separate failure-aware summaries. It checks audit identity and allocated cost, fixed-population versus regenerated-population repetition units, original failure counts, and vector validity. It does not rerun the large experiment or overwrite original results.

`failure_sensitivity.csv` separates success-conditional loss, finite nonconverged raw-value sensitivity, and hypothetical failed-run losses in [0,1] when that bound is applicable. This envelope is not a confidence interval or imputation. `paired_complete_cases.csv` evaluates both methods on exactly the same jointly successful randomizations; success-conditioned selection can still matter. Do not mix fixed-audit and regenerated-population variability.

Final report tables can be regenerated from archived results and the committed final audit:

```bash
python scripts/write_final_reports.py
```

This rewrites only the new final English/Chinese reports. It does not rewrite historical Phase 1/2 reports or CSVs.

## Optional full historic replication

Large replication is not part of the default command and was not performed during finalization. Use a separate working copy, retain the archived outputs, and ensure the output directories are empty before calling the legacy scripts. `run_phase1.py` can overwrite Phase 1 outputs; `run_phase2.py` refuses an existing results.csv. Do not run either on the authoritative archived output directories.

The commands are:

```bash
python scripts/prepare_data.py
python scripts/run_phase1.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_phase2.py
```

The last environment-variable syntax is for POSIX shells. Frozen protocols, seeds, selection version and numerical environments are preserved under `experiments/phase1` and `experiments/phase2`. Phase 2 requires the original Phase 1 split manifest and matching data SHA256. The bounded fetch reads only 2,000 comparisons, but upstream ordering/content are not revision-pinned: a different hash must be reported as a different input. Exact real-data replication is therefore conditional on access to the matching original public-data bytes, which are not redistributed here. The small offline reproduction has no such external-data requirement.

## Static checks, packaging and CI

```bash
python -m ruff check .
python -m ruff format --check .
python -m pip wheel --no-deps . -w dist
python scripts/check_wheel.py dist
```

Use an empty dist directory or keep exactly one ConsensusCensor wheel there. The checker verifies required source data modules are present and real JSON/Parquet data are absent. CI has read-only repository permissions, tests, static checks, offline reproduction, archived-result audit and wheel inspection. It contains no deployment or publishing step. The workflow is configured, not claimed to have passed on GitHub before a remote exists.

## Interpreting results

- The target is a finite complete annotation reference, not world preference truth.
- Group contributions divide by two; overall contributions average the two groups.
- B4 has a design expectation guarantee, not universal variance dominance.
- B5 uses conditional fold probabilities under fixed-size SRS; numerical-success subsets do not inherit an unconditional unbiasedness claim.
- Raw non-probability vectors use half-L1 loss; standard TV terminology is reserved for normalized probability vectors. No clipping/projection is hidden.
- Every numerical failure and unavailable method remains explicit. No accuracy claim is based only on the most favorable seed.
