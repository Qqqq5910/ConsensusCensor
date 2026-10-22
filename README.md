# ConsensusCensor

**An Open Research Artifact on Consensus Filtering and Robust Statistical Estimation from Human Preference Data.**

**Engineering Artifact: READY**

**Original Research Contribution: NOT ESTABLISHED**

Research is complete. The research lead has decided against further original-algorithm work and there is no Phase 3. This CPU-only artifact implements and compares established statistical estimators under controlled annotation censoring. It is not a new PPI/AIPW/GREG method or a claim about downstream LLM quality.

[Final research report](docs/final_research_report.md) · [Reproducibility guide](docs/reproducibility_guide.md) · [中文说明](README_CN.md)

## Quick start

Python 3.11+; run from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,data]"
python scripts/reproduce_minimal.py
```

Windows activation: `.venv\Scripts\activate`. The one-command reproduction runs correctness tests, independent finite-design enumeration and 162 small synthetic comparisons of existing estimators. No real-data download, GPU, paid API or server is needed. It refuses nonempty outputs; use `--output` with a new directory for repeats. See the guide for full archived experiments rather than rerunning 28,800 method runs by default.

## What the artifact establishes

- Consensus filtering can change the observed preference distribution.
- A correctly specified selection-aware likelihood can improve precision; model misspecification can produce persistent bias.
- Design-based residual correction offers a different robustness guarantee and can increase variance.
- Ordinary M2 post-stratification is already a strong baseline in several conditions.
- Model assistance does not uniformly beat either likelihood or simple post-stratification.

Targets are the empirical A/Tie/B distribution in four original annotation slots per comparison, with two normal and two expert slots and equal group weights. Tie is preserved; raw five-level strengths remain in local preprocessing. M0 randomly removes one slot, M1 chooses the closest three-label subset, and M2 hides whole comparisons lacking three matching labels. These are research simulations, not exact replicas of a production dataset's filtering rule.

B0–B3 are naive, audit-only, stratified-audit and exact 81-state selection-aware likelihood baselines. B4 is a Horvitz–Thompson residual estimate; B4-M2 is post-stratification; B5a/B5b are cross-fitted model-assisted residual corrections. All use the same allowed observations and allocated audit budget, according to their baseline definitions. See the mathematical documents for positivity and cross-fitting conditions.

## Results and limitations

All original Phase 1/2 results and frozen protocols remain under `experiments/`. Original reports are preserved as historical outputs; the final report qualifies success-only statistics explicitly. Three B3 and twelve B5b numerical-failure runs remain in the archived data. M2 test B5b at 20% audit has 29/30 successful runs. Failed-run sensitivity and identical-success-pair comparisons are recorded separately in `experiments/finalization/audit/`.

Half-L1 loss is used for raw corrected vectors; it is standard TV only for legal probability distributions. No silent clipping or projection is performed. Within-fixed-population audit variation and regenerated-population variation are kept separate. No global likelihood-identification proof, population confidence interval, downstream-model improvement or original statistical theory is established.

## Data and licensing

The real pilot used the first 2,000 comparisons from [MultiPref](https://huggingface.co/datasets/allenai/multipref), default configuration, upstream train split. It is an ordered prefix, not a random sample of the full dataset. Cite Miranda et al., *Hybrid Preferences*, ACL 2025 / arXiv:2410.19133; see [CITATION.bib](CITATION.bib).

Software and project-authored documentation: [MIT](LICENSE). MultiPref: ODC-BY; see [DATA_LICENSE.md](DATA_LICENSE.md). Dataset-derived outputs retain their attribution and applicable data-license obligations. Original real annotations, full text, secrets and caches are excluded from Git and the distribution. An exact historical real-data rerun requires matching input SHA256; the download is not upstream-revision-pinned. Offline synthetic reproduction is self-contained.

[CITATION.cff](CITATION.cff) provides software citation metadata using a collective contributor name; no DOI or remote repository is invented.

## Validation and publication state

The repository includes a read-only GitHub Actions workflow for Python 3.11/3.12, local tests/static checks, offline reproduction, archived-result auditing and wheel inspection. GitHub-hosted CI has not run because this closeout does not create a remote, push, publish a release or deploy anything. Local evidence and the change log are in `experiments/finalization/` and [docs/final_change_log.md](docs/final_change_log.md).
