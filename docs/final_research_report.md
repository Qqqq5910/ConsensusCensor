# ConsensusCensor: An Open Research Artifact on Consensus Filtering and Robust Statistical Estimation from Human Preference Data

**Project status: completed; no Phase 3.**

**Engineering Artifact: READY**

**Original Research Contribution: NOT ESTABLISHED**

## 1. Abstract

We examine how controlled consensus filtering changes the observed distribution of four human-preference annotations per comparison, and how existing estimators use a limited random complete audit to recover a finite annotation distribution. An independent categorical likelihood model can be efficient under its working assumptions and persistently biased under shared latent factors or heterogeneous comparison difficulty. Horvitz–Thompson residual correction offers a design-based expectation guarantee but can increase variance or yield unnormalized raw vectors. Known-count post-stratification is a strong simple baseline for whole-comparison censoring. Cross-fitted model assistance has conditional benefits and does not uniformly improve upon either likelihood or post-stratification. The artifact implements and compares established methods; it establishes neither a new statistical estimation theory nor an original algorithm.

## 2. Research Problem

Each comparison contains two normal-worker slots and two expert-worker slots. Labels are A, Tie and B; Tie is never assigned to a winner. The target is the empirical distribution of the original four slots within a fixed public-data collection, with equal group weights. Separate group distributions use two slots per comparison. Support scores are (1, 0.5, 0), and the group contrast is expert minus normal support. These targets are not global human preferences, unique answer quality or a causal effect on downstream alignment.

## 3. Related Work

Filtering-induced label-distribution changes and substantive annotator disagreement are prior findings: Fleisig et al., *Balancing Quality and Variation* ([ACL 2025](https://aclanthology.org/2025.nlperspectives-1.5/)); Zhang et al., *Diverging Preferences* ([ICML 2025](https://proceedings.mlr.press/v267/zhang25bx.html)); and Ghafouri et al., *RLHF May Not Reflect Genuine Preferences* ([ICML 2026](https://proceedings.mlr.press/v306/ghafouri26a.html)). MultiPref comes from Miranda et al., *Hybrid Preferences* ([ACL 2025](https://aclanthology.org/2025.acl-long.355/), arXiv:2410.19133). The NVIDIA HelpSteer3 dataset motivates the context, but our M1/M2 rules do not claim to reproduce its actual filtering procedure.

Horvitz and Thompson (1952), DOI 10.1080/01621459.1952.10483446, provide the sampling-estimation background. Post-stratification, generalized difference/GREG and augmented inverse-probability correction are established ideas. [Prediction-Powered Inference](https://arxiv.org/abs/2301.09633), [Cross-Prediction-Powered Inference](https://arxiv.org/abs/2309.16598) and [ppi_py](https://github.com/aangelopoulos/ppi_py) provide related prediction-plus-residual and cross-fitting ideas. We do not claim to invent PPI, AIPW, GREG or unbiased estimation, or transfer their assumptions and coverage guarantees without proof. Source checks and their limits are recorded in phase2_existing_methods.md.

## 4. Statistical Formulation

For group $g$, define $z_{igc}=\frac12\sum_{j\in g}1(Y_{ij}=c)$, $o_{igc}=\frac12\sum_{j\in g}M_{ij}1(Y_{ij}=c)$, and $r_{igc}=z_{igc}-o_{igc}$. The target is $\theta_{gc}=N^{-1}\sum_i z_{igc}$; overall targets average the two groups. B4 is $N^{-1}\sum_i[o_i+A_i r_i/\pi_i]$. For a fixed finite population and observed view, known positive inclusion probabilities give $E(A_i/\pi_i)=1$, hence the design expectation equals the target. Label independence is unnecessary. Under fixed-size SRS, actual inclusion probability is k/N with k=floor(rate*N). The variance depends on residual variation, not the variation of complete labels; it need not dominate audit-only estimation.

B5 is $N^{-1}\sum_i[o_i+\hat m_i+A_i(r_i-\hat m_i)/\pi_i]$. With fixed prompt folds under global fixed-size SRS, the implementation uses conditional fold probabilities k_f/N_f. Conditional on the fold counts, within-fold SRS selections are independent across folds; predictors trained outside a target fold can then be used in the generalized-difference argument. An unknown fold with zero audits is not estimable. This conditional argument requires a defined predictor on the relevant design support; it is not an unconditional unbiasedness guarantee after conditioning on numerical success. Detailed derivations, positivity requirements and exceptions appear in phase2_mathematics.md.

## 5. Data and Selection Mechanisms

The real pilot uses the preserved first 2,000 default/train MultiPref comparisons: 8,000 valid annotation slots and 1,010 distinct prompts, with no unmapped labels in this prefix. The five raw strength levels remain in preprocessing; estimation uses only A/Tie/B. Prompt-group partitions contain 1,201 training, 399 validation and 400 test comparisons. This ordered prefix is not a probability sample of the whole dataset. MultiPref is labeled ODC-BY; see DATA_LICENSE.md and CITATION.bib.

M0 uniformly deletes one of four slots. M1 keeps the three-slot subset minimizing the sum of pairwise distances on A=0, Tie=1, B=2, with uniform random tie-breaking. M2 keeps all four slots only if at least three labels have the same class, otherwise hiding all labels while preserving metadata and the selection indicator. These rules were unchanged between phases. All are controlled censoring simulations of originally complete public data.

## 6. Estimators B0–B5

| Estimator | Permitted information and assumptions |
| --- | --- |
| B0 | Pooled visible annotations; no audit correction. Visible group counts can differ. |
| B1 | Complete randomly audited comparisons only. |
| B2 | Group-stratified audit with equal target weights; identical to B1 under these balanced complete audits. |
| B3 | Group-specific independent categorical model; exact 81-state observed likelihood with the known selection kernel. |
| B4 | All visible contributions plus inverse-probability-weighted audited missing contributions. |
| B4-M2 | Known kept/deleted layer weights; exact visible retained contribution plus the audited deleted-layer mean. |
| B5a | Cross-fitted missing-group frequencies, fixed 0.5 smoothing, plus residual correction. |
| B5b | Cross-fitted B3 posterior expectations of hidden slots, plus residual correction. |

For M2 post-stratification, the missing layer has weight D/N, and group contributions still use denominator two. If D>0 but no deleted comparison is audited, the method returns not estimable. B5b conditions on the actual retained subset and tie-breaking kernel for M1 and the rejected event for M2; it never replaces hidden labels with the unconditional B3 point estimate.

## 7. Experimental Design

Phase 1 used five fixed audit/selection seeds. Phase 2 fixed each synthetic population and its observed censoring view, then used 50 independent audit randomizations per scenario. A separate experiment used 20 independent population generation seeds per scenario, with common audit-index and selection random streams. The latter measures population-generation variation under those fixed assignments, not 20 additional within-population audit replications. Real train/validation/test targets each used 30 audit randomizations. Audit rates were 0%, 5%, 10% and 20%. Each synthetic population had N=800 in Phase 2.

S1 satisfies the independent group model. S2 uses an unobserved shared factor to correlate annotations. S3 mixes high-consensus and high-disagreement difficulty classes with explicitly configured, group-specific probabilities. The latent classes are not given to selection or prediction. All methods in a cell share the same observed view, audit indices and allocated budget. Frozen protocols are preserved. There was no test-Oracle model or hyperparameter selection.

## 8. Results

All following numerical tables are generated from saved Phase 2 artifacts, not new large experiments. Entries summarize the finite complete-label target. **Half-L1 loss is named consistently below:** it is standard TV only when the relevant raw vector is a normalized probability distribution. No clipping or projection was applied.

### 8.1 Fixed synthetic populations, M2, 20% audit

| dataset | B0 | B1 | B2 | B3 | B4 | B4-M2 | B5a | B5b |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | 0.14027 | 0.01792 | 0.01792 | 0.00394 | 0.01507 | 0.00972 | 0.00967 | 0.00966 |
| S2 | 0.13055 | 0.02369 | 0.02369 | 0.04001 | 0.01867 | 0.00831 | 0.00836 | 0.00995 |
| S3 | 0.14190 | 0.02358 | 0.02358 | 0.02832 | 0.01652 | 0.00868 | 0.00868 | 0.00876 |

Under correct specification, B3 is more efficient in this comparison. Under S2/S3 misspecification, its low audit-randomization variability can coexist with substantial persistent error. B4 trades that model dependence for audit variance. Simple M2 post-stratification is already competitive with the more expensive model-assisted variants; this is a result about existing baselines, not evidence of a new method contribution.

### 8.2 Preserved MultiPref test target, 20% audit

| mechanism | method | successful_runs | overall_half_L1 | normal_half_L1 | expert_half_L1 | group_gap_error_mean |
| --- | --- | --- | --- | --- | --- | --- |
| M0 | B0 | 30 | 0.01042 | 0.01390 | 0.00908 | 0.00699 |
| M0 | B1 | 30 | 0.03619 | 0.04629 | 0.03658 | 0.02567 |
| M0 | B2 | 30 | 0.03619 | 0.04629 | 0.03658 | 0.02567 |
| M0 | B3 | 30 | 0.00861 | 0.01010 | 0.00874 | 0.00542 |
| M0 | B4 | 30 | 0.01290 | 0.02352 | 0.02215 | 0.02604 |
| M0 | B4-M2 | 0 | not available | not available | not available | not available |
| M0 | B5a | 30 | 0.01253 | 0.02232 | 0.02034 | 0.01939 |
| M0 | B5b | 30 | 0.01283 | 0.02237 | 0.02064 | 0.01826 |
| M1 | B0 | 30 | 0.04208 | 0.04194 | 0.04233 | 0.00132 |
| M1 | B1 | 30 | 0.03619 | 0.04629 | 0.03658 | 0.02567 |
| M1 | B2 | 30 | 0.03619 | 0.04629 | 0.03658 | 0.02567 |
| M1 | B3 | 30 | 0.01049 | 0.01099 | 0.01078 | 0.00406 |
| M1 | B4 | 30 | 0.01315 | 0.01875 | 0.02100 | 0.02200 |
| M1 | B4-M2 | 0 | not available | not available | not available | not available |
| M1 | B5a | 30 | 0.01339 | 0.01787 | 0.02018 | 0.01977 |
| M1 | B5b | 30 | 0.01210 | 0.01566 | 0.01954 | 0.01693 |
| M2 | B0 | 30 | 0.07522 | 0.07462 | 0.07582 | 0.00125 |
| M2 | B1 | 30 | 0.03619 | 0.04629 | 0.03658 | 0.02567 |
| M2 | B2 | 30 | 0.03619 | 0.04629 | 0.03658 | 0.02567 |
| M2 | B3 | 30 | 0.00946 | 0.01110 | 0.01145 | 0.00711 |
| M2 | B4 | 30 | 0.02075 | 0.03046 | 0.02867 | 0.02146 |
| M2 | B4-M2 | 30 | 0.01427 | 0.02503 | 0.02256 | 0.02140 |
| M2 | B5a | 30 | 0.01506 | 0.02509 | 0.02280 | 0.01993 |
| M2 | B5b | 29 | 0.01488 | 0.02578 | 0.02315 | 0.02182 |

These are success-conditional means, not unconditional algorithmic risks. In particular, M2 B5b has 29 successful runs out of 30. The real pilot does not show design-based/model-assisted methods uniformly outperforming B3, including for the normal-versus-expert support contrast. Synthetic misspecification benefits cannot be assumed to transfer to the real-data target.

### 8.3 Same-randomization paired comparisons, real test, 20% audit

| mechanism | comparison | paired_n | lhs_same_pairs_mean | rhs_same_pairs_mean | paired_mean_difference |
| --- | --- | --- | --- | --- | --- |
| M0 | B4 minus B1 | 30 | 0.01290 | 0.03619 | -0.02329 |
| M1 | B4 minus B1 | 30 | 0.01315 | 0.03619 | -0.02304 |
| M2 | B4 minus B1 | 30 | 0.02075 | 0.03619 | -0.01544 |
| M2 | B4-M2 minus B4 | 30 | 0.01427 | 0.02075 | -0.00648 |
| M0 | B5a minus B4 | 30 | 0.01253 | 0.01290 | -0.00037 |
| M1 | B5a minus B4 | 30 | 0.01339 | 0.01315 | 0.00024 |
| M2 | B5a minus B4 | 30 | 0.01506 | 0.02075 | -0.00569 |
| M0 | B5b minus B4 | 30 | 0.01283 | 0.01290 | -0.00007 |
| M1 | B5b minus B4 | 30 | 0.01210 | 0.01315 | -0.00104 |
| M2 | B5b minus B4 | 29 | 0.01488 | 0.02004 | -0.00516 |
| M0 | B5b minus B3 | 30 | 0.01283 | 0.00861 | 0.00422 |
| M1 | B5b minus B3 | 30 | 0.01210 | 0.01049 | 0.00161 |
| M2 | B5b minus B3 | 29 | 0.01488 | 0.00950 | 0.00538 |
| M2 | B5b minus B4-M2 | 29 | 0.01488 | 0.01459 | 0.00029 |

Both means in each row use the identical jointly successful randomizations; negative differences favor the first method. Complete-case comparisons can still be affected by label-dependent numerical success. They are accompanied by the failure accounting below, rather than presented as unconditional improvements.

## 9. Failure Analysis

The final audit checked 28,800 archived rows and 1,200 sampling records. It confirmed three B3 and 12 B5b numerical-failure runs. The three B3 runs retain finite raw estimates; the B5b failures contain no returned estimate. Neither is silently reclassified as success. Zero-audit unavailability and method/mechanism non-applicability are separately recorded.

| experiment | dataset | split | mechanism | audit_rate | method | success_n | numeric_failure_n | success_conditional_half_l1 | all_finite_raw_half_l1 | failure_loss_0_lower | failure_loss_1_upper |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_audit | S2 | full | M0 | 0.05000 | B5b | 49 | 1 | 0.02092 | 0.02092 | 0.02050 | 0.04050 |
| fixed_audit | S3 | full | M0 | 0.10000 | B5b | 49 | 1 | 0.01257 | 0.01257 | 0.01232 | 0.03232 |
| fixed_audit | S3 | full | M2 | 0.10000 | B3 | 49 | 1 | 0.03261 | 0.03262 | 0.03196 | 0.05196 |
| fixed_audit | S3 | full | M1 | 0.20000 | B5b | 49 | 1 | 0.01045 | 0.01045 | 0.01024 | 0.03024 |
| regenerated | S3 | full | M1 | 0.05000 | B5b | 19 | 1 | 0.02112 | 0.02112 | 0.02007 | 0.07007 |
| real_audit | MultiPref | train | M0 | 0.05000 | B3 | 29 | 1 | 0.00434 | 0.00430 | 0.00419 | 0.03753 |
| real_audit | MultiPref | train | M0 | 0.10000 | B5b | 29 | 1 | 0.01327 | 0.01327 | 0.01283 | 0.04616 |
| real_audit | MultiPref | train | M1 | 0.10000 | B5b | 28 | 2 | 0.01313 | 0.01313 | 0.01226 | 0.07892 |
| real_audit | MultiPref | train | M0 | 0.20000 | B5b | 29 | 1 | 0.00825 | 0.00825 | 0.00798 | 0.04131 |
| real_audit | MultiPref | train | M1 | 0.20000 | B5b | 29 | 1 | 0.00842 | 0.00842 | 0.00814 | 0.04148 |
| real_audit | MultiPref | validation | M2 | 0.05000 | B5b | 29 | 1 | 0.03856 | 0.03856 | 0.03728 | 0.07061 |
| real_audit | MultiPref | validation | M2 | 0.10000 | B3 | 29 | 1 | 0.00677 | 0.00680 | 0.00654 | 0.03987 |
| real_audit | MultiPref | test | M1 | 0.05000 | B5b | 29 | 1 | 0.03115 | 0.03115 | 0.03011 | 0.06344 |
| real_audit | MultiPref | test | M2 | 0.20000 | B5b | 29 | 1 | 0.01488 | 0.01488 | 0.01439 | 0.04772 |

`success_conditional_half_l1` excludes numerical failures and is explicitly conditional. `all_finite_raw_half_l1` is a sensitivity description that also retains finite nonconverged raw B3 values; it does not certify convergence. For successful normalized vectors, the last two columns assign a hypothetical failed-run loss between 0 and 1 across all eligible runs. This is a failure-loss sensitivity envelope, not an imputed estimate, a confidence interval or a proof of unbiasedness. Unknown B5b results cannot be recovered from success-only means. The original phase summaries remain intact and are superseded in interpretation by this explicit qualification.

B4 raw coordinates are nonnegative but their sum may differ from one; for M0/M1 the overall mass can be one while group masses differ. B5 can in principle return negative corrected coordinates, although no negative raw coordinate occurred in the saved Phase 2 runs. Vector-specific diagnostics and tests retain this distinction. Non-probability raw vectors are evaluated using half L1, not misrepresented as probability TV.

## 10. Limitations

The pilot is an ordered prefix, audit labels are simulated releases, and real selection kernels or nonrandom audit nonresponse are not studied. Conditional independence, sample homogeneity and annotator dependence can be violated. No global likelihood-identification proof, population confidence interval, coverage study, downstream LLM experiment or causal alignment claim is provided. The observed four annotations are a reference, not universal preference truth. Design unbiasedness does not guarantee a legal probability vector, low variance or universal dominance over audit-only estimation. B5 exact design statements are conditional and do not automatically survive numerical-success selection. The optional targeted-deleted-layer audit design was not run.

## 11. Reproducibility

Install Python 3.11+ and run from the repository root:

```bash
python -m pip install -e ".[dev,data]"
python scripts/reproduce_minimal.py
```

This command runs all correctness tests, exhaustive finite-design checks and 162 small synthetic method runs using existing estimators. It requires no real data, GPU or paid API. Its outputs are a smoke reproduction, not a replacement for the archived experiments. The data extra supplies PyArrow for preserved Parquet preprocessing tests; core numerical dependencies are NumPy, SciPy and pandas. See reproducibility_guide.md for empty output directories, isolated full replication, input SHA256 checks, numerical version dependence and CI scope. CI is configured for Python 3.11/3.12; GitHub-hosted execution has not occurred because no remote repository or push was authorized.

B4 and conditional B5a exhaustive design checks agree with the finite targets to a maximum absolute error of 1.11e-16. Tests independently check the B3 posterior and cross-fit audit-label isolation. Original Phase 1/2 result and protocol bytes are preserved; finalization products are separate.

## 12. Conclusion

Consensus filtering can materially change observed preference distributions. Classical likelihood can reduce error under its assumptions and fail under misspecification. Design-based residual correction and model assistance offer different robustness/variance tradeoffs, while ordinary post-stratification already gives strong results in several whole-comparison censoring conditions. No new statistical estimation theory or original algorithm has been established. The research lead has closed the project and decided against a six-month original-algorithm effort. This decision does not claim that all possible censoring problems are solved. No Phase 3 is proposed.
