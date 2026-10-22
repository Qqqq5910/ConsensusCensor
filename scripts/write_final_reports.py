"""Generate final reports from immutable phase outputs and final audit artifacts."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "experiments/finalization"


def table(frame: pd.DataFrame) -> str:
    def cell(x: object) -> str:
        if isinstance(x, float):
            return f"{x:.5f}" if pd.notna(x) else "not available"
        return str(x).replace("|", "/")

    return "\n".join(
        [
            "| " + " | ".join(map(str, frame.columns)) + " |",
            "| " + " | ".join(["---"] * len(frame.columns)) + " |",
        ]
        + [
            "| " + " | ".join(cell(x) for x in row) + " |"
            for row in frame.itertuples(index=False, name=None)
        ]
    )


def main() -> None:
    summary = pd.read_csv(ROOT / "experiments/phase2/summary.csv")
    failures = pd.read_csv(FINAL / "audit/failure_sensitivity.csv")
    pairs = pd.read_csv(FINAL / "audit/paired_complete_cases.csv")
    audit = json.loads((FINAL / "audit/audit.json").read_text())
    s = summary[
        (summary.experiment == "fixed_audit")
        & (summary.mechanism == "M2")
        & (summary.audit_rate == 0.2)
    ]
    synth = s.pivot(index="dataset", columns="method", values="tv_mean").reset_index()
    r = summary[
        (summary.experiment == "real_audit")
        & (summary.split == "test")
        & (summary.audit_rate == 0.2)
    ]
    real = r[
        [
            "mechanism",
            "method",
            "ok_n",
            "tv_mean",
            "normal_tv_mean",
            "expert_tv_mean",
            "group_gap_error_mean",
        ]
    ].rename(
        columns={
            "ok_n": "successful_runs",
            "tv_mean": "overall_half_L1",
            "normal_tv_mean": "normal_half_L1",
            "expert_tv_mean": "expert_half_L1",
        }
    )
    failure_rows = failures[failures.numeric_failure_n > 0][
        [
            "experiment",
            "dataset",
            "split",
            "mechanism",
            "audit_rate",
            "method",
            "success_n",
            "numeric_failure_n",
            "success_conditional_half_l1",
            "all_finite_raw_half_l1",
            "failure_loss_0_lower",
            "failure_loss_1_upper",
        ]
    ]
    pair_rows = pairs[
        (pairs.experiment == "real_audit") & (pairs.split == "test") & (pairs.audit_rate == 0.2)
    ][
        [
            "mechanism",
            "comparison",
            "paired_n",
            "lhs_same_pairs_mean",
            "rhs_same_pairs_mean",
            "paired_mean_difference",
        ]
    ]
    theory_error = audit["design_enumeration_max_error"]
    report = [
        "# ConsensusCensor: An Open Research Artifact on Consensus Filtering and Robust "
        "Statistical Estimation from Human Preference Data",
        "**Project status: completed; no Phase 3.**\n\n"
        "**Engineering Artifact: READY**\n\n"
        "**Original Research Contribution: NOT ESTABLISHED**",
        "## 1. Abstract",
        "We examine how controlled consensus filtering changes the observed distribution of "
        "four human-preference annotations per comparison, and how existing estimators use "
        "a limited random complete audit to recover a finite annotation distribution. "
        "An independent categorical likelihood model can be efficient under its working "
        "assumptions and persistently biased under shared latent factors or heterogeneous "
        "comparison difficulty. Horvitz–Thompson residual correction offers a design-based "
        "expectation guarantee but can increase variance or yield unnormalized raw vectors. "
        "Known-count post-stratification is a strong simple baseline for whole-comparison "
        "censoring. Cross-fitted model assistance has conditional benefits and does not "
        "uniformly improve upon either likelihood or post-stratification. The artifact "
        "implements and compares established methods; it establishes neither a new "
        "statistical estimation theory nor an original algorithm.",
        "## 2. Research Problem",
        "Each comparison contains two normal-worker slots and two expert-worker slots. "
        "Labels are A, Tie and B; Tie is never assigned to a winner. The target is the "
        "empirical distribution of the original four slots within a fixed public-data "
        "collection, with equal group weights. Separate group distributions use two slots "
        "per comparison. Support scores are (1, 0.5, 0), and the group contrast is expert "
        "minus normal support. These targets are not global human preferences, unique "
        "answer quality or a causal effect on downstream alignment.",
        "## 3. Related Work",
        "Filtering-induced label-distribution changes and substantive annotator disagreement "
        "are prior findings: Fleisig et al., *Balancing Quality and Variation* "
        "([ACL 2025](https://aclanthology.org/2025.nlperspectives-1.5/)); Zhang et al., "
        "*Diverging Preferences* ([ICML 2025](https://proceedings.mlr.press/v267/zhang25bx.html)); "
        "and Ghafouri et al., *RLHF May Not Reflect Genuine Preferences* "
        "([ICML 2026](https://proceedings.mlr.press/v306/ghafouri26a.html)). "
        "MultiPref comes from Miranda et al., *Hybrid Preferences* "
        "([ACL 2025](https://aclanthology.org/2025.acl-long.355/), arXiv:2410.19133). "
        "The NVIDIA HelpSteer3 dataset motivates the context, but our M1/M2 rules do not "
        "claim to reproduce its actual filtering procedure.",
        "Horvitz and Thompson (1952), DOI 10.1080/01621459.1952.10483446, provide the "
        "sampling-estimation background. Post-stratification, generalized difference/GREG "
        "and augmented inverse-probability correction are established ideas. "
        "[Prediction-Powered Inference](https://arxiv.org/abs/2301.09633), "
        "[Cross-Prediction-Powered Inference](https://arxiv.org/abs/2309.16598) and "
        "[ppi_py](https://github.com/aangelopoulos/ppi_py) provide rela"
        "ted prediction-plus-residual "
        "and cross-fitting ideas. We do not claim to invent PPI, AIPW, GREG or unbiased "
        "estimation, or transfer their assumptions and coverage guarantees without proof. "
        "Source checks and their limits are recorded in phase2_existing_methods.md.",
        "## 4. Statistical Formulation",
        r"For group $g$, define $z_{igc}=\frac12\sum_{j\in g}1(Y_{ij}=c)$, "
        r"$o_{igc}=\frac12\sum_{j\in g}M_{ij}1(Y_{ij}=c)$, and $r_{igc}=z_{igc}-o_{igc}$. "
        "The target is $\\theta_{gc}=N^{-1}\\sum_i z_{igc}$; overall targ"
        "ets average the two groups. "
        r"B4 is $N^{-1}\sum_i[o_i+A_i r_i/\pi_i]$. For a fixed finite population and observed "
        r"view, known positive inclusion probabilities give $E(A_i/\pi_i)=1$, hence the "
        "design expectation equals the target. Label independence is unnecessary. "
        "Under fixed-size SRS, actual inclusion probability is k/N with k=floor(rate*N). "
        "The variance depends on residual variation, not the variation of complete labels; "
        "it need not dominate audit-only estimation.",
        r"B5 is $N^{-1}\sum_i[o_i+\hat m_i+A_i(r_i-\hat m_i)/\pi_i]$. "
        "With fixed prompt folds under global fixed-size SRS, the implementation uses "
        "conditional fold probabilities k_f/N_f. Conditional on the fold counts, "
        "within-fold SRS selections are independent across folds; predictors trained "
        "outside a target fold can then be used in the generalized-difference argument. "
        "An unknown fold with zero audits is not estimable. This conditional argument "
        "requires a defined predictor on the relevant design support; it is not an "
        "unconditional unbiasedness guarantee after conditioning on numerical success. "
        "Detailed derivations, positivity requirements and exceptions appear in "
        "phase2_mathematics.md.",
        "## 5. Data and Selection Mechanisms",
        "The real pilot uses the preserved first 2,000 default/train MultiPref comparisons: "
        "8,000 valid annotation slots and 1,010 distinct prompts, with no unmapped labels "
        "in this prefix. The five raw strength levels remain in preprocessing; estimation "
        "uses only A/Tie/B. Prompt-group partitions contain 1,201 training, 399 validation "
        "and 400 test comparisons. This ordered prefix is not a probability sample of "
        "the whole dataset. MultiPref is labeled ODC-BY; see DATA_LICENSE.md and CITATION.bib.",
        "M0 uniformly deletes one of four slots. M1 keeps the three-slot subset minimizing "
        "the sum of pairwise distances on A=0, Tie=1, B=2, with uniform random tie-breaking. "
        "M2 keeps all four slots only if at least three labels have the same class, otherwise "
        "hiding all labels while preserving metadata and the selection indicator. These "
        "rules were unchanged between phases. All are controlled censoring simulations "
        "of originally complete public data.",
        "## 6. Estimators B0–B5",
        "| Estimator | Permitted information and assumptions |\n"
        "| --- | --- |\n"
        "| B0 | Pooled visible annotations; no audit correction. Visibl"
        "e group counts can differ. |\n"
        "| B1 | Complete randomly audited comparisons only. |\n"
        "| B2 | Group-stratified audit with equal target weights; ident"
        "ical to B1 under these balanced complete audits. |\n"
        "| B3 | Group-specific independent categorical model; exact 81-"
        "state observed likelihood with the known selection kernel. |\n"
        "| B4 | All visible contributions plus inverse-probability-weig"
        "hted audited missing contributions. |\n"
        "| B4-M2 | Known kept/deleted layer weights; exact visible reta"
        "ined contribution plus the audited deleted-layer mean. |\n"
        "| B5a | Cross-fitted missing-group frequencies, fixed 0.5 smoo"
        "thing, plus residual correction. |\n"
        "| B5b | Cross-fitted B3 posterior expectations of hidden slots"
        ", plus residual correction. |",
        "For M2 post-stratification, the missing layer has weight D/N, and group contributions "
        "still use denominator two. If D>0 but no deleted comparison is audited, the method "
        "returns not estimable. B5b conditions on the actual retained subset and tie-breaking "
        "kernel for M1 and the rejected event for M2; it never replaces hidden labels "
        "with the unconditional B3 point estimate.",
        "## 7. Experimental Design",
        "Phase 1 used five fixed audit/selection seeds. Phase 2 fixed each synthetic "
        "population and its observed censoring view, then used 50 independent audit "
        "randomizations per scenario. A separate experiment used 20 independent population "
        "generation seeds per scenario, with common audit-index and selection random streams. "
        "The latter measures population-generation variation under those fixed assignments, "
        "not 20 additional within-population audit replications. Real train/validation/test "
        "targets each used 30 audit randomizations. Audit rates were 0%, 5%, 10% and 20%. "
        "Each synthetic population had N=800 in Phase 2.",
        "S1 satisfies the independent group model. S2 uses an unobserved shared factor "
        "to correlate annotations. S3 mixes high-consensus and high-disagreement difficulty "
        "classes with explicitly configured, group-specific probabilities. The latent "
        "classes are not given to selection or prediction. All methods in a cell share "
        "the same observed view, audit indices and allocated budget. Frozen protocols "
        "are preserved. There was no test-Oracle model or hyperparameter selection.",
        "## 8. Results",
        "All following numerical tables are generated from saved Phase 2 artifacts, not "
        "new large experiments. Entries summarize the finite complete-label target. "
        "**Half-L1 loss is named consistently below:** it is standard TV only when "
        "the relevant raw vector is a normalized probability distribution. No clipping "
        "or projection was applied.",
        "### 8.1 Fixed synthetic populations, M2, 20% audit",
        table(synth),
        "Under correct specification, B3 is more efficient in this comparison. "
        "Under S2/S3 misspecification, its low audit-randomization variability can coexist "
        "with substantial persistent error. B4 trades that model dependence for audit "
        "variance. Simple M2 post-stratification is already competitive with the more "
        "expensive model-assisted variants; this is a result about existing baselines, "
        "not evidence of a new method contribution.",
        "### 8.2 Preserved MultiPref test target, 20% audit",
        table(real),
        "These are success-conditional means, not unconditional algorithmic risks. "
        "In particular, M2 B5b has 29 successful runs out of 30. The real pilot does not "
        "show design-based/model-assisted methods uniformly outperforming B3, including "
        "for the normal-versus-expert support contrast. Synthetic misspecification benefits "
        "cannot be assumed to transfer to the real-data target.",
        "### 8.3 Same-randomization paired comparisons, real test, 20% audit",
        table(pair_rows),
        "Both means in each row use the identical jointly successful randomizations; "
        "negative differences favor the first method. Complete-case comparisons can still "
        "be affected by label-dependent numerical success. They are accompanied by the "
        "failure accounting below, rather than presented as unconditional improvements.",
        "## 9. Failure Analysis",
        f"The final audit checked {audit['archived_rows']:,} archived rows and "
        f"{audit['audit_records_checked']:,} sampling records. It confirmed three B3 and "
        "12 B5b numerical-failure runs. The three B3 runs retain finite raw estimates; "
        "the B5b failures contain no returned estimate. Neither is silently reclassified "
        "as success. Zero-audit unavailability and method/mechanism non-applicability "
        "are separately recorded.",
        table(failure_rows),
        "`success_conditional_half_l1` excludes numerical failures and is explicitly "
        "conditional. `all_finite_raw_half_l1` is a sensitivity description that also "
        "retains finite nonconverged raw B3 values; it does not certify convergence. "
        "For successful normalized vectors, the last two columns assign a hypothetical "
        "failed-run loss between 0 and 1 across all eligible runs. This is a failure-loss "
        "sensitivity envelope, not an imputed estimate, a confidence interval or a proof "
        "of unbiasedness. Unknown B5b results cannot be recovered from success-only means. "
        "The original phase summaries remain intact and are superseded in interpretation "
        "by this explicit qualification.",
        "B4 raw coordinates are nonnegative but their sum may differ from one; for M0/M1 "
        "the overall mass can be one while group masses differ. B5 can in principle return "
        "negative corrected coordinates, although no negative raw coordinate occurred in "
        "the saved Phase 2 runs. Vector-specific diagnostics and tests retain this "
        "distinction. Non-probability raw vectors are evaluated using half L1, not "
        "misrepresented as probability TV.",
        "## 10. Limitations",
        "The pilot is an ordered prefix, audit labels are simulated releases, and real "
        "selection kernels or nonrandom audit nonresponse are not studied. Conditional "
        "independence, sample homogeneity and annotator dependence can be violated. "
        "No global likelihood-identification proof, population confidence interval, "
        "coverage study, downstream LLM experiment or causal alignment claim is provided. "
        "The observed four annotations are a reference, not universal preference truth. "
        "Design unbiasedness does not guarantee a legal probability vector, low variance "
        "or universal dominance over audit-only estimation. B5 exact design statements "
        "are conditional and do not automatically survive numerical-success selection. "
        "The optional targeted-deleted-layer audit design was not run.",
        "## 11. Reproducibility",
        "Install Python 3.11+ and run from the repository root:\n\n"
        '```bash\npython -m pip install -e ".[dev,data]"\n'
        "python scripts/reproduce_minimal.py\n```",
        "This command runs all correctness tests, exhaustive finite-design checks and "
        "162 small synthetic method runs using existing estimators. It requires no real "
        "data, GPU or paid API. Its outputs are a smoke reproduction, not a replacement "
        "for the archived experiments. The data extra supplies PyArrow for preserved "
        "Parquet preprocessing tests; core numerical dependencies are NumPy, SciPy and "
        "pandas. See reproducibility_guide.md for empty output directories, isolated full "
        "replication, input SHA256 checks, numerical version dependence and CI scope. "
        "CI is configured for Python 3.11/3.12; GitHub-hosted execution has not occurred "
        "because no remote repository or push was authorized.",
        f"B4 and conditional B5a exhaustive design checks agree with the finite targets "
        f"to a maximum absolute error of {theory_error:.3g}. Tests independently check "
        "the B3 posterior and cross-fit audit-label isolation. Original Phase 1/2 "
        "result and protocol bytes are preserved; finalization products are separate.",
        "## 12. Conclusion",
        "Consensus filtering can materially change observed preference distributions. "
        "Classical likelihood can reduce error under its assumptions and fail under "
        "misspecification. Design-based residual correction and model assistance offer "
        "different robustness/variance tradeoffs, while ordinary post-stratification "
        "already gives strong results in several whole-comparison censoring conditions. "
        "No new statistical estimation theory or original algorithm has been established. "
        "The research lead has closed the project and decided against a six-month "
        "original-algorithm effort. This decision does not claim that all possible "
        "censoring problems are solved. No Phase 3 is proposed.",
    ]
    (ROOT / "docs/final_research_report.md").write_text("\n\n".join(report) + "\n")
    chinese = [
        "# ConsensusCensor 最终项目总结",
        "**Engineering Artifact: READY**\n\n**Original Research Contribution: NOT ESTABLISHED**",
        "## 研究背景与边界",
        "研究共识筛选如何改变人类偏好标签的观察分布，以及少量随机完整审计能够恢复多少信息。"
        "目标是公开数据内四个标注槽位的经验分布，不是全球偏好或下游模型质量。"
        "项目已正式收口，不进入 Phase 3。",
        "## 两阶段完成内容",
        "Phase 1 实现 M0–M2、B0–B3、81 状态似然、数据转换和 2000 比较 Pilot。"
        "Phase 2 实现既有 HT 残差估计、M2 后分层、两种交叉拟合辅助估计；"
        "各场景 50 次固定总体审计、20 个独立总体，真实分区各 30 次审计，保存 28800 个方法运行。"
        "最终收口只新增小规模复现和检查，没有重新运行大型实验或修改冻结协议。",
        "## 经确认的主要发现",
        table(synth),
        "表格对应 M2、20% 审计的固定合成总体半 L1 平均损失。模型正确时 B3 更精确；"
        "共享因素和难度异质性会使其产生稳定偏差。B4 通过估计缺失残差降低模型依赖，"
        "但不保证误差或方差总是优于仅审计。M2 简单后分层已是强基线，复杂 B3 辅助没有形成稳定优势。"
        "真实 Pilot 上 B3 的总体误差仍较低，设计稳健性没有自动变成所有指标的优势。",
        "## 统计正确性与最终核查",
        f"设计期望独立穷举最大差异 {theory_error:.3g}。B4 证明基于已知正入样概率；"
        "B5 采用固定 prompt 折、折外训练和实际折审计数形成的条件入样概率。"
        "组内分母为 2，总体为两组等权平均；M2 使用真实删除层权重。"
        "所有方法共享同一观察视图及审计索引，允许信息的不同使用方式属于基线定义。",
        "3 次 B3 数值失败有有限原始值；12 次 B5b 失败没有输出。历史主要均值只针对成功运行，"
        "不能理解为无条件算法风险。最终新增失败损失敏感性范围、共同成功运行配对和失败率。"
        "真实 M2/20% 的 B5b 均值来自 29/30 次成功运行。原始结果没有被替换。",
        "B4 总体或组内概率和可能不为 1，统一以半 L1 损失描述原始向量；只有合法概率向量才称标准 TV。"
        "不裁剪、不投影。固定总体审计变化与重新生成总体变化分别统计，不把审计 SD 当人群置信区间。",
        "## 研究局限与停止原因",
        "真实数据只是顺序前缀，筛选和审计均为受控模拟；未知机制、非随机审计、全局识别、"
        "置信区间覆盖率与下游效果均未建立。成功子集还可能受数值收敛选择影响。"
        "本项目没有新的 PPI/AIPW/GREG/HT 理论或原创算法。",
        "研究负责人决定停止新增算法研究：现有方法已解决相当一部分目标问题，"
        "本轮证据不足以支持六个月的原创算法研究投入。代码量、测试数量和实验次数不改变这个判断。"
        "不会再建议或自动进行 Phase 3。",
        "## 最小复现命令",
        '```bash\npython -m pip install -e ".[dev,data]"\npython scripts/reproduce_minimal.py\n```',
        "默认输出 experiments/minimal；已有输出时请用 --output 指定新的空目录。"
        "无需下载真实数据或运行全部大型实验。详见 reproducibility_guide.md。",
        "## GitHub 发布前检查",
        "- 本地验证证据见 experiments/finalization。",
        "- 发布前确认 CITATION.cff 的集体署名。",
        "- 确认 MIT 的项目代码授权范围，并保留 MultiPref 的 ODC-BY 署名与数据许可说明。",
        "- 创建或选择远端、添加真实链接、首次推送和查看 GitHub CI 实际结果由负责人另行操作。"
        "本轮仅完成本地提交，不创建远端、不推送、不发布 Release。",
        "- 公开发布时保留关于无原创理论贡献、数值失败、条件均值、非法向量和数据范围的说明。",
    ]
    (ROOT / "docs/final_project_summary_CN.md").write_text("\n\n".join(chinese) + "\n")
    print("Generated final English research report and Chinese project summary from archived data")


if __name__ == "__main__":
    main()
