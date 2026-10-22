"""All numerical Phase 2 tables derive from actual run products."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/phase2"
KEYS = ["experiment", "dataset", "split", "mechanism", "audit_rate", "method"]
ERRORS = [
    "tv",
    "normal_tv",
    "expert_tv",
    "score_error",
    "group_gap_error",
    "mse",
    "normal_mse",
    "expert_mse",
    "group_gap_squared_error",
]


def table(frame: pd.DataFrame, digits: int = 5) -> str:
    def cell(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{value:.{digits}f}" if np.isfinite(value) else "—"
        return str(value).replace("|", "/")

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


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(KEYS, sort=False):
        ok = group[group.status == "ok"]
        applicable = group[group.status != "not_applicable"]
        record = dict(zip(KEYS, key, strict=True))
        record.update(
            {
                "runs": len(group),
                "ok_n": len(ok),
                "unavailable_rate": float((group.status == "not_estimable").mean()),
                "numeric_failure_rate": float((group.status == "numeric_failure").mean()),
                "applicable_n": len(applicable),
                "invalid_raw_rate_among_ok": float((~ok.raw_valid).mean()),
                "extra_labels_mean": group.extra_labels.mean(),
                "cpu_seconds_mean": group.cpu_seconds.mean(),
            }
        )
        for name in ["overall", "normal", "expert"]:
            record[f"{name}_invalid_raw_rate_among_ok"] = float((~ok[f"{name}_raw_valid"]).mean())
        for metric in ERRORS:
            record[metric + "_mean"] = ok[metric].mean()
            record[metric + "_sd"] = ok[metric].std()
        for name in ["overall", "normal", "expert"]:
            for c in "ATB":
                record[f"{name}_{c}_bias"] = ok[f"{name}_{c}_error"].mean()
                record[f"{name}_{c}_variance"] = ok[f"{name}_{c}"].var()
                record[f"{name}_{c}_error_variance"] = ok[f"{name}_{c}_error"].var()
            biases = [record[f"{name}_{c}_bias"] for c in "ATB"]
            record[f"{name}_bias_linf"] = max(abs(x) for x in biases)
        record["group_gap_signed_bias"] = ok.group_gap_signed_error.mean()
        rows.append(record)
    return pd.DataFrame(rows)


def paired(frame: pd.DataFrame) -> pd.DataFrame:
    index = [
        "experiment",
        "dataset",
        "split",
        "population_seed",
        "audit_seed",
        "mechanism",
        "audit_rate",
        "audit_hash",
    ]
    ok = frame[frame.status == "ok"]
    rows = []
    for metric in ["tv", "mse", "normal_tv", "expert_tv", "group_gap_error"]:
        wide = ok.pivot(index=index, columns="method", values=metric)
        for lhs, rhs in [
            ("B4", "B1"),
            ("B4-M2", "B4"),
            ("B5a", "B4"),
            ("B5b", "B4"),
            ("B5b", "B3"),
            ("B4", "B3"),
        ]:
            if lhs not in wide or rhs not in wide:
                continue
            difference = (wide[lhs] - wide[rhs]).dropna().rename("difference").reset_index()
            group_keys = ["experiment", "dataset", "split", "mechanism", "audit_rate"]
            for key, group in difference.groupby(group_keys):
                row = dict(zip(group_keys, key, strict=True))
                row.update(
                    {
                        "metric": metric,
                        "comparison": lhs + " minus " + rhs,
                        "paired_n": len(group),
                        "mean_difference": group.difference.mean(),
                        "sd_difference": group.difference.std(),
                        "win_fraction": float((group.difference < 0).mean()),
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def pivot(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    return summary.pivot(
        index=["mechanism", "audit_rate"], columns="method", values=metric
    ).reset_index()


def main() -> None:
    frame = pd.read_csv(OUT / "results.csv")
    # Derive vector-specific diagnostics without altering any recorded estimates.
    raw_diagnostics = frame[
        [
            "experiment",
            "dataset",
            "split",
            "population_seed",
            "audit_seed",
            "mechanism",
            "audit_rate",
            "method",
            "status",
        ]
    ].copy()
    for name in ["overall", "normal", "expert"]:
        values = frame[[f"{name}_{c}" for c in "ATB"]].to_numpy()
        finite = np.isfinite(values).all(axis=1)
        mass_error = np.abs(values.sum(axis=1) - 1)
        valid = finite & (values >= -1e-10).all(axis=1) & (values <= 1 + 1e-10).all(axis=1)
        valid &= mass_error <= 1e-10
        frame[f"{name}_raw_valid"] = valid
        raw_diagnostics[f"{name}_mass_error"] = mass_error
        raw_diagnostics[f"{name}_min"] = values.min(axis=1)
        raw_diagnostics[f"{name}_valid"] = valid
    raw_diagnostics.to_csv(OUT / "raw_probability_diagnostics.csv", index=False)
    summary = summarize(frame)
    pairs = paired(frame)
    summary.to_csv(OUT / "summary.csv", index=False)
    pairs.to_csv(OUT / "paired_comparisons.csv", index=False)
    tests = (OUT / "pytest.txt").read_text().strip()
    theory = pd.read_csv(OUT / "design_enumeration.csv")
    protocol = json.loads((OUT / "frozen_protocol.json").read_text())
    fixed = summary[summary.experiment == "fixed_audit"]
    regen = summary[summary.experiment == "regenerated"]
    real = summary[(summary.dataset == "MultiPref") & (summary.split == "test")]
    text = [
        "# Phase 2 实际结果（自动生成）",
        "表格全部读取 experiments/phase2 实验产物；未复制第一阶段结果作为本轮证据。"
        "误差单位 0–1。对不合法原始概率向量，TV 列表示半 L1 损失，严格意义的"
        "概率分布 TV 只适用于合法概率。无投影/裁剪变体。",
        "## 执行范围",
        tests,
        f"记录 {len(frame)} 个方法运行；合成固定总体每场景 "
        f"{len(protocol['fixed_audit_seeds'])} 次审计随机化、每场景 "
        f"{len(protocol['population_seeds'])} 个独立生成总体；"
        f"每个真实分区 {len(protocol['real_audit_seeds'])} 次审计随机化。"
        f"合成总体大小 {protocol['synthetic_n']}。",
        "M0/M1/M2 的筛选种子固定，固定总体实验中只重新抽审计。"
        "重新生成总体实验使用共同固定审计索引/筛选随机种子，变化来自独立生成总体；"
        "两类变异分开统计，不合并成一个独立样本池。",
        "## 理论验证",
        table(theory),
        "B4 在已知正入样概率下设计无偏，证明见 phase2_mathematics.md；"
        "枚举是独立实现验证，不以某次实验准确冒充证明。"
        "B5 仅在固定折、折内条件入样概率、折外训练和可估计折计数条件下作条件设计期望声明。",
        "## 实验一：固定完整总体、仅审计随机化",
    ]
    for scenario in ["S1", "S2", "S3"]:
        subset = fixed[fixed.dataset == scenario]
        text += [
            f"### {scenario}：总体误差，低审计及较高审计",
            table(
                subset[subset.audit_rate.isin([0.05, 0.2])][
                    [
                        "mechanism",
                        "audit_rate",
                        "method",
                        "ok_n",
                        "tv_mean",
                        "tv_sd",
                        "mse_mean",
                        "overall_bias_linf",
                        "group_gap_error_mean",
                    ]
                ]
            ),
            f"### {scenario}：M2 全审计比例及原始输出合法性",
            table(
                subset[subset.mechanism == "M2"][
                    [
                        "audit_rate",
                        "method",
                        "tv_mean",
                        "mse_mean",
                        "invalid_raw_rate_among_ok",
                        "unavailable_rate",
                        "numeric_failure_rate",
                    ]
                ]
            ),
        ]
    text += [
        "固定总体表中 SD 是审计设计随机化的离散程度，不是人群抽样置信区间。"
        "各类有符号偏差、组内方差和误差方差完整保存在 summary.csv。",
        "## 实验二：重新生成完整总体",
        "下表每单元对应独立生成总体。审计索引与随机筛选流固定；"
        "不能把跨总体变化解释为固定总体内审计方差。",
        table(
            regen[(regen.mechanism == "M2") & regen.audit_rate.isin([0.05, 0.2])][
                [
                    "dataset",
                    "audit_rate",
                    "method",
                    "ok_n",
                    "tv_mean",
                    "tv_sd",
                    "mse_mean",
                    "normal_tv_mean",
                    "expert_tv_mean",
                    "group_gap_error_mean",
                ]
            ]
        ),
        "其余机制与比例见 summary.csv；相对生成参数误差另保存在 results.csv 的 truth_* 列。",
        "## 真实 MultiPref Pilot：沿用原 2000 比较与问题划分",
        table(
            pd.DataFrame(
                list(json.loads((OUT / "original_split_counts.json").read_text()).items()),
                columns=["partition", "comparisons"],
            )
        ),
        "原始数据 SHA256 与第一阶段记录核验一致；不重新下载或改变原分区。"
        "方法和 alpha=.5、两折、系数=1 在测试评分前固定，无测试 Oracle 调参。"
        "以下主表针对锁定测试分区；train/validation 全部结果也保留。",
        "### 总体 TV / 半 L1 平均损失",
        table(pivot(real, "tv_mean")),
        "### 普通组 TV / 半 L1 平均损失",
        table(pivot(real, "normal_tv_mean")),
        "### 专家组 TV / 半 L1 平均损失",
        table(pivot(real, "expert_tv_mean")),
        "### 两组支持度差的平均绝对误差",
        table(pivot(real, "group_gap_error_mean")),
        "### 20% 审计的偏差、MSE 与原始概率合法性",
        table(
            real[real.audit_rate == 0.2][
                [
                    "mechanism",
                    "method",
                    "ok_n",
                    "overall_bias_linf",
                    "mse_mean",
                    "invalid_raw_rate_among_ok",
                    "unavailable_rate",
                    "numeric_failure_rate",
                ]
            ]
        ),
        "## 固定协议的配对比较",
        "负差值表示前一种方法损失更低。只在同一目标、机制、审计索引与双方成功时配对；"
        "不把失败运行偷偷计为成功。样本数与失败率必须结合阅读。",
        table(
            pairs[
                (pairs.experiment == "real_audit")
                & (pairs.split == "test")
                & (pairs.metric == "tv")
                & (pairs.audit_rate.isin([0.05, 0.2]))
            ][
                [
                    "mechanism",
                    "audit_rate",
                    "comparison",
                    "paired_n",
                    "mean_difference",
                    "sd_difference",
                    "win_fraction",
                ]
            ]
        ),
        "## 新增标签获取成本与 CPU 时间",
        table(
            real[real.audit_rate.isin([0.05, 0.2])][
                ["mechanism", "audit_rate", "method", "extra_labels_mean", "cpu_seconds_mean"]
            ]
        ),
        "extra_labels 仅计所选审计比较中原来不可见的槽位。audit_full_slots=4k "
        "另存名义完整审计槽位数。M2 已保留样本的完整标签免费可见，不重复计获取成本。"
        "相同审计抽样对所有方法成本相同；未开展优先审计删除层的补充设计。"
        "计时为估计器 CPU 时间，包含交叉拟合训练，不含数据加载、筛选和 I/O。",
        "## 数值失败、不可估计与原始异常输出",
        table(frame.groupby(["method", "status"]).size().rename("runs").reset_index()),
        "不可估计（包括零审计、未知折无审计、M2 删除层无审计）与优化失败分开；"
        "B4-M2 在其他机制属于不适用，不能算失败。所有原始向量都保留，无静默裁剪。"
        "非收敛 B3 的有限数值也保留，但主均值按 status=ok 计算。",
        "## 范围与限制",
        "真实数据是原有顺序前缀 Pilot，不代表整个人群；目标是本次完整标注槽位分布。"
        "B4 设计无偏不保证每次结果更准或有合法概率和。B5 的预测质量影响方差，"
        "交叉拟合不能自动克服固定量抽样依赖，条件权重及正折审计量不可省略。"
        "实际机制未知、审计非随机/非响应、单个标注者跨比较依赖等均未扩展验证。"
        "本轮无显著性声明、无新的算法命名，不自动开展下一阶段。",
    ]
    # Data-driven paired evidence prevents a qualitative claim from silently outrunning results.
    focal = pairs[
        (pairs.experiment == "real_audit")
        & (pairs.split == "test")
        & (pairs.metric == "tv")
        & (pairs.audit_rate == 0.2)
    ]
    s2 = fixed[(fixed.dataset == "S2") & (fixed.mechanism == "M2") & (fixed.audit_rate == 0.2)]
    failure_text = [
        "# Phase 2 失败与限制分析（自动生成）",
        "## S2 / M2：原先的失配问题，本轮重新验证",
        table(
            s2[
                [
                    "method",
                    "ok_n",
                    "tv_mean",
                    "tv_sd",
                    "mse_mean",
                    "normal_tv_mean",
                    "expert_tv_mean",
                    "group_gap_error_mean",
                ]
            ]
        ),
        "## 真实测试的群体差异问题",
        table(
            real[(real.mechanism == "M2") & real.audit_rate.isin([0.05, 0.2])][
                [
                    "audit_rate",
                    "method",
                    "normal_tv_mean",
                    "expert_tv_mean",
                    "group_gap_error_mean",
                    "group_gap_signed_bias",
                ]
            ]
        ),
        "## 低审计、模型失配和概率合法性",
        table(
            summary[
                (summary.audit_rate == 0.05)
                & (
                    (summary.experiment == "fixed_audit")
                    | ((summary.experiment == "real_audit") & (summary.split == "test"))
                )
                & summary.method.isin(["B4", "B4-M2", "B5a", "B5b"])
            ][
                [
                    "experiment",
                    "dataset",
                    "mechanism",
                    "method",
                    "tv_mean",
                    "invalid_raw_rate_among_ok",
                    "unavailable_rate",
                    "numeric_failure_rate",
                ]
            ]
        ),
        "B4 并非方差普遍支配 Audit-Only；原有完整 z 在总体中恒定而缺失残差随样本"
        "变化时，Audit-Only 可有零方差而 B4 非零，已加反例测试。",
        "普通 B4 在 M2 下对删除层数量做 HT 扩张，有限审计可能高估/低估总概率质量。"
        "B4-M2 使用实际层占比，但删除层无审计时不可估计；其设计期望声明有条件。",
        "B5 预测质量差可能增加方差；alpha=.5 与系数=1 固定，无按真值优化。"
        "B5b 折训练不收敛或触边即记录失败，不用 B3 总体点估计替代后验，不回退隐藏真值。",
        "两组和总体的非负/和为1均检查，原始预测与误差全部保存。"
        "未做投影，因而也没有以投影后的结果宣称设计无偏。",
        "## 数值失败原始记录",
        table(
            frame[frame.status == "numeric_failure"][
                [
                    "experiment",
                    "dataset",
                    "split",
                    "population_seed",
                    "audit_seed",
                    "mechanism",
                    "audit_rate",
                    "method",
                    "message",
                ]
            ].head(80)
        ),
        "超过表中展示行数的完整失败记录仍在 results.csv；头部展示不代表删去其他运行。",
    ]
    evidence = [
        "# Phase 2 决策证据（自动生成）",
        "## 已证明或由已知理论直接适用的性质",
        "- B4：固定完整有限总体与观测视图、已知正入样概率下，线性期望给出设计无偏。",
        "- B4：SRS 下方差由缺失残差方差决定，不保证小于完整 z 的审计估计方差。",
        "- B5：固定折与条件折审计计数、折外训练、目标折正条件概率下的条件设计期望；"
        "不能声称任意数据依赖下无条件有限样本无偏。",
        "- M2 后分层：给定删除层有正审计量，层内 SRS 均值按已知层权重合成；空层不可估计。",
        table(theory),
        "## 合成观察：S2/M2、20% 审计",
        table(s2[["method", "tv_mean", "mse_mean", "group_gap_error_mean"]]),
        "## 真实 Pilot：20% 审计的配对 TV 差",
        table(
            focal[
                [
                    "mechanism",
                    "comparison",
                    "paired_n",
                    "mean_difference",
                    "sd_difference",
                    "win_fraction",
                ]
            ]
        ),
        "## 现有方法是否足够：仅列已观察困难",
        "比较必须同时看误差、方差、群体差异、合法性、失败率和标签成本；"
        "设计无偏不等于所有单次估计更好。当前真实测试各指标和场景并非统一排序，"
        "低审计与模型失配仍由本轮表格量化。",
        "这些方法都是现有 HT/后分层/广义差分/PPI/交叉拟合思想，"
        "无需为了六个月计划而创造算法。若其误差和获取成本已满足负责人预定容忍度，"
        "本轮没有额外证据表明必须发明新算法。",
        "未预先给定业务/研究可接受误差阈值，不能用本次误差自行宣布充分解决所有目标。"
        "本轮不提出新算法或新课题，停止等待负责人判断。",
    ]

    # Evidence-led interpretation: fixed protocol, no parameter changes from these results.
    def value(dataset: str, method: str, metric: str = "tv_mean") -> float:
        row = fixed[
            (fixed.dataset == dataset)
            & (fixed.mechanism == "M2")
            & (fixed.audit_rate == 0.2)
            & (fixed.method == method)
        ]
        return float(row.iloc[0][metric])

    overview = (
        fixed[(fixed.mechanism == "M2") & (fixed.audit_rate == 0.2)]
        .pivot(index="dataset", columns="method", values="tv_mean")
        .reset_index()
    )
    interpretation = [
        "## 主要发现与解释",
        "下面是同一固定总体、同一审计预算下 M2/20% 的平均损失对照；"
        "排序变化体现模型精度与设计稳健性的取舍，不能只看一种场景。",
        table(overview),
        f"S1 模型正确时，B3={value('S1', 'B3'):.5f}，B5b={value('S1', 'B5b'):.5f}。"
        "残差校正引入审计波动，并未保留正确模型下 B3 的全部精度优势。",
        f"S2 模型失配时，B3 最大坐标经验偏差={value('S2', 'B3', 'overall_bias_linf'):.5f}，"
        f"TV 的审计随机化 SD={value('S2', 'B3', 'tv_sd'):.5f}；"
        "这是在本固定总体上的稳定偏差表现，而非审计波动很大。"
        f"B4 的对应经验偏差={value('S2', 'B4', 'overall_bias_linf'):.5f}，"
        f"但 TV SD={value('S2', 'B4', 'tv_sd'):.5f}，稳健性伴随更多审计方差。",
        f"S2 的 B4-M2={value('S2', 'B4-M2'):.5f}，B5a={value('S2', 'B5a'):.5f}，"
        f"B5b={value('S2', 'B5b'):.5f}；"
        "此处简单后分层已消除大量损失，复杂 B3 辅助没有超过它。"
        "重新生成总体的排序可能改变，跨总体表必须独立阅读。",
        "B4 使用已知可见贡献，主要价值是把审计要估计的量从完整 z 缩小为残差 r。"
        "B5 进一步尝试降低残差方差，但折内条件权重与有限训练样本也会增加波动；"
        "没有普遍优于 B4 或后分层的保证。",
    ]
    if len(real):
        real_m2 = real[(real.mechanism == "M2") & (real.audit_rate == 0.2)].set_index("method")
        interpretation += [
            f"真实测试 M2/20% 下，B1={real_m2.loc['B1', 'tv_mean']:.5f}，"
            f"B4={real_m2.loc['B4', 'tv_mean']:.5f}，"
            f"B4-M2={real_m2.loc['B4-M2', 'tv_mean']:.5f}，"
            f"B5b={real_m2.loc['B5b', 'tv_mean']:.5f}，"
            f"B3={real_m2.loc['B3', 'tv_mean']:.5f}。"
            "本 Pilot 未显示设计稳健方法在总体误差上超过 B3，不能把合成失配场景的收益直接外推。",
            f"同一真实场景，两组支持度差误差 B3={real_m2.loc['B3', 'group_gap_error_mean']:.5f}，"
            f"B4={real_m2.loc['B4', 'group_gap_error_mean']:.5f}，"
            f"B5b={real_m2.loc['B5b', 'group_gap_error_mean']:.5f}。"
            "群体差异仍需较多审计信息，不能用总体分布的改善替代这个目标。",
        ]
    low = fixed[
        (fixed.dataset == "S1") & (fixed.mechanism == "M2") & (fixed.audit_rate == 0.05)
    ].set_index("method")
    interpretation += [
        f"低审计也有明确反例：S1/M2/5% 下 B4={low.loc['B4', 'tv_mean']:.5f}，"
        f"B1={low.loc['B1', 'tv_mean']:.5f}。"
        "即使 B4 设计无偏且用了所有可见标签，有限审计下平均损失仍可能更高。",
        f"独立审计集合枚举的最大期望误差为 "
        f"{theory.max_design_expectation_error.max():.3g}；"
        "这是实现的数值核验精度，理论期望依据显式推导。",
    ]
    text[2:2] = interpretation
    evidence[2:2] = interpretation
    raw_table = summary[
        (summary.audit_rate == 0.2)
        & (
            (summary.experiment == "fixed_audit")
            | ((summary.experiment == "real_audit") & (summary.split == "test"))
        )
        & summary.method.isin(["B4", "B4-M2", "B5a", "B5b"])
    ][
        [
            "experiment",
            "dataset",
            "mechanism",
            "method",
            "overall_invalid_raw_rate_among_ok",
            "normal_invalid_raw_rate_among_ok",
            "expert_invalid_raw_rate_among_ok",
        ]
    ]
    text += [
        "## 原始概率合法性分开核验",
        table(raw_table),
        "M0/M1 中普通 B4 的总体质量可正确为1，而组内质量仍可能不为1；"
        "因此不能把整体合法性检查失败一概解释为总体分布不合法。"
        "raw_probability_diagnostics.csv 逐次保留各向量质量误差与最小值。",
    ]
    (ROOT / "docs/phase2_results.md").write_text("\n\n".join(text) + "\n")
    (ROOT / "docs/phase2_failure_analysis.md").write_text("\n\n".join(failure_text) + "\n")
    (ROOT / "docs/phase2_decision_evidence.md").write_text("\n\n".join(evidence) + "\n")
    print(f"Generated summaries for {len(frame)} method runs and {len(pairs)} paired contrasts")


if __name__ == "__main__":
    main()
