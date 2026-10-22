"""Generate reports solely from recorded experiments; never insert invented results."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/phase1"


def table(frame: pd.DataFrame, digits: int = 5) -> str:
    def value(x: object) -> str:
        if isinstance(x, (float, np.floating)):
            return f"{x:.{digits}f}" if np.isfinite(x) else "不可用"
        return str(x)

    return "\n".join(
        [
            "| " + " | ".join(map(str, frame.columns)) + " |",
            "| " + " | ".join(["---"] * len(frame.columns)) + " |",
        ]
        + [
            "| " + " | ".join(value(x) for x in row) + " |"
            for row in frame.itertuples(index=False, name=None)
        ]
    )


def mean_sd(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    result = frame.groupby(["mechanism", "audit_rate", "method"])[metric].agg(["mean", "std"])
    result["value"] = result.apply(
        lambda x: f"{x['mean']:.5f} ± {x['std']:.5f}" if np.isfinite(x["mean"]) else "不可用",
        axis=1,
    )
    return result.value.unstack("method").reset_index()


def main() -> None:
    results = pd.read_csv(OUT / "results.csv")
    impacts = pd.read_csv(OUT / "selection_impact.csv")
    real = results[(results.dataset == "MultiPref") & (results.split == "test")]
    text = [
        "# ConsensusCensor 第一阶段实际结果",
        "自动生成。数值来自 results.csv、selection_impact.csv 与数据审计。"
        "所有 TV 和误差均为 0–1 单位；例如 0.01 表示一个百分点。"
        "± 为五次固定目标删失/审计随机化的样本标准差，不是置信区间。",
        "## 执行范围与检查",
        (OUT / "pytest.txt").read_text().strip(),
        "未调用付费 API，未训练大模型，未创建远端仓库。"
        + (
            "完整测试、合成后真实 Pilot 已执行。真实标签不提交。"
            if len(real)
            else "仅执行合成实验，真实数据实验未执行。"
        ),
    ]
    tests = (OUT / "pytest.txt").read_text().strip()
    if len(real) and (OUT / "data_audit.json").exists():
        audit = json.loads((OUT / "data_audit.json").read_text())
        probs = np.array(audit["distribution_normal_expert"])
        text += [
            "## 数据事实",
            f"MultiPref default/train 实际读取 {audit['rows_read']} 个比较；"
            f"完整可用 {audit['usable']}；唯一 prompt {audit['unique_prompts']}；"
            f"异常记录 {len(audit['anomalies'])}。数据 API 顺序前 n 行，不是全库随机样本。",
            "真实五档原始计数：",
            table(
                pd.DataFrame(list(audit["raw_label_counts"].items()), columns=["原始值", "槽位数"])
            ),
            "全体输入的参考分布（不是世界人群真值）：",
            table(
                pd.DataFrame(
                    [*probs, probs.mean(axis=0)],
                    index=["normal", "expert", "overall"],
                    columns=["A", "T", "B"],
                )
                .rename_axis("group")
                .reset_index()
            ),
            "实际 Schema 完整保存于 data_audit.json。overall_pref/evaluator "
            "位于 normal_worker_annotations/expert_worker_annotations 的元素中。"
            "匿名 evaluator 与原始五档在本地 Parquet 保留；强度未进入模型。",
            "ODC-BY；来源与引用见 prior_work.md、CITATION.bib。",
            "分区计数：",
            table(
                pd.DataFrame(
                    [
                        {"split": name, "comparisons": item["n"], "prompts": item["prompts"]}
                        for name, item in json.loads(
                            (OUT / "real_references.json").read_text()
                        ).items()
                    ]
                )
            ),
        ]
    text += [
        "## 合成验证：正确设定与错误设定",
        f"S1/S2 各固定 {int(results[results.dataset == 'S1'].n.iloc[0])} 比较，"
        "生成种子与已知边缘概率见 frozen_protocol.json。"
        "下面既保留相对有限完整标签参考的误差，也保留相对生成参数的误差。"
        "跨五次不改变完整目标，只改变筛选/审计，不能称为五次独立人群模拟。",
    ]
    for scenario in ["S1", "S2"]:
        subset = results[results.dataset == scenario]
        text += [
            f"### {scenario}：有限参考 TV（均值 ± 标准差）",
            table(mean_sd(subset, "tv")),
            f"### {scenario}：相对生成参数 TV（均值）",
            table(
                subset.pivot_table(
                    index=["mechanism", "audit_rate"],
                    columns="method",
                    values="truth_tv",
                    dropna=False,
                ).reset_index()
            ),
        ]
    s2 = results[
        (results.dataset == "S2") & (results.mechanism == "M2") & (results.audit_rate == 0.2)
    ]
    b3s2 = s2[s2.method == "B3"].tv.mean()
    b1s2 = s2[s2.method == "B1"].tv.mean()
    text += [
        "共享潜在因素的 S2 中，M2 是已观察到的模型失配失败例。"
        f"20% 审计时 B3 平均 TV={b3s2:.5f}，B1={b1s2:.5f}。"
        "该方向由预设潜变量场景产生；不能推广所有相关标注机制。"
        "S2 的 M1 未表现同等程度失败，也不能反推独立性无关紧要。"
    ]
    if len(real):
        selection = impacts[(impacts.dataset == "MultiPref") & (impacts.split == "test")]
        text += [
            "## 真实 Pilot：锁定测试目标的筛选影响",
            f"以下均针对 {int(real.n.iloc[0])} 比较测试分区，"
            "完整参考分布见 real_references.json。"
            "B0 不读取完整审计标签，所以其误差跨审计比例不变。",
            table(
                selection.groupby("mechanism")[
                    [
                        "tv",
                        "normal_tv",
                        "expert_tv",
                        "annotation_retention",
                        "sample_retention",
                        "deletion_A",
                        "deletion_T",
                        "deletion_B",
                    ]
                ]
                .mean()
                .reset_index()
            ),
            "M0 理论上与标签无关，其有限目标误差是随机删除波动。"
            "M1 的 A/T/B 序数距离赋予中间 Tie 特殊几何位置；M2 至少三同类"
            "天然偏好高一致性组合。删除方向是本轮人为核与输入分布共同作用，"
            "不是实际 HelpSteer3 或 RLHF 的因果影响。",
            "## 真实 Pilot：四种方法",
            "B1/B2 在当前完整四槽位审计与等权目标下代数相同。零审计的两者不可用。"
            "未按 category 分层，未通过稀疏类别回填或真值调参。",
            "### 总体 TV：均值 ± 标准差",
            table(mean_sd(real, "tv")),
            "### 普通组 TV：均值 ± 标准差",
            table(mean_sd(real, "normal_tv")),
            "### 专家组 TV：均值 ± 标准差",
            table(mean_sd(real, "expert_tv")),
            "### 总体支持度绝对误差：均值",
            table(
                real.pivot_table(
                    index=["mechanism", "audit_rate"],
                    columns="method",
                    values="score_error",
                    dropna=False,
                ).reset_index()
            ),
            "### 群体支持度差误差：均值",
            table(
                real.pivot_table(
                    index=["mechanism", "audit_rate"],
                    columns="method",
                    values="group_gap_error",
                    dropna=False,
                ).reset_index()
            ),
            "组间 TV 差误差及每个概率估计另见完整 CSV。",
        ]
        paired = real.pivot(
            index=["mechanism", "audit_rate", "seed"], columns="method", values="tv"
        ).reset_index()
        paired["B3_minus_B1"] = paired.B3 - paired.B1
        paired["B3_minus_B0"] = paired.B3 - paired.B0
        text += [
            "### 配对差：负数表示 B3 误差更低（描述性）",
            table(
                paired.groupby(["mechanism", "audit_rate"])[["B3_minus_B1", "B3_minus_B0"]]
                .agg(["mean", "std"])
                .pipe(lambda x: x.set_axis(["_".join(c) for c in x.columns], axis=1))
                .reset_index()
            ),
        ]
        agreement = pd.DataFrame(json.loads((OUT / "real_agreement_diagnostics.json").read_text()))
        text += [
            "## 模型假设的经验警示",
            table(agreement),
            f"在这 {audit['usable']} 比较中，槽位对实际同类率比独立边缘乘积高出 "
            f"{agreement.excess_agreement.min():.5f}–{agreement.excess_agreement.max():.5f}。"
            "这是组别独立工作模型的描述性失配警示，可能来自问题/回答异质性、"
            "标注者等多种来源，未归因给单一原因、未作显著性检验。"
            "真实 Pilot B3 的误差较低并不能证明该模型正确。",
        ]
    if len(real):
        m2 = real[(real.mechanism == "M2") & (real.audit_rate == 0.2)]
        gap0 = m2[m2.method == "B0"].group_gap_error.mean()
        gap3 = m2[m2.method == "B3"].group_gap_error.mean()
        text += [
            "### 总体改善不能替代群体差异评估",
            f"真实 M2、20% 审计下，群体支持度差误差 B0={gap0:.5f}，"
            f"B3={gap3:.5f}。该指标必须与总体 TV 分开判断。",
        ]
    fitted = results[results.method == "B3"]
    failures = fitted[~fitted.success]
    text += [
        "## 时间、优化与可识别性诊断",
        table(
            results.groupby(["dataset", "split", "method"])
            .agg(
                runs=("method", "size"),
                success=("success", "sum"),
                mean_seconds=("seconds", "mean"),
                max_seconds=("seconds", "max"),
            )
            .reset_index()
        ),
        f"B3 共 {len(fitted)} 次拟合；结果非收敛 {len(failures)} 次；"
        f"失败起点合计 {int(fitted.failed_starts.sum())}；"
        f"触边 {int(fitted.boundary.sum())}；局部 Jacobian 秩最小 "
        f"{int(fitted.local_rank.min())}；最小奇异值最小 {fitted.min_singular.min():.7f}。"
        "数值秩仅作局部结构诊断，不代表全局识别证明、良好置信区间或无模型恢复。",
        table(failures[["dataset", "split", "mechanism", "audit_rate", "seed", "message"]])
        if len(failures)
        else "本次未记录非收敛结果。",
        "失败拟合的有限数值仍在原表中，未删除或替换成最好成功运行。"
        "零审计 B1/B2 不可用单独标记，不是数值优化失败。"
        "执行时间仅为估计器时间，数据下载、筛选和文件 I/O 不包含在内。",
        "## 负面结果与适用限制",
        "1. S2/M2 存在结构性残余误差；增加到 20% 审计后仍可能不如仅审计。",
        "2. 审计量增加在某个固定目标/五个种子上不保证误差逐点单调下降。"
        "真实 M1 中增加审计不必然改善 B3；小规模试验不能把这种波动解释为普遍规律。",
        "3. 两组总体分布相近不代表逐比较偏好相同；总体 TV 改善不保证组别/差异估计改善。"
        "本报告单列两组误差，原表保留全部重复与组别指标。",
        "4. S1 的良好恢复以组别独立模型和已知筛选核为前提；零审计表现不代表任意缺失可恢复。",
        "5. 真实样本是有界顺序前缀；不同类别、模型配对与 evaluator 的可推广性未验证。"
        "无下游 LLM 训练、性能、因果或全球人群结论。",
        "6. 本轮未证明全局参数可识别，未做统计显著性或覆盖率实验。"
        "不以局部满秩、近似多起点一致或低误差冒充证明。",
        "训练/验证所有对应结果见 summary.csv；不只展示最优种子。方法与优化设置未用测试集调参。",
    ]
    (ROOT / "docs/phase1_results.md").write_text("\n\n".join(text) + "\n")
    evidence = [
        "# 留给研究负责人的第一阶段客观决策证据",
        "本文由实际输出生成，不决定后续课题或投入期限。",
        tests,
        f"已执行 {len(results)} 个方法运行（含不可用的零审计基线），"
        f"{len(fitted)} 个 B3 拟合；优化失败 {len(failures)}。",
        "## 筛选严重程度（真实测试目标）",
    ]
    if len(real):
        evidence += [
            table(
                selection.groupby("mechanism")[["tv", "annotation_retention", "sample_retention"]]
                .mean()
                .reset_index()
            ),
            "## 相同预算下恢复程度：真实测试、20%审计",
            table(
                real[real.audit_rate == 0.2]
                .groupby(["mechanism", "method"])[
                    ["tv", "normal_tv", "expert_tv", "group_gap_error"]
                ]
                .mean()
                .reset_index()
            ),
        ]
    evidence += [
        "## 已观察到的失败边界",
        f"S2/M2、20%审计 B3 TV={b3s2:.5f}，仅审计 TV={b1s2:.5f}。"
        "条件独立错误时，选择感知似然未必胜过代表性审计。",
        "真实总体误差、组别误差和群体差异误差可能不一致，需按预定目标评判。"
        "本轮没有大规模种子或跨数据集验证，已有五次只覆盖删失与审计随机化。",
        "## 尚无证据回答的判断",
        "- 是否需要放宽独立性模型，以及收益能否超过新增复杂度，尚未建立。",
        "- 真实核未知或审计非随机时的恢复能力未验证。",
        "- 81 状态工作模型的局部数值秩不能替代全局识别论证。",
        "- 当前实现是经典似然和审计统计，不构成原创方法贡献认定。",
        "- 是否继续投入由负责人综合上述效果、失败边界与研究价值判断；本轮不自动进入第二阶段。",
    ]
    (ROOT / "docs/phase1_decision_inputs.md").write_text("\n\n".join(evidence) + "\n")


if __name__ == "__main__":
    main()
