# Phase 2 失败与限制分析（自动生成）

## S2 / M2：原先的失配问题，本轮重新验证

| method | ok_n | tv_mean | tv_sd | mse_mean | normal_tv_mean | expert_tv_mean | group_gap_error_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | 50 | 0.13055 | 0.00000 | 0.00876 | 0.10844 | 0.15566 | 0.07450 |
| B1 | 50 | 0.02369 | 0.01305 | 0.00040 | 0.02957 | 0.02624 | 0.01456 |
| B2 | 50 | 0.02369 | 0.01305 | 0.00040 | 0.02957 | 0.02624 | 0.01456 |
| B3 | 50 | 0.04001 | 0.00234 | 0.00089 | 0.03834 | 0.05092 | 0.02883 |
| B4 | 50 | 0.01867 | 0.01047 | 0.00023 | 0.02279 | 0.02107 | 0.01497 |
| B4-M2 | 50 | 0.00831 | 0.00523 | 0.00005 | 0.01561 | 0.01328 | 0.01260 |
| B5a | 50 | 0.00836 | 0.00521 | 0.00005 | 0.01610 | 0.01318 | 0.01239 |
| B5b | 50 | 0.00995 | 0.00594 | 0.00007 | 0.01739 | 0.01339 | 0.01228 |

## 真实测试的群体差异问题

| audit_rate | method | normal_tv_mean | expert_tv_mean | group_gap_error_mean | group_gap_signed_bias |
| --- | --- | --- | --- | --- | --- |
| 0.05000 | B0 | 0.07462 | 0.07582 | 0.00125 | -0.00125 |
| 0.05000 | B1 | 0.09587 | 0.08458 | 0.07417 | 0.00792 |
| 0.05000 | B2 | 0.09587 | 0.08458 | 0.07417 | 0.00792 |
| 0.05000 | B3 | 0.01163 | 0.01217 | 0.00539 | -0.00157 |
| 0.05000 | B4 | 0.07483 | 0.06562 | 0.05400 | 0.00167 |
| 0.05000 | B4-M2 | 0.06295 | 0.04828 | 0.05458 | 0.00125 |
| 0.05000 | B5a | 0.06203 | 0.05630 | 0.06028 | 0.00738 |
| 0.05000 | B5b | 0.06421 | 0.05490 | 0.05634 | 0.00594 |
| 0.20000 | B0 | 0.07462 | 0.07582 | 0.00125 | -0.00125 |
| 0.20000 | B1 | 0.04629 | 0.03658 | 0.02567 | 0.00437 |
| 0.20000 | B2 | 0.04629 | 0.03658 | 0.02567 | 0.00437 |
| 0.20000 | B3 | 0.01110 | 0.01145 | 0.00711 | -0.00113 |
| 0.20000 | B4 | 0.03046 | 0.02867 | 0.02146 | 0.00062 |
| 0.20000 | B4-M2 | 0.02503 | 0.02256 | 0.02140 | 0.00003 |
| 0.20000 | B5a | 0.02509 | 0.02280 | 0.01993 | 0.00021 |
| 0.20000 | B5b | 0.02578 | 0.02315 | 0.02182 | 0.00052 |

## 低审计、模型失配和概率合法性

| experiment | dataset | mechanism | method | tv_mean | invalid_raw_rate_among_ok | unavailable_rate | numeric_failure_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_audit | S1 | M0 | B4 | 0.02074 | 0.88000 | 0.00000 | 0.00000 |
| fixed_audit | S1 | M0 | B4-M2 | — | — | 0.00000 | 0.00000 |
| fixed_audit | S1 | M0 | B5a | 0.02080 | 0.00000 | 0.00000 | 0.00000 |
| fixed_audit | S1 | M0 | B5b | 0.02084 | 0.00000 | 0.00000 | 0.00000 |
| fixed_audit | S1 | M1 | B4 | 0.02191 | 1.00000 | 0.00000 | 0.00000 |
| fixed_audit | S1 | M1 | B4-M2 | — | — | 0.00000 | 0.00000 |
| fixed_audit | S1 | M1 | B5a | 0.02038 | 0.00000 | 0.00000 | 0.00000 |
| fixed_audit | S1 | M1 | B5b | 0.01583 | 0.00000 | 0.00000 | 0.00000 |
| fixed_audit | S1 | M2 | B4 | 0.03939 | 1.00000 | 0.00000 | 0.00000 |
| fixed_audit | S1 | M2 | B4-M2 | 0.02497 | 0.00000 | 0.00000 | 0.00000 |
| fixed_audit | S1 | M2 | B5a | 0.02495 | 0.00000 | 0.00000 | 0.00000 |
| fixed_audit | S1 | M2 | B5b | 0.02466 | 0.00000 | 0.00000 | 0.00000 |
| fixed_audit | S2 | M0 | B4 | 0.02132 | 0.88000 | 0.00000 | 0.00000 |
| fixed_audit | S2 | M0 | B4-M2 | — | — | 0.00000 | 0.00000 |
| fixed_audit | S2 | M0 | B5a | 0.02015 | 0.00000 | 0.00000 | 0.00000 |
| fixed_audit | S2 | M0 | B5b | 0.02092 | 0.00000 | 0.00000 | 0.02000 |
| fixed_audit | S2 | M1 | B4 | 0.01962 | 1.00000 | 0.00000 | 0.00000 |
| fixed_audit | S2 | M1 | B4-M2 | — | — | 0.00000 | 0.00000 |
| fixed_audit | S2 | M1 | B5a | 0.01953 | 0.00000 | 0.00000 | 0.00000 |
| fixed_audit | S2 | M1 | B5b | 0.01925 | 0.00000 | 0.00000 | 0.00000 |
| fixed_audit | S2 | M2 | B4 | 0.03752 | 1.00000 | 0.00000 | 0.00000 |
| fixed_audit | S2 | M2 | B4-M2 | 0.01947 | 0.00000 | 0.00000 | 0.00000 |
| fixed_audit | S2 | M2 | B5a | 0.02002 | 0.00000 | 0.00000 | 0.00000 |
| fixed_audit | S2 | M2 | B5b | 0.02240 | 0.00000 | 0.00000 | 0.00000 |
| fixed_audit | S3 | M0 | B4 | 0.01783 | 0.88000 | 0.00000 | 0.00000 |
| fixed_audit | S3 | M0 | B4-M2 | — | — | 0.00000 | 0.00000 |
| fixed_audit | S3 | M0 | B5a | 0.01839 | 0.00000 | 0.00000 | 0.00000 |
| fixed_audit | S3 | M0 | B5b | 0.01782 | 0.00000 | 0.00000 | 0.00000 |
| fixed_audit | S3 | M1 | B4 | 0.01923 | 1.00000 | 0.00000 | 0.00000 |
| fixed_audit | S3 | M1 | B4-M2 | — | — | 0.00000 | 0.00000 |
| fixed_audit | S3 | M1 | B5a | 0.02036 | 0.00000 | 0.00000 | 0.00000 |
| fixed_audit | S3 | M1 | B5b | 0.02069 | 0.00000 | 0.00000 | 0.00000 |
| fixed_audit | S3 | M2 | B4 | 0.03767 | 1.00000 | 0.00000 | 0.00000 |
| fixed_audit | S3 | M2 | B4-M2 | 0.01751 | 0.00000 | 0.00000 | 0.00000 |
| fixed_audit | S3 | M2 | B5a | 0.01823 | 0.00000 | 0.00000 | 0.00000 |
| fixed_audit | S3 | M2 | B5b | 0.01928 | 0.00000 | 0.00000 | 0.00000 |
| real_audit | MultiPref | M0 | B4 | 0.03283 | 1.00000 | 0.00000 | 0.00000 |
| real_audit | MultiPref | M0 | B4-M2 | — | — | 0.00000 | 0.00000 |
| real_audit | MultiPref | M0 | B5a | 0.03415 | 0.00000 | 0.00000 | 0.00000 |
| real_audit | MultiPref | M0 | B5b | 0.03426 | 0.00000 | 0.00000 | 0.00000 |
| real_audit | MultiPref | M1 | B4 | 0.03125 | 1.00000 | 0.00000 | 0.00000 |
| real_audit | MultiPref | M1 | B4-M2 | — | — | 0.00000 | 0.00000 |
| real_audit | MultiPref | M1 | B5a | 0.03385 | 0.00000 | 0.00000 | 0.00000 |
| real_audit | MultiPref | M1 | B5b | 0.03115 | 0.00000 | 0.00000 | 0.03333 |
| real_audit | MultiPref | M2 | B4 | 0.04890 | 1.00000 | 0.00000 | 0.00000 |
| real_audit | MultiPref | M2 | B4-M2 | 0.03031 | 0.00000 | 0.00000 | 0.00000 |
| real_audit | MultiPref | M2 | B5a | 0.03283 | 0.00000 | 0.00000 | 0.00000 |
| real_audit | MultiPref | M2 | B5b | 0.03234 | 0.00000 | 0.00000 | 0.00000 |

B4 并非方差普遍支配 Audit-Only；原有完整 z 在总体中恒定而缺失残差随样本变化时，Audit-Only 可有零方差而 B4 非零，已加反例测试。

普通 B4 在 M2 下对删除层数量做 HT 扩张，有限审计可能高估/低估总概率质量。B4-M2 使用实际层占比，但删除层无审计时不可估计；其设计期望声明有条件。

B5 预测质量差可能增加方差；alpha=.5 与系数=1 固定，无按真值优化。B5b 折训练不收敛或触边即记录失败，不用 B3 总体点估计替代后验，不回退隐藏真值。

两组和总体的非负/和为1均检查，原始预测与误差全部保存。未做投影，因而也没有以投影后的结果宣称设计无偏。

## 数值失败原始记录

| experiment | dataset | split | population_seed | audit_seed | mechanism | audit_rate | method | message |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_audit | S2 | full | 20261022 | 11047 | M0 | 0.05000 | B5b | B3 training fold failed or reached logit boundary |
| fixed_audit | S3 | full | 20261022 | 11006 | M0 | 0.10000 | B5b | B3 training fold failed or reached logit boundary |
| fixed_audit | S3 | full | 20261022 | 11016 | M2 | 0.10000 | B3 | ABNORMAL:  |
| fixed_audit | S3 | full | 20261022 | 11018 | M1 | 0.20000 | B5b | B3 training fold failed or reached logit boundary |
| regenerated | S3 | full | 21008 | 7007 | M1 | 0.05000 | B5b | B3 training fold failed or reached logit boundary |
| real_audit | MultiPref | train | 0 | 31002 | M0 | 0.05000 | B3 | ABNORMAL:  |
| real_audit | MultiPref | train | 0 | 31004 | M0 | 0.10000 | B5b | B3 training fold failed or reached logit boundary |
| real_audit | MultiPref | train | 0 | 31008 | M1 | 0.10000 | B5b | B3 training fold failed or reached logit boundary |
| real_audit | MultiPref | train | 0 | 31014 | M1 | 0.20000 | B5b | B3 training fold failed or reached logit boundary |
| real_audit | MultiPref | train | 0 | 31019 | M0 | 0.20000 | B5b | B3 training fold failed or reached logit boundary |
| real_audit | MultiPref | train | 0 | 31022 | M1 | 0.10000 | B5b | B3 training fold failed or reached logit boundary |
| real_audit | MultiPref | validation | 0 | 31004 | M2 | 0.10000 | B3 | ABNORMAL:  |
| real_audit | MultiPref | validation | 0 | 31024 | M2 | 0.05000 | B5b | B3 training fold failed or reached logit boundary |
| real_audit | MultiPref | test | 0 | 31005 | M2 | 0.20000 | B5b | B3 training fold failed or reached logit boundary |
| real_audit | MultiPref | test | 0 | 31026 | M1 | 0.05000 | B5b | B3 training fold failed or reached logit boundary |

超过表中展示行数的完整失败记录仍在 results.csv；头部展示不代表删去其他运行。
