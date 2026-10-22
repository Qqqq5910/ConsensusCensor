# Phase 2 既有方法关系与来源核对

本轮不进行新课题搜索或原创性包装。以下均为既有思想：Horvitz–Thompson、后分层、广义回归/Model-Assisted Estimation、AIPW、Prediction-Powered Inference 与 Cross-Fitting。

| 本轮实现 | 已有思想 | 本轮边界 |
|---|---|---|
| B4 | HT 作用于缺失残差；广义差分估计 | 已知概率、正性下的有限设计期望，不是新算法 |
| B4-M2 | 已知层权重的后分层估计 | 删除层无审计则不可估计，不偷偷回填 |
| B5a/B5b | 预测加逆概率残差校正；与 PPI、AIPW 有共同代数结构 | 不将因果 AIPW 定理或 i.i.d. PPI 覆盖率定理直接搬到有限 SRS |
| 两折预测 | Cross-Fitting | 固定总量审计需条件折概率与有效性条件，不因“交叉”就自动无偏 |
| B5b 后验 | 原 B3 的经典潜变量条件期望 | 模型错配影响效率，且需正确使用选择核 |

## 已核对的指定代码

通过用户选择的 GitHub 插件读取 [aangelopoulos/ppi_py README](https://github.com/aangelopoulos/ppi_py/blob/main/README.md)，内容 SHA `66941940269d071970502e244f8359c162087137`；读取 [ppi_py/ppi.py](https://github.com/aangelopoulos/ppi_py/blob/main/ppi_py/ppi.py) 第1–160行，内容 SHA `9d61900f7e1d0868f12c786e57be154b2b5d51c7`。核对 ppi_mean_pointestimate 的固定 lambda 预测加残差结构，以及 README 指向 Cross-Prediction-Powered Inference 的关系。未安装 ppi_py、未复制其整套框架、未直接调用其 CI；本轮有限总体全局 SRS 与该实现的常见独立标注/未标注设定不同。

## 论文与核验范围

1. Horvitz, D. G. & Thompson, D. J. (1952). A Generalization of Sampling Without Replacement from a Finite Universe. JASA 47(260):663–685. [DOI](https://doi.org/10.1080/01621459.1952.10483446)，[大学存档的原论文](https://www.stat.cmu.edu/~brian/905-2008/papers/Horvitz-Thompson-1952-jasa.pdf)。出版站点直接访问403，大学存档可读；本轮核对书目信息，B4 的期望与 SRS 方差另在本项目数学文档显式推导。
2. Angelopoulos, A. N., Bates, S., Fannjiang, C., Jordan, M. I., & Zrnic, T. (2023). [Prediction-Powered Inference](https://arxiv.org/abs/2301.09633)，Science 382:669–674。核对官方 arXiv 页面、摘要、作者与指定代码。
3. Zrnic, T. & Candès, E. J. [Cross-Prediction-Powered Inference](https://arxiv.org/abs/2309.16598)，2023预印本，2024修订。核对官方页面与训练/去偏的核心设定。未宣称本轮逐项复现其渐近置信区间证明。
4. Särndal、Swensson、Wretman 的 Model Assisted Survey Sampling (1992) 是广义差分/GREG 的经典背景。本轮未取得该书完整内容，未将其未核验章节当证明来源。尝试访问 Statistics Canada 的相关 model-assisted 论文页面遇到超时/暂不可用，故不依赖该页面支撑额外定理。

本轮提供的是现有方法在既定四槽位删失问题中的实现与受控比较；不把模型辅助、PPI 或逆概率残差校正称为本项目新算法。B4 精确设计期望、B5 的条件设计期望与不适用条件由 phase2_mathematics.md 直接列出，经验误差不替代理论。
