# 指定既有研究与重合边界

2026-10-22 核对指定六个公开页面（五项研究及 MultiPref 数据页），记录于 sources.json。核验主要覆盖官方页面元数据、摘要和数据卡，未声称逐行审计论文全部证明或作者代码。

| 指定研究 | 已确认的先行内容与本项目关系 |
|---|---|
| Fleisig et al., Balancing Quality and Variation: Spam Filtering Distorts Data Label Distributions, 2025 | 标注过滤与标签分布/多样性变化已有研究。过滤偏差不是本项目首次发现。https://aclanthology.org/2025.nlperspectives-1.5/ |
| Zhang et al., Diverging Preferences: When do Annotators Disagree and do Models Know?, ICML 2025 | 研究偏好分歧来源、偏好模型局限；不能将分歧全部当成噪声。https://proceedings.mlr.press/v267/zhang25bx.html |
| Ghafouri et al., Position: RLHF May Not Reflect Genuine Preferences, ICML 2026 | 讨论测量效度、回应是否等于真实偏好；本项目有限标注分布不是人类真实偏好真值。https://proceedings.mlr.press/v306/ghafouri26a.html |
| Miranda et al., Hybrid Preferences: Learning to Route Instances for Human vs. AI Feedback, ACL 2025 | MultiPref 来源，采用其公开标注，不声称新建数据集。https://aclanthology.org/2025.acl-long.355/ |
| NVIDIA HelpSteer3-Preference, 2025 | 数据页面已核对。M1/M2 是指定四槽位研究模拟，未审计出与其真实筛选规则逐项等价，不能声称精确复现。https://huggingface.co/datasets/nvidia/HelpSteer3 |

B3 是已知选择核下对潜在缺失标签边缘化的经典观测数据最大似然，不起新算法名称。精确 81 状态枚举、随机审计估计与分层估计也不是原创统计思想。本轮不进行广泛新颖性调查，未找到重合不等于证明原创，实际实验不构成新方法贡献认定。

MultiPref: https://huggingface.co/datasets/allenai/multipref，ODC-BY；使用需署名。数据卡要求引用 Miranda 等 Hybrid Preferences（原 arXiv:2410.19133，2024；正式 ACL 2025）。本包 README、本文与 CITATION.bib 已标明来源；原始及转换真实标签不提交 Git。
