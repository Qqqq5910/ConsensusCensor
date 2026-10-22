# ConsensusCensor

项目定位：**共识筛选与人类偏好数据稳健统计估计的开放研究产物。**

**Engineering Artifact: READY**

**Original Research Contribution: NOT ESTABLISHED**

项目已正式收口，不进入 Phase 3，也不继续六个月的原创算法研究。已有代码实现、比较经典似然、随机审计、HT 残差、后分层与交叉拟合辅助方法，不声称发明 PPI、AIPW、GREG 或新的无偏估计理论。

- [英文最终研究报告](docs/final_research_report.md)
- [中文最终总结](docs/final_project_summary_CN.md)
- [复现指南](docs/reproducibility_guide.md)
- [最终变更记录](docs/final_change_log.md)

安装后一个命令完成全部测试、有限总体穷举及小规模合成比较：

```bash
python -m pip install -e ".[dev,data]"
python scripts/reproduce_minimal.py
```

Python 3.11+，普通 CPU 即可；无需真实数据下载、GPU、付费 API 或服务器。输出 experiments/minimal，重复运行请用 --output 选择新的空目录。完整大型实验的原始结果和冻结协议全部保留，默认不重跑。

主要发现是模型精度和设计稳健性之间存在取舍，简单后分层在部分场景已很强。3 次 B3、12 次 B5b 数值失败全部保留；历史成功条件下均值不能当作无条件算法风险。原始非法概率向量用半 L1 损失，不静默裁剪；固定总体审计变化与跨总体变化分开解释。

软件采用 MIT；MultiPref 数据及其派生输出保留 ODC-BY 来源、署名与适用义务，见 DATA_LICENSE.md。真实原始标注和缓存不提交。CI 已配置，尚未在 GitHub 实际运行；本轮仅本地提交，不创建远端、不推送、不发布 Release。
