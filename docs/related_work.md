# 相关工作与当前研究边界

核查日期：2026-10-21。以下是与本项目最直接重叠的工作，不是穷尽性文献综述。“未找到完全相同的实验”不能作为新颖性证明；代码框架的条件 NB 也不应被表述为已经成立的新方法。

| 工作 | 与本项目的已知重叠 | 对实验设计的约束 |
| --- | --- | --- |
| **ImageCCA**：Ash 等，*Joint analysis of expression levels and histological images identifies genes associated with tissue morphology*，Nature Communications，2021。[原论文](https://doi.org/10.1038/s41467-021-21727-x) | 已使用图像表示与表达的稀疏 CCA 学习共享成分 | “学习形态—RNA 共享子空间”本身不是新贡献；需比较线性共享表示 |
| **Probabilistic Partial CCA**：Mukuta 与 Harada，ICML，2014。[正式论文](https://proceedings.mlr.press/v32/mukuta14.html) | 已研究移除第三变量影响后的相关子空间及概率估计 | “控制组成再做 CCA”有成熟方法基础；有限样本稳定性和维数估计也不是新问题 |
| **Path2Space**：Shulman 等，Cell，2026；此处使用作者公布的混杂分析实现作为具体依据。[Table S2 分析代码](https://github.com/eldadshulman/path2space-extended/blob/main/confounder_analysis/README.md) | 已分别控制染色强度、癌细胞比例和 RNA 总量，分析预测与测量的部分相关 | 不应声称现有研究从未考虑组成或技术因素；逐 spot 控制不等于完成逐细胞连续状态验证 |
| **SLIDE-EX**：*Deep learning inference of cell type-specific gene expression from breast tumor histopathology*，npj Precision Oncology，2026。[原论文](https://doi.org/10.1038/s41698-026-01419-9) | 已预测细胞类型特异表达，使用 bulk 去卷积标签，并提供独立单细胞表达验证 | 不能把“类型内部表达”单独作为新题目；需明确患者均值、空间位置和逐细胞状态是不同评价单位 |
| **Stem**：*Diffusion Generative Modeling for Spatially Resolved Gene Expression Inference from Histology Images*，ICLR，2025。[论文](https://arxiv.org/abs/2501.15598) | 已将形态到表达的一对多映射纳入生成建模 | 概率输出或承认形态不能唯一决定 RNA 均已有前作；NB 似然属于基线设计 |
| **Prediction residuals in spatial transcriptomics reveal microenvironmental heterogeneity in cancer**，2026。[PubMed 论文记录与摘要](https://pubmed.ncbi.nlm.nih.gov/42887393/) | 已利用预测残差分析微环境异质性，并涉及组成调整 | “研究预测残差的生物学意义”也非空白；本项目要区分残差的关联分析与残差能否由图像预测 |

## 本仓库目前在做什么

当前代码实现一个可检查的对照实验：图像或协变量 ridge、残差 ridge、协变量 NB，以及冻结协变量基线后的低秩图像增量。数据和模型选择按供者划分，并区分真实图像输入与 RNA-oracle 输入。

它尚未实现独立的污染生成机制、理论可识别性分析、稳健共享子空间估计或完整 Partial CCA 对照。低秩隐变量存在旋转等不确定性，不能仅凭某个基因载荷较高就宣称发现了患者特异分子程序。

## 值得检验、但未得到证明的假设

在严格留患者和类型/组成控制后，某些现有表达预测器可能主要恢复类型模板，而没有恢复细胞内部的连续状态。需要用相同输入、相同划分和相同目标检验这一点，不能从整体 PCC 推断成立或不成立。

如果简单残差 ridge 或 Partial CCA 已能解释所观察的增益，复杂模型应被视为缺少必要性。如果状态增益只出现在 RNA-oracle 分析、某种去污染输出或同患者拆分中，也不能称为可部署的跨患者状态恢复。

潜在的方法研究应建立在可重复失败上，例如：协变量由图像估计且存在误差时如何估计真实增量，或不同患者与测量条件下如何识别稳定共享方向。当前仓库为这些问题建立起点，不预先声称已解决。
