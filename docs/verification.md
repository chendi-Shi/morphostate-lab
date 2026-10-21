# 软件验证记录

更新日期：2026-10-21。本文区分实现、软件测试和真实生物实验。测试数量与 CI 通过状态应以当前 commit 的实际日志为准，不沿用旧版本结果冒充本次完整验证。

## 覆盖范围

本次本地全量测试 **202 passed，0 skipped**，用时 17.50 秒；`ruff check src tests` 通过。`weak_demo` 已完成包含 **21 个模型输出**的合成闭环：11 个弱监督目标、组成 NB 和 9 个非 oracle 常规基线。`core_only` 的选择目标为 core 子集，不能与其余模型全 RNA 指标混成主排名。

另已实际完成：`weak_suite` 的 3 折 × 2 种子 × 2 配置，共 **12 个训练任务**及患者汇总；完整 `demo.yaml` baseline 运行（含新增常规对照）；两种 null 可行性反例生成；完整反例的 LP 原始／对偶、membership、joint/box 与 KL 修正诊断。`stress_suite` 的 18 个任务清单已检查生成，未把它的长训练说成已完成。真实第三方作者网络未运行。

测试代码覆盖以下检查，发布时运行全量 `python -m pytest -q` 与 `python -m ruff check src tests`：

- 数据数值、唯一 ID、Xenium 导入与严格行列对齐、跨患者划分及 oracle 输入阻断。
- NB 数值稳定性、冻结基线、验证集模型选择、检查点重载、无测试 RNA 推理。
- 强基线可复现性、患者交叉拟合 nuisance 排除、加权均值 exposure 处理与类型 oracle 隔离。
- 分子守恒、几何/背景预算、不可行约束、LP 精确解、边际范围和线性对比范围。
- 原始分子构图、固定 owner 与 prepared counts 一致性、坐标/来源声明及图持久化。
- 联合/区间/固定/乐观/绝对准似然训练、core 子集与聚合背景规则、ensemble、无 RNA 弱监督推理。
- 供者 folds、分组隔离、配置变体、任务清单、恢复与失败记录、先平均种子再按患者比较。
- 地标变换、裁图边界、图像协变量训练来源、独立状态评分与对比界。
- pinned 作者 checkout 的子进程执行、日志、失败审计、权重哈希、GHIST 作者 CSV 和 Pixel2Gene pickle 网格转换；normalized-rate 协议的外部评价闭环。
- 可行 null 导致状态抹除的构造反例、原始／对偶间隙与可行性、完整／投影 membership、近似可行性 KL 误差修正、严格保留 edge cost 的质量聚合。
- 选择性风险的患者身份与单位契约、全体对比覆盖、相同接受集合上的固定基线、真值改变不影响接受规则。

外部执行测试使用临时脚本生成符合作者格式的 fixture，验证执行器与格式契约，**不是训练过作者神经网络**。LP/图测试验证声明约束内的计算正确性，不能验证真实候选集合是否包含生物真值。

## 可复核命令

```bash
python -m pytest -q
python -m ruff check src tests
morphostate demo --output data/weak_demo --donors 6 --cells-per-donor 12 --genes 8 --features 4
morphostate synthetic-graphs --data data/weak_demo --output data/weak_demo_graphs
morphostate weak-run --config configs/weak_demo.yaml
morphostate suite --config configs/weak_suite.yaml --plan-only
morphostate suite --config configs/weak_suite.yaml --resume
morphostate summarize --suite runs/weak_suite --reference fixed
```

输出目录要求为空或满足明确的 suite 恢复规则。重复验证使用新路径；合成数值只能检查流程，不能证明选题创新。

## CI

[GitHub Actions](https://github.com/chendi-Shi/morphostate-lab/actions) 是版本化验证入口。请查看目标 commit 的 workflow、实际日志和 artifacts。工作流是否加入弱监督或 suite smoke，以该次 YAML 和日志为准；本文不会把待运行 CI 标为已通过。

本地开发使用 Windows、Python 3.12 和 CPU PyTorch 环境；准确版本以运行记录/环境 artifact 为准。不要把本地 CPU 测试推广成 CUDA、所有操作系统或全部依赖组合均已验证。

## 仍需实测的科学与工程范围

- 真实患者配准 QA、核/膜边界与候选归属合理性、预算的独立校准。
- 病理 encoder 授权权重、完整 WSI/OpenSlide 与 CUDA 执行、大规模 LP/内存/显存耗时。
- GHIST、Pixel2Gene 作者网络在锁定队列上的训练和公平调参；未定位到发布入口的工作不虚报复现。
- 真实患者主结果、预注册消融、多数据集外部验证、独立蛋白/IF 状态证据。
- 超越已有聚合/区间/悲观学习的理论贡献、状态识别保证和论文录用判断。

核心弱监督方法已经写成可执行代码；上述内容需要数据、运行和研究证明，不能由单元测试替代。
