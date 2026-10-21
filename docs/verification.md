# 软件验证记录

日期：2026-10-21。这里记录的是软件正确性和可执行性检查，不是生物学性能。

## 本地结果

- Windows / Python 3.12 / PyTorch 2.6.0+cpu。
- `pytest -q`：**77 passed**。
- `ruff check src tests`：通过。
- CLI `demo → validate → run`：通过；960个合成细胞、12名模拟供者、32个模拟基因。
- 训练/验证/测试细胞数为640/160/160，供者互不重叠；两个候选rank由验证NLL选择。
- 完整测试包含删除计数文件后的检查点推理、协变量ID重排、外部预测单位/顺序对齐、oracle阻断、常量预测PCC、部分回归及分子守恒原型。

早期完整测试发现并修复了两个问题：float32均值给常量预测制造微小方差；残差回归没有同时对图像特征做协变量残差化。保留了回归测试。

## CI

[GitHub Actions](https://github.com/chendi-Shi/morphostate-lab/actions) 在每次提交运行安装、lint、pytest和完整合成示例。成功运行时 `synthetic-smoke-test` artifact 含报告、检查点和 `environment.txt`，后者记录该次实际依赖版本。请按具体commit查看状态，本文不把正在执行的CI写为已通过。

## 尚未验证

尚未运行真实患者训练、H&E配准、可选timm/病理权重提取、GPU执行、第三方SOTA复现或完整歧义训练。没有编造这些环节的性能、显存或耗时。合成训练分数不会用于论证选题创新或生物状态恢复。
