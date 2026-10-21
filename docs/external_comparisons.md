# 接入外部基线与 SOTA 预测

本仓库提供统一评分入口，不捆绑第三方模型，也不声称已经复现 SOTA。先按 [research_plan.md](research_plan.md) 确定输入组、患者划分、评价分辨率、基因及调参预算；在作者代码中完成训练。外部方法的超参数只用内部验证选择。

需要固定作者代码版本、执行训练／预测命令及转换实际输出时，使用已实现的 `reproduce`；GHIST 与 Pixel2Gene 的格式适配、示例配置及单位要求见 [sota.md](sota.md)。下面的 `evaluate-external` 用于已经完成格式转换的预测。

将最终冻结预测导出为 `.npz`，三个字段：

- `cell_ids`：锁定测试集合的完整细胞 ID，唯一、无额外/缺失行。
- `gene_ids`：锁定测试集合的完整基因 ID，唯一、无额外/缺失列。
- `rates`：细胞×基因，有限非负，指定标准 library reference 下的预期计数。

明确 `--reference-library` 的单位。例如模型输出每一万总转录本的 rates，填10000。评分器只把 rates 缩放到训练参考深度；不从测试真值校准预测。`log1p` 表达、z-score、任意 decoder 数值需要在作者定义下正确反变换，不能直接塞入 rates。spot 或固定网格输出不能通过复制到每个细胞伪装为单细胞预测；应另建共同区域评价。

填写 `examples/external_model_metadata.json` 的实际信息。评分器会拒绝声明的 train/validation 与 test 供者重叠，但不能独立核验声明、预训练泄漏或训练时是否看过 test。

```bash
morphostate evaluate-external --run runs/xenium_pilot --predictions outputs/external_rates.npz --reference-library 10000 --model-name author_method --metadata data/external_model_metadata.json --output runs/external_author_method
```

输出每患者指标及配对差异。`input_mode=image/rna_oracle/multimodal` 分开报告，oracle 不参加 H&E-only 排名。这个入口不自动选择外部模型，不能在反复查看测试分数后继续调参并仍宣称测试集合未使用。
