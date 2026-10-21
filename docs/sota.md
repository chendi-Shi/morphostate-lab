# 外部 SOTA 的执行、适配与可比性

此处实现的是**调用作者代码的复现执行器及真实输出适配器**，不是把其他论文重新命名成自己的实现，也不代表已完成真实数据复现。作者环境、图像配准、获准权重和数据需要单独准备。所有访问与版本核验日期为 2026-10-21；完整 revision 在 `configs/sota/upstreams.json`。

## 已实现的代码

`morphostate.comparisons.fit_predict_comparisons` 提供同编码器的加权常数 Poisson MLE、低秩回归、残差 PCA 后 ridge、正则化 partial CCA、Poisson MLP，以及按训练患者内部交叉拟合的 residual ridge。rank 与 MLP epoch 只由验证损失选择；验证损失接受患者均衡权重。crossfit nuisance 逐折排除被预测患者；最后拟合的 nuisance 使用完整训练集。该实现不声称因果识别或正交性定理。类型均值必须显式传入类型标签，输出名为 `type_mean_oracle`，不会混入默认可部署模型。所有默认预测都不接受测试 RNA。

`morphostate.sota.run_reproduction(config_path, output=None)` 会核验完整 Git SHA、拒绝已修改的 tracked 源代码、核验患者划分和声明、输出 train/val/test 清单、执行显式 argv、记录每条命令日志与退出状态、哈希实际权重文件，再转换输出。失败也会保存 `audit.json`。记录和清单不能证明任意外部程序确实遵守清单；作者配置仍需审查。训练清单不可包含 test 患者；编码器预训练重叠需单独审核。

```python
from morphostate.sota import run_reproduction
audit = run_reproduction("configs/sota/ghist.yaml")
```

YAML 中本地路径、donor IDs、作者配置和选定 checkpoint 必须根据真实数据填写。配置示例中的命令参数来自核验过的作者代码，不会自动下载模型、默许未知许可证或替用户申请权重。命令只使用 `shell=False`，无需 shell 字符串拼接。

## GHIST：真实 CSV 适配

- [作者仓库](https://github.com/SydneyBioX/GHIST)；GPL-3.0。只调用，不复制作者网络代码。
- [inference.py（固定 revision）](https://github.com/SydneyBioX/GHIST/blob/917456be305fc82e92293ea272812e79675e821c/inference.py) 将预测除以 `expr_scale`，然后输出 `epoch_{epoch}_expr.csv`，行是 cell IDs、列是 genes。
- [dataset_input.py](https://github.com/SydneyBioX/GHIST/blob/917456be305fc82e92293ea272812e79675e821c/dataio/dataset_input.py) 对输入表达执行 `expr_scale * log1p`。所以 CSV 要用 **expm1** 还原，不能再次除以 expr_scale。
- `convert_ghist_csv` 检查重复 IDs、负值、缺失列和一一对应 ID map。测试 fixture 精确复现作者 `DataFrame(...).to_csv(...)` 格式，包括前导零 IDs。

```python
from morphostate.sota import convert_ghist_csv
convert_ghist_csv("epoch_50_expr.csv", "ghist_native.npz",
                 id_map="source_id_to_cell_id.csv")
```

**单位问题必须显式解决。** 原始 GHIST 学习 raw counts，适配后保存键 `expected_counts`；本项目主要评价的是训练参考 library 下的 `rates`。两者不会自动混用，也禁止用测试真实 library 把 raw count 预测除成 rate。要在同一 rate 表里比较，需预先将作者训练/验证目标转换为本项目的训练参考 library 单位，记录这属于统一目标协议的适配版本，然后设置 `training_target_normalization: reference_library` 和 `reference_library`。执行器仅在这种明确声明下调用 `external.evaluate_external`。原生目标版本保留独立的 counts 结果。

作者 `val` 模式可扫描 epoch；epoch 必须由验证集确定，之后固定运行 `predict` 模式。官方 demo 是区域划分示例，不能直接声称跨患者泛化；需要将作者数据配置与锁定患者清单对应。真实核 mask、作者数字 ID 与本项目全局 ID 的对应关系需要在配准阶段建立。

## Pixel2Gene：真实预测网格与共同掩码

- [作者仓库](https://github.com/yaosicong1999/Pixel2Gene)；API 返回许可证 `NOASSERTION`，使用前检查作者 LICENSE，不能推断其为无限制开源。
- [impute_filter_predict.py](https://github.com/yaosicong1999/Pixel2Gene/blob/4f690ca2c629cc1a079fd07bf16ab2044e8cc460/scripts/impute_filter_predict.py) 实际写出 `cnts-super/{gene}.pickle` 二维 float16 网格及 `predict-gene-names.txt`。该版本虽创建 Zarr，写入 Zarr 的代码被注释；不能拿这个空 Zarr 评价。
- `convert_pixel2gene_pickles` 按配准后的共同评价 mask 汇总网格，不用 nearest-cell 把网格冒充单细胞。映射 NPZ 包含 `cell_ids,row,col,cell_index,weight`，后四个为等长向量。每个网格像素分给各 cell 的权重和不得超过 1；所有评价 cell 必须覆盖。权重是几何面积份额，不能从测试表达推断。
- 作者将训练 counts 按 mask_size 转为网格密度；须确保映射使用作者输出分辨率、坐标变换和面积单位。原生 counts 仍不自动等同 normalized rates。
- pickle 只能加载可信本地作者输出；默认拒绝，必须显式 `trust_pickle=True`。测试覆盖两细胞共享像素及守恒失败情形。

`configs/sota/pixel2gene.yaml` 是经过核验的预测命令。已有训练模型必须来自锁定患者训练/验证集，权重和 metadata 均记录哈希。作者原始 demo 使用单张切片内 k-fold，不等于本项目跨患者协议；需要在独立 checkout 中完成并提交数据加载适配，再固定新 revision。

## 其他必须比较的作者代码及边界

| 方法 | 实际可用入口 | 本项目中的角色与限制 |
|---|---|---|
| Path2Space | [extended](https://github.com/eldadshulman/path2space-extended)、其 README 指向的 [companion](https://github.com/eldadshulman/path2space-companion)、[Zenodo](https://doi.org/10.5281/zenodo.20171390) | extended 是分析与评价仓库；不能当成单细胞训练实现。先建立共同 spot/region 协议，再比较；不把 tile 预测广播到 cells 参加单细胞排名。 |
| VOICE | [论文](https://arxiv.org/abs/2608.08366) | 论文及定向搜索未定位到作者发布的代码/权重入口；不伪造实现，不把独立实现的 retrieval baseline 叫 VOICE。获得作者代码后再锁版本。 |
| Proseg | [作者仓库](https://github.com/dcjones/proseg)，`proseg --xenium transcripts.parquet` | 上游重新分割。当前 Proseg 3 默认 SpatialData Zarr，`sdata.tables['table'].X` 为整数估计；可选 expected counts 为连续值。不可套用旧版 CSV 格式或静默四舍五入。对处理后的标签训练相同图像头。 |
| SPLIT | [作者仓库](https://github.com/bdsc-tds/SPLIT)，R `SPLIT::purify(...)` | 上游去污染；依赖参考表达和分解结果。默认 `purified_counts` 及 `cell_meta`，也支持 SCE。参考只能来自允许的训练/外部参考来源。 |
| ResolVI | [复现仓库](https://github.com/YosefLab/resolvi-reproducibility)，[scvi-tools 官方文档](https://docs.scvi-tools.org/en/stable/api/reference/scvi.external.RESOLVI.html) | 上游去污染；在其作者环境执行，latent 或 normalized expression 不直接冒充原始整数 counts。复现仓库 license 未声明，需要另查 scvi-tools 实现许可证与使用条件。 |
| DeSpotX | [作者仓库](https://github.com/Gentles-lab/DeSpotX)，`despotx --h5ad ... --cell-type-col ... --spatial-key spatial --out ... --device cuda` | 上游去污染；`adjusted_float.npy` 是 cells×genes，行列由 cells.tsv/genes.tsv 定义；保留浮点，不伪造整数。输出只能作为训练标签处理对照，不能作为 H&E-only predictor 排名。 |

上述预处理方法已有完整作者实现，但本仓库没有声称重写其内部算法。必须报告未经处理标签与每种处理标签下的同一个图像预测器结果，以及处理使用了哪些参考和哪些患者；不能把去污染器直接与 H&E-only 模型排成一个 SOTA 表。

## 验证范围

本仓库测试实际执行了一个临时 pinned Git checkout 的预测脚本、记录日志和 checkpoint 哈希、转换作者格式 CSV，并检查错误 revision、患者泄漏、修改代码、失败命令、越界 cwd 和单位误用。这个小型闭环验证**执行器**，不是声称跑过 GHIST/Pixel2Gene 作者网络。完整真实数据复现的证据必须是各方法保存的 audit、checkpoint 和患者级评价文件。
