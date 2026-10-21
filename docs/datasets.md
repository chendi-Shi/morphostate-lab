# 数据集、配准与输入格式

本仓库不包含患者数据、模型权重或真实队列的实验结果，也不自动下载大型队列。下面是可执行的数据准备路线；链接和公开信息核查日期为 **2026-10-21**。具体文件、使用条款和配对关系以提供方当前记录为准。

## 数据选择

| 队列 | 已核实的内容 | 获取方式 | 本项目中的用途与限制 |
| --- | --- | --- | --- |
| [GSE311609](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE311609) | 肺癌、乳癌 Xenium，多种基因面板；细胞计数、分子位置和分割边界；部分样本有 post-Xenium H&E 和 IHC | GEO 公开文件，按样本选择下载 | 首个癌症可行性实验。GEO 的 **41 条记录不等于 41 位患者**；已核实肺癌 5K 子集含 L1、L3、L5、L6、L7、L8 六位患者的配对 H&E。不能假设全部记录均有可用配准 |
| [GSE250346](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE250346) | 肺纤维化及对照，35 位供者、343 基因面板；包含配准 H&E 文件 | GEO 公开文件 | 供者数较多的单细胞状态研究。组织块、重复测量和切片均须归回同一供者；目标局限于面板内基因 |
| [MOSAIC Window](https://www.mosaic-research.com/mosaic-window) | 60 位患者的多模态样本，涵盖 5 个癌种；包含 H&E、空间和单细胞/单核表达等模态 | 经 EGA/DAC 正式申请 | 外部验证候选。已核实的卵巢癌子队列使用 **Visium SD**；空间 spot 包含混合细胞，不能作为逐细胞状态真值 |

GSE311609 的相关论文是 [Resolving sensitivity, specificity and signal contamination in Xenium spatial transcriptomics](https://doi.org/10.1038/s41592-026-03089-8)。H&E 补充上传有[作者公开说明](https://github.com/bdsc-tds/xenium_analysis_pipeline/issues/111#issuecomment-4620206770)。部分大图包含多个组织块，必须建立组织块到原始 Xenium 样本的映射。

GSE250346 参考正式论文 [Spatial transcriptomics identifies molecular niche dysregulation associated with distal lung remodeling in pulmonary fibrosis](https://doi.org/10.1038/s41588-025-02080-x) 和[作者分析代码](https://github.com/Banovich-Lab/Spatial_PF)。早期预印本和最终版本的样本规模不同，分析时记录使用的版本。约 45 个组织样本不能按 45 个独立患者计算。

MOSAIC 的申请入口和项目摘要草稿见 [data_access_request.md](data_access_request.md)。不同癌种的数据不可直接拼接后声称实现了跨队列验证；先明确相同细胞类型、目标基因和评价单位。

## 原始数据清单

先建立本地 `data/raw/sample_manifest.csv`。它是人工审计清单，**当前 CLI 不自动读取或下载此清单**。下列字段建议保留：

| 字段 | 含义 |
| --- | --- |
| `dataset_id`, `accession`, `donor_id`, `slide_id` | 队列、原始条目、供者和唯一切片标识；所有重复样本必须使用一致的供者标识 |
| `tissue_block_id`, `panel_id`, `assay_version` | 区分组织块、基因面板和测量版本 |
| `matrix_path`, `cells_path`, `he_path` | 本地表达矩阵、细胞元数据和真正的 H&E 图像 |
| `registration_path`, `registration_direction` | 配准变换及方向，例如 `xenium_um_to_he_level0_px` |
| `he_mpp_x`, `he_mpp_y`, `coordinate_unit` | 图像 level-0 的微米/像素比例，以及原坐标单位 |
| `same_section`, `registration_qc`, `qc_note` | 同切片还是相邻切片、配准验收结论、排除原因 |
| `source_url`, `download_date`, `sha256`, `terms_reference` | 来源、版本和许可记录 |

建议本地目录如下，名称仅作示例：

```text
data/
  raw/
    sample_manifest.csv
    GSE311609/<slide_id>/
      cell_feature_matrix.h5
      cells.parquet
      histology.ndpi
      registration.json
  derived/
    <slide_id>/
      image_features.npz
      image_covariates.npz
  prepared/
    <cohort_id>/
      counts.npz
      features.npy
      covariates.npy
      cells.csv
      genes.csv
      provenance.json
```

数据、图像、派生特征、检查点和患者对应表应保留在本地受控存储中，不提交到 Git。公开可下载不表示任何再分发都被允许；原始使用条款仍适用。

## H&E 与坐标验收

`morphology.ome.tif` / `morphology_focus.ome.tif` 通常是 Xenium 的荧光形态通道，**不是 H&E**。不能把这些图像送入编码器后宣称完成了 H&E-only 实验。图像来源须在 provenance 中标明。

细胞中心通常处于 Xenium 坐标系，H&E patch 使用像素坐标。二者不是默认相同的坐标系，即使同切片也可能有旋转、翻转、缩放和非刚性变形。

1. 核对样本、组织块、切片和测量面板；先排除相邻切片被误当逐细胞配对的情况。
2. 明确输入坐标单位、图像层级、原点和轴方向。若变换输入为微米，不要先额外除以 mpp；若变换输出为 level-0 像素，再按读取层级的 downsample 转换。
3. 用分布在多个组织区域的标志点和细胞边界叠图验收，并记录残差的微米尺度。阈值应结合细胞尺寸和任务预先确定，不能查看预测结果后放宽。
4. 在组织边缘、折叠、裂隙和低配准质量区域建立排除掩码，保存排除数量。
5. 固定物理 patch 尺寸，记录像素尺寸、mpp、编码器、权重版本和图像预处理。不要让不同扫描倍率隐式改变视野。

GSE311609 的图像可用性已核实，但本项目尚未逐一下载并验证配准。仓库已有地标仿射拟合、验证残差、配准裁图和训练患者粗类型监督的线性图像分类器；这些不等于自动/非刚性配准器或成熟病理细胞分型模型。全部队列的批处理下载不在自动执行范围。

## Xenium 导入接口

当前导入器读取标准 10x/Xenium `cell_feature_matrix.h5` 的 `/matrix` 稀疏矩阵，选择 `feature_type == "Gene Expression"` 的行，将原始“基因 × 细胞”转为“细胞 × 基因”。基因标识使用 H5 中的 `features/id`。

另需准备：

- 细胞元数据：CSV、压缩 CSV 或 Parquet，至少包含 `cell_id`；Parquet 需要可用的相应依赖。没有 `cell_type` 时记为 `unknown`，不会自动调用 RNA 分型。
- 图像特征 NPZ：数组键名为 `cell_ids` 和 `features`，其中 `features` 的形状是 `[N, D]`。
- 协变量 NPZ：数组键名为 `cell_ids` 和 `covariates`，其中 `covariates` 的形状是 `[N, C]`。仅图像可获得的类型概率、邻域组成等才可标记为 `image`。

元数据、特征、协变量与表达矩阵的 cell ID 集合必须完全一致。导入器按 ID 重排，不做静默 inner join 或丢弃细胞。需要 QC 子集时，应明确且一致地过滤全部输入并保留排除记录。

单次导入传入一个 `donor_id` 和 `slide_id`。已有同名元数据列须与参数相符；多供者 TMA 须先按经过核实的组织块映射拆分。导入后 ID 为 `slide_id::原始cell_id`，另存 `source_cell_id`；额外元数据列保留。

`merge_datasets` 要求基因及其顺序完全相同、特征和协变量维数及来源兼容，并拒绝重复 cell ID。它不自动做跨面板填零。不同面板需要显式确定可比较基因集合并统一预处理后再导入；**未测基因不是表达为零的基因**。

## Prepared 数据契约

| 文件 | 要求 |
| --- | --- |
| `counts.npz` | SciPy CSR，`[N, G]`，有限、非负整数原始计数；零总计数细胞不接受 |
| `features.npy` | 数值数组 `[N, D]`，有限值，建议 float32；真实实验应为图像特征 |
| `covariates.npy` | 数值数组 `[N, C]`，有限值，建议 float32；列顺序必须固定 |
| `cells.csv` | 必含 `cell_id, donor_id, slide_id, cell_type`；cell ID 唯一且非空 |
| `genes.csv` | 一列 `gene_id`，唯一且非空，顺序对应计数列 |
| `provenance.json` | 必须如实记录 `dataset_id`, `feature_source`, `covariate_source`；来源类别为 `image`, `rna_oracle`, `synthetic` |

所有行必须一致对应同一细胞。加载器不在全队列上自动标准化或筛选基因；训练变换只能由训练供者拟合。

下面是应由用户补全的 provenance 示例，路径和版本都是占位值：

```json
{
  "dataset_id": "GSE311609-verified-subset",
  "feature_source": "image",
  "covariate_source": "image",
  "image_modality": "post-Xenium H&E",
  "encoder": "REPLACE_WITH_ENCODER_AND_VERSION",
  "encoder_weights_sha256": "REPLACE_WITH_HASH",
  "encoder_pretraining_overlap": "not_yet_audited",
  "patch_size_um": 64,
  "coordinate_unit": "he_level0_pixel",
  "registration_qc": "REPLACE_WITH_LOCAL_QC_RECORD",
  "covariate_columns": ["REPLACE_WITH_IMAGE_DERIVED_COLUMN_NAMES"],
  "cell_type_annotation_source": "REPLACE_WITH_SOURCE_AND_GENE_SET"
}
```

`cells.csv` 的 `cell_type` 默认是分层评价标签，不会自动成为表达模型输入。若直接把实测 RNA 分型、RNA 去卷积或目标基因作为推理协变量，必须标记 `covariate_source: "rna_oracle"` 并显式启用 oracle 实验；结果不能称为 H&E-only 部署性能。`image-covariates` 允许仅用训练/验证供者粗类型注释监督和选择图像分类器，测试协变量从图像预测，并记录分类器拟合/选择供者；每个外层折必须重新生成。这与直接输入测试 RNA 类型不同，仍需审计训练注释与预训练患者重叠。

## 分子候选图准备

`build-graphs` 接受预先配准的 CSV：`slide_id,x,y,gene_id,cell_id`，可选 `mass,is_core`。prepared cells 另需同坐标单位的 `x_centroid,y_centroid`。空 owner ID 表示背景；非空 ID 对应每张切片的 `source_cell_id`（存在时），否则对应 prepared `cell_id`。计数过滤、基因 ID 和分子过滤必须同步，图中原始归属汇总需精确等于 prepared counts。

半径定义的是中心距离候选代理，不能冒充膜分割误差分布；原始 owner 必须已经在半径内。`is_core` 只接受独立确定归属声明，不能由中心距离自动推断。几何预算是按 `mass*(distance/radius)^2` 汇总的每切片上限，背景预算单独指定。预算应由训练/独立配准证据确定，不能根据测试预测分数调整。

共享预算图不能任意拆分、移除细胞或删除观测。原始数据太大时，应先定义明确的小型患者内研究区域，统一处理其分子、细胞、边界与背景，再构图，并记录该条件化研究范围。
