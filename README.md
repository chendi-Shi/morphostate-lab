# morphostate-lab

面向 **H&E 病理图像与空间转录组** 的实验框架：在严格留患者的条件下，检查图像是否提供超出细胞类型和邻域组成基线的分子预测信息。

仓库：[chendi-Shi/morphostate-lab](https://github.com/chendi-Shi/morphostate-lab)。当前版本提供数据校验、Xenium 导入、五类基线、模型选择、负对照和可追溯输出。它是研究起点，**不是已经验证的 A 类论文方法**；没有附带真实患者实验结果。

进一步的方法问题是：分子可能归属多个相邻细胞时，如何利用联合守恒约束学习跨患者的病理—分子状态映射，并明确哪些状态结论无法可靠识别。完整研究假设、SOTA、强基线、11组消融和退出条件见 [research_plan.md](docs/research_plan.md)。相关 ambiguity 工具属于研究原型，不代表已解决状态识别或已获得统计保证。

## 已实现的实验

| 模型 | 输入与训练方式 | 用途 |
| --- | --- | --- |
| `image_ridge` | 图像特征 → 标准化 log 表达 | 直接表达预测基线 |
| `covariate_ridge` | 协变量 → 标准化 log 表达 | 组成等已知因素的线性基线 |
| `residual_ridge` | 先拟合协变量，再用图像拟合训练残差 | 判断复杂模型是否优于简单残差学习 |
| `covariate_nb` | 协变量 → 负二项计数模型 | 包含测量深度的计数基线 |
| `conditional_nb` | 冻结已训练的协变量 NB，再学习低秩图像增量 | 检查图像增量及 rank 选择 |

条件 NB 使用训练集拟合的线性回归去除图像特征中可由协变量线性解释的部分。这不是独立性保证，也不能完整消除组成混杂。当前 residual ridge 和 NB 基线没有 donor cross-fitting。

实验流程按供者划分训练、验证和测试集；标准化、检测过滤和方差选基因仅拟合训练供者。NB 早停和 rank 选择使用验证供者加权 NLL；被选中的条件模型才进入测试报告。评价包含患者等权指标、患者 bootstrap，以及同供者同粗类型内的图像置换。后者是诊断性负对照，不提供有效性已证明的条件检验 p 值。

完整规则及解释边界见 [protocol.md](docs/protocol.md)。

## 安装

建议新建 **Python 3.11** 环境，兼容目标为 Python 3.10–3.12。CPU 可以运行合成示例；GPU 版本应按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)选择与本机驱动兼容的构建。

```bash
git clone https://github.com/chendi-Shi/morphostate-lab.git
cd morphostate-lab
python -m venv .venv
```

Windows PowerShell 激活环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS 激活环境：

```bash
source .venv/bin/activate
```

CPU 安装：

```bash
python -m pip install --upgrade pip
python -m pip install "torch>=2.2,<3" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
morphostate --help
```

可选的图像编码与 Parquet 支持：

```bash
python -m pip install -e ".[images]"
```

私有仓库 clone 需要当前 GitHub 账号具有访问权限。第三方病理编码器的权重许可、账号认证和下载需单独完成；仓库不分发这些权重。

## 先跑通一个合成实验

在仓库根目录运行：

```bash
morphostate demo --output data/demo --seed 7
morphostate validate --data data/demo
morphostate run --config configs/demo.yaml
```

查看 `runs/demo/report.md`、`metrics.json` 和逐患者 CSV。合成数据包含人工设置的细胞身份、状态、供者和深度变化，**分数仅用于检查代码路径，不能作为真实生物学证据**。

配置中的 `data`、`output` 路径相对于 YAML 文件所在目录解析。实验输出目录必须为空；重复运行时指定新目录：

```bash
morphostate run --config configs/demo.yaml --output runs/demo_repeat
```

运行检查：

```bash
python -m pytest -q
python -m ruff check src tests
```

这些是可执行检查命令；README 不将尚未运行的检查描述为已通过。

## 没有测试 RNA 的推理

`predict` 读取条件 NB 检查点和两个 NPZ 文件：

- 图像文件：`cell_ids`、`features`。
- 协变量文件：`cell_ids`、`covariates`。

两者 cell ID 集合须一致，列含义和编码器必须与训练时相同。推理不读取 `counts.npz`、测试 RNA 总量或 RNA 类型标签。

例如，先将合成示例的已有特征转换成推理输入。下面的 Python 片段不读取计数文件：

```python
import numpy as np
import pandas as pd

cells = pd.read_csv("data/demo/cells.csv", dtype=str)
ids = cells["cell_id"].to_numpy(dtype=str)
np.savez_compressed(
    "data/demo/predict_features.npz",
    cell_ids=ids,
    features=np.load("data/demo/features.npy"),
)
np.savez_compressed(
    "data/demo/predict_covariates.npz",
    cell_ids=ids,
    covariates=np.load("data/demo/covariates.npy"),
)
```

```bash
morphostate predict --checkpoint runs/demo/conditional_nb.pt --features data/demo/predict_features.npz --covariates data/demo/predict_covariates.npz --output outputs/demo_rates.npz
```

输出包含 `cell_ids`、`gene_ids` 和 `rates`，表示训练参照深度下的期望计数。它不是 H&E 独立预测的绝对 RNA 总量。训练和评价时使用测量深度作为 exposure，也不改变这一点。上述合成推理示例只验证接口，不能代替留患者测试。

## 接入真实 Xenium 数据

首先完成样本/供者映射、H&E 配准及细胞级 patch 提取。**Xenium 的 `morphology.ome.tif` 通常是荧光通道，不是 H&E**；同切片也不意味着坐标已对齐。注册变换、物理尺度、图像层级、边缘排除和配准质量必须记录。

本仓库提供一个可选 `timm` 编码适配器，输入已经裁好的、配准通过的图像。CSV manifest 格式见 [examples/patch_manifest.csv](examples/patch_manifest.csv)：

```csv
cell_id,patch_path
cell_001,patches/cell_001.png
cell_002,patches/cell_002.png
```

patch 路径相对于 manifest 所在目录。示例 CSV 仅说明格式，仓库不附带对应患者图像。调用示例：

```bash
morphostate extract-features --manifest data/derived/slide01/patch_manifest.csv --model resnet18 --pretrained --output data/derived/slide01/image_features.npz --device auto
```

特征提取会同时生成 `image_features.json`，记录 `feature_source=image` 和 `feature_encoder`（model、weights_sha256、transform）。真实数据导入要求这个同名 sidecar；使用外部 encoder 时也须提供对应元数据。合并与真实图像 checkpoint 推理会核对权重和变换身份，不能仅因向量维度相同就混用模型。输入数据的供者与 encoder 预训练重叠仍需另外审计。

`--pretrained` 会显式请求下载相应预训练权重。这里的 ResNet18 用于接口检查，不是病理领域强基线；正式实验应替换为许可允许且已审计预训练重叠的固定病理编码器。也可用 `--checkpoint` 传入兼容 `timm` 的本地权重，不能与 `--pretrained` 同时使用。不兼容的编码器需要单独适配。

还需自行准备图像协变量 NPZ：每行对应一个 cell ID，包含从 H&E 估计的粗类型概率、邻域组成等，键名为 `cell_ids`、`covariates`。仓库尚未实现病理细胞分型器；不要用零向量或 RNA 标签冒充已估计的图像组成。

导入一份**单供者**样本：

```bash
morphostate import-xenium --matrix data/raw/slide01/cell_feature_matrix.h5 --cells data/raw/slide01/cells.parquet --features data/derived/slide01/image_features.npz --covariates data/derived/slide01/image_covariates.npz --donor-id patient01 --slide-id slide01 --dataset-id GSE311609 --covariate-source image --output data/prepared/slide01
```

导入器读取标准 10x H5 的 `Gene Expression` 特征，按 cell ID 严格对齐，拒绝静默丢行。输出 ID 为 `slide_id::原始cell_id`。多供者 TMA 必须先拆分或建立准确的细胞到供者映射，不能直接把整张 TMA 赋成一个患者。

将多个相同面板及相同特征/协变量定义的样本合并：

```bash
morphostate merge --inputs data/prepared/slide01 data/prepared/slide02 data/prepared/slide03 --output data/prepared/xenium
morphostate validate --data data/prepared/xenium
morphostate run --config configs/xenium_pilot.yaml
```

此处样本路径为占位示例，需要实际准备；正式研究必须有足够的独立供者。合并要求基因顺序完全一致，不自动跨面板填零或取交集。未测基因不能当作零表达。

若协变量来自 RNA 分型或去卷积，导入时使用 `--covariate-source rna_oracle`，配置中显式设置 `allow_oracle: true`。此类运行会标记为 `oracle`；检查点推理也要求 `--allow-oracle`。它用于机制/敏感性分析，不代表 H&E-only 部署性能。`cells.csv` 中的 `cell_type` 默认仅用于分层评价。

## 数据格式

```text
prepared_dataset/
  counts.npz       # SciPy CSR: cells × genes，非负整数原始计数
  features.npy     # cells × image_features
  covariates.npy   # cells × covariates
  cells.csv        # cell_id, donor_id, slide_id, cell_type
  genes.csv        # gene_id
  provenance.json # dataset_id, feature_source, covariate_source 等
```

cell ID、gene ID 必须唯一非空；数组行序必须一致，数值须有限，零总计数细胞不接受。来源类别使用 `image`、`rna_oracle` 或 `synthetic`。图像编码器、权重版本、配准和协变量列定义需要加入 provenance 并核实，不会因导入成功而自动获得验证。

详细 manifest、数据源和坐标检查见 [datasets.md](docs/datasets.md)。

## 可用数据与申请入口

| 数据 | 获取方式 | 实验角色 |
| --- | --- | --- |
| [GSE311609](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE311609) | 公开处理数据、部分配对 post-Xenium H&E/IHC | 癌症可行性实验；41 条 GEO 记录不是 41 位患者，配准需要逐样本验收 |
| [GSE250346](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE250346) | 公开 Xenium 和配准 H&E | 肺纤维化及对照，35 位供者、343 基因面板；结论限于已测面板 |
| [MOSAIC Window](https://www.mosaic-research.com/mosaic-window) | [EGA 正式申请](https://ega-archive.org/studies/EGAS50000000689) | 外部区域/患者级验证候选；Visium SD 不提供逐细胞空间真值 |

MOSAIC 的项目摘要、所需材料和子队列入口见 [data_access_request.md](docs/data_access_request.md)。模板是未提交草稿。仓库不自动申请、下载、再分发患者数据。

## 配置与单卡运行

优先固定病理编码器，将特征提取一次并缓存，再训练小型回归头。先用同一面板、有限基因和预先规定的每供者细胞上限完成可行性实验，再扩展样本量。

[configs/xenium_pilot.yaml](configs/xenium_pilot.yaml) 包含基因数、每供者细胞上限、rank 候选、batch size、早停及设备设置。大计数矩阵保持稀疏直到选定基因；最终 pilot 训练会转为稠密数组，并有 `max_dense_elements` 限制。超出限制时需降低规模或实现新的流式训练接口，不能假设当前 runner 可直接处理整个百万细胞队列。

固定特征能降低 GPU 需求，但内存、耗时仍取决于细胞数、基因数和特征维数。当前没有给出未经实测的显存或运行时保证。

## 输出与解释

| 输出 | 内容 |
| --- | --- |
| `report.md`, `metrics.json` | 测试供者宏平均、模型对比与置换诊断 |
| `*_per_donor.csv` | 每位测试供者的误差、相关性及有效数量 |
| `split.csv`, `split_donors.json` | 实际使用的细胞及供者划分 |
| `selection.json`, `history_*.json` | 验证集 rank 选择及训练轨迹 |
| `conditional_nb.pt` | 条件模型、训练变换、基因顺序及来源标记 |
| `*_ridge_parameters.npz` | ridge 参数 |
| `test_predictions.npz` | 五类模型在锁定测试集合上的预测 |
| `config.json`, `provenance.json` | 配置、版本、种子、来源及元数据指纹 |

NB 模型以验证 NLL 选择；统一测试表使用 Poisson deviance、log-rate MSE 和 within-type PCC。within-type PCC 是按供者内粗类型去均值后的描述性指标，不能替代充分的组成匹配。只有少数测试患者时，bootstrap 区间不支持强泛化结论。

当前指纹覆盖元数据文件，不覆盖大型计数和图像数组；需要完整内容寻址时应另建校验清单。`.gitignore` 排除本地数据、权重和运行输出；上传前仍应检查文件列表。

## 尚未实现与研究边界

- 自动 H&E–Xenium 配准、全切片细胞检测/类型预测及完整数据集下载。
- SPLIT、ResolVI、PACE、DeSpotX、Proseg 等 SOTA 的统一训练和评价适配器。
- Partial CCA、donor cross-fitting、完整污染生成模型及经验证的状态可识别性保证。
- 真实患者外部验证、独立蛋白状态真值及完整预训练患者重叠审计。

新方法需要证明它解决了强基线的可重复失败。若简单残差回归已足够，或所谓增益只存在于 RNA-oracle 输入，复杂结构不能作为必要贡献。已知重叠和主要参考文献见 [related_work.md](docs/related_work.md)，下一阶段研究路线见 [research_plan.md](docs/research_plan.md)。
