# morphostate-lab

**病理图像—空间转录组的结构化弱监督实验代码**：在分子归属不唯一时，保留分子守恒、几何与背景预算，训练跨患者表达预测器，并计算预定义状态对比的可行范围。

仓库：[chendi-Shi/morphostate-lab](https://github.com/chendi-Shi/morphostate-lab)。核心训练、稀疏 LP、候选归属图、强基线、消融、多折多种子调度、无测试 RNA 推理和状态评价均已实现。**代码实现不等于已获得 A 类创新、真实患者效果或统计保证。** 本仓库不附带患者数据、第三方授权权重或编造的 SOTA 结果。

**2026-10-22 研究决策：** 放宽训练预算的三个随机种子开发诊断中，`joint_relative` 仍落后于固定标签和降秩回归；该目标降为保守性/失败模式对照，不再作为已有性能支持的主方法。候选新方向、必须击败的最新对手与真实数据准备状态见 [方法重审](docs/method_reassessment_20261022.md)。真实 SOTA 性能比较尚未完成。

- [研究问题、前作、完整比较与失败判据](docs/research_plan.md)
- [实验协议和科学解释边界](docs/protocol.md)
- [数学定义、推导、反例与适用条件](docs/theory.md)
- [A10 状态抹除反例与 A11 覆盖率—风险协议](docs/stress_and_selection.md)
- [第三方 SOTA 执行与真实输出适配](docs/sota.md)
- [软件验证与尚未实测的环节](docs/verification.md)

## 已实现功能

| 部分 | 实现 |
|---|---|
| 数据 | Xenium 10x H5 导入、ID 对齐、同面板合并、来源与供者校验 |
| 图像 | 地标仿射拟合与验证残差、配准裁图、Pillow/OpenSlide 后端、冻结 timm 特征接口 |
| 图像协变量 | 训练患者粗类型监督的线性图像分类器、验证早停、同切片邻域概率；每个外层折单独拟合 |
| 候选归属 | 原始转录本到稀疏 owner 图、显式背景、质量守恒、几何/背景预算、可行性校验 |
| 内层优化 | 无共享预算的精确逐行求解；有共享预算的 HiGHS 稀疏 LP；精确边际界和线性对比界 |
| 外层学习 | 联合相对悲观目标、独立区间、固定标签、乐观目标、绝对准似然、core 子集、随机归属、ensemble、连通分量聚合、去除预算 |
| 强基线 | ridge、组成 NB、条件 NB、训练均值、Poisson MLP、降秩回归、残差 PCA、正则化 partial CCA、患者交叉拟合 residual ridge |
| 实验调度 | 供者/组外层 folds、LOO、固定种子、预定义消融变体、断点恢复、配置摘要校验、患者配对汇总 |
| 推理与评价 | 标准化 rates、患者指标与 bootstrap、独立蛋白/IF/专家状态评价、预定义线性状态对比范围 |
| 第三方 | 固定 Git revision 执行、日志和 checkpoint 哈希；GHIST CSV、Pixel2Gene 网格输出转换 |

候选图采用**已配准坐标上的中心距离代理**，不是自动识别出的真实膜边界。预算由研究者预先校准；LP 的精确性不证明可行集合包含生物真值。

## 安装

建议独立 Python 3.11 环境；CPU 可运行小型合成实验。GPU 构建请按 [PyTorch 官方说明](https://pytorch.org/get-started/locally/)选择。

```bash
git clone https://github.com/chendi-Shi/morphostate-lab.git
cd morphostate-lab
python -m venv .venv
```

Windows PowerShell：`.\.venv\Scripts\Activate.ps1`；Linux/macOS：`source .venv/bin/activate`。

```bash
python -m pip install --upgrade pip
python -m pip install "torch>=2.2,<3" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
python -m pip install -e ".[images]"
morphostate --help
```

`[images]` 用于裁图、timm 和 Parquet；OpenSlide WSI 需另安装其 Python 包与系统运行库。私有仓库访问、病理编码器授权与权重下载由使用者账号完成，仓库不分发第三方权重。

## 跑通核心方法与消融

生成小型验证数据和归属图；合成数据仅检查计算路径，不作为真实生物证据。

```bash
morphostate demo --output data/weak_demo --seed 7 --donors 6 --cells-per-donor 12 --genes 8 --features 4
morphostate validate --data data/weak_demo
morphostate synthetic-graphs --data data/weak_demo --output data/weak_demo_graphs
morphostate weak-run --config configs/weak_demo.yaml
```

更改数据规模后也需检查 rank 和基因筛选设置。`data`、`graphs`、`output` 相对于 YAML 所在目录解析。已有输出不会静默覆盖；重复执行使用新的输出目录。弱监督示例的短训练预算用于集成检查，不是正式调参配置。

多折、多种子和配置变体：

```bash
morphostate suite --config configs/weak_suite.yaml --plan-only
morphostate suite --config configs/weak_suite.yaml --resume
morphostate summarize --suite runs/weak_suite --reference fixed
```

先生成锁定清单，再执行；失败保留在 `status.json`，不会悄悄排除。示例 suite 为 3 折×2 种子×2 变体。汇总先在同患者内部平均种子，再以患者为单位比较，不能把种子或细胞数当独立患者数。自动分组不是疾病分层，特定分层设计使用显式 folds。

传统基线及新增同特征强基线：

```bash
morphostate demo --output data/demo --seed 7
morphostate run --config configs/demo.yaml
```

新增比较头由 `comparisons.enabled` 开启。默认模型不使用测试类型标签；显式类型均值仅作为 `type_mean_oracle` 单独解释。

## 方法名称与评价范围

| 配置方法名 | 含义 |
|---|---|
| `fixed` | 固定分割标签，统一 Poisson 预测头 |
| `joint_relative` | 在联合守恒和预算集合上，最大化相对冻结基线的 Poisson 损失差 |
| `interval_relative` | 用同一集合的精确边际界组成独立盒子，隔离联合耦合的价值 |
| `optimistic_relative` | 选择有利归属的对照，检查自我确认偏差 |
| `joint_absolute` | 最坏 Poisson 准损失；省略依赖标签的 factorial，不能称完整最坏 NLL |
| `core_only` | 只监督结构上唯一 owner 的分子子集，用子集验证选择；不是全 RNA，排除主排名 |
| `random_assignment` / `ensemble` | 随机 LP 目标产生可行顶点，分别作增强/集成；不是多面体均匀采样 |
| `aggregate` | 候选 owner 图的真实细胞连通分量聚合；不是固定网格或 Pixel2Gene 复现 |
| `no_geometry` / `no_background_budget` | 分别移除几何/背景共享预算 |

去除组成、容量/rank、encoder/数据替换由预定义 suite 变体控制。固定网格、真实核内分子、稀有状态压力场景及独立蛋白数据必须有相应输入；代码开关不代表这些生物实验已经运行。真实 Xenium 配置默认不启用 `core_only`，除非有独立确定归属注释。弱监督运行也可启用同 split 的常规强基线，统一保存患者表。

## 无测试 RNA 推理

两个输入 NPZ 分别包含 `cell_ids,features` 和 `cell_ids,covariates`。ID 集合、特征列语义、图像编码器和变换必须与训练一致。

```bash
morphostate weak-predict --checkpoint runs/weak_demo/joint_relative.pt --features data/inference/features.npz --covariates data/inference/covariates.npz --output outputs/joint_rates.npz
```

推理不读取测试计数、测量深度或测试 RNA 类型标签。输出 `cell_ids,gene_ids,rates` 是训练参照 library 下的期望计数，**不是 H&E 预测的绝对 RNA 总量**。传统条件 NB 检查点使用 `morphostate predict`。oracle 检查点需显式 `--allow-oracle`。示例推理文件需要由已对齐的实际图像特征准备。

## 接入真实数据

1. 建立供者、组织块、切片、panel、坐标和 encoder 重叠表。重复供者始终绑定。
2. 用人工/外部配准地标拟合仿射变换，保留未参与拟合的验证地标。`fit-affine` 不会自动发现地标或处理非刚性形变。
3. `extract-patches` 从明确的 RGB H&E 裁图；`extract-features` 用已许可的固定 encoder 缓存特征。Xenium `morphology.ome.tif` 的荧光通道不能冒充 H&E。
4. `import-xenium` 导入单供者样本，`merge` 合并同基因顺序和同特征定义。多供者 TMA 先明确细胞到供者的映射。
5. 锁定外层 split 后，用 `image-covariates` 拟合图像粗类型与邻域概率；每折重建，不能复用曾接触该折测试患者的分类器。
6. `build-graphs` 读取已注册到共同坐标系的转录本表，严格核验原始 owner 汇总与 prepared counts 一致。
7. 修改 [weak_xenium.yaml](configs/weak_xenium.yaml)，先在完整的小型图块上运行，再扩大。

CLI 参数见 `morphostate <命令> --help`；文件入口见 [datasets.md](docs/datasets.md)。

```text
prepared_dataset/
  counts.npz       # cells × genes，SciPy CSR，原始非负整数计数
  features.npy     # cells × image features
  covariates.npy   # cells × covariates
  cells.csv        # cell_id, donor_id, slide_id, cell_type；构图另需 x_centroid,y_centroid
  genes.csv        # gene_id
  provenance.json # 来源、encoder、协变量拟合/选择供者等

registered_transcripts.csv
  slide_id,x,y,gene_id,cell_id[,mass,is_core]
```

转录本表空 `cell_id` 表示背景。`is_core` 是独立注释的确定归属声明，默认 false；不能仅凭离中心近就宣称核内真值。`radius` 与细胞/分子坐标必须采用声明的相同单位；几何预算是每切片 `mass*(distance/radius)^2` 总和的上限，背景比例是质量上限。原始 owner 不在半径内时直接拒绝，要求检查配准与半径。

协变量为 `rna_oracle` 时必须显式允许并单独报告。训练/验证粗类型注释可监督图像分类器，但推理协变量只由图像特征及同切片邻域生成，不读取测试注释。线性分类器是可审计基线，不是已验证的病理细胞分型 SOTA。

## 状态证据、范围与外部 SOTA

`evaluate-states` 读取预测、独立标签 CSV 和预定义有符号基因程序，报告患者×粗类型内 AUROC、average precision 与分数组间差。来源限定为 protein、IF、expert 或 synthetic；声明仍需人工核实独立性。

`contrast-bounds` 为 `contrast,cell_id,gene_id,coefficient` 表定义的线性功能量计算上下界及方向。**这是给定候选图和预算的可行范围，不是置信区间，也不是预测器已恢复真实状态的证明。**

`bound-risk` 用同单位的独立/合成真实对比值，评价预先规定宽度阈值下的全体覆盖率与方向风险。每个对比必须属于单一患者；阈值不得看测试真值后选择，也不能把目标 RNA 派生的伪标签称独立真值。

```bash
morphostate bound-risk --bounds outputs/contrast_bounds.csv --truth data/independent/contrast_truth.csv --output outputs/bound_risk --thresholds 0 10 100 --truth-source independent_assay
```

[数学说明](docs/theory.md)列出联合目标、LP、退化反例与统计适用条件。标准结果的正确应用提供理论支撑，但不是已经证明新颖性或达到 CCF A 的保证。

可运行的反例与数学诊断：

```bash
morphostate stress-case --output data/stress
morphostate stress-case --output data/stress_excluded --exclude-null
morphostate theory-diagnostics --data data/stress/data --graphs data/stress/graphs --predictions data/stress/truth.npz --output runs/stress_theory --reference-library 80 --model-key true_rates --baseline-key null_rates
morphostate suite --config configs/stress_suite.yaml
```

诊断会检查 LP 原始／对偶解、基线可行性、联合风险与精确边际盒子风险，并保存可行见证。`80` 只适用于这个合成反例；真实运行必须使用其训练 provenance 记录的 reference library。浮点残差不会被直接当成精确可行性的数学证明。

`stress-case` 构造形态明确可见的状态、逐基因质量守恒的相邻细胞对，以及 null 归属可行/被预算排除的两种场景，并保存显式可行见证。它直接检验悲观目标退回基线的机制；实际训练的 NB 基线未必等于生成的 null，不能把解析反例当作 Adam 必然收敛结论。参数矩阵与使用方法见 [stress_and_selection.md](docs/stress_and_selection.md)。

```bash
morphostate reproduce --config configs/sota/ghist.yaml
```

先准备作者环境、锁定数据配置与 checkpoint，再运行外部复现。GHIST 原始 CSV 是 log1p(raw counts)，Pixel2Gene 当前作者版本真正写出的是 per-gene pickle 网格；适配器处理真实格式并显式保留单位。raw counts 不会偷偷改名为标准化 rate。详细步骤和各作者代码可获得性见 [sota.md](docs/sota.md)。

## 数据来源

| 数据 | 获取方式 | 用途 |
|---|---|---|
| [GSE250346](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE250346) | 公开 Xenium 与配准 H&E；35 供者、343 基因 | 肺纤维化及对照跨患者开发；不是肺癌或全转录组 |
| [GSE311609](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE311609) | 公开分子、计数、边界、部分 H&E/IHC | 癌症标签歧义与处理敏感性；记录数不等于患者数 |
| [MOSAIC/EGA](https://ega-archive.org/studies/EGAS50000000689) | 正式 DAC 申请 | 区域/患者级外部验证；Visium SD 不能提供单细胞真值 |

[申请材料草稿](docs/data_access_request.md)尚未提交。正交蛋白候选、已核验范围与数据局限见研究计划。

## 输出、资源与检查

单次实验保存配置、split、provenance、逐候选训练轨迹、选择结果、方法 checkpoint、测试预测、患者指标与报告。弱监督记录内层求解次数/时间、实际训练预算和背景策略。suite 保存任务清单、配置摘要、失败账本、逐患者种子平均与配对汇总；第三方执行保存每条命令日志、代码 revision 和权重哈希。

图使用稀疏边，但选定 cells×genes 标签矩阵仍会稠密化。`max_edges`、`max_label_entries`、`max_dense_elements` 和精确 marginal LP 查询上限主动拒绝过大试验。共享预算下不能任意拆块、丢分子或下采样 cells；那会改变可行集合。选基因时，去掉的基因按固定归属条件化并扣除预算，属于明确的条件分析，不等于原集合的无条件投影。

```bash
python -m pytest -q
python -m ruff check src tests
```

最新验证以 [GitHub Actions](https://github.com/chendi-Shi/morphostate-lab/actions) 的具体 commit 与 [verification.md](docs/verification.md) 为准。真实患者训练、GPU/大规模资源消耗、病理权重效果、独立蛋白验证、SOTA 作者网络及新理论仍需实际证据，不能把合成流程通过写成论文主结果。
