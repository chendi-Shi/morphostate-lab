# 状态抹除压力测试与选择性报告

本页对应 A10（悲观训练是否抹掉真实状态）和 A11（报告范围与弃权是否只是在回避难例）。所有合成输出都是软件和机制反例；不能充当真实患者验证。

## 1. A10：把会失败的情况明确构造出来

`stress-case` 生成同类型、同供者的相邻细胞对。一对内两个细胞分别增加／减少 `STATE_UP` 和 `STATE_DOWN` 的分子，交换前后每对逐基因总量守恒，每个细胞的 library 都是 80。形态特征明确包含状态信号，组成协变量只包含粗类型，因此这是监督目标的困难，而不是故意把图像信号藏起来。

```bash
morphostate stress-case --output data/stress_null_feasible --seed 7 --donors 6 --pairs-per-donor 20 --state-strength 0.5 --state-prevalence 0.25 --ambiguity 0.6
morphostate stress-case --output data/stress_null_excluded --seed 7 --donors 6 --pairs-per-donor 20 --state-strength 0.5 --state-prevalence 0.25 --ambiguity 0.6 --exclude-null
```

两次生成使用相同种子和数据参数，仅改变位移预算。第一种允许一个没有细胞内状态的 `null` 归属；第二种把位移预算设为零，排除不同于真实归属的 null。程序保存并验证真实归属与 null 归属的显式见证，不只比较两个标签矩阵的边际总量。

参数含义：

| 参数 | 实际定义 | 注意 |
|---|---|---|
| `state-strength` | 20 个基线状态分子的交换比例，离散成 0 或 1–19 个分子 | 读取 `proof.json` 的 `integer_state_shift`，不要仅报连续输入 |
| `state-prevalence` | 活跃**细胞对**的比例 | 正状态细胞约占该比例的一半；非零比例在小样本中至少产生每供者一对，实际比例另行记录 |
| `ambiguity` | 每个细胞／基因计数允许两个 owner 的比例，向下取整 | 若不足以支持指定的 null 交换，程序报错，绝不偷偷扩大集合 |
| `exclude-null` | 保持候选支持，位移预算改为零 | 它是已知反例中的受控干预，不代表真实几何预算能如此精确校准 |

输出目录包含 `data/`、`graphs/`、`truth.npz`、`state_labels.csv`、`programs.json` 和 `proof.json`。`truth.npz` 包含真实／观测／null 计数、标准化 rates、潜在状态及每个 block 的 owner 见证；这里观测计数刻意等于合成真值。

如果 null 均值 (b) 本身属于可行标签集合，相对 Poisson 最坏损失至少为 (D_{\mathrm{Poisson}}(b\Vert\mu)\ge 0)，而预测 (b) 的相对目标恰为零。因此，即使另一种真实状态可由图像预测，该目标也不能严格偏好它。`proof.json` 对生成的实例计算这一比较。**这是一条标准损失性质及其反例，不是新的泛化定理。** 实际训练得到的冻结 NB 基线未必恰等于生成器的 null，更未必属于可行集合；不能把该解析结果说成有限神经网络训练必然收敛到 null。

训练时复制 `configs/weak_demo.yaml` 为新的配置，令 `data` 和 `graphs` 指向某一个 case 的对应子目录，输出使用新目录，基因数设为 4；以相同患者划分和种子比较 `fixed`、`joint_relative`、`interval_relative`、`random_assignment`、`ensemble` 和 `aggregate`。生成参数及矩阵应在看效果前固定。建议矩阵：

| 轴 | 建议取值 | 要检查什么 |
|---|---|---|
| 状态强度 | 0.25、0.5 | 更强状态是否仍被目标压制 |
| 活跃细胞对比例 | 0.05、0.25、1.0 | 稀有状态是否比常见状态更易丢失 |
| 歧义比例 | 0.4、0.6、1.0 | 守恒可行集合变大后的退化 |
| null 可行性 | 默认、`--exclude-null` | 问题来自监督信息还是优化不足 |
| 训练／生成种子 | 7、17、29 | 分开记录生成变化和优化变化；都不是新增患者 |
| 无状态负对照 | strength=0、prevalence=0 | 是否凭空产生状态；单独报告 null 检测 |

这个矩阵是预声明建议，不会因为列在文档中就自动运行。每个 case 的规模、离散后的实际比例和失败都要保留。当前生成器不是细胞膜、背景污染或跨平台生物机制的完整模拟器，也没有连续扫描“邻居类型与状态方向夹角”的机制；不能把方向随机化冒充该因素实验。

## 2. A11：先固定状态对比，再计算可行范围

一个对比是预定义的线性计数功能量 (t_j=\sum_{i,g}a^{(j)}_{ig}Y_{ig})。系数 CSV 的列为 `contrast,cell_id,gene_id,coefficient`。一个 `contrast` 可有多行系数，但 A11 中必须只属于一个供者。不同供者的对比使用不同 ID，不能把多患者总和当一个患者再计算覆盖率。

```csv
contrast,cell_id,gene_id,coefficient
patientA_state_contrast,patientA_cell1,STATE_UP,1
patientA_state_contrast,patientA_cell1,STATE_DOWN,-1
patientA_state_contrast,patientA_cell2,STATE_UP,-1
patientA_state_contrast,patientA_cell2,STATE_DOWN,1
```

上面是文件结构示例；必须替换为当前 graph 中真实存在的细胞和基因 ID。系数选取和方向不能根据被评价的 test truth 符号反向调整。

```bash
morphostate contrast-bounds --graphs data/stress_null_feasible/graphs --coefficients data/contrasts.csv --output outputs/contrast_bounds.csv
```

输出包含 `lower,upper,fixed_assignment_value,n_donors,donor_id`。上下界是在给定 owner 集合和预算下对**同一线性对比**进行优化的极值，不是患者抽样置信区间，也不提供集合包含现实的概率保证。

独立真值 CSV 必须覆盖输出的**全部**对比，不能删掉难例：

```csv
contrast,donor_id,truth
patientA_state_contrast,patientA,12
```

`truth` 必须是完全相同系数、相同计数单位和测量尺度的数值。在合成数据中，从 `truth.npz` 的 `true_counts` 计算同一 (a^T Y)。独立测定只有在相同功能量／计数尺度得到验证后才能使用 `independent_assay`；蛋白荧光强度、Ki67 标签、预测的标准化 rate 或另一 panel 的值不能直接代替计数真值。蛋白／IF 的状态标签使用独立的 `evaluate-states` 命令。

```bash
morphostate bound-risk --bounds outputs/contrast_bounds.csv --truth data/contrast_truth.csv --output outputs/bound_risk --thresholds 0 2 5 10 20 --truth-source synthetic
```

阈值单位与对比宽度相同，不存在跨任意对比缩放都通用的阈值。若某些对比系数整体放大，应在预声明阶段统一其尺度；不能用 test truth 大小归一化后再选择阈值。程序记录输入与阈值的 SHA-256，但不会验证用户是否真的提前登记了阈值。

## 3. 覆盖率和风险的精确定义

对供者 (d) 的全部预定义对比集合 (J_d)，阈值 τ 的接受集合为

\[
A_d(\tau)=\{j\in J_d:\ U_j-L_j\le\tau,\ L_j>\epsilon_j\ \text{或}\ U_j<-\epsilon_j\},
\]

其中 \(\epsilon_j=10^{-7}(1+|L_j|+|U_j|+|t_j^{fixed}|)\) 仅用于数值容差。接受只依赖范围、宽度和预给阈值，不依赖真值或误差。跨过零的范围弃权，即使宽度很小。无接受对比的供者依然贡献覆盖率 0，其风险记为空值，不假装风险为零。

- **供者覆盖率**：\(|A_d|/|J_d|\)；报告的 `coverage` 是全部供者的等权平均。
- **供者条件风险**：接受对比中，范围确定的符号与真值符号不一致的比例。
- **报告的 `risk`**：仅在至少接受一个对比的供者中等权平均。因此必须同时读 `n_donors_with_accepted_contrasts` 和覆盖率；它不是所有供者的无条件错误率。
- **`accepted_contrast_fixed_risk`**：同一接受集合上的固定归属符号错误率，用来分开“选择了容易例子”和“范围规则真的更可靠”。
- **`all_contrast_fixed_risk`**：全部预定义对比上的固定归属符号错误率。固定值恰为零但真值非零计为符号错误。
- **`truth_containment`**：每供者真值落入给定可行范围的比例，再对全部供者平均。这是经验容纳率，不是置信水平。

真值恰为零的对比没有正／负二元符号，程序明确拒绝；应提前定义单独的 null 检测任务，而不是在本曲线运行后删除这些对比。对比应全量保存在 `all_contrasts.csv`，患者明细在 `coverage_risk_per_donor.csv`，宏平均曲线在 `coverage_risk.csv`。若所有人都被弃权，`risk` 为缺失值，接受人数和数量均为零。

阈值曲线只做描述；若要根据验证供者选择一个阈值，需要在独立测试供者上一次评价该阈值，当前命令不会自动进行这种选择。不能看同一测试曲线再挑最有利的点作为预注册主结果。
