# 数学定义、可证明性质与适用边界

本文件针对 `budgeted.py`、`geometry.py`、`weak_training.py` 与 `stress.py` 实际实现给出推导。以下结论主要是线性规划、凸分析和标准集中不等式在本问题中的应用，**不声称是新的理论贡献，也不能据此认定达到 CCF A 的创新标准**。它们用于核对实现、界定能够证明的结论，并揭示方法可能失败的原因。

特别需要区分：可行标签集合上的确定性保证、神经网络优化结果、以及新患者上的统计泛化保证。这三件事互不等同。

## 1. 对齐代码的符号与假设

考虑一个不能被拆开的患者纯净 graph block。它包含：

- $n$ 个真实细胞、一个显式背景 compartment，以及 $G$ 个基因；
- $T$ 个转录本观测或合法聚合观测，观测 $t$ 的基因为 $g_t$，质量为 $m_t\geq0$；
- 候选边集合 $E=\{(t,i):i\text{ 是 }t\text{ 的候选归属}\}$，每个观测至少一条边。

这里的质量允许分数。一个质量为 10 的聚合观测可以在候选归属之间分配为 3.5 和 6.5；具有共享预算时，LP 最优解也不必对应逐分子整数归属。因此，下面讨论的是**连续质量分配模型**。

令 $z\in\mathbb R^{|E|}$ 表示每条边实际分得的质量，而不是归属概率。观测—边关联矩阵 $S\in\{0,1\}^{T\times|E|}$ 满足

$$
S_{t,e}=\mathbf 1\{e\text{ 属于观测 }t\}.
$$

代码中的质量守恒约束为 $Sz=m$。几何成本向量 $c\geq0$ 与背景指示向量 $a_{\rm bg}\in\{0,1\}^{|E|}$ 分别对应

$$
c^Tz\leq d_{\rm geom},\qquad
a_{\rm bg}^Tz\leq d_{\rm bg}.
$$

省略某个预算时，也省略相应约束。统一写成 $Cz\leq d$，其中 $C$ 可以没有行。几何构图的默认边成本是归一化中心距离平方；它是明确声明的 centroid proxy，不是经证明的细胞边界模型。

定义边质量可行集

$$
\mathcal Z=\{z\geq0:Sz=m,\ Cz\leq d\}.
$$

令 $\widetilde Pz$ 为包含背景在内的完整 cell-by-gene 标签，令 $Pz$ 只保留真实细胞坐标。全文主标签集合为

$$
\mathcal U=\{Pz:z\in\mathcal Z\}\subset\mathbb R_+^{p},\qquad p=nG.
$$

因此，**$b\in\mathcal U$ 表示真实细胞的 baseline 计数均值存在某个可行背景扩展**，不要求背景标签为零。代码的 LP 包含背景列，但损失给背景的系数为零；这一点正好对应投影 $P$。

对真实细胞—基因坐标 $j$，固定 exposure $s_j>0$，baseline 和候选模型的计数均值分别为

$$
b_j=s_jr_{b,j}>0,\qquad \mu_j=s_jr_{f,j}>0.
$$

同一细胞的基因使用相同 exposure。**这里是乘过 exposure 的计数均值，不是 checkpoint 输出的 reference-library rates。** `weak_training.py` 固定使用原始分割 library / 训练患者的 median library 作为 exposure；内层改变标签归属时不重算 exposure。下文的证明均保持这个 offset 固定，不是重新分配分子后再估计细胞总 RNA 的联合模型。

令 $w_j>0$，$W=\operatorname{diag}(w)$。一个 block 的 `poisson_assignment_loss` 使用 $w_j=1/(nG)$。跨 block 的训练权重为 $n_k/(D N_{d(k)})$，其中 $D$ 是训练患者数、$N_{d(k)}$ 是该患者全部训练细胞数；乘上 block 内均值后，每个真实细胞—基因的总体权重是

$$
w_j=\frac{1}{D N_{d(j)}G}.
$$

block 内所有坐标同时乘同一个正权重不改变 LP 最优归属。不同 block 不共享预算，因此完整训练标签集合是 block 集合的直积。以下证明可以逐 block 应用，也可以对拼接后的集合与患者权重应用。

## 2. 命题一：可行标签集合紧、凸，支持函数可由 LP 对偶验证

**假设：**候选图有限、质量有限、每个观测至少一个候选，且 $\mathcal Z\ne\varnothing$。最后一个条件不是自动成立的；两个分别合理的预算可能联合不可行，构造器会显式调用 LP 验证。

由于 $z\geq0$ 且 $Sz=m$，每条属于观测 $t$ 的边满足 $0\leq z_e\leq m_t$。所以 $\mathcal Z$ 有界；它由线性等式、不等式定义，故闭且凸。有限维空间中闭且有界意味着紧。线性映射保持凸性，连续映射把紧集映为紧集，因此 $\mathcal U=P\mathcal Z$ 是非空紧凸多面体。

完整标签严格守恒：对任一基因 $g$，

$$
\sum_{i=1}^{n+1}\widetilde Y_{ig}
=\sum_{t:g_t=g}m_t.
$$

最后一个细胞是背景。若分子可以进出背景，**真实细胞标签之和不一定恒定**；不能删除背景项后继续声称真实细胞总量守恒。

对任意系数 $v\in\mathbb R^p$，支持函数是

$$
h_{\mathcal U}(v)=\max_{y\in\mathcal U}v^Ty
=\max_{z\geq0}\{(P^Tv)^Tz:Sz=m,\ Cz\leq d\}.
$$

其对偶为

$$
\boxed{
\min_{\alpha\in\mathbb R^T,\ \lambda\geq0}
\quad m^T\alpha+d^T\lambda
\quad\text{s.t.}\quad S^T\alpha+C^T\lambda\geq P^Tv.
}
$$

符号可以直接从最大化问题的 Lagrangian 核对：

$$
\mathscr L(z,\alpha,\lambda)
=m^T\alpha+d^T\lambda+
\big(P^Tv-S^T\alpha-C^T\lambda\big)^Tz.
$$

这里 $\alpha$ 对应等式，符号自由；$\lambda\geq0$ 对应 $Cz\leq d$。对 $z\geq0$ 求上确界有限，恰好要求括号内每个分量不大于零。取 $\lambda=0$、每个 $\alpha_t$ 大于等于该观测候选边的最大目标系数，即可构造对偶可行点。

原问题非空紧且目标有限，因此最优值有限并达到；有限维 LP 强对偶给出原、对偶最优值相等，且这里双方均存在最优解。**不需要预算具有严格内点**，零背景预算等退化情况也在 LP 强对偶范围内。浮点求解器仍需检查状态、守恒残差、预算残差和对偶间隙；数学定理不是忽略数值失败的理由。

这是标准 LP 对偶与凸集性质的应用，可参见 [Boyd 与 Vandenberghe 的作者版教材](https://web.stanford.edu/~boyd/cvxbook/)。

## 3. 命题二：联合集合比精确边际区间盒子更小，但不必带来生物学优势

为每个真实细胞—基因坐标定义**精确**边际界

$$
\ell_j=\min_{y\in\mathcal U}y_j=-h_{\mathcal U}(-e_j),\qquad
u_j=\max_{y\in\mathcal U}y_j=h_{\mathcal U}(e_j),
$$

并定义 $\mathcal B=\prod_j[\ell_j,u_j]$。由定义，$\mathcal U\subseteq\mathcal B$，所以

$$
h_{\mathcal U}(v)\leq h_{\mathcal B}(v)
=\sum_{j:v_j\geq0}v_ju_j+\sum_{j:v_j<0}v_j\ell_j.
$$

取相同模型、baseline、exposure 和权重，联合标签集合上的相对 worst-case risk 不大于区间盒子的 risk。这个比较要求 interval baseline 使用同一 $\mathcal U$ 的**精确边际界**；代码 `count_bounds(exact=False)` 可能忽略共享预算，返回更松的界，不能冒充“相同边际、仅移除联合依赖”的消融。

严格差距可由两个真实细胞、一个基因、一个质量为 1 的两归属观测构造：

$$
\mathcal U=\{(t,1-t):0\leq t\leq1\},\qquad
\mathcal B=[0,1]^2.
$$

背景列可以保留但没有候选边。对 $v=(1,1)$，有 $h_{\mathcal U}(v)=1<2=h_{\mathcal B}(v)$。区间盒子允许同时把同一个分子给两个细胞。

具体到相对目标，取未归一化权重 $w=(1,1)$、$b=(1,1)$、$\mu=(e^{-1},e^{-1})$，则

$$
R_b^{\mathcal U}(\mu)=2/e-1<0,
\qquad
R_b^{\mathcal B}(\mu)=2/e>0.
$$

若按两个坐标取均值，两边同时除以 2，严格差距保留。

**仅靠聚合总量仍不足以发现共享预算限制的例子。**考虑两个细胞、两个基因，每个基因各有一个质量为 1 的两归属观测。只有边“基因 1 → 细胞 1”和“基因 2 → 细胞 2”成本为 1，其余成本为 0，总预算为 1。可行标签满足每列和为 1，且 $Y_{11}+Y_{22}\leq1$。取

$$
b=\begin{pmatrix}0.8&0.2\\0.2&0.8\end{pmatrix}.
$$

它满足每个基因的聚合总量，也在精确边际盒子 $[0,1]^4$ 内，但成本为 1.6，故不属于联合集合。令 $v$ 的两个对角元素为 1、非对角为 0，取未归一化权重，则 $h_{\mathcal U}(v)=1$、$v^Tb=1.6$。沿 $\mu(t)=b\odot\exp(-tv)$ 有

$$
R_b^{\mathcal U}(\mu(t))=1.6(e^{-t}-1)+t,
$$

在 $t=0$ 的右导数为 $-0.6$，所以小正步长可以严格改进。另一方面，盒子包含 $b$，由命题四，其相对 worst-case risk 对所有候选均值都非负。这个例子隔离了共享预算的信息，不能仅用“每个基因总量守恒”解释。

这些例子证明集合结构可以改变优化结论；**并不证明真实归属属于所选集合，也不证明联合训练优于其他方法的真实患者表现**。独立区间监督已有直接前作，例如 [NeurIPS 2025: Learning from Interval Targets](https://proceedings.neurips.cc/paper_files/paper/2025/hash/570641fd4b0340cc8ba6320955de891f-Abstract-Conference.html)；不能把区间到联合约束的形式变化本身当作已成立的新理论贡献。

## 4. 命题三：相对 Poisson 目标与 detach 内层最优解

对固定标签 $y\geq0$，忽略与模型无关的项，Poisson loss 为

$$
L(\mu,y)=\sum_jw_j(\mu_j-y_j\log\mu_j).
$$

对整数标签，完整 NLL 还含 $\log(y_j!)$，但在同一标签上与 baseline 相减时该项严格抵消。对分数质量，可以使用相同 Poisson quasi-loss 差；下面的代数关系不要求 $y$ 是整数。

于是

$$
\boxed{
R_b(\mu)
=\max_{y\in\mathcal U}[L(\mu,y)-L(b,y)]
=\sum_jw_j(\mu_j-b_j)
+h_{\mathcal U}\big(W(\log b-\log\mu)\big).
}
$$

这对应代码 `joint_relative`，而 `joint_absolute` 最大化的是省略 factorial 的绝对 quasi-loss，**不是**完整 Poisson NLL 的 worst-case 值：factorial 对未知标签并非常数。

令 $\eta=\log\mu$，定义

$$
F(\eta,y)=\sum_jw_j(e^{\eta_j}-b_j)
+\sum_jw_j(\log b_j-\eta_j)y_j.
$$

在 $\mathcal U$ 固定且非空紧的前提下，$F(\cdot,y)$ 光滑凸，$R_b(e^\eta)=\max_yF(\eta,y)$ 有限、连续且凸。设

$$
\mathcal A(\eta)=\arg\max_{y\in\mathcal U}F(\eta,y).
$$

Danskin 定理给出

$$
\partial_\eta R_b(e^\eta)
=\operatorname{conv}\{W(e^\eta-y):y\in\mathcal A(\eta)\}.
$$

因此，任取 LP 返回的精确最优标签 $y^*$，

$$
g=W(\mu-y^*)
$$

就是 log-mean 空间的合法凸次梯度。将 $y^*$ detach 后对 $F(\eta,y^*)$ 反向传播，得到的正是这个量，**不需要对 LP 求解器或归属选择求导**。若内层只得到近似最优解，则应另行报告最优性误差，不能直接称为精确次梯度。

存在多个最优标签时，目标可能不可微，不能随意称为“梯度”。在 kink 处任选合法次梯度也不保证一次有限步更新使目标下降。若最大化标签虽不唯一但都给出相同真实细胞梯度，目标仍可能可微。

对模型参数 $\phi$，若 $\eta(\phi)$ 在当前点邻域光滑，则选中的链式向量为

$$
J_\eta(\phi)^T W(\mu-y^*).
$$

由于有限多面体的线性最大化可以写成有限个顶点对应的光滑函数之最大值，该向量在这些光滑条件下属于参数目标的 Clarke 广义梯度。**这不使参数目标变凸，也不提供全局最优保证。**实际低秩 encoder/loading 是双线性参数化；使用 MLP、ReLU 和 clamp 后还会出现额外非光滑点。在这些点，自动微分的分支导数需要额外链式正则性才能被严格解释为 Clarke 次梯度，不能无条件宣称任意计算图次梯度都满足同一个定理。

此外，固定预算是本推导的必要条件：如果训练时让 $\mathcal U_\phi$ 随模型参数改变，不能继续忽略集合边界对导数的影响。本实现的预算由输入图固定。关于 Danskin 定理可参见 [Bertsekas 作者版《Convex Optimization Theory》附录 A.3.2](https://web.mit.edu/dimitrib/www/Convex_Theory_Entire_Book.pdf)。

## 5. 命题四：baseline 可行时，悲观相对目标必然偏好 baseline

**假设：**所有真实细胞权重严格为正，$b_j>0$，且 $b\in\mathcal U$。因为 $b$ 本身是内层可选标签，

$$
\begin{aligned}
R_b(\mu)
&\geq L(\mu,b)-L(b,b)\\
&=\sum_jw_j\left[\mu_j-b_j+b_j\log\frac{b_j}{\mu_j}\right]\\
&=:D_w\big(\operatorname{Pois}(b)\,\|\,\operatorname{Pois}(\mu)\big)
\geq0.
\end{aligned}
$$

最后一步来自 $r-1-\log r\geq0$。严格正权重、正 $b$ 下，等号仅在 $\mu=b$ 时成立；同时 $R_b(b)=0$。因此，在包含 baseline 的预测集合上，**唯一最优计数均值是 $b$**，虽然实现这一均值的神经网络参数可能不唯一。

这不是“安全且能恢复真实状态”的证明，而是一个明确失败条件：即使图像携带非常清晰的真实状态信号，只要未排除同样可行的 null baseline，最坏情况目标就没有理由接受该状态。

`stress.py` 显式构造同一可行集合内的真实状态归属与 null 归属，验证每对细胞每个基因的总量以及每个细胞 library 均不改变，保存 null assignment witness，并计算上述 KL 下界。该反例说明**固定有限标签集合上的目标偏好**，不证明任意神经优化运行都收敛到 baseline，也不是对真实生物过程的经验断言。

## 6. 命题五：baseline 不可行时的 converse，以及受限 head 的正确条件

### 6.1 全部 log-means 自由时：不可行等价于存在严格改进

令 $\mu=e^\eta$，允许 $\eta\in\mathbb R^p$ 任意变化。由于 $\mathcal U$ 非空紧凸，若 $b\notin\mathcal U$，严格分离定理保证存在 $v$ 满足

$$
v^Tb>h_{\mathcal U}(v).
$$

取方向 $d=-W^{-1}v$，从 baseline 出发 $\eta(t)=\log b+td$，$t>0$。利用支持函数的正齐次性与指数函数展开，

$$
\begin{aligned}
R_b(e^{\eta(t)})
&=\sum_jw_jb_j(e^{td_j}-1)+t h_{\mathcal U}(-Wd)\\
&=t\big[-v^Tb+h_{\mathcal U}(v)\big]+o(t).
\end{aligned}
$$

一阶项严格为负，所以存在足够小的 $t>0$ 使 $R_b(e^{\eta(t)})<0$。结合命题四可得

$$
\boxed{
b\notin\mathcal U
\iff
\exists\mu>0:\ R_b(\mu)<0
\quad\text{（全部 log-means 自由）}.
}
$$

这只是存在性结论，不说明有限样本上哪个神经网络可以表达该方向，也不说明优化器能找到它。

### 6.2 固定仿射 log-mean head：必须检查投影后的集合

现在只允许

$$
\log\mu=\log b+A\theta,
\qquad A\in\mathbb R^{p\times q}\text{ 固定},\quad \theta\in\mathbb R^q.
$$

这里必须包含 $\theta=0$ 的 baseline，且不能附加未说明的参数边界或正则项。定义投影集合

$$
\mathcal K=\{A^TWy:y\in\mathcal U\}.
$$

参数目标是凸函数，在 $\theta=0$ 处所有标签都是 active maximizer，并且

$$
\partial_\theta R_b(e^{\log b+A\theta})\big|_{\theta=0}
=\{A^TW(b-y):y\in\mathcal U\}.
$$

于是

$$
\boxed{
\exists\theta:R_b(e^{\log b+A\theta})<0
\iff A^TWb\notin\mathcal K.
}
$$

若投影在集合内，存在 $y_0\in\mathcal U$ 使 $A^TWy_0=A^TWb$。直接代入该标签即可得到对任意 $\theta$，

$$
R_b(e^{\log b+A\theta})
\geq\sum_jw_jb_j\left[e^{(A\theta)_j}-1-(A\theta)_j\right]\geq0.
$$

若投影不在集合内，对 $\mathcal K$ 应用严格分离并沿相反参数方向作一阶展开，就得到严格负目标。因此，仅有 $b\notin\mathcal U$ 对受限 head **并不足够**。

例如 $\mathcal U=\{(1,1)\}$、$b=(1/2,3/2)$、$W=I$，若 head 仅允许共同 log-scale $A=(1,1)^T$，则 $b\notin\mathcal U$，但 $A^Tb=2=A^T(1,1)$，不存在上述严格改进。

实际低秩双线性 head 不能直接当成“固定 $A$”模型。若只使用当前点的 Jacobian $J$，投影不在 $\{J^TWy\}$ 中可以在光滑条件下给出一个局部一阶下降方向；投影在其中只排除这类一阶分离方向，**不能推出整个非线性模型的全局最优为 baseline**。clamp、权重衰减、容量约束和提前停止也不包含在自由 log-mean converse 中。

## 7. 命题六：条件于集合正确的经验改进证书

若真实标签 $y^{\rm true}\in\mathcal U$，则对任何固定模型

$$
L(\mu,y^{\rm true})-L(b,y^{\rm true})\leq R_b(\mu).
$$

所以 $R_b(\mu)\leq0$ 确实能保证：对当前 block、固定 exposure 和权重，真实标签上的**加权总损失**不比 baseline 差。这不保证每个细胞、每个基因或每个稀有状态都改进。

这个证书依赖真实标签在集合内。候选半径漏掉真实归属、错误的 `is_core` 标注、过紧的几何/背景预算，都会使前提失效。按基因子集固定未选分子的原归属时，证明也只针对该条件集合；其真值覆盖需要额外假设。

当前训练历史记录的是更新过程中不同参数上的 batch objective；不能把最后一条训练历史直接当成最终固定模型的全样本证书。要报告证书，应锁定模型后，以原始完整 blocks、相同固定权重重新求解并汇总全部内层最优值，同时检查数值误差与可行性假设。

该结论是确定性、条件性、经验性的。训练集上成立不推出新患者上成立，也不说明预测识别了真实生物状态。代码默认 checkpoint/capacity selection 使用原始 fixed-owner validation deviance；它与这个 robust certificate 是不同量。

## 8. 可选命题：独立患者、固定有限候选集的泛化界

下面是可以用于未来验证设计的标准 Hoeffding + union bound，**不是当前实验已经获得的总体保证**。

条件于一份独立训练集 $\mathcal T$，预先冻结 $K$ 个模型、共同 baseline、以及构图和预算规则。令 $Z_1,\ldots,Z_D$ 为独立同分布的验证患者；每位患者可以包含任意相关细胞，单位始终是患者。对模型 $k$，令 $r_k(Z)$ 为按患者归一化的相对 robust loss，并假设已知

$$
-B\leq r_k(Z)\leq B\quad\text{几乎处处，所有 }k.
$$

写作 $\widehat r_k=D^{-1}\sum_dr_k(Z_d)$、$\bar r_k=\mathbb E[r_k(Z)\mid\mathcal T]$。单侧 Hoeffding 不等式及对 $K$ 个模型取 union bound 给出，以至少 $1-\delta$ 的条件概率同时成立

$$
\boxed{
\bar r_k\leq\widehat r_k+
2B\sqrt{\frac{\log(K/\delta)}{2D}},\qquad k=1,\ldots,K.
}
$$

若额外有 $Y^{\rm true}(Z)\in\mathcal U(Z)$ 几乎处处，则模型 $k$ 在新患者真实标签上的期望相对损失不大于右侧。因为界对所有冻结候选同时成立，也可以用于从这个固定有限集合中选择的候选；但必须实际计算该候选的 robust validation loss。

适用限制包括：

- 训练集独立，候选模型/几何预算规则不能根据这批验证患者不断新增或调整。数据依赖的 checkpoint/架构搜索若未被事先声明的有限候选集合完整涵盖，不能简单代入一个事后较小的 $K$。
- 同一患者多张切片不能被当成独立患者；患者数是 $D$，不是细胞数。患者之间分布不同或未来中心发生漂移时，该 i.i.d. 界不自动适用。
- $B$ 必须是有依据的统一上界。只 clamp 网络 log-rate 不够，因为 exposure、分子质量及患者分布可能无界。若有 $\sum_jw_j=1$、$\lvert\mu_j-b_j\rvert\leq M$、$\lvert\log(\mu_j/b_j)\rvert\leq H$、以及 $\sup_{y\in\mathcal U}\sum_jw_jy_j\leq C_0$，则可以取 $B=M+HC_0$。这些条件需要在数据范围和建模规则中另行证明，不能从一次运行的最大值推断总体上界。
- 当前主实验使用 fixed-owner validation deviance，未自动计算上述 robust validation bound；训练过程中也有提前停止和容量选择。因此此命题不能直接认证当前实验。
- 最关键的“真实归属属于集合”的几乎处处条件目前没有统计覆盖证明。若覆盖未知，不能把该公式当成总体安全或临床可靠性保证。

## 9. 命题七：线性对比界是条件识别界，不是统计置信区间

对预先声明的线性对比 $a^Ty$，紧性保证以下上下界均可达到：

$$
\boxed{
\underline\tau=-h_{\mathcal U}(-a),\qquad
\overline\tau=h_{\mathcal U}(a).
}
$$

它们对**给定可行集模型**是 sharp bounds；不存在更窄且仍覆盖该集合全部标签的区间。若两端相等，该对比在集合内点识别；若不等，则该集合允许多个对比值。

但这不是 95% confidence interval，没有重复抽样覆盖率参数；“sharp”也不表示候选图或预算正确。真实标签不在集合内时，真值可以落在区间外。患者 bootstrap 描述另一个层面的患者抽样不确定性，不能代替归属集合的覆盖验证，也不能与这些确定性区间混称。

## 10. 可运行的数学诊断

`theory.py` 和 `theory-diagnostics` 命令将部分数学条件变为可检查的数值诊断：

- LP primal/dual objective、对偶符号、可行性残差与 gap；
- baseline 计数均值到可行标签集合的距离/可行性见证；
- 同一均值对上的联合与精确边际盒子 risk；
- 在 baseline 仅近似可行时，用见证误差修正的 KL 下界。

例如，默认 stress fixture 的所有细胞 library 为 80，因此可以直接运行：

```sh
python -m morphostate stress-case --output data/stress-proof
python -m morphostate theory-diagnostics --data data/stress-proof/data --graphs data/stress-proof/graphs --predictions data/stress-proof/truth.npz --model-key true_rates --baseline-key null_rates --reference-library 80 --output runs/stress-theory
```

真实实验必须使用该预测模型记录的训练 reference library，不能用测试患者重新估计。诊断要求完整 graph blocks；不能在只预测了一部分 block 细胞时，默默删除其他细胞来放松原有预算。

若可行见证 $y_0$ 只有 $\lVert y_0-b\rVert_\infty\leq\varepsilon$，则正确的近似下界是

$$
R_b(\mu)\geq D_w(b\|\mu)
-\varepsilon\,\lVert W(\log b-\log\mu)\rVert_1.
$$

不能仅因为 LP 在某个 tolerance 内返回“近似可行”，就无修正地套用精确的 $R_b\geq D_w$。这些 diagnostics 是浮点数值见证，而不是形式化证明，也不验证真实归属确实被输入集合覆盖。

## 11. 理论真正支持什么

这些推导支持：求解器和梯度可以被检验；联合约束与区间外包络之间存在可证明差异；baseline 可行性构成一个明确且可构造的失败条件；经验改进证书与识别界都有清晰前提。

它们尚不支持：新的可识别性定理已经完成、真实标签集合具有统计覆盖、实际非凸训练达到全局最优、或者方法已经优于 SOTA。要形成独立研究贡献，需要针对这些现有理论无法直接解决的部分提出新的可验证结果，并用真实患者与正交状态证据检验，而不是把标准凸分析重新命名为新方法理论。
