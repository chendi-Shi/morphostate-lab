# MOSAIC Window 数据申请草稿

**状态：草稿，未提交、未发送，也不表示已获得数据许可。** 请由实际申请人补齐机构、PI、项目和存储条件，并以数据访问委员会（DAC）的当前表格及协议为准。

## 申请入口

[MOSAIC 官方说明](https://www.mosaic-research.com/mosaic-window)提供经 EGA 申请的流程，申请需要 Owkin 和合作方审核。数据规模为 60 位患者、5 个癌种；审核结果和周期无法预先保证。

- [总研究 EGAS50000000689](https://ega-archive.org/studies/EGAS50000000689)
- [卵巢癌 EGAD50000001704](https://ega-archive.org/datasets/EGAD50000001704)：15 位患者，公开说明含 Visium SD、单细胞 RNA、bulk RNA、H&E 和临床资料。
- [膀胱癌 EGAD50000001251](https://ega-archive.org/datasets/EGAD50000001251)：15 位患者的子队列。
- [DLBCL EGAD50000001702](https://ega-archive.org/datasets/EGAD50000001702)：10 位患者的子队列。

从与研究范围相符的具体数据集点击 **Request Access**，不要默认一次申请自动覆盖全部子队列。至少卵巢癌页面列有项目和机构范围限制及非商业使用等条件，最终应按完整 DUA 核实并履行。

## 项目摘要（可粘贴到申请表后修改）

**Proposed project title**

Evaluating composition-adjusted molecular expression prediction from histopathology across patients

**Applicant and institution**

- Principal investigator: [full name, title, institution]
- Responsible researcher(s): [names and institutional email addresses]
- Institution and department: [official names]
- Project duration: [proposed dates]
- Funding and commercial status: [actual information]
- Institutional ethics review / exemption status: [actual determination, if applicable]

**Scientific objective**

We propose to evaluate whether histopathology images provide reproducible information about molecular expression beyond broad cellular composition. The project will compare image-only, composition-only, residual regression and conditional count models using patient-separated training, validation and testing. We will examine the extent to which predictive performance is maintained across patients and assess how conclusions change when composition estimates are obtained from histology rather than measured RNA.

**Requested data**

For the approved cohort(s), we request paired H&E images, processed spatial expression counts, spatial coordinates and scaling/registration metadata, pseudonymous links between patients, tissue blocks and sections, and the minimal sample annotations needed to separate repeated samples and assess dataset shift. If available and within the approved scope, processed single-cell or single-nucleus expression and existing cell-type annotations would support reference construction and interpretation. We do not require identifiable patient information or raw sequencing reads for the initial study.

**Analysis plan**

All samples from the same patient will remain in one data partition. Data-dependent preprocessing and model selection will use training and validation patients only. We will report per-patient performance and paired differences against simple baselines. RNA-derived composition will be treated as a separately labelled oracle analysis and will not be represented as an image-only deployment input. Visium SD results will be interpreted at the spatial region or patient level; they will not be presented as direct single-cell spatial validation.

**Data management and outputs**

Data will be stored in [institution-approved location] with access limited to [authorized researchers]. Retention, deletion, permitted computation environments, sharing and publication procedures will follow the executed data use agreement. We plan to release analysis code and permitted aggregate results, without redistributing controlled data or patient-level derived artifacts unless explicitly allowed. Publications will acknowledge the data contributors as required.

## 申请前应补齐的具体信息

1. 选择实际需要的癌种和分析单位。当前公开肺部 Xenium 实验与 MOSAIC 卵巢癌不是相同疾病外部复现；应明确是方法可迁移性研究还是另一个独立队列实验。
2. 说明所需模态是否确实来自同一患者、同一组织块或同切片，并请求对应关系与重复样本标识。
3. 确认图像和空间坐标是否已配准，是否提供 Space Ranger 输出、比例因子及样本级 QC。
4. 填写真实的 PI、机构资质、计算地点和访问人员，不得用占位信息直接提交。
5. 核查获批范围是否允许 GPU/云环境、派生特征留存及聚合结果发布；按实际协议执行，不在代码仓库保存协议附件、账户凭据或患者对应表。

GSE311609 和 GSE250346 的公开处理数据可先用于本地可行性实验；正式受控申请无需成为代码框架开发的前置条件。
