# 零面积原始标注：限定CPU兼容性审查

2026-10-19；任务 `sahi-original-c-supplemental-20261018-001/zero-area-review-001`；PR #9。

**结论：审查原型CPU通过，建议版本化评价输入协议扩展；生产仍STOP。** 固定官方参考已规定这两条整数坐标行的行为，不建议新增零面积排除、修框或删图。现项目reader/主评价接口只接受正面积GT，故接入兼容处理仍须明确登记新版本，不能称为普通路径修复或旧协议完全不变。本轮只实现隔离审查工具，未接入生产reader、评价器、合同或启动入口；不接续补充评价。

## 原始行与逐阶段行为

原文件、全部物理行及原认证SHA保持；下表为原字节，不是训练转换标签。

| 原始位置 | 原始行（保留CRLF） | 固定参考与兼容原型的行为 |
|---|---|---|
| `9999985_00000_d_0000020.txt:12` | `611,158,4,0,1,4,0,0` | 类别4、score1、面积0；仍接受普通raw-ignore预过滤。未被过滤时GT计数/匹配分母+1，正面积预测与其IoU为0，不能匹配；COCO保留原area0，在all/small计分母，medium/large按面积忽略。 |
| `9999999_00590_d_0000267.txt:89` | `545,414,10,0,0,0,0,0` | 类别0，不计评分GT分母；`max(1,xywh)`与MATLAB闭区间栅格化形成11×2=22像素。预过滤使用原积分图公式，比例≥0.5删除GT/预测，不补位；不将原始高度改写成1。 |

文件SHA分别为 `999d84a9fcc6632645ffb9b2eca416f4e2d08a966d811d24277960e593f8c814`、`ccdc1ba4f8b821ef06539696767d1270876d0781c3db07d68eea7235e5c95155`；物理行SHA及原文本见 `returned-001/REAL_LINES.json`。两份文件执行后重新核SHA一致。

## 依据与现接口差异

VisDrone官方固定commit：`005445782213e20cb91bc50a597db3dd949e749a`。8份必要源码逐文件与仓库既有 `manifests/visdrone_det_toolkit_005445.json` 核SHA一致。

- [dropObjectsInIgr.m](https://github.com/VisDrone/VisDrone2018-DET-toolkit/blob/005445782213e20cb91bc50a597db3dd949e749a/utils/dropObjectsInIgr.m#L5-L47)：按类别0提取忽略区域；9/14行规定最小1及闭区间栅格；19–29、37–47行规定GT/预测预过滤，没有按GT面积0排除的规则。
- [evalRes.m](https://github.com/VisDrone/VisDrone2018-DET-toolkit/blob/005445782213e20cb91bc50a597db3dd949e749a/utils/evalRes.m#L54-L106)和[compOas.m](https://github.com/VisDrone/VisDrone2018-DET-toolkit/blob/005445782213e20cb91bc50a597db3dd949e749a/utils/compOas.m)：匹配保持原宽高；交集宽或高不正时重叠0。栅格路径的最小1不等于修正匹配框。
- [calcAccuracy.m](https://github.com/VisDrone/VisDrone2018-DET-toolkit/blob/005445782213e20cb91bc50a597db3dd949e749a/utils/calcAccuracy.m#L26-L39)：类别筛选后收集所有GT匹配状态，`numel(gtMatch)`进入召回分母；没有零面积剔除。官方主评价不提供APS/APM/APL。
- 项目 `src/sparse_rtdetr/localization/data.py` 的 `parse_raw` 保留物理行并标记 `nonpositive_area`；训练consumer的正面积过滤属于训练转换，不能移作评价规则。生产封存reader目前对此STOP。
- 项目 `baseline/primary_evaluator.py` 的GT校验要求正宽高且area>0；共享坐标转换也拒绝零宽高，所以现主评价直接拒绝。`r0/evaluation.py::truth`从全部raw rows构造GT；COCO桥接复用主评价的栅格/过滤helper，不能仅绕开reader而认定双评价均已兼容。
- COCO桥接应保留原始area0。实际 `pycocotools 2.0.11` 的匹配与面积分组已执行核对；`cocoeval.py` SHA `988485f79624cd42ceed9335d284df4cdcdf7b3d583e052fcbed2dd6bccf0267`，编译mask SHA见门禁。

无MATLAB/Octave：本次是固定源码审查、独立逐式CPU转写与项目算法对照，**不是官方MATLAB运行时已执行**。仅覆盖本次整数行及确定性案例，不外推任意小数/负尺寸GT兼容性。

## CPU证据

`tools/zero_area_gt_review.py`复制函数到隔离namespace验证GT非负扩展，不全局替换生产模块。原预测入口继续严格要求有限值、正宽高；GT仍拒绝负尺寸、非有限值、面积与原宽高不符。不调用模型或读取模型预测。

12案例涵盖：两真实行各自的空预测、附近预测、重叠预测、0.5边界；类别4行加合成ignore、类别0行加合成正面积GT；两完整原文件的空预测和确定性重叠预测。逐阶段GT/预测保留理由、栅格、计数和实际COCO匹配均留证。

| 最小案例 | GT计数前→后 | 预测前→后 | 结果 |
|---|---:|---:|---|
| 类别4行，空预测 | 1→1 | 0→0 | 分母1、TP0 |
| 类别4行，附近或重叠预测 | 1→1 | 1→1 | 分母1、TP0、FP1 |
| 类别4行，加合成ignore覆盖 | 1→0 | 1→0 | 普通raw-ignore删除，非面积规则 |
| 类别0行，空预测 | 0→0 | 0→0 | 栅格22像素 |
| 类别0行，附近预测 | 0→0 | 1→1 | ignore比例0，FP1 |
| 类别0行，重叠预测 | 0→0 | 1→0 | ignore比例1 |
| 类别0行，边界预测 | 0→0 | 1→0 | ignore比例恰0.5，删除 |
| 类别0行，加同处正面积GT | 1→0 | 1→0 | 正常GT也按同一栅格过滤 |
| 两完整原文件，确定性预测 | 129→129 | 2→1 | 类别4分母79、剩余预测FP1 |

另6拒绝案例通过：负/非有限GT、GT面积冲突；零/负/非有限预测。独立参考转写与项目原型的栅格、预过滤和主评价匹配计数一致；COCO执行evaluate匹配，但禁止accumulate/summarize，不保留案例AP作为科学结果。

development复用认证manifest及548份原标注SHA，共40169物理行、零条非正面积行。每图仅构造一个确定性正框，不读取旧模型预测。原接口/原型的primary输入、bridge数据/预测/计数完全一致：GT 38759→38599，预测548→545。实际 `confirmation_prepare.evaluator_inputs → primary_inputs/bridge_inputs` 在600/300两预算通过；两预算canonical SHA均 `6438c1ec0e3cea70e6300a9b00f0d0b4bf1c46132c9a4df49436fdbfb0799650`，bridge payload SHA `3b8809ac0aa2798355cf1bb590f24a58bd6ca97b886edc90bc99c343b3064897`。这证明此次接口/输入等价，**不冒充原模型预测的完整重评分证明**；未重算旧AP，未重复无关准备链。

## 唯一推荐及协议影响

建议版本名 `supplemental-zero-gt-input-extension-v1`，仅用于这次补充评价；审查原型名带 `review-v1`，不作为生产认证版本。

1. 审计层保持原始字节、行序、SHA及角色，不修框/删行/删图。生产若接入，必须显式绑定新协议/reader/GT适配身份及父证据，不改旧release或旧认证。
2. 评价GT允许有限非负宽高及精确原area=w×h；负尺寸、非有限、身份或面积冲突继续拒绝。类别4零面积行保留分母，类别0零面积行保留官方栅格意义；不为通过检查增加排除规则。
3. 双评价沿用原类别、score、匹配、预算→raw-ignore→不补位和COCO面积规则。仅GT边界扩展；预测严格正几何、数值、身份保护不变，不能共享放宽到模型输出。
4. 相对现正面积接口，这是可审查的评价输入协议扩展；相对固定官方参考没有新增零面积排除规则。删除类别4行会减分母，跳过类别0行会改变预测过滤，均不推荐。对1602全集合的实际AP影响未知，本轮未计算。
5. 本轮不实现生产适配、不新增合同/令牌，不解除原reader STOP。后续须单独接受该版本化扩展，并认证受影响真实reader/双评价入口及生产绑定后，才能在新attempt考虑原补充链；独立确认资格始终BLOCKED，历史使用、近重复、flight/site仍未知。

## 版本、成本与交付

审查工具初版 `4807bfcaf744968a421a29775ff28a378b308e0f`；COCO诊断ID的NumPy整数JSON序列化缺陷保留在CPU-001，显式转原生int并读回核对后，新审查版本 `38e92ffce23d2aab7ba6271efa8b64510f8c5049` / 375文件manifest SHA `52f9673d35a91706ee2fa6ad0f99250562900c7db61119296763c8db569c1e51`。实际development接口增量工具提交 `a9c8b3069939d360bb1649111e266827f8f09879`，字节SHA `3905bbbd386f09867887c6035c306b3a2fdfbfaffb967c111e69e7094d7bf56d`；未改变算法或生产代码。

CPU成功审查6.869621573s，实际development接口检查17.517881300s，非重叠已测新增下界24.387502873s；失败工具准备、开发/网络未全计，不记零，旧成本保留、不重复相加。GPU forward0、训练0、生产科学评分0；无后台接续或本机持续轮询。

交付目录：`workspace_management/delivery/experiments/localization-distribution/sahi-original-c-supplemental-20261018-001/zero-area-review-001`。8份必要回执/摘要91650B已回D逐SHA通过；完整235080B案例trace留服务器，SHA `1d592b29e4d9d3c6245463b3b08f3796b5e9136d65dd3d2b9b31b985b116da21`。原数据/标注/checkpoint/预测不回传。源码提交、推送、PR/CI与本次审查CPU通过单列，不等于生产准入或科学/成本门槛通过。
