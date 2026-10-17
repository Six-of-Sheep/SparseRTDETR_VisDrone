# 唯一双臂准备：整图＋切片混合微调

状态：方案冻结、仅CPU准备；不授权GPU/正式训练。常规实用对照，不主张原创。
配置：`configs/research/r18_slice_finetune_v1.json`。旧A/C三seed、Soft-NMS失败关闭均不改。

## 问题与已知边界

固定C相对A的AP/APS在原3seed均通过，C/A实测4.67–4.71倍。
这支持固定切片推理的可重复实用收益，不能区分采样密度、放大、上下文、
融合/候选预算等解释。此双臂检验：同样追加训练后，显式固定切片训练是否
比仅原增强更有价值；不预设“训练/推理分布差异”已经是根因。

SAHI论文3节明确使用切片与原图共同微调、原图有助保留大目标，同时指出
截断大目标风险。本方案属于该类常规适配，采用本项目固定四窗口和原增强，
不复现其随机窗口、保持长宽比resize等全部细节。
2026-10-18核查[论文正文](https://arxiv.org/html/2202.06934)及
[作者切片代码](https://github.com/obss/sahi/blob/main/sahi/slicing.py)。
50:50、12轮和LR是预先选择的探索强度，不是文献证明的最优值。

## 实际增强与差异

沿认证种子0endpoint→binding/contract→source-manifest核对，终点源为
`ee1204a2c47f768baebfb6f93f2227cdc1ddb59c3173bbc8bfc37363a08175d4`。
main与该源的RawAugmentation/CaptureCrop逐字相同；data.py差异仅seed1/2
支持和seed透传，seed0不变。不是仅根据YAML推断实际路径。

实际：解码原图→原GT裁到画布→（epoch<118：PhotometricDistort p=.5、
RandomZoomOut默认、RandomIoUCrop另加p=.8门）→sanitize min_size1→
水平翻转p=.5→非等比resize768×1344→sanitize min_size1→归一化cxcywh。
ignore不参与裁剪采样，随同一几何变换；原损失只遮蔽普通/aux/encoder
未匹配负类，DN不改。原epoch118–120关闭前三种强增强，仍翻转。

RandomIoUCrop是GT驱动的IoU候选试探，随机位置/比例、中心纳入，再裁框。
新增固定窗口由原图尺寸和位置ID决定，不查GT，无IoU/中心/可见比例筛选，
恰为C的ceil(5H/9)×ceil(5W/9)四角半开窗口；先选视图，之后仍走原增强。
因此切片也可能再被RandomIoUCrop裁剪；不把它说成训练时完全复制推理输入。

## 唯一训练方案

| 项 | 两臂共同规则 |
|---|---|
| 父状态 | 原共同协议R18 seed0/e120的EMA.module，SHA `13a35283a02d2b728e3449917586ded5cdf2d578ebe6847510748e0dc4d21757` |
| 初始化 | 两臂分别严格载同一EMA全部参数与BN/几何buffer；不用raw-model，不继承旧optimizer/RNG/EMA计数 |
| 预算 | 12个pass×304=3648更新/臂；物理8×累积2；每次16视图；58368视图/臂；终点FT012 EMA |
| 抽样/增强时钟 | 原train_core4869、seed0原sampler的epoch109–120；每pass丢5，原增强前9强/后3弱；两臂原图顺序相同 |
| T-control | 每次16原整图视图，原增强完整复用 |
| T-slice | 每次8整图＋8切片，每个microbatch4整图4切片；TL/TR/BL/BR各2/更新；无额外读图/选区网络 |
| 混合确定性 | 全局pass位置偶数整图，奇数tile=1+(position//2)%4，不耗额外随机数 |
| 学习率 | 新局部clock；前200更新线性到1e-5，之后cosine到第3648更新1e-6；无扫描 |
| optimizer | 新AdamW，原名称分组/WD/betas/clip，moment/step清零；原两microbatch各0.5，无额外loss |
| EMA | 初值复制父EMA模型、updates=0，原decay=.9999/warmups2000；终点只用新EMA |
| RNG/BN | 独立构建后两臂重置Python/NumPy/torchCPU/CUDA seed0；原BN/frozen norm训练行为，不校准/改动 |
| 架构/监督 | 原共同R18、VFL、各头自身Hungarian、L1/GIoU、DN完全保留；无GO/pre/MAL/Soft-NMS |

58368视图中混合臂29184整图、29184切片，四切片各7296。
两臂训练输入视图数相等，不额外为每张原图运行五视图；原图重复次数由共同sampler决定。
旧已训练epoch120不是这轮未训练初态：这是清楚标注的“从已训练e120 EMA微调”对照，
不能称全新120ep配方复现，也不能将原终点计数混入新optimizer/EMA时钟。

## 标签与忽略冻结

固定裁剪直接从原RGB图和raw标注生成，不放大已缩小图。
GT/ignore与窗口求交、平移到局部；保留所有正面积交集，之后原sanitize最小1px；
不加面积比例/中心阈值、不重标类别或score，不扩大ignore豁免。
截断GT以交集为监督框；保留原stable annotation ID、物理行、原始面积/类别/遮挡截断元数据。
原面积不替换为局部面积；它不改变原回归权重。窗口外框删除，边界零面积删除。
同一GT跨视图可重复出现但原身份不变，配套记录view_id，不能当作新GT。
raw category0空间ignore、score0分类ignore用相同裁剪；category11语义不变。
被后续RandomIoUCrop移除及小于1px的GT沿用原处理，不造额外ignore区域。
空GT切片保留、不换图/重采样；原分母min1与无DN路径照旧。
窄碎片、切断大目标及更多背景负样本是风险，须由大目标/整图保护拒绝不合格结果。

## 预冻结评价与裁决

两臂各自FT012 EMA运行原A与原C，548图、Q300、FP32、主top600次top300、
原双评价器、原raw-ignore；C的窗口/候选600/view/排序/class-NMS .5不变。
预算截断→raw-ignore→评价，不补位；原A/C点值直接复用，不新增bootstrap。
AP75-small是原COCO bridge小目标面积口径；不混用诊断原图≤1024定义。

| 比较 | top600 AP | APS | AP75-small | APM/APL | top300 AP/APS |
|---|---:|---:|---:|---:|---:|
| C(T-slice)−C(T-control) | ≥+.30pp | ≥+.50pp | ≥−.20pp | 各≥−.50pp | 各≥−.20pp |
| C(T-slice)−原seed0 C | ≥+.30pp | ≥+.50pp | ≥−.20pp | 各≥−.50pp | 各≥−.20pp |
| A(T-slice)−A(T-control) | ≥−.20pp | ≥−.20pp | ≥−.20pp | 各≥−.50pp | 各≥−.20pp |
| A(T-slice)−原seed0 A | ≥−.20pp | ≥−.20pp | ≥−.20pp | 各≥−.50pp | 各≥−.20pp |

28项未舍入条件全过才记“单seed双臂实用初筛通过”，不允许均值抵消。
T-control对原父模型的C/A变化全列，帮助辨别一般追加训练收益。
官方风格另列，原分数→pp只转换一次。任一科学门槛失败仍完成两臂固定终点评价，
收尾后关闭本配方，不自动扩轮/改比例/选epoch。通过也不等于新机制或跨seed重复证明。

## 工程验收、预算与后续授权边界

本轮实际CPU入口`slice_ft_entry.py cpu-check`：固定规则/标签、两臂真实
workers2/prefetch2有限两窗、T-control像素/GT/ignore逐字对齐；认证e120载EMA且
两个新optimizer/EMA一致；原生R18 128×128合成CPU每臂1→2→恢复1重放2，
比较输入/DN/输出/loss/裁剪前梯度/完整state/RNG及跨arm/plan/SHA拒绝。
合成CPU不是生产GPU验收，更不代表真实数据训练或AP收益。

未来GPU授权后先补独立formal/监控/精简导出部署接线及新鲜资源准入；
每臂固定3更新认证（合计6），全部过后正式两臂重新载父EMA，从0训练，不能接验收state。
GPU新数值/恢复/身份/资源异常STOP、不自动重试；无进展保护与CPU125/150W、
GPU1500MHz、CPU90C/GPU80C约束保持，不设任意总墙钟终止。

成本依据：原种子0e088→120末32轮实测432.84–468.12秒/304更新，均值约456秒；
seed2完整36480更新56780.26秒，约1.557秒/更新。按同画布/批次，两臂7296
更新估约3.0–3.2GPU小时；切片目标数量/CPU读图和审计I/O可能改变耗时，
预留后训练约3–4小时，验收/双臂A+C与评价收尾再预留0.2–0.5小时，
总预算约3.2–4.5小时（非截止；不含排队/新故障）。不是把视图数当成实测倍率。
未来总更新科学成本7296+验收6；原父训练沉没成本另列。端到端推理不因缓存复用漏计。
未来同64图同batch4核对两臂C/A≤6且C(T-slice)/原C≤1.10，实际值另列。
当前无GPU实测、无质量结果；CPU实际耗时按回执记录。

完整数据/checkpoint/预测/审计留服务器；D仅报告/方案/源码索引与必要CPU结果。
不访问holdout，不恢复本机轮询；不自动启动训练。
