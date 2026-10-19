# 数据条件与方法来源表

> revision-002补充：新增ISPRS2022扩展版15页已核身份并读方法、构建和实验正文；[逐项审查与页码](NWD_EXPANDED_REVIEW_20261019.md)为当前补充依据。初审保留在`acb6d8487d664fa02588af0a6097e0159b195630`；本修订不改任何冻结实验，仍保留0候选。

日期2026-10-19；对应[一页决定](PUBLIC_DATA_TOPIC_DECISION_20261019.md)与[来源索引](PUBLIC_DATA_TOPIC_SOURCES_20261019.json)。这是有界审查，不是截至今日的穷尽查新；“不足以保留本轮候选”不等于所有单次整图方向均无价值。

## 两个外部数据集

|核查项|AI-TOD家族：条件首选v2|UAVDT：本轮不选|
|---|---|---|
|官方来源|[AI-TOD作者仓库](https://github.com/jwwangchn/AI-TOD)；v1论文Tiny Object Detection in Aerial Images，Wang等，ICPR2021。v2对应Xu等ISPRS2022扩展论文；原文入口下列|[作者项目页](https://sites.google.com/site/daviddo0323/projects/uavdt)本轮访问超时，没有取得可用官方正文|
|使用条件|作者v1 README明示数据CC BY-NC-SA4.0、研究使用；代码MIT另列。原图合成需xView训练数据＋AI-TOD_wo_xview，底层xView条款和v2包的授权范围未单独核完；不能把v1声明当全部来源权利认证|条款未知；未用镜像页或“公开下载”推定可用性|
|版本与成员|v1作者README：28,036图、700,621框、均尺度12.8px。扩展版§3.1确认v2同28,036图及8类、重标注；均尺度12.7px。正文752,745与引言／Table1的752,754有冲突，未验下载包manifest；不自行修正，v1/v2标注不能混用|未核实；不填写未经本轮官方来源支持的计数|
|类别／标注|v1作者NWD适配器`mmdet/datasets/aitod.py`实际CLASSES：airplane、bridge、storage-tank、ship、swimming-pool、vehicle、person、wind-mill；COCO式水平框。v2实际category ID、area、ignore/crowd仍须对照其认证标注；不能把vehicle等同VisDrone某一类|类别、遮挡／忽略规范未知，不能直接沿用VisDrone标签|
|原划分|扩展版确认trainval训练、test评价，Table1是实例数，未提供精确图像名单／父图split依据。11,214／2,804／14,018仍是DQ-DETR报告，不能改称本次原文确认；官方manifest未认证|原划分／视频序列关联未知|
|尺度和单位|扩展版§3.2支持约86%小于16px、均值12.7／SD5.6，Fig6仍有large项；撤回二手“全部最大不超过64px”的严格保证。这些是原发布尺度，不是768×1344变换后尺寸。v1作者图像生成器明确subimage_size800、gap200，按名单生成xView块；不能将发布块称未裁切原航拍大场景。v2继承同图已由原文确认，具体lineage仍待认证|目标尺度分布未知；不能只由“航拍”推定与本题同难度|
|评价工具|[作者cocoapi-aitod](https://github.com/jwwangchn/cocoapi-aitod)，已读入口及cocoeval前120行；完整参数末段未读到，raw源访问超时。另已读NWD完整适配器：`proposal_nums=(100,300,1500)`写入maxDets、IoU .50:.05:.95、vt/t/s/m输出。扩展版确认AP .50:.05:.95／vt/t/s/m、AR1500，但还描述预分数.05、NMS.5、top3000。后处理与评价maxDets分列；尺度桶精确端点和v2固定生效代码尚未认证，不签发正式评价配置|工具版本／测试服务器未知，未自行换COCO工具充当官方协议|
|测试可用性|v1作者页宣布完整标注和train/val/trainval/test公开；DQ作者仓库提供v1/v2整理包入口。未下载，不保证当前所有链接、v2测试GT或成员完整。公开test须人为保持开发边界|测试图像／GT／提交服务器是否可用未核实|
|选择理由|适合检验“极小目标的一次图像推理”而非再证明切片有效；多类和密度变化可限定问题。**条件首选，不是已完成正式准入或证明独立**|当前权利与验证边界证据弱于AI-TOD；本轮不扩展到第三集合|

扩展版§3明确底层来源包括DOTA、xView、VisDrone、Airbus、DIOR，v2继承v1图像。重标注分别在trainval和test训练检测器作提示，属于基准构建，不能称test从未被模型使用；不据此断言本项目泄漏。AI-TOD数据制作源码[generate_aitod_imgs.py](https://github.com/jwwangchn/AI-TOD/blob/master/aitodtoolkit/generate_aitod_imgs.py)采用`split_image(...subsize=800,gap=200)`及每split固定名单。它提示应检查父图／重叠块关系，**本轮未证实跨split存在或不存在泄漏**。还须区分同源公开图像与本项目VisDrone／公共预训练成员是否重叠；只有图像SHA不重叠不能证明飞行／地点独立。若元数据不足，限制结论，不临时重划以制造“独立”集合。

## 两个方向的具体审查（均不进入实现）

### 方向1：仅在训练匹配成本中引入NWD

问题是假设微小GT在Hungarian分配中因IoU项的像素敏感性得到不稳定监督。候选的最具体形式是保留原分类及L1项，将几何匹配项替为`1-exp(-sqrt(W2_squared)/C)`；W2_squared为(cx,cy,w/2,h/2)表示的平方欧氏距离，C只能由训练数据统计固定。这里是**待否定的常规移植设想**，不是冻结配置：本轮不选择C、系数或新门槛，不把它写入代码。GT只供训练，不进入部署。

最近已读原文：[A Normalized Gaussian Wasserstein Distance for Tiny Object Detection](https://arxiv.org/html/2110.13389v2)，Wang、Xu、Yang、Yu，arXiv2110.13389v2（2022-06-14）；读完§3公式、§4分配／回归／NMS消融，而非仅摘要。已读作者[官方代码](https://github.com/jwwangchn/NWD)根README和数据适配器；未审完整NWD算子实现。它主要论证anchor分配的正样本问题，不能据此认定R18一对一匹配同因。

- **实质差异**：上述设想与原文仅是部署在一对一匹配接口，不足以独立构成方法增量；更没有证明原R18确有该故障。扩展版正文现已取得并读完，其NWD-RKA直接覆盖每GT排名补样本、尺度监督不平衡与IoU为0的补偿；正文未做R18并不构成研究空白。
- **最近常规控制**：原Hungarian成本与仅移植原NWD成本；训练日程、输入／Q／optimizer／loss不混改。若以后存在真正候选，原NWD控制不能省略。不要把loss、assignment、NMS同时改再宣称唯一因素。
- **最小否定条件（仅说明）**：在同一数据原train/val及同初态、更新预算下成对训练两臂；若训练期预注册的小框匹配稳定性改善没有兑现成最终定位／检测收益及尺度保护，否定该“匹配即可改善”假设。仅几何覆盖增加、总loss变小或一次AP正值均不足。本轮不新增匹配统计或评分。
- **成本**：至少2个完整训练运行单位／数据集用于常规双臂，而非缓存重评分；匹配仍有Q×GT距离与Hungarian成本，墙钟未知。推理不带匹配，理论上仍同一单视图网络，但实际耗时不能未经测量写成完全相同。NWD官方旧MMCV自定义算子环境不直接安装进冻结环境。
- **旧记录**：MAL是分类损失，固定混合微调是增强对照，都不等于已否定NWD匹配；它们的失败也不构成此方向有效的证据。Soft-NMS／框投票等闭合配方不重开。决定：**不保留**。

### 方向2：计数引导query分配

实际问题是假设Q300在高密度发布图像上限制覆盖，而稀疏图使用更多query浪费。具体可用输入是当前整图encoder特征，由训练GT计数监督的head估计密度；推理只读图像／特征，不能读GT或切片输出。最具体的现成方案是DQ-DETR的四档query数量300/500/900/1500，配合计数引导位置选择；这描述**原方法**，不是本项目新算法或执行计划。

最近已读原文：[DQ-DETR: DETR with Dynamic Query for Tiny Object Detection](https://arxiv.org/html/2404.03507v6)，Huang、Liu、Shuai、Cheng，arXiv2404.03507v6（2024-11-01），ECCV2024论文，Springer卷出版年2025。已读§3三模块、§4.1–4.4数据／训练／固定与动态query消融及附录。[官方代码](https://github.com/hoiliu-0801/DQ-DETR)已开放，root所见commit `c5246797b111d7e5da9e6c20667fe0901aae35ba`；仅README／组织核查，不冒称完成代码复现。它已同时研究数量、位置、密度增强及分组验证；不是本文可以重新提出的空白。

- **实质差异**：单改R18、固定Q或另设计数阈值均不足。若做固定Q300空间配额，也与DDQ去冗余及本项目旧query选择问题邻近；目前没有区别现成规则的可靠信号／目标函数。不能用“轻量化”名称代替差异和实测。
- **最近常规控制**：原固定Q300、同上限的固定较大Q、作者计数引导分配；比较精度时还须匹配实际端到端成本，不能把K不同称同计算。DDQ原文审查只复用旧报告，本轮没有再读DDQ正文。
- **最小否定条件（仅说明）**：若未来形成明确新规则，至少原Q300／固定大Q／该规则三臂完整同配方训练及真实成本比较；未优于同成本常规控制或密集图提升伴随保护失败则否定。现有缓存不能模拟新query经过decoder的输出；不安排即时替换或新forward。
- **成本**：至少3个完整训练运行单位／数据集；decoder成本随K变化（self-attention部分可含K²），一图一次forward不等于原A成本。DQ原文为R50、6层encoder/decoder、5尺度、3090 batch1；不能套用本R18三层／单4090D吞吐和显存。官方README多GPU示例不证明单卡公平复现已可行，未做资源验收。
- **旧记录**：已读2026-10-01 Q300决策，原fullpool替换及固定NMS@.8即时选择均停止；当时配方不同，不能当共同R18的直接因果结果，也不能换数据名重放。P12失败不支持query分配必然有效。决定：**不保留**。

## 验证类型和资源边界

如果未来出现合格方法，只做各数据集**分别训练**的同方法检验：VisDrone保留既有A/C及失败证据；新方法须按授权重新成对训练，不能把已开发基线当盲测。AI-TOD用独立类别头和同来源公共预训练初态，epoch、batch、更新数据其train成员重新冻结；120ep与36,480更新不能跨数据直接同时照搬。对外部数据的原C数值目前不存在；这里只保留VisDrone高计算锚点，不计划外部切片实验。

这不支持零样本迁移／域泛化；后两者需冻结源训练集、类别映射、目标访问与是否适配，并受不同标注限制。本轮不提出这条额外任务。不同集合AP仅在各自协议内成对比较，不把AI-TOD tiny桶当COCO small，也不把其maxDets1500当VisDronetop600、Q300或COCOmaxDets100。原VisDrone raw-ignore和限定零面积GT扩展不外推到外部数据。

历史环境回执记载实验室RTX4090D／24564MiB、PyTorch2.4.1／CUDA12.4；只是本地已认证摘要，不是新鲜资源观测。没有在新集合运行基线，故不报GPU小时、峰值显存或“必能装下”。上述训练运行单位只解释否定一个假设所需最低成本，不是已经分配的执行预算；本轮无任何实验入口或自动队列。

## 最少缺件与已读来源

|来源|核查程度／需要内容|用途与限制|
|---|---|---|
|[NWD扩展版：Detecting Tiny Objects in Aerial Images: A Normalized Wasserstein Distance and a New Benchmark](https://www.sciencedirect.com/science/article/pii/S0924271622001599?dgcid=author)，Xu、Wang、Yang、Yu、Yu、Xia，ISPRS JPRS190:79–93(2022)|用户提供15页期刊PDF，已核6作者、DOI、大小／SHA并读§3–6，表图重点目视；作者项目页已读，未取得单独补充文件|正文缺口已关闭，先例覆盖进一步增加；数据计数冲突、精确split／底层条件／固定实现仍未闭合；详见[NWD补充审查](NWD_EXPANDED_REVIEW_20261019.md)|
|[AI-TOD工具与名单入口](https://github.com/jwwangchn/AI-TOD/tree/master/aitodtoolkit)|已读生成源码；v2具体包版本、原成员名单／父图映射与适用许可仍缺|若以后选集需先取得权利、身份、划分依据，不是当前下载请求；无需提供全图到D盘|
|[作者评价器cocoeval.py](https://github.com/jwwangchn/cocoapi-aitod/blob/master/aitodpycocotools/aitodpycocotools/cocoeval.py)|仅前120行已读，末段参数未完成；raw重定向超时。已读NWD适配器补证maxDets赋值|需要固定完整源及v2兼容版本才能签发评价协议；不靠注释默认值推定生效值|
|[AI-TOD类别／评价适配器](https://github.com/jwwangchn/NWD/blob/9775ac2d24354895fbdf5b8db1f1600c01c54e33/mmdet/datasets/aitod.py)|本轮在main页面读完整247行；固定根commit来自作者页；未安装运行|类别和生效maxDets的代码依据；v1类名不能替代v2文件身份|
|[DQ-DETR正文](https://arxiv.org/html/2404.03507v6)及[官方实现](https://github.com/hoiliu-0801/DQ-DETR)|正文与消融已读，官方README核查；未逐文件审计算法|直接先例；方法公开结果不能与共同协议AP直接相减|
|[NWD正文](https://arxiv.org/html/2110.13389v2)及[官方实现](https://github.com/jwwangchn/NWD)|正文§3–4已读；根commit9775ac2d24354895fbdf5b8db1f1600c01c54e33；Apache2.0是代码许可|直接先例；anchor正样本结论不自动适用Hungarian|

公开访问限制分开记录：Bing保存权限拒绝后未重试／换搜索通道；UAVDT官网与GitHub raw是网络超时；ScienceDirect是验证码阻断。可读的作者GitHub与arXiv用于直接核查，不是绕过被拒绝的Bing检索。未宣称完成2025–2026最新文献穷尽检索。本轮至此结束，不因缺件无限扩大收集。

revision-002访问补记：出版社验证码和Bing拒绝为历史记录，未重试；本次直接读论文所列作者项目页成功，v2标注入口存在但未打开／下载，官方mmdet-aitod入口连接关闭，未换通道。没有找到独立补充文件链接不等于证明不存在。AI-TOD-v2含VisDrone来源，外部数据独立性不能直接成立；不追加身份分析或科学运行。
