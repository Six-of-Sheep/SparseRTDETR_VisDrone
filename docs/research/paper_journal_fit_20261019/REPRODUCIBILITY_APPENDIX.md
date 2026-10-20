# 复现附录：版本、环境来源与工程证据

本附录与[正文](CHINESE_DRAFT.md)分开。工程闭合／CI仅证明所查接口或文档检查，不代替研究贡献、投稿适配、数据独立性或科学收益。所有原release、旧STOP、合同、已消费令牌、成本与故障证据保持原样；本文写作不重放入口。本轮没有读取checkpoint、预测载荷或封存图像／标注。

## A.1 科学材料与运行身份

|对象|冻结身份或位置|含义|
|---|---|---|
|原A/B/C成果登记|`manifests/frozen_abc_results_20261019.json`|三个训练seed、两个集合、两预算分别登记；不是新的实验授权|
|标量包|`docs/research/abc_results_20261019/SOURCE_SCALARS.json`，SHA `8fb15f6d0580b0351ff36c6fb1e269ba4a3884f596f5d7426542dad0611d1f5c`|绝对值、成对增量及表图的认证来源；正文表仅摘录冻结CSV|
|补充评价运行commit|`97c40e6610d8a75bf635c1801dc42e79767d029c`|实际运行版本，不使用论文分支commit替换历史源码|
|补充评价运行manifest|`ddfd3260697388e42a8b9fa5b30b983e90055f6a538deca5ba1be24f55c14733`|381文件实际运行清单；不代表本轮重复全量核验|
|模型骨架作者commit|`1c8ac3f7ba84f14bd5651ab7b1b70d69a5f55f47`|共同协议的代码来源，不是声称完整官方配方复现|
|VisDrone工具commit|`005445782213e20cb91bc50a597db3dd949e749a`|toolkit 1.0.4、语义1.0.3，源码由`manifests/visdrone_det_toolkit_005445.json`绑定；未执行MATLAB／Octave|
|SAHI现代作者代码commit|`b5a153ee7bcb1fd84a7b868575ff14131e7423ce`|本地三份入口／合并代码快照，不代表2022论文版本或完整后端数值认证|

|seed|原e120 checkpoint SHA|
|---|---|
|0|`13a35283a02d2b728e3449917586ded5cdf2d578ebe6847510748e0dc4d21757`|
|1|`834cd4de9d4018ac9093f48c9ab459b88a19cb039bea1f780aa2d3ace11f9731`|
|2|`0a356a41c9b62e9ef5a934749ddb8bd12b088ca7954b96ba5ffe9b2de6ca4058`|

seed0历史推理以原源码manifest追溯；seed1/2实际推理身份及后续源码导入身份分开，完整对应见冻结登记。不会把回溯导入commit冒称原启动commit。

## A.2 环境：已从回执提取与仍未知分开

主来源为既有补充评价`returned-preparation-001/PRELAUNCH.json`，本轮重新核对其字节与原`PREPARATION_SHA_RECEIPT.json`一致。它是2026-10-19历史准入的观测，不是本轮服务器实时探测。历史development `BASE_BINDINGS.json`的torch／torchvision／CUDA构建／cuDNN期望与之相同；不能单凭此推断所有训练与评价阶段的软件完整一致。

|字段|有据可写值|核查边界|
|---|---|---|
|GPU|NVIDIA GeForce RTX 4090 D|补充评价实际准入回执；固定64成本所用同机卡身份已在原合同绑定|
|GPU显存总量|24564 MiB（设备回报）|不是模型峰值，也不是24 GiB与24564 MiB混算|
|逻辑CPU数|24|不代表24个物理核心或具体CPU型号|
|PyTorch／torchvision|2.4.1／0.19.1|实际环境回执，非本机CI依赖版本|
|CUDA构建版本|12.4|不等同于驱动版本或系统CUDA Toolkit的单独核验|
|cuDNN|90100（原回执整数，对应9.1.0）|沿框架版本编码解释，原值保留|
|推理batch与精度|4／FP32|原成本协议与固定结果登记，不是部署最优配置|
|pycocotools|零GT兼容性历史审查登记2.0.11|该审查不是全部生产阶段包版本再认证；正文不由此扩大软件一致性声明|
|Python版本、CPU具体型号、OS发行版／内核、NVIDIA驱动、安装RAM|在本轮选定绑定回执中未完整找到|不以Windows本机Python、可用RAM或逻辑CPU数填充；如目标期刊需要，从已有归档环境记录再定向补取，不启动模型|

本轮环境提取回执位于D盘`workspace_management/delivery/publication/paper-journal-fit-20261019-001/revision-003/ENVIRONMENT_EXTRACTION.json`。未取得的字段保持未知，并不伪装为“完整环境已审计”。

## A.3 数据资格与评价扩展的精确登记

正文继续披露历史开发使用及近重复／飞行地点关联未知。原独立确认资格为`BLOCKED_PENDING_ACCESS_HISTORY`，用户三项人工说明均为“用户无法确认”；补充评价不能解除这一资格。原1602 manifest SHA为`1af7eb4248be3097122d2168e18519c8db3ac11e17faff46aca8e3d78e5cfef4`。SHA无交集只证明认证字节级重复关系，不证明独立场景。

限定输入扩展名为`supplemental-zero-gt-input-extension-v1`。精确原行身份：`9999985_00000_d_0000020.txt`物理行12（类别4）、`9999999_00590_d_0000267.txt`物理行89（类别0）。原标注字节及SHA、全部物理行不变；前者原raw-ignore后若保留仍计分母，后者保持22像素栅格和原预测过滤。负尺寸、非有限、未认证退化输入、身份／面积冲突拒绝；预测仍严格正几何。正文说明这些影响结论的语义，精确成员绑定留本附录及原审查。

官方主评价AP/AR使用其maxDets定义，不输出APS/APM/APL；COCO桥接另报面积指标。桥接复用项目固定忽略栅格和预算后过滤，不补位；COCO每类maxDets100与官方maxDets500、外部top600/300不互换。本轮只静态读源码，未运行两个评价器或MATLAB。

## A.4 本轮复读的一手源码与历史文献边界

以下仅引用已存在小文件，不vendor或重新下载第三方源码／论文。每份文件的本轮本地字节SHA与已知旧来源分开记账；SHA证明所读字节，不代表论文或代码已运行。

|对象|文件与所核位置|本轮层级|
|---|---|---|
|SAHI|`predict.py`的整图开关及默认合并；`combine.py`的后端分派、`_apply_merge`与GreedyNMM类；`utils.py`的`get_merged_bbox`／`get_merged_score`|三份2026-10-17已下载快照重新核SHA后静态阅读；后端依赖并未齐全，论文正文未补得|
|RT-DETRv2|2026-10-13作者README、R18顶层config、optimizer快照；本库固定R18 config与decoder|README/config先核旧内容SHA；报告正文未补得，不用COCO/T4公开速度移植本稿|
|VisDrone|`evalDET.m`、`calcAccuracy.m`、`dropObjectsInIgr.m`、`evalRes.m`、`compOas.m`、`createIntImg.m`、`saveAnnoRes.m`、`VOCap.m`|8份与官方固定commit清单逐SHA一致后阅读；不是MATLAB运行测试或数据基准论文全文|
|AutoFocus|已保存正文提取文本第4节、FocusChip作者代码与配置|文本原件与源PDF哈希分开，作为早期范围先例，未当最近航拍工作|

26份已读本地来源（其中17份与既有认证SHA逐项一致）的精确路径、大小和SHA在`revision-003/SOURCE_REVIEW_RECEIPT.json`；没有复制原论文大文件。这个数不表示26篇文献或全部全文核查。最近航拍论文和三刊每刊至多两篇近作仍待用户页面／PDF。

## A.5 工程沿革与结果范围

完整故障流水只保留在原材料，正文不按修复尝试叙事。对应索引：

- `docs/research/ORIGINAL_C_ZERO_AREA_GT_REVIEW_20261019.md`：原始零面积GT审查与旧STOP，属于历史限定审查；
- `docs/research/ORIGINAL_C_GT_EXTENSION_EXECUTION_20261019.md`：已授权扩展的实际生产闭合与交付；
- `docs/research/abc_results_20261019/README.md`：冻结数值、图表与失败对照；
- D盘`workspace_management/delivery/experiments/localization-distribution/sahi-original-c-supplemental-20261018-001/protocol-extension-001/REPORT.md`及`VERSION_LINKAGE.json`：生产18组双评价与累计成本；
- 本文开发分支`codex/paper-journal-fit`／草稿PR #10：文稿审查，未替换实验release、未合并main。

原生产合同、源码身份、终态与SHA回执均为历史证据，不因本轮读到它们而重放。科学失败候选保留原门槛与裁决，工程CI通过不追认科学通过；正文的局部收益也不追认独立资格。源码／checkpoint／预测／审计大材料留服务器，D盘只留文稿与小型引用，索引不是备份。
