# CPU准备验收：范围、证据与未完成项

2026-10-20生产准备002补充：Data Distillation正文及最终15阶段执行入口见[生产执行准备](PRODUCTION_EXECUTION.md)和[正文核对](DATA_DISTILLATION_REVIEW.md)。本文件下文保留7578876原prepare证据的范围，不将其追认为完整生产入口。新增最终CLI的Linux/WSL合成CPU闭环结果、命令、逐项通过/拒绝及源字节绑定在delivery `production-preparation-002/CPU_GATE.json`与`REPORT.md`独立记录；真实材料/BF16/GPU仍未验收。原17项算法检查因缓存fixture身份接口变化已定向重跑，17/17、0 skip；旧真实小型元数据证据复用，未重复读取训练图像/父checkpoint。

任务 `r18-multiview-distill-20261020-001`；以[唯一计划](PLAN.md)、[配置](../../../configs/research/r18_multiview_distill_v1.json)和[来源记录](SOURCES.json)为准。状态为 **CPU_PREPARATION_PASS / NOT_GPU_READY**，不是科学质量通过。代码从main `458d645f70839cda3d1ff35d67bf3ff812e68bd1`隔离新增，无未合并PR源码依赖；旧实验目录、release、指标及PR #10不变。

## 实际执行

精简证据根：`D:/Lzy_CV/workspace_management/delivery/research/r18-multiview-distill-20261020-001`。所有输出用新目录，不覆盖旧现场。

|证据|实际范围|结论|
|---|---|---|
|`entry-007/GATE.json`，`normal/RESULT.json`及`normal/fixture/RESULT.json`|独立子进程从repo cwd启动最终`r18_distill_prepare.py cpu-check`；无额外PYTHONPATH；worker执行17项检查、原生R18 128×128 FP32物理batch1、五视图合成教师forward、raw→紧凑候选写读和重建、三臂各3个actual CPU更新、完整保存恢复重放、双评价器输入适配、RESULT/CLOSED及退出|PASS；0 skip；9次合成更新；5个合成教师视图；AP调用0|
|同一GATE的duplicate/bad-plan/worker-exception|重复启动拒绝且旧输出逐字节不变；计划冲突拒绝；独立worker异常形成STOP、不形成CLOSED|预期拒绝PASS，非生产失败重试|
|`METADATA_GATE.json`及`metadata-002/RESULT.json`|最终prepare入口读取真实4869训练manifest及原seed0父绑定；逐大小/SHA、全成员稳定ID、split/顺序、父scope/seed核验|PASS；没有读取真实checkpoint、训练图像或教师缓存|
|同一元数据GATE|缺少显式输入、错误父绑定SHA|预期拒绝PASS|
|`test_multiview_distill.py` 17项|几何/奇偶尺寸；raw完整性/绑定；精确IoU=.5与>.5；同分类/同分排序；置信门与300预算；实际弱增强/RNG/flip对应；训练ignore；同学生像素和GT；strong/错图拒绝；LR；27个未舍入门槛；实际Dataset.__getitem__及磁盘成员SHA；raw SHA和重建拒绝；原生Hungarian而非query索引；KL手算、teacher detach、空教师/非法值；原监督loss逐项不变|全部执行，不以skip为通过|

entry-007正常/重复/错误计划/worker异常的外层墙钟分别约28.000、0.078、2.688、2.843秒，记录的是本轮CPU准备工况，不是GPU训练耗时。正常保存恢复逐项精确比较model、optimizer、EMA、完整RNG、DN噪声、输出、loss、匹配及clip前梯度；恢复错cache绑定在加载前拒绝，并将该engine置失败。成功的临时**合成**checkpoint不长期保留，仅留SHA/大小/恢复证明；没有生成真实训练state。

真实小型训练元数据复用路径只由精确SHA认证，不以历史目录名认定身份。manifest SHA `df87e6a377a5546d3d3fe0cd0e860d15c2fe5e8003ea87d887153579999a4e9d`；顺序SHA `bd4f14136a52df84fdd0763ecc17cfe6fec56a4c85a9ac27d5a82bc8cfb79027`；父绑定SHA `93d50992a27583832f2ab4ae4039b02db590ffbbb0850f050507bb1a0bdbc938`。这不代替未来服务器原件准入。

## 版本与既有证据

`distill_engine.train_window`与共同Engine完整方法作文本核对，唯一方法内差异是36480→3648终点，LR在新模块局部导入；原Engine、增强、NMS、原生matcher未修改。对未来正式执行另绑定完整source manifest，不能以当前入口6文件记录替代全依赖清单。实际运行字节、Git LF对象与本机CRLF分别记录，不把commit当字节验收。

保留cpu-001/002、entry-004及其他准备输出。cpu-001发现本地TorchVision强增强trace不闭合，故在任何科学执行前将三臂唯一配方共同固定为弱增强；cpu-002是合成空图row身份重复，entry-004是合成父state未按真实safe schema构造。entry-007补齐精确NMS边界、raw SHA与重建检查，并改用独立合成teacher副本，避免fixture解冻学生原本冻结参数。早期成功结果不覆盖、不追认为最终版本；未改变生产历史。旧各次CPU成本保留，不把最终9次合成更新说成准备过程累计只有9次。

## 明确未做／执行前缺口

- 无服务器连接、GPU调用、科学训练、模型AP评分、548或1602训练监督、真实缓存生成、正式合同或令牌。合成CPU forward/backward是工程测试，必须与科学训练0分别记录。
- 双评价器的**输入适配器**已执行；没有运行完整科学AP，也不宣称生产终点评价闭合已经验收。
- 当前入口只开放`prepare`和`cpu-check`。最终GPU监督器、资源保护、教师缓存生产合同、正式三臂入口、终点评价/导出总闭合尚未实现验收；未来须先补齐与验收，不能改一个授权布尔值直接执行。
- 真实checkpoint严格load、真实训练文件、768×1344/物理8/BF16、每臂GPU保存恢复及资源验收待新鲜准入。本地CPU环境不代替原实验室环境。
- DETRDistill v4正文已读；作者代码/补充未核。多变换蒸馏原文Data Distillation尚缺，最近性/原创性不作结论。详见PLAN第6节，取得正文后先核实与本问题重合，不扩展成参数搜索。

最终源码、提交/推送、PR及CI状态由D盘`DELIVERY_INDEX.json`记录；CI排队/运行不写成通过。未来GPU预算28791 batch forward、约5.5–8h仅为估计；本轮实际科学训练更新、GPU及科学评价均为0。
