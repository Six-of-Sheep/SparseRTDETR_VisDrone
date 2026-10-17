# Soft-NMS计数修复／真实数据流验收未完成

**当前为STOP，未开始科学评分。** 本次已实现过滤前/后计数修复和独立准备门禁，最终源码的合成CLI检查通过；随后的父元数据快照检查因准备启动脚本填错路径而退出。未执行18组真实输入证明、真实缓存重建或E/D评分。原SAHI三seed结果、旧失败现场和冻结门槛保持。

## 修复及计数字段

运行commit：7e010d66e3a17d8316c1a10a5dfc163ee0eec2ca，原codex/sahi-softnms-fixed；362文件冻结release已逐文件核验。source.tar SHA为ef89743660177be661f620be5c6b5d9d3007ca60ab238a9976c7b21009cddfe1。

| 阶段 | 字段/规则 | 本次适配 |
|---|---|---|
| 原始query/class及每视图选取 | 每视图Q300×10原分数，按原规则选600 | 不变；准备缓存时增加实际计数留痕 |
| 后处理 | A保持整图候选；B/C原hard-NMS；E/D固定Gaussian σ=.5 | 算法/FP32输入/FP64衰减/排序不变 |
| 预算截断 | top600/300；300为600前缀 | 在raw-ignore之前，不补位 |
| 完整评价输入 | official.canonical_input_sha256 | 原primary_inputs及_input_payload生成，完整SHA必须相同；包括原始GT区域和排除类别 |
| 过滤前 | spatial_filter_counts.detections_before、GT_before | 直接复用原bridge_inputs生成并与原before精确比较；GT_before只计类别1..10，不等于全部原始标注行数 |
| raw-ignore后 | spatial_filter_counts.detections_after、GT_after | 直接复用原bridge_inputs及官方raster primitive，不另写过滤器 |
| 评价计数 | official.evaluated_detection_count、evaluated_ground_truth_count | 必须等于实际after计数；保持原后续匹配/类过滤/maxDets规则 |

保存计数必须为原生整数，SHA、before、after和evaluated分别严格核验。没有不等式放宽、nan_to_num、默认填值或旧记录修改。

## 工程结构与已验证范围

新增input-check监督一个低优先级CPU worker，沿用RAM、磁盘、CPU温度和1800秒无进展保护；无总墙钟上限，CUDA隐藏，禁止模型构造/backward。准备阶段显式禁止原/新AP评分。设计为依次生成METADATA_GATE、18组EVALUATION_INPUT_PROOF、CACHE_REBUILD_GATE，最后REAL_DATA_GATE绑定源码、父材料及成员。正式评分入口必须认证该证据和准备好的E/D文件，不能用元数据PASS代替真实输入或缓存重建。

已通过：

- 原23项Soft-NMS/JSON单元检查：23/23，0.121s；算法与冻结配置字节不变。
- 开发版合成实际CLI：4合成图×3合成seed，原处理分支及新认证准备消费分支各12次双评价器调用，19事件闭合/导出通过。
- 最终362文件release的实际CLI再次通过，外层20.408475643s。五个手算案例：无过滤、部分过滤、全过滤、top600和top300先截断后过滤不补位；24项错误计数/输入SHA拒绝通过。新旧准备消费分支的合成scores完全相同，新增分支禁止再次调用prepare_seed。

上述调用仅为合成工程检查，不是548图科学评分。最终真实父快照CPU检查未完成：在读取快照时退出2，未执行其真实正例/126负例。旧revision004的126项PASS仍只代表旧来源及其原范围。

## 新STOP及准确路径

本次准备启动脚本错误传入：

`revision-004/inventory-002/SNAPSHOT_AUTHENTICATED.json`

旧PARENT_CPU_GATE实际登记：

`revision-004/SNAPSHOT_AUTHENTICATED.json`

这是本次启动脚本的路径接线错误。STOP后只读核对准确文件存在，10,812,081B，SHA为6fa6d6b079eef4058e8c12150dc5ceb0182279173edf32b41b347b28352ba50c，与旧门禁引用精确一致。原材料未证实缺失或损坏；未把路径检查当成重新验收，也未修改脚本后重跑。

此前部署脚本生成器曾把占位符误替换进文件名，Python在执行前语法拒绝；已在准备范围改为分隔占位符并做本地AST检查，保留错误脚本及PREPARATION_ENGINEERING_NOTE。该错误发生在任何最终验收进程启动前，与后来的真实快照路径STOP分开记录。

按用户“新执行异常STOP，不自动重试”，父检查失败后控制器立即退出。STOP_READBACK记录控制器PID1126753已不存在；real-data-001未创建，没有新评分契约/attempt。18组真实输入0/18，真实缓存准入0、ABC重建0、E/D科学score_budget0、GPU/模型/新推理/训练0。没有质量裁决，也未将本候选关闭为科学FAIL。

## 交付与下一接手点

PR #4沿用并更新，不合并main。源码提交/推送、CI、合成工程PASS、真实准入未完成及科学未评价分别报告。最终源码、STOP及已消费旧令牌均保留；大型快照、合成现场、预测及原checkpoint仅远端保存。

D盘必要回执7文件7232B已逐文件核SHA，见SLIM_RETURN_RECEIPT.json；另有本轮源码/报告/小索引。没有大型预测或快照回传。成本见COST.json，历史成本不覆盖；合成工程计时不冒充总研究墙钟。

下一接手需要重新授权准备入口修复/重验：从已认证的旧门禁引用读取准确快照路径并校验，不再手拼目录；复用本次未受影响的合成CPU结果。其后仍必须先完成真实元数据、18输入证明、缓存重建，才允许新评分attempt。以上仅建议，未实施或启动。
