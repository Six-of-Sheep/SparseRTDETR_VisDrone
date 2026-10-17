# 固定 Gaussian Soft-NMS：实现准备的 CPU STOP

2026-10-17，任务 `sahi-softnms-fixed-20261017-001`。**状态：CPU_GATE 未通过；没有三seed科学评分、没有正式执行release或条件接续进程。** 这是工程准备失败，不是Soft-NMS质量失败，旧A/B/C结果和门槛不变。

## 固定方案与已实现范围

承接决策commit `84005348eac8154ca148800fdb8b1f7a54fd8d46` 与用户本轮授权。机器规则位于[configs/sahi_softnms_fixed.json](../../configs/sahi_softnms_fixed.json)，评分前固定：Gaussian σ=.5；每视图原FP32 sigmoid top600；FP64连续IoU与逐轮累计衰减；每轮重新选当前最高分；同分view/query/category升序；不clip、不平均坐标、不设score cutoff，原始零分和有限下溢成零保留并计数；top300是top600前缀；原raw-ignore在预算截断后处理且不补位。

每seed D−C主600 AP≥+.20、APS≥+.30、AP75-small≥0、APM/APL≥−.20pp，次300 AP/APS≥−.20pp；D−A沿原七门槛。三个seed分别全部通过，不以均值掩盖保护失败。E−B、D−E及配对均值/样本SD/最差值另报，原点值复用。质量通过只能标记`QUALITY_PASS_COST_PENDING`，失败关闭固定候选；不扫描参数、追加GPU/训练或自动启动备用。

新增代码是独立CPU入口、Soft-NMS纯计算核、严格缓存读回/原A/B/C重建、原双评价器调用、事件顺序/完整性检查、单次任务保护及小文件导出。生产接续代码尚未完成验收，不可启动。原训练/推理源码均未编辑；新入口不调用旧launcher，模型构造、CUDA初始化和backward在CPU检查中主动禁止。

## 实际检查与失败

本地17项单元检查PASS：动态重排、累计衰减、逐轮同分、类别隔离、原始零分、实际subnormal下溢、400次累计衰减、超过600候选的前缀、几何不变、连续坐标/无硬IoU开关、独立scalar oracle、重复身份/坏数值拒绝、空池、plan门槛对应、pp一次转换及初步事件拒绝。

实验室使用已认证CPU解释器、隐藏CUDA、单线程、nice19/ionice3，通过**实际新入口** `softnms_job.py cpu-check` 运行四张合成图/三个合成seed的集成夹具；不读取训练图或构造模型。这次调用6.814382秒后退出1：

```text
prepare_seed -> publish(COUNTS.json) -> json.dumps
TypeError: Object of type int64 is not JSON serializable
```

静态定位是`softnms_core.py`的`subnormal_scores`计数：与`np.finfo(np.float64).tiny`比较产生NumPy布尔值，`sum`结果成为NumPy整数，不能直接JSON编码。分数/坐标计算未因该错误被修改，但目前不能将整个统计输出、评价接口或条件执行链认证为PASS。

真实入口在第一个合成seed的COUNTS落盘前停止。生命周期只有`started → parents_closed`，实际`score_budget`调用为0，双评价器及其后的闭合/导出分支**未执行到**。夹具虽已写好，不能把未到达的检查算通过。实际三seed缓存准入、完整原B/C精确重建、三seed12次评分、终态读回/退出、全部负例仍待认证。

用户第5条要求“工程、数值或完整性异常STOP、不自动重试”，本轮据此保留现场并停止，**没有修复后复跑或自动部署等待进程**。以下仅是可审阅的最小修复建议，不是已应用补丁：

```diff
- 'subnormal_scores': sum(0 < c.score < np.finfo(np.float64).tiny for c in output),
+ 'subnormal_scores': int(sum(0 < c.score < np.finfo(np.float64).tiny for c in output)),
```

接续需先补所有统计字段的原生JSON数值类型检查及整条真实入口回归；保留本次preparation-001，另建修订验收记录，不能改写CPU_GATE或补PASS。未通过的生产父状态准入/seed2终点绑定/退出归档分支也必须完成检查，不能只修计数字段便直接评分。不得更改冻结科学公式或门槛。

## 外部状态与保全

16:55左右只读观测原seed2到20325/36480更新（e067），无原链完成记录；这是带时间历史观测。未改原链、硬件、配置或Windows观察器。没有合格三seed全部材料的声明；本轮只读了少量已有父契约/结果/代码元数据以及seed0预测schema，不把它们算完整缓存准入。

服务器现场：`/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace/experiments/localization-distribution/sahi-softnms-fixed-20261017-001/`；STOP.json、preparation-001/source.tar、源码、CPU_CHECK.log/CPU_EXIT.json和合成现场全保留。独立STOP是本轮退出后补充的事实回执，不改写失败入口自身日志或旧实验。

D盘同名delivery目录仅必要材料：STOP/小日志/退出回执已核大小SHA（3文件5249B），PLAN_FREEZE_AND_STOP.json记录评分前方案SHA和失败边界。源码准备包5,416,960B是本轮必要源码上传件，不含checkpoint/科研预测；完整合成现场仅远端。本轮无新增GPU/模型推理/训练、bootstrap或holdout；新增实际CPU验收墙钟6.814382秒，本地单元0.076秒，其他编写/网络/静态阅读未全程计时。既有累计科研成本不改。

代码将按任务分支提交/推送并交付**草稿PR**保全；仓库管理检查或通用CI通过不替代本次失败的CPU集成验收。正式release、启动身份/令牌及CPU评分均未发放/启动。
