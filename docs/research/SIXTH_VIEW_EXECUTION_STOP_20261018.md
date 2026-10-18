# 第六视图执行收尾：工程 STOP，质量和成本未评价

任务 `sahi-sixth-view-20261018-001`，唯一 `execution-001/production-001`。原科学 PLAN、A/C、Soft-NMS、微调、框投票及旧结论保持原样。没有自动修复、重试、训练或后续 seed 启动。

| 门禁 | 实际状态 |
|---|---|
| 新几何9项、原切片/NMS21项、管理11项 | PASS；复用或已运行的定向CPU检查 |
| 最终真实进程入口的合成CPU闭环 | PASS；七阶段、四组原双评价器、闭合、导出和退出，以及worker异常/重复拒绝/坏终态/8种资源拒绝 |
| 三seed父状态和CPU模型info核对 | PASS；CUDA未初始化，不能代替真实GPU恢复 |
| seed0真实accept-A | STOP：`KeyError('storage')` |
| 64图A/C/center/flip工程及成本 | 未完成；0 forward，无64图AP |
| 548图center/flip及两预算双评价器 | 未开始 |
| seed0十四项质量、同视图控制、成本≤6 | NOT_EVALUATED；不判为科学FAIL或PASS |
| seed1/2 | 未开始 |

失败路径：`sixth_view_job.worker → sixth_view_gpu.run → restore → base.prepare_cuda → live_admission → contract['storage']['identity']`。新合同使用`storage_identity`，新适配层未向旧接口提供`storage.identity`。这是新增生产恢复入口的字段接线缺陷；不是观察到存储资源不足。实际资源身份检查没有被绕过。

已核冻结源码：`prepare_cuda`先调用`live_admission`，随后才`torch.cuda.init()`；模型构造和checkpoint恢复更晚。失败日志及进度只有`begin`，无`cuda_initialized`、`restored`、`warmup`或batch事件，没有RESTORE/RESULT/CLOSED。因此本次生产CUDA初始化、模型构造、恢复、forward、检测器更新及科学评分均为0。准备阶段CPU曾检查真实父状态及构造模型，不能与本次生产计数混淆。

CPU闭环使用实际最终CLI、原guard/supervisor/channel和真实双评价器，但合成GPU阶段未经过`base.prepare_cuda`真实合同适配。这是验收覆盖缺口，CPU PASS不追认真实GPU恢复或整体工程PASS。

## 退出和证据

2026-10-18T19:33:52.105605+08:00只读核验：chain1588319、supervisor1588351、worker1588364均不存在；supervisor回执`remaining_live/unknown_pids/cleanup_errors`均空，`visible_gpu_pid=null`；单次NVML查询无compute进程。只有seed0/accept-A目录；无根RESULT/CLOSED。11份SSD现场与实验目录runtime-archive逐项SHA一致；这是失败现场完整性，不是成功闭合。

34份必要材料、203837字节已精简回传D盘并逐文件SHA_PASS，见`SHA_RECEIPT.json`。完整运行现场、旧checkpoint/raw缓存和历史材料仅服务器保留；没有下载checkpoint、预测或全量日志。SHA索引不等于独立备份，没有声称全量回传闭合。

## 版本和成本

运行commit：`d1b4bcbfe21e758ae810e59e25131a21e18814be`，366个tracked文件；manifest SHA `b48a8cbbc3f83c2c15ee7d898516644be0dd5c56916e425f748e5cbb385ac47e`。

合同文件SHA：`586cb24ad50128bb72dbedddddebb6c22677357c2fd7f5b86212df401ee7ee47`；合同内部canonical digest：`cc0f55e575bee15cc8e61f340e0d9cec01f45a0282771de56ff9631f2c0f1f00`。二者用途不同。

CPU004门禁内部40.781278853秒（外层41.090593496秒，不重复加）；生产根阶段4.225069658秒。历史准备和失败CPU耗时保留，已计量不重叠墙钟下界82.960823847秒，详见`COST.json`；开发、网络等未全计，不视为零。没有实际GPU推理耗时、峰值显存或候选/A成本倍率。

PR #8保持开放、不合并。收尾仅更新文档和索引，运行release不改。最新PR/CI状态见`PULL_REQUEST.json`和`CI_READBACK.json`；CI不能替代生产验收。

## 后续边界

本次到此STOP，没有后台接续或本机持续轮询。若后续另获修复授权，应显式适配真实旧入口所需字段，并在CUDA隐藏条件下覆盖真实pre-admission接口及全部所需合同字段，再冻结新版本和新attempt；不得重放已消费production-001。此处仅记录修复方向，没有实施。
