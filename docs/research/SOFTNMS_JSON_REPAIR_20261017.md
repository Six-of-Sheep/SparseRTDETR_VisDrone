# 固定 Soft-NMS：统计 JSON 修复与 CPU 准备通过

任务 `sahi-softnms-fixed-20261017-001`，独立修订 `revision-002`。旧 `preparation-001`、STOP、失败 CPU_GATE 和提交 `34626d2a92e656bc085e03e7e818a21f5cf70f9b` 保留；此文不改写旧结果。

## 修复边界

修复提交 `33ddb2e2d00cce1cbdfc2fa5f1946046040e6125`：`candidate_count`、零分、下溢、subnormal 和各视图统计计数显式转换为 Python `int`。严格 JSON 编码器未更改，拒绝 NumPy 标量和非有限值，不使用 `default=str`。分数/坐标算法与冻结配置均未改变。

固定方案 SHA256 仍为 `0a6fe19f0505a31a2c7271f53078bd043549b6982fbe6d33cf4ffba700fb611d`。σ=0.5、FP64累计衰减、动态重排、每视图原候选池、同分规则、零分保留、600/300前缀、D−C/D−A门槛全部保持。模型、原切片/NMS、双评价器和seed2运行版本未编辑。

## CPU 验收事实

- 原17项检查保留，加6项统计JSON回归，共23项本地和服务器检查通过：空池、零分、保留subnormal、实际下溢后的写入/读回/类型/数值一致；严格编码拒绝及写入失败不覆盖旧文件。
- 从该提交 `git archive` 导出355文件，服务器逐文件核对实际字节，再调用真实 `softnms_job.py cpu-check`。独立输出、禁CUDA、单CPU线程、nice19/ionice3，无检测器构造、backward或训练。
- 实际新入口完成4张合成图×3个合成seed的抑制前缓存解码、原A/B/C精确重建、E/D两预算，共12次 `score_budget` 调用。每次均运行未修改的COCO bridge和官方风格评价器；不是手工填写成功事件。**这些调用仅为合成验收，不是548图科学评分。**
- `started → parents_closed → 三seed缓存事件 → all_caches_admitted → 12次评分事件 → complete` 共19个事件闭合；RESULT、CLOSED和精简导出生成成功。导出23个引用共2,340,828字节在服务器核验，未将全部合成预测回传本机。
- 29个预期拒绝检查通过：错误checkpoint、缺view、错误几何、非有限raw、错误SHA、文件缺失、大小错误（各seed）；事件缺失/重复/乱序/错误binding/损坏，闭合覆盖，以及实际CLI重复输出和不允许的生产契约参数。被拒CLI未改写成功现场。
- 各seed统计读回保持原生整数和候选数，视图计数分别合计600/300；原raw-ignore/no-refill、类别ignore、输出前缀、CPU RNG不变、CUDA未初始化也通过。真实入口内测得13.654358秒，进程总时间及单元时间见远端退出回执，不能混为全部开发墙钟。

CPU_GATE 为 **PASS（上述合成工程范围）**；生产父终态和真实三seed科学缓存准入仍为PENDING。生产父状态/数据绑定只在真实材料齐全后认证，不能用合成PASS替代。

## 为什么尚未执行固定三seed评分

2026-10-17 17:28:18+08单次只读核验，原seed2链存活、无终态STOP，完成21,613/36,480更新、完整epoch71，处于e072；`CHAIN_RESULT.json`及`SLIM_EXPORT.json`尚不存在。该快照不能当作持续实时状态。

按本轮“材料未齐就交付准备状态”要求，本次收尾为 **READY_CPU / WAITING_PARENT_DELIVERY**。没有启动服务器等待器、本机轮询、补推理、GPU或训练；没有真实三seed缓存准入或新科学评分，没有质量PASS/FAIL判断。后续仍按既有授权核对原链完整交付、三seed抑制前缓存及原B/C精确重建，通过后仅一次固定E/D评分；新异常STOP，不自动重试。

## 证据与版本

- 源码commit：`33ddb2e2d00cce1cbdfc2fa5f1946046040e6125`；运行源码包SHA256：`85f7d28d9a9eb11e07dc9353024fb13783f9e91495f1287147a57771a47135b2`（5,089,280字节）。本说明为验收后文档，不能冒充运行时源码。
- CPU_GATE SHA256：`89d03d9ebd9e01bed4b53b3521adfdb1c8bb6213b53a126afe54a7badde929e7`。
- 服务器完整现场：`/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace/experiments/localization-distribution/sahi-softnms-fixed-20261017-001/revision-002/`。
- D盘必要交付：`D:/Lzy_CV/workspace_management/delivery/experiments/localization-distribution/sahi-softnms-fixed-20261017-001/revision-002/`；7份退出/门禁/小日志共21,628字节逐SHA核验，另有源码身份、修复差异、单次状态及Git回执。完整合成场景和旧科学预测/checkpoint仅服务器。
- 原STOP说明见[上一轮记录](SOFTNMS_PREPARATION_STOP_20261017.md)。Git推送、PR/CI及科学准入各自独立记录；本轮不合并main。
