# 固定双臂微调准备：CPU结果

本轮仅准备；正式训练与GPU未启动，质量未评价。Soft-NMS保持关闭。

运行commit `b9175febfeccf3e7caab0cc4ad2a999918367f8b`，从main
`cd3b8b788df3852070695db10edd199b1d412eda`新建`codex/r18-slice-finetune`。
355个冻结文件，方案SHA `6e695d96682aabbe08d3787b8b9e75f778e2501c60ab71fde1c1a0b730b10acb`。
后续文档提交不改变已验收的执行字节。

服务器证据位于Workspace/experiments/localization-distribution/
`r18-slice-finetune-20261018-001/release-002/cpu-check-001`。
D盘同系列`returned/`只含必要小文件，checkpoint在服务器。

| 范围 | 实际验收结果 |
|---|---|
| 固定规则15项 | PASS：每micro4整图4切片、总量、局部LR、原LR默认hook等价、交集/截断、原身份/面积、ignore、空GT、窄片sanitize、RNG隔离、control原增强字节一致 |
| 实际e120父状态，无forward | PASS：完整外部SHA/binding校验，严格EMA载入，两次独立构建model/BN/新EMA字节相等，optimizer空、EMA计数0；旧终点计数/绑定冲突拒绝 |
| 真实train_core入口 | PASS：workers2/prefetch2，两臂各原epoch109首2窗/32输入；control32视图像素/GT/ignore与原路径完全一致，原图顺序成对相同，混合比例及view_id固定；原图/标注逐项原SHA检查 |
| 有限预取及退出 | 只提供windows0..1，第三窗不在sampler内；子进程不初始化CUDA；4个真实数据worker均已退出 |
| 原生R18合成CPU恢复 | 每臂3次实际更新，1保存→2→恢复1重放2；输入/DN/输出/loss/裁剪前梯度、全部model/optimizer/EMA/模式/游标/RNG精确；有DN21项、空GT无DN12项，未改变VFL/L1/GIoU |
| 拒绝恢复 | 两臂各拒绝跨arm、错误SHA、错误plan，失败engine不可继续更新 |
| 实际CLI边界 | 重复输出拒绝、formal模式未暴露、错误plan在载父模型前STOP；spawn模块导入无分派且不导入torch |
| 管理检查 | 11项管理测试及staged检查PASS；Git CRLF文件按cr-at-eol规则diff检查PASS |

真实父EMA初始模型摘要
`a322373726cc3c666c8ba0ea79a5dadcc80c149d87f8c4094e9c9c807652d676`。
两臂CPU恢复使用128×128/FP32/物理1的合成原生R18，不是768×1344/BF16/物理8
生产数值验收；合成fake-parent `a`×64明确隔离。实际父检查使用768×1344构造对象，
无forward/backward。没有在真实训练图上更新模型，没有GPU/新终点推理/AP评分。

保存恢复覆盖的是新FT局部clock、数据恢复后的下一窗和完整数值状态。
尚未实现或验收生产formal/supervisor/资源准入/自动评价导出链，未来授权执行前
须补这些接线并完成每臂3更新GPU认证；不能把本门禁称为GPU可直接启动许可。
未来生产checkpoint还须绑定run/source/contract/runtime身份，不能只复用CPU的plan/arm/parent字段。

## 准备阶段故障与真实成本

release001遗漏原launcher的MKL_THREADING_LAYER=GNU，15规则及父初始化已通过后，
spawn子进程发生MKL/libgomp导入错误，父进程停在spawn管道写入。该CPU进程
已被定向SIGINT终止，原STOP、冻结源码和退出回执保留；不追认整套PASS。
修订仅在入口导入前安装原GNU层，模型/数据/方案不改。

release001墙钟14413.165219940秒，包括等待/停滞（不是等量CPU忙时或GPU成本）；
release002完整CPU墙钟44.570985733秒，两段合计14457.736205673秒，
开发、传输、管理测试与后续拒绝检查未完整计时，未将内部阶段重复相加。
GPU实际成本0，合成CPU实际原生更新6。后续两臂正式训练预算约3–4GPU小时，
连同验收/评价/收尾约3.2–4.5小时，仅估计、非截止。

12个必要小文件249918B已在D盘逐文件SHA通过，包括旧STOP和新门禁；
完整CPUcheckpoint及控制台仅服务器。旧报告/全部三seed结果不改；新方案尚无质量结果。
