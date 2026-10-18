# 实验导航

2026-10-18补充：[原C三seed阶段诊断完成](research/C_STAGE_RESULT_20261018.md)，
原C逐框重建通过，无新AP/GPU/训练；[固定框投票下一步方案](research/BOX_VOTING_PROPOSAL_20261018.md)
仅设计未执行。下表保留2026-10-16历史快照，不代表最新seed1/2进度；三seed固定C现已交付。

[机器可读登记](../manifests/experiments.json)记录截至2026-10-16的状态快照，不是实时监控。

| 实验 | 角色 | 状态 |
|---|---|---|
| rtdetrv2-r18-common-protocol-seed0-e120-20261013-001 | 主开发基线 | seed0 e120及development评价完成 |
| r18-sahi-fixed-20261016-001 | 固定A/B/C推理 | 单seed收益/保护/成本初筛通过 |
| r18-sahi-seed12-20261016-001 | 成对跨seed复验 | 已启动，最终结果待回传；禁止重启 |
| rtdetrv2-r18-mal-continue-20261015-001 | 分类损失对照 | 完整训练，整体科学门槛未通过 |

报告相对项目资料根目录的路径见JSON；本机主资料根D:/Lzy_CV，服务器按共享PROJECT的Workspace。大型材料不进Git。

当前仅seed1/2共同源码完成精确Git导入；其他实验先登记原manifest，source_commit=null表示精确映射尚未完成。F0、OR、R-small等旧结论与材料保持，不由本次整理重启。

新实验在任务分支增加代码/配置，提交后记录实际执行commit。结果确定后另作小提交更新科学状态及报告引用，不能把结果文档commit误当运行源码commit。
