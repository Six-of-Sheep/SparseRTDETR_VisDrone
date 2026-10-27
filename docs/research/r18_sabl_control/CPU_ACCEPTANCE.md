# PR26接线与CPU验收

2026-10-27正式24轮链追加接线见[FORMAL_EXECUTION](FORMAL_EXECUTION.md)：旧短验收与下文初次CPU记录保留为历史；正式链使用独立入口/state scope，等待单独正式执行授权。300份既有计算文件原始Git SHA不变，复用已闭合短GPU工程证据；新增正式链只做合成CPU和CI，不声称真实终点评价/成本已经执行。最终验收回执见D盘唯一formal24-preparation交付。

N/S/A最小模型已实现；本轮仅合成CPU构造，不读真实图像/权重，不连接服务器或做科学评分。CPU_PASS不是GPU验收或质量收益。

- 39项原生模型检查：N旁路逐位一致；共有参数/BN/cache及S/A头初态一致；原训练输出、21损失、DN一致；四边/标签/分母/空匹配/贴边/域外/逆序处理；新增梯度到头/F8/backbone，不能经坐标回原decoder；S可响应空间排列，A保持均值控制。常值特征控制及逆序框反例保留。
- 26项实际CLI：Git导出包→SHA/解包→plan→N/S/A实际两微批更新→完整C1保存/恢复及下一更新精确重放→精简SHA/CLOSED/EXIT。调用cwd为独立目录且剥离PYTHONPATH，不依赖预设导入夹具。错误合同/资源预算/重复/源码损坏/材料绑定/伪GPU及表格缺原N均拒绝。
- 6项真实状态适配器拒绝：错误SHA/arm/schema/NaN/EMA/optimizer，失败engine封存。没有替换原生恢复分支。

原N历史评分只读核SHA，加入RESULT_TABLE_TEMPLATE.md及表格入口；没有重新计算AP/AR。同期质量门槛未变；只胜同期而仍低于原N时给出明确解释标签。结果工具只接受既有评分JSON并检查SHA/两预算/原N认证哈希，不代替未来正式链的终点身份与闭合审计；成本仍须附原实测回执，不能把quality_controls_pass当整体科学PASS。

全Q300物理batch8共2400 RoI，单RoI输出FP32共120422400B，两个pre-conv输出240844800B；选定张量字节之和1145835264B不是同时活跃峰值。头前向149831270400 MAC/299662540800 FLOPs（仅乘加按2 FLOPs，不含激活/池化/插值/反向/原模型）；解析参数1837572已由CPU实例核对。六个完整验收state的模型/optimizer/EMA解析下界约2046684672B，实际还含BN/cache/元数据。大state不入Git或D盘，只在WSL隔离目录。详细项见RESOURCE_ESTIMATE.json；CUDA峰值/耗时UNKNOWN，不能据CPU运行时间担保15分钟或18GiB可满足。

短GPU入口只支持plan/run验收：[1,2,2]每臂、9实际更新/18forward、900秒GPU阶段/1800秒wall、18GiB/30GiB/5MiB。运行后必须EXIT，无训练/评分/测速子命令；正式N24/S24/A24入口本轮没有实现，也没有自动阶段连接。验收state是专属format/scope，不可进入正式训练；后续44589总预算必须继承实用18而非退款重置。启动前仍需授权、新鲜认证N/材料/环境/物理GPU/存储/无重复作业核验；包中没有授权文件。

原N构造旁路只认证CPU随机原生模型，不认证真实权重on/off；真实GPU BF16、显存峰、RoIAlign确定性CUDA后端及完整真实材料尚待一次短验收。首轮RoI坐标检查的FP32差3.8147e-6超过3e-6，定位为检查容差，改1e-5后三ULP检查通过，公式不变；旧失败工具轨迹保留。精确CPU消耗和回执见CPU_ACCEPTANCE.json。

新包将从已提交、CI通过的精确head导出，完整manifest与COMMANDS及最终CI留D盘唯一交付目录，不复制checkpoint。CI会在干净Linux环境重跑同一原生模型、状态和实际CLI检查；最终状态以交付回执为准。
