# 固定框投票：授权执行方案

继承origin/main cd3b8b788df3852070695db10edd199b1d412eda，并显式依赖工程分支commit 039bdcc89f3f1a6c23339bc0040a718986e06000（PR4，尚未合并）。认证运行源是c4a8f124d655fa6c4d5a20506b0fb191f49f2c18。复用原始候选解码、hard NMS、父材料格式认证和双评价器；不是复用Soft-NMS科学结果。原Soft-NMS和微调质量FAIL保持。

GT-free坐标投票位于boxvote_core.py。原P先classwise_nms(.5)；原同类邻域IoU64>.5，按view/query/category身份顺序逐项FP64累加s与s*b。权重0保原框。保留分数、类别、身份、排序，不clip/调分/cutoff/迭代/二次NMS。投票邻域只读原框；同一邻框可贡献多个保留者。输出top300为top600前缀，先截断，再调用原bridge_inputs raw-ignore，不补位。预算后过滤数可以不同。

计数阶段：3000 query-class项/视图 → 原每视图top600 → 原hard NMS保留者 → 一次坐标投票（数量、排名不变） → 600/300截断 → raw-ignore → 原评价器规则。COUNTS报告投票前预算NMS幸存者的移动与零权重，FILTER_COUNTS和双评价记录报告预算过滤前后。

三seed各548development，V-whole/V-five各600/300，共12次原双评价器。原A/B/C不重评分，不训练/新推理/GPU/bootstrap/holdout。主V-five−C、另V-five−A七条件逐seed未舍入值判断，合计42条件，均通过才限定64图GPU端到端测速；V-whole不升格。完整阈值在configs/sahi_boxvote_fixed.json。质量失败关闭固定方案；评分后新异常STOP不重试。

最终入口boxvote_score_launch.py → boxvote_job.py run →真实supervisor→worker→原双评价器→RESULT/CLOSED→SLIM_EXPORT→退出。合成success/worker failure/duplicate及坏终态验收，真实材料准入单列。准备原18组输入证明通过原严格验证器和新旧context绑定复用，411批逐SHA解码原A/B/C精确重建后冻结新评分attempt。参数不按结果调整。

D只保必要报告/配置/指标/SHA，完整预测与逐图坐标统计仅服务器。条件GPU成本门槛V-five/A≤6、V-five/C≤1.10，包含全5视图forward，不能以缓存CPU用时代替。旧报告/运行目录不改；PR不自动合并。

准备版release001的合成raw-ignore边界测试错误使用0起点；原评价器round/max(1,x)使原框与移动框同一像素起点，不能作为预期过滤变化例。原实现没有变更；测试平移到(10,10)以跨越实际离散边界。旧CPU日志/源码保留，定向重验完整最终入口。未读真实缓存、未评分。

准备版release002最终入口发现pipeline/fixture同时发布RESOURCE_START造成不可变记录冲突；只统一写入责任到对应backend，修订保留旧现场，重验真实进程链。科学执行尚未启动。
