# 固定SAHI seed1/2：协议修复与剩余链

原seed1正式训练已完成e120/36480，A548闭合。失败发生于accept-A完成实际工作后：`infer_seed.gpu`没有发出`sahi_complete`，入口直接发`seed_phase_complete`，被Reader闭合保护拒绝。旧release、STOP、失败现场、令牌和成本不改，不将内部RESULT追认为阶段PASS。

修复仅补生产事件链，完整输出/状态/RNG验证和RESULT落盘后发`sahi_complete`，然后入口发`seed_phase_complete`。Writer、Reader及归档保持同一顺序规则，并拒绝缺失、乱序、重复终态。模型、损失、数据、采样、切片/NMS、评价及训练日程保持原字节。

`test_sahi_entry_integration.py --output <新CPU目录>`实际进入`new_entry.main → supervise → 独立worker → infer_seed.gpu/evaluate`。小CPU模块/输入/硬件和评分为明确标注的合成后端；生产事件发射、Writer、Reader及归档不替换。覆盖infer-A、accept-A/B/C、full-C、六预算及阶段切换；负例来自生产记录变异，禁止用手写正确事件充当入口验收。这是工程集成证据，不是AP或真实GPU验收。

新attempt `protocol-resume-20261017-001`从固定Git提交导出独立release，记录实际manifest；不重放原`new_chain.main`或旧启动令牌。

- seed1：CPU定向认证原完整checkpoint、EMA、binding、formal闭合和A548身份/完整性/闭合证据。新生命周期绑定新契约；严格完整恢复仍使用未改的旧binding及旧checkpoint，不给旧state换身份。新attempt执行accept-A/B/C、full-C、六预算评价。
- seed2：沿原固定3更新保存恢复验收后，从对应seed独立未训练ImageNet初态训练120ep/36480，再完成自身A/B/C。验收state不能正式续用。
- `resume_support.py`集中解析原父状态和A缓存；禁止seed1走formal、GPU短窗训练验收或重放A548。原A只按原身份读取，不补造新CLOSED。
- `resume_chain.py`为唯一剩余链；`resume_admin.py prepare/freeze/launch`分别认证、冻结和一次性启动，启动记录`RESUME_CONSUMED.json`。普通CPU错误可修复留痕；新GPU/数值/身份/资源异常STOP，无自动重试。

原PLAN-v002及七项C−A门槛、matched64端到端C/A≤6、跨seed逐项PASS规则不变。科学/成本FAIL仍完成固定两seed及评价，不据结果改配置、换seed或追加训练。aggregate走默认契约和评价闭合认证分支，复用seed0原点值，汇总mean/sampleSD(ddof1)/worst及C−B/APL，不bootstrap或访问holdout。

科学原件仅实验室。精简导出增加父状态认证、协议差异、源码Git及manifest索引，不复制大checkpoint、预测或audit到Windows。旧观察器无退出回执，终止原因未知；新工具见[SAHI_OBSERVER.md](SAHI_OBSERVER.md)，实时状态必须fresh读取，历史快照有过期标记。

提交/推送/PR及工程门禁与科学结果分别报告。PR不自动合并main。本修复不授权额外实验。
