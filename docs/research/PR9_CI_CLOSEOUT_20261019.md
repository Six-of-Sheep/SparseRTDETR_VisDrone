# PR #9 CPU CI覆盖与合并前工程收尾

本轮仅补工程回归，不启动实验、不连接实验室。保留原审查`abc_results_20261019/PR9_REVIEW.md`及全部失败证据。本文件是原P2 CI缺口的后续处置，不追认旧CI已经覆盖新增接口。

## 必跑清单

在Python 3.10/3.12 CI矩阵中，`tools/check_confirmation_ci.py`运行以下三个真实测试文件；缺依赖报错，发现任何skip即失败。固定CPU依赖见`environment/requirements-confirmation-ci.txt`，torch 2.8.0+cpu／torchvision 0.23.0+cpu为[官方配对版本](https://pytorch.org/get-started/previous-versions/)，传递依赖也固定以支持两个解释器。

|检查|数量|实际范围|
|---|---:|---|
|test_confirmation_prepare.py|8|成员、配置与解析GT区分、真实seed/checkpoint身份、末批、原raw-ignore及预算输入、真实Writer/Reader闭合与拒绝|
|test_supplemental_plan.py|10|用途、三seed预算、原条件协议、独立资格BLOCKED、人工未知、旧令牌禁复用、SHA及两种明确检出字节身份|
|test_supplemental_zero_gt.py|15|生产reader扩展引用、精确物理行及协议绑定、默认严格几何、两条例外、负/非有限/小数/面积冲突、身份/flags冲突、预测保护、22像素ignore及过滤|
|原管理／slice／readout|11／21／8|已有工程检查继续保留|

本地独立Python 3.12.14 CPU环境：受影响33项通过、0跳过；原40项通过，相关管理检查和历史源码身份检查通过。此前缺torch跳过的`test_raw_ignore_and_budget_inputs`、`test_wrong_GT_identity`、`test_channel_refusals`本次均实际执行。Linux矩阵是否通过必须以**最终推送head**的Actions回执为准，不借用本地结果或旧head success。

测试用两条已审查原行的固定CRLF字节、物理行号和记录SHA；其他填充行为合成category11，不是原文件恢复、完整文件SHA认证或封存全集核验。调用真实`parse_raw`、`supplemental_data.extension`、`SupplementalZeroGT.bind`、`evaluator_inputs`、`primary_inputs`、`bridge_inputs`及主评价器输入校验/栅格函数，未用隔离原型替代。运行器禁止模型构造、checkpoint加载、CUDA初始化、backward及AP调用；不需要SSH、实验室路径、图像、模型权重或完整数据集。

## 初次CI失败及限定修复

提交7182e0fcad8d3f3c6903914d3b6a9c53743536fc的push39173621225／PR39173624300实际运行31项，零GT15项及原确认8项通过；方案测试在Linux因历史父方案CRLF SHA与Git LF检出字节不同而失败，保留日志，不记PASS。Windows同head31项通过，不能替代Linux。

仅在测试文件加入两个固定来源的LF／CRLF精确SHA映射：先核已记录Git内容SHA，再核重建历史CRLF字节SHA；混合换行、内容变化及未登记来源均拒绝。补两项正/拒绝测试。没有修改期望历史SHA、配置、原回执、`.gitattributes`、生产`exact/reference`或任何运行源码。该测试辅助函数不导出到生产准入。

## 完整差异与集成建议

本轮相对48c3048357d80a6bc99bc6d302d2531dae8980f3只修改CI、固定测试依赖、测试、测试运行器及文档。`src/`、`configs/`、生产入口、模型/日程/切片/NMS/原过滤/科学门槛与实际运行97c40e6保持相同；不修改服务器冻结release。原独立资格BLOCKED和历史/关联未知、零GT协议扩展及未MATLAB限制不变。

完整PR相对main的审查沿用原已逐模块检查，并复核本轮新增差异、实际CI执行和受保护文件无差异。没有确认需要修改生产逻辑的缺陷。本轮为作者自审，不冒充外部独立批准；精确head、完整文件清单、patch SHA、Actions执行/跳过步骤及本地日志在D盘工程回执索引中登记。

最终head的Python 3.10/3.12全部必跑检查通过后，可提交人工合并决定。#9无#2–8硬前置；同时集成其他PR时，共享实验索引按ID合并、CI取用例并集，不能整文件覆盖；#7包含#4的关系保留。工程合并不改变科学结论或数据独立性，不自动启动实验。当前未获合并授权，不合并main、不删除历史分支。
