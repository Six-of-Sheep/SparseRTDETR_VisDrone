# 正式表图的本地复现

唯一入口只读本地已回传JSON，不连接服务器，不导入torch或评价器，不加载模型/预测/checkpoint，也不重算AP：

```powershell
python -B tools/paper_sampling_terminal.py --receipts <已核SHA的精简receipts目录> --output <不存在的新输出目录>
```

依赖Python 3.10+及matplotlib；不安装或修改原训练环境。路径由调用者替换，不能直接执行历史生产入口。实际本轮本地receipts目录为`D:/Lzy_CV/workspace_management/delivery/research/r18-sampling-span-20261020-001/terminal-delivery-001/receipts`；Git保留轻量表图、精确SHA绑定与核验摘要，不包含全套环境、数据或大状态。公开复现所需数据成员/环境打包仍是待办，保留正文数据关联限制。

`TERMINAL_BINDING.json`绑定本次formal-003的commit、source、合同文件SHA/规范化摘要、根闭合、原六份评分及必要引用。根实际10阶段与独立B0认证引用分别检查，不伪造B0原CLOSED。核对固定e120/36480更新、独立初态/共同输入摘要、六评分/12评价调用、原22质量条件、成本边界和累计账本。所有已引用小JSON按SHA和大小定位；大型事件、完整checkpoint等只保留原回执引用，本地不重新认证其内容。

`paper_sampling_tables.py`与`paper_sampling_mechanism.py`从PR17的固定提交逐字节复用，原始绑定另存`PR17_ORIGINAL_BINDING.json`。PR17旧CLI仍绑定formal-001，应在原PR17检出使用；PR18入口只导入其表格差值/绘图与机制字段归一化函数，新增严格formal-003读取适配，不修改PR17、生产执行代码或门槛。P−B0/P−B1的COCO差值和主评价器AP差值取原比较，其余主评价器差值及B1−B0只作既有已评分标量相减。算术一致性检查不产生新的科学裁决。

输出完整两预算120个绝对值、120个配对差值、22原门槛、四模型固定64成本/四次实测，以及18份已存机制文件的108个组/层记录。分位数与计数保持各自source/window/layer，三层计数汇总明确为重复reference层实例。B0/B1原生“低于阈值”不等于实际施加下限；DN包含padding。缺失GT尺度关联、有效DN/padding分离和梯度证据保持UNKNOWN。`official_APS`原评分为null，不制造原主评价器APS列。

SVG为论文引用版本，PNG留本地供查看。`results/VERIFICATION.json`是表图输入核验及引用索引，不是全量预测/模型备份。所有数字只支持本固定seed0/development548条件下的结论，质量FAIL不因工程/成本PASS改写。后续问题见`NEXT_RESEARCH_QUESTION.md`，仅提案审查，当前新增实验执行为0。
