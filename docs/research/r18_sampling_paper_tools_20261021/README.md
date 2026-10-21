# 三臂采样探索：直接先例、方法图与终态表图工具

本目录是本地展示工具，不是实验入口。独立分支从 `origin/main` 的
`458d645f70839cda3d1ff35d67bf3ff812e68bd1` 建立，不依赖合并PR #16。
只读参照其冻结提交 `ab88a9b3b2c8fc1f3203269f4ae022d52b0dd8cb`、manifest
`bb073d33bdd4a935d30cf63bfc936dd9940ac85c6eed1a65a476f8f1a2af5197`。
`attempt-formal-001` 的release、合同及后链不由这些工具修改。

- [公式差异表](FORMULA_COMPARISON.md)：复用三篇已审正文，新增DAB正文、Focal DETR与RoIAlign直接先例。
- [来源与SHA](REFERENCES.json)：PDF单份留既有参考目录，不入Git；未取得本地字节的来源明确留空SHA。
- [图注与坐标说明](FIGURE_NOTES.md)：构造图例，不是模型观测；有可编辑SVG、PNG及可核对数值。
- [机制字段及解释限制](MECHANISM_FIELDS.md)：只显示已有记录，普通query与DN分开，缺失为UNKNOWN。
- [冻结输出绑定](FROZEN_OUTPUT_BINDING.json)：指明输入格式与原阈值，不授予执行权限。

## 环境和检查

Python 3.10+；三个工具的读取、格式化及几何检查只需要标准库。
绘图另用 `environment/requirements-paper-sampling.txt`（Matplotlib 3.10.8）。
安装到独立本地制图环境即可，不修改R18环境。

```powershell
python -B -m unittest discover -s tests -p 'test_paper_sampling_*.py' -v
python -B tools/paper_sampling_diagram.py --output D:/paper-output/sampling-method
python -B tools/paper_sampling_tables.py template --output D:/paper-output/results-pending
```

`D:/paper-output`为自选输出目录示例，不是实验材料目录。每次使用新目录，工具拒绝覆盖已有输出。
方法图为原生初始化下的一个head，显式标记 **CONSTRUCTED ILLUSTRATION - NOT MODEL OBSERVATIONS**。
待终态模板的指标、差值和成本为空，门槛仅展示冻结值；不绘制收益。

## 终态结果：使用已经回传的JSON

```powershell
python -B tools/paper_sampling_tables.py render `
  --receipts D:/paper-input/formal-terminal-json `
  --closed D:/paper-input/formal-terminal-json/CLOSED.json `
  --contract D:/paper-input/formal-terminal-json/contract.json `
  --plots --output D:/paper-output/formal-results
```

工具只在 `--receipts` 内建立JSON大小/SHA索引，用原引用SHA解析迁移后的文件。
不按远端绝对路径打开文件，不重写原绑定。只接受上述冻结manifest/合同及
`attempt-formal-001`；校验根闭合、11阶段闭合/结果/进程退出及身份、固定更新/forward总数，
再读取score结果和compare结果。事件、数据和tensor留服务器，工具不是完整运行审计器。

必要小文件：

1. 根 `CLOSED.json`、`RESULT.json`、原合同和source manifest。
2. 根引用的11份阶段CLOSED、对应RESULT、worker-exit和supervisor-exit。
3. score RESULT所引用的六份 `B0/B1/P × 600/300` 评分JSON。

**原自动slim export没有六份评分JSON，不能仅凭comparison生成绝对指标。**
这些评分文件已是冻结科学链的预定产物；终态后按既有授权精简回传它们即可，
不需要新增推理、重算AP或改运行中的exporter。本轮不连接服务器获取它们。
缺少任何必需引用，工具明确拒绝；模板继续留空。

输出：`absolute_metrics.csv`、`paired_deltas.csv`、`cost.csv`、`gates.csv`、`TABLES.md`、
输入来源及SHA `PROVENANCE.json`，以及可选SVG/PNG。
两档预算分别展示13项COCO指标与7项原评价器指标；原评价器值已为百分数，
COCO值由0–1转为百分数，差值单位为pp。
P−B0/P−B1的COCO差值及原评价器AP差值直接取compare，门槛/布尔裁决也原样取compare。
B1−B0及未保存的原评价器其他差值只做已评分数值的相减，明确标记为描述性差值；
不新增评价器、统计显著性、误差条或门槛。原P−A/P−C仍在compare中，属于历史语境。

同轮成本为固定64图、FP32/batch4、两遍暖机和四遍测量。
表中保留均值、样本SD、min/max、allocated/reserved峰值，成本图的误差线仅为同轮重复测量SD。
不把这些波动当作跨seed不确定性或batch1实时部署证据。负向指标全部保留。

`--test-fixture`只接受CPU夹具，输出全程标记 **SYNTHETIC TEST ONLY**，不能用于正式模式；
工程CPU输入里的门槛布尔值保留为测试字段，整体科学/成本裁决为null。
测试夹具在测试代码中构造，不在正式结果目录放入伪造指标。

## 已有机制展示

```powershell
python -B tools/paper_sampling_mechanism.py `
  --input D:/paper-input/MECHANISM_SUMMARY.json --sha256 <原文件SHA> `
  --effects D:/paper-input/MECHANISM_EFFECT_SUMMARY.json --effects-sha256 <原文件SHA> `
  --scope gpu-acceptance-only --plots --output D:/paper-output/acceptance-mechanism
```

参数占位SHA须替换为实际回传文件SHA。可直接显示本地已有GPU短验收摘要，但图中会注明
**GPU engineering acceptance only, not formal training**，不得填入正式结果表。
正式已有 `sampling-eNNN-w000.json` 单窗口记录使用 `selected-training-windows`；
这些仍只是冻结的选定窗口，不能称全训练分布。
缺位置/offset/分位数/stencil字段时显示UNKNOWN，不要求运行任务补采。

## 结论边界

DAB及相关方法已研究框几何/注意范围；RoIAlign已有单feature-cell宽高下限。
P的具体最细level稀疏extent适配存在可说明的计算差异，但原创性未建立。
`max(w,1/W0)`不保证实际采样跨度，offset可为零；同stencil也不保证同插值权重。
这不改变已授权三臂实验的执行，也不预设其收益或发表可行性。
