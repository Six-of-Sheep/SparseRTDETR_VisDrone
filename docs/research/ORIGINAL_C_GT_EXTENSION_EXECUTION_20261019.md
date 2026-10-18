# 原C补充评价：限定GT输入扩展生产执行

用户2026-10-19接受 `supplemental-zero-gt-input-extension-v1` 并授权新attempt。此文是新的生产扩展记录，不覆盖旧STOP、零面积审查、原独立确认协议或原标注。

本版仅允许指定1602图manifest下两条已认证、已审查整数坐标零高度GT；规范独立冻结在 `configs/supplemental_zero_gt_extension_v1.json`。类别4经原raw-ignore后仍保留则计匹配分母；类别0保持22像素栅格和原过滤；原始物理行/框/文件SHA不写回。负尺寸、非有限、未审查零面积、小数退化、身份/面积冲突继续拒绝；预测框保持原严格正几何。

显式 `SupplementalZeroGT.bind` 从认证原始行绑定成员/文件/物理行SHA，传入真实reader、校验、主评价与COCO桥接；没有全局替换评价器。默认不传扩展的接口仍拒绝零面积GT。结果另带扩展规范和合同绑定，不冒称旧正面积接口未变化。正式A/B/C三seed统一使用该扩展，原科学配置、公式、预算→raw-ignore→不补位、门槛、推理和训练零更新保持。

CPU验收通过：最终 `supplemental_launch.sh → start → run → guard.supervise → worker → reader → primary_inputs/bridge_inputs → RESULT/CLOSED → slim export → exit`，真实1602图/83478物理行/2条例外的独立CPU输入worker；含这两条完整原标注文件的6图三seed夹具实际双评价闭环，实际末批2；worker异常、重复start/run、缺窗/乱序/重复终态及原资源拒绝。13项扩展拒绝检查通过；548图development两预算实际输入/桥接payload与原结果精确一致，未读模型预测或重算原AP。1602图CPU输入worker只构造确定性预测，不执行科学评分。主/桥接的小型夹具评分是接口验收，不能作为1602图质量结果。

CPU准备首次因runtime名称CPU-001与历史验收目录冲突被保护拒绝，未到worker；旧现场保留。同一源码在唯一 `GT-EXT-CPU-002` 下验收通过，没有覆盖或重放旧生产入口。成功CPU外层193.246002529s，失败/开发/网络未全计，旧成本继续保留。

运行commit `97c40e6610d8a75bf635c1801dc42e79767d029c`，381文件manifest SHA `ddfd3260697388e42a8b9fa5b30b983e90055f6a538deca5ba1be24f55c14733`；CPU_GATE SHA `2cc69aa874647cc0ace4d0385d7b8dd4ea6a3128ca80c751d6de7f9182ae1f11`；合同文件SHA `367cdd4e00923957c3397c2587ca79db85e09040b6b1f5f26c0917c9eaf4ddcd`。运行代码已推送。后续纯文档/索引提交不替换该冻结release。

唯一attempt `sahi-original-c-supplemental-20261018-001/protocol-extension-001/production-001`，唯一tmux `r18-c-supp-gt-v1-20261019-001`，新合同/令牌；新鲜三父checkpoint完整SHA/载荷、环境/资源/机器及CUDA-hidden合同检查通过后启动。固定64图三seed独立A/B/C工程/端到端成本先行，成本通过后才执行原1602图三seed18组双评价任务。总6423次batch forward，训练0；科学收益失败仍完成所有seed，新生产异常STOP，不自动重试。

本次唯一生产attempt现已完整闭合：16个阶段、18组双评价（A/B/C × 三seed × 两预算），6423次batch forward含暖机，训练更新0；全部三seed各七项质量条件及C/A≤6成本条件通过。32项阶段进程身份退出检查通过，根EXIT与wrapper均返回0，自有tmux/启动器/worker/监督器已退出。完整结果、状态及成本入口为delivery本任务 `protocol-extension-001/REPORT.md` 和服务器同路径 `production-001/SUMMARY.json` 或 `STOP.json`。未恢复Windows持续轮询；仅当前授权执行和阶段收尾。

独立确认资格仍 `BLOCKED_PENDING_ACCESS_HISTORY`。本任务仅“历史使用情况未完全核实的补充评价”；历史开发使用、近重复、flight/site关联仍未知。未执行MATLAB/Octave，官方依据为固定源码审查/独立CPU转写和实际COCO匹配，不宣称官方运行时认证。不得由本结果声称未见数据、独立确认或完全无泄漏。PR #9不自动合并main；大文件仅服务器，D只精简材料并核SHA。


## 固定终点结果

主top600，C−A，单位pp；裁决使用未舍入数值。

| seed | AP | APS | AP75-small | APM | APL | 次300 AP | 次300 APS | C/A |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | +3.290235 | +4.547564 | +4.697207 | +2.826822 | +0.532306 | +3.239643 | +4.451371 | 4.291805 |
| 1 | +3.341686 | +4.676722 | +5.092637 | +2.746756 | +0.947353 | +3.289074 | +4.565023 | 4.822899 |
| 2 | +3.330030 | +4.235013 | +4.427892 | +2.996261 | +0.671783 | +3.247140 | +4.113844 | 4.780110 |

C−A主AP成对增量均值3.320650pp、样本SD 0.026978pp、最差3.290235pp；APS均值4.486433pp、样本SD 0.227111pp、最差4.235013pp。仅三seed描述统计，不新增bootstrap、CI或显著性声明。

主top600的B−A AP为+0.245831/+0.190014/+0.263270pp；C−B AP为+3.044404/+3.151672/+3.066760pp。C−B的APL为−0.222843/−0.001906/−0.117745pp，不能把相对A通过解释成所有指标相对B均改善。两预算的全部C−A、B−A、C−B、绝对值、原主评价器pp及未舍入门槛记录见精简报告和CSV；pp只转换一次。没有重算development原AP。

匹配固定64图、batch4、每臂两暖机：A秒数2.518841/2.336136/2.339232，C秒数10.810375/11.266947/11.181788；包含各自真实整图/四切片、预处理、回映射和NMS。没有省略整图成本。完整生产根墙钟16268.644356s，约4.52小时，包含工程/CPU评价/推理及等待，不能称GPU忙时；本补充评价准备与恢复等已计量不重叠累计下界16611.107041s，未计量失败准备/开发/网络不记零，原训练成本仍在父报告。

seed1 full-C曾明显变慢，定向采样显示I/O等待，原因未知；原保护继续有效，没有改优先级、重启、重试或加墙钟截断。新科学结果不改变旧STOP/成本。

终态SUMMARY SHA `ca71948ea748b86dd004a0a2d624658a56f0f8da160a6ff3f6f0d4e5b19df59b`；远端科学索引SHA `581d0c91270a7571e14fee122afe8ce708b975a99c46eeca130628bef5667c82`。本机精简终态16文件158310B逐SHA通过；此前10份准备材料72933B逐SHA通过。完整checkpoint、图像/原标注、预测、raw、逐阶段审计和456668B科学索引仅远端保存；路径/SHA索引不是备份。

精简入口：项目delivery `sahi-original-c-supplemental-20261018-001/protocol-extension-001/REPORT.md`、`ABSOLUTE_METRICS.csv`、`PAIRED_DELTAS_PP.csv`、`QUALITY_GATES.csv`、`COST.json`和SHA回执；服务器同路径`production-001`保留全部科学原件。运行commit保持97c40e6，收尾提交只更新本说明和实验登记，不替换运行release。PR #9仅更新，不合并main。

结论仅为披露历史不确定性后的集合内补充评价质量/成本通过；独立资格仍BLOCKED，不支持“未见数据”“无开发接触”或完全无泄漏声明。近重复和flight/site关联仍未知，未开展另列核查；MATLAB/Octave未执行，限制保持。到此收尾，不自动追加实验、候选、训练或Windows持续轮询。
