# 32图终点有效支持诊断

本分支只新增诊断工具，生产代码沿用a97e1ea冻结release，只读加载P/B0固定e120 EMA。完整取数前规则在PR18既有`NEXT_RESEARCH_QUESTION.md`，运行合同保存该规则原字节SHA及32个稳定ID。P固定质量FAIL不变。

`r18_support_math.py`提供实际双线性邻点及I/O/V守恒统计；原core的返回值保持原样，P局部无下限计算复用actual value/offset/attention weights，不再运行投影、decoder或检测器。普通300 query全量统计，最终Hungarian匹配的小GT逐层诊断。坐标采用原align_corners=False顺序，目标边界半开；GT条件占比是几何代理，不是特征内容或训练根因。

`r18_support_diagnostic.py --mode contract-check`只核对源、选定输入、e120文件及历史回执。`r18_support_launch.py`对一次worker实施600秒、16 batch forward、18GiB、20MiB输出及原新鲜资源门禁，独占创建输出；失败只STOP，不重试。只终止本任务自己的子进程以执行预算保护，不触及其他作业。公开工具没有训练、反向或AP入口。

CPU检查：`python -B tests/test_r18_support.py -v`，仅构造张量与原attention叶层、原matcher，无检测器/checkpoint/图像。覆盖坐标/插值/半开边界、零分母、不同头权重分母、无效stencil移动反例、P/B0钩子原输出/状态/RNG不变、原matcher空GT及预算/不可覆盖拒绝。需要torch/torchvision匹配环境，不改训练环境。

本地准备器`prepare_r18_support.py`只接收既有终态回执目录、development成员清单、已冻结提案及一个不存在的输出目录；生成独立合同和源码包，不连接服务器或启动。服务器执行步骤为独立source/material/resource准入、隐藏CUDA contract-check、真实CUDA身份precheck、唯一tmux启动、一次即时快照，随后按当前授权另做终态只读审计和精简回传。不得重放历史正式链。

无有效配对/无触发/样本不足保持原状，不扩图。实际取数与不足之处在终态交付登记；CPU通过不构成真实机制或方法收益证据。后续方案只可提出审查，不自动训练。

本次已完成的[终态结果](RESULTS.md)、[完整汇总](SUMMARY.json)和[配对方向表](TABLES.md)保留负向值与解释限制。`r18_support_analyze.py`只读取已有JSON生成展示，无模型、数据或AP依赖；3项独立构造展示检查与正式结果分开。
