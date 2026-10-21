# R18采样约束：本地机制预研究

交付为[数学推导](MATH.md)、[唯一条件性后续问题与选择规则](NEXT_QUESTION.md)和[小型CPU脚本](../../../tools/check_sampling_mechanism_cpu.py)。这不是新生产链或运行实验的中途评价，PR #16/#17/#18及原冻结规则均不改变。

核心结论：最细层P的offset→特征坐标Jacobian在理想实数中逐轴至少为1/8，但实际跨度及损失梯度没有相同保证。逐点offset重缩放可以重现坐标，不能据此证明共享网络函数族或训练轨迹等价。实际attention reference已detach，训练框输出另保留非detach路径；不将裁限的形式偏导当作前层框头的实际采样梯度。

## 执行CPU验证

在现有独立CPU环境运行；不需要数据集、检测器、checkpoint或服务器。`--frozen-root`应指向只读的ab88a9b源码，脚本先核对[SOURCES.json](SOURCES.json)列出的5份文件SHA，之后只提取纯extent函数的AST并加载独立插值core，不导入检测器注册／构建代码。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -B tools/check_sampling_mechanism_cpu.py \
  --frozen-root /path/to/read-only-ab88a9b-source \
  --output /path/to/new/CPU_SYNTHETIC.json
```

依赖为Python3.10+及PyTorch；本轮实际环境为Python3.12／torch2.8.0+cpu，FP64、单CPU线程，CUDA未初始化。此依赖只用于本地构造计算，不安装进或变更原实验环境。原main的CPU CI依赖固定文件可用于复现该环境；本脚本不被现有main CI自动执行，CI通过不能冒称其也运行了本研究。

16组检查、零skip：跨阈值及宽高不对称、局部Jacobian、逐点重参数化、阈值连续性、零偏移、实际gather与CPU grid_sample的双线性输出/feature和grid梯度、内部／边界／支持外坐标、避开折点的有限差分、同邻域不同输出，以及常值/零损失/共享参数抵消/支持外零梯度反例。另验证一个未来代理反向控制的FP64前向一致与指定反向；它不是生产适配或真实模型诊断。

构造结果中：零offset的实际跨度为0；共享标量参数的两项梯度为−0.125与+0.125，总和为0；一个边界例中原生offset梯度最大值为0.01875，P因采样移出支持区为0。这些数值只属于脚本定义的4×8特征，不能填入真实机制图、AP表或用来选择模型配方。

首轮开发检查在“阈值下方坐标不变”的断言中误纳入了没有下限的粗层；已保留失败源码SHA/位置/原因，定向修正为仅最细4点后通过。没有改生产公式或在真实科学异常后重试。脚本拒绝覆盖既有输出，错误保存为FAILED_CPU_STUDY并非PASS；本轮两次输出及SHA见下表，既有失败张量及归档STOP未动。

|本地必要回执|用途|
|---|---|
|`D:/Lzy_CV/workspace_management/delivery/research/r18-sampling-mechanism-20261021-001/CPU_SYNTHETIC_001_FAILURE.json`|开发断言失败留痕；不是科学实验STOP|
|`D:/Lzy_CV/workspace_management/delivery/research/r18-sampling-mechanism-20261021-001/CPU_SYNTHETIC_002.json`|16组构造CPU检查通过；没有AP或真实模型观测|

通过回执SHA256：`34efb6d56f0a3f520c6401a9c84cac4ca822da9398adf0556fbc9282da01c688`；首轮失败留痕SHA256：`a6cee2806ce5c4c96551aa17b8e3d83c881a233ee2ead64aef35ca14d39eaad9`。实际通过的脚本SHA256为`b2858f20ea978f3e49c2ec9985596f19f98e3e3c783e9f705ca27fe293a9feba`，源码索引SHA256为`dab56659e834b4010e30dc10d6ce8040091e4c2617c3797ff3db250abb52260f`。这些标识只绑定本地构造CPU检查，不认证生产运行。

## 引用与范围

5份冻结计算源码与PR #17既有正文审查身份见[SOURCES.json](SOURCES.json)，本轮仅追加一份与反向代理直接相关的原文，无泛综述。源码本地SHA与原manifest比对，不冒称本轮远端实物认证。PDF不下载/入Git，CPU回执不入Git。代码与文档从origin/main458d645建立独立分支，只读引用冻结版本，不把未合并的PR16代码复制为另一份运行版本。

没有真实模型forward、checkpoint加载、数据读取、训练、科学评分、GPU使用或服务器连接。终态分流是本轮提出的后续决策规则，不追写为原实验预设目标；不预设有效、原创或投稿可行。
