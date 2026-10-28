# S/A共同归一化修复：CPU算子范围与GPU提案

## 修复定义

原S/A局部头对x方向沿dim=2、y方向沿dim=3计算：

```text
a = sigmoid(z)
w_i = a_i / sum_axis(a)
p = sum_axis(w * x)
```

对有限实数logit，sigmoid严格为正，故有

```text
w = softmax(logsigmoid(z), dim=original_axis)
```

这是同一实数函数的稳定求值，不是改成`softmax(z)`。例如[0,1,2]上二者最大权重差约0.248。实现直接使用PyTorch log-sigmoid及softmax，无epsilon、clamp、温度、半径或非有限值替换。只接受原实际FP32及验证用FP64；NaN/Inf明确拒绝，不保证非有限输入可计算。

保留归一化权重的链式导数。令p=sum_i(w_i*x_i)，对标量上游g：

```text
dp/dx_i = w_i
dp/dz_i = w_i * (x_i-p) * (1-sigmoid(z_i))
```

共享多个特征通道时，logit梯度对通道的上游加权项求和。新增实现保留`1-sigmoid(z)`因子；直接对z作softmax会丢掉它。有限精度可能使单个权重仍为0，但softmax沿该轴的总质量保持为1（按误差界核验），不再先形成可能为0的sigmoid分母。

## 事前判据与参考

`LOGSIGMOID_CRITERIA.json`在测试前独立提交，SHA `5895115d1e591c25571d3146c0dd1939ae0ca9cc04b45ac7971d383cb578a6f1`。正常域logits[-12,12]、features[-4,4]，两轴及非正方形7×9；旧/新前向atol2e-6/rtol2e-5，梯度atol5e-6/rtol5e-5。失败局部样本与极端输入的新FP32相对FP64，输出atol2e-4/rtol2e-5、梯度atol2e-5/rtol2e-4；FP64自动微分对独立前向及解析梯度atol1e-12/rtol1e-10。FP32/FP64轴和绝对误差上限分别1e-6/1e-12。

allclose判据为atol+rtol×abs(reference)；接近0由绝对界控制，仍记录max_abs、RMS，以及abs(reference)>=1e-5区域的最大相对误差，不把小梯度相对误差隐藏为逐位一致。阈值未在结果后调整。

独立FP64前向用Python math的分段log1p/exp实现，再max-shift归一化；梯度用上式解析计算，另与Torch FP64自动微分交叉核对。Double直接sigmoid/sum仅在其分母仍为正的范围作额外参考；全-1000/-10000等输入连该直接Double公式也会下溢，不能把它当万能真值。

失败样本源于已保存306066B局部tensor（SHA51f994b7f665881a86cbf6552de8fd2d3027e19c102b7a16191d650c3dd8428a）：仅选中query的pre.3和两组attention logits转为193399B轻量JSON，逐Float32字节SHA核对，供CI独立复现。没有将图像、checkpoint、完整匹配或大缓存加入Git。

## CPU结果与反例

服务器CPU109项检查、288组数值比较通过，原失败局部样本的两轴新权重/池化输出/局部梯度均有限并满足预定参考判据。正常域旧/新FP32最大绝对差：权重1.19209289551e-07，输出7.15255737305e-07，特征梯度2.98023223877e-08，logit梯度5.96046447754e-08。32组正常域旧/新比较全部存在位差；不得写成逐位或训练轨迹等价。真实局部输出新FP32对FP64最大绝对差约1.5762e-4；S/A构造局部头前向/梯度最大绝对差8.9407e-8。

保留：旧失败局部公式分母为0（原生产在除法之前被检查拦截，不能把未执行的0/0称原模型输出）；全负极端下直接Double sigmoid也失效；常值零特征使logit梯度严格为0；有限输入的卷积仍可溢出。归一化稳定不保证非零损失梯度、整体模型稳定或检测收益。极端输入的有限输出结论仅针对有界特征及此归一化/池化算子，不推广到任意巨大特征或上游网络。

CPU只运行算子及实际BoundaryHead类片段，不加载完整检测器或权重，不读图像；4次构造局部头前向，自动微分允许，optimizer.step被禁止，真实优化器更新/GPU/评分/测速均0。

## 影响范围与旧证据

S/A共用`normalized_sigmoid_axis`。实际原轴、参数名/数量、初始化完全一致；Model（含N旁路、RoI与detach）、Criterion/损失分母、query/DN、训练配置和Engine AST均不变。VERSION升为`r18-sabl-control-logsigmoid-v2`，导致配置/criterion身份、保存恢复绑定版本发生明确变化；旧v1状态不再能静默按v2恢复。没有状态迁移或新GPU执行入口。

原N前向计算未改；原S/A在正常域即可能出现浮点位差及后续轨迹差异，旧S终点不能直接作为修复后A的同期同配方控制。原N/S终点、原938完整诊断状态、原release/STOP、成本账本保留原字节。未扩大或重置原科学/资源门槛；修复后的质量验证仍待独立方案和授权。

本分支CI是显式operator-only：Torch CPU算子/梯度及管理检查（Python3.10/3.12），不执行旧含optimizer.step的训练/状态CLI回归；旧job跳过状态明确保留，不当作成功。旧冻结诊断仍运行原79ba593科学release，本轮没有更改它。完整模型、optimizer/EMA恢复以及新版本实际GPU路径尚未受验。

## 后续唯一GPU验证提案（本轮不执行）

详见`LOGSIGMOID_GPU_PROPOSAL.json`。先另行实现和CPU验收forward-only入口、显式旧raw权重到v2的严格只读桥接；不能调用旧正式训练入口续跑。固定失败逻辑batch16，完整输入/GT身份核对后仅前8张；A用原938 raw model及RNG，S用自身原e24 raw model及RNG，各1次train-mode/grad-enabled前向，保持原AMP路径，0 backward/optimizer/EMA更新。每模型原/新局部算子复用同一pre-pooling中间量，附加局部计算计入时间，不增加检测器forward。

A到达的选中RoI/pre.3/logits须与原保存Float32字节精确一致，再检验新权重/输出有限及轴和误差<=1e-6。S只检验自身算子兼容性；两者不同历史状态的输出不能当受控质量比较。任一身份/数值/资源或上游字节不一致即STOP，无重试；通过也立即闭合。

提案总计最多2个batch8检测器forward，GPU阶段<=300秒、wall<=600秒、allocated<=18GiB、新增存储<=128MiB、全部stdout/stderr/附件回传<=1MiB。原余额12849forward/49881.05051984998秒不变；未来若执行且耗满上限，余额才减到12847forward/49581.05051984998秒。此为初步工程上限，不是实测完整推理成本。

通过最多证明两个指定batch/state的完整前向数值路径。若要科学质量判断，必须另定S/A共同修复、共同父N/初始化/日程的受控实验；不只补A后拼旧S，不默认恢复或重跑24轮，本轮没有GPU授权。


源码指纹使用事前记录的Python3.10 AST字段模式。初版b2fea043的Python3.12 CI在空type_params字段差异处失败，原日志保留；后续仅在测试指纹中省略这个空字段，非空类型参数必须拒绝，未修改科学实现、旧指纹或数值判据。没有重跑旧CI；新提交触发正常新CI。
