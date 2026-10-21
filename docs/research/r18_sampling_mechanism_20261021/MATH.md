# 参考宽高下限、坐标敏感度与等价性的边界

本文仅对已冻结的 B0/P 计算进行数学辨析。全部示例均为人为构造，不使用检测器、权重、真实图像或实验指标。下文的“原生”指共同协议 B0 使用的四维 reference 采样公式；B0 与 P 都采用同一确定性双线性 gather 后端。该辨析不改变正在执行的三臂实验，也不构成已取得收益、已定位退化原因或已建立原创性的结论。

## 1. 符号与实际计算

固定一个解码器层、图像、query 和注意力头。第 $l$ 个特征尺度为 $H_l\times W_l$，参考中心为 $c=(c_x,c_y)$，参考宽高为 $r=(r_x,r_y)=(w,h)$，均以归一化图像坐标表示。这里的 reference 是解码器传入的预测／DN 参考量，不是按 GT 尺度替换的统计量。第 $k$ 个 offset 为 $d_{lk}\in\mathbb R^2$，该尺度点数为 $n_l$；令 $a_l=1/(2n_l)$。B0 与 P 均为 $n=(4,4,4)$。

在理想实数运算中，原生位置为

$$
s^{\mathrm N}_{lk}=c+a_l\,d_{lk}\odot r. \tag{1}
$$

P 使用采样专属 extent

$$
e_l=\begin{cases}
\max\!\left(r,(1/W_0,1/H_0)\right), & l=0,\\
r, & l=1,2,
\end{cases}
\qquad
s^{\mathrm P}_{lk}=c+a_l\,d_{lk}\odot e_l. \tag{2}
$$

max 为逐轴运算。三个解码器层都采用该规则；同层的普通 query 与 DN query 共同经过该模块。P 不改中心、回归 reference、query 位置编码、点数或注意力权重的生成公式。实现先按点顺序扩展 reference，仅最前四点对应的 extent 接收最细层下限，而不是把三个 level 的张量误当作十二个点。

原生 offset 由共享仿射头生成，$d=W_dq+b_d$；注意力权重由另一仿射头后在全部十二点上做 softmax。代码依据：[offset 与 softmax](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/blob/ab88a9b3b2c8fc1f3203269f4ae022d52b0dd8cb/vendor/rtdetrv2_pytorch/src/zoo/rtdetr/rtdetrv2_decoder.py#L135-L150)，[P 的逐点 extent 适配](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/blob/ab88a9b3b2c8fc1f3203269f4ae022d52b0dd8cb/src/sparse_rtdetr/research/sampling_span.py#L26-L65)。

## 2. Offset 到采样坐标的 Jacobian

以下均为**固定 $c,r$ 及其他输入时的偏导数**，不是完整检测网络的损失梯度。由式（1）和式（2），

$$
J^{\mathrm N}_{s,d}=a_l\operatorname{diag}(r),\qquad
J^{\mathrm P}_{s,d}=a_l\operatorname{diag}(e_l). \tag{3}
$$

令 $D_l=\operatorname{diag}(W_l,H_l)$。align_corners=False 下，归一化位置对应的特征坐标为 $u_{lk}=D_ls_{lk}-(1/2,1/2)$，故

$$
J^{\mathrm N}_{u,d}=a_l\operatorname{diag}(W_lr_x,H_lr_y),\qquad
J^{\mathrm P}_{u,d}=a_l\operatorname{diag}(W_le_{l,x},H_le_{l,y}). \tag{4}
$$

因此，P 在最细尺度对两个对角元素均有

$$
\left(J^{\mathrm P}_{u,d}\right)_{aa}\geq a_0=1/8. \tag{5}
$$

式（5）的单位是“一个 offset 数值单位造成多少特征坐标单位的变化”。它给出最细层 **offset→坐标局部比例** 的下界；不是任意一个实际梯度、一次 SGD 位移或采样范围的下界。原生 reference 为正时式（3）也非零，但随着 reference 逼近零，没有统一的正下界。其他两个特征尺度不受 P 下限改变。

若将 offset 头输入 $q$ 也视为变量，而 reference 暂时固定，则位置对 $q$ 的这一条路径为 $a_l\operatorname{diag}(e_l)W_{d,lk}$。完整 $\partial z/\partial q$ 还包含 softmax 权重、self-attention、残差及后续网络等路径，不能仅用这个局部矩阵替代。

代码先作 $g=2s-1$，再按 $u_x=((g_x+1)W-1)/2$、$u_y=((g_y+1)H-1)/2$ 计算特征坐标。上述 $u_x=Ws_x-1/2$ 是实数化简，不能把式子重排当成浮点实现的逐位同一性。[实际坐标运算](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/blob/ab88a9b3b2c8fc1f3203269f4ae022d52b0dd8cb/src/sparse_rtdetr/baseline/training_v2b_deterministic_sampling.py#L88-L111)，[grid 构造](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/blob/ab88a9b3b2c8fc1f3203269f4ae022d52b0dd8cb/src/sparse_rtdetr/baseline/training_v2b_deterministic_sampling.py#L147-L157)。

## 3. Extent 与实际采样跨度不是同一个量

对同一 query、同一头、同一尺度的一组点，定义轴 $a$ 上的特征坐标跨度

$$
S_{l,a}=\max_k u_{lk,a}-\min_k u_{lk,a}
=D_{l,aa}\,a_l\,e_{l,a}\left(\max_kd_{lk,a}-\min_kd_{lk,a}\right). \tag{6}
$$

式（6）使用同一中心和同一 extent，因此能把公共量提出。固定相同 offset 时，P 与原生的该轴跨度之比为 $e_{l,a}/r_a$（原生跨度非零时），所以 P 不缩小这个特定条件下的跨度。然而，若全部 offset 相同，括号为零，两种方案的跨度都为零；若全部 offset 为零，两种方案都在中心采样。即使 $e_{0,x}\geq1/W_0$，也不能推出 $S_{0,x}\geq1$。

上述同 offset 结论不能直接外推到训练后的两个模型：两者可能学到不同 offset、reference、特征和注意力权重。也不能将一个头内跨度与多个头／多个 level 合并后的范围混用。最细层下限为一个特征单元，仅规定用于乘 offset 的尺度，不规定至少访问两个网格单元或新增多少有效信息。

## 4. 位置变化、插值变化与损失梯度

固定注意力权重 $\alpha_{lk}$ 和投影后的特征场 $F_l$，一个头的聚合为

$$
z=\sum_{l,k}\alpha_{lk}\,\mathcal B(F_l,u_{lk}), \tag{7}
$$

其中 $\mathcal B$ 是双线性插值并使用零填充。在一个不跨网格边界的单元内部，若 $u=(i+\xi,j+\eta)$，$0<\xi,\eta<1$，则

$$
\frac{\partial\mathcal B}{\partial u_x}
=(1-\eta)(F_{i+1,j}-F_{i,j})+\eta(F_{i+1,j+1}-F_{i,j+1}). \tag{8}
$$

$u_y$ 方向同理。令 $g=\nabla_zL$，则在固定上述其他变量及可微区间时，单个 offset 的坐标路径满足

$$
\nabla_{d_{lk}}L
=a_l\operatorname{diag}(W_le_{l,x},H_le_{l,y})\,
\alpha_{lk}\,[\nabla_u\mathcal B(F_l,u_{lk})]^\mathsf T g. \tag{9}
$$

式（9）解释了一个可检验但尚未确证的可能性：P 改变坐标敏感度，也可能改变实际取样处的特征导数。但从式（5）不能推出式（9）的范数增加，更不能推出梯度方向有利于高 IoU 定位。P 同时改变 $u$，因此两方案中的 $\nabla_u\mathcal B$ 及后续 $g$ 一般也变了；不能只比较 $e/r$ 而把其他因子当作相同。

必须保留以下反例：

- **损失梯度为零。** 取 $L=\|z-y\|^2/2$ 且 $z=y$，则 $g=0$，即使坐标 Jacobian 变大，offset 梯度仍为零。这里只是一个构造例子，不主张任意数值为零的函数都有零导数。
- **特征常值。** 若所有可能被访问的相邻特征值均为同一常数，且插值支持不触及零填充边界，则式（8）为零；位置与权重可以变化，而采样输出不变。不能把“换了 stencil”必然解释为“换了有效特征”。
- **共享参数梯度抵消。** 同一 offset bias 由多个 query／样本共享；若两个样本对其贡献分别为 $v$ 与 $-v$，总梯度为零。更一般地，$\nabla_{W_d}L=\sum_q\nabla_{d_q}L\,q^\mathsf T$，局部梯度放大不保证求和后的参数梯度放大。不同点的损失路径也可能在进入公共上游参数时抵消。
- **零填充支持外。** 双线性插值支持外的有限位置可产生输出零和位置梯度零；扩大 extent 可能把点推向该区域。不能把“位移更大”直接等同于“梯度更强”。零填充边界或网格整数边界处可能不可微，应依据实现的分段／单侧导数，不外推式（8）的严格内部导数。
- **同 stencil 仍可能改变输出。** 取单元内 $F_{i,j}=i$ 的水平斜坡，两个点留在相同四邻点中，但 $\xi$ 不同会改变插值权重和输出；反之常值特征中跨 stencil 也可保持输出相同。

因此，stencil 是否改变、插值权重是否改变、采样输出是否改变、损失梯度是否改变是四个不同问题。[确定性 core 的邻点、权重、有效范围及聚合](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/blob/ab88a9b3b2c8fc1f3203269f4ae022d52b0dd8cb/src/sparse_rtdetr/baseline/training_v2b_deterministic_sampling.py#L100-L163)。

## 5. Offset 重缩放：三种等价性必须分开

逐轴定义 $\rho_l=e_l\oslash r$。冻结实现拒绝非正 reference extent；在有限正 $r$、有限 $e_l$ 的理想实数条件下，$\rho_l\geq1$ 且可以定义。固定一组 $q,c,r$、特征和 attention 权重时，

$$
d^{\mathrm N}_{lk}=\rho_l\odot d^{\mathrm P}_{lk}
\quad\Longrightarrow\quad
s^{\mathrm N}_{lk}=s^{\mathrm P}_{lk};\qquad
d^{\mathrm P}_{lk}=d^{\mathrm N}_{lk}\oslash\rho_l
\quad\Longrightarrow\quad
s^{\mathrm P}_{lk}=s^{\mathrm N}_{lk}. \tag{10}
$$

这证明的是**一次输入上的坐标可重参数化**。固定其他量并有相同坐标时，理想实数的插值和聚合值相同。不需要把下限解释为原生坐标完全不可表达的新区域；原生 offset 并无这类固定幅度上限。

这尚未证明**整个可表达函数族相同**。offset 来自所有 query 共享的仿射头 $W_dq+b_d$，而 $\rho_l$ 随 reference、尺度和轴改变。若 $\rho_l$ 对所有输入恒定，可以逐个输出通道缩放 $W_d,b_d$ 来吸收它；若它随输入变化，一般不能由一组固定参数直接完成乘法。以模块独立输入为例，同一个 $q$ 配上两个不同的 $r$ 时，仿射 offset 头输出相同，但式（10）所要求的 $\rho_l\odot d$ 可能不同。在完整网络里 $q$ 与 reference 相关，这个构造不证明两个完整检测器函数族严格不同；它只否定“逐点可以重缩放，所以共享网络一定全域等价”的推理。

这更未证明**训练轨迹等价**。B0 与 P 并未按式（10）重缩放初始 offset 参数，而是使用同一初始参数。即使在一个固定标量 $\rho$ 的简化系统中令 $d_{\mathrm N}=\rho d_{\mathrm P}$，要使普通 SGD 的参数映射在一步后继续保持，仍需对应改变学习率（简化线性情形为 $\eta_{\mathrm N}=\rho^2\eta_{\mathrm P}$）。本实验使用相同优化设置，并涉及 query 变化、AdamW、梯度裁剪、EMA、非线性及跨样本共享，不能援引这个简化条件宣称训练等价。

当 $r$ 很小时，$\rho=e/r$ 可以很大；正值并不保证数值上适合显式重缩放。式（10）的实数恒等式不保证 BF16/FP32 的乘除重排逐位一致，也不保证舍入后的插值 stencil 与梯度完全一致。冻结 core 为采样计算选择 FP32/FP64，但 offset 与位置的上游运算仍受实际 autocast、dtype 和舍入影响。[采样 dtype 选择与转换](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/blob/ab88a9b3b2c8fc1f3203269f4ae022d52b0dd8cb/src/sparse_rtdetr/baseline/training_v2b_deterministic_sampling.py#L58-L90)。

## 6. 真实 detach 路径与辅助回归路径

不能把式（2）的 $\partial\max(r,f)/\partial r$ 直接解释成“P 切断了框预测头的采样梯度”。冻结 decoder 的真实执行关系为：

1. 普通 query 的初始 encoder bbox logits 在进入 decoder 前已 `detach()`；供 encoder 辅助损失使用的未 detach bbox 另外保留。DN 初始框由标签框加噪生成，再与普通 reference 拼接；它不是 P 新建的可训练尺度变量。[初始化及拼接](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/blob/ab88a9b3b2c8fc1f3203269f4ae022d52b0dd8cb/vendor/rtdetrv2_pytorch/src/zoo/rtdetr/rtdetrv2_decoder.py#L508-L527)，[DN 框生成](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/blob/ab88a9b3b2c8fc1f3203269f4ae022d52b0dd8cb/vendor/rtdetrv2_pytorch/src/zoo/rtdetr/denoising.py#L34-L77)。
2. 第 $t$ 层 attention 和 query 位置编码接收 `ref_points_detach`。层内产生 `inter_ref_bbox` 后，将 `inter_ref_bbox.detach()` 作为下一层这两条路径的输入。P 的 max 位于 attention 接口内部，所以也作用于这份已停止反向传播的参考值。[逐层输入与 detach](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/blob/ab88a9b3b2c8fc1f3203269f4ae022d52b0dd8cb/vendor/rtdetrv2_pytorch/src/zoo/rtdetr/rtdetrv2_decoder.py#L258-L282)。
3. 同时，训练时第 $t>0$ 层输出框另按 `sigmoid(bbox_head[t](output)+inverse_sigmoid(ref_points))` 计算，这里的 `ref_points` 保留上一层 `inter_ref_bbox` 的计算图，并不是上述 detached 变量。因此，最终／辅助框监督仍存在通向前层框预测的非 detach 路径。第0层直接输出自己的 `inter_ref_bbox`。这些框再进入最终、辅助和 DN 输出。[训练框输出](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/blob/ab88a9b3b2c8fc1f3203269f4ae022d52b0dd8cb/vendor/rtdetrv2_pytorch/src/zoo/rtdetr/rtdetrv2_decoder.py#L267-L284)，[输出分组](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/blob/ab88a9b3b2c8fc1f3203269f4ae022d52b0dd8cb/vendor/rtdetrv2_pytorch/src/zoo/rtdetr/rtdetrv2_decoder.py#L585-L594)。

为区分两条路径，可记 $b_t=\sigma(f_t(h_t)+\operatorname{logit}(\operatorname{sg}(b_{t-1})))$ 为用于下一层迭代的中间框；attention 和 query 位置编码使用 $\operatorname{sg}(b_{t-1})$。而训练报告的第 $t>0$ 层框为 $\hat b_t=\sigma(f_t(h_t)+\operatorname{logit}(b_{t-1}))$。两者前向数值可一致，反向图不同；实际 `inverse_sigmoid` 还有数值截断，不能忽略其边界来宣称处处同导数。

在真实图里，经当前 attention 的 reference 坐标路径对前层 bbox 输出的梯度被 stop-gradient 阻断；offset 头、注意力权重头、value 特征、query 内容和辅助回归路径仍可接收梯度。若在单独构造张量上让 reference `requires_grad=True`，那么 max 在阈值下方的形式偏导为0、上方为1、等值点需遵循实现规则；这只能验证适配函数本身，不能冒充真实 decoder 中框预测头的梯度路径。

## 7. 能支持的机制陈述

由冻结计算能够确认：P 改变小 reference 下最细层 offset→坐标的局部比例；相同 offset 下可能改变采样位置、插值权重和聚合输出；该变化不新增点数或网络参数。实际跨度、有效特征导数、共享参数梯度和训练终点收益均不由这个局部 Jacobian 单独决定。

尚不能确认：共同协议 B0 的定位瓶颈就是坐标敏感度不足；P 能提高普通 query 的有效梯度；DN 触发能转化为推理收益；或者任意测得的 AP 变化只能由这一条路径解释。这些命题需要与真实终态及已有机制摘要分开判断；若现有摘要缺少某类梯度或分组字段，保留为未知，不要求运行链补采。
