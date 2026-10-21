# 采样几何直接先例与公式差异

核对日期：2026-10-21。对象固定为 `ab88a9b3b2c8fc1f3203269f4ae022d52b0dd8cb` 的 B0 / B1 / P。仅作本地只读文献和源码核对；运行中的 `attempt-formal-001` 沿用原合同，不由本文件改变。文献确认不构成收益或发表结论。

Deformable DETR、RT-DETRv2、Box-DETR沿用冻结版本 `docs/research/r18_sampling_span_20261020/PLAN.md` §3 的已审正文范围，不重做综述。新增核对为 DAB-DETR v4 正文与附录，以及最多两个直接相关先例：Focal DETR 的采样范围/模式、torchvision RoIAlign 的单特征单元下限。原文、版本、访问边界与SHA登记见 [REFERENCES.json](REFERENCES.json)。

## 统一记号

令四维reference为 `r=(c_x,c_y,w,h)`，各坐标相对画布归一化；level `l` 的特征尺寸为 `(H_l,W_l)`，每head的点数为 `n_l`，学习的二维offset为 `d_lmk`。三臂的原生四维reference分支为

```text
s_lmk = c + 0.5 * (d_lmk / n_l) ⊙ e_l
B0: n = (4,4,4), e_l = (w,h)
B1: n = (8,2,2), e_l = (w,h)
P : n = (4,4,4), e_0 = (max(w,1/W_0), max(h,1/H_0))
                         e_1 = e_2 = (w,h)
```

这里的 `e_l` 是用于采样计算的reference extent，不是对回归框的改写。特征坐标为 `u=W_l*s_x-0.5, v=H_l*s_y-0.5`，与原有 `align_corners=False` 双线性取样一致。P在普通query和DN上均按reference本身判断，不使用GT大小。

## 一张公式差异表

| 方法 / 核对范围 | 改变的计算与公式 | 与本候选的已确认差异 | 尚不能主张的内容 |
|---|---|---|---|
| **Deformable DETR**，既有审查：v4，§4/§5、附录A.4 | 稀疏多尺度取样与逐层框更新；四维框分支按reference宽高缩放offset。统一记号即 `s=c+0.5*(d/n)⊙(w,h)`。 | P沿用框引导稀疏取样，仅将最细level的采样extent逐轴取最大值。 | 框尺度调制offset、稀疏双线性取样和径向offset初始化均已有先例，不能作为P原创。 |
| **RT-DETRv2 / B1**，既有审查：v1 §2.1/§2.2/§3.4、固定官方代码 | 允许不同尺度点数 `n_l`；B1采用原生 `(8,2,2)`。总12点联合softmax，原生径向bias及 `1/n_l` 随分配改变。 | P保持 `(4,4,4)`及联合归一化，修改 `e_0`。B1还改变初始细层注意力质量分配，不能描述为“仅改变点位置”。 | 同点数不证明同延迟，B1不是独立的新算法，也不是Box-DETR复现。 |
| **Box-DETR**，既有审查：v1 §3/§4/§5 | 为不同head产生box agent，替换原条件注意力使用的中心reference；即“原框 → head-specific agent → positional attention”。此处为计算关系示意，不冒充原文编号公式。 | P保持中心 `c`、query位置编码及回归reference，修改已有稀疏采样的extent。 | “显式使用框几何”已被覆盖；未固定复现Box-DETR代码，不声称优于它。 |
| **DAB-DETR 主模型**，本轮正文：v4 PDF页6–9，§4.2–4.5、Eq.(1)/(6)/(7)，Tables 2/3 | `P_q=MLP(PE(x,y,w,h))`；dense positional logit为 `[<PE(x),PE(c_x)>*(w_ref/w)+<PE(y),PE(c_y)>*(h_ref/h)]/sqrt(D)`，`(w_ref,h_ref)=sigmoid(MLP(C_q))`，逐层更新anchors。 | 调制的是dense attention的Gaussian-like位置先验及动态query；P只改已有稀疏取样位置中的最细extent，未加入DAB宽高调制或query更新。 | 原文并未给该位置先验严格Gaussian分布保证；读到的公式没有一个feature cell的extent floor。不能把“宽高影响注意范围”称P独有。 |
| **DAB-Deformable-DETR 变体**，本轮附录B/C，PDF页12–15、Fig.8、Table 5 | 将动态anchor-derived positional queries加入Deformable DETR，保留其稀疏注意力。 | 因此“DAB只用dense attention”只适用于主模型，不能用来排除其稀疏变体；该变体仍非P的extent下限操作。 | 附录对照检验query formulation，不检验最细尺度下限；没有本项目R18/VisDrone条件的结论。 |
| **Focal DETR**，本轮PMC正文§3.1/§3.3/§4.4–4.7、Tables 2–4 | `s=p_init(pattern)+Δp_learned`；四种初始化模式：方形边长 `2L0`、`L0/2`，矩形 `L0×2L0`、`2L0×L0`；增加并融合平行采样分支，另有target-aware矩阵。 | 已明确研究过采样范围、形状和学习offset的取舍；P不加分支/矩阵/点数，仅对每个过小reference的最细extent取下限。 | 此处的 `L0` 是论文模式刻度，不能偷换为P的实际feature-cell约束；本轮未认证作者代码。 |
| **torchvision RoIAlign legacy**，本轮官方v0.19.1 Python/C++源码 | `aligned=False` 时 `roi_width=max((x2-x1)*spatial_scale,1)`，高度同理；按该extent均匀划bin并双线性插值。`aligned=True` 不执行该下限。 | **单特征单元宽高下限已有明确算子先例。** RoIAlign以ROI起点为锚、bin内规则采样；P保持中心、使用学习offset，只作用于最细level，非同一完整算子。 | 不能声称“max(width,one-cell)”的数学操作本身原创；该legacy约束也不证明P能提升检测精度。 |
| **本候选P**，冻结适配源码 `sampling_references` / `FinestSpanAttention` | 最细层 `e_0=max((w,h),(1/W0,1/H0))`；其余同原生。固定768×1344时最细特征96×168；应用于3个decoder层。 | 确认的实现差异是“最细采样extent的逐轴单元尺度下限”，不改回归reference、中心、query位置编码、loss、query数或点数。 | 未建立该特定组合的首创性；仍未知是否存在相同的decoder实现、是否有实际收益、机制是否在普通query上充分触发。 |

## 范围与解释限制

DAB正文已核到Table 3中宽高调制、anchor更新及温度的拆分对照，附录Table 5还区分了Deformable变体；这些是COCO上对应结构的证据，不能移植成P的有效性证明。DAB附录I的密集、很小及很大对象失败案例也已读（PDF页16–18）。

Focal DETR并不支持“范围越大越好”的一般结论：Table 4中的采样单模块行，相对同表基线AP为43.4→43.7，但APS为26.3→25.8、AP75为47.5→47.3；完整组合为44.7 AP。§4.7文字归因与表中逐行差值存在口径不一致，本核对优先保留原表，未复现或修正作者实验。它支持存在范围/背景取舍，不能直接证明P的高IoU机制。这里只抄录极少数原表数值作审查，不作为本项目结果。

**reference extent下限不等于实际采样跨度下限。** 固定query/head/level，实际横向跨度为

```text
span_x_cells = 0.5 * W_l * e_l,x / n_l * (max_k d_lmk,x - min_k d_lmk,x)
```

即使 `W0*e_0,x>=1`，offset相等或为零仍可使跨度为零；offset也可学习补偿。相同双线性stencil不表示同插值权重或同输出。新增公式表仅澄清上述差异，不要求运行实验补采，不变更已有机制日志、计算或科学门槛。

本次限定搜索未确认与P完全相同的“只对最细decoder level施加单cell reference-extent下限”实现；搜索无命中不构成不存在的证明。特征尺度约束、范围选择、框引导注意力分别已有直接先例，当前原创性保留未知。已授权三臂实验继续按其原合同决定工程、质量和成本结果。
