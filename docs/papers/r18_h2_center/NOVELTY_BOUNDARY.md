# H2的限定创新边界核查

2026-10-25。目的：核查当前方法差异，不预判正在执行的U/D结果，也不要求证明全领域无人做过。沿用[方法稿](MANUSCRIPT_PREPARATION.md)和运行版本`e50bedbe8e41bdfd3c47d09f7076d9644e438e26`；不改公式、控制或门槛。

## 1 核查结论

**保留H2作为受验假设，但将贡献表述严格收窄到末层中心通路约束。** 本次限定检索未找到已确认的完全等价设计；这不是原创性证明。新增直接先例CF-DETR已覆盖“全局上下文粗定位，再用局部特征精定位”的总体思路，因此不能将这一叙事作为本文创新点。

当前可准确使用的一句话是：

> 在共同协议R18的末层，将中心logit残差限制为框相对四点的无偏置成对差分线性读出，删除直接上下文中心项，同时保留宽高和分类的上下文路径；用共享采样和差分、但加回直接中心项的U检验这一约束。

这句话描述方法差异，不含“首次”“已经有效”或“达到投稿要求”的判断。

## 2 新增先例与阅读范围

|来源|本轮实际读取范围|已有计算及对H2的约束|
|---|---|---|
|Cao等，CF-DETR: Coarse-to-Fine Transformers for End-to-End Object Detection，AAAI 2022，185–193；[官方题录](https://ojs.aaai.org/index.php/AAAI/article/view/19893)、[官方PDF](https://cdn.aaai.org/ojs/19893/19893-13-23906-1-2-20220628.pdf)|9页在线PDF；重点为PDF3–5页、图2/4、ASF/LCA、式1–3及迭代预测段|粗层结合全局上下文，细层从粗框提取多尺度ROI并细化框，分类沿用粗层。粗到细、局部定位、分类和回归不同处理均已有先例；已读方法没有H2的无偏置反对称差分中心约束|
|BorderDet: Border Feature for Dense Object Detection，ECCV 2020；[官方PDF](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460528.pdf)|17页在线PDF；PDF5–8页§3.2–3.4、图3、式1–2及PDF9–11页相关消融|边界采样/池化后保留并融合原点信息，用于分类及框细化。方向性局部信息和框相对采样不是新意；其已有读出不等于H2的受限中心项|
|VarifocalNet: An IoU-aware Dense Object Detector，CVPR 2021；[官方题录与摘要](https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_VarifocalNet_An_IoU-Aware_Dense_Object_Detector_CVPR_2021_paper.html)|官方题录/摘要，以及搜索返回的§4.4–4.5正文片段；PDF直接访问403后停止，**未完成全文审读**|星形框特征和框细化已有先例，进一步限制“框引导固定形状采样加轻量细化”的宽泛主张；片段不足以排除其所有等价实现|

本次CF-DETR指Cao等AAAI 2022论文，不是2025年同名实时调度工作。在线解析未保存PDF到本地，没有新增PDF字节SHA；访问记录不能冒充下载核SHA。

既有SD-DETR、CDC、PiDiNet、PiDiViT、SABL沿用原[来源索引](SOURCE_INDEX.json)及已记录章节，不重复宣称本轮全部重读。未复现任何作者代码。

## 3 等价性与可主张范围

|层面|现有证据支持的表述|不支持的表述|
|---|---|---|
|算子|成对差分是已有PDC形式；固定样本时等于成对反对称权重的线性读出|新差分算子、增加信息量、首次抑制共有分量|
|作用路径|约束第三层中心logit残差的直接输入，宽高/分类保留上下文路径|中心完全不依赖上下文；分类和宽高数值不变|
|表达能力|U将直接中心项置零可以表达D；D是受限函数类，比U少514参数|D表达能力更强、严格等参数、CPU代数等价意味着训练轨迹等价|
|最小控制|D−U检验删除直接上下文中心项这一结构限制是否有用|D−U单独证明反对称性优于任意自由四点读出，或排除所有容量/优化作用|
|经验结果|终态通过后可报告固定配方、单seed、同轮控制间的受验收益|提前认定有效、统计显著、跨seed稳定或原创性成立|

若将论文主张扩大为“反对称读出本身优于任意四点读出”，现有U不足以单独支持它；若主张“与参数减少无关”，现有514参数差异也需被正面处理。这些是解释边界，**本次不据此新增训练臂**。先看原D−U比较能否支持当前更窄的问题。

## 4 与结果相结合的决策

- 若终态质量/成本未通过：保留原失败裁决，结束固定配方；不为保住新颖性叙事自动调参。
- 若终态通过、中心统计一致：保留为单seed有证据的方法候选，再按[条件验证方案](CONDITIONAL_VALIDATION.md)审查重复验证。
- 若终态通过但中心记录不支持预期解释：可报告受控检测收益，机制解释仍未知；不得以选择样本或只报有利均值修补叙事。
- 后续出现完全等价先例：重新界定贡献或转为复现实证工作，不追溯改写现有结果，也不将“换R18”作为原创补偿。

## 5 有限检索记录

本轮共4组、12条查询，2026-10-25；深入范围限制为上述3篇，第三篇访问限制已明示。检索阴性只表示本范围没有确认等价设计。

1. `object detection "bounding box regression" "pixel difference"`；`DETR "center" "difference" "local" regression refinement`；`object detection "zero-sum" localization regression`；`object detection "paired" features "center refinement"`。
2. `CF-DETR coarse fine transformer local cross attention paper arxiv`；`VarifocalNet IoU aware dense object detector star shaped bounding box refinement CVPR 2021 paper`；`BorderDet Border Feature dense object detection ECCV 2020 paper regression`。
3. `"CF-DETR" "Local Cross-Attention"`；`"Coarse-to-Fine" "Detection Transformer" "Adaptive Scale"`。
4. `"object detection" "center" "antisymmetric"`；`"bounding box" "regression" "finite difference" feature`；`"DETR" "paired difference"`。
