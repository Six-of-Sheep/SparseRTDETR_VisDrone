# 原C小目标阶段可达性：限定诊断

2026-10-18。原C三seed实用收益保留；固定Soft-NMS和12pass混合微调均关闭。
本次只对原C认证缓存做CPU诊断，不修改预测，不训练、不推理、不评分、不访问holdout。
不自动生成或运行下一候选。使用全部seed0/1/2、全部548图，主600/次300，IoU固定0.5/0.75。

## 要回答的问题

原C还没有覆盖的小目标，几何可达性最早在哪一层消失？六个嵌套集合依次为：
1. 五视图全部1500个query的几何框（类别无关、无分数门槛）；
2. 每视图top600中的正确类别候选；
3. 原classwise hard NMS全部保留候选；
4. 全图最终top600/top300；
5. 原category0 raster raw-ignore过滤之后（预算在前，不补位）；
6. COCO bridge每图每类maxDets100之后。

使用原bridge过滤后的score>0、iscrowd=0且原面积0<area<=1024的GT。
每GT在各阶段取最大连续坐标IoU，以第一个低于阈值的阶段作互斥桶。
最后仍有合格框归available。各桶和必须等于有效small GT数；两预算共享前3阶段。
同时保留每GT最大值/原身份、按图/类别/序列和seed的计数，不能只挑选失败例。

这是多对多几何可达性，不是AP/AR、一对一检测召回或TIDE的dAP。
第一层允许低分和错误类别query，故只能说明几何是否存在；第二层损失不等于
已证明分类头训练失败。后续阶段损失是固定候选删减的记录，不是独立因果归因。
available也不保证TP：评分排序、同框竞争、crowd及评价匹配仍可能影响指标。
不通过GT选框发布预测、不修改NMS、不扫参数、不求新oracle AP。

## 准入与实施

新脚本复用已冻结Soft-NMS final release中的只读认证、原raw解码和原评价输入过滤。
绑定revision007契约SHA b2ac33e7a99cfa4e8ea3ccce7297f5088272d839f98fae7b7e13c4e2a7f46c28，
逐项核查其370份源码；调用load_contract/context（不调用旧launcher/supervise/pipeline）。
先通过固定合成边界案例，然后一次生产尝试；逐batch核SHA，原C两预算逐候选精确
重建，复用原canonical评价输入证明。模型构建、CUDA、backward和评分入口显式禁止。
任何真实材料/数值/重建不一致即STOP不重试。RAM/磁盘/温度沿原CPU资源门禁。
大型逐GT记录留实验室，D只收汇总、版本、回执和SHA索引。

## 预先固定的解释规则

三seed分别报告，分母和未过滤/过滤数量明确；不以均值掩盖方向变化。
按各seed IoU.75主600第一丢失桶计数，检查同一桶是否均居首，报告其占全部
未达末层GT的比例。若首桶不一致，不选唯一瓶颈；若一致，也仅可排序后续调查，
不能据此宣布某种方法会提高AP。不设置事后“显著”阈值、不bootstrap。
本次结束最多提出一项与实际主损失阶段对应的最小对照；没有足够依据则明确缺证，
不自动启动新训练。GT等权与序列等权并列；不把固定诊断当作多次调参入口。

## 先例与限制

[TIDE ECCV2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/849_ECCV_2020_paper.php)
说明检测错误需区分类型及交互。本诊断没有运行TIDE的反事实AP分析，不能称TIDE复现。
development已反复使用，三seed也不是独立数据确认。本次不解除holdout边界。

原微调完整表：T-slice相对原C AP -.142403、APS -.375528、AP75-small -.734377pp，
但APL +2.082019pp；这种尺寸取舍不能推翻整体FAIL，也不支持“所有指标都退化”。
