# 切片、融合与成本：限定先例核查

检索/核查日期：2026-10-17。服务端搜索接口本轮不可用，改用浏览器检索、原文网页和作者GitHub只读文本。正文阅读与代码静态阅读分开记账；未安装/运行这些项目，未将公开AP数字与本项目排名。检索不是穷尽性证明，未确认代码不等于代码不存在。

| 工作与发表状态 | 正文机制及与本项目最近的对照 | 官方代码及核查程度 | 对候选的约束 |
|---|---|---|---|
| [SAHI，ICIP 2022；正文v5](https://arxiv.org/html/2202.06934v5) | §3切片推理与可选整图、重叠预测合并；另有切片微调，不能混为单一推理干预。已读方法及§4设置/结果；其VisDrone部分采用AP50、maxDets500，切片设置也不同 | [obss/sahi](https://github.com/obss/sahi/tree/b5a153ee7bcb1fd84a7b868575ff14131e7423ce)，读predict及postprocess。现代入口默认GREEDYNMM/IOS、class-aware，可整图联合；这不是2022论文算法版本的逐字证据 | 当前C是SAHI式适配。重新命名“整图+局部多视图”不构成创新。现代包还会根据配置改后处理，不能直接默认接入 |
| [AutoFocus，ICCV 2019正文](https://openaccess.thecvf.com/content_ICCV_2019/papers/Najibi_AutoFocus_Efficient_Multi-Scale_Inference_ICCV_2019_paper.pdf) | §4.1–4.3：训练FocusPixels小目标图，在低尺度筛选更高尺度chip，阈值/膨胀/连通区域/最小chip；Focus Stacking处理人工chip边界截断，原图边界例外。不是简单按whole检测框置信度选两格 | [作者SNIPER](https://github.com/mahyarnajibi/SNIPER/tree/66d7d5bff7f2e1e45e7e299c570e43e8698311aa)，读chips_inference.py的gmask/add_chips及autofocus配置。存在check_valid边界函数，但仅看到定义不能证明完整调用链已核实；未运行/审全模型 | 选择高分辨率局部区域已有明确先例。边界剔除依赖其chip/跨尺度契约，不应原样套在本项目不裁剪的框上 |
| [ROI-Gated SAHI，2026-08-25 arXiv预印本](https://arxiv.org/html/2608.23923v1) | 已读§III方法、§IV表I及限制：低成本proposer定位ROI并扩张，coverage触发静态fallback，refiner与全局NMS。自身设置中静态/自适应的速度和精度存在取舍，不能外推本项目必胜 | 原文可读；未确认作者可用实现/同行评审发表。正文coverage式为面积求和，Algorithm1用union面积，未由代码消除歧义 | 不主张“检测引导切片”新颖；备用只借用已知思想，需单独定义自身规则。未确认实现部分不作为精确复现依据 |
| [Soft-NMS，ICCV 2017](https://openaccess.thecvf.com/content_iccv_2017/html/Bodla_Soft-NMS_--_Improving_ICCV_2017_paper.html)；[正文](https://arxiv.org/html/1704.04503) | 已读§4算法及§6比较：硬删除改为连续分数衰减、每轮重新取最大；linear和Gaussian为不同固定规则；本文候选仅Gaussian σ=.5 | [作者cpu_nms.pyx](https://github.com/bharatsingh430/soft-nms/blob/b8e69bdf8df2ad53025c9d198ded909b50471d4f/lib/nms/cpu_nms.pyx)，读cpu_soft_nms全文：method=2、sigma=.5、inclusive+1面积、threshold=.001剪枝，交换数组后重选。固定commit与先前master读取的SHA相同 | 原始实现并非“永不丢框”；候选只复用衰减公式，连续框/no-cutoff/稳定身份排序属于项目适配。必须比较C而不止A，无创新声明 |
| [WBF，Image and Vision Computing 2021，104117](https://doi.org/10.1016/j.imavis.2021.104117)；[正文v3](https://arxiv.org/html/1910.13302) | 已读§3算法及后续实验设置：同类聚类、分数加权平均坐标、confidence按支持数/模型权重再校准，面向ensemble/TTA；不是单纯改NMS的名字 | [作者实现](https://github.com/ZFTurbo/Weighted-Boxes-Fusion/blob/796d4b1b6125ebef795cf14363dae00da78d5a1c/ensemble_boxes/ensemble_boxes_wbf.py)，读prefilter_boxes/get_weighted_box/cluster与confidence分支；README及arXiv journal reference确认期刊身份 | crop没有覆盖目标不等于模型投反对票；同一view重复query也不能算独立模型支持。作者prefilter会交换角点/clip到[0,1]，不符合项目现有保护；不能直接移植而声称仅变融合 |

作者实现的具体差异：现代SAHI的`merge_object_prediction_pair`取**外包并集框+最大分数**，GreedyNMM和NMM的合并关联不同；IoS=交面积/较小框面积，和IoU不同。WBF取**坐标加权平均**并另校准confidence。Soft-NMS**不改变坐标**，只衰减剩余框的分数。相互替代会改变不同变量；没有逐框证据时不笼统称为“边界修复”。

DAHI（density aided hyper inference）和cross-view agreement rescoring出现在本轮检索线索中，尚未完成可核验的原正文/作者实现阅读；不据摘要作方法差异或创新排除结论，也不把未检索充分写成不存在先例。本轮以以上五项完成候选筛选，不继续扩展列表。

## 固定代码快照

| 仓库/commit | 本轮实际读取文件 | 认证边界 |
|---|---|---|
| obss/sahi `b5a153ee7bcb1fd84a7b868575ff14131e7423ce` | sahi/predict.py、postprocess/combine.py、postprocess/utils.py | 关键合并函数和默认参数静态核对；非包运行测试/论文原始版本认证 |
| mahyarnajibi/SNIPER `66d7d5bff7f2e1e45e7e299c570e43e8698311aa` | lib/chips/chips_inference.py、configs/faster/sniper_res101_e2e_autofocus.yml | chip生成与配置；Focus Stacking整体调用链未完整核实 |
| bharatsingh430/soft-nms `b8e69bdf8df2ad53025c9d198ded909b50471d4f` | lib/nms/cpu_nms.pyx | 公式/逐轮选择/剪枝/坐标口径静态核对；未编译Cython |
| ZFTurbo/Weighted-Boxes-Fusion `796d4b1b6125ebef795cf14363dae00da78d5a1c` | ensemble_boxes/ensemble_boxes_wbf.py、README.md | 融合公式与默认预处理检查；未运行任何框融合 |

逐文件URL/字节/SHA记录保存在D盘任务目录的FETCH_INDEX_01/02/03.json；失败请求留痕。大PDF未作为交付副本保存，仅留AutoFocus正文文本及必要代码阅读快照，总量以收尾清单为准。Git只纳入本报告、决策及小型证据索引，不vendor这些第三方源码。SHA只确认所读快照字节，不代表作者签名或复现成功。

本报告支持的是下一次有限实用对照的排序，不是文献查新证明。正式新方法主张还需针对具体新计算规则补齐最近先例、强常规控制及独立确认，而不能以“未搜到”替代。
