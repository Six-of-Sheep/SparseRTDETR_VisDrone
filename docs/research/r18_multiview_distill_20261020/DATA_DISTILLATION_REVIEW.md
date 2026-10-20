# Data Distillation正文核查（2026-10-20）

用户提供PDF单份保留incoming，不入Git：`Radosavovic_Data_Distillation_Towards_CVPR_2018_paper.pdf`，3,937,008 bytes，SHA256 `db6e6ac4fc7f96e85914614ee4daa86c270f32fe0a6043e48178ee31e2d5fe34`。10页完整可解析、未加密；页内题名 *Data Distillation: Towards Omni-Supervised Learning*，Ilija Radosavovic、Piotr Dollár、Ross Girshick、Georgia Gkioxari、Kaiming He，CVPR2018，印刷4119–4128。未将用户字节说成出版社认证；本文件没有可核实的arXiv版本号/DOI，不补猜。来源入口：https://openaccess.thecvf.com/content_cvpr_2018/html/Radosavovic_Data_Distillation_Towards_CVPR_2018_paper.html （本轮依据本地PDF，不冒称重新联网认证）。

全文方法§3–4、实验§5–6及讨论均已阅读；PDF8页原表4与检测方法经渲染目视核对。页码以下均为PDF物理页。

|证据位置|原文实际做法/结果|本次采用或差别|
|---|---|---|
|2–3页，§3|同一teacher多变换预测、逆变换回统一坐标、聚合生成硬伪标签；标注集与额外未标注数据共同训练学生|一般“多视图teacher→单模型student”思想已有先例，不是本次创新。本次只用原有标注train_core4869，同时保留GT监督，没有额外未标注数据|
|3–4页，§4|关键点采用400–1200、步长100尺度及翻转；单尺度RoI共享，只对关键点head做多变换，热图平均取argmax|不能当作检测框融合实现，也不能说作者没有验证选择变换。本次固定whole+四角、五次完整forward|
|4页，训练设置|以标注集平均实例数量匹配伪标签置信门；标注/伪标注batch6:4；作者从ImageNet重训优于teacher微调，采用retrain|本次固定score≥0.5、上限300、新optimizer/EMA、短12轮续训；不是作者配方复现。该重训观察保留为失败风险，不擅自增加训练预算|
|6页，表2|短90k data-distillation63.6低于监督64.2；长270k/360k为66.5/66.6，监督130k65.1、270k64.7|日程不完全相同，不能拿论文主结果保证短续训有效；本次T0控制续训，T1控制整图teacher信息|
|7页，teacher/数据量/增强消融|teacher质量、未标注量及推理增强均有实验；学生TTA后仍有收益|不能把收益唯一归因为增强不变性；本轮也不预先指定迁移根因|
|8页，§6.1|Faster R-CNN/FPN/RoIAlign检测；多变换框并集，bounding-box voting合并；按类别匹配平均实例数选择门限|本次DETR集合匹配selected-class软KL+正框回归，hard NMS不投票；T1明确是同NMS的B式整图teacher，T2同权重五视图。并非完整Data Distillation或DETRDistill复现|
|8页，表4b|COCO co115+un120：R50 AP37.1→37.9、APS20.0→20.3；R101 AP39.2→40.1、APS21.7→21.7；RX101 AP40.1→41.0、APS22.6→22.9|不同任务/数据与训练日程，不能直接比较本项目AP，更不能保证航拍小目标改善。APS不升的对照也保留|

明确裁决：已有先例覆盖一般思想，当前任务保持**实用可行性探索**，不宣称新蒸馏方法或最近先例穷尽。正文未揭示需要改变当前科学公式/三臂控制的具体错误；因此不取消、不扩搜、不加臂、不改配方。DETRDistill作者实现及独立补充仍未固定核验，不冒称逐项复现；这不是本次实用探索的新增执行阻断。
