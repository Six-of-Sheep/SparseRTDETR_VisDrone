# 固定P12跨seed复验及条件测速

本轮只检查P12：整图view0＋原左上view1＋右上view2。seed0六组筛选与所有失败结果原样保留，不重选组合。seed1/2各自原e120 EMA的认证抑制前缓存，548 development、top600、原双评价器；不训练、不新增候选，不访问1602。

保持原坐标、每视图600、候选身份和同分排序、类内hard NMS严格IoU>.5、先截断再raw-ignore、不补位。复用原`merge_pair`与`screen`原字节。两个seed各自相对自己的A/C判断：AP与APS至少保留80% C−A增益；AP75-small相对C/A均≥−.20pp；APM/APL相对C/A均≥−.50pp，共8条件，未舍入值。AR-large完整报告，不事后新增硬门槛。原A/B/C AP不重算。

两个seed的原始输入SHA、canonical输入/过滤计数、274批缓存及A/B/C精确重建全部通过后，只执行2组新评分/4次评价器调用。工程正常则两个seed全部评分后裁决，科学失败不阻断另一seed。任何新工程/身份/数值/资源异常STOP，不自动重试。任一质量失败则关闭本轮，不执行GPU测速。

只有两个seed的8项全部通过，才准备并执行固定64图A/C/P12真实测速，覆盖seed0/1/2各自权重和固定成员。测速另行在执行前冻结最终入口、暖机/重复、平衡顺序、FP32/batch4、总batch forward预算和资源上限并通过CPU检查。端到端含预处理、传输、真实1/5/3视图forward、回映射、排序/NMS后处理；不得用缓存省略forward。报告各seed峰值显存、重复波动及P12/A与P12/C。新目标每seed P12/C≤.75，不由视图数推算速度。

seed0保留development配置选择身份，seed1/2是同一development的跨训练seed复验；三seed通过也不是独立数据验证。运行使用新release/合同/令牌/attempt，旧release/STOP/成本不变。PR12追加提交，不合并main，D盘仅精简材料并核SHA，无本机持续观察器。
