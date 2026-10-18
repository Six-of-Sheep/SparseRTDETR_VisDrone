# 原C增加一个固定视图：唯一准备方案

本轮仅代码/配置/CPU验收，没有GPU、模型恢复、训练、detector forward或科学评分授权。后续完整执行须另行授权。原共同协议适配R18为开发基线，原C三seed保留，Soft-NMS、固定微调和投票保持关闭。不是未经改动的官方复现；development反复使用及历史确认集限制保持。

## 两个且仅两个方案

C-center = 原C(whole/TL/TR/BL/BR) + 一次固定中心切片。C-flip = 同一原C + 一次原整图水平翻转。各seed使用各自原e120 EMA；不使用微调权重或跨seed权重。每个计分batch的每种新增视图只一次forward；固定暖机单列，不多尺度、不扫位置/比例、不迭代。

中心h=ceil(5H/9)=(5H+8)//9、w=(5W+8)//9；x0=(W-w)//2、y0=(H-h)//2；PIL半开[x0,x0+w)×[y0,y0+h)。尺寸与四角crop完全一致。中心局部FP32解码到crop像素后加整数偏移。翻转在原始RGB上PIL FLIP_LEFT_RIGHT，然后原认证Resize到768×1344；局部预测按连续坐标x1=W−x2'、x2=W−x1'，y不变，映射FP64。都不clip，非有限/非正面积STOP。新视图不读GT；它改变位置/上下文/观察方向，不能叫新增原生图像细节。

原view0..4不变，两方案互斥地使用view5（各自独立variant命名空间）。Q300×10原FP32 sigmoid，每视图跨类top600、零分保留、无cutoff。候选总数3600。原排序score降序，然后view/query/category升序；原NMS为同类连续FP64 IoU严格>.5才抑制。复用原slice_core.classwise_nms/merge_candidates字节，不复制改算术。扩展SixthCandidate只版本化允许view5，旧Candidate仍拒绝view5。无坐标融合/调分/二次NMS。

最终top600、top300同一有序NMS列表前缀；先预算截断，再原raw-ignore，不补位。GT只用于既定评价及其ignore规则。原C能力关闭时直接进入原merge_candidates，原四角几何/像素/候选代码不改。src原评价器原封保留；两档原双评价器，COCO bridge比较与官方风格单列，原AP直接复用。

## 后续唯一分阶段链（本轮不执行）

1. 每个将执行的seed先冻结source/contract、认证自有e120/fixed64/548/cache身份、预检原保护和资源；不写硬件限制。不重放旧launcher。先seed0；seed1/2在种子0全部准入后才启动。
2. 固定该seed原64图、原顺序、batch4/FP32、独立进程A、C、C-center、C-flip，各2暖机原batch。A18、C90、center108、flip108 batch forwards（合324）。整条计时包含原图加载/SHA、预处理、所有真实forward、H2D、坐标映射、排序和NMS；不得用缓存省略原5视图成本。记录峰值allocated/reserved、硬件/频率/温度/环境，不按64图AP挑配置（禁算64 AP）。启动/完整状态审计/写盘另列。原0..4像素/输出及A旧缓存精确一致；model/EMA/BN/optimizer/RNG不变、梯度全空。新数值/身份/资源异常STOP，不自动重试。
3. 候选各自端到端/A≤6，未舍入判断。若工程或成本失败，停止该固定方案，不降batch/精度/换尺寸。相同6视图不是实测等成本。center−flip在匹配64输入下报告实际端到端时间和显存。
4. 64工程/成本通过后seed0做固定548图两个新方案。原5视图使用认证full-C **抑制前raw**（不能用原NMS后C输出回推）。先原C重建/成员/输入绑定全过。每个548 batch仅获取center或flip的view5，分别2暖机+137 batch=139，两方案共278新batch forward；原5视图不再forward。新raw必须绑定seed、checkpoint、variant、view5、原图/批成员、几何、source/contract；不能把不同seed相同query ID当对应框。
5. 合并原五视图pool + 新600，冻结预测，再2variant×2budget=4组双评价器；原A/C点值不重算。原固定64与548分组/位置不同，不假定可逐图复用64新输出；本方案固定全548新增view5各跑一次，避免以未经认证的batch差异复用少计forward。工程正常时完成当前seed两方案全部评价后裁决，不中途选方案。
6. seed0满足下述14质量条件、center对flip的预注册规则及两方案成本≤6，才按相同方式独立执行seed1、seed2。seed0通过才固定启动seed1/2；seed1质量失败也仍完整完成seed2固定比较，不按中途科学结果省略seed；工程/完整性异常仍STOP。未执行项标未执行，不平均掩盖失败。最终每个实际执行seed逐项报告，再另列均值/样本SD/最差值，无bootstrap、无holdout。任何新科学执行异常STOP，禁止自动修复重试。

## 冻结质量及结论规则

每seed，主600 center−C：AP≥+.20pp、APS≥+.30pp、AP75-small≥0、APM/APL各≥−.20pp；次300 AP/APS各≥−.20pp。
同时center−A原七项：主AP≥+.30、APS≥+.50、AP75-small≥−.20、APM/APL≥−.50；次AP/APS≥−.20pp。使用0..1原分数差乘100一次，已有pp不再转换。14项均须通过。

center−flip主600 **AP与APS都严格>0pp**，才称同视图数下中心位置有额外价值（描述性、不是显著性）。相等/权衡/任一不正均写“中心位置额外价值未获支持”，不改阈值、不将flip升为主候选。此条件与14项及实测成本共同决定是否扩到seed1/2。flip−C、center−flip所有指标/两预算/实际成本无论正负都列出；flip其他尺寸差异作为竞争解释/风险保留，不强称唯一下游原因。主center−C/center−A质量通过但center−flip未过时，分开记为“基本质量通过，同视图控制未超过，不进入多seed”。只有每个完成seed全部条件及成本通过才能称预注册复验通过。

## 新增工作量与成本依据

每seed固定GPU验收324 + 全量新增278 = **602次batch4 forward**（2408图像视图，包括全部暖机）。只做seed0时602；全部三seed最多1806。每seed新增4组双评价器，共最多12；原A/B/C不重评分。数据准入可复用投票项目已认证411批原重建和18组输入证明，但必须核验引用/身份/SHA及原source差异；新view5只在新执行中产生，当前不存在合格缓存。检查不自动补推理。

原seed0匹配64实测A2.335102s、C10.910771s、C/A4.672504；六视图的线性工作量估算约C×6/5=13.09s、约5.61A，仅作规划，不当成本倍率证据；NMS池增长、额外预处理、内存、GPU状态可使实际超6。原三seedC/A约4.6725/4.7101/4.6656。按旧阶段完整审计/启动开销，初步每seed GPU阶段墙钟预留5–15分钟，CPU合并/4组评分5–12分钟；三seed全部执行约30–81分钟另加准备/网络。此为估计，不能作为墙钟终止条件，未实测新方案速度/显存。CPU旧12组投票753.68s为评分参考，候选数变化可改成本。64端到端时间与全阶段监督墙钟分列。

CPU验收覆盖奇偶/小尺寸、半开窗口、翻转连续往返及越界、FP32解码、原5视图像素精确、原C关闭精确、同分view5次序/预算前缀、原ignore后预算禁止补位、输入拒绝、实际准备CLI子进程退出与重复启动拒绝。本CLI不暴露GPU或科学run；后续仍需完整production supervisor/资源/恢复/评分/导出接线和受影响CPU验收，不能把准备PASS称整条GPU链就绪。

D只留主表、方案、配置、CPU证据、版本/SHA索引；图像、checkpoint、全raw、预测/审计只服务器。历史记录只引用不改写，main不合并。
