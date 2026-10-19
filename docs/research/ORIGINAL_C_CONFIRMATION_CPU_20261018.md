# 原C独立确认：CPU准备验收

最终运行commit `0382a62593dc8ebc33336703bdf04d14ba3ad4de`，release-002，354文件manifest SHA `1bfe5df04e3c25063efb177f6a5f340a4d13065dbb99a4f8342b526a7b8622e4`。CPU-002门禁PASS，耗时261.340554303秒；含单元与启动外层的总调用263.727400434秒，二者不能相加。

真实最终入口start→run→原guard.supervise→三个独立真实seed worker→认证数据配置/parsed_gt→primary_inputs及bridge_inputs→RESULT/CLOSED→归档/精简导出→退出通过。18组原A/B/C×600/300的canonical输入SHA与原before/after raw-ignore计数精确一致，没有调用双评价器计算分数。三seed各检查development有序前缀1/3/4/5/547/548的实际标注、两个评价输入及末批；1602只检查整数调度401批/末批2，不读取封存成员。零预测图保留覆盖。

worker异常产生STOP、归档、导出和返回2；正常返回0。重复start/run拒绝且原输出树逐文件引用不变。8项定向单元检查PASS，含三seed身份/跨seed冲突、缺失/重复/乱序成员、错误GT身份、配置非parsed_gt、空预测/ignore、缺窗/乱序/重复终态、绑定及损坏记录拒绝。21原slice检查、11管理及317旧source导入证据复用。CPU隐藏CUDA并禁止Module构造、CUDA初始化、backward及评分函数。

CPU-001失败保留：Development548构造不必要的torchvision Module被准备边界哨兵拒绝；明确为工具路径问题，未出现GPU或材料完整性异常。独立release-002改走原认证parsed_gt(rows,data_spec)，新attempt完成受影响验收；旧STOP不改。

此次只认证准备入口和真实评价输入，不认证未来解封reader、checkpoint加载、GPU末批forward或确认指标。数据资格仍BLOCKED_PENDING_ACCESS_HISTORY，工程PASS不代替独立确认资格。
