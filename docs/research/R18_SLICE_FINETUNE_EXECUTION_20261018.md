# 双臂固定微调：执行接线与启动记录

科学方案/门槛保持原配置字节，SHA
`6e695d96682aabbe08d3787b8b9e75f778e2501c60ab71fde1c1a0b730b10acb`。
当前记录为执行中观测，不是终点质量结论。

## 冻结版本与工程验收

生产release的Git commit为`57ae4d1990ef059818825e6d8813435b04046505`，
361文件source manifest SHA
`b989e9f45e50b26a30c36f34132d960b60932c93da8cad7fccaea961d81d4090`。
最终入口`slice_ft_run.py`：launch→真实supervisor→worker→原生训练/推理→
原双评价器→RESULT/CLOSED→精简导出→进程退出。
Writer/Reader/归档统一安装独立FT协议，保留1MiB上限、精确身份、顺序及终态保护。
完整科学审计原子存独立文件，进度只写绑定引用，不搬入日志流。
checkpoint额外绑定run/source/contract/parent/arm/scope/runtime，验收与formal隔离。

最终CPU验收142.863022370秒PASS；合成128px/8图不是生产训练或AP证据。
两臂accept、两epoch formal、两臂A/C、8组预算双评价器及导出都实际完成。
真实worker异常/进程清理、重复启动无副作用、真实事件流缺终态/重复终态/
epoch乱序/评价缺窗拒绝PASS，原15项规则和真实数据/EMA按字节/AST差异绑定复用。
门禁SHA `4c25874ac8de2d79931ac69ff531efc3f2bb167a226587b8e7838e370aa41ed1`。

准备阶段保留两个失败现场：事件字段kind与函数参数重名（91.332559911秒），
CPU临时runtime路径与前次碰撞被拒绝（2.845606737秒）。只修事件命名/检查目录身份，
科学函数AST、原增强、模型/criterion/optimizer math、A/C及评价文件字节不变。
新CPU阶段合计237.041189018秒，旧14457.736205673秒另保留；不重复计内部时间。
GPU资源或正式新异常仍STOP不重试。

## 生产链观测

唯一`production-001`，tmux `r18-ftmix-s0-20261018-001`。
生产契约文件SHA
`0caf2dbabecc4074f0c1d37e99ece7de67a7152578c2dbf537bf506837907f1f`。
启动前资源：4090D，无其他GPU计算进程，空闲24067MiB、RAM52036390912B，
目标盘1421275312128B，CPU125/150W、GPU1500MHz、CPU52C/GPU51C，原保护不变。
隐藏真实入口及独立可见CUDA环境前检均PASS，前检无模型/forward。

两臂GPU各[1保存,2,恢复1重放2]全部PASS并闭合：完整输入/GT/ignore/视图、
DN、输出、loss、裁剪前梯度及model/optimizer/EMA/RNG精确。
各阶段内部耗时24.287533729/23.307024906秒；这不是整个进程的GPU账单。
两次初始模型均为
`a322373726cc3c666c8ba0ea79a5dadcc80c149d87f8c4094e9c9c807652d676`。
正式状态重新载父EMA，optimizer/EMA/游标从局部0新建，不接验收状态。

epoch时间1792297806.5983908的只读观测：T-control完成239/3648更新，
T-slice正式及终点评价未开始，根无STOP。服务器顺序完成各12轮/3648更新，
FT012 EMA后运行548图A/C两预算双评价器、28条件和成本判断。
科学FAIL仍完成剩余固定比较，不调参/追加seed；无holdout/main合并。

## 不轮询的精简收尾

独立return helper commit `df3b3704f395555336fb56f2a815e16f3d3800ab`；
不修改生产release。Windows隐藏进程只通过SSH连接一次，以Linux pidfd内核退出
事件等待chain leader，不定期读取进度、不重连/重试。完成或STOP后读取固定导出
白名单，拒绝大文件/错SHA，复用已有同SHA副本，回传D并生成报告、更新共享交接。
网络中断只登记TRANSFER_STOP，不影响服务器执行或触发重训。

实际CPU结束链回传81小文件602247B SHA通过；错误SHA/重复输出拒绝，
独立真实进程的pidfd退出事件也已检查。
生产返回器PID35704已附着chain PID1318396/start51574330，处于单次事件等待；
这只是带时间的附着事实，最终传输状态以SHA_RECEIPT/TRANSFER_STOP为准。
其运行源码由独立RETURN_HELPER_BINDING登记，不冒充生产模型release的文件。

D报告入口：`workspace_management/delivery/experiments/localization-distribution/`
`r18-slice-finetune-20261018-001/execution-001/`；最终终态目标`terminal/REPORT.md`。
完整checkpoint、raw预测、逐窗审计和控制台只服务器；索引不是备份。
