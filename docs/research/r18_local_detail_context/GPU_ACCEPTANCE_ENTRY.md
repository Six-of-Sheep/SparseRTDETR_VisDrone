# H2独立短GPU验收执行入口

`h2_acceptance.py`只支持prepare、bind、contract-check、cuda-check、launch、run、supervise、worker、verify。不存在formal、评分、推理或测速分发；验收state的scope为`gpu_H2_acceptance`，不能作为正式训练初态。旧ab951f2受验模型、差分公式、初始化、数据、门槛和146,596次总forward提案保持。

## 合同及准入

原始spec必须提供profile、唯一attempt/task_root/job_root、解释器、nonce、boot、完整GPU UUID、存储身份、原train/development成员manifest及根、ImageNet引用、环境和H2成本起点。`base`先构造固定resources/policy/science/source字段，显式冲突拒绝，再调用实际资源接口。bind引用同源码CPU CLI gate、真实CPU prepare gate和本次用户授权；生产拒绝CPU fixture参数和bootstrap gate。绑定合同、发布source manifest、每个依赖源码和原H2科学源码分别核大小/SHA；数据与真实ImageNet不从外部临时补字段绕过。

隐藏CUDA的prepare逐一核train_core4869原图/标注字节，核development548 manifest但不读其图像；实际reader固定epoch1窗口0、1并绑定像素/GT/增强摘要，无模型forward。三臂严格ImageNet独立初始化，逐tensor核D/U共享和允许的行差异，N初态须与已认证共同初态摘要一致。contract-check必须保持CUDA不可见，不能创建job或消费启动令牌；cuda-check只核逻辑CUDA→PCI→物理UUID及BF16能力，零模型forward。

沿用实验室资源约束：≥16 CPU、16GiB可用RAM、80GiB磁盘余量、entry/live shm≥8/4GiB，原25秒wall年龄保护、温度/125W-150W CPU及1500MHz GPU上限，旧功率/时钟不写入。拒绝样本保留完整原值及采样/检查时间，不重写时间后重试。

## 独立执行和计账

启动必须先保证job、H2累计账本、唯一LAUNCH_CONSUMED和tmux名称不存在。只有production的launch能创建一个tmux；CPU只能显式foreground。每臂由独立监督进程启动实际worker，绑定PID/starttime，监测真实资源和worker树；仅清理本次拥有的进程。worker不接受单独手工启动。

固定N/U/D每臂更新1、2、恢复C1后重放2：合计**9次实际更新/18次batch forward**。每个实际forward前原子预记账，逐臂上限6、全局18；每次更新前预记账，逐臂3、全局9。失败/不确定操作不退款，重复启动不创建新账本。GPU累计阶段时间包含进程初始化、CUDA身份、保存恢复和I/O，限制900秒；总wall限制1800秒，留清理余量。allocated限制18GiB，CUDA allocator同时设进程内上限。失败现场保留，STOP后不重试、不接下一臂。

真实BF16从六次实际模型输出logits dtype逐次核验；共同训练、普通/DN、原aux/detach计算仍使用ab951f2完整模型/更新器。C1保存后独立重建模型/optimizer/EMA，从同原始窗口reader重放，核像素/GT/增强、输出、DN噪声、loss、梯度及完整模型/optimizer/EMA/BN/RNG/cursor精确一致。短loss及耗时不是科学评价或完整测速。

## 闭合和精简回传

各臂RESULT、worker退出、supervisor退出和CLOSED均须一致，无残留进程。根RESULT、累计账本、精简导出索引、CLOSED和EXECUTION_EXIT闭合后退出，不能自动转U/D正式训练。精简白名单最多20MiB：合同/source索引、各臂结果/闭合/退出/恢复/C1侧车、根结果或STOP/退出/累计成本及SHA_INDEX。C1状态、原始逐micro日志和资源日志留服务器，不回传tensor。根CLOSED在导出索引之后发布，终态collector单独取该小回执并核SHA，避免循环摘要。

本次实际消耗从H2零起点累计，未来146,596次总预算只扣一次；旧P研究及诊断成本分开保留，不冲抵、不归零。历史B0/N复用限制和原STOP/缺CLOSED保持，本次新N短验收也不是N的正式重训或原B0重新闭合。

## CPU实际CLI检查

```bash
python -B tools/check_h2_acceptance_cli.py --output /tmp/h2-cli-cpu-fresh
```

全部输入/授权依赖是明确标注的构造CPU夹具，模型、更新、序列化、独立子进程监督和CLI分发使用实际实现。检查正常三臂闭合、输入篡改、重复launch/run、资源合同冲突、扩大scope/forward、无权限parser入口、worker异常、forward/update/time/wall/allocated硬界及异常精简导出。不能把该CPU profile说成真实资源或CUDA认证，真实准入/单次GPU执行另有回执。
