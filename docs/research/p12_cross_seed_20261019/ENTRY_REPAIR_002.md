# 授权入口修复与真实prepare验收

旧release-001、preparation-001/STOP、源码和成本原样保留。沿PR12追加修复，只改变入口初始化、来源校验及相应CPU检查，原P12算法、8项条件、原计划配置保持不变。

所有分支在文件入口统一建立this-release的tools、根目录及src搜索顺序；不再由fixture预设。prepare/run均先检查manifest实际文件字节，再导入同release模块、核对实际`__file__`和SHA，再校验计划。相同名称的外部模块不能作为替代。产出ENTRY_IDENTITY.json记录实际cwd、解释器、argv、环境和模块位置。

真实部署使用`/home/lyy/.cache/r0-seed0-e120-20261010-001/environment/bin/python -B <release>/tools/p12_job.py prepare --parent <冻结父合同> --output <新preparation目录>`，cwd为该release，CUDA_VISIBLE_DEVICES为空，PYTHONPATH删除，OMP/MKL/OPENBLAS各1线程、nice15/ionice idle。正式run使用同解释器、cwd及初始化顺序。

新增独立子进程测试直接执行prepare，不通过fixture，不由测试注入sys.path/PYTHONPATH：验证真实模块加载后严格拒绝未认证父合同；缺slice_core文件、源码字节身份冲突均拒绝。CI小夹具不能伪造固定父身份，故不将这些拒绝检查写成真实材料prepare成功。另保留真实fixture双评价器闭合/异常/重复拒绝等5项，共8项0跳过。定向检查曾发现校验计划早于源码身份检查，测试现场entry-cpu-002保留；修正后的entry-cpu-003通过，尚未执行科学评分。

固定提交导出新release后，必须以部署一致的独立子进程实跑真实prepare，认证seed1/2全部274批缓存、原A/B/C精确重建与canonical输入证明。成功应证明CUDA未初始化、科学评分和forward为0；其结果单独记录，才允许首次生成生产合同/令牌。prepare不成功则STOP，不自动修复重试。后续沿原固定链，未更改科学或成本目标。
