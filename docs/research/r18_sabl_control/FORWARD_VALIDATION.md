# S/A专用两前向工程验证

基于992b6b13f0b866700a9bd631ec1a557da96e025e，仅新增入口、显式raw-model桥接及CPU验证。数值公式、科学配置、原轴和误差判据保持。原release、STOP、N/S e24和独立u938状态只读。

## 权重与输入

A来自独立checkpoint-u938.pt（b9e7a005…），S来自原e24 checkpoint（deb67cf4…）。入口核完整旧schema、旧版本/原合同、旧manifest/父N、各自938/7296更新、运行时、全部model layout、参数/缓冲区/不可变cache、dtype、shape、训练mode和源RNG。严格load_state_dict后逐tensor字节核对。只桥接raw model并单独恢复本臂RNG；不调用fullstate.restore，不装载optimizer/EMA/engine，不将v1伪装为v2全状态恢复。

共同固定输入是原e4/window26 logical16，原loader从精确cursor生成一次。先核input_record完整identity、GT/ignore、16ID及像素SHA，再仅用前8张。只访问train_core，不读取548/1602图像。固定A flat-RoI352（b1/q52）在roi-before、A-mean后、pre.3、att_x、att_y逐块与保存FP32原字节一致；首次不一致立即停止。

## 前向与观测

保持train/enable_grad、原BN-layout context、原DN capture/guards、detector BF16和boundary FP32。不计算loss，不创建optimizer，禁止backward/autograd.grad和EMA.update。A/S各一次batch8 detector forward，无暖机。两轴新计算调用原修复函数一次；局部旧计算用相同x/logits，在no_grad下记录零分母、非有限数量、有限池化配对差异。局部比较计入同一计时/显存。新权重/池化及完整后续训练输出必须有限，原轴sum最大绝对误差<=1e-6。A指定旧dx须仍为0。

前后参数hash保持；BN内存变化仅允许running_mean/running_var/num_batches_tracked，逐项记录dtype、前后SHA、maxabs及改变元素数；不写回checkpoint。输出仅工程证据，不给S/A质量排序。

## 生命周期与资源

原准备STOP保留，保护清单读取显式提取protected_files，拒绝平铺、空、错格式或SHA冲突。prepare命令CUDA隐藏，真实两臂模型仅CPU构建和权重桥接，无完整detector forward。fixture实际CLI只使用标记的构造输入、真实BoundaryHead，不能代替真实GPU路径。

冻结Git归档逐文件SHA与受验字节一致；新manifest、definition和一次性authorization相互绑定。正式run仅允许独立attempt-forward-001，FORWARD_ONCE独占创建，拒绝重复。原科学/旧成本不覆盖。异常STOP无修复/重发，PASS写CLOSED立即退出。

硬上限2 forward，0 backward/optimizer/EMA；GPU300s（从首个CUDA资源操作之前计入）、wall600s；allocated18GiB，allocator同额保护；新增128MiB（实际task加32MiB开发/Git预留），全部SSH stdout/stderr/附件合计1MiB。无训练、评分或测速。若通过只证明两个固定模型/批次的前向数值路径，长期训练稳定性与检测收益均UNKNOWN。

## CPU检查范围

实际解释器、release cwd与脚本CLI下检查两臂构造forward/桥接/输出/CLOSED、参数不变及BN变化，拒绝错version、缺tensor、错dtype、错mode、非有限权重、重复输出、预算耗尽；验证原保护清单层级及损坏拒绝、首RoI精确不一致停止、禁止optimizer与backward。原109项/288比较另行复用回归。真实A/S558tensor CPU桥接单列，不冒充GPU前向通过。旧含optimizer的CLI job仍显式跳过，不记PASS。
