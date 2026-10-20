# 合成CPU原型验收

状态CPU_PROTOTYPE_PASS；13项执行、0失败、0错误、0跳过。Python3.12.3、torch2.8.0+cpu/torchvision0.23.0+cpu；本机WSL既有CPU环境，只读复用，未安装或改环境，未连接实验室。CUDA前后均未初始化。不是实际GPU2.4.1、BF16或生产数据验收。

唯一命令（从任意工作目录均由工具自身定位源码；输出必须新目录）：

```bash
CUDA_VISIBLE_DEVICES='' python -B tools/check_sampling_span_cpu.py --output <new-output-directory>
```

|实际执行检查|结果|
|---|---|
|关闭候选时原生输出逐元素一致/原reference不变|PASS|
|reference跨度≥一单元时输出逐元素一致|PASS|
|按12采样点顺序展开/非方形W与H尺度|PASS|
|宽高分别处理/空query参考变换|PASS|
|越界中心不clip/原零填充路径|PASS|
|观察真实native core输入：仅最细4点位置变化|PASS|
|合成叶层CPU autograd有限、offset/输入梯度连通|PASS|
|原维度256/8heads，普通及额外DN query数量形状|PASS|
|相同state keys和参数量/叶层state保存读回输出一致|PASS|
|8/2/2原生control的bias/scale差异、RNG不变|PASS|
|解析小span双线性cell示例（只是构造例）|PASS|
|负/零extent、NaN/Inf、错shape/层级和arm拒绝|PASS|
|完整共同R18构建，三臂20,094,584参数与精确state差异；旧validator拒绝候选|PASS|

执行记录保存在D盘delivery/research/r18-sampling-span-20261020-001/cpu-004/的CPU_REPORT.json及unittest.txt。JSON绑定本进程实际导入的项目/vendor Python SHA；依赖版本和工具字节一并记录。源码入Git，输出不重复拷入仓库。

未运行完整模型图像forward、训练更新、科学AP、模型权重加载、任何GPU或生产闭环。叶层state roundtrip不等于完整optimizer/EMA/RNG恢复验收。生产类型保护保留；这里只证明候选在真实计算叶层上的实现可行性。

CI初次失败保留：d38ea81的工具遍历所有外部模块时，把PyTorch虚拟文件名_ops.py按仓库cwd解析成真实源文件，记录SHA时FileNotFoundError。只修清单收集的模块归属，不改计算/测试/门槛；最终从仓库cwd实际走同一CLI，13项再次通过。旧CI失败不改写，最终head CI按GitHub实际状态另记；不能用本地PASS替代。

修正失败日志后，d367b65的CI明确暴露原生registry导入SummaryWriter所需tensorboard未列入CI依赖，实际仅执行1个导入失败测试而非13项，不能通过。补入本地既有CPU环境同版本tensorboard2.20.0；计算及最终CPU004证据字节未变、可复用。初次SHA记录异常掩盖了测试导入错误，因此不将_ops.py问题称为原失败的唯一原因。两个旧失败均保留；最终CI须实际运行13项且零skip。
