# H2完整模型接线与CPU验收

本次只完成PR21的模型、更新器和完整状态接线；没有连接实验室、读取真实图像/标注或历史checkpoint，没有GPU执行。GPU短验收和正式训练是两个后续授权步骤。配置见[固定范围](../../../configs/research/r18_h2_nud_v1.json)，未来范围见[GPU提案](GPU_PROPOSAL.md)。

## 实现及通路

[local_detail_model.py](../../../src/sparse_rtdetr/research/local_detail_model.py)的N直接使用共同协议原模型；U/D只替换第三层attention和中心残差组合。最细四点为c±(w/4,0)、c±(0,h/4)，按x+、x−、y+、y−排列；其余两层级八点沿用原生offset。删除fine offset的64个投影输出行，保留其余行的原始次序和初值。每head仍12点，所有head共用fine几何位置但保留各自投影value。

`shared_sample_core`每个尺度只调用一次既有gather双线性插值。实际fine样本同时用于原attention加权汇总与差分读出，后者没有再投影或重采样。D_x=(v_x+−v_x−)/2，D_y=(v_y+−v_y−)/2。无bias、零初始化的512→2读出W_D给出中心logit残差。U相加原中心头；D删除其末层中心两行，只保留wh行及原隐藏层。前两层、分类、宽高的context路径、全12点联合softmax、query、loss、匹配与ignore规则均不改。共同的分类/wh路径并不意味着训练后数值不受采样变化影响。

|项目|N|U|D|
|---|---:|---:|---:|
|实测可训练参数|20,094,584|20,079,160|20,078,646|
|第三层fine learned offset输出行|64|0|0|
|新增无bias差分读出参数|0|1,024|1,024|
|原第三层直接中心输出参数|514|514|0|

D/U共有tensor逐项一致；唯一形状差异是第三层box末线性层D只含U的wh两行，恰少514参数。相对N，两者同删16,448 offset参数。构造替换层在CPU RNG fork内完成，不消费外部Torch训练RNG；训练时仍须在各臂初始化后独立重置Python/NumPy/Torch/CUDA seed0。CPU没有真实ImageNet文件，其生产严格SHA加载入口沿用原实现，不能把随机backbone夹具说成真实初始化认证。

采样坐标只读原第三层detached reference；query位置编码仍使用该reference。训练返回的final/aux框继续使用原生上一层未detach的`inter_ref_bbox`作inverse_sigmoid基准。这两种reference角色不能混淆。普通与DN query走同一公式，保留原生DN padding与归一化。第三层bbox MLP的纯函数结果在U/D内复用，避免原实现训练时同一MLP的第二次求值；MLP无dropout或mutable buffer，未改变该函数或梯度。

## 验收入口和覆盖

```bash
python -B tools/local_detail_cpu.py --check-only
python -B tools/local_detail_model_cpu.py --check-only
```

后一个入口只支持构造CPU，隐藏CUDA，既没有GPU命令也没有生产输入参数。完整模型更新使用[local_detail_engine.py](../../../src/sparse_rtdetr/research/local_detail_engine.py)：原共同协议`train_window`的AST逐项相等，损失、累积、LR、clip及EMA更新算术不变；仅显式模型身份和健康验证改为H2范围。旧validator仍拒绝U/D，没有放宽历史模型身份。

[local_detail_checkpoint.py](../../../src/sparse_rtdetr/research/local_detail_checkpoint.py)使用独立H2格式/arm/scope，沿用原原子发布、optimizer布局/逐参数step账、完整RNG、BN/EMA、模式和cursor校验。先流式核文件SHA，再安全加载；拒绝覆盖、错误arm/SHA/scope/cache、半更新及失配状态。只允许`decoder.anchors`在原valid_mask=false的位置为正无穷，所有其他浮点state必须有限，BN running_var非负。EMA整数BN buffer沿原规则不浮点平均，anchor/点数缓存不可变。保存后在独立模型恢复，核验完整state和下一窗口的输入、输出、DN噪声、损失、梯度、optimizer、EMA、RNG精确一致。

完整检查使用128×160、physical1×accum2，覆盖N/U/D×FP32/CPU BF16六种组合，每种更新1→保存→更新2→恢复→重放2，合计18个CPU实际更新、36次该矩阵训练forward；其他构造前向另在回执逐次计数。另构建768×1344生产画布的缓存和E000空optimizer/初始EMA，但不进行该画布图像forward。构造旧格式fixture只测N终点EMA导入接口，绝不是读取历史真实B0。

具体检查包括：N与原共同模型完整训练输出/DN/梯度精确一致；U/D输入、配置、监督、共有初值和零初始化输出一致；采样复用和原core输出精确一致；非方形/不对称reference、无floor/无clip、边界及mask；采样detach与原aux梯度独立核验；差分读出初始梯度可学；DN/空GT与21/12项loss；六组全状态重放；生产缓存非有限例外；历史N只读EMA导入和错误身份拒绝。最终回执及实际受验字节见[full_cpu目录](full_cpu/CPU_REPORT.json)与[源码清单](full_cpu/CPU_SOURCE.json)。

这些是CPU工程证据，不能替代真实reader、ImageNet实物、CUDA/BF16、物理GPU/资源/存储准入，也不能证明速度、定位机制或AP改善。先前一次新增反例断言因对中心二列单独计算与完整四列向量计算的末位舍入差异失败，改为原四列布局核对；原失败回执留在delivery/full-model-cpu-002，生产公式未因此改动。

## 贡献边界与反例

待检验贡献只限于**第三层差分中心残差约束**，不是新差分算子或更强函数类。成对差分是已有PDC特例，可折叠为普通线性权重；U可令原中心项为零表达D。D仍通过前两层框、投影value及特征接收context，不能称完全去上下文。D/U的参数差异也必须公开，不能称严格等参数实验。

原16项叶层检查和[全部反例](DESIGN.md#4-cpu原型及反例)保留：平坦特征和错粗框时D不能靠最后残差纠正；背景坡度可产生假位移；边界零填充破坏共模抵消；小/模糊reference不保证新插值支持。完整模型补充零value时U能由直接中心bias移动而D不能的反例。这些都不能因CPU接线通过而删除。原创性和真实收益保持UNKNOWN。
