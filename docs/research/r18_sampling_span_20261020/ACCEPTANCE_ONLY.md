# 独立三臂GPU工程验收入口（本修订仅CPU准备）

旧9d6454fd/release和gpu-acceptance-preflight-001的BLOCKED保留。该版只支持完整14阶段，本修订新增显式`r18_sampling_acceptance.py`与`r18-sampling-span-acceptance-v1`合同，不改原科学配置、初始化、公式、精度、门槛或219693总forward/累计60GPU小时上限。

## 权限和计数

`GPU_ACCEPTANCE_ONLY`仅允许admission、accept-B0、accept-B1、accept-P。正式训练、科学评分、完整测速及其他阶段既不在解析器中，也被运行时stage_path/identity/predecessor拒绝。旧全链CLI拒绝验收合同及spec；新入口拒绝全链合同。新协议不把合同转换成旧全训练身份，也没有隐藏的截断参数。

每臂复用原acceptance计算：epoch1首两个logical16窗口，batch8×accum2，`update1 → C1 → update2 → 独立重建/restore1 → replay2`。实际更新3、唯一输入窗口2、实际batch forward6；三臂9/18。正式训练未来仍从共同ImageNet初态独立开始，不能接C1或验收末态。

ACCEPTANCE_COUNTS同时保存每臂的updates_reserved/updates_completed和forward_calls_reserved/forward_calls_completed。更新前检查并预留1次更新；模型forward的pre-hook逐microbatch检查并预留1次forward，post-hook记录真实完成及logit dtype；完整更新成功后才记completed。每臂3更新/6forward、全局9/18均不可修改；失败不退款，未完成操作不能重试。数据/模型数值和恢复一致性检查维持原实现；计数hook不改tensor、RNG、optimizer或EMA。实际GPU必须观察到BF16 logits，CPU合成明确只检查FP32。原累计GPU时间/forward预算账本与新工程计数同时执行；不是以视图数量推算forward。

## 闭合和精简导出

四阶段均须监督器/worker/事件链/结果闭合。RESULT记录`result_kind=GPU_ENGINEERING_ACCEPTANCE`和`scientific_quality_evaluated=false`；CPU夹具另标`CPU_SYNTHETIC_ONLY_NOT_GPU_VALIDATED`，不记录质量PASS。先生成工程阶段闭合，再成功白名单导出，最后生成根CLOSED；导出失败即STOP，不能把部分导出写成全链PASS。

ENGINEERING_COST包含各臂阶段/更新耗时、allocated/reserved峰值、开始到三臂闭合的存储增长及原累计账本引用。该存储边界不含之后的小型根结果/导出；非GPU夹具不称实测GPU成本。导出只含配置/合同/source、根与三臂结果/闭合、成本/计数、普通query和DN机制小汇总、服务器C1大小/SHA索引，不含tensor、数据、全量events或任何正式训练产物。成功根闭合后可通过独立export命令导出到新目录；重复目录拒绝。

## 后续GPU执行命令（当前未授权，不运行）

必须使用交付VERSION指向的新commit/release和actual-byte manifest，不能重放旧BLOCKED版本。未来在全新`acceptance-execution-001`元数据目录和`attempt-gpu-acceptance-001`输出目录，重新核验source、真实材料、资源/存储、UUID及逻辑物理映射；任何异常STOP，不修复重试。原两类清单purpose/role/SHA适配保持；prepared spec增加`execution_scope=GPU_ACCEPTANCE_ONLY`，明确新attempt/job_root/nonce，原科学及执行配置hash不变。

以下路径变量须由交付VERSION/新鲜准入产生，不能用默认路径猜测；本轮只冻结release，不创建GPU执行合同或令牌。

```bash
cd "$SPAN_RELEASE"
CUDA_VISIBLE_DEVICES='' "$SPAN_PYTHON" -B r18_sampling_acceptance.py prepare --spec "$SPAN_META/spec.json" --source "$SPAN_SOURCE" --output "$SPAN_META/prepare"
# GPU身份预检无模型forward，未来授权执行时用冻结release中原审计接口。
CUDA_VISIBLE_DEVICES="$SPAN_GPU_UUID" "$SPAN_PYTHON" -B r18_sampling_span.py cuda-preflight --gpu-uuid "$SPAN_GPU_UUID" --output "$SPAN_META/cuda-preflight"
CUDA_VISIBLE_DEVICES='' "$SPAN_PYTHON" -B r18_sampling_acceptance.py bind --spec "$SPAN_META/spec.json" --source "$SPAN_SOURCE" --cpu-gate "$SPAN_CPU_GATE" --authorization "$SPAN_META/authorization.json" --output "$SPAN_META/contract.json"
CUDA_VISIBLE_DEVICES='' "$SPAN_PYTHON" -B r18_sampling_acceptance.py run --contract "$SPAN_META/contract.json"
CUDA_VISIBLE_DEVICES='' "$SPAN_PYTHON" -B r18_sampling_acceptance.py verify --contract "$SPAN_META/contract.json"
CUDA_VISIBLE_DEVICES='' "$SPAN_PYTHON" -B r18_sampling_acceptance.py export --contract "$SPAN_META/contract.json" --output "$SPAN_META/independent-slim-export"
```

新prepare_gate必须为同一新source、同data、LAB_GPU_ONCE及验收协议的PREPARE_PASS。CPU_gate必须为同source的ACCEPTANCE_CPU_ENTRY_PASS，不能拿旧全链CPU标签替代。prior_cost需显式task_id、profile=LAB_GPU_ONCE、非负GPU_seconds/forward_calls及历史attempt证据；未知不能置0，未发生GPU阶段才可附证0。未来单次授权JSON严格包含scope/task/attempt/commit/source/science/execution/acceptance hash/limits/nonce；GPU_authorized=true、science_authorized=false。所有子进程命令、退出码、stdout/stderr保留。

本轮实际CLI CPU工具：`python -B tools/check_sampling_span_acceptance_cpu.py --output <全新目录>`，CPU-only PyTorch、cwd=release、子进程无PYTHONPATH。覆盖三臂3更新保存恢复与9/18实际计数、仅四阶段闭合退出/自动与独立导出、越界/修改限额/混用合同/科学授权/正式训练拒绝、重复启动、worker异常及不重试。CI在Python3.10/3.12执行；合成CPU通过不能替代真实GPU/BF16/deterministic backward/768画布显存或实测成本。

归档迁移仍STOP，本修订不触碰D盘28份旧合成原件、旧partial或其他历史材料。CPU工具仅在自身成功后移除本次新建的可重建合成tensor，小型SHA/闭合证据保留；失败现场不删。
