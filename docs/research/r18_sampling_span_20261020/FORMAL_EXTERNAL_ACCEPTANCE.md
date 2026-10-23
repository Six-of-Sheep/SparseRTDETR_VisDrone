# 正式三臂链：显式复用外部GPU验收

原attempt-gpu-acceptance-002的04fe90d6f41c release、合同、source、ROOT/CLOSED、每臂结果/事件/退出、C1及成本账本保持原字节。正式协议`r18-sampling-span-formal-external-acceptance-v1`仅有11个阶段：admission、train-B0/B1/P、infer-B0/B1/P、score、cost、compare、export。旧全链/验收入口不接受新合同，正式入口不接受accept阶段。

`external_acceptance`引用一份描述符，描述符仅引用原contract/source/closed/execution_exit/cost_ledger。生产身份固定原commit、manifest和合同文件SHA；逐文件检查旧release，再逐文件比较新source。旧文件只允许合同解析/路径解析/调度与CI/导航变更；worker、模型、criterion、训练、增强/采样、观察器、保存恢复、预算和评价计算代码必须同字节。原验收闭合、9/18、恢复、有限梯度、C1大小/SHA/sidecar、退出及成本全部通过才可引用。不得重写旧source校验、复制C1或建立本地假accept目录。

训练函数保持原字节，通过只读解析器读取原验收的initial_model_sha256。正式模型仍从共同ImageNet初态独立构建、optimizer为空、EMA/update0、RNG新建；`used_acceptance_state=false`。没有把C1路径送入训练恢复。原export读取外部accept闭合文件的原字节；根结果/闭合另记录描述符及逐文件差异证明，清楚区分外部证据与本次本地阶段。

累计GPU账本以原验收实际79.265548953088秒/18forward起算（用户展示的六位小数为79.265549）。正式新增预算219675=218880训练+411终点评价推理+384同轮成本；比较中总forward219693包含外部18。原60GPU小时/180秒保存预留、100GiB状态等资源保护不变。新异常STOP、不退款、不自动重试；单纯质量未过仍完成固定三臂/后链。

实际命令统一使用Git内`r18_sampling_formal.py`：lab从已认证原始spec和原验收contract构造新描述符/累计成本、真实prepare/资源/CUDA预检、授权/bind；launch仅一个新tmux；run/supervise/worker执行原计算；根闭合及轻量export均自动完成。每个子进程记录argv/cwd/stdout/stderr/退出码。生产profile无fixture override，不访问1602。

CPU工具`tools/check_sampling_span_formal_cpu.py`使用最终CLI在微型固定合成材料上真实完成一次CPU验收，然后通过外部引用执行11阶段正式CPU链，不靠单元stub。标明合成parent与模拟GPU成本，不宣称生产GPU就绪。覆盖拒绝错误SHA/身份/未闭合/成本重置、正式初态optimizer/EMA/update0、epoch切换、原双评价/末批2、预算累计及最终导出；原accept目录/回执不变，新job无accept目录。成功后只清理该工具自身新建可重建tensor，保留小型SHA证据；D盘28份旧失败原件不动。

未来启动采用新commit/release、精确head四成功CI、新合同/令牌和attempt-formal-001；源码/材料/身份/资源/成本均重新准入。不得在旧release pull或修改，不合并main，不恢复本机持续轮询。正式三臂各120轮/36480更新与冻结科学门槛不变。
# Resource-age observability repair (CPU only)

The first formal CPU attempt stopped in the score supervisor at the unchanged
`0 <= time.time() - sampled_epoch < 25` gate. Its rejected sample was not saved;
the last successful sample cannot establish the cause. A bounded no-model
reproduction (20001 real calls) did not reproduce that failure. Controlled
expired, negative-age, delayed and cross-stage-old-sample cases were rejected.

The confirmed defect addressed here is missing rejection evidence. The screen
now persists the rejected sample, sampling start/completion, actual check time,
wall and monotonic ages, collection duration, unchanged threshold, clock sources
and phase identity before raising. Collector timestamps remain untouched.
Monotonic timing is diagnostic only: a negative wall age is still refused.
No resource threshold, model, training, checkpoint, or scientific computation
changes. This does not claim that the old anomaly's root cause was identified.

