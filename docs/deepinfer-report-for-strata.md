# DeepInfer 设计提取报告 — 面向 Strata dsv41（C++/SYCL，单卡 B70 32 GB）

- 代码基线：`/home/dyy/deepseek-work/llm-for-glm-flash/DeepInfer`，git commit `774ce24`（origin = github.com/deepseek-ai/DeepInfer），444 个 .py + 317 个 C/C++ 文件，24 MB。
- 阅读范围：`deepinfer/`（443 文件）、`csrc/engine/`（调度器与 metadata 生成）、`vendor/DeepGEMM`（MoE 内核与启发式）、`vendor/DeepEP`（EP 通信）、`doc/inference_deployment_qa.md`（312 行官方 Q&A）。
- 结论先行：**这是一个为"64~128 卡 Wide-EP + 全权重驻留 HBM + 大 batch"设计的框架。它没有、也不需要 CPU 专家车道。我们最想要的"CPU/GPU 专家重叠"在框架里不存在。**真正可搬的是 4 个结构性套路：单步一深流水线、全 pinned/设备端决策、按边际收益裁剪验证窗口、按带宽预算给 copy 划 SM。下面逐条给证据。

---

## 1. 专家放置与分级

### 框架怎么做的

**只有"卡间水平复制"，没有"显存/内存/盘分级"。** 三级存储只存在于 KV cache（`doc/inference_deployment_qa.md:131-135` 的表：in-flight block KV → GPU HBM；SWA KV → 本地主机内存；paged block KV → 3FS 盘）。MoE 权重永远是"本 rank 的 N 个专家整块驻留 HBM"：

- `deepinfer/model_executor/models/v4_modules/moe.py:3-8` 定义三层计数：logical（gate 可选范围）/ extra（冗余副本）/ physical（实际分配的参数行数），`physical = logical + extra`，`physical % ep_size == 0`。
- `moe.py:94`：`self.n_local_routed_experts = self.n_routed_experts // self.ep_size` —— **放置粒度就是"每个 EP rank 均分"，没有任何按层配额、热度阈值、换入换出**。
- `deepinfer/model_executor/model_loader.py:83-95` `_configure_moe_redundancy()` 在建模前就用 `resolve_num_extra_experts()`（`eplb/__init__.py:23-31`）把冗余专家数对齐到 ep_size 的整数倍。布局在**构造期固定**，运行期只 shuffle 不换介质。

**EPLB 是"哪张卡放哪个专家"，不是"显卡还是内存"。** 机制：

1. 每步 MoE dispatch 直接把本 rank 各 expert 的 token 数累加进 `expert_count`（`eplb/__init__.py:49`，每步末尾 `zero_()`，见 `:131-138` 的 `step()`；写入点在 `v4_modules/mlp.py:474` 的 `cumulative_local_expert_recv_stats=self._eplb_layer().expert_count`）。
2. 环形窗口缓存最近 N 步（`eplb/__init__.py:114-119`，默认 `DEEPINFER_EXPERT_COUNT_WINDOW_SIZE=100`，`environ.py:115`）。
3. 到周期后 `_rearrange_experts()`（`:223`）：窗口内 physical 计数 scatter_add 回 logical → `dist.all_reduce`（`:254`）→ `rebalance_experts_global()`（`policy.py:73`）算新布局 → 权重就地 shuffle（`:262` 走 `shuffle.py:150` 的 `expert_model_parallel_shuffle_inplace_intranode`，经 `symm_mem` IPC 直接读对端镜像；跨机走 `shuffle.py:65`）。
4. 目标函数写得很清楚（`policy.py:40-43`）：逐层独立，最小化 `sum_step max_rank(该 rank 上所有专家的负载和)`。贪心：每次挑"放到哪个 rank 后峰值最小"，满载 rank 用 `inf` 屏蔽（`policy.py:62-66`）。
5. 生产参数（`doc/inference_deployment_qa.md:97-103`）：prefill 每卡 4 个冗余、1000 步重排；decode 每卡 1 个冗余（EP64 时因整除对齐变成 7 专家/卡）、1000 步统计窗、10000 步重排，实测 `moe_lb`（最慢 rank / 均值）= 1.090。

**"预取下一层专家"：框架里没有，而且我认为不该有。** 全仓检索结论：
- `deepinfer/ csrc/` 内 `prefetch` 命中全部是 kernel 内部的标量预取（如 `tilelang_ops/asc/compress.py:224 STORE_PREFETCH`、`store_kv_cache.py:101`"no benefit from prefetching"）。
- 唯一带 prefetch 名字的 API 是 `vendor/DeepEP/deep_ep/buffers/ep.py:854 lb_prefetch_weights()`，用途是 **EPLB 重排时把本 rank 权重推给对端的冗余槽位**，不是单步预取。
- `csrc/engine/input_metadata/block_load_plan.{hpp,cpp}` 是一个 **KV 实块的 gather 计划**（`block_load_plan.cpp:48 build()` 产出 `[N,4]` 的 `(src_block, begin, end, dst_off)` 到 pinned 内存），把 paged/ring 的 KV 拼成连续 dense K 再进 attention，也不是预取。

为什么没有：路由依赖链是 `h → attn(h) → h' → gate(h') → topk`，层 N 的专家 id 必须等层 N 的 attention 出来才知道，**层间预取在数据上不可行**。DeepInfer 的替代手段是三条"消灭等待"而不是"隐藏等待"：全驻留 HBM、EPLB 让热专家本地化、Mega-MoE 把 dispatch 融进 GEMM 内核（见 §4）。

### 和我们的差距

我们要解决的正是它绕开的那个问题：32 GB 只放得下 ~1500/15360 专家（10%）。它有 64 卡把每卡专家数压到 7 个（`doc:103`），我们是 1 卡 384 个/层、92% 在主机内存。

但它的 EPLB **形状**对我们有用：`窗口统计 → 周期性重排 → 就地搬权重`。我们的 `hotpin-design.md` 目前是一次性读 `expert-profile.bin`（40×384 uint32）静态选 top-K，没有窗口、没有周期刷新。DeepInfer 的窗口语义（100 步/1000 步/10000 步三档分离：**统计窗 vs 重排周期 vs 指标上报间隔**）说明：统计要快、重排要慢且可以很贵（`_rearrange_experts` 里直接 `torch.cuda.synchronize()`，`eplb/__init__.py:225-228`，因为它本来就很少发生）。这个"贵的操作允许同步"的取舍可以照抄。

### 能不能抄 / 抄的形状

**能抄：把静态 hot-pin 升级成窗口式热集刷新。** 改 `ds4_model.dp.cpp` 里 `Chunk` 的 hot arena 逻辑（`hotpin-design.md` §3 的 `hot` / `hot_of` 字段），不动 DMA 路径：

1. 每步把"本步实际从 RAM tier 服务的专家 id"累加到 `uint32 hot_cnt[40][384]`（现有 `Chunk::Item` 已经有命中标志，`ds4_model.dp.cpp:4362-4390` 的 `hit` 判定点顺手 +1）。
2. 每 N=256 步（一次窗口）重算每层 top-K，与当前 hot 集求差集。
3. 差集在**两次 decode step 之间**刷新（此时 GPU 空闲或正在跑下一步的 attention，见 §2 的一步深流水线），单层单专家 blob 7.74 MB，换 32 个/层 = 248 MB，按 36 GB/s 计 ~7 ms，摊到 256 步可忽略。
4. 自检逻辑保留 hotpin 设计里的 memcmp 校验，失败自动 `hot_k=0` 退化。

**预期收益量级**：hotpin 测算 top-10% 专家占 73.9% 访问、每层 32 个（10 GB）覆盖 64.8%。静态 profile 若已选到接近的最优集，动态刷新只赚长尾漂移，估计 **+2~5%**（未验证）。**优先级不高**，列在这里是因为它几乎是唯一"框架里有对应机制"的一条。

---

## 2. CPU/GPU 重叠（对我们价值最高的一条）

### 框架怎么做的

先说结论：**框架里没有"CPU 计算专家"这回事**。全仓 `offload|swap_out|cpu_weights|host_buffer` 检索，命中只有 KV/多模态的 pinned staging（`deepinfer/platform.py:196` 的 `aclrtHostRegisterV2`、`deepinfer/shared_memory.py:83-120`）。DeepInfer 的 CPU 只做**批调度与后处理**，GPU 只做计算，重叠发生在"CPU 调度 vs GPU 前向"之间，不在"CPU 算 vs GPU 算"之间。

它的重叠由 4 个机制组成，每个都可搬：

**(a) 一步深软件流水线：整个 step 只有一处同步，且同步的是"上一步"。**
`deepinfer/worker/worker.py:113-189` 的顺序是硬编码的：
1. `:118` 先用**上一步**的采样结果做 GPU 端 seqlens 修正（见 (c)）；
2. `:122` `execute_main_model(...)` —— **发射即返回，不等待**；
3. `:144` `_post_process_step(self.prev_step_state)` —— 这里才 `sample_end_evt.synchronize()`（`:263`），等的是**上一步**的采样结束事件；
4. `:154` 发射本步 sample；
5. `:183-187` 若开 ZOS（`DEEPINFER_ZERO_OVERHEAD_SCHEDULING`），本步 post-process 推迟到下一步，与下一步 forward 重叠。

所以 CPU 干"post-process(N-1) + schedule(N+1)"时，GPU 正在跑 forward(N)。官方 Q&A 把目标写得很直白（`doc:270`）："GPU 在 post-process(N) 和 schedule(N+1) 期间空转……异步调度的目标是消掉这段空转"，实现方式见 `doc:274`。

**(b) 强制"不许同步"的守卫。** `deepinfer/utils.py:119-132 forbid_cuda_sync()`：进入作用域后，任何 torch 可见的 CPU-GPU 同步（阻塞 copy、stream/device synchronize）直接抛异常；注释特别说明 `Event.synchronize` 不在覆盖范围，**故意保留的事件同步点不受影响**。它被套在 `model_runner.py:322-327`（`execute_main_model`）与 `:357-368`（`execute_sample`）上，docstring 写死"IMPORTANT: This function must NOT contain any cuda synchronize."（`model_runner.py:317`、`:346`）。这是把"不该同步"变成 CI 可检查的约束——**这是最便宜、最该抄的一条工程纪律**。

**(c) 跨步依赖一律在 GPU 上算，绝不回 CPU。**
ZOS 下 CPU 侧的 `seqlens_q/k` 天生滞后一步，DeepInfer 不在 CPU 重算，而是把上一步采样结果留在显存里，用两个 kernel 修正：`worker.py:218-227` 的 `compute_seqlens_q_by_confidence_rates`（置信度裁剪，§5）与 `compute_seqlens_k_zos`（按上一步接受数回退 K 长度）。元数据全部落在 pinned 缓冲（`csrc/engine/zos.cpp:178-183`），H2D 非阻塞，D2H 用一个 event，且**只在开 varlen 时才需要**（`worker.py:229-234`），`d2h_evt.synchronize()` 在 `:149` —— 注意这个 d2h 排在修正 kernel 之后、主前向之前，所以 CPU 只等"小的那一段"，不等 285 ms 的大 kernel。
更进一步：`csrc/engine/zos.cpp:84-98` 的"with overlap, only fresh rows are populated"——同一批请求顺序不变时（`prepare_zos_inputs` 返回 `remap`），**只上传新行，老行直接复用上一步的显存内容**（`remap_ptr[i] >= 0` 即跳过）。这既省 H2D，也让设备侧缓冲地址稳定（CUDA graph 复用的前提）。

**(d) 层内：把通信 kernel 和计算 kernel 同时挂在 GPU 上，用 SM 配额切分。**
`v4_modules/mlp.py:469-490` dispatch 与 `:501-511` combine 都以 `async_with_compute_stream=True`、`allocate_on_comm_stream=True`、`previous_event=dispatch_pre_evt` 发射（DeepEP 事件链，`vendor/DeepEP/deep_ep/buffers/ep.py`）；`:602` 在 combine 在飞的窗口里，用 `limit_num_sms(envs.DEEPINFER_USERBUFFER_COMM_SMS)` 把 deep_gemm/tile_kernels 的可用 SM 数临时减掉，**给通信 kernel 留出 SM 跑 shared expert**（`parallel_utils/userbuffer.py:40-52`：`deep_gemm.set_num_sms(prev - num_sms)`）。DeepEP 侧还有配套的"按带宽算该给 copy 多少 SM"：`ep.py:845-852 lb_get_theoretical_num_sms()` = `max(NVLink GB/s / SM 读带宽, NVLink GB/s / SM 写带宽)` 向上取偶数。

**官方文档里唯一"CPU 线程与 GPU 前向重叠"的实例**（`doc:157`）：KV swap-in 从共享内存搬进 HBM 由引擎进程内的一个线程做，目标是"overlap per-layer H2D with forward computation"；并且因为 CUDA copy engine 对细粒度 KV block 拷贝表现很差，改用 **GDR（RDMA 网卡）搬，不占 SM**。**这段代码不在开源发布里**（我全仓找 swap-in/GDR 实现，只有 `doc` 的这段描述，`deepinfer/` `csrc/` 无对应实现）——属于"框架里没有（未开源）"。但它给出的两个判断对我们直接有效：① 细粒度 H2D 不要走 copy engine；② 用独立硬件引擎搬，省下的 SM 继续算。

### 和我们的差距

- 我们是"设备每层自旋等单线程 CPU 车道"。DeepInfer 是"CPU 永远不等 GPU 的大 kernel，GPU 永远不等 CPU"。
- 我们 22 ms/token（40%）在等车道；DDR5 实测 49–51.6 GB/s、车道跑在 38 GB/s。按 token 级专家流量 1.08 GB、其中约 0.5–0.85 GB 走车道折算，有效带宽约 23–39 GB/s，**已经夹在 memcpy 上限（25.6 GB/s）与 DDR5 峰值之间**。所以车道不是"算得慢"，是"带宽到顶"——加线程、换算法都没用，只有三条路：**搬更少、搬得更早、搬得更并发**。
- DeepInfer 对这三条路的答案分别是：搬更少（宽 EP，流量除以 64）、搬更早（一步深流水线，把 CPU 工作提前到上一个 step）、搬得更并发（`async_with_compute_stream` + SM 配额 + 事件链）。

### 能不能抄 / 抄的形状

**必须纠正一个前提**："预取下一层专家是把 CPU 车道延迟藏起来的唯一办法"——层间预取在依赖链上做不到（§1）。真正可抄的是把"车道延迟"从**层内串行等待**变成**跨 step 流水线**，具体三件事：

1. **深 DMA 队列（最高性价比）**。我们自己的实测（`hotpin-design.md` 事实表）：随机 7.74 MB blob DMA，4/16/32 队列 = 23.5 / 30.5 / **35.0** GB/s。也就是说**同一块硬件，只要把并发深度从 4 提到 32，车道带宽 +49%**。DeepInfer 的对应证据是 `ep.py:845-852` 的"按带宽反推并发度"和 `async_with_compute_stream` 的"一次发射整层需求"。落到我们：把"每专家一次邮箱往返"改成"**每层一次票据，携带 8 个专家槽位**"，设备发出票据后立即继续跑本层 shared expert/norm/KV store，最后只自旋在"8 个完成位全置位"的位图上；服务线程侧改成 32 深 DMA 队列连续提交。
   - 改哪一层：邮箱协议（`Chunk::Item` 的 `src[3]/stage/left`，`ds4_model.dp.cpp:757` 附近）+ 服务线程提交循环。
   - 预期：车道 22 ms → ~15 ms，token 时间 ~45 ms → ~38 ms，**+15~20%**（按 35/23.5 的带宽比线性折算，未验证端到端）。

2. **一步深流水线**。把"解析/剪枝/采样参数准备/日志"从 step 内挪到**下一步 forward 发射之后**执行（`worker.py:183-187` 的形状），一步只保留一次设备端等待。我们目前是否有隐式同步不清楚（见 §3），但 DeepInfer 的做法说明：**允许数据结构滞后一步**（用 remap 表达"哪些行是新的"），是消除气泡的关键。
   - 我们的对应物：`Chunk` 的 `keep`/`left` 已经能表达"在途"，缺的是"上一步结果回来后才补齐的字段"的显式版本化。
   - 预期：消除 1–2 个隐式 stall，**5–10%**（估算，需实测）。

3. **加同步守卫**。照 `utils.py:119-132` 写一个 `forbid_sync()` RAII/宏，在"层循环"作用域内任何 `queue.wait()`/`event.wait()`/阻塞 DMA 都 abort（保留白名单：每步末尾那一次）。工作量 1 天，价值是**让"验证窗口一半时间 GPU 空转"从现象变成可定位的报警点**。

**不要抄**：Mega-MoE 的融合（依赖 SM100 TMA/UMMA + 对称内存 + 2-CTA cluster）；`limit_num_sms` 的 SM 配额（我们等的是 DMA 不是 kernel，没有 SM 竞争对象）。

---

## 3. 内核提交策略

### 框架怎么做的

**一步解码 = 1 次主模型 graph replay +（可选）1 次草稿 graph replay + 少量 eager 采样 kernel。** 不是"几百次提交"。

- 捕获：`deepinfer/worker/cuda_graph_runner.py:125` 用 `torch.cuda.graph(graph, pool=..., stream=..., capture_error_mode="thread_local")` 把 `model(...)` 整个包住，即**全部 40 层、所有 MoE/attention kernel 在一次捕获里**；`capture()` 之前先 warmup（`:112-115`），用 `keep_graph=True` 保留原始 `cudaGraph_t` 供 `graph_timer` 匹配事件节点（`:121-124`）。
- 尺寸分桶：`cuda_graph_runner.py:41-51 build_graph_batch_sizes()` = 2 的幂到 interval，之后固定步长；`environ.py:155` 默认 `INTERVAL=8`。`select()` 用 `bisect_left` 向上取（`:143-148`），即**每步用 bs 向上对齐到最近的已捕获尺寸，padding 吸收**。
- 回放：`cuda_graph_runner.py:150-154 replay()`，`deepinfer/worker/model_runner.py:326-327`。
- 预填充**不捕获**：`model_runner.py:73` `self.use_cuda_graph = not ENFORCE_EAGER and engine_type != EngineType.PROMPT` —— prefill 走 eager，逐 kernel 提交，因此 prefill 的提交开销是真实存在的（这也是它们把 prefill 切成固定 token 预算块的原因之一，`doc:115`：32768 token 上限 + 二次成本模型解算 chunk 大小）。
- 草稿模型另有一套 graph，且**与主模型共享显存池和捕获流**：`dflash_cuda_graph_runner.py:68` 用 `pool=graph_memory_pool, stream=graph_stream`（由 `model_runner.py:543-550` 传入主模型的），`replay` 在 `:89`。
- 通信 kernel 也在 graph 里：NCCL 依赖的注释在 `model_runner.py:635-641`（"when using NCCL inside CUDA graphs, NCCL allocates some extra space"）。

**每层有没有隐式同步？在 decode 路径上，框架的设计目标是"零同步"。** 三条证据：
1. `model_runner.py:317` 的硬约束 + `forbid_cuda_sync()` 包裹（§2b）。
2. 需要跨步信息的地方全部改成"设备端 kernel + pinned 回写"，而不是 CPU 读回（`worker.py:214-234`）。
3. 计时器不靠"插同步"，而是**在 graph 内部插 event-record 节点**：`deepinfer/timer.py:101-167`，捕获后 `graph.get_event_record_nodes()` 把探针名字和节点配对（`:142-167`），`flip_event_buffer` 双缓冲（`:169`），因此**可以在一个已实例化的 graph 内部做分段计时而不引入同步**。想看哪层慢就用这个，而不是改代码加 sync。

**唯一的同步点是 `sample_end_evt.synchronize()`（`worker.py:263`），每步一次。**

### 和我们的差距

我们测到"验证窗口一半时间 GPU 空转"。DeepInfer 的对照告诉我们**提交粒度不是主因**（它一步只提交 1 次），**同步点才是**。我们更可能的三个病灶：

- (i) 每层自旋 = 每层一次同步，40 次/步；
- (ii) 验证窗口的 36 专家/层如果按"每专家一次邮箱往返"发，就是 36 次串行 RTT/层；
- (iii) 若在层循环里读了任何设备端标量（决定验证长度、决定是否提前退出），那就是 40 次隐式同步。

DeepInfer 把这类决策全推到设备 kernel（`schedule_mtp_varlen.py:32`）或上一步（`worker.py:118`），正是为了不产生 (iii)。

### 能不能抄 / 抄的形状

1. **"每步一次同步"的验收标准 + 同步计数器**（1 天）：在邮箱/自旋路径上计数，decode 一步内设备端等待次数必须 == 层数（40），不能再多；`queue.wait` 次数必须 == 1。
2. **SYCL 侧的"一步一提交"**：CUDA graph 在 SYCL 没有直接等价物，但有两条路（**均未验证**，需单独 spike）：① `sycl_ext_oneapi_graph`（命令图扩展，录制后 `exec_graph` 一次提交）；② Level Zero 原生命令列表（`zeCommandListAppendLaunchKernel` 一堆后一次 `zeCommandListClose` + `ExecuteCommandLists`）。后者与 CUDA graph 的差别是**没有跨 kernel 的依赖图优化**，但能消掉 per-kernel 提交的 CPU 开销。
   - 现实建议：**先不要碰 graph**。我们当前的瓶颈是 DMA 等待而不是提交开销（提交 300 个 kernel 的 CPU 时间约 0.3–1 ms，相对 45 ms/token 可忽略）。把 §2 的三件事做完，再评估 graph。
3. **把"插桩"做成零成本**：照 `timer.py` 的思路，用 `sycl::ext::oneapi::experimental::logical_group` 或直接 profiler marker，不要为了计时加 `wait()`。

---

## 4. MoE 内核形状

### 框架怎么做的

**结论：解码时不做"每专家一个 CTA"的 group GEMM，而是用一个融合了 dispatch + 两级 GEMM + combine 的 persistent kernel（Mega MoE）。**

判据在 `v4_modules/mlp.py:184-188`：只要 `mma_type ∈ {fp8xfp4, fp8xfp8, bf16xbf16}` **且** EP 组在 NVLink 域内（`PG.is_within_nvl_domain`），就走 Mega MoE；`:571-572` 直接分派到 `mega_moe_forward`。`doc:231` 明说："Mega MoE 是 scale-up 域内的通用解：不论 EP 大小、batch 大小，都比非融合 MoE 明显更快。"

非融合路径（`mlp.py:565-614`）的形状，是理解"为什么必须融合"的关键：
- `m_grouped_fp8_fp4_gemm_nt_contiguous(..., use_psum_layout=True, expected_m_for_psum_layout=expected_m)`（`mlp.py:119-128`）——**psum layout** 表示"每组（专家）的 token 段按前缀和定位、并按全局对齐填充"。
- 这个对齐是 `heuristics/runtime.hpp:57-64`：SM100 上 `mk_alignment_for_contiguous_layout = 256`（`UMMA_N=256` 允许固定 256），SM90 是 128。注释原文很刺眼："**`expected_m` is ignored, so small values may incur performance loss; use Mega MoE directly for such workloads**"。
- 也就是说：**解码时每个专家只有 ~1 行 token，却要按 256 行对齐填充**——非融合 group GEMM 在 decode 上是灾难。这就是 Mega MoE 存在的理由。

Mega MoE 的 m 相关策略切换，写得非常直白（`vendor/DeepGEMM/csrc/jit_kernels/heuristics/mega_moe.hpp:91-110`）：

```cpp
// Expected tokens per expert, plus a one-sigma routing margin for the tile choice
const float num_expected_tokens = (float)num_tokens * num_ranks * num_topk / num_experts;
const float num_covered_tokens = num_expected_tokens + std::sqrt(num_expected_tokens);
// Every M block costs roughly 128 extra rows (its tasks reload the weight tiles), ...
int block_m = num_expected_tokens <= 10 ? 16 : 32;
if (num_expected_tokens > 24) { ... for (candidate : {64,128,192,240}) ... }
```

即 **block_m 是"每专家期望 token 数"的分段函数：≤10 → 16；11–24 → 32；>24 → 64/128/192/240（按 240 行/块的最少块数选）**。并且 `:103` 那句注释是设计原理："每个 M block 要让它的任务重载一次权重 tile，代价约等于 128 个额外行"——**这是"流式权重"内核的核心成本模型**，对我们近乎字面适用。

其他形状细节：
- swap-AB 恒开（`heuristics/sm100.hpp:35-45`）：`swap_ab=true, block_n=128, block_m=mk_alignment, cluster_n = 2 if 偶数` —— 让权重做 MMA 的 A 操作数、token 做流式 B，这样 384 个专家的权重按块流式喂，不需要 materialize 每专家 dense tile。
- persistent CTA：`jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp:236`（contiguous launcher）的 `grid_dim = dim3(config.launch_config.num_sms,1,1)` —— **grid 恒等于 SM 数**，块调度在 kernel 内做。
- 一个 kernel 同时干 dispatch / GEMM1 / SwiGLU+cast / GEMM2 / combine（`MegaMoEConfig` 字段：`num_dispatch_threads`、`num_non_epilogue_threads`、`num_epilogue_threads`、`block_m/block_n/block_k`、`store_block_m_l1/l2`，`mega_moe.hpp:23-46`）。
- 调用点一次/层：`mlp.py:549-559`，只传 `out + 两组权重 + sym_buffer + cumulative_local_expert_recv_stats`。

### 和我们的差距

- 我们解码是 m=1、8 个活跃专家/层（token 级）。用"按专家块的 group GEMM + 256 行对齐"会浪费 256/1 = 256 倍 MMA 槽位（虽然我们是访存瓶颈，MMA 浪费不直接计时，但**权重 tile 被多个 M block 重载**才是真成本，正好对应 `mega_moe.hpp:103` 那句话）。
- 我们真正缺的是"**把一次 MoE 的权重流式读入和计算合成一条流水线**"，即：给定本层专家 id 集合 → 按 blob 连续流式读 → 边读边算 → 累加输出，中间不 materialize 按专家排布的 token 行。这恰好是 Mega MoE 的 `block_m` 与"每块重载权重"逻辑的 CPU/DMA 版本。
- 我们没有"每 token 重读 6.2 GB 稠密权重"的摊销手段，也没有 8 个专家共享权重的机会（每个专家只有 1 行）。

### 能不能抄 / 抄的形状

**内核本身抄不了（SM100 UMMA/TMA/对称内存），但成本模型可以字面照搬：**

1. **"每 M block 重载一次权重 tile ≙ 128 行"** 换成我们的语言：**每多开一个"专家分组/pipeline stage"，就多读一次该专家的整个 blob**。因此解码时必须做到"**一个专家 blob 只读一次，直接喂给所有用它的 token**"。
   - 我们目前若按 (token, expert) 对逐个流式，6 token × 6 专家/ token 的窗口里，同一专家被 2 个 token 命中就要读 2 次 → 直接翻倍。清理点：`ds4_model.dp.cpp` 的 expert 分派循环里按 expert 归并 token（一个 384 槽的 `vector<pair<blob, span<tokens>>>`）。
   - 预期：验证窗口下 MoE 字节数下降 **15–35%**（取决于窗口内专家重叠率；6 token、384 专家、k=8 时重叠率约 20–40%，需实测）。

2. **block_m 的分段思想**：给"每专家行数"设两个档（≤4 行 → 逐行 GEMV；≥8 行 → 分行 tile），避免为了通用性选一个中庸 kernel。
   - 我们解码永远是小 m，重点是保证 m=1 的 GEMV 路径不被 batch 路径的模板参数拖累（编译期特化 m=1..8）。

3. **persistent/grid-strided**：我们的 DMA+计算循环如果每层一个新 kernel，就有 40 次启动+40 次尾部气泡；改成一个 kernel 内循环所有层（层 id 由计数器推进）能消掉。**但这与"每层自旋等 CPU"冲突**，只能在车道完全隐藏之后做（即先做 §2）。

---

## 5. 投机解码（MTP/DSpark）

### 框架怎么做的

**有，而且是这套代码里最完整的一块。** 关键开关：`DEEPINFER_USE_NEXTN_DECODE`（默认 5，`environ.py:129`）、`DEEPINFER_ENABLE_MTP_VARLEN`（`environ.py:131`）、`DEEPINFER_MAX_BACKTRACKING_TOKENS`（≥ nextn+1，`environ.py:132`）。

**(a) 草稿怎么生成：一次 bulk 前向，不自回归。**
`model_executor/models/mtp/dflash.py:230-239 _embed_draft_init_tokens()`：一条请求一次性喂 `[golden, noise, noise, noise, noise, noise]`（`noise_token_id`，长度 = `max_MTP_value` = 5），`forward_drafts()`（`:241-269`）用固定 stride 的 metadata 一遍跑完所有草稿层，然后在 `sample_drafts()`（`:279-345`）里**沿 `hidden_states[step_i::max_MTP_value]` 切片，逐位置采样**（`:309`），5 个位置各自算 logits、markov 修正、置信度（`:310-321`）。所以 GPU 上是"一次前向 + 5 次轻量采样"，不是 5 次前向。

**(b) 草稿的注意力/KV 怎么处理（这是最值得偷的设计）。**
- 草稿层有**独立 KV**（`nextn_kv_caches`，`dflash.py:202`、`:224`），窗口与主模型 SWA 强制一致（`:143-144` 断言），所以草稿 KV 只占一个 SWA 窗口。
- **"main token" 的 K 在主模型前向里顺手写好**：`update_main_cache()`（`:198-228`）把 `dspark_target_layer_ids`（`:146`）指定的主模型中间层特征，经一个 `projection`（`:148-153`，输入维度 = hidden × 目标层数）+ `projection_norm` + `main_kv_proj`（`:220`，把 N 个草稿层的 K 一次算完）写成草稿 KV。**草稿模型因此不需要跑 golden token 的 prefill，零额外成本拿到上下文**（EAGLE 血统）。
- 注释还解释了一个反直觉点（`:205-208`）：解码时主模型 graph replay 时接受结果还没产生，所以**所有 draft 位置都按 slot 写入**——被拒绝的位置正是随后写 draft K 的位置，在任何读取之前就被覆盖，因此无需按接受结果屏蔽。

**(c) 接受判定：标准拒绝采样 + 前缀截断。**
- 元数据：`tilelang_ops/sampler.py:13-26 prepare_verification_metadata_ref()` 把 `total_q` 行按请求切开，`is_verify = (local_step != seqlen_q-1)` —— **每条请求的最后一行是"新鲜采样行"，不参与验证**，即窗口 = 1 + n_draft 行，产出 = 1 + n_accepted 个 token。
- 判定：`tilelang_ops/sampler.py:227-250 get_accept_result_ref()`：`accept = golden != -1 ? (golden == draft_token_id) : raw_accept`（贪心/强制 token 覆盖），然后 `blocked = ~(is_verify & accept)`，**前缀和**求 `alive`，`effective_accept = alive & is_verify & accept`，`is_final = alive & blocked` —— 这就是"第一个拒绝之后全部作废"的无分支实现。
- 拒绝采样本身：`tilelang_ops/fused_sample.py:63` 文档与 `:232` 代码，`residual = (p_target - exp(logp_draft - DRAFT_PROB_PENALTY)).relu_()`，即 `max(p - q, 0)` 归一后重采样。
- 全部塞进 fused kernel：`sampler.py:411-540 _verification()` 调 `process_sampling`（= `fused_sample`，一次 kernel 完成温度/top-p/top-k/logsoftmax/采样/验证）、`get_accept_result`、`finalize_verification`，**每步约 5–6 个 eager kernel launch**。

**(d) 按置信度裁剪验证窗口（省算力的核心技巧，我们最需要）。**
- 置信度头：`dflash.py:55-88 ConfidenceHead`（`n_embed + markov_rank → 1`，带 per-step 温度），调用在 `:316-321`，写进 `confidence_rates[:, step_i]`。
- **CPU 侧版本**：`csrc/engine/mtp_varlen.hpp:150-267 MTPVarlen::schedule()`。
  - 每请求每位置的接受率入队（`:173-185`），排序（`:210-214`）；
  - 用**在线拟合的耗时模型**估每一步时间：`_forward_time_slope`（`:48-131`，按 graph_size 聚合取两层中位数，`_window_size=5000`）+ `baseline_ms`（装载期测的 decode 性能曲线）+ `sampler_ms`（按 num_seqs 聚合的中位数，`:216-223`）；
  - 然后**穷举 total_q，最大化 `期望产出 token 数 / 预计耗时`**（`:228-258`）：`cur_expected += accept_rate`，`cur_bsz += 1`，`throughput = cur_expected / (baseline + slope*graph_size + sampler_ms)`；
  - 最后按 accept_rate 降序分配每请求的 `seqlen_q = 1 + win_count`（`:262-266`）。
  - 注释还专门处理了 ZOS 滞后（`:160-208`）：**用 N-2 轮的接受率决定 total_q，用 N-1 轮的决定每请求分配**——因为"confidence_{i+1..nextn} 不能影响 draft_i 是否被验证"。
- **GPU 侧版本**（ZOS 下必须，因为 CPU 拿不到上一步的 confidence）：`tilelang_ops/schedule_mtp_varlen.py:11-32` + `:98`：`cum_rates = cumprod(confidence)`，全局展平后 `argsort` 求 rank，`seqlens_q = 1 + (rank < extra_budget).sum()`。**一个 kernel 完成"哪条请求验证几个 token"，零 CPU 参与**（官方 `doc:264` 描述了同一件事）。
- 效果（`doc:249-256` 表）：每步验证的 draft 数 5.00 → 3.2–4.4（-30%），平均接受长度基本不变（3.29 vs 3.20），在 60 并发/rank 的生产点 **+5.4%**，96 并发 **+11%**。

### 和我们的差距

- 我们有 DSpark，实测"草稿 7 ms、接受率 100%、验证一步 285 ms，净亏"。根因：**验证成本随 token 数线性增长，而我们没有 amortization**（因为 MoE 权重不在显存里，6 个 token 路由到 ~36 个不同专家，就要多搬 36 份 blob；而 DeepInfer 的权重全驻留，多验证 6 个 token 只增加算力，不增加权重读取）。
- 换算：285 ms / 6 token = 47.5 ms/token，单 token 是 43–51 ms/token —— **实测摊销率 ≈ 0**。加草稿 7 ms 后就是净亏。这不是"实现不好"，是**在权重流式架构上投机解码的固有性质**。
- 我们缺的正是 (d)：按"边际成本"而不是"固定 6"来选验证窗口。DeepInfer 优化的是"每单位**算力**的边际收益"，我们要优化的是"每单位**字节**的边际收益"。

### 能不能抄 / 抄的形状

**这是全报告最高优先级的一条。**

1. **把 `MTPVarlen::schedule` 的目标函数换成字节成本模型**（`mtp_varlen.hpp:228-258` 直接改）。
   - 装载/预热期测一条曲线 `cost(k)`：k = 1..6 验证窗口的实测耗时（我们已经有 1 token 和 6 token 两个点，补测 2/3/4/5）。
   - 每步：候选位置按 `cumprod(confidence)` 排序（照 `schedule_mtp_varlen.py:98`），逐个加入窗口，**边际成本 = 该位置新引入的专家 blob 字节数 / 38 GB/s + 固定稠密成本/6**，边际收益 = 该位置置信度 × 1 token。
   - 取 argmax `E[accepted] / cost`，得到每步的 k*（大概率落在 2–3）。
2. **k* 的执行为"验证窗口截断"**：`seqlen_q = 1 + k*`，前缀语义天然成立（`get_accept_result_ref` 的前缀和逻辑与 k 无关）。
   - 若无置信度头可用，退化为"按上一步实测 cost(k) 选 k"，仍然能避免固定 6 的净亏。
3. **加一条"专家去重"**（与 §4-1 同一处改动）：窗口内多个 token 若命中同一专家，只搬一次。这条和 k* 选择互补：k 越小，重叠率越高，字节越省。
4. **预期收益**：以 47.5 ms/token 的实测摊销率为 0 为基线，若 k=2 时 cost(2) ≈ 1.5×cost(1) ≈ 70 ms，则 2 token / 70 ms = 35 ms/token → **28 tok/s，+20~35%**；若 k=3 且重叠好，可能到 32 ms/token。**具体曲线必须实测，此处为估算（未验证）**。
5. **明确不要抄的**：DFlash 的"草稿 KV 由主模型前向顺手写入"（`dflash.py:198-228`）需要草稿层与主模型对齐的 projection 权重，我们的模型若没有 dspark_target_layer_ids 那套结构，拿不到收益。

---

## 6. 长上下文注意力

### 框架怎么做的

三层复合结构，**解码成本与上下文长度基本脱钩**（100K 与 4K 的 decode 注意力开销同阶）：

1. **压缩 KV（compressor）**：`models/v4_modules/compress.py:1-6` 定义"gated low-rank 压缩 → norm → 写 main KV cache"，`kv_compress_ratio` 决定压缩比（`attn.py:239-241`）。压缩后的 latents 存进 **block 分页缓存** `ca_cache`。
2. **稀疏索引器（indexer / MQA）**：`index_attn.py:56-57` "对压缩 KV 打分并选 top-k 块"。
   - 打分：`relu(q@k) · weights`，index Q/K **恒为 FP4**（`index_attn.py:86-87` 注释与 `FP4_INDEX_KV_FORMAT`）。
   - 选择：`_topk_decode()`（`:456-515`）：先算 `p_sum`，再 `topk(p_sum, self.topk_tokens, end=topk_lens)`（`:494-504`），最后 `convert_kv_index()`（`:508`）把"序列内逻辑下标"就地转成"KV cache 物理 slot"（`-1` 保留）。**因此 decode 注意力只需要读 topk 个 slot**。
   - **两级候选**（`:486-511` + `:104-121`）：先按 `candidate_block_size ∈ {8,16}` 的块粗选 `candidate_topk_blocks`（≤4096，4 的倍数），再在候选内精排。粗选的 scores 可以**跨层共享**（`is_candidate_topk_source` / `use_candidate_topk`），即"一层算候选块，后续层直接复用"。
   - 分块防降频：`compute_chunk_size = min(8192, ...)`，注释明说"mqa 在 chunk 大时会降频"（`:115-120`）。
3. **SWA 精确窗口**：最近 `window_size` 个 token 的**未压缩** KV 存在独立的 state cache（`cache_engine.py:217` `swa_cache: [num_state_seqs, swa_block_size, 1, 528]`），`swa_block_size = window + 回退余量`（`cache_engine.py:88-108`）。解码时**主 KV（压缩、稀疏 topk）与 SWA（稠密、精确）拼在一起**：`attn.py:641-653` 同时传 `key_cache=swa_cache` + `key_cache_extra=ca_cache` 及两组 indices 给一个融合 kernel。
4. **一次 kernel 干完**：`attn.py:655-668` 调 `decode_mla_mqa_sparse(...)`，带 `fused_norm_rope_cast_kwargs` 时由 `fused_norm_rope_attn_rope_cast.decode` 一个 kernel 完成 Q 归一化 + RoPE + 稀疏 MLA + 输出 RoPE + cast（`layers/attention_backend.py:88-120`）。

**KV 布局与分页**：
- 三类缓存形状（`cache_engine.py:206-217`）：index cache `[num_blocks, block_size//ratio, 1, index_bytes]`；压缩 cache `[num_blocks, block_size//ratio, 1, bytes]`；SWA cache `[num_state_seqs, swa_block_size, 1, 528]`（528 B/token = MXFP8 格式，`doc:169`）。
- 分页粒度按压缩比换算：`tokens_per_block = block_size // compress_ratio`（`csrc/engine/zos.cpp:211`）。
- 预填充时把 paged + ring 的 KV **gather 成连续 dense** 再进 attention：`csrc/engine/input_metadata/block_load_plan.cpp:23-46`（`append_paged_range` / `append_ring_range`），输出 `[N,4]` 段表到 pinned。
- **SWA 单独存主机内存**，128 token 对齐、只存前 20 层（`doc:163-169`），每序列仅 1,415,040 B。

**长上下文不退化靠"有界重放"**（`doc:183-191`）：SWA 命中缺失时只重算最后 128 token，且查询 i 只对 `[max(s,i-127), i]` 的 key 计算；解码器层因 CED 架构只需重算最后 128 token（否则要跑 2560 token × 20 层），**"几乎把预填充计算减半"**。命中策略：`hit = L1` if SWA 命中 else `L1 - 128`，上限 `L - 134`（128 窗口 + 6 回退）。

### 和我们的差距

我们 100K 预填充 9–11 分钟（150–180 tok/s），之后解码掉到 11–13 tok/s（对比短上下文 19.5–23.2）。也就是说**长上下文把每 token 成本推高约 35–45 ms**，这部分几乎全是 attention。DeepInfer 的等价场景下 decode 注意力成本恒定（topk 个 slot + 134 窗口），因为它把"上下文长度"换成了"索引器算力"。

### 能不能抄 / 抄的形状

**结构能抄，内核不能抄（FlashMLA/DeepGEMM 不可移植）。** 优先级取决于我们是否真的常用 100K：

1. **两级候选 + 跨层复用候选块**（`index_attn.py:104-121`, `:486-511`）是最容易移植的部分：粗选可以用我们已有的量化点积 kernel，候选块 scores 复用可以省掉 3/4 层的索引计算。
2. **SWA 精确窗口 + 压缩 KV 并存的双缓存**：我们若目前是"单一 paged KV 全量 attention"，这是最大改动；但收益也最大（100K 解码 11–13 → 20+ tok/s，**+50~80%**，未验证）。
3. **有界重放**（`doc:183-191`）：如果我们有 prefix cache，命中时不要重算全量 SWA，只重算最后 128–134 token。**前提**：模型必须真的是 SWA/混合注意力结构；若 V4.1-Flash 的层配置里 `window_size` 与 `n_prefill_layer` 生效，这个能直接吃到。
   - 注意：这只对**有 prefix cache 的多轮场景**有效；单次 100K 冷预填省不了。
4. **预填充 chunk 调度**：`doc:115` 的"二次成本模型 `cost(q,c)` 反解 chunk 大小、32768 token 上限、DP 组内 all-reduce MIN"——我们单卡，直接可用的是"**按剩余时间预算反解 chunk**"，避免长 prefill 里某个 chunk 拖尾。收益主要体现在 TTFT 稳定性，不是吞吐。

---

## 7. 量化内核

### 框架怎么做的

**只支持 MXFP4 / FP8（含 E2M1、E4M3、UE8M0 scale），完全没有 IQ 系列，没有任何 GGUF/ggml 血统的回退路径。**

- 格式枚举被硬编码：`model_executor/layers/tilelang_ops/cache_common.py:20-21`
  ```python
  QuantizedFormat = Literal["e2m1", "e4m3"]
  ScaleFactorFormat = Literal["ue8m0", "e4m3", "fp32"]
  ```
  `:44-52` 的 `__post_init__` 直接拒绝"混合 quantized/BF16"缓存。
- 各处的具体配方（`doc:217-221`）：main KV = per-16 e4m3（"像 NVFP4 但没有 global scale"）；indexer KV = per-32 ue8m0 MXFP4；**MoE routed 专家权重 = per-32 ue8m0 MXFP4**。
- MoE 的 mma 类型只有三种（`v4_modules/mlp.py:177-182`）：
  ```python
  act_type  = "fp8" if use_fp8_act else "bf16"
  weight_type = "fp4" if use_fp4_weight else "bf16" if ... else "fp8"
  assert self.mma_type in ("fp8xfp4", "fp8xfp8", "bf16xbf16")
  ```
  即**激活 FP8 × 权重 FP4（MXFP4）** 是主力路径，`fp8xfp4` 在 Ascend 上也能走 Mega MoE（`mlp.py:185`）。
- 权重重打包在加载期做：`mlp.py:41-91 process_moe_weight()`（经 `deep_gemm.transform_sf_into_required_layout` 转 SF 布局）、`:93-106 _convert_mega_moe_weight()`（`deep_gemm.transform_weights_for_mega_moe`，转完 `untyped_storage().resize_(0)` 释放原权重，`:104-105`）。
- **IQ2_XXS / IQ1_M / Q4_K / GGUF / GPTQ / AWQ / Marlin：全仓 0 命中**（`grep -i "iq[0-9]|gguf|q2_k|q4_k|awq|gptq|marlin"` 在 `deepinfer/` `csrc/` 无结果）。
- 也没有"未支持格式的 fallback kernel"——不支持的 `mma_type` 直接 `assert` 死（`mlp.py:182`）。

### 和我们的差距

我们是 IQ2_XXS（~2.06 bit/weight）。DeepInfer 是 MXFP4（4 bit + per-32 ue8m0）。**差 2 倍**。这意味着：
- 同一份 32 GB，我们理论能放 ~2× 的专家（实测 ~1500，若换 MXFP4 只能 ~750）。
- 但 MXFP4 有原生 tensor core 支持（UMMA 直接吃 E2M1），我们 IQ2_XXS 要在 kernel 里反量化 → 只能走 SIMT/DPAS 近似，算力利用率差一截。
- **框架对我们这条路"零参考价值"**：没有 IQ kernel，没有 2-bit 的任何代码，也没有"低比特权重 + 主机内存流式"的组合（它假设权重在 HBM 且格式是硬件友好的）。

### 能不能抄 / 抄的形状

**框架里没有，抄不了。**能借的只有两条间接经验：

1. **权重重打包放在加载期、并立刻释放原权重**（`mlp.py:93-106`）：我们的 `expert-profile` + blob 化已经是这个思路，可以再加一层"**按 DMA 对齐（64B 或 4KB）重排 blob 内部布局**"，让一次 `memcpy` 就是一次完整专家（hotpin 设计里 `hot_stride` 已是 64 对齐）。收益：消除 DMA 的 scatter，配合 §2 的深队列才有意义。
2. **per-32 的 scale 布局**（`ue8m0`，`cache_common.py:44-52`）：IQ2_XXS 是 per-256 的 4-bit scale + 符号表。若未来考虑 IQ2 → MXFP4 的上采样/混合（热专家用 MXFP4、冷专家用 IQ2），DeepInfer 的"`QuantizedFormat` + `ScaleFactorFormat` 数据类 + 每层可选格式"（`cache_engine.py:377-478` 按 `layer_id/attn_module` 逐层构造缓存视图）是个可用的组织方式，但**这属于"如果要做混合精度"，不是"抄它的内核"**。

---

## 8. 单卡 32 GB 级别的取舍

### 框架怎么做的

**没有 KTransformers 式的"稀疏专家上 CPU"。整份代码库、文档、以及 GitHub issue/PR 检索都没有这条路径。** 它的取舍是**反向的**：与其把专家放内存，不如把专家摊到更多卡上。

- 支持的硬件写死：`README.md:35-38` "NVIDIA SM100 GPUs / Huawei Ascend 950 NPUs"。没有消费级/单卡档位。
- 官方对"Wide-EP 为什么能救 decode"的回答（`doc:55-59`）："**Wide-EP 大幅降低 MoE 专家部分每步的显存流量**，配合好的 EPLB 有效提升 TPS"，并给了 EP8 vs EP64 的对比图（`doc/images/wide_ep_decode_pareto.png`）。基本原则（`doc:49`）："Decode: EP64–128；用你能拿到的最大 EP/DP"。
- 量化内存账（`doc:103`）：EP64 时每卡 7 个专家（含冗余、不含 shared）。若我们模型 384 个 routed/层，EP64 → 6 个/层/卡，全部驻留 HBM。
- 实测吞吐：`doc` 给的是**每 rank** 的数字（`doc:249-256` 表：EP64/DP64、60 并发/rank 约 13,730 tok/s/rank），没有"32 GB 单卡"的任何数据点。
- GitHub 侧：`repo:deepseek-ai/DeepInfer offload cpu` → **0 条 issue/PR**；`single GPU consumer` → 3 条都是无关 PR（LocalEngram NVLink、finished_payload 泄漏、SM90 支持）。即社区里没有这方面的工作。
- 唯一像"分级"的东西是 KV：`doc:131-135`（HBM / 本地主机内存 / 3FS）+ `doc:157`（swap-in 线程 + GDR）；**权重没有分级**。

### 和我们的差距

我们是它的镜像：它用 64 卡消灭"权重不够放"这个问题；我们只有 1 卡，必须面对。所以这一节它**没有可抄的方案**，但有两个可抄的**判断**：

1. **"每步显存流量"是 decode 的第一性指标**（`doc:57`）。换算到我们：每 token 的专家字节数 = 命中率 × 0 + miss率 × blob。我们实测 1.08 GB/token（`hotpin-design.md`），而 dense 部分 6.2 GB/token —— **稠密权重流量是专家流量的 5.7 倍**！这一点框架给不了帮助，但提醒我们：`6.2 GB @ 483 GB/s = 12.8 ms` 是硬底，专家部分 ~22 ms 是另一块，两者**串行**才是 45 ms。因此**并行化 dense 与 host 专家流**（在等车道时算 dense）是理论上的最大单项收益。
   - 但注意依赖链：dense 投影（QKV/O/shared expert）大多在路由之前，无法与"等本层专家"重叠。**唯一能重叠的是 shared expert（在 routed topk 之后才需要它的输出）**，见 §2。
2. **"宁可少放、不要慢搬"**：它宁可把专家摊到 64 张卡也不做 PCIe 流式。我们的实测支持同一结论的方向——PCIe H2D 36 GB/s 是墙，所以**提高显存命中率（更多热专家常驻）比优化搬运更值钱**。这印证了 hotpin 的方向（pin ≥29% 访问就让 PCIe 成瓶颈 → 推荐每层 32 个 = 10 GB）。

### 能不能抄 / 抄的形状

**方案抄不了（它没有），判断可以照抄。**具体动作：

1. **把"每 token 字节账"做成常驻指标**（照 `doc:57` 的思路，1 天）：每步打印 `dense_bytes`, `routed_vram_bytes`, `routed_host_bytes`, `host_lane_ms`, `gpu_busy_ms`。没有这本账，§2/§4/§5 的任何改动都无法归因。
2. **热集大小按"边际 tok/s / GB"决策**，而不是按覆盖率：hotpin 已有 `coverage` 计算，补一个"每多 pin 1 GB → 车道少 X ms → 多 Y tok/s"的斜率，拐点即最优（hotpin 推算拐点在 29% 访问命中）。
3. **不要走"权重分层 + 换取更大专家驻留"的路**（即不要为了塞下更多专家而降精度到 IQ1_M）：`doc:57` 的逻辑是"减少每步流量"，而降低位宽同时也降低 kernel 效率；我们的 IQ2_XXS 已经在 2 bit，继续降的边际收益远小于把命中率从 10% 抬到 20%。

---

## 9. 按"收益 / 工作量"排序（Top 5）

| # | 动作 | 收益（估算） | 工作量 | 依据（可直接照抄的位置） |
|---|---|---|---|---|
| **1** | **验证窗口按"边际字节成本"裁剪**：装载期测 `cost(k), k=1..6`；每步按 `cumprod(confidence)` 排序，取 argmax `E[accepted]/cost`；执行 `seqlen_q = 1+k*` | DSpark 由净亏转正；**+20~35%**（当前摊销率≈0，实测 285ms/6token=47.5ms/token ≈ 单token 45ms） | 2–3 天（纯 CPU 侧模型 + 窗口截断，无新 kernel） | `csrc/engine/mtp_varlen.hpp:228-266`（目标函数）、`tilelang_ops/schedule_mtp_varlen.py:98`（cumprod+全局 topk 预算）、`sampler.py:227-250`（前缀接受，与 k 无关） |
| **2** | **一层一张票据 + 32 深 DMA 队列 + 设备端完成位图**（一次自旋） | 车道 22→15 ms，**+15~20%** | ~1 周（邮箱协议 + 服务线程，`ds4_model.dp.cpp:757` `Chunk::Item` + `:4362-4390` 命中重定向） | 自有实测：队列 4/16/32 = 23.5/30.5/**35.0** GB/s；DeepInfer 对应形状 `mlp.py:479 async_with_compute_stream`、`ep.py:845-852`（按带宽反推并发度） |
| **3** | **同步审计 + 守卫**：给层循环加 `forbid_sync` 型守卫（白名单：每步末尾一次）；统计每步设备端等待次数（应 == 层数） | 定位"半时空转"；通常能抓出 1–2 处隐式同步，**5–10%** | 1–2 天 | `deepinfer/utils.py:119-132`（forbid_cuda_sync 语义）、`worker.py:262-273`（唯一同步点）、`timer.py:142-167`（插桩但不加同步） |
| **4** | **层内填充：发完路由专家请求后立刻算 shared expert + 本层 norm/rope/KV store**，回来再 combine | **5–10%**（把 ~1–2 ms/层 dense 塞进 22 ms 等待窗） | 3–5 天 | `v4_modules/mlp.py:601-609`（先 `capture()` 事件、再用 `limit_num_sms` 减 SM 跑 `shared_forward`、然后以该事件为 `previous_event` 发 combine，两者并发） |
| **5** | **验证窗口内专家去重**：按 expert 归并 token，一个 blob 只搬一次 | **15–35%** 的 MoE 字节（6 token × 384 专家重叠率 20–40%） | 3–5 天（与 #1 同一处循环，可合并做） | `vendor/DeepGEMM/.../mega_moe.hpp:103`："每个 M block 让任务重载一次权重 tile，约 128 行的代价"——即"多分组=重复读权重"的成本模型 |

**不做/缓做**：CUDA graph 的 SYCL 等价物（提交开销 ~0.3–1 ms，相对 45 ms/token 可忽略；先做 #1–#5 再评估）；热集窗口刷新（+2~5%）；压缩 KV + 稀疏索引（仅长上下文有收益，工作量最大，若 100K 不是主场景则不做）；量化（框架无参考）。

---

## 附：明确的"框架里没有 / 我没找到"

| 我们想找的 | 结论 | 检索依据 |
|---|---|---|
| 专家按显存/内存/盘分级放置 | **没有**。只有卡间 EP 均分 + 冗余副本 | `moe.py:94`、`model_loader.py:83-95`、`eplb/__init__.py:42-138` |
| 按层分配显存配额 | **没有** | 全仓无 per-layer 显存预算代码；`GPUCacheInterface` 只分 KV/state |
| 专家热度统计 + 动态换入换出 | **有统计，但换的是"卡"不是"介质"** | `eplb/__init__.py:146-290`、`doc:97-103` |
| 预取下一层专家 | **没有** | 唯一 `prefetch` API 是 `DeepEP/ep.py:854 lb_prefetch_weights`（EPLB 重排推送）；`block_load_plan.cpp` 是 KV gather |
| CPU 计算专家 / CPU 车道 | **没有**（CPU 只做调度与后处理） | 全仓 `offload|swap_out|cpu_weights` 仅命中 KV/多模态 staging |
| 每层同步等待 | **decode 路径上设计为没有**，每步仅一次 `sample_end_evt.synchronize()` | `worker.py:262-263`、`model_runner.py:317`、`forbid_cuda_sync` |
| CUDA graph 里能否捕获通信 kernel | **能**（NCCL 与 DeepEP 都在 graph 内） | `cuda_graph_runner.py:125`、`model_runner.py:635-641` |
| 内核提交次数 | **一步 decode = 1 次主模型 replay + 1 次草稿 replay + ~5–6 个采样 kernel** | `cuda_graph_runner.py:150-154`、`sampler.py:411-540` |
| m=1 vs m=6 不同内核路径 | **同一内核，按"每专家期望 token 数"切 block_m（16/32/64/128/192/240）** | `mega_moe.hpp:91-110` |
| IQ2_XXS / IQ1_M / GGUF kernel | **完全没有**，也不支持 fallback | `cache_common.py:20-21`、`mlp.py:177-182`；`grep -i iq|gguf|q4_k` 零命中 |
| 单卡 32 GB / KTransformers 式设计 | **没有**，且官方明确推荐 EP64+ | `README.md:35-38`、`doc:49,55-59` |
| KV swap-in 线程 / GDR 实现 | **文档有描述，代码未开源** | `doc:157` 有；`deepinfer/` `csrc/` 无实现 |
