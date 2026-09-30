# 长推理与多轮对话：均值和 norm 漂移诊断与修改建议

日期：2026-09-29。代码基线：`732d440`。模型：`/data/zjx/data-old/model/Llama-3.1-8B-Instruct`。

**建议采用无需训练的分块统计更新，并在检索时补偿块间均值偏移。不要直接打开当前 forward-only EMA，也不要把离线学习一个固定均值方向作为主要解决办法。**

更新状态：分块统计补偿和逐 GQA 查询头打分已接入 Python 缓存、CUDA kernel 和模型生成路径，默认配置为 `block + mean_prob`。已完成 CUDA 回归和 Llama-3.1-8B 在线生成验证，详见 §9。本文 §2 保留修改前的审计结论，§5 保留初始离线诊断。检索预算控制器和持续复用 KV 的多轮输入接口尚未实现；离线检索改善不能直接证明长推理任务准确率改善。论文 PDF 未修改。

后续已补充 124 题、448 次实际生成及三个投影种子的查询聚合消融，见 [查询聚合实测](cometkv_query_aggregation_ablation.md)。没有观察到 mean_prob 稳定优于 q_sum 的证据；效率优先时建议显式使用 `block + q_sum`，默认参数尚未再次修改。

2026-09-30 补充：[尾部采样分析](cometkv_tail_sampling_analysis.md) 给出同题 tail ON/OFF 配对结果及跨层共享时的支持集反例。后续已完成 [独立 tail 配额与预算修复](cometkv_tail_budget_fix.md)：head 容量覆盖完整计划生成长度、tail 额外计量、跨层支持集与重复计入已修正。本报告 §4.5 的容量问题属于修复前审计；§4.6 的质量反馈设计仍是待验证方案。

## 1. 审稿人担忧中需要区分的三件事

当前均值是每个 layer、batch item、KV head 的 **post-RoPE key 均值向量**；norm 是中心化残差 `||k - μ||`，不是 query norm，也不是模型的 RMSNorm。

1. 历史 key 在标准因果推理中已经固定，不会因后续生成而自行漂移。当前模型使用固定 Llama-3.1 RoPE 配置。
2. 新生成 token、新用户输入、话题切换及不同位置会带来新的 key 分布，导致 prompt 均值不再是新 token 的良好中心。固定中心并不破坏精确内积的数学正确性，但可能增加有限位宽签名的误差。
3. 固定 log-norm 范围遇到新的残差大小时会发生饱和截断。这与均值方向变化相关，但不能只靠观察原始 `||k||` 判断。

因此应回答“如何使新增 token 的压缩误差保持受控”，而不是预先承诺“所有任务、位置、轮次的均值都不变化”。

## 2. 修改前实现审计（基线 732d440）

| 位置 | 修改前行为 | 结论 |
|---|---|---|
| `cache_hub/cometkv_cache.py::_build_prefill_signatures` | prefill 计算 μ 和 log-norm 的 min/max，后续新增签名复用统计量 | 默认冻结；超范围 norm 被截断到 0/255 |
| `_update_lockstep_evicted_retrieval` 的 `mean_update_alpha` 分支 | 一个 eviction 批次编码完成后才更新 μ，旧签名保留旧中心 | 不同批次评分缺少均值差补偿；固定 norm 范围也没有随中心变化 |
| 同函数的 `full_recompute_interval` 分支 | 第 0 层写入新 token 后，立即对所有层执行重建 | 后续层尚未写入对应 token，存在未初始化读取 |
| `_full_recompute_stats` | 全量读 K、重算统计量、重建全部签名 | 可作正确性对照；反复读回 CPU K 会破坏轻量追加的性能优势 |
| `_full_recompute_stats` 的 int8 路径 | 直接把 `cpu_key_cache` 转成 float | 没有乘回 `cpu_kv_k_scale`，会在量化码空间而非真实 key 空间重建 |
| `model_hub/LLM.py::generate` | 每次调用 `init_kv_cache` | 连续调用 `generate` 是重建缓存，不等于持续复用同一 KV 的多轮服务 |

修改前三个统计量开关默认都是 0，CUDA score kernel 只接收每个 row 的一套 `norm_lo/norm_step`，没有每块均值、每块量化尺度或均值补偿项。更新后的分组 kernel 接收每块尺度和补偿；旧 kernel 保留给 `frozen + q_sum` 消融。

重建顺序问题已用原始方法体做最小复现：用 CPU tensor 和简化的存储/打包函数隔离调用顺序，在尚未写入的槽位放置 NaN 哨兵。第 0 层触发重建后，第 1 层 μ 变为 NaN；随后第 1 层正常 eviction 仍不能修复。见 `results/drift_diagnostic_llama31/rebuild_order_probe.json`。这证明读取顺序有误，不表示真实未初始化显存一定含 NaN。

## 3. 为什么不能直接 EMA

所有候选使用同一个 μ 时，有

\[
q^\top k_i=q^\top(k_i-\mu)+q^\top\mu.
\]

最后一项对所有候选相同，精确排序不变。这个等价关系本身不要求 μ 是最新均值。

如果第 e 块使用自己的 μ_e，则忽略均值项后的跨块分数差是

\[
q^\top(k_i-\mu_a)-q^\top(k_j-\mu_b)
=q^\top(k_i-k_j)-q^\top(\mu_a-\mu_b).
\]

缺失的是一个**一阶、依赖 query 的偏移**。现有注释称其为“second-order”没有数学依据。例如标量 q=1，k_i=1、μ_a=0，k_j=2、μ_b=10：原始得分 1<2，中心化得分 1>-8，排序直接翻转。

同理，更新一份全局 norm scale 却保留旧字节，会让旧字节被新的标尺错误解码。必须“旧签名和旧统计量一起保留”，或者重编码全部受影响签名。

## 4. 推荐机制：块封存后冻结其统计量

沿用现有 recent window 和 eviction 批次。默认第一次 eviction 为 96 个 token，之后每批 128 个；不要求逐 token 更新历史索引。

1. prefill 的所有签名保留 prompt 的统计量，作为块 0。
2. 新 token 仍在 recent window 时使用完整 KV attention。
3. 一批 token 被移出 recent window、进入检索索引时，利用其仍在 GPU 上的 K 计算本块均值 μ_e，以及本块残差 log-norm 的范围。
4. 用这些统计量编码该批 token，冻结该块的 μ_e、lo_e、step_e；历史签名和对应统计量一起保持有效。
5. 检索时，用候选所属块的 norm 参数解码，并加回块均值的 query 内积。

不依赖未来 token；新块封存时才计算统计量。单独对 query 做中心化会改变目标排序，不能与这里的 key 平移混用。

### 4.1 评分补偿必须与现有投影得分尺度一致

设 x_i=k_i-μ_e，r_i=||x_i||，b_i=sign(Px_i)，当前原始签名分数是

\[
A(q,i)=\hat r_i(Pq)^\top b_i.
\]

对于随机正交投影的 m 个单位行，在不计 norm 量化误差时，投影随机性的期望满足

\[
\mathbb E_P[A(q,i)]=c_{d,m}\,q^\top x_i,
\quad c_{d,m}=m\frac{\Gamma(d/2)}{\sqrt\pi\,\Gamma((d+1)/2)}.
\]

这里 d=128、m=120，c≈8.47939。因此可使用

\[
\boxed{S(q,i)=A(q,i)+c_{d,m}\,q^\top\mu_e}
\]

或者等价地 `A/c + q·μ_e`。为了保持与旧 prompt 分数相同的公共基准，也可以加 `c q·(μ_e-μ_0)`。**不能在未归一化的 A 上直接加裸的 q·μ_e。**

上述关系是对投影随机性的期望；固定投影、120 个 sign bit 和 norm 量化仍有误差。补偿恢复共同的比较目标，不保证压缩后的 top-k 与精确内积完全相同。

代码使用 GQA query 的 sum，补偿也必须使用同一 qsum。若改用 mean，则两部分一起缩放。每一步先算每块一个偏移，scan kernel 按 token 的块号加载偏移，不必为每个 token 重新计算 128 维内积。

### 4.2 Norm 的处理

每块独立保存 `lo_e` 和 `step_e`，并仍在每条 16B 签名的最后一个字节存 norm code。使用封存块的 min/max 时，本块范围内没有越界截断；只剩舍入误差。若 log 量化步长为 Δ，未截断的相对 norm 误差上界为 `exp(Δ/2)-1`。

只做 norm 分块、保持 prompt μ 不变，是很小的对照改动，无需均值补偿。但本次诊断没有显示明显召回收益，不能作为主要结论。固定增加 30% margin 也不能保证覆盖未知漂移，而且会增大 Δ。

### 4.3 修改文件与实现状态

| 文件 | 已完成修改 |
|---|---|
| `cache_hub/cometkv_cache.py` | 增加每层、每 row、每块的 mean/lo/step 和有效范围；eviction 先计算本块统计量再编码；增加 query-block 偏移 buffer |
| `library/cometkv/cometkv/src/cometkv_signature.cu` | 扩展算子参数、shape 检查及绑定，接收块尺度和偏移 |
| `library/cometkv/cometkv/src/cometkv_signature_kernel.cuh` | 候选按所属块加载 lo/step，在分数写回前加同尺度均值补偿 |
| `model_hub/llama.py`、`model_hub/qwen.py`、`config/config.py` | 传递 `stats_mode=block/frozen`、`query_aggregation=mean_prob/q_sum`；保留 frozen 作对照 |
| CUDA/runtime 测试 | 覆盖跨块可比性、旧签名仍按旧尺度解码、第一次 96-token eviction、连续多个 128-token eviction、图捕获与 eager 一致 |

规则块可以由 token offset 计算块号，避免增加每 token 的 int32 epoch ID。可预分配元数据 buffer，并在现有非 CUDA graph 的 eviction 步骤更新有效块数，兼容已有 slide 后重捕获的流程。

全量重建对照也已修复：每层单独计数，只在该层写完新 K 后重建该层；int8 先反量化。非零 forward-only EMA 显式报错，周期重建仅允许在 `frozen` 模式启用。

### 4.4 开销不能漏报

块的 FP32 μ、lo、step 共 `(128+2)*4=520 B` / layer / KV head。B=128 时，每个**新增** token 分摊 4.0625 B，约为 16B 签名的 25.4%；这是额外元数据，不能仍声称整体状态只有 16B/token 且统计量是常数大小。prompt 只需一份元数据，故长 prompt、短续写时整体比例更低。

BF16 μ 加 FP32 lo/step 可将其降至 264 B/block，即每新增 token 2.0625 B，但应先验证补偿的数值误差。更大的块需要更多 recent KV 缓冲，或者使用已知统计量编码后续块，不能为了获得 512-token 块均值而提前读取未来 token。

每层每步的额外 query-mean 计算约为 `O(number_of_blocks * d)`；元数据加载、额外 kernel 和实际 TPOT 必须测量，不能只根据运算量断言“零开销”。

相比之下，Llama-3.1-8B 在 batch=1、128K 上的所有 BF16 K 为 8 GiB。CPU offload 路径每次全量重建至少涉及这些 K 的读回，再做中心化、投影和写索引。固定每 128/256 token 重建不适合作为未经测量的默认方案，且会产生明显延迟尖峰。

### 4.5 检索预算核对与建议（补充更正）

先前把当前 CometKV 预算称为完全 prompt-frozen 过于简化。实际存在两层行为：

- `_compute_topk_for_lengths` 根据当前 `visible_lengths` 计算目标 k；`_update_retrieval_plan` 在初始化及窗口 eviction 时更新它。默认不将 preserved tokens 计入 sparse 预算成本时，目标约为 `max(16, floor(rho * visible_length))`，另受候选数量限制。指定正的 `sig_topk` 时则请求固定数量。
- `_allocate_decode_buffers` 的宽度来自 `_static_prompt_fast_topk_capacity_from_tensors`。后者在 prefill 后仅按 `prompt_length + min(max_new_length-1, static_pattern_end, recent_capacity)` 计算容量；默认 recent=32 时只覆盖 prompt+32。
- `_update_retrieval_plan` 再把目标 k 截断到该固定宽度。因此实际检索量通常只比初始值增加很少，随后封顶。GPU token-cache 容量扩大到约 4 倍不等于 selected-index 缓冲区或检索预算扩大。

用原始容量及 retrieval-plan 方法、CPU tensor 做的复现（rho=2%，sink/recent 不占 sparse 预算，关闭 sampled-tail 以显示总 k）：

| Prompt | 新增 token | 目标 k | 实际 k | 实际 k / 可见长度 |
|---:|---:|---:|---:|---:|
| 1,024 | 0 | 20 | 20 | 1.95% |
| 1,024 | 8,193 | 184 | 21 | 0.23% |
| 8,192 | 0 | 163 | 163 | 1.99% |
| 8,192 | 8,193 | 327 | 164 | 1.00% |

结果见 `results/drift_diagnostic_llama31/budget_capacity_probe.json`。`Exact_TopK` 是另一种实现，它在 `_ensure_plan` 显式冻结 `fixed_topk`；不要把两个 backend 的行为混同。

建议首先提供明确的预算策略：固定数量用于受限延迟场景；固定比例随上下文增长用于长生成精度对照。用 `K_max` 表示用户给定的检索/显存/延迟上限，而不是让旧缓冲区宽度隐式决定上限。更新时取当前可见长度、候选数量和 preserved accounting，计算目标 k，再按明确的 `K_max` 限制。

预分配宽度应覆盖规划的最大预算，或在 eviction 等非捕获步骤按容量档位扩容。需同步审计 selected indices、hit mask、concat 缓冲、sampled-tail 状态和 CUDA graph 重捕获。最近窗口长度随 slide 变化；若计入总预算，必须扣除其当时的完整注意力成本。首次实现保持各层统一 k，便于与现有共享 plan 和 graph 兼容。

更进一步可做质量反馈：长度决定基础预算，定期检查遗漏尾部质量及采样方差，质量不足时提高预算档位，持续充足时缓慢降低，设置滞回和 K_max。签名 score 的熵不能当作真实 attention 覆盖率；错误的 score 也可能很尖锐。可对少量尾部 token 做真实 q·k，结合有完整支持且概率已知的采样分布估计尾部 softmax 质量，但这个比值估计有方差和偏差，需要验证与任务误差的关系，不能称作精度保证。

对纯 top-k，尾部质量较大可作为扩容信号；对当前已有 sampled-tail 的方法，尾部质量大不等于输出误差大，还应看重要性权重方差和 attention 输出误差。控制器需要采用截断之前的诊断量，不能把当前 clipped estimator 的权重直接视作无偏覆盖率。如何调整确定性 head 与 sampled tail 的配额，需要单独消融。

回应漂移审稿意见时，优先实现分块统计补偿和明确的长度感知预算，再评估自适应控制器。必须做“冻结/分块统计 × 固定/增长预算”的交叉对照，并报告相同平均 KV 读取量或相同 TPOT 下的比较，避免把增加检索开销的收益归因于统计量更新。

### 4.6 以推理效率为约束的具体控制器设计

以下是待实现、待测量的设计，不是已经测得无开销的功能。基础预算在现有 eviction 边界更新；质量诊断可先每四次 eviction 执行一次，约每 512 个新增 token 一次。在两个边界之间保持同一个 k，正常 decode 不做 CPU 决策。512、监测层和控制阈值均为验证起点。

**长度基线与资源上限。** 记边界时可见长度为 L_b，候选数为 N_b。以 preserved tokens 另计为例，先计算

\[
K_b^0=\min(N_b,K_{max},\max(K_{min},\lfloor\rho L_b\rfloor)),
\qquad K_b=\min(N_b,K_{max},\lfloor\gamma_b K_b^0\rfloor).
\]

第一版可令 γ 只有 1、1.125、1.25 三档，质量反馈仅增加或撤回这部分余量，不低于长度基线。具体档位和 K_max 需要用实际延迟校准，不能保证增加 k 不影响吞吐。若 sink/recent 计入总成本，先从总额度中扣除它们，再分配 head/tail。

**复用真实计算，而不是增加全量 attention。** `_sampled_tail_attention` 已取得采样 KV、proposal probability，`sampled_tail_attention_merge_kernel` 已计算真实 q·k、重要性修正及 weighted V。可只在诊断步的 0/8/16/24 层（默认 sample_stride=8 时的重新采样层）输出诊断量。这些层的 proposal 排除了当前层自己的 head，便于构造明确的 head/tail 划分；不要直接把其他复用上一层样本的层当作相同划分。

推荐用两个半样本的 attention 输出差异作为一个质量代理。令 H 包含本层的确定性 head 和 preserved tokens，主 attention 已给出 o_H 和 LSE_H。把 m 个独立有放回 tail draws 按奇偶分成 A、B，两组分别计算

\[
\hat Z_T^{(a)}=\frac1{m_a}\sum_{j\in a}\frac{e^{\ell_j}}{p_j},\quad
\hat Y_T^{(a)}=\frac1{m_a}\sum_{j\in a}\frac{e^{\ell_j}v_j}{p_j},\quad
\hat o^{(a)}=\frac{Z_Ho_H+\hat Y_T^{(a)}}{Z_H+\hat Z_T^{(a)}}.
\]

用 `||o_A-o_B|| / (||o_A||+||o_B||+epsilon)` 判断估计是否不稳定。两组大小相同时，复用原本 `1/(m*p_j)` 权重必须乘 2，不能直接把一半权重相加当成完整 tail。所有归约使用稳定的 log/exp 缩放。

这可以在同一 fused kernel 内复用已经加载的 V，用分开的累加器完成，不重复从 CPU 读取 KV，不再次做全长 softmax 或 top-k，也不需要把两个完整输出向量写回。当前 kernel 的四路 V 累加已有奇偶结构可供复用，但增加寄存器、shared-memory 归约和同步是否降低 occupancy，仍须测量。

诊断应保留 clipping 之前的权重，并记录 clipping 影响。clip 后的两组都很稳定，并不能排除共同的截断偏差。半样本差异本身也只是稳定性信号：两组可能一起漏掉低 proposal 概率的重要 token，不能拿它充当真实 attention 误差上界。阈值须在独立数据上对照全 attention 输出/任务误差校准。

**没有 sampled-tail 时。** 默认短上下文常有 m=0，不存在可免费复用的样本。性能优先版本可只保留长度基线、禁用质量反馈；如果需要覆盖短 prompt 的长推理，则在诊断步的少数层额外抽取例如 32 个候选，使用已知概率计算真实 q·k/V。可以对整个候选池均匀有放回抽样，并将命中当前 head 的样本 tail 权重设为 0，以避免再次排序/构造精确 tail 列表。主 attention LSE 也只需在这些 eager 诊断步骤请求。小样本很噪，需要多次观测和滞回。

对 Llama-3.1-8B、BF16、8 个 KV heads，4 层各读 32 个 KV token 的有效数据量为 `4*8*32*(2*128*2)=524288 B`，即 0.5 MiB/次；每 512 token 一次时平均约 1 KiB/token。这里只计算有效 KV 字节，不含采样、传输事务、kernel 启动、归约和等待；不能据此直接宣称延迟小于某个百分比。

**控制规则与同步。** 对少数层的诊断结果做小规模聚合，预热阶段保持 γ=1；连续两次超过高阈值提升一档，连续四次低于低阈值撤回一档，中间保持。两个阈值以及连续次数仅是初始候选。各层的聚合不能因跨层平均而淹没局部问题，可以验证每层的高分位、再跨层取最大值。

诊断结果写入固定 GPU 小缓冲，异步复制到 pinned host memory。在后续 eviction 边界只检查 event 是否完成；未完成就沿用旧档位，不为读诊断量执行 `.item()` 或全设备 synchronize。初版接受至少一个边界的反馈延迟。新一轮或明显话题切换可以额外触发诊断，但低频反馈无法保证及时捕捉所有突变。

**保持 CUDA graph 和实际读取量可控。** 初版对一个请求使用统一预算，tail 的 m 在可行的预算范围内保持固定，反馈只增加/减少确定性 head；避免同时引入逐层可变 k 和频繁重建采样状态。预算只在已有 slide 边界改变，该边界原本已经需要更新 range 并重捕获 graph，因此不必额外增加重捕获时机。

预分配最大容量不等于每步按最大容量执行。当前 `_cg_build_concat` 的 gather 传入 `selected_indices.size(1)`，扩容后应明确区分 capacity 与 active k，使 kernel 的有效扫描宽度按 active k 运行。-1 padding 通常不读有效 CPU KV，但仍有扫描/线程开销，不能把它当成完全免费。还需检查 FlashAttention 的固定宽度 split 配置是否使小 k 承担不必要的工作。

**主要成本来自增加预算本身。** BF16 下，全 32 层、8 个 KV heads 每增加 32 个检索位置，若全部 miss，最多新增 `32*8*32*512=4 MiB` 有效 KV 读取/生成 token；相同数量的 tail 样本通常也缺少跨步缓存复用。因此先固定 tail、仅调整 head，并用 K_max 限制增长。实际代价取决于 hit rate、CPU/GPU store、量化与 overlap；没有办法同时保证任意质量需求和固定延迟上限。

实现验收先分开测两件事：固定 k、仅开启诊断时的 TPOT/P99 开销；开启控制后，相同平均 TPOT 或相同 KV 读取量下的质量。可将诊断本身 TPOT 增量小于 1% 作为工程目标，但这是目标而不是当前实测结果。如果达不到目标，应降低诊断频率/监测层数，或使用长度基线版本。

## 5. 已完成的指定模型诊断

环境：`cometkv`，PyTorch `2.8.0+cu128`，Transformers `5.10.2`，RTX 4090，BF16，HF SDPA 全注意力。

- 数学长回答：226-token prompt，生成 2,048 token，达到长度上限；不是已验证答案正确的推理 benchmark。
- 四轮对话：3,124-token 初始 prompt，文档 → 编程 → 数学 → 回忆原文档，总长度 4,407 token。各轮生成 14/384/384/384 token；后三轮达到上限，下一轮前显式结束该回答。
- 取第 0/7/15/23/31 层，各 8 个 KV heads，每新增 128 token 取一次 query。数学 600、对话 400 个 layer/head/query 观察值，它们相互相关，不能视作这么多独立任务。
- 所有方案在同一全注意力轨迹的 post-RoPE Q/K 上比较。统计量只使用当时已封存的历史。为隔离统计量策略，诊断主动固定预算为 `max(16, floor(0.02*prompt_length))`，数学 k=16、对话 k=62；这不是当前运行时“目标动态增长、实际受缓冲区截断”的逐步复现。短 prompt 触发 k=16 下限，因此不能称数学样本始终使用严格 2% 的可见上下文预算。
- 只比较 selector；没有 sampled tail、完整 sparse attention 输出、任务准确率或 TPOT 的端到端对照。

下表的 recall 是所选 token 对精确 GQA group-sum top-k 集合的召回率，先按每个观察值计算，再等权平均。

| 策略 | 数学长回答 recall | 四轮对话 recall |
|---|---:|---:|
| 冻结 prompt 统计量 | 37.41% | 49.63% |
| 当前 forward EMA，α=0.1 | 37.71% | 46.27% |
| 冻结均值，norm margin=0.3 | 37.08% | 49.58% |
| 冻结均值，块内 norm 范围 | 37.01% | 49.64% |
| 分块均值和 norm，不补偿 | 24.21% | 17.20% |
| **分块均值和 norm，正确补偿** | **44.55%** | **53.35%** |
| 探针时刻全量重建的离线对照 | 40.62% | 50.15% |

正确补偿分块方案相较冻结基线提高 **7.15 / 3.72 个百分点**。全量重建这一行是离线、同步、只读已有效 K 的参考计算，不是打开修改前有缺陷的运行时开关取得的结果，也不保证是所有压缩方案的精度上界。

冻结范围下，新块残差 norm 的上界越界率，在 block/layer/head 上平均为数学 **35.06%**、对话 **0.72%**。数学第 15 层平均约 **70.2%**；可见只报告跨层均值会遮蔽局部问题。均值变化 `||μ_block-μ_prompt|| / RMS_prompt_residual` 平均为 **0.883 / 0.660**。

在检索候选池内，对每个真实 query head 的精确 softmax 计算所选 token 的质量，再对 GQA heads 平均，冻结 → 正确补偿分块分别为：数学 **19.75% → 22.86%**，对话 **38.48% → 40.74%**。这是**候选池条件下的 attention mass**，不含 sink/recent 或 sampled tail，不能直接解释成完整 attention 保留率。

离线冻结评分公式已与真实 `_build_prefill_signatures` 加 CUDA `asym_signature_score_into` 对齐验证，随机 BF16 输入的最大绝对差约 `1.37e-4`，在测试容差内。见 `selector_kernel_check.json`。

这些结果支持分块补偿机制；它们也显示“减少 norm 饱和”不自动等于“提升检索召回”。§9 补充了在线生成和开销验证，仍需更长生成、多任务、多投影种子的任务质量验证。

## 6. 为什么不把离线预设均值作为主方案

离线统计均值通常是 calibration，不一定需要训练模型。它可以作为初始先验或消融，但不能保证匹配当前对话的话题、位置和生成阶段。只有“方向”也不足以确定需要相减的均值向量，模长与残差 norm 的分布都需要定义。

特别是当前压缩 post-RoPE K：对别的任务、长度、位置做 pooled calibration 得到的均值，不能直接假定适用于本请求。换到 pre-RoPE 学均值也不是直接替换；旋转后的均值项随位置变化，需要重新推导补偿。

建议把它设为独立对照：用与测试任务隔离的 calibration 数据估计每层每头 μ，在测试时冻结；另测 `μ_init = λ μ_prompt + (1-λ) μ_calib`，λ 在验证集选定。不要用测试集挑方向或阈值。**本次没有实测这条离线 calibration 基线，因此没有依据宣称它一定更差。** 推荐在线分块的理由是机制直接覆盖请求内分布变化，并且已有初步检索证据。

## 7. 论文如何修改

建议修改 §4.2 的统计量与评分定义、§4.4 的增量维护流程、§5 的实验和附录配置，并同步修改“统计状态常数大小”的复杂度表述。

可用下述文字描述已实现的统计量更新机制；任务质量结论仍需独立实验：

> CometKV does not require the key distribution to remain stationary throughout decoding. Once a batch leaves the recent window, we estimate its key center and log-norm quantization range and freeze these statistics together with its signatures. During retrieval, we decode each norm using its batch-specific scale and compensate for the batch center in the query score. This preserves a common scoring target across batches without rebuilding historical signatures or requiring task-specific training.

紧接着给出带 c 的评分公式、额外元数据规模和真实延迟开销。论文原有的“prompt 统计量始终冻结”应描述为 frozen baseline，并明确区分原始实验结果和本次新增默认配置的验证结果。

至少补两类实验：

1. 短 prompt + 长生成，覆盖 2K/8K/16K 生成；另做 8K/32K/64K prompt 以区分上下文长度和生成长度。按任务报告推理正确率或可执行检查结果，避免将强行忽略 EOS 的重复生成算作长推理。
2. 4/8/16 轮对话，包括同话题延伸、突然切换、返回旧事实、长用户输入；明确测试的是重 prefill 还是持续复用 KV。当前需要实现并验证增量用户输入接口后，才能宣称支持后者。

消融：Full attention、冻结基线、norm-only、现有 EMA、正确的周期重建、分块补偿、离线 calibration；检索预算和 sampled-tail 参数一致。先隔离 selector，再恢复默认 sampled-tail 评测完整方法。

报告随 token/轮次变化的均值位移、norm 上下饱和率、分层 top-k recall、attention mass、任务分数、GPU/CPU 内存、平均 TPOT 和 P95/P99 eviction 延迟。对块大小、校准集、投影种子使用独立验证集，最终任务结果报告多样本波动或置信区间。

## 8. 复现与产物

从仓库根目录运行：

```bash
conda activate cometkv
python scripts/analyze_cometkv_drift.py \
  --model /data/zjx/data-old/model/Llama-3.1-8B-Instruct \
  --output results/drift_diagnostic_llama31
```

已有 token 轨迹时，可追加 `--reuse-trajectories`，仅重放并分析，不重新生成。该选项要求模型与保存轨迹时一致。

结果包括 `metadata.json`、`summary.json`、逐观察值的 `selectors.jsonl` 和 `drift.jsonl`、文本与 token 轨迹。安装了 matplotlib 时脚本也会生成 PNG/PDF；当前 `cometkv` 环境未安装 matplotlib，因此本次以数值表为准。所有生成结果位于 `results/`，不应提交模型输出到仓库。

## 9. 已完成的分块补偿与 GQA 打分

### 9.1 实际评分和更新时机

每个 layer / batch item / KV head 保存一份 prompt 统计量。默认滑动步长 128、overlap 32，因此第一次封存 96 个新增 token，后续每次封存 128 个。新块只使用该批已经生成的 K，计算均值和残差 log-norm 的范围；历史签名及其统计量不重写。尚未封存的 token 继续由 recent 窗口进行精确注意力。

第 e 块保存 `delta_mu_e = mu_e - mu_prompt`、`lo_e`、`step_e`。设 P 有 m=120 个正交单位行，d=128，c≈8.47939，则逐查询头的近似 logit 为：

\[
\widehat\ell_{h,i}
=\frac{\widehat r_i(Pq_h)^\top\operatorname{sign}(P(k_i-\mu_e))/c
+q_h^\top(\mu_e-\mu_0)}{\sqrt d},\qquad
\widehat r_i=\exp(lo_e+code_i\,step_e).
\]

这近似同一个 `q_h·(k_i-mu_prompt)/sqrt(d)`。省略的 `q_h·mu_prompt` 对该头全部候选相同，不影响每头 softmax。不同块均值补偿必须先于概率归一化，且不能漏掉残差签名分数与补偿项之间的 c 比例。

`mean_prob` 的总分为：

\[
S_i=\frac1G\sum_{h=1}^{G}
\frac{\exp(\widehat\ell_{h,i})}
{\sum_{j\in\mathcal C}\exp(\widehat\ell_{h,j})},
\qquad \mathcal I=\operatorname{TopK}_i S_i.
\]

实现存储 `log(S_i)`，与按总概率选 top-k 等价。每头归一化域是检索候选集合 C，不包含 always-visible sink/recent；因此它不是完整注意力概率。直接对线性分数求和仍等价于 `q_sum`，不能解决查询头相互抵消的问题。单测覆盖了 `q` 与 `-q` 的情况：旧的和查询为零，新概率汇总仍可区分 token。

CUDA 实现将 4 个查询头的签名读取合并，采用分块 log-sum-exp、每头归一化、跨头合并三个 kernel；投影和均值补偿由 GPU 矩阵乘完成。工作区按 device 复用，各层顺序共享。每层每步计算中心补偿约 `O(G*E*d)`，扫描签名约 `O(G*T*m)`，没有新增 CPU 读取分数或逐 token 全量重建。

默认启用原有 sampled-tail 时，top-k head 与采样 tail 仍共同使用原来的检索预算。`mean_prob` 将 `log(S_i)/tau` 用作 tail proposal；其 autoscale 默认关闭。`q_sum` 保留旧 proposal 和 autoscale 默认。此处修改打分及与之匹配的采样分布，没有增加读取的 token 数。

### 9.2 验证结果与质量边界

使用 `cometkv` 环境、RTX 4090、BF16、`/data/zjx/data-old/model/Llama-3.1-8B-Instruct`。

- 112 项 CUDA/runtime 测试通过，其中新增 14 项覆盖跨块虚拟 key 的精确点积参照、不同 GQA 大小、64/128-bit 签名、batch=2、旧块不变、相反查询、CUDA Graph 重放和 int8 全量重建修复。
- 74 项轻量回归通过，包括新选项经基准 worker 子进程传递的检查。
- 227-token prompt + 384-token 在线稀疏生成通过，跨过两次窗口滑动，并验证 CUDA Graph 重捕获；另做 8K/32K prompt 的真实模型测速。

在 §5 相同的两条全注意力轨迹、相同固定诊断 k 上重放，得到下列候选条件注意力质量覆盖率（candidate attention mass）：

| 选择方式 | 数学轨迹 | 多轮轨迹 |
|---|---:|---:|
| 冻结 + q_sum | 19.75% | 38.48% |
| 冻结 + mean_prob | 19.76% | 38.35% |
| 分块补偿 + q_sum | 22.86% | 40.74% |
| 分块补偿 + mean_prob | 22.85% | 40.44% |
| 精确逐头概率汇总 top-k 参照 | 33.33% | 52.17% |

数据：`results/block_gqa_reference/summary.json`、`selectors.jsonl`。分块补偿在两个诊断样本上都有改善；逐 Q 概率汇总相比已补偿的 q_sum 尚无稳定收益，多轮样本略低。它避免了线性查询抵消，但近似签名误差仍然存在，因此不能写成“新聚合必然更准”。上述指标也不是最终任务准确率，不能替代长推理/多轮任务评测。

### 9.3 选择器开销

`scripts/benchmark_cometkv_selector.py` 测量单层投影、打分和 top-k 的 CUDA Graph replay：batch=1、8 KV heads、每组 4 Q heads、d=128、128-bit signature。各模式使用相同随机 Q/K、224 个已封存新增 token、固定 k=floor(0.02*prompt)，禁用 tail 以隔离选择器。每项 7 组，每组 200 次，报告组均值的中位数；不包含 KV gather、attention 和 eviction。

| prompt / k | frozen + q_sum | block + q_sum | block + mean_prob |
|---|---:|---:|---:|
| 8K / 163 | 45.16 μs | 47.30 μs | 63.03 μs |
| 32K / 655 | 50.30 μs | 51.91 μs | 68.24 μs |
| 64K / 1310 | 65.42 μs | 68.01 μs | 91.06 μs |

分块补偿本身约增加 1.6–2.6 μs；逐头概率汇总进一步增加 15.7–23.1 μs。新方案的单层选择器比基线慢约 36%–40%，不能称为选择器零开销。实际 TPOT 还受模型矩阵乘、KV gather 和窗口滑动影响，应独立报告端到端结果。

新增分组工作区在这些长度下分别约 1.07 / 4.08 / 8.09 MiB，按 device 跨层共享，不是每层重复分配。工作区随规划的 signature capacity 增长，块元数据另计。数据见 `results/block_gqa_validation/selector.json`。

另在同一张物理 GPU 1 上顺序运行真实 Llama-3.1-8B：FWE 同一输入截到 8K/32K，batch=1，BF16，生成 384 tokens，CUDA Graph，retrieval_budget=0.02，sink/recent=4/32，preserved 不计入 retrieval 预算，默认 sampled-tail，忽略第一个 decode step。TPOT 含余下的窗口滑动和图重捕获，不是只计稳态 graph replay。

| prompt | frozen + q_sum | block + q_sum | block + mean_prob | 新默认相对冻结基线 |
|---|---:|---:|---:|---:|
| 8K | 20.56 ms | 20.65 ms | 21.01 ms | 约 +2.2% |
| 32K | 23.61 ms | 23.77 ms | 24.08 ms | 约 +2.0% |

新默认另一次独立运行得到 21.02 / 24.06 ms。以上是同一输入的工程测速，尚无多任务延迟分布或置信区间；既包含 selector 变化，也包含选中 token / tail proposal 变化带来的 gather 差异，不能把全部 TPOT 差值都归因于 kernel。8K/32K 峰值分配显存约 15.95/18.77 GiB，受 prefill 峰值支配，两位小数下看不到共享工作区的增量。

最终测量分别保存在 `results/block_gqa_validation/e2e_legacy_final/`、`e2e_block_q_sum/`、`e2e_block_mean_prob_repeat/`，每项含 `command.json`、`metrics.jsonl` 和 CSV；`validation_summary.json` 汇总来源和测试数量。初步测速目录 `e2e_legacy/` 早于一次等价的 q_sum 投影路径优化，因此不作为此表最终基线。

### 9.4 配置与下一阶段

```bash
# 当前默认：分块补偿 + 各 Q 独立打分、概率汇总
python simple_test.py \
  --model_name /data/zjx/data-old/model/Llama-3.1-8B-Instruct \
  --attn_type CometKV --use_cuda_graph --gen_len 384 \
  --cometkv_stats_mode block --cometkv_query_aggregation mean_prob

# 选择器开销复现
python scripts/benchmark_cometkv_selector.py
```

消融使用 `--cometkv_stats_mode frozen --cometkv_query_aggregation q_sum`；只测分块统计使用 `block + q_sum`。相应环境变量是 `COMETKV_STATS_MODE` 和 `COMETKV_QUERY_AGG`，环境变量优先于 CLI。修改 CUDA 后必须重新构建扩展。

预算按本轮要求暂不修改。§4.5 的容量上限仍然存在，质量反馈控制器也未接入。后续先修复长度基准预算及其预分配容量，再在 eviction 边界低频引入质量反馈，单独测量预算调整的收益和开销，避免与本次统计量修复混淆。
