# CometKV 尾部合并算子：NCU 定位与优化

本报告记录尾部 merge 优化阶段；后续精确 Top-K 与打分优化见 [selector 优化报告](cometkv_selector_optimization.md)，两阶段的性能基线分别保留。

日期：2026-09-30。环境为 `cometkv`（Python 3.10、PyTorch 2.8.0+cu128），使用 CUDA 12.4 编译，Nsight Compute 2024.1 采集。硬件为 RTX 4090 24 GiB；真实模型为 `/data/zjx/data-old/model/Llama-3.1-8B-Instruct`。

本次优化 `sampled_tail_attention_merge`：构造精确的共享内存 head 集合，替代每个采样槽位重复扫描 Top-K；对 head_dim=128 使用向量化 QK/V 读取和跨 warp 的加权求和。32K 单层合并的 CUDA Event 中位耗时从 **82.53 降至 18.44 μs（4.48 倍速度）**。下面分别报告硬件计数器、完整算子路径和真实模型生成的结果。

## NCU 证据与实现

固定形状：batch=1、8 个 KV heads、每组 4 个 Q heads、head_dim=128、BF16、tail=256、head=655。合成 fixture 强制至少 25% 的 draws 命中本层 head，其余位置随机。

NCU 在物理 GPU 1 上采集单次 NVTX 范围中的 merge，前后使用相同参数：`--set full --clock-control none --cache-control none`。下表是实际采集值，不是 NCU 提示的预测加速比。

| NCU 指标 | 优化前 | 优化后 |
|---|---:|---:|
| kernel duration | 82.528 μs | 18.592 μs |
| 执行指令数 | 3,204,188 | 966,550 |
| threads / block | 128 | 256 |
| achieved occupancy | 8.31% | 16.50% |
| achieved active warps / SM | 3.99 | 7.92 |
| registers / thread | 40 | 43 |

原 kernel 只有 32 个 CTA，4090 有 128 个 SM；每个 warp 为一个样本顺序扫描 head，每个 Q head 重复这一过程，随后输出维度对应的线程串行累加全部 256 个 V。低并行度和串行指令/访存等待比峰值带宽更关键。优化后 CTA 数仍为 32，本次收益来自减少工作量和提高 CTA 内并行度。

实现细节：

1. 每个 CTA 用 `atomicCAS` 构造开放寻址哈希集合，容量为不小于 `2*k` 的二次幂，负载率最多 50%。冲突线性探测并比较完整 token ID，排除结果精确；重复 head ID 可以正常插入，head 缓冲区只读取当前有效长度。
2. 每个样本由 lane 0 查表，再广播 membership。`-1` 哨兵样本走精确扫描；head 超过 4096 或共享内存不足时，也回退到扫描。
3. 对齐的 128 维 BF16 K/V 每个 lane 一次读取 4 个元素；Q 保存在寄存器中。8 个 warp 分担样本维度的 V 累加，再在共享内存中归约。head 哈希空间在 membership 完成后复用为部分和，32K 情况动态共享内存约 9.73 KB。
4. 通用维度、非对齐的连续 tensor，以及向量化临时空间不足的情况使用标量路径。临时共享内存按 16 字节对齐，支持奇数采样数。全部样本被排除时直接返回原 main output。

首次仅加入 hash 的中间版本测得 38.86 μs；加入向量读取及跨 warp 求和后达到最终 18.44 μs。重要性权重仍使用原始 draws 数的 `-log(m*p)`，重复 draws 保留次数，clipping 仍只统计有效 tail 槽位。浮点累加顺序改变，因此使用独立 dense oracle 检查 BF16 误差，而不承诺逐位一致。

## CUDA Event 路径测速

物理 GPU 0 上串行测量；每次先预热、捕获 CUDA Graph，再取 7 组、每组 100 次 replay 的中位耗时。前后固定随机种子和相同 shape。merge 为单层；tail / attention 为 8 层，包含 CPU pinned KV 的窗口预取。后两者使用固定合成 query 和已预热的 head token cache。

| 路径 | 上下文 | 聚合 | 优化前 | 优化后 | 速度比 |
|---|---:|---|---:|---:|---:|
| 单层 merge | 8K | — | 46.77 μs | 17.92 μs | 2.61× |
| 单层 merge | 32K | — | 82.53 μs | 18.44 μs | 4.48× |
| 单层 merge | 64K | — | 131.17 μs | 19.57 μs | 6.70× |
| 8 层 tail：proposal + UVA + merge | 32K | q_sum | 1,059.82 μs | 544.18 μs | 1.95× |
| 8 层 tail：proposal + UVA + merge | 32K | mean_prob | 1,114.05 μs | 522.93 μs | 2.13× |
| 8 层完整 sparse attention | 32K | q_sum | 1,781.47 μs | 1,170.63 μs | 1.52× |

单独的 q_sum selector 测得 419.65 → 403.66 μs；它未修改，这个约 4% 的差异视为测量波动，不作为优化收益。merge 耗时仅由 query、采样 KV、head 等输入决定，不依赖产生 query score 的聚合算法。

## 真实模型解码

物理 GPU 1、BF16、batch=1、CUDA Graph，使用相同 FWE 输入按 16K/32K 截断，各生成 384 tokens。block statistics、head 比例 2%、sink/recent=4/32 另计，**tail=256 为额外配额**。各配置前后独立串行运行两次，报告 TPOT 均值；按现有 benchmark 协议忽略第一个 decode step，包含其余窗口更新和图重捕获。该结果是实际模型生成延迟，合成 kernel 的加速倍数不能替代它。

| 聚合 | Prompt | 优化前 TPOT | 优化后 TPOT | 延迟减少 | 吞吐增加 |
|---|---:|---:|---:|---:|---:|
| q_sum | 16K | 25.76 ms | 24.35 ms | 5.49% | 5.81% |
| q_sum | 32K | 26.73 ms | 24.19 ms | 9.52% | 10.53% |
| mean_prob | 16K | 25.98 ms | 24.56 ms | 5.47% | 5.79% |
| mean_prob | 32K | 26.77 ms | 24.42 ms | 8.77% | 9.62% |

16K/32K 的最后提交 head 分别为 332/660，预分配容量分别为 335/663，额外 tail 均为 256，总检索槽位分别为 588/916。全部 16 次真实生成均检查 `requested_head == actual_head == floor(0.02*budget_plan_visible_length)`、`retrieval_slots == head+tail`，前后预算一致。每次运行的原始 TPOT、命令、日志可在下方结果目录查验；两次重复用于工程验证，不构成跨任务统计。

## 数值与内存验证

新增独立 dense attention oracle 回归，覆盖 k=0/1/17/655/1310/4096/4097、重复 head ID、负数哨兵、有效长度之外的 padding、非有限 correction、clip=0/0.5、非对齐连续存储和奇数采样数。已有回归覆盖空 tail、q_sum/mean_prob、CPU/GPU/int8 store、采样开关、独立预算以及 CUDA Graph 更新 query/noise/head。

验证结果：**155 项 CUDA/runtime 测试、78 项轻量测试通过**。CUDA suite 在单卡隔离运行时 153 passed、2 skipped；随后开放两张 GPU，仅补跑被跳过的两项多 GPU 回归，均通过。Compute Sanitizer memcheck 对 18 个新增数值用例检查为 **0 errors**。扩展已重新编译、安装到 `cometkv` 环境；安装版与工作区扩展 SHA-256 一致。

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:library/cometkv python -m pytest -q \
  library/cometkv/test/test_cometkv_*.py
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=.:library/cometkv python -m pytest -q \
  library/cometkv/test/test_cometkv_runtime.py \
  -k 'lockstep_append_when_current_device_differs or guards_kernel_onto_primary_device'
PYTHONPATH=.:library/cometkv python -m pytest -q \
  test/test_*.py benchmark/longbench benchmark/ruler scripts/test_run_cometkv_benchmarks.py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:library/cometkv \
  compute-sanitizer --tool memcheck --error-exitcode 1 python -m pytest -q \
  library/cometkv/test/test_cometkv_tail_budget.py -k large_heads
```

这次是保持计算定义的算子优化；没有用性能结果推断任务准确率。默认 256 draws 的任务质量仍需单独的多任务评测。

## 复现与后续瓶颈

从项目根目录、`cometkv` 环境执行（先清除外部 `COMETKV_SAMPLE_SIZE`、`COMETKV_SAMPLE_FRAC`、`COMETKV_QUERY_AGG`、`COMETKV_STATS_MODE` 覆盖）：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/profile_cometkv_kernels.py \
  --stage merge --length 32768 --samples 256 \
  --output results/ncu_optimization/recheck_merge.json

CUDA_VISIBLE_DEVICES=0 python scripts/profile_cometkv_kernels.py \
  --stage attention --length 32768 --samples 256 --aggregation q_sum

CUDA_VISIBLE_DEVICES=1 ncu --nvtx --nvtx-include cometkv_profile/ \
  --set full --clock-control none --cache-control none \
  -o results/ncu_optimization/recheck_merge \
  python scripts/profile_cometkv_kernels.py --stage merge --length 32768 --profile

CUDA_VISIBLE_DEVICES=1 COMETKV_SAMPLE_SIZE=256 \
  python benchmark/ruler/bench_cometkv_fwe_sweep.py \
  --model_path /data/zjx/data-old/model/Llama-3.1-8B-Instruct \
  --lengths 16k,32k --batch_sizes 1 --max_new_length 384 --use_cuda_graph \
  --cometkv_stats_mode block --cometkv_query_aggregation q_sum \
  --output_dir results/ncu_optimization/recheck_model
```

本机普通用户采集返回 `ERR_NVGPUCTRPERM`，实际使用已有免密 sudo 权限执行 NCU，并显式传入环境 Python 和用户 site-packages 的 `PYTHONPATH`；未调整驱动配置。NCU 未锁频、不清缓存，反映预热后路径，不能外推为冷缓存或其他 GPU 的速度。独立 CUDA Event 测速支持同一收益方向。

原始 `.ncu-rep`、计数器 CSV、前后 CUDA Event JSON、CUPTI trace、真实模型命令和 metrics 都保存在 `results/ncu_optimization/`。源码与扩展的优化前快照保存在该目录的 `baseline_source/`，用于区分本次优化与此前预算/统计修复；生成物不纳入版本控制。

优化后的 CUPTI trace 中，8 次 merge 累计 kernel 时间从 672.33 降至 128.54 μs，而 8 层 UVA gather 仍约 546 μs，已经成为尾部路径的主要开销。CUPTI 单次 trace 有采集扰动，累计 kernel 时间不能替代独立 CUDA Event 的稳态测速。进一步核对依赖关系发现：当前尾部 gather 与首层 merge 串行，尚未形成跨层传输/计算流水线，具体如下。

## 后续核查：预取时机与其他算子

补充采集 `current_attention_trace.json`：32K、q_sum、8 层、tail=256 的完整合成 attention 路径，已预热 head token cache。其单次 CUPTI kernel 累计时长为：

| 路径 | 累计 kernel 时间 | 优化方向 |
|---|---:|---|
| 尾部 8 层 UVA gather | 542.10 μs | 提前发起，分批就绪，隐藏传输等待 |
| PyTorch Top-K 核心及 scan | 267.06 μs / 120 次 launch | 小 batch 专用精确 Top-K，减少扫描、临时索引和启动次数 |
| 签名打分 | 87.93 μs | 融合投影、补偿及统计读取，评估访存/并行度 |
| 主 attention FlashAttention | 76.09 μs | 评估小检索长度下 split 数和合并开销 |
| 主 head gather | 65.82 μs | 此处为 warm cache；实际 miss 传输需模型 trace 另测 |

这是定位开销的合成 trace，不能直接当作模型各部分的耗时占比，也不是这些优化方向已经取得的加速结果。

当前 BF16 CPU 路径在 `_sampled_tail_attention` 中完成 proposal/draws/correction 后，确实一次读取窗口内各层自己的 K/V。但 `_cg_sparse_attention` 先完成本层 head gather 和 FlashAttention，才调用它；而 owner 层随即等待整个窗口的 `ev_done`。该 trace 中窗口 gather 为 352.95–895.05 μs，首层 merge 从 895.41 μs 才开始。独立 stream 并未消除这条串行依赖。

跨层复用的是样本位置和采样分布修正，各层历史 K/V 仍分别读取。历史 KV 已存在，所以样本位置确定后，可以立即预取后续层的对应 KV，无需等待后续层 query。当前 full-support proposal 也不依赖 owner 层 head 的排除结果，因此可在 score 准备好之后，把 proposal/采样移至侧流，与主流 Top-K、head gather 和 FlashAttention 并行。

下一步建议将采样准备与最终 merge 分离，并比较“当前层 + 后续 7 层”和“前 2 层 + 后续 6 层”的预取分批方式，各批使用独立完成事件。首层只等待所在批次，后续批次与前面层的 merge、投影和 MLP 重叠。单个 CUDA gather launch 的完成事件只能表示整次 launch 完成，不能指示其中某一层已经可用，因而只把原来的 8 层 launch 移到侧流不足以实现按层就绪。

最终收益仍取决于可重叠的计算量、head/tail 对 PCIe 的竞争，以及拆分后批量读取的带宽损失；应以 CUDA Graph 中的依赖 trace 和真实模型 TPOT 验证。以上为本次后续核查的设计结论，尚未修改预取调度。
