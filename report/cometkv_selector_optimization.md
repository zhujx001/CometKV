# 精确 Top-K、查表打分与 Tensor Core 实验

日期：2026-09-30。环境为 `cometkv`，Python 3.10、PyTorch 2.8.0+cu128、CUDA 12.4 编译、RTX 4090 24 GiB。模型为 `/data/zjx/data-old/model/Llama-3.1-8B-Instruct`。

本次在已经优化的尾部 merge 基础上，新增 **6-kernel 精确 Top-K** 和 **q_sum 的 FP32 四位查表打分**。Tensor Core 实际实现并测量了 BF16 和 TF32x3 两种原型，当前形状下未取得收益，因此只保留为实验脚本。

## 实现与预算约束

精确 Top-K 将 FP32 分数转换成保序整数键，先分块统计高位直方图和键范围，再确定包含第 K 项的边界桶。若分数相同，直接按 token ID 输出前 K 项；若键范围较窄，减去最小键并自适应选择第二轮的 radix 位，避免近似相等的分数挤入同一大桶。其他范围按两轮高位直方图筛选。确定高于边界的 token 直接进入结果，边界桶完整收集后继续按剩余分数位及 token ID 精确筛选。它没有按局部固定数量截断候选，边界桶缓冲可容纳整个候选范围。

输出直接写入现有 int32 head 缓冲，不再产生 int64 Top-K 索引并转换、复制。最终按 token ID 升序排列，等分时较小 token ID 优先；NaN 按最大值处理，正负零属于相同分数。独立稳定排序 oracle 验证准确选出 K 个不同位置，且不写有效长度以外的容量槽位。

默认策略如下：

| 设置 | 默认行为 | 对照选项 |
|---|---|---|
| `COMETKV_SCORE_IMPL=auto` | block + q_sum 使用查表；mean_prob 和 frozen/q_sum 使用原打分 | `scalar` / `lookup` |
| `COMETKV_TOPK_IMPL=auto` | 候选数 ≥4096、K≤4096、K≤候选数/4、分数与输出同设备时使用新核 | `torch` / `radix` |

`radix` 可强制用于较短、较密的范围；K>4096 或跨设备输出仍回退到原 `torch.topk`。**4096 是新核的适用边界，不是检索预算上限。** K=4100 的运行时回归验证了完整预算保留。原来的长度更新机制、额外 tail=256 和采样关闭入口继续生效。

四位查表按当前 query 预计算每个 nibble 的 16 种 FP32 加权和，将每 token 的 120 项逐位计算改为 30 次查表累加。索引仍为 16 字节，不做永久解包。norm 解码、块中心补偿及 mean_prob 的归一化公式保留。查表改变 FP32 加法顺序，因此数学公式相同，但不承诺逐位相同。

Top-K 临时存储在同一设备各层之间复用；32K、8 个 KV heads 下，候选索引与直方图约 1.28 MiB，另有每行 7 个 int32 状态。直方图每个 1024-token 块占 288 个 int32，含范围统计及对齐填充。不会为 32 个模型层重复保存这份 workspace。

## Tensor Core 实测

Top-K 的主要操作是比较、计数、扫描和筛选；本次用专用 CUDA 核实现。Tensor Core 实验针对其前面的签名打分矩阵乘法。当前每组只有 1 或 4 个 Q，原型需要补齐到 16 列，同时承担签名解包的开销。

实验使用 8 个 KV heads、120 个有效签名位和 FP32 query projection。合成 fixture 直接生成模型所用布局的随机签名和投影，以独立测量打分；包含三个统计块的 norm/center 补偿。group=1 对应 q_sum 的打分形状，group=4 包含各 Q 的归一化与概率聚合。Top-K 重合率使用相同输入的原标量打分作为参照，不是任务准确率。

Tensor Core 原型在 tile 内解包压缩签名，不存储展开后的全局矩阵。测试 token tile=32/64/128/256、4/8 warps；BF16 最好的已测配置是 128-token tile、4 warps。在同一物理 GPU 0 上复测：

| 32K 打分后端 | q_sum 形状 | mean_prob，4 Q | 与标量 Top-K 重合率 |
|---|---:|---:|---:|
| 原标量 CUDA | 9.19 μs | 21.69 μs | 100% |
| 四位 FP32 查表 | **6.25 μs** | 22.94 μs | 100% |
| BF16 Tensor Core，tile=128 | 21.13 μs | 33.27 μs | 99.866% |
| TF32x3 Tensor Core，tile=32 | 96.23 μs | 136.28 μs | 100% |

TF32x3 的 128-token tile 需要 128 KiB 共享内存，超过本机每 CTA 约 99 KiB 的上限；已记录资源限制。32-token tile 可以运行，速度如表所示。BF16 输入量化会改变少量边界位置，q_sum fixture 最大分数差约 1.075；查表的最大差约 0.000305。mean_prob 查表略慢，默认不启用它。

NCU 的 `tensor_score` 实际出现 Tensor Pipe 活动，`sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` 为 12.17%，每线程使用 119 个寄存器。这确认原型确实执行了 Tensor Core 运算；小查询矩阵、解包和资源占用使当前方案无法兑现峰值算力。该结论限于这些实现、形状和硬件，不表示所有 Tensor Core 方案都不可能更快。

## 精确 Top-K 与 NCU

物理 GPU 1，32K、8 行、K=655、FP32 分数，包含输出 int32 的开销。CUDA Graph 预热后，每组 200 次 replay、7 组取中位数：

| 分数分布 | torch Top-K + 索引转换 | 新精确 Top-K |
|---|---:|---:|
| 正态分布 | 36.33 μs | 19.40 μs |
| mean-prob 风格 log 概率 | 36.39 μs | 19.91 μs |
| 全等分数 | 36.30 μs | 7.92 μs |
| −8 附近的窄分布（跨指数边界） | 36.47 μs | 21.31 μs |

NCU 在 GPU 1 上以 `--set full --clock-control none --cache-control none` 采集正态分布：

| NCU 指标 | 原 torch 路径 | 新精确 Top-K |
|---|---:|---:|
| kernel 数量，含转换/复制 | 17 | **6** |
| 累计 kernel duration | 41.89 μs | **20.96 μs** |
| 累计执行指令数 | 4,590,841 | **1,606,467** |

q_sum 打分的独立 NCU duration 为 9.63 → 6.66 μs，执行指令数为 6,599,320 → 4,353,984。NCU 累计 kernel 时间与 CUDA Event 的 graph wall time 分别报告，不混用为同一指标；未锁频且保留热缓存。

## 完整 selector 和 attention

baseline 显式使用 `COMETKV_SCORE_IMPL=scalar COMETKV_TOPK_IMPL=torch`；优化版两项均为 `auto`。均保留上一轮优化过的尾部 merge。物理 GPU 0、batch=1、BF16、8 层、head 比例 2%、额外 tail=256、block statistics、固定合成 query、CUDA Graph；7 组 ×100 次 replay 的中位数：

| 路径 | 上下文 | 聚合 | 优化前 | 优化后 | 速度比 |
|---|---:|---|---:|---:|---:|
| selector | 8K | q_sum | 371.50 μs | 206.53 μs | 1.80× |
| selector | 8K | mean_prob | 468.03 μs | 295.34 μs | 1.58× |
| selector | 16K | q_sum | 566.03 μs | 220.84 μs | 2.56× |
| selector | 16K | mean_prob | 709.95 μs | 334.28 μs | 2.12× |
| selector | 32K | q_sum | 403.71 μs | 250.96 μs | 1.61× |
| selector | 32K | mean_prob | 535.99 μs | 396.83 μs | 1.35× |
| selector | 64K | q_sum | 540.99 μs | 323.54 μs | 1.67× |
| selector | 64K | mean_prob | 744.53 μs | 576.70 μs | 1.29× |
| attention | 32K | q_sum | 1171.03 μs | 1015.31 μs | 1.15× |
| attention | 32K | mean_prob | 1281.76 μs | 1135.41 μs | 1.13× |

selector 包含 query 投影、补偿打分、GQA 聚合和 Top-K。完整 attention 还包含 head gather、FlashAttention、采样与尾部 KV 读取、merge；其 head cache 已预热。16K 的原 torch Top-K 比 32K 慢，是这些 shape 下实测到的非单调表现，已保留原始数据。

## 真实模型解码

物理 GPU 1、Llama-3.1-8B-Instruct、BF16、batch=1、16K/32K 的相同 FWE 截断输入、每次生成 384 tokens、CUDA Graph。前后各重复两次，并在第二轮反转测量顺序。使用同一编译扩展，通过上述两个环境设置选择前后实现。

| 聚合 | 输入 | 原 selector TPOT | 新 selector TPOT | 延迟减少 | 吞吐增加 |
|---|---:|---:|---:|---:|---:|
| q_sum | 16K | 24.35 ms | 22.82 ms | 6.29% | 6.71% |
| q_sum | 32K | 24.17 ms | 23.29 ms | 3.62% | 3.75% |
| mean_prob | 16K | 24.54 ms | 23.01 ms | 6.25% | 6.67% |
| mean_prob | 32K | 24.33 ms | 23.56 ms | 3.16% | 3.26% |

预算逐条验证：16K/32K 最后提交的 head 分别为 332/660，额外 tail=256，总检索槽位为 588/916。检查 `requested_head == actual_head == floor(0.02*budget_plan_visible_length)`，没有靠减少 K 或采样数获得速度。TPOT 忽略第一个 decode step，包含其余窗口滑动和图重捕获；结果是两次运行的工程对照，不是跨任务准确率统计。

## 验证与复现

178 项 CUDA/runtime、78 项轻量测试通过；Compute Sanitizer memcheck 对 36 项 Top-K 和块/GQA 回归检查为 0 errors。新增覆盖稳定平分、正负零、NaN/Inf、连续浮点近邻、跨指数边界的窄分布、全等分数、K=0/1/4096、非整齐长度、范围外 padding、K>4096 回退以及更新 query/noise/head 的 CUDA Graph。图重放还验证了全等与非全等行混合后恢复普通分布。查表也通过多统计块、不同签名宽度及 GQA 组大小的 dense virtual-key oracle。

最终扩展已重新安装到 `cometkv` conda 环境；从仓库外导入后，确认安装文件 SHA-256 与测试和测量使用的本地扩展一致。源码哈希、安装路径和检查结果记录在 `results/selector_optimization/final/source_sha256.json` 与 `validation.json`。

模型任务准确率未重新评测。新 Top-K 对输入分数精确，平分策略和输出排列明确；查表与 attention 的浮点累加顺序可能导致细小差异。Tensor Core 的 BF16 原型只用于实验，没有接入默认推理。

从项目根目录、`cometkv` 环境复现：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_cometkv_score_backends.py \
  --length 32768 --group 1 --tile 128
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_cometkv_topk.py \
  --length 32768 --distribution logprob

CUDA_VISIBLE_DEVICES=0 COMETKV_SCORE_IMPL=scalar COMETKV_TOPK_IMPL=torch \
  python scripts/profile_cometkv_kernels.py --stage selector --length 32768 --aggregation q_sum
CUDA_VISIBLE_DEVICES=0 COMETKV_SCORE_IMPL=auto COMETKV_TOPK_IMPL=auto \
  python scripts/profile_cometkv_kernels.py --stage selector --length 32768 --aggregation q_sum

CUDA_VISIBLE_DEVICES=1 COMETKV_SAMPLE_SIZE=256 \
  COMETKV_SCORE_IMPL=auto COMETKV_TOPK_IMPL=auto \
  python benchmark/ruler/bench_cometkv_fwe_sweep.py \
  --model_path /data/zjx/data-old/model/Llama-3.1-8B-Instruct \
  --lengths 16k,32k --batch_sizes 1 --max_new_length 384 --use_cuda_graph \
  --cometkv_stats_mode block --cometkv_query_aggregation q_sum \
  --output_dir results/selector_optimization/recheck
```

`benchmark_cometkv_score_backends.py --profile --backend tc_bf16 --tile 128 --group 4` 使用 NVTX 范围 `cometkv_score_profile/`；`benchmark_cometkv_topk.py --profile --backend radix` 使用 `cometkv_topk_profile/`。本机 NCU 通过已有 sudo 权限采集硬件计数器。原始命令、重复运行 JSON、`.ncu-rep`、CUPTI trace、源码/扩展基线快照和测试日志位于 `results/selector_optimization/`；本报告的 Top-K、完整 selector/attention、模型结果和最终 radix NCU 使用其中的 `final/` 数据。Tensor Core 和查表打分结果位于父目录，打分实现在这些测量之后未改变。

后续仍优先处理跨层 tail 预取时机：本次改动集中在 selector，原来的整窗读取后等待关系仍需独立调整和实测。
