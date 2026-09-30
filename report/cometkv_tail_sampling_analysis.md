# 尾部采样的开启条件、正确性与优化

日期：2026-09-30。模型：`/data/zjx/data-old/model/Llama-3.1-8B-Instruct`；环境：`cometkv`。

**后续实现状态：已完成 [独立 tail 配额与预算修复](cometkv_tail_budget_fix.md)。下文记录修复前的默认值、历史评测和问题复现，不能作为当前运行配置说明。**

**建议先修复跨层共享采样的 head/tail 支持集不一致，再决定是否扩大采样的使用范围。当前证据不支持“上下文足够长就必然应该开启”。** 高效推理的对照配置可用 `block + q_sum + sample_frac=0`；本文没有修改运行默认值，也没有实现自适应采样控制器。

## 1. 当前什么时候开启

采样用于稀疏 decode。`cache_hub/cometkv_cache.py::_update_retrieval_plan` 在初始化及窗口滑动边界分配预算；不是每生成一个 token 就在 CPU 上重新决策。

设实际可用检索预算为 K，候选数为 N。默认配置计算

```
if sample_frac > 0 and K > 1 and N > K:
    m = min(round(0.25 * K), K - 1, 160)
    if m < 64:
        m = 0
else:
    m = 0
head = K - m
```

采样槽位从确定性 top-k 中划出，**不是在 K 之外额外补 m 个 token**。这里的“确定性 head”指用近似签名分数选中的 token，其真实 attention 被计算；并不表示它们是精确 q·k 的 top-k。

在 2% 预算、sink/recent 另计、容量没有进一步限制时，初始化附近的大小约为：

| 输入长度 | K | head | tail m |
|---|---:|---:|---:|
| 8K = 8192 | 163 | 163 | 0 |
| 16K = 16384 | 327 | 245 | 82 |
| 32K = 32768 | 655 | 495 | 160 |
| 64K = 65536 | 1310 | 1150 | 160 |

这是约 **12.8K** 上下文的经验门槛，而非算法的必要长度条件。Python 的 `round` 使 K=254 已可能得到 m=64；实际边界还受 preserved 是否计入预算、候选数、预分配容量及窗口滑动影响。若预算改为 1% 或 4%，对应长度门槛也随之变化。

`COMETKV_SAMPLE_STRIDE=8` 指 **每 8 个模型层重新采样**，不是每 8 个生成 token。每个 decode step 都会更新随机噪声。当前预算容量仍主要依据 prompt 分配，短 prompt 的长生成可能一直保持很小的 K，因此也可能一直不会开启 tail；该问题需要后续预算容量修改，单独调采样阈值不能解决。

## 2. 同题配对：采样是否已经证明有益

复用已完成的 448 次生成，不增加质量评测次数。将长输入子集的 38 题、76 个 tail ON 预测，与同题、同聚合方式的 tail OFF 预测配对。检查了输入 token IDs、答案、生成上限、prompt 长度和预分配检索容量相同。block 统计、2% 预算、签名种子 1234 相同；ON 使用默认 frac=0.25、stride=8、clip=4，实际 m=65–82。

### q_sum

| 任务 / 原任务指标 | n | tail OFF | tail ON | ON − OFF |
|---|---:|---:|---:|---:|
| HotpotQA F1 | 22 | 46.66 | 47.93 | +1.27 |
| MultiFieldQA-en F1 | 3 | 72.59 | 70.42 | −2.17 |
| Passage Retrieval-en | 12 | 100.00 | 91.67 | −8.33 |
| GovReport ROUGE-L | 1 | 33.40 | 32.27 | −1.13 |

HotpotQA 是 2 题提高、20 题持平；检索是 1 题从正确变成错误。不能把该结果推广为所有检索或摘要任务的结论，也不能把退化直接归因于下面的支持集问题，因为这里还同时存在 head 缩小、随机采样及 clipping。

### mean_prob

| 任务 / 原任务指标 | n | tail OFF | tail ON | ON − OFF |
|---|---:|---:|---:|---:|
| HotpotQA F1 | 22 | 46.66 | 46.66 | 0.00 |
| MultiFieldQA-en F1 | 3 | 72.59 | 68.23 | −4.36 |
| Passage Retrieval-en | 12 | 100.00 | 100.00 | 0.00 |
| GovReport ROUGE-L | 1 | 26.98 | 37.84 | +10.86 |

所有 n>1 的 95% 配对 bootstrap 区间都包含零；n=1 不计算区间。这个区间仅重采样问题，**不包含在线采样种子间的波动**。观察到全持平所得到的 [0,0] 不能解释成总体效果已精确确定。

结论：长输入只是允许分配足够多样本，不能证明采样收益；目前不能统一主张开启优于关闭。逐题结果、源文件哈希和区间见 `results/tail_analysis/paired_quality.json`。

## 3. 优先修复：跨层共享时支持集不一致

实现当前只在重采样层 a 排除该层的 head H_a，产生支持集 C\\H_a 上的 proposal p_a，然后连续多层共享 draws 和 `-log(m*p_a)`。后续层 l 会重新选择自己的 H_l，但合并 CUDA kernel 没有接收 H_l，也没有检查样本是否已经在主 attention 中。

即使关掉 clip，并令样本数趋向无穷，估计的分母仍是

\[
Z_{H_l}+Z_{C\setminus H_a}
=Z_C+Z_{H_l\setminus H_a}-Z_{H_a\setminus H_l}.
\]

分子有同样的重复/遗漏。preserved token 在主分支中精确计算，不影响此处针对候选池 C 的问题。只有当两个 head 相同，或特殊情况下误差恰好抵消，才能恢复目标。

- H_l\\H_a：可能既进入当前 head，又作为 tail 被加权计入。
- H_a\\H_l：既不在当前 head，也没有任何采样概率。
- **知道实际 proposal probability 是必要条件；proposal 覆盖当前整个 tail 且分区不重叠也是必要条件。** 单靠重要性修正无法恢复概率为零的遗漏部分。

已调用实际 `sampled_tail_attention_merge` CUDA 内核复现一个四候选例子：真实 logits 全为 0，V 的第一个分量为 `[8,4,0,0]`，其余分量为 0。采样层 H_a={0}，复用层 H_l={1}。使用 m=96、三个位置各重复 32 次的平衡 draws，去掉 Monte Carlo 波动：

| 情况 | 输出第一个分量 |
|---|---:|
| 完整 attention | 3.0 |
| 采样层：H_a + 从 C\\H_a 采样 | 3.0 |
| 复用层：H_l + 从 C\\H_a 采样，clip=0 | 2.0 |
| 同上，默认 clip=4 | 2.0 |
| 从完整 C 采样并屏蔽当前 H_l 的 tail 贡献，clip=0 | 3.0 |

这证明了结构性问题；它不量化真实模型中的误差大小。结果见 `results/tail_analysis/support_counterexample.json`。已有 sampled-tail runtime 测试使用单层 cache，覆盖了单层 correction/merge，但没有覆盖 head 随复用层变化的情形。

### 适合保留跨层预取的修复方向

建议让共享 proposal 对整个检索候选池有支持，再在每层对当前 head 的样本贡献置零：

\[
p_a(i)=(1-\epsilon)\widetilde p_a(i)+\epsilon/N,\quad i\in C,
\]
\[
\widehat Z_{T_l}=\frac1m\sum_{j=1}^m
\frac{\mathbf1[i_j\notin H_l]\exp(s_l(i_j))}{p_a(i_j)}.
\]

\(\widetilde p_a\) 从完整候选的分数生成。若先将 H_a 概率置零，再混合完整候选上的 uniform，虽也能恢复支持集，但原 head 只能靠 uniform 部分被抽中，可能产生很大的权重方差。uniform mixture 提供概率下限；epsilon 和温度需要实测，本文不预设最佳值。分子对 `exp(s)*V` 做相同修正。无 clipping 且概率正确时，分子/分母分别无偏，但它们的**比值并不具有有限样本无偏保证**。

这样可以继续每 8 层生成一次 proposal、批量预取 8 层 KV，新增的是每层 head membership 检查。可用固定显存中的索引标记或融合检查，避免每层扫描全候选重建 mask。不过命中当前 head 的 draws 会浪费部分读取，必须测有效 tail 样本数和 TPOT；不能简单宣称修复没有成本。

实现时还必须处理：有效 tail 样本为零时返回主 attention；clipping 的均值只归约有效项，避免将被屏蔽的 `-inf` 计入均值；修正分母仍为总 draws 数 m，不能擅自改成未命中 head 的样本数。单独剔除 overlap 而保留旧 proposal 仍会遗漏 H_a\\H_l。

`stride=1` 可以作为分区正确的参考对照，每层都从自己的 tail 重采样。但它增加 proposal 构造次数并失去原有多层预取方式，不宜未经测速就作为高效默认。proposal 过时造成的方差，以及 clipping/比值估计带来的偏差，是分区修复后仍然存在的不同问题。

## 4. 何时值得开启：预算门槛 + 低频质量证据

推荐把“能够开启”和“值得开启”分开实现：

1. **预算允许**：先保留确定性 head；首个实验档可用 m=64 且 m≤K/4，即 K≥256。这里 64 是当前工程配置下的起始实验值，不是统计定理。
2. **尾部对输出有影响**：关注真实 q·k/V 估计的剩余质量及输出变化。签名分数熵、上下文长度或生成步数本身都不能证明有重要信息遗漏。
3. **估计足够稳定**：质量很大但权重极不稳定时，优先增大 head 或改善 proposal；不能仅因为 tail mass 大就把更多 head 换成噪声大的 tail。

为了保持推理效率，可先离线确定任务/长度档的静态开关，再做低频在线诊断。在已有窗口滑动边界每约 512 个生成 token、少数采样层运行一次诊断，CUDA graph 内不做逐 token CPU 同步。开启后使用两组半样本的补偿输出差异和 ESS 作为方差代理；一半样本的修正需按自身样本数重标定。它们不能检测所有共同偏差，也不构成质量保证。

当 m=0 时不存在免费的 tail 反馈：若需要检测是否应开启，必须额外执行少量诊断采样。作为待验证方案，4 层、各 32 draws、8 个 KV heads、d=128、BF16 的有效 KV 字节为 `4*32*8*(2*128*2)=0.5 MiB/次`。每 512 token 一次约为平均 1 KiB/token 的有效载荷，但 kernel、传输事务、等待及 CPU/GPU 同步另计，不能据此宣称延迟开销已验证。

可以尝试 m∈{0,64,96,128,160} 的少数档位，只在原有 slide 边界调整并设置滞回。没有评测之前不写死“最佳”质量阈值。短输出任务几乎等不到周期诊断，应靠初始化决策或离线校准；单针检索的稀有重要 token 也可能被小规模诊断完全漏掉，因此基础 head 预算不能靠诊断信号任意削减。

## 5. 其他效率优化的优先级

### 新增端到端 ON/OFF 测速

同一张物理 GPU 1（RTX 4090 24 GiB），指定 Llama-3.1-8B-Instruct，BF16、batch=1、block+q_sum、2% 检索预算、sink/recent=4/32 另计、token cache=1024、CUDA Graph。同一条 FWE 输入分别截到 16384/32768 tokens；强制生成 384 tokens，每次忽略第一个 decode step，计时包含后续窗口滑动和 graph 重捕获。关闭逐步同步计时。

按 OFF→ON→ON→OFF 顺序串行运行，每种长度、配置各两次，共 8 次模型生成。ON 固定 frac=0.25、stride=8、min=64、max=160、clip=4、autoscale=1、sigma=2、tau=1。均成功完成。两次 TPOT 的平均值：

| 输入 | tail OFF | tail ON | 增加 |
|---|---:|---:|---:|
| 16K | 21.64 ms/token | 23.15 ms/token | +7.01% |
| 32K | 21.30 ms/token | 23.75 ms/token | +11.48% |

这是当前实现的端到端配置比较，包含采样、gather、merge、graph 更新及生成轨迹/缓存访问变化，不能把全部差值当成某一个 kernel 的成本。每项仅两次运行，不构成多任务延迟分布；同 K 也不是相同的 PCIe 读取量。完整命令、环境、日志和原始时延在 `results/tail_analysis/latency/`；汇总为 `results/tail_analysis/latency_summary.json`。

### 优化顺序

- **减少读回和重复计算**：有放回采样会出现重复 token；可以先 gather 唯一位置，再按 multiplicity c 合并权重为 `log(c)-log(m*p_i)`。不能简单去重后继续当成 m 个独立样本。若开启现有 clip，应保持原始 slot 的 clipping 均值和截断语义，不能先加 log(c) 再套同一个 cap；固定形状工作区和去重开销也需评测。
- **融合 proposal 工作**：现在 q_sum 的均值/方差、exp、分块 CDF 和抽样由多次 Torch 操作完成，有进一步融合空间。但先定位端到端瓶颈，不把 selector-only 测速当成 tail 代价。
- **控制实际 PCIe 流量**：tail 使用绕过 token cache 的 gather，同样 K 不代表同样延迟。长期可测试分层/分块分层抽样来改善覆盖与访问局部性，必须使用完整的两级采样概率进行修正，不能无补偿地偏向 recent blocks。
- **预算与 tail 分开调**：先修复检索容量随长度的扩展，再验证 tail 配额。当前不建议同时改成逐层可变 K、动态 stride、动态 m 和新 proposal，否则难以判断收益来源。

## 6. 复现

```bash
/data/zjx/miniconda3/envs/cometkv/bin/python scripts/analyze_cometkv_tail.py
```

该脚本只读取已保存的质量预测，写配对统计，并执行实际 CUDA 合并内核的支持集反例。它不重新跑模型，不修改采样运行配置。运行默认仍为 `block + mean_prob`、sample_frac=0.25；效率基线实验应显式指定 q_sum 和 sample_frac=0。

若输出目录内已有上述四组完整 latency 结果，脚本还会校验并汇总它们。源码语法检查及 `git diff --check` 已通过；本轮只新增分析工具、报告并纠正原采样注释，没有变更运行算法，因此没有重复全部 runtime 回归测试。
