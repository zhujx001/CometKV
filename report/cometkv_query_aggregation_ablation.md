# CometKV：逐 Q 线性求和、q_sum 与 mean_prob 实测

日期：2026-09-29。模型 `/data/zjx/data-old/model/Llama-3.1-8B-Instruct`，环境 `cometkv`，BF16，RTX 4090。

## 1. 当前评分中的线性等价性

对候选 i，令其所属统计块为 b，残差签名向量为 z_i，解码残差 norm 为 r_i。当前经过块均值补偿的单头分数可写为：

\[
s(q_h,i)=\frac{r_i}{c}(Pq_h)^\top z_i+q_h^\top(\mu_b-\mu_0).
\]

候选的 r_i、z_i、mu_b 与查询头无关，P 是所有头共用的线性投影。因此：

\[
\sum_h s(q_h,i)=s\!\left(\sum_hq_h,i\right).
\]

分块均值补偿没有破坏这个等价性。逐头先算线性分数再相加，并不会提供不同于 q_sum 的排序目标，只会增加计算；浮点舍入可能改变分数极接近的边界位置。逐头引入 softmax、ReLU 等非线性再汇总，才可能改变跨头聚合目标；对汇总后的总分做 softmax 仍然不改变 top-k。

真实 Q/K 数值检查使用两条既有全注意力轨迹、5 个观测层、8 个 KV heads，以及投影种子 1234、2025、42。冻结和分块补偿各做一遍，共 6,000 组 top-k 比较，累计 206,400 个选择位置。

- 5,999 组 top-k 集合完全一致。
- 唯一不同组为 seed=42、多轮轨迹、第 15 层、KV head 7、position=4148、k=62，集合中相差一个位置。
- 最大相对 L2 分数误差为 2.90e-7；最大绝对误差为 0.001953125（未经 softmax 的原始签名评分尺度）。

这与代数等价、FP32 运算顺序不同的预期一致。不能承诺两个计算顺序逐位相同。原始记录包含 `linear_max_abs_error`、`linear_relative_l2_error` 和 `linear_topk_overlap`。

### Softmax 放置位置

设 a_hi 为单头候选分数，softmax 沿候选 token 维度计算。需要区分：

| 流程 | 对应设计 |
|---|---|
| 先跨头求和，再 softmax，再统一 top-k | 与 q_sum 的 top-k 相同 |
| 每个 Q 各自 softmax，再跨头求和/平均，再统一 top-k | 已实现并实测的 mean_prob |
| 每个 Q 各自 softmax、各取 top-k，最后合并 | 各头内 softmax 不改排序；差别来自配额和合并规则 |

第一种情况下，`exp(S_i/tau)` 在 tau>0 时严格单调，且所有候选共用归一化分母，因此 `TopK(softmax(S/tau)) = TopK(S)`。温度在这里也不会改变固定 k 的选择。直接物化低精度概率可能出现下溢和额外并列，因此应直接在 logits 上选 top-k。

第二种情况下，每个头有自己的归一化分母，先归一化再相加与先相加再归一化一般不同。mean_prob 中使用平均而不是求和只是乘了固定的 1/G，不改变 top-k；实现存储其对数也不改变排序。

第三种情况需要独立的配额设计。若每头各取 k 再求并集，总读取量可能达到 G*k；不能把增大总预算的收益归因于 softmax。

补充数值核对：在同模型两条轨迹、5 层、seed=1234 的 1000 组实际 Q/K 评分上，汇总 logits 与其 FP32 softmax 得到的 top-k 集合 1000/1000 一致；逐头概率求和与平均、概率与 log 概率的 top-k 也均为 1000/1000 一致。数据见 `results/aggregation_ablation/softmax_placement_check.json`。

## 2. 三个投影种子的离线检索对照

使用相同完整注意力轨迹、相同候选和固定诊断 k，全部采用分块统计补偿，仅改变查询聚合。数学轨迹 k=16，多轮轨迹 k=62。指标为所选 token 在候选池内的真实逐头注意力概率之和，再对头/层/观测点取平均；不是任务正确率。

| 候选注意力质量覆盖率 | q_sum | mean_prob | 差值 |
|---|---:|---:|---:|
| 数学轨迹，3 个种子均值 | 22.7601% | 22.7635% | +0.0034 个百分点 |
| 多轮轨迹，3 个种子均值 | 40.6769% | 40.4190% | -0.2580 个百分点 |

数学轨迹有的种子升、有的降；多轮轨迹三个种子均小幅下降。三种投影并不是三条独立任务轨迹，不能用增加投影种子数替代增加样本数。

用精确 K 计算时，按真实逐头概率选 top-k 的覆盖率为 33.33% / 52.17%，高于精确 q_sum top-k 的 32.14% / 50.95%。但改为压缩签名估计后，mean_prob 没有实现这个理论目标的预期收益。这说明“对真实概率求和是覆盖率最优”不能推出“对含近似误差的 logits 做 softmax 就一定更好”。尚未单独量化误差来源，不能把下降确定归因于某一个环节。

数据：`results/aggregation_ablation/selector_seed{1234,2025,42}/`，汇总 `selector_multiseed_summary.json`。这些是重放诊断，非持续复用 KV 的在线多轮服务。

## 3. 在线任务对照协议

可复现脚本：`scripts/evaluate_cometkv_aggregation.py`。每个问题分别运行 `q_sum`、`mean_prob` 和 `Full_Flash_Attn`，共享相同 prompt 和停止条件。两种稀疏方案固定 `stats_mode=block`，不使用 ReLU。

- LongBench：hotpotqa、multifieldqa_en、passage_retrieval_en 各 32 题；按照种子 20250929 从完整本地数据中随机选取，所有模式共用同一子集。
- GSM8K：从官方测试集 1319 题中按同一种子随机选 16 题，要求逐步推理，最后输出 `#### 数值`；最大生成 1024 tokens，遇 EOS 正常结束。
- 补充 GovReport：随机选 12 题，最大生成 512 tokens，用仓库 LongBench ROUGE-L 指标比较持续生成质量。
- 最大上下文配置 16384 tokens，沿用仓库 LongBench 中间截断和 chat template。数据来源、SHA256、样本索引、实际输入 token 全部保存。
- batch=1，greedy，2% 检索预算，最小 k=16，sink/recent=4/32，不计入检索预算。GSM8K 短 prompt 主要触发 k=16 下限，不能将其描述为始终严格 2%。
- 主实验 `sample_frac=0`，隔离 top-k。稀疏生成启用 CUDA Graph；完整注意力路径为 eager，故两者的运行时间不能直接用作后端加速比。
- 原有预算容量约束保持不变，逐样本保存检索容量和最终 k。较长生成仍受 prompt 附近容量限制；两个聚合模式使用同一个容量规则。

主实验输出位于 `results/aggregation_ablation/`，摘要补充实验位于 `results/aggregation_ablation_summary/`。每个输出保存预测文本、token IDs、标准答案、逐样本评分、长度、截断状态、预算和计时。

长上下文短答案任务通常只解码几个 token，第一个 token 又由相同的全注意力 prefill 得到，因此这类任务对 decode selector 的敏感性有限。GSM8K 和 GovReport 补充覆盖多次窗口滑动，但此规模仍不足以证明广泛的长推理稳定性。

主实验先完成的 112 题结果如下，分数越高越好；配对区间通过对问题重采样 10,000 次计算，仅反映这个小样本的经验波动。

| 任务 / 指标 | n | full | block + q_sum | block + mean_prob | mean_prob − q_sum |
|---|---:|---:|---:|---:|---:|
| HotpotQA / F1 | 32 | 51.35 | 53.14 | 53.14 | 0.00 |
| MultiFieldQA-en / F1 | 32 | 56.07 | 58.02 | 57.77 | -0.25 |
| Passage Retrieval-en / 检索分数 | 32 | 96.88 | 100.00 | 100.00 | 0.00 |
| GSM8K / 数值准确率 | 16 | 87.50 | 93.75 | 87.50 | -6.25 |

HotpotQA 和 Passage Retrieval 的两个聚合模式连输出 token 都完全相同。MultiFieldQA 的逐题分数为 mean_prob 3 胜、2 负、27 平，平均差 -0.2455 个百分点，95% 配对 bootstrap 区间 [-0.8599, +0.1638]。GSM8K 为 0 胜、1 负、15 平，差值区间 [-18.75, 0.00] 个百分点。全部相同的子集会产生 [0,0] 经验区间，这不代表总体分布必然等价。

16 道 GSM8K 均正常结束，没有撞到 1024-token 上限；每种模式有 15 道生成超过 128 tokens。q_sum 与 mean_prob 平均生成长度分别为 193.44 / 193.19，最长分别为 440 / 370。唯一正确性差异为测试集第 409 题：q_sum 正确计算两个月共 $168,000，mean_prob 和 full 都漏乘每月 4 周，得到 $42,000。这是一次完整生成轨迹差异，不能单凭此题推断某种选择器普遍更强。

两个聚合模式逐题预分配检索容量完全一致，sampled-tail 均为 0。完整逐题结果及配对统计见 `results/aggregation_ablation/summary_pure_topk.json` 和对应 `predictions_pure_topk_shard*.jsonl`。

补充的 12 篇 GovReport 均生成超过 128 tokens，结果为：

| 指标 | full | block + q_sum | block + mean_prob |
|---|---:|---:|---:|
| ROUGE-L | 34.37 | 34.32 | 33.44 |
| 平均生成 tokens | 466.17 | 469.92 | 449.58 |
| 到达 512-token 上限的样本数 | 4 | 3 | 4 |

逐题 mean_prob 7 胜、5 负，但少数下降较大，平均差为 -0.8835 个百分点，95% 配对 bootstrap 区间 [-3.1034, +1.1613]。各模式按照相同 512-token 上限评分，包括达到上限的输出，不事后丢弃这些样本。检索容量仍逐题一致。数据见 `results/aggregation_ablation_summary/summary_pure_topk.json`。

### 默认 sampled-tail 配置复核

从上述已固定的问题中选出全部 38 个输入长度至少 13000 tokens 的问题，使 2% 预算下默认 `sample_frac=0.25` 能达到 64-token 最小 tail。每题仅重跑两个稀疏聚合模式，共 76 次生成。没有根据预测结果选择样本；长输入条件也使该子集的难度分布不同于全部主实验，不能直接跨表比较均值。

| 任务 / 原任务指标 | n | q_sum + tail | mean_prob + tail |
|---|---:|---:|---:|
| HotpotQA | 22 | 47.93 | 46.66 |
| MultiFieldQA-en | 3 | 70.42 | 68.23 |
| Passage Retrieval-en | 12 | 91.67 | 100.00 |
| GovReport | 1 | 32.27 | 37.84 |

所有运行的实际 tail 大小为 65–82，均已启用；head + tail 不超过原检索容量。在 tail 开启的条件下，mean_prob 相比 q_sum 的结果是混合的：检索多答对一题，问答小幅下降，单篇摘要上升；后两项小样本尤其不能作为普遍结论。这不是 tail ON 相比 OFF 的对照。

这项复核同时改变 top-k 打分和 tail proposal：q_sum 使用原有 autoscale，mean_prob 使用原生 log mixture probability。因此它评估的是各聚合模式的默认完整配置，不能把收益或损失单独归因于 top-k。数据见 `results/aggregation_ablation_tail/summary_default_tail.json` 和逐题输出。

后续已将该 38 题与同题、同聚合模式的 tail OFF 输出配对，见 [尾部采样分析](cometkv_tail_sampling_analysis.md)。q_sum 的 ON−OFF 为 HotpotQA +1.27、MultiFieldQA −2.17、Passage Retrieval −8.33、GovReport −1.13 分；尚无稳定改善。另发现当时的跨层共享采样存在 head/tail 支持集不一致，并用实际 CUDA merge 内核复现，详见该文 §3。后续已完成 [独立 tail 配额与预算修复](cometkv_tail_budget_fix.md)，本表仍为旧版历史结果，不能当作修复后的质量评测。重跑需使用新的输出 tag。

## 4. 推理开销对照

此前同机单层选择器测量包含投影、签名评分和 top-k，固定随机 Q/K、k 和 CUDA Graph replay；两种方案均使用分块补偿，关闭 sampled-tail：

| prompt / k | block + q_sum | block + mean_prob |
|---|---:|---:|
| 8K / 163 | 47.30 μs | 63.03 μs |
| 32K / 655 | 51.91 μs | 68.24 μs |
| 64K / 1310 | 68.01 μs | 91.06 μs |

此前默认 sampled-tail 配置下，真实模型生成 384 tokens 的端到端 TPOT：8K 为 20.65 → 21.01 ms，32K 为 23.77 → 24.08 ms。该测量含窗口滑动和重捕获，数据见 `results/block_gqa_validation/e2e_block_q_sum/` 和 `e2e_block_mean_prob_repeat/`。它与主实验关闭 tail 的质量协议不同，单独列出。

## 5. 当前结论与复现

1. 逐 Q 线性求和与 q_sum 在当前评分函数下代数等价，单独实现更昂贵的逐头线性路径没有新的排序目标。
2. mean_prob 是不同的评分目标，但没有观察到稳定的任务质量增益。主实验的几个差异区间均包含零；默认 sampled-tail 的复核也有升有降。不能声称 mean_prob 普遍更差，也不能宣称它已经提升准确率。
3. 考虑已测得的归一化开销，当前建议以 `block + q_sum` 作为高效基线，`mean_prob` 保留作实验选项。分块统计补偿的收益应与查询聚合方式分开评估。
4. 本轮新增评测和报告，没有再次改变生产默认参数。若采用上述基线，明确传入 `--cometkv_stats_mode block --cometkv_query_aggregation q_sum`。

本轮合计 124 个不同问题，主对照 372 次生成，默认 tail 复核 76 次生成，总计 448 次。它们属于探索性子集测试，不是完整 LongBench/GSM8K 成绩；同一模型贪心输出、有限上下文长度和单一在线投影种子限制了外推范围。

从仓库根目录复现主实验（本次下载的 GSM8K 文件已保存在默认 output 目录）：

```bash
conda activate cometkv
python scripts/evaluate_cometkv_aggregation.py prepare
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_cometkv_aggregation.py run --shard 0 --shards 2
CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_cometkv_aggregation.py run --shard 1 --shards 2
python scripts/evaluate_cometkv_aggregation.py summarize

python scripts/evaluate_cometkv_aggregation.py prepare \
  --output results/aggregation_ablation_summary --tasks gov_report --samples 12 --gsm-samples 0
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_cometkv_aggregation.py run \
  --output results/aggregation_ablation_summary --shard 0 --shards 2
CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_cometkv_aggregation.py run \
  --output results/aggregation_ablation_summary --shard 1 --shards 2
python scripts/evaluate_cometkv_aggregation.py summarize --output results/aggregation_ablation_summary

# 使用本次已经固定的 38 题长输入清单，复核默认 tail。
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_cometkv_aggregation.py run \
  --output results/aggregation_ablation_tail --sample-frac 0.25 --tag default_tail \
  --modes q_sum mean_prob --shard 0 --shards 2
CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_cometkv_aggregation.py run \
  --output results/aggregation_ablation_tail --sample-frac 0.25 --tag default_tail \
  --modes q_sum mean_prob --shard 1 --shards 2
python scripts/evaluate_cometkv_aggregation.py summarize \
  --output results/aggregation_ablation_tail --tag default_tail
```

脚本支持按同一个 shard 输出续跑；不同 shard 划分重跑时应使用新的 tag，避免同一问题重复计入。GSM8K 原始来源为 `https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl`；文件 SHA256 为 `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`。输入和协议 manifest 记录其他数据的 SHA256 与精确索引。
