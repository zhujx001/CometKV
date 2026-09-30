# 独立 tail 配额与检索预算修复

日期：2026-09-30。环境：`cometkv`；模型：`/data/zjx/data-old/model/Llama-3.1-8B-Instruct`。

## 已实现的行为

**head 随可见长度更新，tail 额外计量；跨层共享采样现在覆盖完整候选，并在各层排除本层 head 的贡献。** 这些修复不改变分块均值/norm 补偿或查询聚合算法。配置生成器默认仍为 block + mean_prob；本次性能对照显式指定 q_sum。

### head 预算与容量

默认 head 目标是 `max(16, floor(0.02 * visible_length))`，受可检索候选数限制；若 preserved 计入预算，则先扣除它们的数量。正的 `sig_topk` 仍表示固定数量。新增 `COMETKV_MAX_RETRIEVAL_TOPK` / `sig_max_retrieval_topk` 提供显式硬上限，默认 0 表示无额外上限；硬上限优先于最小值。

selected indices、hit mask 和 CUDA Graph concat 缓冲按实际 prompt 加完整计划生成长度预分配，容量不再只覆盖 prompt+recent。容量估计使用最小 preserved 数量和最大候选数作为保守界。GPU token cache 可以较小，只影响命中率，不再隐式限制 head 数量。

实际 k 仍在初始化和窗口滑动时提交，窗口内保持固定以避免每步图重捕获。默认窗口 128 tokens、2% 比例时，对瞬时长度目标的差距最多约 3 个 token（preserved 另计且未触及其他上限时）；不承诺每个 token 都精确按瞬时比例变化。若意外超过已规划容量，现在明确报错，避免再次静默截断。

CUDA Graph gather 只处理实际 head 数量，索引和输出保留预分配行跨度，避免为了未来容量而在每个步骤处理大量填充槽位。日志分别记录 head 请求量、实际量、容量、tail 数量、总检索槽位和预算最近更新时的可见长度。

回归复现：2% 预算、preserved 另计、tail=64 时：

| Prompt | 新增 token | 修复前 head 上限（tail 关闭） | 修复后实际 head | 额外 tail |
|---:|---:|---:|---:|---:|
| 1,024 | 8,193 | 21 | 184 | 64 |
| 8,192 | 8,193 | 164 | 327 | 64 |

### 独立 tail 配额

配置默认已按用户要求调整为 `sample_size=256`，即每层、每个 KV head 有 256 个额外有放回 draws；一个 GQA 组共享这些位置。这是原始采样槽位数，不是去重后或排除 head 后的有效样本数。若 head 已覆盖全部候选，tail 自动关闭。

```
COMETKV_SAMPLE_SIZE=0    # 纯 head
COMETKV_SAMPLE_SIZE=32   # 额外 32
COMETKV_SAMPLE_SIZE=64   # 额外 64
COMETKV_SAMPLE_SIZE=128  # 额外 128
COMETKV_SAMPLE_SIZE=256  # 额外 256，当前默认
```

显式 SIZE 不受旧 `MIN_M=64` 的启用门槛约束，因此 SIZE=32 有效，但仍受 MAX_M=256 限制。兼容的 FRAC 开关改为从 head k 推导额外样本数，不再扣减 head；SIZE 环境变量优先于 FRAC，FRAC 环境变量优先于配置内固定 SIZE。未设置 SIZE 时，旧脚本的 FRAC=0 仍会关闭 tail。直接调用 cache 构造器时，未给出 SIZE 仍使用其显式 sample_frac 参数（默认 0）；模型配置生成器负责提供默认 256。

BF16 Llama-3.1-8B 上，额外 256 draws 的全部层 KV 有效载荷为 32 MiB/生成 token，不含事务开销；不再宣称总读取预算不变。下方 64 draws 的历史测速仍保留原值，不能当作 256 draws 的测速。

### 跨层共享的支持集与去重计入

重采样层使用整个候选池的 score proposal，并混入 10% uniform：

\[
p(i)=0.9\,\operatorname{softmax}(a)_i+0.1/N.
\]

可用 `COMETKV_SAMPLE_UNIFORM_MIX` 在 `(0,1]` 内调整。uniform 提供 fp32 下的概率下限，pad、sink、recent 等非候选没有采样概率。mean_prob 使用原有 log mixture score；q_sum 默认使用标准化 score。每 8 层构造一次 proposal 并预取对应 KV 的方式保留。

每层 CUDA merge 检查 draws 是否命中该层当前 head，命中者的 tail 权重置零。修正使用原始 `-log(m*p)`，不按幸存样本数重新归一化；draws 的重复次数保留，因此没有把有放回采样误当成无放回。membership 检查融合进 q·k kernel，最初使用 warp 协作扫描当前 head；后续 [NCU 优化](cometkv_ncu_optimization.md) 改为共享内存精确哈希集合，并为大 head 保留扫描回退。该路径没有逐 token CPU 同步或完整候选 mask 清零。

clipping 的均值只计算有效 tail 槽位，全部样本被屏蔽时直接保留 main output。clip=0 和默认 clip=4 都覆盖该分支。无 clipping 时重要性分子/分母有正确支持和分区，但它们的比值仍不保证有限样本无偏；默认 clipping 也仍引入偏差。该修复不等于任务质量保证。

四候选反例使用实际 CUDA kernel 验证：完整 attention 输出为 3，旧的 owner-tail 跨层复用输出为 2；新方案在 head={0} 和 head={1} 时均为 3，clip=0/4 都通过。

## 验证

CUDA 12.4 编译扩展（PyTorch 2.8.0+cu128，RTX 4090），在 cometkv 环境执行：

```bash
CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=4 \
  /data/zjx/miniconda3/envs/cometkv/bin/python setup.py build_ext --inplace
# 上一命令在 library/cometkv 目录执行；下面在项目根目录执行。
PYTHONPATH=.:library/cometkv /data/zjx/miniconda3/envs/cometkv/bin/python -m pytest -q \
  library/cometkv/test/test_cometkv_*.py
PYTHONPATH=.:library/cometkv /data/zjx/miniconda3/envs/cometkv/bin/python -m pytest -q \
  test/test_*.py benchmark/longbench benchmark/ruler scripts/test_run_cometkv_benchmarks.py
```

结果：**136 项 CUDA/runtime 测试、78 项轻量测试通过**。新回归覆盖长生成预算、preserved 两种计量方式、显式 head 上限、0/32/64/128 tail 配额、全 head 覆盖、容量不足显式报错、跨层不同 head、重复采样、全部样本命中 head、clipping、q_sum/mean_prob、CPU/GPU/int8 store，以及 CUDA Graph 在更新 query/noise/head 后的输出。

### 真实模型生成与延迟

在物理 GPU 1、RTX 4090 24 GiB、BF16、batch=1、CUDA Graph 上串行完成 6 次真实生成。使用相同 FWE 输入按长度截断，head 比例 2%、sink/recent=4/32 另计，强制生成指定长度，TPOT 忽略第一个 decode step，并包含其余窗口滑动和图重捕获。

| 聚合 / 输入 / 生成长度 | 最后提交的 head | 额外 tail | 预分配 head 容量 | TPOT |
|---|---:|---:|---:|---:|
| q_sum / 16K / 384 | 332 | 0 | 335 | 21.63 ms |
| q_sum / 16K / 384 | 332 | 64 | 335 | 23.29 ms |
| q_sum / 32K / 384 | 660 | 0 | 663 | 21.27 ms |
| q_sum / 32K / 384 | 660 | 64 | 663 | 23.32 ms |
| q_sum / 1K / 1155 | 43 | 64 | 43 | 21.21 ms |
| mean_prob / 8K / 384 | 168 | 64 | 171 | 22.38 ms |

1K 长生成验证了 head 从初始 20 增长到 43，跨越多个封存块；每次生成都检查了 `requested_head == actual_head == floor(0.02*budget_plan_visible_length)` 和 `retrieval_slots == head+tail`。

额外 64 个 tail 的 16K/32K 延迟增量分别为 **7.67% / 9.62%**。这是每项一次的工程检查，包含生成轨迹、读取和采样计算的变化，不是多任务统计或任务质量评测。raw metrics、命令、环境和日志在 `results/tail_budget_fix/`，统一校验结果为 `model_validation.json`。此前旧实现的 split-tail 测速不能直接作为相同总读取量的公平对照。

## 仍需单独评估

- 新版独立 tail 的任务质量和最佳样本数，需比较 `head K`、`head K + tail m` 以及同槽位数的 `head K+m`。此前从 head 中划分 tail 的评测仍是历史结果，不能直接用于宣称新方案的收益。
- 每个请求仍只按长度更新预算，没有在线质量反馈控制器。
- 长生成会实际增加 head 读取和缓冲容量，这正是按长度预算的行为；需要固定成本时设置显式 head 上限。
