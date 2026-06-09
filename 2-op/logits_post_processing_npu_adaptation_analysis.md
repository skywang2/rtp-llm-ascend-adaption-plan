# Logits 后处理 NPU 适配技术分析报告

## —— xllm vs vllm-ascend 对比分析与 rtp-llm 适配路线图

---

## 一、xllm Logits 后处理 NPU 适配架构

### 1.1 整体架构分层

xllm 采用 **C++ 前端 + 宏条件编译 + 多后端算子库** 的架构，NPU 适配通过 `#if defined(USE_NPU)` 编译宏控制：

```
Sampler::forward()  [xllm/core/framework/sampling/sampler.cpp]
  ├─ apply_frequency_presence_penalties()   → 纯 PyTorch 原语 (gather/scatter)
  ├─ apply_repetition_penalties()           → 纯 PyTorch 原语 (gather/where)
  ├─ apply_top_k_top_p()                    → ACLNN 自定义融合算子 (NPU) / 排序+mask (通用)
  │    └─ xllm::kernel::npu::top_k_top_p()  [xllm/core/kernels/npu/xllm_ops/top_k_top_p.cpp]
  │         └─ aclnnApplyTopKTopP()         ← ACLNN 原生 API
  ├─ torch::softmax()                       → torch_npu 自动映射
  ├─ random_sample() / greedy_sample()      → 纯 PyTorch (multinomial/argmax)
  └─ RejectionSampler::forward()            [xllm/core/framework/sampling/rejection_sampler.cpp]
       ├─ random_sample()                   → 纯 PyTorch 原语
       └─ random_sample_fused()             → 编译宏选择 kernel 后端
```

### 1.2 NPU 算子技术路线

xllm 的后处理 NPU 算子分为 **三条技术路线**：

| 路线 | 算子数 | 接口 | 特点 | 典型算子 |
|------|--------|------|------|----------|
| **ACLNN 原生自定义** | 4个 | `aclnn*` API + 显式 workspace | 性能最优，需 NPU 专属代码 | `top_k_top_p`、`beam_search`、`replace_token`、`rec_constrained_topk_fused` |
| **PyTorch → torch_npu** | 2个 | 纯 ATen 操作 | 跨后端兼容，自动映射 | `rejection_sample`、`rec_constrained_topk` |
| **ATB 框架** | 1个 | `atb_speed::LmHead()` | 图级融合 | `LmHead`（logits 计算） |

### 1.3 Top-K/Top-P 实现细节

核心 NPU 路径位于 [logits_utils.cpp](xllm/core/framework/sampling/logits_utils.cpp#L108-L116)：

```
当 top_k 和 top_p 同时有效且 USE_NPU 时:
  → 将 top_k <= 0 的值替换为 INT64_MAX
  → 调用 xllm::kernel::npu::top_k_top_p(logits, processed_top_k, top_p)
```

底层 [top_k_top_p.cpp](xllm/core/kernels/npu/xllm_ops/top_k_top_p.cpp) 实现：
- 将 PyTorch Tensor 转为 `aclTensor*`
- 调用 `aclnnApplyTopKTopPGetWorkspaceSize()` 获取 workspace
- 调用 `aclnnApplyTopKTopP()` 执行融合计算（排序+top-k+top-p 一步完成）
- 同步等待 `aclrtSynchronizeStream()`

> **注**: 只有 top_k 或只有 top_p 时，走通用的排序+mask 路径（纯 PyTorch），不使用 ACLNN 融合算子。

### 1.4 惩罚项实现

xllm 采用**纯 PyTorch 原语**实现惩罚项，不依赖 CUDA/NPU 自定义 kernel：

- **频率/存在惩罚** ([logits_utils.cpp:24-35](xllm/core/framework/sampling/logits_utils.cpp#L24-L35))：`gather → sub_ (scatter_)` 
- **重复惩罚** ([logits_utils.cpp:37-48](xllm/core/framework/sampling/logits_utils.cpp#L37-L48))：`gather → where(score<0, score*penalty, score/penalty) → scatter_`
- **温度缩放** ([logits_utils.cpp:50-59](xllm/core/framework/sampling/logits_utils.cpp#L50-L59))：`logits.div_(temperatures)`

### 1.5 投机采样（Rejection Sampling）

[rejection_sampler.cpp](xllm/core/framework/sampling/rejection_sampler.cpp) 提供两种路径：
- **通用路径** (`random_sample`)：纯 PyTorch 原语（gather/clone/sub/div/scatter_），`torch_npu` 自动映射
- **融合路径** (`random_sample_fused`)：通过 `kernel::rejection_sample()` 调度到后端专用 kernel（NPU 路径将 draft_probs、target_probs 等全部 flatten 后一次传入）

### 1.6 架构特点总结

| 维度 | xllm 实现方式 |
|------|-------------|
| 编程语言 | C++ (LibTorch) |
| 条件编译 | `USE_NPU` / `USE_MLU` / `USE_CUDA` 宏 |
| 核心算子 | ACLNN 融合算子（性能优先路径） |
| 回退策略 | 纯 PyTorch 原语（通用路径） |
| 惩罚项 | 全部 PyTorch 原语，无自定义 kernel |
| 流管理 | 同步等待 (`aclrtSynchronizeStream`) |
| 投机采样 | 支持融合 kernel 路径 |

---

## 二、vllm-ascend Logits 后处理 NPU 适配实现

### 2.1 整体架构

vllm-ascend 遵循 vLLM v1 的 **Python 类继承 + 方法重写** 模式，采样器架构如下：

```
Sampler (vllm.v1.sample.sampler)           ← vLLM 基类
  └─ AscendSampler (vllm_ascend.sample.sampler)  ← 昇腾子类
       ├─ apply_penalties()  → 重写：Triton-Ascend kernel
       ├─ topk_topp_sampler  → AscendTopKTopPSampler
       │    ├─ apply_top_k_top_p  → AscendC 融合算子 (A2/A3) / PyTorch (310P)
       │    └─ forward_native()   → async_exponential + softmax + argmax
       └─ do_async_exponential()  → NPU 异步流指数分布生成

AscendSampler310 (vllm_ascend._310p.sample)  ← 310P 变体
  └─ AscendTopKTopPSampler310  → CPU exponential 生成优化
```

### 2.2 v1 采样器核心实现

#### 2.2.1 惩罚项（Penalties）

vllm-ascend 使用 **Triton-Ascend kernel** 实现惩罚项，路径如下：

```
AscendSampler.apply_penalties() [sampler.py:43-62]
  ├─ 无 Triton  → 回退 vLLM 默认实现
  ├─ no_penalties → 直接返回 logits
  └─ apply_all_penalties() [penalties.py:25-45]
       └─ apply_penalties_triton() [ops/triton/penalty.py:96-123]
            ├─ get_token_bin_counts_and_mask_triton()  → 构建 prompt/output mask
            └─ apply_all_penalties_kernel[grid]()       → Triton kernel
```

Triton kernel ([penalty.py:30-93](vllm_ascend/ops/triton/penalty.py#L30-L93)) 特点：
- 按 sequence 维度并行（`pid = program_id(0)`）
- 对每个 token 的 vocab 分 Block 处理（`BLOCK_SIZE=2048`）
- **三种惩罚一次完成**：repetition（logit>0 除、<0 乘）+ frequency（减 count）+ presence（减 mask）

#### 2.2.2 Top-K/Top-P 筛选

vllm-ascend 实现 **设备感知的算子分发**：

```python
# sampler.py:167-171
apply_top_k_top_p = (
    _apply_top_k_top_p_ascendc    # A2/A3: AscendC 自定义融合算子
    if get_ascend_device_type() in [AscendDeviceType.A2, AscendDeviceType.A3]
    else _apply_top_k_top_p_pytorch  # 310P: 纯 PyTorch 实现
)
```

**AscendC 路径** ([sampler.py:157-164](vllm_ascend/sample/sampler.py#L157-L164))：
```python
def _apply_top_k_top_p_ascendc(logits, k, p):
    return torch.ops._C_ascend.npu_apply_top_k_top_p(logits, k=k, p=p)
```
- 底层对应 C++ 自定义算子 `aclnnApplyTopKTopPCustom`（位于 `csrc/apply_top_k_top_p_custom/`）
- 实现为排序+top-k+top-p 融合的一个 AscendC kernel
- 支持 FP32/FP16/BF16

**PyTorch 回退** ([sampler.py:121-154](vllm_ascend/sample/sampler.py#L121-L154))：
- 先 softmax 转概率
- top-k：sort → gather cutoff → mask fill
- top-p：cumsum → mask fill
- 至少保留一个 token

#### 2.2.3 随机采样（Gumbel-Max Trick）

vllm-ascend 使用 **Gumbel-Max trick** 实现随机采样，而非 `torch.multinomial`（避免 CPU-NPU 同步）：

```python
# sampler.py:15-38
def random_sample(probs, generators):
    with npu_stream_switch(global_stream()):
        q = torch.empty_like(probs)
        q.exponential_()                    # Gumbel 分布的指数部分
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)  # 用户自定义种子
    torch.npu.current_stream().wait_stream(global_stream())
    return probs.div_(q).argmax(dim=-1)     # argmax(p_i / q_i)
```

#### 2.2.4 异步 Exponential 优化

`AscendSampler.do_async_exponential()` ([sampler.py:73-86](vllm_ascend/sample/sampler.py#L73-L86)) 是 vllm-ascend 的关键性能优化：

- **在独立 NPU stream 上异步生成指数分布随机数**
- 与模型前向推理**流重叠**（stream overlap）
- 使用 `Event` 机制同步：在模型执行期间预生成随机数，采样时直接使用
- 310P 变体通过 CPU Generator 进一步优化

### 2.3 v2 采样器

vLLM v2 架构使用 `AscendSampler(Sampler)` ([worker/v2/sample/sampler.py](vllm_ascend/worker/v2/sample/sampler.py))：
- 重写 `sample()` 方法，替换其中的 Triton 算子调用
- 使用 `gumbel_sample()` 进行 Gumbel 采样
- 处理流程与 v1 一致：logit_bias → penalties → bad_words → temperature → min_p → top_k_top_p → gumbel_sample

### 2.4 投机采样（Rejection Sampling）

[rejection_sampler.py](vllm_ascend/sample/rejection_sampler.py) 实现了完整的投机解码拒绝采样：

| 组件 | Triton-Ascend 实现 | PyTorch 回退 |
|------|-------------------|-------------|
| 贪婪验证 | `rejection_greedy_sample_with_triton` | `rejection_greedy_sample_pytorch` |
| 随机验证 | `rejection_random_sample_kernel` | `rejection_random_sample_pytorch` |
| Block Verify | `rejection_random_sample_block_verify_kernel` | `rejection_random_sample_block_verify_pytorch` |
| Token 恢复 | `sample_recovered_tokens_kernel` | `sample_recovered_tokens_pytorch` |
| Batch 展开 | `expand_triton` | `expand_pytorch` |
| Top-K/Top-P 预处理 | `apply_top_k_top_p`（同采样器） | 同左 |

### 2.5 v2 Model Runner 集成

[worker/v2/model_runner.py](vllm_ascend/worker/v2/model_runner.py#L97-L104) 显示 v2 架构显式构造 AscendSampler：

```python
self.sampler: AscendSampler = AscendSampler(
    max_num_reqs=self.max_num_reqs,
    vocab_size=self.vocab_size,
    device=self.device,
    req_states=self.req_states,
    logprobs_mode=self.model_config.logprobs_mode,
    num_speculative_tokens=self.num_speculative_steps + 1,
)
```

### 2.6 架构特点总结

| 维度 | vllm-ascend 实现方式 |
|------|---------------------|
| 编程语言 | Python (PyTorch + Triton + AscendC) |
| 核心算子 | Triton-Ascend kernel（penalties/rejection）+ AscendC 融合算子（top-k-top-p） |
| 设备感知 | 运行时按芯片类型（A2/A3 vs 310P）自动分发算子 |
| 异步优化 | NPU 流重叠（预生成指数分布随机数） |
| 回退策略 | 完整的 PyTorch 回退实现，每一层都有 fallback |
| 投机采样 | Triton-Ascend 全覆盖（贪婪/随机/Block Verify + token 恢复） |
| Logprobs 模式 | 支持 `processed_logits` / `processed_logprobs` / `raw_logprobs` |

---

## 三、对比分析：xllm vs vllm-ascend

### 3.1 架构范式对比

| 维度 | xllm | vllm-ascend |
|------|------|-------------|
| **框架语言** | C++ (LibTorch) | Python (PyTorch) |
| **适配方式** | 编译宏条件编译 | 类继承+方法重写 |
| **算子调度** | 编译时静态分发 | 运行时动态分发 |
| **NPU 算子路线** | ACLNN 原生 API | Triton-Ascend + AscendC 自定义 |
| **核心优势** | 零运行时开销、与 CUDA 路径统一 | 开发效率高、灵活可配置 |

### 3.2 惩罚项实现对比

| | xllm | vllm-ascend |
|------|------|-------------|
| 技术选型 | 纯 PyTorch ATen（gather/scatter） | Triton-Ascend kernel |
| 性能特点 | 依赖 torch_npu 自动映射，可能有多次 kernel launch | 三种惩罚融合在一个 kernel，单次 launch |
| 独特性 | 需要预构建 unique_token_ids 张量 | 运行时构建 prompt/output mask（bincount） |

### 3.3 Top-K/Top-P 实现对比

| | xllm | vllm-ascend |
|------|------|-------------|
| 技术选型 | ACLNN `aclnnApplyTopKTopP` | AscendC 自定义 `npu_apply_top_k_top_p` (A2/A3) |
| 触发条件 | top_k 和 top_p **同时**有效 | 任意一个有效即可 |
| 单参数回退 | 排序+mask（PyTorch） | PyTorch sort+cumsum |
| 实现层级 | 调用 CANN 外部算子库 | 自研 AscendC kernel（C++ host + device） |

### 3.4 采样实现对比

| | xllm | vllm-ascend |
|------|------|-------------|
| 贪婪采样 | `probs.argmax(-1)` (torch_npu) | `argmax` + 异步优化 |
| 随机采样 | `probs.multinomial()` (torch_npu) | Gumbel-Max trick (exponential + div + argmax) |
| 异步优化 | 无 | 独立 NPU stream 预生成指数分布随机数 |
| 种子处理 | 未暴露 per-request 种子 | 支持 per-request Generator |
| 混合采样 | `torch::where(do_sample, random, greedy)` | 同上 |

### 3.5 投机采样对比

| | xllm | vllm-ascend |
|------|------|-------------|
| 融合 kernel | 支持（`random_sample_fused`） | 支持（Triton-Ascend kernel） |
| 通用路径 | PyTorch 原语 | PyTorch 原语 |
| Block Verify | 不支持 | 支持（`rejection_random_sample_block_verify_kernel`） |
| Draft Probs | 支持 dense/selected-only 两种格式 | 支持 dense/ngram 两种模式 |

### 3.6 关键差异总结

1. **语言选择**：xllm 选择 C++ 获得更好的与 CUDA 路径的统一性和编译期优化；vllm-ascend 选择 Python 获得更快的开发迭代速度和 Triton 生态支持。

2. **NPU 算子策略**：xllm 依赖 CANN 外部算子库（`aclnnApplyTopKTopP`），vllm-ascend 自研 AscendC kernel（`npu_apply_top_k_top_p`），后者对芯片特性有更好的定制能力。

3. **流管理**：vllm-ascend 充分利用 NPU 流并发能力（异步 exponential 生成），xllm 目前使用同步等待模式。

4. **运行时灵活性**：vllm-ascend 根据芯片型号（A2/A3/310P）动态选择最优算子，xllm 通过编译宏静态选择。

5. **惩罚项融合度**：vllm-ascend 的 Triton kernel 将三种惩罚融合为单次 launch，xllm 分三次 ATen 操作。

---

## 四、rtp-llm Logits 后处理 NPU 适配路线图

### 4.1 当前状态分析

根据 [CudaSampleOp.cc](rtp_llm/models_py/bindings/core/CudaSampleOp.cc#L285-L297) 的 `#elif USING_ASCEND` 分支代码，rtp-llm 在 Ascend NPU 上的 logits 后处理**当前为纯桩实现（stub）**，两个核心函数均直接抛出 `ERROR_UNIMPLEMENTED` 异常：

```cpp
#elif USING_ASCEND
// Sample ops (Ascend) — stub implementations
GreedyOutput sampleGreedy(const GreedyParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}
void chainSpeculativeSampling(const SpeculativeSamplingParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}
```

这意味着以下功能在 Ascend 上**全部不可用**：

| 功能 | CUDA 实现 | Ascend 当前状态 | 差距 |
|------|----------|----------------|------|
| Temperature 惩罚 | CUDA kernel (half2 向量化) | 未实现 (stub) | ❌ 缺失 |
| Repetition/Frequency/Presence 惩罚 | `batchApplyPenaltyLongSeq` kernel (atomicAdd) | 未实现 (stub) | ❌ 缺失 |
| N-gram 禁止 | `invokeBanRepeatNgram` (TRT-LLM) | 未实现 (stub) | ❌ 缺失 |
| Top-K/Top-P 采样 | FlashInfer kernel (5个) | 未实现 (stub) | ❌ 缺失 |
| 投机采样 | FlashInfer `chain_speculative_sampling` | 未实现 (stub) | ❌ 缺失 |
| Logits Mask | `mask_logits` kernel | 未实现 (stub) | ❌ 缺失 |
| LogitsProcessor 插件链 | `mask_logits` kernel | 未实现 (stub) | ❌ 缺失 |
| Beam Search | TRT-LLM kernel | 未实现 (stub) | ❌ 缺失 |

> **注**：`#elif USING_ASCEND` 分支（[CudaSampleOp.cc:285](rtp_llm/models_py/bindings/core/CudaSampleOp.cc#L285)）仅引入了 `CommonDefines.h` 头文件，未包含任何 penalty kernel、FlashInfer 等 CUDA 路径所依赖的关键头文件和实现。Ascend 路径需要从零开始构建。

### 4.2 适配目标

基于 xllm 和 vllm-ascend 的实践经验，为 rtp-llm 制定以下适配目标：

#### 核心功能目标（Phase 1 — 基础可用）

| 序号 | 功能 | 目标 | 参考方案 |
|------|------|------|----------|
| T1 | Temperature 优化 | 融合 bias + temperature 为单次操作 | 参考 vllm-ascend 的 `logits.div_(temperature.unsqueeze(-1))` 模式 |
| T2 | 惩罚项优化 | 三种惩罚融合为单次 NPU kernel launch | 参考 vllm-ascend Triton penalty kernel 或 xllm gather/scatter 方案 |
| T3 | Top-K/Top-P 加速 | 用融合算子替代 `topk`+`multinomial` 两步操作 | 参考 vllm-ascend AscendC `npu_apply_top_k_top_p` |
| T4 | N-gram 禁止 | 实现 NPU 版本 `BanRepeatNgram` | 可用 PyTorch 实现或封装为 AscendC kernel |
| T5 | 随机采样去同步 | 用 Gumbel-Max trick 替代 `torch.multinomial` | 参考 vllm-ascend `random_sample()` |

#### 性能目标（Phase 2 — 性能优化）

| 序号 | 功能 | 目标 | 参考方案 |
|------|------|------|----------|
| P1 | 异步流重叠 | 模型推理与采样参数准备流重叠 | 参考 vllm-ascend `do_async_exponential()` |
| P2 | Logits Mask 加速 | 将 `mask_logits` kernel 适配为 NPU 高效实现 | 可用 Triton-Ascend 或 AscendC |
| P3 | 投机采样 | 实现 NPU 版本 rejection sampling | 参考 vllm-ascend Triton rejection kernel 或 xllm 融合路径 |
| P4 | Beam Search | NPU 适配 beam search 候选选择 | 参考 xllm ACLNN `beam_search` 算子 |

#### 性能指标目标

| 指标 | 当前 (CUDA) | 目标 (NPU Phase 1) | 目标 (NPU Phase 2) |
|------|------------|-------------------|-------------------|
| 采样延迟（单 batch） | < 0.1ms | < 0.3ms | < 0.15ms |
| 采样延迟（batch=256） | < 0.5ms | < 1.5ms | < 0.8ms |
| 惩罚项吞吐 | N/A | 不低于 CUDA 的 60% | 不低于 CUDA 的 85% |
| Top-K/Top-P 吞吐 | N/A | 不低于 CUDA 的 50% | 不低于 CUDA 的 80% |
| 投机采样接受率 | 与 CUDA 一致 | 与 CUDA 一致 | 与 CUDA 一致 |

### 4.3 推荐技术路线

基于三项目的技术架构差异，推荐 rtp-llm 采用 **分层渐进式** 适配策略：

```
Phase 1: PyTorch 原语适配（快速上线）
  └─ 利用 torch_npu 的自动映射能力，用纯 PyTorch 操作替代 CUDA kernel
     参考: xllm 的惩罚项实现 + vllm-ascend 的 Gumbel-Max 采样

Phase 2: Triton-Ascend / AscendC 加速（性能提升）
  └─ 对热路径（penalties、top-k-top-p、rejection sample）用自定义算子替换
     参考: vllm-ascend 的 Triton penalty kernel + AscendC top-k-top-p

Phase 3: 深度优化（极致性能）
  └─ 流重叠、算子融合、内存优化
     参考: vllm-ascend 的异步 exponential + xllm 的融合 rejection sample
```

### 4.4 具体实施建议

#### 4.4.1 温度/惩罚项适配

```cpp
// rtp-llm CudaSampleOp.cc — 建议的 NPU 路径改造
#if USING_ASCEND
  // Phase 1: 纯 PyTorch 替代（兼容性好）
  // Temperature: logits.div_(temperatures.unsqueeze(1))
  // Penalties: 参考 xllm 的 gather/scatter 模式
  
  // Phase 2: 封装为 AscendC/Triton-Ascend kernel
  // 将 temperature + repetition + frequency + presence
  // 融合为单次 kernel launch
#endif
```

#### 4.4.2 Top-K/Top-P 适配

推荐两种方案按条件选择：
- **方案 A（推荐）**：复用 vllm-ascend 的 AscendC `npu_apply_top_k_top_p` 算子（需要将代码提取为独立库或直接依赖）
- **方案 B**：封装 ACLNN `aclnnApplyTopKTopP`（参考 xllm 方式）

#### 4.4.3 随机采样适配

使用 Gumbel-Max trick 替代 `torch.multinomial`：
```cpp
// 替代 torch::multinomial(probs, 1)
auto q = torch::empty_like(probs).exponential_();
auto samples = probs.div_(q).argmax(-1);
```

#### 4.4.4 投机采样适配

分两步实现：
1. Phase 1：PyTorch 原语实现（参考 vllm-ascend `rejection_random_sample_pytorch`）
2. Phase 2：融合 kernel 实现（参考 vllm-ascend Triton kernel）

### 4.5 文件改造清单

| 文件 | 改造内容 | 优先级 |
|------|---------|--------|
| `models_py/bindings/core/CudaSampleOp.cc` | 新增 `USING_ASCEND` 分支，实现 temperature/penalties 的 PyTorch 替代 | P0 |
| `models_py/bindings/core/CudaSampleOp.cc` | 替换 `torch::multinomial` 为 Gumbel-Max trick | P0 |
| `models_py/bindings/common/kernels/` | 新增 AscendC 算子（top-k-top-p 融合、penalty 融合） | P1 |
| `cpp/normal_engine/speculative/SpeculativeSampler.*` | 新增 NPU 路径投机采样验证 | P1 |
| `models_py/bindings/common/kernels/banRepeatNgram.cu` | 新增 NPU 版本 N-gram 禁止实现 | P1 |
| `models_py/bindings/common/kernels/mask_logits.cu` | 新增 NPU 版本 logits mask 实现 | P2 |

---

## 五、总结

### 5.1 关键发现

1. **xllm 和 vllm-ascend 代表了两种 NPU 适配范式**：xllm 以 C++ 编译宏为核心实现统一后端，vllm-ascend 以 Python 类继承+运行时设备分发为核心实现轻量适配。

2. **ACLNN vs AscendC/Triton 是两种主流 NPU 加速路径**：ACLNN 提供 CANN 官方算子库接口（xllm 采用），AscendC/Triton-Ascend 提供更灵活的自定义 kernel 开发能力（vllm-ascend 采用）。

3. **流异步优化是 NPU 性能的关键**：vllm-ascend 的 `do_async_exponential()` 将采样准备与模型推理流重叠，是显著的性能提升手段。

4. **rtp-llm 当前 NPU 适配处于早期阶段**：大部分功能仅有基础 PyTorch 替代，核心路径（投机采样、N-gram 禁止）尚未实现。

### 5.2 推荐策略

rtp-llm 的 NPU 适配应优先选择 **PyTorch 原语 → Triton-Ascend/AscendC 加速** 的渐进式路线，理由如下：

- rtp-llm 已有 C++/CUDA 代码基础，与 xllm 类似但更复杂
- 可复用 vllm-ascend 已有的 AscendC 算子和 Triton kernel
- PyTorch 原语阶段可快速验证功能正确性
- Triton-Ascend 提供比纯 ACLNN 更灵活的调优能力