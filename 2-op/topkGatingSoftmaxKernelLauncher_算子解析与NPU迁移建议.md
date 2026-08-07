# `topkGatingSoftmaxKernelLauncher` 算子解析与 NPU 迁移建议

> 源码位置:[rtp-llm/rtp_llm/models_py/bindings/common/kernels/moe/moe_routing_kernels.cu](file:///d:/WORK/00018-RTP-LLM/rtp-llm-porting/rtp-llm/rtp_llm/models_py/bindings/common/kernels/moe/moe_routing_kernels.cu)
>
> 用途:MoE 路由的核心算子 — 把 router logits 转成 top-k expert 权重 + 索引
>
> NPU 等价算子:`torch_npu.npu_moe_gating_top_k_softmax`(CANN 原生)

---

## 一、算子全景

`topkGatingSoftmaxKernelLauncher` 是 **MoE Routing 的入口函数**:把 router 输出的 logits 转成每个 token 选 top-k 个 expert 的权重 `output[M, k]` 和索引 `indices[M, k]`,核心是 **softmax + top-K + (可选) renormalize** 的融合算子。

### 1.1 在 MoE 推理流程中的位置

```
hidden_states [M, D]
     │
     ├──► router_linear ──────────► router_logits [M, E]
     │                                  │
     │                                  ▼
     │                  ┌─────────────────────────────────┐
     │                  │ topkGatingSoftmaxKernelLauncher │  ← 本文档分析的算子
     │                  │   softmax → topk → renormalize  │
     │                  └─────────────────────────────────┘
     │                                  │
     │                            topk_weights [M, k]
     │                            topk_ids    [M, k]
     │                                  │
     │                                  ▼
     └──► invoke_fused_moe_kernel(hidden_states, topk_weights, topk_ids, ...)
                                          │
                                          ▼
                                    experts_output [M, H]
```

涉及的关键算子:

| 函数 | 行号 | 作用 |
|------|------|------|
| `topkGatingSoftmaxKernelLauncher` | L436 | 顶层 launcher,按 num_experts 分派 CASE 或 default 路径 |
| `topkGatingSoftmaxLauncherHelper<N, WARPS_PER_TB>` | L451 | 模板特化版,启动融合 kernel |
| `topkGatingSoftmax` (kernel) | L331 | 融合 softmax+topk kernel(E ≤ 256 时使用) |
| `moeSoftmax` | L39 | 独立 softmax kernel(default 路径用) |
| `moeTopK` | L150 | 独立 topk kernel(default 路径用) |

---

## 二、`topkGatingSoftmaxKernelLauncher`(C++ 启动器,L436-478)

### 2.1 函数签名

```cpp
template<typename TOPK_T>
void topkGatingSoftmaxKernelLauncher(
    float const*  input,                // [num_rows, num_experts] router logits
    float*        output,               // [num_rows, k] 输出 top-k 权重
    float*        softmax_temp_output,  // 仅 default 分支用做 softmax 中间结果
    TOPK_T*       indices,              // [num_rows, k] 输出 top-k expert 索引
                                       //   (模板参数,可能是 int32/int8)
    int*          source_row,           // [num_rows, k] 记录原始 token 行号
    int64_t const num_rows,             // token 数 M
    int    const   num_experts,         // E
    int    const   k,                   // top-k 值
    int    const   startk,              // 写入 output 的 k 维起始(支持只写一部分)
    int    const   endk,                // 写入 output 的 k 维结束
    int    const   start_expert,         // expert id 偏移(用于 EP)
    int    const   end_expert,           // expert id 偏移(用于 EP)
    MOEExpertScaleNormalizationMode norm_mode,  // NONE / RENORMALIZE
    cudaStream_t   stream);
```

### 2.2 核心逻辑

```cpp
static constexpr int WARPS_PER_TB = 4;       // 每 block 4 个 warp = 128 线程

#define CASE(N) \
    case N: \
        topkGatingSoftmaxLauncherHelper<N, WARPS_PER_TB>(input, nullptr, output, indices, \
                                                         source_row, num_rows, k, startk, endk, \
                                                         start_expert, end_expert, norm_mode, stream); \
        break;

switch (num_experts) {
    CASE(1) CASE(2) CASE(4) CASE(8) CASE(16) CASE(32) CASE(64) CASE(128) CASE(256)
    default: {
        static constexpr int TPB = 256;
        TLLM_CHECK(softmax_temp_output != nullptr);
        moeSoftmax<TPB><<<num_rows, TPB, 0, stream>>>(
            input, nullptr, softmax_temp_output, num_experts);
        moeTopK<TPB><<<num_rows, TPB, 0, stream>>>(
            softmax_temp_output, nullptr, output, indices, source_row,
            num_experts, k, startk, endk, start_expert, end_expert, norm_mode);
    }
}
```

### 2.3 两种执行路径

| 分支 | 触发条件 | 调用路径 | 实现方式 |
|------|---------|---------|---------|
| **CASE(N)** 各自独立 | `num_experts == N` (N ∈ {1,2,4,8,16,32,64,128,256}) | `topkGatingSoftmaxLauncherHelper<N, 4>` | **融合单 kernel** `topkGatingSoftmax`,编译期模板特化 |
| **default** | `num_experts > 256` 或非 2 的幂 | `moeSoftmax` + `moeTopK` 两步 | **拆分双 kernel**,需要 `softmax_temp_output` 中间 buffer |

---

## 三、CASE(N) 分支:融合单 kernel 路径

### 3.1 `topkGatingSoftmaxLauncherHelper<N, WARPS_PER_TB>`(L451-485)

模板函数,内部:

1. 用 `detail::TopkConstants<EXPERTS, BYTES_PER_LDG>` 在**编译期**算出:
   - `VPT`(Values Per Thread):每 thread 处理的元素数
   - `ROWS_PER_WARP`:每 warp 处理的行数
2. 根据 `num_rows` 算出需要的 warp 数 / block 数
3. 启动融合 kernel [`topkGatingSoftmax`](file:///d:/WORK/00018-RTP-LLM/rtp-llm-porting/rtp-llm/rtp_llm/models_py/bindings/common/kernels/moe/moe_routing_kernels.cu#L331):

```cpp
topkGatingSoftmax<VPT, ROWS_PER_WARP, WARPS_PER_TB, EXPERTS, TOPK_T, BYTES_PER_LDG>
    <<<grid, block, smem_size, stream>>>(
        input, output, indices, source_row, finished,
        num_rows, num_experts, k, startk, endk,
        start_expert, end_expert, norm_mode);
```

### 3.2 融合 kernel 的核心逻辑

[`topkGatingSoftmax` kernel](file:///d:/WORK/00018-RTP-LLM/rtp-llm-porting/rtp-llm/rtp_llm/models_py/bindings/common/kernels/moe/moe_routing_kernels.cu#L331):

```
对每个 row:
  1. 加载整行 logits 到寄存器/共享内存
  2. warp 内 reduction 求最大值 (用 __shfl_sync)
  3. logits - max → exp → sum → divide  (数值稳定 softmax)
  4. iterative argmax 选 top-k(每次找最大值并 mask)
  5. 应用 startk/endk 切片写入
  6. 应用 start_expert 偏移
  7. 按 norm_mode 做 renormalize
```

### 3.3 模板特化的优势

- **编译期常量 N**:`for (int i = 0; i < N; i++)` 完全展开,无分支
- **编译期 VPT**:每 thread 的元素数固定,寄存器分配最优
- **编译期 ROWS_PER_WARP**:warp 内行布局确定,warp shuffle 路径最短
- **BYTES_PER_LDG**:向量化 load 大小确定(LDG.128 等)

---

## 四、default 分支:拆分双 kernel 路径

### 4.1 触发条件

`num_experts > 256` 或非 2 的幂。

### 4.2 `moeSoftmax`(L39)

```cpp
template<int TPB>
__global__ void moeSoftmax(float const* input, bool const* finished,
                            float* output, int64_t const num_cols) {
    int row = blockIdx.x;
    // 1. 读取整行 logits
    // 2. 求 max
    // 3. exp(logits - max) → sum → divide
    // 4. 写回 output (softmax_temp_output)
}
```

**每个 block 处理一行**,256 线程并行做 softmax。

### 4.3 `moeTopK`(L150)

```cpp
template<int TPB>
__global__ void moeTopK(float const* input, bool const* finished,
                        float* output, TOPK_T* indices, int* source_row,
                        int64_t const num_cols, int const k,
                        int const startk, int const endk,
                        int const start_expert, int const end_expert,
                        MOEExpertScaleNormalizationMode norm_mode) {
    int row = blockIdx.x;
    // 1. 读取整行 softmax 结果
    // 2. iterative argmax 选 top-k
    // 3. 应用 startk/endk/start_expert/end_expert/norm_mode
    // 4. 写回 output/indices/source_row
}
```

### 4.4 为什么 default 要拆分?

详见 [rtp-llm Qwen3.5 算子迁移指导.md](file:///d:/WORK/00018-RTP-LLM/rtp-llm-porting/docs/Qwen3.5/rtp-llm%20Qwen3.5%20算子迁移指导.md),核心原因:

| 约束 | 影响 |
|------|------|
| **硬件约束**:寄存器/共享内存容量限制 | E=512 时每 thread 占 4 fp32,开始吃紧;E=2048 占 16 fp32,严重 spill |
| **算法约束**:大 E 需要更优的 topk 算法 | iterative argmax 的 O(k*E) 太贵,需换 bitonic sort 或 radix select |
| **编译约束**:模板特化只能覆盖有限 N 值 | 不能为所有 E 编译特化版本,二进制会爆炸 |

**E ≤ 256 是 NVIDIA GPU 寄存器预算和 occupancy 的甜点**。

---

## 五、关键参数语义

### 5.1 `norm_mode`(归一化模式)

```cpp
enum class MOEExpertScaleNormalizationMode {
    NONE,           // 直接输出 softmax top-k 值
    RENORMALIZE,    // 把选中的 k 个权重重新归一化到和为 1
};
```

- `RENORMALIZE` 的实现(L397-404):
  ```cpp
  float renorm_value = 1.f;
  for (int i = 0; i < k; i++) renorm_value -= output[idx_i];
  renorm_value = 1 / renorm_value;
  for (int i = 0; i < k; i++) output[idx_i] *= renorm_value;
  ```

### 5.2 `startk` / `endk`(top-k 切片)

只把 top-k 结果中 `[startk, endk)` 区间写入 `output`。用于 sparse mixer 之类的多阶段选择。

### 5.3 `start_expert` / `end_expert`(EP 偏移)

kernel 内部算出的索引是 local,加上 `start_expert` 后才是全局 expert id。支持 expert parallel 的分片。

### 5.4 `source_row`(原始行号)

记录"这条 top-k 结果对应原始哪一行 token",便于后续 `scatter`/`gather` 还原 token-expert 映射。

### 5.5 `finished` mask

用于 speculative decoding 的多阶段 routing:
- `finished[i] == true` → 该 token 已完成 routing,跳过
- `finished == nullptr` → 标准路径

---

## 六、融合机制总结

### 6.1 "Fused" 的精确含义

`topkGatingSoftmax` 的 "fused" 指**算子级融合(softmax + topk + renormalize)**:

| 融合维度 | 是否做 | 说明 |
|---------|-------|------|
| **算子级融合(softmax + topk)** | ✅ | E ≤ 256 时融合为单 kernel |
| **算子级融合(+ renormalize)** | ✅ | renormalize 也融入 kernel 尾部 |
| **算子级融合(+ EP 偏移)** | ✅ | `start_expert` 偏移在 kernel 内完成 |
| **跨 token 融合** | ❌ | 每个 token 独立计算 |
| **跨层融合** | ❌ | 不涉及 |

### 6.2 融合收益

| 维度 | 未融合(PyTorch) | 融合后(CUDA CASE 分支) |
|------|------------------|--------------------------|
| softmax | 1 次 launch | 0(融入) |
| topk | 1 次 launch | 0(融入) |
| renormalize | 1 次 launch | 0(融入) |
| EP 偏移 | 1 次 launch | 0(融入) |
| **总 launch 数** | 4 | **1** |
| GMEM 中间落盘 | softmax_out | 0 |

### 6.3 default 分支的折中

E > 256 时融合反而更慢(寄存器溢出),所以拆成 2 个 kernel,但仍有以下优化:
- **比纯 PyTorch 少 2 次 launch**(4 → 2)
- **renormalize + EP 偏移仍融入 moeTopK kernel**
- 中间 `softmax_temp_output` 是 fp32,精度损失小

---

## 七、NPU 等价算子:`torch_npu.npu_moe_gating_top_k_softmax`

### 7.1 算子签名

```python
import torch_npu

# 替换 topkGatingSoftmaxKernelLauncher 的核心计算
topk_weights, topk_ids, source_row = torch_npu.npu_moe_gating_top_k_softmax(
    router_logits,     # [num_rows, num_experts] 输入 logits
    finished=None,     # 可选 finished mask
    k=top_k,           # top-k 值
)
# topk_weights: [num_rows, k]    softmax top-k 权重
# topk_ids:     [num_rows, k]    expert 索引(int32)
# source_row:   [num_rows, k]    原始行号(int32)
```

### 7.2 等价性对比

| 维度 | `topkGatingSoftmaxKernelLauncher` | `npu_moe_gating_top_k_softmax` | 是否等价 |
|------|-----------------------------------|--------------------------------|---------|
| **数学操作** | softmax → topk → (可选) renorm | softmax → topk | ✅ 等价 |
| **实现方式** | CASE:融合单 kernel / default:拆双 kernel | CANN 内部融合 | ⚠️ 实现不同但结果一致 |
| **触发条件** | CASE: E ≤ 256 / default: E > 256 | 任意 E(到 CANN 上限) | ⚠️ 互补 |
| **`finished` mask** | 支持(传 `nullptr` 表示无) | 支持(可传 `None`) | ✅ 等价 |
| **`source_row` / `row_idx`** | 输出 | 输出 | ✅ 等价 |
| **`norm_mode`** (RENORMALIZE) | kernel 内部做 renorm | **不做**,需 Python 后处理 | ❌ 不等价 |
| **`startk` / `endk`** (top-k 切片写) | 支持 | **不支持** | ❌ 不等价 |
| **`start_expert` / `end_expert`** (EP 偏移) | kernel 内部加偏移 | **不支持**,需 Python 后处理 | ❌ 不等价 |
| **`softmax_temp_output`** (中间 buffer) | 必须传入(default 分支) | **不需要**(内部已融合) | ⚠️ 接口简化 |

### 7.3 关键差异

CANN 算子是 **default 分支的简化版** — 只做 softmax+topk 主体,**3 个特性需要 Python 兜底**:
1. `norm_mode = RENORMALIZE`
2. `startk / endk` 切片
3. `start_expert / end_expert` 偏移

---

## 八、CUDA vs CANN 算子关系图

```
topkGatingSoftmaxKernelLauncher(CUDA)
├── CASE(N) 分支  ←─ 融合单 kernel(E≤256,模板特化)
│                   支持 norm_mode / startk/endk / start_expert
│
└── default 分支   ←─ 拆双 kernel(E>256 或非 2 的幂)
                    支持 norm_mode / startk/endk / start_expert
                    ↑
                    └─ 数学上等价于
                       npu_moe_gating_top_k_softmax(CANN)
                       └─ 但只有"softmax + topk"核心,不做 renorm/offset/切片
                          (需 Python 补 3 个补丁)
```

---

## 九、NPU 移植建议

### 9.1 短期方案:用 CANN 算子 + Python 补丁(推荐)

**核心改动**:把 `topkGatingSoftmaxKernelLauncher` 调用替换为 `torch_npu.npu_moe_gating_top_k_softmax` + 3 个 Python 补丁。

```python
import torch_npu

def topk_gating_softmax_npu(
    router_logits,      # [M, E] fp32
    top_k: int,
    startk: int = 0,
    endk: int = None,
    start_expert: int = 0,
    norm_mode: str = "NONE",   # "NONE" or "RENORMALIZE"
    finished: torch.Tensor = None,
):
    if endk is None:
        endk = top_k
    
    # 1. 调用 CANN 算子(等价于 default 分支的 softmax+topk 主体)
    topk_weights, topk_ids, source_row = torch_npu.npu_moe_gating_top_k_softmax(
        router_logits, finished=finished, k=top_k
    )
    
    # 2. 补 default 分支里 moeTopK kernel 做的事:
    
    # (a) start_expert / end_expert (EP 偏移)
    topk_ids = topk_ids + start_expert
    
    # (b) startk / endk 切片
    topk_weights = topk_weights[:, startk:endk]
    topk_ids = topk_ids[:, startk:endk]
    if source_row is not None:
        source_row = source_row[:, startk:endk]
    
    # (c) norm_mode = RENORMALIZE 时
    if norm_mode == "RENORMALIZE":
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    
    return topk_weights, topk_ids, source_row
```

**参考实现**:vllm-ascend 的 [experts_selector.py#L85](file:///d:/WORK/00018-RTP-LLM/rtp-llm-porting/vllm-ascend/vllm_ascend/ops/fused_moe/experts_selector.py#L85) 的 `_select_experts_with_fusion_ops` 和 [experts_selector.py#L194](file:///d:/WORK/00018-RTP-LLM/rtp-llm-porting/vllm-ascend/vllm_ascend/ops/fused_moe/experts_selector.py#L194) 的 `_renormalize_topk_weights`。

### 9.2 中期方案:rtp-llm-npu 的 `SelectTopkOp` 路径

rtp-llm-npu 已有 `SelectTopkOp` C++ binding,见 [select_topk.py#L8](file:///d:/WORK/00018-RTP-LLM/rtp-llm-porting/rtp-llm-npu/rtp_llm/models_py/modules/base/cuda/select_topk.py#L8):

```python
class SelectTopk(nn.Module):
    def __init__(self, config):
        self.select_topk_op = compute_ops.SelectTopkOp(self.config)
    def forward(self, router_logits_fp32, topk_ids, topk_weights):
        self.select_topk_op.forward(router_logits_fp32, topk_ids, topk_weights)
```

**需要验证**:`SelectTopkOp::forward` 在 NPU 编译路径下的底层实现:
- (a) 已走 `aclnnMoeGatingTopKSoftmax` → 直接用,**无需改动**
- (b) 走 PyTorch fallback → 改用 9.1 方案
- (c) 空实现 → 改用 9.1 方案

### 9.3 长期方案:统一用 CANN 算子覆盖所有 E

**核心优势**:CANN 算子内部根据 E 自动选最优 kernel,**不需要在调用方写 CASE/default 分派**:

```python
# 统一路径,适用于任意 E
def select_topk_unified(router_logits, top_k, ...):
    # 不需要 if E <= 256: 走 CASE; else: 走 default
    # CANN 内部自动选最优路径
    return torch_npu.npu_moe_gating_top_k_softmax(router_logits, k=top_k)
```

**好处**:
- 简化代码(消除 switch/case)
- NPU 上 CANN 算子内部已针对 910B/910C 调优,通常优于 triton-ascend 编译产物
- 量化场景可扩展(虽然当前 `npu_moe_gating_top_k_softmax` 不直接支持量化,但未来 CANN 会加)

---

## 十、迁移注意事项

### 10.1 数值精度差异

- **default 分支**:softmax 结果落盘到 `softmax_temp_output`(fp32),topk 再读回
- **CANN 算子**:内部融合,softmax 结果直接在 UB/寄存器传给 topk,无中间落盘
- **影响**:CANN 版数值更稳定(无 fp32→bf16 中间 cast),但 rtp-llm 的 `softmax_temp_output` 是 fp32,实际差异很小

### 10.2 dtype 支持

- **CUDA**:只支持 fp32 输入
- **CANN**:支持 fp16 / bf16 / fp32 输入
- **影响**:NPU 上可以省一次 cast,直接传 router_logits 原 dtype

### 10.3 `group_list` 切换(vllm-ascend 风格)

如果同时迁移 `invoke_fused_moe_kernel` 到 `npu_grouped_matmul`,需要把 `topk_ids` 转成 `group_list` 格式:

```python
# 1. 调 CANN topk softmax
topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k_softmax(router_logits, k=k)

# 2. 转 group_list(用 CANN 算子)
group_list, group_list_type = torch_npu.npu_moe_init_routing_v2(
    topk_ids, pad_idx=..., ...
)

# 3. 调 grouped matmul
out = torch_npu.npu_grouped_matmul(
    x=[x], weight=[w1.T],
    group_list=group_list, group_list_type=group_list_type,
    ...
)[0]
```

### 10.4 不需要移植的部分

以下 CUDA 代码在 NPU 编译路径上**完全排除**:

| 文件/函数 | 处理方式 |
|----------|---------|
| [moe_routing_kernels.cu](file:///d:/WORK/00018-RTP-LLM/rtp-llm-porting/rtp-llm/rtp_llm/models_py/bindings/common/kernels/moe/moe_routing_kernels.cu) | `#ifndef USE_NPU` 排除 |
| `topkGatingSoftmax` kernel | 不需要 |
| `moeSoftmax` kernel | 不需要 |
| `moeTopK` kernel | 不需要 |
| `softmax_temp_output` buffer 分配 | 不需要 |

---

## 十一、关键设计点回顾

1. **模板特化分派(L470)**:`CASE(1) CASE(2) ... CASE(256)` 编译期特化 9 种 E 值,运行时零分支
2. **融合 softmax+topk(L331)**:E ≤ 256 时单 kernel 完成,省 3 次 launch
3. **拆分 default 路径(L470 default)**:E > 256 时拆 moeSoftmax + moeTopK,避免寄存器溢出
4. **`WARPS_PER_TB = 4`(L449)**:每 block 128 线程,适配 SM90 occupancy
5. **iterative argmax 算法**:O(k*E) 复杂度,E 小时寄存器内完成,极快
6. **`norm_mode` kernel 内融合(L397-404)**:renormalize 在 moeTopK kernel 尾部完成
7. **`start_expert` 偏移在 kernel 内**:避免外层 scatter
8. **`finished` mask 支持**:适配 speculative decoding 多阶段 routing
9. **`softmax_temp_output` 中间 buffer**:仅 default 分支需要,fp32 精度

这些设计点共同构成了一个 **"softmax + topk + renormalize 三算子融合 + 模板特化 + adaptive fallback"** 的高性能 MoE Routing kernel。

---

## 十二、总结

| 问题 | 答案 |
|------|------|
| NPU 有直接等价算子吗? | ✅ `torch_npu.npu_moe_gating_top_k_softmax` |
| 数学等价吗? | ✅ 是(softmax+topk 输出一致) |
| 接口等价吗? | ❌ 否(CANN 缺 3 个特性:renorm / startk-endk / start_expert) |
| 替换可行吗? | ✅ 可行,需 Python 补 3 个补丁 |
| 替换收益? | 简化代码(消除 switch/case),CANN 内部已调优 |
| rtp-llm-npu 现状 | `SelectTopkOp` C++ binding 存在,但需验证底层是否走 CANN |
| 移植优先级 | **高**(MoE routing 是热点) |

**核心建议**:确认 `SelectTopkOp` 在 NPU 编译路径下底层实现,如未走 CANN,直接在 NPU 路径用 `torch_npu.npu_moe_gating_top_k_softmax` + 3 个 Python 补丁替换。**不需要移植 CUDA kernel 源码**,整个 [moe_routing_kernels.cu](file:///d:/WORK/00018-RTP-LLM/rtp-llm-porting/rtp-llm/rtp_llm/models_py/bindings/common/kernels/moe/moe_routing_kernels.cu) 在 NPU 编译路径上排除。
