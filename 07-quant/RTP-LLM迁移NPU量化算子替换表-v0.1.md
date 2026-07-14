# RTP-LLM 迁移 NPU MXFP8 量化算子替换表 (v0.1)

> 本文档梳理 RTP-LLM 迁移到华为昇腾 NPU 时，MXFP8（Microscaling FP8，OCP 标准）量化模块需要替换的 CUDA 算子。
>
> **标记说明**：✅🟢=已适配 | ❌🔴=未适配

---

## 适配状态

> 基于 `rtp-llm-npu` 代码检查结果，标注当前算子替换的适配进度。

### 已适配项 ✅🟢

| 适配项 | 已适配代码 | 文件位置 |
|--------|-----------|---------|
| DeviceType.Ascend | `DeviceType.Ascend = 6` | device_type.py L14 |
| AscendImpl 基础框架 | `class AscendImpl(GpuImpl)` 含 `get_device_id`、`_get_mem_info`、`support_dio_load` | device_impl.py L696-710 |
| AscendImpl 注册 | `DeviceType.Ascend → AscendImpl` | device/__init__.py L22-23 |
| AscendF16Linear | `class AscendF16Linear(LinearBase)` + `LinearFactory.register` | impl/ascend/f16_linear.py |
| MoE BF16 Fallback | `class AscendBf16FallbackStrategy(MoeStrategy)` | impl/ascend/strategy/pytorch_fallback.py |
| MoE Ascend 注册 | `DeviceType.Ascend → AscendBf16FallbackStrategy` | fused_moe/__init__.py L62-70 |
| HCCL 链接配置 | `.bazelrc` 中 `--linkopt="-lhccl"`，`def.bzl` 中 `-DUSE_C10D_HCCL` | 构建配置 |

### 未适配项 ❌🔴

| 适配项 | 说明 |
|--------|------|
| AscendImpl MXFP8 方法 | 未重写 `per_block_cast_to_fp8`、`requant_weight_ue8m0` |
| ModelSlimConfig | quant_config.py 中无相关代码 |
| load_from_ckpt ModelSlim 检测 | 无 `quant_model_description.json` 检测 |
| requant_weight_ue8m0 移除 | per_block_fp8_quant_weight.py L780 仍调用 |
| per_block_cast_to_fp8 替换 | per_block_fp8_quant_weight.py L847 仍调用原函数 |
| NpuFp8MXFP8Linear | 无 ascend MXFP8 Linear 实现 |
| NpuMoEMXFP8Executor | 无 ascend MXFP8 MoE 实现 |
| DeepGEMM wrapper 修改 | `has_deep_gemm`/`is_deep_gemm_e8m0_used` 未改 |
| NCCL→HCCL (运行时) | collective_torch.py 仍用 nccl，backend_manager.py 仍 `backend="nccl"` |
| NZ 格式转换 | 无 `npu_format_cast` 调用 |

---

## 一、核心算子替换总览

| 序号 | 功能 | CUDA 算子 | NPU 替换算子 | 替换方式 | 适配状态 |
|------|------|----------|-------------|---------|---------|
| 1 | 动态量化（BF16→MXFP8） | `per_block_cast_to_fp8` | `torch_npu.npu_dynamic_mx_quant` | 直接替换 | ❌🔴 |
| 2 | DenseMLP 的 MXFP8 GEMM | `deep_gemm.fp8_gemm_nt` | `torch_npu.npu_quant_matmul` | 直接替换 | ❌🔴 |
| 3 | MoE 的 MXFP8 Grouped GEMM | `m_grouped_fp8_gemm_nt_masked` | `torch_npu.npu_grouped_matmul_swiglu_quant_v2` | 直接替换 | ❌🔴 |
| 4 | E8M0 scale 格式转换 | `requant_weight_ue8m0` | **移除** | NPU 原生支持 E8M0 | ❌🔴 |

---

## 二、详细替换说明

### 2.1 动态量化算子

| 项目 | CUDA | NPU |
|------|------|-----|
| **算子** | `per_block_cast_to_fp8(weight, group_size=128)` | `torch_npu.npu_dynamic_mx_quant(weight, output_dtype)` |
| **文件位置** | `models_py/kernels/cuda/fp8_kernel/fp8_kernel.py` L329 | `torch_npu` 原生算子 |
| **输入** | BF16/FP16 权重 `[M, N]` | BF16 权重 `[M, N]` |
| **输出** | FP8 E4M3 权重 + FP32 scale | FP8 E4M3 权重 + **E8M0 scale** |
| **block 大小** | 可配置（默认 128） | 固定 128×128 |
| **scale 格式** | FP32（需转换） | E8M0（原生，无需转换） |

**代码对比**：

```python
# CUDA 实现
weight_fp8, scale_fp32 = per_block_cast_to_fp8(weight, group_size=128)
# 后续需要 requant_weight_ue8m0 转换 scale 格式

# NPU 实现
weight_fp8, scale_e8m0 = torch_npu.npu_dynamic_mx_quant(weight, torch.float8_e4m3fn)
# scale 已经是 E8M0 格式，无需额外转换
```

---

### 2.2 DenseMLP 的 MXFP8 GEMM

| 项目 | CUDA | NPU |
|------|------|-----|
| **算子** | `deep_gemm.fp8_gemm_nt` | `torch_npu.npu_quant_matmul` |
| **文件位置** | `models_py/kernels/cuda/deepgemm_wrapper.py` | `torch_npu` 原生算子 |
| **调用位置** | `CudaFp8DeepGEMMLinear.forward` | 新增 `NpuFp8MXFP8Linear.forward` |
| **输入** | `(input_fp8, input_scale), (weight_fp8, weight_scale)` | `x, weight_fp8, scale` |
| **输出** | BF16 结果 | BF16 结果 |
| **scale 格式** | E8M0（需 TMA 对齐） | E8M0（原生） |

**代码对比**：

```python
# CUDA 实现
fp8_gemm_nt(
    (input_fp8, input_scales),
    (weight, weight_scales),
    output,
    disable_ue8m0_cast=not self.scale_ue8m0,
)

# NPU 实现
output = torch_npu.npu_quant_matmul(
    x,                    # BF16 激活
    weight_fp8,           # FP8 E4M3 权重
    scale,                # E8M0 scale
    dtype=torch.float8_e4m3fn
)
```

---

### 2.3 MoE 的 MXFP8 Grouped GEMM

| 项目 | CUDA | NPU |
|------|------|-----|
| **算子** | `m_grouped_fp8_gemm_nt_masked` | `torch_npu.npu_grouped_matmul_swiglu_quant_v2` |
| **文件位置** | `models_py/kernels/cuda/deepgemm_wrapper.py` | `torch_npu` 原生算子 |
| **调用位置** | `DeepGemmMaskedExecutor` | 新增 `NpuMoEMXFP8Executor` |
| **特点** | 分组 GEMM + masked | 分组 GEMM + 融合 SiLU 激活 |
| **权重** | `w1[E, N, K]`, `w2[E, K, N//2]` | 同左 |
| **scale** | `w1_scale`, `w2_scale` | 同左（E8M0 格式） |

**代码对比**：

```python
# CUDA 实现（两步 GEMM）
# Step 1: Gate + Up projection
m_grouped_fp8_gemm_nt_masked(expert_x, self._w1, upgate_output, masked_m, expected_m)
# Step 2: SiLU activation
silu_mul_masked_fp8_post_quant_fwd(...)
# Step 3: Down projection
m_grouped_fp8_gemm_nt_masked(down_input, self._w2, down_output, masked_m, expected_m)

# NPU 实现（融合算子，一步完成）
output = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
    x,                    # 输入激活
    w1,                   # gate_up_proj 权重
    w2,                   # down_proj 权重
    w1_scale,             # w1 的 E8M0 scale
    w2_scale,             # w2 的 E8M0 scale
)
```

---

### 2.4 E8M0 Scale 格式转换 — 移除

| 项目 | CUDA | NPU |
|------|------|-----|
| **算子** | `requant_weight_ue8m0` | **无需替换，直接移除** |
| **文件位置** | `models_py/kernels/cuda/fp8_kernel/fp8_kernel.py` L374 | — |
| **功能** | FP32 scale → E8M0 scale（幂次对齐） | NPU 原生 E8M0，无需转换 |
| **调用位置** | `PerBlockFp8Weight._postprocess` | 移除该调用 |

**说明**：

GPU 需要 `requant_weight_ue8m0` 的原因：
- GPU 的 FP8 GEMM（DeepGEMM）在 Blackwell 架构（SM100/SM120）上才原生支持 E8M0 scale
- 非 Blackwell GPU 需要将 FP32 scale 转换为 E8M0 格式，并做 TMA 内存对齐

NPU 不需要的原因：
- 昇腾 NPU 原生支持 E8M0 格式的 scale
- `npu_dynamic_mx_quant` 输出的 scale 已经是 E8M0
- 无需任何格式转换

---

## 三、替换涉及的代码文件

| 层级 | 文件 | 修改内容 | 适配状态 |
|------|------|---------|---------|
| **Device 层** | `device/device_impl.py` | 新增 `AscendImpl` 类，重写 `per_block_cast_to_fp8` 方法 | ✅🟢（基础框架）/ ❌🔴（MXFP8 方法） |
| **Device 层** | `device/device_type.py` | 新增 `DeviceType.Ascend` 枚举 | ✅🟢 |
| **Device 层** | `device/__init__.py` | 注册 `AscendImpl` | ✅🟢 |
| **权重层** | `model_loader/per_block_fp8_quant_weight.py` | 移除 `_postprocess` 中的 `requant_weight_ue8m0` 调用 | ❌🔴 |
| **权重层** | `model_loader/per_block_fp8_quant_weight.py` | 替换 `_load_raw_tensor` 中的 `per_block_cast_to_fp8` 为 `npu_dynamic_mx_quant` | ❌🔴 |
| **推理层** | `models_py/modules/factory/linear/impl/ascend/` | F16 Linear 已适配；MXFP8 Linear 未适配 | ✅🟢（F16）/ ❌🔴（MXFP8） |
| **推理层** | `models_py/modules/factory/fused_moe/impl/ascend/` | BF16 Fallback 已适配；MXFP8 MoE 未适配 | ✅🟢（BF16）/ ❌🔴（MXFP8） |
| **桥接层** | `models_py/kernels/cuda/deepgemm_wrapper.py` | 修改 `has_deep_gemm()` 返回 False | ❌🔴 |

---

## 四、数据格式对比

### 4.1 权重格式

| 项目 | CUDA (DeepGEMM) | NPU (MXFP8) |
|------|----------------|-------------|
| 权重 dtype | `torch.float8_e4m3fn` | `torch.float8_e4m3fn` |
| Scale dtype | `torch.float32` 或 `torch.int32`（packed E8M0） | `torch.float8_e8m0fnu` |
| Block 大小 | 128×128 | 128×128 |
| Scale 形状 | `[M//128, K//128]` 或 packed | `[M//128, K//128]` |

### 4.2 Scale 数值格式

| 格式 | 位宽 | 表示范围 | 精度 |
|------|------|---------|------|
| FP32 | 32 | ±3.4e38 | 高 |
| E8M0 | 8 | 2^(-127) ~ 2^(127) | 幂次离散（只能表示 2 的幂） |

NPU 原生 E8M0 的优势：
- 无需从 FP32 转换
- 存储量减少 4×（FP32→FP8）
- 硬件原生支持，无需软件模拟

---

## 五、静态量化 vs 动态量化路径

| 路径 | 是否需要量化算子 | 处理方式 |
|------|---------------|---------|
| **动态量化**（`is_quanted=False`） | ✓ 需要 `npu_dynamic_mx_quant` | 加载 BF16 权重 → 在线量化 → FP8 权重 + E8M0 scale |
| **静态量化**（`is_quanted=True`，ModelSlim） | ✗ 不需要量化算子 | 直接加载已量化的 FP8 权重 + E8M0 scale |

**说明**：
- 动态量化路径：推理时才量化，需调用 `npu_dynamic_mx_quant`
- 静态量化路径：权重已预先量化（ModelSlim 输出），直接加载即可，无需在线量化

---

## 六、验证方法

| 算子 | 验证代码 |
|------|---------|
| `npu_dynamic_mx_quant` | 检查输出 dtype 为 `float8_e4m3fn` 和 `float8_e8m0fnu` |
| `npu_quant_matmul` | 检查 MXFP8 GEMM 输出 shape 正确，dtype 为 BF16 |
| `npu_grouped_matmul_swiglu_quant_v2` | 检查 MoE 推理结果与 BF16 基线对比 |

```python
# 验证 NPU 算子
import torch_npu

# 1. 验证动态量化
weight = torch.randn(4096, 4096, dtype=torch.bfloat16, device="npu")
weight_fp8, scale = torch_npu.npu_dynamic_mx_quant(weight, torch.float8_e4m3fn)
assert weight_fp8.dtype == torch.float8_e4m3fn
assert scale.dtype == torch.float8_e8m0fnu

# 2. 验证 MXFP8 GEMM (DenseMLP)
x = torch.randn(128, 4096, dtype=torch.bfloat16, device="npu")
output = torch_npu.npu_quant_matmul(x, weight_fp8, scale, dtype=torch.float8_e4m3fn)
assert output.dtype == torch.bfloat16
assert output.shape == (128, 4096)

# 3. 验证 MXFP8 Grouped GEMM (MoE) — 需要实际 torch_npu 参数
# 参数格式需查阅 torch_npu 官方文档确认
E = 8  # num_experts
M = 128  # num_tokens
K = 4096  # hidden_size
N = 4096 * 2  # intermediate_size * 2

w1 = torch.randn(E, N, K, dtype=torch.float8_e4m3fn, device="npu")
w2 = torch.randn(E, K, N // 2, dtype=torch.float8_e4m3fn, device="npu")
w1_scale = torch.ones(E, N // 128, K // 128, dtype=torch.float8_e8m0fnu, device="npu")
w2_scale = torch.ones(E, K // 128, N // 256, dtype=torch.float8_e8m0fnu, device="npu")

# MoE 算子调用（参数格式需验证）
# output = torch_npu.npu_grouped_matmul_swiglu_quant_v2(...)
```

---

## 七、总结

迁移到 NPU 时，量化模块的算子替换本质上是将 **CUDA DeepGEMM 库**替换为 **torch_npu 原生算子**：

| 变化 | 说明 | 适配状态 |
|------|------|---------|
| 算子来源 | DeepGEMM → torch_npu | ❌🔴 |
| Scale 格式 | FP32 + 转换 → E8M0 原生 | ❌🔴 |
| MoE 融合 | 3 步分开执行 → 1 步融合算子 | ❌🔴 |
| 代码简化 | 移除 `requant_weight_ue8m0` 转换逻辑 | ❌🔴 |

> ✅🟢 **已适配基础部分**：DeviceType.Ascend 枚举、AscendImpl 基础框架（`get_device_id`、`_get_mem_info`、`support_dio_load`）、AscendImpl 注册、AscendF16Linear、MoE BF16 Fallback、HCCL 链接配置已在开发版本中适配。
>
> ❌🔴 **未适配核心部分**：MXFP8 量化的 3 个核心算子替换（`per_block_cast_to_fp8` → `npu_dynamic_mx_quant`、DeepGEMM → `npu_quant_matmul`、DeepGEMM masked → `npu_grouped_matmul_swiglu_quant_v2`）和 1 处移除（`requant_weight_ue8m0`）均未完成。

核心替换只有 **3 个算子** + **1 处移除**，迁移成本较低。