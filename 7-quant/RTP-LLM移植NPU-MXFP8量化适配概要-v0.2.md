# RTP-LLM 移植 NPU W8_MXFP8 量化适配概要 (v0.1)

> 本文档聚焦于将 RTP-LLM 的 **FP8 Per-Block 量化**移植到 NPU 的 **W8_MXFP8** 方案，是 [RTP-LLM移植NPU量化适配概要](./RTP-LLM移植NPU量化适配概要-v0.1.md) 的精简版。

---

## 适配状态

> 基于 `rtp-llm-npu` 代码检查结果，当前各适配项的状态如下：

### 已适配项 ✅🟢

| 适配项 | 已适配代码 | 文件位置 |
|--------|-----------|---------|
| DeviceType.Ascend | `DeviceType.Ascend = 6` | device_type.py L14 |
| AscendImpl 基础框架 | `class AscendImpl(GpuImpl)` 含 `get_device_id`、`_get_mem_info`、`support_dio_load` | device_impl.py L696-710 |
| AscendImpl 注册 | `DeviceType.Ascend → AscendImpl` | device/__init__.py L22-23 |
| is_ascend() 函数 | `get_device_type() == DeviceType.Ascend` | device_type.py L44-45 |
| AscendF16Linear | `class AscendF16Linear(LinearBase)` + `LinearFactory.register` | impl/ascend/f16_linear.py, __init__.py |
| MoE BF16 Fallback | `class AscendBf16FallbackStrategy(MoeStrategy)` | impl/ascend/strategy/pytorch_fallback.py |
| MoE Ascend 注册 | `DeviceType.Ascend → AscendBf16FallbackStrategy` | fused_moe/__init__.py L62-70 |
| HCCL 链接配置 | `.bazelrc` 中 `--linkopt="-lhccl"`，`def.bzl` 中 `-DUSE_C10D_HCCL` | 构建配置 |

### 未适配项 ❌🔴

| 适配项 | 说明 |
|--------|------|
| AscendImpl MXFP8 方法 | 未重写 per_block_cast_to_fp8、requant_weight_ue8m0 |
| ModelSlimConfig | quant_config.py 中无相关代码 |
| load_from_ckpt ModelSlim 检测 | 无 quant_model_description.json 检测 |
| requant_weight_ue8m0 移除 | per_block_fp8_quant_weight.py L780 仍调用 |
| per_block_cast_to_fp8 替换 | per_block_fp8_quant_weight.py L847 仍调用原函数 |
| NpuFp8MXFP8Linear | 无 ascend MXFP8 Linear 实现 |
| NpuMoEMXFP8Executor | 无 ascend MXFP8 MoE 实现 |
| DeepGEMM wrapper 修改 | has_deep_gemm/is_deep_gemm_e8m0_used 未改 |
| NCCL→HCCL (运行时) | collective_torch.py 仍用 nccl，backend_manager.py 仍 backend="nccl" |
| NZ 格式转换 | 无 npu_format_cast 调用 |

---

## 方案说明

> **MXFP8 是 W8A8 量化大类下 W8 子类的一种具体实现**

### 量化方案分类

```
W8A8（量化大类：Weight 8-bit + Activation 8-bit）
├── W8A8 完整方案（权重+激活都量化）
│   ├── W8A8_INT8（SmoothQuant）
│   └── W8A8_MXFP8（权重 FP8 + 激活 FP8）
├── W8 子类（权重-only：仅权重量化）
│   ├── W8_MXFP8 ← 当前 RTP-LLM 实现（Weight-only FP8，group_size=32）
│   ├── W8_INT8
│   └── ...
└── A8 子类（激活-only：仅激活量化）
```

### W8 和 A8 的定义

- **W8（Weight-only 8-bit）**：W8A8 量化方案的一个子类，仅对权重量化为 8-bit，激活保持高精度（如 BF16/FP16）
- **A8（Activation-only 8-bit）**：W8A8 量化方案的一个子类，仅对激活量化为 8-bit，权重保持高精度

### W8_MXFP8 实现细节

- **量化类型**：Weight-only 8-bit（W8 子类）
- **数据格式**：FP8 E4M3（权重）+ BF16（激活）
- **量化方式**：W8_MXFP8（group_size=32）+ E8M0 scale
- **命名来源**：与 vllm-ascend 的 `@register_scheme("W8A8_MXFP8", ...)` 命名一致
- **代码标记**：`# W8_MXFP8: Weight-only FP8 Quantization（W8A8 大类下）`

---

## 一、目标与范围

### 1.1 量化目标

| 维度 | RTP-LLM 原方案 | NPU 目标方案 |
|------|--------------|-------------|
| 量化方法 | FP8 Per-Block (group_size=128) | W8_MXFP8 (group_size=32) |
| 权重格式 | FP8 E4M3 + 手动 E8M0 scale | FP8 E4M3 + 原生 E8M0 scale |
| 推理算子 | DeepGEMM `fp8_gemm_nt` | `torch_npu.npu_quant_matmul`（W8_MXFP8） |
| 量化工具 | Load Quant（框架内） | Load Quant + ModelSlim（预量化） |

### 1.2 为什么选择 W8_MXFP8

1. **RTP-LLM 已有 FP8 Per-Block 实现**，W8_MXFP8 是其 NPU 原生等价方案
2. **对齐 vllm-ascend 标准**：group_size=32 与 vllm-ascend 实现一致
3. **NPU 原生支持**：`torch_npu.npu_dynamic_mx_quant` 原生支持 E8M0 scale
4. **适配工作量最小**：相比 INT4/INT8 方案，W8_MXFP8 改动最少

### 1.3 不在范围内

- INT8 / INT4 / FP4 等其他量化方法（见完整版概要）
- 激活动态量化（NPU 特有，后续引入）
- MLA KV Cache 量化（独立工作项）

---

## 二、适配工作总览

W8_MXFP8 移植涉及 **4 个层面**：

| 层面 | 当前实现 | NPU 目标 | 工作量 | 状态 |
|------|---------|---------|--------|------|
| 配置层 | `Fp8BlockWiseQuantConfig` | 保留，调整 dtype | 低 | ❌🔴 ModelSlimConfig 未适配 |
| 权重层 | `per_block_cast_to_fp8` + `requant_weight_ue8m0` | `npu_dynamic_mx_quant`，移除手动 E8M0 | 中 | ❌🔴 算子替换未完成 |
| 推理层 | DeepGEMM `fp8_gemm_nt` | `torch_npu.npu_quant_matmul`（W8_MXFP8） | 中 | ✅🟢 F16基线已适配 ❌🔴 W8_MXFP8未适配 |
| 桥接层 | `scaled_fp8_quant.cu` | 移除或用 `torch_npu` 替代 | 低 | ❌🔴 未适配 |

> ✅🟢 **Device 基础框架已在开发版本中适配**：`DeviceType.Ascend`、`AscendImpl` 基础类（含 `get_device_id`、`_get_mem_info`、`support_dio_load`）、注册机制、`is_ascend()` 函数均已实现。AscendImpl 尚未重写 W8_MXFP8 相关方法（`per_block_cast_to_fp8`、`requant_weight_ue8m0`），该部分仍在 ❌🔴 未适配状态。

---

## 三、配置层适配 ❌🔴

### 3.1 保留的配置类

```python
# rtp_llm/config/quant_config.py
class Fp8BlockWiseQuantConfig(QuantizationConfig):
    # 保留，仅需调整以下接口
```

### 3.2 需调整的接口

| 接口 | 当前实现 | NPU 适配 |
|------|---------|---------|
| `get_method()` | `FP8_PER_BLOCK` | 保留 |
| `is_quanted()` | `False`（Load Quant） | 保留 |
| `get_supported_compute_dtypes()` | `[torch.float16, torch.bfloat16]` + FP8 | 调整为 NPU 支持的 dtype |

### 3.3 可保留的部分

- `QuantizationConfig` 基类与注册表机制
- `is_quanted` 二分逻辑
- `from_config` / `load_from_ckpt` 入口

---

## 四、权重层适配 ❌🔴

### 4.1 `_postprocess` 算子替换

| CUDA 算子 | 功能 | NPU 替代方案 |
|-----------|------|-------------|
| `per_block_cast_to_fp8` | FP8 Per-Block 在线量化 | `torch_npu.npu_dynamic_mx_quant`（FP8 + E8M0） |
| `requant_weight_ue8m0` | E8M0 scale 重量化 | **移除**（NPU 原生支持 E8M0） |
| `convert_fp8_weight_params` | FP8 权重格式转换 | **移除或简化** |

### 4.2 关键改动：移除手动 E8M0 转换

**RTP-LLM 当前实现**（CUDA）：
```python
# 需要手动将 FP32 scale 转为 E8M0 格式
weight_fp8, weight_scale = per_block_cast_to_fp8(weight)
weight_scale = requant_weight_ue8m0(weight, weight_scale)  # ← 手动转换
output = deepgemm_fp8_gemm(input, weight_fp8, weight_scale)
```

**NPU 适配后**：
```python
# NPU 原生支持 E8M0，无需手动转换
weight_fp8, weight_scale = torch_npu.npu_dynamic_mx_quant(weight, dst_type=torch.float8_e4m3fn)
# weight_scale 直接是 E8M0 格式（uint8 存储，逻辑类型 float8_e8m0fnu）
output = torch_npu.npu_quant_matmul(input, weight_fp8, weight_scale)
```

**关键说明**：
- **group_size=32**：与 vllm-ascend 标准一致（默认值）
- **scale 存储格式**：uint8 物理存储，逻辑类型 float8_e8m0fnu (E8M0)
- **NPU API 参数**：通过 `scale_dtype=FLOAT8_E8M0FNU_DTYPE` 指定逻辑类型

### 4.3 可保留的部分

- `WeightModule` 四步加载流程
- `CompositeWeight` / `QuantWeight` 类层次
- TP 切分策略框架

---

## 五、推理层适配

### 5.0 已适配部分 ✅🟢

> ✅🟢 **AscendF16Linear 已在开发版本中适配**：`class AscendF16Linear(LinearBase)` 已实现并通过 `LinearFactory.register` 注册到 Ascend 设备，提供 BF16/FP16 基线推理能力。该部分已在开发版本中适配。
>
> ✅🟢 **MoE BF16 Fallback 已在开发版本中适配**：`class AscendBf16FallbackStrategy(MoeStrategy)` 已实现，并通过 `DeviceType.Ascend → AscendBf16FallbackStrategy` 注册到 fused_moe 工厂（fused_moe/__init__.py L62-70），提供 BF16 模式下的 MoE 推理降级方案。该部分已在开发版本中适配。

### 5.1 量化 GEMM 替换 ❌🔴

| 推理模块 | CUDA 实现 | NPU 替代 | 状态 |
|---------|----------|---------|------|
| W8_MXFP8 Linear | DeepGEMM `fp8_gemm_nt` | `torch_npu.npu_quant_matmul`（W8_MXFP8） | ❌🔴 |
| W8_MXFP8 MoE | DeepGEMM masked executor | `torch_npu.npu_grouped_matmul`（W8_MXFP8） | ❌🔴 |

### 5.2 代码改动示例 ❌🔴

**RTP-LLM 当前实现**（CUDA）：
```python
# fp8_deepgemm_linear.py
class Fp8DeepGemmLinear(Linear):
    def forward(self, x):
        # DeepGEMM FP8 Per-Block GEMM
        output = deepgemm.fp8_gemm_nt(
            x_fp8, self.weight_fp8,
            x_scale, self.weight_scale  # E8M0 scale
        )
        return output
```

**NPU 适配后**：
```python
# ascend/fp8_mxfp8_linear.py
class NpuFp8MXFP8Linear(Linear):
    def forward(self, x):
        # NPU W8_MXFP8 GEMM
        output = torch_npu.npu_quant_matmul(
            x_fp8, self.weight_fp8,
            self.weight_scale,  # 原生 E8M0 scale
            scale_dtype=FLOAT8_E8M0FNU_DTYPE,
            output_dtype=torch.bfloat16,
            group_sizes=[1, 1, 32]  # group_size=32
        )
        return output
```

### 5.3 目录结构

```
models_py/modules/factory/
├── linear/impl/
│   ├── cuda/
│   │   └── fp8_deepgemm_linear.py      # 现有 CUDA 实现
│   └── ascend/                          # ✅🟢 已创建（含 F16Linear）
│       ├── f16_linear.py                # ✅🟢 已适配
│       └── fp8_mxfp8_linear.py         # ❌🔴 待实现
├── fused_moe/impl/
│   ├── cuda/
│   └── ascend/                          # ✅🟢 已创建（含 BF16 Fallback）
```

---

## 六、桥接层适配 ❌🔴

### 6.1 需处理的 CUDA Kernel

| CUDA Kernel | 处理方式 |
|-------------|---------|
| `scaled_fp8_quant.cu` | **移除**，用 `torch_npu.npu_dynamic_mx_quant` 替代 |
| `librtp_compute_ops`（`per_block_cast_to_fp8` 等） | **移除**，NPU 原生算子替代 |

### 6.2 第三方库依赖

| 第三方库 | 处理方式 |
|---------|---------|
| DeepGEMM | **移除**，用 `torch_npu.npu_quant_matmul` 替代 |
| `torch._scaled_mm` | **移除**（MXFP8 不需要） |

---

## 七、ModelSlim 预量化支持（可选）❌🔴

除了 Load Quant（框架内运行时量化），还可支持 ModelSlim 预生成的 MXFP8 模型：

### 7.1 新增配置类

```python
class ModelSlimConfig(QuantizationConfig):
    @classmethod
    def get_method(cls):
        return "modelslim"

    def is_quanted(self):
        return True  # 预量化模型
```

### 7.2 新增权重加载器

```python
class ModelSlimWeight(QuantWeight):
    def _postprocess(self, tensor, param_info, quant_config, ...):
        # 1. 加载 ModelSlim MXFP8 权重
        # 2. Scale 参数处理（E8M0 格式）
        # 3. 无需 NZ 格式转换（FP8 不需要）
```

---

## 八、适配优先级

### P0（必须，阻塞推理）❌🔴

1. **推理层 GEMM 替换**：DeepGEMM → `torch_npu.npu_quant_matmul`（MXFP8）❌🔴
2. **权重层算子替换**：`per_block_cast_to_fp8` → `npu_dynamic_mx_quant` ❌🔴

### P1（简化代码）❌🔴

3. **移除手动 E8M0 转换**：删除 `requant_weight_ue8m0` 调用 ❌🔴
4. **移除 DeepGEMM 依赖**：删除 `fp8_deepgemm_linear.py` 中的 CUDA 调用 ❌🔴
5. **移除桥接层 CUDA kernel**：`scaled_fp8_quant.cu` ❌🔴

### P2（扩展功能）❌🔴

6. **ModelSlim 预量化支持**：新增 `ModelSlimConfig` + `ModelSlimWeight` ❌🔴
7. **配置层 dtype 调整**：`get_supported_compute_dtypes` ❌🔴

### 已完成的基础设施 ✅🟢

- **Device 层基础框架**：`DeviceType.Ascend`、`AscendImpl`、注册机制 ✅🟢
- **推理层 F16 基线**：`AscendF16Linear` ✅🟢
- **MoE BF16 Fallback**：`AscendBf16FallbackStrategy` + 注册 ✅🟢
- **HCCL 构建配置**：链接选项和编译宏 ✅🟢

---

## 九、总结

W8_MXFP8 移植的核心工作是 **3 个替换 + 1 个移除**：

| 序号 | 工作项 | 具体内容 | 状态 |
|------|--------|---------|------|
| 1 | **替换推理 GEMM** | DeepGEMM `fp8_gemm_nt` → `torch_npu.npu_quant_matmul` | ❌🔴 |
| 2 | **替换权重量化** | `per_block_cast_to_fp8` → `npu_dynamic_mx_quant` | ❌🔴 |
| 3 | **移除手动 E8M0** | 删除 `requant_weight_ue8m0`（NPU 原生支持） | ❌🔴 |
| 4 | **移除第三方依赖** | DeepGEMM / `scaled_fp8_quant.cu` | ❌🔴 |

**关键参数**：
- **方案名称**：W8_MXFP8（与 vllm-ascend 命名一致）
- **group_size**：32（vllm-ascend 标准）
- **scale 存储**：uint8（逻辑类型 float8_e8m0fnu/E8M0）

**优势**：W8_MXFP8 与 vllm-ascend 实现完全一致，对齐行业标准，适配工作量最小，是最优先落地的 NPU 量化方案。

**当前进展**：Device 基础框架、F16 推理基线、MoE BF16 Fallback、HCCL 构建配置已完成 ✅🟢，W8_MXFP8 核心替换（推理 GEMM、权重量化算子、E8M0 移除、DeepGEMM 移除）尚未适配 ❌🔴。
