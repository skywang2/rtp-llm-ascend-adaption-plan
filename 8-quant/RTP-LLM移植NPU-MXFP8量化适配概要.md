# RTP-LLM 移植 NPU W8A8_MXFP8 量化适配概要

> 本文档聚焦于将 RTP-LLM 的 **FP8 Per-Block 量化**移植到 NPU 的 **W8A8_MXFP8** 方案，是 [RTP-LLM移植NPU量化适配概要](./RTP-LLM移植NPU量化适配概要-v0.1.md) 的精简版。

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
| NZ 格式转换 / scale swizzle | FP8 权重无需 NZ（无 npu_format_cast 调用）；scale 布局 swizzle `[N,K//32]→[K//64,N,2]` 未实现 |

---

## 方案说明

> **W8A8_MXFP8 是 W8A8（权重+激活均 8-bit）量化方案的 MXFP8 实现，激活在运行时在线动态量化**

### 量化方案分类

```
W8A8（量化大类：权重+激活都量化为 8-bit）
├── W8A8_INT8（SmoothQuant）
├── W8A8_FP8（Per-Tensor / Per-Channel）
└── W8A8_MXFP8 ← 本方案（权重/激活 FP8 E4M3，group_size=32，E8M0 scale）

W8A16（仅权重量化，激活保持高精度）
├── W8_INT8 / W4_INT8 / ...
```

### W8 和 A8 的定义

- **W8（Weight 8-bit）**：权重量化为 8-bit
- **A8（Activation 8-bit）**：激活量化为 8-bit
- **W8A8**：权重与激活均为 8-bit；本方案的激活通过 `npu_dynamic_mx_quant` 运行时在线动态量化（非 weight-only）

### W8A8_MXFP8 实现细节

- **量化类型**：W8A8（权重量化 + 激活动态量化）
- **数据格式**：FP8 E4M3（权重）+ FP8 E4M3（激活，运行时在线动态 MX 量化）+ BF16（GEMM 输出）
- **量化方式**：group_size=32（沿 K 方向 1D 分组）+ E8M0 scale
- **命名来源**：与 vllm-ascend 的 `@register_scheme("W8A8_MXFP8", ...)` 命名一致
- **代码标记**：`# W8A8_MXFP8: Weight & Activation FP8 Quantization`

---

## 一、目标与范围

### 1.1 量化目标

| 维度 | RTP-LLM 原方案 | NPU 目标方案 |
|------|--------------|-------------|
| 量化方法 | FP8 Per-Block（128×128 二维块） | W8A8_MXFP8（1×32 一维组，沿 K，OCP 标准） |
| 激活处理 | 激活 FP8 在线量化（per-token-group） | 激活 FP8 在线动态 MX 量化（`npu_dynamic_mx_quant`） |
| 权重格式 | FP8 E4M3 + 手动 E8M0 scale | FP8 E4M3 + 原生 E8M0 scale（布局需 swizzle） |
| 推理算子 | DeepGEMM `fp8_gemm_nt` | `torch_npu.npu_quant_matmul`（W8A8_MXFP8） |
| 量化工具 | Load Quant（框架内） | Load Quant + ModelSlim（预量化） |

### 1.2 为什么选择 W8A8_MXFP8

1. **RTP-LLM 已有 FP8 Per-Block 实现**，W8A8_MXFP8 是其 NPU 侧的对位方案（量化粒度不同，见下文说明）
2. **对齐 vllm-ascend 标准**：1×32 分组与 vllm-ascend `W8A8_MXFP8` 实现一致，可直接复用其算子用法
3. **NPU 原生支持**：`torch_npu.npu_dynamic_mx_quant` 原生输出 E8M0 scale
4. **适配工作量最小**：相比 INT4/INT8 方案，W8A8_MXFP8 改动最少

> ⚠️ 注意：W8A8_MXFP8 包含**激活动态量化**（forward 中在线量化激活），这是 NPU MXFP8 GEMM 的必要输入，不属于"后续引入"项。

### 1.3 不在范围内

- INT8 / INT4 / FP4 等其他量化方法（见完整版概要）
- MLA KV Cache 量化（独立工作项）
- GPU FP8_PER_BLOCK 预量化 ckpt 在 NPU 上的加载（128×128 + FP32 scale 与 MXFP8 格式不兼容，需重新量化）

---

## 二、适配工作总览

W8A8_MXFP8 移植涉及 **4 个层面**：

| 层面 | 当前实现 | NPU 目标 | 工作量 | 状态 |
|------|---------|---------|--------|------|
| 配置层 | `Fp8BlockWiseQuantConfig` | 保留，调整 dtype | 低 | ❌🔴 ModelSlimConfig 未适配 |
| 权重层 | `per_block_cast_to_fp8` + `requant_weight_ue8m0` | `npu_dynamic_mx_quant`，移除手动 E8M0 | 中 | ❌🔴 算子替换未完成 |
| 推理层 | DeepGEMM `fp8_gemm_nt` | `torch_npu.npu_quant_matmul`（W8A8_MXFP8） | 中 | ✅🟢 F16基线已适配 ❌🔴 W8A8_MXFP8未适配 |
| 桥接层 | `scaled_fp8_quant.cu` | NPU 构建剥离（保留 CUDA 源码），运行时用 `torch_npu` 替代 | 低 | ❌🔴 未适配 |

> ✅🟢 **Device 基础框架已在开发版本中适配**：`DeviceType.Ascend`、`AscendImpl` 基础类（含 `get_device_id`、`_get_mem_info`、`support_dio_load`）、注册机制、`is_ascend()` 函数均已实现。AscendImpl 尚未重写 W8A8_MXFP8 相关方法（`per_block_cast_to_fp8`、`requant_weight_ue8m0`），该部分仍在 ❌🔴 未适配状态。

### 2.1 调用关系总览

**路径一：动态量化（Load Quant，is_quanted=False）**

```
 CLI --quantization FP8_PER_BLOCK
        │
        ▼
 QuantizationConfig.load_from_ckpt (quant_config.py)
        │
        ▼
 Fp8BlockWiseQuantConfig（复用；NPU 分支 block=32 由算子固定，config.group_size 不生效）
        │
        ▼
 LoadQuantPerBlockFp8Weight._load_raw_tensor (per_block_fp8_quant_weight.py L847)
        │   CUDA: per_block_cast_to_fp8（128×128 二维块）
        │   NPU : torch_npu.npu_dynamic_mx_quant  ★调用点①
        │         （逐权重调用：attn_qkv_w / attn_o_w / ffn_w1~w3 / moe_w1~w2）
        ▼
 PerBlockFp8Weight._postprocess (L780)
        │   CUDA: requant_weight_ue8m0 → NPU: 移除
        │   NPU 新增: scale 布局 swizzle [N,K//32]→[K//64,N,2]；权重不转置
        ▼
 LinearFactory / FusedMoEFactory（按 device + quant_config 分发）
        ├─► Dense: NpuFp8MXFP8Linear.forward
        │       ├─ npu_dynamic_mx_quant  ★调用点②（激活在线量化，每次 forward 每 Linear 各 1 次）
        │       └─ npu_quant_matmul      ★调用点③（每层 Dense Linear 每次 forward 1 次）
        └─► MoE : NpuMoEMXFP8Executor.execute
                ├─ 方案A（融合）: npu_grouped_matmul_swiglu_quant_v2  ★调用点④（每 MoE 层 1 次）
                └─ 方案B（降级）: 分步 grouped GEMM×2 + silu_mul_and_quant
                        └─ npu_dynamic_mx_quant  ★调用点⑤（中间激活量化）
```

**路径二：静态量化（ModelSlim 预量化，is_quanted=True）**

```
 ModelSlim 预量化工具 → quant_model_description.json
        │
        ▼
 QuantizationConfig.load_from_ckpt
        │   新增检测分支：发现 quant_model_description.json
        ▼
 ModelSlimConfig（新增，is_quanted=True）
        │
        ▼
 ModelSlimWeight._load_raw_tensor（新增：直接加载 FP8 权重 + E8M0 scale，不含量化算子调用）
        │   └─ scale 布局 swizzle [N,K//32]→[K//64,N,2]
        ▼
 与路径一共用推理层（NpuFp8MXFP8Linear / NpuMoEMXFP8Executor）
```

**多调用点标注表**

| NPU 算子 | 调用点 | 说明 |
|---------|--------|------|
| `npu_dynamic_mx_quant` | ★① 权重加载量化 | 仅动态量化路径；`_load_raw_tensor` 中对每个权重各调用 1 次 |
| `npu_dynamic_mx_quant` | ★② Dense 激活量化 | `NpuFp8MXFP8Linear.forward`，每次 forward 每 Linear 各 1 次 |
| `npu_dynamic_mx_quant` | ★⑤ MoE 中间激活量化 | 仅方案 B（降级分步路径）使用 |
| `npu_quant_matmul` | ★③ Dense GEMM | 每层 Dense Linear 每次 forward 1 次 |
| `npu_grouped_matmul_swiglu_quant_v2` | ★④ MoE 融合 GEMM | 每 MoE 层 executor 每次 forward 1 次（方案 A） |

### 2.2 跨平台兼容性约束（CUDA/ROCm 零影响）

本方案为增量适配，所有 NPU 逻辑收敛在设备分支内，CUDA/ROCm 原路径不变：

1. **共享模块**（`quant_config.py`、`per_block_fp8_quant_weight.py`、`deepgemm_wrapper.py`、`collective_torch.py`）禁止顶层 `import torch_npu`，仅 Ascend 分支内延迟导入（这些文件在所有平台无条件导入，CUDA 构建无 torch_npu）
2. **运行时分流**统一用 `is_ascend()` / 工厂设备门控；CUDA 分支代码逐字节不变
3. **NPU 专属类**（`NpuFp8MXFP8Linear`、`NpuMoEMXFP8Executor`）放 `impl/ascend/`，仅 Ascend 设备被工厂导入，可顶层 `import torch_npu`
4. **ModelSlim 检测**加 `is_ascend()` 门控；**通信 backend** 为 `"hccl" if is_ascend() else "nccl"`
5. **构建剥离**：DeepGEMM / `scaled_fp8_quant.cu` 按 bazel 架构配置从 NPU 构建剥离，不删 CUDA 源码（详见详设 §1.4）

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
| `get_supported_compute_dtypes()` | `[torch.bfloat16]`（已核实代码） | 无需修改（NPU 支持 BF16），仅验证 |
| `get_supported_kv_cache_dtypes()` | `[fp16, bf16, fp8_e4m3]` | NPU 不支持 KV FP8 时按 `is_ascend()` 分支，不得无条件修改 |

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
# NPU 原生支持 E8M0，无需数值转换（布局 swizzle 在加载时完成）
weight_fp8, weight_scale = torch_npu.npu_dynamic_mx_quant(weight, dst_type=torch.float8_e4m3fn)
# weight_scale 数值上已是 E8M0 格式（uint8 存储，逻辑类型 float8_e8m0fnu）
output = torch_npu.npu_quant_matmul(input_fp8, weight_fp8, weight_scale_swizzled)
```

**关键说明**：
- **量化粒度**：1×32 一维组（沿 K 方向），由 `npu_dynamic_mx_quant` 算子固定，与 vllm-ascend 标准一致；注意 CUDA 侧是 128×128 二维块，属粒度差异而非参数差异
- **scale 存储格式**：uint8 物理存储，逻辑类型 float8_e8m0fnu (E8M0)；量化输出 `[N, K//32]`，加载时 swizzle 为 `[K//64, N, 2]`（参照 vllm-ascend，见详设 §4.2）
- **权重布局**：加载时不转置，保持 `[N, K]`（CUDA 的 NT 转置是 DeepGEMM 专属，NPU 分支跳过）
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
| W8A8_MXFP8 Linear | DeepGEMM `fp8_gemm_nt` | `torch_npu.npu_quant_matmul`（W8A8_MXFP8） | ❌🔴 |
| W8A8_MXFP8 MoE | DeepGEMM masked executor | `torch_npu.npu_grouped_matmul_swiglu_quant_v2`（W8A8_MXFP8） | ❌🔴 |

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
        # W8A8_MXFP8：激活先在线动态量化
        x_fp8, pertoken_scale = torch_npu.npu_dynamic_mx_quant(
            x, dst_type=torch.float8_e4m3fn
        )
        # NPU W8A8_MXFP8 GEMM
        output = torch_npu.npu_quant_matmul(
            x_fp8, self.weight_fp8,
            self.weight_scale,  # E8M0 scale（加载时已 swizzle 为 [K//64, N, 2]）
            scale_dtype=FLOAT8_E8M0FNU_DTYPE,
            pertoken_scale=pertoken_scale,
            pertoken_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
            output_dtype=torch.bfloat16,
            group_sizes=[1, 1, 32]  # 1×32 分组，算子固定
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

> 处理原则：**NPU 构建剥离，保留 CUDA 源码**（本仓库 CUDA/NPU 路径共存，直接删除会破坏 CUDA 构建）。

### 6.1 需处理的 CUDA Kernel

| CUDA Kernel | 处理方式 |
|-------------|---------|
| `scaled_fp8_quant.cu` | NPU 路径不调用（用 `torch_npu.npu_dynamic_mx_quant` 替代），按 bazel 架构配置剥离出 NPU 构建 |
| `librtp_compute_ops`（`per_block_cast_to_fp8` 等） | 同上，NPU 原生算子替代 |

### 6.2 第三方库依赖

| 第三方库 | 处理方式 |
|---------|---------|
| DeepGEMM | NPU 不依赖（`torch_npu.npu_quant_matmul` 替代）；`has_deep_gemm()` 在 NPU 返回 False |
| `torch._scaled_mm` | MXFP8 路径不调用 |

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

> 判定依据：P0 = 缺少该项则量化推理无法跑通。条目与详细设计 §九 编号对齐。

### P0（必须，阻塞推理）❌🔴

1. **推理层 Dense GEMM 替换**：`NpuFp8MXFP8Linear` + `npu_quant_matmul`（含激活动态量化）❌🔴
2. **权重层算子替换与布局处理**：`per_block_cast_to_fp8` → `npu_dynamic_mx_quant`；NPU 分支不转置 + scale swizzle `[N,K//32]→[K//64,N,2]` ❌🔴
3. **DeepGEMM wrapper NPU 分流**：`has_deep_gemm()` / `is_deep_gemm_e8m0_used()` 在 NPU 返回 False ❌🔴
   （现状 `is_deep_gemm_e8m0_used()` 调用 `torch.cuda.get_device_capability()`，NPU 上抛异常，阻塞 `_postprocess` 权重加载，必须先改）
4. **MoE MXFP8 Executor**：`NpuMoEMXFP8Executor` + `npu_grouped_matmul_swiglu_quant_v2` ❌🔴
   （目标模型为 MoE 时本项同样阻塞推理，仅面向 Dense 模型时可后置）
5. **通信层运行时 NCCL→HCCL**：`init_process_group` backend ❌🔴
   （TP>1 部署时阻塞，仅单卡场景可后置）

### P1（简化代码，不做不阻塞）❌🔴

6. **移除手动 E8M0 转换**：删除 `requant_weight_ue8m0` 调用（第 3 项生效后即为死分支）❌🔴
7. **NPU 构建剥离 DeepGEMM / CUDA kernel 依赖**：`scaled_fp8_quant.cu` 等按 bazel 架构配置隔离，**保留 CUDA 源码**（不直接删除，避免破坏 CUDA 路径共存）❌🔴

### P2（扩展功能）❌🔴

8. **ModelSlim 预量化支持**：新增 `ModelSlimConfig` + `ModelSlimWeight` + `load_from_ckpt` 检测 ❌🔴
9. **配置层 dtype 调整**：`get_supported_compute_dtypes` 验证 NPU 支持 ❌🔴

### 已完成的基础设施 ✅🟢

- **Device 层基础框架**：`DeviceType.Ascend`、`AscendImpl`、注册机制 ✅🟢
- **推理层 F16 基线**：`AscendF16Linear` ✅🟢
- **MoE BF16 Fallback**：`AscendBf16FallbackStrategy` + 注册 ✅🟢
- **HCCL 构建配置**：链接选项和编译宏 ✅🟢

---

## 九、总结

W8A8_MXFP8 移植的核心工作是 **3 个替换 + 1 个移除**：

| 序号 | 工作项 | 具体内容 | 状态 |
|------|--------|---------|------|
| 1 | **替换推理 GEMM** | DeepGEMM `fp8_gemm_nt` → `torch_npu.npu_quant_matmul` | ❌🔴 |
| 2 | **替换权重量化** | `per_block_cast_to_fp8` → `npu_dynamic_mx_quant`（含不转置 + scale swizzle） | ❌🔴 |
| 3 | **移除手动 E8M0** | 删除 `requant_weight_ue8m0` 调用（NPU 原生支持） | ❌🔴 |
| 4 | **剥离第三方依赖** | DeepGEMM / `scaled_fp8_quant.cu`（NPU 构建剥离，保留 CUDA 源码） | ❌🔴 |

**关键参数**：
- **方案名称**：W8A8_MXFP8（与 vllm-ascend 命名一致；权重+激活均 FP8，激活动态量化）
- **量化粒度**：1×32 一维组（沿 K，`npu_dynamic_mx_quant` 算子固定，对应 vllm-ascend 标准）
- **scale 存储**：uint8（逻辑类型 float8_e8m0fnu/E8M0），布局加载时 swizzle 为 `[K//64, N, 2]`
- **权重布局**：`[N, K]` 不转置

**优势**：W8A8_MXFP8 与 vllm-ascend 实现完全一致，对齐行业标准，适配工作量最小，是最优先落地的 NPU 量化方案。

**当前进展**：Device 基础框架、F16 推理基线、MoE BF16 Fallback、HCCL 构建配置已完成 ✅🟢，W8A8_MXFP8 核心替换（推理 GEMM、权重量化算子、E8M0 移除、DeepGEMM 剥离）尚未适配 ❌🔴。
