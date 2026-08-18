# RTP-LLM 移植 NPU MXFP8 量化适配详细设计

> 本文档是 NPU MXFP8 量化适配的**可实施详细设计**，基于 [调用流程图](./RTP-LLM-MXFP8量化代码调用流程图.md) 和 [跟读指南](./RTP-LLM移植NPU-MXFP8量化适配跟读指南-v0.1.md)。

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

## 一、概述

### 1.1 目标

将 RTP-LLM 的 FP8 Per-Block 量化移植到 NPU W8A8_MXFP8，支持**静态量化**和**动态量化**两条路径。

### 1.2 核心替换

| 序号 | 工作项 | CUDA 实现 | NPU 替代 | 状态 |
|------|--------|----------|---------|------|
| 1 | 权重量化 | `per_block_cast_to_fp8` | `torch_npu.npu_dynamic_mx_quant` | ❌🔴 |
| 2 | Linear GEMM | `deep_gemm.fp8_gemm_nt` | `torch_npu.npu_quant_matmul` | ❌🔴 |
| 3 | MoE GEMM | `deep_gemm.masked_fp8_gemm` | `torch_npu.npu_grouped_matmul_swiglu_quant_v2` | ❌🔴 |
| 4 | E8M0 转换 | `requant_weight_ue8m0` | **移除**（NPU 原生） | ❌🔴 |

### 1.3 范围

| 路径 | 预处理 | 配置层 | 权重层 | 推理层 |
|------|--------|--------|--------|--------|
| 静态量化 | ModelSlim ❌🔴 | `ModelSlimConfig` ➕❌🔴 | `ModelSlimWeight` ➕❌🔴 | `NpuFp8MXFP8Linear` ➕❌🔴 |
| 动态量化 | 无 | `Fp8BlockWiseQuantConfig` ✅🟢 | `LoadQuantPerBlockFp8Weight` 🔄❌🔴 | 同静态 |

### 1.4 跨平台兼容性约束（CUDA / ROCm 不受影响）

本方案为**增量适配**：所有 NPU 逻辑均在设备分支内完成，CUDA/ROCm 原有路径零修改。实施时必须遵守以下约束：

**约束 1：共享模块禁止顶层 `import torch_npu`**

`per_block_fp8_quant_weight.py`（经 `model_loader/__init__.py:13` 无条件导入）、`quant_config.py`、`deepgemm_wrapper.py`、`collective_torch.py` 等文件在**所有平台**都会被导入，`import torch_npu` 只能写在 Ascend 分支内部（延迟导入），与仓库现有惯例一致（参考 `server_config_setup.py`、`loader.py:444`）。

**约束 2：所有运行时分流使用 `is_ascend()` / `isinstance(exported_device, AscendImpl)`**

分支外的 CUDA/ROCm 代码路径保持逐字节不变；`has_deep_gemm()` / `is_deep_gemm_e8m0_used()` 仅在函数头部追加 `if is_ascend(): return False`，CUDA 分支逻辑不动。

**约束 3：NPU 专属类放在设备目录，利用既有工厂门控**

`NpuFp8MXFP8Linear` / `NpuMoEMXFP8Executor` 放入 `impl/ascend/`，仅在 `get_device_type() == DeviceType.Ascend` 时被工厂导入（`linear/__init__.py:24`、`fused_moe/__init__.py:62-70` 已有该分支）。此类文件**允许**顶层 `import torch_npu`。

**约束 4：配置注册与 ckpt 检测加设备门控**

`ModelSlimConfig` 检测（`quant_model_description.json`）仅在 `is_ascend()` 时生效，避免 CUDA 平台误入 ModelSlim 加载路径。

**约束 5：构建层剥离而非删源码**

DeepGEMM / `scaled_fp8_quant.cu` 等通过 bazel 架构配置从 NPU 构建中剥离，CUDA 源码与测试保持原样。

| 共享文件 | 修改方式 | CUDA/ROCm 影响 |
|---------|---------|---------------|
| `quant_config.py` | 新增 `ModelSlimConfig` + `is_ascend()` 门控检测分支 | 无（分支不进入） |
| `per_block_fp8_quant_weight.py` | `is_ascend()` 分支 + 分支内延迟导入 torch_npu | 无（else 分支为原代码） |
| `deepgemm_wrapper.py` | 两函数头部追加 `if is_ascend(): return False` | 无 |
| `device_impl.py` | `AscendImpl` 追加 MXFP8 方法（纯 torch / torch_npu 仅限方法体内） | 无（不在 CUDA 实例化） |
| `collective_torch.py` / `backend_manager.py` | `backend = "hccl" if is_ascend() else "nccl"` | 无 |
| `impl/ascend/*`（新增） | 顶层 import torch_npu | 无（非 Ascend 设备不导入） |
| bazel 构建 | NPU 配置剥离 DeepGEMM 等依赖 | 无（CUDA 配置不变） |

---

## 二、Device 层适配

### 2.1 新增 DeviceType ✅🟢

> ✅🟢 **该部分已在开发版本中适配**：`DeviceType.Ascend = 6` 已定义于 `device_type.py` L14，`is_ascend()` 函数已实现于 `device_type.py` L44-45。

**文件**：`rtp_llm/device/device_type.py` L7-13

```python
class DeviceType(IntEnum):
    Cpu = 0
    Cuda = 1
    ...
    Ascend = 6  # ✅🟢 已适配
```

### 2.2 新增 AscendImpl ✅🟢 基础框架已适配 / ❌🔴 MXFP8 方法未适配

> ✅🟢 **AscendImpl 基础框架已在开发版本中适配**：`class AscendImpl(GpuImpl)` 已实现于 `device_impl.py` L696-710，包含 `get_device_id`、`_get_mem_info`、`support_dio_load` 方法。该部分已在开发版本中适配。
>
> ❌🔴 **MXFP8 相关方法尚未适配**：以下 `per_block_cast_to_fp8`、`convert_fp8_weight_params`、`requant_weight_ue8m0` 等方法尚未重写，仍保持 CUDA 原实现。

**文件**：`rtp_llm/device/device_impl.py`

**设计**：继承 `GpuImpl`，重写 MXFP8 相关方法，不重写的方法自动继承。

```python
class AscendImpl(GpuImpl):
    """NPU 设备实现，参考 RocmImpl (L696) 的继承模式"""

    @property
    def arch(self) -> int:
        # NPU 架构版本，用于条件判断
        return 910  # Ascend 910

    def per_block_cast_to_fp8(self, weight, group_size=None):
        """🔄 ❌🔴 替换：使用 NPU 原生 W8A8_MXFP8 量化算子

        CUDA 原实现: per_block_cast_to_fp8(weight, group_size) → FP8 + FP32 scale（128×128 二维块）
        NPU 替代: torch_npu.npu_dynamic_mx_quant → FP8 + E8M0 scale（1×32 一维组，沿 K）

        注意：量化粒度由算子固定为 1×32（OCP MXFP8 标准），group_size 参数
        仅为兼容基类签名保留，NPU 分支不生效（config 中 128 为 CUDA 侧默认值，不修改）。

        Args:
            weight: BF16/FP16 权重 tensor [M, N]
            group_size: 忽略（保留签名兼容；如显式传入非 32 的值应告警）

        Returns:
            (weight_fp8, scale)
            weight_fp8: torch.float8_e4m3fn [M, N]
            scale: torch.uint8 [M, N//32] (存储为 uint8，逻辑类型 E8M0；
                   布局 swizzle [N,K//32]→[K//64,N,2] 在权重加载后处理完成)
        """
        if group_size is not None and group_size != 32:
            logger.warning(f"NPU MXFP8 quantization fixes group size to 32, got {group_size}")
        # ⚠️ 延迟导入：device_impl.py 经 device/__init__.py 在所有平台导入，
        # 顶层禁止 import torch_npu（与现状一致，现状该文件无 torch_npu 依赖）
        import torch_npu

        weight_fp8, scale = torch_npu.npu_dynamic_mx_quant(
            weight,
            dst_type=torch.float8_e4m3fn  # 输出 dtype
        )
        return weight_fp8, scale

    def convert_fp8_weight_params(self, weight, weight_scale):
        """🔄 ❌🔴 替换：NPU 无需 FP8 格式转换（e4m3fnuz 等）

        CUDA 原实现: ROCm 需要转 e4m3fnuz, scale × 2
        NPU: 直接返回，无需转换
        """
        return [weight, weight_scale]

    # 🗑️ ❌🔴 不实现以下方法（NPU 原生支持 E8M0 + MX 格式）
    # def requant_weight_ue8m0(self, weight, scale):  # 不需要
    # def swizzle_blockscale(self, scale):            # 不需要

    # ✅ 保留以下方法（其他量化方法使用，不影响 MXFP8）
    # apply_int8 (L215)
    # moe_apply_int8 (L224)
    # preprocess_groupwise_weight_params (L242)
    # convert_fp4_gemm_weight_params (L510)
```

### 2.3 注册 AscendImpl ✅🟢

> ✅🟢 **该部分已在开发版本中适配**：`DeviceType.Ascend → AscendImpl` 注册已实现于 `device/__init__.py` L22-23。该部分已在开发版本中适配。

**文件**：`rtp_llm/device/__init__.py` L11-23

```python
def get_device_cls(type: DeviceType):
    if type == DeviceType.Cuda:
        return CudaImpl
    elif type == DeviceType.ROCm:
        return RocmImpl
    elif type == DeviceType.Ascend:   # ✅🟢 已适配
        return AscendImpl
    ...
```

### 2.4 DeepGEMM Wrapper 适配 ❌🔴

**文件**：`rtp_llm/models_py/kernels/cuda/deepgemm_wrapper.py`

```python
# 🗑️ ❌🔴 NPU 场景下这些函数返回 False（用 is_ascend() 判断，不用 arch 魔数）
@functools.cache
def has_deep_gemm() -> bool:
    if is_ascend():  # NPU
        return False
    return has_module("deep_gemm")

@functools.cache
def is_deep_gemm_e8m0_used() -> bool:
    if is_ascend():  # NPU
        return False  # NPU 原生 E8M0，不需要 DeepGEMM 的 E8M0 转换
    return torch.cuda.get_device_capability()[0] in [10, 12]
```

---

## 三、配置层适配 ❌🔴

### 3.1 保留 Fp8BlockWiseQuantConfig ✅🟢

**文件**：`rtp_llm/config/quant_config.py` L376-404

```python
class Fp8BlockWiseQuantConfig(QuantizationConfig):
    # ✅ 保留 128 不变：该常量同时服务 CUDA 侧 FP8 Per-Block（128×128 块），
    # 全局改成 32 会破坏 CUDA 语义。NPU MXFP8 的 1×32 分组由
    # npu_dynamic_mx_quant 算子固定，config.group_size 在 NPU 分支不生效。
    DEFAULT_FP8_QUANT_BLOCK_SIZE = 128

    @classmethod
    def get_method(cls) -> str:
        return "FP8_PER_BLOCK"  # ✅ 保留

    @classmethod
    def get_algo(cls) -> str:
        return "fp8"  # ✅ 保留

    def get_supported_compute_dtypes(self) -> List[torch.dtype]:
        # ✅ 现状即返回 [bfloat16]（已核实 quant_config.py），NPU 支持 BF16，
        # 无需修改返回值、无需平台分支，仅在 NPU 上验证校验逻辑通过即可
        return [torch.bfloat16]

    def get_supported_kv_cache_dtypes(self) -> List[torch.dtype]:
        # 🔄 ❌🔴 验证 NPU KV Cache FP8 支持；若不支持则按 is_ascend() 分支
        # 返回 [float16, bfloat16]，不得无条件修改（会影响 CUDA 现有行为）
        return [torch.float16, torch.bfloat16, torch.float8_e4m3fn]
```

### 3.2 修改 load_from_ckpt — 新增 ModelSlim 检测 ❌🔴

**文件**：`rtp_llm/config/quant_config.py` L99-273

在 `load_from_ckpt` 方法中，**在 smoothquant.ini 检测之后**新增 ModelSlim 检测分支：

```python
@classmethod
def load_from_ckpt(cls, ckpt_path: str) -> Optional["QuantizationConfig"]:
    # ... 现有 smoothquant.ini / pertensorquant.ini 检测 ...

    # ➕ ❌🔴 新增: ModelSlim 检测（仅 Ascend 生效，不影响 CUDA/ROCm 检测逻辑；
    # ModelSlim 为昇腾量化工具，其它平台不应进入该路径）
    if is_ascend():
        modelslim_config_path = os.path.join(ckpt_path, "quant_model_description.json")
        if os.path.exists(modelslim_config_path):
            with open(modelslim_config_path, "r") as f:
                quant_desc = json.load(f)
            modelslim_config = ModelSlimConfig.from_config(quant_desc)
            if modelslim_config is not None:
                return modelslim_config

    # ... 现有 config.json 检测 ...
```

> `is_ascend()` 来自 `device_type.py`（轻量、无 torch_npu 顶层依赖），quant_config.py 可安全导入。

### 3.3 新增 ModelSlimConfig ❌🔴

**文件**：`rtp_llm/config/quant_config.py` ➕ 新增

```python
class ModelSlimConfig(QuantizationConfig):
    """ModelSlim 预量化模型配置

    ModelSlim 输出 quant_model_description.json，包含每层量化类型:
    - W8A8: 权重和激活均为 INT8/FP8
    - W8A8_DYNAMIC: 动态激活量化
    - FLOAT: 未量化
    """

    def __init__(self, bits=8, group_size=32, is_quanted=True, **kwargs):
        super().__init__(bits=bits, group_size=group_size, is_quanted=is_quanted)

    @classmethod
    def get_method(cls) -> str:
        return "modelslim"

    @classmethod
    def get_algo(cls) -> str:
        return "fp8"  # MXFP8 算法标识

    def is_quanted(self) -> bool:
        return True  # 预量化模型

    @classmethod
    def from_config(cls, quant_desc: Dict[str, Any]) -> "ModelSlimConfig":
        """从 quant_model_description.json 解析配置

        Args:
            quant_desc: quant_model_description.json 的内容
                包含每层的量化类型 (W8A8, W4A8, FLOAT 等)

        Returns:
            ModelSlimConfig 实例
        """
        # 检查是否包含 MXFP8 层
        # 注意：仅匹配显式标注 MXFP8 的层；裸 "W8A8" 可能是 INT8/动态 INT8 方案，
        # 不能按 MXFP8 加载（ModelSlim 同时支持 W8A8-INT8 与 W8A8_MXFP8）
        has_mxfp8 = any(
            "MXFP8" in v.upper()
            for k, v in quant_desc.items()
            if isinstance(v, str) and k != "fa_quant_type"
        )
        if not has_mxfp8:
            return None  # 不包含 MXFP8 量化层，交由其它配置类处理

        return cls(bits=8, group_size=32, is_quanted=True)
```

### 3.4 注册 ModelSlimConfig ❌🔴

ModelSlimConfig 继承 `QuantizationConfig`，通过 `__init_subclass__` 自动注册到 `_registry`，`from_config` 和 `load_from_ckpt` 会自动匹配。

---

## 四、权重层适配 ❌🔴

### 4.1 静态量化：修改 PerBlockFp8Weight._postprocess ❌🔴

**文件**：`rtp_llm/model_loader/per_block_fp8_quant_weight.py` L761-808

**改动**：移除 `requant_weight_ue8m0` 调用

```python
def _postprocess(self, tensor, device, load_config):
    processed_res = super()._postprocess(tensor, device, load_config)
    kernel_weight = processed_res[self.kernel.name]

    # ✅ NPU 场景: is_deep_gemm_e8m0_used() 返回 False, 走非 E8M0 路径
    if not is_deep_gemm_e8m0_used():
        kernel_weight = kernel_weight.reshape(
            kernel_weight.shape[-1], -1
        ) if kernel_weight.dim() == 2 else kernel_weight
    processed_res[self.kernel.name] = kernel_weight

    if self.scale is not None:
        scale_weight = processed_res[self.scale.name]

        if not is_deep_gemm_e8m0_used():
            scale_weight = scale_weight.reshape(
                scale_weight.shape[-1], -1
            ) if scale_weight.dim() == 2 else scale_weight

        # ✅ 保留: device 特定的权重重写
        kernel_weight = load_config.exported_device.maybe_rewrite_weight_by_key(
            "weight", kernel_weight
        )
        scale_weight = load_config.exported_device.maybe_rewrite_weight_by_key(
            "scale", scale_weight
        )

        # 🗑️ ❌🔴 移除这段: NPU 不需要 requant_weight_ue8m0
        # if is_deep_gemm_e8m0_used():
        #     kernel_weight, scale_weight = requant_weight_ue8m0(
        #         kernel_weight, scale_weight
        #     )

        processed_res[self.scale.name] = scale_weight
        processed_res[self.kernel.name] = kernel_weight

    return processed_res
```

**关键点**：
- NPU 场景下 `has_deep_gemm()` 和 `is_deep_gemm_e8m0_used()` 均返回 False
- 所以 NPU 走非 E8M0 路径，`requant_weight_ue8m0` 分支自然不执行
- 但建议**直接删除**这段死代码，保持代码清晰

### 4.2 动态量化：修改 LoadQuantPerBlockFp8Weight._load_raw_tensor ❌🔴

**文件**：`rtp_llm/model_loader/per_block_fp8_quant_weight.py` L854-886

**改动**：替换 `per_block_cast_to_fp8` 为 NPU 算子

```python
def _load_raw_tensor(self, tensor_source, layer_id, device, load_config):
    # 加载原始 BF16 权重
    kernel = self.kernel._load_raw_tensor(tensor_source, layer_id, device, load_config)

    res = {}
    scale = None
    # 用 is_ascend()（device_type.py，轻量无 torch_npu 依赖）分流，
    # 不 import AscendImpl，避免共享模块引入设备耦合
    is_ascend_dev = is_ascend()
    if self.scale:
        # 🔄 ❌🔴 NPU 替换: per_block_cast_to_fp8 → npu_dynamic_mx_quant
        if is_ascend_dev:
            # ⚠️ 延迟导入：本文件经 model_loader/__init__.py 在所有平台导入，
            # 顶层禁止 import torch_npu（CUDA 构建无此包）
            import torch_npu
            from rtp_llm.models_py.kernels.ascend.mx_layout import (
                swizzle_scale_to_npu_layout,
            )
            # NPU 路径: 原生 W8A8_MXFP8 量化（1×32 沿 K，算子固定），
            # 直接输出 E8M0 scale；self.group_size（CUDA 默认 128）在 NPU 分支不生效
            quant_kernel, scale = torch_npu.npu_dynamic_mx_quant(
                kernel.get(self.kernel.name),
                dst_type=torch.float8_e4m3fn
            )
        else:
            # CUDA 路径: 保留原实现（128×128 二维块）
            quant_kernel, scale = per_block_cast_to_fp8(
                kernel.get(self.kernel.name), self.group_size
            )

        if quant_kernel.dim() == 2:
            scale = scale.reshape([scale.shape[0], -1])
    else:
        quant_kernel = cast_to_fp8(kernel.get(self.kernel.name))

    if is_ascend_dev:
        # NPU 分支: 不转置（NT 转置是 DeepGEMM 专属），dense 权重保持 [N, K]，
        # MoE 权重保持 [E, N, K]；scale swizzle [N,K//32] → [K//64,N,2]（参照 vllm-ascend）
        if self.scale is not None and quant_kernel.dim() == 2:
            scale = swizzle_scale_to_npu_layout(scale)  # [N,K//32] → [K//64,N,2]
    else:
        # CUDA 分支: 转置 (非 MoE)
        if self.kernel.name == W.moe_w1 or self.kernel.name == W.moe_w2:
            pass
        elif quant_kernel.dim() == 2:
            quant_kernel = quant_kernel.T
        if self.scale:
            scale = scale.T if scale.dim() == 2 else scale

    res = {self.kernel.name: quant_kernel.contiguous().to(device)}
    if self.scale:
        res.update({self.scale.name: scale.contiguous().to(device)})

    return res
```

**scale 布局 swizzle 辅助函数**（新增于 `models_py/kernels/ascend/mx_layout.py`；**纯 torch 实现，该文件不得 import torch_npu**，供共享模块在 Ascend 分支内延迟导入）：

```python
def swizzle_scale_to_npu_layout(scale: torch.Tensor) -> torch.Tensor:
    """E8M0 scale 布局转换: [N, K//32] → [K//64, N, 2]

    npu_dynamic_mx_quant 输出 [N, K//32]（每行 K 方向每 32 元素 1 个 scale）；
    NPU GEMM 算子要求 [K//64, N, 2] 布局（与 vllm-ascend W8A8_MXFP8 权重
    处理一致）。仅重排内存布局，不改变 E8M0 数值。

    注意: 具体 reshape/permute 顺序需在 NPU 实机上用非方阵 GEMM 数值对齐
    后固化（P0 验证项）。
    """
    N, K_div_32 = scale.shape
    # [N, K//32] → [N, K//64, 2] → [K//64, N, 2]
    return scale.view(N, K_div_32 // 2, 2).permute(1, 0, 2).contiguous()
```

**关键点**：
- NPU 的 `npu_dynamic_mx_quant` 直接输出 E8M0 数值格式的 scale（uint8 存储，逻辑类型 `float8_e8m0fnu`）
- 不需要后续的 `requant_weight_ue8m0` 数值转换
- **NPU 分支跳过 `.T` 转置**：CUDA 的权重/scale 转置是为 DeepGEMM NT 布局服务的；NPU 侧 dense 权重保持 `[N, K]`，scale 需 swizzle 为 `[K//64, N, 2]`（与 vllm-ascend `W8A8_MXFP8` 的权重处理一致）
- 量化粒度差异：CUDA `per_block_cast_to_fp8` 是 128×128 二维块，NPU 是 1×32 一维组（算子固定），`self.group_size` 在 NPU 分支不参与计算
- CUDA 路径保持不变，通过 `is_ascend()` 分流（不 import AscendImpl，避免共享模块引入设备耦合）

### 4.3 新增 ModelSlimWeight ❌🔴

**文件**：`rtp_llm/model_loader/modelslim_weight.py` ➕ 新建

```python
from rtp_llm.model_loader.weight_module import CompositeWeight, QuantWeight
from rtp_llm.model_loader.atomic_weight import AtomicWeight
# ⚠️ 禁止顶层 import torch_npu：本文件经 model_loader 在所有平台导入；
# 且本类逻辑（加载/swizzle）为纯 torch，不需要 torch_npu。
# swizzle 复用 kernels/ascend/mx_layout.py（同样无 torch_npu 依赖），分支内延迟导入。


class ModelSlimWeight(CompositeWeight, QuantWeight):
    """ModelSlim 预量化权重加载器

    与 PerBlockFp8Weight 的区别:
    - ModelSlim 权重已是 FP8 + E8M0 scale 格式（1×32 分组）
    - 无需 requant_weight_ue8m0 数值转换
    - 权重保持 [N, K] 不转置；scale 仍需布局 swizzle [N,K//32]→[K//64,N,2]
    """

    # 权重名称映射 (参考 PerBlockFp8Weight.w8a8_weight_list)
    modelslim_weight_list: Dict[str, str] = {
        W.attn_qkv_w: W.attn_qkv_s,
        W.attn_o_w: W.attn_o_s,
        W.ffn_w1: W.ffn_s1,
        W.ffn_w2: W.ffn_s2,
        W.ffn_w3: W.ffn_s3,
        W.moe_w1: W.moe_s1,
        W.moe_w2: W.moe_s2,
    }

    @classmethod
    def support(cls, quant_config, src_weight_info) -> bool:
        if not isinstance(quant_config, ModelSlimConfig):
            return False
        if not quant_config.is_quanted():
            return False
        name = src_weight_info.name
        return name in cls.modelslim_weight_list

    def __init__(self, src_weight_info, quant_config, *args, **kwargs):
        self.group_size = quant_config.group_size()
        # 创建 kernel 和 scale 子权重
        params = src_weight_info.extract_params(...)
        kernel = AtomicWeight(name=src_weight_info.name, data_type=torch.float8_e4m3fn, ...)
        sub_weights = {kernel.name: kernel}

        scale_name = self.modelslim_weight_list.get(src_weight_info.name)
        if scale_name:
            scale = AtomicWeight(name=scale_name, data_type=torch.float8_e8m0fnu, ...)
            sub_weights[scale.name] = scale

        CompositeWeight.__init__(self, sub_weights, quant_config=quant_config, ...)
        self.kernel = kernel
        self.scale = scale

    def _load_raw_tensor(self, tensor_source, layer_id, device, load_config):
        """直接加载 ModelSlim 预量化的 FP8 权重"""
        kernel = self.kernel._load_raw_tensor(tensor_source, layer_id, device, load_config)
        res = {self.kernel.name: kernel.get(self.kernel.name)}

        if self.scale:
            scale = self.scale._load_raw_tensor(tensor_source, layer_id, device, load_config)
            res[self.scale.name] = scale.get(self.scale.name)

        return res

    def _postprocess(self, tensor, device, load_config):
        """后处理: 保持 [N, K] 布局 + scale swizzle, 无需 requant"""
        processed_res = super()._postprocess(tensor, device, load_config)
        kernel_weight = processed_res[self.kernel.name]

        # NPU 布局约定: dense 权重保持 [N, K] 不转置/不 reshape（与 CUDA 非 E8M0
        # 路径的 reshape(K, N) 不同），MoE 权重保持 [E, N, K]

        if self.scale is not None:
            scale_weight = processed_res[self.scale.name]
            if scale_weight.dim() == 2:
                # scale 布局 swizzle: [N, K//32] → [K//64, N, 2]
                # （uint8 存储，逻辑类型 E8M0；与 vllm-ascend W8A8_MXFP8 一致）
                # 延迟导入（ModelSlimWeight 仅 Ascend 选中，此处为双保险）
                from rtp_llm.models_py.kernels.ascend.mx_layout import (
                    swizzle_scale_to_npu_layout,
                )
                scale_weight = swizzle_scale_to_npu_layout(scale_weight)
            processed_res[self.scale.name] = scale_weight

        # 🗑️ 无需 requant_weight_ue8m0 — ModelSlim 已输出 E8M0 数值格式
        # ⚠️ scale 布局 swizzle 仍需要（见上）

        return processed_res
```

---

## 五、推理层适配

### 5.0 已适配部分 ✅🟢

> ✅🟢 **AscendF16Linear 已在开发版本中适配**：`class AscendF16Linear(LinearBase)` 已实现并通过 `LinearFactory.register` 注册到 Ascend 设备，提供 BF16/FP16 基线推理能力。该部分已在开发版本中适配。
>
> ✅🟢 **MoE BF16 Fallback 已在开发版本中适配**：`class AscendBf16FallbackStrategy(MoeStrategy)` 已实现，并通过 `DeviceType.Ascend → AscendBf16FallbackStrategy` 注册到 fused_moe 工厂（fused_moe/__init__.py L62-70），提供 BF16 模式下的 MoE 推理降级方案。该部分已在开发版本中适配。

### 5.1 Factory 选择机制 ✅🟢 部分 / ❌🔴 MXFP8 未适配

> ✅🟢 **Factory 基础机制已在开发版本中适配**：Ascend 设备分支已注册，`impl/ascend/` 目录已创建并包含 F16Linear 和 BF16 MoE Fallback。
> ❌🔴 **MXFP8 推理层尚未适配**：`NpuFp8MXFP8Linear` 和 `NpuMoEMXFP8Executor` 未实现。

**文件**：`rtp_llm/models_py/modules/factory/linear/__init__.py` L19-27

```python
# ➕ ✅🟢 已适配 NPU 分支
if device_type == DeviceType.Cuda:
    from .impl.cuda import *
elif device_type == DeviceType.Ascend:
    from .impl.ascend import *  # ✅🟢 已适配（含 F16Linear）
```

**目录结构**：

```
models_py/modules/factory/linear/impl/
├── cuda/
│   ├── __init__.py
│   ├── fp8_deepgemm_linear.py      # CUDA DeepGEMM (NPU 不使用)
│   ├── fp8_gemm_linear.py          # CUDA 入口策略
│   └── f16_linear.py               # 基线
└── ascend/                          # ✅🟢 已创建
    ├── __init__.py                  # ✅🟢 已适配
    ├── fp8_mxfp8_linear.py          # ❌🔴 待实现
    └── f16_linear.py                # ✅🟢 已适配
```

### 5.2 新增 NpuFp8MXFP8Linear ❌🔴

**文件**：`rtp_llm/models_py/modules/factory/linear/impl/ascend/fp8_mxfp8_linear.py` ➕ 新建

**CUDA 原实现**：`fp8_deepgemm_linear.py` L171-236

```python
# 本文件位于 impl/ascend/，仅当 DeviceType.Ascend 时被工厂导入
# （linear/__init__.py 设备分支），因此允许顶层 import torch_npu
import torch
import torch_npu
from rtp_llm.models_py.modules.factory.linear.linear import Linear


class NpuFp8MXFP8Linear(Linear):
    """NPU W8A8_MXFP8 Linear 层

    替换 CUDA CudaFp8DeepGEMMLinear (fp8_deepgemm_linear.py L171)

    CUDA 原流程:
        1. sgl_per_token_group_quant_fp8(input) → 在线量化输入
        2. deep_gemm.fp8_gemm_nt(input_fp8, weight_fp8, scales) → GEMM

    NPU 新流程:
        1. torch_npu.npu_dynamic_mx_quant(input) → 在线量化输入
        2. torch_npu.npu_quant_matmul(input_fp8, weight_fp8, scale) → GEMM
    """

    @classmethod
    def can_handle(cls, quant_config, weight, weight_scales, **kwargs) -> bool:
        """匹配条件: W8A8_MXFP8 量化 + FP8 权重"""
        if weight_scales is None or quant_config is None:
            return False
        if weight.dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            return False
        # 匹配 Fp8BlockWiseQuantConfig 和 ModelSlimConfig
        method = quant_config.get_method()
        return method in ("FP8_PER_BLOCK", "modelslim")

    def __init__(self, weight, weight_scales, bias=None, **kwargs):
        super().__init__(weight=weight, weight_scales=weight_scales, bias=bias)
        self.weight = weight          # FP8 E4M3 [N, K]
        self.weight_scale = weight_scales  # E8M0 [N, K//32] (uint8存储)
        self.bias = bias

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """W8A8_MXFP8 GEMM 前向

        Args:
            input: BF16 [M, K] 或 FP8 [M, K]

        Returns:
            BF16 output [M, N]
        """
        M, K = input.shape

        # 1. 在线量化输入 (如果输入是 BF16)
        if input.dtype == torch.bfloat16:
            input_fp8, pertoken_scale = torch_npu.npu_dynamic_mx_quant(
                input,
                dst_type=torch.float8_e4m3fn
            )
        elif input.dtype == torch.float8_e4m3fn:
            input_fp8 = input
            # 构造全 1.0 的 scale (E8M0 格式)
            # 注意: E8M0 为纯指数格式（bias=127），1.0 = 2^0 → 编码 0x7F；
            # 不能用 torch.ones（0x01 = 2^-126，会引入 1e38 量级误差）
            pertoken_scale = torch.full(
                (M, K // 32), 0x7F,
                dtype=torch.uint8,
                device=input.device
            )
        else:
            raise ValueError(f"Unsupported input dtype: {input.dtype}")

        # 2. NPU W8A8_MXFP8 GEMM
        output = torch_npu.npu_quant_matmul(
            input_fp8,           # [M, K] FP8
            self.weight,         # [N, K] FP8
            self.weight_scale,   # [N, K//32] E8M0 (uint8存储)
            scale_dtype=FLOAT8_E8M0FNU_DTYPE,
            pertoken_scale=pertoken_scale,
            pertoken_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
            bias=self.bias,
            output_dtype=torch.bfloat16,
            group_sizes=[1, 1, 32]  # group_size=32
        )

        return output  # [M, N] BF16
```

### 5.3 新增 NpuMoEMXFP8Executor ❌🔴

**文件**：`rtp_llm/models_py/modules/factory/fused_moe/impl/ascend/executors/npu_mxfp8_executor.py` ➕ 新建

**CUDA 原实现**：`deepgemm_masked_executor.py` L147-351

```python
# 本文件位于 impl/ascend/，仅当 DeviceType.Ascend 时被工厂导入，允许顶层 import torch_npu
import torch
import torch_npu


class NpuMoEMXFP8Executor:
    """NPU W8A8_MXFP8 MoE Executor

    替换 CUDA DeepGemmMaskedExecutor (deepgemm_masked_executor.py)

    CUDA 原流程:
        1. m_grouped_fp8_gemm_nt_masked(expert_x, w1) → upgate_output BF16
        2. silu_mul_masked_fp8_post_quant_fwd(upgate) → down_input FP8
        3. m_grouped_fp8_gemm_nt_masked(down_input, w2) → down_output BF16

    NPU 新流程:
        1. npu_grouped_matmul_swiglu_quant_v2 融合 GEMM + SwiGLU + 量化
        (或分步执行，取决于算子支持)
    """

    @classmethod
    def check_conditions(cls, checker, config):
        """条件检查"""
        # 检查 NPU 可用
        checker.check(torch.npu.is_available())
        # 检查 BF16 模式
        checker.check(resolver.is_bf16(config))
        # 量化方法匹配
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method in [None, "FP8_PER_BLOCK", "modelslim"])

    def __init__(self, config, quant_config, weights):
        self._w1 = weights[W.moe_w1]       # [E, N, K] FP8
        self._w2 = weights[W.moe_w2]       # [E, K, N//2] FP8
        self._E, self._N, self._K = self._w1.size()
        self._w1_scale = weights.get(W.moe_s1, None)  # E8M0 (uint8存储, 已 swizzle)
        self._w2_scale = weights.get(W.moe_s2, None)  # E8M0 (uint8存储, 已 swizzle)
        # 进入本 executor 即确定走 MXFP8 FP8 推理（静态/动态量化路径权重均为 FP8）
        self._use_fp8 = True
        # 能力探测一次完成，运行时按标志分流；不用 try/except 兜底（避免掩盖算子真实错误）
        self._use_fused = has_npu_op("npu_grouped_matmul_swiglu_quant_v2")

    def execute(self, payload, activation, expert_map, **kwargs):
        if self._use_fp8:
            return self._execute_fp8(payload, **kwargs)
        else:
            return self._execute_bf16(payload, **kwargs)

    def _execute_fp8(self, payload, expert_x_scale=None, **kwargs):
        expert_x = payload.expert_x           # [E, M, K] BF16（dispatch 输出）
        masked_m = payload.expert_tokens_meta.expert_num_tokens
        expected_m = min(expert_x.shape[1], payload.expert_tokens_meta.expected_m)

        if self._use_fused:
            # ======== 方案 A: 融合算子 (优先) ========
            # 参考 vllm-ascend w8a8_mxfp8.py L312-336
            # 输入为 BF16，激活的 MX 量化由算子内部按 mxfp_act_quant_type 完成
            output = fused_experts(
                hidden_states=expert_x,
                w1=self._w1,
                w2=self._w2,
                quant_type=QuantType.MXFP8,
                mxfp_act_quant_type=torch.float8_e4m3fn,
                mxfp_weight_quant_type=torch.float8_e4m3fn,
                mxfp_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
                mxfp_per_token_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
                w1_scale=self._w1_scale,  # uint8 存储
                w2_scale=self._w2_scale,  # uint8 存储
                ...
            )
        else:
            # ======== 方案 B: 分步执行 (降级) ========
            # Step 0: 激活先 MX 量化（分步路径无算子内量化）
            expert_x_fp8, expert_x_scale = self._mx_quant(expert_x)
            # Step 1: Gate-Up GEMM
            upgate_output = torch_npu.npu_grouped_matmul_quant(
                expert_x_fp8, self._w1, self._w1_scale,
                per_token_scale=expert_x_scale,
                scale_dtype=FLOAT8_E8M0FNU_DTYPE,
                output_dtype=torch.bfloat16,
                group_sizes=[1, 1, 32]
            )
            # Step 2: SiLU + Mul + 量化
            down_input, down_input_scale = self._silu_mul_and_quant(upgate_output)
            # Step 3: Down GEMM
            output = torch_npu.npu_grouped_matmul_quant(
                down_input, self._w2, self._w2_scale,
                per_token_scale=down_input_scale,
                scale_dtype=FLOAT8_E8M0FNU_DTYPE,
                output_dtype=torch.bfloat16,
                group_sizes=[1, 1, 32]
            )

        return CombineForwardPayload(fused_expert_output=output)

    @staticmethod
    def _mx_quant(x):
        """MX 动态量化（1×32 沿最后一维，算子固定）"""
        return torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)

    def _silu_mul_and_quant(self, upgate_output):
        """SiLU + Mul + FP8 量化"""
        # 分割 gate 和 up
        gate, up = upgate_output.chunk(2, dim=-1)
        # SiLU
        act = torch.nn.functional.silu(gate) * up
        # 量化
        act_fp8, act_scale = self._mx_quant(act)
        return act_fp8, act_scale
```

### 5.4 MoE Strategy 注册 ✅🟢 BF16 Fallback 已适配 / ❌🔴 MXFP8 未适配

> ✅🟢 **BF16 Fallback Strategy 已在开发版本中适配**：`DeviceType.Ascend → AscendBf16FallbackStrategy` 已注册到 fused_moe 工厂（fused_moe/__init__.py L62-70），提供 BF16 模式下的 MoE 推理降级方案。该部分已在开发版本中适配。
>
> ❌🔴 **MXFP8 MoE Strategy 尚未适配**：以下 `NpuMoEMXFP8Executor` 和 `NpuMXFP8Strategy` 注册代码尚未实现。

**文件**：`rtp_llm/models_py/modules/factory/fused_moe/impl/ascend/__init__.py` ➕ 新建

```python
# ✅🟢 BF16 Fallback 已适配 (AscendBf16FallbackStrategy)
# ❌🔴 MXFP8 Strategy 待实现
# from .executors.npu_mxfp8_executor import NpuMoEMXFP8Executor
# from .strategy.npu_mxfp8 import NpuMXFP8Strategy

# 注册策略
# FusedMoEFactory.register(NpuMXFP8Strategy)
```

---

## 六、桥接层适配 ❌🔴

### 6.1 CUDA Kernel 处理（NPU 构建剥离，保留 CUDA 源码）

| CUDA Kernel | 文件 | 处理方式 |
|-------------|------|---------|
| `scaled_fp8_quant.cu` | `bindings/cuda/kernels/` | 🔄❌🔴 NPU 路径不调用（用 `npu_dynamic_mx_quant`）；按 bazel 架构配置剥离出 NPU 构建，不删源码 |
| `librtp_compute_ops` | C++ 扩展 | 🔄❌🔴 同上，NPU 原生算子替代 |
| `mla_quant_kernel.cu` | `bindings/cuda/kernels/` | 🔄❌🔴 ACLNN 重写 (独立工作项) |

### 6.2 第三方库处理（NPU 构建剥离，保留 CUDA 源码）

| 库 | 用途 | 处理方式 |
|----|------|---------|
| DeepGEMM | FP8 Per-Block GEMM | 🔄❌🔴 NPU 不依赖（`npu_quant_matmul` 替代）；`has_deep_gemm()` 在 NPU 返回 False（P0 第 3 项） |
| `torch._scaled_mm` | FP8 PerTensor GEMM | 🔄❌🔴 W8A8_MXFP8 路径不调用 |

### 6.3 可保留的第三方库

| 库 | 用途 | 说明 |
|----|------|------|
| FlashInfer | FP8 PerChannel / FP4 | 非 MXFP8 场景保留 |
| CUTLASS | INT8 / FP8 GEMM | 非 MXFP8 场景保留 |

### 6.4 fp8_kernel.py 函数处理 ❌🔴

**文件**：`rtp_llm/models_py/kernels/cuda/fp8_kernel/fp8_kernel.py`

| 函数 | 行号 | NPU 处理 |
|------|------|---------|
| `per_block_cast_to_fp8` | fp8_kernel.py L329（128×128 版）；权重加载实际调用的是 `model_loader/per_block_fp8_quant_weight.py` L99（带 group_size 参数版） | 🔄❌🔴 NPU 分支均不调用 (用 `npu_dynamic_mx_quant`) |
| `requant_weight_ue8m0` | L374 | 🔄❌🔴 NPU 分支不调用 (NPU 原生 E8M0)；函数保留（CUDA 路径/测试在用） |
| `quant_weight_ue8m0` | L348 | 🔄❌🔴 NPU 分支不调用；保留（同上） |
| `ceil_to_ue8m0` | L51 | 🔄❌🔴 NPU 分支不调用；保留（同上） |
| `_transform_scale_ue8m0` | L56 | 🔄❌🔴 NPU 分支不调用；保留（同上） |
| `sgl_per_token_group_quant_fp8` | L110 | 🔄❌🔴 NPU 用 `npu_dynamic_mx_quant` |

---

## 七、通信层适配

### 7.0 HCCL 构建配置 ✅🟢

> ✅🟢 **HCCL 链接配置已在开发版本中适配**：`.bazelrc` 中已配置 `--linkopt="-lhccl"`，`def.bzl` 中已配置 `-DUSE_C10D_HCCL` 编译宏。该部分已在开发版本中适配。

### 7.1 分布式通信后端（运行时）❌🔴

> ⚠️ 必须按设备条件选择 backend，**不能无条件替换为 hccl**（会破坏 CUDA/ROCm 分布式）。

| 文件 | 行号 | 修改 | 状态 |
|------|------|------|------|
| `collective_torch.py` | L586 | `backend = "hccl" if is_ascend() else "nccl"` | ❌🔴 |
| `backend_manager.py` | L50 | 同上（按 `get_device_type()` 分流） | ❌🔴 |

### 7.2 验证 ❌🔴

```python
import torch.distributed as dist
from rtp_llm.device.device_type import is_ascend

# NPU 环境验证 HCCL 可用；CUDA 环境回归验证 nccl 仍正常
dist.init_process_group(backend="hccl" if is_ascend() else "nccl")
```

---

## 八、验证方案

### 8.1 单层验证

#### Device 层 ✅🟢

```python
from rtp_llm.device import get_current_device, DeviceType

# ✅🟢 验证 AscendImpl 注册
device = get_current_device()
assert isinstance(device, AscendImpl)
assert device.arch == 910

# ❌🔴 验证 per_block_cast_to_fp8 (未适配)
# 注意: 用非方阵验证, 方阵无法暴露转置/布局错误
weight = torch.randn(128, 256, dtype=torch.bfloat16, device="npu")
weight_fp8, scale = device.per_block_cast_to_fp8(weight)
assert weight_fp8.dtype == torch.float8_e4m3fn
assert scale.dtype == torch.uint8  # E8M0 以 uint8 物理存储（逻辑类型 float8_e8m0fnu）
assert scale.shape == (128, 256 // 32)
```

#### 配置层 ❌🔴

```python
# ❌🔴 验证 load_from_ckpt 检测 ModelSlim
quant_config = QuantizationConfig.load_from_ckpt("/path/to/modelslim_model")
assert isinstance(quant_config, ModelSlimConfig)
assert quant_config.is_quanted() == True
```

#### 权重层 ❌🔴

```python
# ❌🔴 验证动态量化（非方阵: N=4096, K=11008）
weight = torch.randn(4096, 11008, dtype=torch.bfloat16, device="npu")
weight_fp8, scale = torch_npu.npu_dynamic_mx_quant(weight, torch.float8_e4m3fn)
assert weight_fp8.dtype == torch.float8_e4m3fn
assert scale.dtype == torch.uint8  # E8M0 以 uint8 物理存储
assert weight_fp8.shape == weight.shape
assert scale.shape == (4096, 11008 // 32)

# 验证 scale swizzle 布局: [N, K//32] → [K//64, N, 2]
scale_swizzled = swizzle_scale_to_npu_layout(scale)
assert scale_swizzled.shape == (11008 // 64, 4096, 2)
```

#### 推理层 ❌🔴

```python
# ❌🔴 验证 Linear（W8A8_MXFP8）— 非方阵: N=4096, K=11008
# 注意: torch.randn 不支持 float8 dtype, 需先 bf16 randn 再量化
x = torch.randn(128, 11008, dtype=torch.bfloat16, device="npu")
weight_bf16 = torch.randn(4096, 11008, dtype=torch.bfloat16, device="npu")
weight_fp8, scale = torch_npu.npu_dynamic_mx_quant(weight_bf16, torch.float8_e4m3fn)
scale_swizzled = swizzle_scale_to_npu_layout(scale)  # [K//64, N, 2]

# 激活在线量化（W8A8，不是 W8-only）
x_fp8, pertoken_scale = torch_npu.npu_dynamic_mx_quant(x, torch.float8_e4m3fn)

output = torch_npu.npu_quant_matmul(
    x_fp8, weight_fp8, scale_swizzled,
    scale_dtype=FLOAT8_E8M0FNU_DTYPE,
    pertoken_scale=pertoken_scale,
    pertoken_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
    output_dtype=torch.bfloat16,
    group_sizes=[1, 1, 32]
)
assert output.dtype == torch.bfloat16
assert output.shape == (128, 4096)  # [M, N], 非方阵可捕获转置/布局错误
# 数值验证: 与 BF16 参考结果对比
ref = x @ weight_bf16.T
assert cosine_similarity(output, ref) > 0.99
```

### 8.2 端到端验证 ❌🔴

```python
# 1. 动态量化启动
# python rtp_llm_cli.py --model_name <model> --quantization FP8_PER_BLOCK --device npu

# 2. 静态量化启动 (ModelSlim 预量化模型)
# python rtp_llm_cli.py --model_name <modelslim_model> --device npu

# 3. 验证推理结果
# 对比 BF16 基线输出与 MXFP8 输出的差异
# 允许误差: cos_sim > 0.99, max_diff < 0.1
```

### 8.3 性能验证

| 指标 | BF16 基线 | MXFP8 期望 | 验证方法 |
|------|----------|-----------|---------|
| 显存占用 | 1× | ~0.5× | `torch.npu.memory_allocated()` |
| 推理速度 | 1× | ≥1.5× | 时间测量 |
| 精度损失 | 0 | <1% | cos_sim 对比 |

---

## 九、适配优先级

> 判定依据：P0 = 缺少该项则量化推理无法跑通。条目与概要 §八 编号对齐。

### P0（必须，阻塞推理）

1. **推理层 Dense GEMM 替换**：`NpuFp8MXFP8Linear` + `npu_quant_matmul`（含激活动态量化）❌🔴
2. **权重层算子替换与布局处理**：`npu_dynamic_mx_quant`（动态量化路径）；NPU 分支不转置 + scale swizzle `[N,K//32]→[K//64,N,2]` ❌🔴
   （`AscendImpl` 注册已 ✅🟢；其 `per_block_cast_to_fp8` 等 MXFP8 方法重写是本项前提）
3. **DeepGEMM wrapper NPU 分流**：`has_deep_gemm()` / `is_deep_gemm_e8m0_used()` 在 NPU 返回 False（`is_ascend()` 判断）❌🔴
   （现状 `is_deep_gemm_e8m0_used()` 调用 `torch.cuda.get_device_capability()`，NPU 上抛异常，阻塞 `_postprocess` 权重加载，必须先改）
4. **MoE MXFP8 Executor**：`NpuMoEMXFP8Executor` + `npu_grouped_matmul_swiglu_quant_v2` ❌🔴
   （目标模型为 MoE 时本项同样阻塞推理，仅面向 Dense 模型时可后置）
5. **通信层运行时 NCCL→HCCL** ❌🔴
   （TP>1 部署时阻塞，仅单卡场景可后置）

### P1（简化代码，不做不阻塞）

6. **移除 `requant_weight_ue8m0`**：删除调用（第 3 项生效后即为死分支）❌🔴
7. **NPU 构建剥离第三方依赖**：DeepGEMM / `scaled_fp8_quant.cu` 等按 bazel 架构配置隔离，保留 CUDA 源码（不直接删除，避免破坏 CUDA 路径共存）❌🔴

### P2（扩展功能）

8. **ModelSlim 预量化支持**：`ModelSlimConfig` + `ModelSlimWeight` + `load_from_ckpt` 检测 ❌🔴
9. **配置层 dtype 调整**：`get_supported_compute_dtypes` 验证 NPU 支持 ❌🔴

### P3（独立工作项）

10. **MLA KV Cache 量化**：ACLNN 重写 ❌🔴

### 已完成的基础设施 ✅🟢

- **Device 层基础框架**：`DeviceType.Ascend`、`AscendImpl` 基础方法、注册机制 ✅🟢
- **推理层 F16 基线**：`AscendF16Linear` + Factory 注册 ✅🟢
- **MoE BF16 Fallback**：`AscendBf16FallbackStrategy` + fused_moe 注册 ✅🟢
- **HCCL 构建配置**：`.bazelrc` 链接选项 + `def.bzl` 编译宏 ✅🟢
