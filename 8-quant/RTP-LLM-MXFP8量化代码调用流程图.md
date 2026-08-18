# RTP-LLM MXFP8 量化代码调用流程图

> 本文档展示 **静态量化（预量化模型）** 和 **动态量化（Load Quant）** 两条路径的完整代码调用链，
> 以及 NPU MXFP8 适配后的对应调用链。每处标注 **文件路径 / 类名 / 方法名 / 行号**。

---

## 适配状态

> 基于 `rtp-llm-npu` 代码检查结果，下表列出 MXFP8 量化方案在 Ascend NPU 上的各项适配状态。
> - ✅🟢 已适配：该部分已在开发版本中适配
> - ❌🔴 未适配：尚未适配，保留原始 CUDA/MXFP8 流程供后续开发参考

| 适配项 | 状态 | 说明 | 代码位置 |
|--------|------|------|---------|
| DeviceType.Ascend | ✅🟢 | `DeviceType.Ascend = 6` 枚举值已定义 | device_type.py L14 |
| AscendImpl 基础框架 | ✅🟢 | `class AscendImpl(GpuImpl)` 基础类已创建 | device_impl.py L696-710 |
| AscendImpl 注册 | ✅🟢 | `DeviceType.Ascend → AscendImpl` 已注册到设备工厂 | device/__init__.py L22-23 |
| AscendF16Linear（BF16 基线推理） | ✅🟢 | BF16 基线 Linear 推理已适配 | impl/ascend/f16_linear.py |
| MoE BF16 Fallback | ✅🟢 | BF16 基线 MoE 推理已适配 | impl/ascend/strategy/pytorch_fallback.py |
| AscendImpl MXFP8 方法 | ❌🔴 | `per_block_cast_to_fp8` 未重写，仍继承 GpuImpl | device_impl.py |
| NpuFp8MXFP8Linear | ❌🔴 | 无 Ascend MXFP8 Linear 实现 | — |
| NpuMoEMXFP8Executor | ❌🔴 | 无 Ascend MXFP8 MoE 实现 | — |
| 权重层算子替换 | ❌🔴 | `per_block_cast_to_fp8` / `requant_weight_ue8m0` 未改 | per_block_fp8_quant_weight.py |
| scale swizzle | ❌🔴 | `[N,K//32]→[K//64,N,2]` 布局转换未实现（FP8 权重无需 NZ 转换） | — |
| ModelSlimConfig | ❌🔴 | quant_config.py 中无相关代码 | — |
| load_from_ckpt ModelSlim 检测 | ❌🔴 | 无 quant_model_description.json 检测（需 is_ascend() 门控） | quant_config.py |
| DeepGEMM wrapper | ❌🔴 | `has_deep_gemm` / `is_deep_gemm_e8m0_used` 未改（NPU 上 `get_device_capability()` 抛异常，阻塞加载） | deepgemm_wrapper.py |
| NCCL→HCCL (运行时) | ❌🔴 | 通信后端未按设备分流（需 `"hccl" if is_ascend() else "nccl"`） | collective_torch.py / backend_manager.py |

> **总结**：当前开发版本已完成 **Ascend 设备框架注册** 与 **BF16 基线推理（Linear + MoE）** 的适配，
> 模型可在 Ascend NPU 上以 BF16 精度运行。MXFP8 量化推理路径（权重在线量化、FP8 GEMM、DeepGEMM）尚未适配。

---

## 方案说明

> **W8A8_MXFP8 是 W8A8（权重+激活均 8-bit）量化方案的 MXFP8 实现，激活在运行时在线动态量化**
>
> 本文档中：§二/§三 描述 **CUDA 现状路径**（FP8 Per-Block，128×128 块），§四 描述 **NPU 目标路径**（W8A8_MXFP8，1×32 组）。

### 量化方案分类

```
W8A8（量化大类：权重+激活都量化为 8-bit）
├── W8A8_INT8（SmoothQuant）
├── W8A8_FP8（Per-Tensor / Per-Channel）
└── W8A8_MXFP8 ← NPU 目标方案（权重/激活 FP8 E4M3，group_size=32，E8M0 scale）

W8A16（仅权重量化，激活保持高精度）
├── W8_INT8 / W4_INT8 / ...
```

### W8 和 A8 的定义

- **W8（Weight 8-bit）**：权重量化为 8-bit
- **A8（Activation 8-bit）**：激活量化为 8-bit
- **W8A8**：权重与激活均为 8-bit；NPU 方案的激活通过 `npu_dynamic_mx_quant` 运行时在线动态量化（非 weight-only）

### 两种方案的实现细节对比

| 维度 | CUDA 现状（FP8 Per-Block） | NPU 目标（W8A8_MXFP8） |
|------|--------------------------|----------------------|
| 量化类型 | 权重 128×128 块 + 激活 per-token-group | W8A8（权重 1×32 组 + 激活动态量化） |
| 数据格式 | FP8 E4M3（权重）+ FP8（激活）+ FP32/int32 scale | FP8 E4M3（权重+激活）+ E8M0 scale（uint8 存储） |
| 量化粒度 | 128×128 二维块 | 1×32 一维组（沿 K，算子固定） |
| 代码标记 | — | `# W8A8_MXFP8: Weight & Activation FP8 Quantization` |

---

## 一、总览：两条路径的分流与汇聚

```
                        ┌─────────────────────────────────┐
                        │         模型启动入口             │
                        │  model_config.py L565           │
                        │  init_precision_config()        │
                        └────────┬────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
            ┌───────▼───────┐         ┌───────▼───────┐
            │   静态量化     │         │   动态量化     │
            │ (预量化模型)   │         │ (Load Quant)  │
            │ is_quanted=True│         │is_quanted=False│
            └───────┬───────┘         └───────┬───────┘
                    │                         │
          配置层     │                  配置层  │
          load_from_ckpt()           init_quant_config()
                    │                         │
          权重层     │                  权重层  │
          PerBlockFp8Weight          LoadQuantPerBlockFp8Weight
          (直接加载 FP8)              (在线量化 BF16→FP8)
                    │                         │
                    └────────────┬────────────┘
                                 │
                          ┌──────▼──────┐
                          │   推理层     │  ← 两条路径汇聚
                          │ Linear/MoE  │
                          └─────────────┘
```

---

## 二、静态量化（预量化模型）调用流程

### 2.1 预处理层（离线，框架外）

```
ModelSlim / 外部工具
    │
    ├── 输入: 原始 BF16/FP16 权重 (HuggingFace 格式)
    │
    ├── 执行: FP8 BlockWise 128×128 分块量化
    │
    └── 输出:
        ├── config.json (含 quantization_config 字段)
        │   └── quant_method: "fp8"
        │   └── weight_block_size: [128, 128]
        ├── *.safetensors (FP8 E4M3 权重 + FP32 scale)
        └── quant_model_description.json (ModelSlim 输出)
```

### 2.2 配置层

```
[model_config.py L565] init_precision_config(self, ...)
    │
    ├── [L583] QuantizationConfig.load_from_ckpt(self.ckpt_path)
    │   │
    │   │  [quant_config.py L99] load_from_ckpt(cls, ckpt_path)
    │   │
    │   ├── [L109] 检测 smoothquant.ini → 不存在
    │   ├── [L115] 检测 pertensorquant.ini → 不存在
    │   │
    │   ├── [L127] 读取 config.json
    │   │   └── [L135] quantization_config.quant_method == "fp8"
    │   │
    │   ├── [L157-165] 检测 weight_block_size 字段
    │   │   └── 存在 → quant_method = Fp8BlockWiseQuantConfig.get_method()
    │   │                              = "FP8_PER_BLOCK"
    │   │
    │   └── [L263] cls.from_config({"method": "FP8_PER_BLOCK", "is_quanted": True})
    │       │
    │       └── [L80] from_config() 遍历 _registry
    │           └── 匹配 Fp8BlockWiseQuantConfig.get_method() == "FP8_PER_BLOCK"
    │               └── [L402] Fp8BlockWiseQuantConfig._from_config()
    │                   └── Fp8BlockWiseQuantConfig(bits=8, group_size=128, is_quanted=True)
    │
    ├── [L591-595] quant_algo.setQuantAlgo("fp8", 8, 128)
    │
    └── [L623-629] 强制 data_type = WEIGHT_TYPE.BF16
        └── "fp8_block_wise quantization only supports BF16"
```

### 2.3 权重层

```
[model_weight_info.py L324-325]
    │  if quant_algo.isQuant():
    │      weight_info = weight_info.to_quant_weight_info(quant_config)
    │
    └── [L80] to_quant_weight_info()
        └── weight.create(weight, quant_config)
            │
            │  [weight_module.py L60] WeightModule.create(weight_info, quant_config)
            │
            ├── [L73-77] 遍历 _registry, 调用 support()
            │   │
            │   └── [per_block_fp8_quant_weight.py L243] PerBlockFp8Weight.support()
            │       ├── is_quanted() == True ✓
            │       ├── isinstance(Fp8BlockWiseQuantConfig) ✓
            │       └── name in w8a8_weight_list ✓
            │       → return True
            │
            └── 创建 PerBlockFp8Weight 实例
                │
                │  [L254] __init__()
                │  根据 weight name 分发:
                │  ├── W.attn_qkv_w → _get_qkv_quant_weight (L328)
                │  ├── W.ffn_w1/w2/w3 → _get_ffn_quant_weight (L592)
                │  └── W.moe_w1 → _get_moe_w1_quant_weight (L728)
                │
                │  每个方法创建:
                │  ├── kernel: W8A8Fp8PerBlockAtomicWeight (dtype=float8_e4m3fn)
                │  └── scale: W8A8Fp8PerBlockAtomicWeight (dtype=float32)
                │
                └── [weight_module.py L157] load()
                    │
                    ├── [L164] _load_raw_tensor()
                    │   └── 从 safetensors 读取已量化的 FP8 权重 + FP32 scale
                    │       输入: safetensors 中的 FP8 tensor
                    │       输出: {kernel_name: FP8, scale_name: FP32}
                    │
                    ├── [L175] _split()
                    │   └── TP 张量并行切分
                    │
                    └── [L177] _postprocess()
                        │
                        │  [per_block_fp8_quant_weight.py L761] PerBlockFp8Weight._postprocess()
                        │
                        ├── [L768] super()._postprocess()  # CompositeWeight 递归
                        │
                        ├── [L777-783] 非 E8M0 路径:
                        │   ├── kernel_weight.reshape(N, -1)
                        │   └── scale_weight.reshape(N, -1)
                        │
                        └── [L800-803] E8M0 路径 (Blackwell SM100/120):  ❌🔴 (权重层算子替换未适配)
                            │
                            └── requant_weight_ue8m0(kernel_weight, scale_weight)
                                │
                                │  [fp8_kernel.py L374] requant_weight_ue8m0()  ❌🔴
                                │
                                ├── [L380] block_quant_dequant(weight, scale, [128,128], bf16)
                                │   │  [L305] block_quant_dequant()
                                │   └── FP8 × scale_repeat → BF16 (反量化)
                                │
                                ├── [L386] quant_weight_ue8m0(weight_dequant, [128,128])
                                │   │  [L348] quant_weight_ue8m0()
                                │   └── [L329] per_block_cast_to_fp8(weight, use_ue8m0=True)
                                │       ├── sf = amax / 448.0
                                │       ├── sf = ceil_to_ue8m0(sf) = 2^ceil(log2(sf))  [L51]
                                │       └── x_scaled = (x / sf).to(float8_e4m3fn)
                                │
                                └── [L390] _transform_scale_ue8m0(out_s, mn)
                                    └── deep_gemm.utils.layout.get_mn_major_tma_aligned_packed_ue8m0_tensor()

                        最终输出:
                        ├── kernel_weight: torch.float8_e4m3fn, shape (N, K)
                        └── scale_weight: torch.int32 (UE8M0 打包), shape (N, K//512)
```

### 2.4 推理层 — Linear

> ✅🟢 **BF16 基线 Linear 已适配**：`AscendF16Linear`（`impl/ascend/f16_linear.py`）已实现 BF16 基线推理，
> 该部分已在开发版本中适配。模型可在 Ascend NPU 上以 BF16 精度完成 Linear 前向计算。
>
> ❌🔴 以下 FP8/DeepGEMM MXFP8 调用流程为原始 CUDA 路径，**尚未适配**，保留供后续 MXFP8 适配参考。

```
[factory.py L96] LinearFactory.create_linear(weight, bias, weight_scales, quant_config)
    │
    ├── 遍历注册策略, 调用 can_handle()
    │
    ├── [fp8_gemm_linear.py L22] CudaFp8GEMMLinear.can_handle()
    │   ├── weight_scales != None ✓
    │   ├── weight.dtype == float8_e4m3fn ✓
    │   └── quant_config.get_method() == "FP8_PER_BLOCK" ✓
    │   → return True
    │
    └── [L38] CudaFp8GEMMLinear.__init__()
        ├── 创建 CudaFp8DeepGEMMLinear (主后端)
        └── 创建 CudaFp8FlashinferLinear (备后端, 小 batch)

[fp8_gemm_linear.py L133] forward(input)
    │
    ├── _should_use_flashinfer(input) → 判断 batch 大小
    │   ├── M < FLASHINFER_M_THRESHOLD → FlashInfer 后端
    │   └── M >= threshold → DeepGEMM 后端
    │
    └── [fp8_deepgemm_linear.py L171] CudaFp8DeepGEMMLinear.forward(input)
        │
        ├── 输入检查: input.dtype == bfloat16 或 float8_e4m3fn
        │
        ├── [L196-206] 输入已是 FP8:
        │   └── 直接使用, scale 填 0x7F7F7F7F (UE8M0 的 1.0)
        │
        ├── [L207-215] 输入是 BF16, 需在线量化:
        │   └── [fp8_kernel.py L110] sgl_per_token_group_quant_fp8(input, group_size=128)
        │       ├── per_token_group_quant_fp8_v2(x, x_q, x_s, ...)
        │       └── 输出: (FP8 input [M,K], scale [M, K//128])
        │
        ├── [L224] 准备输出: torch.empty(M, N, dtype=bfloat16)
        │
        └── [L226] fp8_gemm_nt()
            │
            │  [deepgemm_wrapper.py L367] fp8_gemm_nt(a, b, output)  ❌🔴 (DeepGEMM wrapper 未适配)
            │  A = (input_fp8, input_scales)
            │  B = (weight_fp8, weight_scales)
            │
            └── _fp8_gemm_nt_impl(a, b, output)
                └── deep_gemm.fp8_gemm_nt(A, B, C)  # CUDA Tensor Core

        输出: BF16 tensor [M, N]
```

### 2.5 推理层 — MoE

> ✅🟢 **BF16 基线 MoE 已适配**：MoE BF16 Fallback（`impl/ascend/strategy/pytorch_fallback.py`）已实现 BF16 基线 MoE 推理，
> 该部分已在开发版本中适配。模型可在 Ascend NPU 上以 BF16 精度完成 MoE 专家前向计算。
>
> ❌🔴 以下 FP8/DeepGEMM MXFP8 调用流程为原始 CUDA 路径，**尚未适配**，保留供后续 MXFP8 适配参考。

```
[deepgemm_masked_executor.py L473] DeepGemmMaskedExecutor.execute(payload, ...)
    │
    ├── [L477] if self._use_fp8:
    │   └── [L383] _execute_fp8(payload, ...)
    │       └── [L353] _normal_execute(expert_x, masked_m, expected_m, expert_x_scale)
    │           │
    │           └── [L147] _forward_masked_grouped_ffn(start_idx, end_idx, ...)
    │               │
    │               │  ======== GroupGEMM-0: Gate-Up ========
    │               ├── [L176-195] m_grouped_fp8_gemm_nt_masked(
    │               │       a = (expert_x FP8, expert_x_scale),       # (E, M, K)
    │               │       b = (self._w1 FP8, self._w1_scale),       # (E, N, K)
    │               │       output = upgate_output BF16                # (E, M, N)
    │               │   )
    │               │   └── [deepgemm_wrapper.py L468] m_grouped_fp8_gemm_nt_masked()  ❌🔴 (DeepGEMM wrapper 未适配)
    │               │       ├── maybe_pack_ue8m0_scale(a, b)
    │               │       └── _m_grouped_fp8_gemm_nt_masked_impl()
    │               │           └── deep_gemm.masked_fp8_gemm()
    │               │
    │               │  ======== SiLU + Mul + 量化 ========
    │               ├── [L218-242] silu_mul_masked_fp8_post_quant_fwd(
    │               │       input = upgate_output BF16,
    │               │       output = down_input FP8,                # (E, M, N//2)
    │               │       output_scale = down_input_scale,       # FP32 or int32
    │               │   )
    │               │
    │               │  ======== GroupGEMM-1: Down ========
    │               └── [L260-285] m_grouped_fp8_gemm_nt_masked(
    │                       a = (down_input FP8, down_input_scale),  # (E, M, N//2)
    │                       b = (self._w2 FP8, self._w2_scale),      # (E, K, N//2)
    │                       output = down_output BF16                 # (E, M, K)
    │                   )
    │
    └── 输出: BF16 fused_expert_output [M, K]
```

---

## 三、动态量化（Load Quant）调用流程

### 3.1 预处理层

```
无预处理 — 使用原始 HuggingFace 模型权重 (BF16/FP16)
```

### 3.2 配置层

```
[model_config.py L565] init_precision_config(self, ...)
    │
    ├── [L583] QuantizationConfig.load_from_ckpt(self.ckpt_path)
    │   └── 返回 None (checkpoint 中无量化配置)
    │
    ├── [L584-587] if not quant_config and self.quantization:
    │   └── [quant_config.py L845] init_quant_config(self.quantization)
    │       │  输入: "FP8_PER_BLOCK"
    │       │
    │       ├── [L847] json.loads("FP8_PER_BLOCK") → 失败
    │       │
    │       └── [L852] preset_quant_config["FP8_PER_BLOCK".upper()]
    │           └── [L801] DEFAULT_FP8_BLOCK_WISE_QUANT_CONFIG
    │               = Fp8BlockWiseQuantConfig(bits=8, group_size=128, is_quanted=False)
    │
    ├── [L591-595] quant_algo.setQuantAlgo("fp8", 8, 128)
    │
    └── [L623-629] 强制 data_type = WEIGHT_TYPE.BF16
        └── checkpoint 权重以 BF16 读取
```

### 3.3 权重层

```
[weight_module.py L60] WeightModule.create(weight_info, quant_config)
    │
    ├── [L73-77] 遍历 _registry, 调用 support()
    │   │
    │   ├── [per_block_fp8_quant_weight.py L243] PerBlockFp8Weight.support()
    │   │   └── is_quanted() == False → return False ✗
    │   │
    │   └── [L813] LoadQuantPerBlockFp8Weight.support()
    │       ├── is_quanted() == False ✓ (注意: 与静态相反)
    │       ├── isinstance(Fp8BlockWiseQuantConfig) ✓
    │       ├── name in w8a8_weight_list ✓
    │       └── name not in [W.mla_kc, W.mla_vc] ✓
    │       → return True
    │
    └── 创建 LoadQuantPerBlockFp8Weight 实例
        │
        │  [L823] __init__()
        │  ├── self.group_size = quant_config.group_size()  # 128
        │  ├── 创建 kernel: AtomicWeight (原始 dtype, BF16)
        │  └── 创建 scale: AtomicWeight (placeholder)
        │
        └── [weight_module.py L157] load()
            │
            ├── [L164] _load_raw_tensor()
            │   │
            │   │  [per_block_fp8_quant_weight.py L854]
            │   │  LoadQuantPerBlockFp8Weight._load_raw_tensor()
            │   │
            │   ├── [L861] self.kernel._load_raw_tensor()
            │   │   └── 从 safetensors 读取原始 BF16 权重
            │   │       输入: BF16 tensor [M, N]
            │   │
            │   ├── [L868] per_block_cast_to_fp8(kernel, group_size=128)  ❌🔴 (per_block_cast_to_fp8 未改)
            │   │   │
            │   │   │  [per_block_fp8_quant_weight.py L102]
            │   │   │  per_block_cast_to_fp8(x, group_size)
            │   │   │
            │   │   ├── [L110-111] pad 到 128 的倍数
            │   │   ├── [L116-118] reshape 为 (b, M//128, 128, N//128, 128)
            │   │   ├── [L119] x_amax = amax(dim=(2,4)).clamp(1e-4)  # 每 block 最大值
            │   │   ├── [L120] x_scaled = x * (448 / amax) → float8_e4m3fn
            │   │   └── [L122] scales = amax / 448 → float32
            │   │
            │   │   输出: (FP8 [M,N], FP32 scale [M//128, N//128])
            │   │
            │   ├── [L879] quant_kernel.T  # 转置 (非 MoE)
            │   └── [L883] scale.T
            │
            ├── [L175] _split()  # TP 切分
            │
            └── [L177] _postprocess()
                │
                │  [L761] 继承自 PerBlockFp8Weight._postprocess()
                │  (与静态量化相同的后处理逻辑)
                │
                └── [L800-803] E8M0 路径:  ❌🔴 (权重层算子替换未适配)
                    └── requant_weight_ue8m0(kernel_weight, scale_weight)
                        └── (与静态量化完全相同)

            最终输出:
            ├── kernel_weight: torch.float8_e4m3fn, shape (N, K)
            └── scale_weight: torch.int32 (UE8M0) 或 torch.float32
```

### 3.4 推理层

```
与静态量化完全相同 — 两条路径在此汇聚
（见 2.4 Linear 和 2.5 MoE）
```

### 3.5 Device 层（两条路径共用）

> ✅🟢 **AscendImpl 基础框架已适配**：`class AscendImpl(GpuImpl)`（`device_impl.py` L696-710）已创建并注册到设备工厂
> （`DeviceType.Ascend → AscendImpl`，`device/__init__.py` L22-23）。
> 该部分已在开发版本中适配，Ascend 设备可被正确识别和调度。
>
> ❌🔴 AscendImpl 中的 MXFP8 相关方法（`per_block_cast_to_fp8`、`convert_fp8_weight_params`、`requant_weight_ue8m0`）
> 尚未重写，仍继承 GpuImpl 原始实现，**尚未适配**。以下为原始 CUDA Device 层流程，保留供参考。

```
[device_impl.py L141] GpuImpl(DeviceBase)
    │
    ├── [L55-60] maybe_rewrite_weight_by_key(key, weight)
    │   └── preprocess_gemm_weight_by_key()
    │       └── 默认: 直接返回 (无转换)
    │
    └── [L343-346] convert_fp8_weight_params(weight, weight_scale)
        └── 默认: return [weight, weight_scale]  (无转换)

[device_impl.py L349] CudaImpl(GpuImpl)
    │
    ├── [L929] maybe_rewrite_weight_by_key(key, weight)
    │   ├── key == "weight":
    │   │   ├── 非 gfx950: FP8 e4m3fn → e4m3fnuz (ROCm)
    │   │   └── 特定权重: swizzle / shuffle
    │   └── key == "scale":
    │       └── 非 gfx950: scale × 2.0 (ROCm)
    │
    └── [L1000-1018] convert_fp8_weight_params(weight, weight_scale)
        └── ROCm 非 gfx950: e4m3fn → e4m3fnuz, scale × 2
```

---

## 四、NPU 适配后的调用流程

> ❌🔴 **本章整体未适配**：以下 NPU MXFP8 适配流程为目标设计方案，当前开发版本尚未实现。
> 涉及的权重层算子替换（`per_block_cast_to_fp8` / `requant_weight_ue8m0`）、
> `NpuFp8MXFP8Linear`、`NpuMoEMXFP8Executor`、DeepGEMM wrapper 均未适配。

### 4.1 NPU 静态量化（ModelSlim 预量化）

> ⚠️ **兼容性前提**：GPU 预量化 FP8_PER_BLOCK ckpt（128×128 块 + FP32 scale）与 NPU MXFP8 格式
> （1×32 组 + E8M0 scale）不兼容，不能直接加载；NPU 静态路径仅支持 ModelSlim 输出的 MXFP8 ckpt。
> ModelSlim 检测须加 `is_ascend()` 门控，不得影响 CUDA/ROCm 的配置检测逻辑。

```
ModelSlim 离线量化
    │
    └── 输出:
        ├── quant_model_description.json (量化描述, 含 MXFP8 层标注)
        ├── quant_model_weight_*.safetensors (FP8 E4M3 权重 + E8M0 scale)
        └── config.json

[model_config.py L583] load_from_ckpt(ckpt_path)
    │
    │  [quant_config.py L99] load_from_ckpt()
    │
    ├── 检测 config.json → quant_method="fp8" + weight_block_size
    │   └── 匹配 Fp8BlockWiseQuantConfig(is_quanted=True)
    │
    └── ➕ 新增: if is_ascend(): 检测 quant_model_description.json
        └── 值含 "MXFP8" → 匹配 ModelSlimConfig(is_quanted=True, group_size=32)
            （is_ascend() 门控: CUDA 平台不进入该分支）

[权重层] WeightModule.create()
    │
    ├── Fp8BlockWiseQuantConfig → PerBlockFp8Weight
    │   └── _postprocess: is_deep_gemm_e8m0_used() 在 NPU 返回 False,
    │       requant_weight_ue8m0 分支不执行（调用点另行删除）❌🔴
    │
    └── ModelSlimConfig → ➕ ModelSlimWeight  ❌🔴
        └── _postprocess: 直接加载 FP8 + E8M0 scale（无 torch_npu 依赖）
            ├── 权重保持 [N, K] 不转置
            └── scale swizzle [N,K//32] → [K//64,N,2]

[推理层] LinearFactory.create_linear()（仅 DeviceType.Ascend 时导入 impl/ascend）
    │
    └── ➕ NpuFp8MXFP8Linear  ❌🔴 (无 Ascend MXFP8 Linear 实现)
        ├── torch_npu.npu_dynamic_mx_quant(x) → 激活在线量化（W8A8，非 W8-only）
        └── torch_npu.npu_quant_matmul(x_fp8, weight, scale_swizzled, pertoken_scale)

[推理层 MoE] ➕ NpuMoEMXFP8Executor  ❌🔴 (无 Ascend MXFP8 MoE 实现)
    └── torch_npu.npu_grouped_matmul_swiglu_quant_v2(...)（方案 A 融合算子）
```

### 4.2 NPU 动态量化（Load Quant）

> ⚠️ **兼容性约束**：`per_block_fp8_quant_weight.py` / `device_impl.py` 在所有平台无条件导入，
> NPU 分支内必须**延迟导入 torch_npu**；分流统一用 `is_ascend()`，CUDA else 分支逐字节不变（详见详设 §1.4）。

```
命令行: --quantization FP8_PER_BLOCK
    │
    └── init_quant_config("FP8_PER_BLOCK")
        └── Fp8BlockWiseQuantConfig(is_quanted=False, group_size=128)
            （group_size=128 为 CUDA 侧保留；NPU 分支不使用该值，1×32 由算子固定）

[权重层] WeightModule.create()
    │
    └── LoadQuantPerBlockFp8Weight
        │
        ├── _load_raw_tensor:
        │   ├── CUDA 分支: per_block_cast_to_fp8(kernel, 128)（128×128 块，保持不变）
        │   └── NPU 分支 (is_ascend()):  ❌🔴 (per_block_cast_to_fp8 未改)
        │       ├── import torch_npu（分支内延迟导入）
        │       ├── 🔄 torch_npu.npu_dynamic_mx_quant → FP8 + E8M0 scale（1×32 沿 K）
        │       ├── 不执行 CUDA 的 quant_kernel.T / scale.T 转置，权重保持 [N, K]
        │       └── scale swizzle [N,K//32] → [K//64,N,2]
        │
        └── _postprocess:
            ├── has_deep_gemm()/is_deep_gemm_e8m0_used() 在 NPU 返回 False  ❌🔴
            │   （否则 torch.cuda.get_device_capability() 在 NPU 抛异常，阻塞加载）
            └── 🗑️ 删除 requant_weight_ue8m0 调用（NPU 原生 E8M0）  ❌🔴

[Device 层] AscendImpl（已注册 ✅🟢，MXFP8 方法未重写 ❌🔴）
    │
    ├── per_block_cast_to_fp8 → npu_dynamic_mx_quant（方法体内延迟导入）🔄  ❌🔴
    ├── convert_fp8_weight_params → 直接返回 🔄  ❌🔴
    ├── requant_weight_ue8m0 → 不实现 🗑️  ❌🔴
    └── swizzle_blockscale → 不实现 🗑️  ❌🔴

[推理层] 与 NPU 静态量化相同
    └── npu_quant_matmul / npu_grouped_matmul_swiglu_quant_v2
```

---

## 五、数据格式变化表

### 5.1 静态量化数据流

> 前 4 行为 CUDA 现状；最后一行为 NPU 目标。

| 阶段 | Weight dtype | Weight shape | Scale dtype | Scale shape |
|------|-------------|-------------|-------------|-------------|
| ModelSlim 输出（GPU 128×128 ckpt） | float8_e4m3fn | (M, N) | float32 | (M//128, N//128) |
| _load_raw_tensor 后（CUDA） | float8_e4m3fn | (M, N) | float32 | (M//128, N//128) |
| _postprocess 非E8M0（CUDA） | float8_e4m3fn | (N, K) | float32 | (N//128, K//128) |
| _postprocess E8M0（CUDA Blackwell） | float8_e4m3fn | (N, K) | **int32 (UE8M0)** | (N, K//512) |
| NPU _postprocess（ModelSlim MXFP8 ckpt） | float8_e4m3fn | (N, K) 不转置 | **uint8（E8M0）** | 量化输出 (N, K//32) → swizzle **(K//64, N, 2)** |

### 5.2 动态量化数据流

> 前 4 行为 CUDA 现状（128×128 块）；最后一行为 NPU 目标（1×32 组，粒度不同）。

| 阶段 | Weight dtype | Weight shape | Scale dtype | Scale shape |
|------|-------------|-------------|-------------|-------------|
| Checkpoint 存储 | bfloat16 | (M, N) | - | - |
| _load_raw_tensor 后（CUDA） | **float8_e4m3fn** | (M, N) | **float32** | (M//128, N//128) |
| 转置后（CUDA） | float8_e4m3fn | (N, M) | float32 | (N//128, M//128) |
| _postprocess E8M0（CUDA Blackwell） | float8_e4m3fn | (N, K) | int32 (UE8M0) | (N, K//512) |
| NPU _load_raw_tensor / _postprocess | float8_e4m3fn | (N, K) 不转置 | **uint8（E8M0）** | (N, K//32) → swizzle **(K//64, N, 2)** |

### 5.3 推理时激活值数据流（CUDA 现状：per-token-group，group=128）

| 阶段 | Activation dtype | Scale dtype |
|------|-----------------|-------------|
| 输入 | bfloat16 [M, K] | - |
| 在线量化后 | float8_e4m3fn [M, K] | float32/int32 [M, K//128] |
| GEMM 输出 | bfloat16 [M, N] | - |

### 5.4 NPU 适配后数据流（W8A8_MXFP8：激活 MX 动态量化，group=32）

> Scale 均为 uint8 物理存储、E8M0 逻辑类型（float8_e8m0fnu）。

| 阶段 | Weight | Weight Scale | Activation |
|------|--------|--------------|------------|
| 动态量化（权重量化） | npu_dynamic_mx_quant 输出 FP8 | uint8 (E8M0) [N,K//32] → swizzle [K//64,N,2] | - |
| 静态加载（ModelSlim） | FP8 E4M3 | 同上 | - |
| 推理（激活在线量化） | float8_e4m3fn | 已 swizzle | BF16 → npu_dynamic_mx_quant → FP8 + pertoken_scale uint8 [M,K//32] |
| GEMM 输出 | - | - | BF16 |
