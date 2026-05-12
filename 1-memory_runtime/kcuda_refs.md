# `torch::kCUDA` 引用检查结果

| 文件名 | 已处理 | 遗漏 |
|--------|--------|------|
| `BlockPool.cc` | `#if USING_CUDA / elif USING_ASCEND (kPrivateUse1) / elif USING_ROCM / else` 完整条件编译链 | — |
| `KVCacheManager.cc` | `dst_device` 有 `#if USING_ASCEND` guard；`src_device` 运行时判 `is_privateuseone()` | `cudaSyncAndCheck()` (行219, 507) 未 guard |
| `KVCacheMemoryConnector.cc` | `to_device` lambda 有 `#if USING_ASCEND` → `kPrivateUse1` | — |
| `CudaCopyUtil.cc` | `batchCopy*` 中 `#if USING_ASCEND` → `kPrivateUse1` | — |
| `cuda_graph_runner.h` | `options_cuda_int32_` / `options_cuda_float_` 有 `#if USING_ASCEND` | — |
| `cuda_graph_runner.cc` | `arange` / `options_cuda_int32` 有 `#if USING_ASCEND` | — |
| `ExecOps.cc` | `getTorchCudaDevice()`、`execCreateMoeExpertStates` 有 `#if USING_ASCEND` → `kPrivateUse1` | `runtimeCreateEvent()` 只有 `#if USING_CUDA` / `elif USING_ROCM`，没有 Ascend 分支 (行134-140) |
| `WeightsConverter.cc` | — | `CopyTensorToGPU()` 硬编码 `torch::kCUDA` (行14)，无任何条件编译或运行时判决 |
| `RopeCache.cc` | — | 4 处 `torch::kCUDA` 硬编码 (行24,52,65,69)；`is_cuda` 参数仅用于 rope 风格选择，非设备判断 |
| `ModelTypes.cc` | — | `allocBuf` 中 `DEVICE` 分配硬编码 `kCUDA` (行105)；运行时 `is_cuda()` 判打包类型 (行252) 漏 NPU；`cudaSyncAndCheck()` 无条件 (行64,85,299) |
| `PyWrappedModel.cc` | `#if USING_CUDA` 控制 CUDA stream 头文件包含 | `tensorHoldHostAndToCuda()` 中 `.is_cuda()` 判据漏 NPU (行28)；多处 `.to(torch::kCUDA)` / `.device(torch::kCUDA)` (行35,44,145,147,382,611)；`cudaCheckLastError()` 无条件 (行534,540) |
| `Sampler.cc` | — | 6 处硬编码 `torch::kCUDA` (行43,50,53-54,166-170)；ROCm 注释提及但无 NPU 保护 |
| `ExpertBalancer.cc` | — | 全部 tensor 分配硬编码 `torch::kCUDA` (行19,35-36,47,59-60,71,101+) |
| `BaseLogitsProcessor.cc` | `maskLogits` 的 CUDA kernel 路径有 `#if USING_CUDA` guard + `#else` PyTorch fallback | `generateVocabMask` 中 `.to(torch::kCUDA)` (行36) 未保护 |
| `MultiSeqLogitsProcessor.cc` | — | `.to(torch::kCUDA)` (行42) 硬编码 |
| `NormalModelInputGatherer.cc` | 有运行时 `is_cuda()` 判据防止重复拷贝 | `is_cuda()` 漏 NPU `is_privateuseone()` (行134) |
| `NormalOutputDispatcher.cc` | softmax 有 `#if USING_CUDA` + `#else` PyTorch fallback (行139-143) | label tensor `.to(torch::kCUDA)` (行119) 未保护 |
| `NormalSamplerInputGatherer.cc` | — | `all_probs` tensor `.device(torch::kCUDA)` (行64) 硬编码 |
| `MtpBatchStreamProcessor.cc` | — | 4 处 `.to(torch::kCUDA)` / `.device(torch::kCUDA)` (行388,424,477,488)；运行时 `is_cuda()` 判据漏 NPU (行234,361) |
| `MtpExecutor.cc` | — | 多处 `.device(torch::kCUDA)` / `.to(torch::kCUDA)` (行65,67,541,542,841,876)；`cudaSyncAndCheck()` ×3 (行397,646,689) + `cudaProfilerBegin()` (行213) 无条件 |
| `CudaOps.cc` | — | `BatchCopy` 内 `D2D/D2H/H2D` switch 全部硬编码 `torch::kCUDA`（2 处共 6 次，行106-114, 344-352） |
| `CudaSampleOp.cc` | — | 全部 `.to(torch::kCUDA)` / `.device(torch::kCUDA)` 硬编码（行40,65,68,71-74,97,104-105,135,140,226,243,350,358,383,386,389-392 等） |
| `CudaBeamSearchOp.cc` | — | workspace/output buffer 均硬编码 `.device(torch::kCUDA)` (行79,83,84) |
| `CudaXqa.cc` | — | 所有 tensor 硬编码 `.device(torch::kCUDA)` / `.to(torch::kCUDA)` (行30,41,48,87) |
| `GroupTopKOp.cc` | — | CUDA 专属 kernel 绑定 |
| `TRTAttnOp.cc` | — | CUDA 专属 kernel 绑定 |
| `FlashInferMlaParams.cc` | — | CUDA 专属 kernel 绑定 |
| `SparseMlaParams.cc` | — | CUDA 专属 kernel 绑定 |
| `TrtV2FmhaRunner.cc` | — | CUDA 专属 kernel 绑定 |
| `PagedAttn.cc` (bindings/rocm) | ROCm 目录，非目标 | 仍用 `torch::kCUDA` 作 device type（HIP 兼容） |
| `CKAttnUtils.h` (bindings/rocm) | ROCm 目录，非目标 | 同上 |
| `FusedRopeKVCacheOp.cc` (bindings/rocm) | ROCm 目录，非目标 | `.to(torch::kCUDA)` + `is_cuda()` 判据（行81-82） |
