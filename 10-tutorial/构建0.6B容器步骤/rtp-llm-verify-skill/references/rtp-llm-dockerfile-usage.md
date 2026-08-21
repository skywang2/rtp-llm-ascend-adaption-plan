# rtp-llm Dockerfile 使用说明

## 概述

基于 `quay.io/wjunlu/vllm-ascend-daily:main-a5-openEuler-x86` 基础镜像，
构建包含 CANN 9.2.0~weekly.20260814.01 + Python 3.10 rtp-env 的 Ascend NPU 推理容器镜像。

## 核心版本信息

| 组件 | 版本 |
|------|------|
| CANN | 9.2.0~weekly.20260814.01 |
| Python | 3.10 |
| torch | 2.9.0+cpu |
| torch_npu | 2.9.0.post3 |
| triton-ascend | 3.2.2 |
| transformers | 4.51.2 |
| numpy | 2.2.6 |
| Bazel | 6.4.0 |

## 构建步骤

### 1. 构建镜像

```bash
# 可使用-f 指定 Dockerfile 路径，默认使用当前目录下的 Dockerfile
docker build -t rtp-llm-a5-image:cann-9.2.0-x86_64 .
```

> 构建预计耗时 15-30 分钟（视网络状况），CANN 安装包共约 4.7GB。
> pip 已配置阿里云镜像源加速，conda 自动同意 Terms of Service。

### 2. 创建容器

```bash
docker run -itd --name rtp-llm \
  --privileged --network host --security-opt label=disable --shm-size=1g \
  --hostname localhost.localdomain -w /workspace \
  -v /home:/home \
  -v /mnt:/workspace \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /root/.cache:/root/.cache \
  --device /dev/davinci0 --device /dev/davinci1 --device /dev/davinci2 --device /dev/davinci3 \
  --device /dev/davinci4 --device /dev/davinci5 --device /dev/davinci6 --device /dev/davinci7 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  rtp-llm-a5-image:cann-9.2.0-x86_64
```

### 3. 进入容器

```bash
docker exec -it rtp-llm bash
```

### 4. 激活 conda 环境

```bash
conda activate rtp-env
```

## 镜像内关键路径

| 路径 | 说明 |
|------|------|
| /usr/local/Ascend/cann-9.2.0/ | CANN 安装目录 |
| /usr/local/Ascend/cann | CANN 默认版本软链接 |
| /root/miniconda3/envs/rtp-env/ | Python 3.10 conda 环境 |
| /workspace/ | 工作目录 |

