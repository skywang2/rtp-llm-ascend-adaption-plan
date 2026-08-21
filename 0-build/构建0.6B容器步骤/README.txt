## rtp-llm-verify-skill 介绍
1. 根据Dockerfile构建容器镜像
2. 验证rtp-llm推理服务能否拉起并完成Qwen3-0.6B推理

## 使用dockerfile构建容器
1. 参考rtp-llm-dockerfile-usage.md构建容器
2. 参考rtp_llm_workflow_guide.md，第3章，第9~14章，可以让ai辅助执行修改

## 手工构建容器
1. 参考rtp_llm_workflow_guide.md下载指定镜像并按文档操作，可以让ai辅助执行修改

## 独立算子包
1. ops-transformer独立算子包仅在CANN 9.1.0-beta.3版本需要安装，CANN 9.2.0已内置所需算子，无需额外安装
