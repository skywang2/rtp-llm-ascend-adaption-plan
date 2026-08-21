#!/bin/bash
# 一键验证：健康检查 + 推理测试
# 用法: bash verify.sh [PORT]
set -u
PORT="${1:-9100}"

echo "### 1) 健康检查"
curl -s --max-time 10 "http://127.0.0.1:$PORT/health"; echo
echo

echo "### 2) 模型列表"
curl -s --max-time 10 "http://127.0.0.1:$PORT/v1/models"; echo
echo

echo "### 3) 推理测试 (Qwen3-0.6B)"
curl -s --max-time 120 "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen","messages":[{"role":"user","content":"你是什么模型"}]}'
echo
