#!/bin/bash
# vLLM Benchmark Docker 使用示例

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}vLLM Benchmark Docker Tool${NC}"

# 检查参数
if [ $# -lt 2 ]; then
    echo "用法: $0 <vllm_url> <api_key> [model_name]"
    echo ""
    echo "参数说明:"
    echo "  vllm_url    : vLLM 服务地址 (如 http://localhost:8000/v1)"
    echo "  api_key     : API 密钥"
    echo "  model_name  : 模型名称 (可选，默认: deepseek-v4-flash)"
    echo ""
    echo "示例:"
    echo "  $0 http://localhost:8000/v1 sk-xxx"
    echo "  $0 http://localhost:8000/v1 sk-xxx Qwen/Qwen2-7B-Instruct"
    exit 1
fi

VLLM_URL=$1
API_KEY=$2
MODEL_NAME=${3:-"deepseek-v4-flash"}

# 容器名称
CONTAINER_NAME="vllm-benchmark"

echo -e "${YELLOW}配置:${NC}"
echo "  vLLM URL  : $VLLM_URL"
echo "  Model     : $MODEL_NAME"
echo "  Output    : results/"
echo ""

# 确保 results 目录存在
mkdir -p results

# 构建镜像（如果需要）
if ! docker image inspect $CONTAINER_NAME:latest > /dev/null 2>&1; then
    echo -e "${YELLOW}构建镜像...${NC}"
    docker build -t $CONTAINER_NAME:latest .
fi

# 运行测试
echo -e "${GREEN}开始压测...${NC}"
docker run --rm \
    --network host \
    -v $(pwd)/results:/app/results \
    $CONTAINER_NAME:latest \
    --vllm_url "$VLLM_URL" \
    --api_key "$API_KEY" \
    --model "$MODEL_NAME" \
    --output_dir /app/results

echo -e "${GREEN}测试完成!${NC}"
echo "结果保存在: results/"