# vLLM Benchmark

使用 Docker 容器对 vLLM 推理服务进行性能压测的工具。

## 功能特性

- 支持不同并发级别压测（1/10/50/100）
- 关键性能指标：QPS、延迟、Tokens/秒、TTFT（首Token时间）
- 结果自动保存到 `results/` 目录，文件名带模型名和时间戳
- 支持自定义模型名称
- 支持长/短上下文测试

## 快速开始

### 1. 构建镜像

```bash
docker build -t vllm-benchmark:latest .
```

### 2. 运行测试

```bash
# 使用封装脚本（推荐）
chmod +x run-docker.sh
./run-docker.sh http://your-vllm-host:8000/v1 "your-api-key" "model-name"

# 或者直接使用 docker run
docker run --rm -v $(pwd)/results:/app/results vllm-benchmark:latest \
    --vllm_url "http://your-vllm-host:8000/v1" \
    --api_key "your-api-key" \
    --model "deepseek-v4-flash" \
    --output_dir /app/results
```

### 3. 查看结果

```bash
ls results/
# 输出示例: deepseek_v4_flash_20260520_143022.json
```

## 参数说明

| 参数 | 说明 | 必填 | 默认值 |
|------|------|------|--------|
| `--vllm_url` | vLLM 服务地址 | 是 | - |
| `--api_key` | API 密钥 | 是 | - |
| `--model` | 模型名称 | 否 | deepseek-v4-flash |
| `--output_dir` | 结果输出目录 | 否 | results |
| `--use_long_context` | 使用长上下文测试 | 否 | false |

## 输出指标说明

- **requests_per_second (QPS)**: 每秒处理请求数
- **latency**: 响应延迟（average/p50/p95/p99）
- **tokens_per_second**: 令牌生成速度
- **time_to_first_token (TTFT)**: 首Token响应时间
- **successful_requests**: 成功请求数/总请求数

## 项目结构

```
vllm-benchmark/
├── vllm_benchmark.py     # 核心压测逻辑
├── run_benchmarks.py     # 批量测试入口
├── run-docker.sh         # Docker 运行脚本
├── Dockerfile            # Docker 镜像构建文件
├── results/              # 测试结果目录
└── README.md             # 本文件
```

## 依赖

- Python 3.11+
- openai
- numpy

## License

Apache 2.0 License