import asyncio
import json
import time
import argparse
import os
from datetime import datetime
from vllm_benchmark import run_benchmark

async def run_all_benchmarks(vllm_url, api_key, use_long_context, model):
    configurations = [
        {"num_requests": 10, "concurrency": 1, "output_tokens": 100},
        {"num_requests": 100, "concurrency": 10, "output_tokens": 100},
        {"num_requests": 500, "concurrency": 50, "output_tokens": 100},
        {"num_requests": 1000, "concurrency": 100, "output_tokens": 100},
    ]

    all_results = []

    for config in configurations:
        print(f"Running benchmark with concurrency {config['concurrency']}...")
        results = await run_benchmark(
            config['num_requests'], 
            config['concurrency'], 
            30, 
            config['output_tokens'], 
            vllm_url, 
            api_key, 
            use_long_context,
            model
        )
        all_results.append(results)
        time.sleep(5)  # Wait a bit between runs to let the system cool down

    return all_results

def main():
    parser = argparse.ArgumentParser(description="Run vLLM benchmarks with various configurations")
    parser.add_argument("--vllm_url", type=str, required=True, help="URL of the vLLM server")
    parser.add_argument("--api_key", type=str, required=True, help="API key for vLLM server")
    parser.add_argument("--model", type=str, default=None, help="Model name to benchmark (default: NousResearch/Meta-Llama-3.1-8B-Instruct)")
    parser.add_argument("--use_long_context", action="store_true", help="Use long context prompt pairs instead of short prompts")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory for results (default: results)")
    args = parser.parse_args()

    # 获取模型名称（用于文件名）
    model_name = args.model if args.model else "NousResearch/Meta-Llama-3.1-8B-Instruct"
    # 处理模型名称中的特殊字符，替换为下划线
    safe_model_name = model_name.replace("/", "_").replace("-", "_")
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成文件名：模型名称_时间戳.json
    output_filename = f"{safe_model_name}_{timestamp}.json"
    output_path = os.path.join(args.output_dir, output_filename)

    # 运行测试
    all_results = asyncio.run(run_all_benchmarks(args.vllm_url, args.api_key, args.use_long_context, args.model))

    # 保存结果
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nBenchmark results saved to: {output_path}")
    
    # 打印摘要
    print("\n=== Test Summary ===")
    for result in all_results:
        print(f"Concurrency {result['concurrency']}: QPS={result['requests_per_second']:.2f}, "
              f"Latency={result['latency']['average']:.2f}s, "
              f"Tokens/s={result['tokens_per_second']['average']:.2f}, "
              f"Success={result['successful_requests']}/{result['total_requests']}")

if __name__ == "__main__":
    main()