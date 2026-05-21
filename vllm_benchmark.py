import asyncio
import time
import numpy as np
from openai import AsyncOpenAI
import logging
import argparse
import json
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SHORT_PROMPTS = [
    "Explain the concept of artificial intelligence in simple terms.",
    "What are the main causes of climate change?",
    "Describe the process of photosynthesis in plants.",
    "How does the human immune system work?",
    "What were the main causes of World War II?",
    "Explain the theory of relativity in layman's terms.",
    "What are the key principles of effective leadership?",
    "How does blockchain technology work?",
    "What are the main theories about the origin of the universe?",
    "Describe the water cycle and its importance for life on Earth.",
    "What are the major differences between capitalism and socialism?",
    "How does the human brain process and store memories?",
    "What are the main challenges in space exploration?",
    "Explain the concept of supply and demand in economics.",
]

LONG_PROMPT_PAIRS = [
    {
        "prompt": "Explain the concept of artificial intelligence in simple terms.",
        "context": "Artificial intelligence (AI) is a rapidly evolving field of computer science that aims to create intelligent machines that can perform tasks that typically require human intelligence. These tasks include visual perception, speech recognition, decision-making, and language translation. AI systems are designed to learn from experience, adjust to new inputs, and perform human-like tasks. The field of AI encompasses various subfields, including machine learning, neural networks, and deep learning, which have led to significant advancements in areas such as autonomous vehicles, virtual assistants, and recommendation systems."
    },
    {
        "prompt": "What are the main causes of climate change?",
        "context": "Climate change is a complex global phenomenon primarily driven by human activities that release greenhouse gases into the atmosphere. The burning of fossil fuels for energy, deforestation, industrial processes, and agriculture are major contributors to the increased concentration of carbon dioxide and other heat-trapping gases."
    },
    {
        "prompt": "Describe the process of photosynthesis in plants.",
        "context": "Photosynthesis is a fundamental biological process that allows plants to convert light energy into chemical energy. This process occurs in the chloroplasts of plant cells, specifically in structures called thylakoids."
    },
    {
        "prompt": "How does the human immune system work?",
        "context": "The human immune system is a complex network of cells, tissues, and organs that work together to defend the body against harmful pathogens."
    },
    {
        "prompt": "What were the main causes of World War II?",
        "context": "World War II, which lasted from 1939 to 1945, was one of the deadliest conflicts in human history. Its origins can be traced to several complex factors."
    },
    {
        "prompt": "Explain the theory of relativity in layman's terms.",
        "context": "Albert Einstein's theory of relativity, developed in the early 20th century, revolutionized our understanding of space, time, and gravity."
    },
    {
        "prompt": "What are the key principles of effective leadership?",
        "context": "Effective leadership is crucial in guiding organizations, teams, and individuals towards achieving their goals."
    },
    {
        "prompt": "How does blockchain technology work?",
        "context": "Blockchain is a decentralized, distributed ledger technology that underlies cryptocurrencies like Bitcoin, but has potential applications far beyond digital currencies."
    },
    {
        "prompt": "What are the main theories about the origin of the universe?",
        "context": "The origin of the universe has been a subject of intense scientific inquiry and philosophical debate for centuries."
    },
    {
        "prompt": "Describe the water cycle and its importance for life on Earth.",
        "context": "The water cycle, also known as the hydrologic cycle, is the continuous movement of water within the Earth and atmosphere."
    },
    {
        "prompt": "What are the major differences between capitalism and socialism?",
        "context": "Capitalism and socialism are two contrasting economic and political systems that have shaped much of modern history."
    },
    {
        "prompt": "How does the human brain process and store memories?",
        "context": "The human brain's ability to process and store memories is a complex and fascinating process involving various regions and neural networks."
    },
    {
        "prompt": "What are the main challenges in space exploration?",
        "context": "Space exploration, while offering immense potential for scientific discovery and technological advancement, faces numerous challenges."
    },
    {
        "prompt": "Explain the concept of supply and demand in economics.",
        "context": "Supply and demand is a fundamental concept in economics that describes how the price and quantity of a good or service in a market are determined."
    },
]

async def process_stream(stream):
    first_token_time = None
    total_tokens = 0
    async for chunk in stream:
        if first_token_time is None:
            first_token_time = time.time()
        if chunk.choices[0].delta.content:
            total_tokens += 1
        if chunk.choices[0].finish_reason is not None:
            break
    return first_token_time, total_tokens

async def make_request(client, model, output_tokens, request_timeout, use_long_context):
    start_time = time.time()
    if use_long_context:
        prompt_pair = random.choice(LONG_PROMPT_PAIRS)
        content = prompt_pair["context"] + "\n\n" + prompt_pair["prompt"]
    else:
        content = random.choice(SHORT_PROMPTS)

    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": content}
            ],
            max_tokens=output_tokens,
            stream=True
        )
        first_token_time, total_tokens = await asyncio.wait_for(process_stream(stream), timeout=request_timeout)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        ttft = first_token_time - start_time if first_token_time else None
        tokens_per_second = total_tokens / elapsed_time if elapsed_time > 0 else 0
        return total_tokens, elapsed_time, tokens_per_second, ttft

    except asyncio.TimeoutError:
        logging.warning(f"Request timed out after {request_timeout} seconds")
        return None
    except Exception as e:
        logging.error(f"Error during request: {str(e)}")
        return None

async def worker(client, model, semaphore, queue, results, output_tokens, request_timeout, use_long_context):
    while True:
        async with semaphore:
            task_id = await queue.get()
            if task_id is None:
                queue.task_done()
                break
            logging.info(f"Starting request {task_id}")
            result = await make_request(client, model, output_tokens, request_timeout, use_long_context)
            if result:
                results.append(result)
            else:
                logging.warning(f"Request {task_id} failed")
            queue.task_done()
            logging.info(f"Finished request {task_id}")

def calculate_percentile(values, percentile, reverse=False):
    if not values:
        return None
    if reverse:
        return np.percentile(values, 100 - percentile)
    return np.percentile(values, percentile)

async def run_benchmark(num_requests, concurrency, request_timeout, output_tokens, vllm_url, api_key, use_long_context, model=None):
    # 默认模型
    if model is None:
        model = "NousResearch/Meta-Llama-3.1-8B-Instruct"
    
    client = AsyncOpenAI(base_url=vllm_url, api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)
    queue = asyncio.Queue()
    results = []

    # Add tasks to the queue
    for i in range(num_requests):
        await queue.put(i)
    
    # Add sentinel values to stop workers
    for _ in range(concurrency):
        await queue.put(None)

    # Create worker tasks
    workers = [asyncio.create_task(worker(client, model, semaphore, queue, results, output_tokens, request_timeout, use_long_context)) for _ in range(concurrency)]

    start_time = time.time()
    
    # Wait for all tasks to complete
    await queue.join()
    await asyncio.gather(*workers)

    end_time = time.time()

    # Calculate metrics
    total_elapsed_time = end_time - start_time
    total_tokens = sum(tokens for tokens, _, _, _ in results if tokens is not None)
    latencies = [elapsed_time for _, elapsed_time, _, _ in results if elapsed_time is not None]
    tokens_per_second_list = [tps for _, _, tps, _ in results if tps is not None]
    ttft_list = [ttft for _, _, _, ttft in results if ttft is not None]

    successful_requests = len(results)
    requests_per_second = successful_requests / total_elapsed_time if total_elapsed_time > 0 else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    avg_tokens_per_second = sum(tokens_per_second_list) / len(tokens_per_second_list) if tokens_per_second_list else 0
    avg_ttft = sum(ttft_list) / len(ttft_list) if ttft_list else 0
    
    # Calculate percentiles
    percentiles = [50, 95, 99]
    latency_percentiles = [calculate_percentile(latencies, p) for p in percentiles]
    tps_percentiles = [calculate_percentile(tokens_per_second_list, p, reverse=True) for p in percentiles]
    ttft_percentiles = [calculate_percentile(ttft_list, p) for p in percentiles]
    
    return {
        "model": model,
        "total_requests": num_requests,
        "successful_requests": successful_requests,
        "concurrency": concurrency,
        "request_timeout": request_timeout,
        "max_output_tokens": output_tokens,
        "use_long_context": use_long_context,
        "total_time": total_elapsed_time,
        "requests_per_second": requests_per_second,
        "total_output_tokens": total_tokens,
        "latency": {
            "average": avg_latency,
            "p50": latency_percentiles[0],
            "p95": latency_percentiles[1],
            "p99": latency_percentiles[2]
        },
        "tokens_per_second": {
            "average": avg_tokens_per_second,
            "p50": tps_percentiles[0],
            "p95": tps_percentiles[1],
            "p99": tps_percentiles[2]
        },
        "time_to_first_token": {
            "average": avg_ttft,
            "p50": ttft_percentiles[0],
            "p95": ttft_percentiles[1],
            "p99": ttft_percentiles[2]
        }
    }

def print_results(results):
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark LLM with vLLM")
    parser.add_argument("--num_requests", type=int, required=True, help="Number of requests to make")
    parser.add_argument("--concurrency", type=int, required=True, help="Number of concurrent requests")
    parser.add_argument("--request_timeout", type=int, default=30, help="Timeout for each request in seconds (default: 30)")
    parser.add_argument("--output_tokens", type=int, default=50, help="Number of output tokens (default: 50)")
    parser.add_argument("--vllm_url", type=str, required=True, help="URL of the vLLM server")
    parser.add_argument("--api_key", type=str, required=True, help="API key for vLLM server")
    parser.add_argument("--model", type=str, default=None, help="Model name to benchmark (default: NousResearch/Meta-Llama-3.1-8B-Instruct)")
    parser.add_argument("--use_long_context", action="store_true", help="Use long context prompt pairs instead of short prompts")
    args = parser.parse_args()

    results = asyncio.run(run_benchmark(args.num_requests, args.concurrency, args.request_timeout, args.output_tokens, args.vllm_url, args.api_key, args.use_long_context, args.model))
    print_results(results)
else:
    # When imported as a module, provide the run_benchmark function
    __all__ = ['run_benchmark']