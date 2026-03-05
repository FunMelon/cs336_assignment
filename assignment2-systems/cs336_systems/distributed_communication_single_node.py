import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
from typing import List, Tuple, Dict

# 全局变量用于收集结果
results_collection = []

def setup(rank, world_size, backend):
    """
    初始化分布式训练环境
    
    Args:
        rank: 当前进程的编号
        world_size: 总进程数
        backend: 后端类型 ('gloo' 或 'nccl')
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()

def benchmark_all_reduce(rank, world_size, backend, device_type, data_size_mb):
    """
    对指定配置进行all-reduce基准测试
    
    Args:
        rank: 当前进程编号
        world_size: 总进程数
        backend: 通信后端 ('gloo' 或 'nccl')
        device_type: 设备类型 ('cpu' 或 'cuda')
        data_size_mb: 数据大小（MB）
    
    Returns:
        运行时间（秒）
    """
    setup(rank, world_size, backend)
    
    # 设置设备
    if device_type == "cuda":
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    # 计算需要的元素数量（float32 = 4 bytes）
    num_elements = int(data_size_mb * 1024 * 1024 / 4)
    
    # 创建随机数据张量
    data = torch.randn(num_elements, device=device, dtype=torch.float32)
    
    # 预热5次
    for _ in range(5):
        dist.all_reduce(data, op=dist.ReduceOp.SUM)
        if device_type == "cuda":
            torch.cuda.synchronize()
    
    # 进行实际的基准测试（10次取平均）
    num_iterations = 10
    start_time = time.time()
    
    for _ in range(num_iterations):
        dist.all_reduce(data, op=dist.ReduceOp.SUM)
        if device_type == "cuda":
            torch.cuda.synchronize()  # 确保GPU操作完成
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_iterations
    
    # 使用all-gather收集所有rank的结果
    # 创建一个张量来存储时间
    time_tensor = torch.tensor([avg_time], device=device)
    gathered_times = [torch.zeros(1, device=device) for _ in range(world_size)]
    dist.all_gather(gathered_times, time_tensor)
    
    # 只在rank 0收集结果
    result = None
    if rank == 0:
        times_list = [t.item() for t in gathered_times]
        avg_across_ranks = sum(times_list) / len(times_list)
        min_time = min(times_list)
        max_time = max(times_list)
        
        # 计算带宽 (GB/s)
        # all-reduce传输的数据量 = data_size_mb * 2 * (world_size - 1) / world_size
        data_size_gb = data_size_mb / 1024
        bandwidth = (data_size_gb * 2 * (world_size - 1) / world_size) / avg_across_ranks
        
        result = {
            'backend': backend,
            'device': device_type,
            'size_mb': data_size_mb,
            'world_size': world_size,
            'avg_time_ms': avg_across_ranks * 1000,
            'min_time_ms': min_time * 1000,
            'max_time_ms': max_time * 1000,
            'bandwidth_gbps': bandwidth,
        }
    
    cleanup()
    return result

def run_benchmark(rank, world_size, backend, device_type, data_size_mb, results_queue=None):
    """
    运行单个基准测试的包装函数
    
    Args:
        results_queue: 用于进程间传递结果的队列
    """
    try:
        result = benchmark_all_reduce(rank, world_size, backend, device_type, data_size_mb)
        # rank 0 将结果放入队列
        if rank == 0 and results_queue is not None and result is not None:
            results_queue.put(result)
    except Exception as e:
        print(f"Rank {rank} encountered error: {e}")

def print_table_header():
    """打印Markdown格式表格头部"""
    print("| Backend | Device | Size(MB) | Procs | Avg Time(ms) | Min Time(ms) | Max Time(ms) | Bandwidth(GB/s) |")
    print("|---------|--------|----------|-------|--------------|--------------|--------------|-----------------|")

def print_table_row(result: Dict):
    """打印Markdown格式表格行"""
    print(f"| {result['backend']:<7} | {result['device']:<6} | "
          f"{result['size_mb']:>8.0f} | {result['world_size']:>5} | "
          f"{result['avg_time_ms']:>12.4f} | {result['min_time_ms']:>12.4f} | "
          f"{result['max_time_ms']:>12.4f} | {result['bandwidth_gbps']:>15.4f} |")

def main():
    """
    主函数：运行所有基准测试配置
    """
    # 测试配置
    configs = [
        # (backend, device_type, data_sizes_mb, world_sizes)
        ("gloo", "cpu", [1, 10, 100, 1000], [2]),  # Gloo + CPU，只测试2进程
        ("nccl", "cuda", [1, 10, 100, 1000], [2]), # NCCL + GPU，只测试2进程（因为只有2卡）
    ]
    
    print("\n")
    print("=" * 100)
    print("All-Reduce Benchmark - Single Node Multi-Process Setup")
    print("=" * 100)
    print()
    
    # 收集所有结果
    all_results = []
    
    for backend, device_type, data_sizes, world_sizes in configs:
        # 检查CUDA可用性
        if device_type == "cuda" and not torch.cuda.is_available():
            print(f"Skipping {backend} + {device_type}: CUDA not available")
            continue
        
        # 检查GPU数量
        if device_type == "cuda":
            num_gpus = torch.cuda.device_count()
            print(f"Available GPUs: {num_gpus}")
            # 限制world_size不超过可用GPU数量
            world_sizes = [ws for ws in world_sizes if ws <= num_gpus]
        
        print(f"\n{'='*100}")
        print(f"Testing: {backend.upper()} + {device_type.upper()}")
        print(f"{'='*100}")
        
        # 为每个配置创建结果队列
        for world_size in world_sizes:
            for data_size_mb in data_sizes:
                # 创建队列用于收集结果
                results_queue = mp.Queue()
                
                # 使用spawn启动多进程
                mp.spawn(
                    fn=run_benchmark,
                    args=(world_size, backend, device_type, data_size_mb, results_queue),
                    nprocs=world_size,
                    join=True
                )
                
                # 从队列中获取结果
                if not results_queue.empty():
                    result = results_queue.get()
                    all_results.append(result)
                    print(f"Completed: Size={data_size_mb}MB, Procs={world_size}, "
                          f"Time={result['avg_time_ms']:.2f}ms, Bandwidth={result['bandwidth_gbps']:.2f}GB/s")
                
                # 短暂延迟，避免进程组冲突
                time.sleep(0.5)
    
    # 打印汇总表格
    if all_results:
        print("\n")
        print("=" * 100)
        print("SUMMARY TABLE")
        print("=" * 100)
        print()
        
        # 按后端和设备分组打印
        current_backend_device = None
        for result in all_results:
            backend_device = f"{result['backend']}+{result['device']}"
            if backend_device != current_backend_device:
                if current_backend_device is not None:
                    print()  # 组之间空一行
                print(f"## {result['backend'].upper()} + {result['device'].upper()}")
                print()
                print_table_header()
                current_backend_device = backend_device
            print_table_row(result)
        
        print()
    
    print("\n" + "=" * 100)
    print("Benchmark Completed!")
    print("=" * 100)
    print()

if __name__ == "__main__":
    # 设置多进程启动方法
    mp.set_start_method("spawn", force=True)
    main()
