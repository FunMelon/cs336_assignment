"""
Attention基准测试脚本

测试不同规模下的注意力实现性能：
- 批大小固定为8，不使用多头注意力（无头维度）
- d_model: [16, 32, 64, 128]
- sequence_length: [256, 1024, 4096, 8192, 16384]
- 测试100次前向传播和100次反向传播
- 记录内存使用和时间
"""

import torch
import timeit
import argparse
import itertools
import pandas as pd
from cs336_basics.util import scaled_dot_product_attention

# ==================== 全局可配置参数 ====================
batch_size = 8                          # 批大小（固定）
d_model_list = [16, 32, 64, 128]        # 头嵌入维度列表
seq_len_list = [256, 1024, 4096, 8192, 16384]  # 序列长度列表
warmup_steps = 10                       # 预热步数
benchmark_steps = 100                   # 基准测试步数（前向/反向各100次）
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32                   # 数据类型


def generate_qkv(
    batch_size_: int,
    seq_len_: int,
    d_model_: int,
    device_: str,
    dtype_: torch.dtype = dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    生成随机的Q, K, V张量
    形状: (batch_size, seq_len, d_model)
    """
    Q = torch.randn(batch_size_, seq_len_, d_model_, device=device_, dtype=dtype_, requires_grad=True)
    K = torch.randn(batch_size_, seq_len_, d_model_, device=device_, dtype=dtype_, requires_grad=True)
    V = torch.randn(batch_size_, seq_len_, d_model_, device=device_, dtype=dtype_, requires_grad=True)
    return Q, K, V


def benchmark_attention_forward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    warmup_steps_: int,
    benchmark_steps_: int,
) -> float:
    """基准测试注意力前向传播"""
    device_type = Q.device.type
    
    # 预热步骤
    with torch.no_grad():
        for _ in range(warmup_steps_):
            _ = scaled_dot_product_attention(Q, K, V)
            if device_type == "cuda":
                torch.cuda.synchronize()
    
    # 基准测试
    times = []
    with torch.no_grad():
        for _ in range(benchmark_steps_):
            start = timeit.default_timer()
            _ = scaled_dot_product_attention(Q, K, V)
            if device_type == "cuda":
                torch.cuda.synchronize()
            end = timeit.default_timer()
            times.append(end - start)
    
    avg_time = sum(times) / len(times)
    return avg_time


def benchmark_attention_backward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    warmup_steps_: int,
    benchmark_steps_: int,
) -> tuple[float, float]:
    """
    基准测试注意力反向传播
    返回: (反向传播平均时间, 反向传播前内存使用MB)
    """
    device_type = Q.device.type
    
    # 预热步骤
    for _ in range(warmup_steps_):
        # 重新生成需要梯度的张量
        Q_ = Q.detach().clone().requires_grad_(True)
        K_ = K.detach().clone().requires_grad_(True)
        V_ = V.detach().clone().requires_grad_(True)
        
        output = scaled_dot_product_attention(Q_, K_, V_)
        loss = output.sum()
        loss.backward()
        if device_type == "cuda":
            torch.cuda.synchronize()
    
    # 清理内存以便准确测量
    if device_type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # 基准测试
    times = []
    memory_before_backward = 0.0
    
    for i in range(benchmark_steps_):
        # 重新生成需要梯度的张量
        Q_ = Q.detach().clone().requires_grad_(True)
        K_ = K.detach().clone().requires_grad_(True)
        V_ = V.detach().clone().requires_grad_(True)
        
        # 前向传播
        output = scaled_dot_product_attention(Q_, K_, V_)
        loss = output.sum()
        
        if device_type == "cuda":
            torch.cuda.synchronize()
        
        # 测量反向传播前的内存（第一次迭代）
        if i == 0 and device_type == "cuda":
            # 方法1: 当前已分配内存 (更准确反映实际使用)
            memory_before_backward = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            # 方法2: 也可以使用 max_memory_allocated() 获取峰值内存
            # memory_before_backward = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
        # 反向传播计时
        start = timeit.default_timer()
        loss.backward()
        if device_type == "cuda":
            torch.cuda.synchronize()
        end = timeit.default_timer()
        times.append(end - start)
    
    avg_time = sum(times) / len(times)
    return avg_time, memory_before_backward


def run_single_benchmark(
    batch_size_: int,
    seq_len_: int,
    d_model_: int,
    warmup_steps_: int,
    benchmark_steps_: int,
    device_: str,
    dtype_: torch.dtype,
) -> dict:
    """运行单个配置的基准测试"""
    
    # 生成Q, K, V
    Q, K, V = generate_qkv(batch_size_, seq_len_, d_model_, device_, dtype_)
    
    # 前向传播基准测试
    forward_time = benchmark_attention_forward(Q, K, V, warmup_steps_, benchmark_steps_)
    
    # 反向传播基准测试
    backward_time, memory_before_backward = benchmark_attention_backward(
        Q, K, V, warmup_steps_, benchmark_steps_
    )
    
    # 清理GPU内存
    del Q, K, V
    if device_.startswith("cuda"):
        torch.cuda.empty_cache()
    
    return {
        "batch_size": batch_size_,
        "seq_len": seq_len_,
        "d_model": d_model_,
        "forward_time_ms": forward_time * 1000,
        "backward_time_ms": backward_time * 1000,
        "memory_before_backward_MB": memory_before_backward,
    }


def run_benchmark(
    batch_size_: int = batch_size,
    d_model_list_: list = d_model_list,
    seq_len_list_: list = seq_len_list,
    warmup_steps_: int = warmup_steps,
    benchmark_steps_: int = benchmark_steps,
    device_: str = device,
    dtype_: torch.dtype = dtype,
    output_format: str = "both",
) -> pd.DataFrame:
    """运行完整的基准测试"""
    
    print("=" * 70)
    print("Attention 基准测试")
    print("=" * 70)
    print(f"设备: {device_}")
    print(f"数据类型: {dtype_}")
    print(f"基准测试配置:")
    print(f"  - batch_size: {batch_size_} (固定)")
    print(f"  - d_model: {d_model_list_}")
    print(f"  - seq_len: {seq_len_list_}")
    print(f"  - warmup_steps: {warmup_steps_}")
    print(f"  - benchmark_steps: {benchmark_steps_}")
    print(f"  - 多头注意力: 禁用 (头数=1)")
    print("=" * 70)
    
    results = []
    total_configs = len(d_model_list_) * len(seq_len_list_)
    current_config = 0
    
    # 遍历所有配置的笛卡尔积
    for d_model_, seq_len_ in itertools.product(d_model_list_, seq_len_list_):
        current_config += 1
        print(f"\n[{current_config}/{total_configs}] 测试配置: d_model={d_model_}, seq_len={seq_len_}")
        
        try:
            result = run_single_benchmark(
                batch_size_=batch_size_,
                seq_len_=seq_len_,
                d_model_=d_model_,
                warmup_steps_=warmup_steps_,
                benchmark_steps_=benchmark_steps_,
                device_=device_,
                dtype_=dtype_,
            )
            results.append(result)
            
            print(f"  前向传播时间: {result['forward_time_ms']:.4f} ms")
            print(f"  反向传播时间: {result['backward_time_ms']:.4f} ms")
            print(f"  反向传播前内存: {result['memory_before_backward_MB']:.2f} MB")
            
        except RuntimeError as e:
            print(f"  跳过配置 (内存不足或其他错误): {e}")
            results.append({
                "batch_size": batch_size_,
                "seq_len": seq_len_,
                "d_model": d_model_,
                "forward_time_ms": float('nan'),
                "backward_time_ms": float('nan'),
                "memory_before_backward_MB": float('nan'),
            })
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 70)
    print("基准测试完成！结果汇总:")
    print("=" * 70)
    
    # 输出结果表格
    if output_format in ["markdown", "both"]:
        print("\n### Markdown 格式表格:")
        print(df.to_markdown(index=False))
    
    if output_format in ["latex", "both"]:
        print("\n### LaTeX 格式表格:")
        print(df.to_latex(index=False))
    
    # 创建透视表（更直观的展示方式）
    print("\n### 前向传播时间 (ms) 透视表:")
    pivot_forward = df.pivot(index='d_model', columns='seq_len', values='forward_time_ms')
    print(pivot_forward.to_markdown())
    
    print("\n### 反向传播时间 (ms) 透视表:")
    pivot_backward = df.pivot(index='d_model', columns='seq_len', values='backward_time_ms')
    print(pivot_backward.to_markdown())
    
    print("\n### 反向传播前内存 (MB) 透视表:")
    pivot_memory = df.pivot(index='d_model', columns='seq_len', values='memory_before_backward_MB')
    print(pivot_memory.to_markdown())
    
    return df


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Attention 基准测试脚本")
    
    parser.add_argument("--batch_size", type=int, default=batch_size, 
                       help="批大小 (默认: 8)")
    parser.add_argument("--d_model", type=int, nargs='+', default=d_model_list,
                       help="d_model列表 (默认: 16 32 64 128)")
    parser.add_argument("--seq_len", type=int, nargs='+', default=seq_len_list,
                       help="序列长度列表 (默认: 256 1024 4096 8192 16384)")
    parser.add_argument("--warmup_steps", type=int, default=warmup_steps,
                       help="预热步数 (默认: 10)")
    parser.add_argument("--benchmark_steps", type=int, default=benchmark_steps,
                       help="基准测试步数 (默认: 100)")
    parser.add_argument("--device", type=str, default=device,
                       help="设备 (cuda/cpu)")
    parser.add_argument("--dtype", type=str, default="float32",
                       choices=["float32", "float16", "bfloat16"],
                       help="数据类型 (默认: float32)")
    parser.add_argument("--output_format", type=str, default="both",
                       choices=["markdown", "latex", "both"],
                       help="输出格式 (默认: both)")
    parser.add_argument("--save_csv", type=str, default=None,
                       help="保存结果到CSV文件")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # 解析数据类型
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype_ = dtype_map[args.dtype]
    
    # 运行基准测试
    df = run_benchmark(
        batch_size_=args.batch_size,
        d_model_list_=args.d_model,
        seq_len_list_=args.seq_len,
        warmup_steps_=args.warmup_steps,
        benchmark_steps_=args.benchmark_steps,
        device_=args.device,
        dtype_=dtype_,
        output_format=args.output_format,
    )
    
    # 保存结果到CSV
    if args.save_csv:
        df.to_csv(args.save_csv, index=False)
        print(f"\n结果已保存到: {args.save_csv}")
