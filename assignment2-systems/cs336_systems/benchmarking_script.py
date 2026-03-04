from cs336_basics import Transformer
import torch
import timeit
import argparse

# ==================== 全局可配置参数 ====================
# 模型超参数
vocab_size = 10000          # 词汇表大小
context_length = 512        # 最大上下文长度
d_model = 2560              # 模型隐藏维度
d_ff = 10240                # 前馈网络隐藏层维度
num_layers = 32             # Transformer块数量
num_heads = 32              # 注意力头数量

# 基准测试参数
batch_size = 8              # 批大小
warmup_steps = 5            # 预热步数
benchmark_steps = 10        # 基准测试步数
enable_memory_profiling = False  # 是否启用内存分析
enable_autocast = False     # 是否启用混合精度训练
autocast_dtype = torch.bfloat16  # 混合精度数据类型
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32       # 数据类型


def create_model(
    vocab_size_: int = vocab_size,
    context_length_: int = context_length,
    d_model_: int = d_model,
    d_ff_: int = d_ff,
    num_layers_: int = num_layers,
    num_heads_: int = num_heads,
    device_: str = device,
    dtype_: torch.dtype = dtype,
) -> Transformer:
    """创建Transformer模型"""
    model = Transformer(
        vocab_size=vocab_size_,
        context_length=context_length_,
        d_model=d_model_,
        d_ff=d_ff_,
        num_layers=num_layers_,
        nhead=num_heads_,
        device=device_,
        dtype=dtype_,
    )
    return model.to(device_)


def generate_random_batch(
    batch_size_: int,
    context_length_: int,
    vocab_size_: int,
    device_: str,
) -> torch.Tensor:
    """生成随机输入数据批次"""
    return torch.randint(
        low=0,
        high=vocab_size_,
        size=(batch_size_, context_length_),
        device=device_,
    )


def benchmark_forward(
    model: Transformer,
    input_data: torch.Tensor,
    warmup_steps_: int,
    benchmark_steps_: int,
    enable_autocast_: bool = enable_autocast,
    autocast_dtype_: torch.dtype = autocast_dtype,
) -> float:
    """基准测试前向传播"""
    model.eval()
    device_type = input_data.device.type
    
    # 预热步骤
    with torch.no_grad():
        for _ in range(warmup_steps_):
            if enable_autocast_ and device_type == "cuda":
                with torch.autocast(device_type="cuda", dtype=autocast_dtype_):
                    _ = model(input_data)
            else:
                _ = model(input_data)
            if device_type == "cuda":
                torch.cuda.synchronize()
    
    # 基准测试
    times = []
    with torch.no_grad():
        for _ in range(benchmark_steps_):
            start = timeit.default_timer()
            if enable_autocast_ and device_type == "cuda":
                with torch.autocast(device_type="cuda", dtype=autocast_dtype_):
                    _ = model(input_data)
            else:
                _ = model(input_data)
            if device_type == "cuda":
                torch.cuda.synchronize()
            end = timeit.default_timer()
            times.append(end - start)
    
    avg_time = sum(times) / len(times)
    return avg_time


def benchmark_forward_backward(
    model: Transformer,
    input_data: torch.Tensor,
    warmup_steps_: int,
    benchmark_steps_: int,
    enable_autocast_: bool = enable_autocast,
    autocast_dtype_: torch.dtype = autocast_dtype,
) -> float:
    """基准测试前向和反向传播"""
    model.train()
    device_type = input_data.device.type
    
    # 预热步骤
    for _ in range(warmup_steps_):
        if enable_autocast_ and device_type == "cuda":
            with torch.autocast(device_type="cuda", dtype=autocast_dtype_):
                logits = model(input_data)
                # 简单的损失计算：对logits求和
                loss = logits.sum()
        else:
            logits = model(input_data)
            loss = logits.sum()
        loss.backward()
        model.zero_grad()
        if device_type == "cuda":
            torch.cuda.synchronize()
    
    # 基准测试
    times = []
    for _ in range(benchmark_steps_):
        start = timeit.default_timer()
        if enable_autocast_ and device_type == "cuda":
            with torch.autocast(device_type="cuda", dtype=autocast_dtype_):
                logits = model(input_data)
                loss = logits.sum()
        else:
            logits = model(input_data)
            loss = logits.sum()
        loss.backward()
        if device_type == "cuda":
            torch.cuda.synchronize()
        end = timeit.default_timer()
        times.append(end - start)
        model.zero_grad()
    
    avg_time = sum(times) / len(times)
    return avg_time


def run_benchmark(
    d_model_: int = d_model,
    d_ff_: int = d_ff,
    num_layers_: int = num_layers,
    num_heads_: int = num_heads,
    batch_size_: int = batch_size,
    context_length_: int = context_length,
    vocab_size_: int = vocab_size,
    warmup_steps_: int = warmup_steps,
    benchmark_steps_: int = benchmark_steps,
    forward_only: bool = False,
    enable_memory_profiling_: bool = enable_memory_profiling,
    enable_autocast_: bool = enable_autocast,
    autocast_dtype_: torch.dtype = autocast_dtype,
    device_: str = device,
    dtype_: torch.dtype = dtype,
) -> dict:
    """运行完整的基准测试"""

    print("=" * 60)
    print("Transformer 基准测试")
    print("=" * 60)
    print(f"设备: {device_}")
    print(f"数据类型: {dtype_}")
    print(f"模型配置:")
    print(f"  - vocab_size: {vocab_size_}")
    print(f"  - context_length: {context_length_}")
    print(f"  - d_model: {d_model_}")
    print(f"  - d_ff: {d_ff_}")
    print(f"  - num_layers: {num_layers_}")
    print(f"  - num_heads: {num_heads_}")
    print(f"基准测试配置:")
    print(f"  - batch_size: {batch_size_}")
    print(f"  - warmup_steps: {warmup_steps_}")
    print(f"  - benchmark_steps: {benchmark_steps_}")
    print(f"  - forward_only: {forward_only}")
    print(f"  - enable_memory_profiling: {enable_memory_profiling_}")
    print(f"  - enable_autocast: {enable_autocast_}")
    if enable_autocast_:
        print(f"  - autocast_dtype: {autocast_dtype_}")
    print("=" * 60)
    
    # 创建模型
    model = create_model(
        vocab_size_=vocab_size_,
        context_length_=context_length_,
        d_model_=d_model_,
        d_ff_=d_ff_,
        num_layers_=num_layers_,
        num_heads_=num_heads_,
        device_=device_,
        dtype_=dtype_,
    )
    
    # 打印模型参数量
    num_params = model.compute_params()
    print(f"模型参数量: {num_params:,} ({num_params / 1e6:.2f}M)")
    
    # 生成随机数据
    input_data = generate_random_batch(batch_size_, context_length_, vocab_size_, device_)

    # ==================== 内存分析：开始记录 ====================
    if enable_memory_profiling_ and device_.startswith("cuda"):
        print("\n[Memory Profiling] 开始记录内存历史...")
        torch.cuda.memory._record_memory_history(max_entries=1000000)
    # ============================================================

    # 运行基准测试
    if forward_only:
        avg_time = benchmark_forward(
            model, input_data, warmup_steps_, benchmark_steps_,
            enable_autocast_, autocast_dtype_
        )
        mode = "Forward Only"
    else:
        avg_time = benchmark_forward_backward(
            model, input_data, warmup_steps_, benchmark_steps_,
            enable_autocast_, autocast_dtype_
        )
        mode = "Forward + Backward"

    # ==================== 内存分析：保存快照并停止记录 ====================
    if enable_memory_profiling_ and device_.startswith("cuda"):
        print("[Memory Profiling] 保存内存快照到 memory_snapshot.pickle ...")
        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        print("[Memory Profiling] 停止记录内存历史...")
        torch.cuda.memory._record_memory_history(enabled=None)
    # ===================================================================
    
    # 计算吞吐量
    tokens_per_step = batch_size_ * context_length_
    throughput = tokens_per_step / avg_time
    
    print(f"\n结果 ({mode}):")
    print(f"  - 平均每步时间: {avg_time * 1000:.4f} ms")
    print(f"  - 吞吐量: {throughput:.2f} tokens/s")
    print("=" * 60)
    
    return {
        "mode": mode,
        "avg_time_ms": avg_time * 1000,
        "throughput_tokens_per_sec": throughput,
        "num_params": num_params,
    }


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Transformer 基准测试脚本")
    
    # 模型参数
    parser.add_argument("--vocab_size", type=int, default=vocab_size, help="词汇表大小")
    parser.add_argument("--context_length", type=int, default=context_length, help="上下文长度")
    parser.add_argument("--d_model", type=int, default=d_model, help="模型隐藏维度")
    parser.add_argument("--d_ff", type=int, default=d_ff, help="前馈网络隐藏层维度")
    parser.add_argument("--num_layers", type=int, default=num_layers, help="Transformer层数")
    parser.add_argument("--num_heads", type=int, default=num_heads, help="注意力头数")
    
    # 基准测试参数
    parser.add_argument("--batch_size", type=int, default=batch_size, help="批大小")
    parser.add_argument("--warmup_steps", type=int, default=warmup_steps, help="预热步数")
    parser.add_argument("--benchmark_steps", type=int, default=benchmark_steps, help="基准测试步数")
    parser.add_argument("--forward_only", action="store_true", help="仅测试前向传播")
    parser.add_argument("--enable_memory_profiling", action="store_true", help="启用内存分析并保存快照")
    parser.add_argument("--enable_autocast", action="store_true", help="启用混合精度训练(autocast)")
    parser.add_argument("--autocast_dtype", type=str, default="bfloat16",
                       choices=["float16", "bfloat16"], help="混合精度数据类型")
    
    # 设备参数
    parser.add_argument("--device", type=str, default=device, help="设备 (cuda/cpu)")
    parser.add_argument("--dtype", type=str, default="float32", 
                       choices=["float32", "float16", "bfloat16"], help="数据类型")
    
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
    
    # 解析混合精度数据类型
    autocast_dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    autocast_dtype_ = autocast_dtype_map[args.autocast_dtype]
    
    # 运行基准测试
    results = run_benchmark(
        d_model_=args.d_model,
        d_ff_=args.d_ff,
        num_layers_=args.num_layers,
        num_heads_=args.num_heads,
        batch_size_=args.batch_size,
        context_length_=args.context_length,
        vocab_size_=args.vocab_size,
        warmup_steps_=args.warmup_steps,
        benchmark_steps_=args.benchmark_steps,
        forward_only=args.forward_only,
        enable_memory_profiling_=args.enable_memory_profiling,
        enable_autocast_=args.enable_autocast,
        autocast_dtype_=autocast_dtype_,
        device_=args.device,
        dtype_=dtype_,
    )
