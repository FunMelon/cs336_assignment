"""
FlashAttention-2 基准测试脚本

使用 triton.testing.do_bench 对比以下两种实现的性能：
  1. 标准 PyTorch 注意力（直接材化 N×N 注意力矩阵，无 FlashAttention）
  2. Triton FlashAttention-2（自定义内核前向 + torch.compile 反向）

测试矩阵（笛卡尔积）：
  - 序列长度:  128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536
  - 嵌入维度:  16, 32, 64, 128
  - 精度:      torch.bfloat16, torch.float32

固定参数：batch_size=1, is_causal=True

输出：包含前向、反向、端到端延迟的 pandas 表格

用法：
    python -m cs336_systems.flash_benchmarking
    python -m cs336_systems.flash_benchmarking --save_csv results.csv
    python -m cs336_systems.flash_benchmarking --seq_lens 128 256 512 --dims 32 64
"""

import torch
import math
import itertools
import argparse
import pandas as pd
import triton.testing

from .flash_attention import FlashAttentionTriton


# =============================================================================
# 标准 PyTorch 注意力实现（作为对比基准）
# =============================================================================

def pytorch_standard_attention(Q, K, V, is_causal=False):
    """
    标准缩放点积注意力（非 FlashAttention）。
    直接材化完整的 N×N 注意力矩阵，在长序列时显存占用为 O(N²)。

    参数:
        Q, K, V: (batch_size, seq_len, d)
        is_causal: 是否使用因果掩码

    返回:
        O: (batch_size, seq_len, d)
    """
    scale = 1.0 / math.sqrt(Q.shape[-1])
    # S = Q @ K^T / sqrt(d)，形状 (B, N, N) —— 这就是 O(N²) 显存的来源
    S = torch.bmm(Q, K.transpose(1, 2)) * scale
    if is_causal:
        N = Q.shape[1]
        # 上三角掩码：位置 j > i 的元素设为 -inf（未来位置不可见）
        causal_mask = torch.triu(
            torch.ones(N, N, device=Q.device, dtype=torch.bool), diagonal=1
        )
        S.masked_fill_(causal_mask, float('-inf'))
    P = torch.softmax(S, dim=-1)
    O = torch.bmm(P, V)
    return O


# =============================================================================
# 基准测试核心函数
# =============================================================================

def bench_single_config(
    seq_len: int,
    d: int,
    dtype: torch.dtype,
    batch_size: int = 1,
    is_causal: bool = True,
    warmup: int = 25,
    rep: int = 100,
) -> dict:
    """
    对单个配置运行基准测试，返回延迟结果。

    测量 6 个延迟值：
      - PyTorch 标准注意力: 前向 / 反向 / 端到端
      - Triton FlashAttention: 前向 / 反向 / 端到端

    对于标准注意力在大序列长度下的 OOM 情况，记录为 NaN。
    """
    device = "cuda"
    result = {
        "seq_len": seq_len,
        "d": d,
        "dtype": str(dtype).split(".")[-1],
        # PyTorch 标准注意力
        "pytorch_fwd_ms": float('nan'),
        "pytorch_bwd_ms": float('nan'),
        "pytorch_fwd_bwd_ms": float('nan'),
        # Triton FlashAttention
        "triton_fwd_ms": float('nan'),
        "triton_bwd_ms": float('nan'),
        "triton_fwd_bwd_ms": float('nan'),
    }

    # ----- 生成输入数据 -----
    Q = torch.randn(batch_size, seq_len, d, device=device, dtype=dtype, requires_grad=True)
    K = torch.randn(batch_size, seq_len, d, device=device, dtype=dtype, requires_grad=True)
    V = torch.randn(batch_size, seq_len, d, device=device, dtype=dtype, requires_grad=True)

    # ======================== PyTorch 标准注意力 ========================
    try:
        # -- 前向 --
        def pytorch_fwd():
            return pytorch_standard_attention(Q.detach(), K.detach(), V.detach(), is_causal)

        result["pytorch_fwd_ms"] = triton.testing.do_bench(
            pytorch_fwd, warmup=warmup, rep=rep, return_mode="mean"
        )

        # -- 反向 --
        def pytorch_bwd():
            Q_ = Q.detach().requires_grad_(True)
            K_ = K.detach().requires_grad_(True)
            V_ = V.detach().requires_grad_(True)
            O = pytorch_standard_attention(Q_, K_, V_, is_causal)
            O.sum().backward()

        result["pytorch_bwd_ms"] = triton.testing.do_bench(
            pytorch_bwd, warmup=warmup, rep=rep, return_mode="mean"
        )

        # -- 端到端（前向+反向总时间）--
        result["pytorch_fwd_bwd_ms"] = result["pytorch_fwd_ms"] + result["pytorch_bwd_ms"]

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            # 标准注意力在长序列时会 OOM，标记为 NaN
            torch.cuda.empty_cache()
        else:
            raise

    # ======================== Triton FlashAttention ========================
    try:
        # -- 前向 --
        def triton_fwd():
            return FlashAttentionTriton.apply(
                Q.detach(), K.detach(), V.detach(), is_causal
            )

        result["triton_fwd_ms"] = triton.testing.do_bench(
            triton_fwd, warmup=warmup, rep=rep, return_mode="mean"
        )

        # -- 反向 --
        def triton_bwd():
            Q_ = Q.detach().requires_grad_(True)
            K_ = K.detach().requires_grad_(True)
            V_ = V.detach().requires_grad_(True)
            O = FlashAttentionTriton.apply(Q_, K_, V_, is_causal)
            O.sum().backward()

        result["triton_bwd_ms"] = triton.testing.do_bench(
            triton_bwd, warmup=warmup, rep=rep, return_mode="mean"
        )

        # -- 端到端 --
        result["triton_fwd_bwd_ms"] = result["triton_fwd_ms"] + result["triton_bwd_ms"]

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
        else:
            raise

    # 清理显存
    del Q, K, V
    torch.cuda.empty_cache()

    return result


# =============================================================================
# 主流程：扫描参数空间并生成报告
# =============================================================================

def run_benchmark(
    seq_lens: list[int] = None,
    dims: list[int] = None,
    dtypes: list[torch.dtype] = None,
    batch_size: int = 1,
    is_causal: bool = True,
) -> pd.DataFrame:
    """
    遍历所有参数组合，运行基准测试并返回结果表格。
    """
    if seq_lens is None:
        seq_lens = [2**i for i in range(7, 17)]  # 128 ~ 65536
    if dims is None:
        dims = [16, 32, 64, 128]
    if dtypes is None:
        dtypes = [torch.bfloat16, torch.float32]

    total = len(seq_lens) * len(dims) * len(dtypes)
    current = 0
    results = []

    print("=" * 90)
    print("FlashAttention-2 基准测试")
    print("=" * 90)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"batch_size: {batch_size} (固定)")
    print(f"is_causal:  {is_causal}")
    print(f"seq_lens:   {seq_lens}")
    print(f"dims:       {dims}")
    print(f"dtypes:     {[str(d).split('.')[-1] for d in dtypes]}")
    print(f"总配置数:   {total}")
    print("=" * 90)

    for dtype, d, seq_len in itertools.product(dtypes, dims, seq_lens):
        current += 1
        dtype_str = str(dtype).split(".")[-1]
        print(f"\n[{current}/{total}] seq_len={seq_len:>5}, d={d:>3}, dtype={dtype_str:>8}", end=" ... ")

        result = bench_single_config(
            seq_len=seq_len, d=d, dtype=dtype,
            batch_size=batch_size, is_causal=is_causal,
        )
        results.append(result)

        # 打印摘要
        pt_fwd = result["pytorch_fwd_ms"]
        tr_fwd = result["triton_fwd_ms"]
        speedup_str = ""
        if not (math.isnan(pt_fwd) or math.isnan(tr_fwd)) and tr_fwd > 0:
            speedup_str = f"(前向加速比: {pt_fwd / tr_fwd:.2f}x)"
        print(f"PyTorch fwd={pt_fwd:>10.3f} ms | Triton fwd={tr_fwd:>10.3f} ms {speedup_str}")

    df = pd.DataFrame(results)

    # 打印完整结果表
    print("\n")
    print("=" * 90)
    print("完整结果表")
    print("=" * 90)
    print(df.to_markdown(index=False, floatfmt=".3f"))

    # 打印各精度下的前向传播透视表
    for dtype in dtypes:
        dtype_str = str(dtype).split(".")[-1]
        sub = df[df["dtype"] == dtype_str]
        if sub.empty:
            continue

        print(f"\n--- {dtype_str} 前向传播延迟 (ms) ---")
        pivot_pt = sub.pivot(index="d", columns="seq_len", values="pytorch_fwd_ms")
        print(f"\nPyTorch 标准注意力:")
        print(pivot_pt.to_markdown(floatfmt=".3f"))

        pivot_tr = sub.pivot(index="d", columns="seq_len", values="triton_fwd_ms")
        print(f"\nTriton FlashAttention:")
        print(pivot_tr.to_markdown(floatfmt=".3f"))

        # 加速比透视表
        if not pivot_pt.empty and not pivot_tr.empty:
            speedup = pivot_pt / pivot_tr
            print(f"\n前向加速比 (PyTorch / Triton):")
            print(speedup.to_markdown(floatfmt=".2f"))

        print(f"\n--- {dtype_str} 端到端延迟 (ms) ---")
        pivot_pt_e2e = sub.pivot(index="d", columns="seq_len", values="pytorch_fwd_bwd_ms")
        print(f"\nPyTorch 标准注意力 (前向+反向):")
        print(pivot_pt_e2e.to_markdown(floatfmt=".3f"))

        pivot_tr_e2e = sub.pivot(index="d", columns="seq_len", values="triton_fwd_bwd_ms")
        print(f"\nTriton FlashAttention (前向+反向):")
        print(pivot_tr_e2e.to_markdown(floatfmt=".3f"))

    return df


# =============================================================================
# 命令行入口
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="FlashAttention-2 基准测试脚本")
    parser.add_argument("--seq_lens", type=int, nargs="+", default=None,
                        help="序列长度列表 (默认: 128 到 65536 的 2 的幂)")
    parser.add_argument("--dims", type=int, nargs="+", default=None,
                        help="嵌入维度列表 (默认: 16 32 64 128)")
    parser.add_argument("--dtypes", type=str, nargs="+", default=None,
                        choices=["bfloat16", "float32"],
                        help="精度列表 (默认: bfloat16 float32)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="批大小 (默认: 1)")
    parser.add_argument("--save_csv", type=str, default=None,
                        help="保存结果到 CSV 文件")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float32": torch.float32}
    dtypes = [dtype_map[d] for d in args.dtypes] if args.dtypes else None

    df = run_benchmark(
        seq_lens=args.seq_lens,
        dims=args.dims,
        dtypes=dtypes,
        batch_size=args.batch_size,
    )

    if args.save_csv:
        df.to_csv(args.save_csv, index=False)
        print(f"\n结果已保存到: {args.save_csv}")
