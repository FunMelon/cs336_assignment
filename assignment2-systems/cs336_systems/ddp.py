import os
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from contextlib import nullcontext
from torch.amp import autocast, GradScaler
import time
import csv
import matplotlib.pyplot as plt
import pandas as pd
import tqdm

from cs336_basics import (
    Transformer,
    Muon,
    AdamW,
    get_batch,
    cross_entropy_loss,
    cosine_anneal_schedule,
    gradient_clipping,
)


# ==================== 训练配置 ====================
# 输出和数据集路径
output_path = "./out"
train_dataset_path = "../../../cs336_data/id/owt-t-id/owt_train.bin"
valid_dataset_path = "../../../cs336_data/id/owt-v-id/owt_valid.bin"

# 模型超参数
vocab_size = 32000
context_length = 1024
d_model = 768           # 768/12 = 64 (2的幂，支持Flash Attention)
nhead = 12
num_layers = 12
d_ff = 2048
rope_theta = 10000.0
logit_cap = 30.0
use_flash_attention = True
dtype = torch.float32

# 训练超参数
batch_size = 64         # 每个GPU的batch size
iteration = 20000       # 总训练步数
saving_interval = -1    # Checkpoint保存间隔（-1=不保存中间checkpoint）
valid_frequency = 100  # 验证频率
valid_batch_multiples = 8  # 验证时使用的batch数量
accumulation_steps = 1  # 梯度累积步数

# 学习率调度参数
warmup_ratio = 0.05     # 预热阶段占总步数的比例
lr_decay_ratio = 0.1    # 学习率从峰值衰减到最小值的比例

# 梯度裁剪参数
max_grad_norm = 1.0

# 优化器参数
# Muon 参数 (用于 2D 权重矩阵)
muon_lr = 0.02
muon_momentum = 0.95
muon_weight_decay = 0.0
ns_steps = 5

# AdamW 参数 (用于 1D 参数、Embedding 等)
adamw_lr = 1e-3
adamw_betas = (0.9, 0.95)
adamw_eps = 1e-8
adamw_weight_decay = 1e-2

# 性能优化参数
enable_tf32 = False     # TF32优化（Ampere+ GPU）
enable_amp = False      # 混合精度训练（AMP）
amp_dtype = 'float16'   # AMP数据类型：'float16' 或 'bfloat16'
enable_compile = False  # torch.compile优化（需要PyTorch 2.0+）

# 分布式训练参数
world_size = torch.cuda.device_count()  # 可用的GPU数量
dist_url = "env://"


def setup_distributed(rank, world_size):
    """设置分布式训练环境"""
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=dist_url,
        world_size=world_size,
        rank=rank
    )
    torch.manual_seed(42 + rank)


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def evaluate_validation_loss(
    model: torch.nn.Module,
    dataset: np.memmap,
    batch_size: int,
    context_length: int,
    device: str,
    num_batches: int = 5,
    enable_amp: bool = False,
    amp_dtype: torch.dtype = torch.float32,
    rank: int = 0,
    world_size: int = 1,
) -> float:
    """评估验证集上的平均损失（支持AMP加速 + 分布式并行验证）
    
    并行策略：
    - 所有GPU并行评估不同的batch
    - 使用 all_reduce 汇总所有进程的损失
    - 验证速度提升 world_size 倍
    """
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(num_batches):
            input_batch, target_batch = get_batch(
                dataset,
                batch_size=batch_size,
                context_length=context_length,
                device=device,
            )
            input_batch = input_batch.to(dtype=torch.int)
            target_batch = target_batch.to(dtype=torch.int)

            # 使用AMP加速验证（不需要GradScaler，因为没有反向传播）
            with autocast('cuda', dtype=amp_dtype, enabled=enable_amp):
                logits = model(input_batch)
                val_loss = cross_entropy_loss(logits, target_batch)
            
            losses.append(val_loss.item())
    
    # 计算本地平均损失
    local_avg_loss = sum(losses) / len(losses)
    
    # 分布式环境：使用 all_reduce 汇总所有进程的损失
    if world_size > 1:
        loss_tensor = torch.tensor(local_avg_loss, device=device, dtype=torch.float32)
        # 所有进程的损失求和
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        # 计算全局平均
        global_avg_loss = (loss_tensor / world_size).item()
    else:
        global_avg_loss = local_avg_loss
    
    model.train()
    return global_avg_loss


def save_log(
    log_path: str,
    step: int,
    wallclock_time: float,
    train_loss: float,
    val_loss: float | None,
    lr: float,
    rank: int,
) -> None:
    """保存日志到CSV文件（只在rank 0进程保存）"""
    if rank != 0:
        return
        
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                step,
                wallclock_time,
                train_loss,
                val_loss,
                lr,
            ]
        )


def plot_logs(log_path: str, output_dir: str, rank: int) -> None:
    """绘制训练和验证损失曲线（只在rank 0进程执行）"""
    if rank != 0:
        return
        
    df = pd.read_csv(log_path)
    plt.figure(figsize=(10, 6))
    plt.plot(df["step"], df["train_loss"], label="Train Loss")
    plt.plot(df["step"], df["val_loss"], label="Validation Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Iterations")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()


def train_worker(rank, world_size, config):
    """每个训练进程的入口函数"""
    try:
        setup_distributed(rank, world_size)
        device = f"cuda:{rank}"
        
        # 设置TF32（如果启用）
        if config.get('enable_tf32', False):
            torch.set_float32_matmul_precision('high')
            if rank == 0:
                print("TF32 enabled for float32 matmul")
        
        # 创建模型
        model = Transformer(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            d_ff=d_ff,
            rope_theta=rope_theta,
            logit_cap=logit_cap,
            use_flash_attention=use_flash_attention,
            device=device,
            tie_weights=False,
        )
        
        # 使用torch.compile优化（如果启用）- 必须在DDP包装之前！
        if config.get('enable_compile', False):
            compile_mode = config.get('compile_mode', 'default')
            model = torch.compile(model, mode=compile_mode)
            if rank == 0:
                print(f"torch.compile enabled (mode: {compile_mode})")
        
        # 使用DistributedDataParallel包装模型（手动控制通信优化）
        # DDP优化模式：
        #   0: 无优化（朴素DDP，每参数单独通信）
        #   1: 只扁平化（避免分桶overhead）
        #   2: 分桶（启用计算通信重叠）
        
        ddp_mode = config.get('ddp_mode', 1)
        
        if ddp_mode == 0:
            # 模式0：无优化（朴素DDP）
            bucket_size = 1  # 最小桶 = 几乎每个参数单独通信
            use_bucket_view = False
            optimization_level = "naive (no optimization)"
            
        elif ddp_mode == 1:
            # 模式1：只用优化一（扁平化）
            bucket_size = 25000  # 超大桶 = 不分桶 = 只扁平化
            use_bucket_view = False
            optimization_level = "flatten-only (default)"
            
        else:  # ddp_mode == 2
            # 模式2：优化一+二+三（分桶）
            bucket_size = 25
            use_bucket_view = True
            optimization_level = "bucketing+overlap"
        
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            output_device=rank,
            bucket_cap_mb=bucket_size,
            gradient_as_bucket_view=use_bucket_view,
            broadcast_buffers=False,  # Transformer不需要同步buffer（无BatchNorm）
            static_graph=config.get('enable_compile', False),  # torch.compile时启用
        )
        
        if rank == 0:
            print("\n" + "=" * 80)
            print("DDP Training with cs336 Transformer")
            print("=" * 80)
            num_params = sum(p.numel() for p in model.parameters())
            print(f"Model: {num_params:,} parameters ({num_params/1e6:.2f}M)")
            if use_flash_attention:
                print("FlashAttention enabled (using Triton kernels)")
            print(f"GPUs: {world_size} | Batch/GPU: {batch_size} | Global Batch: {batch_size * world_size}")
            print(f"Validation: Parallel across {world_size} GPUs (speedup: {world_size}x)")
            
            print(f"\nDDP Communication Strategy: Mode {ddp_mode} - {optimization_level}")
            print(f"  - Bucket Size: {bucket_size}MB")
            print(f"  - Zero-Copy Gradients: {'Enabled' if use_bucket_view else 'Disabled'}")
            print(f"  - Broadcast Buffers: Disabled (no BatchNorm)")
            
            if ddp_mode == 0:
                print(f"\n  ⚠️  Mode 0: 朴素DDP（无优化）")
                print(f"     - 每个参数单独通信")
            elif ddp_mode == 1:
                print(f"\n  ✅ Mode 1: 只扁平化（推荐2-3卡）")
                print(f"     - 所有梯度拼接后一次通信")
                print(f"     - 避免分桶和重叠的overhead")
            else:  # ddp_mode == 2
                print(f"\n  ✅ Mode 2: 标准分桶（推荐4-7卡）")
                print(f"     - 梯度分桶")
                print(f"     - 计算与通信重叠")
            
            print("=" * 80 + "\n")
        
        # 分离参数：2D 权重矩阵用 Muon，其他参数用 AdamW
        muon_params = []
        adamw_decay_params = []
        adamw_nodecay_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.ndim == 2 and 'embedding' not in name.lower():
                    muon_params.append(param)
                elif 'embedding' in name.lower():
                    adamw_decay_params.append(param)
                else:
                    adamw_nodecay_params.append(param)
        
        if rank == 0:
            print(f"Optimizer Setup:")
            print(f"  - Muon params: {len(muon_params)} tensors (2D weights)")
            print(f"  - AdamW params (w/ decay): {len(adamw_decay_params)} tensors (embeddings)")
            print(f"  - AdamW params (no decay): {len(adamw_nodecay_params)} tensors (bias, norm)\n")
        
        # 创建混合优化器
        muon_opt = Muon(
            muon_params,
            lr=muon_lr,
            momentum=muon_momentum,
            weight_decay=muon_weight_decay,
            ns_steps=ns_steps,
        )
        adamw_opt = AdamW(
            [
                {'params': adamw_decay_params, 'weight_decay': adamw_weight_decay},
                {'params': adamw_nodecay_params, 'weight_decay': 0.0},
            ],
            lr=adamw_lr,
            betas=adamw_betas,
            eps=adamw_eps,
        )
        
        # ==================== AMP设置 ====================
        enable_amp = config.get('enable_amp', False)
        amp_dtype_str = config.get('amp_dtype', 'bfloat16')
        amp_dtype = torch.bfloat16 if amp_dtype_str == 'bfloat16' else torch.float16
        
        # GradScaler只在float16时需要（bfloat16不需要缩放）
        use_scaler = enable_amp and amp_dtype_str == 'float16'
        scaler = GradScaler('cuda', enabled=use_scaler)
        
        if rank == 0 and enable_amp:
            print(f"AMP Configuration:")
            print(f"  - Mixed Precision: Enabled")
            print(f"  - AMP dtype: {amp_dtype_str}")
            print(f"  - GradScaler: {'Enabled' if use_scaler else 'Disabled (not needed for bfloat16)'}\n")
        
        # 创建输出目录（只在rank 0进程）
        if rank == 0:
            os.makedirs(output_path, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            out_dir = os.path.join(output_path, f"{timestamp}")
            os.makedirs(out_dir, exist_ok=True)
            checkpoint_path = os.path.join(out_dir, "checkpoint.pth")
            
            # 创建日志文件
            log_path = os.path.join(out_dir, "log.csv")
            with open(log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "wallclock_time", "train_loss", "val_loss", "lr"])
        else:
            out_dir = ""
            checkpoint_path = ""
            log_path = ""
        
        # 同步输出目录路径给所有进程
        out_dir_tensor = torch.zeros(256, dtype=torch.uint8, device=device)
        if rank == 0:
            out_dir_bytes = [ord(c) for c in out_dir[:255]]
            out_dir_tensor[:len(out_dir_bytes)] = torch.tensor(out_dir_bytes, dtype=torch.uint8, device=device)
        
        dist.broadcast(out_dir_tensor, src=0)
        
        if rank != 0:
            out_dir_chars = out_dir_tensor.cpu().tolist()
            out_dir = ''.join(chr(c) for c in out_dir_chars if c != 0)
            checkpoint_path = os.path.join(out_dir, "checkpoint.pth")
            log_path = os.path.join(out_dir, "log.csv")
        
        # 加载checkpoint（如果存在）
        start_iteration = 0
        checkpoint_exists = False
        
        if rank == 0:
            checkpoint_exists = os.path.exists(checkpoint_path)
        
        checkpoint_exists_tensor = torch.tensor([1 if checkpoint_exists else 0], dtype=torch.int, device=device)
        dist.broadcast(checkpoint_exists_tensor, src=0)
        checkpoint_exists = checkpoint_exists_tensor.item() == 1
        
        if checkpoint_exists:
            if rank == 0:
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                model.module.load_state_dict(checkpoint["model_state_dict"])
                muon_opt.load_state_dict(checkpoint["muon_optimizer_state_dict"])
                adamw_opt.load_state_dict(checkpoint["adamw_optimizer_state_dict"])
                start_iteration = checkpoint["iteration"]
                
                # 如果checkpoint中有scaler状态且当前使用scaler，则加载
                if use_scaler and "scaler_state_dict" in checkpoint:
                    scaler.load_state_dict(checkpoint["scaler_state_dict"])
                    print(f"Loaded checkpoint (with scaler) from iteration {start_iteration}\n")
                else:
                    print(f"Loaded checkpoint from iteration {start_iteration}\n")
            
            start_iteration_tensor = torch.tensor([start_iteration], dtype=torch.long, device=device)
            dist.broadcast(start_iteration_tensor, src=0)
            start_iteration = start_iteration_tensor.item()
            
            for param in model.parameters():
                dist.broadcast(param.data, src=0)
        else:
            if rank == 0:
                print("No checkpoint found, starting from scratch.\n")
        
        # 加载真实数据集
        train_dataset = np.memmap(
            train_dataset_path,
            dtype=np.uint16,
            mode="r",
        )
        valid_dataset = np.memmap(
            valid_dataset_path,
            dtype=np.uint16,
            mode="r",
        )
        total_iterations = iteration
        if rank == 0:
            print(f"Using dataset: {train_dataset_path}")
        
        # 将模型移动到指定设备和数据类型
        model.to(device=device, dtype=dtype)
        muon_opt.to(device=device, dtype=dtype)
        adamw_opt.to(device=device, dtype=dtype)
        
        # 同步所有进程
        dist.barrier()
        
        # 训练循环
        model.train()
        muon_opt.zero_grad()
        adamw_opt.zero_grad()
        
        if rank == 0:
            print(f"Starting training from iteration {start_iteration} to {total_iterations}...")
            print(f"Output directory: {out_dir}\n")
            pbar = tqdm.tqdm(total=total_iterations, initial=start_iteration)
        
        start_time = time.time()
        for iter_idx in range(start_iteration, total_iterations):
            # 获取数据
            input_batch, target_batch = get_batch(
                train_dataset,
                batch_size=batch_size,
                context_length=context_length,
                device=device,
            )
            
            input_batch = input_batch.to(dtype=torch.int)
            target_batch = target_batch.to(dtype=torch.int)
            
            # 计算学习率调度因子
            lr_factor = cosine_anneal_schedule(
                current_step=iter_idx,
                warmup_steps=int(warmup_ratio * iteration),
                cosine_anneal_steps=iteration,
                max_lr=1.0,
                min_lr=lr_decay_ratio,
            )
            
            # 更新学习率
            for param_group in muon_opt.param_groups:
                param_group["lr"] = muon_lr * lr_factor
            for param_group in adamw_opt.param_groups:
                param_group["lr"] = adamw_lr * lr_factor
            
            # 梯度累积优化
            is_sync_step = (iter_idx + 1) % accumulation_steps == 0
            sync_context = nullcontext() if is_sync_step else model.no_sync()
            
            with sync_context:
                # AMP autocast：自动选择精度（前向传播）
                with autocast('cuda', dtype=amp_dtype, enabled=enable_amp):
                    logits = model(input_batch)
                    loss = cross_entropy_loss(logits, target_batch)
                    loss_scaled = loss / accumulation_steps
                
                # 反向传播（使用scaler处理float16的梯度缩放）
                if use_scaler:
                    scaler.scale(loss_scaled).backward()
                else:
                    loss_scaled.backward()
            
            if is_sync_step:
                # 梯度裁剪（需要先unscale以获取真实梯度）
                if use_scaler:
                    scaler.unscale_(muon_opt)
                    scaler.unscale_(adamw_opt)
                
                gradient_clipping(model.parameters(), max_norm=max_grad_norm)
                
                # 优化器更新（使用scaler.step确保梯度有效）
                if use_scaler:
                    scaler.step(muon_opt)
                    scaler.step(adamw_opt)
                    scaler.update()
                else:
                    muon_opt.step()
                    adamw_opt.step()
                
                muon_opt.zero_grad()
                adamw_opt.zero_grad()
            
            # 进度条更新（只在rank 0）
            if rank == 0 and iter_idx % 10 == 0:
                pbar.set_description(
                    f"Loss: {loss.item():.4f}, LR: {lr_factor:.4f}"
                )
            
            # ==================== 并行验证 ====================
            # 所有进程都参与验证，提升验证速度 world_size 倍
            if (
                iter_idx == 0 or (iter_idx + 1) % valid_frequency == 0 or iter_idx == total_iterations - 1
            ):
                val_loss = evaluate_validation_loss(
                    model.module,
                    valid_dataset,
                    batch_size,
                    context_length,
                    device,
                    valid_batch_multiples,
                    enable_amp=enable_amp,
                    amp_dtype=amp_dtype,
                    rank=rank,
                    world_size=world_size,
                )
                
                # 只在rank 0进程记录日志和绘图
                if rank == 0:
                    save_log(
                        log_path,
                        step=iter_idx + 1,
                        wallclock_time=time.time() - start_time,
                        train_loss=loss.item(),
                        val_loss=val_loss,
                        lr=lr_factor,
                        rank=rank,
                    )
                    plot_logs(log_path, out_dir, rank)
            
            if rank == 0:
                # 保存checkpoint
                should_save_checkpoint = saving_interval > 0 and (
                    (iter_idx + 1) % saving_interval == 0 or iter_idx == total_iterations - 1
                )
                if should_save_checkpoint:
                    checkpoint_dict = {
                        "model_state_dict": model.module.state_dict(),
                        "muon_optimizer_state_dict": muon_opt.state_dict(),
                        "adamw_optimizer_state_dict": adamw_opt.state_dict(),
                        "iteration": iter_idx,
                    }
                    # 如果使用scaler，也保存scaler状态
                    if use_scaler:
                        checkpoint_dict["scaler_state_dict"] = scaler.state_dict()
                    
                    torch.save(checkpoint_dict, checkpoint_path)
                
                # 最后一次迭代时，单独保存模型文件
                if iter_idx == total_iterations - 1:
                    model_only_path = os.path.join(out_dir, "model.pth")
                    torch.save(model.module.state_dict(), model_only_path)
                    print(f"\nModel saved to {model_only_path}")
                
                pbar.update(1)
        
        if rank == 0:
            pbar.close()
            print("\n" + "=" * 80)
            print("Training Completed!")
            print("=" * 80 + "\n")
    
    except Exception as e:
        if rank == 0:
            print(f"\nError during training: {e}")
            import traceback
            traceback.print_exc()
        raise
    finally:
        cleanup_distributed()


def main():
    """主函数：启动分布式训练
    
    支持两种启动方式：
    1. torchrun 方式（推荐）: torchrun --nproc_per_node=N ddp.py [options]
    2. 直接运行方式: python ddp.py [options]
    
    示例：
      torchrun --nproc_per_node=2 ddp.py --enable_tf32
      torchrun --nproc_per_node=4 ddp.py --enable_tf32 --enable_compile --enable_amp --amp_dtype bfloat16
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="DDP Training with cs336 Transformer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  torchrun --nproc_per_node=2 ddp.py --enable_tf32
  torchrun --nproc_per_node=4 ddp.py --enable_tf32 --enable_compile --enable_amp
        """
    )
    
    # 仅保留性能优化选项
    parser.add_argument("--enable_tf32", action="store_true", default=enable_tf32,
                       help="启用TF32加速（Ampere+ GPU）")
    parser.add_argument("--enable_amp", action="store_true", default=enable_amp,
                       help="启用混合精度训练（AMP）")
    parser.add_argument("--amp_dtype", type=str, default=amp_dtype, 
                       choices=["float16", "bfloat16"],
                       help="AMP数据类型")
    parser.add_argument("--enable_compile", action="store_true", default=enable_compile,
                       help="启用torch.compile编译优化（PyTorch 2.0+）")
    parser.add_argument("--compile_mode", type=str, default="default",
                       choices=["default", "reduce-overhead", "max-autotune"],
                       help="torch.compile编译模式：default(平衡,快速编译), reduce-overhead(减少开销), max-autotune(最大优化,编译慢)")
    parser.add_argument("--ddp_mode", type=int, default=1,
                       choices=[0, 1, 2],
                       help="DDP通信优化模式：0(无优化), 1(只扁平化,默认), 2(标准分桶)")
    
    args = parser.parse_args()
    
    config = {
        'enable_tf32': args.enable_tf32,
        'enable_amp': args.enable_amp,
        'amp_dtype': args.amp_dtype,
        'enable_compile': args.enable_compile,
        'compile_mode': args.compile_mode,
        'ddp_mode': args.ddp_mode,
    }
    
    # 检查是否由 torchrun 启动（环境变量 RANK 存在）
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        ws = int(os.environ["WORLD_SIZE"])
        
        if rank == 0:
            print(f"Detected {ws} GPUs, starting distributed training (torchrun)...\n")
        
        train_worker(local_rank, ws, config)
    else:
        ws = torch.cuda.device_count()
        
        if ws < 2:
            print("Warning: fewer than 2 GPUs detected, using single GPU...\n")
            ws = 1
        
        print(f"Detected {ws} GPUs, starting distributed training (mp.spawn)...\n")
        
        if ws == 1:
            train_worker(0, 1, config)
        else:
            mp.spawn(
                train_worker,
                args=(ws, config),
                nprocs=ws,
                join=True
            )


if __name__ == "__main__":
    main()
