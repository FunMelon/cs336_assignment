import os
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from contextlib import nullcontext
import time
import csv
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
import argparse

# 导入自定义的DDP和优化器实现
from cs336_systems.ddp import DDPIndividualParameters, DDPBucketed
from cs336_systems.optimizer import ShardedOptimizer

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
batch_size = 48         # 每个GPU的batch size
iteration = 50000       # 总训练步数
saving_interval = -1    # Checkpoint保存间隔（-1=不保存中间checkpoint）
valid_frequency = 1000  # 验证频率
valid_batch_multiples = 8  # 验证时使用的batch数量
accumulation_steps = 2  # 梯度累积步数

# Early Stopping 超参数
early_stopping_patience = 15  # 连续N次验证无改善则停止
early_stopping_min_delta = 0.01  # 最小改善阈值

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

# 默认性能优化参数
default_enable_tf32 = False     # TF32优化（Ampere+ GPU）
default_enable_amp = False      # 混合精度训练（AMP）
default_amp_dtype = 'float16'   # AMP数据类型
default_enable_compile = False  # torch.compile优化

# 默认分布式策略参数
default_ddp_strategy = 'naive' # naive, individual, bucketed
default_bucket_size_mb = 25.0
default_sharded_optim = False

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
) -> float:
    """评估验证集上的平均损失"""
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

            logits = model(input_batch)
            val_loss = cross_entropy_loss(logits, target_batch)
            losses.append(val_loss.item())
    model.train()
    return sum(losses) / len(losses)


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
        
        # DDP 策略选择
        ddp_strategy = config.get('ddp_strategy', 'naive')
        bucket_size_mb = config.get('bucket_size_mb', 25.0)
        
        if rank == 0:
            print("\n" + "=" * 80)
            print("DDP Training with cs336 Transformer")
            print("=" * 80)
            num_params = sum(p.numel() for p in model.parameters())
            print(f"Model: {num_params:,} parameters ({num_params/1e6:.2f}M)")
            if use_flash_attention:
                print("FlashAttention enabled (using Triton kernels)")
            print(f"GPUs: {world_size} | Batch/GPU: {batch_size} | Global Batch: {batch_size * world_size}")
            print(f"DDP Strategy: {ddp_strategy}")
            if ddp_strategy == 'bucketed':
                print(f"Bucket Size: {bucket_size_mb} MB")
            print(f"Optimizer Sharding: {config.get('sharded_optim', False)}")
            print("=" * 80 + "\n")

        # 封装DDP
        if ddp_strategy == 'individual':
            model = DDPIndividualParameters(model)
        elif ddp_strategy == 'bucketed':
            model = DDPBucketed(model, bucket_size_mb=bucket_size_mb)
        else: # naive or pytorch standard
            # 使用 PyTorch 原生 DDP 作为 naive/baseline (或者可以用我们自己写的 naive 实现，
            # 但为了作为可靠 baseline，原生 DDP 更好。如果一定要用自己实现的 naive_ddp.py 里的逻辑，
            # 需要把那个手动同步逻辑搬过来。这里为了方便，我们假设 naive 就是指 PyTorch DDP)
            # 不过根据之前的任务，naive_ddp 确实是手动同步的。
            # 这里我们使用 PyTorch 原生 DDP 作为标准对照。
             model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[rank],
                output_device=rank
            )
        
        # 使用torch.compile优化（如果启用）
        if config.get('enable_compile', False):
            model = torch.compile(model, mode="default")
            if rank == 0:
                print("torch.compile enabled")
        
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
        
        # 优化器选择
        use_sharded_optim = config.get('sharded_optim', False)
        
        def create_optimizer(optim_cls, params, **kwargs):
            if use_sharded_optim:
                return ShardedOptimizer(params, optim_cls, **kwargs)
            else:
                return optim_cls(params, **kwargs)

        # 创建混合优化器
        # 注意：ShardedOptimizer 期望 params 是 list 或 param_groups
        # Muon 和 AdamW 的构造函数参数略有不同，需要适配
        
        if use_sharded_optim:
            muon_opt = ShardedOptimizer(
                muon_params,
                Muon,
                lr=muon_lr,
                momentum=muon_momentum,
                weight_decay=muon_weight_decay,
                ns_steps=ns_steps
            )
            
            adamw_opt = ShardedOptimizer(
                [
                    {'params': adamw_decay_params, 'weight_decay': adamw_weight_decay},
                    {'params': adamw_nodecay_params, 'weight_decay': 0.0},
                ],
                AdamW,
                lr=adamw_lr,
                betas=adamw_betas,
                eps=adamw_eps
            )
        else:
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
            print(f"Using real dataset: {train_dataset_path}")
        
        # 将模型移动到指定设备和数据类型
        model.to(device=device, dtype=dtype)
        # 优化器已经在CPU上初始化，PyTorch优化器通常会自动处理设备
        
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
        
        # Early Stopping 状态
        best_val_loss = float('inf')
        patience_counter = 0
        early_stop_flag = False
        
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
            
            # 根据DDP策略选择同步上下文
            # PyTorch DDP 和 Bucketed/Individual DDP 都有 no_sync() 接口
            if hasattr(model, 'no_sync'):
                sync_context = nullcontext() if is_sync_step else model.no_sync()
            else:
                sync_context = nullcontext()

            with sync_context:
                logits = model(input_batch)
                loss = cross_entropy_loss(logits, target_batch)
                loss_scaled = loss / accumulation_steps
                loss_scaled.backward()
            
            if is_sync_step:
                # 只有自定义 DDP 需要显式调用 finish_gradient_synchronization
                # PyTorch DDP 在 backward 时已自动处理
                if hasattr(model, 'finish_gradient_synchronization'):
                     model.finish_gradient_synchronization()
                
                gradient_clipping(model.parameters(), max_norm=max_grad_norm)
                
                muon_opt.step()
                adamw_opt.step()
                
                muon_opt.zero_grad()
                adamw_opt.zero_grad()
            
            # 只在rank 0进程显示进度和保存日志
            if rank == 0:
                if iter_idx % 10 == 0:
                    pbar.set_description(
                        f"Loss: {loss.item():.4f}, LR: {lr_factor:.4f}"
                    )
                
                # 验证和日志记录
                if (
                    iter_idx == 0 or (iter_idx + 1) % valid_frequency == 0 or iter_idx == total_iterations - 1
                ):
                    val_loss = evaluate_validation_loss(
                        model.module if hasattr(model, 'module') else model,
                        valid_dataset,
                        batch_size,
                        context_length,
                        device,
                        valid_batch_multiples,
                    )
                    
                    # Early Stopping 检查
                    if val_loss < best_val_loss - early_stopping_min_delta:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_model_path = os.path.join(out_dir, "best_model.pth")
                        # 保存模型时，如果用了 DDP 包装，要取 model.module
                        save_model = model.module if hasattr(model, 'module') else model
                        torch.save(save_model.state_dict(), best_model_path)
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            print(f"\nEarly stopping triggered at iteration {iter_idx + 1}")
                            print(f"  Best validation loss: {best_val_loss:.4f}")
                            early_stop_flag = True
                    
                    # 记录日志
                    save_log(
                        log_path,
                        step=iter_idx + 1,
                        wallclock_time=time.time() - start_time,
                        train_loss=loss.item(),
                        val_loss=val_loss,
                        lr=lr_factor,
                        rank=rank,
                    )
                    
                    # 保存损失曲线图
                    plot_logs(log_path, out_dir, rank)
            
            # 检查是否需要 early stop
            early_stop_tensor = torch.tensor([1 if early_stop_flag else 0], dtype=torch.int, device=device)
            dist.broadcast(early_stop_tensor, src=0)
            if early_stop_tensor.item() == 1:
                if rank == 0:
                    print("All processes stopping due to early stopping.")
                break
            
            if rank == 0:
                # 保存checkpoint
                should_save_checkpoint = saving_interval > 0 and (
                    (iter_idx + 1) % saving_interval == 0 or iter_idx == total_iterations - 1
                )
                if should_save_checkpoint:
                    save_model = model.module if hasattr(model, 'module') else model
                    torch.save(
                        {
                            "model_state_dict": save_model.state_dict(),
                            "muon_optimizer_state_dict": muon_opt.state_dict(),
                            "adamw_optimizer_state_dict": adamw_opt.state_dict(),
                            "iteration": iter_idx,
                        },
                        checkpoint_path,
                    )
                
                # 最后一次迭代时，单独保存模型文件
                if iter_idx == total_iterations - 1:
                    model_only_path = os.path.join(out_dir, "model.pth")
                    save_model = model.module if hasattr(model, 'module') else model
                    torch.save(save_model.state_dict(), model_only_path)
                    print(f"\nModel saved to {model_only_path}")
                
                pbar.update(1)
        
        if rank == 0:
            if early_stop_flag:
                print(f"\nTraining ended via early stopping. Best val_loss: {best_val_loss:.4f}")
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
    """主函数：启动分布式训练"""
    
    parser = argparse.ArgumentParser(
        description="DDP Training with cs336 Transformer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # 性能优化选项
    parser.add_argument("--enable_tf32", action="store_true", default=default_enable_tf32,
                       help="启用TF32加速（Ampere+ GPU）")
    parser.add_argument("--enable_amp", action="store_true", default=default_enable_amp,
                       help="启用混合精度训练（AMP）")
    parser.add_argument("--amp_dtype", type=str, default=default_amp_dtype, 
                       choices=["float16", "bfloat16"],
                       help="AMP数据类型")
    parser.add_argument("--enable_compile", action="store_true", default=default_enable_compile,
                       help="启用torch.compile编译优化（PyTorch 2.0+）")
    
    # 分布式策略选项
    parser.add_argument("--ddp_strategy", type=str, default=default_ddp_strategy,
                        choices=['naive', 'individual', 'bucketed'],
                        help="DDP策略: naive(PyTorch DDP), individual(单参数通信), bucketed(分桶通信)")
    parser.add_argument("--bucket_size_mb", type=float, default=default_bucket_size_mb,
                        help="分桶策略下的桶大小 (MB)")
    parser.add_argument("--sharded_optim", action="store_true", default=default_sharded_optim,
                        help="启用优化器状态分片 (Optimizer State Sharding)")
    
    args = parser.parse_args()
    
    config = {
        'enable_tf32': args.enable_tf32,
        'enable_amp': args.enable_amp,
        'amp_dtype': args.amp_dtype,
        'enable_compile': args.enable_compile,
        'ddp_strategy': args.ddp_strategy,
        'bucket_size_mb': args.bucket_size_mb,
        'sharded_optim': args.sharded_optim,
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
