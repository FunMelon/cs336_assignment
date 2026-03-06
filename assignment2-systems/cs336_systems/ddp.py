import os
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from contextlib import nullcontext, contextmanager
import time
import csv
import matplotlib.pyplot as plt
import pandas as pd
import tqdm

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW, get_cosine_lr
from cs336_basics.data import get_batch
from cs336_basics.nn_utils import cross_entropy, clip_gradient


# ==================== 训练配置 ====================
# 输出和数据集路径
output_path = "./out"
train_dataset_path = "../../../cs336_data/id/owt-t-id/owt_train.bin"
valid_dataset_path = "../../../cs336_data/id/owt-v-id/owt_valid.bin"

# 模型超参数
vocab_size = 32000
context_length = 1024
d_model = 768           # 768/12 = 64
nhead = 12
num_layers = 12
d_ff = 2048
rope_theta = 10000.0
# logit_cap 和 use_flash_attention 在 BasicsTransformerLM 中不支持，忽略
dtype = torch.float32

# 训练超参数
batch_size = 48         # 每个GPU的batch size
iteration = 50000       # 总训练步数
saving_interval = -1    # Checkpoint保存间隔
valid_frequency = 1000  # 验证频率
valid_batch_multiples = 8  # 验证时使用的batch数量
accumulation_steps = 2  # 梯度累积步数

# Early Stopping 超参数
early_stopping_patience = 15
early_stopping_min_delta = 0.01

# 学习率调度参数
warmup_ratio = 0.05
lr_decay_ratio = 0.1

# 梯度裁剪参数
max_grad_norm = 1.0

# 优化器参数
adamw_lr = 1e-3
adamw_betas = (0.9, 0.95)
adamw_eps = 1e-8
adamw_weight_decay = 1e-2

# 分布式训练参数
world_size = torch.cuda.device_count()
dist_url = "env://"


class DDPIndividualParameters(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.handles = []
        self.require_backward_grad_sync = True
        
        # Broadcast initial weights from rank 0 to all other processes
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
        
        # Broadcast buffers (e.g. running_mean/var in BatchNorm, though not used in current Transformer)
        for buffer in self.module.buffers():
            dist.broadcast(buffer.data, src=0)
            
        # Register hooks for async communication
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._make_hook())
                
    def _make_hook(self):
        def hook(param):
            if param.grad is None or not self.require_backward_grad_sync:
                return
            
            with torch.no_grad():
                # Divide by world_size BEFORE all_reduce
                # This implements (g1/N + g2/N + ... + gN/N) which is equal to sum(g)/N
                # Using in-place division to ensure memory safety
                param.grad.div_(dist.get_world_size())
                
                # Asynchronously all-reduce the gradient
                handle = dist.all_reduce(param.grad, async_op=True)
                self.handles.append(handle)
        return hook

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    @contextmanager
    def no_sync(self):
        old_val = self.require_backward_grad_sync
        self.require_backward_grad_sync = False
        try:
            yield
        finally:
            self.require_backward_grad_sync = old_val

    def finish_gradient_synchronization(self):
        # Wait for all async communication to finish
        for handle in self.handles:
            handle.wait()
        self.handles.clear()
        # Ensure all CUDA operations (including Gloo reductions) are completed
        torch.cuda.synchronize()
        # Ensure GPU operations are fully finished before optimizer step
        torch.cuda.synchronize()


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
            val_loss = cross_entropy(logits, target_batch)
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
        
        # 创建模型 - 使用 BasicsTransformerLM
        model = BasicsTransformerLM(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=nhead,
            d_ff=d_ff,
            rope_theta=rope_theta,
        )
        
        # 使用 DDPIndividualParameters 包装模型
        model = DDPIndividualParameters(model)
        
        if rank == 0:
            print("\n" + "=" * 80)
            print("DDP Training with cs336 BasicsTransformerLM")
            print("=" * 80)
            num_params = sum(p.numel() for p in model.parameters())
            print(f"Model: {num_params:,} parameters ({num_params/1e6:.2f}M)")
            print(f"GPUs: {world_size} | Batch/GPU: {batch_size} | Global Batch: {batch_size * world_size}")
            print("=" * 80 + "\n")
        
        # 优化器 - 使用 AdamW
        # 分离权重衰减参数
        decay_params = []
        nodecay_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim == 1:
                nodecay_params.append(param)
            else:
                decay_params.append(param)
                
        optimizer = AdamW(
            [
                {'params': decay_params, 'weight_decay': adamw_weight_decay},
                {'params': nodecay_params, 'weight_decay': 0.0},
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
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.module.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
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
        # 优化器已经在CPU上初始化，PyTorch优化器通常会自动处理设备，或者我们需要手动移动状态
        # 这里为了简单，假设optimizer step时会处理，或者我们不移动它因为AdamW实现可能依赖param.grad所在的设备
        
        # 同步所有进程
        dist.barrier()
        
        # 训练循环
        model.train()
        # AdamW from cs336_basics might not have zero_grad, checking... 
        # checked: it inherits from torch.optim.Optimizer, so it has zero_grad
        optimizer.zero_grad()
        
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
            
            # 计算学习率
            lr = get_cosine_lr(
                it=iter_idx,
                max_learning_rate=adamw_lr,
                min_learning_rate=adamw_lr * lr_decay_ratio,
                warmup_iters=int(warmup_ratio * iteration),
                cosine_cycle_iters=iteration,
            )
            
            # 更新学习率
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            
            # 梯度累积优化
            is_sync_step = (iter_idx + 1) % accumulation_steps == 0
            sync_context = nullcontext() if is_sync_step else model.no_sync()
            
            with sync_context:
                logits = model(input_batch)
                loss = cross_entropy(logits, target_batch)
                loss_scaled = loss / accumulation_steps
                loss_scaled.backward()
            
            if is_sync_step:
                model.finish_gradient_synchronization()
                clip_gradient(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
            
            # 只在rank 0进程显示进度和保存日志
            if rank == 0:
                if iter_idx % 10 == 0:
                    pbar.set_description(
                        f"Loss: {loss.item():.4f}, LR: {lr:.4f}"
                    )
                
                # 验证和日志记录
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
                    )
                    
                    # Early Stopping 检查
                    if val_loss < best_val_loss - early_stopping_min_delta:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_model_path = os.path.join(out_dir, "best_model.pth")
                        torch.save(model.module.state_dict(), best_model_path)
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
                        lr=lr,
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
                    torch.save(
                        {
                            "model_state_dict": model.module.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "iteration": iter_idx,
                        },
                        checkpoint_path,
                    )
                
                # 最后一次迭代时，单独保存模型文件
                if iter_idx == total_iterations - 1:
                    model_only_path = os.path.join(out_dir, "model.pth")
                    torch.save(model.module.state_dict(), model_only_path)
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
    import argparse
    
    parser = argparse.ArgumentParser(
        description="DDP Training with cs336 BasicsTransformerLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    args = parser.parse_args()
    
    config = {}
    
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
