# 分布式训练脚本（多GPU支持）
import numpy as np
from contextlib import nullcontext
import torch
# 设置 float32 矩阵乘法精度（A100/H100 等 Ampere+ 架构）
# 注意：此选项只影响矩阵乘法的计算精度，不影响数据存储类型
torch.set_float32_matmul_precision('high')
import torch.distributed as dist
import torch.multiprocessing as mp
import tqdm
import time
import os
import csv
import matplotlib.pyplot as plt
import pandas as pd
from cs336_basics import (
    Transformer,
    Muon,
    AdamW,
    get_batch,
    cross_entropy_loss,
    cosine_anneal_schedule,
    gradient_clipping,
)

# 训练循环超参数
output_path = "./out"
train_dataset_path = "../../cs336_data/id/owt-t-id/owt_train.bin"
valid_dataset_path = "../../cs336_data/id/owt-v-id/owt_valid.bin"
batch_size = 16  # 每个GPU的batch size
iteration = 100000 
saving_interval = -1
valid_frequency = 1000
valid_batch_multiples = 8
accumulation_steps = 4
# Early Stopping 超参数
early_stopping_patience = 15  # 连续 N 次验证无改善则停止
early_stopping_min_delta = 0.01  # 最小改善阈值
# 模型超参数
vocab_size = 32000
context_length = 1024
d_model = 768
nhead = 12
num_layers = 12
d_ff = 2048
rope_theta = 10000.0
logit_cap = 30.0  # Logit Softcapping 阈值，防止logits爆炸
dtype = torch.float32
# 学习率调度参数
warmup_ratio = 0.05           # 预热阶段占总步数的比例
lr_decay_ratio = 0.1          # 学习率从峰值衰减到最小值的比例 (min_lr = peak_lr * ratio)
cosine_anneal_steps = iteration
# 梯度裁剪参数
max_grad_norm = 1.0
# 优化器参数
# Muon 参数 (用于 2D 权重矩阵)
muon_lr = 0.02                # Muon 优化器的峰值学习率
muon_momentum = 0.95          # Muon 动量系数
muon_weight_decay = 0.0       # Muon 通常不需要权重衰减
ns_steps = 5                  # Newton-Schulz 迭代次数
# AdamW 参数 (用于 1D 参数、Embedding 等)
adamw_lr = 1e-3               # AdamW 优化器的峰值学习率
adamw_betas = (0.9, 0.95)
adamw_eps = 1e-8
adamw_weight_decay = 1e-2

# 分布式训练参数
world_size = torch.cuda.device_count()  # 可用的GPU数量
dist_url = "env://"  # 使用环境变量初始化


def setup_distributed(rank, world_size):
    """设置分布式训练环境"""
    # 设置CUDA设备
    torch.cuda.set_device(rank)
    
    # 初始化进程组
    dist.init_process_group(
        backend="nccl",  # 使用NCCL后端进行GPU通信
        init_method=dist_url,
        world_size=world_size,
        rank=rank
    )
    
    # 设置随机种子以确保不同进程使用不同的数据
    torch.manual_seed(42 + rank)
    

def cleanup_distributed():
    """清理分布式训练环境"""
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
            input_batch = input_batch.to(
                dtype=torch.int
            )  # 转换为整型，以匹配嵌入层要求
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
    plt.plot(df["step"], df["train_loss"], label="Train Loss")
    plt.plot(df["step"], df["val_loss"], label="Validation Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Iterations")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()


def train_worker(rank, world_size):
    """每个训练进程的入口函数"""
    # 设置分布式环境
    setup_distributed(rank, world_size)
    
    # 创建本地设备
    device = f"cuda:{rank}"
    
    # 创建模型
    model = Transformer(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        d_ff=d_ff,
        logit_cap=logit_cap,
        device=device,
        tie_weights=False,
    )
    
    # 使用DistributedDataParallel包装模型
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[rank],
        output_device=rank
    )
    
    # 使用 torch.compile 优化模型
    # mode="reduce-overhead" 适合小batch、频繁调用场景
    # mode="max-autotune" 会花更多编译时间但运行更快
    model = torch.compile(model, mode="default")
    if rank == 0:
        print("torch.compile enabled")
    
    # 分离参数：2D 权重矩阵用 Muon，其他参数用 AdamW
    # 关键：Norm 层的仿射参数不应施加 Weight Decay，否则会破坏网络等变性映射
    muon_params = []  # 2D 权重矩阵
    adamw_decay_params = []  # 需要 weight decay 的参数（如 embedding）
    adamw_nodecay_params = []  # 不需要 weight decay 的参数（bias、norm 层参数）
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.ndim == 2 and 'embedding' not in name.lower():
                # 2D 权重矩阵（排除 embedding）用 Muon
                muon_params.append(param)
            elif 'embedding' in name.lower():
                # Embedding 参数需要 weight decay
                adamw_decay_params.append(param)
            else:
                # 1D 参数（bias）和 norm 层参数不需要 weight decay
                # 这包括 LayerNorm/RMSNorm 的 weight(gamma) 和 bias(beta)
                adamw_nodecay_params.append(param)
    
    if rank == 0:
        print(f"Muon params: {len(muon_params)} tensors")
        print(f"AdamW params (with decay): {len(adamw_decay_params)} tensors")
        print(f"AdamW params (no decay): {len(adamw_nodecay_params)} tensors")
    
    # 创建混合优化器
    muon_opt = Muon(muon_params, lr=muon_lr, momentum=muon_momentum, weight_decay=muon_weight_decay, ns_steps=ns_steps)
    # AdamW 使用参数组：embedding 有 weight decay，其他 1D 参数（bias、norm）无 weight decay
    adamw_opt = AdamW([
        {'params': adamw_decay_params, 'weight_decay': adamw_weight_decay},
        {'params': adamw_nodecay_params, 'weight_decay': 0.0}
    ], lr=adamw_lr, betas=adamw_betas, eps=adamw_eps)
    
    # 创建输出目录（只在rank 0进程）
    if rank == 0:
        os.makedirs(output_path, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        out_dir = os.path.join(output_path, f"{timestamp}")
        os.makedirs(out_dir, exist_ok=True)
        checkpoint_path = os.path.join(out_dir, "checkpoint.pth")
        
        # 创建日志文件，写入表头
        log_path = os.path.join(out_dir, "log.csv")
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "wallclock_time", "train_loss", "val_loss", "lr"])
    else:
        out_dir = ""
        checkpoint_path = ""
        log_path = ""
    
    # 同步输出目录路径给所有进程（固定大小256字节）
    out_dir_tensor = torch.zeros(256, dtype=torch.uint8, device=device)
    if rank == 0:
        # 将out_dir编码为固定长度tensor
        out_dir_bytes = [ord(c) for c in out_dir[:255]]  # 最多255个字符
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
    
    # 首先检查 checkpoint 是否存在（只在 rank 0 检查并广播结果）
    if rank == 0:
        checkpoint_exists = os.path.exists(checkpoint_path)
    
    # 广播 checkpoint 是否存在的信息给所有进程
    checkpoint_exists_tensor = torch.tensor([1 if checkpoint_exists else 0], dtype=torch.int, device=device)
    dist.broadcast(checkpoint_exists_tensor, src=0)
    checkpoint_exists = checkpoint_exists_tensor.item() == 1
    
    if checkpoint_exists:
        # 只在rank 0进程加载checkpoint，然后广播给其他进程
        if rank == 0:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.module.load_state_dict(checkpoint["model_state_dict"])
            muon_opt.load_state_dict(checkpoint["muon_optimizer_state_dict"])
            adamw_opt.load_state_dict(checkpoint["adamw_optimizer_state_dict"])
            start_iteration = checkpoint["iteration"]
            print(f"Rank {rank}: Loaded checkpoint from iteration {start_iteration}")
        
        # 将checkpoint信息广播给所有进程
        start_iteration_tensor = torch.tensor([start_iteration], dtype=torch.long, device=device)
        dist.broadcast(start_iteration_tensor, src=0)
        start_iteration = start_iteration_tensor.item()
        
        # 同步模型状态给所有进程
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
    else:
        if rank == 0:
            print("No checkpoint found, starting from scratch.")
    
    # 加载数据集（所有进程都加载完整数据集）
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
    
    # 将模型移动到指定设备和数据类型
    model.to(device=device, dtype=dtype)
    muon_opt.to(device=device, dtype=dtype)
    adamw_opt.to(device=device, dtype=dtype)
    
    # 同步所有进程（确保所有进程都准备好了）
    dist.barrier()
    
    # 训练循环
    model.train()  # 设置模型为训练模式
    muon_opt.zero_grad()  # 清空优化器梯度
    adamw_opt.zero_grad()
    
    if rank == 0:
        print(f"Rank {rank}: Starting distributed training on {world_size} GPUs")
        print(f"Rank {rank}: Model: {model.module}")
        print(f"Rank {rank}: Using device: {device}")
        print(f"Rank {rank}: Model parameters: {model.module.compute_params()}")
        print(f"Rank {rank}: Output directory: {out_dir}")
        pbar = tqdm.tqdm(total=iteration, initial=start_iteration)
    
    # Early Stopping 状态变量
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_flag = False
    
    start_time = time.time()
    for iter in range(start_iteration, iteration):
        # 每个进程处理不同的数据批次（通过rank进行数据分割）
        # 注意：这里简化了数据分割，实际应用可以使用DistributedSampler
        input_batch, target_batch = get_batch(
            train_dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )
        input_batch = input_batch.to(
            dtype=torch.int
        )  # 转换为整型，以匹配嵌入层要求
        target_batch = target_batch.to(dtype=torch.int)
        
        # 计算学习率调度因子 (范围: [lr_decay_ratio, 1.0])
        # 使用归一化参数，让 cosine_anneal_schedule 返回一个比例因子
        lr_factor = cosine_anneal_schedule(
            current_step=iter,
            warmup_steps=int(warmup_ratio * cosine_anneal_steps),
            cosine_anneal_steps=cosine_anneal_steps,
            max_lr=1.0,              # 归一化最大值
            min_lr=lr_decay_ratio,   # 衰减到峰值的 lr_decay_ratio 倍
        )
        
        # 更新各优化器的学习率：峰值学习率 × 调度因子
        for param_group in muon_opt.param_groups:
            param_group["lr"] = muon_lr * lr_factor
        for param_group in adamw_opt.param_groups:
            param_group["lr"] = adamw_lr * lr_factor
        
        # 梯度累积优化：只在需要更新参数时同步梯度，避免中间步骤的无效通信
        is_sync_step = (iter + 1) % accumulation_steps == 0
        sync_context = nullcontext() if is_sync_step else model.no_sync()
        
        with sync_context:
            logits = model(input_batch)  # 前向传播（自动处理分布式）
            loss = cross_entropy_loss(logits, target_batch) # 计算损失
            loss_scaled = loss / accumulation_steps
            loss_scaled.backward()  # 反向传播（非同步步骤不触发AllReduce）
        
        if is_sync_step:
            gradient_clipping(model.parameters(), max_norm=max_grad_norm)  # 梯度裁剪
            muon_opt.step()  # Muon 更新 2D 权重矩阵
            adamw_opt.step()  # AdamW 更新其他参数
            muon_opt.zero_grad()
            adamw_opt.zero_grad()
        
        # 只在rank 0进程显示进度和保存日志
        if rank == 0:
            if iter % 10 == 0:  # 每10次迭代更新一次进度条，显示损失和学习率
                pbar.set_description(
                    f"Iter {iter}, Loss: {loss.item():.4f}, LR factor: {lr_factor:.4f}"
                )
            
            if (
                iter == 0 or (iter + 1) % valid_frequency == 0 or iter == iteration - 1
            ):  # 分别在第一次迭代、每valid_frequency次迭代和最后一次迭代时评估验证损失，记录日志和保存损失曲线图
                # 使用rank 0进程进行验证（避免重复验证）
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
                    # 保存最佳模型
                    best_model_path = os.path.join(out_dir, "best_model.pth")
                    torch.save(model.module.state_dict(), best_model_path)
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping triggered at iteration {iter + 1}")
                        print(f"Best validation loss: {best_val_loss:.4f}")
                        early_stop_flag = True
                
                # 记录日志
                save_log(
                    log_path,
                    step=iter + 1,
                    wallclock_time=time.time() - start_time,
                    train_loss=loss.item(),
                    val_loss=val_loss,
                    lr=lr_factor,  # 记录调度因子，实际 lr = peak_lr * lr_factor
                    rank=rank,
                )
                
                # 保存损失曲线图
                plot_logs(log_path, out_dir, rank)
        
        # 检查是否需要 early stop（广播给所有进程）
        early_stop_tensor = torch.tensor([1 if early_stop_flag else 0], dtype=torch.int, device=device)
        dist.broadcast(early_stop_tensor, src=0)
        if early_stop_tensor.item() == 1:
            if rank == 0:
                print("All processes stopping due to early stopping.")
            break
        
        if rank == 0:
            # 保存checkpoint（saving_interval为-1时不保存checkpoint）
            should_save_checkpoint = saving_interval > 0 and (
                (iter + 1) % saving_interval == 0 or iter == iteration - 1
            )
            if should_save_checkpoint:
                # 只在rank 0进程保存checkpoint
                torch.save(
                    {
                        "model_state_dict": model.module.state_dict(),  # 保存原始模型状态
                        "muon_optimizer_state_dict": muon_opt.state_dict(),
                        "adamw_optimizer_state_dict": adamw_opt.state_dict(),
                        "iteration": iter,
                    },
                    checkpoint_path,
                )
            
            # 最后一次迭代时，单独保存模型文件（不含优化器状态）
            if iter == iteration - 1:
                model_only_path = os.path.join(out_dir, "model.pth")
                torch.save(model.module.state_dict(), model_only_path)
                print(f"Model saved to {model_only_path}")
            
            pbar.update(1)
    
    # 如果是 early stopping 结束，在 rank 0 打印最终信息
    if rank == 0 and early_stop_flag:
        print(f"Training ended via early stopping. Best val_loss: {best_val_loss:.4f}")
    
    if rank == 0:
        pbar.close()
    
    # 清理分布式环境
    cleanup_distributed()


def main():
    """主函数：启动分布式训练
    
    支持两种启动方式：
    1. torchrun 方式（推荐）: torchrun --nproc_per_node=N train_distributed.py
       - 通过环境变量 RANK, LOCAL_RANK, WORLD_SIZE 获取进程信息
    2. 直接运行方式: python train_distributed.py
       - 使用 mp.spawn() 启动多进程
    """
    # 检查是否由 torchrun 启动（环境变量 RANK 存在）
    if "RANK" in os.environ:
        # torchrun 方式：直接从环境变量获取 rank 和 world_size
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        ws = int(os.environ["WORLD_SIZE"])
        
        if rank == 0:
            print(f"检测到 {ws} 个GPU，启动分布式训练（torchrun 方式）...")
        
        # 直接调用 train_worker，使用 local_rank 作为 GPU 索引
        train_worker(local_rank, ws)
    else:
        # 直接运行方式：使用 mp.spawn
        if world_size < 2:
            print("警告：检测到少于2个GPU，建议使用单GPU训练脚本")
            print("如果想要多GPU训练，请确保至少有2个可用GPU")
            return
        
        print(f"检测到 {world_size} 个GPU，启动分布式训练（mp.spawn 方式）...")
        
        # 使用spawn方式启动多进程训练
        mp.spawn(
            train_worker,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )


if __name__ == "__main__":
    main()