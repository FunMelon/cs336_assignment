# 学习率测试脚本
import numpy as np
import torch
import tqdm
import time
import os
import csv
import matplotlib.pyplot as plt
import pandas as pd
from cs336_basics import (
    Transformer,
    AdamW,
    get_batch,
    cross_entropy_loss,
    gradient_clipping,
)

# 训练循环超参数
output_path = "./out"
train_dataset_path = "./data/id/ts-t-id/TinyStoriesV2-GPT4-train.bin"
valid_dataset_path = "./data/id/ts-v-id/TinyStoriesV2-GPT4-valid.bin"
batch_size = 64
saving_interval = 100
valid_frequency = 100
valid_batch_multiples = 5
# 模型超参数
vocab_size = 10000
context_length = 256
d_model = 512
nhead = 16
num_layers = 4
d_ff = 1344
rope_theta = 10000.0
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32
# 学习率范围测试参数
iteration = 1000
lr_start = 1e-7
lr_end = 1e-1
# 梯度裁剪参数
max_grad_norm = 1.0
# 优化器参数
lr = 1e-3
betas = (0.9, 0.999)
eps = 1e-8
weight_decay = 1e-2

model = Transformer(
    vocab_size=vocab_size,
    context_length=context_length,
    d_model=d_model,
    nhead=nhead,
    num_layers=num_layers,
    d_ff=d_ff,
    device=device,
)

opt = AdamW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
# 创建输出目录
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
) -> None:
    """保存日志到CSV文件"""
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


def plot_logs(log_path: str, output_dir: str) -> None:
    """绘制训练和验证损失曲线"""
    df = pd.read_csv(log_path)
    plt.figure()
    plt.plot(df["step"], df["train_loss"], label="Train Loss")
    plt.plot(df["step"], df["val_loss"], label="Validation Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Iterations")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()


if __name__ == "__main__":
    print("Starting LR range test...")
    print(f"Model: {model}")
    print(f"Using device: {device}")
    print(f"Model parameters: {model.compute_params()}")
    print(f"Output directory: {out_dir}")
    start_iteration = 0
    # 加载数据集
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

    # 将模型和优化器移动到指定设备和数据类型
    model.to(device=device, dtype=dtype)
    opt.to(device=device, dtype=dtype)

    # 训练循环
    model.train()  # 设置模型为训练模式
    with tqdm.tqdm(total=iteration, initial=start_iteration) as pbar:
        start_time = time.time()
        for iter in range(iteration):
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

            current_lr = lr_start * (lr_end / lr_start) ** (iter / iteration)   # 指数增长学习率 

            for param_group in opt.param_groups:  # 更新优化器中的学习率
                param_group["lr"] = current_lr

            logits = model(input_batch)  # 前向传播
            loss = cross_entropy_loss(logits, target_batch)  # 计算损失
            opt.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            gradient_clipping(model.parameters(), max_norm=max_grad_norm)  # 梯度裁剪
            opt.step()  # 优化器更新参数

            if iter % 10 == 0:  # 每10次迭代更新一次进度条，显示损失和学习率
                pbar.set_description(
                    f"Iter {iter}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}"
                )

            val_loss = evaluate_validation_loss(
                model,
                valid_dataset,
                batch_size,
                context_length,
                device,
                valid_batch_multiples,
            )

            # 记录日志
            save_log(
                log_path,
                step=iter + 1,
                wallclock_time=time.time() - start_time,
                train_loss=loss.item(),
                val_loss=val_loss,
                lr=current_lr,
            )

            # 保存损失曲线图
            plot_logs(log_path, out_dir)
            val_loss = None  # 重置验证损失以节省内存

            pbar.update(1)
