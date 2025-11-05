# 训练脚本
import numpy as np
import torch
import tqdm
import time
import os
from cs336_basics import (
    Transformer,
    AdamW,
    get_batch,
    cross_entropy_loss,
    cosine_anneal_schedule,
    gradient_clipping,
)

# 训练循环超参数
output_path = "./saves"
checkpoint_path = ""
dataset_path = "./data/id/ts-v-id/TinyStoriesV2-GPT4-train.bin"
iteration = 1000
batch_size = 32
saving_interval = 100
# 模型超参数
vocab_size = 5000
context_length = 128
d_model = 512
nhead = 8
num_layers = 6
d_ff = 2048
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32
# 余弦退火学习率参数
max_lr = 1e-4
min_lr = 1e-5
warmup_steps = 100
cosine_anneal_steps = 1000
# 梯度裁剪参数
max_grad_norm = 1.0


model = Transformer(
    vocab_size=vocab_size,
    context_length=context_length,
    d_model=d_model,
    nhead=nhead,
    num_layers=num_layers,
    d_ff=d_ff,
    device=device,
)

dataset = np.memmap(
    dataset_path,
    dtype=np.uint16,
    mode="r",
)

opt = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)


if __name__ == "__main__":
    # 加载checkpoint（如果存在）
    start_iteration = 0
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        opt.load_state_dict(checkpoint["optimizer_state_dict"])
        start_iteration = checkpoint["iteration"]
        print(f"Loaded checkpoint from iteration {start_iteration}")
    except FileNotFoundError:
        print("No checkpoint found, starting from scratch.")

    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    out_dir = os.path.join(output_path, f"{timestamp}")
    checkpoint_path = os.path.join(out_dir, "checkpoint.pth")

    model.to(device=device, dtype=dtype)
    model.train()   # 设置模型为训练模式
    opt.to(device=device, dtype=dtype)

    # 训练循环
    with tqdm.tqdm(total=iteration, initial=start_iteration) as pbar:
        for iter in range(start_iteration, iteration):
            input_batch, target_batch = get_batch(
                dataset,
                batch_size=batch_size,
                context_length=context_length,
                device=device,
            )

            current_lr = cosine_anneal_schedule(  # 计算当前学习率
                current_step=iter,
                warmup_steps=warmup_steps,
                cosine_anneal_steps=cosine_anneal_steps,
                max_lr=max_lr,
                min_lr=min_lr,
            )

            for param_group in opt.param_groups:  # 更新优化器中的学习率
                param_group["lr"] = current_lr

            logits = model(input_batch)  # 前向传播
            loss = cross_entropy_loss(logits, target_batch)  # 计算损失
            opt.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            gradient_clipping(model.parameters(), max_norm=max_grad_norm)  # 梯度裁剪
            opt.step()  # 优化器更新参数

            pbar.update(1)

            if iter % 10 == 0:  # 每10次迭代更新一次进度条，显示损失和学习率
                pbar.set_description(f"Iter {iter}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}")

            if (iter + 1) % saving_interval == 0 or iter == iteration - 1:  # 保存checkpoint
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "iteration": iter,
                    },
                    checkpoint_path,
                )
