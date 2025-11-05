# 工具函数
import torch
import math
from collections.abc import Iterable
import numpy.typing as npt
import os
import typing


def SiLU(x: torch.Tensor) -> torch.Tensor:
    """
    SiLU激活函数的实现。
    args:
        x (torch.Tensor): 输入张量。
    returns:
        torch.Tensor: 应用SiLU激活函数后的张量。
    """
    return x * torch.sigmoid(x)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    继承自 torch.nn.Module 的自实现的Softmax模块
    args:
        x (torch.Tensor): 输入张量。
        dim (int): 归一化的维度。
    returns:
        torch.Tensor: 输出张量，应用Softmax后的结果。
    """
    max_val = x.max(dim=dim, keepdim=True).values  # 保留维度以便广播
    exp_x = torch.exp(x - max_val)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    计算缩放点积注意力。
    args:
        query (torch.Tensor): 查询张量，形状为 (..., seq_len_q, d_k)。
        key (torch.Tensor): 键张量，形状为 (..., seq_len_k, d_k)。
        value (torch.Tensor): 值张量，形状为 (..., seq_len_v, d_v)。
        mask (torch.Tensor | None): 可选的掩码张量，形状为 (..., seq_len_q, seq_len_k)。
    returns:
        torch.Tensor: 注意力输出张量，形状为 (..., seq_len_q, d_v)。
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
        torch.tensor(d_k, dtype=query.dtype, device=query.device)
    )

    if mask is not None:
        scores = scores.masked_fill(mask == False, float("-inf"))  # 使用掩码

    attn_weights = softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, value)
    return output


def cross_entropy_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    计算交叉熵损失。
    args:
        predictions (torch.Tensor): 预测概率分布，形状为 (batch_size, num_classes)。
        targets (torch.Tensor): 真实标签，形状为 (batch_size,)。
    returns:
        torch.Tensor: 交叉熵损失值。
    """
    x_max = predictions.max(dim=1, keepdim=True).values
    log_sum_exp = (
        torch.log(torch.sum(torch.exp(predictions - x_max), dim=1, keepdim=True))
        + x_max
    )  # 稳定的log-sum-exp计算
    log_probs = predictions - log_sum_exp  # 取对数除法变为减法
    loss = -log_probs[torch.arange(predictions.size(0)), targets].mean()

    return loss


def get_perplexity(loss: torch.Tensor) -> torch.Tensor:
    """
    计算困惑度。
    args:
        loss (torch.Tensor): 交叉熵损失值。
    returns:
        torch.Tensor: 困惑度值。
    """
    return torch.exp(loss).mean()


def cosine_anneal_schedule(
    current_step: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    cosine_anneal_steps: int,
) -> float:
    """
    计算余弦退火学习率调度。
    args:
        current_step (int): 当前训练步骤。
        max_lr (float): 最大学习率。
        min_lr (float): 最小学习率。
        warmup_steps (int): 预热步骤数。
        cosine_anneal_steps (int): 余弦退火步骤数。
    returns:
        float: 计算得到的学习率。
    """
    if current_step < warmup_steps:  # 预热阶段
        lr = max_lr * (current_step / warmup_steps)
    elif (
        current_step >= warmup_steps and current_step < cosine_anneal_steps
    ):  # 余弦退火阶段
        lr = min_lr + 0.5 * (max_lr - min_lr) * (
            1
            + math.cos(
                math.pi
                * (current_step - warmup_steps)
                / (cosine_anneal_steps - warmup_steps)
            )
        )
    else:  # 结束阶段
        lr = min_lr

    return lr


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_norm: float,
) -> None:
    """
    对模型参数进行梯度裁剪。
    args:
        parameters (torch.nn.Parameter): 模型参数。
        max_norm (float): 最大梯度范数。
    """
    l2_norm = torch.sqrt(sum(p.grad.data.norm(2) ** 2 for p in parameters if p.grad is not None))  # type: ignore

    if l2_norm > max_norm:  # 超过最大范数则进行裁剪
        clip_coef = max_norm / (l2_norm + 1e-6)
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    从数据集中获取一个批次的数据。
    args:
        dataset (npt.NDArray): 输入数据集。
        batch_size (int): 批次大小。
        context_length (int): 上下文长度。
        device (str): 设备类型（如 'cpu' 或 'cuda'）。
    returns:
        tuple[torch.Tensor, torch.Tensor]: 输入张量和目标张量。
    """
    dataset_length = dataset.shape[0]
    start_indices = torch.randint(
        0, dataset_length - context_length, (batch_size,)
    )  # 随机选择起始索引

    input_batch = torch.stack(
        [
            torch.tensor(dataset[start_idx : start_idx + context_length], device=device)
            for start_idx in start_indices
        ]
    )

    target_batch = torch.stack(
        [
            torch.tensor(
                dataset[start_idx + 1 : start_idx + context_length + 1],
                device=device,
            )
            for start_idx in start_indices
        ]
    )

    return input_batch, target_batch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
) -> None:
    """
    保存模型检查点。
    args:
        model (torch.nn.Module): 模型。
        optimizer (torch.optim.Optimizer): 优化器。
        iteration (int): 当前迭代次数。
        out (str | os.PathLike | typing.BinaryIO | typing.IO[bytes]): 保存路径或文件对象。
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    加载模型检查点。
    args:
        src (str | os.PathLike | typing.BinaryIO | typing.IO[bytes]): 检查点路径或文件对象。
        model (torch.nn.Module): 模型。
        optimizer (torch.optim.Optimizer): 优化器。
    returns:
        int: 加载的迭代次数。
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    iteration = checkpoint["iteration"]
    return iteration
