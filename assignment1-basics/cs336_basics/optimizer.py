from collections.abc import Callable
from typing import Optional
import torch
import math


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr)  # 为优化器设置默认参数
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """进行单次优化步骤。
        args:
            closure (Callable, optional): 一个闭包函数，用于重新计算损失。
        returns:
            Optional[float]: 如果提供了闭包函数，则返回损失值，否则返回 None。
        """
        loss = None if closure is None else closure()
        for group in self.param_groups:  # 对不同的参数组进行迭代
            lr = group["lr"]
            for p in group["params"]:  # 对参数组中的每个参数进行迭代
                if p.grad is None:
                    continue

                state = self.state[p]  # 获取参数的状态字典
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr * math.sqrt(t + 1) * grad
                state["t"] = t + 1

        return loss


class AdamW(torch.optim.Optimizer):
    """实现 AdamW 优化器。"""

    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-2
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """进行单次优化步骤。
        args:
            closure (Callable, optional): 一个闭包函数，用于重新计算损失。
        returns:
            Optional[float]: 如果提供了闭包函数，则返回损失值，否则返回 None。
        """
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:  # 初始化状态
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)  # 一阶矩估计
                    state["v"] = torch.zeros_like(p.data)  # 二阶矩估计

                m, v = state["m"], state["v"]
                state["t"] += 1
                t = state["t"]
                # 更新一阶和二阶矩估计
                m.mul_(beta1).add_(grad, alpha=1 - beta1)  # 直接修改以节省内存
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                # 计算偏差修正后的学习率
                adapted_lr = (
                    lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                )
                # 更新参数
                p.data -= adapted_lr * m / v.sqrt().add_(eps)
                # 应用权重衰减
                p.data -= lr * weight_decay * p.data

        return loss


if __name__ == "__main__":
    weights = torch.nn.Parameter(
        5 * torch.randn((10, 10))
    )  # 初始化参数，形状为 (10, 10)，值为随机数乘以 5
    opt = SGD([weights], lr=1e1)  # 创建自定义的 SGD 优化器，学习率为 1
    for t in range(10):
        opt.zero_grad()
        loss = (weights**2).mean()
        print(loss.cpu().item())
        loss.backward()
        opt.step()
