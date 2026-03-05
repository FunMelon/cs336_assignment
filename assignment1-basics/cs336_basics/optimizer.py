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
        self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-2, device=None
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


    def to(self, device: torch.device | str, dtype: torch.dtype) -> None:
        """将优化器的状态移动到指定的设备和数据类型。
        args:
            device (torch.device): 目标设备。
            dtype (torch.dtype): 目标数据类型。
        """
        for state in self.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, dtype=dtype)


    @torch.no_grad()
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

                grad = p.grad  # .data会绕过autograd
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
                bias_correction1 = 1 - beta1 ** t 
                bias_correction2 = 1 - beta2 ** t
                adapted_lr = lr * math.sqrt(bias_correction2) / bias_correction1

                # Adam参数更新
                denom = v.sqrt().add(eps)
                p.data.addcdiv_(m, denom, value=-adapted_lr)

                # 应用权重衰减
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)

        return loss


class Muon(torch.optim.Optimizer):
    """实现 Muon 优化器 (MomentUm Orthogonalized by Newton-Schulz)。
    
    Muon 是一种针对神经网络隐藏层参数（尤其是二维权重矩阵）的优化器，
    其核心思想是通过正交化梯度更新矩阵达到显著提升训练效率和模型性能的效果。
    
    注意：Muon 只适用于 2D 参数（矩阵），1D 参数（如 bias、embedding）应使用 AdamW。
    """

    def __init__(
        self,
        params,
        lr=0.02,
        momentum=0.95,
        weight_decay=0.0,
        ns_steps=5,
        ns_coeffs=(3.4445, -4.7750, 2.0315),
        device=None,
    ):
        """
        Args:
            params: 模型参数
            lr: 学习率，默认 0.02
            momentum: 动量系数，默认 0.95
            weight_decay: 权重衰减，默认 0.0
            ns_steps: Newton-Schulz 迭代次数，默认 5
            ns_coeffs: Newton-Schulz 迭代系数 (a, b, c)
            device: 设备
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay value: {weight_decay}")
        if ns_steps < 1:
            raise ValueError(f"Invalid ns_steps value: {ns_steps}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            ns_coeffs=ns_coeffs,
        )
        super().__init__(params, defaults)

    def _newton_schulz(self, G: torch.Tensor, steps: int, coeffs: tuple) -> torch.Tensor:
        """Newton-Schulz 迭代，用于近似计算矩阵的正交因子。
        
        将动量矩阵正交化，使其奇异值趋近于 1。
        
        Args:
            G: 输入梯度/动量矩阵 (2D tensor)
            steps: 迭代次数
            coeffs: 迭代系数 (a, b, c)
            
        Returns:
            正交化后的矩阵
        """
        a, b, c = coeffs
        eps = 1e-7
        
        # 确保 G 是 2D 矩阵，如果 rows > cols，则转置
        transposed = False
        if G.shape[0] > G.shape[1]:
            G = G.T
            transposed = True
        
        # 转换为 bfloat16 以提高计算效率（如果支持）
        original_dtype = G.dtype
        if G.device.type == 'cuda' and hasattr(torch, 'bfloat16'):
            X = G.bfloat16()
        else:
            X = G.float()
        
        # 归一化，确保谱范数 <= 1
        X = X / (X.norm() + eps)
        
        # Newton-Schulz 迭代
        for _ in range(steps):
            # A = X @ X.T (Gram 矩阵)
            A = X @ X.T
            # B = b * A + c * A @ A
            B = b * A + c * A @ A
            # X = a * X + B @ X
            X = a * X + B @ X
        
        # 转换回原始数据类型
        X = X.to(original_dtype)
        
        # 如果之前转置了，现在转置回来
        if transposed:
            X = X.T
        
        return X

    def _adjust_lr(self, lr: float, param_shape: tuple) -> float:
        """根据参数形状调整学习率。
        
        确保正交化后的更新在不同形状的矩阵上具有一致的 RMS。
        
        Args:
            lr: 原始学习率
            param_shape: 参数形状 (rows, cols)
            
        Returns:
            调整后的学习率
        """
        A, B = param_shape[0], param_shape[1]
        # 使用原始 Muon 的学习率调整策略
        adjusted_ratio = math.sqrt(max(1, A / B))
        return lr * adjusted_ratio

    def to(self, device: torch.device | str, dtype: torch.dtype) -> None:
        """将优化器的状态移动到指定的设备和数据类型。"""
        for state in self.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, dtype=dtype)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """进行单次优化步骤。
        
        Args:
            closure (Callable, optional): 一个闭包函数，用于重新计算损失。
            
        Returns:
            Optional[float]: 如果提供了闭包函数，则返回损失值，否则返回 None。
        """
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]
            ns_coeffs = group["ns_coeffs"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # 只对 2D 参数（矩阵）使用 Muon，1D 参数直接使用 SGD-like 更新
                if p.ndim != 2:
                    # 对于非 2D 参数，使用简单的带动量的 SGD
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad)
                    
                    if weight_decay != 0:
                        p.mul_(1 - lr * weight_decay)
                    p.add_(buf, alpha=-lr)
                    continue

                # 对于 2D 参数，使用 Muon 正交化更新
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)

                # 更新动量: m = momentum * m + grad
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)

                # 正交化动量矩阵
                update = self._newton_schulz(buf, ns_steps, ns_coeffs)

                # 调整学习率
                adjusted_lr = self._adjust_lr(lr, p.shape)

                # 应用权重衰减（解耦）
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                # 参数更新
                p.add_(update, alpha=-adjusted_lr)

        return loss


class Magma:
    """Magma (Momentum-aligned Gradient Masking) 优化器包装器。
    
    Magma 是一种基于动量-梯度对齐的随机梯度掩码方法，可作为包装器应用于
    任何自适应优化器（如 AdamW）或动量优化器（如 Muon）。
    
    核心思想：通过计算每个参数块的一阶矩（动量）与当前梯度之间的余弦相似度，
    对对齐度低的参数块按对齐分数缩放梯度，而对齐度高的块保持不变。
    这引入了一种曲率依赖的几何正则化效果，平滑优化轨迹。
    
    参考论文: arxiv 2602.15322
    """

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        tau: float = 2.0,
        p: float = 0.5,
        ema_decay: float = 0.9,
    ):
        """
        Args:
            base_optimizer: 底层优化器实例（AdamW 或 Muon）
            tau: 温度参数，控制对齐分数的锐度，默认 2.0
            p: Bernoulli 存活概率（mask=1 保留原始梯度的概率），默认 0.5
            ema_decay: 对齐分数 EMA 衰减系数，默认 0.9
        """
        self.base_optimizer = base_optimizer
        self.tau = tau
        self.p = p
        self.ema_decay = ema_decay
        # 存储每个参数块的 EMA 对齐分数
        self._alignment_ema = {}

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    def _get_momentum(self, param: torch.Tensor) -> torch.Tensor | None:
        """从底层优化器获取参数的一阶矩估计（动量）。
        
        对于 AdamW，一阶矩为 state["m"]。
        对于 Muon，一阶矩为 state["momentum_buffer"]。
        """
        state = self.base_optimizer.state.get(param, {})
        if "m" in state:
            return state["m"]
        if "momentum_buffer" in state:
            return state["momentum_buffer"]
        return None

    @torch.no_grad()
    def step(self, closure=None):
        """执行 Magma 包装的优化步骤。
        
        步骤：
        1. 对每个参数块计算动量-梯度对齐分数
        2. 按对齐分数和随机掩码修改梯度
        3. 调用底层优化器的 step()
        4. 恢复被修改的梯度（确保梯度累积等场景不受影响）
        """
        # 收集需要处理的参数及其原始梯度
        modified_params = []

        for group in self.base_optimizer.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                momentum = self._get_momentum(param)

                # 第一步没有动量时跳过 Magma 掩码
                if momentum is None:
                    continue

                # 将 momentum 和 grad 展平计算余弦相似度
                mu_flat = momentum.flatten().float()
                g_flat = grad.flatten().float()

                # cossim(μ, g)
                mu_norm = mu_flat.norm()
                g_norm = g_flat.norm()
                if mu_norm == 0 or g_norm == 0:
                    continue

                cossim = torch.dot(mu_flat, g_flat) / (mu_norm * g_norm)

                # 对齐分数: s̃ = sigmoid(cossim / τ)
                alignment = torch.sigmoid(cossim / self.tau).item()

                # EMA 更新对齐分数
                param_id = id(param)
                if param_id in self._alignment_ema:
                    s = self.ema_decay * self._alignment_ema[param_id] + (1 - self.ema_decay) * alignment
                else:
                    s = alignment
                self._alignment_ema[param_id] = s

                # Bernoulli 掩码采样
                mask = 1 if torch.rand(1).item() < self.p else 0

                if mask == 0:
                    # 掩码为 0：用对齐分数缩放梯度
                    original_grad = param.grad.clone()
                    param.grad.mul_(s)
                    modified_params.append((param, original_grad))
                # mask == 1: 保留原始梯度，不做任何修改

        # 调用底层优化器
        loss = self.base_optimizer.step(closure)

        # 恢复被修改的梯度（保证外部逻辑如梯度累积不受影响）
        for param, original_grad in modified_params:
            param.grad.copy_(original_grad)

        return loss

    def zero_grad(self, set_to_none=True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "base_optimizer": self.base_optimizer.state_dict(),
            "alignment_ema": dict(self._alignment_ema),
        }

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict["base_optimizer"])
        self._alignment_ema = state_dict.get("alignment_ema", {})

    def to(self, device, dtype):
        if hasattr(self.base_optimizer, "to"):
            self.base_optimizer.to(device, dtype)


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
