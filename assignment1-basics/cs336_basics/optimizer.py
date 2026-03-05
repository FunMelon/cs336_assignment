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


class RMSProp(torch.optim.Optimizer):
    """实现 RMSProp 优化器（带解耦权重衰减）。
    
    RMSProp 只维护二阶矩（梯度平方的 EMA）用于自适应学习率缩放，
    不包含内置的一阶动量。这使其非常适合与 Magma 配合使用，
    因为 Magma 提供独立的动量-梯度对齐信号，两者职责互补、无冗余。
    
    更新规则：
        v_t = α * v_{t-1} + (1 - α) * g_t²
        θ_t = θ_{t-1} - lr * g_t / (√v_t + ε)
        θ_t = θ_t * (1 - lr * weight_decay)  （解耦权重衰减）
    """

    def __init__(
        self, params, lr=1e-3, alpha=0.99, eps=1e-8, weight_decay=0.0
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= alpha < 1.0:
            raise ValueError(f"Invalid alpha value: {alpha}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay value: {weight_decay}")

        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def to(self, device: torch.device | str, dtype: torch.dtype) -> None:
        for state in self.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, dtype=dtype)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            alpha = group["alpha"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["v"] = torch.zeros_like(p.data)

                v = state["v"]
                # 更新二阶矩: v = α * v + (1 - α) * g²
                v.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

                # 参数更新: θ -= lr * g / (√v + ε)
                denom = v.sqrt().add_(eps)
                p.data.addcdiv_(grad, denom, value=-lr)

                # 解耦权重衰减
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)

        return loss


class Magma:
    """Magma (Momentum-aligned Gradient Masking) 优化器包装器。
    
    Magma 维护自己独立的一阶动量 μ（梯度的 EMA），用于计算与当前梯度的对齐度。
    对齐度低的参数块会被缩放抑制，起到曲率依赖的几何正则化效果。
    
    与底层优化器的关系：
    - RMSProp（推荐）：RMSProp 只有二阶矩，Magma 提供一阶动量信号，职责互补
    - Adam：Adam 自带一阶动量 m，与 Magma 的 μ 存在冗余，可能导致过度干预
    
    参考论文: arxiv 2602.15322
    """

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        tau: float = 2.0,
        p: float = 0.5,
        ema_decay: float = 0.9,
        beta: float = 0.9,
    ):
        """
        Args:
            base_optimizer: 底层优化器实例
            tau: 温度参数，控制对齐分数的锐度，默认 2.0
            p: Bernoulli 存活概率（mask=1 保留原始梯度的概率），默认 0.5
            ema_decay: 对齐分数 EMA 衰减系数，默认 0.9
            beta: Magma 自身动量 μ 的 EMA 系数，默认 0.9
        """
        self.base_optimizer = base_optimizer
        self.tau = tau
        self.p = p
        self.ema_decay = ema_decay
        self.beta = beta
        self._momentum_buffers = {}  # param_id -> μ tensor（Magma 自己的动量）
        self._alignment_ema = {}     # param_id -> EMA 对齐分数

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    @torch.no_grad()
    def step(self, closure=None):
        """执行 Magma 包装的优化步骤。
        
        步骤：
        1. 对每个参数块，更新 Magma 自己的动量 μ = β * μ + (1-β) * g
        2. 计算动量-梯度对齐分数 s̃ = sigmoid(cossim(μ, g) / τ)
        3. EMA 平滑对齐分数 s = 0.9 * s_prev + 0.1 * s̃
        4. Bernoulli(p) 掩码：mask=0 时用 s 缩放梯度，mask=1 时保留原始梯度
        5. 调用底层优化器 step()
        6. 恢复被修改的梯度
        """
        modified_params = []

        for group in self.base_optimizer.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                param_id = id(param)

                # 更新 Magma 自己的动量: μ = β * μ + (1 - β) * g
                if param_id not in self._momentum_buffers:
                    self._momentum_buffers[param_id] = torch.zeros_like(grad)
                mu = self._momentum_buffers[param_id]
                mu.mul_(self.beta).add_(grad, alpha=1 - self.beta)

                # 计算余弦相似度: cossim(μ, g)
                mu_flat = mu.flatten().float()
                g_flat = grad.flatten().float()
                mu_norm = mu_flat.norm()
                g_norm = g_flat.norm()
                if mu_norm == 0 or g_norm == 0:
                    continue

                cossim = torch.dot(mu_flat, g_flat) / (mu_norm * g_norm)

                # 对齐分数: s̃ = sigmoid(cossim / τ)
                alignment = torch.sigmoid(cossim / self.tau).item()

                # EMA 平滑对齐分数
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

        # 调用底层优化器
        loss = self.base_optimizer.step(closure)

        # 恢复被修改的梯度
        for param, original_grad in modified_params:
            param.grad.copy_(original_grad)

        return loss

    def zero_grad(self, set_to_none=True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        # 序列化时用参数索引映射替代 id(param)
        param_to_idx = {}
        idx = 0
        for group in self.base_optimizer.param_groups:
            for param in group["params"]:
                param_to_idx[id(param)] = idx
                idx += 1

        momentum_state = {}
        for pid, buf in self._momentum_buffers.items():
            if pid in param_to_idx:
                momentum_state[param_to_idx[pid]] = buf

        alignment_state = {}
        for pid, val in self._alignment_ema.items():
            if pid in param_to_idx:
                alignment_state[param_to_idx[pid]] = val

        return {
            "base_optimizer": self.base_optimizer.state_dict(),
            "momentum_buffers": momentum_state,
            "alignment_ema": alignment_state,
        }

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict["base_optimizer"])

        # 恢复时用参数索引重建 id 映射
        idx_to_param_id = {}
        idx = 0
        for group in self.base_optimizer.param_groups:
            for param in group["params"]:
                idx_to_param_id[idx] = id(param)
                idx += 1

        self._momentum_buffers = {}
        for str_idx, buf in state_dict.get("momentum_buffers", {}).items():
            i = int(str_idx)
            if i in idx_to_param_id:
                self._momentum_buffers[idx_to_param_id[i]] = buf

        self._alignment_ema = {}
        for str_idx, val in state_dict.get("alignment_ema", {}).items():
            i = int(str_idx)
            if i in idx_to_param_id:
                self._alignment_ema[idx_to_param_id[i]] = val

    def to(self, device, dtype):
        if hasattr(self.base_optimizer, "to"):
            self.base_optimizer.to(device, dtype)
        for pid in self._momentum_buffers:
            self._momentum_buffers[pid] = self._momentum_buffers[pid].to(device=device, dtype=dtype)


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
