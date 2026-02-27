import torch
from .module import TransformerBlock, Embedding, RoPE, Linear, RMSNorm
from .util import softmax
from .tokenizer import Tokenizer

class Transformer(torch.nn.Module):
    """一个完整的Transformer模型类，包含嵌入层、多层Transformer块、归一化层和线性输出层。"""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        logit_cap: float = 0.0,
        tokenizer: Tokenizer | None = None,
        device=None,
        dtype=None,
    ):
        """初始化Transformer模型。
        args:
            vocab_size (int): 词汇表大小。
            context_length (int): 最大上下文长度（序列长度）。
            d_model (int): 模型的隐藏维度。
            nhead (int): 多头注意力机制中的头数。
            num_layers (int): Transformer块的数量。
            d_ff (int): 前馈网络的隐藏层维度。
            tokenizer (Tokenizer, optional): 分词器，用于生成文本输出。
            device: 设备信息（如 'cpu' 或 'cuda'）。
            dtype: 数据类型（如 torch.float32）。
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.logit_cap = logit_cap
        self.tokenizer = tokenizer
        self.device = device == None and torch.device("cpu") or device
        self.dtype = dtype == None and torch.float32 or dtype

        rope = RoPE(
            theta=rope_theta,
            d_k=d_model // nhead,
            max_seq_len=context_length,
            device=device,
        )
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.transformer_blocks = torch.nn.ModuleList(
            [
                TransformerBlock(
                    d_model, nhead, d_ff, rope=rope, device=device, dtype=dtype
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """前向传播函数。
        args:
            input_ids (torch.Tensor): 输入的词ID张量，形状为 (batch_size, sequence_length)。
        returns:
            torch.Tensor: 输出的 logits 张量，形状为 (batch_size, sequence_length, vocab_size)。
        """
        x = self.embedding(input_ids)  # 形状为 (batch_size, sequence_length, d_model)
        for block in self.transformer_blocks:
            x = block(x)  # 形状保持不变
        x = self.ln_final(x)  # 形状为 (batch_size, sequence_length, d_model)
        logits = self.lm_head(x)  # 形状为 (batch_size, sequence_length, vocab_size)
        # Logit Softcapping: 防止logits爆炸，稳定训练
        if self.logit_cap > 0:
            logits = self.logit_cap * torch.tanh(logits / self.logit_cap)
        return logits

    def generate_text(self, prompts: str, temperature: float = 1.0, top_p: float = 1.0, max_length: int | None = None) -> str:
        """生成文本。
        args:
            prompts (str): 输入的提示文本。
            temperature (float): 生成文本的温度参数。
            top_p (float): top-p 采样的累计概率阈值。
            max_length (int): 生成的最大长度。
        returns:
            str: 生成的文本。
        """
        
        assert temperature > 0, "Temperature must be greater than 0."
        if len(prompts) > self.context_length:
            raise ValueError("Input prompt length exceeds model's context length.")
        if max_length is None:
            max_length = self.context_length
        elif max_length > self.context_length:
            raise ValueError("max_length cannot be greater than context_length.")
        elif max_length <= 0:
            raise ValueError("max_length must be a positive integer.")

        if self.tokenizer is None:
            raise ValueError("Tokenizer is not provided for text generation.")

        self.eval() # 设置为评估模式
        with torch.no_grad():
            input_ids = torch.tensor(self.tokenizer.encode(prompts), device=self.device).unsqueeze(0)  # 添加批次维度
            
            special_tokens = self.tokenizer.special_tokens or []
            special_tokens_ids = [
                self.tokenizer.token2id[st.encode('utf-8')] for st in special_tokens
            ]   # 获取特殊标记的ID列表
            for _ in range(max_length - input_ids.size(1)):
                logits = self.forward(input_ids)
                scaled_logits = logits[:, -1, :] / temperature  # 使用温度缩放
                probs = softmax(scaled_logits, dim=-1)

                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                    sorted_indices_to_remove = cumulative_probs > top_p     # 标记要移除的标记
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()  # 将布尔标记整体右移一位
                    sorted_indices_to_remove[..., 0] = False  # 保留至少一个标记

                    # 将超过 top_p 的概率设为0
                    sorted_probs = sorted_probs.masked_fill(sorted_indices_to_remove, 0.0)

                    # 将修改后的概率重新映射回原始索引位置
                    probs.zero_().scatter_(1, sorted_indices, sorted_probs)
                    # 重新归一化
                    probs = probs / probs.sum(dim=-1, keepdim=True)

                next_token = torch.multinomial(probs, num_samples=1)  # 随机采样下一个标记
                if special_tokens_ids and next_token.item() in special_tokens_ids:  # 输出特殊标记则停止生成
                    break
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return self.tokenizer.decode(input_ids.squeeze().tolist())
    

    def compute_params(self) -> int:
        """计算模型的总参数数量。
        returns:
            int: 模型的总参数数量。
        """
        return sum(p.numel() for p in self.parameters())

    def compute_params_self(self) -> int:
        """计算模型的总参数数量，按照公式计算。
        returns:
            int: 模型的总参数数量。
        """
        total_params = 0
        total_params += 2 * self.vocab_size * self.d_model  # Embedding 和 lm_head
        total_params += self.num_layers * (
            4 * self.d_model * self.d_model  # Q, K, V, O 矩阵
            + 3 * self.d_model * self.d_ff  # 前馈网络的权重
            + 2 * self.d_model  # RMSNorm 的参数
        )
        total_params += self.d_model  # RMSNorm 的参数

        return total_params

    def compute_forward_flops(self, batch_size: int = 1, k_soft: int = 5) -> int:
        """
        计算模型在最大长度下的前向传播 FLOPs（近似值）。（AI生成）
        参数:
            batch_size: 批大小（默认为1）
            k_soft: softmax 每元素的常数开销估计（通常取 5~10，可调整）
        返回:
            int: 总 FLOPs（近似）
        假设:
        - QKV 合并线性 (D -> 3D)
        - attention 中 QK^T 和 A@V 各为 2*B*L^2*d_model 总项中的 2*...
        - FFN 按 SwiGLU/常见实现近似为 6 * B * L * D * F
        """
        B = int(batch_size)
        L = int(self.context_length)
        D = int(self.d_model)
        F = int(self.d_ff)
        N = int(self.num_layers)
        H = int(self.nhead)
        V = int(self.vocab_size)

        # QKV 投影: 2 * B * L * D * (3D) = 6 * B * L * D^2
        flops_qkv = 6 * B * L * D * D

        # QK^T: 2 * B * H * L^2 * d_k  where d_k = D / H  => simplifies to 2 * B * L^2 * D
        flops_qk = 2 * B * (L**2) * D

        # softmax 行为近似成本: k_soft * B * H * L^2
        flops_softmax = k_soft * B * H * (L**2)

        # A @ V: same cost as QK^T
        flops_av = 2 * B * (L**2) * D

        # attention 输出线性 proj: 2 * B * L * D^2
        flops_outproj = 2 * B * L * D * D

        # attention 总和（每层）
        flops_attn_per_layer = (
            flops_qkv + flops_qk + flops_softmax + flops_av + flops_outproj
        )
        # print("FLOPs per attention layer:", flops_attn_per_layer)
        # FFN（SwiGLU 风格近似）: 6 * B * L * D * F
        flops_ffn_per_layer = 6 * B * L * D * F
        # print("FLOPs per FFN layer:", flops_ffn_per_layer)
        # 每层总 FLOPs
        flops_per_layer = flops_attn_per_layer + flops_ffn_per_layer

        # 所有层
        total_flops = N * flops_per_layer

        # lm_head（输出层）: 2 * B * L * D * V
        flops_output = 2 * B * L * D * V
        # print("FLOPs for output layer:", flops_output)
        total_flops += flops_output

        return int(total_flops)

    def compute_peek_memory(self, batch_size: int = 1) -> int:
        """
        计算模型在最大长度下的前向传播峰值内存使用量（近似值）。
        参数:
            batch_size: 批大小（默认为1）
        返回:
            int: 峰值内存使用量（近似，单位：字节）
        假设:
        - 每个激活值占用 4 字节（float32）
        - 考虑输入嵌入、每层的中间激活和输出
        """

        if self.dtype == torch.float16 or self.dtype == torch.bfloat16:
            bytes_per_param = 2
        else:
            bytes_per_param = 4

        # 参数内存
        param_memory = self.compute_params() * bytes_per_param
        # 梯度内存
        grad_memory = param_memory
        # 优化器状态内存（假设 AdamW，约为参数的两倍）
        optim_memory = 2 * param_memory
        # 激活内存
        activation_memory = (
            batch_size * self.context_length * self.d_model * bytes_per_param
        )  # 输入层激活
        activation_memory += (
            batch_size
            * self.context_length
            * self.d_model
            * bytes_per_param
            * self.num_layers
        )  # transformer层激活
        activation_memory += (
            batch_size * self.context_length * self.vocab_size * bytes_per_param
        )  # 输出层激活

        total_memory = param_memory + grad_memory + optim_memory + activation_memory

        return int(total_memory)

    def compute_backward_flops(self, batch_size: int = 1) -> int:
        """
        计算模型在最大长度下的反向传播 FLOPs（近似值）。
        参数:
            batch_size: 批大小（默认为1）
        返回:
            int: 反向传播的总 FLOPs（近似）
        假设:
        - 反向传播 FLOPs 约为前向传播的两倍
        """
        forward_flops = self.compute_forward_flops(batch_size)
        backward_flops = 2 * forward_flops
        return int(backward_flops)
    
    def compute_adamw_flops(self, batch_size: int = 1) -> int:
        """
        计算运行一个步骤的 AdamW 所需的 FLOPs（近似值）。
        参数:
            batch_size: 批大小（默认为1）
        返回:
            int: AdamW 优化器的 FLOPs（近似）
        假设:
        - 每个参数有一个梯度
        - 每个参数都有一阶矩 m 和二阶矩 v
        """
        total_params = self.compute_params()
        # 每个参数的 FLOPs 估计
        flops_per_param = 8  # 计算 m, v 更新和参数更新的近似 FLOPs
        total_flops = total_params * flops_per_param
        return int(total_flops)

    def __str__(self) -> str:
        """返回Transformer模型的字符串表示形式。"""
        return (
            f"Transformer(vocab_size={self.vocab_size}, "
            f"context_length={self.context_length}, "
            f"d_model={self.d_model}, "
            f"nhead={self.nhead}, "
            f"num_layers={self.num_layers}, "
            f"d_ff={self.d_ff})"
        )
