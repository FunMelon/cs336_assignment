import torch
from .module import TransformerBlock, Embedding, RoPE, Linear, RMSNorm
from .util import softmax


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
        self.device = device == None and torch.device("cpu") or device
        self.dtype = dtype == None and torch.float32 or dtype

        rope = RoPE(
            theta=10000.0,
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
        x = self.ln_final(x) # 形状为 (batch_size, sequence_length, d_model)
        logits = self.lm_head(
            x
        )  # 形状为 (batch_size, sequence_length, vocab_size)
        return logits

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

    def compute_flops(self, batch_size: int = 1, k_soft: int = 5) -> int:
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
        flops_qk = 2 * B * (L ** 2) * D

        # softmax 行为近似成本: k_soft * B * H * L^2
        flops_softmax = k_soft * B * H * (L ** 2)

        # A @ V: same cost as QK^T
        flops_av = 2 * B * (L ** 2) * D

        # attention 输出线性 proj: 2 * B * L * D^2
        flops_outproj = 2 * B * L * D * D

        # attention 总和（每层）
        flops_attn_per_layer = flops_qkv + flops_qk + flops_softmax + flops_av + flops_outproj
        print("FLOPs per attention layer:", flops_attn_per_layer)
        # FFN（SwiGLU 风格近似）: 6 * B * L * D * F
        flops_ffn_per_layer = 6 * B * L * D * F
        print("FLOPs per FFN layer:", flops_ffn_per_layer)
        # 每层总 FLOPs
        flops_per_layer = flops_attn_per_layer + flops_ffn_per_layer

        # 所有层
        total_flops = N * flops_per_layer

        # lm_head（输出层）: 2 * B * L * D * V
        flops_output = 2 * B * L * D * V
        print("FLOPs for output layer:", flops_output)
        total_flops += flops_output

        return int(total_flops)
        