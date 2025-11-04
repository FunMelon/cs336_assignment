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

    def compute_flops(self) -> int:
        """计算模型在最大长度下的前向传播 FLOPs。（AI生成代码，正确性待验证）
        returns:
            int: 前向传播的 FLOPs。
        """

        # Attention
        flops_attn = 8 * self.context_length**2 * self.d_model + 8 * self.context_length * self.d_model**2

        # FFN
        flops_ffn = 6 * self.context_length * self.d_model * self.d_ff

        # 每层总 FLOPs
        flops_per_layer = flops_attn + flops_ffn

        # 所有层
        total_flops = self.num_layers * flops_per_layer

        # 输出层
        flops_output = 2 * self.context_length * self.d_model * self.vocab_size

        total_flops += flops_output
        return total_flops