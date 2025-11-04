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