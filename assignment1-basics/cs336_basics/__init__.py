# cs336_basics/__init__.py
try:
    from importlib.metadata import version

    # 分词器
    from .tokenizer import Tokenizer
    from .bpe.bpe_trainer import find_chunk_boundaries, bpe

    # module
    from .util import (
        SiLU,
        softmax,
        scaled_dot_product_attention,
        cross_entropy_loss,
    )
    from .module import (
        Linear,
        Embedding,
        RMSNorm,
        PositionwiseFeedForward,
        RoPE,
        MultiheadSelfAttention,
        TransformerBlock,
    )
    from .transformer import Transformer

    __version__ = version("cs336_basics")
    __all__ = [
        "Tokenizer",
        "find_chunk_boundaries",
        "bpe",
        "Linear",
        "Embedding",
        "RMSNorm",
        "SiLU",
        "PositionwiseFeedForward",
        "RoPE",
        "softmax",
        "scaled_dot_product_attention",
        "MultiheadSelfAttention",
        "TransformerBlock",
        "Transformer",
        "cross_entropy_loss",
    ]

except Exception:
    # 开发环境下的版本号
    __version__ = "0.1.0-dev"
