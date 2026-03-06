# cs336_systems/__init__.py
try:
    from importlib.metadata import version

    from .flash_attention_pytorch import FlashAttention as FlashAttentionPyTorch
    from .flash_attention import FlashAttentionTriton as FlashAttention
    from .ddp import DDPIndividualParameters

    __version__ = version("cs336-systems")
    __all__ = [
        "FlashAttentionPyTorch",
        "FlashAttention",
        "DDPIndividualParameters",
    ]

except Exception:
    # 开发环境下的版本号
    __version__ = "0.1.0-dev"
