# cs336_basics/__init__.py
try:
    from importlib.metadata import version
    from .tokenizer import Tokenizer
    from .bpe_trainer import find_chunk_boundaries, bpe

    __version__ = version("cs336_basics")
    __all__ = ['Tokenizer', 'find_chunk_boundaries', 'bpe']

except Exception:
    # 开发环境下的版本号
    __version__ = "0.1.0-dev"