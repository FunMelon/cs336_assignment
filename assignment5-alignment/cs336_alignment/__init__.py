try:
    from importlib.metadata import version

    from .util import (
        tokenize_prompt_and_output,
    )

    __version__ = version("cs336-alignment")
    __all__ = [
        "tokenize_prompt_and_output",
    ]

except Exception:
    # 开发环境下的版本号
    __version__ = "0.1.0-dev"
