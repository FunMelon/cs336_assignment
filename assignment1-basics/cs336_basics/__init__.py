# cs336_basics/__init__.py
try:
    from importlib.metadata import version
    __version__ = version("cs336_basics")
except Exception:
    # 开发环境下的版本号
    __version__ = "0.1.0-dev"