from __future__ import annotations

import os
import importlib
import json
import sys
from typing import Any


def _try_import_rust_module() -> Any | None:
    # Prefer package-local module name, but also allow top-level import.
    try:
        return importlib.import_module("cs336_basics._bpe_rust")
    except Exception:
        try:
            return importlib.import_module("_bpe_rust")
        except Exception:
            return None


def _try_import_rust_module_with_errors() -> tuple[Any | None, list[dict[str, str]]]:
    """Try importing the Rust extension and keep exception details for diagnostics."""

    errors: list[dict[str, str]] = []
    module_names = ["cs336_basics._bpe_rust", "_bpe_rust"]
    for name in module_names:
        try:
            mod = importlib.import_module(name)
            return mod, errors
        except Exception as exc:
            errors.append(
                {
                    "module": name,
                    "exc_type": type(exc).__name__,
                    "exc": str(exc),
                }
            )
    return None, errors


def diagnose_rust_bpe_extension() -> dict[str, Any]:
    """Return a structured diagnosis of Rust extension availability.

    This is meant for debugging why the merge backend falls back to Python.
    """

    disabled = os.getenv("CS336_BPE_DISABLE_RUST") == "1"
    mod, errors = _try_import_rust_module_with_errors()

    diagnosis: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "env": {
            "CS336_BPE_DISABLE_RUST": os.getenv("CS336_BPE_DISABLE_RUST"),
        },
        "rust_extension": {
            "disabled": disabled,
            "importable": mod is not None,
            "import_errors": errors,
        },
    }

    if disabled:
        diagnosis["hint"] = (
            "Rust 后端被环境变量禁用：CS336_BPE_DISABLE_RUST=1。"
        )
    elif mod is None:
        diagnosis["hint"] = (
            "未能导入 Rust 扩展模块（cs336_basics._bpe_rust 或 _bpe_rust）。"
            "通常是还没用 maturin 编译/安装到当前解释器环境，或者用错了解释器（例如 conda/system Python）。"
        )
    else:
        diagnosis["hint"] = "Rust 扩展模块可导入。若仍走 Python，请检查是否启用了 tqdm 进度条（会强制 Python）。"

    return diagnosis


def bpe_merge_cached_rust(
    pre_token2freq: dict[tuple[bytes, ...], int],
    vocab: dict[int, bytes],
    vocab_size: int,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]] | None:
    """Optional Rust-accelerated BPE merge.

    Returns None if the Rust extension is unavailable or disabled.

    Disable via env var `CS336_BPE_DISABLE_RUST=1`.
    """

    if os.getenv("CS336_BPE_DISABLE_RUST") == "1":
        return None

    mod = _try_import_rust_module()
    if mod is None:
        return None

    # Convert to a Rust-friendly nested list format.
    seqs_and_freqs: list[tuple[list[bytes], int]] = [
        (list(seq), int(freq)) for seq, freq in pre_token2freq.items() if int(freq) > 0
    ]
    vocab_list: list[bytes] = [vocab[i] for i in range(len(vocab))]

    vocab_out, merges_out = mod.bpe_merge_cached(seqs_and_freqs, vocab_list, int(vocab_size))

    vocab_dict: dict[int, bytes] = {i: bytes(tok) for i, tok in enumerate(vocab_out)}
    merges_list: list[tuple[bytes, bytes]] = [(bytes(a), bytes(b)) for a, b in merges_out]
    return vocab_dict, merges_list


def _main() -> None:
    diag = diagnose_rust_bpe_extension()
    print(json.dumps(diag, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
