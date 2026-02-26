# Rust 加速（可选）

本目录的 BPE 合并热点（`_bpe_merge_cached`）提供了一个可选的 Rust/PyO3 加速实现。

- 默认行为：如果检测到 Rust 扩展模块可导入，则优先使用 Rust；否则自动回退到纯 Python 实现。
- 可通过环境变量禁用：`CS336_BPE_DISABLE_RUST=1`

## 构建/安装（开发机本地）

前提：已安装 Rust（`cargo`）以及可用的 Python 开发环境。

1) 安装 `maturin`（用于构建 PyO3 扩展）：

- `uv pip install maturin`

2) 构建并安装扩展到当前环境：

- `cd assignment1-basics`
- `uv run maturin develop --release -m cs336_basics/bpe/rust_bpe/Cargo.toml`

> 说明：若你希望模块安装为 `cs336_basics._bpe_rust`（包内子模块形式），可以使用 maturin 的 `--module-name` 选项（如果你的 maturin 版本支持），或调整构建脚本把生成的 `.so` 放到 `cs336_basics/` 目录下；Python 侧也兼容直接导入顶层 `_bpe_rust`。