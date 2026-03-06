#  profile脚本，用来分析bpe训练的性能瓶颈
from __future__ import annotations

import argparse
from pathlib import Path

from cs336_basics.bpe import streaming_bpe_trainer


def _default_input_path() -> Path:
    here = Path(__file__).resolve().parent
    return here / "../../data/owt_train.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile cs336_basics.bpe.bpe_trainer.bpe")
    parser.add_argument(
        "--input",
        type=Path,
        default=_default_input_path(),
        help="Path to training corpus (default: tests/fixtures/corpus.en)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=500,
        help="Vocabulary size including special tokens (default: 500)",
    )
    parser.add_argument(
        "--special-token",
        action="append",
        default=["<|endoftext|>"],
        help="Special token to add (repeatable). Default: <|endoftext|>",
    )

    args = parser.parse_args()

    vocab, merges = streaming_bpe_trainer.bpe_streaming(
        input_path=str(args.input),
        vocab_size=args.vocab_size,
        special_tokens=list(args.special_token),
        chunk_size_mb=256,
    )

    # Small sanity output so it's obvious it ran.
    print(f"Trained vocab size: {len(vocab)}")
    print(f"Trained merges: {len(merges)}")


if __name__ == "__main__":
    main()
