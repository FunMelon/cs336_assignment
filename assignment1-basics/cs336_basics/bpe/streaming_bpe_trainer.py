# 负责从语料中训练bpe编码器，并将结果保存到本地
import os
from typing import BinaryIO, Iterator
import regex as re  # 使用 regex 库，由于re对GPT-2的tokenization支持不好
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import heapq
from typing import Any
import time


class _RevPair:
    """用于 heapq 的 pair 排序包装器（频次相同取字典序更大的 pair）。"""

    __slots__ = ("pair",)

    def __init__(self, pair: tuple[bytes, bytes]):
        self.pair = pair

    def __lt__(self, other: "_RevPair") -> bool:  # type: ignore[override]
        return self.pair > other.pair


def _pair_counts(byte_seq: tuple[bytes, ...]) -> dict[tuple[bytes, bytes], int]:
    counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    for a, b in zip(byte_seq, byte_seq[1:]):
        counts[(a, b)] += 1
    return counts


def _merge_pair_in_seq(
    byte_seq: tuple[bytes, ...], pair: tuple[bytes, bytes], new_token: bytes
) -> tuple[bytes, ...]:
    if len(byte_seq) < 2:
        return byte_seq
    merged: list[bytes] = []
    i = 0
    a, b = pair
    while i < len(byte_seq):
        if i < len(byte_seq) - 1 and byte_seq[i] == a and byte_seq[i + 1] == b:
            merged.append(new_token)
            i += 2
        else:
            merged.append(byte_seq[i])
            i += 1
    return tuple(merged)


def _bpe_merge_cached_py(
    pre_token2freq: dict[tuple[bytes, ...], int],
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    vocab_size: int,
    pbar: Any | None = None,
) -> None:
    """缓存 + 增量更新版 BPE merge，可选 tqdm 进度条。"""

    pair_freq: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair2seqs: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)

    for byte_seq, freq in pre_token2freq.items():
        if len(byte_seq) < 2:
            continue
        for a, b in zip(byte_seq, byte_seq[1:]):
            pair = (a, b)
            pair_freq[pair] += freq
            pair2seqs[pair].add(byte_seq)

    heap: list[tuple[int, _RevPair]] = [
        (-freq, _RevPair(pair)) for pair, freq in pair_freq.items() if freq > 0
    ]
    heapq.heapify(heap)

    def push_pair(pair: tuple[bytes, bytes]) -> None:
        freq = pair_freq.get(pair, 0)
        if freq > 0:
            heapq.heappush(heap, (-freq, _RevPair(pair)))

    while len(vocab) < vocab_size:
        most_frequent_pair: tuple[bytes, bytes] | None = None
        while heap:
            neg_freq, rev_pair = heapq.heappop(heap)
            pair = rev_pair.pair
            freq = -neg_freq
            if pair_freq.get(pair, 0) != freq:
                continue
            if freq <= 0:
                continue
            most_frequent_pair = pair
            break

        if most_frequent_pair is None:
            if pbar is not None:
                pbar.n = pbar.total
                pbar.refresh()
            break

        a, b = most_frequent_pair
        new_token = a + b
        vocab[len(vocab)] = new_token
        merges.append(most_frequent_pair)
        if pbar is not None:
            pbar.update(1)

        affected_seqs = pair2seqs.get(most_frequent_pair)
        if not affected_seqs:
            pair_freq[most_frequent_pair] = 0
            continue
        affected_seqs = set(affected_seqs)
        pair2seqs.pop(most_frequent_pair, None)

        for old_seq in affected_seqs:
            old_freq = pre_token2freq.get(old_seq)
            if old_freq is None:
                continue

            old_counts = _pair_counts(old_seq)
            new_seq = _merge_pair_in_seq(old_seq, most_frequent_pair, new_token)
            if new_seq == old_seq:
                continue
            new_counts = _pair_counts(new_seq)

            del pre_token2freq[old_seq]
            pre_token2freq[new_seq] = pre_token2freq.get(new_seq, 0) + old_freq

            for pair, cnt in old_counts.items():
                pair_freq[pair] -= old_freq * cnt
                if pair_freq[pair] <= 0:
                    pair_freq[pair] = 0
                seqs = pair2seqs.get(pair)
                if seqs is not None:
                    seqs.discard(old_seq)
                    if not seqs:
                        pair2seqs.pop(pair, None)

            for pair, cnt in new_counts.items():
                pair_freq[pair] += old_freq * cnt
                pair2seqs[pair].add(new_seq)

            for pair in set(old_counts.keys()) | set(new_counts.keys()):
                push_pair(pair)


def _bpe_merge_cached(
    pre_token2freq: dict[tuple[bytes, ...], int],
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    vocab_size: int,
    pbar: Any | None = None,
) -> None:
    """优先走 Rust（若可用），否则回退 Python。

    Rust 分支不支持 tqdm 进度条；如果启用 pbar，将强制走 Python。
    """

    if pbar is None:
        rust_result = None
        try:
            try:
                from .rust_accel import bpe_merge_cached_rust
            except Exception:
                # 允许以脚本方式运行：`python streaming_bpe_trainer.py`
                from rust_accel import bpe_merge_cached_rust  # type: ignore

            rust_result = bpe_merge_cached_rust(pre_token2freq, vocab, vocab_size)
        except Exception:
            rust_result = None

        if rust_result is not None:
            print("[cs336_basics.bpe.streaming] merge backend: rust", flush=True)
            vocab_out, merges_out = rust_result
            vocab.clear()
            vocab.update(vocab_out)
            merges.clear()
            merges.extend(merges_out)
            return

        print("[cs336_basics.bpe.streaming] merge backend: python", flush=True)
    else:
        print("[cs336_basics.bpe.streaming] merge backend: python (tqdm enabled)", flush=True)

    _bpe_merge_cached_py(pre_token2freq, vocab, merges, vocab_size, pbar)

def bytes_to_unicode():
    """
    将任意字节映射为可打印的单字符字符串
    returns:
        dict: 映射字典，键为字节(int)，值为对应的可见字符(str)。
    """
    # bs初始为可打印字符的字节值列表
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    # 将不可打印的字节映射到256以上的Unicode码点
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    # 将bs和cs映射为字节到字符的字典
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """
    根据特殊词元拆分文本，防止BPE的合并跨段落合并词元
    args:
        text: 输入字符串。
        special_tokens: 要拆分的特殊词元列表。

    returns:
        拆分后的字符串列表。
    """
    PAT = "|".join(re.escape(tok) for tok in special_tokens)  # 不包含捕获组
    return re.split(PAT, text)


def pre_tokenize(text: str) -> Iterator[str]:
    """
    预分词函数，将文本拆分为预分词单元。
    减轻负载，同时避免语义相近的词被拆散
    args:
        text: 输入字符串。

    returns:
        预分词单元的迭代器。
    """
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # PAT = r"\S+|\s+"    # 按空格拆分
    return (
        m.group(0) for m in re.finditer(PAT, text)
    )  # 返回字符串迭代器而非match对象迭代器


def process_chunk_text(
    chunk: str, special_tokens: list[str]
) -> dict[tuple[bytes, ...], int]:
    """
    处理单个文本块，返回局部预分词频率。
    """
    local_freq = defaultdict(int)
    chunk = chunk.replace("\r\n", "\n").replace("\r", "\n")
    docs = split_on_special_tokens(chunk, special_tokens)
    for doc in docs:
        for pre_token in pre_tokenize(doc):
            token_bytes = pre_token.encode("utf-8")
            byte_seq = tuple(bytes([b]) for b in token_bytes)
            local_freq[byte_seq] += 1
    return local_freq


def process_chunk_range(
    args: tuple[int, int, str, list[str]]
) -> dict[tuple[bytes, ...], int]:
    """处理文件的[start, end)字节区间，返回局部预分词频率。

    设计为可在多进程中运行：worker 自己打开文件并读取对应区间，
    避免把大块 bytes/str 通过 pickle 在进程间传输。
    """

    start, end, input_path, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        data = f.read(end - start)
    if not data:
        return {}
    text = data.decode("utf-8", errors="ignore")
    return process_chunk_text(text, special_tokens)


def bpe_streaming(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    chunk_size_mb: int = 1024,
    num_processes: int = 8,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    训练BPE模型。
    args:
        input_path: 输入文本文件路径。
        vocab_size: 目标词汇表大小（包括特殊词元）。
        special_tokens: 需要保留的特殊词元列表。
    returns:
        词汇表（索引到字节字符串的映射）和合并规则列表。
    """
    vocab: dict[int, bytes] = {
        i: bytes([i]) for i in range(256)
    }  # 存储词汇表，索引到字节字符串的映射
    merges: list[tuple[bytes, bytes]] = []  # 存储合并规则
    pre_token2freq: defaultdict[tuple[bytes, ...], int] = defaultdict(
        int
    )  # 存储预分词单元的频率

    # 处理特殊词元
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    # 流式分词
    stats = os.stat(input_path)
    CHUNK_SIZE = chunk_size_mb * 1024 * 1024
    chunk_num = max(1, stats.st_size // CHUNK_SIZE)
    print(f"The file size is {stats.st_size} bytes, splitting into {chunk_num} chunks.")
    available_cpus = os.cpu_count() or 1
    num_processes = max(1, min(int(num_processes), available_cpus))

    print(f"Start streaming BPE training from {input_path}")
    pretokenize_start_time = time.time()
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, chunk_num, b"<|endoftext|>")

    chunk_count = max(0, len(boundaries) - 1)
    if chunk_count == 0:
        pretokenize_end_time = time.time()
        print(
            f"Finished preprocessing 0 chunks, total time cost: {pretokenize_end_time - pretokenize_start_time:.2f} seconds"
        )
    else:
        print(f"Pre-tokenizing with {num_processes} processes across {chunk_count} chunks")
        chunk_args = (
            (boundaries[i], boundaries[i + 1], str(input_path), special_tokens)
            for i in range(chunk_count)
        )

        with ProcessPoolExecutor(max_workers=num_processes) as ex:
            for chunk_id, local_freq in enumerate(
                ex.map(process_chunk_range, chunk_args, chunksize=1), start=1
            ):
                if local_freq:
                    for k, v in local_freq.items():
                        pre_token2freq[k] += v
                # 释放 worker 返回的字典引用，降低峰值
                del local_freq

                print(
                    f"Processed chunk {chunk_id}, current token units: {len(pre_token2freq):,}, cost time: {time.time() - pretokenize_start_time:.2f} seconds"
                )
    pretokenize_end_time = time.time()
    print(
        f"Finished preprocessing {chunk_count} chunks, total time cost: {pretokenize_end_time - pretokenize_start_time:.2f} seconds"
    )
    # 迭代合并字节对（缓存 + 增量更新版）
    merge_start_time = time.time()
    _bpe_merge_cached(pre_token2freq, vocab, merges, vocab_size, pbar=None)
    merge_end_time = time.time()
    print(
        f"Finished BPE merging, total time cost: {merge_end_time - merge_start_time:.2f} seconds"
    )

    return vocab, merges


def save_vocab_and_merges(output_dir, vocab, merges):
    """
    保存词汇表和合并规则为 HuggingFace 格式。
    args:
        output_dir: 输出目录路径。
        vocab: 词汇表（索引到字节字符串的映射）。
        merges: 合并规则列表。
    """
    os.makedirs(output_dir, exist_ok=True)
    vocab_path = os.path.join(output_dir, "vocab.json")
    merges_path = os.path.join(output_dir, "merges.txt")

    byte_encoder = bytes_to_unicode()  # 获取字节到可见字符的映射

    def decode_bytes(token_bytes: bytes) -> str:
        return "".join(byte_encoder[b] for b in token_bytes)

    vocab_dict = {decode_bytes(v): k for k, v in vocab.items()}  # 构造 vocab.json 字典

    with open(vocab_path, "w", encoding="utf-8") as vf:
        json.dump(vocab_dict, vf, ensure_ascii=False)

    with open(merges_path, "w", encoding="utf-8") as mf:
        mf.write("#version: 0.2\n")
        for a, b in merges:
            mf.write(f"{decode_bytes(a)} {decode_bytes(b)}\n")

    print(f"Vocab and merges saved to {vocab_path} and {merges_path}")


if __name__ == "__main__":
    vocab, merges = bpe_streaming(
        input_path="../../data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
        chunk_size_mb=64,
    )

    # save_vocab_and_merges(output_dir="../data/owt", vocab=vocab, merges=merges)
