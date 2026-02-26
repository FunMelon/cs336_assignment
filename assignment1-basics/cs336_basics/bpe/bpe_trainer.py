# 负责从语料中训练bpe编码器，并将结果保存到本地
import os
from typing import BinaryIO, Iterator
import regex as re  # 使用 regex 库，由于re对GPT-2的tokenization支持不好
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import heapq


class _RevPair:
    """用于 heapq 的 pair 排序包装器。

    目标：在频次相同的情况下，选择字典序更大的 pair（与原先 max(key=(freq, pair)) 行为一致）。
    heapq 是最小堆，因此这里反转 pair 的比较，使得“更大”的 pair 变成“更小”。
    """

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
) -> None:
    """使用缓存 + 增量更新的方式执行 BPE 合并。

    - 维护全局 pair 频次 `pair_freq`
    - 维护 pair -> words(包含该 pair 的 byte_seq) 的倒排索引 `pair2seqs`
    - 使用 heap 以 O(log N) 取出当前最频繁 pair；允许 heap 中存在过期条目
    """

    pair_freq: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair2seqs: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)

    # 初始化 pair 统计与倒排索引
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
        # 取出当前最频繁 pair（跳过过期 heap 条目）
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
            break

        a, b = most_frequent_pair
        new_token = a + b
        vocab[len(vocab)] = new_token
        merges.append(most_frequent_pair)

        affected_seqs = pair2seqs.get(most_frequent_pair)
        if not affected_seqs:
            # 理论上不应发生（除非索引过期），继续下一轮
            pair_freq[most_frequent_pair] = 0
            continue

        # 拷贝一份，避免迭代时被我们更新索引
        affected_seqs = set(affected_seqs)

        # 该 pair 即将被合并，不再需要保留它的倒排索引
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

            # 更新 pre_token2freq（词表类型频次）
            del pre_token2freq[old_seq]
            pre_token2freq[new_seq] = pre_token2freq.get(new_seq, 0) + old_freq

            # 从旧序列涉及的 pair 中移除 old_seq，并扣减全局 pair 频次
            for pair, cnt in old_counts.items():
                pair_freq[pair] -= old_freq * cnt
                if pair_freq[pair] <= 0:
                    pair_freq[pair] = 0
                seqs = pair2seqs.get(pair)
                if seqs is not None:
                    seqs.discard(old_seq)
                    if not seqs:
                        pair2seqs.pop(pair, None)

            # 将新序列加入对应 pair 的索引，并增加全局 pair 频次
            for pair, cnt in new_counts.items():
                pair_freq[pair] += old_freq * cnt
                pair2seqs[pair].add(new_seq)

            # 将受影响的 pair 重新推入 heap（允许过期，pop 时校验）
            for pair in set(old_counts.keys()) | set(new_counts.keys()):
                push_pair(pair)


def _bpe_merge_cached(
    pre_token2freq: dict[tuple[bytes, ...], int],
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    vocab_size: int,
) -> None:
    """BPE merge 主入口。

    - 如果本地已安装 Rust 扩展，则优先走 Rust 加速实现
    - 否则自动回退到纯 Python `_bpe_merge_cached_py`
    """

    # 可选 Rust 加速：见 cs336_basics/bpe/RUST_ACCEL.md
    rust_result = None
    try:
        try:
            from .rust_accel import bpe_merge_cached_rust
        except Exception:
            # 允许以脚本方式运行：`python bpe_trainer.py`
            from rust_accel import bpe_merge_cached_rust  # type: ignore

        rust_result = bpe_merge_cached_rust(pre_token2freq, vocab, vocab_size)
    except Exception:
        rust_result = None

    if rust_result is not None:
        print("[cs336_basics.bpe] merge backend: rust", flush=True)
        vocab_out, merges_out = rust_result
        vocab.clear()
        vocab.update(vocab_out)
        merges.clear()
        merges.extend(merges_out)
        return

    print("[cs336_basics.bpe] merge backend: python", flush=True)
    _bpe_merge_cached_py(pre_token2freq, vocab, merges, vocab_size)


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


def process_chunk(args) -> defaultdict[tuple[bytes, ...], int]:
    """
        处理单个文件块的辅助函数
    args:
        args: 包含块的起始和结束字节索引以及输入文件路径的元组。
    returns:
        预分词单元频率的字典。
    """
    start, end, input_path = args
    local_freq = defaultdict(int)  # 存储预分词单元的频率

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        chunk = chunk.replace("\r\n", "\n").replace("\r", "\n")  # 统一换行符

        docs = split_on_special_tokens(chunk, ["<|endoftext|>"])  # 按文档边界拆分
        for doc in docs:
            for pre_token in pre_tokenize(doc):
                token_bytes = pre_token.encode("utf-8")
                byte_seq = tuple(
                    bytes([b]) for b in token_bytes
                )  # 转换为字节序列的元组
                local_freq[byte_seq] += 1

    return local_freq


def bpe(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]
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

    # 多线程预分词
    with open(input_path, "rb") as f:
        num_processes = 8
        boundaries = find_chunk_boundaries(
            f, num_processes, b"<|endoftext|>"
        )  # 划分特定大小

        chunk_args = [
            (start, end, input_path)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]
        # 如果实际只有一个 chunk，启动进程池反而会有明显开销（尤其是小文件或没有特殊 token 时）
        if len(chunk_args) <= 1:
            if chunk_args:
                result = process_chunk(chunk_args[0])
                for k, v in result.items():
                    pre_token2freq[k] += v
        else:
            with ProcessPoolExecutor(max_workers=num_processes) as ex:
                results = ex.map(process_chunk, chunk_args, chunksize=1)
                for result in results:  # 合并各块的频率
                    for k, v in result.items():
                        pre_token2freq[k] += v

    # 使用缓存 + 增量更新的合并实现（避免每轮全量重算 pair2freq）
    _bpe_merge_cached(pre_token2freq, vocab, merges, vocab_size)

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

    # 构造 vocab.json 字典
    vocab_dict = {decode_bytes(v): k for k, v in vocab.items()}

    with open(vocab_path, "w", encoding="utf-8") as vf:
        json.dump(vocab_dict, vf, ensure_ascii=False)

    with open(merges_path, "w", encoding="utf-8") as mf:
        mf.write("#version: 0.2\n")
        for a, b in merges:
            mf.write(f"{decode_bytes(a)} {decode_bytes(b)}\n")

    print(f"Vocab and merges saved to {vocab_path} and {merges_path}")


if __name__ == "__main__":
    vocab, merges = bpe(
        input_path="../../data/TinyStoriesV2-GPT4-valid.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )

    # save_vocab_and_merges(output_dir="../data/ts-v", vocab=vocab, merges=merges)
