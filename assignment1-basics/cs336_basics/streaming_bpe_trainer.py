# 负责从语料中训练bpe编码器，并将结果保存到本地
import os
from typing import BinaryIO, Iterator
import regex as re  # 使用 regex 库，由于re对GPT-2的tokenization支持不好
import json
from collections import defaultdict
from tqdm import tqdm

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


def bpe_streaming(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    chunk_size_mb: int = 1024,
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
    chunk_id = 0

    print(f"Start streaming BPE training from {input_path}")
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, chunk_num, b"<|endoftext|>")
        for i in range(len(boundaries) - 1):
            f.seek(boundaries[i])
            data = f.read(boundaries[i + 1] - boundaries[i])
            if not data:
                break
            chunk_id += 1
            text = data.decode("utf-8", errors="ignore")
            local_freq = process_chunk_text(text, special_tokens)

            # 合并局部统计到主字典
            for k, v in local_freq.items():
                pre_token2freq[k] += v
            del local_freq  # 释放内存

            print(
                f"Processed chunk {chunk_id}, current token units: {len(pre_token2freq):,}"
            )

    print(f"Finished preprocessing {chunk_id} chunks, begin BPE merge...")
    # 迭代合并字节对
    with tqdm(total=vocab_size - len(vocab), desc="BPE merging") as pbar:
        while len(vocab) < vocab_size:
            # 找到最频繁的字节对
            pair2freq: defaultdict[tuple[bytes, bytes], int] = defaultdict(int)
            for byte_seq, freq in pre_token2freq.items():  # 计算字节对频率
                for i in range(len(byte_seq) - 1):
                    pair2freq[byte_seq[i], byte_seq[i + 1]] += freq
            if not pair2freq:  # 没有更多的字节对可合并
                break
            most_frequent_pair = max(pair2freq.items(), key=lambda x: (x[1], x[0]))[
                0
            ]  # 频率最高的字节对，且字典序最小
            # 创建新词汇项
            new_token = most_frequent_pair[0] + most_frequent_pair[1]
            vocab[len(vocab)] = new_token
            # 更新预分词频率
            to_update: dict[tuple[bytes, ...], int] = {}
            for byte_seq, freq in pre_token2freq.items():  # 检查序列是否包含目标字节对
                has_target_pair = False
                for i in range(len(byte_seq) - 1):
                    if (byte_seq[i], byte_seq[i + 1]) == most_frequent_pair:
                        has_target_pair = True
                        pbar.n = pbar.total
                        pbar.refresh()
                        break

                if has_target_pair:  # 记录需要更新的序列
                    to_update[byte_seq] = freq

            # 只更新包含目标字节对的序列
            for byte_seq, freq in to_update.items():
                del pre_token2freq[byte_seq]  # 从原字典中移除旧序列

                new_byte_seq: list[bytes] = []
                i = 0
                while i < len(byte_seq):
                    if (
                        i < len(byte_seq) - 1
                        and (byte_seq[i], byte_seq[i + 1]) == most_frequent_pair
                    ):
                        new_byte_seq.append(new_token)
                        i += 2
                    else:
                        new_byte_seq.append(byte_seq[i])
                        i += 1

                pre_token2freq[tuple(new_byte_seq)] += freq  # 将合并后的序列添加回字典
            # 记录合并规则
            merges.append(most_frequent_pair)
            pbar.update(1)

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
        input_path="../data/owt_train.txt",
        vocab_size=36000,
        special_tokens=["<|endoftext|>"],
        chunk_size_mb=1024,
    )

    save_vocab_and_merges(output_dir="../data/owt", vocab=vocab, merges=merges)
