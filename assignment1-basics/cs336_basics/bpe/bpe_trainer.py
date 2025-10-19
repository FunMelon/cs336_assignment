import os
from typing import BinaryIO, Iterator
import regex as re  # 使用 regex 库，由于re对GPT-2的tokenization支持不好
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    将文件分块为可以独立计数的部分。
    如果边界重叠，可能会返回更少的块。

    args:
        file: 要分块的文件对象，必须以二进制模式打开。
        desired_num_chunks: 期望的块数。
        split_special_token: 用于词元块边界的特殊字节字符串。
    
    returns:
        块边界的字节索引列表。
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # 获取文件的总字节数（通过移动文件指针）
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # 块边界位置的初始猜测，均匀间隔
    # 块从上一个索引开始，不包括最后一个索引
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # 每次向前读取4k字节

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # 从猜测的位置开始
        while True:
            mini_chunk = file.read(mini_chunk_size)  # 读取一个小块

            # 如果到达文件末尾，则此边界应位于文件末尾
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # 在小块中查找特殊词元
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # 确保边界唯一且排序
    return sorted(set(chunk_boundaries))

def split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """
    根据特殊词元拆分文本，保留词元，防止BPE的合并跨段落合并词元
    args:
        text: 输入字符串。
        special_tokens: 要拆分的特殊词元列表。
    
    returns:
        拆分后的字符串列表。
    """
    PAT = "|".join(re.escape(tok) for tok in special_tokens) # 不包含捕获组
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
    return (m.group(0) for m in re.finditer(PAT, text))  # 返回字符串迭代器而非match对象迭代器

def process_chunk(args) -> defaultdict[tuple[bytes, ...], int]:
    """
        处理单个文件块的辅助函数
    args:
        args: 包含块的起始和结束字节索引以及输入文件路径的元组。
    returns:
        预分词单元频率的字典。
    """
    start, end, input_path = args
    local_freq = defaultdict(int)   # 存储预分词单元的频率
    
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        chunk = chunk.replace('\r\n', '\n').replace('\r', '\n')     # 统一换行符
        
        docs = split_on_special_tokens(chunk, ["<|endoftext|>"])    # 按文档边界拆分
        for doc in docs:
            for pre_token in pre_tokenize(doc):
                token_bytes = pre_token.encode("utf-8")
                byte_seq = tuple(bytes([b]) for b in token_bytes)   # 转换为字节序列的元组
                local_freq[byte_seq] += 1
                
    return local_freq

def bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str]
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
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}    # 存储词汇表，索引到字节字符串的映射
    merges: list[tuple[bytes, bytes]] = []  # 存储合并规则
    pre_token2freq: defaultdict[tuple[bytes, ...], int] = defaultdict(int)  # 存储预分词单元的频率

    # 处理特殊词元
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")  # 划分特定大小

        chunk_args = [(start, end, input_path) 
                     for start, end in zip(boundaries[:-1], boundaries[1:])]
        with ProcessPoolExecutor(max_workers=num_processes) as ex:
            results = ex.map(process_chunk, chunk_args, chunksize=1)
            
        for result in results:  # 合并各块的频率
            for k, v in result.items():
                pre_token2freq[k] += v

    while len(vocab) < vocab_size:
        # 找到最频繁的字节对
        pair2freq: defaultdict[tuple[bytes, bytes], int] = defaultdict(int)
        for byte_seq, freq in pre_token2freq.items():   # 计算字节对频率
            for i in range(len(byte_seq) - 1):
                pair = (byte_seq[i], byte_seq[i + 1])
                pair2freq[pair] += freq
        if not pair2freq:  # 没有更多的字节对可合并
            break
        most_frequent_pair = max(pair2freq.items(), key=lambda x: (x[1], x[0]))[0]  # 频率最高的字节对，且字典序最小
        # 创建新词汇项
        new_token = most_frequent_pair[0] + most_frequent_pair[1]
        vocab[len(vocab)] = new_token
        # 更新预分词频率
        new_pre_token2freq: defaultdict[tuple[bytes, ...], int] = defaultdict(int)
        for byte_seq, freq in pre_token2freq.items():
            i = 0
            if most_frequent_pair not in zip(byte_seq, byte_seq[1:]):   # 如果字节对不在序列中，直接复制
                new_pre_token2freq[byte_seq] += freq
                continue
            new_byte_seq = []
            while i < len(byte_seq):
                if i < len(byte_seq) - 1 and (byte_seq[i], byte_seq[i + 1]) == most_frequent_pair:
                    new_byte_seq.append(new_token)
                    i += 2
                else:
                    new_byte_seq.append(byte_seq[i])
                    i += 1
            new_pre_token2freq[tuple(new_byte_seq)] += freq
        pre_token2freq = new_pre_token2freq
        # 记录合并规则
        merges.append(most_frequent_pair)


    return vocab, merges