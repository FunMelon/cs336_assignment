from typing import Iterable, Iterator
import regex as re
import json
from .bpe_trainer import bytes_to_unicode


def pre_tokenize(text: str) -> Iterator[str]:
    """
    预分词函数，将文本拆分为预分词单元。
    args:
        text: 输入字符串。
    returns:
        预分词单元的迭代器。
    """
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    return (m.group(0) for m in re.finditer(PAT, text))


def get_pairs(word: list[bytes]):
    """
    获取字节序列中的所有相邻字节对。
    args:
        word: 字节序列的列表表示。
    returns:
        相邻字节对的集合。
    """
    return {(word[i], word[i + 1]) for i in range(len(word) - 1)}


def encode_str_to_bytes(s: str) -> bytes:
    """
    将字符串编码为字节序列，使用 bytes_to_unicode 映射。
    args:
        s: 输入字符串。
    returns:
        对应的字节序列。
    """
    byte_encoder = bytes_to_unicode()
    byte_decoder = {v: k for k, v in byte_encoder.items()}
    return bytes([byte_decoder[c] for c in s])


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        从指定的词汇表和合并规则初始化分词器。
        args:
            vocab (dict[int, bytes]): 词汇表，映射 token ID 到字节序列。
            merges (list[tuple[bytes, bytes]]): BPE 合并规则列表。
            special_tokens (list[str], optional): 额外的特殊标记列表。
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.token2id: dict[bytes, int] = {  # 反向映射
            token_bytes: token_id for token_id, token_bytes in vocab.items()
        }

        if self.special_tokens:
            sorted_tokens = sorted(
                self.special_tokens, key=len, reverse=True
            )  # 按长度降序排序，防止子串冲突
            self._special_re = re.compile(
                "(" + "|".join(map(re.escape, sorted_tokens)) + ")"
            )
        else:
            self._special_re = None

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        """
        类方法，从指定的文件路径加载词汇表和合并规则，并初始化分词器实例。
        args:
            vocab_filepath (str): 词汇表文件路径。
            merges_filepath (str): 合并规则文件路径。
            special_tokens (list[str], optional): 额外的特殊标记列表。
        returns:
            Tokenizer: 初始化后的分词器实例。
        """

        # 加载词表
        vocab: dict[int, bytes] = {}
        with open(vocab_filepath, encoding="utf-8") as vf:
            vocab_dict = json.load(vf)

        inv_vocab = {v: k for k, v in vocab_dict.items()}  # 反转词表映射
        vocab: dict[int, bytes] = {}
        for idx, token_str in inv_vocab.items():
            vocab[int(idx)] = encode_str_to_bytes(token_str)

        # 加载合并规则
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as mf:
            lines = [l.strip() for l in mf.readlines()]
            # 跳过 #version: 0.2 行
            for line in lines:
                if not line or line.startswith("#"):
                    continue
                parts = line.split(" ")
                if len(parts) != 2:
                    continue
                a, b = parts
                merges.append((encode_str_to_bytes(a), encode_str_to_bytes(b)))

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        将输入文本编码为 token ID 列表。
        args:
            text (str): 输入文本字符串。
        returns:
            list[int]: 对应的 token ID 列表。
        """

        text = text.replace("\r\n", "\n").replace("\r", "\n")  # 统一换行符
        segments: list[tuple[bool, str]] = []  # (是否为特殊token, 文本片段)
        if self.special_tokens:
            # 用正则切分文本，保留特殊标记位置
            if self._special_re is not None:
                parts = self._special_re.split(text)
            else:
                parts = [text]
            for part in parts:
                if not part:
                    continue
                if part in self.special_tokens:
                    segments.append((True, part))
                else:
                    segments.append((False, part))
        else:
            segments = [(False, text)]

        ids: list[int] = []
        merge_ranks = {
            pair: i for i, pair in enumerate(self.merges)
        }  # 合并规则的排名，按照顺序编号

        for is_special, segment in segments:  # 按照特殊标记分段处理
            # 处理特殊标记，直接查表
            if is_special:
                token_id = self.token2id.get(segment.encode("utf-8"))
                if token_id is not None:
                    ids.append(token_id)
                else:
                    raise ValueError(f"Special token not found in vocab: {segment}")
                continue  # 跳过BPE

            # 处理普通文本，进行BPE编码
            pre_tokens: list[list[bytes]] = []  # 存储预分词单元的字节序列元组
            for pre_token in pre_tokenize(segment):  # 对每个分词单元
                token_bytes = pre_token.encode("utf-8")  # 转换为bytes
                pre_tokens.append([bytes([b]) for b in token_bytes])  # 存为list[bytes]

            for byte_seq in pre_tokens:
                pairs = get_pairs(byte_seq)  # 获取当前字节序列的相邻字节对
                while pairs:
                    best = min(
                        pairs, key=lambda pair: merge_ranks.get(pair, float("inf"))
                    )  # 找到排名最高的字节对
                    if best not in merge_ranks:
                        break  # 没有更多可合并的字节对
                    first, second = best
                    new_seq: list[bytes] = []
                    i = 0
                    while i < len(byte_seq):
                        if (
                            i < len(byte_seq) - 1
                            and byte_seq[i] == first
                            and byte_seq[i + 1] == second
                        ):
                            new_seq.append(first + second)  # 合并字节对
                            i += 2
                        else:
                            new_seq.append(byte_seq[i])
                            i += 1
                    byte_seq = new_seq
                    pairs = get_pairs(byte_seq)  # 更新字节对集合

                # 将最终的字节序列转换为 token ID
                for bseq in byte_seq:
                    token_id = self.token2id.get(bseq)
                    if token_id is not None:
                        ids.append(token_id)
                    else:  # 处理未登录词
                        raise ValueError(f"Token not found in vocab: {bseq}")
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        迭代地将输入文本序列编码为 token ID 序列。
        args:
            iterable (Iterable[str]): 输入文本字符串的可迭代对象。
        returns:
            Iterator[int]: 对应的 token ID 迭代器。
        """
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        """
        将 token ID 列表解码为文本字符串。
        args:
            ids (list[int]): 输入的 token ID 列表。
        returns:
            str: 解码后的文本字符串。
        """
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="ignore")
