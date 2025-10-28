# 将文本文件编码为 uint16 二进制文件，按块处理大文件并生成元信息
import os
import json
import numpy as np
import math
import tqdm
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.bpe_trainer import find_chunk_boundaries


def saveID(input_path: str, save_dir: str, tokenizer: Tokenizer, chunk_size_mb=16):
    """
    按块读取大文件 -> 编码 -> 连续写入 uint16 二进制文件
    同时生成 meta.json 记录元信息
    """
    # 自动创建输出目录
    os.makedirs(save_dir, exist_ok=True)

    # 输出文件路径
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_bin = os.path.join(save_dir, f"{base_name}.bin")
    meta_path = os.path.join(save_dir, "meta.json")

    print(f"Encoding {input_path} → {output_bin}")

    total_tokens = 0

    with open(input_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        chunk_num = max(1, math.ceil(file_size / (chunk_size_mb * 1024 * 1024)))
        print(f"File size: {file_size} bytes, Chunk size: {chunk_size_mb} MB, Chunks: {chunk_num}")
        f.seek(0)
        boundaries = find_chunk_boundaries(f, chunk_num, b"<|endoftext|>")
        f.seek(0, 2)
        file_end = f.tell()
        boundaries.append(file_end)

        with open(output_bin, "wb") as out_f:
            with tqdm.tqdm(total=len(boundaries) - 1, desc="Encoding chunks") as pbar:
                for i in range(len(boundaries) - 1):
                    start, end = boundaries[i], boundaries[i + 1]
                    f.seek(start)
                    raw_bytes = f.read(end - start)
                    text = raw_bytes.decode("utf-8", errors="ignore")

                    # 分块编码
                    token_ids = tokenizer.encode(text)
                    arr = np.array(token_ids, dtype=np.uint16)
                    arr.tofile(out_f)
                    total_tokens += len(arr)
                    
                    pbar.set_postfix({"chunk": i + 1, "bytes": end - start, "tokens": len(arr)})
                    pbar.update(1)

    # 生成 meta.json
    meta = {
        "source_file": os.path.abspath(input_path),
        "output_file": os.path.abspath(output_bin),
        "dtype": "uint16",
        "vocab_size": tokenizer.vocab_size,
        "total_tokens": total_tokens,
        "chunks": len(boundaries) - 1
    }

    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(meta, mf, ensure_ascii=False, indent=4)

    print(f"\n✅ Done. Saved binary to {output_bin}")
    print(f"🧾 Meta info saved to {meta_path}")


if __name__ == "__main__":
    tokenizer_tiny = Tokenizer.from_files("./data/vocab/owt/vocab.json", "./data/vocab/owt/merges.txt")

    saveID(
        input_path="./data/owt_valid.txt",
        chunk_size_mb=16,
        save_dir="./data/id/owt-t-id",
        tokenizer=tokenizer_tiny
    )