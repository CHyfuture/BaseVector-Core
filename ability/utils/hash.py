"""
哈希工具函数
"""
import hashlib
from pathlib import Path
from typing import BinaryIO


def calculate_file_hash(file_path: Path | str, chunk_size: int = 4096) -> str:
    """
    计算文件的MD5哈希值

    Args:
        file_path: 文件路径
        chunk_size: 每次读取的块大小（默认：4096字节）

    Returns:
        MD5哈希值（十六进制字符串）
    """
    md5_hash = hashlib.md5()
    file_path = Path(file_path)

    with open(file_path, "rb") as f:
        # 分块读取，避免大文件占用过多内存
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5_hash.update(chunk)

    return md5_hash.hexdigest()


def calculate_stream_hash(stream: BinaryIO, chunk_size: int = 4096) -> str:
    """
    计算文件流的MD5哈希值

    Args:
        stream: 文件流对象
        chunk_size: 每次读取的块大小

    Returns:
        MD5哈希值（十六进制字符串）
    """
    md5_hash = hashlib.md5()
    stream.seek(0)  # 确保从开始读取

    while chunk := stream.read(chunk_size):
        md5_hash.update(chunk)

    stream.seek(0)  # 重置流位置
    return md5_hash.hexdigest()


def calculate_text_hash(text: str) -> str:
    """
    计算文本的MD5哈希值

    Args:
        text: 文本内容

    Returns:
        MD5哈希值（十六进制字符串）
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()
