"""
文本处理工具函数
"""
import re
from typing import List, Optional


def clean_text(text: str, whitespace_pattern: Optional[str] = None) -> str:
    """
    清洗文本：去除多余空白字符

    Args:
        text: 原始文本
        whitespace_pattern: 空白字符正则表达式（默认：r"\s+"）

    Returns:
        清洗后的文本
    """
    # 去除首尾空白
    text = text.strip()

    # 将多个连续空白字符替换为单个空格
    pattern = whitespace_pattern or r"\s+"
    text = re.sub(pattern, " ", text)

    return text


def split_by_sentences(text: str, sentence_delimiters: Optional[str] = None) -> List[str]:
    """
    按句子分割文本

    Args:
        text: 原始文本
        sentence_delimiters: 句子分隔符正则表达式（默认：r"[。！？.!?]\s*"）

    Returns:
        句子列表
    """
    # 简单的句子分割（可以根据需求使用更复杂的NLP工具）
    pattern = sentence_delimiters or r"[。！？.!?]\s*"
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]


def remove_empty_lines(text: str) -> str:
    """
    移除空行

    Args:
        text: 原始文本

    Returns:
        处理后的文本
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(lines)


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    截断文本到指定长度

    Args:
        text: 原始文本
        max_length: 最大长度
        suffix: 截断后缀

    Returns:
        截断后的文本
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix
