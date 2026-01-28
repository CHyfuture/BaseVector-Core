"""
文档切片器基类
"""
from typing import Any, Dict, List, Optional

from ability.operators.base import BaseOperator


class Chunk:
    """文档块数据类"""

    def __init__(
        self,
        content: str,
        chunk_index: int,
        start_index: int = 0,
        end_index: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
        parent_chunk_id: Optional[int] = None,
    ):
        """
        初始化文档块

        Args:
            content: 块内容
            chunk_index: 块索引
            start_index: 在原文中的起始位置
            end_index: 在原文中的结束位置
            metadata: 元数据
            parent_chunk_id: 父块ID（用于父子切片）
        """
        self.content = content
        self.chunk_index = chunk_index
        self.start_index = start_index
        self.end_index = end_index
        self.metadata = metadata or {}
        self.parent_chunk_id = parent_chunk_id

    def __repr__(self):
        return f"Chunk(index={self.chunk_index}, length={len(self.content)}, parent={self.parent_chunk_id})"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "content": self.content,
            "chunk_index": self.chunk_index,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "metadata": self.metadata,
            "parent_chunk_id": self.parent_chunk_id,
        }


class BaseChunker(BaseOperator):
    """
    文档切片器基类
    所有切片策略都应继承此类
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化切片器

        Args:
            config: 配置字典
        """
        super().__init__(config)
        self.chunk_size = self.get_config("chunk_size", 512)
        self.chunk_overlap = self.get_config("chunk_overlap", 50)

    def validate_input(self, input_data: Any) -> bool:
        """
        验证输入数据（文本字符串）

        Args:
            input_data: 输入文本

        Returns:
            验证是否通过
        """
        if not super().validate_input(input_data):
            return False

        if not isinstance(input_data, str):
            self.logger.error(f"Input must be a string, got {type(input_data)}")
            return False

        if not input_data.strip():
            self.logger.error("Input text is empty")
            return False

        return True

    def process(self, input_data: str, **kwargs) -> List[Chunk]:
        """
        切片文档（抽象方法，子类必须实现）

        Args:
            input_data: 输入文本
            **kwargs: 额外的处理参数

        Returns:
            文档块列表
        """
        self.logger.info(f"Chunking text of length {len(input_data)}")
        chunks = self._chunk(input_data, **kwargs)
        self.logger.info(f"Created {len(chunks)} chunks")
        return chunks

    def _chunk(self, text: str, **kwargs) -> List[Chunk]:
        """
        内部切片方法，子类必须实现

        Args:
            text: 输入文本
            **kwargs: 额外的处理参数

        Returns:
            文档块列表
        """
        raise NotImplementedError("Subclass must implement _chunk method")

    def _clean_text(self, text: str, **kwargs) -> str:
        """
        清洗文本（子类可以重写）

        Args:
            text: 原始文本
            **kwargs: 额外的处理参数，可包含：
                - whitespace_pattern: 空白字符正则表达式
                - sentence_delimiters: 句子分隔符正则表达式

        Returns:
            清洗后的文本
        """
        from ability.utils.text_processing import clean_text, remove_empty_lines

        whitespace_pattern = kwargs.get("whitespace_pattern")
        sentence_delimiters = kwargs.get("sentence_delimiters")

        text = remove_empty_lines(text)
        text = clean_text(text, whitespace_pattern=whitespace_pattern)
        return text
