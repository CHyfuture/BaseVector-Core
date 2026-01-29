"""
父子切片器
生成大粒度父块（含完整上下文）和小粒度子块（用于精准匹配）
"""
from typing import Any, Dict, List, Optional

from ability.operators.chunkers.base_chunker import BaseChunker, Chunk


class ParentChildChunker(BaseChunker):
    """父子切片器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化父子切片器

        Args:
            config: 配置字典
                - parent_size: 父块大小（字符数，默认2000）
                - child_size: 子块大小（字符数，默认512）
                - child_overlap: 子块重叠（字符数，默认50）
                - base_chunker: 用于生成子块的底层切片器（默认FixedChunker）
        """
        super().__init__(config)
        self.parent_size = self.get_config("parent_size", 2000)
        self.child_size = self.get_config("child_size", 512)
        self.child_overlap = self.get_config("child_overlap", 50)
        self.min_content_ratio = self.get_config("min_content_ratio", 0.5)

    def _chunk(self, text: str, **kwargs) -> List[Chunk]:
        """
        生成父子块

        Args:
            text: 输入文本
            **kwargs: 额外的处理参数

        Returns:
            文档块列表（包含父块和子块）
        """
        text = self._clean_text(text, **kwargs)

        # 先创建父块
        parent_chunks = self._create_parent_chunks(text)

        # 为每个父块创建子块
        all_chunks = []
        chunk_index = 0

        for parent_chunk in parent_chunks:
            # 添加父块
            parent_chunk.chunk_index = chunk_index
            all_chunks.append(parent_chunk)
            chunk_index += 1

            # 创建子块
            child_chunks = self._create_child_chunks(
                parent_chunk.content,
                parent_chunk.start_index,
                parent_chunk_index=parent_chunk.chunk_index,
            )

            # 添加子块
            for child_chunk in child_chunks:
                child_chunk.chunk_index = chunk_index
                child_chunk.parent_chunk_id = parent_chunk.chunk_index
                all_chunks.append(child_chunk)
                chunk_index += 1

        return all_chunks

    def _create_parent_chunks(self, text: str) -> List[Chunk]:
        """
        创建父块

        Args:
            text: 输入文本

        Returns:
            父块列表
        """
        parent_chunks = []
        start = 0
        parent_index = 0

        while start < len(text):
            end = min(start + self.parent_size, len(text))

            # 尝试在段落边界处截断
            if end < len(text):
                # 向后查找段落分隔符
                paragraph_end = text.rfind("\n\n", start, end)
                if paragraph_end > start + self.parent_size * self.min_content_ratio:
                    end = paragraph_end + 2

            chunk_content = text[start:end]

            parent_chunk = Chunk(
                content=chunk_content.strip(),
                chunk_index=parent_index,
                start_index=start,
                end_index=end,
                metadata={
                    "strategy": "parent_child",
                    "chunk_type": "parent",
                    "chunk_size": len(chunk_content),
                },
            )
            parent_chunks.append(parent_chunk)

            start = end
            parent_index += 1

        return parent_chunks

    def _create_child_chunks(
        self, text: str, base_start_index: int, parent_chunk_index: int
    ) -> List[Chunk]:
        """
        创建子块

        Args:
            text: 父块文本
            base_start_index: 父块在原文中的起始位置
            parent_chunk_index: 父块索引

        Returns:
            子块列表
        """
        child_chunks = []
        start = 0
        child_index = 0

        while start < len(text):
            end = min(start + self.child_size, len(text))

            # 如果不是最后一块，尝试在句子边界处截断
            if end < len(text) and self.child_overlap > 0:
                chunk_content = text[start:end]
                sentence_end = max(
                    chunk_content.rfind("。"),
                    chunk_content.rfind("."),
                    chunk_content.rfind("！"),
                    chunk_content.rfind("!"),
                    chunk_content.rfind("？"),
                    chunk_content.rfind("?"),
                    chunk_content.rfind("\n"),
                )

                if sentence_end > self.child_size * self.min_content_ratio:
                    chunk_content = chunk_content[: sentence_end + 1]
                    end = start + len(chunk_content)
            else:
                chunk_content = text[start:end]

            child_chunk = Chunk(
                content=chunk_content.strip(),
                chunk_index=child_index,
                start_index=base_start_index + start,
                end_index=base_start_index + end,
                parent_chunk_id=parent_chunk_index,
                metadata={
                    "strategy": "parent_child",
                    "chunk_type": "child",
                    "chunk_size": len(chunk_content),
                },
            )
            child_chunks.append(child_chunk)

            # 已到达末尾，直接结束，避免 overlap 导致重复切片进入死循环
            if end >= len(text):
                break

            # 计算下一个块的起始位置（考虑重叠）
            start = end - self.child_overlap if self.child_overlap > 0 else end
            child_index += 1

            # 防止无限循环
            if start >= end:
                break

        return child_chunks
