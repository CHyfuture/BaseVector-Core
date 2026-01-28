"""
固定窗口切片器
按固定大小（字符数或Token数）切分文档
"""
from typing import Any, Dict, List, Optional

from ability.operators.chunkers.base_chunker import BaseChunker, Chunk


class FixedChunker(BaseChunker):
    """固定窗口切片器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化固定窗口切片器

        Args:
            config: 配置字典
                - chunk_size: 块大小（字符数，默认512）
                - chunk_overlap: 重叠大小（字符数，默认50）
                - use_tokens: 是否使用Token计数（默认False，使用字符数）
        """
        super().__init__(config)
        self.use_tokens = self.get_config("use_tokens", False)
        self.min_content_ratio = self.get_config("min_content_ratio", 0.5)

    def _chunk(self, text: str, **kwargs) -> List[Chunk]:
        """
        按固定窗口切片

        Args:
            text: 输入文本
            **kwargs: 额外的处理参数

        Returns:
            文档块列表
        """
        text = self._clean_text(text, **kwargs)
        chunks = []

        if self.use_tokens:
            # 使用Token计数（简单实现，可以后续集成tiktoken）
            chunks = self._chunk_by_tokens(text)
        else:
            # 使用字符数
            chunks = self._chunk_by_chars(text)

        return chunks

    def _chunk_by_chars(self, text: str) -> List[Chunk]:
        """
        按字符数切片

        Args:
            text: 输入文本

        Returns:
            文档块列表
        """
        # 如果文本为空，返回空列表
        if not text or not text.strip():
            return []
        
        # 如果文本长度小于等于chunk_size，直接返回单个块
        if len(text) <= self.chunk_size:
            return [
                Chunk(
                    content=text.strip(),
                    chunk_index=0,
                    start_index=0,
                    end_index=len(text),
                    metadata={
                        "chunk_size": len(text),
                        "strategy": "fixed",
                    },
                )
            ]
        
        chunks = []
        start = 0
        chunk_index = 0
        max_iterations = len(text) // max(1, self.chunk_size - self.chunk_overlap) + 10  # 防止无限循环
        iteration_count = 0

        while start < len(text):
            iteration_count += 1
            if iteration_count > max_iterations:
                self.logger.warning(f"Reached max iterations ({max_iterations}), breaking loop")
                break

            # 计算结束位置
            end = min(start + self.chunk_size, len(text))

            # 提取块内容
            chunk_content = text[start:end]

            # 如果不是最后一块，尝试在句子边界处截断
            if end < len(text) and self.chunk_overlap > 0:
                # 向后查找句子结束符
                sentence_end = max(
                    chunk_content.rfind("。"),
                    chunk_content.rfind("."),
                    chunk_content.rfind("！"),
                    chunk_content.rfind("!"),
                    chunk_content.rfind("？"),
                    chunk_content.rfind("?"),
                    chunk_content.rfind("\n"),
                )

                if sentence_end > self.chunk_size * self.min_content_ratio:
                    chunk_content = chunk_content[: sentence_end + 1]
                    end = start + len(chunk_content)

            # 确保end不会小于start
            if end <= start:
                end = start + 1
                chunk_content = text[start:end]

            # 创建块（使用原始内容，不strip，保持位置准确性）
            chunk = Chunk(
                content=chunk_content.strip(),
                chunk_index=chunk_index,
                start_index=start,
                end_index=end,
                metadata={
                    "chunk_size": len(chunk_content),
                    "strategy": "fixed",
                },
            )
            chunks.append(chunk)

            # 计算下一个块的起始位置（考虑重叠）
            next_start = end - self.chunk_overlap if self.chunk_overlap > 0 else end
            
            # 确保start真正前进（防止无限循环）
            if next_start <= start:
                next_start = start + 1
            
            # 如果next_start已经超过文本长度，退出循环
            if next_start >= len(text):
                break
                
            start = next_start
            chunk_index += 1

        return chunks

    def _chunk_by_tokens(self, text: str) -> List[Chunk]:
        """
        按Token数切片（简化实现）

        Args:
            text: 输入文本

        Returns:
            文档块列表
        """
        # 简化实现：将Token数近似为字符数的1/3（中文）或1/4（英文）
        # 实际应用中应使用tiktoken等库
        approximate_chars_per_token = 3
        char_chunk_size = self.chunk_size * approximate_chars_per_token
        char_overlap = self.chunk_overlap * approximate_chars_per_token

        # 临时调整配置
        original_chunk_size = self.chunk_size
        original_overlap = self.chunk_overlap

        self.chunk_size = char_chunk_size
        self.chunk_overlap = char_overlap

        try:
            chunks = self._chunk_by_chars(text)
        finally:
            # 恢复配置
            self.chunk_size = original_chunk_size
            self.chunk_overlap = original_overlap

        return chunks
