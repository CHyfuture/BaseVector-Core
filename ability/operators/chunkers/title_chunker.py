"""
标题切片器
按照Markdown标题层级进行切片
"""
import re
from typing import Any, Dict, List, Optional

from ability.operators.chunkers.base_chunker import BaseChunker, Chunk


class TitleChunker(BaseChunker):
    """标题切片器，按照Markdown标题层级切片"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化标题切片器

        Args:
            config: 配置字典
                - max_depth: 最大标题层级（默认4，即H1-H4）
                - include_headers: 是否在块中包含标题（默认True）
        """
        super().__init__(config)
        self.max_depth = self.get_config("max_depth", 4)
        self.include_headers = self.get_config("include_headers", True)

    def _chunk(self, text: str, **kwargs) -> List[Chunk]:
        """
        按标题层级切片

        Args:
            text: 输入文本（Markdown格式）
            **kwargs: 额外的处理参数

        Returns:
            文档块列表
        """
        text = self._clean_text(text, **kwargs)
        lines = text.split("\n")

        chunks = []
        current_chunk_lines = []
        current_title = ""
        current_level = 0
        chunk_index = 0
        start_index = 0
        current_pos = 0

        for line in lines:
            # 检查是否是标题
            title_match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if title_match:
                # 如果当前块有内容，先保存
                if current_chunk_lines:
                    chunk_text = "\n".join(current_chunk_lines)
                    chunk = Chunk(
                        content=chunk_text.strip(),
                        chunk_index=chunk_index,
                        start_index=start_index,
                        end_index=current_pos,
                        metadata={
                            "strategy": "title",
                            "title": current_title,
                            "title_level": current_level,
                        },
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    start_index = current_pos

                # 更新当前标题
                title_level = len(title_match.group(1))
                current_title = title_match.group(2).strip()
                current_level = title_level

                # 重置当前块
                current_chunk_lines = []
                if self.include_headers:
                    current_chunk_lines.append(line)

            else:
                # 普通内容行
                current_chunk_lines.append(line)

            current_pos += len(line) + 1  # +1 for newline

        # 处理最后一个块
        if current_chunk_lines:
            chunk_text = "\n".join(current_chunk_lines)
            chunk = Chunk(
                content=chunk_text.strip(),
                chunk_index=chunk_index,
                start_index=start_index,
                end_index=current_pos,
                metadata={
                    "strategy": "title",
                    "title": current_title,
                    "title_level": current_level,
                },
            )
            chunks.append(chunk)

        # 如果没有找到标题，返回整个文本作为一个块
        if not chunks:
            chunk = Chunk(
                content=text,
                chunk_index=0,
                start_index=0,
                end_index=len(text),
                metadata={
                    "strategy": "title",
                    "title": "",
                    "title_level": 0,
                },
            )
            chunks.append(chunk)

        return chunks
