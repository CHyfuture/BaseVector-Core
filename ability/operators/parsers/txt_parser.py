"""
TXT文档解析器
"""
from pathlib import Path
from typing import Any, Dict

from ability.operators.parsers.base_parser import BaseParser


class TXTParser(BaseParser):
    """TXT文档解析器"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化TXT解析器

        Args:
            config: 配置字典
        """
        super().__init__(config)
        # 支持带点和不带点的扩展名格式
        self.supported_extensions = [".txt", "txt"]

    def _initialize(self) -> None:
        """初始化TXT解析器"""
        self.logger.info("TXT parser initialized")

    def _parse(self, file_path: Path, **kwargs) -> Dict[str, Any]:
        """
        解析TXT文档

        Args:
            file_path: TXT文件路径
            **kwargs: 额外的处理参数

        Returns:
            解析结果字典
        """
        # 尝试多种编码
        encodings = ["utf-8", "gbk", "gb2312", "latin-1"]
        content = None

        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue

        if content is None:
            raise ValueError(f"Failed to decode file: {file_path}")

        # 简单的结构提取
        structure = {
            "lines": len(content.split("\n")),
            "paragraphs": len([p for p in content.split("\n\n") if p.strip()]),
        }

        return {
            "content": content,
            "metadata": {},
            "structure": structure,
        }
