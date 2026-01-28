"""
Markdown文档解析器
"""
from pathlib import Path
from typing import Any, Dict

from ability.operators.parsers.base_parser import BaseParser


class MarkdownParser(BaseParser):
    """Markdown文档解析器"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化Markdown解析器

        Args:
            config: 配置字典
        """
        super().__init__(config)
        self.supported_extensions = [".md", ".markdown"]

    def _initialize(self) -> None:
        """初始化Markdown解析器"""
        self.logger.info("Markdown parser initialized")

    def _parse(self, file_path: Path, **kwargs) -> Dict[str, Any]:
        """
        解析Markdown文档

        Args:
            file_path: Markdown文件路径
            **kwargs: 额外的处理参数

        Returns:
            解析结果字典
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 提取元数据（如果存在Front Matter）
        metadata = {}
        structure = {
            "headings": [],
            "code_blocks": [],
        }

        # 检查是否有YAML Front Matter
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                front_matter = parts[1].strip()
                content = parts[2].strip()

                # 简单解析YAML（可以后续使用PyYAML增强）
                for line in front_matter.split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        metadata[key.strip()] = value.strip().strip('"').strip("'")

        # 提取结构信息
        lines = content.split("\n")
        for line in lines:
            # 提取标题
            if line.startswith("#"):
                level = len(line) - len(line.lstrip("#"))
                text = line.lstrip("#").strip()
                structure["headings"].append(
                    {
                        "level": level,
                        "text": text,
                    }
                )

        return {
            "content": content,
            "metadata": metadata,
            "structure": structure,
        }
